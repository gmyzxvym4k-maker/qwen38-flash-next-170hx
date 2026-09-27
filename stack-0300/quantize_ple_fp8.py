#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""quantize_ple_fp8.py —— 把 PLE n-gram 表离线量化成 **0.30.0 原生支持的 FP8-e4m3**
格式，产出一个新的模型目录（其余分片软链，不占额外磁盘）。

为什么这么做（2026-09-27 用户的「PLE 走 int8/硬盘，启动更快」建议在 0.30.0 栈上的落地）：
  · 官方 0.30.0 **没有** 旧 chroot 栈那套「INT8 + 磁盘 mmap + 查表时反量化」路径：
    `config/engram.py` 只有 `cpu_offload`（锁页内存/进显存）两档，查表在 GPU 侧 UVA。
  · 但 0.30.0 **原生支持 FP8 PLE**：`models/qwen4_exp/nvidia/ngram_embedding.py`
    的 `Qwen4ExpPLEFp8EmbeddingMethod`（weight=F8_E4M3 + 每个张量**一个全局 scale**），
    由 `config.ple_embedding_dtype == "float8_e4m3fn"` 选中（该文件 698 行）。
  · 本模型 BF16 表 = 128 张量 × [2500012,160] = 102.4 GB（单分片 model-00016）。
    量化成 FP8 后 = 51.2 GB ⇒ 启动读表时间腰斩、常驻内存 95.4GiB → ~48GiB
    （顺带把 KVOFF pinned 档位的内存压力也释放一半）。

产物：DST/config.json（打上 ple_embedding_dtype）+ DST/model.safetensors.index.json
      + DST/model-00016-of-00017.safetensors（FP8 权重 + `<名>.weight_scale` 标量 F32）
      + 其余文件全部 symlink 回原模型目录（不复制）。

用法（部署机）：
  /home/ll/vllm-env/bin/python quantize_ple_fp8.py \
      --src /media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound \
      --dst /media/ll/data/models-fp8ple/Qwen3.8-Flash-Next-W4A16-AutoRound-fp8ple
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open

FP8_MAX = 448.0  # torch.float8_e4m3fn 的最大可表示值
SHARD = "model-00016-of-00017.safetensors"
NEVER_LINK = {SHARD, "config.json", "model.safetensors.index.json"}
DTYPE_STR = {torch.float8_e4m3fn: "F8_E4M3", torch.float32: "F32"}


def write_header(fh, entries):
    """entries: [(name, dtype_str, shape)] → 写 safetensors 头，返回数据区起点。"""
    off = 0
    header = {}
    for name, dts, shape in entries:
        n = 1
        for d in shape:
            n *= int(d)
        itemsize = 1 if dts == "F8_E4M3" else 4
        size = n * itemsize
        header[name] = {"dtype": dts, "shape": list(shape),
                        "data_offsets": [off, off + size]}
        off += size
    blob = json.dumps(header, separators=(",", ":")).encode()
    pad = (8 - (len(blob) % 8)) % 8
    blob += b" " * pad
    fh.write(struct.pack("<Q", len(blob)))
    fh.write(blob)
    return 8 + len(blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只量化前 N 个（调试用）")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)

    # 1) 其余文件 symlink（绝不覆盖/写穿源目录）
    for f in sorted(os.listdir(a.src)):
        if f in NEVER_LINK:
            continue
        dst = os.path.join(a.dst, f)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(a.src, f), dst)

    src_shard = os.path.join(a.src, SHARD)
    with safe_open(src_shard, framework="pt") as f:
        keys = sorted(f.keys())
        if a.limit:
            keys = keys[: a.limit]
        names = []
        for k in keys:
            names.append((k, torch.float8_e4m3fn, tuple(f.get_slice(k).get_shape())))
            names.append((k + "_scale", torch.float32, (1,)))
        entries = [(n, DTYPE_STR[d], s) for n, d, s in names]

        out_shard = os.path.join(a.dst, SHARD)
        with open(out_shard, "wb") as fh:
            data_start = write_header(fh, entries)
            fh.seek(data_start)
            for i, k in enumerate(keys):
                w = f.get_tensor(k).to(torch.float32)
                amax = float(w.abs().max())
                scale = (amax / FP8_MAX) if amax > 0 else 1.0
                q = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                fh.write(q.view(torch.uint8).numpy().tobytes())
                fh.write(np.asarray([scale], dtype=np.float32).tobytes())
                del w, q
                if i % 8 == 0 or i == len(keys) - 1:
                    print("  [%3d/%d] %s scale=%.6g" % (i + 1, len(keys), k, scale),
                          flush=True)

    # 2) index：加 scale 条目 + 重算 total_size
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    for k in keys:
        wm[k] = SHARD
        wm[k + "_scale"] = SHARD
    files = set(wm.values())
    idx["metadata"]["total_size"] = sum(
        os.path.getsize(os.path.join(a.dst, v)) for v in files
    )
    json.dump(idx, open(os.path.join(a.dst, "model.safetensors.index.json"), "w"),
              indent=2)

    # 3) config：声明 FP8 PLE（ngram_embedding.py:698 按这个字段选实现）
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    (cfg.get("text_config") or cfg)["ple_embedding_dtype"] = "float8_e4m3fn"
    json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=2)

    print("DONE ->", a.dst)
    print("shard size GB:", round(os.path.getsize(os.path.join(a.dst, SHARD)) / 1e9, 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
