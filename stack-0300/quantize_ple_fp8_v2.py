#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""quantize_ple_fp8_v2.py —— 按 0.30.0 **FP8 PLE 的真实加载契约**重做产物。

第 11 轮踩到的坑（已实机复现）：
  `Qwen4ExpNGramEmbedding.load_weights` 只拦截 `ngram_embedding.shard_<i>.weight`，
  其余名字交给 AutoWeightsLoader。上一版我给每个 shard 都配了 `shard_i.weight_scale`
  ⇒ 名字落到 AutoWeightsLoader ⇒ 报
  `ValueError: There is no module or parameter named 'ngram_embedding.shard_0'`。

正确契约（源码 ngram_embedding.py:880-930 + Qwen4ExpPLEFp8EmbeddingMethod）：
  · 128 个 `...ngram_embedding.shard_<i>.weight`（fp8_e4m3，行序 = shard_index×shard_size）
  · **一个** `...ngram_embedding.weight_scale`（f32 标量，PerTensor 全局 scale）
  · config.json 里 `text_config.ple_embedding_dtype = "float8_e4m3fn"`
  ⇒ 所有 shard 必须共用**同一个** scale（本表各 shard 的 amax 实测差异 <3%，代价可忽略）。

做法：先从既有 v1 产物读出 128 个 per-shard scale 取最大值作为全局 S，
再流式重写 128 个 fp8 分片（量化用 S）+ 追加唯一 scale，其余文件软链。
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open

FP8_MAX = 448.0
SHARD = "model-00016-of-00017.safetensors"
PREFIX = ("model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
          ".shard_%d.weight")
SCALE_NAME = ("model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
              ".weight_scale")
NEVER_LINK = {SHARD, "config.json", "model.safetensors.index.json"}


def write_header(fh, entries):
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
    blob += b" " * ((8 - (len(blob) % 8)) % 8)
    fh.write(struct.pack("<Q", len(blob)))
    fh.write(blob)
    return 8 + len(blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原始 BF16 模型目录")
    ap.add_argument("--v1", required=True, help="上一版产物（取 per-shard scale）")
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)

    for f in sorted(os.listdir(a.src)):
        if f in NEVER_LINK:
            continue
        dst = os.path.join(a.dst, f)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(a.src, f), dst)

    # 1) 从 v1 产物聚合全局 scale
    with safe_open(os.path.join(a.v1, SHARD), framework="pt") as g:
        scales = [float(g.get_tensor(k + "_scale")[0])
                  for k in sorted(g.keys()) if k.endswith(".weight")]
    if not scales:
        print("v1 产物里没有 scale，退出", file=sys.stderr)
        return 2
    S = max(scales)
    print("per-shard scale 数=%d min=%.6g max=%.6g -> 全局 S=%.6g"
          % (len(scales), min(scales), S, S), flush=True)

    with safe_open(os.path.join(a.src, SHARD), framework="pt") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("shard_")[1].split(".")[0]))
        entries = []
        for k in keys:
            entries.append((k, "F8_E4M3", tuple(f.get_slice(k).get_shape())))
        entries.append((SCALE_NAME, "F32", (1,)))

        out = os.path.join(a.dst, SHARD)
        with open(out, "wb") as fh:
            fh.seek(write_header(fh, entries))
            for i, k in enumerate(keys):
                w = f.get_tensor(k).to(torch.float32)
                q = (w / S).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                fh.write(q.view(torch.uint8).numpy().tobytes())
                del w, q
                if i % 16 == 0 or i == len(keys) - 1:
                    print("  [%3d/%d] %s" % (i + 1, len(keys), k), flush=True)
            fh.write(np.asarray([S], dtype=np.float32).tobytes())

    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    for k in keys:
        wm[k] = SHARD
    wm[SCALE_NAME] = SHARD
    idx["metadata"]["total_size"] = sum(
        os.path.getsize(os.path.join(a.dst, v)) for v in set(wm.values()))
    json.dump(idx, open(os.path.join(a.dst, "model.safetensors.index.json"), "w"),
              indent=2)

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    (cfg.get("text_config") or cfg)["ple_embedding_dtype"] = "float8_e4m3fn"
    json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=2)

    print("DONE ->", a.dst)
    print("shard GB:", round(os.path.getsize(out) / 1e9, 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
