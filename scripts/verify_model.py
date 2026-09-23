#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_model.py —— 校验 Qwen3.8-Flash-Next-W4A16-AutoRound 的完整性。

只做四件事（都是本项目真实踩过的坑）：
  1. index.json 引用的每个分片都存在，且 `metadata.total_size` 与实际字节一致；
  2. 每个分片的 safetensors 头部可解析、data_offsets 连续且不越界（截断下载的典型特征）；
  3. config.json 的架构与量化字段符合预期（**特别检查 quant_method 必须是 auto-round**）；
  4. PLE n-gram 表：128 个 [2500012,160] BF16 张量首尾相接，总行数 320,001,536。

注意：仓库本身没有 model-00002-of-00017.safetensors，index 也不引用它 —— 这不是缺漏。

退出码：0=全部通过；1=有 FAIL。
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

EXPECT_ARCH = "Qwen4ExpForConditionalGeneration"
EXPECT_QUANT = "auto-round"
PLE_ROWS = 320001536
PLE_HIDDEN = 160
PLE_SHARDS = 128
PLE_SHARD_ROWS = 2500012

fails = 0


def ck(name, expect, got, must=True):
    global fails
    ok = (expect == "-" or str(expect) == str(got))
    if not ok and not must:
        print(f"  \033[33m!\033[0m {name:<52} 期望 {expect} / 实得 {got}")
        return
    print(f"  {'\033[32m✓\033[0m' if ok else '\033[31m✗\033[0m'} {name:<52} {got if ok else f'期望 {expect} / 实得 {got}'}")
    if not ok:
        fails += 1


def header_of(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > 64 * 2**20:
            raise ValueError(f"头部长度异常：{n}")
        return json.loads(f.read(n)), 8 + n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="模型目录")
    ap.add_argument("--skip-shard-parse", action="store_true", help="只查存在性与大小，不解析每个分片头部（快）")
    args = ap.parse_args()
    d = os.path.abspath(args.dir)
    if not os.path.isdir(d):
        print(f"目录不存在：{d}", file=sys.stderr)
        return 1

    print("── 1. 文件与 index ──")
    idxp = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(idxp):
        print("  ✗ 缺 model.safetensors.index.json"); return 1
    idx = json.load(open(idxp, encoding="utf-8"))
    wm = idx["weight_map"]
    shards = sorted(set(wm.values()))
    print(f"  index 引用分片 {len(shards)} 个，张量 {len(wm)} 个")
    missing = [s for s in shards if not os.path.exists(os.path.join(d, s))]
    ck("index 引用的分片全部存在", "0 个缺失", f"{len(missing)} 个缺失" if missing else "0 个缺失")
    if missing:
        print("   ", missing[:5])
    real = [f for f in os.listdir(d) if f.endswith(".safetensors")]
    print(f"  磁盘上 safetensors = {len(real)}（编号缺 00002 属正常，仓库本身如此）")
    actual_total = sum(os.path.getsize(os.path.join(d, s)) for s in shards if os.path.exists(os.path.join(d, s)))
    exp_total = idx.get("metadata", {}).get("total_size")
    if exp_total:
        ck("metadata.total_size == 实际字节", exp_total, actual_total, must=False)

    print("── 2. 各分片 safetensors 头部 ──")
    if args.skip_shard_parse:
        print("  （已跳过）")
    else:
        bad = 0
        for s in shards:
            p = os.path.join(d, s)
            if not os.path.exists(p):
                continue
            try:
                head, db = header_of(p)
                size = os.path.getsize(p)
                end = max((m["data_offsets"][1] for k, m in head.items() if k != "__metadata__"), default=0)
                if db + end > size:
                    print(f"  \033[31m✗\033[0m {s}: 声明数据到 {db+end}，文件只有 {size} → **截断**")
                    bad += 1
            except Exception as e:
                print(f"  \033[31m✗\033[0m {s}: 头部解析失败 {e}")
                bad += 1
        ck("全部分片头部可解析且未截断", "0 个异常", f"{bad} 个异常")

    print("── 3. config.json ──")
    cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
    tc = cfg.get("text_config", cfg)
    ck("architectures", f"['{EXPECT_ARCH}']", cfg.get("architectures"))
    qm = (cfg.get("quantization_config") or {}).get("quant_method")
    ck("quantization_config.quant_method（**不得改成 gptq**）", EXPECT_QUANT, qm)
    ck("bits / group_size", "4 / 128",
       f"{(cfg.get('quantization_config') or {}).get('bits')} / {(cfg.get('quantization_config') or {}).get('group_size')}")
    ck("text_config.max_position_embeddings", 262144, tc.get("max_position_embeddings"), must=False)
    print(f"  层数 {tc.get('num_hidden_layers')}  hidden {tc.get('hidden_size')}  "
          f"experts {tc.get('num_experts')}  vocab {tc.get('vocab_size')}")
    print(f"  ple_layer_ids {tc.get('ple_layer_ids')}  ngram_size {tc.get('ngram_size')}  hc_count {tc.get('hc_count')}")
    mtp = [k for k in wm if k.startswith("mtp.")]
    ck("MTP 草稿头权重存在", ">0", f"{len(mtp)} 个键")

    print("── 4. PLE n-gram 表 ──")
    ngram = {k: m for k, m in wm.items() if "ngram" in k.lower()}
    if not ngram:
        print("  ! index 里没找到 ngram 张量名（若模型版本不同请人工确认）")
    else:
        # 表集中在某个分片里；找出含 ngram 最多的分片
        byshard = {}
        for k, s in ngram.items():
            byshard.setdefault(s, []).append(k)
        big = max(byshard.items(), key=lambda kv: len(kv[1]))
        shard, keys = big
        ck("n-gram 张量数", PLE_SHARDS, len(keys))
        head, db = header_of(os.path.join(d, shard))
        shapes = {tuple(head[k]["shape"]) for k in keys}
        ck("张量形状", f"[({PLE_SHARD_ROWS}, {PLE_HIDDEN})]", str(sorted(shapes)), must=False)
        offs = sorted((head[k]["data_offsets"][0], head[k]["data_offsets"][1]) for k in keys)
        contiguous = all(offs[i][1] == offs[i + 1][0] for i in range(len(offs) - 1))
        ck("分片在数据段内首尾相接", True, contiguous)
        total_rows = sum(head[k]["shape"][0] for k in keys)
        ck("n-gram 表总行数", PLE_ROWS, total_rows)
        dtypes = {head[k]["dtype"] for k in keys}
        ck("n-gram 表 dtype（BF16 才能走本仓库 INT8 量化）", "{'BF16'}", str(dtypes))
        print(f"  表所在分片：{shard}（{os.path.getsize(os.path.join(d, shard))/2**30:.1f} GiB），"
              f"data_base={db}，表起点={db + offs[0][0]}")

    print()
    print("══════════ " + ("✓ 模型完整，可以继续（docs/04 做 INT8 量化）" if fails == 0
                            else f"✗ {fails} 项不通过，先修模型再说"))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
