#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make-longctx-copy.py —— 制作长上下文（512K / 1M）模型副本。

原理（为什么必须做副本，而不是只加 --hf-overrides）：
  Qwen3.8-Flash-Next 的原生 `max_position_embeddings = 262144` 写在 **text_config** 里
  （顶层 config 没有这个字段）。要跑更长上下文必须同时满足两件事：
    1) vLLM 侧 `--max-model-len` 放大 + `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`（否则 ModelConfig
       校验直接拒启）；
    2) RoPE 必须做 YaRN 缩放，否则位置越过原生上限后注意力打分退化（长文"看不太懂"）。
  本仓库的启动脚本走的是"副本"路线：**把 YaRN 烘进副本的 config.json**，
  其余文件全部软链回原目录（零额外磁盘占用，且不动原模型）。
  这样 inner 脚本只需切换模型目录，不需要 --hf-overrides，也不会与
  "dict 型 hf_overrides 不传给草稿模型"之类的上游行为纠缠。

副本 config.json 的改动（只改 text_config 两处，其余逐字节保留）：
  text_config.max_position_embeddings            = <target>
  text_config.rope_parameters.rope_type          = "yarn"
  text_config.rope_parameters.factor             = <target / 262144>
  text_config.rope_parameters.original_max_position_embeddings = 262144
  （rope_parameters 里原有的 mrope_section / partial_rotary_factor / rope_theta 等一律保留）

用法：
  python3 make-longctx-copy.py --src /media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound \
                               --dst /media/ll/data/models-1m/Qwen3.8-Flash-Next-W4A16-AutoRound-1M \
                               --target-len 1048576
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ORIGIN = 262144


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原始模型目录（权重所在处）")
    ap.add_argument("--dst", required=True, help="副本目录（新建）")
    ap.add_argument("--target-len", type=int, required=True, help="副本要支持的上下文长度，如 524288 / 1048576")
    ap.add_argument("--overwrite-config", action="store_true", help="已存在 config.json 时重写")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    cfg_path = os.path.join(src, "config.json")
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"源目录里没有 config.json：{cfg_path}")
    if args.target_len <= ORIGIN:
        raise SystemExit(f"--target-len 必须 > {ORIGIN}，否则不需要副本")
    if args.target_len % ORIGIN:
        print(f"[warn] {args.target_len} 不是 {ORIGIN} 的整数倍，factor 取比值", file=sys.stderr)
    os.makedirs(dst, exist_ok=True)

    # 1) 除 config.json 外全部软链（含 safetensors 分片、tokenizer、index）
    linked = skipped = 0
    for name in sorted(os.listdir(src)):
        if name == "config.json":
            continue
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if os.path.islink(d) and os.path.realpath(d) == os.path.realpath(s):
            skipped += 1
            continue
        if os.path.exists(d) or os.path.islink(d):
            raise SystemExit(f"目标已存在且不是指向源的软链，拒绝覆盖：{d}")
        os.symlink(s, d)
        linked += 1

    # 2) 写副本 config.json
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    tc = cfg.get("text_config")
    target = cfg if tc is None else tc
    if "max_position_embeddings" not in target:
        raise SystemExit("config.json 的 text_config 里没有 max_position_embeddings，"
                         "确认这是 Qwen3.8-Flash-Next 的 checkpoint")
    orig = int(target["max_position_embeddings"])
    if orig != ORIGIN:
        print(f"[warn] 源模型 max_position_embeddings={orig}（预期 {ORIGIN}），"
              f"factor 按 {args.target_len}/{orig} 计算", file=sys.stderr)

    rp = dict(target.get("rope_parameters") or {})
    factor = round(args.target_len / orig, 6)
    rp.update({
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": orig,
    })
    target["max_position_embeddings"] = args.target_len
    target["rope_parameters"] = rp

    out = os.path.join(dst, "config.json")
    if os.path.exists(out) and not args.overwrite_config:
        raise SystemExit(f"{out} 已存在（要重写请加 --overwrite-config）")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[ok] 副本目录 {dst}")
    print(f"     软链 {linked} 个文件（复用 {skipped}），config.json 已重写")
    print(f"     text_config.max_position_embeddings = {args.target_len}")
    print(f"     rope_parameters = {{rope_type: yarn, factor: {factor}, "
          f"original_max_position_embeddings: {orig}, ...保留原有键}}")
    print(f"     磁盘增量：{os.path.getsize(out)} 字节（权重全部软链，零拷贝）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
