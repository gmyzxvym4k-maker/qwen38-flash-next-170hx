#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantize_ple.py —— 把 Qwen3.8-Flash-Next 的 PLE n-gram 表（BF16，95.4 GiB）
离线量化成 **逐行对称 INT8 + 逐行 BF16 scale**（47.7 GiB + 0.6 GiB）。

为什么是 INT8 而不是 FP8-e4m3（实测结论，见 docs/04-ple-int8.md）：
  全表 |x| 动态范围只有约 480×（max|x| = 0.0447），FP8 的 4 位指数纯属浪费。
  同一次扫描代价下：INT8 逐行 scale relMSE = 4.36e-05 / 行余弦 0.9999784，
  FP8-e4m3 relMSE = 6.70e-04 —— INT8 精确 15 倍，体积还只有 BF16 的一半。

产物（与引擎侧补丁 `_attach_int8_disk_resident_ngram_table` 的读法逐字节对齐）：
  <out>/ple_ngram_int8.bin   int8      [rows, hidden]  row-major，rows*hidden 字节
  <out>/ple_ngram_scale.bin  bfloat16  [rows]          rows*2 字节
  <out>/ple_ngram_meta.json  几何 / 量化方案 / 全表与抽样误差
  <out>/_state.json          断点状态（完成后改名 _state.json.done）

数值定义（必须与引擎侧 `q.to(bf16) * s.to(bf16)` 互逆）：
  scale = bf16(row_absmax / 127)            # 先算 fp32 再落 bf16
  q     = clamp(round_half_even(x / fp32(scale)), -127, 127) -> int8
  row_absmax == 0 时 scale=0、q=0（本表实测 0 行）

用法：
  # 量化（约 8 分钟，需 ~2 GB 内存，纯 CPU，可断点续传）
  python3 quantize_ple.py --model /path/to/Qwen3.8-Flash-Next-W4A16-AutoRound \
                          --out   /path/to/ple
  # 只校验既有产物（抽样重算并逐字节比对，几分钟内完成，不写文件）
  python3 quantize_ple.py --model ... --out /path/to/ple --verify-only --rows 20000
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import torch

CHUNK_ROWS = 524288  # 512K 行 ≈ 160 MiB bf16 / 80 MiB int8，与生产产物一致


def read_safetensors_header(path: str):
    """返回 (header_dict, data_base)。data_base = 8 + len(header_json)。

    【关键坑】safetensors 的 data_offsets 是**相对数据段起点**的，不是相对文件头。
    切片必须用 data_base + offset，漏加 8+header_len 会把 header 当数据，
    整张表错位若干元素 —— 症状是"内容词全对、只有标点乱"，极难反推。
    """
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    return head, 8 + n


def find_ngram_tensors(head: dict):
    """挑出 n-gram 表张量并按 data_offset 排序，校验连续性。

    W4A16-AutoRound 的 checkpoint 把表拆成 128 个 [2500012, 160] BF16 张量，
    在数据段里首尾相接；拼接后即 [320001536, 160] 的一张表。
    """
    cands = []
    for name, meta in head.items():
        if name == "__metadata__":
            continue
        if "ngram" not in name and "n_gram" not in name:
            continue
        if meta.get("dtype") != "BF16":
            continue
        cands.append((meta["data_offsets"][0], meta["data_offsets"][1], name, meta))
    if not cands:
        raise SystemExit("在 header 里没找到 BF16 的 n-gram 张量，检查 --model 是否指对分片")
    cands.sort()
    s0 = cands[0][0]
    for a, b, name, meta in cands:
        if meta["shape"][1] != cands[0][3]["shape"][1]:
            raise SystemExit(f"{name} hidden 维度不一致：{meta['shape']}")
    # 连续性：后一块的起点 == 前一块的终点
    cur = s0
    for a, b, name, meta in cands:
        if a != cur:
            raise SystemExit(f"分片不连续：{name} 起点 {a} != 期望 {cur}")
        cur = b
    return cands, s0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型目录（含 model-00016-of-00017.safetensors）")
    ap.add_argument("--shard", default="", help="指定含 n-gram 表的分片文件；默认自动扫描所有分片")
    ap.add_argument("--out", required=True, help="产物目录")
    ap.add_argument("--verify-only", action="store_true", help="只抽样重算并比对既有产物字节")
    ap.add_argument("--rows", type=int, default=20000, help="--verify-only 的抽样行数")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    # 1) 定位含表的分片
    shards = [args.shard] if args.shard else sorted(
        os.path.join(args.model, f) for f in os.listdir(args.model)
        if f.endswith(".safetensors")
    )
    chosen = None
    for s in shards:
        if not os.path.exists(s):
            continue
        head, db = read_safetensors_header(s)
        try:
            tensors, s0 = find_ngram_tensors(head)
        except SystemExit:
            continue
        if chosen is None:
            chosen = (s, head, db, tensors, s0)
    if chosen is None:
        raise SystemExit("没有任何分片含 BF16 n-gram 张量")
    src, head, data_base, tensors, first_off = chosen
    hidden = int(tensors[0][3]["shape"][1])
    rows = sum(int(t[3]["shape"][0]) for t in tensors)
    table_start = data_base + first_off
    print(f"[geo] src={os.path.basename(src)} shards={len(tensors)} rows={rows} hidden={hidden}")
    print(f"[geo] header_len={data_base - 8} data_base={data_base} table_data_start_in_file={table_start}")
    assert rows * hidden * 2 == sum(t[1] - t[0] for t in tensors), "表字节数与分片总和不符"

    os.makedirs(args.out, exist_ok=True)
    i8p = os.path.join(args.out, "ple_ngram_int8.bin")
    scp = os.path.join(args.out, "ple_ngram_scale.bin")
    metap = os.path.join(args.out, "ple_ngram_meta.json")
    statep = os.path.join(args.out, "_state.json")
    donep = os.path.join(args.out, "_state.json.done")

    # 2) 抽样校验模式：不写任何文件
    if args.verify_only:
        if not (os.path.exists(i8p) and os.path.exists(scp)):
            raise SystemExit("产物不存在，无法校验")
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randint(0, rows, (args.rows,), generator=g)
        bad_i8 = bad_sc = 0
        with open(src, "rb") as fs, open(i8p, "rb") as fi, open(scp, "rb") as fsc:
            for r in idx.tolist():
                x = read_row(fs, table_start, r, hidden)
                q, s = quantize_row(x)
                fs_off = r * hidden
                fi.seek(fs_off); a = fi.read(hidden)
                fsc.seek(r * 2); b = fsc.read(2)
                if a != q.numpy().tobytes():
                    bad_i8 += 1
                if b != s.view(torch.uint16).numpy().tobytes():
                    bad_sc += 1
        print(f"[verify] rows={args.rows} int8_mismatch={bad_i8} scale_mismatch={bad_sc}")
        if bad_i8 or bad_sc:
            print("[verify] ✗ 与既有产物字节不一致（量化定义不同？）", file=sys.stderr)
            return 1
        print("[verify] ✓ 抽样字节完全一致：量化定义与引擎侧产物互逆")
        return 0

    # 3) 断点续传
    start_chunk = 0
    if os.path.exists(donep):
        print("[state] 已完成（存在 _state.json.done）。要重做请先删产物目录。")
        return 0
    if os.path.exists(statep):
        start_chunk = int(json.load(open(statep))["next_chunk"])
        print(f"[state] 从 chunk {start_chunk} 续传")

    n_chunks = (rows + CHUNK_ROWS - 1) // CHUNK_ROWS
    fi = open(i8p, "r+b" if start_chunk else "wb")
    fsc = open(scp, "r+b" if start_chunk else "wb")
    fi.truncate(rows * hidden); fsc.truncate(rows * 2)
    if start_chunk:
        fi.seek(start_chunk * CHUNK_ROWS * hidden); fsc.seek(start_chunk * CHUNK_ROWS * 2)

    # 全表误差累加器（fp64）
    sum_x2 = sum_e2 = cos_sum = 0.0
    rows_done = zero_scale = rounded_zero = 0
    if start_chunk:
        st = json.load(open(statep)).get("accum", {})
        sum_x2, sum_e2 = st.get("sum_x2", 0.0), st.get("sum_e2", 0.0)
        cos_sum, cos_count = st.get("cos_sum", 0.0), st.get("cos_count", 0)
        rows_done, zero_scale, rounded_zero = (
            st.get("rows_done", 0), st.get("zero_scale_rows", 0), st.get("rounded_zero_rows", 0))

    t0 = time.time()
    with open(src, "rb") as fs:
        for c in range(start_chunk, n_chunks):
            r0 = c * CHUNK_ROWS
            n = min(CHUNK_ROWS, rows - r0)
            x = read_rows(fs, table_start, r0, n, hidden)          # [n,hidden] fp32
            q, s = quantize_block(x)                                # int8, bf16[n]
            deq = q.to(torch.float32) * s.to(torch.float32).unsqueeze(-1)
            e = deq - x
            sum_x2 += float((x * x).sum()); sum_e2 += float((e * e).sum())
            cos_sum += float(cosine_rows(x, deq).sum()); cos_count = rows_done + n
            rows_done += n
            zero_scale += int((s == 0).sum())
            rounded_zero += int(((q == 0) & (x != 0)).sum())
            fi.write(q.numpy().tobytes()); fsc.write(s.view(torch.uint16).numpy().tobytes())
            json.dump({"next_chunk": c + 1, "chunk_rows": CHUNK_ROWS, "src": src,
                       "accum": {"rows_done": rows_done, "sum_x2": sum_x2, "sum_e2": sum_e2,
                                 "cos_sum": cos_sum, "cos_count": cos_count,
                                 "zero_scale_rows": zero_scale, "rounded_zero_rows": rounded_zero}},
                      open(statep, "w"))
            if c % 20 == 0 or c == n_chunks - 1:
                el = time.time() - t0
                print(f"[{c+1}/{n_chunks}] rows={rows_done} elapsed={el:.0f}s "
                      f"rate={rows_done/max(el,1):.0f} rows/s", flush=True)
    fi.close(); fsc.close()

    # 4) 抽样复核（写完后立刻自证）
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randint(0, rows, (100000,), generator=g)
    with open(src, "rb") as fs, open(i8p, "rb") as a, open(scp, "rb") as b:
        bad = 0
        s2 = e2 = cs = 0.0
        for r in idx.tolist():
            x = read_row(fs, table_start, r, hidden)
            q, s = quantize_row(x)
            a.seek(r * hidden); av = a.read(hidden)
            b.seek(r * 2); bv = b.read(2)
            if av != q.numpy().tobytes() or bv != s.view(torch.uint16).numpy().tobytes():
                bad += 1
            deq = q.to(torch.float32) * s.to(torch.float32)
            s2 += float((x * x).sum()); e2 += float(((deq - x) ** 2).sum())
            cs += float(torch.dot(deq, x) / (deq.norm() * x.norm() + 1e-30))
    print(f"[verify] 抽样 100000 行：字节不一致={bad} relMSE={e2/max(s2,1e-30):.3e} "
          f"行余弦均值={cs/100000:.7f}")
    if bad:
        return 1

    meta = {
        "rows": rows, "hidden": hidden,
        "dtype_int8": "int8 row-major [rows, hidden]",
        "dtype_scale": "bfloat16 [rows]",
        "src_file": src, "header_len": data_base - 8, "data_base": data_base,
        "table_data_start_in_file": table_start,
        "shard_order_check": {
            "shards": len(tensors), "contiguous": True,
            "shard_shapes_first_last": [list(tensors[0][3]["shape"]), list(tensors[-1][3]["shape"])],
        },
        "quant": {
            "scheme": "symmetric per-row INT8",
            "scale": "row_absmax/127, stored bfloat16, division done with stored bf16 scale",
            "clamp": [-127, 127],
            "rounding": "round-half-to-even (torch.round)",
        },
        "stats_full_table": {
            "rel_mse": sum_e2 / max(sum_x2, 1e-30),
            "row_cosine_mean": cos_sum / max(cos_count, 1),
            "zero_scale_rows": zero_scale,
            "quantized_to_zero_rows": rounded_zero,
        },
        "sample_verify": {
            "sampled_rows": 100000, "seed": args.seed,
            "sample_rel_mse": e2 / max(s2, 1e-30),
            "sample_row_cosine_mean": cs / 100000,
            "byte_mismatch": bad,
        },
        "int8_bytes": rows * hidden, "scale_bytes": rows * 2,
        "elapsed_s": round(time.time() - t0, 3),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    json.dump(meta, open(metap, "w"), indent=2, ensure_ascii=False)
    os.rename(statep, donep)
    print(f"[done] -> {args.out}  ({rows*hidden/2**30:.2f} GiB int8 + {rows*2/2**30:.3f} GiB scale)")
    return 0


def read_rows(fs, table_start: int, r0: int, n: int, hidden: int) -> torch.Tensor:
    fs.seek(table_start + r0 * hidden * 2)
    buf = fs.read(n * hidden * 2)
    t = torch.frombuffer(bytearray(buf), dtype=torch.uint16)
    return t.view(torch.bfloat16).view(n, hidden).to(torch.float32)


def read_row(fs, table_start: int, r: int, hidden: int) -> torch.Tensor:
    return read_rows(fs, table_start, r, 1, hidden)[0]


def quantize_block(x: torch.Tensor):
    absmax = x.abs().amax(dim=-1)
    scale = (absmax / 127.0).to(torch.bfloat16)
    sf = scale.to(torch.float32)
    sf_safe = torch.where(sf == 0, torch.ones_like(sf), sf)
    q = torch.clamp(torch.round(x / sf_safe.unsqueeze(-1)), -127, 127).to(torch.int8)
    q = torch.where((sf == 0).unsqueeze(-1), torch.zeros_like(q), q)
    return q, scale


def quantize_row(x: torch.Tensor):
    q, s = quantize_block(x.unsqueeze(0))
    return q[0], s[0]


def cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    num = (a * b).sum(-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-30
    return num / den


if __name__ == "__main__":
    sys.exit(main())
