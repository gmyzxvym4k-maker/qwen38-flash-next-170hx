#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kvoff-mon —— 每 2s 采样 CPU 档关键指标，观察「store→ready→可用」的生命周期。

用法：python3 kvoff-mon.py <秒数> <输出csv> [endpoint]
输出列：t, usage_perc, write_usage, read_usage, alloc_cnt, alloc_sum,
        store_bytes, load_bytes, ext_hits, ext_queries, running, waiting, deferred
"""
import re
import sys
import time
import urllib.request

WANT = {
    "vllm:kv_offload_cpu_cache_usage_perc": "usage_perc",
    "vllm:kv_offload_cpu_cache_write_usage_perc": "write_usage",
    "vllm:kv_offload_cpu_cache_read_usage_perc": "read_usage",
    "vllm:kv_offload_cpu_allocation_size_count": "alloc_cnt",
    "vllm:kv_offload_cpu_allocation_size_sum": "alloc_sum",
    "vllm:kv_offload_store_size_sum": "store_bytes",
    "vllm:kv_offload_load_size_sum": "load_bytes",
    "vllm:external_prefix_cache_hits_total": "ext_hits",
    "vllm:external_prefix_cache_queries_total": "ext_queries",
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
}
COLS = ["t", "usage_perc", "write_usage", "read_usage", "alloc_cnt", "alloc_sum",
        "store_bytes", "load_bytes", "ext_hits", "ext_queries", "running", "waiting"]


def fetch(base):
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
            return r.read().decode().splitlines()
    except Exception:
        return []


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 90
    outp = sys.argv[2] if len(sys.argv) > 2 else "/tmp/kvoff-mon.csv"
    base = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:18420"
    t0 = time.time()
    with open(outp, "w") as fh:
        fh.write(",".join(COLS) + "\n")
        while time.time() - t0 < secs:
            vals = {c: 0.0 for c in COLS}
            vals["t"] = round(time.time() - t0, 1)
            for line in fetch(base):
                if line.startswith("#"):
                    continue
                name = line.split("{")[0].split(" ")[0]
                key = WANT.get(name)
                if not key:
                    continue
                try:
                    v = float(line.rsplit(" ", 1)[-1])
                except ValueError:
                    continue
                vals[key] = max(vals[key], v)
            fh.write(",".join(str(vals[c]) for c in COLS) + "\n")
            fh.flush()
            time.sleep(2)
    print("done ->", outp)


if __name__ == "__main__":
    main()
