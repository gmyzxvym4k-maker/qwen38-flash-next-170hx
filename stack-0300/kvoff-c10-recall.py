#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kvoff-c10-recall —— 内存二级缓存「回载内容无损」定向复述探针（不重启，只发请求）。

为什么要单独写：
  · 2026-09-27 上一版探针（kvoff-c10-probe.py）在带 c11 的窗口里已经拿到
    external hits=25,856 token / CPU→GPU=828MB，但**回载那一问的 text 为空**，
    导致「验证码复述」判 false —— 无法区分「回载数据坏了」与「探针取数/预算问题」。
  · 本探针把这条链路单独做干净：①预算给足（缺省 2048）；②打印**原始响应**的
    choices[0].message 全部字段 + finish_reason + usage；③命中/回载字节/复述三判据。

序列：标定 → 建档 A(带验证码) → 挤池 N 篇 → 重发 A 问验证码。
判据（全部满足才算 PASS）：
  hits  delta > 0   （CPU 档命中）
  load  delta > 0   （CPU→GPU 真搬了字节）
  code 出现在复述文本里（content 与 reasoning_content 都算）
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango uniform victor "
         "whiskey xray yankee zulu").split()
_BYTE_COUNTERS = ("vllm:kv_offload_load_bytes", "vllm:kv_offload_store_bytes",
                  "vllm:kv_offload_total_bytes_total")
_FORBID = ("_created", "_time", "_size", "usage_perc", "_bucket")


def post(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def metric_lines(base):
    murl = re.sub(r"/v1/?$", "", base) + "/metrics"
    try:
        with urllib.request.urlopen(murl, timeout=20) as r:
            return r.read().decode().splitlines()
    except Exception as exc:
        print("[recall] /metrics 抓取失败：%s" % exc, file=sys.stderr)
        return []


def counters(base):
    """返回 {CPU_to_GPU, GPU_to_CPU, external_hits, external_queries, usage_perc}。"""
    out = {"CPU_to_GPU": 0.0, "GPU_to_CPU": 0.0, "external_hits": 0.0,
           "external_queries": 0.0, "usage_perc": 0.0}
    for line in metric_lines(base):
        if line.startswith("#"):
            continue
        name = line.split("{")[0].split(" ")[0]
        try:
            val = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        if name in _BYTE_COUNTERS and not any(f in name for f in _FORBID):
            if name.endswith("load_bytes"):
                out["CPU_to_GPU"] += val
            elif name.endswith("store_bytes"):
                out["GPU_to_CPU"] += val
            elif 'transfer_type="CPU_to_GPU"' in line:
                out["CPU_to_GPU"] += val
            elif 'transfer_type="GPU_to_CPU"' in line:
                out["GPU_to_CPU"] += val
        elif name == "vllm:external_prefix_cache_hits_total":
            out["external_hits"] += val
        elif name == "vllm:external_prefix_cache_queries_total":
            out["external_queries"] += val
        elif name == "vllm:kv_offload_cpu_cache_usage_perc":
            out["usage_perc"] = max(out["usage_perc"], val)
    return out


def make_doc(seed, tokens, ratio, code):
    """约 tokens 个 token 的英文自然文本，把验证码埋在 1/3 处。"""
    rng = random.Random(seed)
    n_words = max(64, int(tokens / (ratio * 1.30)))
    parts, ln = [], 0
    target = n_words
    while ln < target:
        tag = " ".join(rng.choices(WORDS, k=14))
        parts.append(tag)
        ln += 15
        if len(parts) == int(target / 3) // 15:
            parts.append("The verification code is %s . Remember it." % code)
    return " ".join(parts)


def ask(base, model, doc, question, max_tokens):
    payload = {"model": model,
               "messages": [{"role": "user", "content": doc + "\n\n" + question}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
    j = post(base + "/chat/completions", payload)
    ch = (j.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    usage = j.get("usage") or {}
    det = usage.get("prompt_tokens_details") or {}
    text = " ".join(filter(None, [msg.get("content") or "",
                                  msg.get("reasoning_content") or ""])).strip()
    return {"raw_keys": sorted(msg.keys()),
            "finish_reason": ch.get("finish_reason"),
            "content_len": len(msg.get("content") or ""),
            "reasoning_len": len(msg.get("reasoning_content") or ""),
            "text": text,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": det.get("cached_tokens")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:18420/v1")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--tokens", type=int, default=30000)
    ap.add_argument("--flush", type=int, default=5)
    ap.add_argument("--gap", type=float, default=3.0)
    ap.add_argument("--answer-tokens", type=int, default=2048)
    ap.add_argument("--ratio", type=float, default=0.0)
    ap.add_argument("--max-prompt-tokens", type=int, default=55000)
    a = ap.parse_args()

    out = {"endpoint": a.endpoint, "steps": []}

    # 标定 token/字符比
    if a.ratio <= 0:
        probe = " ".join(random.Random(7).choices(WORDS, k=1200))
        r = ask(a.endpoint, a.model, probe, "Reply OK.", 8)
        a.ratio = (r["prompt_tokens"] or 1200) / max(1, len(probe))
        out["ratio_tok_per_char"] = round(a.ratio, 4)
    print("[recall] ratio=%.4f tokens/char" % a.ratio, file=sys.stderr)

    code = "CODE-RCL%02d-0000" % random.Random(int(time.time())).randint(10, 99)
    docA = make_doc(1001, a.tokens, a.ratio, code)
    if a.max_prompt_tokens and (len(docA) * a.ratio) > a.max_prompt_tokens:
        docA = docA[: int(a.max_prompt_tokens / a.ratio)]

    c0 = counters(a.endpoint)
    r = ask(a.endpoint, a.model, docA, "Reply with the single word OK.", 8)
    out["steps"].append({"phase": "build", "code": code,
                         "prompt_tokens": r["prompt_tokens"]})
    print("[recall] 建档 %s tok（验证码 %s）" % (r["prompt_tokens"], code), file=sys.stderr)
    time.sleep(a.gap)

    for i in range(a.flush):
        d = make_doc(2000 + i, a.tokens, a.ratio, "CODE-NONE-0000")
        rr = ask(a.endpoint, a.model, d, "Reply with the single word OK.", 8)
        out["steps"].append({"phase": "flush", "i": i, "prompt_tokens": rr["prompt_tokens"]})
        time.sleep(a.gap)

    time.sleep(max(8.0, a.gap))  # 等异步 store 落盘 + 提交延迟
    c1 = counters(a.endpoint)

    q = ("Repeat the verification code that appears in the document above, "
         "exactly as written. Answer with the code only.")
    r2 = ask(a.endpoint, a.model, docA, q, a.answer_tokens)
    c2 = counters(a.endpoint)

    d_hits = c2["external_hits"] - c1["external_hits"]
    d_load = c2["CPU_to_GPU"] - c1["CPU_to_GPU"]
    recalled = (code in r2["text"]) or (code.split("-")[1] in r2["text"])
    out["reload"] = {**r2, "text": r2["text"][:600], "code": code}
    out["deltas"] = {"external_hits_tokens": d_hits, "cpu_to_gpu_bytes": d_load}
    out["cumulative"] = {k: c2[k] for k in ("external_hits", "external_queries",
                                            "CPU_to_GPU", "GPU_to_CPU", "usage_perc")}
    out["codes"] = {"expected": code, "recalled": bool(recalled)}
    out["verdict"] = bool(d_hits > 0 and d_load > 0 and recalled)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["verdict"] else 1


if __name__ == "__main__":
    sys.exit(main())
