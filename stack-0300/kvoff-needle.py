#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kvoff-needle —— 内存档回载「内容无损」正向验证（自然文本 + 验证码）。

为什么再写一个：
  · kvoff-dbg/kvoff-tail 证明了**命中与回载**（ext hits>0 / CPU→GPU>0），但问的是 "Reply OK."，不校验内容；
  · kvoff-c10-recall2 的合成文档是「随机词沙拉」，模型在两轮里都退化成复读
    （content_len 18427、finish=length），虽可对比但不给出正向证据。
本探针用自然句库拼长文档、把验证码埋在文中，再问验证码——命中的那一轮必须精准复述。

【必须记住的坑】建档/挤池/重发三步的 chat_template_kwargs 必须完全一致：
vLLM 的块哈希把模板前缀算进去，一边开思考一边关思考 ⇒ 首块哈希不同 ⇒ 永远不可能命中。
"""
import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

SENTS = [
    "The storage subsystem keeps a compressed copy of every block it has evicted.",
    "Operators prefer predictable latency over peak throughput in interactive workloads.",
    "Each pipeline stage exchanges a small tensor with its neighbour once per step.",
    "Memory bandwidth is often the limiting factor for table lookups on the host side.",
    "The scheduler postpones a request when a transfer is already in flight for it.",
    "Hardware counters expose the number of bytes moved between the two cache tiers.",
    "A block hash covers the token contents together with the identifiers of its parents.",
    "Long prefixes are reused most often when the same document is revisited later.",
    "Thermal headroom determines how long the accelerator can stay at its boost clock.",
    "Log files rotate on a schedule so that the oldest entries are dropped first.",
    "The verification table is rebuilt from the checkpoint whenever the process restarts.",
    "Engineers measure the round trip time of every control message during bring up.",
    "A single slow worker can stall the whole collective operation for seconds.",
    "The cache policy ranks entries by recency and evicts the least recently used one.",
    "Quantised weights reduce the footprint at the cost of a small accuracy penalty.",
    "Prefill work is dominated by large matrix products and scales with the batch size.",
    "Decoding is dominated by the sequential dependency between successive tokens.",
    "Watching the memory controller counters tells you whether the link is saturated.",
]
NEEDLE = "The verification code is {code} . Please remember this code."


def post(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        return {"__http_error__": exc.code}
    except Exception as exc:
        return {"__error__": "%s: %s" % (type(exc).__name__, exc)}


def ask(base, model, doc, question, max_tokens, no_think=True):
    payload = {"model": model,
               "messages": [{"role": "user", "content": doc + "\n\n" + question}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.time()
    j = post(base + "/chat/completions", payload)
    el = round(time.time() - t0, 1)
    if "__http_error__" in j or "__error__" in j:
        return {"id": None, "error": j, "elapsed_s": el}
    ch = (j.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    usage = j.get("usage") or {}
    det = usage.get("prompt_tokens_details") or {}
    return {"id": j.get("id"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": det.get("cached_tokens"),
            "finish_reason": ch.get("finish_reason"),
            "text": ((msg.get("content") or "") + (msg.get("reasoning") or ""))[:300],
            "elapsed_s": el}


def build_doc(rng, sentences, needle=None):
    """自然句拼装：句序随机、句量由 sentences 决定；needle 埋在 70% 处。"""
    out, at = [], int(sentences * 0.7)
    for i in range(sentences):
        if needle and i == at:
            out.append(needle)
        out.append(rng.choice(SENTS))
    if needle:
        out.append(needle)
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:18420/v1")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--a-sentences", type=int, default=3000)
    ap.add_argument("--b-sentences", type=int, default=5600)
    ap.add_argument("--flushes", type=int, default=2)
    ap.add_argument("--gap", type=float, default=5.0)
    ap.add_argument("--answer-tokens", type=int, default=256)
    a = ap.parse_args()
    salt = int(time.time())
    code = "CODE-NL%03d-%04d" % (salt % 1000, salt % 9973)

    def counters():
        try:
            with urllib.request.urlopen(
                    a.endpoint.rstrip("/").replace("/v1", "") + "/metrics", timeout=15) as r:
                lines = r.read().decode().splitlines()
        except Exception:
            return {}
        out = {}
        for ln in lines:
            if ln.startswith("#"):
                continue
            name = ln.split("{")[0].split(" ")[0]
            try:
                v = float(ln.rsplit(" ", 1)[-1])
            except ValueError:
                continue
            if name == "vllm:kv_offload_total_bytes_total":
                if 'transfer_type="CPU_to_GPU"' in ln:
                    out["CPU_to_GPU"] = out.get("CPU_to_GPU", 0.0) + v
                elif 'transfer_type="GPU_to_CPU"' in ln:
                    out["GPU_to_CPU"] = out.get("GPU_to_CPU", 0.0) + v
            elif name == "vllm:external_prefix_cache_hits_total":
                out["ext_hits"] = out.get("ext_hits", 0.0) + v
        return out

    rng = random.Random(salt)
    docA = build_doc(rng, a.a_sentences, NEEDLE.format(code=code))
    docB = [build_doc(random.Random(salt + 7 + i), a.b_sentences) for i in range(a.flushes)]

    out = {"salt": salt, "code": code, "steps": []}
    c0 = counters()
    r = ask(a.endpoint, a.model, docA, "Reply with the single word OK.", 8)
    out["steps"].append({"phase": "build", **r})
    print("[needle] 建档 %s tok（%s）" % (r.get("prompt_tokens"), code), file=sys.stderr)
    time.sleep(a.gap)
    for i, d in enumerate(docB):
        rr = ask(a.endpoint, a.model, d, "Reply with the single word OK.", 8)
        out["steps"].append({"phase": "flush-%d" % i, **rr})
        print("[needle] 挤池%d %s tok" % (i, rr.get("prompt_tokens")), file=sys.stderr)
        time.sleep(a.gap)
    time.sleep(max(8.0, a.gap))
    c1 = counters()
    q = ("What is the verification code in the document above? "
         "Reply with just the code, nothing else.")
    r2 = ask(a.endpoint, a.model, docA, q, a.answer_tokens)
    c2 = counters()
    out["steps"].append({"phase": "reload", **r2})
    d_hits = c2.get("ext_hits", 0.0) - c1.get("ext_hits", 0.0)
    d_load = c2.get("CPU_to_GPU", 0.0) - c1.get("CPU_to_GPU", 0.0)
    recalled = code in (r2.get("text") or "")
    print("[needle] 重发 id=%s prompt=%s cached=%s 复述=%s (%ss %s)"
          % (r2.get("id"), r2.get("prompt_tokens"), r2.get("cached_tokens"), recalled,
             r2.get("elapsed_s"), r2.get("finish_reason")), file=sys.stderr)
    print("[needle] ext_hits_delta=%s cpu_to_gpu_delta=%s" % (d_hits, d_load), file=sys.stderr)
    print("[needle] 文本=%r" % ((r2.get("text") or "")[:200],), file=sys.stderr)
    out["deltas"] = {"external_hits_tokens": d_hits, "cpu_to_gpu_bytes": d_load}
    out["recalled"] = bool(recalled)
    out["verdict"] = bool(d_hits > 0 and d_load > 0 and recalled)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["verdict"] else 1


if __name__ == "__main__":
    sys.exit(main())
