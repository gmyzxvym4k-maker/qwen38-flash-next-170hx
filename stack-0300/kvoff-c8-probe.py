#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kvoff-c8-probe —— CPU KV 二级缓存「回载」真值探针（不重启、只发请求）。

判据口径（历史定版，别改）：
  · `vllm:external_prefix_cache_hits_total` 增量 > 0  = 命中发生在 **CPU 档**
    （GPU 显存前缀缓存命中走 prefix_cache_hits_total，不算二级缓存收益）；
  · `CPU_to_GPU` 字节增量 > 0 = 真的有把 KV 从内存搬回显存；
  · 验证码精准复述 = 回载内容无损（光有计数器不算数）。
  只看 cached_tokens 会被显存命中污染 ⇒ 必须先把 GPU 池挤干净再重发。

用法：
  python3 kvoff-c8-probe.py                 # 缺省 2 篇建档 + 6 篇挤池 + 重发
  python3 kvoff-c8-probe.py --docs 2 --flush 6 --tokens 120000
  python3 kvoff-c8-probe.py --endpoint http://127.0.0.1:18420/v1
退出码：0=三项判据全绿；1=任一失败（详情打印 JSON）。
"""
import argparse
import json
import random
import re
import string
import sys
import time
import urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango uniform victor "
         "whiskey xray yankee zulu").split()


def post(url, payload, timeout=1800):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()), time.time() - t0


def metrics(base):
    """抓 /metrics，返回 {指标名: 数值}（同名多标签取求和）。"""
    murl = re.sub(r"/v1/?$", "", base) + "/metrics"
    try:
        with urllib.request.urlopen(murl, timeout=20) as r:
            txt = r.read().decode()
    except Exception as exc:
        print("[probe] /metrics 抓取失败：%s" % exc, file=sys.stderr)
        return {}
    out = {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{")[0].split(" ")[0]
        val = line.rsplit(" ", 1)[-1].strip()
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            pass
    return out


def offload_bytes(m):
    """CPU_to_GPU / GPU_to_CPU 累计字节（histogram 的 _sum 或 counter 皆可）。"""
    res = {}
    for key in ("CPU_to_GPU", "GPU_to_CPU"):
        tot = 0.0
        for name, val in m.items():
            if "kv_offload" in name and key.lower().replace("_", "") in name.lower().replace("_", ""):
                tot += val
        res[key] = tot
    # 兜底：label 形态（direction="CPU_to_GPU"）时上面按名字匹配不到，逐行再算一次
    return res


def offload_bytes_labeled(base):
    """按标签精确取 CPU_to_GPU / GPU_to_GPU 方向的字节累计。"""
    murl = re.sub(r"/v1/?$", "", base) + "/metrics"
    want = {"CPU_to_GPU": 0.0, "GPU_to_CPU": 0.0}
    try:
        with urllib.request.urlopen(murl, timeout=20) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("#") or "kv_offload" not in line:
                    continue
                for k in want:
                    if k in line:
                        try:
                            want[k] += float(line.rsplit(" ", 1)[-1])
                        except ValueError:
                            pass
    except Exception:
        pass
    return want


def calibrate_ratio(endpoint, model):
    """实测本文本形态下的 token/字符比（缺省 0.5299 是中文标定值，
    探针用的是英文词表，实测约 1.6 —— 不标定就会把文档做长 3 倍）。"""
    filler = " ".join(WORDS * 40)[:4000]
    j, _ = post(endpoint + "/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": filler + "\n\nReply OK."}],
        "max_tokens": 1, "temperature": 0.0, "stream": False})
    pt = (j.get("usage") or {}).get("prompt_tokens") or 0
    n = len(filler) + len("\n\nReply OK.")
    if pt <= 0 or n <= 0:
        return None
    return pt / n


def make_doc(idx, tokens, ratio, rng):
    """构造约 tokens 个 token 的自然文本，内嵌唯一验证码。"""
    code = "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(7))
    n = "CODE-%s-%04d" % (code, idx)
    body = []
    for i in range(int(tokens / ratio / 12) + 8):
        body.append(" ".join(rng.choices(WORDS, k=12)))
        if i == len(body) // 3:
            body.append("THE VERIFICATION TOKEN IS %s END OF TOKEN." % n)
    return n, " ".join(body)


def ask(endpoint, model, doc, question, max_tokens=64):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": doc + "\n\n" + question}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    j, el = post(endpoint + "/chat/completions", payload)
    ch = (j.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    text = " ".join(filter(None, [msg.get("content") or "",
                                  msg.get("reasoning_content") or ""]))
    usage = j.get("usage") or {}
    det = (usage.get("prompt_tokens_details") or {})
    return {"elapsed": round(el, 2), "text": text.strip(),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": det.get("cached_tokens")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:18420/v1")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--tokens", type=int, default=100000, help="每篇文档目标 token")
    ap.add_argument("--ratio", type=float, default=0.0,
                    help="token/字符；0=自动标定（本探针用英文词表，约 1.6）")
    ap.add_argument("--docs", type=int, default=2, help="建档文档数")
    ap.add_argument("--flush", type=int, default=11,
                    help="挤池文档数：总 token 必须 > GPU KV 池（本机 120.7 万）才会把建档"
                         "文档从显存挤出去；同时 建档+挤池 总量必须 < CPU 档容量"
                         "（48 GiB≈160 万 tok），否则 CPU 侧也被 LRU 掉 = 必 miss")
    ap.add_argument("--gap", type=float, default=5.0, help="请求间隔秒")
    args = ap.parse_args()

    rng = random.Random(20260927)
    out = {"endpoint": args.endpoint, "checks": {}, "detail": []}

    if args.ratio <= 0:
        got = calibrate_ratio(args.endpoint, args.model)
        args.ratio = got or 1.6
    out["ratio_tok_per_char"] = round(args.ratio, 4)

    def snap():
        return metrics(args.endpoint)

    m0 = snap()
    b0 = offload_bytes_labeled(args.endpoint)

    # 1) 建档：发 docs 篇长文档（触发 GPU→CPU store）
    docs = []
    for i in range(args.docs):
        code, doc = make_doc(i, args.tokens, args.ratio, rng)
        q = "What is the verification token in the document above? Answer with the token only."
        r = ask(args.endpoint, args.model, doc, q)
        docs.append((code, doc, q, r))
        out["detail"].append({"phase": "build", "i": i, "code": code, "answer": r["text"][:120],
                              "prompt_tokens": r["prompt_tokens"],
                              "cached_tokens": r["cached_tokens"],
                              "elapsed": r["elapsed"]})
        time.sleep(args.gap)

    # 2) 挤池：发 flush 篇互不相同的长文档，把 GPU KV 池占满、把建档请求的块挤出去
    for i in range(args.flush):
        _, doc = make_doc(1000 + i, args.tokens, args.ratio, rng)
        r = ask(args.endpoint, args.model, doc,
                "Reply with the single word OK.", max_tokens=8)
        out["detail"].append({"phase": "flush", "i": i,
                              "prompt_tokens": r["prompt_tokens"],
                              "cached_tokens": r["cached_tokens"],
                              "elapsed": r["elapsed"]})
        time.sleep(args.gap)

    # 挤池后再等一会儿：二级缓存 store 是异步的，且历史实测块提交有延迟
    time.sleep(max(10.0, args.gap))

    # 3) 重发第一篇：期望 CPU 档命中（external hits + CPU_to_GPU 字节 + 内容无损）
    code, doc, q, _ = docs[0]
    m1 = snap()
    b1 = offload_bytes_labeled(args.endpoint)
    r = ask(args.endpoint, args.model, doc, q, max_tokens=256)
    m2 = snap()
    b2 = offload_bytes_labeled(args.endpoint)
    out["detail"].append({"phase": "reload", "code": code,
                          "prompt_tokens": r["prompt_tokens"],
                          "cached_tokens": r["cached_tokens"],
                          "elapsed": r["elapsed"], "text": r["text"][:200]})

    d_ext = m2.get("vllm:external_prefix_cache_hits_total", 0.0) - \
        m1.get("vllm:external_prefix_cache_hits_total", 0.0)
    d_load = b2["CPU_to_GPU"] - b1["CPU_to_GPU"]
    d_store = b2["GPU_to_CPU"] - b1["GPU_to_CPU"]
    secret = code.split("-")[1] if "-" in code else code
    recalled = (code in r["text"]) or (secret in r["text"])

    out["checks"]["external_hit_tokens"] = d_ext
    out["checks"]["cpu_to_gpu_bytes"] = d_load
    out["checks"]["gpu_to_cpu_bytes_during_reload"] = d_store
    out["checks"]["recalled_verification_code"] = bool(recalled)
    out["checks"]["cumulative"] = {
        "external_hits_total": m2.get("vllm:external_prefix_cache_hits_total"),
        "external_queries_total": m2.get("vllm:external_prefix_cache_queries_total"),
        "prefix_hits_total": m2.get("vllm:prefix_cache_hits_total"),
        "cpu_to_gpu_total": b2["CPU_to_GPU"],
        "gpu_to_cpu_total": b2["GPU_to_CPU"],
        "cpu_cache_usage_perc": m2.get("vllm:kv_offload_cpu_cache_usage_perc"),
        "cpu_cache_fill_perc": m2.get("vllm:kv_offload_cpu_cache_fill_perc"),
    }
    out["verdict"] = bool(d_ext > 0 and d_load > 0 and recalled)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["verdict"] else 1


if __name__ == "__main__":
    sys.exit(main())
