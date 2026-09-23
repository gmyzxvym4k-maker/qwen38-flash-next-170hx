#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3.8-Flash-Next 基准测试：TTFT / decode tok/s / 稳定性

测量口径（两条都是踩坑换来的，见 docs/07-performance.md §5）：
  · 本实例没有 /v1/tokenize，prompt 长度靠 UNIT × PER_UNIT_TOKENS 近似标定；
  · 这是思考模型，流式首个增量可能在 delta.reasoning_content 而非 content，
    TTFT 必须按"任意第一个增量"算（本脚本已如此处理）；
  · decode tok/s = (completion_tokens - 1) / (结束 - 首 token)，按 SSE chunk 计数会因
    MTP 一步多 token 而严重低估；
  · 默认每轮换盐，避免前缀缓存把吞吐测成缓存速度；要测缓存加 --same。

用法示例：
  python3 fnx-bench.py --base http://127.0.0.1:18420 --ctx 8192 --out 256 --repeat 2
  python3 fnx-bench.py --base http://127.0.0.1:18420 --ctx 131072 --out 128 --timeout 1800
  python3 fnx-bench.py --base http://127.0.0.1:18420 --ctx 8192 --same   # 复用同一 prompt 测前缀缓存

注意：decode tok/s 的正确口径 = (completion_tokens - 1) / (结束时刻 - 首 token 时刻)。
按 SSE chunk 计数会因 MTP 一步多 token 而严重低估。
"""
import argparse
import json
import random
import string
import sys
import time
import urllib.request

UNIT = "在模型推理部署的实践中，我们需要持续关注显存占用、批处理规模、上下文长度与吞吐之间的平衡关系，并结合实测数据做出取舍。"

# UNIT 的近似 token 数（中文约 1.5~1.7 字符/token，本串约 62 字）
PER_UNIT_TOKENS = 40


def build_prompt(approx_tokens, salt):
    reps = max(1, int(round(approx_tokens / float(PER_UNIT_TOKENS))))
    head = "【批次标识 %s】" % salt
    return head + UNIT * reps


def stream_chat(base, model, prompt, max_tokens, timeout, temperature):
    url = base.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    t0 = time.time()
    t_first = None
    t_last = None
    n_chunks = 0
    usage = None

    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except Exception as e:
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}

    try:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    n_chunks += 1
    finally:
        try:
            resp.close()
        except Exception:
            pass

    t_end = time.time()
    total = t_end - t0
    ttft = (t_first - t0) if t_first else None
    ct = (usage or {}).get("completion_tokens")
    pt = (usage or {}).get("prompt_tokens")
    decode_s = (t_end - t_first) if t_first else None
    decode_tps = None
    if ct and decode_s and decode_s > 0:
        decode_tps = (ct - 1) / decode_s

    return {
        "ok": True,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "total_s": round(total, 3),
        "decode_s": round(decode_s, 3) if decode_s is not None else None,
        "decode_tps": round(decode_tps, 2) if decode_tps else None,
        "sse_chunks": n_chunks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18420")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--ctx", default="8192", help="目标 prompt token 数，逗号分隔多档")
    ap.add_argument("--out", type=int, default=256, help="max_tokens")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--same", action="store_true", help="各轮复用同一 prompt（测前缀缓存）")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    ctxs = [int(x) for x in str(args.ctx).split(",") if x.strip()]
    salt_base = args.tag or "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))

    print("== fnx-bench tag=%s base=%s model=%s out=%d repeat=%d same=%s ==" % (
        salt_base, args.base, args.model, args.out, args.repeat, args.same))
    sys.stdout.flush()

    rows = []
    for ctx in ctxs:
        salt = salt_base if args.same else salt_base
        prompt = build_prompt(ctx, salt)
        for i in range(args.repeat):
            tag = "ctx%-7d r%d" % (ctx, i + 1)
            if not args.same and i > 0:
                # 换盐：保证前缀不同，测真实冷 TTFT
                prompt = build_prompt(ctx, salt_base + "-r%d" % (i + 1))
            r = stream_chat(args.base, args.model, prompt, args.out, args.timeout, args.temperature)
            r["ctx_target"] = ctx
            r["round"] = i + 1
            rows.append(r)
            if not r.get("ok"):
                print("%s  FAILED  %s" % (tag, r.get("error")))
            else:
                print("%s  prompt=%-7s out=%-5s TTFT=%-8s total=%-8s decode=%-7s tok/s  chunks=%s" % (
                    tag, r.get("prompt_tokens"), r.get("completion_tokens"),
                    r.get("ttft_s"), r.get("total_s"), r.get("decode_tps"), r.get("sse_chunks")))
            sys.stdout.flush()

    print("== SUMMARY(JSON) ==")
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
