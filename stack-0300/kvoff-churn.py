#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kvoff-churn —— 生产档（大 GPU 池 ~1.2M token）下的内存二级缓存验收探针。

生产池太大，普通小探针的挤池量（~120k）不足以把文档挤出显存，命中会落在**本地**前缀
缓存上、根本走不到 CPU 档。本探针用 N 篇 ~120k token 的唯一文档把池子搅翻（>池容），
再用 external_prefix_cache_hits_total 增量判定"这次到底有没有走内存档"，
最后用验证码复述判定回载内容是否正确。

用法：python3 kvoff-churn.py [--flushes 12] [--a-sentences 2000] [--b-sentences 8000]

2026-09-27 加固（针对上一轮「hits=0 但答案正确」这种无法定论的窗口）：
  ① 累计挤池 token 数与 GPU 池容量对比，输出 tier_exercised 字段——若挤池量 < 池容，
     则「hits=0」是构造性必然、该窗口对二级缓存零信息量，脚本会显式说出来；
  ② 复述预算从 64 提到 --answer-tokens（缺省 256）且在「正文为空/被预算截断」时自动
     加预算重试一次，避免思考模型把预算吃光导致假阴性（历史踩坑：recalled=false 但 text 为空）；
  ③ 记录 reload 的 prompt_tokens 与建档时是否一致（不一致=前缀都不同，判据无效）。
"""
import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from importlib.machinery import SourceFileLoader

N = SourceFileLoader("kvne", "/home/ll/deploy/kvoff-needle.py").load_module()
EP, MODEL = "http://127.0.0.1:18420/v1", "qwen3.8-flash-next"
QOK = "Reply with the single word OK."
QCODE = ("What is the verification code in the document above? "
         "Reply with just the code, nothing else.")


def counters():
    try:
        with urllib.request.urlopen("http://127.0.0.1:18420/metrics", timeout=20) as r:
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
        if name == "vllm:external_prefix_cache_hits_total":
            out["ext_hits"] = out.get("ext_hits", 0.0) + v
    return out


def reload_probe(doc, code, budget):
    """复述验证码；正文为空或被预算截断时加预算重试一次，返回 (响应, 是否命中, 尝试次数)。"""
    r, tries, ok = None, 0, False
    for tok in (budget, max(budget * 2, 512)):
        tries += 1
        r = N.ask(EP, MODEL, doc, QCODE, tok)
        text = r.get("text") or ""
        if code in text:
            ok = True
            break
        if text.strip() and r.get("finish_reason") != "length":
            break          # 有正文且不是被预算截断 → 如实判负，不重试
    return r, ok, tries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flushes", type=int, default=12)
    ap.add_argument("--a-sentences", type=int, default=2000)
    ap.add_argument("--b-sentences", type=int, default=8000)
    ap.add_argument("--answer-tokens", type=int, default=256)
    ap.add_argument("--gpu-pool-tokens", type=int, default=1207262,
                    help="GPU KV 池 token（日志 GPU KV cache size），用于判断挤池量是否够")
    a = ap.parse_args()
    salt = int(time.time())
    code = "CODE-CH%04d" % (salt % 9973)
    doc = N.build_doc(random.Random(salt), a.a_sentences, N.NEEDLE.format(code=code))

    r0 = N.ask(EP, MODEL, doc, QCODE, 64)
    print(json.dumps({"step": "build", "prompt": r0.get("prompt_tokens"),
                      "text": (r0.get("text") or "")[:40], "code": code}), flush=True)

    flushed = 0
    for i in range(a.flushes):
        d = N.build_doc(random.Random(salt + 100 + i), a.b_sentences)
        rr = N.ask(EP, MODEL, d, QOK, 8)
        flushed += int(rr.get("prompt_tokens") or 0)
        print(json.dumps({"step": "flush-%d" % i, "prompt": rr.get("prompt_tokens"),
                          "flushed_total": flushed}), flush=True)
        time.sleep(1)

    time.sleep(6)
    c1 = counters()
    r1, rec, tries = reload_probe(doc, code, a.answer_tokens)
    c2 = counters()
    d_hits = c2.get("ext_hits", 0.0) - c1.get("ext_hits", 0.0)
    exercised = flushed >= a.gpu_pool_tokens
    same_prefix = (r1.get("prompt_tokens") == r0.get("prompt_tokens"))
    out = {
        "step": "reload", "prompt": r1.get("prompt_tokens"),
        "cached": r1.get("cached_tokens"), "finish": r1.get("finish_reason"),
        "text": (r1.get("text") or "")[:60], "recalled": rec,
        "answer_tries": tries,
        "external_hits_delta": d_hits,
        "flushed_tokens": flushed, "gpu_pool_tokens": a.gpu_pool_tokens,
        "tier_exercised": exercised,
        "same_prefix_as_build": same_prefix,
        "verdict": bool(d_hits > 0 and rec),
        "code": code,
    }
    print(json.dumps(out, ensure_ascii=False), flush=True)
    if not exercised:
        print(json.dumps({"step": "warning",
                          "msg": "挤池 %d < GPU 池 %d：本次窗口挤不动显存，hits=0 是构造性必然，"
                                 "对二级缓存无信息量（请加大 --flushes）"
                                 % (flushed, a.gpu_pool_tokens)}, ensure_ascii=False), flush=True)
    if d_hits > 0 and not rec:
        print(json.dumps({"step": "warning",
                          "msg": "走通了内存档（external hits > 0）但复述不正确：回载内容与本地不等价"},
                         ensure_ascii=False), flush=True)
    if not same_prefix:
        print(json.dumps({"step": "warning",
                          "msg": "reload prompt_tokens(%s) != build(%s)：前缀不同，判据无效"
                                 % (r1.get("prompt_tokens"), r0.get("prompt_tokens"))},
                         ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
