#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kvoff-scale —— 定标：**回载多少 token 时输出开始失真**。

对每个尺寸 s：建档（验证码埋在文档最前面）→ 3 篇 36k 挤池（把该文档挤出显存并触发 store）
→ 重发同一 prompt 问验证码。逐档记录 prompt / 外部命中 / 答案是否精准。
判读：
  · 小尺寸对、大尺寸错 ⇒ 与「非前缀可缓存的组状态」有关（QSA 环形组 / mamba 状态），
    破位点就是线索（如 8 块 × 1616 = 12,928 token）。
  · 所有尺寸都错 ⇒ 回载基本路径就有问题。
"""
import json
import random
import sys
import time
import urllib.error
import urllib.request
from importlib.machinery import SourceFileLoader

N = SourceFileLoader("kvne", "/home/ll/deploy/kvoff-needle.py").load_module()
EP, MODEL = "http://127.0.0.1:18420/v1", "qwen3.8-flash-next"
Q = ("What is the verification code in the document above? Reply with just the code, nothing else.")


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


def ask(doc, question, mt=64):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": doc + "\n\n" + question}],
               "max_tokens": mt, "temperature": 0.0, "stream": False,
               "chat_template_kwargs": {"enable_thinking": False}}
    j = post(EP + "/chat/completions", payload)
    if "__http_error__" in j or "__error__" in j:
        return {"err": j}
    ch = (j.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    u = j.get("usage") or {}
    return {"prompt": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "finish": ch.get("finish_reason"),
            "text": ((msg.get("content") or "") + (msg.get("reasoning") or ""))[:80]}


def counters():
    try:
        with urllib.request.urlopen("http://127.0.0.1:18420/metrics", timeout=15) as r:
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
            out["ext_hits"] = v
        elif name == "vllm:kv_offload_total_bytes_total" and 'CPU_to_GPU"' in ln:
            out["cpu_to_gpu"] = out.get("cpu_to_gpu", 0.0) + v
    return out


sizes = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["2000", "6000", "12000", "20000"])]
salt = int(time.time())
for s in sizes:
    code = "CODE-SC%02d-%04d" % (s // 1000, salt % 9973)
    # 句子数 ≈ s/26（每句 ~26 token），验证码埋在最前面
    doc = N.build_doc(random.Random(salt + s), max(60, int(s / 26)), N.NEEDLE.format(code=code))
    r0 = ask(doc, Q, 64)
    for i in range(3):
        d = N.build_doc(random.Random(salt + 100 + i), 4200)
        N.ask(EP, MODEL, d, "Reply with the single word OK.", 8)
        time.sleep(2)
    time.sleep(6)
    c1 = counters()
    r1 = ask(doc, Q, 64)
    c2 = counters()
    rec = code in (r1.get("text") or "")
    print(json.dumps({"size_req": s, "code": code,
                      "fresh": {"prompt": r0.get("prompt"), "text": (r0.get("text") or "")[:40]},
                      "reload": {"prompt": r1.get("prompt"), "cached": r1.get("cached"),
                                 "finish": r1.get("finish"), "text": (r1.get("text") or "")[:60]},
                      "ext_hits_delta": round(c2.get("ext_hits", 0) - c1.get("ext_hits", 0), 1),
                      "cpu_to_gpu_delta": round(c2.get("cpu_to_gpu", 0) - c1.get("cpu_to_gpu", 0), 1),
                      "recalled": rec}, ensure_ascii=False))
    time.sleep(3)
