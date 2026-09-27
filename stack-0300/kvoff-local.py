#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kvoff-local —— 对照：同一 prompt 连发两次（第二次走**本地**前缀缓存），检验本地缓存路径是否无损。
用来把「回载乱码」的锅分给 CPU 档还是上游本地前缀缓存。"""
import json, random, sys, time, urllib.request
from importlib.machinery import SourceFileLoader
N = SourceFileLoader("kvne", "/home/ll/deploy/kvoff-needle.py").load_module()
EP, MODEL = "http://127.0.0.1:18420/v1", "qwen3.8-flash-next"
Q = ("What is the verification code in the document above? Reply with just the code, nothing else.")

def ask(doc):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": doc + "\n\n" + Q}],
               "max_tokens": 128, "temperature": 0.0, "stream": False,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(EP + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.loads(r.read().decode())
    ch = (j.get("choices") or [{}])[0]; msg = ch.get("message") or {}; u = j.get("usage") or {}
    return {"prompt": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "finish": ch.get("finish_reason"),
            "text": ((msg.get("content") or "") + (msg.get("reasoning") or ""))[:120]}

salt = int(time.time()); code = "CODE-LC%03d-%04d" % (salt % 1000, salt % 9973)
doc = N.build_doc(random.Random(salt), 2200, N.NEEDLE.format(code=code))
print(json.dumps({"code": code, "step": "1-fresh", **ask(doc)}, ensure_ascii=False))
time.sleep(2)
print(json.dumps({"step": "2-local-reuse", **ask(doc)}, ensure_ascii=False))
time.sleep(2)
print(json.dumps({"step": "3-local-reuse-again", **ask(doc)}, ensure_ascii=False))
