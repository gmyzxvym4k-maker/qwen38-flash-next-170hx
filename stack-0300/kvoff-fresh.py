#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kvoff-fresh —— 对照：**全新**（无任何缓存命中）的自然文档 + 验证码提问。
用来判定「回答退化成 Register 复读」是模型/提示词问题还是缓存回载问题。"""
import json, random, sys, time, urllib.request
sys.path.insert(0, "/home/ll/deploy")
from importlib.machinery import SourceFileLoader
N = SourceFileLoader("kvne", "/home/ll/deploy/kvoff-needle.py").load_module()

salt = int(time.time())
code = "CODE-FR%03d-%04d" % (salt % 1000, salt % 9973)
doc = N.build_doc(random.Random(salt), 2200, N.NEEDLE.format(code=code))
for q, mt in (("What is the verification code in the document above? Reply with just the code, nothing else.", 256),
              ("Summarise the document above in one short sentence.", 128)):
    r = N.ask("http://127.0.0.1:18420/v1", "qwen3.8-flash-next", doc, q, mt)
    print(json.dumps({"code": code, "q": q[:40], "prompt": r.get("prompt_tokens"),
                      "cached": r.get("cached_tokens"), "finish": r.get("finish_reason"),
                      "tokens": r.get("completion_tokens"), "text": (r.get("text") or "")[:160]},
                     ensure_ascii=False))
    doc = N.build_doc(random.Random(salt + 1), 2200, N.NEEDLE.format(code=code))
