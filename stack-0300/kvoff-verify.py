#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kvoff-verify —— 内存档回载「内容正确性」判定（同一 prompt 的本地算 vs 回载对比）。

步骤（温度 0，同一提示词）：
  1) 全新 docX + 验证码提问            → answer0（本地前向，正确基准）
  2) 两篇 60k 挤池文档                  → 把 docX 挤出显存（同时 store 进内存档）
  3) docX + 同问题（正常）              → answer1（走 CPU→GPU 回载）
  4) docX + 同问题 + kv_load_tiers:[]   → answer2（显式禁用外部加载，本地重算）
判据：answer1 == answer0 ⇒ 回载无损；answer1 乱码而 answer2 == answer0 ⇒ 回载数据错。
"""
import json, random, sys, time, urllib.request
from importlib.machinery import SourceFileLoader
N = SourceFileLoader("kvne", "/home/ll/deploy/kvoff-needle.py").load_module()

EP, MODEL = "http://127.0.0.1:18420/v1", "qwen3.8-flash-next"
Q = ("What is the verification code in the document above? "
     "Reply with just the code, nothing else.")


def ask_tiers(doc, question, tiers=None, mt=128):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": doc + "\n\n" + question}],
               "max_tokens": mt, "temperature": 0.0, "stream": False,
               "chat_template_kwargs": {"enable_thinking": False}}
    if tiers is not None:
        payload["kv_transfer_params"] = {"kv_load_tiers": tiers}
    req = urllib.request.Request(EP + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.loads(r.read().decode())
    ch = (j.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    u = j.get("usage") or {}
    return {"id": j.get("id"), "prompt": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "finish": ch.get("finish_reason"),
            "text": ((msg.get("content") or "") + (msg.get("reasoning") or ""))[:160]}


salt = int(time.time())
code = "CODE-VF%03d-%04d" % (salt % 1000, salt % 9973)
docX = N.build_doc(random.Random(salt), 2200, N.NEEDLE.format(code=code))
print(json.dumps({"code": code, "step": "0-fresh", **ask_tiers(docX, Q)}, ensure_ascii=False))

for i in range(2):
    d = N.build_doc(random.Random(salt + 11 + i), 4200)
    r = N.ask(EP, MODEL, d, "Reply with the single word OK.", 8)
    print(json.dumps({"step": "flush-%d" % i, "prompt": r.get("prompt_tokens")}, ensure_ascii=False))
    time.sleep(3)

time.sleep(6)
print(json.dumps({"step": "1-reload(cpu)", **ask_tiers(docX, Q)}, ensure_ascii=False))
time.sleep(3)
print(json.dumps({"step": "2-reload(no-external)", **ask_tiers(docX, Q, tiers=[])}, ensure_ascii=False))
