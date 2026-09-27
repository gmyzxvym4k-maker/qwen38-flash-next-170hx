#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kvoff-exact —— 在「长度精确等于 N×1616+1」的 prompt 上验证回载正确性。

动机：mamba(GDN) 组的状态是**按 1616 边界 checkpoint** 的，而 OffloadingConnector 存/取
的 mamba 状态必须与装载的注意力前缀**同边界**。用一个 token 数恰好落在边界上的 prompt，
可以把「状态比 KV 靠前/靠后」这类语义错位排除掉：
  · 若这种 prompt 回载后答案正确 ⇒ 语义没错，之前的乱码来自「状态与装载前缀不同边界」；
  · 若仍乱码 ⇒ 状态/装配链路本身有问题（与边界无关）。

做法：把 doc + 填充词 + 问题 拼到目标 token 数（填充词插在问题之前，保证首块内容不变），
逐轮测量微调；然后 建档 → 两篇 60k 挤池 → 重发问验证码。
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
QOK = "Reply with the single word OK."
QCODE = ("What is the verification code in the document above? "
         "Reply with just the code, nothing else.")
BLOCK = 1616


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
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": doc + "\n\n" + question}],
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


def main():
    nblocks = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    target = BLOCK * nblocks + 1
    salt = int(time.time())
    code = "CODE-EX%d-%04d" % (nblocks, salt % 9973)

    # 句子数按目标长度估（每句 ~26 token），验证码埋在最前面
    doc = N.build_doc(random.Random(salt), max(60, int(target / 26)),
                      N.NEEDLE.format(code=code))
    pad = 0
    measured = None
    for it in range(8):
        r = ask(doc + (" x" * pad), QOK, 8)
        if "err" in r:
            print(json.dumps({"step": "measure", "iter": it, "err": r["err"]}))
            return 2
        measured = r["prompt"]
        print(json.dumps({"step": "measure", "iter": it, "pad": pad,
                          "prompt": measured, "target": target}), flush=True)
        if measured == target:
            break
        pad += target - measured
        pad = max(0, pad)
    if measured != target:
        print(json.dumps({"step": "give-up", "measured": measured, "target": target}))
        return 3
    padded = doc + (" x" * pad)

    r0 = ask(padded, QCODE, 64)
    print(json.dumps({"step": "build", "prompt": r0.get("prompt"),
                      "text": (r0.get("text") or "")[:40]}), flush=True)
    for i in range(2):
        d = N.build_doc(random.Random(salt + 100 + i), 4200)
        N.ask(EP, MODEL, d, QOK, 8)
        time.sleep(2)
    time.sleep(6)
    c1 = counters()
    r1 = ask(padded, QCODE, 64)
    c2 = counters()
    rec = code in (r1.get("text") or "")
    print(json.dumps({"step": "reload", "prompt": r1.get("prompt"),
                      "cached": r1.get("cached"), "finish": r1.get("finish"),
                      "text": (r1.get("text") or "")[:60], "recalled": rec,
                      "ext_hits_delta": round(c2.get("ext_hits", 0) - c1.get("ext_hits", 0), 1),
                      "cpu_to_gpu_delta": round(c2.get("cpu_to_gpu", 0) - c1.get("cpu_to_gpu", 0), 1),
                      "code": code}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
