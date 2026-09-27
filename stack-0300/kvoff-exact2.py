#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kvoff-exact2 —— 自然词填充，把 prompt 精确补到 `1616*N + 1`。

目的：让**回载覆盖 99.98% 的 prompt**（只留 1 个 token 重算），从而把
「整体装载是否错」与「装载/重算交界是否错」分开：
  · 若这种 prompt 回载后答案正确 ⇒ 装载本身没问题，问题在交界/记账；
  · 若仍乱码 ⇒ 装载进来的 KV 整体就是错的。

与 v1（kvoff-exact.py）的区别：v1 用 `" x" * 1382` 填充，模型自己就退化成复读
（连"全新请求"那一问都是乱码），实验被污染；这里用**随机单词**填充（与
kvoff-scale 建档文档同款，实测模型能正常作答），并逐轮测量收敛到精确长度。
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
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
         "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
         "xray yankee zulu system value record block memory cache kernel tensor").split()


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
    nblocks = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    target = BLOCK * nblocks + 1
    salt = int(time.time())
    code = "CODE-EX2-%04d" % (salt % 9973)
    rng = random.Random(salt)

    base = N.build_doc(rng, max(80, int(target / 40)), N.NEEDLE.format(code=code))
    pad = []
    measured = None
    for it in range(10):
        doc = base + (" " + " ".join(pad) if pad else "")
        r = ask(doc, QCODE, 4)
        if "err" in r:
            print(json.dumps({"step": "measure", "iter": it, "err": r["err"]}), flush=True)
            return 2
        measured = r["prompt"]
        print(json.dumps({"step": "measure", "iter": it, "pad": len(pad),
                          "prompt": measured, "target": target}), flush=True)
        if measured == target:
            break
        delta = target - measured
        if delta > 0:
            pad.extend(rng.choices(WORDS, k=delta))
        else:
            pad = pad[:max(0, len(pad) + delta)]
    if abs(measured - target) > 16:
        print(json.dumps({"step": "give-up", "measured": measured, "target": target}))
        return 3
    # 允许 ±16 token 误差：只要 prompt 落在 [boundary+1, boundary+16]，
    # 回载就覆盖 ≥99.75%，重算尾部 ≤16 token —— 实验目的已达成。
    print(json.dumps({"step": "converged", "measured": measured, "target": target,
                      "tail_tokens": measured - (target - 1)}), flush=True)
    final = base + (" " + " ".join(pad) if pad else "")
    print(json.dumps({"step": "build", "prompt": measured,
                      "text": (r.get("text") or "")[:60], "code": code}), flush=True)

    for i in range(2):
        d = N.build_doc(random.Random(salt + 100 + i), 4200)
        N.ask(EP, MODEL, d, QOK, 8)
        time.sleep(2)
    time.sleep(6)
    c1 = counters()
    r1 = ask(final, QCODE, 64)
    c2 = counters()
    rec = code in (r1.get("text") or "")
    print(json.dumps({"step": "reload", "prompt": r1.get("prompt"),
                      "cached": r1.get("cached"), "cached_pct":
                      round(100.0 * (r1.get("cached") or 0) / max(1, r1.get("prompt") or 1), 2),
                      "finish": r1.get("finish"), "text": (r1.get("text") or "")[:60],
                      "recalled": rec,
                      "ext_hits_delta": round(c2.get("ext_hits", 0) - c1.get("ext_hits", 0), 1),
                      "cpu_to_gpu_delta": round(c2.get("cpu_to_gpu", 0) - c1.get("cpu_to_gpu", 0), 1),
                      "code": code}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
