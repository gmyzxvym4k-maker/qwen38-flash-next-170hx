#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kvoff-tail —— 控制实验：同一段长前缀、**不同的尾部提问**，第二次能否走内存档命中？

背景：recall2 探针里「建档」与「重发」用的是同一份文档但**尾部提问不同**，
      实测两侧的首块 OffloadKey 哈希竟然不同（07b78147 vs 6121b263），
      而 kvoff-dbg（两侧提问都是 "Reply OK."）的哈希完全一致并能命中。
      本实验就是判定：尾部不同会不会改变前缀块的 key。

序列：P1 = doc + Q1 → 挤池大文档 ×N → P2 = doc + Q2（Q2 更长/更短都试）
输出每一步的响应 id，配合引擎日志 dbg[lookup]/dbg[store] 的 key 对比。
"""
import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango uniform victor "
         "whiskey xray yankee zulu").split()


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


def ask(base, model, doc, question, max_tokens=32):
    payload = {"model": model,
               "messages": [{"role": "user", "content": doc + "\n\n" + question}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
               "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    j = post(base + "/chat/completions", payload)
    el = round(time.time() - t0, 1)
    if "__http_error__" in j or "__error__" in j:
        return {"id": None, "error": j, "elapsed_s": el}
    usage = j.get("usage") or {}
    det = usage.get("prompt_tokens_details") or {}
    return {"id": j.get("id"), "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": det.get("cached_tokens"), "elapsed_s": el}


def make_doc(salt, words):
    rng = random.Random(hash(salt) & 0xFFFFFFFF)
    parts = ["SALT-%s ." % salt]
    ln = 0
    while ln < words:
        parts.append(" ".join(rng.choices(WORDS, k=14)))
        ln += 15
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:18420/v1")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--doc-tokens", type=int, default=20000)
    ap.add_argument("--flush-tokens", type=int, default=36000)
    ap.add_argument("--flushes", type=int, default=2)
    ap.add_argument("--gap", type=float, default=5.0)
    a = ap.parse_args()
    salt = "TL%d" % int(time.time())

    r0 = ask(a.endpoint, a.model, " ".join(random.Random(7).choices(WORDS, k=1200)),
             "Reply OK.", 8)
    tpw = (r0.get("prompt_tokens") or 1800) / 1200.0
    print("[tail] 标定 %.3f tok/word salt=%s" % (tpw, salt), file=sys.stderr)

    def doc(tag, tokens):
        return make_doc("%s-%s" % (salt, tag), int(tokens / tpw))

    Q1 = "Reply with the single word OK."
    Q2 = ("Repeat the verification code that appears in the document above, "
          "exactly as written. Answer with the code only.")
    dA = doc("A", a.doc_tokens)

    out = {"salt": salt, "steps": []}
    r1 = ask(a.endpoint, a.model, dA, Q1)
    out["steps"].append({"phase": "P1-doc+Q1", **r1})
    print("[tail] P1 id=%s prompt=%s cached=%s" % (r1.get("id"), r1.get("prompt_tokens"),
                                                   r1.get("cached_tokens")), file=sys.stderr)
    time.sleep(2)
    for i in range(a.flushes):
        r = ask(a.endpoint, a.model, doc("B%d" % i, a.flush_tokens), Q1)
        out["steps"].append({"phase": "flush-%d" % i, **r})
        print("[tail] flush%d id=%s prompt=%s" % (i, r.get("id"), r.get("prompt_tokens")),
              file=sys.stderr)
        time.sleep(2)
    time.sleep(a.gap)

    r2 = ask(a.endpoint, a.model, dA, Q2)
    out["steps"].append({"phase": "P2-doc+Q2", **r2})
    print("[tail] P2 id=%s prompt=%s cached=%s (%ss)"
          % (r2.get("id"), r2.get("prompt_tokens"), r2.get("cached_tokens"),
             r2.get("elapsed_s")), file=sys.stderr)
    time.sleep(a.gap)
    r3 = ask(a.endpoint, a.model, dA, Q1)
    out["steps"].append({"phase": "P3-doc+Q1-again", **r3})
    print("[tail] P3 id=%s prompt=%s cached=%s (%ss)"
          % (r3.get("id"), r3.get("prompt_tokens"), r3.get("cached_tokens"),
             r3.get("elapsed_s")), file=sys.stderr)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
