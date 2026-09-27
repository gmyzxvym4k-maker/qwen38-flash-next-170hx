#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kvoff-dbg —— 定向 key 诊断探针：建档 A(唯一) → 挤池 B/C(唯一，大) → 重发 A。

与 recall2 的区别：**只求把每步的响应 id 打出来**，好把引擎日志里的
`dbg[store] req=<id>` / `dbg[lookup] req=<id>` 逐条捞出来对账——
「存了哪些 key、查了哪些 key、结果如何」是区分「key 不一致」与「policy 里没有」的唯一办法。

用法：
  python3 kvoff-dbg.py --a-tokens 8000 --b-tokens 40000 --gap 5
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
        body = ""
        try:
            body = exc.read().decode()[:300]
        except Exception:
            pass
        return {"__http_error__": exc.code, "__body__": body}
    except Exception as exc:
        return {"__error__": "%s: %s" % (type(exc).__name__, exc)}


def ask(base, model, doc, question, max_tokens=64, no_think=True):
    payload = {"model": model,
               "messages": [{"role": "user", "content": doc + "\n\n" + question}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.time()
    j = post(base + "/chat/completions", payload)
    el = round(time.time() - t0, 1)
    if "__http_error__" in j or "__error__" in j:
        return {"id": None, "error": j, "elapsed_s": el}
    ch = (j.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    usage = j.get("usage") or {}
    det = usage.get("prompt_tokens_details") or {}
    return {"id": j.get("id"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": det.get("cached_tokens"),
            "text": ((msg.get("content") or "") + (msg.get("reasoning") or ""))[:80],
            "finish_reason": ch.get("finish_reason"),
            "elapsed_s": el}


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
    ap.add_argument("--a-tokens", type=int, default=8000)
    ap.add_argument("--b-tokens", type=int, default=40000)
    ap.add_argument("--flushes", type=int, default=2)
    ap.add_argument("--gap", type=float, default=5.0)
    ap.add_argument("--max-prompt-tokens", type=int, default=45000)
    a = ap.parse_args()
    salt = "T%d" % int(time.time())

    # 标定 tokens/word
    probe = " ".join(random.Random(7).choices(WORDS, k=1200))
    r0 = ask(a.endpoint, a.model, probe, "Reply OK.", 8)
    if not r0.get("prompt_tokens"):
        print("[dbg] 标定失败：%r" % (r0,), file=sys.stderr)
        return 2
    tpw = r0["prompt_tokens"] / 1200.0
    print("[dbg] 标定 %.3f tok/word（salt=%s）" % (tpw, salt), file=sys.stderr)

    def doc(tag, tokens):
        return make_doc("%s-%s" % (salt, tag), int(tokens / tpw))

    steps = []
    docA = doc("A", a.a_tokens)
    r = ask(a.endpoint, a.model, docA, "Reply OK.")
    steps.append({"phase": "build-A", **{k: r.get(k) for k in
                 ("id", "prompt_tokens", "cached_tokens", "elapsed_s")}})
    print("[dbg] build A id=%s prompt=%s cached=%s (%ss)"
          % (r.get("id"), r.get("prompt_tokens"), r.get("cached_tokens"), r.get("elapsed_s")),
          file=sys.stderr)
    time.sleep(2)

    for i in range(a.flushes):
        r = ask(a.endpoint, a.model, doc("B%d" % i, a.b_tokens), "Reply OK.")
        steps.append({"phase": "flush-%d" % i, **{k: r.get(k) for k in
                     ("id", "prompt_tokens", "cached_tokens", "elapsed_s")}})
        print("[dbg] flush%d id=%s prompt=%s cached=%s (%ss)"
              % (i, r.get("id"), r.get("prompt_tokens"), r.get("cached_tokens"),
                 r.get("elapsed_s")), file=sys.stderr)
        time.sleep(2)

    time.sleep(a.gap)
    r2 = ask(a.endpoint, a.model, docA, "Reply OK.")
    steps.append({"phase": "reload-A", **{k: r2.get(k) for k in
                 ("id", "prompt_tokens", "cached_tokens", "elapsed_s")}})
    print("[dbg] reload A id=%s prompt=%s cached=%s (%ss)"
          % (r2.get("id"), r2.get("prompt_tokens"), r2.get("cached_tokens"),
             r2.get("elapsed_s")), file=sys.stderr)
    print(json.dumps({"salt": salt, "steps": steps}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
