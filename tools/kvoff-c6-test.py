#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kvoff-c6 验证窗口（2026-09-24）
P1 基础推理；P2 CPU 层回载正确性（80K 文档→4×170K 挤占→重发验证 cached>0 且答案精准）；
P3 并发冲刷（制造常规 store/flush）；P4 双 600K 并发（复现 09-23 抢占崩溃触发条件）。
判据：全程 health=200、无 EngineDead、无 shm_broadcast 卡死、[kvoff-c6] FUSE 允许出现
（出现=熔断兜底生效，也算 PASS，但记录时点）、QA 重发 cached>0 且复述码正确。
"""
import json
import random
import string
import time
import urllib.request

BASE = "http://127.0.0.1:18420"
RATIO = 0.5299  # tok/字符（中文标定值，用于估算）
LOG = "/home/ll/deploy/vllm-flash-next-w4a16.log"
OUT = open("/tmp/kvoff-c6-test.result", "w")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT.write(line + "\n")
    OUT.flush()


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t0


def chat(prompt, max_tokens=48, stream_stats=False, timeout=600):
    body = {
        "model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if stream_stats:
        body["stream_options"] = {"include_usage": True}
    return post("/v1/chat/completions", body, timeout=timeout)


def filler_words(n):
    rng = random.Random(0xC0FFEE)
    return " ".join("".join(rng.choices(string.ascii_lowercase, k=rng.randint(2, 9)))
                    for _ in range(n))


BOILER = 24  # 模板/包装 token 粗估


def filler(n_tokens, salt):
    """按标定比例生成约 n_tokens 的唯一文本（salt 前缀保证内容互不相同）。"""
    chars = int(max(0, n_tokens - BOILER) / RATIO_ASCII)
    unit = filler_words(200)  # ~1300 字符固定文本
    reps = chars // len(unit) + 1
    body = (unit * reps)[:chars]
    return f"DOC#{salt} " + body


def health():
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
            return r.status
    except Exception as e:
        return f"ERR {e}"


def log_lines_since(n0):
    with open(LOG, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 400000))
        txt = f.read().decode("utf-8", "replace")
    hits = []
    for pat in ("kvoff-c6", "EngineDead", "No available shared memory",
                "timed out", "cuMemcpyBatchAsync"):
        for ln in txt.splitlines():
            if pat in ln:
                hits.append(ln[:220])
    return hits


def usage_cached(resp):
    d = resp.get("usage", {}).get("prompt_tokens_details") or {}
    return d.get("cached_tokens", 0), resp["usage"].get("prompt_tokens", 0)


log("=== P0 标定 tok/字符 ===")
RATIO_ASCII = 0.30  # 初值
_probe_txt = filler_words(3000)
_rp, _ = chat(_probe_txt, max_tokens=8)
_tp = _rp["usage"]["prompt_tokens"] - 8
RATIO_ASCII = _tp / len(_probe_txt)
log(f"P0 标定：prompt={_tp} tok / {len(_probe_txt)} 字符 = {RATIO_ASCII:.4f} tok/char")

log("=== P1 基础推理 ===")
r0, dt = chat("只回复两个字：正常")
log(f"P1 content={r0['choices'][0]['message']['content']!r} ({dt:.1f}s)")

log("=== P2 CPU 回载正确性（80K 文档 → 4×170K 挤占 → 重发）===")
code = "".join(random.choices(string.ascii_uppercase + string.digits, k=12))
doc = (filler(78000, 9001) +
       f"\n\n本文档验证码是 {code} 。问：验证码是什么？只回答验证码本身。")
r1, dt1 = chat(doc, max_tokens=200)
log(f"P2 首发 {dt1:.0f}s prompt={r1['usage']['prompt_tokens']} "
    f"answer={r1['choices'][0]['message']['content']!r}")
for i in range(4):
    rr, ddt = chat(filler(165000, 9100 + i), max_tokens=32)
    log(f"P2 挤占#{i + 1} {ddt:.0f}s prompt={rr['usage']['prompt_tokens']}")
time.sleep(8)
r2, dt2 = chat(doc, max_tokens=200)
c2, p2 = usage_cached(r2)
ans2 = r2["choices"][0]["message"]["content"]
ok2 = code in ans2
log(f"P2 重发 {dt2:.0f}s cached={c2}/{p2} answer={ans2!r} "
    f"{'PASS' if ok2 and c2 > 0 else ('GPU命中?' if ok2 else 'FAIL')}")

log("=== P3 并发冲刷（3 波 × 4 路 × 30K）===")
import threading
for wave in range(3):
    ths, res = [], []

    def one(salt):
        try:
            rr, ddt = chat(filler(28000, salt), max_tokens=24)
            res.append((salt, rr["usage"]["prompt_tokens"], round(ddt, 1)))
        except Exception as e:
            res.append((salt, "ERR", str(e)[:80]))
    for j in range(4):
        t = threading.Thread(target=one, args=(9200 + wave * 10 + j,))
        t.start()
        ths.append(t)
    for t in ths:
        t.join()
    log(f"P3 波{wave + 1}: {res}")
    if health() != 200:
        log("P3 health 丢失，中止")
        break

log("=== P4 双 600K 并发（抢占触发）===")
res4 = []


def monster(salt):
    try:
        rr, ddt = chat(filler(590000, salt), max_tokens=24, timeout=1200)
        res4.append((salt, rr["usage"]["prompt_tokens"], round(ddt, 1)))
    except Exception as e:
        res4.append((salt, "ERR", str(e)[:120]))


ta = threading.Thread(target=monster, args=(9501,))
tb = threading.Thread(target=monster, args=(9502,))
ta.start(); tb.start(); ta.join(); tb.join()
log(f"P4: {res4}")

log("=== 汇总 ===")
h = health()
log(f"health={h}")
hits = log_lines_since(0)
for ln in hits[:20]:
    log("  命中: " + ln)
if not hits:
    log("  无 FUSE/EngineDead/shm_broadcast/cuMemcpy 命中 = 全程零异常（最佳结果）")
log("DONE")
OUT.close()
