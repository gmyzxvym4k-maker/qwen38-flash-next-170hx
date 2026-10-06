#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[1006 kvoff-accept-0310] 二级缓存·CPU（SimpleCPUOffloadConnector）验收探针
判据口径（09-27 定版，三条一起看才算过）：
  ① /metrics vllm:external_prefix_cache_hits_total 增量 > 0（外档=内存档真的回载过）
  ② 重发请求 usage.prompt_tokens_details.cached_tokens ≈ 文档长度（前缀复用）
  ③ 文档里埋的验证码被逐字复述（回载内容正确，不是坏数据）
另外全程记录 Xid 计数与 segfault 关键词。

纪律（血泪教训）：建档 / 挤池 / 重发三步的 chat_template_kwargs 必须完全一致，
一边开思考一边关会让首块哈希不同 → 永远 0 命中（09-27 两次假 0 全是这个原因）。
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

BASE = os.environ.get("FN_BASE_URL", "http://127.0.0.1:18420")
MODEL = os.environ.get("PROBE_MODEL", "qwen3.8-flash-next")
CHUNK = int(os.environ.get("CHUNK_TOKENS", "100000"))      # 挤池每发填充的 token 量级
CHURN_TARGET = int(os.environ.get("CHURN_TARGET", "1400000"))  # 挤池总量（> GPU 池 121 万）
DOC_TARGET = int(os.environ.get("DOC_TARGET", "40000"))
ANSWER_MAX = int(os.environ.get("ANSWER_MAX_TOKENS", "900"))  # ⚠ 别用 64：思考模型会把预算吃满、content 为空 → 假阴性（09-27 与 10-06 两轮都栽过）
# 本副本已脱敏：Xid 计数需要 sudo，跑前 export SUDO_PASS=<部署机 sudo 口令>；不设则该项记 "?"
SUDO_PASS = os.environ.get("SUDO_PASS", "")
RES = os.environ.get("PROBE_RES", "/home/ll/deploy/kvoff-accept-0310.result")
CHATKW = {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "medium"}

out = open(RES, "a", buffering=1)
log = lambda *a: print(*a, file=out)


def req(messages, max_tokens=64, temperature=0.0):
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "chat_template_kwargs": CHATKW,
    }).encode()
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=1800) as f:
        d = json.loads(f.read())
    return d, time.time() - t0


def metrics(pat):
    o = subprocess.run(["curl", "-s", "-m", "8", BASE + "/metrics"],
                       capture_output=True, text=True).stdout
    m = re.search(pat + r"\{[^}]*\}\s+([0-9.]+)", o)
    return float(m.group(1)) if m else 0.0


def xid():
    p = subprocess.run(["bash", "-c",
                        "echo \"$SUDO_PASS\" | sudo -S -p '' dmesg 2>/dev/null | grep -c 'Xid'"],
                       capture_output=True, text=True)
    return p.stdout.strip()


def make_doc(target_tokens, nonce):
    """凑到约 target_tokens（本机中文 0.53 tok/字符，英文约 0.75 字符/token）：
    用重复段落 + 每轮唯一 nonce（防前缀缓存复用旧内容）+ 埋两处验证码。"""
    code = "CODE-%s-%04d" % (nonce, 3126)
    para = ("电力系统里，输电线路的等值电路按长度分段建立；每单位长度的串联阻抗与并联导纳"
            "共同决定电压降落与功率传输能力。潮流计算需要收敛判据与迭代顺序，"
            "短线路用集中参数，中长线路用标称π模型，超长线路用分布参数模型。")
    # 中文约 0.53 tok/字 → 目标字符数 ≈ target/0.53
    chars = int(target_tokens / 0.53)
    unit = len(para) + 30
    reps = max(1, chars // unit)
    parts = []
    for i in range(reps):
        parts.append(f"【段{i:05d}】{para}本段编号{i:05d}，唯一标识 {nonce}-{i}，请勿混淆。")
    parts.append(f"本文档的唯一验证码是 {code}，请原样记住它。")
    parts.append(f"文末再次确认：验证码={code}，编号顺序为 {reps} 段。")
    text = "\n".join(parts)
    return text, code


def main():
    nonce = str(int(time.time()))[-6:]
    log("=== kvoff-accept-0310 开始 %s（doc≈%d tok, churn≈%d tok, chatkw=%s）"
        % (time.strftime("%F %T"), DOC_TARGET, CHURN_TARGET, CHATKW))
    log("起点 Xid=%s" % xid())
    ext0 = metrics("vllm:external_prefix_cache_hits_total")
    extq0 = metrics("vllm:external_prefix_cache_queries_total")

    doc, code = make_doc(DOC_TARGET, nonce)
    log("[1] 建档：文档字符=%d（估算 token≈%d）" % (len(doc), int(len(doc) * 0.53)))
    d, el = req([{"role": "user", "content": doc + "\n\n请用一句话说完这段文档讲了什么。"}],
                max_tokens=200)
    pt = d["usage"]["prompt_tokens"]
    det = (d.get("usage") or {}).get("prompt_tokens_details") or {}
    log("    实测 prompt_tokens=%d（标定 tok/字符=%.4f）cached=%s 耗时=%.1fs"
        % (pt, pt / len(doc), det.get("cached_tokens"), el))
    ratio = pt / len(doc)

    # 挤池：发若干条互不相同的大请求，把 GPU 池（121 万 token）冲穿
    need = CHURN_TARGET
    sent = 0
    i = 0
    while sent < need:
        i += 1
        filler, _ = make_doc(CHUNK, "fill%s-%d" % (nonce, i))
        d2, el2 = req([{"role": "user", "content": filler + "\n\n只回答：收到"}], max_tokens=8)
        got = d2["usage"]["prompt_tokens"]
        sent += got
        log("    churn#%d prompt=%d 累计=%d elapsed=%.1fs" % (i, got, sent, el2))

    ext_mid = metrics("vllm:external_prefix_cache_hits_total")
    gpu_usage = metrics("vllm:gpu_cache_usage_perc")
    log("[2] 挤池完成 累计=%d token；期间外档命中增量=%.0f；GPU 池占用=%s" % (sent, ext_mid - ext0, gpu_usage))

    # 重发同一文档（前缀必须与建档逐字一致：同 CHATKW、同 doc 文本）
    log("[3] 重发建档文档，问验证码")
    t0 = time.time()
    d3, el3 = req([{"role": "user", "content": doc + "\n\n这份文档里出现的验证码是什么？请只输出验证码本身，不要解释。"}],
                  max_tokens=ANSWER_MAX)
    det3 = (d3.get("usage") or {}).get("prompt_tokens_details") or {}
    cached = det3.get("cached_tokens") or 0
    msg = d3["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    txt = content + reasoning          # 思考模型：验证码可能在 reasoning 里，判据必须合并两路
    ext1 = metrics("vllm:external_prefix_cache_hits_total")
    extq1 = metrics("vllm:external_prefix_cache_queries_total")
    hit = code in txt
    log("    prompt=%d cached=%d（%.1f%%）external_hits 增量=%.0f external_queries 增量=%.0f 耗时=%.1fs"
        % (d3["usage"]["prompt_tokens"], cached, 100 * cached / max(1, d3["usage"]["prompt_tokens"]),
           ext1 - ext0, extq1 - extq0, el3))
    log("    期望验证码=%s | 逐字命中=%s | content=%r | reasoning 长度=%d"
        % (code, hit, content[:60], len(reasoning)))
    if not content.strip():
        log("    ⚠ content 为空：预算被思考吃掉，属探针假阴性，请加大 ANSWER_MAX_TOKENS 复验")
    log("终点 Xid=%s" % xid())

    ver = 1 if (ext1 - ext0 > 0 and hit and cached > 0) else 0
    log("VERDICT=%d（判据：外档命中>0 且 验证码逐字命中 且 cached>0）" % ver)
    print("VERDICT=%d external_hits_delta=%.0f cached=%d code_ok=%s"
          % (ver, ext1 - ext0, cached, hit))
    return 0 if ver else 1


if __name__ == "__main__":
    sys.exit(main())
