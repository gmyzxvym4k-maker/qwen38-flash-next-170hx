#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kvoff-c8-probe —— CPU KV 二级缓存「回载」真值探针（不重启、只发请求）。

判据口径（历史定版，别改）：
  · `vllm:external_prefix_cache_hits_total` 增量 > 0  = 命中发生在 **CPU 档**
    （GPU 显存前缀缓存命中走 prefix_cache_hits_total，不算二级缓存收益）；
  · `CPU_to_GPU` 字节增量 > 0 = 真的有把 KV 从内存搬回显存；
  · 验证码精准复述 = 回载内容无损（光有计数器不算数）。
  只看 cached_tokens 会被显存命中污染 ⇒ 必须先把 GPU 池挤干净再重发。

用法：
  python3 kvoff-c8-probe.py                 # 缺省 2 篇建档 + 6 篇挤池 + 重发
  python3 kvoff-c8-probe.py --docs 2 --flush 6 --tokens 120000
  python3 kvoff-c8-probe.py --endpoint http://127.0.0.1:18420/v1
退出码：0=三项判据全绿；1=任一失败（详情打印 JSON）。
"""
import argparse
import json
import os
import random
import re
import string
import sys
import time
import urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango uniform victor "
         "whiskey xray yankee zulu").split()


def post(url, payload, timeout=1800):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()), time.time() - t0


def metrics(base):
    """抓 /metrics，返回 {指标名: 数值}（同名多标签取求和）。"""
    murl = re.sub(r"/v1/?$", "", base) + "/metrics"
    try:
        with urllib.request.urlopen(murl, timeout=20) as r:
            txt = r.read().decode()
    except Exception as exc:
        print("[probe] /metrics 抓取失败：%s" % exc, file=sys.stderr)
        return {}
    out = {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{")[0].split(" ")[0]
        val = line.rsplit(" ", 1)[-1].strip()
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            pass
    return out


def offload_bytes(m):
    """CPU_to_GPU / GPU_to_CPU 累计字节（histogram 的 _sum 或 counter 皆可）。"""
    res = {}
    for key in ("CPU_to_GPU", "GPU_to_CPU"):
        tot = 0.0
        for name, val in m.items():
            if "kv_offload" in name and key.lower().replace("_", "") in name.lower().replace("_", ""):
                tot += val
        res[key] = tot
    # 兜底：label 形态（direction="CPU_to_GPU"）时上面按名字匹配不到，逐行再算一次
    return res


# 【口径铁律】只累加**真字节 counter**，绝不按子串扫 "kv_offload"：
#   2026-09-27 实测踩坑 —— 旧实现把 `kv_offload_*_created`（**unix 时间戳 gauge**，
#   ≈1.79e9）与直方图桶一起加进来，凭空造出"5.37 GB CPU→GPU 回载"的假象，
#   而真值 `kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"}` 一直是 0。
_BYTE_COUNTERS = ("vllm:kv_offload_load_bytes", "vllm:kv_offload_store_bytes",
                  "vllm:kv_offload_total_bytes_total")
_FORBID = ("_created", "_time", "_size", "usage_perc", "_bucket")


def _metric_lines(base):
    murl = re.sub(r"/v1/?$", "", base) + "/metrics"
    try:
        with urllib.request.urlopen(murl, timeout=20) as r:
            return r.read().decode().splitlines()
    except Exception:
        return []


def offload_bytes_labeled(base):
    """按**指标名 + transfer_type 标签**精确取字节累计（load=CPU_to_GPU）。"""
    want = {"CPU_to_GPU": 0.0, "GPU_to_CPU": 0.0}
    for line in _metric_lines(base):
        if line.startswith("#") or "kv_offload" not in line:
            continue
        name = line.split("{")[0].split(" ")[0]
        if name not in _BYTE_COUNTERS or any(f in name for f in _FORBID):
            continue
        if name.endswith("load_bytes"):
            k = "CPU_to_GPU"
        elif name.endswith("store_bytes"):
            k = "GPU_to_CPU"
        else:
            k = next((d for d in want if 'transfer_type="%s"' % d in line), None)
            if k is None:
                continue
        try:
            want[k] += float(line.rsplit(" ", 1)[-1])
        except ValueError:
            pass
    return want


def calibrate_ratio(endpoint, model):
    """实测本文本形态下的 token/字符比（缺省 0.5299 是中文标定值，
    探针用的是英文词表，实测约 1.6 —— 不标定就会把文档做长 3 倍）。"""
    filler = " ".join(WORDS * 40)[:4000]
    j, _ = post(endpoint + "/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": filler + "\n\nReply OK."}],
        "max_tokens": 1, "temperature": 0.0, "stream": False})
    pt = (j.get("usage") or {}).get("prompt_tokens") or 0
    n = len(filler) + len("\n\nReply OK.")
    if pt <= 0 or n <= 0:
        return None
    return pt / n


def gpu_pool_tokens(base):
    """从 /metrics 的 cache_config_info 里取 kv_cache_size_tokens。"""
    murl = re.sub(r"/v1/?$", "", base) + "/metrics"
    try:
        with urllib.request.urlopen(murl, timeout=20) as r:
            for line in r.read().decode().splitlines():
                if "kv_cache_size_tokens" in line and "cache_config_info" in line:
                    m = re.search(r'kv_cache_size_tokens="(\d+)"', line)
                    if m:
                        return int(m.group(1))
    except Exception:
        pass
    return None


def make_doc(idx, tokens, ratio, rng):
    """构造约 tokens 个 token 的自然文本，内嵌唯一验证码。

    ⚠️ 单位铁律（2026-09-27 踩坑）：`ratio` 是 **token/字符**（calibrate_ratio 实测），
    所以文档长度必须按**字符预算**生成：chars = tokens / ratio。
    旧实现写的是 `tokens/ratio/12` 个"12 词块"（= tokens/ratio **个单词**），
    把 ratio 当成了 token/单词 ⇒ 实际文档大 6 倍（target 5000 → 30285 token，
    target 45000 → 270022 token），小池位测试里直接顶穿 max_model_len 吃 400。
    """
    code = "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(7))
    n = "CODE-%s-%04d" % (code, idx)
    target_chars = int(tokens / max(ratio, 1e-6))
    parts, ln, i = [], 0, 0
    while ln < target_chars:
        w = " ".join(rng.choices(WORDS, k=12))
        parts.append(w)
        ln += len(w) + 1
        i += 1
        if i == 3:  # 验证码埋在开头 1/3 处（与旧版一致）
            tag = "THE VERIFICATION TOKEN IS %s END OF TOKEN." % n
            parts.append(tag)
            ln += len(tag) + 1
    return n, " ".join(parts)


def ask(endpoint, model, doc, question, max_tokens=64):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": doc + "\n\n" + question}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    j, el = post(endpoint + "/chat/completions", payload)
    ch = (j.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    text = " ".join(filter(None, [msg.get("content") or "",
                                  msg.get("reasoning_content") or ""]))
    usage = j.get("usage") or {}
    det = (usage.get("prompt_tokens_details") or {})
    return {"elapsed": round(el, 2), "text": text.strip(),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": det.get("cached_tokens")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:18420/v1")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--tokens", type=int, default=100000, help="每篇文档目标 token")
    ap.add_argument("--ratio", type=float, default=0.0,
                    help="token/字符；0=自动标定（本探针用英文词表，约 1.6）")
    ap.add_argument("--docs", type=int, default=2, help="建档文档数")
    ap.add_argument("--flush", type=int, default=11,
                    help="挤池文档数：总 token 必须 > GPU KV 池（本机 120.7 万）才会把建档"
                         "文档从显存挤出去；同时 建档+挤池 总量必须 < CPU 档容量"
                         "（48 GiB≈160 万 tok），否则 CPU 侧也被 LRU 掉 = 必 miss")
    ap.add_argument("--gap", type=float, default=5.0, help="请求间隔秒")
    ap.add_argument("--answer-tokens", type=int, default=1024,
                    help="问答预算：本模型是思考模型，预算给小会被思考吃满、"
                         "正文空 → 复述判定假阴性（09-27 实测 256 不够）")
    ap.add_argument("--calib-tokens", type=int, default=20000,
                    help="容量自标定用的全新文档 token 数；0=跳过标定")
    ap.add_argument("--max-prompt-tokens", type=int, default=0,
                    help="单请求 prompt 上限（=实例 max-model-len − 回答预算）。>0 时自适应"
                         "选参会把建档/挤池文档压到该值以内，否则小池位测试必吃 400")
    ap.add_argument("--shm-baseline-mb", type=float, default=248.0,
                    help="/dev/shm 非本区域占用（基线），用于反推共享区字节数")
    ap.add_argument("--auto", type=int, default=1,
                    help="1=按标定结果自动选建档/挤池规模（推荐）；0=用命令行给的")
    args = ap.parse_args()

    rng = random.Random(20260927)
    out = {"endpoint": args.endpoint, "checks": {}, "detail": []}

    if args.ratio <= 0:
        got = calibrate_ratio(args.endpoint, args.model)
        args.ratio = got or 1.6
    out["ratio_tok_per_char"] = round(args.ratio, 4)

    def snap():
        return metrics(args.endpoint)

    m0 = snap()
    b0 = offload_bytes_labeled(args.endpoint)

    # ---- 容量自标定：CPU 档每 token 占多少字节、要装下 GPU 池得配多大 ----
    pool = gpu_pool_tokens(args.endpoint)
    if args.calib_tokens > 0:
        cu0 = m0.get("vllm:kv_offload_cpu_cache_usage_perc", 0.0)
        cb0 = offload_bytes_labeled(args.endpoint)["GPU_to_CPU"]
        _, cdoc = make_doc(9999, args.calib_tokens, args.ratio, rng)
        ask(args.endpoint, args.model, cdoc, "Reply OK.", max_tokens=4)
        import time as _t
        _t.sleep(max(6.0, args.gap))
        mc = snap()
        cb1 = offload_bytes_labeled(args.endpoint)["GPU_to_CPU"]
        d_store = cb1 - cb0
        d_usage = mc.get("vllm:kv_offload_cpu_cache_usage_perc", 0.0) - cu0
        # 只算真正落盘的 token（最后不足一个 chunk 的部分不算）：粗估按 85%
        stored_tok = max(1.0, args.calib_tokens * 0.85)
        bpt = d_store / stored_tok
        out["calibration"] = {
            "calib_prompt_tokens": args.calib_tokens,
            "delta_store_bytes": d_store,
            "delta_usage_perc": d_usage,
            "cpu_bytes_per_token_est": round(bpt, 1),
            "gpu_pool_tokens": pool,
            "gib_needed_to_beat_pool": (
                round((pool * 1.15) * bpt / 2**30, 1) if pool else None),
        }
        if pool and d_store > 0:
            need_gib = (pool * 1.15 + args.docs * args.tokens) * bpt / 2**30
            out["calibration"]["gib_needed_with_build_docs"] = round(need_gib, 1)
            out["calibration"]["note"] = (
                "若 config < 该值，本轮挤池必然把建档文档从 CPU 侧 LRU 掉 ⇒ 回载必 miss，"
                "属探针容量配置问题而非 c8 缺陷")
            # ---- 实测 CPU 档容量，并自动选规模（P+B < F <= C-B）----
            try:
                st = os.statvfs("/dev/shm")
                shm_used = (st.f_blocks - st.f_bfree) * st.f_frsize
                region = max(0, shm_used - int(args.shm_baseline_mb * 1e6))
                # 容量优先用 usage_perc 标定：C = 标定token / Δusage。
                # 字节法（region/bpt）会高估——实测 offload 给每个 chunk 预留的 slot
                # 远大于实际写入的 KV（90MB/块 vs 实际 ≈25MB/块），区域里大量是空洞。
                if d_usage > 0.004:
                    C = args.calib_tokens / d_usage
                    out["calibration"]["capacity_method"] = "usage_perc"
                else:
                    C = region / bpt if bpt > 0 else 0
                    out["calibration"]["capacity_method"] = "bytes(高估,Δusage过小)"
                out["calibration"]["shm_used_bytes"] = shm_used
                out["calibration"]["region_bytes_est"] = region
                out["calibration"]["cpu_capacity_tokens_est"] = int(C)
                if args.auto and C > 0:
                    B = int((C - pool) / 2.6)
                    # 下限 15k：小池位测量（--num-gpu-blocks-override 压小 GPU 池）时
                    # C-P 只有几万 token，固定 60k 下限会让约束无解、自适应用不起来。
                    B = max(15000, min(B, 200000, args.tokens))
                    if args.max_prompt_tokens:      # 小池位测试：不得顶穿 max-model-len
                        B = min(B, max(15000, args.max_prompt_tokens))
                    Fmax = int(C - B)
                    F = int(min(Fmax, pool + B + max(int(0.12 * pool), 60000)))
                    nflush = max(1, int(round(F / B)))
                    # 挤池必须真能挤出建档文档（F > P+B），至少 3 篇留余量
                    if Fmax >= 3 * B and (nflush < 3 or B * nflush <= pool + B):
                        nflush = 3
                    out["calibration"]["auto_plan"] = {
                        "doc_tokens": B, "flush_docs": nflush,
                        "flush_tokens": B * nflush,
                        "gpu_pool": pool,
                        "ok_gpu_evicts_build": (B * nflush) > (pool + B),
                        "ok_cpu_keeps_build": (B * nflush + B) <= C,
                    }
                    _msg = ("[probe] 标定：bpt=%.1f B/token（%s）C=%.2fM token "
                            "P=%.2fM → 建档 %d tok + 挤池 %d × %d tok（挤池 %.2fM）"
                            % (bpt, out["calibration"]["capacity_method"], C / 1e6,
                               pool / 1e6, B, nflush, B, B * nflush / 1e6))
                    print(_msg, file=sys.stderr, flush=True)
                    print(_msg, flush=True)
                    args.tokens = B
                    args.flush = nflush
            except Exception as exc:
                print("[probe] 自适应选参失败，用命令行参数：%s" % exc,
                      file=sys.stderr, flush=True)

    # 1) 建档：发 docs 篇长文档（触发 GPU→CPU store）
    docs = []
    for i in range(args.docs):
        code, doc = make_doc(i, args.tokens, args.ratio, rng)
        q = "What is the verification token in the document above? Answer with the token only."
        r = ask(args.endpoint, args.model, doc, q)
        docs.append((code, doc, q, r))
        out["detail"].append({"phase": "build", "i": i, "code": code, "answer": r["text"][:120],
                              "prompt_tokens": r["prompt_tokens"],
                              "cached_tokens": r["cached_tokens"],
                              "elapsed": r["elapsed"]})
        time.sleep(args.gap)

    # 2) 挤池：发 flush 篇互不相同的长文档，把 GPU KV 池占满、把建档请求的块挤出去
    for i in range(args.flush):
        _, doc = make_doc(1000 + i, args.tokens, args.ratio, rng)
        r = ask(args.endpoint, args.model, doc,
                "Reply with the single word OK.", max_tokens=8)
        out["detail"].append({"phase": "flush", "i": i,
                              "prompt_tokens": r["prompt_tokens"],
                              "cached_tokens": r["cached_tokens"],
                              "elapsed": r["elapsed"]})
        time.sleep(args.gap)

    # 挤池后再等一会儿：二级缓存 store 是异步的，且历史实测块提交有延迟
    time.sleep(max(10.0, args.gap))

    # 3) 重发第一篇：期望 CPU 档命中（external hits + CPU_to_GPU 字节 + 内容无损）
    code, doc, q, _ = docs[0]
    m1 = snap()
    b1 = offload_bytes_labeled(args.endpoint)
    r = ask(args.endpoint, args.model, doc, q, max_tokens=args.answer_tokens)
    m2 = snap()
    b2 = offload_bytes_labeled(args.endpoint)
    out["detail"].append({"phase": "reload", "code": code,
                          "prompt_tokens": r["prompt_tokens"],
                          "cached_tokens": r["cached_tokens"],
                          "elapsed": r["elapsed"], "text": r["text"][:200]})

    d_ext = m2.get("vllm:external_prefix_cache_hits_total", 0.0) - \
        m1.get("vllm:external_prefix_cache_hits_total", 0.0)
    d_load = b2["CPU_to_GPU"] - b1["CPU_to_GPU"]
    d_store = b2["GPU_to_CPU"] - b1["GPU_to_CPU"]
    secret = code.split("-")[1] if "-" in code else code
    recalled = (code in r["text"]) or (secret in r["text"])

    out["checks"]["external_hit_tokens"] = d_ext
    out["checks"]["cpu_to_gpu_bytes"] = d_load
    out["checks"]["gpu_to_cpu_bytes_during_reload"] = d_store
    out["checks"]["recalled_verification_code"] = bool(recalled)
    out["checks"]["cumulative"] = {
        "external_hits_total": m2.get("vllm:external_prefix_cache_hits_total"),
        "external_queries_total": m2.get("vllm:external_prefix_cache_queries_total"),
        "prefix_hits_total": m2.get("vllm:prefix_cache_hits_total"),
        "cpu_to_gpu_total": b2["CPU_to_GPU"],
        "gpu_to_cpu_total": b2["GPU_to_CPU"],
        "cpu_cache_usage_perc": m2.get("vllm:kv_offload_cpu_cache_usage_perc"),
        "cpu_cache_fill_perc": m2.get("vllm:kv_offload_cpu_cache_fill_perc"),
    }
    out["verdict"] = bool(d_ext > 0 and d_load > 0 and recalled)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["verdict"] else 1


if __name__ == "__main__":
    sys.exit(main())
