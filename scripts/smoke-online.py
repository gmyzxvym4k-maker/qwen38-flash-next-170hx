#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在线功能验收：工具调用 / 思考模式 / 长上下文 / 流式。只读，不改任何状态。"""
import json, time, urllib.request, sys

B = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18420"
M = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-flash-next"
UNIT = "在模型推理部署的实践中，我们需要持续关注显存占用、批处理规模、上下文长度与吞吐之间的平衡关系，并结合实测数据做出取舍。"


def post(path, payload, timeout=900):
    req = urllib.request.Request(B + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t0


def main():
    ok = True
    # 1) 工具调用
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "查询城市天气",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string", "description": "城市名"}},
                       "required": ["city"]}}}]
    d, t = post("/v1/chat/completions", {
        "model": M, "messages": [{"role": "user", "content": "北京现在天气怎么样？必须调用工具查询。"}],
        "tools": tools, "tool_choice": "auto", "max_tokens": 512, "temperature": 0})
    tc = d["choices"][0]["message"].get("tool_calls")
    name = tc[0]["function"]["name"] if tc else None
    args = tc[0]["function"]["arguments"] if tc else None
    print("[1 工具调用] %.1fs finish=%s name=%s args=%s" % (t, d["choices"][0]["finish_reason"], name, args))
    ok &= (name == "get_weather")

    # 2) 思考模式
    d, t = post("/v1/chat/completions", {
        "model": M, "messages": [{"role": "user", "content": "9.11 和 9.8 哪个大？一句话回答。"}],
        "max_tokens": 2048, "temperature": 0})
    m = d["choices"][0]["message"]
    r = m.get("reasoning") or m.get("reasoning_content") or ""
    c = (m.get("content") or "").replace("\n", " ")
    print("[2 思考模式] %.1fs reasoning=%d 字 content=%r" % (t, len(r), c[:70]))
    ok &= len(r) > 0 and len(c) > 0

    # 3) 长上下文（约 128K token）+ 标识复述自检
    salt = "zz-%d" % int(time.time())
    p = "【唯一标识 " + salt + "】" + UNIT * 2600
    # max_tokens 要给足：这是思考模型，预算太小时 96 个 token 全被 reasoning 吃掉，
    # content 会是空的（不是服务故障）。判据同 docs/08#P15。
    d, t = post("/v1/chat/completions", {
        "model": M,
        "messages": [{"role": "user", "content": p + "\n\n问题：请原样复述上文开头的【唯一标识】内容。"}],
        "max_tokens": 1024, "temperature": 0})
    u = d["usage"]
    det = u.get("prompt_tokens_details") or {}
    msg = d["choices"][0]["message"]
    out = (msg.get("content") or "").replace("\n", " ")
    rea = (msg.get("reasoning") or msg.get("reasoning_content") or "")
    hit = salt in out
    print("[3 长上下文] %.1fs prompt=%s cached=%s finish=%s reasoning=%d 标识命中=%s 输出=%r"
          % (t, u.get("prompt_tokens"), det.get("cached_tokens"),
             d["choices"][0]["finish_reason"], len(rea), hit, out[:60]))
    ok &= hit

    # 4) 流式 TTFT
    req = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps({
        "model": M, "messages": [{"role": "user", "content": "写一首四行诗"}],
        "max_tokens": 200, "temperature": 0.7, "stream": True,
        "stream_options": {"include_usage": True}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; n = 0
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            s = line.decode().strip()
            if s.startswith("data:") and "[DONE]" not in s:
                o = json.loads(s[5:])
                dl = (o.get("choices") or [{}])[0].get("delta") or {}
                if dl.get("content") or dl.get("reasoning_content"):
                    if first is None:
                        first = time.time()
                    n += 1
    print("[4 流式] chunks=%d TTFT=%.2fs" % (n, (first - t0) if first else -1))
    ok &= (first is not None)

    # 5) 标点病灶探针（P13：表错位时最先坏的就是标点）
    d, t = post("/v1/chat/completions", {
        "model": M, "messages": [{"role": "user", "content": "输出下面这句话，一字不改：今天天气很好，我们去爬山；山上有风，也有云。"}],
        "max_tokens": 256, "temperature": 0})
    out = (d["choices"][0]["message"].get("content") or "")
    bad = sum(out.count(x) for x in ("、。", "。，", "，，", "。。"))
    print("[5 标点自检] %.1fs 异常标点组合=%d 输出=%r" % (t, bad, out[:50]))
    ok &= (bad == 0)

    print("\n" + ("✓ 在线功能验收全部通过" if ok else "✗ 有用例未通过，见 docs/08 对照"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
