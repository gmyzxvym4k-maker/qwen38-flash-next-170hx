#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复控制台栈路由被 SGLang 哨兵劫持（2026-10-03）

现象
  · 控制台点「启动」→ 拉起的是 SGLang（而非官方 vLLM 0.30.0）
  · 控制台点「停止」→ 调用 SGLang 停止脚本，杀不到 vLLM 进程 → 报
    `[quickstart-script] stop … (WARN: 仍有残留 pid=…)`
  · 连带：vLLM 专属的「二级缓存（CPU KV offload）」在控制台里"没法使用"
    （SGLang 无该机制；且启动路径根本走不到 vLLM inner）

根因
  server.js 的栈路由按哨兵文件判定，且 SGLang 优先级高于 0.30.0：
    681  if (sm.stopScriptSglang && sglangActive() && …) return sm.stopScriptSglang;
    685  if (sm.scriptSglang      && sglangActive() && …) return sm.scriptSglang;
  而 `sglangActive()` = fs.existsSync('/home/ll/deploy/sglang-18420/ACTIVE')。
  该哨兵 10-03 07:31 启用 SGLang 时创建，回切 vLLM 时**无人清理**（全仓库无任何脚本
  写/删它，server.js 也只读）→ 路由恒指向 SGLang。

修法（两步）
  1) 清掉孤儿哨兵 → 路由立即回到 vLLM 0.30.0（sglangActive() 每次实时读文件，无需重启控制台）
  2) 根治：让哨兵随**实际启动的栈**自动同步——vLLM 启动脚本清哨兵、SGLang 启动脚本置哨兵，
     两栈互斥自洽，今后切换不会再留下孤儿状态。
"""
import os
import shutil
import sys
import time

DEPLOY = "/home/ll/deploy"
SENTINEL = os.path.join(DEPLOY, "sglang-18420/ACTIVE")
VLLM_START = os.path.join(DEPLOY, "vllm-0300/start-flash-next-0300.sh")
SG_START = os.path.join(DEPLOY, "sglang-18420/start-flash-next-sglang.sh")

# ---- vLLM 启动脚本：清 SGLang 哨兵（紧随现有的人工停止闩锁清理）----
V_OLD = """# 启动 = 操作者声明实例应在运行：清除人工停止闩锁，恢复看门狗托管（09-26）
rm -f /home/ll/deploy/fnx-manual-stop"""

V_NEW = """# 启动 = 操作者声明实例应在运行：清除人工停止闩锁，恢复看门狗托管（09-26）
rm -f /home/ll/deploy/fnx-manual-stop
# [stack-route-fix-1003] 清除 SGLang 栈哨兵：控制台 resolveStart/StopScript 以
# sglang-18420/ACTIVE 是否存在判定栈路由，且 SGLang 优先级高于 0.30.0。该哨兵若
# 残留（10-03 切回 vLLM 时无人清），控制台的启动/停止会双双路由到 SGLang——表现为
# 「停止报仍有残留」「启动拉起 SGLang」「vLLM 专属的 CPU KV 二级缓存没法用」。
# 两栈互斥：本脚本（vLLM 栈）启动即声明栈归属，SGLang 启动脚本负责置位。
rm -f /home/ll/deploy/sglang-18420/ACTIVE"""

# ---- SGLang 启动脚本：置哨兵（对称声明）----
SG_ANCHOR = "BASE=/home/ll/deploy/sglang-18420"
SG_NEW = """BASE=/home/ll/deploy/sglang-18420
# [stack-route-fix-1003] 置 SGLang 栈哨兵：与 vLLM 栈启动脚本的清哨兵动作互斥配对，
# 保证控制台栈路由（resolveStartScript/resolveStopScript）始终跟随实际启动的栈。
: > /home/ll/deploy/sglang-18420/ACTIVE"""


def patch(path, edits, tag):
    src = open(path, encoding="utf-8").read()
    orig = src
    rep = []
    for name, old, new in edits:
        if new in src and old not in src:
            rep.append("%-34s SKIP(已应用)" % name)
            continue
        c = src.count(old)
        if c != 1:
            print("FAIL[%s]: 锚点出现 %d 次（需 1 次）→ 中止，未写文件" % (name, c))
            return False
        src = src.replace(old, new, 1)
        rep.append("%-34s OK" % name)
    if src == orig:
        print("[%s] 无改动" % os.path.basename(path))
        for r in rep:
            print("   " + r)
        return True
    b = "%s.bak-%s-%s" % (path, tag, time.strftime("%m%d-%H%M%S"))
    shutil.copy2(path, b)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("[%s] 已写入；备份 %s" % (os.path.basename(path), b))
    for r in rep:
        print("   " + r)
    return True


def main():
    if "--check" in sys.argv:
        print("哨兵存在:", os.path.exists(SENTINEL))
        for p, edits, _ in ((VLLM_START, [(("清哨兵", V_OLD, V_NEW))], "v"),
                            (SG_START, [(("置哨兵", SG_ANCHOR, SG_NEW))], "s")):
            src = open(p, encoding="utf-8").read()
            for n, o, w in edits:
                print("%-28s %s" % (os.path.basename(p) + ":" + n,
                                    "done" if w in src else ("todo" if o in src else "ANCHOR-MISSING")))
        return

    ok1 = patch(VLLM_START, [("vLLM 启动清 SGLang 哨兵", V_OLD, V_NEW)], "stackroute")
    ok2 = patch(SG_START, [("SGLang 启动置哨兵", SG_ANCHOR, SG_NEW)], "stackroute")
    if not (ok1 and ok2):
        sys.exit(2)

    if os.path.exists(SENTINEL):
        shutil.copy2(SENTINEL, SENTINEL + ".orphan-bak-" + time.strftime("%m%d-%H%M%S"))
        os.remove(SENTINEL)
        print("已删除孤儿哨兵 %s（原件留 .orphan-bak-* 备份）" % SENTINEL)
    else:
        print("哨兵已不存在")
    print("DONE")


main()
