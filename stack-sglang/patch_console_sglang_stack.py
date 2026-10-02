#!/usr/bin/env python3
# 8889 控制台栈感知扩展：现行生产 = SGLang@18420（sglang-18420/ACTIVE 哨兵）。
# 幂等；--revert 恢复 server.js.bak-sglangstack-*。
# 注入点：
#  1) sglangActive() 哨兵 + resolveStop/StartScript 的 sglang 优先分支
#  2) SCRIPT_MODELS['qwen3.8-flash-next-w4a16'] 加 scriptSglang/stopScriptSglang/logSglang
#  3) scriptModelInstance 的 pgrep 认 sglang 主进程（否则状态/停止全瞎）
#  4) pickScriptModelLogFile 哨兵短路（日志面板读 sglang-18420.log）
import re
import shutil
import sys
import time

F = "/home/ll/deploy/server.js"
TAG = "dsh-sglang-stack-1003"

ts = time.strftime("%m%d-%H%M%S")
BAK = F + ".bak-sglangstack-" + ts


def apply():
    s = open(F).read()
    if TAG in s:
        print("already applied")
        return
    shutil.copy(F, BAK)

    # 1) 哨兵 + resolve 函数
    anchor = "const STACK0300_DISABLED = '/home/ll/deploy/vllm-0300/DISABLED';"
    assert s.count(anchor) == 1, "anchor1"
    s = s.replace(
        anchor,
        anchor
        + f"""
// [{TAG}] 第三栈：SGLang@18420。sglang-18420/ACTIVE 哨兵在位 = 现行生产走 sglang 脚本对，
// 优先级高于 vllm-0300/DISABLED 双栈判定。移除哨兵文件即回退旧路由。
const SGLANG_ACTIVE = '/home/ll/deploy/sglang-18420/ACTIVE';
function sglangActive() {{
  try {{ return fs.existsSync(SGLANG_ACTIVE); }} catch (e) {{ return false; }}
}}""",
    )
    for fn, script_field in (("resolveStopScript", "stopScriptSglang"), ("resolveStartScript", "scriptSglang")):
        old = "function %s(sm) {\n  try {\n" % fn
        assert s.count(old) == 1, "resolve fn " + fn + " count=" + str(s.count(old))
        ins = (
            "    if (sm && sm.%s && sglangActive() && fs.existsSync(sm.%s)) return sm.%s;  // [%s]\n"
            % (script_field, script_field, script_field, TAG)
        )
        s = s.replace(old, old + ins, 1)

    # 2) SCRIPT_MODELS 条目字段
    anchor2 = "  scriptNew: '/home/ll/deploy/vllm-0300/start-flash-next-0300.sh',"
    assert s.count(anchor2) == 1, "anchor2"
    s = s.replace(
        anchor2,
        "  // [%s] SGLang 栈脚本对与日志（ACTIVE 哨兵在位时 resolve* 优先选它们）\n"
        "  scriptSglang: '/home/ll/deploy/sglang-18420/start-flash-next-sglang.sh',\n"
        "  stopScriptSglang: '/home/ll/deploy/sglang-18420/stop-flash-next-sglang.sh',\n"
        "  logSglang: '/home/ll/deploy/sglang-18420.log',\n"
        % TAG
        + anchor2,
    )

    # 3) scriptModelInstance 认 sglang 主进程
    anchor3 = "pgrep -f \"[v]llm.entrypoints|[v]llm serve\""
    assert s.count(anchor3) == 1, "anchor3"
    s = s.replace(anchor3, "pgrep -f \"[v]llm.entrypoints|[v]llm serve|[s]glang.launch_server\"  ")

    # 4) 日志面板短路
    anchor4 = "function pickScriptModelLogFile(inst, sm) {"
    assert s.count(anchor4) == 1, "anchor4"
    s = s.replace(
        anchor4,
        anchor4
        + f"""
  // [{TAG}] sglang 栈在位 → 直接读 sglang 日志（打分链候选不含该文件名）
  try {{
    if (sm && sm.logSglang && sglangActive() && fs.existsSync(sm.logSglang)) return sm.logSglang;
  }} catch (e) {{}}""",
    )

    open(F, "w").write(s)
    print("APPLIED (bak=%s)" % BAK)


def revert():
    import glob

    baks = sorted(glob.glob(F + ".bak-sglangstack-*"))
    if not baks:
        print("no backup")
        return
    shutil.copy(baks[-1], F)
    print("reverted from", baks[-1])


if __name__ == "__main__":
    revert() if "--revert" in sys.argv else apply()
