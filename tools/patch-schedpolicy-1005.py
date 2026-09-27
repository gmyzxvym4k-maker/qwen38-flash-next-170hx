#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""0.30.0 栈接上 FN_SCHED_POLICY（1005）

背景：server.js scriptModelLaunchPlan 在用户选非 fcfs 调度策略时下发 FN_SCHED_POLICY
（server.js:828-829 `if (sp && sp !== 'fcfs') env.FN_SCHED_POLICY = sp;`），
而 vllm-0300/bin/flash-next-0300-inner.sh 既不消费它、也没有 NOOP_NOTE_ 说明
→ 落到"参数体检"的 UNKNOWN_FN 分支，只打一条「收到本脚本未实现的参数」警告，
引擎仍走 fcfs。与 09-26 FN_GENCFG 事故同型（弹窗可填字段静默/半静默失效）。

修法：ARGS 构造段加 --scheduling-policy，并把 FN_SCHED_POLICY 登记进 CONSUMED。
兼容性已核实：AsyncScheduler(Scheduler)（v1/core/sched/async_scheduler.py:12）只覆写
_update_after_schedule / _update_request_with_output，waiting 队列与抢占的 policy 逻辑
在基类（scheduler.py:194-204 create_request_queue(self.policy)、746-759 抢占排序），
故 --scheduling-policy priority 与 --async-scheduling 可共存，无需互斥。
合法值 Literal["fcfs","priority"]（config/scheduler.py:22），缺省 fcfs。

幂等；--revert 还原。
"""
import shutil
import sys

P = '/home/ll/deploy/vllm-0300/bin/flash-next-0300-inner.sh'
BAK = P + '.bak-schedpolicy-1005'

ASYNC_ANCHOR = ('if [ "${FN_ASYNC:-1}" = "0" ]; then ARGS+=(--no-async-scheduling); '
                'else ARGS+=(--async-scheduling); fi\n')

BLOCK = """
# 调度策略（1005 补）：plan 在用户选非 fcfs 时下发 FN_SCHED_POLICY，此前本脚本不消费
# → 弹窗选「优先级」只落一条"未实现"警告、引擎仍走 fcfs。AsyncScheduler 继承 Scheduler
# 的 waiting 队列与抢占逻辑（async_scheduler.py:12 仅覆写 _update_after_schedule /
# _update_request_with_output），故与 --async-scheduling 兼容，无需互斥。
# 合法值 Literal["fcfs","priority"]（config/scheduler.py:22）；fcfs 即引擎缺省，不下发。
if [ -n "${FN_SCHED_POLICY:-}" ]; then ARGS+=(--scheduling-policy "$FN_SCHED_POLICY"); fi
"""

CONSUMED_OLD = 'FN_KVOFF FN_KVOFF_BYTES FN_KVOFF_WAIT_TIMEOUT "'
CONSUMED_NEW = 'FN_KVOFF FN_KVOFF_BYTES FN_KVOFF_WAIT_TIMEOUT FN_SCHED_POLICY "'


def main():
    src = open(P, encoding='utf-8').read()

    if '--revert' in sys.argv:
        bak = open(BAK, encoding='utf-8').read()
        open(P, 'w', encoding='utf-8').write(bak)
        print('REVERTED from %s' % BAK)
        return 0

    if '--scheduling-policy' in src:
        print('ALREADY_PATCHED（已含 --scheduling-policy，无动作）')
        return 0

    n = src.count(ASYNC_ANCHOR)
    if n != 1:
        print('FAIL: async 锚点命中 %d 次（期望 1）' % n)
        return 2
    if src.count(CONSUMED_OLD) != 1:
        print('FAIL: CONSUMED 锚点命中 %d 次（期望 1）' % src.count(CONSUMED_OLD))
        return 2

    src = src.replace(ASYNC_ANCHOR, ASYNC_ANCHOR + BLOCK, 1)
    src = src.replace(CONSUMED_OLD, CONSUMED_NEW, 1)

    shutil.copy2(P, BAK)
    open(P, 'w', encoding='utf-8').write(src)
    print('OK patched, bytes %d -> %d, backup %s' % (len(src), len(src), BAK))
    return 0


if __name__ == '__main__':
    sys.exit(main())
