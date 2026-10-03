#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vLLM 管理页对应修复（第二批）：/v1/internal/stats 的 vLLM 实例补 runtime 字段。

背景：stats.instances 是前端运行时徽标（index.html:2943/4023）的判据来源。
SGLang 侧主/从实例对象都带 runtime:'sglang'（且代码注释自陈「缺 runtime → 徽标误显示 vllm」），
而 vLLM 侧主实例与从实例对象都漏了该字段 → instances[0].runtime === undefined：
  · index.html:2943 二级缓存卡的「SGLang 不适用」分支永不触发；
  · index.html:4023 启动弹窗 runtime 默认只能靠回落值，拿不到实例真实运行时。
按 SGLang 侧对称补齐 'vllm'。
"""
import os
import shutil
import sys
import time

SERVER = "/home/ll/deploy/server.js"

E1_OLD = """    kv: buildKvCacheInfo(m),
    primary: false,
    gen_speed_1s: (ticker.lastSecond && ticker.lastSecond.speed) || 0,
  };
}"""

E1_NEW = """    kv: buildKvCacheInfo(m),
    primary: false,
    runtime: 'vllm',   // [vllm-page-1003] 对称 SGLang 从实例；前端运行时徽标/口径分支的判据
    gen_speed_1s: (ticker.lastSecond && ticker.lastSecond.speed) || 0,
  };
}"""

E2_OLD = """              kv: kvCache,
              primary: true,
              gen_speed_1s: (result.last_second && result.last_second.speed) || 0,
            }];"""

E2_NEW = """              kv: kvCache,
              primary: true,
              runtime: 'vllm',   // [vllm-page-1003] 对称 _sglPrimaryInst；缺该字段时前端徽标判据拿不到运行时
              gen_speed_1s: (result.last_second && result.last_second.speed) || 0,
            }];"""

EDITS = [("C1 vLLM 从实例补 runtime", E1_OLD, E1_NEW),
         ("C2 vLLM 主实例补 runtime", E2_OLD, E2_NEW)]


def main():
    src = open(SERVER, encoding="utf-8").read()
    if "--check" in sys.argv:
        for n, o, w in EDITS:
            print("%-30s %s" % (n, "done" if w in src else ("todo" if o in src else "ANCHOR-MISSING")))
        return
    orig = src
    report = []
    for n, o, w in EDITS:
        if w in src and o not in src:
            report.append("%-30s SKIP(已是新版)" % n)
            continue
        c = src.count(o)
        if c != 1:
            print("FAIL[%s]: 锚点出现 %d 次（需恰好 1 次）" % (n, c))
            sys.exit(2)
        src = src.replace(o, w, 1)
        report.append("%-30s OK" % n)
    if src == orig:
        print("无改动")
        return
    b = "%s.bak-vllm-runtime2-%s" % (SERVER, time.strftime("%m%d-%H%M%S"))
    shutil.copy2(SERVER, b)
    with open(SERVER, "w", encoding="utf-8") as f:
        f.write(src)
    print("已写入；备份 %s" % b)
    for r in report:
        print("   " + r)


main()
