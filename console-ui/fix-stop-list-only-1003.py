#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STOP_LIST_ONLY 透传修复（2026-10-03）

缺陷：stop-flash-next-0300.sh 自提权时用 `sudo -S -p '' bash "$0" "$PORT"` 重跑自己，
而 sudo 默认清环境 → 调用方设的 `STOP_LIST_ONLY=1`（只列目标、不动手）丢失，
只读探测退化为"真停实例"（本机 10-03 已因此误停过一次生产实例）。

修法：提权时以 `VAR=value` 形式显式透传（sudo 支持该语法）。

实现说明：锚点用行级正则匹配（不内联任何口令字面量），因此本脚本可原样进公开仓库。
"""
import os
import re
import shutil
import sys
import time

DEPLOY = "/home/ll/deploy"
F = os.path.join(DEPLOY, "vllm-0300/stop-flash-next-0300.sh")

# 匹配提权那一行的 "| sudo -S -p '' bash "$0" "$PORT" "，在前面插入 STOP_LIST_ONLY 透传。
PAT = re.compile(r"(\| sudo -S -p '' )bash \"\$0\" \"\$PORT\"")
REPL = r'\1STOP_LIST_ONLY="${STOP_LIST_ONLY:-}" bash "$0" "$PORT"'


def main():
    src = open(F, encoding="utf-8").read()
    m = PAT.search(src)
    already = 'sudo -S -p \'\' STOP_LIST_ONLY="${STOP_LIST_ONLY:-}"' in src
    if "--check" in sys.argv:
        print("已修复:", already, "| 锚点可匹配:", bool(m))
        return
    if already:
        print("SKIP: 已修复")
        return
    if not m:
        print("FAIL: 未匹配到提权行（脚本可能已改）→ 未写文件")
        sys.exit(2)
    b = F + ".bak-stoplistonly-" + time.strftime("%m%d-%H%M%S")
    shutil.copy2(F, b)
    open(F, "w", encoding="utf-8").write(PAT.sub(REPL, src, count=1))
    print("OK 已写入；备份 %s" % b)


main()
