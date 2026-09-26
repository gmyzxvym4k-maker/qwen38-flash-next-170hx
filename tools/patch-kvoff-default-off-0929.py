#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KVOFF 缺省关闭三处同步（0929，A/B 定案：21h 零外部命中 + 释放 107GB pinned）：
1) server.js SCRIPT_MODELS base: kvoff:'1' -> kvoff:'0'（含注释同步）
2) server.js scriptModelLaunchPlan 回落分支: 字段缺失=关闭（原=开启）
3) index.html 脚本模型弹窗缺省: kvOff: '1' -> '0'
幂等；备份 .bak-kvoffoff-0929；node --check 失败自动回滚。
用法: python3 patch-kvoff-default-off-0929.py [--revert]
"""
import sys, shutil, subprocess

SJ = "/home/ll/deploy/server.js"
IH = "/home/ll/deploy/index.html"
BAK = ".bak-kvoffoff-0929"
revert = "--revert" in sys.argv

def patch(path, pairs):
    src = open(path, encoding="utf-8").read()
    changed = 0
    for old, new in pairs:
        a, b = (new, old) if revert else (old, new)
        if b in src and a not in src:
            print("  已改过（跳过）: %s..." % b[:50])
            continue
        n = src.count(a)
        if n != 1:
            print("  [SKIP] 锚点命中 %d 次（期望 1）: %s..." % (n, a[:60]))
            continue
        src = src.replace(a, b)
        changed += 1
        print("  OK: %s..." % b[:60].replace("\n", "\\n"))
    if changed:
        shutil.copy2(path, path + BAK)
        open(path, "w", encoding="utf-8").write(src)
    return changed

print("== server.js ==")
pairs = [
    # 1) base 缺省
    ("kvoff: '1', kvOffGiB: 96",
     "kvoff: '0', kvOffGiB: 96"),
    ("    // [kvoff-sync 09-23] 生产真值：二级缓存开 96GiB（c5a 修复后命中已通）；",
     "    // [kvoff-off 0929] 生产真值：二级缓存关（A/B 定案 21h 零外部命中，省 107GB pinned）；"),
    # 2) plan 回落分支：缺字段=关
    ("""  // [kvoff-toggle 09-22] CPU KV 二级缓存：inner 脚本按 FN_KVOFF(缺省开)/FN_KVOFF_BYTES 决定。
  // 弹窗显式传 '0'/'1'；旧快启预设无 kvoff 字段 → 落到开启（与 inner 缺省一致，行为不漂移）。
  if (String(d.kvoff) === '0') {
    env.FN_KVOFF = '0';
  } else {
    env.FN_KVOFF = '1';
    const koG = int(d.kvoffGiB, 96);
    if (koG > 0) env.FN_KVOFF_BYTES = String(koG * 1073741824);
  }""",
     """  // [kvoff-toggle 09-22][kvoff-off 0929] CPU KV 二级缓存：inner 脚本按 FN_KVOFF(缺省关)/FN_KVOFF_BYTES 决定。
  // 弹窗显式传 '0'/'1'；字段缺失 → 关闭（与 inner 缺省 :-0 一致；0929 A/B 定案后翻转）。
  if (String(d.kvoff) === '1') {
    env.FN_KVOFF = '1';
    const koG = int(d.kvoffGiB, 96);
    if (koG > 0) env.FN_KVOFF_BYTES = String(koG * 1073741824);
  } else {
    env.FN_KVOFF = '0';
  }"""),
]
n1 = patch(SJ, pairs)
if n1:
    r = subprocess.run(["node", "--check", SJ], capture_output=True, text=True)
    if r.returncode != 0:
        print("[FATAL] node --check 失败，回滚: %s" % r.stderr[:300])
        shutil.copy2(SJ + BAK, SJ)
        sys.exit(1)
    print("node --check OK")

print("== index.html ==")
n2 = patch(IH, [
    ("kvOff: '1', kvOffGiB: Number(smd.kvOffGiB) || 96,",
     "kvOff: '0', kvOffGiB: Number(smd.kvOffGiB) || 96, // 0929 A/B 定案：缺省关"),
])
print("server.js 改 %d 处 / index.html 改 %d 处" % (n1, n2))
print("DONE")
