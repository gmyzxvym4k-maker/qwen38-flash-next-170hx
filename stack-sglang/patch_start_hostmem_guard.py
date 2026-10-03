#!/usr/bin/env python3
# 给 start-flash-next-sglang.sh 加 host 内存门禁：
# 重启窗口里旧实例 PLE 锁页(~128GB)未回收即拉起新实例 = OOM killer
# （2026-10-03 09:05 实锤：scheduler exit -9, shmem-rss 119GB）。
# 幂等；--revert 还原。
import shutil
import sys

F = "/home/ll/deploy/sglang-18420/start-flash-next-sglang.sh"

if "--revert" in sys.argv:
    shutil.copy(F + ".bak-hostmem", F)
    print("reverted")
    sys.exit(0)

s = open(F).read()
if "MemAvailable" in s:
    print("already applied")
    sys.exit(0)

shutil.copy(F, F + ".bak-hostmem")
anchor = "ENVFILE=$BASE/launch.env"
assert s.count(anchor) == 1, "anchor"
guard = """# host 内存门禁：可用 <180GB 说明上一实例锁页未回收，最多等 120s（SG_FORCE=1 跳过）
_i=0
while [ "${SG_FORCE:-0}" != "1" ] && [ $_i -lt 24 ]; do
  _av=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$_av" -ge 180 ] && break
  echo "[sg-start] MemAvailable=${_av}GB <180GB（旧实例锁页回收中）等待…"
  sleep 5; _i=$((_i+1))
done
"""
s = s.replace(anchor, guard + anchor, 1)
open(F, "w").write(s)
print("APPLIED")
