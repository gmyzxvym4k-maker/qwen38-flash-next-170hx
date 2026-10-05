#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
heap→disk 低内存强制钳位（126 机 32GB 内存，2026-10-05 第二层修复）
背景：pagefix-1005 修好了弹窗默认值，但用户浏览器里可能还挂着 **补丁前加载的旧页面**
（前端 JS 写死 pleLoc:'heap'），或有人纯手动传 FN_PLE_LOC=heap——旧页面提交的
启动请求仍会带着 heap 打进来，32GB 机上 PLE 表(48.3GiB 匿名堆)必被 oom-killer 杀
（10-05 实锤：08:58/09:01/09:04/09:2x 四连 OOM → EngineDead → 看门狗循环）。
本补丁在 **后端 plan** 与 **inner 脚本** 两处加机器内存判据：
  MemTotal < 100GiB 且请求 heap → 强制降级 disk（可回收页缓存），并打警告。
纯手动 `FN_PLE_LOC=heap bash start-...` 同样被 inner 兜住。
幂等标记：__pleclamp_1005__
"""
import os, shutil, subprocess

MARK = "__pleclamp_1005__"
SVR = "/home/ll/deploy/server.js"
INN = "/home/ll/deploy/flash-next-w4a16-inner.sh"


def patch(path, old, new, check=None):
    s = open(path, encoding="utf-8").read()
    if old not in s and new in s:
        print("[skip] %s" % path)
        return
    assert s.count(old) == 1, "锚点在 %s 出现 %d 次:\n%r" % (path, s.count(old), old[:160])
    if not os.path.exists(path + ".bak-pleclamp-1005"):
        shutil.copy2(path, path + ".bak-pleclamp-1005")
    s2 = s.replace(old, new)
    tmp = path + ".tmp-pleclamp" + (".js" if check == "node" else ".sh")
    open(tmp, "w", encoding="utf-8").write(s2)
    if check == "node":
        assert subprocess.run(["node", "--check", tmp]).returncode == 0, "node --check 失败"
    elif check == "bash":
        assert subprocess.run(["bash", "-n", tmp]).returncode == 0, "bash -n 失败"
    os.replace(tmp, path)
    print("[ok] %s" % path)


# ── 1) server.js：scriptModelLaunchPlan 里 FN_PLE_LOC 赋值处加钳位 ──
SVR_OLD = "  env.FN_PLE_LOC = String(d.pleLoc) === 'heap' ? 'heap' : 'disk';"
SVR_NEW = """  env.FN_PLE_LOC = String(d.pleLoc) === 'heap' ? 'heap' : 'disk';
  // [pleclamp-1005] 低内存机防线：heap=48.3GiB(INT8)/95.4GiB(BF16) 匿名堆不可回收，
  // MemTotal <100GiB 的机器（本机 32GB）必被 oom-killer 杀 → EngineDead → 看门狗循环。
  // 旧缓存页面/手选/旧预设传来的 heap 一律在此强制降级 disk（inner 侧第二道兜底同判据）。
  if (env.FN_PLE_LOC === 'heap' && require('os').totalmem() < 100 * 1024 * 1024 * 1024) {
    env.FN_PLE_LOC = 'disk';
    warnings.push('本机物理内存 <100GiB，「PLE 表位置=放内存」会被 OOM 杀（历史四连 oom-kill）——已自动降级为「放硬盘」（mmap 可回收页缓存）');
  }"""
patch(SVR, SVR_OLD, SVR_NEW, check="node")

# ── 2) inner：heap 判据前加同款钳位（覆盖手动命令/看门狗按 envfile 重拉） ──
INN_OLD = """# 精度定精度、位置定内存/硬盘，两者正交（四种组合都成立）：
#   INT8+内存 = VLLM_PLE_INT8_MEMORY 匿名堆 48.3GiB（[FN-PLE-INT8MEM] 引擎侧新增）
if [ "$FN_PLE_LOC" = "heap" ]"""
INN_NEW = """# [pleclamp-1005] 低内存机防线（与 server.js plan 同判据的第二道兜底）：
# heap 匿名堆 48.3GiB(INT8)/95.4GiB(BF16) 不可回收，MemTotal <100GiB 必触发 oom-killer
# （10-05 该机四连 OOM→PLE worker exited→EngineDead→看门狗循环）。强制降级 disk。
if [ "$FN_PLE_LOC" = "heap" ]; then
  MEM_KIB=$(awk '/MemTotal/{print $2}' /proc/meminfo)
  if [ "${MEM_KIB:-0}" -lt $((100 * 1024 * 1024)) ]; then
    echo "[FN-PLE-CLAMP] MemTotal=$((MEM_KIB / 1048576))GiB <100GiB：heap 会被 OOM 杀 -> 强制降级 disk（可回收页缓存）" >&2
    FN_PLE_LOC=disk
  fi
fi
# 精度定精度、位置定内存/硬盘，两者正交（四种组合都成立）：
#   INT8+内存 = VLLM_PLE_INT8_MEMORY 匿名堆 48.3GiB（[FN-PLE-INT8MEM] 引擎侧新增）
if [ "$FN_PLE_LOC" = "heap" ]"""
patch(INN, INN_OLD, INN_NEW, check="bash")

# 校验 inner 里 CLAMP 块位于「兼容缺省推导」之后（FN_PLE_LOC 未传时先推导再钳位，
# 推导出的 heap（BF16 兼容路径）在小内存机同样被钳到 disk？——不：BF16+heap 兼容语义
# 属于显式回滚命令，降级 disk 同样能起（BF16 mmap 零堆），语义安全。
print("DONE")
