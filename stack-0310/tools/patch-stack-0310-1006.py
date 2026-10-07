#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[stack-0310-1006] 幂等：把 8889 控制台与看门狗的「新栈」从官方 vLLM 0.30.0 升到 0.31.0，
并把生产参数（1M/PP2/MTP4/block1616/seqs2 + SimpleCPU 内存二级缓存 96GiB + PLE INT8 mmap）
固化进 SCRIPT_MODELS.base 与快启预设。

层级（自顶向下）：
  sglang-18420/ACTIVE（最高，本栈不用）
    → vllm-0310/DISABLED 在位 ? 退回 chroot 旧栈 : 用 0.31.0 新栈
  回滚 0.30.0 = 把 SCRIPT_MODELS.scriptNew/stopScriptNew 改回 vllm-0300 那份（本脚本 --revert 即做此事）

用法：python3 patch-stack-0310-1006.py [--revert]
"""
import json
import os
import re
import shutil
import subprocess
import sys

D = "/home/ll/deploy"
SERVER = f"{D}/server.js"
PRESETS = f"{D}/quickstart-presets.json"
WATCHDOG = f"{D}/fnx-18420-watchdog.sh"
MARK = "stack-0310-1006"
S310 = f"{D}/vllm-0310"
S300 = f"{D}/vllm-0300"
LOG310 = f"{D}/vllm-flash-next-0310.log"
SENT310 = f"{S310}/DISABLED"
SENT300 = f"{S300}/DISABLED"

revert = "--revert" in sys.argv
log = lambda *a: print("[patch]", *a)


def backup(p):
    b = p + f".bak-{MARK}"
    if os.path.exists(p) and not os.path.exists(b):
        shutil.copy2(p, b)


def node_check(path):
    tmp = "/tmp/_check-stack0310.js"
    shutil.copy2(path, tmp)
    p = subprocess.run(["node", "--check", tmp], capture_output=True, text=True)
    os.remove(tmp)
    if p.returncode != 0:
        raise SystemExit("[fail] node --check 不通过：\n" + p.stderr[-1500:])
    log("node --check 通过")


# ------------------------------------------------------------------ server.js
ENTRY_KEY = "SCRIPT_MODELS['qwen3.8-flash-next-w4a16'] = {"
NEW_VALUES = {
    "scriptNew": f"{S310}/start-flash-next-0310.sh",
    "stopScriptNew": f"{S310}/stop-flash-next-0310.sh",
    "altLogs": f"  altLogs: ['{LOG310}', '/home/ll/deploy/vllm-flash-next-0300.log'],",
    "note": ("  note: '官方 vLLM 0.31.0 运行时补丁栈 TP1×PP2（现行生产=W4A16-AutoRound-1M，"
             "PLE=INT8 磁盘 mmap，内存二级缓存 96GiB），加载约 5~8 分钟；"
             "回滚：touch /home/ll/deploy/vllm-0310/DISABLED → 退回 chroot 旧栈',  // [stack-0310-1006]"),
    "basekv": ("    kvoff: 'simple', kvoffGiB: 96, pleInt8: '1', pleLoc: 'disk',  // [stack-0310-1006] "
               "二级缓存=SimpleCPUOffloadConnector 96GiB（每 rank ≈48GiB 锁页）；"
               "PLE=INT8 产物磁盘 mmap（/media/ll/data/ple，47.7+0.6 GiB 可回收页缓存）。"
               "0.31 只有两档：mmap-INT8 / pinned-BF16——弹窗「精度=BF16」或「位置=heap」都会走官方锁页 BF16 95.4GiB"),
}
OLD_VALUES = {
    "scriptNew": f"{S300}/start-flash-next-0300.sh",
    "stopScriptNew": f"{S300}/stop-flash-next-0300.sh",
    "altLogs": "  altLogs: ['/home/ll/deploy/vllm-flash-next-0300.log'],",
    "note": ("  note: 'chroot 镜像 TP1×PP2 脚本启动（现行生产=W4A16-AutoRound，1M=YaRN×4 副本，"
             "PLE BF16 磁盘驻留），加载约 5~11 分钟',  // [w4a16-restore-1006]"),
    "basekv": ("    kvoff: '0', pleInt8: '0', pleLoc: 'disk', // [w4a16-restore-1006] "
               "二级缓存关 + PLE BF16 硬盘驻留"),
}
STACKFN_OLD = ("function stack0300Active() {\n  try { return !fs.existsSync(STACK0300_DISABLED); } catch (e) { return false; }\n}")
STACKFN_NEW = ("function stack0300Active() {\n"
               "  // [stack-0310-1006] 「新栈」已升到官方 vLLM 0.31.0（vllm-0310）：\n"
               "  // 哨兵换名生效——vllm-0310/DISABLED 在位 = 退回 chroot 旧栈；不在位 = 走 0.31.0。\n"
               "  // （0.30.0 已被 0.31.0 取代，其 DISABLED 不再参与判定；回滚 0.30 用本补丁 --revert。）\n"
               "  try { return !fs.existsSync('/home/ll/deploy/vllm-0310/DISABLED'); } catch (e) { return false; }\n}")


def entry_span(src):
    i = src.index(ENTRY_KEY)
    j = src.index("\n};", i) + 3
    return i, j


def patch_server():
    src = open(SERVER, encoding="utf-8").read()
    V = OLD_VALUES if revert else NEW_VALUES
    ch = []
    i0, i1 = entry_span(src)
    e = src[i0:i1]
    # 1) scriptNew / stopScriptNew
    for key in ("scriptNew", "stopScriptNew"):
        pat = re.compile("^  " + re.escape(key) + ": '.*',?$", re.M)
        if not pat.search(e):
            raise SystemExit(f"[fail] 条目内缺 {key}")
        new_line = f"  {key}: '{V[key]}',"
        if pat.search(e).group(0) == new_line:
            log(f"server.js {key} 已是目标值")
        else:
            e = pat.sub(lambda m: new_line, e, count=1); ch.append(key)
    # 2) altLogs
    pat = re.compile(r"^  altLogs: \[.*\],$", re.M)
    if pat.search(e).group(0) == V["altLogs"]:
        log("server.js altLogs 已是目标值")
    else:
        e = pat.sub(lambda m: V["altLogs"], e, count=1); ch.append("altLogs")
    # 3) note（条目内第一处 note）
    pat = re.compile(r"^  note: .*$", re.M)
    if pat.search(e).group(0) == V["note"]:
        log("server.js note 已是目标值")
    else:
        e = pat.sub(lambda m: V["note"], e, count=1); ch.append("note")
    # 4) base 的 kvoff/pleInt8 行 + 其后两行旧注释
    pat = re.compile(r"^    kvoff: .*?\n(?:    //.*\n){0,2}", re.M)
    m = pat.search(e)
    if not m:
        raise SystemExit("[fail] base kvoff 锚点未找到")
    if m.group(0).startswith(V["basekv"]):
        log("server.js base.kvoff 已是目标值")
    else:
        e = e[:m.start()] + V["basekv"] + "\n" + e[m.end():]
        ch.append("base.kvoff/ple")
    # 5) base.maxNumSeqs 对齐实跑（2）
    if not revert:
        e = e.replace("maxModelLen: 262144, gpuMemUtil: 0.95, maxNumSeqs: 4, maxBatchedTokens: 8192,",
                      "maxModelLen: 262144, gpuMemUtil: 0.95, maxNumSeqs: 2, maxBatchedTokens: 8192,  // [stack-0310-1006] 与 18420 实跑一致（1M 档 seqs2）", 1)
    src = src[:i0] + e + src[i1:]
    # 6) 栈哨兵函数
    fn_new = STACKFN_OLD if revert else STACKFN_NEW
    fn_old = STACKFN_NEW if revert else STACKFN_OLD
    if fn_new in src:
        log("server.js stack0300Active 已是目标实现")
    elif src.count(fn_old) == 1:
        src = src.replace(fn_old, fn_new); ch.append("stack0300Active")
    else:
        log("[warn] stack0300Active 锚点数=", src.count(fn_old))
    if ch:
        backup(SERVER)
        open(SERVER, "w", encoding="utf-8").write(src)
        node_check(SERVER)
        log("server.js 已修改:", ",".join(ch))
    else:
        log("server.js noop")


# -------------------------------------------------------------- watchdog
def patch_watchdog():
    if revert:
        b = WATCHDOG + f".bak-{MARK}"
        if os.path.exists(b):
            shutil.copy2(b, WATCHDOG); log("看门狗已回滚")
        return
    src = open(WATCHDOG, encoding="utf-8").read()
    if MARK in src:
        log("看门狗已改过"); return
    backup(WATCHDOG)
    old_sel = """BASE_NEW=/home/ll/deploy/vllm-0300
STACK_DISABLED=0
[ -f "$BASE_NEW/DISABLED" ] && STACK_DISABLED=1
if [ "$STACK_DISABLED" = "0" ]; then
  START_SCRIPT=$BASE_NEW/start-flash-next-0300.sh
  STOP_SCRIPT=$BASE_NEW/stop-flash-next-0300.sh
  ENVF=$BASE_NEW/launch.env
  INNER_PAT='flash-next-0300-inner.sh'
else"""
    new_sel = """# [stack-0310-1006] 栈阶梯：0.31.0（vllm-0310）→ chroot 旧栈（vllm-0310/DISABLED 在位时）
BASE_NEW=/home/ll/deploy/vllm-0310
BASE_MID=/home/ll/deploy/vllm-0300
STACK_DISABLED=0
if [ ! -x "$BASE_NEW/start-flash-next-0310.sh" ] || [ -f "$BASE_NEW/DISABLED" ]; then
  STACK_DISABLED=1
fi
if [ "$STACK_DISABLED" = "0" ]; then
  START_SCRIPT=$BASE_NEW/start-flash-next-0310.sh
  STOP_SCRIPT=$BASE_NEW/stop-flash-next-0310.sh
  ENVF=$BASE_NEW/launch.env
  INNER_PAT='flash-next-0310-inner.sh'
elif [ -x "$BASE_MID/start-flash-next-0300.sh" ] && [ ! -f "$BASE_MID/DISABLED" ]; then
  START_SCRIPT=$BASE_MID/start-flash-next-0300.sh
  STOP_SCRIPT=$BASE_MID/stop-flash-next-0300.sh
  ENVF=$BASE_MID/launch.env
  INNER_PAT='flash-next-0300-inner.sh'
else"""
    assert src.count(old_sel) == 1, "看门狗栈选择块锚点未找到"
    src = src.replace(old_sel, new_sel)
    old_log = "for f in /home/ll/deploy/vllm-flash-next-0300.log /home/ll/deploy/vllm-flash-next-w4a16.log; do"
    new_log = f"for f in {LOG310} /home/ll/deploy/vllm-flash-next-0300.log /home/ll/deploy/vllm-flash-next-w4a16.log; do"
    if src.count(old_log) == 1:
        src = src.replace(old_log, new_log)
    old_alive = "inner_alive(){ ps -eo args= 2>/dev/null | grep -qE \"[f]lash-next-(0300|w4a16)-inner\\.sh\"; }"
    new_alive = "inner_alive(){ ps -eo args= 2>/dev/null | grep -qE \"[f]lash-next-(0310|0300|w4a16)-inner\\.sh\"; }"
    if src.count(old_alive) == 1:
        src = src.replace(old_alive, new_alive)
    open(WATCHDOG, "w", encoding="utf-8").write(src)
    p = subprocess.run(["bash", "-n", WATCHDOG], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("[fail] 看门狗语法错：" + p.stderr[-600:])
    log("看门狗已改：栈阶梯含 0.31.0")


# --------------------------------------------------------------- presets
def patch_presets():
    if revert:
        b = PRESETS + f".bak-{MARK}"
        if os.path.exists(b):
            shutil.copy2(b, PRESETS); log("预设已回滚")
        return
    d = json.load(open(PRESETS, encoding="utf-8"))
    card = d.get("flashnext")
    if not card:
        raise SystemExit("[fail] 无 flashnext 卡")
    backup(PRESETS)
    for p in card.get("presets", []):
        pr = p.setdefault("params", {})
        k = p.get("key", "")
        if k.startswith("current-pp2-1m-mtp4") or k.startswith("current-pp3-1m-mtp4"):
            pr["kvoff"] = "simple"; pr["kvoffGiB"] = "96"
            pr["pleInt8"] = "1"; pr["pleLoc"] = "disk"
            pr["gpuCount"] = "2"; pr["parallelMode"] = "pp"
            pr["maxNumSeqs"] = "2"; pr["blockSize"] = "1616"
            pr["mtp"] = "1"; pr["mtpTokens"] = "4"; pr["gpuMemUtil"] = "0.95"
            p["name"] = "当前固化-双卡PP2-1M-MTP4-二级缓存96G-PLE-INT8磁盘驻留（10-06 vLLM 0.31.0 验收配置）"
            p["note"] = ("【10-06 定版】官方 vLLM 0.31.0 运行时补丁栈（/home/ll/deploy/vllm-0310）。"
                         "与 chroot 旧栈逐键同源：1M(YaRN×4)+PP2(26,22)+block1616+MTP4+seqs2+mbt8192+gpu0.95"
                         "+async+前缀缓存+思考 medium+采样 t1.0·pp0·rp1.0；"
                         "新增内存二级缓存=SimpleCPUOffloadConnector 96GiB（PP2 每 rank ≈48GiB 锁页，"
                         "rt-patch #13 提供 PP 下 CPU 块 id 握手 clamp+逐块拷贝守卫）。"
                         "PLE=INT8 表磁盘 mmap（/media/ll/data/ple，47.7+0.6 GiB 可回收页缓存）。"
                         "验收（19:47 窗口）：挤池 1,434,381 token 后重发 55,927-token 文档 →"
                         " cached=53,328（95.4%）全部由内存档回载、验证码 CODE-287365-3126 逐字命中、Xid=0。"
                         "⚠ 采样仍是裸档（复刻 10-06 实跑），若出现 uct/duct 硬循环改回 0.6/0.2/1.15 定档。")
        elif k.startswith("current-pp3-1m-mtp5") or k.startswith("current-pp2-1m-mtp5"):
            pr["kvoff"] = "0"
            p["name"] = "对照档-双卡PP2-1M-MTP5-GPU97%-无二级缓存-PLE-INT8磁盘驻留"
            p["note"] = (p.get("note", "") + " 【10-06】作为「关掉内存二级缓存」的对照档保留：block 1632 对 MTP5 合法"
                         "（ring=4*cdiv(4+5,4)=12，1632/12=136）。")
    json.dump(d, open(PRESETS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log("预设已更新（二级缓存 96GiB 固化进当前档）")


def sentinels():
    if revert:
        if not os.path.exists(SENT300):
            shutil.copy2(f"{S300}/DISABLED.bak-superseded-1006", SENT300) if os.path.exists(
                f"{S300}/DISABLED.bak-superseded-1006") else None
            log("已恢复 vllm-0300/DISABLED")
        return
    if os.path.exists(SENT300):
        shutil.move(SENT300, f"{S300}/DISABLED.bak-superseded-1006")
        log("vllm-0300/DISABLED 已移开（0.30.0 被 0.31.0 取代）→ 控制台/看门狗走 0.31.0")
    else:
        log("vllm-0300/DISABLED 本就不在位")


def main():
    log("回滚模式" if revert else "应用模式")
    patch_server()
    patch_watchdog()
    patch_presets()
    sentinels()
    log("完成。server.js 生效需 systemctl --user restart dsh-console")


main()
