#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[1006 w4a16-restore] 幂等修复：本机（HUANANZHI X99-T8 / E5-2696 v4 / 251GiB 内存 / 2×CMP 170HX）
被 10-05 那批「换装 EPYC+3 卡+32GiB」的配置文件残留描述成三卡 PP3 + w8a8 生产模型，
导致 8889 控制台/看门狗/预设一按下去就是 FN_PP=3（只有 2 张卡 → Worker 起不来）。

本脚本一次收口六处：
  A server.js  SCRIPT_MODELS['qwen3.8-flash-next-w4a16']：模型目录/1M 副本/base.pp/base.pleInt8/note
  B server.js  plan 下发 FN_PLE_INT8_DIR（新增 pleInt8Dir 字段，缺=/media/ll/data/ple）
  C quickstart-presets.json 两档预设：gpuCount 3→2、pleInt8 1→0、名称/备注纠偏
  D wrapper start-flash-next-w4a16.sh：RA_LOAD 缺省 16→128（本机 251GiB 内存，服务期零 I/O）
  E udev 99-nvme-readahead.rules：16→128 + udevadm trigger
  F fnx-18420-watchdog.sh 内置回退档 FN_PLE_INT8=1→0；删除 fnx-manual-stop 闩锁（恢复自愈）

用法：python3 patch-w4a16-restore-1006.py [--revert]
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
WRAPPER = f"{D}/start-flash-next-w4a16.sh"
UDEV = "/etc/udev/rules.d/99-nvme-readahead.rules"
WATCHDOG = f"{D}/fnx-18420-watchdog.sh"
LATCH = f"{D}/fnx-manual-stop"
MARK = "w4a16-restore-1006"

W4A16_BASE = "/media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound"
W4A16_1M = "/media/ll/data/models-1m/Qwen3.8-Flash-Next-W4A16-AutoRound-1M"
W8A8_BASE = "/media/ll/data/models/Qwen3.8-Flash-Next-Channel-INT8-w8a8"
W8A8_1M = "/media/ll/data/models-1m/Qwen3.8-Flash-Next-Channel-INT8-w8a8-1M"
PLE_INT8_DIR = "/media/ll/data/ple"

revert = "--revert" in sys.argv
log = lambda *a: print("[patch]", *a)


def backup(path):
    b = path + f".bak-{MARK}"
    if os.path.exists(path) and not os.path.exists(b):
        shutil.copy2(path, b)
        log("备份", b)


def sudo(args, **kw):
    """sudo -S -p ''；口令只从环境变量 SUDO_PASS 取（仓库内不留明文）。

    若目标机已配 NOPASSWD sudoers（推荐），SUDO_PASS 可为空串。
    """
    pw = os.environ.get("SUDO_PASS", "")
    stdin = (pw + "\n") if pw else ""
    p = subprocess.run(["sudo"] + (["-S", "-p", ""] if pw else []) + args,
                       input=stdin, capture_output=True, text=True, **kw)
    return p


# ---------------------------------------------------------------- server.js
def patch_server():
    src = open(SERVER, encoding="utf-8").read()
    changed = []
    if revert:
        b = SERVER + f".bak-{MARK}"
        if os.path.exists(b):
            shutil.copy2(b, SERVER)
            log("server.js 已回滚")
        return

    # A1 dirNames 主映射
    old = "  dirNames: ['Qwen3.8-Flash-Next-Channel-INT8-w8a8', 'Qwen3.8-Flash-Next-W4A16-AutoRound'],  // [w8a8-1005] 主映射=w8a8 生产模型（W4A16 目录保留兼容其重下完成后的手动档）"
    new = f"  dirNames: ['Qwen3.8-Flash-Next-W4A16-AutoRound', 'Qwen3.8-Flash-Next-Channel-INT8-w8a8'],  // [{MARK}] 主映射=W4A16-AutoRound（重下完成+验收通过，10-06 切回；w8a8 目录保留兼容）"
    if new in src:
        log("server.js dirNames 已切换")
    elif src.count(old) == 1:
        src = src.replace(old, new); changed.append("dirNames")
    else:
        raise SystemExit(f"[fail] dirNames 锚点数={src.count(old)}")

    # A2 modelPath
    old = f"  modelPath: '{W8A8_BASE}',"
    new = f"  modelPath: '{W4A16_BASE}',  // [{MARK}]"
    if new in src:
        log("server.js modelPath 已切换")
    elif src.count(old) == 1:
        src = src.replace(old, new); changed.append("modelPath")
    else:
        raise SystemExit(f"[fail] modelPath 锚点数={src.count(old)}")

    # A3 altModelPaths / longCtxModelPath
    old = f"  altModelPaths: ['{W8A8_1M}', '{W4A16_1M}'],  // [w8a8-1005] w8a8 无 512K 副本，512 档已隐藏"
    new = f"  altModelPaths: ['{W4A16_1M}'],  // [{MARK}] 1M=YaRN×4 W4A16 副本"
    if new in src:
        log("server.js altModelPaths 已切换")
    elif src.count(old) == 1:
        src = src.replace(old, new); changed.append("altModelPaths")
    else:
        raise SystemExit(f"[fail] altModelPaths 锚点数={src.count(old)}")

    old = f"  longCtxModelPath: '{W8A8_1M}',"
    new = f"  longCtxModelPath: '{W4A16_1M}',  // [{MARK}]\n  pleInt8Dir: '{PLE_INT8_DIR}',  // [{MARK}] W4A16 专属 INT8 表产物目录（不存在则 inner 回落 BF16 磁盘驻留）；禁止跨 checkpoint 复用（inner 缺省是 ple-w8a8）"
    if new in src:
        log("server.js longCtxModelPath+pleInt8Dir 已切换")
    elif src.count(old) == 1:
        src = src.replace(old, new); changed.append("longCtxModelPath+pleInt8Dir")
    else:
        raise SystemExit(f"[fail] longCtxModelPath 锚点数={src.count(old)}")

    # A4 base.pp 3 -> 2（本机现 2 卡在位：GPU0=03:00 / GPU1=04:00）
    old = "presencePenalty: 0.2, repetitionPenalty: 1.15, pp: 3, mtpTokens: 4,"
    new = "presencePenalty: 0.2, repetitionPenalty: 1.15, pp: 2, mtpTokens: 4,  // [w4a16-restore-1006] 本机在位 2×CMP 170HX（lspci 仅 03:00/04:00），PP 基准 3→2；三卡机勿套此值"
    if new in src:
        log("server.js base.pp 已是 2")
    elif src.count(old) == 1:
        src = src.replace(old, new); changed.append("base.pp")
    else:
        raise SystemExit(f"[fail] base.pp 锚点数={src.count(old)}")

    # A5 base.pleInt8 1 -> 0（W4A16 的 INT8 产物未生成，251GiB 内存走 BF16 磁盘驻留）
    old = "kvoff: '0', pleInt8: '1', pleLoc: 'disk', // 1005 生产定版=二级缓存关 + PLE INT8 硬盘驻留"
    new = ("kvoff: '0', pleInt8: '0', pleLoc: 'disk', // [w4a16-restore-1006] 二级缓存关 + PLE BF16 硬盘驻留"
           "\n    //   （W4A16 的 INT8 n-gram 表产物 /media/ll/data/ple 尚未生成；ple-w8a8 那份属另一 checkpoint，"
           "\n    //     跨 checkpoint 复用会内容对/标点乱，禁止。要 INT8 档先跑 quantize_ple.py 生成 ple/ 再切 pleInt8=1）")
    if new in src:
        log("server.js base.pleInt8 已是 0")
    elif src.count(old) == 1:
        src = src.replace(old, new); changed.append("base.pleInt8")
    else:
        raise SystemExit(f"[fail] base.pleInt8 锚点数={src.count(old)}")

    # A6 note
    old = "note: 'chroot 镜像 PP3 脚本启动（现行生产=Channel-INT8-w8a8，1M=YaRN×4 副本），加载约 3~9 分钟',"
    new = ("note: 'chroot 镜像 TP1×PP2 脚本启动（现行生产=W4A16-AutoRound，1M=YaRN×4 副本，"
           "PLE BF16 磁盘驻留），加载约 5~11 分钟',  // [w4a16-restore-1006]")
    if new in src:
        log("server.js note 已是新值")
    elif src.count(old) == 1:
        src = src.replace(old, new); changed.append("note")
    else:
        raise SystemExit(f"[fail] note 锚点数={src.count(old)}")

    # B plan 下发 FN_PLE_INT8_DIR
    old = "  const env = { FN_MODEL_PATH: sm.modelPath };"
    new = ("  const env = { FN_MODEL_PATH: sm.modelPath };\n"
           "  if (sm.pleInt8Dir) env.FN_PLE_INT8_DIR = sm.pleInt8Dir;  // [w4a16-restore-1006] 防落 inner 缺省(w8a8 表)")
    if "sm.pleInt8Dir) env.FN_PLE_INT8_DIR" in src:
        log("server.js plan 已下发 FN_PLE_INT8_DIR")
    elif src.count(old) == 1:
        src = src.replace(old, new); changed.append("plan-env")
    else:
        raise SystemExit(f"[fail] plan 锚点数={src.count(old)}")

    if changed:
        backup(SERVER)
        open(SERVER, "w", encoding="utf-8").write(src)
        log("server.js 已修改:", ",".join(changed))
    else:
        log("server.js noop")

    # 语法门禁：临时文件必须以 .js 结尾（本项目已两次踩坑）
    tmp = "/tmp/_check-w4a16-restore.js"
    shutil.copy2(SERVER, tmp)
    p = subprocess.run(["node", "--check", tmp], capture_output=True, text=True)
    os.remove(tmp)
    if p.returncode != 0:
        raise SystemExit("[fail] node --check 不通过：\n" + p.stderr[-1200:])
    log("server.js 语法校验通过")


# ------------------------------------------------------ quickstart-presets
def patch_presets():
    if revert:
        b = PRESETS + f".bak-{MARK}"
        if os.path.exists(b):
            shutil.copy2(b, PRESETS); log("预设已回滚")
        return
    d = json.load(open(PRESETS, encoding="utf-8"))
    card = d.get("flashnext")
    if not card:
        raise SystemExit("[fail] quickstart-presets.json 无 flashnext 卡")
    backup(PRESETS)
    for p in card.get("presets", []):
        pr = p.setdefault("params", {})
        pr["gpuCount"] = "2"
        pr["parallelMode"] = "pp"
        pr["pleInt8"] = "0"
        pr["pleLoc"] = "disk"
        pr["kvoff"] = "0"
        k = p.get("key", "")
        if k.startswith("current-pp3-1m-mtp4"):
            pr.update({
                "blockSize": "1616", "mtp": "1", "mtpTokens": "4",
                "gpuMemUtil": "0.95", "maxNumSeqs": "2",
                "maxModelLen": "1048576", "longCtx": "1m",
                "thinking": "1", "thinkingEffort": "medium",
                "temperature": "1", "topP": "0.95", "topK": "20",
                "minP": "0.0", "presencePenalty": "0", "repetitionPenalty": "1",
                "prefixCaching": "1", "chunkedPrefill": "1",
                "asyncScheduling": "1", "enforceEager": "0",
            })
            p["name"] = "当前固化-双卡PP2-1M-MTP4-PLE-BF16硬盘驻留-无二级缓存（10-06 实跑复刻）"
            p["note"] = (p.get("note", "") + " 【10-06 纠偏】本机在位 2×CMP 170HX（GPU 03:00/04:00），"
                         "原「三卡 PP3」预设属 10-05 换装机描述，按下去必起不来，已改 PP2；"
                         "PLE 精度回 BF16（W4A16 的 INT8 表产物未生成，ple-w8a8 那份禁止跨 checkpoint 用）；"
                         "max-num-seqs 2 / 显存 0.95 / 采样 t1.0·pp0·rp1（逐键复刻当日 launch.env；"
                         "⚠ 采样为裸档，与 09-29 反循环定档 0.6/0.2/1.15 不同，如出现 uct/duct 硬循环改回定档）。")
        elif k.startswith("current-pp3-1m-mtp5"):
            pr["gpuMemUtil"] = "0.97"
            p["name"] = "固化-双卡PP2-1M-MTP5-GPU97%-PLE-BF16硬盘驻留"
            p["note"] = (p.get("note", "") + " 【10-06 纠偏】PP 档随本机改为 2（原 3 卡档）；"
                         "block 1632 与 MTP5 的 QSA ring=12 合法（1632/12=136）。")
    json.dump(d, open(PRESETS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log("预设已更新：", ", ".join(p.get("key", "?") for p in card["presets"]))


# ------------------------------------------------------------------ wrapper
def patch_wrapper():
    src = open(WRAPPER, encoding="utf-8").read()
    if revert:
        b = WRAPPER + f".bak-{MARK}"
        if os.path.exists(b):
            shutil.copy2(b, WRAPPER); log("wrapper 已回滚")
        return
    if f"RA_LOAD=${{RA_LOAD:-128}}  # [{MARK}]" in src:
        log("wrapper RA_LOAD 已是 128")
        return
    m = re.search(r"^RA_LOAD=\$\{RA_LOAD:-\d+\}.*$", src, re.M)
    if not m:
        raise SystemExit("[fail] wrapper 未找到 RA_LOAD 行")
    backup(WRAPPER)
    new = ("RA_LOAD=${RA_LOAD:-128}  # [w4a16-restore-1006] 本机 251GiB 内存：PLE 表(95.4GiB)+权重(79GiB) "
           "可全驻留页缓存，服务期盘读≈0，RA 无读放大代价；09-23 实测 128KB 即到拐点（0→297 / 128→3011 MB/s），"
           "RA=16 会把加载拖慢数倍（16 是 10-05 那台 32GiB 三卡机的降预读档，不适用于本机）")
    src = src[:m.start()] + new + src[m.end():]
    open(WRAPPER, "w", encoding="utf-8").write(src)
    log("wrapper RA_LOAD 16→128")
    p = subprocess.run(["bash", "-n", WRAPPER], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("[fail] wrapper 语法错：" + p.stderr[-600:])


# ------------------------------------------------------------------- udev
def data_disk_name():
    p = subprocess.run("readlink -f /dev/disk/by-id/nvme-JZ-SSD2T-XW_* | head -1",
                       shell=True, capture_output=True, text=True).stdout.strip()
    return os.path.basename(p)  # 例 nvme0n1（盘符重启会互换，故按型号解析）


def patch_udev():
    name = data_disk_name()
    cur = open(UDEV, encoding="utf-8").read() if os.path.exists(UDEV) else ""
    want = re.sub(r'read_ahead_kb\}="\d+"', 'read_ahead_kb}="128"', cur)
    if want != cur:
        tmp = "/tmp/_99-nvme-readahead.new"
        open(tmp, "w", encoding="utf-8").write(want)
        # sudo -S 的 stdin 已被密码占用，不能再用 tee 喂正文 -> 先落临时文件再 sudo cp
        p = sudo(["sh", "-c",
                  f"cp -a {UDEV} {UDEV}.bak-{MARK} && cp {tmp} {UDEV} && chmod 644 {UDEV}"])
        if p.returncode != 0:
            log("udev 规则写入失败（sudo）:", p.stderr[-200:])
        else:
            log(f"udev 规则 read_ahead_kb 已置 128（原备份 {UDEV}.bak-{MARK}）")
    if name:
        p = sudo(["sh", "-c",
                  f"udevadm control --reload-rules && udevadm trigger --action=change /sys/block/{name}"])
        if p.returncode != 0:
            log("udevadm trigger 失败:", p.stderr[-200:])
        ra = subprocess.run(f"cat /sys/block/{name}/queue/read_ahead_kb",
                            shell=True, capture_output=True, text=True).stdout.strip()
        log(f"数据盘 {name} 当前 read_ahead_kb =", ra)
    else:
        log("[warn] 未按 by-id 找到数据盘，跳过 trigger")


# ---------------------------------------------------------------- watchdog
def patch_watchdog():
    src = open(WATCHDOG, encoding="utf-8").read()
    if revert:
        b = WATCHDOG + f".bak-{MARK}"
        if os.path.exists(b):
            shutil.copy2(b, WATCHDOG); log("看门狗已回滚")
        return
    old = '[ "$STACK_DISABLED" = "1" ] && export FN_PLE_INT8=1 FN_PLE_LOC=disk FN_KVOFF=0'
    new = '[ "$STACK_DISABLED" = "1" ] && export FN_PLE_INT8=0 FN_PLE_LOC=disk FN_KVOFF=0  # [w4a16-restore-1006] W4A16 INT8 产物未生成→回退档也走 BF16 磁盘驻留'
    if new in src:
        log("看门狗回退档已是 BF16")
    elif src.count(old) == 1:
        backup(WATCHDOG)
        open(WATCHDOG, "w", encoding="utf-8").write(src.replace(old, new))
        log("看门狗回退档 FN_PLE_INT8 1→0")
    else:
        log("[warn] 看门狗回退档锚点数=", src.count(old), "，跳过")
    p = subprocess.run(["bash", "-n", WATCHDOG], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("[fail] 看门狗语法错：" + p.stderr[-600:])


def remove_latch():
    if revert:
        open(LATCH, "a").close()
        log("已恢复 fnx-manual-stop 闩锁")
        return
    if os.path.exists(LATCH):
        os.remove(LATCH)
        log("已删除 fnx-manual-stop 闩锁（看门狗自愈恢复生效）")
    else:
        log("闩锁本就不在")


def main():
    if revert:
        log("=== 回滚模式 ===")
        patch_server(); patch_presets(); patch_wrapper(); patch_watchdog(); remove_latch()
        log("回滚完成（需自行 systemctl --user restart dsh-console）")
        return
    log("=== 应用模式 ===")
    patch_server()
    patch_presets()
    patch_wrapper()
    patch_udev()
    patch_watchdog()
    remove_latch()
    log("全部完成")


main()
