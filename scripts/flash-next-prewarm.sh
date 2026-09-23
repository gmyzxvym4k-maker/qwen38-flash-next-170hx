#!/bin/bash
# ---- sudo 凭据（脱敏版）----
# 生产机上 chroot 内进程属 root，宿主脚本需要 sudo。两种给法二选一：
#   A) 推荐：/etc/sudoers.d/ 配 NOPASSWD（见 docs/01-host-prep.md），无需本变量
#   B) 环境变量 SUDO_PASS 传入密码（不落盘、不写进脚本）
if [ -n "${SUDO_PASS:-}" ]; then
  SUDO() { printf '%s\n' "$SUDO_PASS" | sudo -S -p '' "$@"; }
else
  SUDO() { sudo -n "$@"; }
fi

# PLE n-gram 表页缓存预热（2026-09-19 重建；原 flash-next-ra-watch.sh 在重装中丢失）
#
# 为什么需要：95.4 GiB 的 BF16 表由 PleOffloadWorker 读进【匿名堆内存】（实测 Rss 96.82G /
# Private_Dirty 96.42G），不是页缓存，进程一死就没了；实测服务期 fincore 只剩 25 MiB，
# 于是每次启动都要从盘重读 95.4 GiB（实测那一片单独吃掉 97 秒，约 1.0GB/s）。
# 实例停止后会释放那 96.8 GiB 匿名内存 —— 此时把表读进页缓存，下次启动即缓存命中（~20s）。
#
# 只在「没有实例在跑」时预热：运行期预热会与那 96.8 GiB 堆内存抢 RAM，纯属互相挤掉。
set -u
TABLE="${1:-/media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound/model-00016-of-00017.safetensors}"
LOG=/home/ll/deploy/flash-next-prewarm.log
LOCK=/tmp/flash-next-prewarm.lock
say(){ echo "[$(date -Is)] $*" >> "$LOG"; }

exec 9>"$LOCK"
if ! flock -n 9; then say "已有预热在跑，跳过（互斥）"; exit 0; fi

[ -f "$TABLE" ] || { say "表文件不存在: $TABLE"; exit 1; }
if ps -eo comm | grep -q '^VLLM::'; then say "有实例在跑，跳过预热（避免与堆内存抢 RAM）"; exit 0; fi

SZ=$(stat -c %s "$TABLE")
say "开始预热: $(basename "$TABLE") ($((SZ/1024/1024/1024)) GiB)"
T0=$(date +%s)
python3 - "$TABLE" >> "$LOG" 2>&1 <<'INNERPY'
import sys, os, time
f = sys.argv[1]
CH = 1 << 30
t0 = time.time(); n = 0
with open(f, 'rb') as fh:
    while True:
        b = fh.read(CH)
        if not b:
            break
        n += len(b)
try:
    fd = os.open(f, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
    os.close(fd)
except Exception as e:
    print('  fadvise 跳过:', e)
print("  读完 %.2f GiB，用时 %.1fs" % (n / 2**30, time.time() - t0))
INNERPY
T1=$(date +%s)
say "预热完成，用时 $((T1-T0))s"
# 驻留率自检（有 fincore 就报一下，作为下次启动快慢的判据）
if command -v fincore >/dev/null 2>&1; then
  say "驻留自检: $(fincore --res --bytes "$TABLE" 2>/dev/null | tail -1)"
fi
