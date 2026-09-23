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

# Flash-Next W4A16 (18420) 看门狗 —— 2026-09-23 上线
# 背景：机器反复重启 + 外部自动化脚本频繁误杀实例，18420 被杀后完全不自愈。
# 逻辑：每 30s 探 /health
#   200                  -> 清零失败计数，退出
#   非200 但进程还在     -> 视为启动中/停止中，不动
#   非200 且无实例进程   -> 连续 2 次（约 60s）确认离线 -> 清残留 -> 等显存归零 -> 拉起
# 启动参数与 2026-09-23 14:49 手动启动逐字一致（1M 档 + MTP4 + PLE INT8 heap + KVOFF 关）。
set -u
PORT=18420
LOG=/home/ll/deploy/fnx-watchdog.log
STATE=/tmp/fnx-watchdog-fail
LOCK=/tmp/fnx-watchdog.lock
LAUNCH_LOG=/home/ll/deploy/vllm-flash-next-w4a16.log
NVSMI=/usr/bin/nvidia-smi

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

mkdir -p "$(dirname "$LOCK")" 2>/dev/null || true
exec 9>"$LOCK" || exit 0
flock -n 9 || exit 0

# 实例进程判据：主进程 comm 恒为 python3，只能按 cmdline 匹配（括号防自匹配）；
# worker/engine 经 setproctitle 改名为 VLLM::*（comm 被截断 15 字符）。
vllm_alive(){
  ps -eo args= 2>/dev/null | grep -q "[e]ntrypoints[./]cli[./]main serve" && return 0
  ps -eo comm= 2>/dev/null | grep -q "^VLLM::" && return 0
  return 1
}
inner_alive(){ ps -eo args= 2>/dev/null | grep -q "[f]lash-next-w4a16-inner.sh"; }

code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "http://127.0.0.1:${PORT}/health" 2>/dev/null)
[ -z "$code" ] && code=000

if [ "$code" = "200" ]; then
  if [ -f "$STATE" ]; then log "health=200 恢复，清零失败计数"; rm -f "$STATE"; fi
  exit 0
fi

if inner_alive || vllm_alive; then
  exit 0
fi

n=$(cat "$STATE" 2>/dev/null || echo 0)
case "$n" in ''|*[!0-9]*) n=0;; esac
n=$((n+1)); echo "$n" > "$STATE"
log "health=$code 且无实例进程（连续第 $n 次）"
[ "$n" -lt 2 ] && exit 0

log "确认离线 -> 清理残留"
bash /home/ll/deploy/stop-flash-next-w4a16.sh >> "$LOG" 2>&1

used=0
for i in $(seq 1 30); do
  used=$("$NVSMI" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  [ -z "$used" ] && used=0
  [ "$used" -lt 500 ] && break
  sleep 2
done
if [ "$used" -ge 500 ]; then
  log "显存未归零（${used}MiB），本轮不启动，等下一轮"
  exit 0
fi

log "显存已归零 -> 拉起实例（1M + MTP4 + PLE INT8 heap + KVOFF=0）"
cd /home/ll/deploy && env \
  FN_MODEL_PATH=/media/ll/data/models-1m/Qwen3.8-Flash-Next-W4A16-AutoRound-1M \
  FN_LONGCTX=1 FN_1M_MODEL_PATH=/media/ll/data/models-1m/Qwen3.8-Flash-Next-W4A16-AutoRound-1M FN_YARN_FACTOR=4.0 \
  FN_MAXLEN=1048576 FN_GPUMEM=0.96 FN_SEQS=3 FN_MBTOKENS=8192 FN_BLOCK=1616 \
  FN_SPEC=mtp4 FN_ASYNC=1 FN_PLE_INT8=1 FN_PLE_LOC=heap FN_KVOFF=1 FN_KVOFF_BYTES=68719476736 \
  FN_SERVED=qwen3.8-flash-next FN_PORT=18420 \
  setsid nohup bash /home/ll/deploy/start-flash-next-w4a16.sh >> "$LAUNCH_LOG" 2>&1 < /dev/null &
rm -f "$STATE"
log "启动已发起（约 6~8 分钟就绪）"
exit 0
