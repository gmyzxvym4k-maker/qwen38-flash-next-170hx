#!/bin/bash
# 把 18420 从「kvoff-c8 窗口遗留的挂死测试档」恢复到生产态（官方 0.30.0 栈 + inner 内置定稿参数）。
# 用法（部署机 ll 用户）：setsid nohup bash restore-prod.sh > restore-prod.log 2>&1 &
set -u
LOG=/home/ll/deploy/kvoff-restore-prod.log
log() { echo "[$(date +%H:%M:%S)] $*"; }
exec >>"$LOG" 2>&1

BASE=/home/ll/deploy/vllm-0300
PORT=18420
export XDG_RUNTIME_DIR=/run/user/1000

log "=== 1) 停看门狗 timer（防操作期间自愈抢跑）"
systemctl --user stop fnx-18420-watchdog.timer || true

log "=== 2) 清掉挂死的 c8 窗口/探针（它们是遗留进程，探针已挂 30+ 分钟）"
for pat in kvoff-c8-probe.py kvoff-c8-window.sh; do
  pids=$(pgrep -f "$pat" | grep -v $$ || true)
  [ -n "$pids" ] && { log "kill $pat -> $pids"; kill $pids 2>/dev/null; }
done
sleep 2

log "=== 3) 优雅停实例"
bash "$BASE/stop-flash-next-0300.sh" "$PORT"

log "=== 4) 等显存归零（≤120s）"
for i in $(seq 1 24); do
  used=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  log "  显存合计 ${used} MiB"
  [ "${used:-99999}" -lt 4000 ] && break
  sleep 5
done

log "=== 5) 裸启（不带任何 FN_* ⇒ inner 内置定稿参数：1M/YaRN×4 + block1616 + MTP4 + KVOFF=0）"
rm -f "$BASE/launch.env.bak-preprod-$(date +%H%M)"
[ -f "$BASE/launch.env" ] && cp -a "$BASE/launch.env" "$BASE/launch.env.bak-preprod-$(date +%H%M)"
cd "$BASE"
setsid env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/ll \
  bash "$BASE/start-flash-next-0300.sh" </dev/null >/dev/null 2>&1 &
rm -f /home/ll/deploy/fnx-manual-stop

log "=== 6) 等 health 200（≤900s）"
ok=0
for i in $(seq 1 180); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/health" || true)
  if [ "$code" = "200" ]; then ok=1; log "  health=200 @ $((i*5))s"; break; fi
  sleep 5
done

if [ "$ok" = "1" ]; then
  log "=== 7) 校验生产参数"
  pid=$(for p in $(ls /proc | grep -E '^[0-9]+$'); do c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null); case "$c" in *vllm*serve*18420*) echo $p; break;; esac; done)
  log "  pid=$pid"
  tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | grep -nE 'max-model-len|kv-transfer-config|num-gpu-blocks-override|max-num-seqs' | sed 's/^/    /'
  for v in "$(tr '\0' '\n' < /proc/$pid/cmdline | grep -A1 -- '--max-model-len' | tail -1)"; do log "  max-model-len=$v"; done
  log "  KV offload 应缺席：$(tr '\0' '\n' < /proc/$pid/cmdline | grep -c 'kv-transfer-config') 处"
else
  log "!!! health 未就绪，请看 $BASE/../vllm-flash-next-0300.log 与 vllm-run.log"
fi

log "=== 8) 恢复看门狗 timer"
systemctl --user start fnx-18420-watchdog.timer || true
systemctl --user is-active fnx-18420-watchdog.timer
log "=== DONE ok=$ok"
