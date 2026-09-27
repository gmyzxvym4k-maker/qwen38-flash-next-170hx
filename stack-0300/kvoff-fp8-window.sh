#!/bin/bash
# kvoff-fp8-window.sh —— FP8 PLE（v2 单 scale 格式）+ simple 卸载的联合验证窗口
# 用法：bash kvoff-fp8-window.sh   结果追加写 /home/ll/deploy/kvoff-fp8-window.result
set -uo pipefail
BASE=/home/ll/deploy/vllm-0300
RES=/home/ll/deploy/kvoff-fp8-window.result
FP8DIR=${FP8DIR:-/media/ll/data/models-fp8ple/Qwen3.8-Flash-Next-W4A16-AutoRound-fp8ple-v2}
KVOFF_GIB=${KVOFF_GIB:-64}
FN_MAXLEN_V=${FN_MAXLEN_V:-131072}
BLOCKS=${BLOCKS:-100}
export XDG_RUNTIME_DIR=/run/user/1000
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RES"; }

: > "$RES"
log "=== FP8 PLE + simple 卸载验证：dir=$(basename "$FP8DIR") maxlen=$FN_MAXLEN_V blocks=$BLOCKS tier=${KVOFF_GIB}GiB ==="
systemctl --user stop fnx-18420-watchdog.timer 2>/dev/null || true
log "启动时刻 $(date +%H:%M:%S)（用于算启动耗时）"
bash "$BASE/stop-flash-next-0300.sh" 18420 >> "$RES" 2>&1

cd "$BASE" || exit 1
setsid env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/ll \
  FN_MODEL_PATH="$FP8DIR" FN_MAXLEN="$FN_MAXLEN_V" \
  FN_EXTRA_ARGS="--num-gpu-blocks-override $BLOCKS" \
  FN_SIMPLE_OFFLOAD="$KVOFF_GIB" \
  bash "$BASE/start-flash-next-0300.sh" </dev/null >/dev/null 2>&1 &

H=000
for i in $(seq 1 80); do
  H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:18420/health)
  [ "$H" = "200" ] && break
  sleep 10
done
log "health=$H（第 $i 轮）"
grep -a 'Initialized PLE embedding' /home/ll/deploy/vllm-flash-next-0300.log | tail -1 | tee -a "$RES"
grep -a 'SimpleCPUOffloadConnector: role' /home/ll/deploy/vllm-flash-next-0300.log | tail -1 | tee -a "$RES"
if [ "$H" != "200" ]; then
  log "启动失败；末尾错误："
  tail -c 60000 /home/ll/deploy/vllm-flash-next-0300.log | grep -aE 'Error|error|assert' | tail -4 | tee -a "$RES"
else
  log "跑探针 kvoff-scale 6000 --flushes 3（~180k 挤池 > 100 块池 ~161k，强制造走 CPU 档）"
  /home/ll/vllm-env/bin/python /home/ll/deploy/kvoff-scale.py 6000 --flushes 3 >> "$RES" 2>&1
  log "外部命中累计："
  curl -s --max-time 8 http://127.0.0.1:18420/metrics | grep -a 'external_prefix_cache_hits_total' | head -1 >> "$RES"
fi
log "恢复生产（含看门狗）"
setsid nohup bash /home/ll/deploy/kvoff-restore-prod.sh >/dev/null 2>&1 < /dev/null &
log "=== 窗口脚本结束 ==="
