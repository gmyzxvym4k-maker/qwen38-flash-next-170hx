#!/bin/bash
# kvoff-accept.sh —— 生产档（1M 配置）内存二级缓存验收：起实例 → 挤池探针 → 恢复生产
# 结果写 /home/ll/deploy/kvoff-accept.result（一路追加，抗 ssh 超时/外部干扰取证）
set -uo pipefail
BASE=/home/ll/deploy/vllm-0300
RES=/home/ll/deploy/kvoff-accept.result
KVOFF_GIB=${KVOFF_GIB:-32}
FLUSHES=${FLUSHES:-9}
export XDG_RUNTIME_DIR=/run/user/1000
export SUDO_PASS=${SUDO_PASS:?本副本已脱敏：先 export SUDO_PASS=<部署机 sudo 口令>}
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RES"; }

: > "$RES"
log "=== 验收开始：KVOFF_GIB=$KVOFF_GIB flushes=$FLUSHES ==="
log "停看门狗 + 停实例"
systemctl --user stop fnx-18420-watchdog.timer 2>/dev/null || true
bash "$BASE/stop-flash-next-0300.sh" 18420 >> "$RES" 2>&1

log "启动：生产档（inner 缺省 1M/YaRN×4/block1616/MTP4）+ SimpleCPUOffloadConnector"
cd "$BASE" || exit 1
setsid env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/ll \
  FN_EXTRA_ENV='VLLM_USE_SIMPLE_KV_OFFLOAD=1' \
  FN_EXTRA_ARGS="--kv-offloading-size $KVOFF_GIB" \
  bash "$BASE/start-flash-next-0300.sh" </dev/null >/dev/null 2>&1 &

H=000
for i in $(seq 1 80); do
  H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:18420/health)
  log "等待就绪 #$i health=$H"
  [ "$H" = "200" ] && break
  sleep 10
done
if [ "$H" != "200" ]; then
  log "启动失败/超时，直接恢复生产"
  setsid nohup bash /home/ll/deploy/kvoff-restore-prod.sh >/dev/null 2>&1 < /dev/null &
  exit 1
fi

log "连接器判据："
grep -a 'SimpleCPUOffloadConnector: role' /home/ll/deploy/vllm-flash-next-0300.log | tail -1 | tee -a "$RES"

log "跑挤池验收探针（$FLUSHES × ~116k token）"
/home/ll/vllm-env/bin/python /home/ll/deploy/kvoff-churn.py \
  --flushes "$FLUSHES" --a-sentences 2000 --b-sentences 8000 >> "$RES" 2>&1

log "外部命中累计："
curl -s --max-time 8 http://127.0.0.1:18420/metrics \
  | grep -a 'external_prefix_cache_hits_total' | head -1 >> "$RES"

log "本轮 <EXTERNAL_HOST> 登录次数（外部干扰取证）：$(echo "$SUDO_PASS" | sudo -S -p '' journalctl _COMM=sshd --since '-20min' --no-pager 2>/dev/null | grep -ac 'from <EXTERNAL_HOST>')"
log "恢复生产（含看门狗）"
setsid nohup bash /home/ll/deploy/kvoff-restore-prod.sh >/dev/null 2>&1 < /dev/null &
log "=== 验收脚本结束（生产恢复已在后台执行）==="
