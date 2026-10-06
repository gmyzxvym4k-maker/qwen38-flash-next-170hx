#!/bin/bash
# soak-0310-monitor.sh — 0.31.0 + SimpleCPU 二级缓存 长跑看护（每 300s 记一行）
# 判读：segv 计数必须停在当前值（历史病灶=带档就绪后 26~71 分钟 PP1 原生 segfault）；
#      ext_hits 只增不减；xid 恒 0；shared/pinned 不再增长。
# 用法：setsid nohup bash soak-0310-monitor.sh [时长秒，缺省 21600=6h] >/dev/null 2>&1 &
set -u
export SUDO_PASS="${SUDO_PASS:?本副本已脱敏：先 export SUDO_PASS=<部署机 sudo 口令>}"   # [脱敏]
LOG=/home/ll/deploy/soak-0310.log
VLLM_LOG=/home/ll/deploy/vllm-flash-next-0310.log
DUR=${1:-21600}
END=$(( $(date +%s) + DUR ))
echo "[soak] 开始 $(date -Is) 观察 $((DUR/60)) 分钟" >> "$LOG"
while [ "$(date +%s)" -lt "$END" ]; do
  H=$(curl -s -o /dev/null -m 5 -w '%{http_code}' http://127.0.0.1:18420/health)
  SEGV=$(grep -ac "Segfault encountered" "$VLLM_LOG" 2>/dev/null || echo 0)
  ERR=$(grep -acE "EngineDeadError|Worker.*exit code|illegal memory access" "$VLLM_LOG" 2>/dev/null || echo 0)
  EXT=$(curl -s -m 5 http://127.0.0.1:18420/metrics 2>/dev/null | awk '/vllm:external_prefix_cache_hits_total\{/{print $2}' | head -1)
  QRY=$(curl -s -m 5 http://127.0.0.1:18420/metrics 2>/dev/null | awk '/vllm:external_prefix_cache_queries_total\{/{print $2}' | head -1)
  XID=$(echo "$SUDO_PASS" | sudo -S -p '' dmesg 2>/dev/null | grep -c 'Xid' || echo '?')
  MEM=$(free -g | awk 'NR==2{printf "used=%s free=%s cache=%s avail=%s", $3,$4,$6,$7}')
  PIN=$(free -g | awk 'NR==2{print $5}')
  GPU=$(timeout 15 nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr '\n' '/' )
  echo "$(date '+%m-%d %H:%M:%S') health=$H segv=$SEGV engerr=$ERR ext_hits=${EXT:-?} ext_q=${QRY:-?} xid=$XID pinnedGiB=${PIN:-?} $MEM gpu=${GPU:-?}" >> "$LOG"
  sleep 300
done
echo "[soak] 结束 $(date -Is)" >> "$LOG"
