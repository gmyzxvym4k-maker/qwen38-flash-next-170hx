#!/bin/bash
# kvoff-soak-monitor.sh —— 二级缓存 soak 记录器（每 5 分钟一行，缺省 4 小时）
# 用法：setsid nohup bash kvoff-soak-monitor.sh [小时数] >/dev/null 2>&1 &
# 记录：health / 并发 / KV 占用 / 外部(内存档)命中 / 本地命中 / GPU Xid 计数 / 内存水位
# 异常：health 连续 2 次非 200、GPU 新增 Xid —— 都追加醒目标记（看门狗自愈会剥离二级缓存档位）
OUT=/home/ll/deploy/kvoff-soak.log
HOURS=${1:-4}
PW="${SUDO_PASS:?本副本已脱敏：先 export SUDO_PASS=<部署机 sudo 口令>}"
xid_count(){ echo "$PW" | sudo -S -p '' dmesg 2>/dev/null | grep -ac "NVRM: Xid"; }
X0=$(xid_count)
i=0
bad=0
echo "$(date '+%F %T') === soak 监控开始（${HOURS}h，外部命中基线 xid=$X0）===" >> "$OUT"
while [ "$i" -lt $((HOURS * 12)) ]; do
  TS=$(date '+%F %T')
  H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:18420/health)
  M=$(curl -s --max-time 8 http://127.0.0.1:18420/metrics 2>/dev/null)
  EXT=$(echo "$M" | awk '/^vllm:external_prefix_cache_hits_total\{/{print $NF; exit}')
  LOC=$(echo "$M" | awk '/^vllm:prefix_cache_hits_total\{/{print $NF; exit}')
  QUERY=$(echo "$M" | awk '/^vllm:prefix_cache_queries_total\{/{print $NF; exit}')
  KV=$(echo "$M" | awk '/^vllm:kv_cache_usage_perc\{/{print $NF; exit}')
  RUN=$(echo "$M" | awk '/^vllm:num_requests_running\{/{print $NF; exit}')
  XN=$(xid_count)
  MEM=$(free -g | awk 'NR==2{printf "%s/%s/%s", $3, $4, $7}')
  GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd/ -)
  echo "$TS health=$H run=$RUN kv=$KV ext_hits=${EXT:-NA} local_hits=${LOC:-NA} queries=${QUERY:-NA} xid=$XN mem(u/f/a)=$MEM gpu_used_mib=$GPU" >> "$OUT"
  if [ "$H" != "200" ]; then bad=$((bad + 1)); else bad=0; fi
  if [ "$XN" -gt "$X0" ]; then
    echo "$TS !! 新增 GPU Xid：$X0 -> $XN（soak 记录：看门狗若拉起会剥离二级缓存档位）" >> "$OUT"
    X0=$XN
  fi
  if [ "$bad" -ge 2 ]; then
    echo "$TS !! health 连续 $bad 次非 200（已到自愈阈值，看门狗应介入）" >> "$OUT"
  fi
  sleep 300
  i=$((i + 1))
done
echo "$(date '+%F %T') === soak 监控结束（${HOURS}h）===" >> "$OUT"
