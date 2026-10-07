#!/bin/bash
# soak-0310-monitor.sh v3 — 0.31.0 + SimpleCPU 二级缓存 长跑看护（每 300s 一行）
#
# 判读（口径全用引擎真值，别信 wrapper 的"成功"）：
#   dsegv / dengine_err 必须恒 0 —— 历史病灶（P59~P61）= 带 96GiB 档就绪后 26~71 分钟 Worker_PP1 原生 segfault
#   xid 恒 0
#   cpu_used 逼近 cpu_blocks 上限 = 内存档在被真实使用；cpu_loaded_blk 只增 = 回载在发生
#   ext_hits 只增不减 = 前缀被挤出显存池后由内存档接住
#
# 用法：export SUDO_PASS=<部署机口令>          # 本副本已脱敏
#       setsid nohup bash soak-0310-monitor.sh 21600 >>/tmp/soak-run.out 2>&1 &
#       bash soak-0310-monitor.sh --reset      # 只重记基线（日志含换栈前的陈旧行时用）
set -u
export SUDO_PASS="${SUDO_PASS:?本副本已脱敏：先 export SUDO_PASS=<部署机 sudo 口令>}"
LOG=/home/ll/deploy/soak-0310.log
VLLM_LOG=/home/ll/deploy/vllm-flash-next-0310.log
BASE=/home/ll/deploy/.soak-0310.baseline
DUR=${1:-21600}
END=$(( $(date +%s) + DUR ))

MET=/tmp/soak-metrics.txt
fetch() { curl -s -m 6 http://127.0.0.1:18420/metrics -o "$MET" 2>/dev/null; }
mval() {   # 单值指标（带标签取第一条）
  awk -v pat="^$1[{ ]" '$0 ~ pat && $0 !~ /_created/ {print $NF; exit}' "$MET" 2>/dev/null
}
msum() {   # 多标签计数求和（store 结果按 outcome 分了多条）
  awk -v pat="^$1{" '$0 ~ pat && $0 !~ /_created/ {s+=$NF} END{printf "%.0f", s+0}' "$MET" 2>/dev/null
}
mcap() {   # SimpleCPU 容量块数写在 info 的 label 里（=clamp 后的最窄 rank）
  awk '/^vllm:simple_kv_offload_info{/ {if (match($0, /capacity_blocks="[0-9]+"/)) {print substr($0, RSTART+17, RLENGTH-18); exit}}' "$MET" 2>/dev/null
}
cnt() {    # grep -c 无匹配时也返回 0；绝不能再 `|| echo 0`（会拼成两行把日志搞脏，10-06 实锤）
  local n
  n=$(grep -ac "$1" "$VLLM_LOG" 2>/dev/null); echo "${n:-0}"
}

if [ ! -f "$BASE" ] || [ "${1:-}" = "--reset" ]; then
  printf 'segv=%s\nengerr=%s\n' "$(cnt 'Segfault encountered')" \
    "$(cnt -E 'EngineDeadError|Worker.*exit code|illegal memory access')" > "$BASE"
  echo "[soak] 基线已记录 $(date -Is) -> $BASE" >> "$LOG"
  [ "${1:-}" = "--reset" ] && exit 0
fi
BASE_SEGV=$(awk -F= '/^segv=/{print $2}' "$BASE")
BASE_ERR=$(awk -F= '/^engerr=/{print $2}' "$BASE")

echo "[soak] 开始 $(date -Is) 观察 $((DUR/60)) 分钟（基线 segv=$BASE_SEGV engerr=$BASE_ERR）" >> "$LOG"
while [ "$(date +%s)" -lt "$END" ]; do
  H=$(curl -s -o /dev/null -m 5 -w '%{http_code}' http://127.0.0.1:18420/health)
  SEGV=$(cnt 'Segfault encountered'); ERR=$(cnt -E 'EngineDeadError|Worker.*exit code|illegal memory access')
  XID=$(echo "$SUDO_PASS" | sudo -S -p '' dmesg 2>/dev/null | grep -c 'Xid'); XID=${XID:-?}
  fetch
  EXT=$(mval 'vllm:external_prefix_cache_hits_total'); QRY=$(mval 'vllm:external_prefix_cache_queries_total')
  CAP=$(mcap); UB=$(mval 'vllm:simple_kv_offload_used_blocks')
  LB=$(mval 'vllm:simple_kv_offload_load_blocks_total'); PB=$(mval 'vllm:simple_kv_offload_pending_store_blocks')
  SV=$(awk '/^vllm:simple_kv_offload_save_outcomes_total\{.*outcome="stored"/ {print $NF; exit}' "$MET" 2>/dev/null); GPU=$(mval 'vllm:kv_cache_usage_perc')
  MEM=$(free -g | awk 'NR==2{printf "used=%s avail=%s", $3,$7}')
  echo "$(date '+%m-%d %H:%M:%S') health=$H dsegv=$((SEGV-BASE_SEGV)) dengine_err=$((ERR-BASE_ERR)) xid=$XID ext_hits=${EXT:-?} ext_q=${QRY:-?} cpu_cap_blk=${CAP:-?} cpu_cap_tok=$(( ${CAP:-0} * 1616 )) cpu_used=${UB:-?} cpu_loaded_blk=${LB:-?} cpu_pend_store=${PB:-?} cpu_saved=${SV:-?} kv_usage=${GPU:-?} $MEM" >> "$LOG"
  sleep 300
done
echo "[soak] 结束 $(date -Is)" >> "$LOG"
