#!/bin/bash
# kvoff-c8-window.sh —— 18420「CPU KV 二级缓存 · 公共区（c8）」实机验证窗口
#
# 干什么：停看门狗 → 优雅停实例 → 带 FN_KVOFF=1 + FN_KVOFF_SHARED=1 原样重启 →
#   按判据逐项核对（公共区协商、物理钉住=配置值、store 落地、CPU→GPU 回载、
#   验证码无损复述、零 Xid）→ 任一步硬失败自动回滚成 FN_KVOFF=0 的生产态。
#
# 为什么需要窗口：c8 改的是 host 侧内存布局，离线自检（selftest_kvoff_rt.py，
#   67 项）能证明"算术与协议正确"，但证明不了"引擎真的会用这块内存搬 KV"。
#
# 用法（部署机，ll 用户）：
#   SUDO_PASS=**** bash /home/ll/deploy/kvoff-c8-window.sh              # 缺省 48 GiB
#   KVOFF_BYTES=$((64*1024*1024*1024)) SUDO_PASS=**** bash ...          # 换 64 GiB 档
#   MODE=rollback bash /home/ll/deploy/kvoff-c8-window.sh               # 只做回滚
#   MODE=judge   bash /home/ll/deploy/kvoff-c8-window.sh                # 只跑判据（服务已在跑）
#
# 结果：/home/ll/deploy/kvoff-c8-window.result（逐项 PASS/FAIL + 关键数字）
# 停机时长：约 8~10 分钟（冷启动 5~8 min + 探针 3~6 min，探针期间服务可用但会被
#   长文档挤占，别在业务高峰跑）。
set -uo pipefail

BASE=${BASE:-/home/ll/deploy/vllm-0300}
PORT=${FN_PORT:-18420}
LOG=${LOG:-/home/ll/deploy/vllm-flash-next-0300.log}
ENVF=$BASE/launch.env
RES=${RES:-/home/ll/deploy/kvoff-c8-window.result}
PROBE=${PROBE:-/home/ll/deploy/kvoff-c8-probe.py}
PY=${PY:-/home/ll/vllm-env/bin/python}
KVOFF_BYTES=${KVOFF_BYTES:-51539607552}          # 48 GiB：容量≈160 万 tok > GPU 池 120.7 万
PROBE_ARGS=${PROBE_ARGS:---tokens 100000 --docs 2 --flush 11 --gap 3}
MODE=${MODE:-full}                                # full | rollback | judge
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-900}
SUDO_PASS=${SUDO_PASS:?需要 SUDO_PASS（root 起实例与读 smaps）}
WATCHDOG=fnx-18420-watchdog.timer

PASS=0; FAIL=0
say() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$RES"; }
chk() { # chk <名称> <0/1> <详情>
  if [ "$2" = "0" ]; then PASS=$((PASS+1)); printf 'PASS  %s  %s\n' "$1" "$3" | tee -a "$RES"
  else FAIL=$((FAIL+1)); printf 'FAIL  %s  %s\n' "$1" "$3" | tee -a "$RES"; fi
}
SUDO() { echo "$SUDO_PASS" | sudo -S -p '' "$@"; }
health() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/health"; }
xid_count() { SUDO dmesg 2>/dev/null | grep -cE "NVRM: *Xid" || true; }
shm_used() { df -B1 --output=used /dev/shm 2>/dev/null | tail -1 | tr -d ' '; }
mem_avail() { awk '/^MemAvailable:/{print $2*1024}' /proc/meminfo; }

start_engine() { # start_engine <kvoff:0|1>
  local kv=$1
  # 原样重启的铁律（09-24 定版）：wrapper 会重写 launch.env，必须先把现行 env
  # source 进环境，否则未列出的 FN_* 全部静默回落到 inner 内置缺省。
  if [ -f "$ENVF" ]; then set -a; # shellcheck disable=SC1090
    . "$ENVF"; set +a; fi
  export FN_KVOFF=$kv
  [ "$kv" = "1" ] && { export FN_KVOFF_BYTES=$KVOFF_BYTES; export FN_KVOFF_SHARED=1; }
  setsid bash "$BASE/start-flash-next-0300.sh" >> /home/ll/deploy/kvoff-c8-window.log 2>&1 < /dev/null &
  disown || true
}

wait_health() {
  local t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt "$HEALTH_TIMEOUT" ]; do
    [ "$(health)" = "200" ] && return 0
    sleep 10
  done
  return 1
}

judge() { # judge <日志起始字节偏移>
  local off=$1 seg
  seg=$(mktemp)
  tail -c +"$((off+1))" "$LOG" > "$seg" 2>/dev/null || cp "$LOG" "$seg"

  # 1) 两个 worker 都宣告公共区协商成功
  local nw=$(grep -c "c8\[worker rank.*公共区已协商" "$seg")
  chk "c8 worker 侧公共区协商（期望 2）" "$([ "$nw" -ge 2 ] && echo 0 || echo 1)" "命中 $nw 行"
  # 2) 调度器侧采纳同一口径（decision=['shared'] 且 num_chunks>0）
  local sl=$(grep -o "c8\[调度器侧\]: decision=\[[^]]*\][^(]*num_chunks=[0-9]*" "$seg" | tail -1)
  local sn=$(printf '%s' "$sl" | grep -o "num_chunks=[0-9]*" | cut -d= -f2)
  chk "c8 调度器侧采纳公共区（decision=shared 且 num_chunks>0）" \
      "$(echo "$sl" | grep -q "shared" && [ "${sn:-0}" -gt 0 ] && echo 0 || echo 1)" "${sl:-未出现}"
  # 3) 三侧分组一致（c1 语义，环形分组被剔除）
  local g=$(grep -o "offload 分组 \[[0-9, ]*\]" "$seg" | sort -u | wc -l)
  chk "c1 offload 分组三侧一致（唯一分组集合=1）" "$([ "$g" = "1" ] && echo 0 || echo 1)" "distinct=$g"
  # 4) 共享区只创建一次（两 rank 共用一块），且 barrier 后被 unlink（自愈不留尸）
  local cc=$(grep -c "Created mmap file /dev/shm/vllm_offload_" "$seg")
  local ul=$(grep -c "Unlinked mmap file /dev/shm/vllm_offload_" "$seg")
  chk "共享区创建恰 1 次（两 rank 同一块内存）" "$([ "$cc" = "1" ] && echo 0 || echo 1)" "created=$cc"
  chk "共享区 barrier 后已 unlink（进程退出即回收）" "$([ "$ul" = "1" ] && echo 0 || echo 1)" "unlinked=$ul"
  # 5) 真的 pin 上了（cudaHostRegister 成功；失败会是 warning 且退化成非 pinned）
  local pr=$(grep -c "cudaHostRegister failed" "$seg")
  chk "cudaHostRegister 无失败" "$([ "$pr" = "0" ] && echo 0 || echo 1)" "failed=$pr"
  # 6) 没有断言 / 没有熔断
  local ae=$(grep -c "AssertionError" "$seg"); local fu=$(grep -c "FUSE: store 方向" "$seg")
  chk "零 AssertionError（分组/长度口径一致）" "$([ "$ae" = "0" ] && echo 0 || echo 1)" "ae=$ae"
  chk "零 store 熔断（c6 未触发）" "$([ "$fu" = "0" ] && echo 0 || echo 1)" "fuse=$fu"
  # 7) 物理钉住 = 配置值（公共区是 tmpfs 文件；df used 应≈KVOFF_BYTES，
  #    私有路径会是 1.56~2× —— 这条就是 c8 的存在理由）
  sleep 5
  local used=$(shm_used)
  local ratio=$(awk -v u="${used:-0}" -v c="$KVOFF_BYTES" 'BEGIN{printf "%.2f", (c>0)?u/c:0}')
  chk "物理钉住≈配置值（tmpfs used/配置 ∈[0.9,1.15]）" \
      "$(awk -v r="$ratio" 'BEGIN{print (r>=0.9 && r<=1.15)?0:1}')" \
      "used=${used} B 配置=${KVOFF_BYTES} B 比值=${ratio}（私有路径历史值≈1.56~2.0）"
  # 8) 两 rank 都建立了 CPU 档（Allocating N CPU tensors 各一次）
  local ac=$(grep -c "Allocating .* CPU tensors" "$seg")
  chk "两 rank 均分配 CPU KV 缓冲（期望 2）" "$([ "$ac" -ge 2 ] && echo 0 || echo 1)" "命中 $ac 行"
  rm -f "$seg"
}

metrics_dump() {
  curl -s --max-time 20 "http://127.0.0.1:$((PORT))/metrics" \
    | grep -E "kv_offload|external_prefix" | grep -v "^#" | tee -a "$RES"
}

rollback() {
  say ">>> 回滚：FN_KVOFF=0 重启（生产态）"
  local off=$(wc -c < "$LOG")
  bash "$BASE/stop-flash-next-0300.sh" "$PORT" >> /home/ll/deploy/kvoff-c8-window.log 2>&1
  start_engine 0
  if wait_health; then say "回滚完成：health=200"; else say "回滚后 health 仍非 200，人工介入！"; fi
  systemctl --user start "$WATCHDOG" 2>/dev/null || true
  say "看门狗已恢复（timer start）"
}

# ---------------------------------------------------------------- 主流程
: > "$RES"
say "kvoff-c8 验证窗口 MODE=$MODE KVOFF_BYTES=$KVOFF_BYTES ($((KVOFF_BYTES/1073741824)) GiB) port=$PORT"

if [ "$MODE" = "rollback" ]; then rollback; exit 0; fi

X0=$(xid_count); say "基线：Xid=$X0 health=$(health) shm_used=$(shm_used) MemAvailable=$(mem_avail)"
if [ "$MODE" != "judge" ]; then
  say ">>> 停看门狗 timer（防自愈抢跑）"
  systemctl --user stop "$WATCHDOG" 2>/dev/null || true
  say ">>> 优雅停止实例"
  bash "$BASE/stop-flash-next-0300.sh" "$PORT" >> /home/ll/deploy/kvoff-c8-window.log 2>&1
  # 等显存归零（幽灵显存 = 启动失败第一嫌疑）
  for i in $(seq 1 30); do
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    [ "$busy" = "0" ] && break
    sleep 5
  done
  say "显存占用进程数=$busy（等归零 ${i} 轮）"
fi

OFF=$(wc -c < "$LOG")
say ">>> 启动 FN_KVOFF=1 FN_KVOFF_SHARED=1（日志偏移 $OFF）"
start_engine 1
if ! wait_health; then
  chk "health=200（启动完成）" 1 "超时 ${HEALTH_TIMEOUT}s"
  tail -60 "$LOG" | tee -a "$RES"
  rollback
  say "总结：PASS=$PASS FAIL=$FAIL（启动失败已回滚）"
  exit 1
fi
chk "health=200（启动完成）" 0 "耗时见日志"

judge "$OFF"
say ">>> 探针（$PROBE_ARGS）"
"$PY" "$PROBE" --endpoint "http://127.0.0.1:$PORT/v1" $PROBE_ARGS | tee -a "$RES"
PR=$?
chk "探针三项判据（external hits>0 且 CPU→GPU>0 且验证码复述）" "$PR" "退出码 $PR"
metrics_dump
X1=$(xid_count)
chk "零新增 Xid（GPU 未被拷贝路径打挂）" "$([ "$X1" = "$X0" ] && echo 0 || echo 1)" "基线=$X0 现在=$X1"

if [ "$FAIL" -gt 0 ]; then
  say ">>> 有 $FAIL 项失败 ⇒ 自动回滚 FN_KVOFF=0"
  rollback
else
  say ">>> 全绿。要不要把公共区固化成生产态：写进 launch.env + 快启预设 + 弹窗默认（三源同步）"
  systemctl --user start "$WATCHDOG" 2>/dev/null || true
fi
say "总结：PASS=$PASS FAIL=$FAIL  结果文件 $RES"
[ "$FAIL" = "0" ] || exit 1
