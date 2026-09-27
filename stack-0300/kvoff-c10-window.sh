#!/bin/bash
# kvoff-c10-window.sh —— 18420「CPU KV 二级缓存 · c10 悬挂自愈 + 定向命中」实机验证窗口
#
# 与 c8 窗口的三点区别：
#   1) 带 c10 自愈 TTL：FN_KVOFF_PENDING_TTL=45 / FN_KVOFF_JOB_TTL=60（故意短，
#      让 window7 那种「pending 永挂」在窗口内必被自愈并打出判据行）；
#   2) 更小的 GPU 池（--num-gpu-blocks-override 40 ≈ 64.6k token）+ 15k token 文档
#      ⇒ 建档 1 篇 + 挤池 4 篇 = 75k > 池容，A 必被逐出显存 ⇒ 重发只能走 CPU 档；
#   3) 探针关掉自适应（--auto 0）并显式给参，避免 09-27 两次「自适应选参过冲」。
#
# 判据 = c8 结构判据 + c10 自愈判据 + 命中判据 + 「不再永久 deferred」收尾判据。
#
# 用法（部署机，ll 用户）：
#   SUDO_PASS=**** bash /home/ll/deploy/kvoff-c10-window.sh      # 全流程
#   MODE=rollback SUDO_PASS=**** bash ...                        # 只回滚到生产态
#   MODE=judge bash ...                                          # 只跑判据
set -uo pipefail

BASE=${BASE:-/home/ll/deploy/vllm-0300}
PORT=${FN_PORT:-18420}
LOG=${LOG:-/home/ll/deploy/vllm-flash-next-0300.log}
ENVF=$BASE/launch.env
RES=${RES:-/home/ll/deploy/kvoff-c10-window.result}
PROBE=${PROBE:-/home/ll/deploy/kvoff-c10-probe.py}
PY=${PY:-/home/ll/vllm-env/bin/python}
C10_WINDOW_LOG=${C10_WINDOW_LOG:-/home/ll/deploy/kvoff-c10-window.log}
KVOFF_BYTES=${KVOFF_BYTES:-68719476736}          # 64 GiB
PENDING_TTL=${PENDING_TTL:-45}
JOB_TTL=${JOB_TTL:-60}
PROBE_ARGS=${PROBE_ARGS:---tokens 15000 --docs 1 --flush 4 --auto 0 --gap 3 --answer-tokens 512}
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
xid_count() { SUDO timeout 30 dmesg 2>/dev/null | grep -cE "NVRM: *Xid" || true; }
shm_used() { df -B1 --output=used /dev/shm 2>/dev/null | tail -1 | tr -d ' '; }
mem_avail() { awk '/^MemAvailable:/{print $2*1024}' /proc/meminfo; }

start_engine() { # start_engine <kvoff:0|1>
  local kv=$1
  # 原样重启的铁律（09-24 定版）：wrapper 会重写 launch.env，必须先把现行 env
  # source 进环境，否则未列出的 FN_* 全部静默回落到 inner 内置缺省。
  if [ -f "$ENVF" ]; then set -a; # shellcheck disable=SC1090
    . "$ENVF"; set +a; fi
  # 可选：**在 launch.env 之后**再套一层覆盖（用于"小池位测量"这类临时配置）。
  # 必须是文件而不是环境变量：FN_EXTRA_ARGS 这类值含空格，且 launch.env 会覆盖同名变量。
  # ⚠️【只在测试启动（kv=1）时套用】—— 2026-09-27 事故：早期实现无条件套用，
  #    回滚也带上测试配置，把生产实例留在 max-model-len 65536 + 80 blocks 上。
  if [ "$kv" = "1" ] && [ -n "${FN_OVERRIDES_FILE:-}" ] && [ -f "$FN_OVERRIDES_FILE" ]; then
    set -a; # shellcheck disable=SC1090
    . "$FN_OVERRIDES_FILE"; set +a
    echo "[window] 应用 FN 覆盖（仅测试档）：$FN_OVERRIDES_FILE" >&2
  fi
  export FN_KVOFF=$kv
  if [ "$kv" = "1" ]; then
    export FN_KVOFF_BYTES=$KVOFF_BYTES
    export FN_KVOFF_SHARED=1
    # c10 自愈闸门（故意设短：窗口内即可观察到自愈判据行）
    export FN_KVOFF_PENDING_TTL=$PENDING_TTL
    export FN_KVOFF_JOB_TTL=$JOB_TTL
  fi
  setsid bash "$BASE/start-flash-next-0300.sh" >> "$C10_WINDOW_LOG" 2>&1 < /dev/null &
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
  #    【判据修正·2026-09-27】旧版 grep 的是旧 chroot 栈的字符串 "FUSE: store 方向"，
  #    而 0.30.0 栈 c6 的熔断日志是 "[dsh-kvoff c6] worker store fuse tripped"
  #    ⇒ 这条判据一直假 PASS（无论熔断与否都算「零熔断」）。现改按真实文案匹配。
  local ae=$(grep -c "AssertionError" "$seg")
  local fu=$(grep -c "worker store fuse tripped" "$seg")
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
  # 9) c10 自愈已挂（管理器侧一定打；scheduler 侧同批挂载）
  local c10a=$(grep -c "c10a: CPUOffloadingManager pending 自愈已挂" "$seg")
  chk "c10a 管理器自愈已挂载" "$([ "$c10a" -ge 1 ] && echo 0 || echo 1)" "命中 $c10a 行"
  rm -f "$seg"
}

c10_judge() { # c10_judge <日志起始偏移>：自愈判据（出现过即证明本窗口复现过悬挂）
  local off=$1 seg n_a n_b n_d
  seg=$(mktemp)
  tail -c +"$((off+1))" "$LOG" > "$seg" 2>/dev/null || cp "$LOG" "$seg"
  n_a=$(grep -c "c10a\[manager\]" "$seg")
  n_b=$(grep -c "c10b: 强制收尾超时 job" "$seg")
  n_d=$(grep -c "c10a: .*reaped" "$seg")
  say "c10 观测：c10a 诊断行=$n_a  c10b 自愈行=$n_b（0 行 = 本窗口 ack 没丢，属正常）"
  grep -E "c10a\[manager\]|c10b" "$seg" | tail -5 | tee -a "$RES"
  rm -f "$seg"
  # 收尾判据：探针跑完后不得再有 deferred 请求（window7 的病征）
  local dfr
  dfr=$(curl -s --max-time 10 "http://127.0.0.1:$PORT/metrics" \
        | awk '/^vllm:num_requests_waiting_by_reason/ && /reason="deferred"/ {print $2; exit}')
  chk "探针后无 deferred 请求残留（不再永久挂）" \
      "$(awk -v d="${dfr:-0}" 'BEGIN{print (d+0)==0?0:1}')" "deferred=${dfr:-0}"
}

metrics_dump() {
  curl -s --max-time 20 "http://127.0.0.1:$((PORT))/metrics" \
    | grep -E "kv_offload|external_prefix" | grep -v "^#" | tee -a "$RES"
}

rollback() {
  say ">>> 回滚：FN_KVOFF=0 裸启（生产态：inner 内置 1M/YaRN×4 + block1616 + MTP4，无测试覆盖）"
  bash "$BASE/stop-flash-next-0300.sh" "$PORT" >> "$C10_WINDOW_LOG" 2>&1
  # 铁律（09-27 事故）：绝不能在带测试 overrides 的环境里回滚 —— 那会把生产实例
  # 留在 65536/小池位上。这里用 env -i 从零环境启动：不带任何 FN_* ⇒ inner 定稿参数。
  (
    cd "$BASE" || exit 1
    setsid env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/ll \
      bash "$BASE/start-flash-next-0300.sh" >> "$C10_WINDOW_LOG" 2>&1 < /dev/null &
  )
  rm -f /home/ll/deploy/fnx-manual-stop
  if wait_health; then
    local mml
    mml=$(tr '\0' '\n' < "/proc/$(pgrep -f 'vllm.entrypoints.cli.main serve' | head -1)/cmdline" 2>/dev/null \
          | grep -A1 '^--max-model-len' | tail -1)
    say "回滚完成：health=200 max-model-len=${mml:-?}（期望 1048576）"
  else
    say "回滚后 health 仍非 200，人工介入！"
  fi
  systemctl --user start "$WATCHDOG" 2>/dev/null || true
  say "看门狗已恢复（timer start）"
}

# ---------------------------------------------------------------- 主流程
: > "$RES"
say "kvoff-c10 验证窗口 MODE=$MODE KVOFF_BYTES=$KVOFF_BYTES ($((KVOFF_BYTES/1073741824)) GiB) port=$PORT"

if [ "$MODE" = "rollback" ]; then rollback; exit 0; fi

X0=$(xid_count); say "基线：Xid=$X0 health=$(health) shm_used=$(shm_used) MemAvailable=$(mem_avail)"
if [ "$MODE" != "judge" ]; then
  say ">>> 停看门狗 timer（防自愈抢跑）"
  systemctl --user stop "$WATCHDOG" 2>/dev/null || true
  say ">>> 优雅停止实例"
  bash "$BASE/stop-flash-next-0300.sh" "$PORT" >> "$C10_WINDOW_LOG" 2>&1
  # 等显存归零（幽灵显存 = 启动失败第一嫌疑）
  for i in $(seq 1 30); do
    busy=$(timeout 20 nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    [ "$busy" = "0" ] && break
    sleep 5
  done
  say "显存占用进程数=$busy（等归零 ${i} 轮）"
fi

OFF=$(wc -c < "$LOG")
say ">>> 启动 FN_KVOFF=1 FN_KVOFF_SHARED=1（日志偏移 $OFF）"
start_engine 1
if ! wait_health; then
  # 【判据补强·2026-09-27】把「启动参数本身无效」与「探针失败」区分开：
  # --num-gpu-blocks-override 给小了（如 40 块）时引擎会直接拒启，日志指纹如下。
  if tail -c 300000 "$LOG" | grep -q "KV cache is needed, which is larger than the available"; then
    chk "启动参数有效性（KV 池 ≥ max-model-len）" 1 \
        "num-gpu-blocks-override 太小，引擎拒启（65536 上下文需 1.42 GiB KV；40 块仅 0.84 GiB）"
  fi
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
c10_judge "$OFF"
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
