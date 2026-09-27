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
#
# 2026-09-27 窗口 7 后的加固（每一条都有实机教训）：
#   · launch.env 先备份、回滚/中断时还原 —— 否则测试档（max-model-len/池大小）会留在盘上，
#     成为看门狗与控制台下次启动的参数源。
#   · systemctl --user 显式给 XDG_RUNTIME_DIR/DBUS 并校验生效 —— 旧写法 `2>/dev/null || true`
#     让"停看门狗"静默失败，看门狗整场在跑并与回滚抢启动。
#   · 探针套 timeout（PROBE_TIMEOUT，缺省 900s）+ 卡死取证 —— 引擎卡死时探针会一直等
#     （post 超时 1800s），窗口空转且结果里看不出是卡死还是慢。
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
xid_count() { SUDO timeout 30 dmesg 2>/dev/null | grep -cE "NVRM: *Xid" || true; }
shm_used() { df -B1 --output=used /dev/shm 2>/dev/null | tail -1 | tr -d ' '; }
mem_avail() { awk '/^MemAvailable:/{print $2*1024}' /proc/meminfo; }

# ---- 用户级 systemd 的两个必需变量（2026-09-27 事故根因）----------------------
# ssh 非登录会话（尤其经 sudo / 无 tty）缺 XDG_RUNTIME_DIR / DBUS_SESSION_BUS_ADDRESS 时
# `systemctl --user` 直接失败；旧代码写成 `... 2>/dev/null || true` = 静默失效。
# 实测后果：窗口开头"停看门狗 timer"没生效，看门狗整场每 35s 照跑，收尾时与窗口回滚
# 抢启动、还把 launch.env 覆盖成"只有一行头"，生产参数靠 inner 缺省兜住（运气，不是设计）。
# 以 root 跑本脚本时不能用 /run/user/0（不存在），必须回落到调用者（ll）的会话目录。
RUN_UID=${SUDO_UID:-$(id -u)}
RT_DIR=/run/user/$RUN_UID
[ -d "$RT_DIR" ] || RT_DIR=/run/user/1000
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-$RT_DIR}
export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}
LAUNCH_BAK=${LAUNCH_BAK:-/home/ll/deploy/kvoff-c8-launch.env.bak}

wd() { # wd stop|start —— 带生效校验，不再静默失败
  local act=$1 st
  systemctl --user "$act" "$WATCHDOG" >/dev/null 2>&1
  sleep 1
  st=$(systemctl --user is-active "$WATCHDOG" 2>/dev/null || true)
  case "$act:$st" in
    stop:inactive|start:active) say "[wd] 看门狗 $act 生效（state=$st）" ;;
    *) say "[wd] ⚠️ 看门狗 $act 未生效（state=$st）——本次判读需考虑看门狗干扰" ;;
  esac
}

restore_launch_env() {
  if [ -s "${LAUNCH_BAK:-}" ]; then
    cp -f "$LAUNCH_BAK" "$ENVF" && say "已还原 launch.env（生产档，$(grep -c . "$LAUNCH_BAK") 行）"
  fi
}

# 中断/被杀时别把测试配置留在 launch.env（它是看门狗与控制台下次启动的唯一参数源）
trap 'say "收到中断信号 ⇒ 还原 launch.env 并恢复看门狗"; restore_launch_env; wd start; exit 130' INT TERM

stall_evidence() { # 引擎卡死取证：Waiting>0 + 零吞吐 = 调度/拷贝路径卡住，与探针无关
  local seg; seg=$(mktemp)
  tail -c 300000 "$LOG" > "$seg" 2>/dev/null || true
  local w d z
  w=$(grep -c "Waiting: [1-9]" "$seg" || true)
  d=$(grep -c "Deferred: [1-9]" "$seg" || true)
  z=$(grep -c "throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: [1-9]" "$seg" || true)
  say "卡死取证：Waiting>0 行=$w、Deferred>0 行=$d、零吞吐且仍有排队=$z（z>0 即引擎卡住，不是探针慢）"
  grep -a "Avg prompt throughput" "$seg" | tail -3 | sed 's/^/    /' | tee -a "$RES"
  rm -f "$seg"
}

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
  restore_launch_env          # 先还原生产参数，再让 wrapper 重写 launch.env（否则测试档留在盘上）
  bash "$BASE/stop-flash-next-0300.sh" "$PORT" >> /home/ll/deploy/kvoff-c8-window.log 2>&1
  start_engine 0
  if wait_health; then say "回滚完成：health=200"; else say "回滚后 health 仍非 200，人工介入！"; fi
  wd start
}

# ---------------------------------------------------------------- 主流程
: > "$RES"
say "kvoff-c8 验证窗口 MODE=$MODE KVOFF_BYTES=$KVOFF_BYTES ($((KVOFF_BYTES/1073741824)) GiB) port=$PORT"

if [ "$MODE" = "rollback" ]; then rollback; exit 0; fi

X0=$(xid_count); say "基线：Xid=$X0 health=$(health) shm_used=$(shm_used) MemAvailable=$(mem_avail)"
# 生产参数快照：launch.env 是看门狗与控制台下次启动的唯一参数源，必须留底
# （2026-09-27 事故：测试档留在 launch.env 里，生产被起成 65536 上下文 + 80 块小池）
if [ -s "$ENVF" ]; then
  cp -f "$ENVF" "$LAUNCH_BAK" && say ">>> 已备份 launch.env → $LAUNCH_BAK（$(grep -c . "$LAUNCH_BAK") 行）"
else
  say ">>> ⚠️ launch.env 为空/不存在，无可备份（本次回滚将只能靠 inner 缺省）"
fi
if [ "$MODE" != "judge" ]; then
  say ">>> 停看门狗 timer（防自愈抢跑）"
  wd stop
  say ">>> 优雅停止实例"
  bash "$BASE/stop-flash-next-0300.sh" "$PORT" >> /home/ll/deploy/kvoff-c8-window.log 2>&1
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
  chk "health=200（启动完成）" 1 "超时 ${HEALTH_TIMEOUT}s"
  tail -60 "$LOG" | tee -a "$RES"
  rollback
  say "总结：PASS=$PASS FAIL=$FAIL（启动失败已回滚）"
  exit 1
fi
chk "health=200（启动完成）" 0 "耗时见日志"

judge "$OFF"
say ">>> 探针（$PROBE_ARGS）上限 ${PROBE_TIMEOUT:-900}s"
# timeout：2026-09-27 窗口 7 实测探针会被引擎卡死拖着不返回（post 超时 1800s），
# 单靠探针自身保护会让窗口空转十几分钟，还在结果里看不出是卡死还是慢。
timeout "${PROBE_TIMEOUT:-900}" "$PY" "$PROBE" --endpoint "http://127.0.0.1:$PORT/v1" $PROBE_ARGS | tee -a "$RES"
PR=${PIPESTATUS[0]}
chk "探针三项判据（external hits>0 且 CPU→GPU>0 且验证码复述）" "$PR" "退出码 $PR（124=超时/143=被信号打断）"
stall_evidence
metrics_dump
X1=$(xid_count)
chk "零新增 Xid（GPU 未被拷贝路径打挂）" "$([ "$X1" = "$X0" ] && echo 0 || echo 1)" "基线=$X0 现在=$X1"

if [ "$FAIL" -gt 0 ]; then
  say ">>> 有 $FAIL 项失败 ⇒ 自动回滚 FN_KVOFF=0"
  rollback
else
  say ">>> 全绿。要不要把公共区固化成生产态：写进 launch.env + 快启预设 + 弹窗默认（三源同步）"
  wd start
fi
say "总结：PASS=$PASS FAIL=$FAIL  结果文件 $RES"
[ "$FAIL" = "0" ] || exit 1
