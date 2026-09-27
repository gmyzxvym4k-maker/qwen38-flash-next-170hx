#!/bin/bash
# Flash-Next W4A16 (18420) 看门狗 —— 2026-09-23 上线
# 背景：机器反复重启 + 外部 <EXTERNAL_HOST> 自动化频繁操作，18420 被杀后完全不自愈。
# 逻辑：每 30s 探 /health
#   200                  -> 清零失败计数，退出
#   非200 但进程还在     -> 视为启动中/停止中，不动
#   非200 且无实例进程   -> 连续 2 次（约 60s）确认离线 -> 清残留 -> 等显存归零 -> 拉起
# 启动参数与生产定版一致（1M + MTP4 + PLE INT8 heap + KVOFF=1 96GiB，09-24 同步）。
# 09-26 改：栈路由不再写死。默认托管【官方 vLLM 0.30.0 新栈】(/home/ll/deploy/vllm-0300)；
#   出现回滚哨兵 /home/ll/deploy/vllm-0300/DISABLED 时自动退回 chroot 旧栈脚本。
#   判活/取证一律与栈无关（见 vllm_alive：按 --port 圈定，两种 APIServer 形态都认）。
# 09-24 新增：卡死侦测——health=200 但引擎日志近 3 分钟出现 shm_broadcast hanging 警告时，
# 用 py-spy 抓 worker 栈存 /home/ll/deploy/stall-dumps/（判内存硬件卡死 vs connector 拷贝路径卡死）。
set -u
# py-spy 装在 ~/.local/bin，systemd user 环境的默认 PATH 不含它；不导出则卡死取证
# 永远只产出 "timeout: 无法运行命令 py-spy"（2026-09-24 实锤，stall-dumps 3 个空文件）。
export PATH="$HOME/.local/bin:$PATH"
PORT=18420
LOG=/home/ll/deploy/fnx-watchdog.log
STATE=/tmp/fnx-watchdog-fail
LOCK=/tmp/fnx-watchdog.lock
NVSMI=/usr/bin/nvidia-smi
BASE_NEW=/home/ll/deploy/vllm-0300
STACK_DISABLED=0
[ -f "$BASE_NEW/DISABLED" ] && STACK_DISABLED=1
if [ "$STACK_DISABLED" = "0" ]; then
  START_SCRIPT=$BASE_NEW/start-flash-next-0300.sh
  STOP_SCRIPT=$BASE_NEW/stop-flash-next-0300.sh
  ENVF=$BASE_NEW/launch.env
  INNER_PAT='flash-next-0300-inner.sh'
else
  START_SCRIPT=/home/ll/deploy/start-flash-next-w4a16.sh
  STOP_SCRIPT=/home/ll/deploy/stop-flash-next-w4a16.sh
  ENVF=/home/ll/deploy/flash-next-w4a16-launch.env
  INNER_PAT='flash-next-w4a16-inner.sh'
fi
# 引擎日志取两个候选里 mtime 最新的那个：正在写日志的就是当前活的栈，
# 这样回滚/换栈后卡死取证不必再改脚本（旧版写死 w4a16.log，换栈即哑）。
pick_log(){
  local best="" bt=0 f t
  for f in /home/ll/deploy/vllm-flash-next-0300.log /home/ll/deploy/vllm-flash-next-w4a16.log; do
    [ -f "$f" ] || continue
    t=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    [ "$t" -gt "$bt" ] && { bt=$t; best=$f; }
  done
  echo "${best:-/home/ll/deploy/vllm-flash-next-0300.log}"
}
LAUNCH_LOG=$(pick_log)

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# ---------- 自愈安全模式（2026-09-27）----------
# 看门狗的职责是把**服务**拉起来，不是重放某个实验档位。内存二级缓存（官方 simple offload
# 或经典 KVOFF）属可选优化：若它恰是崩溃诱因，按 launch.env 原样重放就变成「崩溃→重启→再崩」
# 的循环，用户看到的是服务一直不起来。故自愈时一律剥掉 offload 档位（并把删掉的记进日志），
# 让生产回到已验证的无二级缓存定版；要带档位跑请手工/控制台启动（闩锁 MANUAL_STOP 同理适用）。
# 返回被剥离项的摘要（空串=无需剥离）。
# 【坑】必须**直接调用**、由函数写全局 TIER_DROPPED：写成 D=$(selfheal_sanitize) 时函数体在
# 命令替换子壳里跑，里面的 unset 影响不到本进程——日志会说"已剥离"而实际档位还在（2026-09-27
# 隔离测试抓到的假绿灯）。故此处只写全局变量、不回显。
TIER_DROPPED=""
selfheal_sanitize(){
  local dropped="" pat_env='VLLM_USE_SIMPLE_KV_OFFLOAD'
  if [ -n "${FN_SIMPLE_OFFLOAD:-}" ]; then
    dropped="$dropped FN_SIMPLE_OFFLOAD=${FN_SIMPLE_OFFLOAD}"; unset FN_SIMPLE_OFFLOAD
  fi
  if [ "${FN_KVOFF:-0}" = "1" ]; then
    dropped="$dropped FN_KVOFF=1"; unset FN_KVOFF
  fi
  # FN_EXTRA_ENV/FN_EXTRA_ARGS 在本项目只被 offload 实验用过；含 offload 关键字就整体丢弃，
  # 不做逐词过滤（args 里的 JSON 带空格，逐词重组有改写风险）。命中即记录原值片段便于取证。
  case "${FN_EXTRA_ENV:-}" in
    *"$pat_env"*) dropped="$dropped FN_EXTRA_ENV=[${FN_EXTRA_ENV:0:80}]"; unset FN_EXTRA_ENV;;
  esac
  case "${FN_EXTRA_ARGS:-}" in
    *kv-offloading-size*|*kv-transfer-config*)
      dropped="$dropped FN_EXTRA_ARGS=[${FN_EXTRA_ARGS:0:80}]"; unset FN_EXTRA_ARGS;;
  esac
  TIER_DROPPED="${dropped# }"
}

mkdir -p "$(dirname "$LOCK")" 2>/dev/null || true
exec 9>"$LOCK" || exit 0
flock -n 9 || exit 0

# ---- 人工停止闩锁（09-26）----
# 控制台/手工执行 stop 脚本会 touch /home/ll/deploy/fnx-manual-stop；存在期间看门狗完全静默
# （不探活、不卡死取证、不清理、不拉起）——根治「控制台点停止，60s 后又被拉起来」的表象。
# 任一 start 脚本入口清除本闩；看门狗自己的拉起分支在发起启动前也会清除（清残留步骤走
# stop 脚本会置闩，属预期，不能让自愈流程自己把自愈锁死）。
MANUAL_STOP=/home/ll/deploy/fnx-manual-stop
if [ -f "$MANUAL_STOP" ]; then exit 0; fi

# 实例进程判据：主进程 comm 恒为 python3，只能按 cmdline 匹配（括号防自匹配）；
# worker/engine 经 setproctitle 改名为 VLLM::*（comm 被截断 15 字符）。
vllm_alive(){
  # 只认主 APIServer（cmdline 含 entrypoints.cli.main serve）。worker 经 setproctitle 改名
  # VLLM::，主进程死后它们成孤儿仍匹配 ^VLLM:: —— 旧版据此误判"实例还活着"而静默不自愈
  # （2026-09-24 10:11 事故：主进程 EngineDead 退出，两个 Worker_PP 孤儿占 126GB 显存，
  #  看门狗 12 分钟无动作）。孤儿在本函数返回假后走清理路径，由 stop 脚本按 ^VLLM:: 收走。
  # 与栈无关的判据：主 APIServer 的 comm 是 vllm（宿主 venv 0.30.0 实测）或 python*（chroot 旧栈
  # 的 -m vllm.entrypoints.cli.main 形态），cmdline 必含 serve 与本端口。绝不能再按某一种命令行
  # 字样写死 —— 换栈后判活恒假会让看门狗在实例健康时反复清理重启（09-26 迁栈必改项）。
  local d pid comm args
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    comm=$(cat "$d/comm" 2>/dev/null) || continue
    case "$comm" in vllm|python*) ;; *) continue;; esac
    args=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    case "$args" in *serve*) ;; *) continue;; esac
    case "$args" in *"--port $PORT"*) return 0;; esac
  done
  return 1
}
inner_alive(){ ps -eo args= 2>/dev/null | grep -qE "[f]lash-next-(0300|w4a16)-inner\.sh"; }

code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "http://127.0.0.1:${PORT}/health" 2>/dev/null)
[ -z "$code" ] && code=000

# ---- 卡死侦测（09-24 新增）：引擎卡住时 APIServer 仍回 200，普通探活看不见 ----
STALL_MARK=/tmp/fnx-stall-caught
# 补丁5：卡死后 APIServer 的 /health 也会转 000（19:26 实锤），故进程活着就查
if [ "$code" = "200" ] || vllm_alive; then
  # 窗口 400→5000 行：16:25 实锤警告距尾部 2230 行（KVOFF 调试日志+访问日志洪峰），400 行看不见；
  # 加时间校验：警告行时间戳（UTC）须在最近 6 分钟内，防旧警告滞留窗口引起误 dump。
  wline=$(tail -n 5000 "$LAUNCH_LOG" 2>/dev/null | grep -aE "No available shared memory broadcast block found in (60|[0-9]{3,}) seconds" | tail -1)
  wts=$(echo "$wline" | grep -oE "[0-9]{2}:[0-9]{2}:[0-9]{2}" | tail -1)
  if [ -n "$wts" ]; then
    w_epoch=$(date -u -d "today $wts" +%s 2>/dev/null || echo 0)
    n_epoch=$(date -u +%s)
    [ "$w_epoch" -gt 0 ] && [ $((n_epoch - w_epoch)) -gt 360 ] && wts=""
  fi
  if [ -n "$wts" ]; then
    now=$(date +%s); last=0
    [ -f "$STALL_MARK" ] && last=$(cat "$STALL_MARK" 2>/dev/null || echo 0)
    case "$last" in ''|*[!0-9]*) last=0;; esac
    if [ $((now-last)) -ge 90 ]; then
      echo "$now" > "$STALL_MARK"
      DD=/home/ll/deploy/stall-dumps; mkdir -p "$DD"
      # 补丁5：GPU 侧快照（冻结时 SM/显存控制器占用是判据）
      timeout 15 nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,power.draw \
        --format=csv > "$DD/gpu-$(date +%m%d-%H%M%S).txt" 2>&1 || true
      for p in $(ps -eo pid,comm= 2>/dev/null | awk '/VLLM::(Worker|EngineCore)/ {print $1}'); do
        n=$(ps -o comm= -p "$p" 2>/dev/null | tr -d ':')
        f="$DD/stall-$(date +%m%d-%H%M%S)-$p-$n.txt"
        if command -v py-spy >/dev/null 2>&1; then
          # worker 属 root（chroot 实例），裸 py-spy 报 Permission Denied 只落 99 字节空壳；
          # sudo 又会重置 PATH，必须 env "PATH=$PATH" 才找得到 ~/.local/bin/py-spy（09-24 实锤）
          echo "${SUDO_PASS:?本副本已脱敏：先 export SUDO_PASS=<部署机 sudo 口令>}" | sudo -S -p '' env "PATH=$PATH" py-spy dump --pid "$p" > "$f" 2>&1
        else
          { echo "=== py-spy 不可用，退化为 /proc 取证 ==="
            echo "--- threads: tid comm wchan state ---"
            for t in /proc/$p/task/*; do
              echo "$(basename "$t") $(cat "$t/comm" 2>/dev/null) wchan=$(cat "$t/wchan" 2>/dev/null) state=$(awk '{print $3}' "$t/stat" 2>/dev/null)"
            done
            echo "--- kernel stack ---"
            cat "/proc/$p/stack" 2>/dev/null
          } > "$f" 2>&1
        fi
        # 补丁5：附内核栈与 wchan（py-spy 只见 Python 帧，native 阻塞点看这里）
        { echo "--- kernel stack ---"
          echo "${SUDO_PASS:?本副本已脱敏：先 export SUDO_PASS=<部署机 sudo 口令>}" | sudo -S -p '' cat /proc/$p/stack 2>/dev/null
          echo "--- wchan ---"; cat /proc/$p/wchan 2>/dev/null; echo
        } >> "$f"
        [ -s "$f" ] && log "卡死取证: pid=$p -> $(basename "$f")（$(wc -c < "$f") 字节）"
      done
    fi
  else
    rm -f "$STALL_MARK"
  fi
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
"$STOP_SCRIPT" "$PORT" >> "$LOG" 2>&1

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

# 拉起参数优先跟随「控制台最近一次启动」落盘的 launch.env，消除看门狗硬编码与生产配置
# 漂移（实锤差异：PLE_INT8 1/0、FN_SEQS 3/4、FN_GPUMEM 0.96/0.95）；无 launch.env 时回退内置定版。
cd /home/ll/deploy || exit 0
if [ -f "$ENVF" ]; then
  log "显存已归零 -> 按 launch.env 原样拉起（$(grep -c . "$ENVF" 2>/dev/null) 行参数）"
  set -a; . "$ENVF"; set +a
else
  log "显存已归零 -> 无 launch.env，按内置定版拉起（disabled=$STACK_DISABLED；1M + MTP4 + block1616，2026-09-26 与生产同步）"
  export FN_MODEL_PATH=/media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound
  export FN_LONGCTX=1 FN_1M_MODEL_PATH=/media/ll/data/models-1m/Qwen3.8-Flash-Next-W4A16-AutoRound-1M FN_YARN_FACTOR=4
  export FN_MAXLEN=1048576 FN_GPUMEM=0.95 FN_SEQS=4 FN_MBTOKENS=8192 FN_BLOCK=1616
  export FN_SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":4,\"use_local_argmax_reduction\":false}" FN_ASYNC=1
  # 新栈无 PLE 精度/位置与 KVOFF 档位（0.30.0 只有 BF16 锁页；传了会触发 inner 的忽略告警）
  [ "$STACK_DISABLED" = "1" ] && export FN_PLE_INT8=1 FN_PLE_LOC=disk FN_KVOFF=0
  export FN_SERVED=qwen3.8-flash-next FN_PORT=18420
fi
selfheal_sanitize
[ -n "$TIER_DROPPED" ] && log "自愈安全模式：已剥离内存二级缓存档位[$TIER_DROPPED]，按无二级缓存定版拉起"
rm -f "$MANUAL_STOP"   # 本看门狗的拉起意图优先于闩锁（前面"清残留"调 stop 脚本会置闩）
setsid nohup bash "$START_SCRIPT" >> "$LAUNCH_LOG" 2>&1 < /dev/null &
rm -f "$STATE"
log "启动已发起（约 6~8 分钟就绪）"
exit 0
