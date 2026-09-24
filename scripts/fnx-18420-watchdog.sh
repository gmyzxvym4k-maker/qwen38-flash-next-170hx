#!/bin/bash
# Flash-Next W4A16 (18420) 看门狗 —— 2026-09-23 上线
# 背景：机器反复重启 + 外部 192.168.1.36 自动化频繁操作，18420 被杀后完全不自愈。
# 逻辑：每 30s 探 /health
#   200                  -> 清零失败计数，退出
#   非200 但进程还在     -> 视为启动中/停止中，不动
#   非200 且无实例进程   -> 连续 2 次（约 60s）确认离线 -> 清残留 -> 等显存归零 -> 拉起
# 启动参数与生产定版一致（1M + MTP4 + PLE INT8 heap + KVOFF=1 96GiB，09-24 同步）。
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
LAUNCH_LOG=/home/ll/deploy/vllm-flash-next-w4a16.log
NVSMI=/usr/bin/nvidia-smi

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

mkdir -p "$(dirname "$LOCK")" 2>/dev/null || true
exec 9>"$LOCK" || exit 0
flock -n 9 || exit 0

# 实例进程判据：主进程 comm 恒为 python3，只能按 cmdline 匹配（括号防自匹配）；
# worker/engine 经 setproctitle 改名为 VLLM::*（comm 被截断 15 字符）。
vllm_alive(){
  # 只认主 APIServer（cmdline 含 entrypoints.cli.main serve）。worker 经 setproctitle 改名
  # VLLM::，主进程死后它们成孤儿仍匹配 ^VLLM:: —— 旧版据此误判"实例还活着"而静默不自愈
  # （2026-09-24 10:11 事故：主进程 EngineDead 退出，两个 Worker_PP 孤儿占 126GB 显存，
  #  看门狗 12 分钟无动作）。孤儿在本函数返回假后走清理路径，由 stop 脚本按 ^VLLM:: 收走。
  ps -eo args= 2>/dev/null | grep -q "[e]ntrypoints[./]cli[./]main serve" && return 0
  return 1
}
inner_alive(){ ps -eo args= 2>/dev/null | grep -q "[f]lash-next-w4a16-inner.sh"; }

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
          echo 3124 | sudo -S -p '' env "PATH=$PATH" py-spy dump --pid "$p" > "$f" 2>&1
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
          echo 3124 | sudo -S -p '' cat /proc/$p/stack 2>/dev/null
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
bash /home/ll/deploy/stop-flash-next-w4a16.sh >> "$LOG" 2>&1

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
ENVF=/home/ll/deploy/flash-next-w4a16-launch.env
cd /home/ll/deploy || exit 0
if [ -f "$ENVF" ]; then
  log "显存已归零 -> 按 launch.env 原样拉起（$(grep -c . "$ENVF" 2>/dev/null) 行参数）"
  set -a; . "$ENVF"; set +a
else
  log "显存已归零 -> 无 launch.env，按内置定版拉起（1M + MTP4 + PLE INT8 heap + KVOFF=1 96GiB）"
  export FN_MODEL_PATH=/media/ll/data/models-1m/Qwen3.8-Flash-Next-W4A16-AutoRound-1M
  export FN_LONGCTX=1 FN_1M_MODEL_PATH=/media/ll/data/models-1m/Qwen3.8-Flash-Next-W4A16-AutoRound-1M FN_YARN_FACTOR=4.0
  export FN_MAXLEN=1048576 FN_GPUMEM=0.96 FN_SEQS=3 FN_MBTOKENS=8192 FN_BLOCK=1616
  export FN_SPEC=mtp4 FN_ASYNC=1 FN_PLE_INT8=1 FN_PLE_LOC=heap FN_KVOFF=1 FN_KVOFF_BYTES=103079215104
  export FN_SERVED=qwen3.8-flash-next FN_PORT=18420
fi
setsid nohup bash /home/ll/deploy/start-flash-next-w4a16.sh >> "$LAUNCH_LOG" 2>&1 < /dev/null &
rm -f "$STATE"
log "启动已发起（约 6~8 分钟就绪）"
exit 0
