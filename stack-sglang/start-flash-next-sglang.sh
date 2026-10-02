#!/bin/bash
# 宿主 wrapper：root 起 SGLang 版 18420（PLE 锁页需 memlock）。
# 门禁：①fnx vLLM 看门狗必须已停 ②端口空闲 ③显存归零（SG_FORCE=1 跳过③）。
# 参数透传：SG_* 环境变量落盘 envfile，sudo 内 source（与 FN_*/launch.env 同机制）。
set -u
BASE=/home/ll/deploy/sglang-18420
LOG=${SG_LOG:-/home/ll/deploy/sglang-18420.log}
PORT=${SG_PORT:-18420}
SUDO_PASS=${SUDO_PASS:?需要先 export SUDO_PASS=部署机sudo口令，仓库不携带口令}
ENVFILE=$BASE/launch.env

{
  echo "# generated $(date -Is)"
  for v in $(compgen -e | grep -E "^SG_[A-Z0-9_]+$" | grep -v "^SG_LOG$" | sort); do
    val=$(printenv "$v")
    [ -n "$val" ] && printf "%s=%q\n" "$v" "$val"
  done
} > "$ENVFILE"
chmod 644 "$ENVFILE"

if systemctl --user is-active --quiet fnx-18420-watchdog.timer; then
  echo "[sg-start] 拒绝：fnx-18420-watchdog.timer 仍 active（会重拉 vLLM 抢端口）。先 systemctl --user stop fnx-18420-watchdog.timer" | tee -a "$LOG"; exit 1
fi
if curl -s -o /dev/null -m 3 "http://127.0.0.1:$PORT/health"; then
  echo "[sg-start] 拒绝：端口 $PORT 已有服务（先跑 stop-flash-next-sglang.sh）" | tee -a "$LOG"; exit 1
fi
if [ "${SG_FORCE:-0}" != "1" ]; then
  BUSY=$(timeout 20 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F", " "\$2 > 2000 {print \$1\":\"\$2\"MiB\"}" | tr "\n" " ")
  if [ -n "$BUSY" ]; then
    echo "[sg-start] 拒绝：显存未归零 -> $BUSY（确认无孤儿后 SG_FORCE=1 强启）" | tee -a "$LOG"; exit 1
  fi
fi

# sudo 铁律：密码管道 + </dev/null 放 sh -c 内部；setsid 脱 ssh
echo "$SUDO_PASS" | sudo -S -p "" sh -c "set -a; . $ENVFILE 2>/dev/null; set +a; exec setsid bash $BASE/sglang-inner.sh >> $LOG 2>&1 </dev/null" &
echo "[sg-start] $(date -Is) 已发起（日志 $LOG，envfile $ENVFILE）" | tee -a "$LOG"
