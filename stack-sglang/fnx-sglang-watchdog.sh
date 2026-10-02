#!/bin/bash
# SGLang 18420 看门狗（timer 驱动）：health 非 200 且主进程消失 → 清残留 → 等显存归零 → 重拉。
# 与 vLLM 栈看门狗互斥：本脚本在岗时 fnx-18420-watchdog.timer 应保持停止。
set -u
LOG=/home/ll/deploy/fnx-sglang-watchdog.log
BASE=/home/ll/deploy/sglang-18420
SUDO_PASS=${SUDO_PASS:?需要先 export SUDO_PASS=部署机sudo口令，仓库不携带口令}
say(){ echo "[$(date -Is)] $*" >> "$LOG"; }

HC=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:18420/health)
MAIN=$(ps -eo pid,args | grep "[s]glang.launch_server" | awk "{print \$1}" | head -1)
[ "$HC" = "200" ] && exit 0
# health 不通但进程在：可能启动中/卡死。启动宽限 15 分钟（1M 冷启 ~10min）。
if [ -n "$MAIN" ]; then
  AGE_S=$(( $(date +%s) - $(stat -c %Y /proc/$MAIN 2>/dev/null || echo 0) ))
  if [ "$AGE_S" -lt 900 ]; then exit 0; fi
  say "主进程 $MAIN 存在但 health=$HC 且年龄 ${AGE_S}s>900s → 视为卡死，SIGTERM"
  echo "$SUDO_PASS" | sudo -S -p "" kill -TERM $MAIN 2>/dev/null
  sleep 20
fi
# 清残留（含 sglang:: 子进程），等显存归零（≤120s）
for i in $(seq 1 12); do
  LEFT=$(ps -eo pid,args | grep "[s]glang.launch_server\|[s]glang::" | awk "{print \$1}")
  [ -z "$LEFT" ] && break
  echo "$SUDO_PASS" | sudo -S -p "" kill -TERM $LEFT 2>/dev/null
  sleep 5
done
LEFT=$(ps -eo pid,args | grep "[s]glang.launch_server\|[s]glang::" | awk "{print \$1}")
if [ -n "$LEFT" ]; then
  say "警告：SIGTERM 未退净 $LEFT，本轮放弃（防强杀诱发 Xid），下轮再试"; exit 0
fi
for i in $(seq 1 24); do
  BUSY=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk "\$1>2000{c++} END{print c+0}")
  [ "$BUSY" = "0" ] && break
  sleep 5
done
[ "$BUSY" != "0" ] && { say "显存未归零，放弃本轮"; exit 0; }
say "重拉 SGLang 18420"
echo "$SUDO_PASS" | sudo -S -p "" sh -c "set -a; . $BASE/launch.env 2>/dev/null; set +a; exec setsid bash $BASE/sglang-inner.sh >> /home/ll/deploy/sglang-18420.log 2>&1 </dev/null" &
exit 0
