#!/bin/bash
# 优雅停 SGLang 18420：SIGTERM 主进程组，等净（绝不无差别 -9 持 CUDA context 进程）。
set -u
SUDO_PASS=${SUDO_PASS:?需要先 export SUDO_PASS=部署机sudo口令，仓库不携带口令}
PIDS=$(ps -eo pid,args | grep "[s]glang.launch_server" | awk "{print \$1}")
CHILD=$(pgrep -f "^sglang::" 2>/dev/null | tr "\n" " ")
ALL="$PIDS $CHILD"
[ -z "$(echo $ALL | tr -d " ")" ] && { echo "[sg-stop] 无 sglang 进程"; exit 0; }
echo "[sg-stop] SIGTERM -> $ALL"
echo "$SUDO_PASS" | sudo -S -p "" kill -TERM $ALL 2>/dev/null
for i in $(seq 1 60); do
  LEFT=$(ps -eo pid,args | grep "[s]glang.launch_server" | awk "{print \$1}")
  [ -z "$LEFT" ] && { echo "[sg-stop] 已干净退出 (${i}s)"; exit 0; }
  sleep 1
done
echo "[sg-stop] 60s 未退净，再发一轮 SIGTERM（仍不 -9；持 CUDA context 强杀有 Xid31 风险）"
echo "$SUDO_PASS" | sudo -S -p "" kill -TERM $LEFT 2>/dev/null
sleep 10
LEFT=$(ps -eo pid,args | grep "[s]glang.launch_server" | awk "{print \$1}")
if [ -n "$LEFT" ]; then
  echo "[sg-stop] 警告：仍存活 $LEFT —— 人工确认后再 SIGKILL"
  exit 1
fi
echo "[sg-stop] 完成"
