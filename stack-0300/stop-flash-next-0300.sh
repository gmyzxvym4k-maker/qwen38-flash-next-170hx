#!/bin/bash
# 停止 0.30.0 路线的 Flash-Next 实例。
#
#   用法：bash /home/ll/deploy/vllm-0300/stop-flash-next-0300.sh [端口]     缺省 18420
#
# 【为什么必须按端口圈定】8889 控制台的停止按钮会调 `bash <stopScript> <port>`，
# 而本机可能同时跑着别的 vLLM 实例（如 8000/8001）。旧版按 "vllm serve" 广匹配会
# 把它们一起 SIGTERM —— 跨端口误杀。现在：
#   1) 主进程 = comm 是 python*/VLLM::* 且 cmdline 含 "vllm serve" 且含 "--port <P>"；
#   2) 后代进程 = 沿 /proc/*/stat 的 ppid 链自顶向下收集（EngineCore / Worker_PP*）；
#   3) 孤儿 = ppid==1 且 /proc/<pid>/environ 里有本栈指纹（PYTHONPATH 含 vllm-0300
#      或 VLLM_RT_PATCHES=1）—— 09-24 教训：PLE worker / resource_tracker 是
#      `python -c from multiprocessing.spawn spawn_main` 形态，comm 不是 VLLM::，
#      漏抓会留下占显存/占内存的孤儿，导致下次启动 shm_broadcast 超时。
#      用指纹而不是"所有 multiprocessing 进程"，避免误杀别人的 python。
# 铁律（09-15/19/24 实锤）：
#   · 绝不先手 SIGKILL 持 CUDA 上下文的进程 —— 会诱发 Xid31 → GPU 降级 → 只能整机重启。
#     正解：SIGTERM 等满 90s；僵尸(Z)不计入存活；超时不退才 SIGKILL 兜底。
#   · 启动新实例前必须确认两卡显存归零（本脚本末尾会等并报告）。
set -u

PORT=${1:-${FN_PORT:-18420}}
case "$PORT" in (*[!0-9]*|'') PORT=18420;; esac

# ---- 自提权：进程属 root，ll 直接 kill 会静默 EPERM（09-18 实锤）
# 09-26 修复：原先缺 SUDO_PASS 时只试 sudo -n，控制台 spawn 不带该变量也没有免密 sudoers，
# 直接"需要密码"失败——停止按钮形同虚设。改为本机既定模式：密码管道（口令经 SUDO_PASS 注入，本副本已脱敏；机器版可用
# SUDO_PASS 覆盖），与 start-flash-next-*.sh / stop-flash-next-w4a16.sh 一致。
# 铁律：sudo -S 的密码走管道，命令尾部绝不加 < /dev/null（会覆盖管道）。
if [ "$(id -u)" != "0" ]; then
  echo "${SUDO_PASS:?本副本已脱敏：先 export SUDO_PASS=<部署机 sudo 口令>}" | sudo -S -p '' bash "$0" "$PORT"
  exit $?
fi

MARK_DIR="vllm-0300"

# 人工停止闩锁（09-26）：代操作者声明"这次是要主动停"，看门狗见此文件全程静默、不再拉起；
# 启动脚本入口负责清除。看门狗自愈流程中的"清残留"也走本脚本会置闩，但它在拉起前自行 rm，
# 不影响自愈。放在提权之后：闩锁文件属 ll，root 创建后要改属主，免得 ll 的启动脚本删不掉。
if [ -z "${STOP_LIST_ONLY:-}" ]; then
  : > /home/ll/deploy/fnx-manual-stop
  chown ll:ll /home/ll/deploy/fnx-manual-stop 2>/dev/null || true
  chmod 644 /home/ll/deploy/fnx-manual-stop 2>/dev/null || true
  echo "[stop] 已置人工停止闩锁 fnx-manual-stop（看门狗将不再拉起，直到任一 start 脚本清除）"
fi

# 主进程判据：comm 白名单必须含 **vllm** —— 0.30.0 栈的 APIServer comm 实测就是 "vllm"
# （venv/bin/vllm 入口脚本，不是 python3.11；旧 chroot 栈才是 python3.12），
# 只写 python*|VLLM::* 会一个都找不到（09-26 实锤：stop 报「无存活进程」而实例在跑）。
# 同时显式排除 shell/观察类 comm，避免把运维命令自身或 stop 脚本算进目标。
root_pids() {
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    self_guard "$pid" && continue
    comm=$(cat "$d/comm" 2>/dev/null) || continue
    case "$comm" in vllm|python*|VLLM::*) ;; *) continue;; esac
    args=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    case "$args" in *stop-flash-next*|*start-flash-next*) continue;; esac
    case "$args" in *"serve "*|"--port $PORT"*) ;; *) continue;; esac
    case "$args" in *vllm*) ;; *) continue;; esac
    case "$args" in *"--port $PORT"*) echo "$pid";; esac
  done
}

descendants() {  # $1=空格分隔的父 pid 列表；输出含自身
  # 【09-26 实锤修复】原先写成 `local frontier="$1" out="$frontier"`：同一条 local 语句里
  # 右侧 $frontier 展开时左边的局部变量尚未创建，读到的是不存在的全局变量 → set -u 下
  # 函数当场炸死，且调用方 find_pids 的 2>/dev/null 把报错吞净 → PIDS 恒空、"无存活进程"、
  # alive_count 恒 0（90s 等待秒过）。本脚本上线以来从未真正杀过进程，直到控制台停止按钮
  # 失灵被用户实锤才现形。local 必须分行；循环变量 f 也一并局部化。
  local frontier="$1"
  local out="$frontier" next p d ppid f
  while [ -n "$frontier" ]; do
    next=""
    for d in /proc/[0-9]*; do
      p=${d#/proc/}
      ppid=$(awk '{print $4}' "$d/stat" 2>/dev/null) || continue
      for f in $frontier; do
        [ "$ppid" = "$f" ] && { case " $out " in *" $p "*) ;; *) out="$out $p"; next="$next $p";; esac; }
      done
    done
    frontier="$next"
  done
  echo "$out"
}

self_guard() {  # 绝不把「自己和自己所在的调用链」当成目标（setsid/sudo 包装下会自杀）
  case "$1" in "$$"|"${PPID}"|"$BASHPID") return 0;; esac
  return 1
}

orphans() {  # 本栈遗留孤儿（ppid=1 + 指纹命中）
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    self_guard "$pid" && continue
    ppid=$(awk '{print $4}' "$d/stat" 2>/dev/null) || continue
    [ "$ppid" = "1" ] || continue
    comm=$(cat "$d/comm" 2>/dev/null) || continue
    case "$comm" in vllm|python*|VLLM::*) ;; *) continue;; esac   # 只认解释器/vllm/VLLM::，sh/bash 一律跳过
    args=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    case "$args" in *stop-flash-next*) continue;; esac
    case "$args" in *multiprocessing*|*VLLM::*|*vllm*) ;; *) continue;; esac
    env0=$(tr '\0' '\n' < "$d/environ" 2>/dev/null) || continue
    case "$env0" in
      *"$MARK_DIR"*|*VLLM_RT_PATCHES=1*) echo "$pid" ;;
    esac
  done
}

find_pids() {  # 主进程+后代，外加孤儿；统一 sort -u 去重（alive_count 不能重复计数）
  {
    R=$(root_pids)
    [ -n "$R" ] && descendants "$R"
    orphans
  } 2>/dev/null | sort -u
}

alive_count() {
  n=0
  for pid in $(find_pids); do
    st=$(sed "s/.*) //" /proc/$pid/stat 2>/dev/null | cut -d" " -f1)
    [ -n "$st" ] && [ "$st" != "Z" ] && n=$((n+1))
  done
  echo "$n"
}

PIDS=$(find_pids | tr '\n' ' ')
if [ -n "${STOP_LIST_ONLY:-}" ]; then
  echo "[stop] (只列目标，不动手) 端口 $PORT -> ${PIDS:-无}"
  exit 0
fi
if [ -z "${PIDS// /}" ]; then
  echo "[stop] 端口 $PORT 无存活进程"
else
  echo "[stop] 端口 $PORT 目标：$PIDS"
  kill -TERM $PIDS 2>/dev/null
  for i in $(seq 1 90); do
    [ "$(alive_count)" = "0" ] && break
    sleep 1
  done
  LEFT=$(alive_count)
  if [ "$LEFT" != "0" ]; then
    echo "[stop] 90s 后仍存活 $LEFT 个，SIGKILL 兜底（随后务必查 dmesg 是否新增 Xid）"
    kill -9 $(find_pids) 2>/dev/null
    sleep 3
  else
    echo "[stop] 全部优雅退出（第 ${i}s 秒清零）"
  fi
fi

echo "[stop] 等待显存释放："
MAX=99999
for i in $(seq 1 60); do
  OUT=$(timeout 20 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')
  MAX=$(echo $OUT | awk '{m=0; for(i=1;i<=NF;i+=2){v=$(i+1); if(v>m)m=v} print m}')
  echo "  [${i}] $OUT (max=${MAX:-?} MiB)"
  [ "${MAX:-99999}" -lt 2000 ] && break
  sleep 2
done
if [ "${MAX:-99999}" -ge 2000 ]; then
  echo "[stop] 显存未归零：多为驱动回收延迟或幽灵显存（compute-apps 已空但 memory.used 仍高），等 1~2 分钟再查；持续不归零=GPU 已降级，需 reboot"
  exit 1
fi
echo "[stop] 显存已归零，可以启动新实例"
