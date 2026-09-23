#!/bin/bash
# 等驱动就绪后设置 GPU 功耗上限与持久模式（CMP 170HX 允许 100-300W）
# 由 gpu-power-limit.service 开机调用。所有 nvidia-smi 调用带 timeout，
# 防止坏卡/驱动未就绪时 nvidia-smi 阻塞挂死导致开机设置静默失效。
PL=${PL:-210}
NVS="timeout 20 nvidia-smi"

verify_all() {
  # 所有 GPU 的 power.limit 都等于 PL 才返回 0
  local vals n
  vals=$($NVS --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | awk '{printf "%.0f\n", $1}')
  n=$(printf "%s\n" "$vals" | grep -c .)
  [ "$n" -ge 1 ] || return 1
  ! printf "%s\n" "$vals" | grep -qv "^${PL}$"
}

for i in $(seq 1 90); do
  if $NVS -L >/dev/null 2>&1; then
    $NVS -pm 1 >/dev/null 2>&1
    if verify_all; then
      echo "功耗上限已是 ${PL}W（无需改动）"
      exit 0
    fi
    $NVS -pl "$PL" >/dev/null 2>&1
    if verify_all; then
      echo "已设置功耗上限 ${PL}W"
      exit 0
    fi
  fi
  sleep 2
done
echo "设置功耗上限 ${PL}W 失败：180s 内未确认生效"
exit 1
