#!/bin/bash
# kvoff-start-test.sh —— 起 KVOFF 测试实例（64GiB 公共区 / 65536 上下文 / 80 blocks / 带 key 诊断）
#
# 用途：迭代期反复「停→起」用，不跑窗口判据、不回滚。收尾务必手工回生产：
#   bash /home/ll/deploy/kvoff-restore-prod.sh
set -uo pipefail
BASE=${BASE:-/home/ll/deploy/vllm-0300}
ENVF=$BASE/launch.env
OVR=${FN_OVERRIDES_FILE:-/home/ll/deploy/kvoff-c10-window.overrides}
WLOG=${C10_WINDOW_LOG:-/home/ll/deploy/kvoff-c10-window.log}
SUDO_PASS=${SUDO_PASS:?本副本已脱敏：先 export SUDO_PASS=<部署机 sudo 口令>}

[ -f "$ENVF" ] && { set -a; . "$ENVF"; set +a; }
[ -f "$OVR" ] && { set -a; . "$OVR"; set +a; }
export FN_KVOFF=${FN_KVOFF:-1}
export FN_KVOFF_SHARED=${FN_KVOFF_SHARED:-1}
export FN_KVOFF_BYTES=${FN_KVOFF_BYTES:-68719476736}
export FN_KVOFF_PENDING_TTL=${FN_KVOFF_PENDING_TTL:-120}
export FN_KVOFF_JOB_TTL=${FN_KVOFF_JOB_TTL:-180}
export FN_KVOFF_DEBUG=${FN_KVOFF_DEBUG:-1}
export SUDO_PASS

setsid bash "$BASE/start-flash-next-0300.sh" >> "$WLOG" 2>&1 < /dev/null &
disown || true
echo "submitted: FN_KVOFF=$FN_KVOFF bytes=$FN_KVOFF_BYTES debug=$FN_KVOFF_DEBUG (log $WLOG)"
