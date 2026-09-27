#!/bin/bash
# kvoff-accept.sh —— 生产档（1M 配置）内存二级缓存验收：起实例 → 挤池探针 → 恢复生产
# 结果写 /home/ll/deploy/kvoff-accept.result（一路追加，抗 ssh 超时/外部干扰取证）
#
# 2026-09-27 加固（事故：18:16 窗口停掉了「正在启动」的生产实例，用户察觉"好像蹦了"）：
#   ① 前置门禁 preflight()：停服前必须先确认生产 health=200；若进程在但未就绪（判定为启动中）
#      则最多等 KVOFF_ACCEPT_WAIT 秒（缺省 240）等它就绪，仍不健康就中止（exit 2），绝不硬停；
#   ② flock 单实例锁，防两个验收窗口互相踩；
#   ③ KVOFF_ACCEPT_FORCE=1 可显式越过门禁（仅在明知生产已离线时使用）。
set -uo pipefail
BASE=/home/ll/deploy/vllm-0300
RES=/home/ll/deploy/kvoff-accept.result
KVOFF_GIB=${KVOFF_GIB:-48}          # 缺省 48：32 档在 1M 池上容量≈池容，结构上测不出二级缓存
FLUSHES=${FLUSHES:-13}              # 缺省 13：13×~116k≈1.51M > GPU 池 1.207M，才挤得动显存
LOG=/home/ll/deploy/vllm-flash-next-0300.log
WAIT=${KVOFF_ACCEPT_WAIT:-240}     # 前置门禁最多等生产就绪的秒数（0=不等，直接判定）
FORCE=${KVOFF_ACCEPT_FORCE:-0}
SOAK=${SOAK_IF_PASS:-0}            # 1=探针判定 PASS 后不恢复生产，直接带着二级缓存进入 soak
export XDG_RUNTIME_DIR=/run/user/1000
EXT_HOST="${EXT_HOST:-}"
export SUDO_PASS="${SUDO_PASS:?本副本已脱敏：先 export SUDO_PASS=<部署机 sudo 口令>}"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RES"; }

# 18420 上是否已有 vLLM serve 进程（用 python 直读 /proc，避免 ps|grep 自匹配）
inst_pid() {
  /home/ll/vllm-env/bin/python - <<'PY' 2>/dev/null
import os
for pid in os.listdir('/proc'):
    if not pid.isdigit():
        continue
    try:
        argv = [x.decode('utf-8', 'replace') for x in open(f'/proc/{pid}/cmdline', 'rb').read().split(b'\0') if x]
    except OSError:
        continue
    if not argv:
        continue
    if 'serve' in argv and '18420' in argv:
        print(pid)
        break
PY
}

health() { curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:18420/health; }

# 返回 0=可以停服开窗口；1=不可以（调用方应中止）
preflight() {
  local h p waited=0
  h=$(health)
  if [ "$h" = "200" ]; then
    log "前置门禁：生产 health=200，可开窗口"
    return 0
  fi
  if [ "$FORCE" = "1" ]; then
    log "前置门禁：health=$h 但 KVOFF_ACCEPT_FORCE=1，强制继续"
    return 0
  fi
  p=$(inst_pid || true)
  if [ -z "$p" ]; then
    log "前置门禁：health=$h 且无 18420 实例进程（生产本就离线），按现状继续"
    return 0
  fi
  log "前置门禁：health=$h 但实例进程存在（pid=$p）= 启动中，等它就绪（最多 ${WAIT}s），绝不停掉半启动实例"
  while [ "$waited" -lt "$WAIT" ]; do
    sleep 10; waited=$((waited + 10))
    h=$(health)
    log "  门禁等待 ${waited}s health=$h"
    [ "$h" = "200" ] && { log "前置门禁：已就绪（等了 ${waited}s），可开窗口"; return 0; }
  done
  log "前置门禁：实例仍未就绪（health=$h，等了 ${waited}s）→ 中止本次验收，不动生产（可设 KVOFF_ACCEPT_FORCE=1 强制）"
  return 1
}

mkdir -p /tmp
exec 9>/tmp/kvoff-accept.lock
if ! flock -n 9; then
  echo "[$(date +%H:%M:%S)] 已有验收窗口在跑（/tmp/kvoff-accept.lock），本次退出" | tee -a "$RES"
  exit 3
fi

: > "$RES"
log "=== 验收开始：KVOFF_GIB=$KVOFF_GIB flushes=$FLUSHES（wait=${WAIT}s force=$FORCE）==="
if ! preflight; then
  log "=== 验收中止（生产未动，看门狗未停）==="
  exit 2
fi

log "停看门狗 + 停实例"
systemctl --user stop fnx-18420-watchdog.timer 2>/dev/null || true
bash "$BASE/stop-flash-next-0300.sh" 18420 >> "$RES" 2>&1

log "启动：生产档（inner 缺省 1M/YaRN×4/block1616/MTP4）+ SimpleCPUOffloadConnector ${KVOFF_GIB}GiB"
cd "$BASE" || exit 1
# 用 inner 的一等开关 FN_SIMPLE_OFFLOAD（它自己 export VLLM_USE_SIMPLE_KV_OFFLOAD=1 + 追加
# --kv-offloading-size），比 FN_EXTRA_ENV/FN_EXTRA_ARGS 旁路更可读；2026-09-27 已 dry-run 验证。
setsid env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/ll \
  FN_SIMPLE_OFFLOAD="$KVOFF_GIB" \
  bash "$BASE/start-flash-next-0300.sh" </dev/null >/dev/null 2>&1 &

H=000
for i in $(seq 1 80); do
  H=$(health)
  log "等待就绪 #$i health=$H"
  [ "$H" = "200" ] && break
  sleep 10
done
if [ "$H" != "200" ]; then
  log "启动失败/超时，直接恢复生产"
  setsid nohup bash /home/ll/deploy/kvoff-restore-prod.sh >/dev/null 2>&1 < /dev/null &
  exit 1
fi

log "连接器判据："
grep -a 'SimpleCPUOffloadConnector: role' "$LOG" | tail -1 | tee -a "$RES"

# 容量自检（2026-09-27 加）：上一轮 32GiB+9 挤池窗口 hits=0 是构造性必然——容量≈GPU 池、
# 挤池量<池容。这里在跑探针前先算清楚「挤得动显存吗 / 挤完还留在 CPU 档吗」，避免再浪费窗口。
log "容量自检（本次启动日志）："
SELFCHECK=$(/home/ll/vllm-env/bin/python - "$LOG" "$FLUSHES" <<'PY' 2>&1
import math, re, sys
log, flushes = sys.argv[1], int(sys.argv[2])
txt = open(log, errors="ignore").read()
blocks = [int(m) for m in re.findall(r"SimpleCPUOffloadWorker \[CPU\]: 1 tensors, (\d+) CPU blocks", txt)][-2:]
pools = [int(m.replace(",", "")) for m in re.findall(r"GPU KV cache size: ([\d,]+) tokens", txt)]
if not blocks or not pools:
    print("[容量自检] 日志里缺 SimpleCPUOffloadWorker / GPU KV cache size 行，跳过自检")
    raise SystemExit(0)
PER = 116000                       # 每篇挤池文档的 token 量级
cap = min(blocks) * 1616
pool = pools[-1]
churn = flushes * PER
print("[容量自检] CPU 档 blocks/rank=%s → 容量≈%d token；GPU 池=%d token；容量/池=%.2fx"
      % (blocks, cap, pool, cap / pool))
print("[容量自检] 计划挤池≈%d token：挤出显存 %s；挤完仍留在 CPU 档 %s"
      % (churn, "✓" if churn >= pool else "✗（挤不动，hits 必为 0）",
         "✓" if churn < cap else "✗（会被 LRU 冲掉，请减小 FLUSHES 或加大 KVOFF_GIB）"))
need = math.ceil(pool * 1.05 / PER)          # 至少比池容多 5% 才挤得动
maxf = max(1, int(cap * 0.90 // PER))        # 最多用掉 CPU 档 90%
rec = min(max(need, 1), maxf) if maxf >= need else maxf
print("[容量自检] 推荐 FLUSHES：need=%d（>池 5%%）、max=%d（≤档 90%%）→ RECOMMEND_FLUSHES=%d%s"
      % (need, maxf, rec, "" if maxf >= need else "（容量压不住：以 max 为准，命中判据会打折）"))
PY
)
echo "$SELFCHECK" | tee -a "$RES"
REC=$(echo "$SELFCHECK" | sed -n 's/.*RECOMMEND_FLUSHES=\([0-9][0-9]*\).*/\1/p' | tail -1)
if [ -n "$REC" ] && [ "$REC" != "$FLUSHES" ]; then
  log "按容量自检调整挤池轮数：$FLUSHES → $REC"
  FLUSHES=$REC
fi

log "跑挤池验收探针（$FLUSHES × ~116k token）"
CHURN_OUT=/tmp/kvoff-churn-out.txt
/home/ll/vllm-env/bin/python /home/ll/deploy/kvoff-churn.py \
  --flushes "$FLUSHES" --a-sentences 2000 --b-sentences 8000 2>&1 | tee "$CHURN_OUT" >> "$RES"

log "外部命中累计："
curl -s --max-time 8 http://127.0.0.1:18420/metrics \
  | grep -a 'external_prefix_cache_hits_total{' | head -1 >> "$RES"

VERDICT=$(grep -a '"step": "reload"' "$CHURN_OUT" | tail -1 | grep -c '"verdict": true')
HITS=$(grep -a '"step": "reload"' "$CHURN_OUT" | tail -1 | sed -n 's/.*"external_hits_delta": \([0-9.]*\).*/\1/p')
EXER=$(grep -a '"step": "reload"' "$CHURN_OUT" | tail -1 | sed -nE 's/.*"tier_exercised": (true|false).*/\1/p')
log "判定：verdict=$([ "$VERDICT" = "1" ] && echo PASS || echo FAIL) tier_exercised=${EXER:-?} external_hits_delta=${HITS:-?}"

log "本轮外部自动化主机登录次数（干扰取证，EXT_HOST 未设则跳过）：${EXT_HOST:+$(echo "$SUDO_PASS" | sudo -S -p '' journalctl _COMM=sshd --since '-20min' --no-pager 2>/dev/null | grep -ac "from $EXT_HOST")}${EXT_HOST:-（未设置 EXT_HOST=<来源IP>，计 0）}"

if [ "$SOAK" = "1" ] && [ "$VERDICT" = "1" ]; then
  # soak 模式：判定通过就**不恢复**，让二级缓存带着真实流量继续跑；看门狗恢复计时，
  # 且其自愈安全模式（2026-09-27）会在实例死亡时退回无二级缓存定版，不会陷入重启循环。
  log "PASS 且 SOAK_IF_PASS=1 → 保持二级缓存运行进入 soak（不恢复生产）"
  log "soak 基线：$(free -g | sed -n '2p')"
  systemctl --user start fnx-18420-watchdog.timer 2>/dev/null || true
  log "看门狗 timer 已恢复（自愈时会剥离二级缓存档位并记日志）"
  log "=== 验收脚本结束（soak 模式，实例保持运行）==="
  exit 0
fi

log "恢复生产（含看门狗）"
setsid nohup bash /home/ll/deploy/kvoff-restore-prod.sh >/dev/null 2>&1 < /dev/null &
log "=== 验收脚本结束（生产恢复已在后台执行）==="
