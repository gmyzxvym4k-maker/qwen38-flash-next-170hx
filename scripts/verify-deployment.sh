#!/bin/bash
# verify-deployment.sh —— 一键体检：主机前置 / 镜像补丁 / 模型产物 / 在线服务 / 稳定性。
#
# 每项都打印 **期望值** 与实测值，末尾给 PASS/FAIL/WARN 汇总；有任何 FAIL 退出码 1。
# 只读操作，不改动任何东西（推理冒烟测试除外，它会产生一次 completion）。
#
# 用法：
#   bash verify-deployment.sh                 # 全量
#   bash verify-deployment.sh --offline       # 跳过在线服务段（实例没起来时用）
#   PORT=18420 ROOT=/media/ll/data/vllm-image bash verify-deployment.sh
set -uo pipefail

PORT=${PORT:-18420}
ROOT=${ROOT:-/media/ll/data/vllm-image}
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
DATA=${DATA:-/media/ll/data}
MODEL=${MODEL:-$DATA/models/Qwen3.8-Flash-Next-W4A16-AutoRound}
PLE=${PLE:-$DATA/ple}
SERVED=${SERVED:-qwen3.8-flash-next}
VLLM_DIR=$ROOT/rootfs/usr/local/lib/python3.12/dist-packages/vllm
OFFLINE=0
[ "${1:-}" = "--offline" ] && OFFLINE=1

# sudo 凭据：优先 SUDO_PASS 环境变量，其次 NOPASSWD（docs/01 §9）
if [ -n "${SUDO_PASS:-}" ]; then
  SUDO() { printf '%s\n' "$SUDO_PASS" | sudo -S -p '' "$@"; }
else
  SUDO() { sudo -n "$@"; }
fi

# free 的输出标签随 locale 变（中文系统是「内存：/交换空间：」），按标签匹配会
# 静默取到空值 —— 一律按行号取：1=表头 2=Mem 3=Swap。
free_g(){ free -g | awk -v n="$1" 'NR==n{print $2+0}'; }

PASS=0; FAIL=0; WARN=0
ck(){ # ck <名称> <期望> <实测> [级别]
  local name=$1 exp=$2 got=$3 lvl=${4:-must}
  if [ "$exp" = "-" ] || [ "$exp" = "$got" ]; then
    printf '  \033[32m✓\033[0m %-42s %s\n' "$name" "$got"; PASS=$((PASS+1))
  elif [ "$lvl" = "warn" ]; then
    printf '  \033[33m!\033[0m %-42s 期望 %s / 实得 %s\n' "$name" "$exp" "$got"; WARN=$((WARN+1))
  else
    printf '  \033[31m✗\033[0m %-42s 期望 %s / 实得 %s\n' "$name" "$exp" "$got"; FAIL=$((FAIL+1))
  fi
}
sec(){ echo; echo "── $1 ──────────────────────────────────────────"; }

# ============ A. 主机前置 ============
sec "A. 主机前置"
ck "内核 vm.overcommit_memory" "1" "$(sysctl -n vm.overcommit_memory 2>/dev/null)"
NG=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
ck "GPU 数量（PP2 需 2 张）" "2" "$NG"
ck "单卡显存 MiB（解锁后 64GB）" "65536" "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')"
G1=$(nvidia-smi --query-gpu=pcie.link.gen.current --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
ck "PCIe Gen（判据只看这条，别信 dmesg）" "2" "${G1:-?}" warn
for i in $(seq 0 $((NG>0?NG-1:0))); do
  W=$(nvidia-smi --query-gpu=pcie.link.width.current --format=csv,noheader,nounits -i $i 2>/dev/null | tr -d ' ')
  ck "GPU$i PCIe 位宽" "16" "${W:-?}" warn
done
ck "数据盘 read_ahead_kb" "128" "$(cat /sys/block/$(basename "$(readlink -f /dev/disk/by-id/nvme-JZ-SSD2T-XW_* 2>/dev/null|head -1)")/queue/read_ahead_kb 2>/dev/null)" warn
ck "udev 规则 99-nvme-readahead" "存在" "$([ -f /etc/udev/rules.d/99-nvme-readahead.rules ] && echo 存在 || echo 缺失)"
ck "udev 规则 98-cmp-gen2-early" "存在" "$([ -f /etc/udev/rules.d/98-cmp-gen2-early.rules ] && echo 存在 || echo 缺失)"
ck "gen2 早钩子可执行" "存在" "$([ -x /usr/local/sbin/gen2-early-launch ] && echo 存在 || echo 缺失)"
ck "功耗服务已 enabled" "enabled" "$(systemctl is-enabled gpu-power-limit.service 2>/dev/null)" warn
ck "总内存 GiB（≥128 才够 BF16 表；本方案 INT8 可 64）" "-" "$(free_g 2)"
ck "swap GiB（建议关，与 PLE 同盘会抢 I/O）" "0" "$(free_g 3)" warn

# ============ B. 镜像与补丁 ============
sec "B. 镜像 rootfs 与 vLLM 补丁"
ck "rootfs 目录" "存在" "$([ -d $ROOT/rootfs ] && echo 存在 || echo 缺失)"
ck "镜像内 python3.12" "存在" "$([ -x $ROOT/rootfs/usr/bin/python3.12 ] && echo 存在 || echo 缺失)"
ck "vllm 包目录" "存在" "$([ -d "$VLLM_DIR" ] && echo 存在 || echo 缺失)"
if [ -d "$VLLM_DIR" ]; then
  OUT=$(python3 "$REPO/scripts/apply-patches.py" --target "$VLLM_DIR" --check 2>&1)
  NP=$(echo "$OUT" | grep -c "patched ")
  NU=$(echo "$OUT" | grep -cE "pristine |UNKNOWN |✗")
  ck "补丁数（25 个全部 patched）" "25" "$NP"
  [ "$NU" != "0" ] && printf '  \033[31m✗\033[0m 未达期望状态 %d 个：\n%s\n' "$NU" "$(echo "$OUT"|grep -E 'pristine |UNKNOWN |✗'|head -5)"
  FAIL=$((FAIL+NU))
  VER=$(SUDO chroot "$ROOT/rootfs" /usr/bin/python3.12 -c 'import vllm;print(vllm.__version__)' 2>/dev/null)
  case "${VER:-}" in *0.1.dev20073*) VOK=匹配;; *) VOK="不匹配（${VER:-取不到，需 root}）";; esac
  ck "镜像 vLLM 版本含 0.1.dev20073" "匹配" "$VOK" warn
fi

# ============ C. 模型与离线产物 ============
sec "C. 模型与离线产物"
ck "模型目录" "存在" "$([ -d "$MODEL" ] && echo 存在 || echo 缺失)"
NS=$(ls "$MODEL"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
ck "权重分片数（仓库无 00002，共 16 片 + model_extra）" "17" "$NS" warn
SZ=$(stat -c %s "$MODEL/model-00016-of-00017.safetensors" 2>/dev/null)
ck "PLE 表分片 00016 字节" "102400512256" "${SZ:-缺失}"
ck "config.json quant_method" "auto-round" "$(python3 -c "import json;print(json.load(open('$MODEL/config.json')).get('quantization_config',{}).get('quant_method','?'))" 2>/dev/null)"
ck "PLE INT8 产物 int8.bin" "51200245760" "$(stat -c %s $PLE/ple_ngram_int8.bin 2>/dev/null || echo 缺失)"
ck "PLE INT8 产物 scale.bin" "640003072" "$(stat -c %s $PLE/ple_ngram_scale.bin 2>/dev/null || echo 缺失)"
ck "PLE INT8 meta" "存在" "$([ -f $PLE/ple_ngram_meta.json ] && echo 存在 || echo 缺失)"

# ============ D. 在线服务 ============
if [ "$OFFLINE" = "0" ]; then
  sec "D. 在线服务（端口 $PORT）"
  ck "GET /health" "200" "$(curl -s -m 10 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)"
  MODELS=$(curl -s -m 10 http://127.0.0.1:$PORT/v1/models | python3 -c "import sys,json;d=json.load(sys.stdin);print(','.join(m['id'] for m in d['data']))" 2>/dev/null)
  ck "served 模型名" "$SERVED" "${MODELS:-无响应}"
  MLEN=$(curl -s -m 10 http://127.0.0.1:$PORT/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0].get('max_model_len','?'))" 2>/dev/null)
  ck "max_model_len" "-" "${MLEN:-?}"
  PID=$(ps -eo pid=,args= | grep "[e]ntrypoints[./]cli[./]main" | awk -v p="--port $PORT" 'index($0,p){print $1}' | head -1)
  CMD=$( [ -n "$PID" ] && (tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null || SUDO sh -c "tr '\\0' ' ' < /proc/$PID/cmdline" 2>/dev/null) )
  for f in "--block-size 1616" "--mamba-ssm-cache-dtype float32" "--pipeline-parallel-size 2" "--moe-backend auto"; do
    ck "cmdline 含 $f" "yes" "$(case "$CMD" in *"$f"*) echo yes;; *) echo no;; esac)"
  done
  ck "cmdline 含 num_speculative_tokens\":4" "yes" "$(case "$CMD" in *'\"num_speculative_tokens\":4'*) echo yes;; *"num_speculative_tokens\":4"*) echo yes;; *) echo no;; esac)"
  ck "cmdline 不含 --enforce-eager" "no" "$(case "$CMD" in *enforce-eager*) echo yes;; *) echo no;; esac)"
  HIT=$(curl -s -m 10 http://127.0.0.1:$PORT/metrics | awk '/^vllm:prefix_cache_queries_total/{q=$2} /^vllm:prefix_cache_hits_total/{h=$2} END{if(q>0)printf "%.1f%%", h/q*100; else print "n/a"}')
  ck "前缀缓存累计命中率（>50% 健康）" "-" "${HIT:-n/a}"
  ACC=$(curl -s -m 10 http://127.0.0.1:$PORT/metrics | awk '/^vllm:spec_decode_num_drafts_total/{d+=$2} /^vllm:spec_decode_num_accepted_tokens_total/{a+=$2} END{if(d>0)printf "接受长度 %.2f", 1+a/d; else print "无投机数据"}')
  ck "MTP 平均接受长度（预期 ~3~4）" "-" "${ACC:-n/a}"
  T0=$(date +%s)
  RESP=$(curl -s -m 180 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{
    "model":"'"$SERVED"'","messages":[{"role":"user","content":"只回答两个汉字：你好"}],
    "max_tokens":48,"temperature":0}')
  TXT=$(echo "$RESP" | python3 -c "import sys,json;d=json.load(sys.stdin);c=d['choices'][0]['message'];print((c.get('content') or '').strip()[:40] or '(空)')" 2>/dev/null)
  [ -z "$TXT" ] && TXT="解析失败: $(echo "$RESP"|head -c 120)"
  ck "冒烟推理（应返回简短中文）" "-" "[$(( $(date +%s)-T0 ))s] $TXT"
  CT=$(echo "$RESP" | python3 -c "import sys,json;print(json.load(sys.stdin)['usage'].get('prompt_tokens_details') is not None)" 2>/dev/null)
  ck "响应含 prompt_tokens_details（需 --enable-prompt-tokens-details）" "True" "${CT:-?}" warn
fi

# ============ E. 稳定性 ============
sec "E. 稳定性"
NX=$( (dmesg 2>/dev/null || SUDO dmesg 2>/dev/null) | grep -cE 'NVRM: Xid' )
ck "dmesg Xid 计数（本次开机，期望 0）" "0" "${NX:-?}" warn
NM=$( (dmesg 2>/dev/null || SUDO dmesg 2>/dev/null) | grep -c 'mce: \[Hardware Error\]' )
ck "dmesg MCE 计数（内存不稳会硬挂，期望 0）" "0" "${NM:-?}" warn
LOGF=${LOGF:-/home/ll/deploy/vllm-flash-next-w4a16.log}
DEAD=$(grep -acE "EngineDead|CUDA error|illegal memory access" "$LOGF" 2>/dev/null); DEAD=${DEAD:-0}
ck "当前日志崩溃签名数（期望 0）" "0" "$DEAD" warn

echo
echo "══════════ 汇总：PASS=$PASS  WARN=$WARN  FAIL=$FAIL ══════════"
[ "$FAIL" = "0" ] && echo "✓ 部署与文档一致，可用" || echo "✗ 有 $FAIL 项不达期望，见 docs/08-pitfalls.md 对照排查"
[ "$FAIL" = "0" ]
