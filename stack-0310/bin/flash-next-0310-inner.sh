#!/bin/bash
# flash-next-0310-inner.sh — 官方 vLLM 0.31.0 + 运行时补丁栈（site-packages 零改动）
#
# 本机画像（`<DEPLOY_HOST>`（DHCP 会变）/ HUANANZHI X99-T8 / E5-2696 v4 / 251GiB 内存 / 2×CMP 170HX 64GiB）
#   并行 = TP1 × PP2，层切分 26,22（48 层）；CUDA_VISIBLE_DEVICES 缺省 0,1
#   PLE  n-gram 表默认走【INT8 磁盘 mmap】（DSH_PLE_MMAP=1 + DSH_PLE_INT8_DIR=/media/ll/data/ple，
#        表 47.7+0.6 GiB 可回收页缓存）；FN_PLE_MMAP=0 则回官方 pinned-host BF16（95.4 GiB 锁页）。
#   KV   二级缓存=SimpleCPUOffloadConnector（--kv-offloading-size + VLLM_USE_SIMPLE_KV_OFFLOAD=1），
#        缺省 96 GiB；由 rt-patch #13 提供 PP 下 CPU 块 id 空间握手 clamp 与逐块拷贝守卫。
#
# 本文件缺省值 = 2026-10-06 生产逐键（1M YaRN×4 / block1616 / MTP4 / seqs2 / gpu0.95 / async /
#   思考 medium / 二级缓存关→由 launch.env 显式开），与 8889 控制台 SCRIPT_MODELS.base 对齐；
#   采样缺省=10-10 用户定档官方默认 1/0.95/20/0/0/1（控制台 base 相同时不下发 FN_GENCFG）。
#
# 用法：
#   FN_DRY_RUN=1 bash flash-next-0310-inner.sh     # 只打印 argv（离线可跑，不碰 GPU）
#   FN_SIMPLE_OFFLOAD=0 bash ...                    # 关二级缓存
#   FN_PLE_MMAP=0 bash ...                          # PLE 回官方锁页 BF16
set -u

BASE_DIR=${BASE_DIR:-/home/ll/deploy/vllm-0310}
RT_DIR=${RT_DIR:-$BASE_DIR/patches}
RT_EXTRA_DIR=${RT_EXTRA_DIR:-$BASE_DIR/patches-extra}
VENV=${FN_VENV:-/media/ll/data/vllm-0310-env}

# ============ 0. 参数通道：launch.env ============
# 宿主 wrapper 是 `sudo … setsid bash inner` 起的，sudo 的 env_reset 会清掉控制台传来的
# FN_*，所以参数唯一可靠通道是 wrapper 落盘的 launch.env（与 0.30 栈同机制，
# 09-26「弹窗采样参数静默失效」事故的根治就是这套）。不要改用 sudo -E。
FN_ENVFILE=${FN_ENVFILE:-$BASE_DIR/launch.env}
if [ -f "$FN_ENVFILE" ]; then
  # set -a：文件里的 FN_* 变成导出变量，后面「未消费参数」体检才看得见它们
  set -a
  # shellcheck disable=SC1090
  . "$FN_ENVFILE"
  set +a
  echo "[FN-0310] 已加载参数文件 $FN_ENVFILE（$(grep -c '^FN_' "$FN_ENVFILE" 2>/dev/null) 项）" >&2
else
  echo "[FN-0310] 无 $FN_ENVFILE，使用本脚本内置缺省" >&2
fi

PORT=${FN_PORT:-18420}
SERVED=${FN_SERVED:-qwen3.8-flash-next}
MODEL=${FN_MODEL_PATH:-/media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound}
PP=${FN_PP:-2}
TP=${FN_TP:-1}
MAXLEN=${FN_MAXLEN:-${FN_CTX:-262144}}
GPUMEM=${FN_GPUMEM:-0.95}
SEQS=${FN_SEQS:-2}
BLOCK=${FN_BLOCK:-1616}
MBT=${FN_MBTOKENS:-8192}
MOE=${FN_MOE:-auto}
SSMDTYPE=${FN_SSMDTYPE:-float32}
GENCFG=${FN_GENCFG:-'{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}'}
CHATKW=${FN_CHATKWARGS:-'{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}'}
CUDAGRAPH=${FN_CUDAGRAPH:-FULL_AND_PIECEWISE}
CAPTURE_SIZES=${FN_CAPTURE_SIZES:-[1,2,4,8,16,24,32,40]}
SPEC=${FN_SPEC:-'{"method":"mtp","num_speculative_tokens":4,"use_local_argmax_reduction":false}'}
# PLE 形态：本栈两档 = ①INT8 产物 mmap（可回收页缓存，省 44 GiB）②官方 pinned-host BF16（95.4 GiB 锁页）。
# 8889 弹窗的两个下拉在语义上正好映射到这两档，故此处把它们接上（不再是死参数）：
#   精度 INT8(pleInt8=1) + 位置 disk  ⇒ mmap=1
#   精度 BF16(pleInt8=0) 或 位置 heap ⇒ mmap=0（官方锁页 BF16）
# 显式 FN_PLE_MMAP 优先。
if [ -n "${FN_PLE_MMAP:-}" ]; then
  PLE_MMAP=$FN_PLE_MMAP
  PLE_ANON=0
elif [ "${FN_PLE_INT8:-1}" = "0" ]; then
  PLE_MMAP=0
  PLE_ANON=0
elif [ "${FN_PLE_LOC:-disk}" = "heap" ]; then
  # [anon-ple 1010] INT8+放内存 = 匿名堆驻留：不可回收且**计入「已用」内存**。
  # （mlock 文件页只保证驻留，内核记账仍留在缓冲/缓存；要"占用已用"必须匿名页。）
  PLE_MMAP=1
  PLE_ANON=1
else
  PLE_MMAP=1
  PLE_ANON=0
fi
PLE_INT8_DIR=${FN_PLE_INT8_DIR:-${DSH_PLE_INT8_DIR:-/media/ll/data/ple}}
CUDA_DEVS=${FN_CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((PP*TP-1)))}

die() { echo "[FATAL][FN-0310] $*" >&2; exit 1; }
warn() { echo "[WARN][FN-0310] $*" >&2; }

# ============ 1. 投机档位与 block-size 的 QSA ring 整除自检 ============
# ring = compress_ratio(4) × cdiv(4+K, 4)，block-size 必须被 ring 整除（09-19 定版；
# 0.31 上游把 ring 放宽过，rt-patch hook8 在「legacy 整除 block」时收缩回 legacy）。
if [ -n "$SPEC" ] && [ "$SPEC" != "none" ]; then
  K=$("$VENV/bin/python" -c "
import json,sys
try:
    print(int(json.loads('''$SPEC''').get('num_speculative_tokens',0)))
except Exception:
    print(0)
" 2>/dev/null || echo 0)
  if [ "${K:-0}" -ge 1 ]; then
    RING=$(( 4 * ( (4 + K + 3) / 4 ) ))
    if [ $(( BLOCK % RING )) -ne 0 ]; then
      die "MTP K=$K ⇒ QSA ring=$RING 不整除 block-size=$BLOCK（加载后才 AssertionError）。block 1616 合法 K=1..4/9..12；K=5 请 FN_BLOCK=1680"
    fi
    echo "[FN-0310] MTP K=$K ring=$RING 整除 block=$BLOCK ✓" >&2
  fi
fi

# ============ 2. 长上下文档位（YaRN 配置副本） ============
if [ "${FN_LONGCTX:-0}" = "1" ]; then
  LC=${FN_1M_MODEL_PATH:-}
  [ -n "$LC" ] && [ -f "$LC/config.json" ] || die "要求长上下文档但 YaRN 副本不可用：'${LC:-<未下发 FN_1M_MODEL_PATH>}'（拒绝用未缩放 RoPE 跑 $MAXLEN）"
  MODEL="$LC"
  CAP=$("$VENV/bin/python" -c "
import json
c=json.load(open('$MODEL/config.json')); t=c.get('text_config',c)
print(t.get('max_position_embeddings',0))
" 2>/dev/null || echo 0)
  if [ "${CAP:-0}" -gt 0 ] && [ "$MAXLEN" -gt "$CAP" ] 2>/dev/null; then
    warn "max-model-len $MAXLEN 钳到副本上限 $CAP"; MAXLEN="$CAP"
  fi
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  echo "[FN-LONGCTX] 模型切到 YaRN 副本 $MODEL（factor=${FN_YARN_FACTOR:-?}，上限=$CAP，本次 max-model-len=$MAXLEN）" >&2
fi
[ -f "$MODEL/config.json" ] || die "模型目录不可用：$MODEL（config.json 缺失＝软链断/下载未完）"

# ============ 3. PLE 表形态 ============
if [ "$PLE_MMAP" = "1" ]; then
  # 完整性判据：ple_ngram_meta.json 由 quantize_ple.py 在全部 611 块写完后才生成
  # （两个 .bin 是预分配的，跑一半就存在，只查体积会误判"产物就绪"）。
  if [ -f "$PLE_INT8_DIR/ple_ngram_int8.bin" ] && [ -f "$PLE_INT8_DIR/ple_ngram_scale.bin" ] \
     && [ -f "$PLE_INT8_DIR/ple_ngram_meta.json" ]; then
    export DSH_PLE_MMAP=1
    export DSH_PLE_INT8_DIR="$PLE_INT8_DIR"
    if [ "$PLE_ANON" = "1" ]; then
      # [anon-ple 1010] 匿名堆驻留（用户要求：在内存中不可回收、计入已用）。
      # 代价：启动多 ~20-40s 顺序读 48GB；物理占用与原 mlock 档相同，只是记账桶从
      # 缓冲/缓存变为已用（诚实反映"表在内存里"）。
      export DSH_PLE_MEM_RESIDENT=1
      echo "[FN-PLE] INT8 内存驻留（匿名堆）：$PLE_INT8_DIR（47.7+0.6 GiB，不可回收、计入已用）" >&2
    else
      # FN_PLE_LOCK=1（缺省）⇒ 对表做 mlock：保证常驻不被回收（注意：内核记账仍在
      # 缓冲/缓存，不进「已用」；要计入已用请选「PLE 表位置=放内存」匿名堆档）。
      # 想回到纯可回收页缓存：FN_PLE_LOCK=0。
      export DSH_PLE_MMAP_LOCK=${FN_PLE_LOCK:-1}
      echo "[FN-PLE] INT8 磁盘 mmap：$PLE_INT8_DIR（47.7+0.6 GiB，mlock=${DSH_PLE_MMAP_LOCK}）" >&2
    fi
  else
    die "FN_PLE_MMAP=1 但产物不全：$PLE_INT8_DIR/ple_ngram_int8.bin|ple_ngram_scale.bin。先生成（scripts/quantize_ple.py）或改 FN_PLE_MMAP=0 走官方锁页 BF16（95.4 GiB）"
  fi
else
  export DSH_PLE_MMAP=0
  echo "[FN-PLE] 官方 pinned-host BF16（95.4 GiB 锁页，需 root + memlock 不限）" >&2
fi

# ============ 4. 二级缓存（CPU KV 分层） ============
if [ "${FN_KVOFF:-0}" = "1" ]; then
  die "本栈不支持经典 OffloadingConnector（FN_KVOFF=1）——该路径对本 hybrid 模型已定案退役（回载不等价）。二级缓存请用 FN_SIMPLE_OFFLOAD=<GiB>"
fi
OFFLOAD_ARGS=()
if [ -n "${FN_SIMPLE_OFFLOAD:-}" ] && [ "${FN_SIMPLE_OFFLOAD:-0}" != "0" ]; then
  GiB=${FN_SIMPLE_OFFLOAD}
  [ "$GiB" -ge 8 ] 2>/dev/null || die "FN_SIMPLE_OFFLOAD=$GiB 非法（≥8 GiB）"
  PIN=$(( GiB + 8 ))
  if [ "$(id -u)" != "0" ]; then
    die "开二级缓存必须 root（cudaHostRegister 需 memlock 不限；ll 被 user@1000.service LimitMEMLOCK=65536 钉死）"
  fi
  MEM_GIB=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)
  if [ "$PLE_MMAP" = "1" ]; then NEED=$(( PIN + 60 )); else NEED=$(( PIN + 110 )); fi
  [ "$NEED" -le "$MEM_GIB" ] || warn "内存预算紧：pinned ${PIN}GiB + PLE/引擎预留 ≈${NEED}GiB > 物理 ${MEM_GIB}GiB（第一刀回退 FN_SIMPLE_OFFLOAD=$(( GiB / 2 ))）"
  export VLLM_USE_SIMPLE_KV_OFFLOAD=1
  OFFLOAD_ARGS=(--kv-offloading-backend native --kv-offloading-size "$GiB")
  echo "[FN-KVOFF] SimpleCPUOffloadConnector：CPU 档 ${GiB} GiB（world_size=$((PP*TP)) 均分；rt-patch #13 提供 PP 握手 clamp+越界守卫）" >&2
else
  echo "[FN-KVOFF] 二级缓存关（无 --kv-offloading-size）" >&2
fi

# ============ 5. 运行时环境 ============
export PYTHONPATH="$RT_DIR:$RT_EXTRA_DIR${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_RT_PATCHES=1
export CUDA_VISIBLE_DEVICES="$CUDA_DEVS"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=${FN_V2RUNNER:-1}
export VLLM_PP_LAYER_PARTITION=${FN_PP_PARTITION:-26,22}
export VLLM_LOGGING_LEVEL=${FN_LOGLEVEL:-INFO}
export VLLM_CACHE_ROOT=${FN_CACHE_ROOT:-/root/.cache/vllm-18420-0310}
export VLLM_SKIP_MM_WARMUP=1
export VLLM_USE_FLASHINFER_SAMPLER=${FN_FLASHINFER_SAMPLER:-0}
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 本机 BAR1 P2P 数据通路被强制补丁刷成"假 OK"且静默损坏（10-01 实锤）→ NCCL 走 host SHM
export NCCL_P2P_DISABLE=${FN_NCCL_P2P_DISABLE:-1}
export NCCL_SHM_DISABLE=0
export NCCL_CUMEM_ENABLE=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
mkdir -p "$VLLM_CACHE_ROOT" 2>/dev/null || true

if [ "${FN_EAGER:-0}" = "1" ] || [ "${FN_ENFORCE_EAGER:-0}" = "1" ]; then
  CUDAGRAPH=NONE
fi

# ============ 6. argv ============
ARGS=(
  "$MODEL"
  --served-model-name "$SERVED"
  --host 0.0.0.0 --port "$PORT"
  --load-format safetensors
  --safetensors-load-strategy lazy
  --distributed-executor-backend mp
  --tensor-parallel-size "$TP"
  --pipeline-parallel-size "$PP"
  --disable-custom-all-reduce
  --dtype bfloat16
  --max-model-len "$MAXLEN"
  --block-size "$BLOCK"
  --mamba-ssm-cache-dtype "$SSMDTYPE"
  --max-num-seqs "$SEQS"
  --gpu-memory-utilization "$GPUMEM"
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --max-num-batched-tokens "$MBT"
  --moe-backend "$MOE"
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --trust-remote-code
  --default-chat-template-kwargs "$CHATKW"
  --override-generation-config "$GENCFG"
  --no-enable-flashinfer-autotune
  --enable-chunked-prefill
  "-cc.cudagraph_mode=$CUDAGRAPH"
  "-cc.cudagraph_capture_sizes=$CAPTURE_SIZES"
  '-cc.inductor_compile_config={"combo_kernels":false,"benchmark_combo_kernel":false}'
)
if [ "${FN_ASYNC:-1}" = "1" ]; then ARGS+=(--async-scheduling); else ARGS+=(--no-async-scheduling); fi
[ "${FN_PREFIX_CACHE:-1}" = "0" ] && ARGS+=(--no-enable-prefix-caching)
[ "${FN_CHUNKED:-1}" = "0" ] && ARGS+=(--no-enable-chunked-prefill)
[ -n "${FN_SEED:-}" ] && ARGS+=(--seed "$FN_SEED")
[ "${FN_NOLOG:-0}" = "1" ] && ARGS+=(--disable-log-requests)
if [ -n "$SPEC" ] && [ "$SPEC" != "none" ]; then ARGS+=(--speculative-config "$SPEC"); fi
if [ "${#OFFLOAD_ARGS[@]}" -gt 0 ]; then ARGS+=("${OFFLOAD_ARGS[@]}"); fi
[ "${FN_EP:-0}" = "1" ] && ARGS+=(--enable-expert-parallel)

# ============ 7. 参数体检：控制台下发了但本脚本不消费的键 ============
CONSUMED=" FN_VENV FN_BASE FN_ENVFILE FN_MODEL_PATH FN_1M_MODEL_PATH FN_LONGCTX FN_YARN_FACTOR \
FN_CTX FN_MAXLEN FN_PORT FN_SERVED FN_TP FN_PP FN_PP_PARTITION FN_CUDA_VISIBLE_DEVICES \
FN_SEQS FN_GPUMEM FN_BLOCK FN_SSMDTYPE FN_MBTOKENS FN_MOE FN_EP FN_ASYNC FN_PREFIX_CACHE FN_CHUNKED \
FN_EAGER FN_ENFORCE_EAGER FN_CUDAGRAPH FN_CAPTURE_SIZES FN_SPEC FN_SEED FN_NOLOG FN_NOLOG \
FN_GENCFG FN_CHATKWARGS FN_KVOFF FN_KVOFF_BYTES FN_KVOFF_WAIT_TIMEOUT FN_KVOFF_SHARED \
FN_SIMPLE_OFFLOAD FN_PLE_MMAP FN_PLE_INT8_DIR FN_V2RUNNER FN_LOGLEVEL FN_CACHE_ROOT FN_LOG \
FN_DRY_RUN FN_FORCE FN_EXTRA_ENV FN_EXTRA_ARGS FN_LOG_LEVEL FN_NOLOG \
FN_PLE_INT8 FN_PLE_LOC FN_LIMIT_MM FN_MAX_SCHED_TOKENS FN_CPU_OFFLOAD_GB FN_SCHED_POLICY \
FN_DISABLE_ALLREDUCE FN_MTP_MOE_BACKEND FN_PP1_FULL_DECODE FN_DIAG_SAMPLE FN_DTYPE "
for v in $(compgen -e | grep -E '^FN_[A-Z0-9_]+$' | sort); do
  case " $CONSUMED " in *" $v "*) continue ;; esac
  case "$v" in
    FN_PLE_INT8|FN_PLE_LOC)
      # 官方 0.31.0 无 INT8-内存/匿名堆形态：精度只由「INT8 产物在不在」决定，位置由 FN_PLE_MMAP 决定
      warn "忽略 $v=${!v}（0.31.0 栈 PLE 只有 mmap-INT8 / pinned-BF16 两档，用 FN_PLE_MMAP 与 FN_PLE_INT8_DIR 表达）" ;;
    *) warn "未知参数 $v=${!v:0:60}（本 inner 未消费）" ;;
  esac
done

# 附加 env / 附加 argv（控制台透传位）
if [ -n "${FN_EXTRA_ENV:-}" ]; then
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in \#*) continue ;; esac
    export "$line" 2>/dev/null || warn "FN_EXTRA_ENV 行无法 export：$line"
  done <<< "$FN_EXTRA_ENV"
fi

CMD="\"$VENV/bin/python\" -m vllm.entrypoints.cli.main serve \"${ARGS[@]}\" ${FN_EXTRA_ARGS:-}"
if [ "${FN_DRY_RUN:-0}" = "1" ]; then
  echo "[dry-run] $CMD"
  echo "[dry-run] PLE_MMAP=$DSH_PLE_MMAP DIR=${DSH_PLE_INT8_DIR:-(未设)} | KVOFF=${FN_SIMPLE_OFFLOAD:-0}GiB | PP=$PP TP=$TP PARTITION=$VLLM_PP_LAYER_PARTITION DEVS=$CUDA_DEVS"
  echo "[dry-run] CUDAGRAPH=$CUDAGRAPH MAXLEN=$MAXLEN BLOCK=$BLOCK SEQS=$SEQS GPUMEM=$GPUMEM ASYNC=${FN_ASYNC:-1} CACHE_ROOT=$VLLM_CACHE_ROOT"
  exit 0
fi

if [ "${FN_PRECHECK_ONLY:-0}" = "1" ]; then echo "[precheck] 通过：argv 可生成、档位与产物齐备"; exit 0; fi

echo "[FN-0310] $(date -Is) 启动 vLLM 0.31.0 栈 port=$PORT model=$MODEL pp=$PP offload=${FN_SIMPLE_OFFLOAD:-0}GiB" >&2
cd /root 2>/dev/null || true
# shellcheck disable=SC2086
exec "$VENV/bin/python" -m vllm.entrypoints.cli.main serve "${ARGS[@]}" ${FN_EXTRA_ARGS:-}
