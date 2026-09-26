#!/bin/bash
# 18420 Flash-Next W4A16 —— 官方 vLLM 0.30.0 + PYTHONPATH 运行时补丁（18432 路线）
#
# 与旧栈（/home/ll/deploy/flash-next-w4a16-inner.sh，chroot 定制镜像 v0.1.dev20073）
# 的关系：同一个模型、同一个端口、同一套 FN_* 参数契约；但
#   · 不 chroot：直接用宿主 /home/ll/vllm-env（vLLM 0.30.0 + torch 2.13.0+cu130 + py3.11）
#   · 不改 site-packages：补丁经 PYTHONPATH 的 sitecustomize.py 运行时注入，
#     删掉 PYTHONPATH 重启 = 回到纯原厂 0.30.0（一键回滚）
#   · PLE n-gram 表：官方只有 BF16 锁页（pinned，95.4 GiB）一条路，旧栈的
#     INT8 匿名堆 / mmap 磁盘驻留两套自研加载器**不随迁移保留**（详见 README-0300.md §3）
#
# 权限：必须 root。锁页 95.4 GiB 要 cuMemHostRegister，ll 用户 memlock 硬上限 64 MiB
#   且 PAM limits.d 在 systemd user@1000 下不生效；root 的 memlock=unlimited。
#   启动方式：宿主 wrapper 用 sudo 起本脚本（不要直接 sudo -E，参数走 launch.env）。
#
# 用法：
#   sudo bash flash-next-0300-inner.sh              # 内置缺省=定稿参数
#   FN_SPEC=none sudo -E bash ...                   # 关投机
#   FN_DRY_RUN=1 sudo -E bash ...                   # 只打印 argv（不占 GPU）
set -u

VENV=${FN_VENV:-/home/ll/vllm-env}
BASE=${FN_BASE:-/home/ll/deploy/vllm-0300}
RT_DIR=$BASE/patches
RT_EXTRA=$BASE/patches-extra

FN_ENVFILE=${FN_ENVFILE:-$BASE/launch.env}
if [ -f "$FN_ENVFILE" ]; then
  # set -a：让文件里的 FN_* 变成导出变量，后面的「未消费参数」体检才能看见它们
  # （控制台 spawn 的那批本来就是导出的，wrapper 落盘的这批默认只是 shell 变量）。
  set -a; # shellcheck disable=SC1090
  . "$FN_ENVFILE"; set +a
fi

# ================================================================ 模型档位
# 三档副本（只改 config.json、其余软链回原目录，原 checkpoint 零改动）：
#   原生     /media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound        cap 262144 rope=default
#   512K     /media/ll/data/models-1m/...-AutoRound-512K                     cap 524288 rope=yarn×2
#   1M       /media/ll/data/models-1m/...-AutoRound-1M                       cap 1048576 rope=yarn×4
# 【必须自动选档】8889 控制台选长档时下发 FN_MODEL_PATH=原生目录 + FN_1M_MODEL_PATH=副本
#   目录 + FN_MAXLEN=1048576。若只认 FN_MODEL_PATH，就会拿未缩放的 262144 模型跑 1M
#   —— VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 会放行启动，但 RoPE 位置越过 original_max 后
#   NaN/越界（09-16 18:28 那类事故的根因形态）。故这里按「请求长度 ↔ 副本 cap」自配，
#   并在任何情况下都不允许 cap < 请求长度 的组合进 argv。
ORIG_MODEL_DIR=${FN_MODEL_PATH:-/media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound}
REQ_MAXLEN=${FN_MAXLEN:-1048576}
TIER_HINT=${FN_1M_MODEL_PATH:-}          # 控制台字段名沿用 1M，实际可能是 512K 档
BN=$(basename "$ORIG_MODEL_DIR")

cap_of() {  # 读某目录的位置上限（text_config 里，兼容顶层）
  python3 -c "
import json,sys
try:
    c=json.load(open(sys.argv[1]+'/config.json'))
    t=c.get('text_config',c) if isinstance(c,dict) else {}
    print(int(t.get('max_position_embeddings') or 0))
except Exception:
    print(0)" "$1" 2>/dev/null || echo 0
}

CANDIDATES=""
[ -n "$TIER_HINT" ] && CANDIDATES="$TIER_HINT"
CANDIDATES="$CANDIDATES /media/ll/data/models-1m/${BN}-1M /media/ll/data/models-1m/${BN}-512K $ORIG_MODEL_DIR"

# 选取规则：① 控制台显式指定的档位（FN_1M_MODEL_PATH）cap 够用就直接采纳；
# ② 否则在全部候选里取「cap 最小但 >= 请求长度」的那一档 —— 首命中是错的：
#    要原生 262144 却会撞上 cap=1048576 的 YaRN×4 副本（RoPE 被缩放 4 倍，短上下文
#    与原生不等价）。③ 全都不够 → 回落原生并把长度钳回原生 cap。
MODEL_PATH=""; CAP=0
for d in $CANDIDATES; do
  [ -d "$d" ] || continue
  c=$(cap_of "$d")
  if [ "$d" = "$TIER_HINT" ] && [ "${c:-0}" -ge "$REQ_MAXLEN" ]; then
    MODEL_PATH="$d"; CAP="$c"; break
  fi
done
if [ -z "$MODEL_PATH" ]; then
  for d in $CANDIDATES; do
    [ -d "$d" ] || continue
    c=$(cap_of "$d"); [ "${c:-0}" -ge "$REQ_MAXLEN" ] || continue
    if [ -z "$MODEL_PATH" ] || [ "$c" -lt "$CAP" ]; then MODEL_PATH="$d"; CAP="$c"; fi
  done
fi
if [ -z "$MODEL_PATH" ]; then
  MODEL_PATH="$ORIG_MODEL_DIR"; CAP=$(cap_of "$ORIG_MODEL_DIR")
  echo "[FN-0300] 警告：找不到 cap>=${REQ_MAXLEN} 的 YaRN 副本，max-model-len 钳回原生上限 ${CAP}（否则位置越界）" >&2
  REQ_MAXLEN="$CAP"
fi
FN_MAXLEN_EFF="$REQ_MAXLEN"

# ---------------------------------------------------------------- 预检
if [ ! -x "$VENV/bin/vllm" ]; then
  echo "[FN-0300] 找不到 $VENV/bin/vllm，放弃" >&2; exit 1
fi
if [ ! -f "$RT_DIR/sitecustomize.py" ]; then
  echo "[FN-0300] 找不到运行时补丁 $RT_DIR/sitecustomize.py，放弃（裸原厂 0.30.0 在 PP2+PLE 下必被硬拒）" >&2
  exit 1
fi
EXPECT_RT_SHA=9b84700fee6dbe3da9da9b6f2f071bcb6049926a9acc14199927d15185de29b7
RT_SHA=$(sha256sum "$RT_DIR/sitecustomize.py" | awk '{print $1}')
if [ "$RT_SHA" != "$EXPECT_RT_SHA" ]; then
  echo "[FN-0300] 警告：上游补丁 sha256 与登记值不符（实=$RT_SHA）——被改过或换了版本，请人工确认" >&2
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "[FN-0300] 模型目录不存在：$MODEL_PATH" >&2; exit 1
fi
MEMLOCK=$(ulimit -l)
[ "$MEMLOCK" != "unlimited" ] && \
  echo "[FN-0300] 警告：memlock=$MEMLOCK（KB，非 unlimited），95.4 GiB PLE 锁页会失败 ⇒ 必须 root 运行" >&2

# ---------------------------------------------------------------- 运行时补丁注入
# 顺序：上游目录在前，本地扩展在后 —— 上游 _chain_stock_sitecustomize() 会按 sys.path
# 跳过自己目录、加载第一个外部 sitecustomize.py，即 patches-extra/sitecustomize.py。
export PYTHONPATH="$RT_DIR:$RT_EXTRA${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_RT_PATCHES=1

# ---------------------------------------------------------------- PLE（Engram）
# 官方 0.30.0 的选择逻辑（models/qwen4_exp/nvidia/ngram_embedding.py:702-706）：
#   engram_config 缺省 None ⇒ 模型有 n-gram 层（ple_layer_ids=[2]）则自动造 EngramConfig()，
#   其 cpu_offload 取 envs.VLLM_PLE_CPU_OFFLOAD（缺省 1）⇒ Qwen4ExpPLEPinnedHostEmbedding
#   （BF16 锁页）。这里显式 export，防止将来 env 缺省值翻转把表静默搬进显存（必 OOM）。
export VLLM_PLE_CPU_OFFLOAD=1
echo "[FN-0300] PLE 表：官方 BF16 锁页（pinned CPU 95.4 GiB，PP rank0 独占），由 rt-patch 分块 cuMemHostRegister(<=60 GiB)" >&2

# ---------------------------------------------------------------- 并行/显存
export VLLM_PP_LAYER_PARTITION="${FN_PP_PARTITION:-26,22}"
export CUDA_VISIBLE_DEVICES="${FN_CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_LOGGING_LEVEL="${FN_LOGLEVEL:-INFO}"
# 编译/图缓存按栈隔离，别和旧 chroot 栈、也别和宿主其它实例共用 torch_compile_cache
# （09-06 实锤：共享 AOT 缓存 + max_num_seqs 不同 → DFlash2 stride 断言；同型风险通用化）
export VLLM_CACHE_ROOT="${FN_CACHE_ROOT:-/root/.cache/vllm-18420-0300}"
# flashinfer 采样器需要现场 JIT，本机无 ninja/nvcc（关掉走 torch 采样，统计等价）
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
# 多模态 warmup 跳过（rt-patch #8，本地扩展补丁）
export VLLM_SKIP_MM_WARMUP=1
# NCCL：与旧栈实跑逐项一致（P2P 已打通，走 PHB 级放行）
export NCCL_CUMEM_ENABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_NET_GDR_LEVEL=0
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-PHB}"
export NCCL_SHM_DISABLE=0

# 旧栈专有、官方 0.30.0 无读取方的 env 不再下发（VLLM_PLE_MMAP*、VLLM_PLE_GDS*、
# VLLM_PLE_NVFP4_GPU、VLLM_PLE_DISK_RESIDENT、VLLM_PLE_INT8*）。

# ---------------------------------------------------------------- argv
# 采样参数：09-21 定版三源一致值（治循环复读）。旧栈实跑 argv 是漂移态
# （temperature 1 / presence 0 / repetition 1），此处按定版值，见 README-0300.md §4。
GENCFG_DEFAULT='{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.1,"repetition_penalty":1.05}'
CHATKW_DEFAULT='{"enable_thinking":true,"preserve_thinking":true}'

BLOCK=${FN_BLOCK:-1616}
ARGS=(
  serve "$MODEL_PATH"
  --served-model-name "${FN_SERVED:-qwen3.8-flash-next}"
  --host 0.0.0.0
  --port "${FN_PORT:-18420}"
  --load-format safetensors
  --safetensors-load-strategy lazy
  --distributed-executor-backend mp
  --tensor-parallel-size "${FN_TP:-1}"
  --pipeline-parallel-size "${FN_PP:-2}"
  --dtype "${FN_DTYPE:-bfloat16}"
  --max-model-len "$FN_MAXLEN_EFF"
  # 两级 CSA + linear 层 block 不一致 → 1616 必传（gavinxym 手册）
  --block-size "$BLOCK"
  --mamba-ssm-cache-dtype "${FN_SSMDTYPE:-float32}"
  --max-num-seqs "${FN_SEQS:-4}"
  --gpu-memory-utilization "${FN_GPUMEM:-0.95}"
  --enable-prompt-tokens-details
  --max-num-batched-tokens "${FN_MBTOKENS:-8192}"
  # 必须 auto：草稿 MoE 层未量化，显式 marlin 会 ValueError
  --moe-backend "${FN_MOE:-auto}"
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --trust-remote-code
  --default-chat-template-kwargs "${FN_CHATKWARGS:-$CHATKW_DEFAULT}"
  --override-generation-config "${FN_GENCFG:-$GENCFG_DEFAULT}"
  '-cc.inductor_compile_config={"combo_kernels":false,"benchmark_combo_kernel":false}'
  --no-enable-flashinfer-autotune
  '-cc.cudagraph_mode=FULL_AND_PIECEWISE'
  '-cc.cudagraph_capture_sizes=[1,2,4,8,16,24,32,40]'
)
# 控制台可关的三项（0926 补：以前硬编码，弹窗勾选会静默失效）
if [ "${FN_PREFIX_CACHE:-1}" = "0" ]; then ARGS+=(--no-enable-prefix-caching); else ARGS+=(--enable-prefix-caching); fi
if [ "${FN_CHUNKED:-1}" = "0" ]; then ARGS+=(--no-enable-chunked-prefill); else ARGS+=(--enable-chunked-prefill); fi
if [ "${FN_ASYNC:-1}" = "0" ]; then ARGS+=(--no-async-scheduling); else ARGS+=(--async-scheduling); fi
# 【与旧栈的 argv 偏离】
#  · 不下发 -cc.splitting_ops：旧栈那 18 项是为了把自研 ple_mmap_lookup/ple_gds_lookup 请出
#    计算图；官方 0.30.0 的 CompilationConfig._attention_ops（config/compilation.py:772）已
#    含 vllm::qwen4_exp_ple_short_conv / vllm::qwen4_exp_qsa_with_output，显式覆盖反而会把
#    官方新增的分裂算子挤掉 → 省略即可，语义等价。
#  · --kv-transfer-config 仅在 FN_KVOFF=1 时下发：CPU KV 二级缓存 0929 曾定案退役，
#    1005 起经 rt-patch #9（patches-extra/dsh_kvoff_rt.py）移植回 0.30.0，缺省仍关。

# ------------------------------------------------------- CPU KV 二级缓存
# 1005 移植定版（rt-patch #9，语义=旧栈 c1/c2/c6/c7）：缺省关。
# 旧栈结论全部继承：
#  · 物理钉住 ≈1.56x 配置值（PP2 每 rank 私有 pinned，容量铁律 09-22）；
#  · 容量必须 > GPU KV 池（≈122 万 tok）才有回载收益，store 侧 ≈40.4KB/token；
#  · 对本机流量形态收益存疑（0929 退役依据：21h 生产 external hits=0，GPU 池自扛 ~90%）；
#  · store 熔断只读降级 + 有界等待已内置（FN_KVOFF_WAIT_TIMEOUT 缺省 15s）。
if [ "${FN_KVOFF:-0}" = "1" ]; then
  KVOFF_BYTES="${FN_KVOFF_BYTES:-68719476736}"
  ARGS+=(--kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${KVOFF_BYTES}}}")
  export FN_KVOFF_WAIT_TIMEOUT="${FN_KVOFF_WAIT_TIMEOUT:-15}"
  echo "[FN-0300] CPU KV 二级缓存：开 cpu_bytes_to_use=${KVOFF_BYTES} ($(( KVOFF_BYTES / 1073741824 )) GiB) wait_timeout=${FN_KVOFF_WAIT_TIMEOUT}s（rt-patch#9 c1/c2/c6/c7 已挂）" >&2
fi

if [ "${FN_EP:-0}" = "1" ]; then ARGS+=(--enable-expert-parallel); fi
if [ "${FN_EAGER:-0}" = "1" ] || [ "${FN_ENFORCE_EAGER:-0}" = "1" ]; then
  ARGS+=(--enforce-eager)
fi
if [ -n "${FN_SEED:-}" ]; then ARGS+=(--seed "$FN_SEED"); fi

SPEC="${FN_SPEC-mtp4}"
if [ "$SPEC" = "none" ] || [ "$SPEC" = "0" ] || [ -z "$SPEC" ]; then
  echo "[FN-0300] 投机：关闭" >&2
else
  case "$SPEC" in
    mtp*) N="${SPEC#mtp}"; SPEC_JSON='{"method":"mtp","num_speculative_tokens":'"$N"',"use_local_argmax_reduction":false}' ;;
    '{"method"'*) SPEC_JSON="$SPEC"; N=$(python3 -c "import json,sys;print(json.loads(sys.argv[1]).get('num_speculative_tokens','?'))" "$SPEC_JSON" 2>/dev/null || echo '?') ;;
    '{'*) SPEC_JSON="$SPEC"; N='?' ;;
    *) echo "[FN-0300] 无法识别的 FN_SPEC=$SPEC" >&2; exit 1 ;;
  esac
  # K 合法档受 QSA ring 整除约束：ring = 4*cdiv(4+K,4) 必须整除 block-size
  #   block 1616 → K=1..4 或 9..12 合法（5..8 启动即 AssertionError）
  #   block 1680 → K=5 亦合法（1680/12=140）
  RING=$(( 4 * ( (4 + ${N:-0} + 3) / 4 ) ))
  if [ "${N:-0}" -gt 0 ] 2>/dev/null && [ $(( BLOCK % RING )) -ne 0 ]; then
    echo "[FN-0300] 拒绝启动：MTP K=$N ⇒ QSA ring=$RING 不整除 block-size=$BLOCK（引擎会在加载权重后 AssertionError）。block 1616 的合法 K=1..4/9..12；要用 K=5 请 FN_BLOCK=1680" >&2
    exit 1
  fi
  ARGS+=(--speculative-config "$SPEC_JSON")
  echo "[FN-0300] 投机：$SPEC_JSON (block=$BLOCK, ring=$RING ✓)" >&2
fi

if [ -n "${FN_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  EXTRA=($FN_EXTRA_ARGS); ARGS+=("${EXTRA[@]}")
fi
if [ -n "${FN_EXTRA_ENV:-}" ]; then
  # 控制台「vLLM 附加环境变量」：KEY=VAL 每行一条（与旧栈同契约）
  for kv in $FN_EXTRA_ENV; do export "$kv"; done
fi

# ---------------------------------------------------------------- 参数体检
# 目标：任何「弹窗里能填、本脚本不消费」的 FN_* 都必须显式出声。
# FN_PLE_INT8 / FN_KVOFF 在旧栈就是靠缺省值生效、切档静默失效（09-18 定版事故），
# 这里用穷举比对代替人工记忆，新增字段忘了接也会立刻暴露。
CONSUMED=" FN_VENV FN_BASE FN_ENVFILE FN_MODEL_PATH FN_1M_MODEL_PATH FN_LONGCTX FN_YARN_FACTOR \
FN_MAXLEN FN_MAXLEN_EFF FN_PORT FN_SERVED FN_TP FN_PP FN_PP_PARTITION FN_DTYPE FN_SSMDTYPE \
FN_SEQS FN_GPUMEM FN_BLOCK FN_MBTOKENS FN_MOE FN_EP FN_EAGER FN_ENFORCE_EAGER FN_SPEC \
FN_PREFIX_CACHE FN_CHUNKED FN_ASYNC FN_SEED FN_GENCFG FN_CHATKWARGS FN_CACHE_ROOT \
FN_LOGLEVEL FN_CUDA_VISIBLE_DEVICES FN_EXTRA_ARGS FN_EXTRA_ENV FN_DRY_RUN \
FN_KVOFF FN_KVOFF_BYTES FN_KVOFF_WAIT_TIMEOUT "
NOOP_NOTE_FN_CPU_OFFLOAD_GB="FN_KVOFF=1 时用 FN_KVOFF_BYTES（字节数）指定容量，本变量未接"
NOOP_NOTE_FN_PLE_INT8="官方 0.30.0 只有 BF16 锁页一档，INT8/磁盘驻留是旧镜像自研加载器（README-0300.md §3）"
NOOP_NOTE_FN_PLE_LOC="$NOOP_NOTE_FN_PLE_INT8"
NOOP_NOTE_FN_KV_DTYPE="QSA 要求主 KV 必须 BF16，非 bfloat16 会在建模期 NotImplementedError，故不接受覆盖"
NOOP_NOTE_FN_MAX_SCHED_TOKENS="本档未接（如需请用 FN_EXTRA_ARGS）"
NOOP_NOTE_FN_DISABLE_ALLREDUCE="本档未接（PP2 无此项需求）"
NOOP_NOTE_FN_NOLOG="本档未接（vLLM 0.30 已移除该 flag）"
NOOP_NOTE_FN_LIMIT_MM="多模态上限：本档已跳过多模态 warmup，未接"
UNKNOWN_FN=""
for v in $(env | sed -n 's/^\(FN_[A-Z0-9_]*\)=.*/\1/p' | sort -u); do
  case " $CONSUMED " in *" $v "*) continue;; esac
  eval "note=\${NOOP_NOTE_$v:-}"
  if [ -n "${note:-}" ]; then
    echo "[FN-0300] 忽略 $v=$(printenv "$v")：$note" >&2
  else
    UNKNOWN_FN="$UNKNOWN_FN $v"
  fi
done
[ -n "$UNKNOWN_FN" ] && echo "[FN-0300] 警告：收到本脚本未实现的参数$UNKNOWN_FN（值未生效，需要就写进 FN_EXTRA_ARGS 或补脚本）" >&2

echo "[FN-0300] 模型=$MODEL_PATH (cap=${CAP}) max-model-len=${FN_MAXLEN_EFF}${FN_LONGCTX:+ 长档位=$FN_LONGCTX}" >&2
echo "[FN-0300] PYTHONPATH=$PYTHONPATH" >&2
echo "[FN-0300] 补丁 sha256(上游)=$RT_SHA" >&2
echo "[FN-0300] VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT" >&2

if [ "${FN_DRY_RUN:-0}" = "1" ]; then
  printf '%q ' "$VENV/bin/vllm" "${ARGS[@]}"; echo
  exit 0
fi

mkdir -p "$VLLM_CACHE_ROOT"
exec "$VENV/bin/vllm" "${ARGS[@]}"
