#!/bin/bash
# 18420 Flash-Next W4A16-AutoRound-1M —— SGLang 0.5.21（源码安装，rust 扩展关闭）
# 必须 root：PLE 95.4 GiB 锁页要 memlock=unlimited（ll 被钉 64 KiB，与 vLLM 0300 栈同理）。
# 参数对齐生产 vLLM 0.30.0 实例：
#   PP2(26,22) / 1M(YaRN 已在 config) / QSA 接管 page_size=64 / seqs 4 / cbt 8192 /
#   NEXTN x4 / 采样 0.6,0.95,20,min_p0,pres0.2,rep1.15 / xhigh 思考 / qwen3+qwen3_coder
# 注意：SGLang 没有 --disable-custom-all-reduce（vLLM 专属）；TP=1 无 all-reduce，
#       跨卡 PP 中继走 NCCL，由 NCCL_P2P_DISABLE=1 兜底（10-01 BAR1 P2P 静默损坏铁律）。
set -u
MODEL=${SG_MODEL:-/media/ll/data/models-1m/Qwen3.8-Flash-Next-W4A16-AutoRound-1M}
PORT=${SG_PORT:-18420}
SERVED=${SG_SERVED:-qwen3.8-flash-next}
TP=${SG_TP:-1}
PP=${SG_PP:-2}
CTX=${SG_CTX:-1048576}
MEMFRAC=${SG_MEMFRAC:-0.88}
SEQS=${SG_SEQS:-4}
CBT=${SG_CHUNKED_PREFILL:-8192}
SPEC=${SG_SPEC:-nextn}          # nextn | none
GENCFG=${SG_GENCFG:-{\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20,\"min_p\":0,\"presence_penalty\":0.2,\"repetition_penalty\":1.15}}
CTKwargs=${SG_CT_KWARGS:-{\"enable_thinking\":true,\"preserve_thinking\":true,\"reasoning_effort\":\"xhigh\"}}

MEMLOCK=$(ulimit -l)
if [ "$MEMLOCK" != "unlimited" ]; then
  echo "[SG-18420] 致命：memlock=$MEMLOCK，PLE 锁页会失败 ⇒ 必须 root 运行" >&2
  exit 1
fi

# ── 本机铁律环境（10-01 BAR1 P2P 静默损坏 / 08-30 Xid31）──
export NCCL_P2P_DISABLE=1
export NCCL_CUMEM_ENABLE=0
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
export CUDA_VISIBLE_DEVICES="${SG_CVD:-0,1}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TORCHINDUCTOR_COMPILE_THREADS=1
# PP2+投机(NEXTN→EAGLE) 非 PD 模式官方开关：validation_hook 的 else 分支只放行 PD prefill，
# 本开关走 pp_spec_relay 中继（末段草稿、各段重建 verify 输入），要求非 adaptive、非 dp-attention。
export SGLANG_ENABLE_PP_SPEC=1
# JIT（gptq_marlin_repack 等 sgl-kernel JIT）工具链：ninja 在 venv/bin，nvcc 在 cu13 包内；
# 系统 g++=9.4 无 <concepts>；JIT 头用 std::bit_cast 需 GCC 11+，装 gcc-12（toolchain-r PPA）。
# gcc10bin 是目录名遗留，内链现指 gcc-12/g++-12。
# nvcc 13.4 不认 CXX/CUDAHOSTCXX env，host compiler 只从 PATH 找（实测），必须 shim 目录前置。
# tilelang/TVM 找 nvcc 优先 PATH（找不到会回落 /usr/local/cuda=12.9 与 cu13 头不兼容，实测），cu13/bin 必须在 PATH。
export PATH="/home/ll/gcc10bin:/home/ll/sglang-env/bin:/home/ll/sglang-env/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH"
export CUDA_HOME=/home/ll/sglang-env/lib/python3.12/site-packages/nvidia/cu13
export CXX=/home/ll/gcc10bin/g++
export CUDAHOSTCXX=/home/ll/gcc10bin/g++
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"   # tilelang/contrib nvcc 编译点通用注入（本机 pip cu13 与 /usr/local/cuda-12.9 共存时 cccl 自检误报）
# tilelang(QSA verify 内核 JIT)数值与编译头治理：禁 --use_fast_math（fp16 scale GEMM 保精度）；
# CUDA_HOME 指 pip cu13 包根，include 与 nvcc 同源（/usr/local/cuda-12.9 的头会打架）。
export CUDA_HOME=/home/ll/sglang-env/lib/python3.12/site-packages/nvidia/cu13
# PP 层切分（与 vLLM 栈 26,22 一致：末段兼 lm_head+NEXTN 草稿，前段多两层平衡）
# 内存探针（SG_PROBE=1）：PYTHONPATH 挂 sitecustomize，每层 repack 记账
if [ "${SG_PROBE:-0}" = "1" ]; then export PYTHONPATH="/home/ll/deploy/sglang-18420/probe${PYTHONPATH:+:$PYTHONPATH}"; fi

export SGLANG_PP_LAYER_PARTITION="${SG_PP_PART:-26,22}"

PY=/home/ll/sglang-env/bin/python
ARGS=(
  -m sglang.launch_server
  --model-path "$MODEL"
  --served-model-name "$SERVED"
  --host 0.0.0.0 --port "$PORT"
  --tp-size "$TP" --pp-size "$PP"
  --context-length "$CTX"
  --dtype bfloat16
  --mem-fraction-static "$MEMFRAC"
  --max-running-requests "$SEQS"
  --chunked-prefill-size "$CBT"
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --trust-remote-code
  --language-model-only
  --enable-cache-report
  --default-chat-template-kwargs "$CTKwargs"
  --preferred-sampling-params "$GENCFG"
)
if [ "$SPEC" = "nextn" ]; then
  ARGS+=(--speculative-algorithm NEXTN
         --speculative-draft-model-quantization unquant
         --speculative-num-steps 3
         --speculative-eagle-topk 1
         --speculative-num-draft-tokens 4)
fi
if [ "${SG_DRY_RUN:-0}" = "1" ]; then
  printf "%q " "$PY" "${ARGS[@]}"; echo; exit 0
fi
exec "$PY" "${ARGS[@]}"
