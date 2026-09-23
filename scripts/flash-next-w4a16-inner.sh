#!/bin/bash
# ---- sudo 凭据（脱敏版）----
# 生产机上 chroot 内进程属 root，宿主脚本需要 sudo。两种给法二选一：
#   A) 推荐：/etc/sudoers.d/ 配 NOPASSWD（见 docs/01-host-prep.md），无需本变量
#   B) 环境变量 SUDO_PASS 传入密码（不落盘、不写进脚本）
if [ -n "${SUDO_PASS:-}" ]; then
  SUDO() { printf '%s\n' "$SUDO_PASS" | sudo -S -p '' "$@"; }
else
  SUDO() { sudo -n "$@"; }
fi

# chroot 内启动 Qwen3.8-Flash-Next W4A16-AutoRound —— gavinxym/170hx-2 手册方案（2026-09-15）
#
# 依据：github.com/gavinxym/170hx-2-qwen3.8-flash-next（W4A16-AutoRound 档位手册）：
#   PP2 + MTP4 + block-size 1616 + mamba-ssm-cache-dtype float32
#   + VLLM_PP_LAYER_PARTITION=26,22 + seqs 4 + mbtokens 8192 + gpu-mem 0.95
#
# 与手册的偏离（本机硬约束，均已实测依据）：
#   · PLE 表 95.4GiB(BF16) 手册要求常驻 CPU 内存(需≥128GB RAM)，本机仅 31GB
#     → 走 PLE mmap（SSD 零拷贝 + PLE worker 异步 gather，复用 NVFP4 已验证链路；
#       vllm_ple_mmap.py 已补 BF16 支持，备份 .bak-bf16-0915）
#   · quantization_config 已由 auto-round 改写为 GPTQ（本镜像 vLLM 无 auto-round
#     quant method；张量本就是 GPTQ 打包。备份 config.json.bak-autoround2gptq-0915）
#   · CUDA 图模式（FULL_AND_PIECEWISE）在本机 PLE 栈上 >8K prompt 必触发 GPU Xid31
#     （09-14/09-15 多次实锤，含 eager+MTP4 下 GDN ssm_state 非法访问一次）
#     → --enforce-eager 定版；手册 ~115 tok/s 数据基于其平台的图模式，本机 eager 会低
#
# dry-run：FN_DRY_RUN=1 只打印将执行的命令。
set -u
export PATH=/usr/local/cuda/bin:$PATH

FN_ENVFILE=${FN_ENVFILE:-/home/ll/deploy/flash-next-w4a16-launch.env}
if [ -f "$FN_ENVFILE" ]; then
  # shellcheck disable=SC1090
  . "$FN_ENVFILE"
fi

# ---------- PLE：mmap 路径（手册的 CPU offload 需 96GB RAM，本机改 mmap）----------
export VLLM_PLE_MMAP=1
export VLLM_PLE_CPU_OFFLOAD=1
# 【2026-09-22 PLE 表改走 INT8 磁盘驻留（宿主内存账复盘后定版）；2026-09-23 拆出位置开关】
# 磁盘驻留（disk）= mmap 直读产物、零堆分配、页缓存可回收：内存紧张时 OS 可逐出，
#   不会与 CPU KV 二级缓存 64GiB pinned 抢不可回收内存（09-22 18:08 卡死故障的教训，
#   见 /home/ll/deploy/HANG-0922-18420.md）。精度二选一：INT8（47.7+0.6GiB，产物
#   /media/ll/data/ple，quantize_ple.py 生成，几何/字节自校验不过自动回落 BF16 mmap）
#   或 BF16 mmap（95.4GiB，直读 checkpoint shard16）。实测 INT8 decode 与 BF16 持平。
# 内存驻留（heap）= 旧匿名堆路径：PLE 表被 PleOffloadWorker 读进匿名内存 95.4GiB，
#   全驻留零缺页但完全不可回收（本机 swap 已关），只存在 BF16 形态——INT8 产物没有
#   内存驻留实现，此模式下精度强制按 BF16 处理。
# 开关：FN_PLE_LOC=disk|heap（8889 弹窗「PLE 表位置」，只管放内存/放硬盘）；
#   FN_PLE_INT8=1|0（弹窗「PLE 表加载（n-gram 精度）」，只管精度）。
#   两者正交，四种组合：INT8+disk(VLLM_PLE_INT8_DIR mmap 47.7GiB 可回收) /
#   INT8+heap(VLLM_PLE_INT8_MEMORY=1 匿名堆 48.3GiB 不可回收) /
#   BF16+disk(mmap safetensors 零堆) / BF16+heap(匿名堆 95.4GiB 不可回收)。
# 兼容：FN_PLE_LOC 未传（旧手动命令/旧预设）→ INT8=0 视作 heap（与 09-22 前逐字一致）、
#   INT8=1 视作 disk。回滚手动命令 FN_PLE_INT8=0 语义不变。
FN_PLE_INT8="${FN_PLE_INT8:-1}"
if [ -z "${FN_PLE_LOC:-}" ]; then
  if [ "$FN_PLE_INT8" = "0" ]; then FN_PLE_LOC=heap; else FN_PLE_LOC=disk; fi
fi
# 精度定精度、位置定内存/硬盘，两者正交（四种组合都成立）：
#   INT8+内存 = VLLM_PLE_INT8_MEMORY 匿名堆 48.3GiB（[FN-PLE-INT8MEM] 引擎侧新增）
if [ "$FN_PLE_LOC" = "heap" ] && [ "$FN_PLE_INT8" = "1" ] && [ -f /media/ll/data/ple/ple_ngram_meta.json ]; then
  export VLLM_PLE_DISK_RESIDENT=1
  export VLLM_PLE_INT8_DIR=/media/ll/data/ple
  export VLLM_PLE_INT8_MEMORY=1
  echo "[FN-PLE-INT8] PLE 表走 INT8 内存驻留：/media/ll/data/ple (47.7+0.6 GiB 匿名堆，不可回收，零磁盘 I/O)" >&2
elif [ "$FN_PLE_LOC" = "heap" ]; then
  echo "[FN-PLE-LOC] PLE 表 BF16 内存驻留（匿名堆 95.4GiB，不可回收）" >&2
elif [ "$FN_PLE_INT8" = "1" ] && [ -f /media/ll/data/ple/ple_ngram_meta.json ]; then
  export VLLM_PLE_DISK_RESIDENT=1
  export VLLM_PLE_INT8_DIR=/media/ll/data/ple
  echo "[FN-PLE-INT8] PLE 表走 INT8 磁盘驻留：/media/ll/data/ple (47.7+0.6 GiB, 可回收页缓存)" >&2
else
  export VLLM_PLE_DISK_RESIDENT=1
  echo "[FN-PLE-LOC] PLE 表走 BF16 磁盘驻留（mmap safetensors，零堆，可回收页缓存）" >&2
fi
export VLLM_PLE_GDS=0
export VLLM_PLE_GDS_RUNNER=0
export VLLM_PLE_MMAP_WORKERS=32
export VLLM_PLE_MMAP_CHUNK=2048
export VLLM_PLE_MMAP_PREWARM=0
export VLLM_PLE_NVFP4_GPU=0
# 【2026-09-15 按参考机对齐】参考机环境变量只有 4 个（CPU_OFFLOAD/MMAP/GDS/FLASHINFER），
# 无 VLLM_PLE_MTP_MOE_BACKEND=triton；该项曾触发 "Unknown vLLM env" 警告，
# 疑为 MTP4 乱码来源之一 → 缺省不设，需要时 FN_MTP_MOE_BACKEND=triton 显式开
if [ -n "${FN_MTP_MOE_BACKEND:-}" ]; then
  export VLLM_PLE_MTP_MOE_BACKEND="$FN_MTP_MOE_BACKEND"
fi

# PP 形态（TP=1）下三个 TP=2 专属 hook 必须关（draft INT8 / TP_PLE / projection
# 内部硬断言 world_size==2）；hc_gemv 校验已改为切分无关（.bak-ppguard-0914）→ 保留
# export VLLM_USE_V2_MODEL_RUNNER=0  # V1 撞 self.drafter 未守卫（上游 PP+MTP 未完成）；改走给 V2 补权重加载
export Q38_DRAFT_INT8=0
export Q38_TP_PLE=0
export Q38_PROJECTION=0
export Q38_HC_GEMV=${FN_HC_GEMV:-1}
export Q38_TP_PLE_VERIFY_STEPS=8
export R38_PP1_FULL_DECODE=${FN_PP1_FULL_DECODE:-0}
export FN_DIAG_SAMPLE=${FN_DIAG_SAMPLE:-0}
# 以下四个是官方 GDS+TEP2 组合的运行时开关，本机 mmap 栈加了会崩 → 不启用
# export QWEN_GDN_REPLAY=1 / GDN_DIAG_DISABLE_JIT_MONITOR=1 / CUDA_MODULE_LOADING=LAZY / PYTORCH_NVML_BASED_CUDA_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 参考机 18430 §2.2：绕过 flashinfer cubin/package 版本不一致报错
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL="${FN_LOG_LEVEL:-INFO}"

# gavinxym 手册：26/22 层切分（39.20/39.95 GiB 最均衡，KV 池最大）
# 【2026-09-15 按参考机对齐】参考机 18430 不设 VLLM_PP_LAYER_PARTITION
# （§4："用 fork 默认，不要抄手册"）→ FN_PARTITION=none 时不设该 env，
# 交回镜像默认切分逻辑。缺省仍为 26,22（手册档，已长期验证）。
if [ "${FN_PARTITION:-26,22}" != "none" ]; then
  export VLLM_PP_LAYER_PARTITION="${FN_PARTITION:-26,22}"
fi

# 纯文本服务：跳过 ~33s 的多模态 dummy warmup（补丁见 renderers/base.py）
export VLLM_SKIP_MM_WARMUP=1
export VLLM_CACHE_ROOT="${FN_CACHE_ROOT:-/root/.cache/vllm-flash-next-w4a16}"
mkdir -p "$VLLM_CACHE_ROOT" 2>/dev/null || true

# 无 P2P（CNS）：NCCL 走 host SHM
# 0919 P2P 实验：平台已换 X99-T8，topo -p2p r=OK，实测跨卡 copy 5.27GB/s，启用 P2P（回滚=恢复 .bak-p2p-0919）
# export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
# 0919 关键：GA100 双卡为 PHB 拓扑（同桥异根端口），NCCL 默认 P2P 级别 LOC 不跨 PHB，必须显式放行（见 CMP170HX-P2P-打通记录.md §2.5）
export NCCL_P2P_LEVEL=PHB  # 0919 A/B 定版：P2P 生效（via P2P/IPC），性能与 SHM 持平，保留 P2P（省 CPU 中继）
export NCCL_CUMEM_ENABLE=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"  # 0919 实验结束回 WARN（实验期曾临时 INFO）

FN_MAXLEN_EFF="${FN_MAXLEN:-262144}"
if [ "$FN_MAXLEN_EFF" != "auto" ] && [ "$FN_MAXLEN_EFF" -gt 262144 ] 2>/dev/null; then
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
fi

# 附加环境变量（每行 KEY=VALUE）
if [ -n "${FN_EXTRA_ENV:-}" ]; then
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in \#*) continue ;; esac
    export "$line" 2>/dev/null || true
  done <<< "$FN_EXTRA_ENV"
fi

MODEL_PATH="${FN_MODEL_PATH:-/media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound}"
# ===== 长上下文档位（2026-09-19 补齐）=====
# 控制台 scriptModelLaunchPlan 在 max-model-len > 262144 时下发：
#   FN_LONGCTX=1  FN_1M_MODEL_PATH=<YaRN 副本目录>  FN_YARN_FACTOR=<2.0|4.0>
# 副本是「只改 config.json（YaRN 已烘进 rope_parameters）、其余文件软链回原目录」，
# 所以这里只需切换模型目录，不需要 --hf-overrides。
# 副本缺失时直接拒启：用原生权重跑超长上下文 = 未缩放 RoPE，位置越过原生上限会退化甚至越界。
if [ "${FN_LONGCTX:-0}" = "1" ]; then
  LC_PATH="${FN_1M_MODEL_PATH:-}"
  if [ -z "$LC_PATH" ] || [ ! -f "$LC_PATH/config.json" ]; then
    echo "[FATAL][FN-LONGCTX] 要求长上下文档，但 YaRN 副本不可用：'${LC_PATH:-<未下发 FN_1M_MODEL_PATH>}'。拒绝以未缩放 RoPE 启动。" >&2
    exit 1
  fi
  MODEL_PATH="$LC_PATH"
  LC_CAP="$(/usr/bin/python3.12 -c "import json;c=json.load(open('$MODEL_PATH/config.json'));print(c.get('text_config',c).get('max_position_embeddings',0))" 2>/dev/null || echo 0)"
  if [ "${LC_CAP:-0}" -gt 0 ] && [ "$FN_MAXLEN_EFF" -gt "$LC_CAP" ] 2>/dev/null; then
    echo "[FN-LONGCTX] max-model-len $FN_MAXLEN_EFF 钳到副本上限 $LC_CAP" >&2
    FN_MAXLEN_EFF="$LC_CAP"
  fi
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  echo "[FN-LONGCTX] 模型切到 YaRN 副本 $MODEL_PATH (factor=${FN_YARN_FACTOR:-?}, 副本上限=${LC_CAP:-?}, 本次 max-model-len=$FN_MAXLEN_EFF)" >&2
fi
# 采样参数缺省（可被 FN_GENCFG 覆盖；与 server.js SCRIPT_MODELS.base、快启预设 p2p-mtp4 一致）
GENCFG_DEFAULT='{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.1,"repetition_penalty":1.05}'

ARGS=(
  serve "$MODEL_PATH"
  --served-model-name "${FN_SERVED:-qwen3.8-flash-next}"
  --host 0.0.0.0 --port "${FN_PORT:-18420}"
  --load-format safetensors
  --safetensors-load-strategy lazy
  --distributed-executor-backend mp
  --tensor-parallel-size "${FN_TP:-1}"
  --pipeline-parallel-size "${FN_PP:-2}"
  --dtype "${FN_DTYPE:-bfloat16}"
  --max-model-len "$FN_MAXLEN_EFF"
  # gavinxym 手册：两级 CSA + linear 层 block size 不一致 → 1616 必传
  --block-size "${FN_BLOCK:-1616}"
  # gavinxym 手册：sharded mamba cache dtype 不一致 → float32 必传
  --mamba-ssm-cache-dtype "${FN_SSMDTYPE:-float32}"
  --max-num-seqs "${FN_SEQS:-4}"
  --gpu-memory-utilization "${FN_GPUMEM:-0.95}"
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --max-num-batched-tokens "${FN_MBTOKENS:-8192}"

  --moe-backend "${FN_MOE:-auto}"

  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --trust-remote-code
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'
  # 【2026-09-21 复读修复】temperature 0.3→0.6、repetition_penalty 1.0→1.05、presence_penalty 0→0.1。
  # 0.3 是 09-19 为 MTP 接受率（34.1%→38.9%、decode 92→100 tok/s）刻意调低的，代价=循环复读；
  # 用户拍板优先治复读。presence 仅给 0.1（09-19 实验证高 presence 显著伤 MTP 接受率）。
  # 弹窗/快启预设的采样值经 FN_GENCFG 覆盖本缺省（此前 inner 不读 FN_GENCFG，预设采样值静默失效，已修）。
  # 与 server.js SCRIPT_MODELS.base、快启预设 p2p-mtp4 三处保持一致。
  --override-generation-config "${FN_GENCFG:-$GENCFG_DEFAULT}"
  # splitting_ops：PLE mmap lookup 等 14 个算子必须留在图外分段执行
  # （我们的官方镜像版图模式必需；参考机 fork 镜像内部已默认）
  '-cc.splitting_ops=["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention","vllm::hpc_rope_norm_forward","vllm::ple_mmap_lookup","vllm::ple_gds_lookup"]'
  '-cc.inductor_compile_config={"combo_kernels":false,"benchmark_combo_kernel":false}'
  --no-enable-flashinfer-autotune
)
# 官方 TEP2 才需要专家并行；PP2 下不加
[ "${FN_EP:-0}" = "1" ] && ARGS+=(--enable-expert-parallel)

# 【2026-09-15 按参考机 18430（同卡型 CMP 170HX，MTP4 稳定运行 9h）配置修复】
# 参考机用 FULL_AND_PIECEWISE + 显式 capture_sizes 且 MTP4 输出正常；
# 我们此前用 --enforce-eager 时 MTP4 输出乱码（疑与 CUDA 图 padding 语义
# 不同 → PP+MTP 的 draft_tokens 中继错位、污染验证输入有关）。
# 此前"图模式 >8K prompt 必 Xid31"的实验未限制 capture_sizes（默认捕获大量
# 尺寸），本次严格照抄参考机的 [1,2,4,8,16,24,32,40]。
# 回滚：FN_EAGER=1 恢复 --enforce-eager。
if [ "${FN_EAGER:-0}" = "1" ]; then
  ARGS+=(-cc.cudagraph_mode=NONE)
  ARGS+=(--enforce-eager)
else
  ARGS+=(-cc.cudagraph_mode=FULL_AND_PIECEWISE
         '-cc.cudagraph_capture_sizes=[1,2,4,8,16,24,32,40]')
fi
# async-scheduling：参考机未启用 → 缺省关；FN_ASYNC=1 显式开
[ "${FN_ASYNC:-0}" = "1" ] && ARGS+=(--async-scheduling)

# 投机解码：默认**关闭**（先验证基线稳定性）。
# 手册定稿为 MTP4；注意本机 PP2+MTP 残余 bug（09-15 05:20 NVFP4 栈 MTP4 首个
# prefill 即 GDN ssm_state 非法访问 → Xid31），MTP4 实验需在基线稳定后单独开。
case "${FN_SPEC:-none}" in
  none|"") : ;;
  mtp4) ARGS+=(--speculative-config '{"method":"mtp","num_speculative_tokens":4}') ;;
  mtp6) ARGS+=(--speculative-config '{"method":"mtp","num_speculative_tokens":6,"use_local_argmax_reduction":true}') ;;
  *)    ARGS+=(--speculative-config "$FN_SPEC") ;;
esac

# 【2026-09-22 KV 二级缓存】vLLM OffloadingConnector：GPU KV 池 → 宿主内存 64GiB
# 依赖镜像内 kvoff-c1/c2 补丁（QSA 环形分组排除且保留 group_idx 位置 / PP>1 私有
# pinned 缓冲 / c5a 解除查找侧 eagle 双罚——09-22「只存不命中」根因已修，命中实测打通）。
# 容量：96GiB ≈ 126 万 token（76KB/token，09-23 实测口径）> GPU 池 122 万；
# 物理钉住 ≈1.56×配置（≈150GB），须配 PLE heap（disk 模式页缓存被挤有 09-19 缺页事故模式）。
# store_threshold=2：只存被查过≥2 次的块（write_back 类比，防一次性文档冲刷档位——
# 参照 ChinaBoy0618/170hx 仓库 v1.0.0 write_through→write_back 的演进经验）。
# 回滚：FN_KVOFF=0；调容量：FN_KVOFF_BYTES=<字节数>
if [ "${FN_KVOFF:-1}" = "1" ]; then
  KVOFF_BYTES="${FN_KVOFF_BYTES:-103079215104}"
  ARGS+=(--kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${KVOFF_BYTES},\"store_threshold\":2}}")
  echo "[FN-KVOFF] CPU KV 二级缓存：cpu_bytes_to_use=${KVOFF_BYTES} (96 GiB), store_threshold=2" >&2
fi

if [ "${FN_DRY_RUN:-0}" = "1" ]; then
  echo "[dry-run] /usr/bin/python3.12 -m vllm.entrypoints.cli.main ${ARGS[*]} ${FN_EXTRA_ARGS:-}"
  echo "[dry-run] PLE: MMAP=$VLLM_PLE_MMAP CPU_OFFLOAD=$VLLM_PLE_CPU_OFFLOAD GDS=$VLLM_PLE_GDS | PARTITION=$VLLM_PP_LAYER_PARTITION | HC_GEMV=$Q38_HC_GEMV"
  echo "[dry-run] CACHE_ROOT=$VLLM_CACHE_ROOT MAXLEN=$FN_MAXLEN_EFF BLOCK=${FN_BLOCK:-1616} SSMDTYPE=${FN_SSMDTYPE:-float32}"
  exit 0
fi

exec /usr/bin/python3.12 -m vllm.entrypoints.cli.main "${ARGS[@]}" ${FN_EXTRA_ARGS:-}
