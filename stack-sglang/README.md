# SGLang 栈：18420 部署与避坑（2026-10-02 定版）

> 本目录 = 用 **SGLang 0.5.21**（替代旧 vLLM 定制栈）服务
> `Qwen3.8-Flash-Next-W4A16-AutoRound-1M` 的全套部署件与踩坑记录。
> 旧 vLLM 栈复现见 `../stack-0300/` 与根 README；两套看门狗互斥（同一时刻只跑一套）。

## 0. 性能基准（定长 `ignore_eos` `/generate` 测法，自然文本 gen 长度不定不可比）

| 档位 | 吞吐 | 说明 |
|---|---|---|
| 无投机（SG_SPEC=none） | 64.5 tok/s | 基线 |
| **NEXTN steps=3/draft=4（生产）** | **117~119 tok/s** | accept len 2.80/rate 0.60（自然文本）|
| 旧 vLLM MTP4 栈 | ~90 tok/s | 回滚路径仍可用 |

冷启动 ~10.7 分钟（load 493s + KV 31s + 图 7s）。验收：验证码复述 / 病理正则 /
> ⚠️ **2026-10-03 生产降级记录：NEXTN 全档虽吞吐 118 tok/s，但流式长输出出现质量退化**
> （电报体短句循环、markdown 失衡、句中早 EOS——同 prompt A/B：spec=on 退化密度 9 vs
> spec=none 1，none 版输出高质量表格/代码）。**launch.env 已钉 SG_SPEC=none（64.5 tok/s）**。
> 退化根因指向 draft unquant 后 verify 数值链路仍有污染（嫌疑：QSA ring=4 与 draft 4-token
> verify 的 index-key 环形缓冲复用、或 BF16 draft GEMM 精度）；修复验证前勿开 nextn。
> 另注意：xhigh 思考预算可吃掉整个 max_tokens（实测 3000 预算纯 thinking 不出 content），
> 客户端预算要给思考留头寸。

缓存报告 cached_tokens / 56k 长文召回 —— 全绿。

## 1. 环境

- venv：`/home/ll/sglang-env`（py3.12.14，torch 2.13.0+cu130，flashinfer 0.6.18，
  sgl_kernel 0.4.7，transformers 5.12.1）
- sglang 0.5.21 源码安装（`/home/ll/sglang-src/sglang-0.5.21`，`SGLANG_BUILD_RUST_EXTS=none`）
- 额外 pip 件：**flash-attn 2.8.3.post1 源码编译**（QSA decode 必需，见 §4.8）
- gcc-12：`ppa:ubuntu-toolchain-r/test`（focal 主源最高 gcc-10，JIT 头用 `std::bit_cast` 需 GCC≥11）
  - apt 源文件必须 `*.list` 后缀（sourceparts 不认 `.conf`）；本机走 http（https 不通）+ `[trusted=yes]`

## 2. 启动/停止/看门狗

```bash
export SUDO_PASS=…                      # 部署机 sudo 口令，仓库不携带
bash start-flash-next-sglang.sh         # 读 launch.env（sg_* 参数）
bash stop-flash-next-sglang.sh          # SIGTERM-only（绝不 -9 持 CUDA context 进程）
# 看门狗（systemd user timer，60s tick，15min 启动宽限）：
systemctl --user enable --now fnx-sglang-watchdog.timer
```

launch.env（生产钉版）：`SG_SPEC=nextn`、`SG_SPEC_STEPS=3`；其余默认全在
`sglang-inner.sh`（PP2 26,22 / ctx 1048576 / mem-frac 0.88 / seqs 4 / chunked 8192 /
BF16 KV / page_size=64 自动 / 采样 0.6,0.95,20,0,0.2,1.15 / xhigh 思考 /
qwen3+qwen3_coder / NEXTN steps3 topk1 draft4 + **`--speculative-draft-model-quantization unquant`**）。

## 3. 必带环境变量（inner 已内置）

| 变量 | 原因 |
|---|---|
| `NCCL_P2P_DISABLE=1` | 本机 BAR1 P2P 数据通路**静默损坏**（cmpunlocker 0011/0015/0016 补丁后遗症，10-01 实锤）|
| `SGLANG_ENABLE_PP_SPEC=1` | 放行非 PD 模式的 PP+投机（validation_hook else 分支默认只允许 PD prefill；NEXTN 会别名成 EAGLE）|
| `SGLANG_PP_LAYER_PARTITION=26,22` | 与 vLLM 栈同切分（PP1 兼 lm_head+NEXTN 草稿段略轻）|
| `PATH=/home/ll/gcc10bin:$VENV/bin:$VENV/…/nvidia/cu13/bin:$PATH` | gcc10bin 是 shim 目录（现链 **gcc-12**）：nvcc 13.4 **不认 CXX/CUDAHOSTCXX 环境变量**，host compiler 只从 PATH 找；venv/bin 供 ninja；cu13/bin 供 tilelang 找对版本 nvcc |
| `CUDA_HOME=…nvidia/cu13`、`CXX/CUDAHOSTCXX=gcc10bin/g++` | JIT 同源工具链 |
| `NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK` | pip cu13 与残留 /usr/local/cuda-12.9 共存时 cccl 自检**误报** incompatible（实测代码生成正常；vLLM 栈 FLASHINFER_EXTRA_CUDAFLAGS 同款）|
| `TORCHINDUCTOR_COMPILE_THREADS=1`、`FLASHINFER_DISABLE_VERSION_CHECK=1` | 编译内存/版本噪音 |

root 必须：PLE 95.4GiB 锁页要 memlock=unlimited（`ulimit -l` 门禁在 inner 里）。

## 4. 九个必踩坑（全部已解决，复现时照抄即可绕开）

1. **`--language-model-only` 白名单缺 `Qwen4ExpForConditionalGeneration`** →
   server_args.py `LANGUAGE_MODEL_ONLY_ARCHITECTURES` 加名字（模型侧原生支持 lm-only，白名单过时）。
2. **PP2+投机被拒** → `SGLANG_ENABLE_PP_SPEC=1`（见 §3；要求非 adaptive、非 dp-attention、非 PD）。
3. **sgl-kernel JIT 缺 ninja** → venv/bin 前置 PATH（uv 环境无 pip 但带 ninja）。
4. **JIT 头 `<concepts>`/`std::bit_cast` 编译失败** → gcc≥11（装 gcc-12 做 shim）；
   **nvcc 只从 PATH 认 host compiler**，CXX/CUDAHOSTCXX 环境变量全部无效（实测）。
5. **权重加载内存翻倍 → PP0 OOM（sglang 真 bug）**：marlin repack 后旧 packed Parameter
   处于循环引用（post_load `_TensorState` 快照 + 参数自引用），引用计数不释放，
   **每层 FusedMoE 净漏 1.24GB**（PP0 62.5GB 死 / PP1 58.2GB 侥幸存活）。
   修复=loader.py `postprocess_weights` 每模块后 `gc.collect()`（`patch_dsh_sglang_fixes.py`）。
   判据：`Load weight end … mem usage=` 应≈层权重理论值（PP1 31.7GB / PP0 37.5GB）。
6. **QSA ring 整除约束**：`--speculative-num-draft-tokens ≤ compress_ratio(4)`
   → NEXTN 合法档 = steps≤3 + draft_tokens=4（steps=4/draft=5 在 verify 图捕获期报
   `pending index-key ring holds one group`）。
7. **tilelang（QSA verify 内核）nvcc 版本打架**：tilelang 内置 tvm 找到 cu13 nvcc 后，
   默认 include 命中 /usr/local/cuda-12.9 的 cuda.h → cccl CTK 自检 #error。
   解法=§3 的 `NVCC_PREPEND_FLAGS`（一行覆盖 libgen 与 contrib 两条编译路径）；
   `p_cccl.sh` 是 libgen.py 的等价文件补丁（双保险，可 --revert）。
8. **QSA decode 选到 FA4-cute（Blackwell 专属）→ sm80 MLIR `crd2idx` 崩**：
   机器只装了 flash_attn_4 预览版（纯 Python cute/），sglang 回退链要求 FA2 顶层
   `flash_attn_varlen_func`。**必须编译安装 FA2**：
   `MAX_JOBS=6`（18 并行会 OOM 杀 cicc——bwd 内核每个 4-8GB）、
   `TORCH_CUDA_ARCH_LIST=8.0`、cccl 旁路、`setup.py bdist_wheel` 直编（绕开 uv 网络层）、
   增量复用 .o 可省一半时间（wheel 在 uv sdist 缓存 dist/ 下）。
9. **NEXTN draft 被 4bit 误建 → 首 token 即 EOS（matched_stop）**：sglang
   `_mtp_quant_config` 只处理 modelopt/quark/NPU，**auto-round 落到 passthrough**
   → draft 按 marlin-4bit 建，而 checkpoint `mtp.*` 1565 键全是 BF16 明文 → 垃圾 draft
   污染全链路。官方解=`--speculative-draft-model-quantization unquant`（本仓库 inner 已带）。
   附带教训：marlin 内核 **act/scale 必须同 dtype**（bf16×bf16/fp16×fp16 模板对），
   AutoRound 的 fp16 组 scale 要在 postprocess 转 bf16（`patch_scale_bf16.py`，
   同时恢复 fused_marlin_moe 的两条 dtype 断言——断言是真话）。

## 5. 补丁清单（全部幂等、带 .bak、可 --revert）

| 补丁 | 文件 | 应用脚本 |
|---|---|---|
| lm-only 白名单 | srt/server_args.py | 手动（一行 tuple 追加）|
| 每层 gc.collect() + scale 断言还原 | srt/model_loader/loader.py, fused_marlin_moe.py | `patch_dsh_sglang_fixes.py` |
| scale→bf16 | srt/hardware_backend/gpu/quantization/gptq_kernels.py | `patch_scale_bf16.py` |
| tilelang cccl | tilelang/jit/adapter/libgen.py | `p_cccl.sh` |
| FA2 | 新增包 flash-attn 2.8.3.post1 | §4.8 编译步骤 |

**改 .py 后必删对应 `__pycache__/*.pyc`**（否则改动不生效且无报错）。

## 6. 主机既有铁律（沿用 vLLM 时代，勿动）

- 绝不 SIGKILL 持 CUDA context 进程（Xid31 风险）；启动前必确认显存归零。
- BF16 KV 锁死（fp8 会 QSA NotImplementedError）；chunked-prefill 8192（勿 32768）。
- 127 机 nvme 盘符重启互换（数据盘按 by-id：JZ-SSD2T-XW）；RTC 会跳（判时序用 uptime）。
- 回滚 vLLM 栈：`stack-0300/` 脚本 + 停本看门狗（`systemctl --user disable --now fnx-sglang-watchdog.timer`）。
