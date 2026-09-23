# Qwen3.8-Flash-Next（W4A16-AutoRound）在 2× CMP 170HX 上的 vLLM 部署：可复现手册

> 这是一套**已经在生产上跑着的**服务的完整复刻件：推理端点 `http://<部署机>:18420/v1`，
> 模型 `qwen3.8-flash-next`（176B 总参 / 约 6B 激活的 MoE + PLE n-gram 嵌入 + GDN 线性注意力混合架构），
> 跑在两张 64 GB 的 **NVIDIA CMP 170HX**（GA100 die，SM80，矿卡解锁）上，用 **PP2 + MTP4**，
> 上下文 **1M token**，decode 稳态 **90~130 tok/s**，前缀缓存命中率 **91%**。
>
> 本仓库包含：镜像获取脚本、**22 个 vLLM 补丁（逐条带根因说明）**、启动/停止/看门狗脚本、
> PLE 表离线 INT8 量化器、长上下文副本生成器、一键体检脚本，以及 **40+ 条踩坑记录**。
>
> 最后同步生产机状态：**2026-09-23**（所有数值为该日实测，非引用）。

---

## 0. 这个模型为什么难部署（先读这段，否则后面看不懂）

Qwen3.8-Flash-Next 不是普通 Transformer，它有三处"常规 vLLM 配方会直接撞墙"的设计：

| 特性 | 后果 |
|---|---|
| **PLE / n-gram 嵌入表**：一张 `[320,001,536 × 160]` 的 BF16 表（**95.4 GiB**），第 2 层用 | 官方手册要求 **≥128 GB CPU 内存常驻**。95 GB 表放 SSD 会慢到不可用——本仓库用**离线 INT8（48.3 GiB）+ 页缓存驻留**绕过，见 [`docs/04-ple-int8.md`](docs/04-ple-int8.md) |
| **混合注意力**：48 层里每 4 层才 1 层全注意力，其余是 GDN 线性注意力 + 两级 CSA + QSA | block size 必须 `1616`、mamba cache 必须 `float32`，**漏一个直接起不来** |
| **MTP 投机解码头**（checkpoint 内含 31 个 `mtp.*` 权重） | `num_speculative_tokens` 不是任意值：受 **QSA ring capacity 整除 block-size** 约束，block 1616 时合法档只有 1~4 和 9~12，见 [`docs/05-patches.md#补丁-01`](docs/05-patches.md) |

再加上硬件侧的两个坑：矿卡 **CMP 170HX 需要解锁固件**（8 GB→64 GB）且 **PCIe Gen2 只在驱动 probe 的几秒窗口内可写**；
无 NVLink、双卡 P2P 走 PHB → **TP2 反而更慢，PP2 才是正解**。

结论：**上游 vLLM 在 "PP>1 + MTP + PLE CPU offload" 三者同开时有硬伤，必须打补丁**。这就是本仓库的主体。

---

## 1. 最终形态一览

```
客户端 ──OpenAI 兼容 API──► :18420  (host 0.0.0.0)
                              │
              宿主 bash 脚本（chroot 启动，进程属 root）
                              │
   chroot /media/ll/data/vllm-image/rootfs
     └─ python3.12 -m vllm.entrypoints.cli.main serve ...
          ├─ vLLM v0.1.dev20073（官方定制镜像 vllm/vllm-openai:qwen38-flash-next）
          │    └─ 本仓库 22 个补丁（PP 解禁 PLE / MTP 中继 / 磁盘驻留 INT8 表 / KV 二级缓存 …）
          ├─ EngineCore
          ├─ Worker_PP0  → GPU0（层 1~26）
          ├─ Worker_PP1  → GPU1（层 27~48 + lm_head + MTP 草稿 + 采样）
          └─ PleOffloadWorker（CPU 侧 n-gram 查表，INT8 + 逐行 scale 反量化）
```

| 项 | 生产值 |
|---|---|
| served 模型名 | `qwen3.8-flash-next` |
| 并行 | **PP2**（`VLLM_PP_LAYER_PARTITION=26,22`），TP1 |
| 投机 | **MTP4** `{"method":"mtp","num_speculative_tokens":4,"use_local_argmax_reduction":false}` |
| 上下文 | **1,048,576**（YaRN×4 副本，见 `scripts/make-longctx-copy.py`） |
| CUDA 图 | `FULL_AND_PIECEWISE` + `capture_sizes [1,2,4,8,16,24,32,40]`（**不是** enforce-eager） |
| PLE 表 | **INT8 逐行 scale**，48.3 GiB，当前驻留形态=内存堆 |
| KV 二级缓存 | OffloadingConnector → 宿主内存 96 GiB（`FN_KVOFF=1`，需 c1~c5a 补丁） |
| GPU KV 池 | **1,224,366 token**（18.16 GiB，@1M 上下文并发 1.17×） |
| 显存 | 39.6 + 40.3 GiB / 卡（`gpu-memory-utilization 0.95`） |
| 冷启动 | **约 4 分钟**（权重 78 s + PLE 挂载 + init engine 61 s） |
| decode | 90~130 tok/s（MTP 接受长度 3.95） |
| 预填充 | 6,900~8,700 tok/s |
| 前缀缓存命中率 | 累计 **91%** |
| 采样缺省 | `temperature 0.6, top_p 0.95, top_k 20, min_p 0, presence_penalty 0.1, repetition_penalty 1.05` |

完整 argv 与环境变量快照：[`tools/live-cmdline.txt`](tools/live-cmdline.txt)、[`tools/live-env.txt`](tools/live-env.txt)。

---

## 2. 最短复现路径

前提：一台已装好 NVIDIA 驱动、已解锁显存、rootfs 已解出的机器。若从零开始，按顺序读
[`docs/01`](docs/01-host-prep.md) → [`docs/02`](docs/02-image-rootfs.md) → [`docs/03`](docs/03-model.md)。

```bash
git clone <本仓库> flashnext && cd flashnext

# ① 打 vLLM 补丁（幂等、带 sha256 校验、可 --revert 回滚）
sudo python3 scripts/apply-patches.py \
     --target /media/ll/data/vllm-image/rootfs/usr/local/lib/python3.12/dist-packages/vllm --apply

# ② 离线量化 PLE n-gram 表（一次性，约 8 分钟；不做则退回 95 GiB BF16 表，需 ≥128 GB 内存）
python3 scripts/quantize_ple.py \
     --model /media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound \
     --out   /media/ll/data/ple

# ③（可选）做 1M 上下文副本；只做 262144 原生档可跳过
python3 scripts/make-longctx-copy.py \
     --src /media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound \
     --dst /media/ll/data/models-1m/Qwen3.8-Flash-Next-W4A16-AutoRound-1M \
     --target-len 1048576

# ④ 装脚本到部署目录并按本机改路径（脚本内路径是变量，见 docs/06）
mkdir -p /home/ll/deploy && cp scripts/{start-flash-next-w4a16.sh,flash-next-w4a16-inner.sh,stop-flash-next-w4a16.sh} /home/ll/deploy/

# ⑤ 启动（后台，日志落 /home/ll/deploy/vllm-flash-next-w4a16.log）
FN_MAXLEN=1048576 FN_SEQS=3 FN_ASYNC=1 FN_PLE_INT8=1 FN_PLE_LOC=heap \
FN_1M_MODEL_PATH=/media/ll/data/models-1m/Qwen3.8-Flash-Next-W4A16-AutoRound-1M FN_LONGCTX=1 \
setsid nohup bash /home/ll/deploy/start-flash-next-w4a16.sh >/dev/null 2>&1 &

# ⑥ 验收（每项都打印期望值 vs 实测值）
bash scripts/verify-deployment.sh
```

**先预演不启动**：`FN_DRY_RUN=1 bash /home/ll/deploy/start-flash-next-w4a16.sh` 会打印 inner 将要执行的完整命令。

---

## 3. 目录导航

| 路径 | 内容 |
|---|---|
| `docs/01-host-prep.md` | 主机前置：驱动、CMP 解锁、**PCIe Gen2 早钩子**、内核 sysctl、udev 读预读、功耗墙、swap |
| `docs/02-image-rootfs.md` | 不用 docker，按 **digest 锁定**拉官方定制镜像并解成 rootfs + chroot 挂载 |
| `docs/03-model.md` | 模型获取/校验、量化配置为什么**不能改**、长上下文 YaRN 副本 |
| `docs/04-ple-int8.md` | PLE n-gram 表 INT8 量化：方案、误差、四种驻留形态、内存账 |
| **`docs/05-patches.md`** | **22 个补丁逐条**：文件、落点、根因、上游依据、影响面、是否必需 |
| `docs/06-launch.md` | 启动/停止/看门狗、`FN_*` 参数矩阵、危险参数黑名单、验收清单 |
| `docs/07-performance.md` | 性能画像、与手册对比、测量方法（含两个必踩的坑） |
| **`docs/08-pitfalls.md`** | **踩坑全集**：崩溃类 / 静默失效类 / 显示误导类 / 硬件类 / 运维禁忌 |
| `docs/09-verification.md` | 本仓库的自验证记录（补丁闭环、量化字节闭环、在线验收输出原文） |
| `patches/` | 22 个 `.patch` + `MANIFEST.tsv`（pristine/patched 双哈希） |
| `scripts/` | 全部可执行件（含 `apply-patches.py`、`verify-deployment.sh`） |
| `scripts/host/`、`systemd-units/` | udev 规则、gen2 钩子、功耗脚本、systemd 单元 |
| `tools/` | 生产机实跑快照（cmdline / env / 启动画像 / metrics 摘录） |

---

## 4. 硬约束速查（改错必挂）

| # | 约束 | 依据 |
|---|---|---|
| 1 | `--block-size 1616` **必传** | 两级 CSA + linear 层 block 不一致，漏了建模阶段直接报错 |
| 2 | `--mamba-ssm-cache-dtype float32` **必传** | 同上，sharded cache dtype 不一致 |
| 3 | `num_speculative_tokens ≤ 4`（block 1616 档） | QSA ring=4×cdiv(4+k,4) 必须整除 1616；k=5~8 → ring 12 → 134.67 崩 |
| 4 | `--moe-backend auto`，**不得显式写 marlin** | MTP 草稿的 MoE 层未量化，marlin 不接受 → `moe_backend='marlin' is not supported for unquantized MoE` |
| 5 | **不得改模型 config.json 的 quant_method** | vLLM `inc.py override_quantization_method` 自动把 auto-round→inc；改写会丢 2340 条 fp16 层映射 |
| 6 | 此模型 **不能用 fp8 KV cache** | QSA 模块硬性要求主 KV 为 BF16，`NotImplementedError` |
| 7 | `--max-num-batched-tokens` 锁 **8192** | 16384 实测引擎崩 + 触发 GPU Xid31 |
| 8 | **绝不 SIGKILL 持 CUDA 上下文的进程** | 会诱发 Xid31 → GPU 降级 → 唯一修复是整机重启 |
| 9 | 启新实例前必须确认两卡显存归零 | 孤儿 worker 占显存 → 下次启动 shm_broadcast 超时 |
| 10 | 改镜像内 `.py` 后必须删对应 `__pycache__` | 否则改动不生效（症状：加了日志却一条不出） |

展开见 [`docs/08-pitfalls.md`](docs/08-pitfalls.md)。

---

## 5. 与官方手册的偏离（本仓库的两处自主决策）

唯一权威手册：[`gavinxym/170hx-2-qwen3.8-flash-next`](https://github.com/gavinxym/170hx-2-qwen3.8-flash-next)（只有一份 README，不含补丁源码）。
手册定稿参数本仓库**逐字照抄**，除两处——都是硬件不匹配逼出来的，均有实测依据：

| # | 手册 | 本仓库 | 理由 |
|---|---|---|---|
| 1 | PLE 表 BF16 常驻 CPU RAM（要求 ≥128 GB） | **离线 INT8 + 逐行 scale**（95.4→48.3 GiB），可选驻留内存或 mmap 磁盘 | 全表 `max|x|=0.0447`，动态范围仅约 480×，FP8 的指数位纯浪费。同代价实测 INT8 逐行 `relMSE=4.36e-05`、行余弦 `0.9999784`，比 FP8-e4m3（`6.70e-04`）**精确 15 倍**。decode 与 BF16 持平（87~93 tok/s）。 |
| 2 | MTP=4 | **不变** | 扫描 k=1/2/3/4 = 72.1 / 82.2 / 90.4 / 92.7 tok/s，单调递增。**手册是对的。** |

性能达成度（同协议实测）：预填充 / TTFT = 手册的 **93~101%**；decode = **70~75%**。
差距已定位为**主机平台代差**（本机 E5 v4 + DDR3-1866，STREAM 实测 50.3 GB/s；手册机 ≥128 GB 新平台），
软件侧无对齐空间——证据链与否决项见 [`docs/07-performance.md`](docs/07-performance.md)。

---

## 6. 免责声明

- **不含模型权重**。Qwen3.8-Flash-Next-W4A16-AutoRound 需自行从模型站下载（181 GB，见 `docs/03`）。
- 依赖**官方定制镜像** `vllm/vllm-openai:qwen38-flash-next`（含 `Qwen4ExpForConditionalGeneration` 架构与 PLE offload 框架）。上游若重推 tag 会使补丁哈希失配——本仓库按 digest 锁定并在 `apply-patches.py` 里做了硬校验，失配会明确报错而不是静默打歪。
- CMP 170HX 是矿卡，解锁后 ECC/坏页软件层查不到；本仓库记录了 Xid31/MCE 的判据与处置，但**不保证硬件可靠**。
- 文中所有主机路径、账号、密码均已脱敏（`<SUDO_PASS>` / `<DEPLOY_HOST>`），请按本机替换。

## License

补丁与脚本以 **Apache-2.0** 发布（与 vLLM 上游一致，补丁修改的是 Apache-2.0 代码）。
文档中的分析、实测数据为本仓库原创。详见 [`LICENSE`](LICENSE)。
