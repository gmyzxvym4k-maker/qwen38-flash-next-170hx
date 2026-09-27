# Qwen3.8-Flash-Next（W4A16-AutoRound）在 2× CMP 170HX 上的 vLLM 部署：可复现手册

> 这是一套**已经在生产上跑着的**服务的完整复刻件：推理端点 `http://<部署机>:18420/v1`，
> 模型 `qwen3.8-flash-next`（176B 总参 / 约 6B 激活的 MoE + PLE n-gram 嵌入 + GDN 线性注意力混合架构），
> 跑在两张 64 GB 的 **NVIDIA CMP 170HX**（GA100 die，SM80，矿卡解锁）上，用 **PP2 + MTP4**，
> 上下文 **1M token**，decode 稳态 **90~130 tok/s**，前缀缓存命中率 **89~91%**。
>
> **当前生产 = 官方 vLLM 0.30.0 运行时补丁栈，并已开启「KV 缓存内存二级层」
> （GPU 显存 → 宿主内存 96 GiB，`SimpleCPUOffloadConnector`，实测被挤出显存的前缀 80~97% 由内存档回载）**——
> 复刻入口 [`stack-0300/`](stack-0300/README.md)，权威文档 [`stack-0300/README-0300.md`](stack-0300/README-0300.md) **§11 生产定版**。
> 本仓库主体（`patches/` 22 补丁组 + `docs/01~09`，本文 §2 起）是上一代 chroot 定制镜像栈的复刻件，
> 保留为回滚路线，补丁闭环验证最完整（§2 的图景与命令都是旧栈的）。
>
> 本仓库包含：两代栈的启动/停止/看门狗脚本、运行时补丁（site-packages 零改动）与 **22 个 vLLM 补丁
> （逐条带根因说明）**、**内存二级缓存的验收/挤池/soak 工具链与三轮攻关记录**、PLE 表离线 INT8 量化器、
> 长上下文副本生成器、一键体检脚本，以及 **50+ 条踩坑记录**。
>
> 最后同步生产机状态：**2026-09-27 20:20**（当前生产栈真值快照，读自 `/proc/<APIServer>`，非引用）。

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

## 1. 当前生产形态：官方 vLLM 0.30.0 + 内存二级缓存（2026-09-26 起）

```
客户端 ──OpenAI 兼容 API──► :18420 (host 0.0.0.0)
   │
   └─ 宿主 venv（python3.11，vLLM 0.30.0，site-packages 零改动）
        PYTHONPATH = stack-0300/patches : stack-0300/patches-extra   ← 运行时补丁注入点
        └─ vllm serve /media/ll/data/models-1m/Qwen3.8-Flash-Next-W4A16-AutoRound-1M ...
             ├─ EngineCore
             ├─ Worker_PP0 → GPU0（层 1~26）
             ├─ Worker_PP1 → GPU1（层 27~48 + lm_head + MTP 草稿 + 采样）
             ├─ PleOffloadWorker（CPU 侧 n-gram 查表，BF16 锁页 95.4 GiB）
             └─ SimpleCPUOffloadConnector（KV 二级层：宿主内存 96 GiB 锁页，后台线程 DMA 回载）
```

| 项 | 当前生产值（2026-09-27 20:20 实跑快照） |
|---|---|
| served 模型名 | `qwen3.8-flash-next` |
| 引擎 | **官方 vLLM 0.30.0**（torch 2.13.0+cu130 / flashinfer 0.6.18.post1）+ PYTHONPATH 运行时补丁（上游 8 hook + 本地 14 hook，含 #10 attrIdxs 修复，见 §1.1 末条） |
| 并行 / 投机 | **PP2**（26,22）TP1 / **MTP4**；block 1616；mamba float32；CUDA 图 `FULL_AND_PIECEWISE` |
| 上下文 | **1,048,576**（YaRN×4 副本） |
| **KV 二级缓存** | **SimpleCPUOffloadConnector `--kv-offloading-size 96`（GiB，两 rank 各 48 GB 锁页）**，经 rt-patch #10 修复上游 attrIdxs 越界 UB 后启用；经典 OffloadingConnector（c1~c11）已退役 |
| GPU KV 池 | **1,207,262 token / 776 块**（@1M 并发 1.15×） |
| CPU 档容量 | blocks/rank=[2224,2157] → **≈348.6 万 token = 2.9× GPU 池** |
| PLE 表 | 官方 cpu_offload **BF16 锁页 95.4 GiB**（旧栈 INT8/磁盘档在新栈无实现，见 stack-0300 README §4） |
| 冷启动 | 约 **8 分钟**（就绪判据 `/health` 200） |
| decode | K4 中位 **96.6 tok/s**（历史 K5 档 123.6，档位不同勿混比）；prefill ≈8,300~8,900 tok/s |
| 前缀缓存命中率 | 本地累计 **88.8%**；外部（内存档）回载命中运行 45 分钟即 **143,824 token** |
| 采样缺省（实跑） | `temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0, repetition_penalty 1.0` + `reasoning_effort=xhigh`（09-27 回到 HF 原生；09-21 的 0.6 系可用 `FN_GENCFG` 下发） |
| 自愈 | `fnx-18420-watchdog.{service,timer}`（30s 探活；自愈时**剥离 offload 档位**防崩溃循环，带档回归需手工重放 launch.env） |

逐字快照：[`stack-0300/tools/live-cmdline-0300.txt`](stack-0300/tools/live-cmdline-0300.txt)、
[`stack-0300/tools/live-env-0300.txt`](stack-0300/tools/live-env-0300.txt)。
完整复刻六步：[`stack-0300/README-0300.md` §11.3](stack-0300/README-0300.md)。

### 1.1 KV 缓存内存二级层（本次交付的主角）

- **机制**：GPU 前缀缓存（1.2M token）装不下、但宿主内存档里还在的前缀，由
  `SimpleCPUOffloadConnector` 在 prefill 前经后台线程 DMA 回载进显存，跳过重算。
  CPU 档查找**复用核心** `cpu_coordinator.find_longest_cache_hit`，且显式跳过
  `has_positionally_stable_blocks=False` 的组（mamba/GDN）——只卸载位置稳定的注意力组，
  这是它与我们移植的经典连接器（在 hybrid 模型上因 mamba 活写竞态退役）的本质区别。
- **实测收益**（挤池 >1.2M token 后重发，全由内存档回载）：~100k prompt 命中 **97%**、
  ~40k **93%**、~12k **80%**（零头=每请求末块 ≤1616 不入库）；内嵌验证码精准复述=内容无损。
- **代价**：96 GiB 物理常驻（机器 251 GB、有 MCE 硬挂史）。复发卡死/硬挂第一刀降 `FN_SIMPLE_OFFLOAD=48`
  （命中率实测持平，只减存活时长），第二刀去掉该变量回无二级缓存定版。
- **适用判断**：只对「前缀被挤出 GPU 池后又被重发」的负载有收益；若流量重复度低（历史 21.5h
  观测 GPU 池自扛 ~90%、外部命中 0），不开它才是正解——开关是一等 `FN_SIMPLE_OFFLOAD=<GiB>`，
  按自己流量画像决定。
- **踩坑全录**（c1~c11 三轮攻关：假 0 命中、pending 悬挂、容量口径勘误、mamba 竞态、拷贝 API 冻结流…）：
  [`stack-0300/README-0300.md` §6.1~6.3](stack-0300/README-0300.md) + ROUND1/2/3 文档。
- **segfault 根因定案与修复（rt-patch #10，2026-09-27 深夜）**：带二级缓存实例当日两次原生段错误
  （17:38 / 20:38，均 Worker_PP1 猝死、栈仅 glibc pthread 帧、无 Xid 前导）——根因是上游
  `vllm/v1/simple_kv_offload/cuda_mem_ops.py::copy_blocks` 把**单个 `c_size_t` 标量**当 `attrIdxs`
  数组传给 `cuMemcpyBatchAsync`，驱动按契约读 `count` 个元素 ⇒ 标量之后全是堆上随机字节，
  非零值即索引 `attrs[垃圾]` ⇒ 间歇性越界/段错误（上游 issue #53860 同判，本栈代码逐字命中；
  此前经典连接器的 `error 1` 崩溃亦属同族 API 面）。修复=每次调用传 `np.zeros(cnt, uint64)`
  零索引数组（语义不变，逐描述符仍用 attrs[0]），以 `patches-extra/dsh_simple_offload_rt.py`
  运行时钩子落地，**功能（线程/流/事件排程/拷贝路径）零改动**；附带 src/dst 块数与负 id 入参加固。
  回退开关 `DSH_SIMPLE_OFFLOAD_UPSTREAM=1`；自检 `selftest_simple_rt.py`。详见 docs/08 P59。

### 1.2 两代栈的关系

chroot 旧栈（本文 §2 起）仍是**验证最完整的复刻件**：官方定制镜像 digest 锁定 + 22 补丁
逐条双哈希闭环。新栈在同一端口替换生产后，旧栈通过哨兵文件
`vllm-0300/DISABLED` 一键回滚（看门狗与控制台都识别）。要理解当前生产为什么长成这样，
旧栈文档里的主机前置（docs/01）、模型获取与量化（docs/03/04）、长上下文副本（`make-longctx-copy.py`）
**全部照常适用**。

---

## 2. 旧 chroot 栈最终形态一览（回滚路线）

> 以下本节与 §3 的目录/命令描述的是**上一代 chroot 定制镜像栈**（vLLM v0.1.dev20073）。
> 当前生产请以 §1 为准。

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
| KV 二级缓存 | （旧栈实验形态）OffloadingConnector `FN_KVOFF=1` —— **已退役**，生产用 §1 的 Simple 方案 |
| GPU KV 池 | **1,224,366 token**（18.16 GiB，@1M 上下文并发 1.17×） |
| 显存 | 39.6 + 40.3 GiB / 卡（`gpu-memory-utilization 0.95`） |
| 冷启动 | **约 4 分钟**（权重 78 s + PLE 挂载 + init engine 61 s） |
| decode | 90~130 tok/s（MTP 接受长度 3.95） |
| 预填充 | 6,900~8,700 tok/s |
| 前缀缓存命中率 | 累计 **91%** |
| 采样缺省 | `temperature 0.6, top_p 0.95, top_k 20, min_p 0, presence_penalty 0.1, repetition_penalty 1.05` |

完整 argv 与环境变量快照（旧栈 2026-09-23）：[`tools/live-cmdline.txt`](tools/live-cmdline.txt)、[`tools/live-env.txt`](tools/live-env.txt)。

---

## 3. 旧 chroot 栈最短复现路径

前提：一台已装好 NVIDIA 驱动、已解锁显存、rootfs 已解出的机器。若从零开始，按顺序读
[`docs/01`](docs/01-host-prep.md) → [`docs/02`](docs/02-image-rootfs.md) → [`docs/03`](docs/03-model.md)。

```bash
git clone https://github.com/gmyzxvym4k-maker/qwen38-flash-next-170hx.git flashnext && cd flashnext

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

## 4. 目录导航

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
| `tools/` | 旧栈生产快照（cmdline / env / 启动画像 / metrics 摘录，2026-09-23） |
| **`stack-0300/`** | **当前生产栈全套**：运行时补丁、启动/停止/看门狗、**内存二级缓存（KVCache）实现与验收工具链**、生产真值快照、README-0300.md（§11=生产定版） |

---

## 5. 硬约束速查（改错必挂）

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
| 11 | 内存二级缓存只用 `FN_SIMPLE_OFFLOAD=<GiB>`，**与 `FN_KVOFF=1` 互斥** | 同开 inner 直接拒启；经典连接器在 hybrid 模型上回载内容不等价（§1.1） |
| 12 | 缓存命中实验两侧 `chat_template_kwargs` 必须完全一致 | 块哈希含模板前缀，`enable_thinking` 一边开一边关 ⇒ 永远 0 命中（stack-0300 ROUND2 实锤） |
| 13 | 看门狗自愈后需手工带回 offload 档 | 自愈安全模式故意剥离档位（防「崩溃→重启→再崩」循环），见 `fnx-watchdog.log` 的 TIER_DROPPED 行 |

展开见 [`docs/08-pitfalls.md`](docs/08-pitfalls.md)。

---

## 6. 与官方手册的偏离（本仓库的两处自主决策）

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

## 7. 免责声明

- **不含模型权重**。Qwen3.8-Flash-Next-W4A16-AutoRound 需自行从模型站下载（181 GB，见 `docs/03`）。
- 依赖**官方定制镜像** `vllm/vllm-openai:qwen38-flash-next`（含 `Qwen4ExpForConditionalGeneration` 架构与 PLE offload 框架）。上游若重推 tag 会使补丁哈希失配——本仓库按 digest 锁定并在 `apply-patches.py` 里做了硬校验，失配会明确报错而不是静默打歪。
- CMP 170HX 是矿卡，解锁后 ECC/坏页软件层查不到；本仓库记录了 Xid31/MCE 的判据与处置，但**不保证硬件可靠**。
- 文中所有主机路径、账号、密码均已脱敏（`<SUDO_PASS>` / `<DEPLOY_HOST>`），请按本机替换。

## License

补丁与脚本以 **Apache-2.0** 发布（与 vLLM 上游一致，补丁修改的是 Apache-2.0 代码）。
文档中的分析、实测数据为本仓库原创。详见 [`LICENSE`](LICENSE)。
