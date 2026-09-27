# 18420 Qwen3.8-Flash-Next @ 官方 vLLM 0.30.0（<DEPLOY_HOST>）

迁移日期：2026-09-26（北京时间 21:10 起服务）。本文是这套栈的唯一操作说明，
旧 chroot 定制镜像（vLLM v0.1.dev20073+g8e685d198）的文档见
`/home/ll/deploy/OPTIMAL-CONFIG-FLASH-NEXT-W4A16.md`。

## 1. 栈形态

| 项 | 旧（已停，保留作回滚） | 新（生产中） |
|---|---|---|
| 运行位置 | chroot `/media/ll/data/vllm-image/rootfs`（py3.12 容器镜像） | 宿主 venv `/home/ll/vllm-env`（py3.11.16） |
| vLLM | 0.1.dev20073+g8e685d198（定制镜像，无上游版本语义） | **0.30.0**（PyPI 官方 wheel） |
| torch / flashinfer | 镜像内置 | 2.13.0+cu130 / 0.6.18.post1 |
| 补丁方式 | 直接改 site-packages（22 文件，重装即丢） | **PYTHONPATH 运行时注入，site-packages 零改动** |
| PLE n-gram 表 | INT8 47.7 GiB（匿名堆）或 BF16 95.4 GiB | **BF16 95.4 GiB 锁页（cuMemHostRegister）** |
| 上下文 | 1M（YaRN×4 副本） | 1M（同一 YaRN×4 副本，`--max-model-len 1048576`） |
| 投机 | MTP K=4/5 | MTP K=4（`--speculative-config`，与旧生产一致） |
| 权限 | root（chroot 内） | **root（宿主，必须，见 §5）** |

并行/关键 argv 与旧栈逐项一致：PP2（`VLLM_PP_LAYER_PARTITION=26,22`）、TP1、
`--block-size 1616`、`--mamba-ssm-cache-dtype float32`、`--max-num-seqs 4`、
`--max-num-batched-tokens 8192`、`--gpu-memory-utilization 0.95`、
`--moe-backend auto`、`FULL_AND_PIECEWISE` + capture sizes `[1,2,4,8,16,24,32,40]`、
`--async-scheduling`、`--enable-prefix-caching`、`--enable-prompt-tokens-details`、
BF16 KV（QSA 强制，不可 fp8）。

## 2. 文件与启停

```
/home/ll/deploy/vllm-0300/
├── start-flash-next-0300.sh      # 宿主入口：落盘 FN_* → 显存归零门禁 → sudo 拉起
├── stop-flash-next-0300.sh       # 停止：SIGTERM 90s → SIGKILL 兜底 → 等显存归零
├── bin/flash-next-0300-inner.sh  # 实际 vllm 命令构造（含 preflight）
├── patches/sitecustomize.py      # 上游 8 处补丁（正源 CyrilCN/qwen38-flash-170hx-patches）
├── patches-extra/sitecustomize.py# 本地 rt-patch #8（跳过多模态 warmup）
├── launch.env                    # 最近一次启动的 FN_*（排障用，勿手改）
├── rollback-old-stack.env        # 旧 chroot 栈的生产参数快照（09-26 18:42）
├── rollback-old-cmdline.txt      # 旧栈实跑 argv（59 token）
├── selftest_extra.py             # 本地补丁 + 启动参数装配自检
├── redirect-console-watchdog-0300.py  # 控制台/看门狗重定向（幂等，--revert 可退）
└── plan_argv_check.py            # 控制台按钮 → argv 与生产 cmdline 逐 token 对账（离线）
```

inner 脚本除了构造 argv，还兜三件与栈无关的正确性（09-26 补）：

1. **长上下文自动配档**：控制台选 1M/512K 档时下发 `FN_MODEL_PATH=原生目录` +
   `FN_1M_MODEL_PATH=YaRN 副本` + `FN_MAXLEN=1048576`，只认 `FN_MODEL_PATH` 就会拿
   未缩放的 262144 模型跑 1M（`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` 会放行启动，但位置越过
   original_max 后 NaN/越界）。现在按「请求长度 ↔ 各副本 cap」**取最小够用档**，
   显式 hint 优先；找不到够档副本就钳回原生 cap 并告警。三档实测 cap：原生 262144(rope=default)、
   `-512K` 524288(yarn×2)、`-1M` 1048576(yarn×4)。
2. **MTP 档位整除门禁**：`ring = 4×cdiv(4+K,4)` 必须整除 `--block-size`，不整除直接拒启并说明
   （block 1616 → K=1..4/9..12；block 1680 → K=5 合法）。以前是加载完 17 分片才 AssertionError，
   浪费 4 分钟。
3. **FN_* 体检**：穷举比对「本脚本消费的变量」与「实际收到的变量」，凡是弹窗里能填、
   脚本不吃的项一律出声（`[FN-0300] 忽略 FN_XXX=… ：原因`）。根治旧栈 FN_PLE_INT8 /
   FN_GENCFG 那类「配置长期静默失效」——新增字段忘了接也会当场暴露。

stop 脚本按**端口**圈定目标（主进程 + 后代树 + 带本栈指纹的孤儿），不再按 `vllm serve` 广匹配；
否则控制台点停止会连带杀掉本机其它 vLLM 实例。`STOP_LIST_ONLY=1` 可只列目标不动手。

启动（默认 MTP4 / 1M / INT8 关闭，与生产一致）：

```bash
bash /home/ll/deploy/vllm-0300/start-flash-next-0300.sh
tail -f /home/ll/deploy/vllm-flash-next-0300.log      # 就绪约 8 分钟
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18420/health   # 期望 200
```

可覆盖项（全部走 `FN_*`，语义同旧栈）：`FN_SPEC=none|mtp<N>|<raw json>`、
`FN_MAXLEN`、`FN_SEQS`、`FN_PP_PARTITION`、`FN_EAGER=1`、`FN_EXTRA_ARGS`、
`FN_EXTRA_ENV`、`FN_FORCE=1`（跳过显存归零门禁，慎用）、`FN_DRY_RUN=1`（只打印 argv）。

停止：`bash /home/ll/deploy/vllm-0300/stop-flash-next-0300.sh`
（绝不 `pkill -9`：强杀持 CUDA context 的进程会诱发 Xid31，唯一修复是重启整机。）

## 3. 为什么放弃了 INT8 / 磁盘驻留 PLE（inner 脚本注释所指）

旧 chroot 镜像里有自研的三套 PLE 装载路径：BF16 mmap（`VLLM_PLE_MMAP`）、
INT8 磁盘驻留（`VLLM_PLE_INT8_DIR` + `VLLM_PLE_DISK_RESIDENT`）、INT8 匿名堆
（`VLLM_PLE_INT8_MEMORY`）。**官方 0.30.0 里这些开关一个都没有读取方**——
上游只有 `EngramConfig`（n-gram 表上游改名 Engram），且只提供两种形态：

- `cpu_offload=1` → `Qwen4ExpPLEPinnedHostEmbedding`，BF16 全表 **锁页 95.4 GiB**；
- `cpu_offload=0` → 表进显存（本机 2×64 GB 装不下，不可用）。

INT8 压缩是本地补丁的产物，不属于上游 8 处补丁的范围，迁移时**没有移植**
（移植=在运行时注入里重做一个 CPU 侧反量化加载器，风险与收益不成比例）。
后果与代价：

1. 内存常驻从 47.7 GiB（INT8 heap）涨到 **95.4 GiB 锁页**，`free` 的 used 从
   ~60 GB 涨到 ~107 GB；锁页内存不可换出，是硬占用。
2. 锁页 95.4 GiB 必须 **memlock 不限**，只有 root 拿得到（见 §5）。
3. 启动时表要一次性读进来：日志
   `[rt-patch] PLE pinned alloc: 95.368 GiB registered in 2 chunk(s) via cuMemHostRegister in 23.2s`
   （单块 >64 GiB 驱动不支持，补丁按 60 GiB 分块注册，这是补丁存在的理由）。
4. 好处：查表不再依赖页缓存驻留，旧栈「表自己吃自己的缓存」「disk 档与大 pinned
   层共存被页缓存挤压缺页」两类问题在这条路径上不存在。

选择器无需命令行参数：`config/vllm.py:_resolve_and_verify_engram_config` 在
CUDA + 模型含 engram 层（本 checkpoint `ple_layer_ids=[2]`）时自动构造
`EngramConfig()`，其 `cpu_offload` 取自 `envs.VLLM_PLE_CPU_OFFLOAD`（缺省 1），
inner 脚本显式 export 该变量为 1，`ngram_embedding.py:702-706` 据此选锁页实现。

## 4. 采样默认值与旧生产的偏离（inner 脚本注释所指）

`--override-generation-config` 取 **09-21 定版**：

```json
{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.1,"repetition_penalty":1.05}
```

旧 chroot 栈实跑（09-26 快照 `rollback-old-stack.env`）的 `FN_GENCFG` 是
`temperature:1 / min_p:0 / repetition_penalty:1`——那是历史上无人记录的漂移，
不是有意决策。本次迁移按 09-21「治循环复读」的定版取值，**这是一处刻意的行为变化**：
同一 prompt 的输出会与旧栈不同（温度 0.6 vs 1.0）。基准测试不受影响，
因为 `fnx-bench.py` 显式钉 `temperature=0`。

要回到旧栈那组值：`FN_GENCFG='{"temperature":1,...}' bash start-flash-next-0300.sh`。

> **09-27 更新（另一会话的用户定档，本仓库如实记录）**：采样基准改为
> `t1.0 / p0.95 / k20 / minp0 / pp0 / rp1.0`，同步落在 `server.js` 的
> `SCRIPT_MODELS[..].base` 与两套栈的 inner `GENCFG_DEFAULT`。
> **但三个快启预设仍显式带 `temperature:0.6`** ⇒ 从预设启动时 plan 会下发
> `FN_GENCFG`，实跑值还是 0.6（当前 18420 的 `/proc/<pid>/cmdline` 实测即 0.6）。
> 这构成"弹窗/基准显示 ≠ 预设启动后的引擎真值"的三源不一致，与本项目一贯的
> 三源同步铁律相违；**未擅自改动**（改预设=改生产行为），待用户拍板：
> 要么把预设的采样字段删掉（跟随 base），要么把 base 改回 0.6。

另外两处刻意偏离：

- `VLLM_USE_FLASHINFER_SAMPLER=0`：宿主 venv 无 ninja/nvcc，FlashInfer 采样器 JIT
  编不出来；关掉走 torch 原生采样，统计等价。
- `-cc.splitting_ops` 不再显式传：官方 `config/compilation.py:772` 的 `_attention_ops`
  已内置 `vllm::qwen4_exp_ple_short_conv` / `vllm::qwen4_exp_qsa_with_output`，
  再传一份反而可能与上游清单打架。
- `--kv-transfer-config` 不传：CPU KV 二级缓存（OffloadingConnector）0929 已定案退役
  （21 h 生产观测 `external_prefix_cache_hits_total=0`，却常驻 107 GB pinned，
  且 PP2 抢占路径有 `cuMemcpyBatchAsync error 1` 崩溃史）。

## 5. 为什么必须 root

锁页 95.4 GiB 走 `cuMemHostRegister`，需要 memlock 不限。`ll` 账号被
`DefaultLimitMEMLOCK=65536` 钉在 64 MiB（`user@1000.service LimitMEMLOCK=65536`），
而 `/etc/security/limits.d/99-vllm-memlock.conf` 在 systemd 下**根本不生效**；
抬高它要重启 `user@1000.service`——那个 cgroup 里住着 dsh-console（8889 与
生产实例的父进程），禁止重启。因此入口脚本用
`echo <pw> | sudo -S -p '' sh -c "exec setsid bash ... </dev/null"` 以 root 起服务。

两个必须知道的坑：
1. **脚本里每条 sudo 都要自带密码管道**。`sudo -S true` 建立的时间戳在同一 ssh
   会话里可能对后续 `sudo -n` 失效（21:10 首次启动就因此失败过一次）。
2. **绝不要在 `sudo -S` 那行后面接 `< /dev/null`**——它会覆盖密码管道，sudo 变成
   交互式要密码然后超时（本项目第三次踩）。脱离 tty 的动作要写进 `sh -c` 里面。

## 6. 运行时补丁映射（旧 22 补丁 → 新 9 处）

上游 `patches/sitecustomize.py`（sha256 `9b84700f…29b7`，与
CyrilCN/qwen38-flash-170hx-patches 逐字节一致）注册 8 个 hook，生效时打印
`[rt-patch] patching <module>`：

| 上游 hook | 覆盖的旧补丁 | 作用 |
|---|---|---|
| `vllm.config.vllm`（models.config） | 08 | PP>1 + PLE 的建模期硬拒 |
| `vllm.models.qwen4_exp.{nvidia,amd}.model_state` | 08/07 | 段缺失层与 PP 禁令 |
| `vllm.model_executor.layers.nvidia.ngram_embedding` | 19/10 | 锁页分块注册（>64 GiB 驱动限制） |
| `vllm.model_executor.model_loader.weight_utils` | — | INC PLE 嵌入按未量化接受（bits=16） |
| `vllm.models.qwen4_exp.{nvidia,amd}.mtp` | 09/01 | 草稿只建在末 PP rank 的 assert |
| `vllm.v1.core.kv_cache_utils` | — | 空占位 KV 组 |

本地新增 **#8**（`patches-extra/sitecustomize.py`）＝旧补丁 11：
`VLLM_SKIP_MM_WARMUP=1` 时让 `BaseRenderer._warmup_mm_processor` 早退，省 ~25 s。
链式加载是安全的：上游 `_chain_stock_sitecustomize()` 在装它自己的 `_PatchFinder`
之前遍历 `sys.path`、跳过自身目录、只加载**第一个**其它 `sitecustomize.py` 后 `break`，
所以 `patches-extra` 放在 PYTHONPATH 的**后面**正是设计好的扩展点；
extra 侧再用 `_PATCHES_ROOT` 排除整棵补丁树，避免回链成无限递归。
幂等哨兵必须是 `sys._dsh_rt_extra_installed`（**进程级**）——早先用 env 变量，
被 mp 子进程继承导致「哨兵命中，跳过重复安装」×4，子进程会静默丢掉未来的 worker 侧 hook。

**未移植（已判定过时或被替代）**：05 `distributed/utils` 切分回退（上游已回退）、
06 `envs` INT8 开关、18/19 PLE 磁盘驻留加载器（§3）、20/21/22 worker/PP 中继
（上游 0.30.0 已修或形态改变）。KVOFF 系列 02/03/04/13/14/15/16/23/24/25 曾随
0929 退役不迁；**2026-10-05 经本地新增 #9 恢复**（见 §6.1），其中 03/13/15
（metrics 断言降级与 fill 观测指标）与 c5a（eagle 双罚，上游 from_spec 已原生
「无标注则全部按 non-draft」）确认不再需要。
**待办**：12/17（8889 控制台逐请求实时预填充速度行）需要给 `SchedulerStats`
做事后 `setattr` 并包裹 `schedule()` 返回 `prefill_progress`，列为第二阶段。

## 6.1 KV 二级缓存恢复（rt-patch #9，2026-10-05）

`patches-extra/dsh_kvoff_rt.py` 把旧栈验证过的 c1/c2/c6/c7 移植到 0.30.0，
经 `patches-extra/sitecustomize.py` 挂进同一套导入钩子。**不配
--kv-transfer-config 时相关模块不被导入、钩子不触发**——缺省（FN_KVOFF=0）
对生产零影响；紧急总闸 `DSH_KVOFF_RT_DISABLE=1` 可让全部回调变 no-op。

| 组 | 落点（0.30.0） | 语义 |
|---|---|---|
| c1 | `offloading.config.get_offloading_group_ids` | **只**剔除 `prefix_cacheable=False` 的分组（QSA 环形 CircularBufferSpec，ring 8 ∤ 1616）；**空 `layer_names` 分组必须保留**——PP>1 时各 rank 只带自己的层名，空组是合法占位，剔除会让 worker 的组数与 scheduler 不一致（2026-09-27 实测修正） |
| c2 | `cpu.spec.CPUOffloadingSpec._uses_shared_region` | pp_size>1 → 每 rank 私有缓冲（**必要**：共享区行布局的 slot 起点 = rank × cpu_page_size，而该值是 per-rank 量；PP 各 rank 层子集不同 ⇒ 行 stride 分歧 + total_size 取整各异（09-18 实测 34.32 vs 34.3566 GB）→ joiner 30s 超时）。**该私有路径是真 pinned**（`pin_memory=PIN_MEMORY`，本栈实测 True；09-22 smaps 实测 /dev/zero = cudaHostAlloc 特征），09-27 曾误判为"无 pin 退化路径"，09-30 已推翻 |
| c6 | `cpu.gpu_worker`（handler.wait 有界+熔断、worker.wait 回传未完成集）、`offloading.common`（meta 增 failed_jobs/fuse_tripped + aggregate 合并）、`offloading.worker`（提交 try/except、失败 ack、迟到完成去重、熔断后只读降级）、`offloading.scheduler`（update_connector_output 完整替换：失败 job 走 `complete_store(success=False)` 撤登记+释放块；fuse 后两个 store 构建函数短路返回 {}） | 拆掉「worker 主线程无限 event 等待 × PP2 NCCL 中继」锁死环（09-24 七次卡死根因）；store 失败只丢缓存条目，load 正确性零妥协、永不熔断 |
| c7 | `cpu.swap_blocks_triton.MIN_N=0` + `cpu.gpu_worker._select_swap_blocks_fn` **仅 load 方向**强制 Triton | load(CPU→GPU) 绕开 cuMemcpyBatchAsync（与 PP2 NCCL-P2P 并发会冻结 compute 流）；**store(GPU→CPU) 一律保持上游 C++ DMA**——上游 `gpu_worker.py:41-43` 显式禁止该方向用 Triton（"GPU->CPU is bandwidth-bound; the dedicated copy engine beats Triton"），强开会让 SM 内核解引用 host 指针 → MMU Fault VIRT_WRITE → Xid31（2026-09-27 实测） |

启用方式（任一）：
- 手动：`FN_KVOFF=1 [FN_KVOFF_BYTES=<字节>] bash start-flash-next-0300.sh`（容量缺省 64 GiB）
- 控制台：启动弹窗「CPU KV 二级缓存」=开 + 容量 GiB（plan 下发 FN_KVOFF/FN_KVOFF_BYTES，链路现成）
- 等待上限：`FN_KVOFF_WAIT_TIMEOUT`（秒，缺省 15）

沿用旧栈铁律：物理钉住 ≈1.56×配置（PP2 私有 pinned）；容量必须 > GPU 池
（≈122 万 tok，store 侧 ≈40.4 KB/token）才有回载收益；本机历史结论是
「GPU 池自扛 ~90% 命中、21h 零外部回载」——开之前想清楚负载形态。

自检：`PYTHONPATH=$PWD/patches:$PWD/patches-extra /home/ll/vllm-env/bin/python selftest_kvoff_rt.py`
（24 项判据，全绿 PASS；不占 GPU、不动服务）。argv 装配四态已验证：
缺省/FN_KVOFF=0 不带 --kv-transfer-config 且体检无告警；=1 正确注入 JSON。
### 2026-09-27 实机验证结论：c1 修复有效，但 KVOFF 在本栈仍不可用 → 保持 FN_KVOFF=0

验证动作：停看门狗 → 停实例（优雅退出、显存归零）→ 装 c1 修复 → `FN_KVOFF=1`
（96 GiB）按 launch.env 原样重启 → 探针 4×387k token 文档 + 重发第 1 篇。

- ✅ **c1 修复被证实**：启动日志三侧分组一致
  `c1: kv_cache_groups=6 -> offload 分组 [0, 1, 2, 3, 5]`（scheduler 侧 group 4 是
  `CircularBufferSpec`，worker 侧同一组以 `UniformTypeKVCacheSpecs` 包装器出现，
  `prefix_cacheable=False` 判定一致），**零 AssertionError、零熔断**；
  store 真实跑起来：2 篇文档累积 `GPU_to_CPU` 21.4 GB、CPU 档填充至 10.3%。
  修复前是首个 store job 即断言失败 → `fuse tripped` 只读降级（09-27 08:07 实录）。
- ❌ **但第 3 篇文档时 Worker_PP1 崩**：`cudaErrorIllegalAddress` → `EngineDeadError`；
  dmesg 实锤 **Xid 31 MMU Fault VIRT_WRITE**（PCI 04:00 = GPU1，pid = Worker_PP1，
  报错栈落在 GDN attention 的 `a.contiguous()`，属异步上报）。dmesg 现场：
  `MMU Fault: ENGINE GRAPHICS GPC3 GPCCLIENT_T1_4 faulted @ 0x7fb4_0c000000,
  Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_WRITE`。
- ⚠️ **根因定性已于 2026-09-30 复核推翻并纠正**（原文写的是「c2 把 CUDA 生产路径推到
  非 CUDA 的无 pin 退化 tensor 路径 → host 缓冲不可被 device 访问」，这是错的）：
  - 私有路径**本来就是 pinned**：`cpu/gpu_worker.py` 该分支为
    `torch.zeros((num_chunks, cpu_page_size_bytes), dtype=torch.int8, pin_memory=PIN_MEMORY)`，
    而 `PIN_MEMORY = is_pin_memory_available()` 在宿主 venv 实测 = **True**；09-22 曾用
    smaps 实测该路径为 `/dev/zero` 映射（cudaHostAlloc 特征），是独立佐证。
    ⇒ **不存在「host 缓冲不可被 device 访问」这个前提**，c2 不是 store 崩溃的根因。
  - 「统一尺寸后恢复共享区」**不是**修法：共享区行布局为 `|--- W0-C0---|--- W1-C0---| ... |`，
    其 slot 起点 = `rank × cpu_page_size`，而
    `cpu_page_size_per_worker = worker_kv_bytes_per_block × blocks_per_chunk` 是 **per-rank**
    量 —— TP 下各 rank 层相同故协议成立，PP>1 各 rank 层子集不同则行 stride 与 slot 划分
    **整体分歧**（旧栈 c2 补丁注释当年就已写明此点），故 PP>1 必须回退私有缓冲。
  - **真正的形态是「版本相关」**（尚未定论）：旧 chroot 栈（vLLM 0.1.dev20073 /
    torch 2.13.0+cu130）在**同一条 c2 私有 pinned 路径**上 store 累计 **99.3 GB、零 Xid**
    （09-18 P6 深测；且实例能启动本身就证明 c2 生效——否则共享区在 PP>1 必然 30s 超时）；
    而 0.30.0 同路径、同 CUDA 崩：C++ DMA `cuMemcpyBatchAsync` → error 1
    （CUDA_ERROR_INVALID_VALUE，09-23）、Triton SM 内核 → MMU Fault VIRT_WRITE → Xid31
    （09-27）。两栈的 `_custom_ops.swap_blocks_batch` **逐字相同**、torch/CUDA 版本相同、
    旧栈镜像同样基于 cu130 ⇒ 差异收敛在**各自编译的 C++ kernel**
    （`csrc/cache_kernels.cu` 的 `swap_blocks_batch`）或其调用参数，**尚未定论**。
  - 结论不变、理由更正：**保持 FN_KVOFF=0**。不仅因为 store 方向在本栈不可用，更因为
    即便修好，本机负载下收益也为 0（0929 实测 21.5 h 零外部回载、GPU 池自扛 ~90% 命中，
    代价却是 107 GB pinned 与该机 MCE 硬挂风险敞口）。若要继续追根因，入口是**对比两版
    上游 `csrc/cache_kernels.cu` 的 `swap_blocks_batch`**，而不是继续改 c2/c7。
- **本次处置**：① c1 修复保留；② c7 收窄为仅 load 方向（store 保持上游 C++ DMA，
  消除强开 Triton 带来的 Xid31 危险）；③ c2 保留（PP>1 的必要回退，非引入缺陷），
  启动时只打说明性日志；④ 生产回滚 `FN_KVOFF=0`（launch.env 已写回），重启 200s 内
  恢复 health 200、看门狗 timer 已恢复。

## 6.2 公共区（c8）：PP2 下 CPU KV 二级缓存的物理账对得上（2026-10-06 实现）

### 动机：不是优化，是"配置值 ≠ 物理值"这个隐性风险

c2 的每 rank 私有 pinned 缓冲，各 rank 都按**自己的**每块字节去铺满 `cpu_bytes_to_use`：

| 口径 | rank0（26 层） | rank1（22 层） | 合计物理 | manager 可用档数 |
|---|---|---|---|---|
| 配置 64 GiB | 64 GiB | ≈54 GiB | **≈107 GiB**（09-22/09-24 smaps 实测 1.56×） | 64 GiB ÷ rank0 每块 |

这台机器 09-22 起有"大内存操作整机硬断电"的病史，**配 64 GiB 实际钉 107 GiB** 是不可接受的
——容量规划做不了，看门狗/仪表盘显示的还都是配置值。

### 做法：把"slot 起点 = rank × cpu_page_size"换成前缀和

上游共享区（`SharedOffloadRegion`）的行布局是 `|W0-C0|W1-C0|...|`，slot 起点写成
`rank × cpu_page_size_per_worker`，而后者是 **per-rank 量**：TP 下各 rank 层相同才成立，
PP>1 各 rank 层子集不同 ⇒ 行 stride 与总尺寸分歧 ⇒ 创建者 ftruncate 自己的字节数、
joiner 等自己的字节数 → 30 s 超时（09-18 实锤，也是当年判定"统一尺寸也救不了共享区"的依据）。

c8 只改这一处算术，其余（O_EXCL 创建、ftruncate、barrier 后 unlink、整区 cudaHostRegister、
`(row_stride, 1)` 跨步视图）全部沿用上游：

```
slot_i     = round_up(rank_i 自己的每块字节, 4096)
offset_i   = Σ_{j<i} slot_j            ← 前缀和，天然两两不相交
row_stride = Σ_i slot_i                ← 全体一致
num_chunks = cpu_bytes_to_use // row_stride
```

⇒ **物理钉住 = 配置值**（公共区是 `/dev/shm` 上的 tmpfs 文件，`df --output=used /dev/shm`
可直接核对），且不再有"每 rank 各铺一份"的浪费；每个 rank 只在自己 `[offset_i, offset_i+slot_i)`
内读写，跨步视图与上游 `compute_sub_block_ptrs` 的地址算术完全兼容（自检里有逐指针断言）。

### 协商：无中心、可判陈旧、失败即降级

worker 在 `/dev/shm/vllm_kvoff_slot.<engine_id>.r<rank>.json` 发布自己的每块字节，收齐
`world_size` 份后各自算出同一套布局，再发布第二次带 `decision`+`rows`。**调度器侧**
（EngineCore 进程，构造时机在 `initialize_from_config` 之后，见 `v1/engine/core.py:162/354`）
只读这些文件并采纳 `num_chunks = min(各 rank 公布的 rows)`。

- 取 min 是安全性的全部来源：manager 发出的 chunk id 必须落在**每一个** rank 的 CPU 缓冲行数
  以内，越界就是 device-side assert / Xid31。顺带堵住上游一个潜在越界——当后置 PP rank 的
  每块字节 **大于** rank0 时，上游按 rank0 算的 `num_chunks` 会超过该 rank 的实际行数
  （自检有这一项：`c8 rank1 块更大 -> 采纳下界`）。
- 陈旧文件判据用 `/proc/<pid>/stat` 的 **starttime**（进程存活 + 与本进程启动时刻相差 ≤
  `FN_KVOFF_LAYOUT_WINDOW` 秒，缺省 900），**不用墙上时间**——本机 RTC 会跳到 2161 年（09-18 定案）。
- 收不齐 / `/dev/shm` 放不下 / 本 rank 实际需求超过自己那格 ⇒ 该 rank 退回 c2 私有缓冲并
  如实公布；调度器收不齐时把 CPU 档行数置 **0**（`prepare_store` 恒返回 None = 不做 offload），
  引擎照常服务，绝不带着错几何跑。

### 适用范围与旋钮

自动生效条件：`pp_size>1` 且 `tp_size==1` 且 非 replicated(MLA) 且 非 canonical 布局 且 单节点 mp。
不满足即静默退回 c2（行为与加补丁前逐字一致）。

| 旋钮 | 缺省 | 说明 |
|---|---|---|
| `FN_KVOFF_SHARED` | 1 | 0=强制走 c2 私有缓冲（对照/应急） |
| `FN_KVOFF_LAYOUT_TIMEOUT` | 120 s | 协商等待上限 |
| `FN_KVOFF_LAYOUT_WINDOW` | 900 s | 同一次启动窗口的 starttime 容差 |
| `FN_KVOFF_WAIT_TIMEOUT` | 15 s | c6 有界等待（沿用） |

容量口径（本机 18420，1M 档）：GPU 池 1,207,262 tok、全局 ≈32.2 KB/token（每 rank 18.06 GiB×2）。
⇒ 48 GiB ≈ 160 万 tok（**1.33× GPU 池**，够回载实验）；64 GiB ≈ 213 万 tok。
内存账：PLE BF16 锁页 95.4 GiB + 公共区 + 引擎 ≈15 GiB + 权重页缓存 79 GiB（可回收）≤ 251 GiB
⇒ 首窗口建议 48 GiB，稳了再上 64 GiB。

### 判据日志行（启动后 grep 实例日志）

```
[rt-patch-kvoff] c8[worker rank0 pid...]: 公共区已协商 —— row_stride=... 本 rank 区段 [0, ...) num_chunks=... ⇒ 物理钉住 48.00 GiB（配置 48.00 GiB，不再 ×world_size）
[rt-patch-kvoff] c8[worker rank1 pid...]: ... 本 rank 区段 [50331648, ...) ...
[rt-patch-kvoff] c8[调度器侧]: decision=['shared'] 各 rank 行数=[...] ⇒ 采纳 num_chunks=...
[rt-patch-kvoff] c1: kv_cache_groups=6 -> offload 分组 [0, 1, 2, 3, 5]   # 三侧必须一致
```

### 验证状态

- **离线自检 67 项全绿**（`selftest_kvoff_rt.py`，不占显存、不动服务）：布局数学、前缀和
  区段不相交、双 rank 打开同一 region 的字节可见性、喂给上游 `compute_sub_block_ptrs` 的
  逐指针断言、调度器/worker 口径一致、四类门控退回私有、死进程陈旧文件被忽略、
  `/dev/shm` 不足退回私有、混合决策取 min、区段溢出被断言拦住。
- **实机窗口 1（2026-09-27 09:48–10:11，48 GiB）已跑**：12 项结构判据**全 PASS**，
  探针三项判据 FAIL——但那是**探针自己的参数 bug**，不是 c8 缺陷（见下）。

| 判据 | 结果 |
|---|---|
| worker/调度器两侧公共区协商 | PASS（row_stride=170.14 MB，slots=90.0+80.2 MB，num_chunks=302） |
| 共享区单文件两 rank 共用 | PASS（PP1 创建 51.38 GB，PP0 join，barrier 后 unlink） |
| **物理钉住 = 配置值** | PASS：`df used` 51,586,396,160 B vs 配置 51,539,607,552 B，**比值 1.00**（c2 私有路径历史 1.56~2.0） |
| cudaHostRegister / 零断言 / 零熔断 | PASS（failed=0，ae=0，fuse=0） |
| store（GPU→CPU）真实落地 | PASS：累计 **32.15 GB**，零异常 |
| **load（CPU→GPU）真实回载** | ❌ **恒为 0**（见下方勘误：早期"5.37 GB"是探针口径 bug 造成的假值） |
| 定向命中（external_prefix_cache_hits_total） | **0** —— 见下「为什么 0」 |

  **为什么 external hits = 0（探针参数问题，已修）**：窗口 1 的探针用中文标定比例
  （0.5299 tok/字符）估算英文合成文档长度，实际每篇生成 **599k token**（而非 100k），
  11 篇挤池 = **7.8M token** 灌进 48 GiB 档（容量 ≈1.44M token）⇒ 建档文档必被 LRU 冲掉，
  定向重发当然 0 命中（`allocation_failure_total` 同步涨到 1495，正是"档满"的旁证）。
  已修：探针增加**比例自动标定**（实测 tok/字符）+ **容量自诊断/自适应选参**。

  **容量口径（本轮实测校准，2026-09-27 勘误后）**：CPU 档 ≈ **35.7 KB/token**
  ⇒ `容量_token ≈ cpu_bytes_to_use / 35.7KB`：48 GiB≈1.44M、64 GiB≈**1.12M**（实标）、96 GiB≈2.9M。
  要演示回载必须同时满足 `GPU池 + 建档 < 挤池 ≤ 容量 − 建档`（GPU 池 1.207M）——
  64 GiB 档**只差 7%**（早期"差 6 倍"的算法分母错了，见 §6.3 勘误）。

- **实机窗口 2（64 GiB + 自适应探针）**：`KVOFF_BYTES=68719476736
  PROBE_ARGS="--docs 1 --answer-tokens 1024" bash kvoff-c8-window.sh`——探针先标定
  bytes/token 与 CPU 容量，再按 `P+B < F ≤ C−B` **自动**定文档/挤池规模。
- **实机窗口 7（64 GiB + `--num-gpu-blocks-override 80`）**：**12 项结构判据全 PASS**
  （含物理钉住比值 1.00、store 8.299 GB 真实落地），但探针被**引擎卡死**打断（退出码 143）——
  详细现场与机制指向见 §6.3「窗口 7」。
- 仍未定论的一项：**store 在抢占边界**（09-23 的 `cuMemcpyBatchAsync error 1` 现场）会不会崩。
  窗口 1 的 store 累计 32 GB 正常；**load 恒为 0**（无命中 ⇒ 无需搬回），也从未把 KV 池压到抢占阈值。
  窗口 7 首次把池压到 6.6%（80 块）——**没崩，但卡了 9 分半**，见 §6.3。

### ⚠️ 勘误（2026-09-27）：早期「CPU→GPU 回载 5.37 GB」是假值

探针早期按子串累加 "kv_offload" 指标，把 `kv_offload_*_created`（**unix 时间戳 gauge**，
≈1.79e9）与直方图桶一起当成字节数 ⇒ 凭空造出 ≈5.37 GB。**真值**
`vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"}` 与 `kv_offload_load_bytes`
在五轮窗口里**一直是 0**。探针已改为只累加真字节 counter（回归测试含"时间戳 gauge 不得计入"）。

⇒ 截至本轮，本机 L2 档的状态是：**存得进（32 GB / 8.3 GB）、从未命中、从未搬回**。
机制解释见 §6.3：`C ≈ 1.12 M < P = 1.207 M` 且驱逐是纯 LRU ⇒
CPU 档里的块**必然同时还在 GPU 池里** ⇒ 查表永远在 GPU 侧满足（只差 7%，不是差 6 倍）。

### 判定「L2 是否真的能工作」的正确实验：人为把 GPU 池压小

用 `--num-gpu-blocks-override`（本机 0.30.0 支持，`arg_utils.py:1277`）把 P 压到 < C，
再跑「建档 → 挤池(>P) → 重发」：此时"显存里没了、CPU 档里还在"第一次在数学上成立，
若命中则 `external_prefix_cache_hits_total > 0` 且 `kv_offload_load_bytes > 0`。
窗口 5/6/7 用的就是这套（`FN_MAXLEN=65536` + `--num-gpu-blocks-override 80` ⇒ P=79 437 token < C≈1.12 M）。
窗口 5/6 因探针**单位 bug**（文档大 6 倍、顶穿 65536 上限吃 400）未测成；
窗口 7 修好后**结构判据全绿但引擎卡死**（见 §6.3）。

### 窗口脚本的三条加固（2026-09-27 窗口 7 事故后，每条都有实机教训）

1. **`launch.env` 先备份、回滚/中断时还原**（`LAUNCH_BAK`，缺省
   `/home/ll/deploy/kvoff-c8-launch.env.bak`）。它是**看门狗与控制台下次启动的唯一参数源**；
   测试档留在盘上 = 生产被静默起成测试参数。本次实际发生（生产一度跑在
   `max-model-len 65536 + 80 blocks`），已恢复（`1048576`、无 override）。
2. **`systemctl --user` 必须显式给 `XDG_RUNTIME_DIR` / `DBUS_SESSION_BUS_ADDRESS` 并校验生效**
   （新 `wd()` 助手打印 state）。旧写法 `systemctl --user stop ... 2>/dev/null || true`
   **静默失败**：窗口开头"停看门狗"没生效，看门狗整场每 35 s 照跑，收尾时与窗口回滚
   **抢启动**，还把 `launch.env` 覆盖成"只有一行头"（生产参数靠 inner 缺省兜住，属运气）。
3. **探针套 `timeout`（`PROBE_TIMEOUT` 缺省 900 s）+ 卡死取证**（`stall_evidence()`：
   统计 `Waiting>0` / `Deferred>0` / 零吞吐行数并回显最后 3 行吞吐）。
   引擎卡死时探针会一直等（`post` 超时 1800 s），旧版窗口会空转十几分钟且结果里看不出是卡死还是慢。


## 6.3 容量口径实测与策略方向（2026-09-27 实测 + 同日勘误，含外部参考）

### ⚠️ 先看勘误：**「CPU 档只有 0.20 M token / 340–400 KB per token」是错的（已撤回）**

错在**分母**：窗口 3 标定时用「名义 20 000 token」当存储量，而探针实际发出去的是
**≈120 000 token** 的文档（`ratio` 是 token/字符，`make_doc` 早期按"词数"折算 ⇒ 文档大 6 倍）。
于是 `bpt = Δstore / 20000` 把每 token 字节数放大了 6 倍、`C` 缩小了 6 倍。

**校正后的真值（同一次实测数据、只换分母）**：

| 项 | 值 | 口径 |
|---|---|---|
| CPU 档每 token 字节 | **≈35.7 KB/token** | `bpt = Δbyte / Δusage × C`（usage_perc 标定） |
| **CPU 档容量 C @64 GiB** | **≈1.12 M token** | `C = 标定 token / Δusage_perc`（实测 bpt=35 690.1 B/token） |
| GPU 池 P（大池位） | 1.207 M token | `kv_cache_size_tokens` |
| GPU 侧每 token | 27–32 KB/token | `kv_cache_size_tokens` ÷ 池字节 |

⇒ **C 与 P 同量级（仅差 ~7%），不是"差 6 倍"**。
"需要 ~400 GB 内存才压过 GPU 池"的推论、以及"L2 结构性不可能有收益"的结论**一并撤回**。

那 09-29「跑 21.5 h、外部命中恒 0」怎么解释？——是**纯 LRU + C < P**：
只要 `C ≤ P`，任何被 CPU 档留住的块**必然同时还在 GPU 池里**（CPU 档只是 GPU 驱逐的副产品），
查表在 GPU 侧就满足了，永远轮不到 CPU 回载。**不是容量小 6 倍，而是小 7%**——
这 7% 的差距让"零命中"从"结构性不可能"变成"刚好差一点"，方向完全不同。

### 为什么之前会得出 6 倍：探针的两个 bug（都已修 + 回归测试）

1. **单位 bug**：`make_doc` 按词数折算 ⇒ 文档大 6 倍（目标 5 000 实际 30 285 token 等）；
   已改为按**字符预算** `target_chars = tokens / ratio`（实测 5 000→5 009、46 000→46 014 token）。
2. **假字节 bug**：早期按子串累加 `kv_offload*` 指标，把 `*_created`（unix 时间戳 gauge
   ≈1.79e9）当成字节 ⇒ 凭空造出「CPU→GPU 回载 5.37 GB」。真值恒为 0；
   已改为只累加真字节 counter，并加"时间戳 gauge 不得计入"的回归测试。

### 窗口 7（64 GiB 公共区 + `--num-gpu-blocks-override 80`，P=79 437 token < C）：结构性全绿，探针被卡死打断

| 判据 | 结果 |
|---|---|
| c8 worker 侧公共区协商（两 rank 各一行） | PASS |
| 调度器侧采纳公共区 | PASS：`decision=['shared']` 各 rank 行数 `[786, 786]` ⇒ `num_chunks=786` |
| c1 offload 分组三侧一致 | PASS（`distinct=1`，环形分组已剔） |
| 共享区创建恰 1 次 / barrier 后 unlink | PASS（created=1、unlinked=1） |
| cudaHostRegister / 零断言 / 零熔断 | PASS（failed=0、ae=0、fuse=0） |
| **物理钉住 = 配置值** | PASS：`df used`=68 899 057 664 B vs 配置 68 719 476 736 B，**比值 1.00** |
| 两 rank 均分配 CPU KV 缓冲 | PASS |
| store（GPU→CPU） | 真实落地 8.299 GB（c9 打包后 rank0 89.98→43.64 MB、rank1 80.15→43.75 MB） |
| **探针（hits>0 且 CPU→GPU>0 且验证码复述）** | **FAIL：退出码 143** —— 不是探针报错，是**引擎卡死**（见下） |

**卡死现场（这是本轮最有价值的发现）**：探针发完 55 000 token 建档请求后，第二个请求
（挤池文档）进入 `Waiting: 1 / Deferred: 1`，然后**引擎日志整段静默 9 分 30 秒**
（11:20:00 → 11:29:30；统计行是 per-step 打的，静默 = 这 9 分半里**只跑了极少数 step**，
其中一次吞吐 17.0 tok/s，正常 5 200+），`Running: 0`、GPU KV 占用 0.0%、
`kv_offload_cpu_cache_usage_perc` 钉在 0.2404 不再增长。
时间上**紧跟着一次 store 突发**（11:19:40–11:19:50 单区间 store 651 MB、累计 1.98 GB）。

**机制指向（源码级，2026-09-27 补充）**：`Deferred` 不是队列慢，而是**调度器主动拒绝调度**——
`offloading/scheduler.py:643 _maximal_prefix_lookup()` 只有在整段前缀要么 `HIT` 要么 `MISS` 时才
返回可调度；一旦某个 chunk 返回 `LookupResult.HIT_PENDING`（"在档里但还不可读"）或 `RETRY`，
函数**返回 `None` ⇒ 该请求本轮不调度**（`scheduler.py:665/695/986`）。而 `HIT_PENDING` 的判据是
`cpu/manager.py:134` 的 `chunk.is_ready`——**只有 store 作业完成才会置 ready**。
⇒ **"某个 store 作业永远不完成" = 该请求被无限期 defer**，这正是窗口 7 的现场
（store 累计停在 8.299 GB、`write_usage_perc` 钉在 0.2404 不再增长、请求 9 分半不动）。
另外 `store` 方向仍是上游 **C++ 批量拷贝（`cuMemcpyBatchAsync`）**（启动日志自证：
`c7: 仅 CPU->GPU(load) 方向强制 Triton swap 内核；store 方向保持上游 C++ DMA`），
09-24 的 c7 定案正是它在 PP2 下冻结 compute 流 ⇒ **"store 作业卡住"的第一嫌疑就是它**（待坐实）。

**跨会话交叉证据（并行会话的 c10 窗口，2026-09-27 11:42–12:02，脚本/补丁未入本仓库）**：
该会话给 offload 作业加了 TTL（`FN_KVOFF_JOB_TTL=60` / `FN_KVOFF_PENDING_TTL=45`，即"卡住就放弃"），
在**更极端的 40 块池**（≈39.7 k token = GPU 池的 3.3%）跑 `--tokens 30000 --docs 1 --flush 5`：

| 项 | 结果 |
|---|---|
| 结构判据（含共享区、物理钉住 1.00、零 Xid） | 14 PASS / 1 FAIL（FAIL=探针三项判据） |
| **"探针后无 deferred 请求残留"** | **PASS（deferred=0）** ⇒ 加了 TTL 后不再永久挂住 |
| store（GPU→CPU） | 真实落地 7.55 GB（≈200 k token 的量级，即内容确实进档了） |
| CPU 档使用率 | 13.4%（远未满 ⇒ 不存在 LRU 驱逐） |
| `external_prefix_cache_hits_total` | **0**（`external_queries=248 533`，`prefix_hits_total` 也是 0） |
| CPU→GPU 回载 | **0** |

⇒ **两个独立会话、两种池大小（80 块 / 40 块）、都确认"存得进、档未满、但一次都不命中"。**
这把问题的性质从"容量不够"彻底改写成 **"命中路径（key 对不上或 store 未 ready）"**：
- 若是 key 对不上：store 时算的 offload key 与 lookup 时算的不一致（本模型是多组 hybrid +
  MTP4 投机，key 由各分组块哈希合成，分组/draft 标注差异足以让两侧 key 分叉）；
- 若是 ready 判定：`is_ready` 依赖 store 作业完成回调，作业卡住 ⇒ 永远是 `HIT_PENDING`
  （此时 `lookup` 会返回 HIT_PENDING 而不是 MISS ⇒ 请求被 defer，与"两个窗口都没崩却都没命中"吻合）。

**下一步的决定性实验**：在一次窗口里**把 store 时的 key 与 reload 时的 key 各打印前 N 条做比对**
（`cpu/manager.py` 的 `prepare_store` / `lookup` 各插一行日志），一次即可区分上面两条。
这比再调容量/池大小有意义得多。

### 下一步的三条候选（按性价比排序）

1. **先做归因 A/B（不改代码、最快）**：同一 `--num-gpu-blocks-override 80` 但 `FN_KVOFF=0`，
   跑同一探针。若也卡 ⇒ 卡死与小 GPU 池本身有关（测量手法不成立），KVOFF 无罪；
   若不卡 ⇒ 坐实是 store 路径。
2. **换更温和的压力档**：80 块 = 79 437 token 只剩 GPU 池的 6.6%，属极端档；
   命中只需 `P < C`，用 `--num-gpu-blocks-override 700`（≈695 k token < C≈1.12 M）即可，
   同时把 `P+B < F ≤ C−B` 的窗口放大到可操作。
3. **c10 候选：store 也走 Triton SM 内核**（c7 当年只改 load，理由是"私有路径缓冲未注册、
   Triton 会 MMU fault"；**c8 的公共区已 `cudaHostRegister` 成功**，该前提已不成立）。
   这是唯一能绕开 `cuMemcpyBatchAsync` 的现成手段，但上游在 `gpu_worker.py` 显式写着
   "GPU→CPU 不要用 Triton"，须带 Xid 监控做受控实验。

### 目标口径（与外部同类机一致）

宿主档 = **1.0× GPU 池**（本机 ≈1.2 M token）。按校正后的 35.7 KB/token，1.25 M token
只需 **≈45 GiB**（64 GiB 档已够，无需 400 GB）——与外部参考机 `--hicache-ratio 1.0` 的容量铁律一致。

### 若容量仍不够：走「策略」而不是「堆容量」

参考仓库 `github.com/ChinaBoy0618/170hx-qwen3.8-27b-fullstack`（v1.0.0，4×170HX/256GB，SGLang）
的同族结论与做法：
- `--hicache-ratio 1.0`（1.5 被 256 GB RAM 硬约束否决）——容量铁律与本机一致；
- 命中真正靠**分层+钉住**：单暖→tier1（压力下先逐）、双暖（hit_count≥2）→tier2（最后逐 + 12 h TTL）、
  `POST /admin/pin_prefix`→tier4 永久（实测 evictable 20 191 → 63）。
  **我们的演示失败正是"单暖被逐"这一条**；解法不是把档做大，而是给重要前缀晋级/钉住。
- vLLM 侧有现成落点：`--kv-transfer-config` 的 `cache_policy_module_path` 可加载**外置
  CachePolicy**（`v1/kv_offload/cpu/policies/factory.py`，源码注释「out-of-tree, no fork/patch」），
  另有 `v1/kv_offload/tiering/`（fs/obj/p2p）分层骨架 ⇒ **c10 候选：外置 tier/TTL/pin 策略模块**。
- ⚠️ **铁律（它用事故換来的）**：驱逐器绝不能把 O(1) 的 `popitem` 换成 O(n) 全索引扫描——v2 补丁
  在 L3 近满 + write_back 压力下形成"多分钟扫描风暴"，backup 线程 100% CPU 钉死 → 写停摆 →
  调度主循环挂死 →「容器 Up / HTTP 200 / 引擎死」僵尸签名。修复=**有界扫描窗口 + 驱逐预算上限**
  （16 384 次 ≈50 ms worst-case）。与我们的 c6（有界等待 + store 熔断只读降级）同源，
  且它的僵尸签名与我们历史 `shm_broadcast → RPC sample_tokens 超时 → EngineDead` 完全同类。

### 运维铁律（本轮新增）

**实例启动期间绝不覆盖补丁文件**：`scp` 非原子，worker 在启动早期 import `patches-extra/*.py`，
覆盖瞬间可能被读到半截文件。改补丁要在无实例启动时做。

## 6.4 segfault 根因与 rt-patch #10：attrIdxs 越界 UB 修复（2026-09-27）

SimpleCPUOffloadConnector 生产档（96 GiB）soak 当日两次原生段错误（17:38:35 / 20:38:30，
Worker_PP1 猝死，栈仅 glibc pthread 帧，无 Xid/MCE/Python 异常）。定位=上游 issue
[vllm-project/vllm#53860](https://github.com/vllm-project/vllm/issues/53860)：
`vllm/v1/simple_kv_offload/cuda_mem_ops.py::copy_blocks` 用 `ctypes.byref(单个 c_size_t)`
冒充 `attrIdxs` 数组传给 `cuMemcpyBatchAsync`；驱动按契约读 `count` 个 size_t ⇒ 标量之后
全是堆相邻随机字节，非零即索引 `attrs[垃圾]` ⇒ 布局依赖的间歇性 UB/段错误。

- **修复**：`patches-extra/dsh_simple_offload_rt.py`（rt-patch #10）替换 `copy_blocks`，
  每次调用传 `np.zeros(cnt, uint64)` 零索引数组（语义与原期望完全一致），另加块数相等/负 id
  入参校验。线程、流、事件排程、DMA 拷贝机制零改动 ⇒ **二级缓存功能原样保留**。
- **挂载**：`patches-extra/sitecustomize.py` 并入 `PATCHES`；钩子只挂
  `vllm.v1.simple_kv_offload.cuda_mem_ops` 一个模块，不开 `--kv-offloading-size` 时天然惰性。
  幂等哨兵=模块属性 `_dsh_attridxs_fixed`。回退开关 `DSH_SIMPLE_OFFLOAD_UPSTREAM=1`。
- **验证**：离线自检 `selftest_simple_rt.py`（G1 挂钩幂等 / G2 attrIdxs 零数组+描述符逐值 /
  G4 入参加固 / G5 回退开关 / G6 copy_backend 绑定传导，G5/G6 用 fresh-import 子进程——
  `importlib.reload` 不清命名空间，测不出新进程语义）；上线判据=启动日志每个进程一条
  `[dsh-simple-rt] ... attrIdxs 单标量→count 元素零数组` + `SimpleCPUOffloadWorker [CPU]` 块数
  + `external_prefix_cache_queries_total` 递增；soak 监控（kvoff-soak-monitor.sh）持续记录
  ext_hits/xid/health 到 `kvoff-soak.log`。
- **取证加固**：inner 脚本 `ulimit -c unlimited` + `core_pattern=/media/ll/data/cores/core.%e.%p.%t`
  （worker RSS 上百 GB，core 必须落数据盘）；gdb 已装。若再次 segfault，直接
  `gdb <python3.11> core.* -batch -ex bt` 拿原生栈。

## 7. 回滚

新栈出问题就整条丢掉 PYTHONPATH 概念、直接复活旧栈（旧文件全部未动）：

```bash
touch /home/ll/deploy/vllm-0300/DISABLED                 # ① 先摘看门狗：让它改回托管旧栈
bash /home/ll/deploy/vllm-0300/stop-flash-next-0300.sh   # ② 停新栈（脚本已按端口圈定，会等显存归零）
set -a; . /home/ll/deploy/vllm-0300/rollback-old-stack.env; set +a
bash /home/ll/deploy/start-flash-next-w4a16.sh           # ③ 起旧栈（注意它会重写自己的 launch.env）
```

①是关键：看门狗默认托管新栈，不回滚哨兵的话它会在你停掉新栈后 60s 内又把新栈拉起来，
和手工回滚对打。`DISABLED` 存在时看门狗自动改用旧栈的 start/stop/launch.env；
删掉该文件即恢复托管新栈。控制台侧不必动——`SCRIPT_MODELS` 的键名没变，
回滚后把 `script/inner/stopScript/log` 四处改回 `/home/ll/deploy/*-w4a16.*` 即可
（或直接 `python3 /home/ll/deploy/redirect-console-watchdog-0300.py --revert`）。

更轻的「只退补丁不退版本」：把 inner 脚本的 `VLLM_RT_PATCHES=1` 改掉即可回到纯原厂 0.30.0
——但那样 PP2+PLE 会被上游硬拒，起不来，仅作取证用。

## 8. 已验证事实（09-26 上线时）

- 启动 21:10:07 → `Application startup complete` / health=200 @ 21:18:15，**8 min 8 s**。
- 补丁逐条在日志留痕：INC PLE 按未量化接受、95.368 GiB 分 2 块 23.2 s 注册、
  `weight_device=cpu, pinned=True`、PP>1 PLE 检查放宽、`[rt-patch-extra] … 早退`。
- KV：`Available KV cache memory: 18.05 GiB`，`GPU KV cache size: 1,207,262 tokens`
  （旧栈 1,224,366，-1.4%），1M 请求并发 1.15×。
- 权重 17/17 分片、CUDA 图 PIECEWISE 8/8 + FULL 2/2、mamba page padding 0.62%。
- `/v1/models` ctx=1048576；chat 冒烟正常；**`usage.prompt_tokens_details.cached_tokens`
  现在有值**（旧 chroot 镜像恒 null，DSH 状态栏缓存命中率因此一直是 0%）。
  受控复测：同 7745-token prompt 第二次 `cached=4848`（=3×1616 整块，末块不写属预期）。
- 8889 控制台自动识别新实例（`/home/ll/vllm-env/bin/vllm serve` 形式已被
  `isVllmProcCmd` 覆盖），别名路由与 18420 端口显示正常。
- 非致命告警：`kv_cache_utils.py:2206 Speculative decoding (method=mtp) is enabled but
  no KV cache group could be identified as the draft model's.` —— 上游对 PP2+MTP 的
  保守提示，实测 `spec_decode_num_drafts_total` 正常递增、接受长度 2.77。

## 9. 控制台与看门狗接线（09-26 完成；10-05 审计修订）

**8889 控制台已指向本栈**。09-26 首版补丁 `redirect-console-watchdog-0300.py`（幂等、marker 判重）
把字段**整体改到新栈**；当晚已演进为可回滚的「**双字段 + 哨兵**」方案，下述为现行实现：

- `SCRIPT_MODELS['qwen3.8-flash-next-w4a16']` **同时保留两套字段**：`script/inner/stopScript`
  （chroot 旧栈）与 `scriptNew/stopScriptNew`（官方 0.30.0）；由 `resolveStartScript()` /
  `resolveStopScript()`（server.js:640 / 631）按 `DISABLED` 哨兵动态选——判据与看门狗
  `fnx-18420-watchdog.sh` 同源，`touch /home/ll/deploy/vllm-0300/DISABLED` 即全线退回旧栈。
  键名保持不变 → 快启预设、代理路由、别名表零改动。
- **`inner` 故意没有 `innerNew` 对应字段**：新栈 wrapper 的 `INNER` 缺省即
  `$BASE/bin/flash-next-0300-inner.sh`（start-flash-next-0300.sh:18），而 spawn 侧以
  `startScript === sm.script` 为守卫（server.js:8448 / 9284）——解析到新栈时 **envPrefix 为空**，
  绝不把旧栈 inner 灌进新栈。若在这里补一个 `innerNew`，反而会引入
  「新栈环境 + 旧栈 inner」的杂交命令（09-26 评审已识别）。
- 新栈 wrapper 的参数透传是**动态**的：`compgen -e | grep -E '^FN_[A-Z0-9_]+$'`（仅排除
  `FN_ENVFILE`），不再维护手写白名单——这是 09-26 `FN_GENCFG` 静默失效事故的根治
  （旧栈手写白名单漏项会让弹窗字段静默不生效，本项目在 `FN_PLE_INT8`/`FN_KVOFF` 上累计踩过两次）。
- 顺带修了一个**跨栈通用缺陷**：`scriptModelInstance()` 与 `findVllmPidByPort()` 的兜底判据写死
  `pgrep -f "[v]llm.entrypoints"`，只认 chroot 形态的 `-m vllm.entrypoints.cli.main serve`；
  新栈主进程 cmdline 是 `bin/vllm serve` → 判活恒假。后果不止日志面板回落 `vllm.log`，
  更严重的是**「已在运行」防重复启动失效**（点启动会试图再起一个）。已放宽为
  `"[v]llm.entrypoints|[v]llm serve"`。教训与 09-19 那条同源：凡按 cmdline 字样找 vllm 主进程的地方，
  换栈/换入口都会失明，必须覆盖全部形态。
- 三个快启预设的采样值同步回到 09-21 定版（原预设里是 18:42 实跑的漂移态 1/0/1，
  **点预设会把新栈刚定版的采样又覆盖回去**——这是本次最容易漏的一层）；
  `kvoff/pleInt8/pleLoc` 字段从预设删除（本栈无此档位，留着只会误导）。
- 端到端等价验证：`python3 /home/ll/deploy/plan_argv_check.py`（node vm 真跑 plan → inner dry-run
  → 与 `/proc/<pid>/cmdline` 逐 token 对账）。当前结论：`current-1m-mtp4-*` 预设与生产命令
  仅差一个显式 `--enable-chunked-prefill`（=0.30.0 默认值，无行为差异）。

- **三层 `FN_*` 穷举对账（10-05）**：对 `plan → wrapper → inner` 逐层取变量集合做差集。结论——
  wrapper 动态透传无缺口；`inner` 不串栈；长上下文三档链路完整（`maxModelLenLong` 1048576 /
  `maxModelLen512` 524288 / `longCtxModelPath` / `longCtx512ModelPath` / `altModelPaths` 均在
  `qwen3.8-flash-next-w4a16` 条目内，位于 `base` 块之后）；采样 inner `GENCFG_DEFAULT`
  ≡ `SCRIPT_MODELS[..].base`（09-27 起同为 t1.0/p0.95/k20/minp0/pp0/rp1.0），
  **但三个快启预设显式带 `temperature:0.6`** → 预设启动会覆盖成 0.6（见 §4 注，待拍板）。**唯一真缺口 = `FN_SCHED_POLICY`**：plan 在用户
  选非 fcfs 时下发（server.js:828-829 `if (sp && sp !== 'fcfs') env.FN_SCHED_POLICY = sp;`），
  而 inner 既不消费它、也没有对应的 `NOOP_NOTE_` 说明 → 落到「参数体检」的 UNKNOWN 分支，
  只打一条 `[FN-0300] 警告：收到本脚本未实现的参数 FN_SCHED_POLICY`，引擎仍走 fcfs。
- **接线做法与验证**：inner 在「控制台可关的三项」之后加
  `if [ -n "${FN_SCHED_POLICY:-}" ]; then ARGS+=(--scheduling-policy "$FN_SCHED_POLICY"); fi`，
  并把 `FN_SCHED_POLICY` 登记进 `CONSUMED`（从而不再进 UNKNOWN）。兼容性已核源码：
  `AsyncScheduler(Scheduler)`（`v1/core/sched/async_scheduler.py:12`）只覆写
  `_update_after_schedule` / `_update_request_with_output`，waiting 队列与抢占的 policy 逻辑在基类
  （`scheduler.py:194-204` `create_request_queue(self.policy)`、`746-759` 抢占排序），故
  `--scheduling-policy priority` 与 `--async-scheduling` 可共存、无需互斥；合法值
  `Literal["fcfs","priority"]`（`config/scheduler.py:22`），`fcfs` 即引擎缺省、plan 不下发。
  三重验证：`bash -n` 通过；**不带该变量时 dry-run argv 与改动前逐字节相同（缺省路径零漂移）**；
  带 `=priority` 时参数落到 argv 且 UNKNOWN 警告消失、`FN_PLE_INT8/LOC` 的 `NOOP_NOTE` 说明保持。
  备份 `bin/flash-next-0300-inner.sh.bak-schedpolicy-1005`，幂等补丁 `tools/patch-schedpolicy-1005.py`
  （`--revert` 可回滚）。

**看门狗 `fnx-18420-watchdog` 已改为与栈无关并已 re-arm**（`systemctl --user is-active` = active）：

- 判活 `vllm_alive()` 从「按 cmdline 字样」改为「扫 /proc：comm ∈ {vllm, python*} 且含 serve 且
  含 `--port 18420`」→ 新旧两种 APIServer 形态都认，且按端口圈定，不会把别的实例算进来。
- 引擎日志文件不再写死：`pick_log()` 取两个候选里 mtime 最新的那个（换栈后卡死取证不必再改脚本）。
- 停/起脚本、`launch.env` 路径按栈切换。
- **回滚哨兵**：`touch /home/ll/deploy/vllm-0300/DISABLED` → 看门狗自动退回托管 chroot 旧栈。
  所以 §7 的回滚流程现在**不必**先 stop timer（详见 §7）。

## 10. 待办（不影响当前服务）

1. 与旧栈基线（prefill 8858 tok/s @55K、decode 均值 123.6；18432 参考 130.6，两组都是
   K5+block1680 档）做同档位 A/B：预设 `bak-1m-mtp5-kv96` 已备好 K5/1680，需 8 分钟停机重启。
2. 跑 `selftest_extra.py` 与上游 `selftest.py` 各一遍（本次上线前只跑了上游那份 PASS=23）。
3. plan 的 MTP 警告文案在 `bak`（block 1680）档位下仍写「1616 下合法档 1~4/9~12」，
   对 K5 是误导（1680 下 K5 合法），择机按 block 分档出文案。
4. 把新栈与结论提交进 `repo18420/qwen38-flash-next-170hx`（脚本里的 sudo 明文口令要先脱敏）。
5. 决定 `/etc/security/limits.d/99-vllm-memlock.conf` 的去留（systemd 下无效，留着误导人）。
6. 弹窗 UI 上「PLE 表加载（精度/位置）」两个字段对本栈无效（inner 会打
   `[FN-0300] 忽略 …` 并说明原因；官方 0.30.0 只有 BF16 锁页一档）。
   「CPU KV 二级缓存」字段自 1005（§6.1）起已真实生效，缺省关；
   「调度策略」字段自 10-05（§9）起已真实生效（此前只落一条「未实现」警告）。
7. `FN_KVOFF_WAIT_TIMEOUT`（c6 熔断的有界等待，缺省 15 s）inner 已消费、但 plan 不下发 →
   弹窗无字段，只能 envfile 手填。要暴露需按 §9 的多层清单同步改 index.html 与 server.js。
   低优先（不影响正确性，只影响可调性）。
8. **把现行「双字段 + 哨兵」栈路由做成幂等重打脚本**并收入本仓库：目前唯一的重打脚本
   `redirect-console-watchdog-0300.py` 是已被取代的首版做法，误用会废掉旧栈回滚能力（见 §9 末）。
9. 【KVOFF 归因，最高优先】**打印 store 时的 key 与 reload 时的 key 比对本**（在
   `v1/kv_offload/cpu/manager.py` 的 `prepare_store` / `lookup` 各插一行，只打前 N 条哈希）——
   一次窗口即可区分「key 对不上」还是「store 未 ready（HIT_PENDING 一直挂）」。
   这比继续调池大小/容量有意义得多（两种池大小 80/40 块都已确认"存得进、档未满、零命中"）。
10. 【KVOFF 换温和压力档】命中只需 `P < C`，用 `--num-gpu-blocks-override 700`（≈695 k token
    < C≈1.12 M）即可，比 40/80 块温和得多，`P+B < F ≤ C−B` 的可操作窗口也更大。
11. 【c10 候选】把 **store 方向**也切到 Triton SM 内核：c7 当年只改 load，理由是"私有路径
    缓冲未注册、Triton 会 MMU fault"；c8 的公共区已 `cudaHostRegister` 成功 ⇒ 该前提不成立。
    上游 `gpu_worker.py` 显式写着 "GPU→CPU 不要用 Triton"，故须带 Xid 监控做受控实验。
12. 【跨会话协调·长期】同一时刻只允许一个会话在 18420 上跑验证窗口：2026-09-27 两个会话
    的窗口**互相打断**（11:36 我的窗口实例被另一个会话 stop，我的探针/脚本被 SIGTERM，
    窗口 7 没跑到回滚；12:02 起对方又连跑两轮），结论容易被污染。
    判据：`ps -eo pid,args | grep [k]voff` + `/home/ll/deploy/*kvoff*` 的 mtime。

**并行会话风险（本机长期事实）**：`/home/ll/deploy/server.js` 会被其它会话基于旧基线整文件写回。
本次 22:02:44 就发生过一次——重定向与判据补丁被抹掉、控制台被重启。判据：改完记下 md5，
隔几分钟复查。

⚠️ **被写回后重打，勿再用 `redirect-console-watchdog-0300.py`**：它是 09-26 22:05 的**首版做法**
（把 `script/inner/stopScript/envFile/log` 整体替换成新栈路径，见该脚本 `:81-87`），
在现行「双字段 + 哨兵」方案上重打会把 `script` 也改成新栈路径 → 旧栈字段丢失、
`DISABLED` 哨兵回滚随之失效。自查现行方案是否完整：

```bash
grep -n "function stack0300Active\|scriptNew:\|stopScriptNew:\|startScript === sm.script" \
  /home/ll/deploy/server.js
```

应看到 `stack0300Active` 定义、`scriptNew:`/`stopScriptNew:` 各一处、`startScript === sm.script`
两处（spawn 侧守卫）；缺哪类即被写回。恢复＝按本节上面三条重新实施，**本仓库尚未收该方案的
幂等重打脚本**（见 §10 待办 8）。


---

## 11. 生产定版（2026-09-27 20:20 快照）：内存二级缓存 96 GiB 常驻开启

> 本节是**当前真实运行态**的逐字记录。实例 19:34:40 拉起，至快照时刻无卡死、无新增 Xid。
> 完整 argv / env 快照（从 `/proc/<APIServer>/cmdline`、`/proc/<APIServer>/environ` 直读）：
> [`tools/live-cmdline-0300.txt`](tools/live-cmdline-0300.txt)、[`tools/live-env-0300.txt`](tools/live-env-0300.txt)。

### 11.1 形态一览

| 项 | 值 | 判据（怎么自己验） |
|---|---|---|
| 引擎 | 官方 vLLM **0.30.0**（宿主 venv py3.11，site-packages 零改动）+ `PYTHONPATH=<本目录>/patches:<本目录>/patches-extra` | `pip show vllm`；日志 `[rt-patch-extra] 本地扩展钩子已挂载` |
| 并行/投机 | PP2（26,22）+ **MTP4**；block 1616；mamba float32；1M YaRN×4 | `/metrics vllm:cache_config_info` |
| **内存二级缓存** | **SimpleCPUOffloadConnector，`--kv-offloading-size 96`（GiB，两 rank 均分 48 GB/rank）** | 日志 `SimpleCPUOffloadConnector: role=SCHEDULER, per_rank=48.00 GB` + `SimpleCPUOffloadWorker [CPU]: N CPU blocks (≈48 GB)` |
| GPU KV 池 | **1,207,262 token / 776 块**（18.06 GiB，@1M 并发 1.15×） | 日志 `GPU KV cache size: 1,207,262 tokens` |
| CPU 档容量 | blocks/rank = [2224, 2157] → min×1616 = **≈348.6 万 token = 2.9× GPU 池**（≈36.3 万 tok/GiB） | 日志 worker 行 + 换算 |
| launch.env 钉法 | `FN_KVOFF=0`（经典退役）+ `FN_SIMPLE_OFFLOAD=96`；两者同开 inner 直接拒启 | `cat /home/ll/deploy/vllm-0300/launch.env` |
| 回载收益（实测） | 挤池 >1.2M token 后重发：~100k prompt 命中 **97%**、~40k **93%**、~12k **80%**（零头=每请求末块 ≤1616 不入库）；验证码精准复述=内容无损 | `vllm:external_prefix_cache_hits_total` 增量 + `kvoff-accept.sh` |
| 快照时刻累计 | external hits = **143,824 token**（运行 45 分钟）；本地前缀缓存累计命中 **88.8%**（22.65M/25.51M queries） | `curl :18420/metrics` |
| 内存账 | PLE BF16 锁页 95.4 GiB + CPU 档 96 GiB 常驻 + 引擎 ≈ 204 GB / 251 GB（free≈7、avail≈44） | `free -g`（按行号取，勿按中文标签） |

### 11.2 为什么是 Simple 而不是经典连接器（定版理由，细节见 ROUND1/2 文档）

- 经典 OffloadingConnector（我们移植的 c1~c11）在本模型（hybrid：全注意力 + GDN/mamba 组）上
  **store 拷贝期间 mamba 状态块仍被模型活写** → CPU 档回载内容不等价（注意力组逐字节精确、
  mamba 组不等），属数据面竞态，判定退役。
- 官方 `SimpleCPUOffloadConnector` 显式 `SupportsHMA`，`v1/simple_kv_offload/manager.py:834-836`
  **跳过 `has_positionally_stable_blocks=False` 的组**——只卸载/回载位置稳定的注意力组；
  CPU 档查找复用核心 `cpu_coordinator.find_longest_cache_hit`；拷贝走后台线程 DMA
  （独立流 + compute-done 事件排序），不碰 worker 主线程 → 与 PP2 的 NCCL 中继无锁死面。
- 收益边界（如实）：只对「前缀已被挤出 GPU 池后又被重发」的负载有收益；本机历史（GPU 池
  自扛 ~90% 命中）说明多数流量用不到它，但档位常驻的代价只有内存、无稳定性投诉
  （soak 记录见 §11.4 工具）。

### 11.3 复现要点（他人从零跑通本形态）

1. `uv venv --python 3.11 && pip install vllm==0.30.0`（torch 2.13.0+cu130 / flashinfer 0.6.18.post1 随附）；
   补 cu13 软链：`site-packages/nvidia/cu13` 下 `lib64→lib`、`lib/libcudart.so→libcudart.so.13`。
2. 内核 `memlock` 不限（root 运行即可；PLE 95.4 GiB `cuMemHostRegister` 与大页锁 pinned 都要它）。
3. 把本目录的 `patches/` 与 `patches-extra/` 放上机器，
   `PYTHONPATH=<绝对路径>/patches:<绝对路径>/patches-extra` 注入启动环境（这是 site-packages 零改动的
   运行时补丁机制；缺了它 PP>1 / auto-round PLE 会被上游硬拒）。
4. `FN_*` 环境按 §11.1 表（或直接抄 `tools/live-env-0300.txt`），跑
   `bash start-flash-next-0300.sh`（脚本内 sudo 口令走 `SUDO_PASS` 注入）。
5. 就绪判据四行日志（期望值逐字）：
   `GPU KV cache size: 1,207,262 tokens`、`Creating v1 connector with name: SimpleCPUOffloadConnector`、
   `SimpleCPUOffloadWorker [CPU]: 2224 CPU blocks (47.98 GB)`（PP0；PP1=2157）、
   `Capturing CUDA graphs` 正常走完；`curl :18420/health` = 200。
6. 验收：`bash kvoff-accept.sh`（preflight + 挤池 + 重发对照 + 失败自动恢复，详见
   [`ROUND3-simple-offload-验收手册.md`](ROUND3-simple-offload-验收手册.md)）。

### 11.4 运维件（本目录已收齐，均为生产现行版本的脱敏副本）

| 文件 | 机器路径 | 用途 |
|---|---|---|
| `patches/sitecustomize.py` | `vllm-0300/patches/sitecustomize.py` | 上游运行时补丁入口（8 处 hook，正源公开仓库 `CyrilCN/qwen38-flash-170hx-patches`，本副本含 09-27 FP8-PLE 加载契约修正） |
| `patches-extra/sitecustomize.py` | `vllm-0300/patches-extra/sitecustomize.py` | 本地扩展入口（#8 mm-warmup 跳过 + #9 挂 dsh_kvoff_rt + #10 挂 dsh_simple_offload_rt） |
| `patches-extra/dsh_kvoff_rt.py` | 同名 | 经典连接器移植层（c1~c11；现已退役但保留代码，缺省惰性零影响，总闸 `DSH_KVOFF_RT_DISABLE=1`） |
| `patches-extra/dsh_simple_offload_rt.py` | 同名 | SimpleCPUOffload attrIdxs 越界 UB 修复（§6.4，上游 #53860）；回退 `DSH_SIMPLE_OFFLOAD_UPSTREAM=1` |
| `patches-extra/selftest_simple_rt.py` | 同名 | #10 离线自检（G1-G6，PYTHONPATH=patches:patches-extra 跑） |
| `fnx-18420-watchdog.sh` + `systemd/fnx-18420-watchdog.{service,timer}` | `/home/ll/deploy/…`、`~/.config/systemd/user/…` | 30s 探活自愈；**自愈安全模式会剥离 offload 档位**（防崩溃循环），带档自愈加回需手工重放 `launch.env` |
| `kvoff-accept.sh` / `kvoff-churn.py` / `kvoff-soak-monitor.sh` | `/home/ll/deploy/…` | 验收 / 挤池判据 / 每 5 分钟 soak 记录（health·KV 水位·外部命中·Xid·内存） |
| `tools/live-cmdline-0300.txt` / `tools/live-env-0300.txt` | —— | 生产实跑真值快照（2026-09-27 20:20） |

### 11.5 风险与第一手处置

- 96 GiB 档 = 物理常驻；本机 251 GB 内存且有**内存混插 MCE 硬挂史**（docs/08 P30~P33 语境）。
  若复发整机硬挂/卡死：第一刀 `FN_SIMPLE_OFFLOAD=48`（命中率实测与 96 持平，只减存活时长），
  第二刀 = 去掉该变量回无二级缓存定版。
- 看门狗自愈后档位被剥离属**设计行为**（§11.4），检查 `launch.env` 与 `fnx-watchdog.log`
  的 `TIER_DROPPED` 行即可确认。
