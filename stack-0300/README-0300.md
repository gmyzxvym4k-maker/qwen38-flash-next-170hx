# 18420 Qwen3.8-Flash-Next @ 官方 vLLM 0.30.0（192.168.1.127）

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
| c1 | `offloading.config.get_offloading_group_ids` | 剔除 QSA 环形分组（CircularBufferSpec，ring 8 ∤ 1616）与空占位分组；0.30.0 的 scheduler/worker 按显式 group_id 索引，源头过滤即完整闭环（不再需要旧栈的保位 hack） |
| c2 | `cpu.spec.CPUOffloadingSpec._uses_shared_region` | pp_size>1 → 每 rank 私有 pinned 缓冲（共享 mmap 区在 PP 各 rank 尺寸不一致） |
| c6 | `cpu.gpu_worker`（handler.wait 有界+熔断、worker.wait 回传未完成集）、`offloading.common`（meta 增 failed_jobs/fuse_tripped + aggregate 合并）、`offloading.worker`（提交 try/except、失败 ack、迟到完成去重、熔断后只读降级）、`offloading.scheduler`（update_connector_output 完整替换：失败 job 走 `complete_store(success=False)` 撤登记+释放块；fuse 后两个 store 构建函数短路返回 {}） | 拆掉「worker 主线程无限 event 等待 × PP2 NCCL 中继」锁死环（09-24 七次卡死根因）；store 失败只丢缓存条目，load 正确性零妥协、永不熔断 |
| c7 | `cpu.swap_blocks_triton.MIN_N=0` + `cpu.gpu_worker._select_swap_blocks_fn` 双向强制 Triton | 彻底绕开 cuMemcpyBatchAsync（09-23 崩溃/09-24 卡死共同毒点：与 PP2 NCCL-P2P 并发冻结 compute 流）。本模型块页 MB 级 >上游 28KB 阈值，必须显式覆盖 |

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
**尚未做**：FN_KVOFF=1 的实机验证窗口（需停机重启 8 分钟，等需要启用时执行）。

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

## 9. 控制台与看门狗接线（09-26 已完成）

**8889 控制台已指向本栈**（补丁脚本 `redirect-console-watchdog-0300.py`，幂等、marker 优先判重）：

- `SCRIPT_MODELS['qwen3.8-flash-next-w4a16']` 的 `script/inner/stopScript/log/envFile` 全部改到新栈，
  键名保持不变 → 快启预设、代理路由、别名表零改动。
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
   「CPU KV 二级缓存」字段自 1005（§6.1）起已真实生效，缺省关。

**并行会话风险（本机长期事实）**：`/home/ll/deploy/server.js` 会被其它会话基于旧基线整文件写回。
本次 22:02:44 就发生过一次——重定向与判据补丁被抹掉、控制台被重启。判据：改完记下 md5，
隔几分钟复查；丢了就用 `redirect-console-watchdog-0300.py` 重打（幂等，会先核锚点再动手）。

