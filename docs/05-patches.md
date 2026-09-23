# 05 · 22 个 vLLM 补丁逐条说明

补丁全部打在**官方定制镜像内的 vLLM**（`vllm/vllm-openai:qwen38-flash-next`，版本 `v0.1.dev20073`，
Python 3.12，路径 `.../dist-packages/vllm`）。

```bash
# 体检 / 打补丁 / 回滚（幂等，逐文件 sha256 校验）
python3 scripts/apply-patches.py --target <rootfs>/usr/local/lib/python3.12/dist-packages/vllm --check
sudo python3 scripts/apply-patches.py --target <同上> --apply
sudo python3 scripts/apply-patches.py --target <同上> --revert
```

`patches/MANIFEST.tsv` 给每个文件两把哈希：`pristine_sha256`（上游原版）与 `patched_sha256`（打完应为）。
**应用后立即比对哈希，不符就自动回滚该文件**——这是为了杜绝"看着打上了其实打歪了"。

## 分组与必需性

| 组 | 补丁 | 作用 | 不打会怎样 |
|---|---|---|---|
| **A. PP>1 解禁** | 05 07 08 09 18 20 21 22 | 让 `PP2 + MTP + PLE CPU offload` 三者同开 | 启动即失败，或首个请求乱码/崩溃 |
| **B. PLE 表驻留形态** | 06 10 19 | 磁盘驻留 / INT8 / 内存堆四态 | 只能按手册要 128 GB 内存常驻 BF16 |
| **C. MTP 并行校验** | 01 | 草稿模型不按 PP 段校验 | 启动即失败 |
| **D. KV 二级缓存** | 02 03 04 13 14 15 16 | `OffloadingConnector` 在本栈可用 | `FN_KVOFF=1` 时启动或首请求必挂 |
| **E. 提速与观测** | 11 12 17 | 跳过多模态 warmup、每请求实时进度 | 只是慢 33 s / 仪表盘无数据，**不影响可用性** |

> A+B+C 是本仓库生产实例的**必需集**（9 + 3 + 1 = 13 个）。
> D 组 7 个只在开 `FN_KVOFF=1` 时需要；关掉二级缓存时它们存在也无害。
> E 组 3 个纯优化。

---

## A 组 · PP>1 解禁（8 个）

上游 vLLM 对 "PLE n-gram 嵌入 + 流水线并行" 的态度是**明确禁止**的，而本模型不放 TP（无 P2P，TP2 每步约 192 次 all-reduce 走 host SHM，实测比 PP2 慢）也无法单卡放下 79 GB 权重 + 显存 KV，所以必须把这道禁令连带的三处结构性缺陷一起改掉。
语义上等价于 vLLM **PR #52295**（pp_utils / model_runner / speculative）与 **PR #46994 / issue #54709**（mtp 按输入分支）；本仓库是**在定制镜像上重新实现**，不是 cherry-pick。

### 22 · `v1/worker/gpu_worker.py` — 撤掉 "PP≠1 不支持" 的硬拒
```
if parallel_config.pipeline_parallel_size != 1:
    unsupported.append(f"PP={...}")          ← 删除
```
**症状（不打）**：日志 `PLE CPU offload is not supported: PP=2`，引擎初始化阶段直接退出。
**根因**：该守卫假设 PLE 层只能在单一 rank 上；实际上每个 PP 段只持有自己那几层，PLE 层（第 2 层）落在 rank0。
**为什么放在最后讲**：它是"开关"，剩下 7 个才是让它成立的工作。

### 08 · `models/qwen3_8_flash_next/nvidia/model_state.py` — 撤掉第二道 PP 禁令
```
- if vllm_config.parallel_config.pipeline_parallel_size > 1:
-     raise RuntimeError("N-gram PLE embedding currently requires
-         pipeline_parallel_size=1 because non-first pipeline ranks do
-         not receive the raw input_ids required by PLE. ...")
+ # [FN-PLE-PP] PLE n-gram context is carried across PP stages via the
+ # PP broadcast payloads, so PP>1 no longer has to be rejected here.
```
**症状**：`RuntimeError: N-gram PLE embedding currently requires pipeline_parallel_size=1`。
**依据**：禁令的理由（非首 rank 拿不到 raw input_ids）由补丁 21 的 PP 载荷扩展解决——n-gram 上下文随 PP 广播一起过段。

### 07 · `models/qwen3_8_flash_next/nvidia/model.py` — 给非末段一个权重落点
```
- self.hyper_connection_mixer = None
+ self.hyper_connection_mixer = StageMissingLayer("hyper_connection_mixer")
```
**症状**：加载权重时 `ValueError: could not determine the shape of object type 'torch.storage.UntypedStorage'`（PP1 报）或 "found weights not used"，看着像 checkpoint 坏了。
**根因（重要，值得单独记）**：`AutoWeightsLoader` 只有 `_can_skip(skip_prefixes/skip_substrs)` 与 `_can_ignore_unexpected()` 两条跳过出口；模型用 `self.xxx = None` 表示"本段没有这个模块"时，`named_children()` 不包含 None，loader 找不到落点就 raise。**必须用占位 Module 而不是 None。**
**排障教训**：PP1 那条报错其实是 PP0 早死 9 秒引发的连锁误报——**多 rank 同时启动时永远先找最早死的那个**。

### 05 · `distributed/utils.py` — 层切分不匹配时回退而不是致命
```
- if len(partitions) != pp_size: raise ValueError(...)
- if sum(partitions) != num_hidden_layers: raise ValueError(...)
+ # [FN-PP-FALLBACK] 不匹配就回退默认切分
+ if len(partitions) != pp_size or sum(partitions) != num_hidden_layers:
+     partitions = None
```
**症状**：`ValueError: len(partitions)=2 does not match pp_size=1`，崩在 `PleOffloadWorker` 启动时。
**根因**：`VLLM_PP_LAYER_PARTITION=26,22` 是全局 env，PLE offload worker 是个 **pp_size=1 的独立进程**，继承同一个 env 后按自己的 pp_size 校验必然失败。

### 09 · `models/qwen3_8_flash_next/nvidia/mtp.py` — 草稿按输入分支，不按 rank 分支
```
- if get_pp_group().is_first_rank:
+ if intermediate_tensors is None:
      assert hidden_states is not None
      ...
  else:
-     assert intermediate_tensors is not None
      hidden_states = intermediate_tensors["hidden_states"]
```
**根因**：MTP 草稿头**整个建在最后一个 PP rank** 上，那里 `is_first_rank` 是 False，于是草稿走了"接收 intermediate_tensors"分支，但它自己是起点、根本没有上游张量 → 断言失败或拿到垃圾。判据必须换成"输入里有没有 intermediate_tensors"。
这正是上游 PR #46994 的 `is_local_drafter_forward` 语义。

### 21 · `v1/worker/gpu/pp_utils.py` — PP 广播载荷带上 draft_tokens（8 处）
```
- sampled_tokens: torch.Tensor            # [num_reqs, max_sample_len]
+ token_payload: torch.Tensor             # [num_reqs, max_sample_len + draft_token_width]
...
def __init__(..., sync_draft_tokens: bool = False):
+   self.draft_token_width = num_speculative_steps if sync_draft_tokens else 0
```
`receive()` 里把 `token_payload[:, :max_sample_len]` 当采样 token、`[:, max_sample_len:]` 当草稿 token，并按 `exclude_mask` 用 `valid_rows` 做 `index_select` 对齐行与槽位（`draft_idx_mapping`）。
**症状（不打）**：PP2 + MTP 下**输出 100% 乱码**——草稿位恒定吐 `220 / 199082 / 200263`（`' '`/`utente`/`Werktage`），而 MTP 接受率还显示正常（1.96/4），因为垃圾 token 是被"验证接受"的。
**攻关过程值得记**：曾长期怀疑"中继行数与 `idx_mapping` 长度不一致（CUDA graph padding）"，加了四点诊断补丁实测 `PPDRAFT-MISMATCH` **出现 0 次**——假说被证伪，真因是**载荷里压根没有 draft_tokens 这条通路**。诊断补丁要敢于先加再撤。

### 18 · `v1/ple_offload/connector.py` — 无 PLE 层的 rank 不注册（3 处）
```
+ # [FN-PLE-PP] All attributes above are initialised, but this
+ # pipeline stage owns no PLE layers. Do not connect, register
+ # or start the request thread: that would publish field-less
+ # messages to the shared CPU worker and crash it.
```
`_setup_layers` 在本段没有 PLE 层时返回 `{}`，且构造函数在属性初始化完成后、进 try 块**之前**提前 return。
**症状**：rank1 的 connector 线程向全局 CPU worker 发"无字段"消息 → PleOffloadWorker 崩 → 整个引擎 dead（表现为"启动正常、首个请求 500"）。

### 20 · `v1/worker/gpu/model_runner.py` — 按段构造 connector + 首 rank 才有 encoder（8 处）
```
+ # [FN-PLE-PP] Only the pipeline stage that actually owns PLE layers
+ # may build a connector. Other stages leave self._ple_offload_connector
+ # as None, so every guarded call site in this runner skips PLE.
...
+ # The encoder runner exists only on the first PP rank, so later ranks
+ # have no cached embeddings to gather.
```
**要点**：该 runner 里所有 connector 使用点都有 `is not None` 守卫，所以**置 None 是设计内正解**，不是绕过。
**症状**：非首 rank 去 gather encoder 缓存 → KeyError / 非法访问。

---

## B 组 · PLE n-gram 表的驻留形态（3 个）

### 06 · `envs.py` — 新增三个开关
```
VLLM_PLE_DISK_RESIDENT: bool   # 磁盘驻留（mmap，零堆分配）
VLLM_PLE_INT8_DIR: str         # INT8 产物目录（空=关）
VLLM_PLE_INT8_MEMORY: bool     # INT8 读进匿名堆（不可回收，零磁盘 I/O）
```
配合镜像自带的 `VLLM_PLE_CPU_OFFLOAD` / `VLLM_PLE_MMAP`，得到**精度 × 位置**四种组合（详见 `docs/04`）。

### 19 · `v1/ple_offload/worker.py` — 磁盘驻留加载器（+276 行，本仓库最大的补丁）
两个新方法，挂载顺序 = **INT8 → BF16 mmap → 匿名堆回退**：

- `_attach_disk_resident_ngram_table()`：直接 mmap checkpoint 里的 `model-00016-of-00017.safetensors`（102.4 GB 单文件，128 个 `[2500012,160]` BF16 张量首尾相接），**零堆分配**，页缓存可被内核回收。
- `_attach_int8_disk_resident_ngram_table()`：读 `ple_ngram_int8.bin` + `ple_ngram_scale.bin` + `ple_ngram_meta.json`；几何（rows/hidden）与模型 embedding 不符就返回 False 自动回落 BF16。`VLLM_PLE_INT8_MEMORY=1` 时用 `bytearray readinto + torch.frombuffer` 落成匿名堆。

**本补丁内埋着整个项目最贵的一行代码**：
```python
# [FN-PLE-DISK] safetensors 写了一个 8 字节长度前缀在 JSON header 之前，
# data_offsets 是相对【数据段】的，不是相对文件。
data_base = 8 + header_len
```
**漏加的后果**：整张表错位 `header_len/2 = 10368` 个元素，输出**内容词全对、只有标点乱**（`、。`/`。，`/三连字）。这种症状指纹在任何"嵌入表/查表"类模块上都成立——**看到"语义对、标点坏"就直接查 n-gram/嵌入层偏移**，不要去怀疑采样或量化。

另外两处细节：
- 映射一律用 `mmap.ACCESS_COPY`：读走页缓存，误写落到私有页，不会污染 102 GB 的 checkpoint。
- `torch.frombuffer` 借用的内存必须**显式持有 owner**（映射/bytearray 存进 self），否则 GC 回收后张量指向野内存。

### 10 · `models/qwen3_8_flash_next/nvidia/ple_layer.py` — INT8 查表 + CPU 侧反量化（2 处）
```python
q = torch.index_select(weight, 0, flat)                 # int8 行
s = torch.index_select(row_scales, 0, flat).unsqueeze(-1)
torch.mul(q.to(output.dtype), s.to(output.dtype), out=output.reshape(-1, head_dim))
```
外加一道保险：若表是 int8 却试图走普通 embedding 路径，直接 `raise RuntimeError`（否则静默吐垃圾）。
**设计要点**：反量化全部发生在 **CPU worker 内**，GPU 侧 IPC 契约仍是 bf16 —— 所以**显存占用、CUDA 图、跨卡通信与 BF16 路径逐字节相同**，换精度不动其它任何一层。这也是 INT8 能"无痛上线"的原因。

---

## C 组 · MTP 配置校验（1 个）

### 01 · `config/speculative.py` — 草稿模型不按 PP 段数校验
```python
draft_parallel_config = self.draft_parallel_config
if draft_parallel_config.pipeline_parallel_size > 1:
    draft_parallel_config = copy.copy(draft_parallel_config)
    draft_parallel_config.pipeline_parallel_size = 1
self.draft_model_config.verify_with_parallel_config(draft_parallel_config)
```
**症状**：`ValueError: ... pipeline_parallel_size 2 ...` 在 ModelConfig 校验阶段，权重还没开始加载就退出。
**根因**：草稿头全部建在末段（等价 pp=1），但校验拿引擎的 pp_size 去核对草稿的分片尺寸。

> **顺带记录一个由本补丁牵出的重要事实**（不属于代码改动，但决定了 MTP 能开几档）：
> QSA 的 key 环形缓冲 `ring = compress_ratio × cdiv(compress_ratio + num_spec, compress_ratio)`（compress_ratio=4），
> 且要求 **ring 整除 attention block size**。block=1616 时：
> `k=1..4 → ring 8 → 202 ✅` ｜ `k=5..8 → ring 12 → 134.67 ❌ 必崩` ｜ `k=9..12 → ring 16 → 101 ✅`。
> 报错原文：`AssertionError: QSA ring capacity 12 must divide the attention block size 1616`。
> 这解释了手册"k≥5 被封死"的说法——**不是产品限制，是整除约束**；NVFP4 路线用 block 1632（1632/12=136）就能跑 MTP6。
> **切勿用 A 路线的 MTP 档位去推 B 路线。**

---

## D 组 · KV 二级缓存（7 个，仅 `FN_KVOFF=1` 需要）

vLLM 自带 `OffloadingConnector`（GPU KV → 宿主内存），默认关闭。本栈开不起来不是功能缺失，而是上游几处假设与"混合架构 + PP2"冲突。

| 补丁 | 文件 | 症状（不打） | 做法 |
|---|---|---|---|
| 02 | `kv_connector/v1/offloading/config.py` | `AssertionError: tokens_per_block=8 not divisible by tokens_per_hash=1616` | 整除校验只对 `prefix_cacheable` 分组做——**核心 `kv_cache_coordinator` 本来就是这么过滤的，connector 漏了**。真凶是 QSA 的 `CircularBufferSpec`（block_size=8，每请求 1 块），不是 PLE 层 |
| 04 | `offloading/scheduler.py`（17 hunk） | 依次撞：`assert isinstance(spec, FullAttentionSpec)`（`get_sliding_window_size_in_chunks` 只认 4 种 spec）；把环形分组当 GDN 索引；CPU 档**只存不命中** | 把非 prefix_cacheable 分组从 store/load/lookup/touch/hit-chunk/storable **六条链路**排除，但**保留它在 `kv_group_configs` 里的位置**（`group_idx` 是承重下标，压缩数组会让调度器拿环当 GDN）。c5a 解除查找侧 eagle `+1/-1` 双罚：MTP 开启时 `from_spec` 兜底把所有分组标成 eagle，每轮多轮 delta（通常 1 chunk）被折成 0，CPU 命中**结构性不可能**；c4 对 align 分段(sw)分组跳过 +1 |
| 16 | `v1/kv_offload/cpu/spec.py` | 两个 rank 各等对方的字节数 → 30 s 超时；即便过了槽位也会重叠 | `pp_size > 1` 时 `_uses_shared_region()` 返回 False → 每 rank 用**私有 pinned 缓冲**。共享区的协议是"单文件 + 创建者 ftruncate 成自己的大小 + joiner 等自己的大小"，而 PP 各段层数不同（实测 34.32 GB vs 34.3566 GB） |
| 03 | `offloading/metrics.py` | **首个请求即 HTTP 500**：`assert key in self._offloading_metric_defs`（`observe()` 在每请求 record 路径上，确定性炸） | 未知 key 降级为"丢弃 + 一次性 warning"，不再 assert |
| 14 | `v1/kv_offload/cpu/gpu_worker.py` | 抢占换出时 `cuMemcpyBatchAsync failed at index 8 with error 1` → Worker_PP1 死 → EngineDead，只知道第 8 个块失败 | 加 `_kvoff_c3_ranges/_kvoff_c3_dump_failed_batch`：崩溃前把每个块的 src/dst 地址区间、是否落在已注册缓冲内 dump 出来再原样上抛 |
| 13 / 15 | `cpu/common.py`、`cpu/manager.py` | 只有 `..._usage_perc`（在途传输占比），看不出档位到底有没有被填 | 新增 `vllm:kv_offload_cpu_cache_fill_perc` = 已分配块/总块 |

> ⚠️ **诚实的现状说明**：D 组把"能不能开、会不会炸、命中不命中"三层都修通了，但**该功能仍属高风险**。
> 已知未解：PP2 下超长请求把 KV 顶到 ~89% 触发 preemption 时，`submit_store → swap_blocks_batch` 会报
> `CUDA_ERROR_INVALID_VALUE`（补丁 14 就是为了把这个错误看清楚而存在）。
> 另外**物理内存账必须实测**：配置 64 GiB → 两 rank 的 `/dev/zero` 映射合计 **107.5 GB ≈ 1.56×**（PP 私有 pinned + 对齐 padding）。
> 容量规则：`内存档 token ≈ cpu_bytes_to_use / 40.4 KB`，**小于 GPU 池就是负收益**（本实例 GPU 池 1,224,366 token → 至少要 47 GiB，推荐 ≥96 GiB）。
> 不想碰这些就把 `FN_KVOFF=0`，其余 15 个补丁照常工作。

---

## E 组 · 提速与观测（3 个，可选）

### 11 · `renderers/base.py` — 跳过 ~33 s 多模态 warmup
```python
if os.environ.get("VLLM_SKIP_MM_WARMUP", "") == "1":
    logger.info("%s warmup skipped (VLLM_SKIP_MM_WARMUP=1)", log_prefix); return
```
纯文本服务不必为最大图像特征做 dummy 处理。**代价**：跳过后第一次图片请求一次性付这笔开销。

### 12 · `v1/core/sched/scheduler.py` + 17 · `v1/metrics/stats.py` — 每请求实时进度
`make_stats()` 里新增 `prefill_progress`：对每个 running 请求输出
`{request_id, arrival, prompt_total, computed = max(0, num_computed_tokens − num_in_flight_tokens), cached}`，
并新增 `self._dsh_cached`（admission 时记下本地前缀命中数）。`SchedulerStats` 相应加字段（补丁 17）。
**为什么必须走调度器**：本版本 Prometheus 的 prefill 指标都是**prefill 完成后才更新**，实时进度只有 `num_computed_tokens` 这一个来源。
消费方是 `dsh_vllm_logger` 插件（写 `vllm-live-prefill.jsonl`）；不接监控面板可以不打。

---

## 补丁生效判据（不靠读代码，看日志）

| 判据（日志/命令行） | 证明 |
|---|---|
| `[FN-PLE-PP] no PleOffloadLayer on this pipeline stage; PLE offload connector not created for this rank.` | 补丁 18/20 生效 |
| `[FN-PLE-INT8] n-gram table attached from /media/ll/data/ple: rows=320001536 hidden=160 int8=47.68 GiB + bf16 row scales=0.596 GiB (mmap, zero heap, ...)` | 补丁 06/10/19 生效；括号里 `(anonymous heap, non-reclaimable)` = INT8 内存堆形态 |
| `Creating offloading spec with name: CPUOffloadingSpec` | D 组补丁生效（否则启动就断言失败） |
| `[kvoff-c3] offloading stats key without metric def (dropped): ...` | 补丁 03 生效（旧版此处是 AssertionError） |
| `warmup skipped (VLLM_SKIP_MM_WARMUP=1)` | 补丁 11 生效 |
| `grep -c 'prefill_progress' .../v1/core/sched/scheduler.py` ≥ 1 | 补丁 12 生效 |
| `python3 scripts/apply-patches.py --check` 全 `patched` | **22 个补丁与仓库逐字节一致**（最可靠） |

> **改完 `.py` 必须删 `__pycache__`**（`apply-patches.py` 已自动处理）。
> 症状：加了日志一行都不出——因为加载的是旧 `.pyc`。手工清理：
> `find <vllm目录> -name __pycache__ -type d -exec rm -rf {} +`
