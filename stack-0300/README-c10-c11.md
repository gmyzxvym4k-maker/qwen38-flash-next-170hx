# KVOFF c10（悬挂自愈）+ c11（真根因修复）—— 内存二级缓存第一次真正出命中

> 2026-09-27。适用于 `stack-0300/`（官方 vLLM 0.30.0 + PYTHONPATH 运行时补丁）。
> 上游文件零改动；全部逻辑在 `patches-extra/dsh_kvoff_rt.py`。

---

## 1. 一句话

内存二级缓存（`--kv-transfer-config OffloadingConnector`）此前在本机「跑得起来、从不命中」，
根因**不在上游**，而在我们自己 c6 补丁的一行聚合：

> `OffloadingWorkerMetadata.aggregate()` 写成 `completed_jobs=dict(self.completed_jobs)`，
> **丢掉了 `other.completed_jobs`**。

PP2 下每个 job 要累到 `pending_count == num_workers(=2)` 才算完成（`scheduler.py:1180/1286/1407/1714`
建 job 时 `pending_count=self.config.num_workers`，1857-1861 判定），丢一半 ⇒ job 永远算不完
⇒ `complete_store()` 永不调用 ⇒ chunk 永远 `ref_cnt=-1` ⇒ `lookup()` 恒返回 `HIT_PENDING`
⇒ 调度器把请求标成 deferred（**永久挂死**）且 CPU→GPU 装载永不发生（**hits 恒 0**）。

## 2. 修了什么

| 编号 | 位置 | 作用 |
|---|---|---|
| **c11** | `_patch_offloading_common` 的 `aggregate()` | 恢复上游逐 job 求和语义（`merged[job] += other.completed_jobs[job]`），保留我们自己的 `failed_jobs` / `fuse_tripped` 通道。**这是「能不能命中」的那一刀。** |
| **c10a** | `_patch_cpu_manager`（新增） | `prepare_store()` 记下每个 `keys_to_store` 的时刻；`lookup()` / `get_stats()` 把超 `FN_KVOFF_PENDING_TTL`（缺省 120 s）仍未 ready 的 chunk 摘除 + 归还 chunk 池 + 扣 `_num_write_pending_chunks`（等价 `complete_store(success=False)`）。该 key 随即变 MISS，请求走本地 prefill，不再 deferred。 |
| **c10b** | `_patch_offloading_scheduler` 内 | 每步 `update_connector_output()` 清扫 `_jobs` 中超 `FN_KVOFF_JOB_TTL`（缺省 180 s）未收尾的 job：撤登记、清 `transfer_jobs` / `_block_id_to_pending_jobs`，让 `has_pending_push_work()` 能落回 False（引擎不再带空批空转）。 |

两个 TTL 可调，设 0 关闭。c10 是**保险**（即使将来再丢一次 ack 也不会永久挂死）；c11 是**治病**。

## 3. 判据与期望值（带 c11 的实测）

命令（部署机 ll）：

```bash
cd /home/ll/deploy
FN_OVERRIDES_FILE=/home/ll/deploy/kvoff-c10-window.overrides SUDO_PASS=**** \
PROBE=/home/ll/deploy/kvoff-c10-recall.py \
PROBE_ARGS='--tokens 30000 --flush 5 --gap 3 --answer-tokens 2048' \
bash /home/ll/deploy/kvoff-c10-window.sh
```

窗口配置：`FN_KVOFF=1` + `FN_KVOFF_SHARED=1` + 64 GiB + 测试档
（`FN_MAXLEN=65536` + `--num-gpu-blocks-override 80` ⇒ 引擎自报 `kv_cache_size_tokens=79437`）。

| 判据 | 期望值 | 实测（2026-09-27） |
|---|---|---|
| `external_prefix_cache_hits_total` 增量 | **> 0** | **25,856 token**（prompt 28,542 ⇒ 命中 90.6%） |
| `kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"}` 增量 | **> 0** | **828,575,744 B（828 MB）** |
| `kv_offload_total_bytes_total{transfer_type="GPU_to_CPU"}` 累计 | 数 GB | 6.72 GB |
| 验证码复述（`kvoff-c10-recall.py`） | 复述出 `CODE-…` | 见 §4 |
| `c10b: 强制收尾超时 job` 行数 | 0（修好后不应再有） |  **0**（修前 6） |
| `c10a[manager]: write_pending / oldest` | oldest 秒级 | 15/786、**1 s**（修前 105/786、45 s） |
| `num_requests_waiting{reason="deferred"}` 探针后 | 0 | **0** |
| 新增 Xid | 0 | **0** |
| 物理钉住 / 配置（`df /dev/shm`） | ∈[0.9, 1.15] | **1.00** |

## 4. 两个必须知道的坑（都是实测踩出来的）

1. **`--num-gpu-blocks-override` 有下限**：`max-model-len=65536` 时引擎要求 ≥1.42 GiB KV，
   40 块只有 0.84 GiB ⇒ **引擎直接拒启**
   （日志指纹 `To serve at least one request with the model's max seq len ... is larger than the available`）。
   本机 80 块（79,437 token）可用。
2. **窗口判据曾有两处「假 PASS」**（已在 `kvoff-c10-window.sh` 修正）：
   ① 「零 store 熔断」原先 grep 旧 chroot 栈的字符串 `FUSE: store 方向`，而现行日志是
   `[dsh-kvoff c6] worker store fuse tripped` ⇒ 无论熔断与否都判 PASS；
   ② 引擎因参数无效拒启时，不应记成「探针失败」，现单列一条
   「启动参数有效性（KV 池 ≥ max-model-len）」。

## 5. 复验（离线，不占 GPU）

```bash
cd /home/ll/deploy/vllm-0300
PYTHONPATH=$PWD/patches:$PWD/patches-extra FN_KVOFF_PENDING_TTL=1 FN_KVOFF_JOB_TTL=1 \
  /home/ll/vllm-env/bin/python selftest_kvoff_c10.py     # c10a/c10b + c11 聚合语义（G/G2/G3）
PYTHONPATH=$PWD/patches:$PWD/patches-extra \
  /home/ll/vllm-env/bin/python selftest_kvoff_rt.py      # c1/c2/c6/c7/c8 无回归
```

## 6. 回滚

- 代码：`patches-extra/dsh_kvoff_rt.py` 的 `.bak-c11-*` / `.bak-c10-*`（逐版本备份）。
- 运行态：窗口脚本任何失败都会自动 `FN_KVOFF=0` 裸启生产
  （`tools/kvoff-restore-prod.sh` 是同一条路的独立入口）。
- 生产缺省仍是 `FN_KVOFF=0`（inner 内置）；要不要常开见 `README-0300.md` §6.3 与容量评估。
