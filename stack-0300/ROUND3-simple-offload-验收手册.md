# 18420 内存二级缓存（SimpleCPUOffloadConnector）验收手册 —— 2026-09-27

> 目标：在生产档（1M 上下文 / PP2 / MTP4 / BF16 锁页 PLE）上验证 **GPU 显存 → 宿主内存** 的
> 二级 KV 缓存真正被走到（回载命中 > 0）且回载内容正确，然后决定是否常驻生产。
> 本文只讲「怎么跑、怎么读结果、失败怎么办」；原理与历史结论见 `ROUND1/ROUND2-*.md` 与
> 交付仓库 `stack-0300/README-0300.md`。

## 0. 一句话现状（2026-09-27 晚）

| 项 | 状态 | 证据 |
|---|---|---|
| 引擎能带二级缓存启动 | ✅ | `SimpleCPUOffloadWorker [CPU]: N CPU blocks` + `SimpleCPUOffloadConnector: role` 出现在日志 |
| CPU→GPU 回载能发生且内容正确 | ✅（小规模） | 3,426 / 8,408 / 31,825 token 三档，分别从 CPU 装载 1,616 / 6,464 / 29,088 token，验证码精准复述（`recalled=true`），累计 `external_prefix_cache_hits_total` 37,168 |
| 生产档（GPU 池 1,207,262 token）下被真正走到 | ❌ 未验 | 32GiB 档 + 9 挤池窗口 `external_hits_delta=0`（构造性必然，见 §3） |
| 常驻生产的稳定性 soak | ❌ 未做 | —— |
| 经典 OffloadingConnector（c1~c11） | ⛔ 退役 | 本模型 hybrid 递归状态回载语义不成立（内容退化），详见 `ROUND2-verification.md` §7 |

## 1. 为什么用官方 simple 实现而不是我们移植的经典连接器

源码直读（0.30.0，`site-packages/vllm/`）：

- `distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py:55`
  `class SimpleCPUOffloadConnector(KVConnectorBase_V1, SupportsHMA)` —— 显式 `SupportsHMA`
  （hybrid/mamba 感知），是官方为混合模型写的实现。
- CPU 档查找直接复用**核心**的 CPU 侧协调器：`v1/simple_kv_offload/manager.py:359`
  `self.cpu_coordinator.find_longest_cache_hit(...)`，并跳过 null 块
  （`manager.py:364-365 if not blk.is_null`）。也就是说 hybrid 分组/边界/padding 的语义由
  核心 KV 管理器负责，而不是由连接器自己重推 —— 这正是经典连接器出问题的位置。
- 容量按显存比例缩放：`manager.py:272`
  `num_cpu_blocks = max(1, num_gpu_blocks * cpu_capacity_bytes // gpu_total_bytes)`。
- CPU 档内存分配（`v1/simple_kv_offload/worker.py:232-236`）：先 `torch.zeros`（非 pinned），
  再 `cuda_mem_ops.pin_tensor()` → `cudaHostRegister`。**注释说明**：绕开 PyTorch
  `CUDACachingHostAllocator` 的「向上取整到 2 的幂」（100GB → 128GB），所以日志里的
  `N CPU blocks (X GB)` 就是真实锁页量。
- 拷贝后端：`copy_backend.py:26` `DmaCopyBackend`（**后台线程**）→ `cuda_mem_ops.copy_blocks`
  → `cuMemcpyBatchAsync`。⚠ 这正是经典连接器 c7 收敛出的高危 API（PP2 + NCCL-P2P 下冻结
  compute 流）；simple 实现的差别是它在**独立 load/store 流 + 后台线程**上跑，且 store 前
  记录 compute-done 事件再排序（`worker.py` `wait_for_save` docstring，引用上游 #45704/#39306）。
  风险等级：中——需要 soak 观察，而不是假定安全。
- 命中指标是引擎原生的：`v1/metrics/loggers.py:626 name="vllm:external_prefix_cache_hits"`，
  Prometheus 输出为 `vllm:external_prefix_cache_hits_total`（实测线上存在，带 `_created`
  伴生行，统计时**只能累加 `_total` 那一行**，历史踩过把 `_created` unix 时间戳当字节的坑）。
- `--kv-transfer-config` 的 extra config 还支持 `cpu_bytes_to_use_per_rank`、`lazy_offload`、
  `kv_offload_backend`（`cpu`/`disk`）、`disk_path`、`disk_capacity_bytes`（默认 100GiB）、
  `use_page_cache` —— 将来要做「内存档 + 磁盘三级」不用改代码。

## 2. 一条命令跑窗口（含 soak）

```bash
# 前置：生产必须健康（脚本会自己门禁检查，健康才继续；正在启动则最多等 240s，仍不健康就中止）
ssh ll@<DEPLOY_HOST>
cd /home/ll/deploy
SOAK_IF_PASS=1 KVOFF_GIB=48 bash kvoff-accept.sh
```

- 停机时长：约 **15~20 分钟**（冷启动 6~8 分钟 + 探针 11 × 116k token ≈ 10 分钟）。
- `SOAK_IF_PASS=1`：探针判定 PASS 时**不恢复**生产，保持二级缓存运行进入 soak，并恢复看门狗；
  判定 FAIL 时照常恢复无二级缓存的生产。
- 结果文件：`/home/ll/deploy/kvoff-accept.result`（一路追加，抗 ssh 断线）。
- 监控：`tail -f /home/ll/deploy/kvoff-accept.result`。

前置门禁（2026-09-27 事故后加的，`preflight()`）：停服前必须 `health=200`；若 18420 进程存在
但 health≠200（＝启动中）则最多等 `KVOFF_ACCEPT_WAIT`（缺省 240s），仍不健康就 **exit 2 不动
生产**。`KVOFF_ACCEPT_FORCE=1` 可强制越过。另加 `flock` 单窗口锁。

## 3. 容量数学（先算再跑，否则白跑）

`--kv-offloading-size`（= `FN_SIMPLE_OFFLOAD`，GiB）按 `world_size` 均分到两个 PP rank。
本机实测（32GB/rank → 1483/1438 块；16GB/rank → 741/719 块；GPU 池 1,207,262 token，
block 1616，每 rank GPU 747 块）：

| 配置 | 每 rank 锁页 | CPU 块/rank | CPU 档容量 | 容量/GPU 池 | 要挤出显存需挤池 |
|---|---|---|---|---|---|
| 32 GiB | 16 GB | 741 / 719 | 1,161,904 token | **0.96×** | 结构上不可能有收益 |
| 48 GiB | 24 GB | ~1035 / ~1004 | ~1,622,464 token | 1.34× | > 1,207,262，脚本自动推荐 11 轮 |
| 64 GiB | 32 GB | 1483 / 1438 | 2,323,808 token | 1.92× | 11~18 轮都行，但锁页更多 |

- 结论：**1M 档必须 ≥48GiB 才有意义**（32GiB 档容量比 GPU 池还小，历史那次 `hits=0` 不是 bug）。
- 脚本会在探针前自检并把结论写进 result：`挤出显存 ✓/✗`、`挤完仍留在 CPU 档 ✓/✗`，
  并按 `need=ceil(池×1.05/116k)`、`max=floor(档×0.90/116k)` 自动调整 `--flushes` 轮数。
- 内存账（251GB 物理，`swap=0`）：PLE BF16 锁页 95.4GiB + 48GiB 档锁页 ≈ 143GiB 锁页，
  权重页缓存 ~79GB（可回收）+ 引擎 ~12GB ≈ 234GB。**64GiB 档请先确认内存水位**（历史上大锁页
  分配与 MCE 硬挂强相关）。

## 4. 结果怎么读

`kvoff-accept.result` 里探针会写一行终判（`--answer-tokens` 缺省 256，正文为空会自动加预算重试）：

```json
{"step": "reload", "cached": 25856, "recalled": true, "answer_tries": 1,
 "external_hits_delta": 25856.0, "flushed_tokens": 1508000,
 "gpu_pool_tokens": 1207262, "tier_exercised": true, "verdict": true}
```

判定规则（脚本已实现，见 `kvoff-accept.sh` 的 `VERDICT/HITS/EXER` 解析）：

| 组合 | 含义 | 处置 |
|---|---|---|
| `external_hits_delta > 0` 且 `recalled=true` | 二级缓存真被走到且回载无损 | ✅ PASS（soak 模式下保持运行） |
| `external_hits_delta > 0` 但 `recalled=false` | 走通了但内容不等价 → 语义/边界 bug | ⛔ 恢复生产，取证 `dbg[xfer]`/日志 |
| `external_hits_delta = 0` 且 `tier_exercised=false` | 挤池量不够，本轮无信息量 | 加大 `--flushes` 重跑 |
| `external_hits_delta = 0` 且 `tier_exercised=true` | 挤到位却零回载 → 容量/查找侧问题 | 看容量自检的「容量/池」比值 |

## 5. 失败与回滚

- 脚本任何失败分支都会后台执行 `kvoff-restore-prod.sh`（裸启生产 + 恢复看门狗 timer）。
- 手工回滚：`bash /home/ll/deploy/kvoff-restore-prod.sh`。
- **自愈安全模式**（2026-09-27 部署，`fnx-18420-watchdog.sh` 的 `selfheal_sanitize`）：看门狗
  自愈拉起时会剥离 `FN_SIMPLE_OFFLOAD` / `FN_KVOFF=1` / 含 offload 关键字的
  `FN_EXTRA_ENV`/`FN_EXTRA_ARGS` 并写日志 —— 保证「二级缓存是崩溃诱因」时不会陷入
  崩溃→重启→再崩的循环，生产会回到无二级缓存定版。要带档位跑请手工/控制台启动。
- 本次已验证：剥离逻辑在部署后的文件上 5 个用例全对（含「无关 `FN_EXTRA_ARGS` 必须保留」）、
  调用点位于 `setsid nohup` 之前、`health=200` 时手工跑看门狗不动服务。

## 6. 已知坑（都踩过）

1. `ps -eo args | grep kv-transfer-config` 会**自匹配 ssh 命令行**（假阳性，第 8 次踩）；
   判据一律用 python 直读 `/proc/<pid>/cmdline`（工具 `/tmp/fnx-argv-check.py`）。
2. `D=$(func)` 形式的命令替换里 `unset` 不生效（子壳）——日志说"已剥离"而实际没剥离；
   必须直接调用函数 + 全局变量回传（已在看门狗里修正并测出）。
3. 缓存命中实验里建档/挤池/重发三步的 `chat_template_kwargs`（`enable_thinking`）必须完全一致，
   否则块哈希不同、永远 0 命中。
4. 探针复述题若只给 64 token，思考模型可能把预算吃光 → 正文为空 → 假阴性；现已缺省 256 并
   在空正文时自动加预算重试一次。
5. 别用 `free` 判断 PLE/锁页是否常驻 —— 锁页计入 shared，判据要用 `fincore` / `smaps`。
