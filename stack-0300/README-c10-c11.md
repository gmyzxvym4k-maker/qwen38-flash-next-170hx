# KVOFF c10（悬挂自愈）+ c11（真根因修复）+ 第 2 轮验证（命中已证实 / 回载正确性未过）

> 2026-09-27。适用于 `stack-0300/`（官方 vLLM 0.30.0 + `PYTHONPATH` 运行时补丁，上游零改动）。
> 全部逻辑在 `patches-extra/dsh_kvoff_rt.py`。

---

## 0. 结论速查（先看这张表）

| 项 | 状态 |
|---|---|
| store / lookup / load 数据面 | ✅ 通（存得进、查得到、装得动，零 Xid、零断言、零熔断） |
| 真命中 | ✅ **实测有**：单次 `external_prefix_cache_hits_total` **37,168 token**（prompt 39,256 的 94.6%），`CPU_to_GPU` 单次 **1.14 GB**；累计 **177,760 token / 5.59 GB** |
| 回载内容正确性 | ❌ **未过**：同一 prompt「本地算」答案是精准的 `CODE-…`，走 CPU 档回载却退化成 `duct Register Register …` ⇒ **在修好前不要在生产开 `FN_KVOFF=1`** |
| 生产缺省 | `FN_KVOFF=0`（默认关，零影响） |

细节与证据见 `ROUND2-verification.md`（本轮）与 `ROUND1-…`（第 1 轮的 c10/c11）。

---

## 1. 第 1 轮：c11 真根因（命中为什么曾经恒 0）

`OffloadingWorkerMetadata.aggregate()` 被我们自己的 c6 补丁写成 `completed_jobs=dict(self.completed_jobs)`，
**丢掉 `other.completed_jobs`**。PP2 下每个 job 要累到 `pending_count == num_workers(=2)` 才算完成
（`scheduler.py:1180/1286/1407/1714` 建 job 时 `pending_count=self.config.num_workers`，1857-1861 判定），
丢一半 ⇒ `complete_store()` 永不调用 ⇒ chunk 永远 `ref_cnt=-1` ⇒ `lookup()` 恒 `HIT_PENDING`
⇒ 调度器把请求标 deferred（**永久挂死**）+ CPU→GPU 永不发生（**hits 恒 0**）。

**c11 修法**：恢复上游逐 job 求和（`merged[job] += other.completed_jobs[job]`），保留自加的
`failed_jobs` / `fuse_tripped` 通道。自检 `selftest_kvoff_c10.py` 的 G/G2/G3。

**c10（保险）**：`c10a` manager 侧对超 `FN_KVOFF_PENDING_TTL`（缺省 120 s）未 ready 的 chunk
摘除归池；`c10b` scheduler 侧对超 `FN_KVOFF_JOB_TTL`（缺省 180 s）未收尾的 job 强制
`complete_store(success=False)`。两个 TTL 设 0 关闭。

---

## 2. 第 2 轮：为什么三次窗口判「0 命中」——**探针自己的 bug**

`kvoff-c10-recall2.py` 早期版本建档请求没带 `chat_template_kwargs`、重发请求带了
`enable_thinking=False`。模板前缀进了块哈希，于是同一份文档两侧首块哈希不同，任何缓存都不可能命中：

```
dbg[lookup] req=…a073b46a i=1 g=5 key=07b781474f59bc4e24e1 -> MISS   （建档，思考开）
dbg[lookup] req=…9b47df2b i=1 g=5 key=6121b26361fd65a13659 -> MISS   （重发，思考关；同一文档）
```

**修法**：建档 / 挤池 / 重发三步的 `chat_template_kwargs` 必须完全一致。修后同一命令立刻转绿
（37,168 token 命中、1.14 GB 回载）。**教训**：凡做缓存命中实验，先把所有进 prompt 的开关固定住
（思考开关、系统提示、工具定义、salt）。

---

## 3. 本轮新增的诊断与探针（都在 `stack-0300/`）

| 文件 | 用途 |
|---|---|
| `patches-extra/dsh_kvoff_rt.py` 的 `FN_KVOFF_DEBUG=1` | key 级诊断：`dbg[store]` / `dbg[lookup]` / `dbg[sched]`（每请求最终匹配 token 数 + 分组画像） |
| `kvoff-verify.py` | **正确性判定**：同一 prompt 的「全新算 / CPU 回载」对比（本轮的核心工具） |
| `kvoff-local.py` | 对照：本地前缀缓存两次连发是否无损 |
| `kvoff-needle.py` / `kvoff-fresh.py` | 自然文本 + 验证码的正向复述校验 / 全新对照 |
| `kvoff-dbg.py` / `kvoff-tail.py` | 建档→挤池→重发的 key 对比；尾部提问是否影响前缀 key |
| `kvoff-mon.py` | 每 2 s 采样 CPU 档指标（usage/write/read/alloc/store/load/hits） |
| `kvoff-c10-window.sh` | 窗口脚本（新增 `KEEP_ANY=1`：判据不全绿也保留实例，省 6~8 分钟冷启动） |
| `kvoff-start-test.sh` | 迭代期「停→起测试档」；收尾用 `kvoff-restore-prod.sh` |

命中机制（dbg 实证）：**全注意力组的前缀链**与 **mamba/GDN 组的滑窗边界块（window=1）**
必须同时在档；调度器会把边界逐块回退（11 块 → 10 块）直到两者都成立。任一组在首块 MISS
（`_lookup_complete_chunks` 里 `num_hit_chunks == 0 → return 0`）整请求即归零。

---

## 4. ❗未过项：回载内容与本地 KV 不等价

`kvoff-verify.py`（temperature=0，同一 prompt）：

```
0-fresh          cached=0     → "CODE-VF519-3910"                 ✅
1-reload(cpu)    cached=29088 → "duct Register Register …"        ❌
2-reload(no-ext) cached=29088 → "duct Register Register …"        ❌（第 1 步装载的块已被本地缓存接住）
```

`kvoff-local.py`（本地前缀缓存同样命中 29,088）：三次全部精准 `CODE-…` ✅
⇒ 本地缓存路径无损，**问题在 KVOFF 的 CPU→GPU 回载**。

已排除（三种配置复现同样乱码）：

- c8 公共区（前缀和布局）+ c7 Triton 回载 → 乱码
- c8 公共区 + **C++** 回载（`FN_KVOFF_LOAD_CPP=1`）→ 乱码
- **私有 pinned**（`FN_KVOFF_SHARED=0`）+ C++ 回载 → 乱码

⇒ 不是 c7 内核、不是 c8 偏移（c8/c9 此前只做过结构/字节验证，本轮补上了内容验证）。

剩余嫌疑（下一轮按序二分，用 `kvoff-verify.py` 一票判定）：

1. **c1 在 0.30.0 的分组剔除方式**：旧 chroot 栈是「剔除环形组但**保位**」；
   0.30.0 移植按「显式 group_id 索引、无需保位」——若 worker 侧按过滤后下标取值，
   组 5 的数据会落到别处 ⇒ 回载错位。**验证法**：改成保位变体后重跑。
2. **QSA 环形组（group 4，`prefix_cacheable=False`）状态未回载**：本地命中时可重算补齐，
   外部装载路径可能没有 ⇒ 长前缀的 QSA 上下文错。验证法：只回载短于一个 QSA 周期的请求，
   或放宽整除断言把该组也纳入 offload。
3. 回载后的 `num_computed_tokens` 记账与实际写入块数是否一致（c10/c11 自愈是否误伤）。

---

## 5. 怎么用（部署机 ll）

```bash
# 测试档（65536 上下文 + 80 块池 + 64 GiB 内存档）
FN_OVERRIDES_FILE=/home/ll/deploy/kvoff-c10-window.overrides SUDO_PASS=**** \
  bash /home/ll/deploy/kvoff-start-test.sh
# 正确性一票判定
/home/ll/vllm-env/bin/python /home/ll/deploy/kvoff-verify.py
# 收尾：回生产（1M/YaRN×4 + block1616 + MTP4 + KVOFF=0，并恢复看门狗）
bash /home/ll/deploy/kvoff-restore-prod.sh
```

排障开关（缺省关）：`FN_KVOFF_DEBUG=1`（key 级诊断）、`FN_KVOFF_LOAD_CPP=1`（回载走上游 C++ 路径）、
`FN_KVOFF_PENDING_TTL` / `FN_KVOFF_JOB_TTL`（自愈 TTL，0 = 关）。
