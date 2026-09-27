# 18420 内存二级缓存（KVOFF）：命中已证实、**回载内容正确性未过**（2026-09-27 第 2 轮）

> 承接 `ROUND1-c10-hang-evidence.md`。第 1 轮定位并修掉了「store 做完却永不算 ready」的
> c11 聚合 bug；第 2 轮把「到底有没有命中」查清（**有**，且量不小），同时查出一个
> **必须修掉才能上生产的问题：回载进来的 KV 与本地算出来的不等价**。

---

## 0. 一句话结论

| 项 | 结论 |
|---|---|
| 命中有没有？ | **有**。定向探针实测 `external_prefix_cache_hits_total` 单次 **37,168 token**（占 39,256 token prompt 的 **94.6%**），`CPU_to_GPU` 单次 **1.14 GB**；整轮累计命中 **177,760 token**、CPU→GPU **5.59 GB**（GPU→CPU 20.2 GB） |
| 为什么 11:5x~13:1x 三次窗口判「0 命中」？ | **探针自己的 bug**：建档请求没带 `chat_template_kwargs`、重发请求带了 `enable_thinking=False`，模板前缀进了块哈希 ⇒ **两侧首块哈希根本不同**，任何缓存都不可能命中（详见 §2） |
| 能上生产吗？ | **不能**。回载 29,088 token 后模型输出退化成 `duct Register Register …`，而**同一 prompt 全新计算 / 本地前缀缓存命中都是精准答案**（`CODE-…`）⇒ 回载没有还原出等价 KV（详见 §4） |
| 生产态 | `FN_KVOFF=0`（默认关），窗口/实验实例已恢复生产档（1M YaRN×4 + block1616 + MTP4 + 无 `--kv-transfer-config`） |

---

## 1. 第 2 轮做了什么（工具与方法）

除第 1 轮的窗口脚本外，本轮新增**运行时 key 级诊断**与四个定向探针：

| 工具 | 作用 |
|---|---|
| `patches-extra/dsh_kvoff_rt.py` 的 `FN_KVOFF_DEBUG=1` | 打印 `dbg[store]`（存了哪些 key、按组计数）、`dbg[lookup]`（每个 key 查出来 HIT/HIT_PENDING/MISS）、`dbg[sched]`（每请求最终匹配到多少 token + 分组画像）。**这是把「存了但从不命中」拆成「key 不一致」与「policy 里没有」的唯一手段。** |
| `kvoff-dbg.py` | 建档 A → 挤池 B×N → 重发 A，打印每步响应 id，用来把日志里的 dbg 行按请求对上 |
| `kvoff-tail.py` | 同一前缀 + **不同尾部提问**，验证「尾部不影响前缀块的 key」 |
| `kvoff-needle.py` | 自然文本 + 埋验证码，重发时要求复述（正向内容校验） |
| `kvoff-verify.py` | **正确性判定**：同一 prompt 的「全新算」「CPU 回载」「禁用外部加载」三态对比 |
| `kvoff-local.py` | 对照：同一 prompt 连发两次（第二次走**本地**前缀缓存）是否无损 |
| `kvoff-mon.py` | 每 2 s 采样 CPU 档指标（usage/write/read/alloc/store/load/hits） |
| `kvoff-start-test.sh` | 迭代期「停→起测试档」脚本（不回滚）；收尾仍用 `kvoff-restore-prod.sh` |

窗口脚本新增 `KEEP_ANY=1`：判据不全绿也保留测试实例，省掉每次 6~8 分钟的冷启动。

---

## 2. 假故障根因：探针把「思考开关」在两侧设成了不同值（已修）

`kvoff-c10-recall2.py` 早期版本：

```python
r  = ask(..., "Reply with the single word OK.", 8)              # no_think=False → 不带 chat_template_kwargs
r2 = ask(..., q, a.answer_tokens, no_think=not a.think)          # no_think=True → enable_thinking=False
```

vLLM 的块哈希把**模板前缀**算进去，于是同一份文档在两次请求里首块哈希不同。日志铁证：

```
dbg[lookup] req=...a073b46a i=1 g=5 key=07b781474f59bc4e24e1 -> MISS      （建档，开思考缺省）
dbg[lookup] req=...9b47df2b i=1 g=5 key=6121b26361fd65a13659 -> MISS      （重发，关思考；同一文档！）
dbg[store]  req=...9b47df2b n=5 groups=[(5,5)] first=6121b26361fd65a13659  （它的首块就是 6121…）
```

两次窗口（12:42、12:53）判「0 命中」全是这条造成的。**修法**：建档/挤池/重发三步的
`chat_template_kwargs` 必须完全一致（修后同一命令立刻转绿，见 §3）。
教训：**凡做缓存命中实验，先固定所有会进 prompt 的开关**（思考开关、系统提示、工具定义、salt）。

---

## 3. 命中是真的（已修探针后的实测）

### 3.1 修好后第一次跑（`kvoff-c10-recall2.py --tokens 20000 --flush 3 --flush-tokens 30000 --max-prompt-tokens 40000`）

```
[recall2] 建档 39256 tok（验证码 CODE-RC253-0000）
[recall2] pass1: ext_hits=37168.0 cpu_to_gpu=1139158016.0 cached_local=37168 recalled=False (10.8s, finish=length)
```

- `external_prefix_cache_hits_total` **+37,168 token** = 23 个 1616-token 块（prompt 39,256）
- `kv_offload_total_bytes_total{CPU_to_GPU}` **+1,139,158,016 B（1.14 GB）**
- `dbg[sched] -> 37168 async=True`：调度器确实把这批 token 交给 connector 异步装载

### 3.2 机制（dbg 行给出的完整链路）

```
dbg[lookup] req=…ab4819 i=1..11 g=5 key=… -> HIT ×11        （全注意力组的前缀链命中）
dbg[lookup] req=…ab4819 i=12 g=0 key=ffb9ffac… -> MISS       （mamba 组的滑窗（window=1）边界块不在档里）
dbg[lookup] req=…ab4819 i=13..16 g=0..3 key=e5e53ec7… -> HIT （沿滑窗往回找到可用边界）
dbg[sched]  req=…ab4819 prompt=18694 computed=0 -> 16160     （收敛到 10 块 = 16,160 token）
```

要点：**命中要求「全注意力组的前缀链」与「mamba/GDN 组的滑窗边界块」同时在场**；
调度器会把边界逐块回退直到两者都成立（11 块 → 10 块）。任何一个组在首块就 MISS
（`_lookup_complete_chunks` 里 `num_hit_chunks == 0 → return 0`），整请求就归零。

### 3.3 其他实测（同一实例，累计）

| 实验 | external hits | CPU→GPU | 说明 |
|---|---|---|---|
| `kvoff-dbg`（A=18.7k，2×36.5k 挤池） | **16,160 token** | 562,362,368 B | 首次确认命中 |
| `kvoff-tail`（**不同尾部提问**） | 16,160 token | — | 尾部不影响前缀块 key（前缀缓存语义正确） |
| `kvoff-needle`（自然文本 + 验证码） | **29,088 token** | 917,313,536 B | 机制正常，但答案乱码 → §4 |
| 累计（实例结束前） | **177,760 token** | **5,592,619,008 B** | GPU→CPU 20.2 GB；零新增 Xid；health 200 |

---

## 4. ❗未过项：回载进来的 KV 与本地计算**不等价**

### 4.1 判定实验（`kvoff-verify.py`，temperature=0，同一 prompt）

```
0-fresh          prompt=31918 cached=0     finish=stop    text="CODE-VF519-3910"   ✅ 精准
1-reload(cpu)    prompt=31918 cached=29088 finish=length  text="duct Register Register Register …"  ❌
2-reload(no-ext) prompt=31918 cached=29088 finish=length  text="duct Register Register Register …"  ❌
```

- 唯一变量是「这 29,088 token 是从 CPU 档装进来的」。
- 第 2 步 `kv_load_tiers: []` 仍乱码，是因为第 1 步装载进来的块已经进了**本地**前缀缓存，
  第 2 步命中的是那份被污染的本地副本（并非禁用外部加载无效）。

### 4.2 对照实验（`kvoff-local.py`）：本地前缀缓存同样 29,088 命中 → **输出正确**

```
1-fresh          cached=0     text="CODE-LC379-4770"  ✅
2-local-reuse    cached=29088 text="CODE-LC379-4770"  ✅
3-local-reuse    cached=29088 text="CODE-LC379-4770"  ✅
```

⇒ 本地缓存路径无损；**问题出在 KVOFF 的 CPU→GPU 回载**。

### 4.3 已排除的假设（三种配置都复现同样乱码）

| 配置 | 结果 |
|---|---|
| c8 公共区（前缀和布局）+ c7 Triton 回载 | 乱码 |
| c8 公共区 + **C++ 回载**（`FN_KVOFF_LOAD_CPP=1`） | 乱码 |
| **私有 pinned 缓冲**（`FN_KVOFF_SHARED=0`）+ C++ 回载 | 乱码 |

⇒ 不是 c7 的 Triton 内核、也不是 c8 的共享区偏移（**c8/c9 之前只做过结构/字节验证，
从未做过内容验证**——这次补上了，结论是「不是它们」）。

### 4.4 第三批定位（13:5x–14:5x，本轮新增）

**① 尺寸扫描 `kvoff-scale.py`：只要从 CPU 档装 ≥1 块就乱码，与规模无关**

| 目标尺寸 | 实际 prompt | 重发 cached（外部装载块数） | 答案 |
|---|---|---|---|
| 2,000 | 1,173 | 0（不足一块，未走档） | ✅ `CODE-SC02-5679` |
| 6,000 | 3,379 | 1,616（1 块） | ❌ 乱码 |
| 12,000 | 6,780 | 4,848（3 块） | ❌ 乱码 |
| 20,000 | 11,244 | 8,080（5 块） | ❌ 乱码 |

⇒ 不是「QSA 环形周期/状态累积」类规模效应（1 块就坏）。
**本地对照**：同一 3,418-token prompt 连发两次（第二次本地命中 1,616）→ 三次全部精准
⇒ 本地路径 1 块复用完全正常，问题确实只在 CPU 档装载。

**② `dbg[xfer]`（新增诊断）证明 store/load 的指针算术是对称的**

```
#1..#4 store group_sizes=[1,0,0,0,0] / [0,1,0,0,0] / [0,0,1,0,0] / [0,0,0,1,0]  dst_ids=[0]/[1]/[2]/[3]
#5     store group_sizes=[0,0,0,0,1]                                             dst_ids=[4]
#1     load  group_sizes=[1,1,1,1,1]  src_ids=[0,1,2,3,4]  dst_ids=[6,5,28,40,17]
       load  num_bytes=86,525,952(rank0) / 76,516,352(rank1) ≈ 163MB（与指标 delta 一致）
```

即：host 行 = 每（逻辑块 × 组）一行，store 按组各写一行，load 恰好读回那 5 行，
字节数与 store 对称，行步长 = row_stride 87.4MB（c8 前缀和布局）自洽
⇒ **不是偏移错位 / 布局冲突**。

**③ 又排除两项**

- 强制 `supports_partial_tail=True`（新增实验开关 c13，`FN_KVOFF_FORCE_PARTIAL_TAIL=1`）
  后现象不变 ⇒ 部分尾块协同不是根因。
- host 行指纹（`host_fp`）：store 入队那一刻 host 行全 0（异步拷贝尚未发生，符合预期），
  而 load 读之前指纹非 0 ⇒ **数据确实落到 host 行了**。

**④ 当前第一嫌疑（有具体证据）：mamba/GDN 状态的「边界 key」与「状态源块」差一块**

证据链：请求 3,416 token（mamba 对齐边界 = 3,232 = 第 1 块末）；mamba 组的 store 作业是
`src_ids=[1]`（GPU 侧第 1 个 mamba 块 = 3,232 处的状态）；而重发时调度器在 **chunk 0**
（1,616）命中该 key，据此只装 1 块（`cached=1616`）⇒ **注意力 KV 是 1,616 的、GDN 状态是
3,232 的**，错位一块。GDN 是线性注意力，状态错一块等于上下文全错 ⇒ 复读乱码，与「任何
长度都乱码」「本地路径正常」两个现象同时吻合。

确认方法（下一轮第一步）：在 `_build_aligned_boundary_store_jobs` 里打印
`boundary_tokens` / `hash_idx` / `src block id` 三元组，看 key 与源块是否差一块；
若确认，修点在该函数的 key↔块配对（或 `_make_boundary_key` 的 `hash_idx = boundary//tph - 1`）。

### 4.5 原先的嫌疑清单（保留供对照）

1. ~~c1 分组剔除方式~~：dbg[xfer] 显示 store/load 布局自洽，此项降级。
2. ~~QSA 环形组状态未回载~~：1 块即坏，与环形周期无关，此项降级。
3. c10/c11 自愈路径误伤：仍未完全排除（但窗口内 `reaped=0`）。

---


1. **c1 在 0.30.0 的分组剔除方式**：旧 chroot 栈的 c1 是「剔除环形组但**保位**」
   （`group_sizes.append(0)` 占位），0.30.0 移植时判断「按显式 group_id 索引、无需保位」。
   若 worker 侧 `layer_refs_per_group` 是**按过滤后列表的下标**取值，而 KV 缓存缓冲是
   按真实 group_id 布局，则组 5 的数据会落到别处 ⇒ 回载内容错位。**验证法**：把 c1 改成
   「保留位置、置空占位」的变体后重跑 `kvoff-verify.py`。
2. **QSA 环形组（group 4，`prefix_cacheable=False`）的状态未回载**：本地命中时它的状态
   由「命中后重算」补齐，外部装载路径可能没有这一步 ⇒ 长前缀的 QSA 注意力上下文错。
   验证法：把该组也纳入 offload（`block_size=8` 的整除断言需要放宽），或对比「只回载
   短于一个 QSA 周期（8 块）」的请求是否正常。
3. 回载后的 `num_computed_tokens` 记账与实际写入的块数是否一致（c10/c11 自愈路径是否误伤）。

---

## 5. 生产状态与回滚

- 生产**保持** `FN_KVOFF=0`（inner 缺省就是 0）；收尾用
  `bash /home/ll/deploy/kvoff-restore-prod.sh` 裸启（1M/YaRN×4 + block1616 + MTP4 + 无
  `--kv-transfer-config`），恢复看门狗 `fnx-18420-watchdog.timer`。
- 测试档（65536 上下文 + `--num-gpu-blocks-override 80` + 64 GiB）需要时：
  `FN_OVERRIDES_FILE=/home/ll/deploy/kvoff-c10-window.overrides SUDO_PASS=… bash /home/ll/deploy/kvoff-start-test.sh`
- 代码侧新增两个**排障开关**（缺省关，不影响生产）：
  `FN_KVOFF_DEBUG=1`（key 级诊断）、`FN_KVOFF_LOAD_CPP=1`（回载走上游 C++ 路径）。
- 备份：`patches-extra/dsh_kvoff_rt.py.bak-c12dbg-0927-1305`（加诊断前）。

## 6. 结论（给使用者的三句话）

1. **内存二级缓存的引擎侧已经真正工作**：存、查、装、命中、字节数、自愈、零 Xid 全部实测通过，
   单次命中可达 94.6% 的长前缀，说明 c1/c2/c6/c7/c8 + c10/c11 这套移植是通的。
2. **但回载的数据目前不等价于本地 KV**（唯一变量实验已实锤），表现形式是输出退化；
   在此之前**不能在生产开 `FN_KVOFF=1`**。
3. 下一轮只做一件事：先确认 §4.4④ 的「mamba 边界 key ↔ 状态源块 错位一块」——
   在 `_build_aligned_boundary_store_jobs` 打印 `boundary_tokens / hash_idx / src block id`，
   确认后修 key↔块配对（或 `_make_boundary_key` 的 `hash_idx`），再用 `kvoff-verify.py`
   一票判定（fresh 正确 **且** 回载正确 = 通过）。

---

## 7. 第 4 轮（设备级同步指纹）：store 无罪、load 不落地、边界嫌疑被推翻

本轮把指纹改成 **拷贝前/后各打一次 + `torch.cuda.synchronize()`**（v1 用 `self.wait([job_id])`，
事件尚未注册时它立即返回 ⇒ 读到的是拷贝前的旧值，之前的「GDN 行全 0」是这么来的假象；
另修正了多组 job 的 id 取法：第 gi 组要用**第 `pos` 个** id 而不是一律 `ids[0]`）。

### 7.1 store（GPU→CPU）：逐组字节精确 ✅

```
#1 store g=0 src/src'=2305843009213680610/2305843009213680610 dst/dst'=0/2305843009213680610 dst_changed=True src==dst'=True
#2 store g=1 src=1241133990970385820 dst'=1241133990970385820  src==dst'=True
#3 store g=2 src=1837575868171699818 dst'=1837575868171699818  src==dst'=True
#5 store g=4 src=1859139792594929026 dst'=1859139792594929026  src==dst'=True
```

每个组都是「dst 由 0 变成 src、且等于 src」⇒ **store 的源选择与行布局都对**。

### 7.2 load（CPU→GPU）：目标块内容 ≠ 源行内容 ❌

```
#1 load g=0 src=2305843009213680610(源行=store 写进去的值) dst'=1783450266022643288  src==dst'=False
```

即使 `FN_KVOFF_LOAD_CPP=1`（回载走**上游 C++** 而不是我们 c7 强制的 Triton）也同样不落地。
⇒ **回载没有把数据写进"模型会去读的那些块"**；这与「本地缓存命中=正确答案、CPU 档回载=乱码」
完全吻合（本地命中不需要这次拷贝）。

### 7.3 「mamba 边界错位」假设被推翻 ✅（重要）

新增 `dbg[boundary]` 打印 `_build_aligned_boundary_store_jobs` 的 hand-off 三元组：

```
req=…b45fb729 entries=[(0, 1, 6464), (1, 6, 6464), (2, 11, 6464), (3, 16, 6464)]   # 8469 token prompt
req=…b3f7b575 entries=[(0, 55, 58176), (1, 56, 58176), (2, 57, 58176), (3, 58, 58176)]  # ~61k 挤池文档
```

每个请求每组**只有一个边界**，且就是「最后的 1616 对齐位置」（8,469→6,464；61k→58,176）；
而重发时装载量 `cached` 恰好等于该边界（`cached=6464`）。
⇒ **mamba 状态与注意力前缀是同一边界**，「状态比前缀靠前/靠后」的错位假设不成立。

### 7.4 下一轮唯一要做的实验

把 load 的**目标侧**查清：在 `update_state_after_alloc` 里打印
`(group_idx, dst_block_ids, block_indices, num_external_tokens)`，并与请求实际的
block table 对照——判定是「拷贝写错位置」还是「填的块不是模型要读的块」。
辅助实验：用**自然句填充**（不要 `x x x`，会让模型退化成复读）把 prompt 补到
恰好 `1616*N+1`，使回载覆盖 99.98% 的 prompt（无重算尾部），看输出是否恢复连贯——
用于区分「整体装载错」与「装载/重算交界错」。

### 7.5 第 5 轮补充（整行指纹 + canonical 判定 + 两个新开关）

1. **指纹从「每张量前 2KB」扩到「整行」**，store 侧结论升级为：**整行**逐组字节精确
   （`head_ok=True` 且 `full_ok=True`），g=0/1/2/4 全中 ⇒ store 无罪更硬了。
2. **布局判定**：`dbg[xfer] layout: canonical=False src_bpc=1 dst_bpc=1`
   ⇒ 本机走的是 **direct**（非 canonical）拷贝计划，`is_writer` 轮转/规范页编号**不参与**
   ⇒ 「c1 剔除环形组扰乱 canonical page 编号」的假设**排除**。
3. **c1 剔除是必需的**：新增 `FN_KVOFF_NO_C1=1` 做对照，不做剔除时引擎直接拒启
   （`AssertionError: tokens_per_block=8 not divisible by tokens_per_hash=1616`）。
4. **回载换上游 C++ 路径（`FN_KVOFF_LOAD_CPP=1`）依然乱码** ⇒ 不是我们 c7 强制 Triton 的锅。
5. **测量本身的局限（必须记下）**：load 侧「目标块内容 ≠ 源行」的读数**不可靠**——
   拷贝前后我读目标块时，同一块可能正被 compute 流/后续 job 写（`torch.cuda.synchronize()`
   只保证"等到此刻为止"，之后的写会把读数改掉）；而且哪几个组"对上"逐轮不同（g=0/1/2/3/4
   随机组合），这是竞态特征而非确定性 bug。⇒ **判定回载正确性不能再依赖这个指纹**，
   要回到"输出是否正确"这个地面真值，或改成读"拷贝完成后立刻"的单点快照（在 worker 的
   `get_finished()` 里读）。
6. **下一轮的两个实验（按序）**：
   a) 在 `get_finished()`（拷贝完成事件已 query 通过）里对刚完成的 job 打一次指纹，
      避开并发写；这能给出 load 是否真的把行内容写进目标块的**确定性**答案；
   b) 自然句填充把 prompt 补到恰好 `1616*N+1`，让回载覆盖 99.98%（无重算尾部），
      判断问题是"整体装载错"还是"装载/重算交界错"。

### 7.6 第 6 轮：把快照挪到"拷贝完成后"，缺陷锁定到 **mamba/GDN 状态回载**

1. **新诊断 `dbg[xferDONE]`**：在 worker 的 `get_finished()`（`end_event.query()` 已通过）里
   对刚完成的 job 打一次快照，避开"拷贝还没跑完"的读。结果：

```
#8 load g=4 head src=1670906557403536246 dst_pre=858775443939331979 dst_post=1670906557403536246 | head_ok=True full_ok=True
#8 load g=4 head src=1004204176743227098 dst_pre=604128206416349686 dst_post=1004204176743227098 | head_ok=True full_ok=True
#8 load g=0/1/2 … head_ok=False full_ok=False（逐轮不同组合）
```

   ⇒ **全注意力组（g=4）的回载逐次字节精确**；mamba/GDN 组（g=0/1/2）读数不稳定。
2. **为什么不稳定的原因找到了**：mamba 状态块是**活的**——请求一进入 decode 就每步更新它，
   所以我"拷贝完成后"读到的已经是模型后续写进去的状态，不是回载进去的那份。
   ⇒ 这类块**无法用事后快照验证**（g=4 的注意力 KV 是只读的，所以稳定且对得上）。
3. 由此得到的确定结论：
   * store（含 mamba 组）**字节精确** ✓（store 方向行是静态的，快照可信）；
   * 回载的**注意力 KV 字节精确** ✓；
   * ⇒ **缺陷只在 mamba/GDN 状态的恢复环节**（要么回载没落到模型读的状态块，要么
     CoW hand-off 给的状态不是"该边界的状态"）。
4. **精确长度实验被证伪**：`kvoff-exact2.py` 用随机词填充把 prompt 补到 `1616*N+1`
   （实测 prompt 6,467 / tail 3 token），但**连"全新请求"那一问都退化成复读**
   ⇒ 这类填充 prompt 本身就超出模型的稳定区，"答案对不对"不能再当判据
   （自然文本的 kvoff-needle/scale 才是可用判据：fresh 正确、回载乱码）。
5. **下一轮两个候选（按性价比）**：
   a) **V2 runner + `--mamba-cache-mode all`**：0.30.0 源码里 `mamba cache mode 'all'`
      被列为 V1 runner 的 unsupported feature（`config/vllm.py:2878`）⇒ all 模式需要
      `VLLM_USE_V2_MODEL_RUNNER=1`。all 模式给**每个块边界**做状态 checkpoint，
      hand-off 就能覆盖"装载边界"这个位置（当前 align 模式每请求只有 1 个边界）。
      注意本模型可能 `supports_mamba_prefix_caching=False`（会静默降级回 align），
      需先确认/补丁。
   b) 调度器侧打印 `(group_idx, dst_block_ids, block_indices, num_external_tokens)`
      并在随后几步打印该请求 mamba 组的实际状态块 id，判定"回载块 ≠ 模型读的块"。

---

## 8. 【重要发现·第 7 轮】0.30.0 有**两条** KV 卸载实现，我们一直在调的是旧的那条

用户问「0.30.0 是不是不支持内存二级缓存」——**支持，而且有两条**：

| 实现 | 启用方式 | 代码 | 对 hybrid/mamba 的处理 |
|---|---|---|---|
| `OffloadingConnector`（经典，我们移植/调试的那条） | `--kv-transfer-config '{"kv_connector":"OffloadingConnector",...}'` | `v1/kv_offload/cpu/` | 自定义分组/chunk + CoW hand-off；**对本模型的 mamba 状态回载不成立**（§7 的全部结论） |
| **`SimpleCPUOffloadConnector`**（新） | `VLLM_USE_SIMPLE_KV_OFFLOAD=1` + `--kv-offloading-size <GiB>`（`--kv-offloading-backend native\|lmcache`，缺省 native） | `v1/simple_kv_offload/` + `v1/.../simple_cpu_offload_connector.py` | **`class SimpleCPUOffloadConnector(KVConnectorBase_V1, SupportsHMA)`**；manager 里**显式**处理 `MambaSpec`（“keep their own block size”）、空块（“sliding window or mamba padding”）、hybrid 联合查找与边界；建在核心 BlockPool/KVCacheCoordinator 之上 |
| 另外：`v1/simple_kv_offload/disk_backend.py` | 由 `disk_capacity_bytes>0` 启用（`kv_event_medium=MEDIUM_STORAGE`） | 同上 | **自带 NVMe 磁盘层**：独立 store/load IO 线程 + pinned staging buffer（用户提的“硬盘方案”在这条路径上是内建的） |

⇒ 下一轮的正确实验是 **换成 `SimpleCPUOffloadConnector`**（`VLLM_USE_SIMPLE_KV_OFFLOAD=1 --kv-offloading-size 64`），**不用我们那套 rt-patch #9**：
判据 = 同一 prompt「本地算 vs 中途 flush 后再算」答案是否都正确（经典连接器就死在这一条）+ `SupportsHMA` 是否让 mamba 状态按块正确回载。
