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

### 4.4 剩余嫌疑（下一轮按序验证）

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
3. 下一轮只做一件事：按 §4.4 的顺序二分（先 c1 保位变体），用 `kvoff-verify.py` 一票判定
   （fresh 正确 + 回载正确 = 通过）。
