# 04 · PLE n-gram 表：INT8 量化与四种驻留形态

## 1. 这张表是什么

第 2 层（`ple_layer_ids=[2]`）除了常规注意力，还有一张 **n-gram 嵌入表**：
以最近 `ngram_size=3` 个 token 组成的 n-gram 为下标去查一行向量，加到隐状态上。

| 项 | 值 |
|---|---|
| 形状 | `[320,001,536 × 160]` |
| dtype | BF16 |
| 体积 | **95.37 GiB**（checkpoint 里是 102.4 GB 的 `model-00016-of-00017.safetensors`，含 128 个 `[2500012,160]` 张量首尾相接） |
| 每 token 访问量 | 10 行 × 160 B ≈ **1.6 KB** |
| 查表位置 | **CPU 侧** `PleOffloadWorker` 的 `torch.index_select`（不是自定义 op、不是 triton） |

官方手册要求这张表**常驻 CPU 内存**（≥128 GB RAM），并明言"放 SSD mmap 会慢到不可用"。
生产机内存不足以按手册来，于是有了本仓库的两项自主决策：**离线 INT8** + **精度×位置四态**。

## 2. 为什么选 INT8 而不是 FP8（实测，不是猜）

全表 `max|x| = 0.0447`，`|x|` 的动态范围只有约 **480×**。
FP8-e4m3 有 4 位指数——在这么窄的范围里，指数位**纯属浪费**，省下来的比特不如给尾数。

同一遍扫描的实测误差（全表 3.2 亿行，不是抽样）：

| 方案 | relMSE | 行余弦均值 | 体积 |
|---|---|---|---|
| **INT8 + 逐行 scale** | **4.36e-05** | **0.9999784** | 48.28 GiB |
| FP8 e4m3 | 6.70e-04 | — | 47.7 GiB |
| BF16（基准） | 0 | 1 | 95.37 GiB |

**INT8 比 FP8 精确 15 倍，体积还一样**——没有理由选 FP8。

量化方案（与引擎侧反量化严格互逆）：

```
scale = bf16(row_absmax / 127)                    # 先 fp32 计算，再落 bf16
q     = clamp(round_half_even(x / fp32(scale)), -127, 127) → int8
反量化（CPU worker 内）:  out_bf16 = q.to(bf16) * scale.to(bf16)
```

三个必须一致的细节：
1. **除法用的是"已落 bf16 的 scale"**，不是原始 fp32 商——否则量化值与产物字节不一致（`--verify-only` 会报 mismatch）。
2. `row_absmax == 0` 时 `scale=0, q=0`（本表实测 0 行，但代码必须有这个分支，否则除零出 NaN）。
3. 反量化在 **CPU worker 里做完再送 GPU**，所以 **GPU 侧 IPC 契约仍是 bf16**：
   显存占用、CUDA 图、跨卡通信与 BF16 路径**逐字节相同**。这是它能"无痛上线"的原因。

## 3. 生成产物

```bash
python3 scripts/quantize_ple.py \
  --model /media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound \
  --out   /media/ll/data/ple
# 约 8 分钟（实测 486 s），纯 CPU，可断点续传（_state.json）
```

产物：

| 文件 | 字节 | 内容 |
|---|---|---|
| `ple_ngram_int8.bin` | 51,200,245,760 | int8 `[rows, hidden]` row-major |
| `ple_ngram_scale.bin` | 640,003,072 | bfloat16 `[rows]` |
| `ple_ngram_meta.json` | — | 几何、量化方案、**全表误差**、抽样复核 |
| `_state.json.done` | — | 完成后改名，脚本据此拒绝重跑 |

脚本在写完后会**自动抽样 10 万行重算并逐字节比对**，不一致就返回非零。
随时可以只校验（不写文件）：

```bash
python3 scripts/quantize_ple.py --model <同上> --out /media/ll/data/ple --verify-only --rows 20000
# 期望：int8_mismatch=0 scale_mismatch=0
```

**safetensors 偏移（本仓库最贵的一个知识点）**：`data_offsets` 是**相对数据段起点**的，
数据段前面还有一个 8 字节小端长度 + JSON header。所以：

```
table_start_in_file = 8 + header_len + data_offset      # 本模型：8 + 20728 = 20736
```

漏加的后果是整表错位 10368 个元素，症状为**"内容词全对、只有标点乱"**（`docs/08#P13`）。

## 4. 四种驻留形态（精度 × 位置）

由 `FN_PLE_INT8`（精度）与 `FN_PLE_LOC`（位置）两个**正交**开关组合：

| 形态 | 引擎 env | 内存代价 | 磁盘 I/O | 启动 | decode |
|---|---|---|---|---|---|
| **INT8 + 放硬盘**（disk） | `VLLM_PLE_DISK_RESIDENT=1` + `VLLM_PLE_INT8_DIR=…` | 48.3 GiB **可回收页缓存** | 冷启动读、驻留后≈0 | ~3.5 min | 87~93 tok/s |
| **INT8 + 放内存**（heap） | 再加 `VLLM_PLE_INT8_MEMORY=1` | 49.2 GiB **匿名堆，不可回收** | 0 | ~6 min | 87~93 tok/s |
| BF16 + 放硬盘 | `VLLM_PLE_DISK_RESIDENT=1`（无 INT8_DIR） | 95.4 GiB 可回收页缓存 | 大 | 慢 | 与 INT8 持平（表驻留时） |
| BF16 + 放内存 | 都不设（旧匿名堆路径） | **95.4 GiB 不可回收** | 0 | ~9 min | 87~93 tok/s |

生产当前用 **INT8 + 放内存**（`FN_PLE_INT8=1 FN_PLE_LOC=heap`），因为这台机器内存够（157~220 GB）
且要把页缓存留给权重文件。判据（看引擎日志，不要猜）：

```
[PleOffloadWorker ...] [FN-PLE-INT8] n-gram table attached from /media/ll/data/ple:
  rows=320001536 hidden=160 int8=47.68 GiB + bf16 row scales=0.596 GiB
  (mmap, zero heap, dequantised in the CPU worker)          ← 磁盘形态
  (anonymous heap, non-reclaimable, ...)                    ← 内存形态
```

兼容性铁律：`FN_PLE_LOC` **未传**时，`FN_PLE_INT8=0` 视作 heap、`=1` 视作 disk
（保证旧命令/旧预设的语义不变）。

## 5. 收益与代价（实测）

| 指标 | BF16 匿名堆 | INT8 |
|---|---|---|
| decode（短上下文热态） | 26 tok/s（表被换出时）→ 87~93（全驻留） | **87~93 tok/s（持平）** |
| 冷启动 | 555~630 s | **210 s** |
| PP 权重加载 | 516 / 548 s | **48 / 74 s**（页缓存不被表挤掉） |
| 服务期读盘 | 有缺页 | **≈0 MB/s**，majflt≈0 |
| 内存 | 95.4 GiB 不可回收 | 48.3 GiB（可回收或匿名，看形态） |

INT8 的真实收益不在"查表更快"（每 token 才 1.6 KB，带宽根本不构成瓶颈），
而在**省下的 47 GiB 内存让权重文件能常驻页缓存**，从而把加载与 warmup 全提速。
参见 `docs/08#P27`（为什么 BF16 档预热无效）与 `docs/08#P29`（为什么带宽与本题无关）。

## 6. 回滚

```bash
FN_PLE_INT8=0 FN_PLE_LOC=heap bash /home/ll/deploy/start-flash-next-w4a16.sh   # 退回 BF16 匿名堆
```
产物目录 `/media/ll/data/ple` 留着不碍事（没有它，INT8 分支会自动回落 BF16 mmap 并打 warning）。
