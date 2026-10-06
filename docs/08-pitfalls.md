# 08 · 踩坑全集

按"会不会要命"排序。每条给 **症状 / 根因 / 判据 / 处置**。编号可跨文档引用（如 `#P07`）。

> 一条总纲：**这套栈对未文档化的参数偏差零容忍**。凡是"看起来是优化"的改动（换 backend、加 env、调 batch tokens），
> 在本栈上大概率是硬断言或 GPU 崩溃。改之前先确认它有实测依据，改之后必须跑 `scripts/verify-deployment.sh`。

---

## 一、起不来 / 崩溃类

### P01 漏 `--block-size 1616`
- **症状**：建模阶段直接报错，起不来。
- **根因**：本模型两级 CSA + linear 层的 block size 不一致，必须由用户显式给一个能同时对齐的值。
- **判据**：`tr '\0' ' ' < /proc/<pid>/cmdline | grep -o -- '--block-size [0-9]*'` → 期望 `1616`。
- **处置**：照抄。不要"优化"成 1632（那是 NVFP4 路线的值，会连带改变 MTP 合法档位，见 `docs/05#C组`）。

### P02 漏 `--mamba-ssm-cache-dtype float32`
- 同 P01，另一个"漏了直接起不来"的参数。**两个必须同时给。**

### P03 `--max-num-batched-tokens 16384`
- **症状**：引擎崩，且伴随 GPU1 `Xid 31`（MMU Fault）→ 之后一切 CUDA init 报 unknown error → **唯一修复是整机重启**。
- **处置**：锁 8192。已验证走不通，别再试。

### P04 MTP 档位开成 5~8
- **症状**：加载权重后约 4.5 分钟崩：
  `AssertionError: QSA ring capacity 12 must divide the attention block size 1616`
- **根因**：`ring = 4 × cdiv(4 + k, 4)`，block=1616 时 k=1..4 → 8 ✅，k=5..8 → 12 ❌，k=9..12 → 16 ✅。
- **判据**：合法档 = `{1,2,3,4} ∪ {9,10,11,12}`；生产用 4（实测 k=1/2/3/4 = 72.1/82.2/90.4/92.7 tok/s 单调递增）。

### P05 `--moe-backend marlin`（显式写）
- **症状**：`ValueError: moe_backend='marlin' is not supported for unquantized MoE`。
- **根因**：MTP 草稿的 MoE 层是 BF16 未量化；官方 GDS 路线有 runner hook 给草稿单独切 triton，本路线没有。
- **处置**：写 `auto`。主模型 NVFP4/W4A16 仍会自动落到 MARLIN（日志 `Using 'MARLIN' NvFp4 MoE backend`），不损失性能。

### P06 `--kv-cache-dtype fp8`
- **症状**：建模阶段 `NotImplementedError` 直接崩。
- **根因**：本模型 QSA 模块（`models/qwen3_8_flash_next/nvidia/qsa.py`）要求主 KV cache 必须 BF16。
- **附带**：这与 vLLM 通用的"fp8 KV 需 FLASHINFER 后端"是两回事——就算换了后端，这个模型也不行。

### P07 PP>1 的三道禁令只拆了一半
- **症状**：三种不同的死法，取决于漏了哪个补丁：
  `PLE CPU offload is not supported: PP=2`（漏 22）/ `N-gram PLE embedding currently requires pipeline_parallel_size=1`（漏 08）/ `ValueError: len(partitions)=2 does not match pp_size=1`（漏 05，崩在 PleOffloadWorker）。
- **处置**：A 组 8 个补丁是一个整体，缺一不可。见 `docs/05#A组`。

### P08 CUDA 图与长序列冲突（历史坑，现已用别的方式解决）
- **症状**：图模式（`FULL_AND_PIECEWISE`）下 >8K prompt 必触发 `Xid31`（ENGINE GRAPHICS 虚拟写越界，地址/GPC 每次随机）。
- **曾经的处置**：`--enforce-eager`（decode 掉到 26 tok/s）。
- **现在的正解**：保留图模式，但**严格限制 `cudagraph_capture_sizes=[1,2,4,8,16,24,32,40]`**，并补上 A 组的 PP+MTP 中继（P09）。当初"图模式必炸"的实验**没有限制 capture_sizes**（默认捕获几十个尺寸），结论被过度外推了。
- **教训**：证伪一个假设前，先确认实验只改了那一个变量。

### P09 PP2 + MTP 输出乱码（本项目最难的一个）
- **症状**：能启动、能出 token、MTP 接受率还显示正常（1.96/4），但输出是 `utente`/`Werktage` 这类垃圾；draft 位恒定错（t0 对，t1~t3 永远是同一组 token）。
- **走过的弯路（都已实锤排除，勿重复排查）**：MoE backend（marlin/humming/triton/emulation）、`use_local_argmax_reduction`、`Q38_HC_GEMV`、logits/hidden 是否 NaN、lm_head 塌缩、draft embedding 是否随机初始化、pp_utils 中继行数与 `idx_mapping` 长度不一致（加诊断实测 `PPDRAFT-MISMATCH` 出现 **0 次**）。
- **真因**：PP 广播载荷里**根本没有 draft_tokens 这条通路**（补丁 21 补上）。
- **方法论**：①"t0 对、后续恒定为同一组垃圾"⇒ 错在依赖 draft 输入的环节，而不是随机内存；
  ②**对照实验（关掉 MTP）能一次性排除整条嫌疑链**，比逐参数试快得多；
  ③**黏性错误陷阱**：首个 device assert 之后，任何 kernel 启动都会被误报为崩溃点（本项目曾把锅甩给 Triton `load_binary`、NCCL `isend`）。定位手法=临时加 `CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1`，**只认第一个 assert**。

### P10 `--enforce-eager` 与图模式的取舍
- 生产**不要** eager（decode 腰斩）。判据：日志应出现 `Capturing CUDA graphs (FULL): n/n` 与 `(PIECEWISE): 8/8`。
- 若 `Capturing CUDA graphs (FULL): 2/2` 而 `capture_sizes` 写了 8 个值 ⇒ `max_num_seqs` 太小，只有前两个 batch 尺寸进了 FULL 图，超出后掉 PIECEWISE（decode 变慢）。**FULL 捕获数量 = min(capture_sizes ≤ max_num_seqs)**。

### P11 改了 `.py` 但行为没变
- **根因**：`__pycache__/*.pyc` 还在。
- **处置**：`apply-patches.py` 自动删；手工改就 `find <dir> -name __pycache__ -type d -exec rm -rf {} +`。
- **症状特征**：加了 `logger.info` 却一条都不出。

### P12 启动"诡异 OOM"（空卡上几十 KB 分配失败）
- **根因**：GPU 之前吃过 `Xid31`，显存尾部映射已进入坏态。
- **判据**：`sudo dmesg | grep -E "Xid|_scrubWaitAndSave timeout"`。
- **处置**：先跑 CUDA 分配测试（`torch.cuda.mem_get_info()` + 大块 alloc）；健康就直接拉起，不健康**只能整机重启**（`nvidia-smi --gpu-reset` 会被 Xorg 挡住）。

---

## 二、输出质量类

### P13 内容词全对、**只有标点乱**（`、。`/`。，`/三连字）
- **根因**：n-gram / 嵌入类查表偏移错位。本项目真实 bug = safetensors `data_offsets` 是**相对数据段**而非文件，正确起点 `8 + header_len + offset`；漏加导致整表错位 10368 个元素。
- **通用价值**：这个症状指纹在任何有嵌入表的服务上都成立——**看到"语义对、标点坏"就查表偏移，不要怀疑采样/量化**。
- **判据**：`[FN-PLE-INT8] n-gram table attached ... rows=320001536 hidden=160`，且量化产物 meta 的 `data_base` 与实际一致（`python3 scripts/quantize_ple.py --verify-only` 逐字节比对）。

### P14 长文循环复读（★ 本交付最重要的质量问题，完整攻关与定档见 P62）
- **根因**：`temperature` 太低（曾为把 MTP 接受率从 34.1% 提到 38.9% 调到 0.3）。
- **处置**：生产缺省 `temperature 0.6 / top_p 0.95 / top_k 20 / min_p 0 / presence_penalty 0.1 / repetition_penalty 1.05`。
- **注意**：presence 给高了会显著伤 MTP 接受率，0.1 是折中。**质量与接受率是一对矛盾，改之前先想清楚要哪一个。**
- **后续**：该组合在 09-27 被改回裸档后又复发，最终定档与全链路排查见 **P62**（复刻请直接按 P62 执行）。

### P15 思考模型的"首个增量"取不到
- **症状**：流式测试脚本 `TypeError`，TTFT 永远拿不到。
- **根因**：本模型开了 `enable_thinking`，流式时**第一个增量在 `delta.reasoning_content`，不是 `delta.content`**。
- **处置**：TTFT 一律按"任意第一个增量"计算。`scripts/bench-fnx.py` 已如此实现。

### P16 本实例没有 `/v1/tokenize` 端点（404）
- 要构造指定长度的 prompt，只能**字符→token 比率标定**：先发一次已知长度的请求读 `usage.prompt_tokens`。
- 本模型中文比率 ≈ 0.53 tok/字符（不同语言不同，要重新标定）。

---

## 三、静默失效类（最阴险：没有报错，只是没生效）

### P17 控制台/预设里的采样参数不生效
- **根因**：chroot 内 `flash-next-w4a16-inner.sh` 曾**硬编码** `--override-generation-config`，不读 `FN_GENCFG`。
- **判据**：`FN_GENCFG='{"temperature":0.11}' FN_DRY_RUN=1 bash <inner>` 看命令行里有没有 0.11。
- **处置**：已改成 `"${FN_GENCFG:-$GENCFG_DEFAULT}"`。**三处一致性铁律**：inner 的 `GENCFG_DEFAULT`、控制台 `SCRIPT_MODELS.base`、快启预设，三处采样值必须同步（预设与 base 相同时不下发 `FN_GENCFG`，走 inner 缺省）。

### P18 附加环境变量对脚本化模型长期静默失效
- **根因链**：控制台把附加 env 塞进 `FN_EXTRA_ENV`（一个字符串），脚本侧没人展开 ⇒ 你切的档从来没生效过，全靠 inner 的缺省值。
- **处置**：inner 里加了逐行 `export` 展开（`FN_EXTRA_ENV` 段）。
- **推广**：**任何"配置项→环境变量→脚本"的链路，都要在末端验证一次**，不要相信中间层。

### P19 宿主 wrapper 的 `FN_*` 透传白名单漏加新变量
- **根因**：`sudo` 会清环境，wrapper 靠一份白名单把控制台传来的 `FN_*` 落盘成 `launch.env` 给 chroot 内 `source`。白名单漏一个 ⇒ 该字段永远不生效。
- **踩过的**：`FN_PLE_INT8`、`FN_KVOFF` 两次。
- **判据**：`cat /home/ll/deploy/flash-next-w4a16-launch.env` 里有没有那一行。

### P20 shell 缺省值写法把 JSON 截断
- **症状**：`FN_SPEC=none` 变成 `none}`。
- **根因**：`${V:-{json}}` 里的 `}` 会**提前终止参数展开**。
- **处置**：JSON 型缺省值单独用变量（`SPEC_DEFAULT='...'`）承载，别塞进展开式。

### P21 bash 里 vLLM 的 JSON 参数必须单引号包裹
- `--default-chat-template-kwargs '{"enable_thinking":true}'`。不加引号会被 brace expansion 剥掉，启动即失败。

### P22 `FN_ASYNC=0` 与"没下发"等价
- inner 判据是 `[ "${FN_ASYNC:-0}" = "1" ]`。做"配置与实跑逐 token 对账"时不要把它误判成差异。

### P23 缓存命中率显示 0%（客户端侧）
- **根因**：启动命令缺 `--enable-prompt-tokens-details` ⇒ 响应 `usage.prompt_tokens_details` 恒 `null`，客户端无字段可读。**不是镜像缺陷**，代码链是完整的。
- **真实口径**：引擎 `/metrics` 的 `vllm:prefix_cache_hits_total / queries_total` 增量。

### P24 前缀缓存"看着不命中"的两个固有折扣
- ①块**异步提交延迟 ~20 s**：紧跟着重发只能命中更早的部分；
- ②每请求**最后一个块（≤1616 token）永不写入**。
- **判据**：受控实验"同 prompt 重发 + 等 20 s"，而不是拿单次 0 命中定故障。同尺寸 ≠ 同内容。

---

## 四、性能与启动速度类

### P25 权重加载慢 ≠ 盘慢（本项目最大的误判之一）
- **实测**：预先 dd 30 GiB 权重进页缓存，`Loading weights` 仍 142.93 s（全冷 138.26 s）——**零提速**。
- **探针结论**：每个 PP rank **恰好只有 1 个烧核线程**（主线程，85% 时间在用户态跑），盘仅 445 MiB/s。真瓶颈=单线程 mmap 拷贝 + AutoRound W4A16 解包/重排（约 287 MB/s/核）。
- **判据方法**：分离 utime/stime 增量 + 线程名 + `wchan`，比看 `iostat` 靠谱。

### P26 vLLM 自带多线程加载器**反而更慢**
- `--model-loader-extra-config '{"enable_multithread_load":true,"num_threads":4}'`：`Loading weights` 164.16 s vs 基线均值 136.4 s（**慢 20.4%**）。
- **根因**：`multi_thread_safetensors_weights_iterator` 是**整文件**粒度，每 worker 物化整个 5.37 GB 分片（4 线程 → 21.5 GB 在飞）+ GIL 争用，冷数据把 95 GB 的 PLE 表挤出页缓存。
- **结构性结论**：本机"并行加载"与"保 PLE 表驻留"互相打架。**不要再动加载器。**

### P27 "预热 PLE 表页缓存"在 BF16 档结构性无效
- **机制**：BF16 表被 PleOffloadWorker 读进**匿名堆**（smaps：Rss 96.82 GiB / Anonymous 96.30）。读它这件事本身要申请 95.4 GiB 物理页，内核只能靠丢弃页缓存腾地方——**丢的正是它自己要读的源页**。157 GiB 装不下"95.4 匿名 + 95.4 缓存"。
- **推论**：想让预热有效，表必须小到能同时以两种形态存在 ⇒ **只有 INT8（48.3 GiB）成立**。
- **判据**：`fincore` 看驻留率，**不能用 `free`**（匿名堆和页缓存在 `free` 里分不开）。

### P28 `read_ahead_kb` 不是全局即时生效的旋钮
- **机制**：内核在 **open/mmap 时刻**把 `bdi->ra_pages` 快照进 `file->f_ra`，之后写 sysfs 对已建立的 fd **不生效**。
- **后果**："加载期高、就绪后回落 0"这类两段式方案对长期持有的映射（PLE 表）**不成立**，只能全程取一个值。用 `dd` 测会"写完即变"，极易误判成可热更新。
- **实测拐点**（2T 盘 2.70 GB 分片，每轮 `fadvise DONTNEED`）：`0→297 / 128→3011 / 256→3113 / 512→3080 / 1024→3301 / 2048→3094 / 8192→3303 MB/s`。
- **定版**：全程 **128 KB**。设 0 会让权重加载慢 10.9×；2048 无吞吐收益却把缺页读放大 16×（n-gram 一行才 160 B）。
- **边界**：RA 只管页缓存投机预读，不影响 O_DIRECT / fault-around / 缓存命中 / 队列深度。

### P29 判断 GPU 是否带宽受限，必须看 dmon
- `nvidia-smi dmon -s u`：decode 期 `sm 51~69%` 而 `mem 14~17%` ⇒ **不是显存带宽瓶颈**。
- 别凭"PCIe 被固件锁在 Gen2"这类静态事实推断瓶颈。同理，PLE 每 token 只需约 1.6 KB，@90 tok/s ≈ 144 KB/s，比内存/NVMe 低 5 个数量级——**带宽与本题无关，相关的是延迟**（页缓存 ~100 ns vs NVMe 随机读 ~100 µs）。

### P30 功率墙的真实影响要看是哪张卡
- 限 200 W 时 **GPU1（PP1 rank，跑 MTP 草稿 + lm_head + 采样）冲顶 200.17 W**，SM 频率 1470→1440~1455 MHz（约 1~2% 损失）；GPU0 峰值仅 ~127 W 不受影响。
- **早先"功率非瓶颈"的结论是错的**——当时只看了 GPU0。**只看一张卡得出的功耗结论不可信。**
- 判据：`nvidia-smi -q -d PERFORMANCE`（`SW Power Cap: Active`）+ 逐卡 `--query-gpu=power.draw,clocks.sm`。

### P31 改功率必须先开 persistence mode
- `nvidia-smi -pm 1` 之后再 `-pl`，否则重启即失效（nvidia-smi 自己会警告）。

---

## 五、硬件与主机类

### P32 重启后 nvme 盘符与系统盘**互换**
- 任何写死 `/dev/nvmeXn1` 或 `/sys/block/nvmeXn1` 的脚本会**静默打错盘**（本项目曾长期把 read_ahead 写到系统盘）。
- **正解**：按 by-id 定位 `/dev/disk/by-id/nvme-JZ-SSD2T-XW_*`。
- udev 规则两个坑：①编号必须 **99-**（`ID_SERIAL` 由 `60-persistent-storage.rules` 导入，60- 之前匹配不到）；②匹配键必须写 **`ENV{DEVTYPE}`**，直接写 `DEVTYPE` 会让 `udevadm test` 报 `Invalid key`（报错但规则静默不生效）。
- 验证：`udevadm test --action=change /sys/block/<dev>` 看是否命中，再实弹（置 2048 → `udevadm trigger --action=change --settle` → 读回应为 128）。

### P33 CMP 170HX 的 PCIe Gen2 只有一个几秒的窗口
- **机制**：驱动在 GSP bootstrap 期写好 CYA_0/LINK_CONFIG_0/VSEC_DEVICE 后，端点仅在**极短窗口**（开机 3~4 s）对外声明 Gen2（`LnkCap2=0x06`）；必须在窗口内由**上游根端口**写 `LNKCTL2 TLS=2 + Retrain Link`。窗口关后 `LnkCap` 退回 `0x00456101`，此后任何 retrain 无效。
- **判据铁律**：只看 `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.gen.max`（成功=2,2）。
  **dmesg 里的 `SEC2_DEBUG ... OPT_GEN23 FAILED` / `booter FAILED` 是所有 170HX 必现的假阴性**（OPT_GEN23 是纯 OTP 熔丝反射、无写端口），拿它判成败会被完全带偏。
- **时机才是根因**：官方 `gen2.service`（`After=sysinit.target`）在本机被 `plymouth-quit-wait` 拖到开机后 92 s，错过窗口 80 余秒 → 600 次全落空。
- **修复**：udev 钩子在 **nvidia 模块 add 事件**（t≈2.3 s）就起 hammer。注意 `RUN+=` 会阻塞事件队列 ⇒ 启动器必须立即返回 + `setsid` 脱离（见 `scripts/host/98-cmp-gen2-early.rules`）。实测约第 26~32 次迭代（≈2 s）成功。

### P34 `Xid 31` 的两种模式与"整机硬挂"
- 模式一（GPU 侧）：`ENGINE GRAPHICS GPC0 VIRT_WRITE @0x7fXX`（地址/GPC 随机）= 页表子系统间歇失效，**不是 vLLM 软件 bug**。诱因=`gpu-memory-utilization` 顶太高逼近 64 GB 显存尾部 + 长上下文 + 图模式。
- 模式二（主机侧）：加内存条后**任何大内存读回校验**都可能整机硬断电（无 shutdown 序列）。vLLM 启动要连续读 79 GB 权重 + 95 GB 表，必在 2 分钟内触发。
- **判据**：`journalctl -b <n>` 末尾**没有** `Reached target Shutdown/Power-Off` = 硬挂。
- **注意**：`dmesg` 里 170HX 的 `SEC2_DEBUG/OPT_GEN23 FAILED` 与本条无关，别混用。
- **归因纪律**：本项目实测 EDAC / rasdaemon / MSR 直读 MCA 四路证据**全为零**，即 `mce: [Hardware Error]` 只是"记录已交给用户态"的通知，**不带 bank/地址**，无法反推到具体 DIMM。措辞上必须区分"大内存读会硬挂"（现象，已证）与"某条内存 ECC 报错"（**无证据**）。

### P35 系统时钟会跳到 2161 年（RTC 掉电）
- **症状**：任何依赖绝对时间的数据（trace 时间窗、日志排序、命中率统计）全乱。
- **判据**：`timedatectl` 的 `system clock synchronized`；`sudo journalctl -k | grep "setting system clock"` 看本次 boot 设成了哪一年。
- **软件侧护栏**：读历史数据时过滤"未来时间戳"（`t < 1e9` 或 `t > now + 300s` 判脏），并且**先过滤再截断窗口**（先截后滤会让脏记录占满窗口）。根治只能换 CR2032 + 配内网 NTP。

### P36 内存通道数与理论带宽的算法坑
- 通道数 = `dmidecode -t 17` 的 **Locator 剥掉槽位尾号后去重**（`DIMM_A1/DIMM_D2 → A/D`）。**绝不可用 Bank Locator 兜底**——服务器平台 `Bank=NODE` 是内存控制器封装（E5 v4 每颗含 2 通道），两颗 NODE 会被误判"双通道"。
- 理论带宽 = `min(Data Width,64)/8 × MT/s × 通道数`；白牌 X99 板 BIOS 把 ECC 板的 `Data Width` 虚报 72，不钳制会虚高 12.5%。
- 本机：4 通道 DDR3-1866，理论 59.7 GB/s，**STREAM Triad 18 线程实测 50.3 GB/s** —— 这是 CPU 侧 PLE 查表的硬上限。

---

## 六、运维禁忌与进程管理

### P37 **绝不 `kill -9` 持 CUDA 上下文的进程**
- 强杀会诱发 `Xid31` → GPU 降级（CUDA 可分配但 NCCL 起不来）→ 下次启动失败，唯一修复是重启。
- **正确停法**（`scripts/stop-flash-next-w4a16.sh`）：SIGTERM → 轮询等满 **90 s** → 仍未退才 SIGKILL 兜底。

### P38 "按 `VLLM::` 前缀杀进程"永远杀不到主进程
- vLLM 用 setproctitle 把 Worker/EngineCore 改名 `VLLM::Worker_PP0`（**comm 被内核截断为 15 字符**），但**主 APIServer 的 comm 恒为 `python3`**，只能按 cmdline 的 `entrypoints.cli.main` 匹配。
- 症状：主进程收 SIGTERM 后挂死 10+ 分钟不退、显存已空但 health=000。
- **僵尸（`STAT=Z`）绝不能计入"存活"**，否则"等退净"循环永远等不到 0，白拖满超时并让 SIGKILL 兜底**误触**（回到 P37）。
- 正解：`find_pids = (ps comm ~ ^VLLM::) ∪ (ps args ~ [e]ntrypoints[./]cli[./]main)`；`alive_count` 逐个读 `/proc/PID/stat` 第 3 字段排除 `Z`。

### P39 启新实例前必须确认两卡显存归零
- 孤儿进程常见形态：`ppid=1` 的 `python3.12 -c from multiprocessing.spawn import spawn_main`，pkill 模式匹配不到 → 必须 `nvidia-smi` 查 PID 后按 PID 杀。
- 症状：下次启动 `shm_broadcast: No available shared memory broadcast block found in 60 seconds`。
- **幽灵显存**：`compute-apps` 列里已死 PID 仍占额度，等驱动回收或重启。

### P40 chroot 内进程属 root，普通用户 kill 静默失败
- `kill` 返回 0 但进程还在（EPERM 被 `2>/dev/null` 吞）。必须 `sudo kill`。
- 本项目曾因此留下 2 个孤儿 worker 占 120 GB 显存。

### P41 `sudo -S` 的密码提示会吞掉 stdout 首行
- 症状：脚本首行输出神秘消失。
- **处置**：一律 `sudo -S -p ''`。

### P42 先建立 sudo 时间戳，再跑 `setsid chroot ... < /dev/null`
- `echo $pw | sudo -S setsid chroot ... </dev/null` 里 `</dev/null` 会**覆盖管道**，sudo 收不到密码。
- 处置：先跑一条 `sudo -S sh -c true` 预热时间戳，第二条才免密成功（wrapper 里 read_ahead 那行顺带干了这事）。

### P43 ssh 里的 `pgrep/pkill -f` 会**自匹配**自己的会话
- 只要命令行里出现了 `vllm.entrypoints` 字样，`bash -c` 就会匹配到自己，轻则误判"进程还在"，重则把 ssh 会话自己杀断（exit 255）。
- **本项目踩了 7+ 次。** 写法：`[v]llm`、`entrypoints[.]cli[.]main`、`resource[_]tracker`。
- 判定"运行进程有没有某个 flag"必须读 `/proc/PID/cmdline`，不要用 `pgrep -fa | grep`。

### P44 ssh/bash 单次调用有 10 分钟上限
- 长任务（冷启动 4~10 分钟、量化 8 分钟、压测）必须 `setsid` 后台跑 + 轮询结果文件，不要前台等。

---

## 七、观测口径误导

### P45 `read_bytes` 不能当加载 I/O 指标
- mmap 缺页读盘**不计入** `io.stat` 的 `read_bytes`（实测恒 0）。判断加载瓶颈看日志两行：`Loading weights took ...` / `Model loading took ...`。

### P46 `/metrics` 增量会被并发污染
- 每请求真值口径应取 trace（`request-traces.jsonl` 的 `cached_tokens`）；`/metrics` 增量只适合看累计趋势。

### P47 "按命名约定推断文件归属"一定会被骗
- 同一端口换过启动栈（NVFP4 → W4A16）后，旧日志文件还在，按 `vllm-flash-next-<port>.log` 存在即返回会永远看不到真实日志。
- **通用规则**：凡按名字推断归属的地方，必须再用**内容命中 + 新鲜度**校验一次。

### P48 前端同名函数后者静默覆盖前者
- 单体 HTML 里两张卡都叫 `renderCpuCard` ⇒ 页面无报错但永远"读取中…"，API 全 200。
- 规范：新函数一律加功能域前缀，插入前 `grep "^function <名>"` 查重。

### P49 解析器默认值必须与引擎真实默认一致
- 实例明明开着 chunked prefill，面板显示"❌关闭"——因为解析器把 `enable_chunked_prefill` 的默认写成了 `false`（vLLM 引擎默认 `True`）。
- 新加显示字段前，**先查引擎默认值**再定解析器默认值。

### P50 判断"驻留"只能用 `fincore`，不能用 `free`
- `free` 里匿名堆与页缓存不可分；P27 的全部教训都源于此。
- 判据：`fincore -b <文件>` 看 RES/SIZE 比例；进程侧看 `smaps_rollup` 的 `Rss / Anonymous / Private_Dirty`。

---

## 八、内存二级缓存（KV offload）专属坑（2026-09-22~27，全史见 `stack-0300/`）

### P51 给引擎 stats 自加 key，必须同时注册 metric defs
- 移植版 connector 在 stats 里加了新 key，而 `offloading/metrics.py` 的 `observe()` 要求每个 key 命中
  `_offloading_metric_defs` → **首个请求即 AssertionError→500**，且每次 record 都走这条路（确定性炸）。
- 判据：崩栈停在 `metrics.py observe → assert key in self._offloading_metric_defs`。
  修 defs 比砍 stats 字段好，或干脆不加自造 key。

### P52 覆盖上游方法时丢字段 = 语义性损坏（最阴的一种）
- 我们 c6 覆盖 `OffloadingWorkerMetadata.aggregate()` 时写成 `completed_jobs=dict(self.completed_jobs)`，
  **丢掉了 `other.completed_jobs`**。调度器等每个 job 的 pending_count（=worker 数，PP2 为 2）减到 0
  才调 `complete_store` → 每个 job 永远只记 1 次 → chunk 永久 `ref_cnt=-1` → lookup 恒 HIT_PENDING
  → **请求被永久 defer、CPU→GPU 恒 0**。store 日志看起来一切正常、字节在涨。
- 教训：monkeypatch 上游方法必须逐字段保语义；"写得进、永远不命中"第一嫌疑是聚合/引用计数被吞。

### P53 缓存命中实验的建档/重发两侧 `chat_template_kwargs` 必须完全一致
- vLLM 块哈希把**模板渲染后的前缀**算进去；建档时 `enable_thinking=true`、重发时缺省 false ⇒
  首块哈希就不同 ⇒ **永远 0 命中**，且现象酷似"缓存坏了"。两次窗口全栽在这。
- 做缓存实验前先固定所有会进 prompt 的开关（本项目判据文档已把它列为前置检查）。

### P54 hybrid 模型上"卸载 mamba 状态"是数据面竞态，不是索引 bug
- 经典连接器回载内容与本地 KV 不等价的真因：store 拷贝在途时，mamba 状态块**仍在被模型活写**
  （指纹：注意力组逐字节精确、mamba 组对不上）。off-by-one、指针布局、拷贝 API 全是伪线索。
- 正解=官方 `SimpleCPUOffloadConnector`：源码直接跳过 `has_positionally_stable_blocks=False` 的组，
  **只卸载位置稳定的注意力组**。设计准则：二级层只碰"写后不再变"的数据，否则必须串行化写窗。

### P55 拷贝 API 与并行中继的相互作用：换 API 只改失败签名，不改前提
- `cuMemcpyBatchAsync` 与 PP2 NCCL-P2P 并发会冻结 compute 流（worker 卡死、py-spy 抓在 kernel
  launcher）；而把 GPU→CPU 换成 Triton SM 内核又撞上游明令禁止的两点（带宽 + host 缓冲未经
  `cudaHostRegister` 时 device 不可访问 → MMU Fault/Xid31）。
- 结论：**GPU→CPU 只走"已注册锁页 + DMA 且独立于主计算流"** —— Simple 连接器的
  后台线程 + 独立流 + compute-done 事件排序就是这套正确姿势。

### P56 pinned 内存：配置值≠物理值；计数器也要验明正身
- PP2 私有 pinned 实测 ≈1.56×配置（96 GiB 档实际钉 107 GB，且 64→96 非线性/有上限）；
  公共区改前缀和偏移后实测比值 1.00。**任何"配了 N GiB 就以为占 N×系数"都要用 smaps 实测**：
  `sudo awk '/\/dev\/zero/{f=1;next} f&&/^Rss:/{t+=$2;f=0} END{print t}' /proc/<worker>/smaps`。
- 另一个假绿灯：把 `*_created`（unix 时间戳）当字节累加，编出过"回载 5.37 GB"的假数据。
  判据必须盯带方向的 counter（如 `kv_offload_total_bytes_total{direction="CPU_to_GPU"}`）。

### P57 编排脚本的 flock fd 会被守护进程继承
- `exec 9>lock && flock -n 9` 编排里 setsid 拉起 vLLM 时不关 fd ⇒ 被拉起的进程树**长期持锁**，
  之后所有窗口 `flock -n` 静默出局。解法：spawn 命令尾部加 `9>&-`。
  排障：`fuser -v /tmp/<lock>`（持有者可能属 root，/proc 自扫看不到）。
- 同轮教训：setsid 后台编排的 stdout/stderr 绝不能进 `/dev/null`——撞锁/语法错全成静默死亡，
  统一落 `/tmp/<tag>.out`。

### P58 验证窗口自身会撒谎：判据要防"旧串匹配"和"行数窗口漂移"
- 窗口判据 grep 的是**旧栈日志字符串**（`FUSE: store 方向`），现行日志早已换成
  `[dsh-kvoff c6] worker store fuse tripped` → 永远 grep 不到 → **假 PASS**。验证补丁要盯
  「预期日志是否出现」，零触发是假设证伪信号，不是修复成功。
- 高日志量下"取尾部 N 行找警告"会漂移漏检（实锤：死亡警告距文件尾 2230 行），
  行数窗口必须配时间戳校验（警告距今 ≤N 分钟）。

### P59 SimpleCPU 二级缓存的间歇性原生 segfault：上游 attrIdxs 越界 UB（issue #53860）
- 症状指纹（09-27 两次，17:38:35 / 20:38:30）：`!!!!!!! Segfault encountered !!!!!!!` →
  `Worker proc VllmWorker-1 died unexpectedly (exit code: None)` → EngineDeadError → APIServer 退出；
  **无 GPU Xid 前导、无 MCE、无 Python Traceback**，vLLM 自带 segfault handler 打的栈只有
  glibc `pthread_create/start_thread` ⇒ 死在原生线程（后台 DMA copy loop 正是 such a thread）。
  判别价值：与经典连接器的 Xid31 现场（P54/P55 族）、09-24 shm_broadcast 卡死型都能区分开。
- 根因（读上游源码逐字命中，非猜测）：`copy_blocks()` 传给 `cuMemcpyBatchAsync` 的
  `attrIdxs` 是 `ctypes.byref(params.attrs_idx)` —— **单个 `c_size_t` 标量**；而 CUDA Driver API
  契约要求该参数是 `count` 个元素的数组（每个拷贝描述符一个属性索引，值 < numAttrs）。
  驱动会读 `attrIdxs[0..cnt-1]`，越过 8 字节标量后取到的是堆上相邻随机字节，任何非零值都会
  让驱动去索引 `attrs[垃圾]` ⇒ 未定义行为，间歇性崩溃、依赖堆布局（上游同判：观测到"in the wild"
  的 segfault、nightly 上"自愈"消失——正是布局敏感 UB 的表现）。CUDA 侧 `num_attrs=1`，
  attrIdxs 必然被消费，本崩溃面在 CUDA 上成立。
- 修复（rt-patch #10，`patches-extra/dsh_simple_offload_rt.py`）：钩子替换 `copy_blocks`，
  每次调用显式传 `np.zeros(cnt, dtype=np.uint64)`（全零 = 每描述符仍用 attrs[0]，与原实现期望
  语义完全一致），另加 src/dst 块数相等与负 id 入参校验。**线程/流/事件排程/拷贝机制零改动**
  ⇒ 二级缓存功能原样保留。回退开关 `DSH_SIMPLE_OFFLOAD_UPSTREAM=1`；离线自检
  `selftest_simple_rt.py`（G1-G6，含 fresh-import 子进程验证绑定传导）。
- 取证加固（同轮上线）：inner 脚本 `ulimit -c unlimited` + `core_pattern=/media/ll/data/cores/core.%e.%p.%t`
  （落数据盘——worker RSS 上百 GB，落系统盘必写爆）。gdb 已就位。
- 教训（方法论，长期有效）：
  ① "开 X 功能就崩、关 X 就稳"是比读码更快的**功能级二分定位**——今天三次崩溃全落在带二级缓存
  的实例上，直接把嫌疑钉死在 connector 的数据面；
  ② ctypes 直调驱动 API 时，凡"数组指针 + count"型参数，逐个对照 API 文档的数组语义，
  `byref(标量)` 冒充数组是最容易被"碰巧能跑"掩盖的 UB；
  ③ 间歇性原生崩溃，先找上游 issue tracker 的同签名——本项目 30 秒命中 #53860，
  比自建复现+core 分析快一个数量级。

### P60 【更正 P59】attrIdxs 是误诊；根因=批量 cuMemcpyBatchAsync API 本身在本机驱动上不稳定，正解=逐块 cuMemcpyAsync（rt-patch #11）

> ⚠️ **本节的「批量 API 不稳定」结论也已被 P61 推翻**（#11 后生产仍崩）：真正根因是 PP2 CPU 块 id 空间越界，两种 API 只是同吃一个坏地址。保留本节以存证推理链。
- **P59 的根因判断错了，勿采信**。09-27 打上 rt-patch #10（attrIdxs 改 count 元素零数组）的实例
  （21:06 就绪）仍在 21:51 同签名 segfault（core：`cuda-EvtHandlr.7142`，栈全在 libcuda，无
  Xid/MCE/Python 帧），当日第三次崩溃。三次都发生在带二级缓存的实例、时间尺度一致（26~59 分钟），
  无 offload 实例稳定 ⇒ 功能级二分把嫌疑钉在 connector，但 attrIdxs 这条具体假设被证伪。
- **契约重读**（cuda.h 本机实测两处原型 + cuda-python 绑定 `inspect.signature` = 8 参、无 failIdx；
  driver API 文档原话："attrs 和 attrsIdxs 必须同长，长度由 **numAttrs** 指定"）：
  `attrsIdxs` 是「每个 attr 项对应的起始块索引」数组，长度 = **numAttrs**（本模型 numAttrs=1），
  上游传 `byref(c_size_t(0))`（1 个元素）**本就合法**——#10 把它换成 `np.zeros(count)` 是
  no-op，所以打了照崩。P59 里"驱动读 attrIdxs[0..count-1] 越界"不成立。
- **真正的根因**：`cuMemcpyBatchAsync`（批量 API）在本机驱动 **610.43.03 + CMP 170HX 定制固件**
  上间歇性原生 segfault——与 09-23 经典连接器同一 API 面的 `cuMemcpyBatchAsync error 1`、
  09-24 的卡死同族。用 vLLM 同款取符号方式（`cuGetProcAddress("cuMemcpyBatchAsync",12080)`）
  实测该符号在本机是 **9 参含 failIdx** 版；无论按 8 参还是 9 参调用，批量路径都会间歇炸
  （纯 ctypes 探针 60 次即段错误退出），而**逐块 `cuMemcpyAsync` 离线压测 2000 轮 ×16 块 ×64KB
  （33.5 GB）零错误**，pinned H2D 吞吐 3975 MB/s（达标）。
- **修复（rt-patch #11，`patches-extra/dsh_simple_offload_rt.py` 整版升级）**：钩子把 `copy_blocks`
  换成逐块 `cuMemcpyAsync`（同 `params.stream_handle` 上入队），彻底绕开批量 API 的宿主端描述符数组 /
  属性索引 / 完成回收机制。地址算式与批量版逐字节等价，store/load 语义、事件排程、线程模型、连接层
  零改动 ⇒ **二级缓存功能原样保留**。挂载自检：双向 + 乱序映射 + 2000 轮浸泡 全 PASS；上线后
  `vllm:external_prefix_cache_hits_total` 正常增长（二级缓存真命中）。
- **A/B 回退开关**：`DSH_SIMPLE_BATCH=1` ⇒ 退回 #10 批量行为（已知会崩，仅取证）；
  `DSH_SIMPLE_OFFLOAD_UPSTREAM=1` ⇒ 完全不打钩（上游原码）。
- **教训（方法论，长期有效）**：
  ① "上游 issue 有同签名"≠"根因相同"——#53860 谈的是 attrIdxs 数组语义，我未验证本机契约就照抄，
  方向跑偏一轮；**照抄外部修复前，必须用本机头文件/绑定把涉及的 API 签名与数组语义逐字核一遍**；
  ② 判"数组参数该多长"要读文档原文（这里是 numAttrs 不是 count），不能望文生义；
  ③ 间歇性原生崩溃，若"关功能稳、开功能崩"，最稳的止血是把该功能依赖的**具体系统调用**换成
  久经考验的等价原语（批量→逐块 async memcpy），而不是继续在可疑参数上打补丁；
  ④ 最小可复现探针（纯 ctypes、脱开 vLLM）是判定 API 稳定性的金标准，比 core 分析快且可量化。

---

## 九、复读（循环输出）问题完整攻关与解决方案 ★复刻必读

### P62 长文/token 级循环复读：三层因果链与最终解法（2026-09-19 → 10-01 定案）

**症状谱系**（从轻到重，同一个病的三个阶段）：
1. 长文（数千字输出）中出现句子级复读（同一段落反复出现）；
2. 思考模型（本模型默认 `reasoning_effort=xhigh`）在 content 里吐出 token 级硬循环碎片——典型指纹 **`uct`/`duct` 连排**、或提示词碎片回显；
3. 循环一旦开始不会自愈，只能中断请求；同一会话内倾向复发。

**三层因果链（缺一不可地解释了过去一个月的全部现象）**：
- **直接触发＝会话上下文污染**。一旦某轮输出出现病理循环串、且它进入了后续对话历史（或被人为粘回对话框），它就成为复读吸引子，该会话怎么采样都会复读。
  **判别法（永远是第一道题）**：新开一个干净会话问同样的问题——不复读＝症状在会话不在引擎；同时刻直连引擎发干净上下文探测可交叉确认。
- **土壤＝模型本身的数值退化边缘**。1M 上下文 + xhigh 思考 + PLE n-gram 表参与 logits，低温时 logits 分布极尖，容易跌进重复 attractor。实证：`temperature=0` 下同一 prompt 走「全新计算」与「前缀缓存命中」两条路径的续写就不逐字相同（vLLM 前缀缓存路径间本就非确定性）——**所以"字节级等价"不能作为任何故障的定罪判据，必须用语义级探针（验证码复述 + 病理正则）**。
- **历史放大器＝SM74 回收补丁窗口**（09-29~30 病发期与 SM74 冷启动窗口重合；09-25~28 热装期零循环；09-29 同日还有 Xid13 GPC0 SM Out Of Range）。10-01 已回滚 SM70。SM74 是放大器不是必要条件——裸采样在无 SM74 时也发作过。

**采样档位时间线（这就是"解决了复读"的确切含义）**：

| 日期 | 档位 | 动机 | 结果 |
|---|---|---|---|
| 09-19 前 | t1.0 原生 | 默认 | 长文句子级复读（P14） |
| 09-19 | t0.3 | 提 MTP 接受率 34.1→38.9% | 接受率涨了，复读恶化 |
| 09-21 | **t0.6 / p0.95 / k20 / minp0 / pres0.1 / rep1.05** | 治复读，牺牲少量接受率 | 句子级复读平息 |
| 09-27 | 回到裸 1/0/1 | 追求原生质量 | — |
| 09-29 | **uct/duct token 级硬循环复发** → 用户拍板回 0.6 组合 | — | 定档 |
| 10-01 | 临时加固 **t0.6 / pres0.2 / rep1.15**（观察期） | SM74 归因窗内从严 | 观察期内零发作 |
| 10-03 夜 | **当前生产定档 = t0.7 / p0.95 / k20 / minp0 / pres0 / rep1.15** | 旧 chroot 栈上稳定运行、无 uct/duct 病例复发；快启预设名直接叫「不复读采样t0.7-rp1.15」 | 实跑 cmdline 与该档逐字一致（`tools/live-cmdline-w4a16.txt`） |

**最终解法（复刻者照此配，五条全部做到才不会复读）**：
1. **采样档位（当前生产定版）**：
   `--override-generation-config '{"temperature":0.7,"top_p":0.95,"top_k":20,"min_p":0,"presence_penalty":0,"repetition_penalty":1.15}'`
   ——要点是 **repetition_penalty>1 + 中等温度**的组合；历史上的 0.6/pres0.1/rep1.05 与 0.6/pres0.2/rep1.15
   是同一家族的过渡档，都可接受，**唯独不要回到 t≤0.3 或裸 1.0/0/1.0**。两套 inner 的
   `GENCFG_DEFAULT`（`scripts/flash-next-w4a16-inner.sh`、`stack-0300/bin/…0300-inner.sh`）与
   快启预设必须同档。
2. **病理串卫生**：任何复读输出的原文**绝不回流到对话历史**（只存文件归档）。已污染的会话无药可救，开新会话是唯一出路。
3. **硬件侧**：不使用 SM74 回收补丁（保持 stock SM70），直至该补丁通过显存正确性金丝雀测试（见 CHANGELOG 09-29/10-01 段）。
4. **投机档**：MTP4 + block1616 合法组合（ring 8｜1616）；`num_speculative_tokens≥5` 在 block1616 下非法（P04 的整除约束），不要为了提速乱开高档。
5. **参数一致性**：采样参数的权威值分布在**四处**（新栈 inner / 旧栈 inner / server.js SCRIPT_MODELS.base / launch.env 的 FN_GENCFG），改任何一处必须四处同步，且**重启后终验必须读 `/proc/<APIServer pid>/cmdline` 的 `--override-generation-config` 真值**——历史上多次"文件都对、运行时却是别的值"（envfile 残留 + 并发启动竞写，09-29 实锤）。wrapper 已改为全量透传 `FN_*`（根治 09-26 前的手写白名单漏键导致的"弹窗改采样静默失效"）。

**已排除项（都有实锤，勿重复排查）**：SimpleCPU/经典连接器回载污染（验证码探针通过）、超 KV 池抢占（`num_preemptions_total=0` 且循环时段占用≤53%）、segfault 守卫（v13 零触发）、单一采样档（裸档与定档都发作过，说明档位是调节量不是充分病因）、MoE backend、CUDA graph 模式。

**监控工具**：机器上有 `loop-sentinel.py`（定时探针 + 病理正则 `uct|duct` + 污染探测，P1-P4 四类检查）；判据日志 `/home/ll/deploy/loop-sentinel.log`。

### P63b 盘读速塌方的标准处置顺序（2026-10-05 二次发作后定版，先于 P63 修复细则执行）

1. **抢救不可重下载件**（只有这些值得从 50 MB/s 的故障盘里抠）：`vllm-image/`（chroot 镜像 rootfs，本机连不上 dockerhub 无法 docker pull）、小配置目录（如 `models-1m/` 的 YaRN config）。其余模型权重、PLE 表产物一律**擦后重下**——ModelScope 4 路并发 ~100 MB/s，比从故障盘抠还快 2~4 倍且拿到的是干净数据。抢救打包命令：`sudo tar cf /home/ll/rescue-1005/vllm-image.tar -C /media/ll/data vllm-image`（tar 前先 umount rootfs 上的 bind 挂载避免递归）。
2. **可选一枪 `nvme format --ses=0`**（不擦数据重格式化）：规范上不动用户数据，赌控制器状态机复位。**实际成功率低**（多数控制器 ses=0 近似空操作，09-25 的修复机制是 block erase 重建 FTL 空闲表），且白牌固件行为不可保证（可能等同全清）——只把它当免费彩票，中奖预期不要高于两三成。打完立刻测读速。
3. **主力方案 `nvme format --lbaf=0 --ses=1`（用户数据擦除）**：约 4 秒完成，本次实测读速恢复 **4.7 GB/s + 7.4 万 IOPS**（比 09-25 的 1.4 GB/s 还好——本机槽位 Gen4）。擦后必须先过测速关再 mkfs：`dd if=/dev/disk/by-id/... of=/dev/null bs=1M count=2048 iflag=direct`。
4. **ses=1 也救不回 = 盘报废**，换盘。
5. 擦后重建：`mkfs.ext4 -F -L data2t -m 1 /dev/disk/by-id/nvme-...`（整盘无分区）→ fstab 改 UUID → mount → `nvme format` 后 **dockerd 的 data-root 在数据盘上，要先 systemctl stop docker.socket docker 才能 umount**。
6. 长期教训：该盘 30 天内两次读塌方（09-25、10-05），791 次不安全关机 + 166 TB 累计读是主因。**该机 UPS/正常关机纪律要做**（看门狗/实例 SIGTERM 正常流程都算安全关机；整机断电硬挂才是杀手），并把"盘健康只信实测读速，不信 SMART"写进巡检。

### P63 服务机换板后数据盘读速塌方（28~45 MB/s）+ mmap 随机查表的预读放大——"输出/预填充没有旧平台快"的双重根因（2026-10-05）

**症状**：Qwen3.8-Flash-Next-Channel-INT8-w8a8（3×170HX PP3）decode 只有 13~25 tok/s（旧 x99 平台 ~90），实例每隔 20~40 分钟卡死→`TimeoutError: RPC call to sample_tokens timed out`→EngineDead→看门狗重启循环。

**排查链（每步都有实测数）**：
1. PLE INT8 表 48.3 GiB 在 32 GiB 内存机上 fincore 只驻留 30%（14.6/47.7 GiB）→ 服务期每个 token 的 n-gram 随机查表都在缺页读盘。
2. 全局 `read_ahead_kb=128`（权重加载最优值）对 PLE 映射是灾难：每 160 B 行缺页投机读 128 KB（**约 800× 读放大**），页缓存被冲刷、盘被喂爆。内核在 open/mmap 时刻把 `bdi->ra_pages` 快照进 `file->f_ra`，事后改 sysfs 无效（P29 同型机制）。
3. 盘本身也在退化：同一块 JZ-SSD2T-XW（serial 30166625132，累计读 166 TB、不安全关机 791 次）换到新主板 H12D-8D 后 **O_DIRECT 单流 512 MB 全偏移恒 28~40 MB/s**（同机对照三星 PM981 = 2669 MB/s；SMART 温度 45 ℃、Media Errors=0、Spare 100%）——盘没有"坏块"，是 DRAM-less QLC 主控在读重试/GC 风暴里的整体性塌方。**任何"这台机盘没问题，因为 SMART 干净"的判断都不可信，判据只能是有负载直读测速**。

**修复（两件，都要做）**：
1. `tools/patch-ple-fadvise-1005.py`：给 `v1/ple_offload/worker.py` 三个 PLE mmap fd（BF16 磁盘驻留路径 + INT8 表/scale）加 `posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM)`——`f_ra.ra_pages=0` 随 fd 生效，缺页只读所需页；权重文件不受影响。改完必删 `__pycache__` 下对应 .pyc。
2. **把 PLE 表迁到健康的盘**：最终落在 `/home/ll/deploy/ple-w8a8`（系统盘三星 PM981，49 GB，复制后多窗口 dd+md5 界内校验；注意校验偏移必须落在文件大小内，越界窗口 dd 读 0 字节会"假 OK"）。**坑中坑：chroot 只 bind 了 `/media/ll/data` 与 `/home/ll/deploy` 两个宿主目录，PLE 目录必须放其中之一**——第一次放 `/home/ll/ple-w8a8` 时 chroot 内 `[ -f meta ]` 判空，静默回落 BF16 分片加载路径（多花 40 分钟才暴露）。同时把 inner 缺省 `PLE_INT8_DIR` 与 launch.env 的 `FN_PLE_INT8_DIR` 一起钉到该目录（控制台/看门狗/手动三条启动链路都要覆盖）。

**残余事实（如实）**：32 GiB 内存装不下 48.3 GiB 表 → 服务期永远有 cold miss，根治需 ≥96 GiB RAM（该板 8 槽、旧平台留有 7×32 GB DDR4 ECC RDIMM 可直接搬）+ 更换数据盘（权重冷启动加载现在也要 ~40 分钟，因为盘只有 ~40 MB/s）。INT8+heap（匿名 48.3 GiB）在此机结构性不可行。

**判据**：修复后 decode 采样 `disk_read_mbs≈0~50` 而 `gpu0_util` 不再周期性掉 0；日志 `[FN-PLE-INT8] n-gram table attached from /home/ll/ple-w8a8`。

---

### P64 沿用另一台机器的配置档（卡数/内存档不同）→「按启动没反应」型失败（2026-10-06）

症状：8889 控制台点「启动」后端口始终不 LISTEN、实例日志尾部只有一轮 `[shutdown] … SIGTERM`，
或引擎在 WorkerProc 初始化处报错；`dmesg` Xid=0、机器不硬挂。看着像「模型坏了」，实为**配置档与在位硬件不符**。

本机实锤链：这台机是 HUANANZHI X99-T8 + E5-2696 v4 + 8×32 GiB（251 GiB 内存）+ **2×CMP 170HX**
（`lspci -d 10de:` 只有 `03:00.0`/`04:00.0`，第三张卡 10-06 傍晚已拆走），
但 10-05 那批换装 EPYC/3 卡/32 GiB 机器的描述一路留在配置里：
`server.js` `SCRIPT_MODELS.base.pp=3` 与 `pleInt8='1'`、两张快启预设 `gpuCount:'3'`、
wrapper `RA_LOAD` 缺省 16、udev 规则 `read_ahead_kb=16`、inner 的 INT8 目录缺省 `ple-w8a8`。
⇒ 任何一次 UI/看门狗重放都会下发 `FN_PP=3`，2 张卡起 3 个 PP rank 必然失败。

四处对账判据（换卡/换内存/换盘/换模型后必须逐条核）：

| 位置 | 查法 | 期望 |
|---|---|---|
| 在位卡数 | `lspci -d 10de: \| wc -l` | 与 base.pp / 预设 gpuCount 相等 |
| 生产模型目录 | `curl :8889/v1/models` 与 `ls /media/ll/data/models` | `SCRIPT_MODELS.modelPath` = 真要跑的 checkpoint |
| PLE 精度可行性 | `free -g` 内存 vs 表体积（INT8 48 GiB / BF16 95 GiB）；`ls <INT8 目录>/ple_ngram_meta.json` | 产物不存在就别选 INT8（inner 会回落 BF16，但 UI 显示会骗人） |
| 预读 | `cat /sys/block/<by-id 解析出的数据盘>/queue/read_ahead_kb` | 大内存机 128（09-23 实测拐点），只有内存装不下表时才降 |

一次收口件：`tools/patch-w4a16-restore-1006.py`（幂等，`--revert` 可回滚，六处=
server.js base/模型目录/`pleInt8Dir`+plan 下发、两预设、wrapper `RA_LOAD`、udev 规则、看门狗回退档、
删 `fnx-manual-stop` 闩锁）。

**验证手法（别看 JSON 就收工）**：用 node vm 把 `SCRIPT_MODELS` 条目与 `scriptModelLaunchPlan` 抽出来真跑，
分别喂「弹窗默认（空 params）」与「两条预设」，把得到的 `FN_*` 与在跑实例 `/proc/<pid>/cmdline` 逐 token 对账；
再 `FN_DRY_RUN=1 bash start-flash-next-w4a16.sh` 看 inner 生成的 argv。
两个长期坑顺带记牢：① wrapper 每次启动都会用当前环境**重写** `launch.env`，
所以 dry-run 时导出的 `FN_DRY_RUN=1`/`FN_LOG=/tmp/…` 也会被写进去 ⇒ 事后必须删掉这两行，
否则看门狗会「忠实重放」出一个只打印命令的假启动；
② `udevadm test` 以普通用户跑时报 `Failed to write ATTR… Permission denied` 属正常（规则由 root 的 udevd 执行），
复核要 `sudo udevadm trigger --action=change --sysname-match=<盘>` 后读回数值。

---

### P65 自己新写的 inner 忘了「参数通道」→ 静默按内置缺省起（症状：像启动成功了，但档位全错）（2026-10-06）

给新栈写 wrapper+inner 时，只做了「wrapper 把控制台 FN_* 落盘 launch.env」这一半，
忘了 inner 那一半必须 `set -a; . "$FN_ENVFILE"; set +a`——**sudo 的 env_reset 会把控制台传来的 FN_* 全部清掉**，
落盘文件不被人读就等于没落。后果不是报错而是**静默回落内置缺省**：

| 现象 | 本次实锤 |
|---|---|
| 上下文缩水 | 引擎 `/v1/models` 的 `max_model_len=262144`（应为 1048576），`GPU KV cache size … Maximum concurrency for **262,144** tokens` |
| 二级缓存没开 | `cache_config_info{… kv_offloading_size="None", num_cpu_blocks="None"}`，日志里 SimpleCPUWorker 一行都不出 |
| 采样/思考档漂移 | 实际生效的是 inner 内置值，不是弹窗/预设值 |

**判据（别看 wrapper 有没有报错，它一定"成功"）**：
① `tr '\0' '\n' < /proc/<APIServer pid>/cmdline | grep -aE 'max-model-len|kv-offloading'`；
② `curl -s :18420/metrics | grep -a cache_config_info`（`kv_offloading_size`/`num_cpu_blocks` 必须非 None）；
③ inner 启动即打一行 `[FN-0310] 已加载参数文件 …（N 项）`——没有这行就是通道断了。
根治约定：**任何新写的 inner 第一段就是 source launch.env**（与 0.30/旧 chroot 栈同构），
并在 wrapper 里禁止 `sudo -E`（会绕过落盘、留下两份真值）。

### P66 vLLM 0.30.0 → 0.31.0 升级 + 「二级缓存·CPU」落地清单（2026-10-06 实测）

一次做对的顺序与判据（全部在生产机验证过，命令见 `stack-0310/README-0310.md`）：

1. **rt-patch #13 不用改**：`selftest_simple_rt_v13.py` 对 0.31.0 环境直接 **13/13 PASS**（含金丝雀）。
   0.31 的 `simple_kv_offload/{manager,worker}.py` 虽改了（stats/boundary/`prefix_cacheable_group_ids`），
   三个钩子锚点 `_derive_cpu_config` / `_init_cpu_mode` / `copy_blocks`+`build_params` 形状未变；
   **但 0.31 上游仍是 `generate_scheduler_kv_cache_config` deepcopy `kv_cache_configs[0]` +
   `_derive_cpu_config` 按 rank0 推块数** ⇒ P61 的口径分裂还在，#13 必须挂。
2. **只给 `--kv-offloading-size` 不会启用 SimpleCPU**：`config/vllm.py` 需要
   `envs.VLLM_USE_SIMPLE_KV_OFFLOAD=1`，否则 backend=native 走的是 `OffloadingConnector`（P54 已定案退役）。
   inner 里两件事一起做，并与 `FN_KVOFF=1` 互斥拒绝启动。
3. **hook8（QSA ring 收缩）保住 block 1616**：0.31 把 ring 放宽，`MTP K=5 ⇒ ring 12`，`1616%12≠0` 会在
   加载权重后 AssertionError；hook 在「legacy 整除 block」时收缩。inner 另做同型前置校验（早失败、给人话）。
4. **PP 两 rank 的 CPU 块数天生不等**：实跑 `PP0 2224 / PP1 2157 块`（PP1 多 LM+draft 注意力层、每块更大），
   判据行 `[dsh-simple-rt] 握手 clamp：调度器 CPU 块数 2224 -> 2157（对齐最窄 worker，防 PP 越界）`；
   没有这行 = clamp 没生效 = 迟早复现 P61 的原生 segfault。
5. **换栈后日志文件属主坑**：新实例由 root 创建日志，宿主 wrapper 是 ll ⇒ `[wrapper] … 权限不够`（丢两行诊断，
   不影响启动）。正解 `sudo chown ll:ll <log>`（root 仍写得进）。
6. **PLE INT8 产物完整性判据**：`quantize_ple.py` 的两个 `.bin` 是**预分配**的（跑一半就在，体积正确），
   只查 `.bin` 存在会误判"产物就绪" ⇒ 必须同时要求 `ple_ngram_meta.json`（收尾才生成），
   或直接看 `[done]` / `[verify] … 字节不一致=0 relMSE=4.360e-05` 两行。
7. **升级带来的内存好处**：`DSH_PLE_MMAP=1` 用 INT8 表 mmap（51.8 GiB 可回收）替代 0.30 的锁页 BF16（95.4 GiB），
   于是「96 GiB CPU 档 + 51.8 GiB 表 + 权重页缓存」在 251 GiB 机上从容（实测 used 106 / cache 143 / avail 142）。

---

### P67 二级缓存验收的两种假阴性（都实锤过，别拿它们当"回载坏了"）（2026-10-06 第二轮）

同一套验收（`tools/kvoff-accept-0310.py`）跑第二轮，指标全绿却判 `VERDICT=0`：

| 假阴性 | 症状 | 真因 | 修法 |
|---|---|---|---|
| **回答预算被思考吃满** | `external_prefix_cache_hits_total` 增量 80,800、`cached=80,800（96.2%）` 都对，但"验证码复述"判 False，`content` 是空串 | `max_tokens=64` 对 xhigh/medium 思考模型不够，reasoning 吃光预算 → 正文 0 字 | 探针 `ANSWER_MAX_TOKENS` 缺省提到 **900**；判定必须 **合并 `content` + `reasoning_content`**；正文为空时直接打"假阴性，请加大预算复验"的告警行（09-27 就留过同样的 TODO，这次补齐） |
| **curl 传长文档** | `OSError: [Errno 7] Argument list too long: 'curl'` | 单个命令行参数受 `MAX_ARG_STRLEN`=128 KB 限制（中文 84k token ≈ 340 KB），**不是** ARG_MAX 总量 | 发请求走 `urllib.request`（脚本本就是），任何临时复验也别用 `curl -d "$文档"` |

**复验结论**（同一份文档、同 nonce 重发，预算 900）：`content='\n\nCODE-288925-3126'`
→ **逐字命中 True**，`prompt=83,959 / cached=80,800 / 1.7 s`；第二轮 `external_hits` 增量 0 属正常
（前一轮已把该前缀装回 GPU 池，这次命中的是显存档）。

判据纪律：**"外档命中"与"回载正确性"必须分开取数**——前者只看 `/metrics` 计数器增量，
后者需要一次真正产出正文的问答；任何一侧被探针参数卡住都会把健康的系统判成故障。

---

### P68 `pkill -f` 第四次自杀：同一条命令里既写模式又写脚本路径，照样把自己杀掉（2026-10-06）

`pkill -f 'soak-0310-mo[n]itor]'` 这种"括号防自匹配"只保护**模式字符串本身**不被匹配；
可同一条 `bash -c` 里后面还有 `setsid … /home/ll/deploy/soak-0310-monitor.sh` ——
**那句普通文本才是被匹配到的东西**，于是 pkill 先把自己的 ssh 会话杀了（表现：整条命令零输出，
后面的启动全没执行，看护反而变成 0 个）。本次连踩两次。

正解（三选一，按稳妥度排序）：
1. **拆成两条命令**：先 `pgrep -f 'soak-0310-mo[n]itor' | xargs -r kill`（这条命令行里绝不出现普通脚本名），
   再单独一条去 `install/start`。
2. 按 PID 杀：先 `pgrep` 落盘，再 `kill $(cat /tmp/pids)`。
3. 让常驻进程自己写 pidfile，启停脚本按 pidfile 操作（长期做法）。

通用纪律：**任何 `pkill -f` / `pgrep -f` 的调用行里，都不能出现被匹配字符串的"普通形态"**；
清理与启动不要写在同一条命令里。

---

### P69 吞吐数字必须在空载下测：同一台机器上并行跑"挤池验收"会把探针数据打成 6.9 tok/s（2026-10-06）

我把 4 并发探针和二级缓存挤池验收**同时**跑，得到"聚合 27.4 tok/s、单流最低 6.9 tok/s"，
差点据此判 4 并发是负优化。等验收结束后空载重测同一脚本：**聚合 257.8 tok/s、单流 64.5~68.8 tok/s**。
相差近 10 倍，全部来自测量污染（churn 每条 14 万 token 的 prefill 独占 SM）。

纪律：任何 tok/s 数字都要注明**当时机器上还有谁**——首选空载；无法空载就在报告里同时给出
并发请求数与 `vllm:num_requests_running`/`kv_cache_usage_perc` 快照，否则数字没有可比性
（与 P45「报数附配置」、09-26「测速方法论」同源，但这次是**自测污染**，更隐蔽）。

---

## 附：已验证走不通的死路（别再试）

| 尝试 | 结论 |
|---|---|
| `--max-num-batched-tokens 16384` | 引擎崩 + Xid31（P03） |
| `--kv-cache-dtype fp8` | QSA 要求 BF16，直接 `NotImplementedError`（P06） |
| `--moe-backend marlin` 显式 | 未量化草稿 MoE 报错（P05） |
| `--mamba-ssm-cache-dtype bfloat16` | 首个请求即崩在 CUDA 图 replay |
| `QWEN_GDN_REPLAY` / `GDN_DIAG_DISABLE_JIT_MONITOR` / `CUDA_MODULE_LOADING=LAZY` / `PYTORCH_NVML_BASED_CUDA_CHECK` | 官方 GDS+TEP2 组合专用，加进 mmap 栈必崩 |
| 改模型 `config.json` 的 `quant_method` 为 gptq | vLLM 自动把 auto-round→inc；改写会丢 2340 条 fp16 层映射 |
| vLLM 多线程加载器 | 慢 20.4% 且挤掉 PLE 表驻留（P26） |
| BF16 档预热 PLE 页缓存 | 结构性无效（P27） |
| `read_ahead_kb=0` | 权重加载慢 10.9×（P28） |
| TP2 代替 PP2 | 无 P2P 时每步约 192 次 all-reduce 走 host SHM，更慢 |
| 经典 OffloadingConnector 在本模型常驻（含 c1~c11 全部修法） | hybrid mamba 活写竞态，回载不等价（P54），退役；用 SimpleCPUOffloadConnector |
| SimpleCPU 二级缓存间歇 PP1 segfault，先后按「attrIdxs 越界（#10）」「批量 API 不稳定（#11）」修复均失败 | 两条都是**误诊**（P59→P60→P61 两次更正）。最终根因=**PP2 下调度器与 worker 的 CPU 块 id 空间口径分裂**，越界地址喂给驱动（P61）。正解=rt-patch #13 握手 clamp |
| 「加大 pinned 到 >GPU 池」在 251 GB 内存机上硬做 | PLE 95 GB + 私有 pinned 1.56× 必超配，MCE 硬挂风险（P56） |
| MTP6 / MTP1 在本档 | MTP6 触发 QSA ring 断言（P04）；MTP1 实测比 MTP4 慢 22% |
| 沿用另一台机器的 launch.env/控制台预设（卡数、内存档不同）直接按启动 | PP 档/PLE 档与在位硬件冲突，症状是「按了没反应」而非报错（P64） |
| 在 chroot 旧栈上直接开 SimpleCPU 二级缓存（镜像里有 `simple_kv_offload/` 模块） | 该镜像版本缺 #13 握手 clamp 与逐块拷贝守卫 ⇒ 原样复现 P61 的 PP1 原生 segfault；要开就升 0.31.0 栈（P66） |
