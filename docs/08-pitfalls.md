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

### P14 长文循环复读
- **根因**：`temperature` 太低（曾为把 MTP 接受率从 34.1% 提到 38.9% 调到 0.3）。
- **处置**：生产缺省 `temperature 0.6 / top_p 0.95 / top_k 20 / min_p 0 / presence_penalty 0.1 / repetition_penalty 1.05`。
- **注意**：presence 给高了会显著伤 MTP 接受率，0.1 是折中。**质量与接受率是一对矛盾，改之前先想清楚要哪一个。**

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
| MTP6 / MTP1 在本档 | MTP6 触发 QSA ring 断言（P04）；MTP1 实测比 MTP4 慢 22% |
