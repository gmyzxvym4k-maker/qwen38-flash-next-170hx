# stack-0310 —— 官方 vLLM 0.31.0 + 运行时补丁栈（当前生产，2026-10-06 起）

## 1. 是什么 / 为什么换
0.30.0 栈的功能等价升级版：**site-packages 零改动**，全靠 `PYTHONPATH` 注入运行时补丁。
换它的直接动因是要把「二级缓存·CPU」做稳：0.30 栈上验证过的 rt-patch **#13（PP 下 CPU 块 id 空间握手 clamp
+ 逐块拷贝守卫）** 在 0.31 上 **无需改一行即可挂载**（`selftest_simple_rt_v13.py` 对 0.31 环境 13/13 PASS，含金丝雀），
而 0.31 的运行时补丁集另有两个决定性 hook：

| hook | 作用 |
|---|---|
| `hook8`（`common.qsa_cache`） | 0.31 把 QSA ring capacity 放宽（K=5→12），会撞 `block-size 1616` 整除断言；hook 在「legacy 值整除 block」时收缩回去 ⇒ **1M + block 1616 + MTP4 组合继续可用** |
| `DSH_PLE_MMAP`（hook3/3c + `DshPLEMmapEmbedding`） | PLE n-gram 表走 **INT8 产物磁盘 mmap**（47.7+0.6 GiB 可回收页缓存），不再锁页 95.4 GiB BF16 ⇒ 省 ~44 GiB 内存，给 96 GiB CPU 档腾出安全余量 |

官方 0.31 对 PLE 只有两档：**pinned-host**（全表 `cuMemHostRegister`）或**进显存**，无磁盘驻留路径——
mmap 档是本补丁集提供的。

## 2. 目录
```
stack-0310/
├─ bin/flash-next-0310-inner.sh        # 参数装配 + 档位自检（ring 整除/长上下文/PLE/二级缓存/root+memlock）
├─ start-flash-next-0310.sh            # 宿主 wrapper：FN_* 落盘 launch.env、RA=128、显存归零门禁、root setsid
├─ stop-flash-next-0310.sh             # 按端口优雅停（SIGTERM 90s 后才兜底；僵尸不计存活）
├─ launch.env                          # 生产参数真值（与 8889 base/预设三源同步）
├─ patches/sitecustomize.py            # 上游 8+2 钩（0.31.0 版，含 hook8/hook10 与 DSH_PLE_MMAP）
├─ patches-extra/{sitecustomize,dsh_simple_offload_rt}.py   # rt-patch #11/#13（逐块拷贝+握手 clamp）
└─ selftest_simple_rt_v13.py           # 13 用例离线自检（含"故意破坏必须被抓"金丝雀）
```
> 本副本对 `start/stop/探针` 做了 **sudo 口令脱敏**（改走 `SUDO_PASS` 环境变量），与生产机逐字节不同属预期；
> `patches*/`、`bin/` 与生产 **md5 一致**。

## 3. 从零复现（七步）
```bash
# ① 装 vLLM 0.31.0（torch 2.13.0+cu130，PyPI 直连可达）
uv venv /media/ll/data/vllm-0310-env --python 3.11
uv pip install --python /media/ll/data/vllm-0310-env/bin/python vllm==0.31.0
# ② 必做两件（缺任一：flashinfer 链接失败 / 仪表盘无 trace）
cd /media/ll/data/vllm-0310-env/lib/python3.11/site-packages/nvidia/cu13 && ln -sfn lib lib64 && ln -sfn libcudart.so.13 lib/libcudart.so
uv pip install --python /media/ll/data/vllm-0310-env/bin/python /home/ll/deploy/dsh-logger-pkg/
# ③ 生成 W4A16 的 INT8 PLE 表（约 11 分钟；data_base=8+header_len 已在脚本内处理）
python3 scripts/quantize_ple.py --model /media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound --out /media/ll/data/ple
#    期望末两行：[verify] 抽样 100000 行：字节不一致=0 relMSE=4.360e-05 行余弦均值=0.9999784 / [done] …
# ④ 自检 rt-patch #13 对新版本仍生效（期望 PASS=13 FAIL=0）
cd stack-0310 && /media/ll/data/vllm-0310-env/bin/python selftest_simple_rt_v13.py
# ⑤ 预演 argv（不碰 GPU）：期望 max-model-len 1048576 / --kv-offloading-size 96 / PP2 / block 1616
set -a; . launch.env; set +a; FN_DRY_RUN=1 bash bin/flash-next-0310-inner.sh
# ⑥ 实启（必须 root；PLE 锁页/注册要 memlock 不限）
set -a; . launch.env; set +a; bash start-flash-next-0310.sh
# ⑦ 就绪判据（见 §4），期望 5~8 分钟
```

## 4. 验收判据（每条都带期望值，命令与判据分开写）
```bash
L=/home/ll/deploy/vllm-flash-next-0310.log
grep -a "Initializing a V1 LLM engine" $L | tail -1        # 期望含 v0.31.0 与 models-1m/…-1M
grep -aE "SimpleCPUOffloadWorker \[CPU\]" $L | tail -2      # 期望两 rank ≈2157/2224 块、各 ≈48 GiB
grep -a "握手 clamp" $L | tail -1                            # 期望「2224 -> 2157（对齐最窄 worker，防 PP 越界）」
grep -a "GPU KV cache size" $L | tail -1                     # 期望 1,210,374 tokens / 1M 并发 1.15x
curl -s :18420/metrics | grep -a kv_cache_size_tokens        # 与上行一致
echo <部署机 sudo 口令>|sudo -S -p '' dmesg | grep -c Xid                  # 期望 0（本次全程）
```
二级缓存有效性（`tools/kvoff-accept-0310.py`，2026-10-06 19:47 实测）：

| 步骤 | 期望/实测值 |
|---|---|
| 建档 55,924-token 文档 | `prompt_tokens=55924`、首次 `cached=0`、9.7 s |
| 挤池（发互不相同的大请求） | 累计 **1,434,381 token** ＞ GPU 池 1,210,374 |
| 重发同一文档问验证码 | `cached=53,328`（**95.4%**）、耗时 **1.2 s**、`external_prefix_cache_hits_total` 增量 **53,328** |
| 内容正确性 | 验证码 `CODE-287365-3126` **逐字复述**（回载无损） |
| 第 2 轮复测（20:21，60k-token 文档） | 挤池 **1,579,107 token** → `prompt=83,955 cached=80,800`（**96.2%**）、`external_hits` 增量 **80,800**、耗时 **1.5 s**；验证码 `CODE-288925-3126` 在把回答预算从 64 提到 900 后**逐字命中**（64 时正文为空 = 探针假阴性，见 `docs/08` **P67**） |
| 稳定性 | 全程 `Xid=0`；实例段 `Segfault/EngineDead/Traceback` 计数 **0**；**带 96GiB 档连续运行 70 分钟（越过历史 26~71 分钟病灶窗口）零 segfault** |
| 重启后复测（20:53→20:58） | 挤池 130 万 token 后重发：`cached=53,328（95.3%）`、`external_hits` 增量 **53,328**、验证码逐字命中、`Xid=0` |
| **PLE 是否在内存**（20:53 重启后实测） | `fincore`：INT8 表 **驻留 98.83%**（47.13/47.68 GiB）、scale 全驻留；worker 对产物的映射 61.95 GiB 中 **RSS 45.74 GiB**；空闲 30 s **majflt 增量=0**、20 s **盘读=0 MiB** ⇒ 查表全程在 RAM，只是"可回收页缓存"而非锁页 |
| **4 并发吞吐**（seqs 2→4 后，空载实测 4×300 token） | **聚合 257.8 tok/s**，单流 64.5~68.8 tok/s（几乎不掉速）；FULL CUDA 图从捕获 2 个尺寸变 **3 个（bs=1,2,4）**；MTP 平均接受长度 4.66 |
| 加压连测（20:22~20:33，3 轮挤池各 1,579,107 token） | **3/3 VERDICT=1**：每轮 `external_hits` 增量 **59,792**、`cached=94.9%`、验证码逐字命中；累计 `ext_hits=877,488 token`、`load_blocks=563`、`save_outcomes(stored)=272`、`pending_store` 稳定在 15（不增长＝无卡死传输）、`Xid=0`、`dsegv=dengine_err=0` |

## 5. 参数真值与三源同步（10-10 与生产实测逐字节对齐）
`launch.env` ＝ 8889 `SCRIPT_MODELS['qwen3.8-flash-next-w4a16'].base`（`modelVariants` 按 checkpoint 分派模型路径/1M 副本/PLE 表目录/PP 下限）＝ 快启预设 `w8a8-pp3-1m-mtp4`。
当前生产真值（本仓库 `launch.env` 的 md5 与机器上那份一致）：
模型 `Qwen3.8-Flash-Next-Channel-INT8-w8a8`（1M 副本 `models-1m/…-1M`）、`FN_PP=3`（本机在位 3 卡）、`FN_TP=1`、
`FN_BLOCK=1616`、`FN_MAXLEN=1048576` + `FN_YARN_FACTOR=4`、**`FN_SEQS=4`**、`FN_GPUMEM=0.95`、`FN_MBTOKENS=8192`、
`FN_SIMPLE_OFFLOAD=96`、`FN_KVOFF=0`、`FN_SPEC=mtp×4`、
**`FN_PLE_INT8=1` + `FN_PLE_LOC=heap` + `FN_PLE_INT8_DIR=/media/ll/data/ple-w8a8`**（PLE 驻留四档见 §8）。
采样真值＝inner 的 `GENCFG_DEFAULT`（launch.env 不带 `FN_GENCFG` 即沿用）：
`t0.6/top_p0.95/top_k20/min_p0/presence0.2/repetition1.15`（10-01 反循环加固档，沿革见 `docs/08` P46/P62）。
改任何一项三处齐改；重启后终验读 `/proc/<APIServer pid>/cmdline`（10-01 实锤过「文件都对但运行时不同」的 envfile 竞写）。

## 6. 回滚阶梯
```bash
# ① 关二级缓存（保留 0.31）：launch.env 里 FN_SIMPLE_OFFLOAD=0 → 重启实例
# ② PLE 回官方锁页 BF16（无 INT8 产物时）：FN_PLE_MMAP=0（或弹窗精度=BF16 / 位置=heap）
# ③ 退回 chroot 旧栈：touch /home/ll/deploy/vllm-0310/DISABLED   （控制台+看门狗同时生效）
# ④ 退回官方 vLLM 0.30.0 栈：bash tools/patch-stack-0310-1006.py --revert && systemctl --user restart dsh-console
```
内存档回退第一刀：`FN_SIMPLE_OFFLOAD=48`（本机 251 GiB 内存、有 MCE 硬挂史）。

## 7. 已知风险与未验证清单（如实）
1. **soak 仍在计时**：历史病灶（P59~P61）发生在带 96 GiB 档就绪后 26~71 分钟。#13 clamp 已确认生效
   （日志有 clamp 行），但 **≥6 小时无 segfault 才算稳**；看护件 `tools/soak-0310-monitor.sh`
   → `/home/ll/deploy/soak-0310.log`（每 5 分钟一行：health/segv/ext_hits/xid/内存）。
2. `DSH_PLE_MMAP` 的 mmap-INT8 查表路径在 **0.31 上首跑**：标点/内容探针与验证码已过（§4），
   但长文与多并发下的 PLE 命中质量仍需日常观察（症状指纹=「内容词对、标点乱」→ 立即 `FN_PLE_MMAP=0`）。
3. `patches-extra/sitecustomize.py` 仍尝试 `import dsh_kvoff_rt`（经典连接器移植件，本栈刻意不放）
   ⇒ 每个进程启动多一行 `[rt-patch-extra] dsh_kvoff_rt 加载失败…` 的**无害**告警。
4. 二级缓存的收益边界不变：只有「前缀被挤出 GPU 池（121 万 token）之后又被重发」才吃到；
   历史 21.5 h 观测里 GPU 池自扛 ~90% 命中、外档命中为 0。

## 8. PLE n-gram 表驻留四档与「内存·匿名堆」（10-10 上线，生产档=第 2 行）
控制台「PLE 表驻留」卡的值来自引擎日志判据行（显示即真值），不是配置文件。四档与内存记账：

| 弹窗（精度 / 位置） | inner 推导出的 env | 引擎实现 | 体积 | `free` 里算在哪 |
|---|---|---|---|---|
| INT8 / 放硬盘 | `DSH_PLE_MMAP=1` + `DSH_PLE_MMAP_LOCK=1` | mmap 产物文件（mlock 的是**文件页**） | 47.7+0.6 GiB | **buff/cache**（仍可被回收） |
| **INT8 / 放内存**（生产） | `DSH_PLE_MMAP=1` + `DSH_PLE_MEM_RESIDENT=1` | **私有匿名堆**：`np.empty` + 并行 `os.pread` 分块读入 | 47.7+0.6 GiB | **已用**（不可回收）✅ |
| BF16 / 放内存 | `FN_PLE_MMAP=0` | 官方 pinned-host（`cuMemHostRegister`，要 root+memlock） | 95.4 GiB | 已用（且不可换出） |
| 显存 | — | 官方 device 驻留 | — | 显存 |

要点：
1. **mlock 的文件页仍算 buff/cache**，只有私有匿名页算「已用」。所以「PLE 表要体现在内存已用量」的正解是走匿名堆档，而不是给 mmap 加锁——这是本档存在的全部理由。
2. 判据日志（缺一行就是没吃到该档）：
   `[FN-PLE] INT8 内存驻留（匿名堆）：…（47.7+0.6 GiB，不可回收、计入已用）`、
   `[rt-patch-0310] PLE anon-heap storage attached: 47.7 GiB … (resident, non-reclaimable)`、
   引擎侧 `Initialized PLE embedding … weight_dtype=torch.int8, weight_device=cpu, pinned=False`。
3. 首跑必崩已修（`docs/08` **P71**）：`memoryview(arr.data)` 继承调用方的 2-D 形状 ⇒ 切片赋值 `NotImplementedError: memoryview slice assignments are currently restricted to ndim = 1`；正解 `memoryview(arr).cast("B")`。重打件 `tools/patch-anonple-memview-1010.py`（幂等、`--revert` 可撤、自带 2-D/1-D 逐字节自检，须用 venv 的 python 跑）。
4. 内存账（251 GiB 机，10-10 实测）：PLE 匿名 48.3 + SimpleCPU 96 GiB pinned + 引擎 ~12 ≈ **已用 156 GiB**，buff/cache 只剩 ~2.7 GiB ⇒ 权重页缓存被挤光，下次重启多付 ~57 s 权重加载。内存告警第一刀仍是 `FN_SIMPLE_OFFLOAD=48`。
5. 控制台侧 `server.js` 的 `matchPleLine` 已认 `PLE anon-heap storage attached` / `[FN-PLE] INT8 内存驻留` 两类新行；**改完必须 `systemctl --user restart dsh-console`**——10-10 实锤：磁盘文件已更新但进程还是旧的，卡片一直显示旧驻留档（`docs/08` **P70**）。
