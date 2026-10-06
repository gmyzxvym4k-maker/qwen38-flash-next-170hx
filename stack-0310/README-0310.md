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
| 稳定性 | 全程 `Xid=0`；实例段 `Segfault/EngineDead/Traceback` 计数 **0** |

## 5. 参数真值与三源同步
`launch.env` ＝ 8889 `SCRIPT_MODELS['qwen3.8-flash-next-w4a16'].base` ＝ 快启预设
`current-pp3-1m-mtp4-int8disk-32g`（显示名「当前固化-双卡PP2-1M-MTP4-**二级缓存96G**…」）。
要点：`FN_PP=2`（本机在位 2 卡，`lspci -d 10de:` 数卡）、`FN_SIMPLE_OFFLOAD=96`、
`FN_PLE_MMAP=1` + `FN_PLE_INT8_DIR=/media/ll/data/ple`、采样仍是复刻当轮的裸档
`t1.0/top_p0.95/top_k20/min_p0/presence0/repetition1.0`（反循环定档是 0.6/0.95/20/0/0.2/1.15，
如复发言题三源一起改，见 `docs/08` P46/P62）。

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
