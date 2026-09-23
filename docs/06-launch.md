# 06 · 启动、停止、参数矩阵、验收

## 1. 启动链路

```
你（或控制台/看门狗）
  └─ bash scripts/start-flash-next-w4a16.sh          ①宿主 wrapper
       ├─ 把显式给出的 FN_* 变量落盘 → flash-next-w4a16-launch.env
       │    （sudo 会清环境，chroot 内看不到你的 shell 变量 ⇒ 必须落盘传递）
       ├─ 设数据盘 read_ahead_kb（by-id 定位）
       └─ SUDO setsid chroot <rootfs> /bin/bash flash-next-w4a16-inner.sh   ②chroot 内
              ├─ source 那个 env 文件
              ├─ 按 FN_* 推导 PLE 驻留形态 / 长上下文副本 / 采样 / 投机 / KV 二级缓存
              └─ exec python3.12 -m vllm.entrypoints.cli.main serve ...      ③引擎
                     ├─ VLLM::EngineCore
                     ├─ VLLM::Worker_PP0 → GPU0（层 1~26，含 PLE 层）
                     ├─ VLLM::Worker_PP1 → GPU1（层 27~48 + lm_head + MTP 草稿 + 采样）
                     └─ PleOffloadWorker（CPU 侧 n-gram 查表）
```

## 2. 启动

```bash
# 预演（强烈建议第一次先跑这个）：只打印将要执行的完整命令
FN_DRY_RUN=1 bash /home/ll/deploy/start-flash-next-w4a16.sh

# 真启动（脱离终端，日志落盘）
setsid nohup bash /home/ll/deploy/start-flash-next-w4a16.sh >/dev/null 2>&1 &

# 等就绪（约 3.5~6 分钟，取决于表是否已驻留）
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18420/health)" = "200" ]; do sleep 10; done
```

生产实例的完整 `FN_*` 取值见 [`../tools/launch.env`](../tools/launch.env)；
inner 的**缺省值就是生产值**，所以什么都不传也能起出同一个实例（262144 档）。

## 3. `FN_*` 参数矩阵

| 变量 | 缺省 | 说明 |
|---|---|---|
| `FN_PORT` | `18420` | 监听端口（host 固定 `0.0.0.0`） |
| `FN_SERVED` | `qwen3.8-flash-next` | served 模型名 |
| `FN_MODEL_PATH` | `…/Qwen3.8-Flash-Next-W4A16-AutoRound` | 模型目录 |
| `FN_MAXLEN` | `262144` | `--max-model-len`；>262144 自动加 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` |
| `FN_LONGCTX` / `FN_1M_MODEL_PATH` / `FN_YARN_FACTOR` | `0` / — / — | 长上下文档：切到 YaRN 副本，**副本缺失直接拒启** |
| `FN_PP` / `FN_TP` | `2` / `1` | 并行。**无 P2P 时只有 PP2 是对的**（TP2 每步约 192 次 all-reduce 走 host SHM，实测更慢） |
| `FN_PARTITION` | `26,22` | `VLLM_PP_LAYER_PARTITION`；`none`=交回镜像默认 |
| `FN_BLOCK` | `1616` | **必传值**（P01） |
| `FN_SSMDTYPE` | `float32` | **必传值**（P02） |
| `FN_SEQS` | `4` | `--max-num-seqs`。改小会减少 FULL 图捕获数量（P10） |
| `FN_MBTOKENS` | `8192` | `--max-num-batched-tokens`。**不要 16384**（P03） |
| `FN_GPUMEM` | `0.95` | OOM 时第一刀降到 0.92 |
| `FN_MOE` | `auto` | **不要显式写 marlin**（P05） |
| `FN_SPEC` | `{"method":"mtp","num_speculative_tokens":4,...}` | `none`=关投机；档位合法性见 P04 |
| `FN_GENCFG` | 见 §6 | 采样 JSON；不传则用 inner 缺省 |
| `FN_ASYNC` | `0` | `--async-scheduling`（`0` 与未传等价，P22） |
| `FN_EAGER` | `0` | `1`=`--enforce-eager`（decode 腰斩，仅排障用） |
| `FN_PLE_INT8` | `1` | 精度：INT8 / BF16 |
| `FN_PLE_LOC` | 由 INT8 推导 | 位置：`disk` / `heap`（见 `docs/04`） |
| `FN_KVOFF` / `FN_KVOFF_BYTES` | `1` / `103079215104`(96 GiB) | CPU KV 二级缓存。**高风险，建议先 0**（`docs/05#D组`） |
| `FN_CACHE_ROOT` | `/root/.cache/vllm-flash-next-w4a16` | **按实例隔离**！与别的 vLLM 实例共享 `torch_compile_cache` 会踩 AOT stride 冲突 |
| `FN_EXTRA_ENV` | — | 多行 `KEY=VALUE`，inner 逐行 export（P18） |
| `FN_EXTRA_ARGS` | — | 追加到命令行的原始参数 |
| `FN_DRY_RUN` | `0` | 只打印命令 |
| `FN_LOG_LEVEL` / `FN_DIAG_SAMPLE` / `FN_HC_GEMV` / `FN_PP1_FULL_DECODE` | `INFO` / `0` / `1` / `0` | 调试与本地扩展开关 |

### 危险参数黑名单（inner 会剔除，别费劲加）

```
--mamba-ssm-cache-dtype bfloat16        → 首个请求即崩在 CUDA 图 replay
--kv-cache-dtype fp8                    → QSA 要求 BF16，NotImplementedError
--moe-backend marlin                    → 草稿 MoE 未量化
QWEN_GDN_REPLAY / GDN_DIAG_DISABLE_JIT_MONITOR / CUDA_MODULE_LOADING / PYTORCH_NVML_BASED_CUDA_CHECK
                                        → 官方 GDS+TEP2 路线专用，本栈加了必崩
```

## 4. 停止

```bash
bash /home/ll/deploy/stop-flash-next-w4a16.sh
```

铁律与实现要点见 `docs/08#P37~P40`，一句话版：
**SIGTERM 等满 90 s，僵尸不计入存活，超时才 SIGKILL 兜底；主进程 comm 是 `python3` 不是 `VLLM::`；绝不无脑 `pkill -9`。**

停止脚本会顺手触发 PLE 表页缓存预热（此时刚释放的匿名内存正好装得下）——
**这只在 INT8 形态有意义**，BF16 形态无效（P27）。

## 5. 看门狗（可选，防实例被误杀/机器重启后不自愈）

```bash
sudo cp scripts/fnx-18420-watchdog.sh /home/ll/deploy/
sudo cp systemd-units/fnx-watchdog.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now fnx-watchdog.timer
```

逻辑：每 30 s 探 `/health`；死亡 → 清残留 → **等两卡显存归零** → `vmtouch` 预热 → 跑启动脚本；
连续失败 3 次退避 300 s。为什么需要它：这台机器历史上实例常被外部自动化 SIGTERM 掉，
而被杀后不会自愈（只有控制台启动时才拉起一次）。

## 6. 采样缺省

```json
{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.1,"repetition_penalty":1.05}
```

- 模型自带 `generation_config.json` 是 `1.0 / 0.95 / 20`，这里用 `--override-generation-config` 覆盖。
- 为什么是 0.6 而不是 0.3：0.3 是为了 MTP 接受率（34.1%→38.9%）刻意调的，代价是循环复读；
  实测下来**质量优先**。折中档可试 0.5。
- **三处一致性**：inner 的 `GENCFG_DEFAULT`、（若接控制台）`SCRIPT_MODELS.base`、快启预设，三处必须同步。

## 7. 验收清单

`bash scripts/verify-deployment.sh` 会全部覆盖；手工核对则看这四组：

**① 引擎起来了**
```bash
curl -s http://127.0.0.1:18420/v1/models | python3 -m json.tool | head -8
# 期望：id=qwen3.8-flash-next，max_model_len=1048576
grep -aE "GPU KV cache size|Maximum concurrency|init engine" /home/ll/deploy/vllm-flash-next-w4a16.log | tail -3
# 期望：GPU KV cache size: 1,224,366 tokens（@1M 档）；init engine 60~190 s
```

**② 关键参数确实生效**（读 `/proc/PID/cmdline`，别信 `pgrep -fa`，P43）
```bash
PID=$(ps -eo pid=,args= | grep "[e]ntrypoints[./]cli[./]main" | awk '/--port 18420/{print $1; exit}')
sudo tr '\0' '\n' < /proc/$PID/cmdline | grep -A1 -E "block-size|mamba-ssm|pipeline-parallel|max-num-seqs|speculative"
# 期望：1616 / float32 / 2 / 3~4 / num_speculative_tokens":4
```

**③ 补丁与驻留形态**
```bash
grep -aE "FN-PLE|no PleOffloadLayer|CPUOffloadingSpec|warmup skipped" <日志> | tail
# 期望：n-gram table attached ... (anonymous heap 或 mmap, zero heap)
#       [FN-PLE-PP] no PleOffloadLayer on this pipeline stage ...（rank1 一行，正常）
```

**④ 功能与性能冒烟**
```bash
python3 scripts/smoke-online.py http://<部署机>:18420 qwen3.8-flash-next   # 5 项功能验收（工具/思考/80K 标识复述/流式/标点）
curl -s http://127.0.0.1:18420/v1/chat/completions -H 'Content-Type: application/json' \
 -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"只回答两个汉字：你好"}],"max_tokens":48,"temperature":0}'
python3 scripts/bench-fnx.py 32768,32768,32768 512     # 预填充/TTFT/decode 三档
sudo dmesg | grep -c "NVRM: Xid"    # 期望 0（注意 r8169 网卡的 "XID 541" 是芯片 ID，不是 GPU 故障）
```

工具类端点：`/v1/models`、`/health`、`/metrics`。
**注意本实例没有 `/v1/tokenize`**（P16）。
