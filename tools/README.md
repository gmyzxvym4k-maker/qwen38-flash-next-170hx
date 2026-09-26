# tools/ — 生产机实跑快照

这些文件是**从运行中的实例直接抓取**的，用来回答"文档写的和生产跑的是不是同一件事"。
所有快照时间：**2026-09-23 19:0x~19:5x CST**（引擎日志时间是 UTC，比对北京时间 **+8**）。

| 文件 | 内容 | 怎么复核 |
|---|---|---|
| `live-cmdline.txt` | APIServer 的完整 argv（逐行） | `sudo tr '\0' '\n' < /proc/<pid>/cmdline` |
| `live-env.txt` | 引擎进程环境变量（`VLLM_*`/`Q38_*`/`NCCL_*`/…） | `sudo tr '\0' '\n' < /proc/<pid>/environ` |
| `live-proctree.txt` | PP2 进程树（EngineCore / Worker_PP0 / Worker_PP1） | `ps -eo pid,ppid,stat,etime,comm` |
| `launch.env` | wrapper 落盘的 `FN_*`（本次启动的输入） | `cat /home/ll/deploy/flash-next-w4a16-launch.env` |
| `startup-profile.log` | 启动画像：PLE 挂载 / 权重加载 / KV 池 / init engine / 补丁指纹 | `grep -aE 'FN-PLE\|Loading weights\|GPU KV cache size\|init engine' <日志>` |
| `metrics-snapshot.txt` | `/metrics` 关键计数器（命中率、接受长度、KV offload 填充率） | `curl -s :18420/metrics \| grep -E '^vllm:(prefix\|spec\|kv_offload)'` |
| `model-files.tsv` | 模型目录逐文件字节数（181.24 GB / 28 文件） | `stat -c '%s\t%n' <model>/*` |

## 三个"看快照而不是看文档"的理由

1. **`pgrep -fa | grep` 会自匹配**执行它的那条 ssh 命令（本项目踩了 7+ 次）。
   判定运行参数**必须**读 `/proc/PID/cmdline`。
2. **文档与代码各说各话是复现类文档最大的敌人**。本仓库每条"生产值"都能在这些快照里找到出处；
   对不上就是文档错，不是生产错。
3. **同一端口可能换过启动栈**（本端口先跑过 NVFP4 栈、后跑 W4A16 栈），
   按文件名推断日志归属会被骗——快照里保留了 pid 与时间戳，可交叉验证。

## 快照与仓库脚本的一致性

`launch.env` 里的 `FN_*` 经过 inner 脚本推导后，应与 `live-cmdline.txt` 完全对应。
对账方法（零风险，不启进程）：

```bash
FN_MODEL_PATH=/media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound \
FN_MAXLEN=1048576 FN_LONGCTX=1 FN_1M_MODEL_PATH=<1M副本> FN_YARN_FACTOR=4 \
FN_SEQS=3 FN_PP=2 FN_ASYNC=1 FN_PLE_INT8=1 FN_PLE_LOC=heap \
FN_KVOFF=1 FN_KVOFF_BYTES=103079215104 FN_SPEC='{"method":"mtp","num_speculative_tokens":4,"use_local_argmax_reduction":false}' \
FN_DRY_RUN=1 FN_ENVFILE=/dev/null bash scripts/flash-next-w4a16-inner.sh
# 把输出的 argv 与 tools/live-cmdline.txt 逐 token 比
```

**两个已知的等价差异**（对账时不要误判）：
- `FN_ASYNC=0` 与"未下发"行为等价（inner 判据是 `[ "${FN_ASYNC:-0}" = "1" ]`）；
- dry-run 输出的 JSON 被 shell 解析后引号形态可能不同，要**两侧引号归一**后再比。

## 非快照件：运维补丁脚本

| 文件 | 用途 | 校验 |
|---|---|---|
| `patch-kvoff-default-off-0929.py` | 把 8889 控制台三源（server.js SCRIPT_MODELS base/fallback、index.html 弹窗缺省）的 KVOFF 翻为默认关；幂等、`--revert` 回滚、`node --check` 语法门禁+失败自动回滚 | `python3 tools/patch-kvoff-default-off-0929.py --check` |
