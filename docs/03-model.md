# 03 · 模型：获取、校验、量化配置、长上下文副本

## 1. 需要的模型

| 项 | 值 |
|---|---|
| 仓库名 | `Qwen3.8-Flash-Next-W4A16-AutoRound` |
| 架构 | `Qwen4ExpForConditionalGeneration`（`model_type: qwen4_exp`） |
| 规模 | 48 层 / hidden 2560 / 512 experts（激活 10）/ vocab 248320 / 原生 ctx 262144 |
| 参数量 | index 记 `total_parameters = 123,958,298,771`（不含 PLE n-gram 表） |
| 量化 | AutoRound **W4A16**，`bits=4, group_size=128, sym=true`，packing `auto_round:auto_gptq` |
| PLE | `ple_layer_ids=[2]`、`ngram_size=3`、`hc_count=4` |
| MTP | index 内含 **1565 个 `mtp.*` 键**（草稿头权重齐全，可用） |
| 体积 | **181.24 GB / 28 个文件**（17 分片 + extra + tokenizer/config 等） |

完整文件清单与字节数：[`../tools/model-files.tsv`](../tools/model-files.tsv)。

> **注意：仓库里没有 `model-00002-of-00017.safetensors`，`index.json` 也不引用它。**
> 分片编号从 00001 跳到 00003 是**上游仓库本身如此**，不是下载缺漏。
> 校验脚本按 index 核对，不要按"1..17 必须齐全"去判缺文件——那是假警报。

## 2. 下载

国内直连 HuggingFace 通常不通，实测走 **ModelScope** 可用：

```bash
# 机器上有 venv 就装 modelscope CLI；实测 4 并发聚合约 55~100 MB/s，单流约 28 MB/s
modelscope download --model <repo_id> \
  --local_dir /media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound --max-workers 4
```

单文件直取（做局部补下/校验时方便）：
```
https://modelscope.cn/api/v1/models/<repo_id>/repo?Revision=master&FilePath=<file>
列目录：.../repo/files?Revision=master&Recursive=true      （repo/tree 端点是 404）
```

**放盘位置**：模型和 PLE 产物都放大数据盘（1.9 TB）。系统盘只剩几十 GB，放 181 GB 会把机器做死。

## 3. 校验（下完必做）

```bash
python3 scripts/verify_model.py --dir /media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound
```

检查项：
1. `model.safetensors.index.json` 引用的每个分片都存在、`metadata.total_size` 与实际字节一致；
2. 每个分片的 **safetensors 头部**能解析、`data_offsets` 连续且不超过文件大小（截断下载的典型特征）；
3. `config.json` 的架构 / 量化字段符合预期（见 §4）；
4. **PLE n-gram 表完整性**：128 个 `[2500012,160]` BF16 张量首尾相接，总行数 `320,001,536`、总字节 `102,400,512,256 - header`。

> 本项目吃过"下载看似完成、其实某分片被截断"的亏：**只看文件大小不够**，必须解 safetensors 头并核对 offset 链。

## 4. 量化配置为什么**不能改**

`config.json` 里的 `quantization_config.quant_method` 必须是 **`auto-round`**（原样）。

- vLLM 的 INC 量化入口有 `override_quantization_method`：`quant_method == "auto-round"` → 自动改用 `inc` 加载，
  并识别 `packing_format: auto_round:auto_gptq` 的 GPTQ 打包张量。**不需要你帮忙改成 gptq。**
- 曾经有人（包括本项目早期）把 `quant_method` 改写成 `gptq` 以求"更通用"，结果**丢掉 `extra_config` 里 2340 条
  per-layer fp16 映射**（AutoRound 逐层精度配置），量化层的选择就错了。
- 目录里可能残留两个 sidecar：`quantization_config.json`（94 KB，AutoRound 原始导出）与
  `quantize_config.json`（98 B，写着 `gptq`，是早期实验留下的）。**vLLM 只读 `config.json` 里的
  `quantization_config` 字段，这两个 sidecar 都不参与**；看到 `quantize_config.json` 写着 gptq 不要以为配置被改了，
  以 `config.json` 为准（`scripts/verify_model.py` 也是这么判的）。

## 5. 长上下文（512K / 1M）副本

原生 `max_position_embeddings = 262144` 写在 **`text_config`** 里（顶层没有）。要跑更长上下文：

| 做法 | 结果 |
|---|---|
| 只加 `--max-model-len 1048576` + `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` | 能起，但 **RoPE 未缩放**，位置越过原生上限后注意力打分退化（长文"看不太懂"） |
| 用 `--hf-overrides` 注入 YaRN | dict 型 `hf_overrides` 是递归合并、只写增量键即可，**但不会传给草稿模型**（上游 `speculative.py` 注释明说）→ 投机侧仍按原长度建 cos/sin 缓存，位置一大就越界 → device-side assert → `Xid 43` |
| **本仓库做法：做一份只改 config.json 的副本** | 权重全部软链回原目录（零额外空间，只多一个 266 KB 的 config.json），主模型与草稿共用同一份 config，缩放一致 |

```bash
python3 scripts/make-longctx-copy.py \
  --src /media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound \
  --dst /media/ll/data/models-1m/Qwen3.8-Flash-Next-W4A16-AutoRound-1M \
  --target-len 1048576
```

副本改动只有两处（其余键逐字节保留）：

```jsonc
"text_config": {
  "max_position_embeddings": 1048576,
  "rope_parameters": {
    "rope_type": "yarn",
    "factor": 4.0,                              // = 1048576 / 262144
    "original_max_position_embeddings": 262144,
    "mrope_section": [11,11,10], "partial_rotary_factor": 0.25, "rope_theta": 10000000  // 保留原值
  }
}
```

**机制备忘**：YaRN 的 cos/sin 缓存长度 = `max_position × factor`；而 mrope 无 YaRN 时是 `max_position × 4`。
这解释了"主模型 512K 不越界、草稿越界"的历史现象（缓存长度算法不同）。

启动时通过 `FN_LONGCTX=1 FN_1M_MODEL_PATH=<副本> FN_YARN_FACTOR=4` 切换；
inner 脚本会**在副本缺失时直接拒启**（宁可起不来，也不要静默用未缩放 RoPE 跑超长上下文），
并把 `--max-model-len` 钳到副本声明的上限。

## 6. 采样缺省

模型自带 `generation_config.json`（`temperature 1.0 / top_p 0.95 / top_k 20`）。
本服务用 `--override-generation-config` 覆盖为 `temperature 0.6 / top_p 0.95 / top_k 20 / min_p 0 /
presence_penalty 0.1 / repetition_penalty 1.05`（治循环复读，权衡见 `docs/08#P14`）。
