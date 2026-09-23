# patches/

22 个针对 **官方定制镜像 `vllm/vllm-openai:qwen38-flash-next`（vLLM v0.1.dev20073 / py3.12）** 内
`vllm` 包的补丁。逐条说明见 [`../docs/05-patches.md`](../docs/05-patches.md)。

## 格式

每个文件是 `diff -u a/<相对路径> b/<相对路径>`，**相对路径以 `.../dist-packages/vllm/` 为根**，
所以用 `patch -p1` 在 vllm 目录下应用。

文件名 = `序号-父目录__文件名.patch`（父目录用于消歧，因为仓库里有 `scheduler.py`、`gpu_worker.py` 各两个）。

## MANIFEST.tsv

| 列 | 含义 |
|---|---|
| `id` | 序号（= 补丁文件名前缀，也是 docs/05 的编号） |
| `path` | 相对 vllm 包的源文件路径 |
| `pristine_sha256` | **上游原版**哈希（打补丁前必须等于它） |
| `patched_sha256` | **打完应为**的哈希（打完立即校验） |
| `patch_sha16` | 补丁文件自身哈希前 16 位 |
| `patch_file` | 补丁文件名 |

双哈希是这个仓库对"可复现"的核心承诺：**别人拿到的镜像层若与生产机不同，
`apply-patches.py` 会在第一秒报出来，而不是打完一个歪补丁、跑出莫名其妙的乱码。**

## 用法

```bash
V=/media/ll/data/vllm-image/rootfs/usr/local/lib/python3.12/dist-packages/vllm
python3 ../scripts/apply-patches.py --target $V --check     # 体检
sudo python3 ../scripts/apply-patches.py --target $V --apply
sudo python3 ../scripts/apply-patches.py --target $V --revert
```

## 这些补丁是怎么做出来的

不是手抄上游 PR，而是**从生产机反向导出**：

1. 用镜像的 32 个层（`blobs/` + `layers.txt` 顺序）解出一棵纯净 vllm 树（`docs/02#5`）；
2. `LC_ALL=C diff -rq` 纯净树 vs 生产 rootfs → 恰好 22 个文件不同（排除 `__pycache__` 与 `.bak-*`）；
3. 逐文件 `diff -u` 导出补丁，并同时记录两侧 sha256。

因此本目录与生产实例**逐字节一致**，验证记录见 [`../docs/09-verification.md`](../docs/09-verification.md)。

## 上游依据对照

| 本仓库补丁 | 对应上游 |
|---|---|
| 20 `gpu/model_runner.py`、21 `gpu/pp_utils.py`、01 `config/speculative.py` | vLLM **PR #52295**（PP + 投机解码的中间张量/draft token 中继） |
| 09 `nvidia/mtp.py` | **issue #54709 / PR #46994**（`is_local_drafter_forward`：按输入分支而非按 rank 分支） |
| 05 / 06 / 07 / 08 / 10 / 18 / 19 / 22 | 本地实现（无公开源码），对应手册 §3 里"本档自有开关"的那几个 |
| 02 / 03 / 04 / 13 / 14 / 15 / 16 | 本地实现（KV 二级缓存 c1→c5a 演进链） |
| 11 / 12 / 17 | 本地运维向（提速 + 每请求实时进度） |
