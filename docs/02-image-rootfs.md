# 02 · 获取镜像并搭 chroot（不用 docker）

## 1. 为什么必须用这个镜像

`Qwen3.8-Flash-Next` 的架构是 **`Qwen4ExpForConditionalGeneration`**（`model_type: qwen4_exp`），
PyPI 上的任何 vLLM stable/nightly **都不认识它**；而且 PLE offload 框架（`v1/ple_offload/`）、
GDN 融合内核、QSA 缓存这些实现都在**官方 day-0 定制镜像**里。所以：

```
镜像：vllm/vllm-openai:qwen38-flash-next
多架构 index digest：sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8
linux/amd64 manifest：sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3
镜像内 vLLM 版本：v0.1.dev20073（Python 3.12）
层数：32（压缩后约 8.1 GB，解出后 vllm 包 756 MB / 4864 个文件）
```

> **一定要按 digest 锁，不要按 tag 拉。** tag 会被上游重推，一旦内容变了，`patches/MANIFEST.tsv`
> 里的 `pristine_sha256` 就对不上——`apply-patches.py` 会明确报"上游哈希失配"而不是静默打歪。
> 复核当前 tag 指向：
> ```bash
> TOK=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:vllm/vllm-openai:pull" | jq -r .token)
> curl -sI -H "Authorization: Bearer $TOK" \
>   -H "Accept: application/vnd.docker.distribution.manifest.list.v2+json" \
>   https://registry-1.docker.io/v2/vllm/vllm-openai/manifests/qwen38-flash-next | grep -i docker-content-digest
> ```

## 2. 解出 rootfs

```bash
bash scripts/fetch-image-rootfs.sh /media/ll/data/vllm-image
# 产物：amd64.json（manifest）、layers.txt（层序）、blobs/sha256:*（gzip 层）、rootfs/（根文件系统）
```

脚本做的事：registry token → 按 digest 取 amd64 manifest → 逐层下载（`.done` 标记 + 大小校验，可断点续传）
→ **按 `layers.txt` 顺序**解包到 `rootfs/`，并处理 OCI 的 whiteout（`.wh.<name>` 删除标记、`.wh..wh..opq` 目录清空）。

> 层顺序不可省：后面的层覆盖前面的层，whiteout 表示"上层删掉了下层这个文件"。
> 乱序或跳过 whiteout，会得到一棵"看着完整其实混了旧版本"的树——补丁哈希会告诉你出了事。

## 3. 挂 chroot 并验证

```bash
sudo bash scripts/setup-chroot.sh /media/ll/data/vllm-image/rootfs
# 挂 /proc /sys /dev + bind 数据盘 + 拷 NVIDIA 用户态库进 rootfs
sudo chroot /media/ll/data/vllm-image/rootfs /usr/bin/python3.12 -c \
  'import torch,vllm; print(torch.cuda.device_count(), vllm.__version__)'
# 期望：2 v0.1.dev20073
```

常见失败：`Unable to open /dev/urandom` = `/dev` 没挂进去；`No CUDA devices` = NVIDIA 用户态库没拷/驱动版本不匹配。

## 4. 打补丁

见 [`05-patches.md`](05-patches.md)。**这一步之前 rootfs 必须是纯净上游态**：

```bash
python3 scripts/apply-patches.py --target /media/ll/data/vllm-image/rootfs/usr/local/lib/python3.12/dist-packages/vllm --check
# 首次应输出：22 个 pristine；若混着 patched/unknown，说明树被动过
```

## 5. 重建"纯净基准树"（做补丁 / 排查时非常有用）

想知道某个文件上游原版长什么样、或者怀疑有人偷偷改过镜像：

```bash
P=/tmp/pristine-vllm; REL=usr/local/lib/python3.12/dist-packages/vllm
mkdir -p $P
while read -r d; do tar -xzf /media/ll/data/vllm-image/blobs/$d -C $P --wildcards "$REL/*"; done \
  < /media/ll/data/vllm-image/layers.txt
diff -rq $P/$REL /media/ll/data/vllm-image/rootfs/$REL | grep -v __pycache__ | grep -v '\.bak-'
# 期望：恰好 22 行 differ，且与本仓库 patches/ 一一对应
```

本仓库的 22 个补丁就是用这个方法从生产机导出的（`LC_ALL=C` 才能被脚本解析，中文 locale 下 `diff` 输出是"文件 … 和 … 不同"）。

## 6. 为什么不用 docker

- 本方案的 PLE 走**内存/mmap**，不需要 GDS、不需要 `nvidia-container-toolkit`、不需要 P2PDMA 直通；
- chroot 少一个 daemon，端口与进程都在宿主命名空间里，监控与停止脚本能直接管；
- **代价**要知道：chroot 不隔离 PID，也不改文件属主——镜像内进程一律 **root**，
  所以 `lsof` 看不到监听端口、普通用户 `kill` 静默失败（EPERM），必须走 sudo（见 `docs/08#P40`）。

> 如果你确实要走 docker（例如参考官方 GDS+TEP2 路线）：控制台/脚本的端口探测要从
> `/proc/<pid>/cgroup` 取容器 id 再用 `docker port` 映射宿主端口，否则所有监控都会空转。
