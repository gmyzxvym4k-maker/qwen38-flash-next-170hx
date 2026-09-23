# 09 · 自验证记录

本仓库交付前跑了 **8 项验证**，全部在**生产机（跑着 18420 实例的那台）**上执行。
下面是原始输出摘录与复跑方法。验证时间：2026-09-23 19:3x~20:0x CST。

---

## V1 · 补丁闭环（最关键的一项）

**命题**：任何人按 `docs/02` 从官方镜像解出纯净树，再打上本仓库 22 个补丁，得到的代码树
应与生产机正在运行的 rootfs **逐字节相同**。

做法：镜像 32 层 → 纯净 vllm 树（4864 文件）→ `cp -a` 到 `/tmp/verify-tree` → `apply-patches.py --apply`
→ 逐文件 `sha256` 与生产 rootfs 比对。

```
--- apply-patches --check (before) ---
[check] 共 22 个补丁：pristine=22 patched=0 unknown=0 失败=0
--- apply-patches --apply ---
  [22] APPLIED   v1/worker/gpu_worker.py
[apply] 共 22 个补丁：pristine=22 patched=0 unknown=0 失败=0
--- 逐字节比对 verify-tree vs 生产 rootfs ---
V1_RESULT ok=22 mismatch=0
```

✅ **22/22 字节一致，0 不一致。** 这也反向证明了 `patches/MANIFEST.tsv` 里
`pristine_sha256` 确实是上游原版（否则 `--apply` 第一步就会拒打）。

## V2 · 生产 rootfs 与仓库一致

```
$ python3 scripts/apply-patches.py --target <生产 vllm 目录> --check
[check] 共 22 个补丁：pristine=0 patched=22 unknown=0 失败=0
[OK] 补丁状态一致
```
✅ 生产机上没有任何"仓库里没有的第 23 个改动"，也没有漏打的补丁。

## V3 · INT8 量化器字节级闭环

**命题**：`scripts/quantize_ple.py` 是从零重写的（原脚本已在一次系统重装中丢失），
必须证明它与生产上那份 48.3 GiB 产物是**同一套量化定义**。

```
$ python3 scripts/quantize_ple.py --model <模型> --out /media/ll/data/ple --verify-only --rows 4000
[geo] src=model-00016-of-00017.safetensors shards=128 rows=320001536 hidden=160
[geo] header_len=20728 data_base=20736 table_data_start_in_file=20736
[verify] rows=4000 int8_mismatch=0 scale_mismatch=0
[verify] ✓ 抽样字节完全一致：量化定义与引擎侧产物互逆
```
✅ 随机 4000 行，int8 与 scale **全部逐字节相同**；`data_base=20736` 与产物 meta 记录一致
（正是 `docs/08#P13` 那条最贵的偏移知识）。

## V4 · 部署体检

```
$ SUDO_PASS=… bash scripts/verify-deployment.sh
── A. 主机前置 ──  overcommit=1 ✓  GPU 2 ✓  显存 65536 ✓  PCIe Gen 2 ✓  位宽 16/16 ✓
                   read_ahead 128 ✓  udev×2 ✓  gen2 钩子 ✓  功耗服务 enabled ✓
── B. 镜像与补丁 ── 补丁数 22/22 patched ✓
── C. 模型与产物 ── 分片 17 ✓  00016=102400512256 ✓  quant_method=auto-round ✓
                   int8.bin=51200245760 ✓  scale.bin=640003072 ✓  meta ✓
── D. 在线服务 ──── /health 200 ✓  served=qwen3.8-flash-next ✓  max_model_len=1048576 ✓
                   cmdline: block-size 1616 ✓ / mamba float32 ✓ / PP2 ✓ / moe auto ✓ / mtp 4 ✓ / 无 enforce-eager ✓
                   前缀缓存累计命中率 95.3% ✓   MTP 平均接受长度 3.74 ✓
                   冒烟推理 [1s] 你好 ✓   响应含 prompt_tokens_details ✓
── E. 稳定性 ────── dmesg Xid = 0 ✓
══════════ 汇总：PASS=37  WARN=4  FAIL=0 ══════════
```
✅ **FAIL=0**。4 个 WARN 的处置：
| WARN | 性质 | 处置 |
|---|---|---|
| 镜像版本串 `0.1.dev20073+g8e685d198` 与期望不完全等值 | 脚本判据过严 | 已改成"包含 `0.1.dev20073`" |
| swap 值取空 | `free -g` 解析边角 | 已加空值兜底 |
| 崩溃签名计数打印成 `0\n0` | `grep -c … \|\| echo 0` 双输出 | 已修 |
| **dmesg MCE 计数 = 1** | **真实硬件状况**（该机内存曾出过硬挂，见 `docs/08#P34`） | **保留为 WARN 不掩盖** |

## V5 · 脚本语法

```
bash -n：fetch-image-rootfs / flash-next-prewarm / flash-next-w4a16-inner / fnx-18420-watchdog /
         setup-chroot / start-flash-next-w4a16 / stop-flash-next-w4a16 / verify-deployment /
         gen2-early-launch / gen2-early-run / gpu-power-limit            → 11/11 ok
python3 -m py_compile：apply-patches / bench-fnx / make-longctx-copy / quantize_ple / verify_model / smoke-online
                                                                        → 6/6 ok
```

## V6 · 脱敏扫描

```
grep -rniE "<旧sudo密码>|192\.168\.[0-9]+\.[0-9]+|password\s*[:=]"  →  无命中
```
（唯一命中是 `MANIFEST.tsv` 里某个 sha256 恰好含子串 `3124`，属误报。）
所有脚本的 sudo 凭据改为 `SUDO_PASS` 环境变量或 NOPASSWD sudoers；主机地址写作 `<DEPLOY_HOST>`。

## V7 · 在线功能验收

```
$ python3 scripts/smoke-online.py
[1 工具调用] 0.7s finish=tool_calls name=get_weather args={"city": "北京"}
[2 思考模式] 0.8s reasoning=82 字 content='  9.8 大。'
[3 长上下文] 10.4s prompt=80684 cached=0 finish=stop reasoning=91 标识命中=True 输出='  zz-1790166214'
[4 流式] chunks=12 TTFT=1.13s
[5 标点自检] 0.9s 异常标点组合=0 输出='\n\n今天天气很好，我们去爬山；山上有风，也有云。'

✓ 在线功能验收全部通过
```
✅ 5/5。要点：80,684 token 长上下文里的**随机标识被正确复述**（说明 PLE 表与 RoPE 缩放都没坏）；
标点探针 0 异常（这是 `P13` 那类表错位最灵敏的早期信号）。

> **两点如实说明**：
> 1. 用例 3 第一次跑出 `content=''` —— 不是服务故障，是 `max_tokens=96` 全被思考吃掉了。
>    已把该用例调到 1024 并在脚本注释里写明（`docs/08#P15` 同源）。
> 2. 用例 2 模型答"9.8 大"，**事实错误**（9.11 > 9.8）。这是模型能力问题，不是部署问题；
>    本仓库的验收只判"reasoning 与 content 都非空、无乱码"，不判答案对错。

## V8 · 文档内链

遍历全部 `.md` 的相对链接并检查目标文件存在 → **0 断链**。

## V9 · 镜像引用仍然有效（2026-09-23 复核）

```
$ curl -sI -H "Authorization: Bearer …" -H "Accept: …manifest.list.v2+json" \
    https://registry-1.docker.io/v2/vllm/vllm-openai/manifests/qwen38-flash-next
docker-content-digest: sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8
```
✅ 与 `docs/02` 记录的 index digest 一致；本机 `blobs/` 解出的 amd64 manifest 为
`sha256:0aea3024…`，也与文档一致。

---

## 复跑清单

```bash
# 1) 体检（只读）
bash scripts/verify-deployment.sh
bash scripts/smoke-online.py http://<部署机>:18420 qwen3.8-flash-next

# 2) 补丁闭环（需要 rootfs 与镜像层；不需要停机）
bash scripts/apply-patches.py --target <生产 vllm 目录> --check

# 3) 量化器闭环（只读产物与 checkpoint，几分钟）
python3 scripts/quantize_ple.py --model <模型> --out <产物> --verify-only --rows 4000

# 4) 语法与脱敏
for f in scripts/*.sh scripts/host/*; do bash -n "$f" || echo "FAIL $f"; done
grep -rniE "<旧sudo密码>|192\.168\.[0-9]+\.[0-9]+" . | grep -v sha256
```

## 未验证 / 已知限制（不装懂）

| 项 | 状态 |
|---|---|
| 从零走完 `fetch-image-rootfs.sh → apply-patches → 启动` 全流程 | **未在一台干净机器上跑过**（只有一台生产机，不能停机做全量重建）。各环节已分别验证：镜像 digest 有效（V9）、补丁闭环（V1）、脚本语法（V5）。 |
| `fnx-watchdog.{service,timer}` 的开机自启 | 单元文件由生产脚本反推整理，**未在本仓库环境下实测触发** |
| D 组 KV 二级缓存的**抢占换出路径** | 生产开着（`FN_KVOFF=1`，96 GiB，`fill=2.6%`），但超长请求把 KV 顶到 ~89% 触发 preemption 时仍会 `cuMemcpyBatchAsync … error 1`（补丁 14 就是为看清它而加的 dump）。**结论：该功能仍属高风险**，见 `docs/05#D组` |
| decode 追平手册 | **做不到**，已归因主机平台代差（`docs/07#4`），需换平台 |
| 模型答案质量 | 不在本仓库验收范围（见 V7 附注 2） |
