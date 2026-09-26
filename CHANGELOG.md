# Changelog

## 2026-10-05 — KV 二级缓存移植回官方 0.30.0 新栈（rt-patch #9）

- 背景：2026-09-26 生产接管方已迁至官方 vLLM 0.30.0（宿主 venv + PYTHONPATH 运行时补丁，
  site-packages 零改动；操作文档在机器 `/home/ll/deploy/vllm-0300/README-0300.md`，
  本仓库 `stack-0300/` 收录其关键脚本与补丁产物，脚本已脱敏为 SUDO_PASS 注入）。
- **恢复功能不恢复缺省**：`patches-extra/dsh_kvoff_rt.py` 把旧栈 D 组验证过的
  c1（QSA 环形/空占位分组源头过滤）、c2（PP>1 私有 pinned）、c6（有界等待 + store 熔断
  只读降级 + 失败 ack 走 complete_store(success=False)）、c7（双向强制 Triton swap，
  绕开 cuMemcpyBatchAsync×PP2 NCCL-P2P 冻结流）移植为运行时钩子；c3/c5a/13/15 经源码
  核对确认已被上游吸收或不再触发。FN_KVOFF 缺省仍为 0，未配 connector 时钩子天然惰性。
- 验证：离线自检 24 项全绿（`stack-0300/selftest_kvoff_rt.py`）；inner 四态 dry-run
  正确。**未做 FN_KVOFF=1 实机验证窗口**（需停机 ~8 min，使用者择时执行）。


## 2026-09-29 — 生产收口（三场定案，脚本与文档增量同步）

- **KVOFF（CPU KV 二级缓存）正式退役**：生产 21.5 h `external_prefix_cache_hits_total=0`、物理钉住 ≈107 GiB、
  PP2 拷贝路径两种故障模式（09-23 抢占崩 / 09-24 卡死）→ 三处缺省（inner、server.js base+fallback、index.html 弹窗）
  翻为**关**。新增 `tools/patch-kvoff-default-off-0929.py`（幂等、`--revert`、`node --check` 门禁+自动回滚）。
- **YaRN×4 无罪（保留 1M 档）**：同协议 A/B —— 1M YaRN 接受率 30.2%/decode 93.7 vs 原生 262144 30.5%/97.4，
  差异在噪声内；参考机 65.7% 属 prompt 协议差异。详见 `docs/07-performance.md §7`。
- **PLE 统一 INT8+heap**：消除 envfile 跑 BF16+heap（95.4 GiB 不可回收）与三源缺省（INT8，49.2 GiB，decode 持平）的漂移；
  产物用 `scripts/quantize_ple.py` 再生（约 10 min、可断点续传、`--verify-only` 字节级自检）。


## v1.0.0 — 2026-09-23（首次公开，与生产实例同步）

生产实例：`http://<部署机>:18420/v1`，served `qwen3.8-flash-next`，1M 上下文，PP2 + MTP4。

### 新增
- `patches/`：22 个 vLLM 补丁（PP>1 解禁 8 / PLE 驻留 3 / MTP 校验 1 / KV 二级缓存 7 / 提速观测 3），
  含 `MANIFEST.tsv` 双哈希（pristine + patched）。
- `scripts/apply-patches.py`：幂等打补丁，应用后逐文件 sha256 校验、不符自动回滚、自动清 `__pycache__`。
- `scripts/quantize_ple.py`：PLE n-gram 表离线 INT8 量化器（95.4→48.3 GiB），带 `--verify-only` 字节级自检。
- `scripts/make-longctx-copy.py`：512K/1M 的 YaRN 配置副本（权重软链，零额外空间）。
- `scripts/fetch-image-rootfs.sh`：不用 docker，按 digest 拉官方定制镜像并解 rootfs（含 whiteout 处理）。
- `scripts/verify-deployment.sh`：五段体检（主机/补丁/模型/在线/稳定性），每项带期望值。
- `scripts/smoke-online.py`：在线功能验收（工具调用 / 思考 / 80K 长上下文标识复述 / 流式 / 标点探针）。
- `scripts/bench-fnx.py`：TTFT / decode / 预填充基准（已按思考模型口径修正 TTFT）。
- `scripts/host/`、`systemd-units/`：Gen2 早钩子、read_ahead udev、overcommit sysctl、功耗服务、chroot 服务、看门狗。
- `docs/01…09`：主机前置 → 镜像 → 模型 → PLE INT8 → 补丁详解 → 启动 → 性能 → **踩坑全集（50 条）** → 自验证记录。
- `tools/`：生产机实跑快照（argv / env / 进程树 / 启动画像 / metrics / 模型清单）。

### 与官方手册的偏离（两处，均有实测依据）
1. PLE n-gram 表由「BF16 常驻 128 GB 内存」改为「离线 INT8 + 逐行 scale（48.3 GiB），精度×位置四态」。
2. MTP 档位保持手册的 4（实测 k=1/2/3/4 = 72.1/82.2/90.4/92.7 tok/s 单调递增）。

### 已知限制
- KV 二级缓存（`FN_KVOFF=1`）在 PP2 的**抢占换出路径**仍会崩（`cuMemcpyBatchAsync error 1`）；
  不需要就设 `FN_KVOFF=0`。
- decode 为手册参考机的 70~75%，已归因主机平台代差（内存带宽/延迟），软件侧无对齐空间。
- 未在一台干净机器上端到端重跑全流程（只有一台生产机，不能停机重建）；各环节已分别闭环，见 `docs/09`。
