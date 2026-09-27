# Changelog

## 2026-09-27 深夜 — rt-patch #11：二级缓存 segfault 根治 + 生产真值快照刷新（v1.1.2）

- **P59 误诊更正（P60 入档）**：当日三次间歇 segfault（17:38 / 20:38 / 21:51）的"attrIdxs 单标量
  越界"结论被证伪——按 cuMemcpyBatchAsync 契约 `attrsIdxs` 长度=numAttrs（本栈=1），上游标量传参
  本就合法，rt-patch #10 系 no-op（打满 #10 的实例 21:51 仍崩，core=`cuda-EvtHandlr.7142`、
  栈全在 libcuda）。真根因=**批量 API 本身在本机驱动 610.43.03 + CMP 170HX 定制固件上不稳定**：
  纯 ctypes 探针 60 次内复现段错误；逐块 `cuMemcpyAsync` 压测 2000 轮 ×16 块 ×64KB（33.5 GB）零错误。
- **rt-patch #11（生产现行）**：`patches-extra/dsh_simple_offload_rt.py` 整版升级，copy_blocks
  改逐块 `cuMemcpyAsync` 同流入队（地址算式与批量版逐字节等价、事件/线程模型零改动，
  二级缓存功能保留）。A/B 开关 `DSH_SIMPLE_BATCH=1`（回批量路径，会崩，仅取证）/
  `DSH_SIMPLE_OFFLOAD_UPSTREAM=1`（不打钩）。
- **自检体系 v11**：`selftest_simple_rt.py` 重写为纯 CPU mock 的 G1-G4（挂载幂等 / 地址算式对照 /
  入参防御 / 开关行为），全绿；真机端到端另跑双向+乱序+浸泡 2000 轮全 PASS。
- **inner 新增 core dump 取证链**（`ulimit -c unlimited` + `core_pattern=/media/ll/data/cores/…`，
  强制落数据盘）——本次根因判定即靠该 core + gdb 完成。
- **生产真值快照刷新（2026-09-27 23:11）**：`tools/live-cmdline-0300.txt` / `live-env-0300.txt` /
  `live-launch.env` 三件套改读自 #11 上线后实例（pid 464646，22:24 就绪）。快照时点运行 40 分钟、
  **已越过历史崩溃窗（26~59 分钟）**：external hits 2,526.8 万 token / 外部命中率 91.6%，
  零 segfault、零 Xid，8889 控制台识别正常（18420 / GPU 0+1）。
- README-0300 §11 标题/引擎行/累计行同步为 #11 现行态；docs/08 死路表该行改写（批量 API 整体进死路）。

## 2026-09-27 晚 — 生产定版发布：内存二级缓存 96 GiB 常驻（v1.1.0）

- **当前生产形态首次完整入仓**（此前仓库主体是旧 chroot 栈）：官方 vLLM 0.30.0 运行时补丁栈 +
  `SimpleCPUOffloadConnector` 宿主内存 KV 二级层 96 GiB。生产实例 19:34:40 拉起、至快照无卡死无新增 Xid。
- 新增生产真值快照：`stack-0300/tools/live-cmdline-0300.txt`、`live-env-0300.txt`（读自 `/proc/<APIServer>`）。
- `README-0300.md` 新增 **§11 生产定版**：形态表 / 容量账（CPU 档 ≈348.6 万 token = 2.9× GPU 池）/
  实测命中率（~100k→97%、~40k→93%、~12k→80%、external hits 45 分钟 143,824）/ 从零复现六步 /
  运维件清单 / 风险与第一手处置。
- 补齐复刻缺口：`stack-0300/patches/sitecustomize.py`（上游 8-hook 运行时补丁生产现行版）、
  `fnx-18420-watchdog.sh` + systemd 单元（含自愈安全模式：剥离 offload 档防崩溃循环）、
  `kvoff-soak-monitor.sh`。根 README 重构：§1=当前生产（含内存二级缓存专节），旧 chroot 内容
  整体标注为回滚路线（§2 起）。
- `docs/08-pitfalls.md` 新增「八、内存二级缓存专属坑」P51~P58（metrics defs 断言、aggregate 丢字段、
  模板开关假 0 命中、mamba 活写竞态、拷贝 API×中继冻结、pinned 物理账、flock fd 继承、判据假 PASS）。
- **安全**：清除了仓库中泄漏的真实 sudo 口令（3 处）与内网 IP（7 处），全部改为
  `SUDO_PASS` 注入 / `<DEPLOY_HOST>` 占位，并**重写了 git 历史**（旧 tip 的口令不再可达；
  建议部署机择期改口令）。


## 2026-10-05 — 控制台启动链路审计：`FN_SCHED_POLICY` 接线 + §9 纠偏

- **审计结论**：`8889` 控制台 → 官方 0.30.0 的启动链路**已正确指向**（09-26 完成）。本次对
  `plan → wrapper → inner` 三层做 `FN_*` 穷举对账，7 项健康：栈感知「双字段 + 哨兵」、
  `inner` 不串栈（`startScript === sm.script` 守卫）、wrapper 动态透传（`compgen -e`）、
  长上下文三档链路完整（`maxModelLenLong/512` 与两个 `longCtx*ModelPath` 均在条目内）、
  采样三处一致、看门狗 enabled+active、三个 0.30.0 快启预设齐备。
- **唯一真缺口已修**：`FN_SCHED_POLICY` —— plan 在用户选非 fcfs 时下发（`server.js:828-829`），
  而 inner 既不消费也无 `NOOP_NOTE_` → 只打一条「收到本脚本未实现的参数」警告、引擎仍走 fcfs。
  已在 inner 接线 `--scheduling-policy` 并登记进 `CONSUMED`
  （`stack-0300/inner-flash-next-0300.sh`，备份 `.bak-schedpolicy-1005`，
  幂等补丁 `tools/patch-schedpolicy-1005.py`，`--revert` 可回滚）。
  验证三重：`bash -n` 通过 / **缺省路径 dry-run argv 与改动前逐字节相同（零漂移）** /
  `=priority` 落到 argv 且警告消失。兼容性核过源码：`AsyncScheduler` 继承 `Scheduler` 的
  waiting 队列与抢占逻辑，故与 `--async-scheduling` 可共存。
- **文档纠偏**：`stack-0300/README-0300.md` §9 原写「`script/inner/stopScript/log/envFile`
  全部改到新栈」，那是 09-26 首版做法；生产当晚已演进为可回滚的双字段方案。§9 已改写，
  并**标注 `redirect-console-watchdog-0300.py` 已过时——在现行方案上重打会把 `script` 也改成
  新栈路径、废掉 `DISABLED` 回滚**（新增 §10 待办 8：补一个与现行方案一致的幂等重打脚本）。
- 生产零影响：该变量只在用户显式选非 fcfs 时进 argv，缺省（fcfs）路径逐字节不变；
  实例全程 health=200，未重启。

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
