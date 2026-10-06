# Changelog

## 2026-10-06 晚 — ★vLLM 升到 0.31.0 + 「二级缓存·CPU」真正开起来（96 GiB，实测 95.4% 外档回载）

- **【起因】** 用户问「二级缓存·CPU 没正常开启？」→ 核实：当时 18420 跑 chroot 旧栈（`0.1.dev20073`），
  `cmdline` 无 kv-transfer/kv-offloading、`external_prefix_cache_queries=0`；旧栈上三条路都不通
  （经典 `OffloadingConnector`=P54 回载不等价；`FN_SIMPLE_OFFLOAD` 旧 inner 不接线；
  镜像自带的 SimpleCPU 模块缺 #13 clamp ⇒ 必复现 P61 segfault）。
- **【做法】** 新建/复用宿主 venv `/media/ll/data/vllm-0310-env`（vLLM 0.31.0 + torch 2.13.0+cu130 +
  flashinfer 0.7.0.post1 + cu13 软链 + dsh-logger-pkg），补丁全部运行时注入
  （`stack-0310/patches/` 上游 8+2 钩、`stack-0310/patches-extra/` rt-patch #11/#13）。
  先离线证明 **#13 对 0.31.0 直接可用**（自检 13/13 PASS 含金丝雀），再用 `vllm serve --help=all`
  逐 flag 核对（`--kv-offloading-size/-backend`、`-cc.*`、`--no-enable-flashinfer-autotune` 等全在）。
  PLE 表用 `scripts/quantize_ple.py` 从 W4A16 checkpoint 重新生成 INT8 产物
  （`/media/ll/data/ple`，抽样 10 万行 **字节不一致=0 / relMSE=4.360e-05 / 行余弦 0.9999784**），
  经 `DSH_PLE_MMAP=1` 走可回收页缓存（省 44 GiB）。
- **【踩坑 P65】** 新写的 inner 忘了 source `launch.env` ⇒ 参数静默回落内置缺省
  （`max_model_len=262144`、`kv_offloading_size="None"`），已补通道并加「已加载参数文件 N 项」判据行。
- **【生产参数（与 chroot 那轮逐键同源）】** 1M(YaRN×4 副本) + TP1×PP2(26,22) + block1616 +
  mamba float32 + moe auto + MTP4 + seqs2 + mbt8192 + gpu-mem0.95 + async + prefix-caching +
  prompt-tokens-details + BF16 KV + `--disable-custom-all-reduce` + NCCL_P2P_DISABLE=1 +
  思考 medium + 采样裸档 + **FN_SIMPLE_OFFLOAD=96** + **FN_PLE_MMAP=1**。
- **【验收】** GPU KV 池 **1,210,374 token**（1M 并发 1.15×）；两 rank CPU 档 2224/2157 块（各 ≈48 GiB）
  → **clamp 行确认 `2224 -> 2157`**；`tools/kvoff-accept-0310.py`：建档 55,924 tok → 挤池 **1,434,381 tok**
  → 重发同文档 `cached=53,328（95.4%）`、`external_prefix_cache_hits_total` 增量 **53,328**、
  验证码 `CODE-287365-3126` 逐字复述、耗时 9.7 s → **1.2 s**；全程 **Xid=0**、实例段零 segfault/EngineDead。
  速度口径：端到端 72~82 tok/s（与 chroot 栈持平），MTP 接受计数正常增长。
- **【管理台收口】** `tools/patch-stack-0310-1006.py`：`SCRIPT_MODELS.scriptNew/stopScriptNew` → 0.31 对、
  栈哨兵改判 `vllm-0310/DISABLED`、`altLogs` 加 0310 日志、base 同步 `kvoff=simple/96 + pleInt8=1 + seqs=2`、
  快启预设固化「双卡PP2-1M-MTP4-二级缓存96G」、看门狗栈阶梯加 0.31；已验证
  `resolveStartScript → vllm-0310/start-flash-next-0310.sh`，控制台「vLLM」页版本徽章显示 `0.31.0`。
- **【仍在观察】** `DSH_PLE_MMAP` 查表路径在 0.31 属首跑；历史 segfault 窗口是带档就绪后 26~71 分钟，
  已挂 `tools/soak-0310-monitor.sh`（6 h，每 5 分钟一行写 `/home/ll/deploy/soak-0310.log`）。
  回滚阶梯与判据见 `stack-0310/README-0310.md` §6。
## 2026-10-06 傍晚 — ★X99 双卡机（251 GiB 内存）配置档纠偏：W4A16-AutoRound 恢复生产 + PP 档对齐在位卡数

- **【故障】** `<DEPLOY_HOST>`（HUANANZHI X99-T8 / E5-2696 v4 / 8×32 GiB / 2×CMP 170HX）按 8889「启动」
  起不来 `Qwen3.8-Flash-Next-W4A16-AutoRound`：实例日志尾部只见上一轮 `[shutdown] … SIGTERM`，
  新启动连 Worker 都起不齐；`dmesg` Xid=0、MCE=0，机器不硬挂。
- **【根因】** 配置档整体还是 10-05 那台「EPYC+3 卡+32 GiB」机的描述：
  `SCRIPT_MODELS.base.pp=3`、两条快启预设 `gpuCount:'3'`、`base.pleInt8='1'`（而 W4A16 的
  INT8 n-gram 表产物 `/media/ll/data/ple` 从未生成，`ple-w8a8` 那份属另一 checkpoint，跨用会
  「内容词对、标点乱」）、wrapper `RA_LOAD` 缺省 16、udev `read_ahead_kb=16`；
  且 `modelPath/longCtxModelPath` 仍指 `Channel-INT8-w8a8`。
  2 张卡在位却下发 `--pipeline-parallel-size 3` ⇒ 必然起不来（判据 `lspci -d 10de:`）。
- **【修复】** `tools/patch-w4a16-restore-1006.py`（幂等 + `--revert`，一次收口六处：
  server.js 模型目录/base.pp=2/base.pleInt8='0'/`pleInt8Dir`+plan 下发 `FN_PLE_INT8_DIR`、
  两预设 gpuCount 2 + pleInt8 0 + 名称纠偏、wrapper `RA_LOAD` 128、udev 128、
  看门狗内置回退档 `FN_PLE_INT8=0`、删 `fnx-manual-stop` 闩锁恢复自愈）。详见 `docs/08-pitfalls.md` **P64**。
- **【启动（=刚才的参数，逐键复刻 16:11 那轮 launch.env，只改两处硬件不匹配项）】**
  1M(YaRN×4 副本) + TP1×**PP2** + block 1616 + mamba float32 + moe auto + MTP4 + seqs2 + mbt8192 +
  gpu-mem 0.95 + async-scheduling + prefix caching + BF16 KV + `--disable-custom-all-reduce` +
  NCCL_P2P_DISABLE=1 + PLE=**BF16 磁盘驻留** + 思考 medium + 采样 t1.0/top_p0.95/top_k20/pp0/rp1.0 + 二级缓存关。
  18:40 发起 → 18:51 health=200（约 11 分钟：PLE 表 mmap → 权重 143 s → compile+建池+图 176 s）。
- **【验收】** `GPU KV cache size: 1,199,694 tokens`（1 M 请求并发 1.14×，比 3 卡 PP3 的 1,099,915 更大）；
  判据行 `[FN-PLE-LOC] PLE 表走 BF16 磁盘驻留（mmap safetensors，零堆…）` +
  `[FN-PLE-DISK] n-gram table attached from model-00016-of-00017.safetensors: dtype=torch.bfloat16 contiguous=95.37 GiB`；
  temp0 冒烟问答正确、无 uct/duct 病理；`/metrics` MTP 接受 token 计数递增（投机生效）；
  直连 18420 与走 8889 代理（别名 `qwen3.8-flash-next`）两条路都通；`prompt_tokens_details.cached_tokens` 有值；
  Xid=0、MCE=0、内存 used ~8 GiB + 页缓存 77 GiB（表全驻留）；数据盘 read_ahead_kb=128（root trigger 复验通过）。
- **【如实说明】** ①本机 W4A16 的 INT8 表产物不存在 ⇒ 这一轮走 BF16（与 INT8 档历史实测速度持平，
  代价是冷启动多 2~4 分钟、页缓存多吃 ~47 GiB）；要 INT8 档需先跑
  `scripts/quantize_ple.py --model <W4A16 目录> --out /media/ll/data/ple`（约 11 分钟）再把预设/弹窗的
  PLE 精度切回 INT8。②运行实例的采样是「刚才」那轮的裸档 t1.0/pp0/rp1.0，与 09-29 反循环定档
  （0.6/0.2/1.15）不同，属按用户要求原样复刻；如复发言题，改 `launch.env` 的 `FN_GENCFG` 并重启即可。
## 2026-10-05 上午 — ★126 机（32GB 内存）启动页 heap 参数 OOM 循环定版 + 双层钳位闸

- **【故障签名】** 管理页启动 18420 → 引擎在加载期死：日志 `RuntimeError: PLE offload worker exited during startup` + `Engine core initialization failed`；`dmesg` 连续 `oom-kill`（被杀进程 anon-rss ~24GiB 持续增长中）。08:58/09:01/09:04/09:2x/09:2x 五连，看门狗每 3 分钟按旧 envfile 原样重拉 → 无限循环。
- **【根因链】** ①`index.html` 弹窗脚本模型覆盖块把 `pleLoc` 写死 `'heap'`、二级缓存缺省写死 `simple 96GiB`（09-23 大内存机时代的"生产现状"）；②`server.js scriptModelDefaults()` 未把 10-05 换装定版进 `SCRIPT_MODELS.base` 的 `pleLoc:'disk'/pleInt8:'1'/kvoff:'0'` 透传给前端；③浏览器里补丁前加载的旧页面缓存仍提交 heap；④`launch.env` 由失败启动落盘为 heap，看门狗忠实重放。本机（华勤 H12D-8D，32GB 内存）INT8 heap=48.3GiB 匿名堆不可回收 → PleOffloadWorker 必被 OOM 杀。
- **【修复（三层，均可幂等重放）】**
  1. `tools/patch-pagefix-1005.py`：`scriptModelDefaults()` 透传 `pleInt8/pleLoc/kvoff/kvoffGiB`（base 未声明回落安全档）；前端 smd 块三行改为跟随 `smd.*`（单一真源=SCRIPT_MODELS.base）。
  2. `tools/patch-pleclamp-1005.py`：低内存双闸——plan 层 `heap && os.totalmem()<100GiB → 强制 disk + 页面警告`；inner 层同判据 `[FN-PLE-CLAMP]`（覆盖手动命令/看门狗/旧 envfile 路径）。本包 `scripts/flash-next-w4a16-inner.sh` 已含钳位段。
  3. `flash-next-w4a16-launch.env` 的 `FN_PLE_LOC` 同步为 `disk`（备份 `.bak-pleclamp-1005`）。
- **【验证】** 看门狗 09:34 拉起 → 09:37:18 health=200；判据行 `[FN-PLE-INT8] n-gram table attached ... (mmap, zero heap)`；KV 池 1,129,450 tok（262144 档 + gpu-mem 0.97）；chat 冒烟正常；此后零新 oom-kill、Xid=0；内存 used 11Gi + 可回收 18Gi。plan 钳位离线证据：node vm 执行 `scriptModelLaunchPlan` 以 `pleLoc:'heap'` 提交 → 返回 `env.FN_PLE_LOC='disk'` + 中文警告。
- **【运维判据】** 32GB 机三档红线：INT8 heap / BF16 heap / 大 GiB 二级缓存 全部不可用；唯一可行=INT8+disk+kvoff=0。补内存 ≥96GB 后 heap 档才重新有意义（钳位阈值 <100GiB 自动放行，无需回退补丁）。
## 2026-10-03 深夜 — ★复读问题解决定版 + 生产切回 chroot 旧栈 + 功耗墙 210 W

- **【主题：复读（循环输出）问题已解决】** 完整攻关归档 → `docs/08-pitfalls.md` 新增第九节 **P62**：
  - 三层因果链定案：①直接触发=会话上下文污染（病理串进历史后无可救药，新会话是唯一出路；
    「新会话是否循环」=第一判别题）；②土壤=1M 上下文 × xhigh 思考 × PLE n-gram 参与 logits 的
    数值退化边缘（temp0 下 fresh/prefix-cache 两条路径续写即句级漂移——字节级等价不可作定罪判据）；
    ③历史放大器=SM74 回收补丁窗口（09-29~30 病发期与其冷启动窗口重合，10-01 已回滚 SM70，
    rmmod 不清粘滞 live-override 寄存器、必须冷重启）。
  - **生产定档采样（10-03 夜最终）= `t0.7 / top_p0.95 / top_k20 / min_p0 / presence0 / rep1.15`**
    （演变：t1.0→t0.3（治接受率、引发 P14 复读）→09-21 定 0.6 组合→09-27 回裸档→
    09-29 uct/duct token 级硬循环复发→用户拍板回 0.6 组合→10-01 观察期 0.2/1.15→
    10-03 夜旧栈上定档 0.7/0/1.15=当前生产）。
  - **防复读复刻五条清单**（P62 内详述）：采样档写进两套 inner 的 `GENCFG_DEFAULT`（本仓库
    `scripts/flash-next-w4a16-inner.sh` 已同步为 t0.6/pres0.2/rep1.15 现役档）；病理串绝不回流；
    不用 SM74 补丁；MTP 档保持 block1616 合法的 K≤4；采样参数四源同步 + 读
    `/proc/<APIServer>/cmdline` 终验（历史上多次「文件都对、运行时另一个值」=envfile 残留+竞写）。
  - 监控件：机器侧 `loop-sentinel.py`（病理正则 `uct|duct` + 污染探针）。
- **生产栈路由现状（如实记录）**：18420 实跑=**chroot 旧栈 vLLM v0.1.dev20073**
  （17:27:47 起，PP2+MTP4+block1616+1M YaRN+OffloadingConnector 96 GiB 经典连接器；
  `vllm-0300/DISABLED` 哨兵在位，控制台/看门狗随之路由旧栈）。三套栈路由优先级：
  `sglang-18420/ACTIVE` ＞ `vllm-0300/DISABLED` ＞ 默认新栈。实跑真值快照入仓
  `tools/live-cmdline-w4a16.txt` / `live-env-w4a16.txt` / `launch.env.w4a16-current`。
  定档演进（10-03 夜最终态）：实跑与服务端定档统一为 **t0.7 / top_p0.95 / top_k20 / min_p0 /
  presence0 / rep1.15**（旧 chroot 栈上验证的稳定反循环档），快启预设固化为
  `current-norepeat-1m-mtp4-kvoff96-oldstack`（名称含「不复读」；旧的 t0.3 档降级为
  legacy 条目）。注意 09-29 记忆里的 0.6/0.1/1.05 组合已被该档接替——P62 表格里两档都有效，
  当前生产用 0.7/0/1.15。
- **功耗墙 210 W 定版**：双卡 `power.limit=210W`、persistence=Enabled、开机自动重放。
  生效链与陷阱（systemd drop-in 覆盖脚本缺省值）见新增件
  `systemd-units/gpu-power-limit.service.d-pl-console.conf`。沿革 200→250→210→300→250→210。
- **脚本现行版同步（md5 对账机器）**：`scripts/flash-next-w4a16-inner.sh`（9bbc814f…，含
  PLE_INT8_DIR 可覆盖、FN_ENFORCE_EAGER 键名修复、SimpleCPU 旧栈防呆 WARN、采样缺省
  t0.6/pres0.2/rep1.15）、`start-/stop-flash-next-w4a16.sh`（全量透传 FN_* 动态扫 +
  manual-stop 闩锁，脱敏为 SUDO()/NOPASSWD 框架）；`stack-0300/quickstart-presets.json`
  同步机器现行版（flashnext=current-1m-mtp4-simplecpu100，uncensored=unc-1m-mtp4-xhigh）。
- 诚实声明：实跑与定档之间的上述漂移**未被本仓库擅自修改**——机器现态以 `tools/live-*`
  快照为准，目标态以 P62 清单为准，处置（是否回调采样、是否清旧栈连接器）由部署者拍板。

## 2026-10-03 — 8889 管理台新增「vLLM」标签页 + 两处运行时归因修复；生产栈切回 vLLM 0.30.0

- **生产栈切换（用户指令）**：18420 由 SGLang 0.5.21 切回官方 vLLM 0.30.0 栈。
  SGLang SIGTERM 优雅退出（7 s、显存 10 s 归零）→ `source vllm-0300/launch.env` →
  `start-flash-next-0300.sh`，**就绪 4.5 min**；health 200、验证码精准复述、MTP 接受率 39.5%、Xid 0。
  托管切换：`fnx-sglang-watchdog.timer` disabled、`fnx-18420-watchdog.timer` enabled、`fnx-manual-stop` 清除。
- **SGLang 启动失败根因（同时修掉，留档）**：`sglang-inner.sh:20-21` 的 `${SG_GENCFG:-{json}}` /
  `${SG_CT_KWARGS:-{json}}` 写法——bash 在 default 第一个未转义 `}` 处终止展开，**变量有值时尾部多余
  `}` 被字面附加** → `argparse invalid loads value: '…"xhigh"}}'` → 秒退 → 看门狗每 60 s 重拉、
  连续失败 4 h。修法：`VAR="${SG_X:-}"; [ -n "$VAR" ] || VAR='{json}'`（= 本项目 docs/08 **P20** 早已记录的坑）。
- **新增 8889「vLLM」标签页**（形态复刻同日 SGLang 页）→ [`console-ui/`](console-ui/README.md)：
  独立页 `vllm.html`（MTP 逐位接受率 / KV 池与 block 配置取自 `cache_config_info` 标签 /
  前缀缓存 / CPU KV 二级缓存 / 延迟分位 / 累计吞吐按来源拆分），＋两处归因修复补丁 ＋渲染实测脚本
  （DOM stub + 真实端点数据，**17/17 PASS**）。
- **归因修复①**：`findVllmPidByPort` / model-manager 实例发现补 `listVllmInstances()` 兜底——
  root 启动让 `lsof` 看不见监听端口，旧兜底 `pgrep -f "[v]llm.entrypoints"` 又匹配不到
  0.30.0 实跑的 `vllm serve` CLI 形式 → 实例 `runtime=unknown / gpu=null / pid=null`。
  修后：`runtime='vllm' gpu='0' gpus=[0,1] pid=<非空>`。
- **归因修复②**：`/v1/internal/stats` 的 vLLM 主/从实例对象补 `runtime: 'vllm'`
  （SGLang 侧两处都有、vLLM 侧两处都漏；前端运行时徽标与二级缓存口径分支依赖该字段）。
- 配套：补丁幂等（锚点缺失明确报错退出）、server.js 语法门禁、`PAGE_VERSION` r7 → r8。
- **栈路由哨兵修复（同日追加，用户报「二级缓存·CPU 功能没法使用」）**：`server.js` 的
  `resolveStartScript`/`resolveStopScript` **SGLang 优先级高于 0.30.0**，判据是
  `sglang-18420/ACTIVE` 哨兵；该哨兵 07:31 启用 SGLang 时创建、回切 vLLM 时无人清（全仓无脚本
  维护它）→ 控制台「启动」拉 SGLang、「停止」调 SGLang 脚本（`WARN: 仍有残留 pid=…`），
  vLLM 专属的 CPU KV 二级缓存因此不可达。修法：①清孤儿哨兵；②根治=两栈启动脚本互斥配对
  （vLLM 启动清哨兵 / SGLang 启动置哨兵）。验证：提 server.js 真实函数在沙箱执行，
  `resolve*Script` 均返回 `vllm-0300/*`（`ROUTE_OK`）。
- **附带修复 `STOP_LIST_ONLY` 透传**：停止脚本自提权时 `sudo bash "$0"` 清环境 → 只读探测标志
  丢失、退化成真停实例（排查中误停过一次生产实例）；改为 `VAR=value` 显式透传，复验只读语义生效。
- **二级缓存按用户选择开启 48 GiB**（同日收尾）：栈路由修好后经控制台同款链路重启，
  `FN_SIMPLE_OFFLOAD=48` 已钉入 `vllm-0300/launch.env`（重启/看门狗自愈会保持）。
  引擎侧判据：`SimpleCPUOffloadConnector role=SCHEDULER per_rank=24.00 GB world_size=2`
  + `Allocating 1078 offload blocks`（rt-patch #13 握手 clamp 1112→1078，防 PP 越界 segfault 生效）；
  内存 105 → **153 GB**（+48，与预估一致，余 95 GB 供权重页缓存）；推理验证码无损。
  选 48 而非 96 的理由：本机 PLE 表在 0.30.0 栈恒为 BF16 锁页 95.4 GiB（`FN_PLE_INT8` 在该栈是
  死变量），96 档会把权重页缓存压到 ~50 GB（需 79 GB），且该机有 MCE 硬挂史。
  前端口径：`metric_kind=simple`（SimpleCPU 不暴露 `kv_offload_*` 族，只有
  `external_prefix_cache_*`），卡片显示「48 GiB · 已启用 / SimpleCPU档 · 查 N/中 M」；
  `hits=0` 属正常（需前缀被挤出 GPU 池后重发才产生回载命中）。
  真值快照已更新 `stack-0300/tools/live-launch.env`。

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
