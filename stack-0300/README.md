# stack-0300 —— 官方 vLLM 0.30.0 生产栈产物（18420，含内存二级缓存 KVCache）

本目录收录 **当前生产栈**（2026-09-26 起接管 18420）的完整复刻件：官方 `vllm==0.30.0`
（宿主 venv，**site-packages 零改动**）+ `PYTHONPATH` 运行时补丁（上游 8 hook + 本地扩展），
以及 **KV 缓存二级层（GPU 显存 → 宿主内存 96 GiB，SimpleCPUOffloadConnector）**——
生产实跑、有回载命中证据。唯一权威操作文档 = [`README-0300.md`](README-0300.md)
（**§11 = 当前生产定版快照**，含逐字判据与从零复现步骤）。

与仓库其余部分的关系：仓库主体（`patches/` 22 补丁 + `docs/01~09`）是**旧 chroot 定制镜像栈**
（vLLM v0.1.dev20073）的复刻件，保留为回滚路线；两代栈同一模型、同一端口，补丁语义映射见
`README-0300.md` §6。

## 目录内容

| 文件 | 机器路径（部署机） | 说明 |
|---|---|---|
| `README-0300.md` | `vllm-0300/README-0300.md` | **权威文档**：补丁架构 / KV 二级缓存全史（§6.1~6.3）/ 回滚 / **§11 生产定版** |
| `patches/sitecustomize.py` | `vllm-0300/patches/` | 上游运行时补丁入口（PP>1 解禁、auto-round PLE、>64GiB 锁页分块、FakeTensorMode 泄漏诊断等 8 处 hook；含 09-27 FP8-PLE 加载契约修正）。正源公开仓库 `CyrilCN/qwen38-flash-170hx-patches`，本副本=生产现行版 |
| `patches-extra/sitecustomize.py` | `vllm-0300/patches-extra/` | 本地扩展入口（#8 多模态 warmup 跳过 + #9 合并 dsh_kvoff_rt 钩子） |
| `patches-extra/dsh_kvoff_rt.py` | 同上 | 经典 OffloadingConnector 移植层 c1~c11（**已退役**，缺省惰性零影响；总闸 `DSH_KVOFF_RT_DISABLE=1`） |
| `inner-flash-next-0300.sh` | `vllm-0300/bin/flash-next-0300-inner.sh` | argv 装配（`FN_SIMPLE_OFFLOAD=<GiB>` 一等开关，与 `FN_KVOFF=1` 互斥拒启；参数体检 CONSUMED/NOOP_NOTE） |
| `start-flash-next-0300.sh` / `stop-flash-next-0300.sh` | `vllm-0300/` | 宿主入口 / 停止（SIGTERM→90s→SIGKILL 兜底，绝不先杀 worker） |
| `fnx-18420-watchdog.sh` + `systemd/fnx-18420-watchdog.{service,timer}` | `/home/ll/deploy/`、`~/.config/systemd/user/` | 30s 探活自愈；卡死 py-spy 取证；**自愈安全模式会剥离 offload 档位**（防崩溃循环，带档回归需手工重放 launch.env） |
| `kvoff-accept.sh` / `kvoff-churn.py` / `kvoff-soak-monitor.sh` / `tools/kvoff-restore-prod.sh` | `/home/ll/deploy/` | 生产档验收 / 挤池判据 / 5 分钟粒度 soak 记录（health·KV 水位·外部命中·Xid·内存）/ 恢复生产 |
| `selftest_kvoff_rt.py` / `selftest_kvoff_c10.py` / `selftest.py` / `selftest_extra.py` | `vllm-0300/` | 补丁层离线自检（不占 GPU；改动补丁后必跑） |
| `kvoff-c8-window.sh` / `kvoff-c10-window.sh` / `kvoff-fp8-window.sh` / `kvoff-*-{probe,verify,fresh,local,needle,scale,exact,tail,dbg,mon}.py` | `/home/ll/deploy/` | c1~c11 攻关期的实验窗口与判定小工具（历史证据可复跑） |
| `ROUND2-verification.md` / `ROUND3-simple-offload-验收手册.md` / `README-c10-c11.md` | —— | KV 二级缓存三轮攻关证据链（假 0 命中根因、mamba 竞态、Simple 换道与验收） |
| `quantize_ple_fp8.py` / `quantize_ple_fp8_v2.py` | —— | PLE FP8 离线量化器（v2 符合加载契约；GA100/SM80 无 FP8 单元硬件封路，留作记录） |
| `tools/live-cmdline-0300.txt` / `tools/live-env-0300.txt` | —— | **生产实跑真值快照（2026-09-27 20:20，读自 /proc/<APIServer>）** |

## 内存二级缓存（KVCache）一分钟版

- **开法**：`FN_SIMPLE_OFFLOAD=96 bash start-flash-next-0300.sh`（launch.env / 8889 控制台同效）；
  实际给引擎的是 `--kv-offloading-size 96` + `VLLM_USE_SIMPLE_KV_OFFLOAD=1`。
- **容量**：CPU 档 ≈ 348.6 万 token = **2.9× GPU 池**（1,207,262 token / 776 块）；
  只对「已被挤出 GPU 池、之后又被重发」的长前缀产生收益。
- **实测**：挤池 >1.2M token 后重发，~100k prompt 命中 **97%**、~40k **93%**、~12k **80%**
  由内存档回载（零头=每请求末块 ≤1616 不入库）；验证码精准复述=内容无损；
  生产 45 分钟 `vllm:external_prefix_cache_hits_total`=143,824。
- **为什么安全**：Simple 连接器显式 `SupportsHMA`，源码跳过 `has_positionally_stable_blocks=False`
  的组（mamba/GDN）→ 只卸载/回载位置稳定的注意力组；拷贝在后台线程独立流上按
  compute-done 事件排序完成，不回踩经典连接器在 PP2 上的两类事故
  （cuMemcpyBatchAsync 冻结 compute 流 / store 期间 mamba 活写竞态）。
- **经典连接器退役史**（c1~c11：环形分组排除 / PP 私有 pinned / 公共区前缀和 / 有界等待+熔断 /
  aggregate 丢件悬挂根因…）全文见 `README-0300.md` §6.1~6.3 与 ROUND 文档——这段弯路本身就是
  本交付件最有价值的踩坑记录。

## 脱敏说明

脚本内 sudo 凭据一律 `SUDO_PASS` 环境变量注入（未设即报错退出）；内网 IP 写作
`<DEPLOY_HOST>` / `<EXTERNAL_HOST>`；机器路径按现状保留；**git 历史已经净化重写**
（发布时全文扫描口令/IP 零命中）。
