# stack-0300 —— 官方 vLLM 0.30.0 新栈产物（192.168.1.127:18420）

本目录收录 2026-09-26 起接管 18420 生产的**官方 vLLM 0.30.0 栈**（宿主
`/home/ll/vllm-env` + PYTHONPATH 运行时补丁）的关键产物。与仓库其余部分
（旧 chroot 定制镜像 v0.1.dev20073 路线的 22 文件补丁组）的关系：同一模型、
同一端口的两代栈；旧栈的补丁语义如何映射到新栈见这里与机器上的
`/home/ll/deploy/vllm-0300/README-0300.md`（**唯一权威操作文档**）。

## 目录内容

| 文件 | 机器路径 | 说明 |
|---|---|---|
| `patches-extra/dsh_kvoff_rt.py` | 同名 | **rt-patch #9（2026-10-05）**：KV 二级缓存（OffloadingConnector）PP2 移植，语义对应旧栈 c1/c2/c6/c7（D 组） |
| `patches-extra/sitecustomize.py` | 同名 | 本地扩展入口（#8 多模态 warmup 跳过 + #9 合并 dsh_kvoff_rt.PATCHES） |
| `inner-flash-next-0300.sh` | `bin/flash-next-0300-inner.sh` | argv 装配（含 1005 新增的 FN_KVOFF/FN_KVOFF_BYTES/FN_KVOFF_WAIT_TIMEOUT 块） |
| `start-flash-next-0300.sh` / `stop-flash-next-0300.sh` | 同名 | 宿主入口/停止 |
| `selftest_kvoff_rt.py` | 同名 | rt-patch #9 离线自检（24 项判据，不占 GPU） |

## rt-patch #9（KV 二级缓存恢复）要点

- **缺省惰性**：不配 `--kv-transfer-config` 时 offloading 模块不被导入，钩子不触发，
  对生产零影响；紧急总闸 `DSH_KVOFF_RT_DISABLE=1` 使全部回调变 no-op。
- 启用：`FN_KVOFF=1 [FN_KVOFF_BYTES=<字节>] bash start-flash-next-0300.sh`，
  或 8889 弹窗「CPU KV 二级缓存」=开（plan 下发 FN_KVOFF/FN_KVOFF_BYTES）。
- 移植取舍（对照旧栈 D 组）：c1 分组过滤、c2 PP 私有 pinned、c6 有界等待+store
  熔断只读降级、c7 双向强制 Triton 绕开 cuMemcpyBatchAsync；
  c3/c5a 与 fill 观测指标（03/13/15）不再需要——metrics 断言源头是我们自加
  的 key，eagle 双罚上游 0.30.0 已原生修复（无标注分组时全部按 non-draft）。
- 容量铁律沿用旧栈：物理钉住 ≈1.56×配置；容量须 > GPU 池（≈122 万 tok、
  store 侧 ≈40.4 KB/token）才有回载收益；本机历史结论「GPU 池自扛 ~90%、
  21h 零外部回载」——启用前先想清楚负载形态。

## 验证状态（1005）

- 自检 24 项全绿：`cd /home/ll/deploy/vllm-0300 && PYTHONPATH=$PWD/patches:$PWD/patches-extra \
  /home/ll/vllm-env/bin/python selftest_kvoff_rt.py`
- inner 四态 dry-run 正确（缺省/=0 不带参数且体检无告警；=1 注入 JSON）。
- **未做**：FN_KVOFF=1 实机验证窗口（需停机重启 ~8 分钟，由使用者择时执行）。

## 脱敏说明

机器版脚本内置本机 sudo 口令；**本仓库副本一律改为 `${SUDO_PASS:?}` 注入**，
直接执行前先 `export SUDO_PASS=<口令>`。内网 IP 保留（自用部署仓库）。
