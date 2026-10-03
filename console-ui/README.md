# console-ui — 8889 管理台「vLLM」标签页（独立监控页）

2026-10-03 新增。形态**复刻**同日上线的「SGLang」标签页（`/home/ll/deploy/sglang.html` +
`index.html` 内嵌 iframe），把监控口径整体换成 vLLM 0.30.0 实测指标族。

同时修掉两处**运行时归因**缺陷（vLLM 侧缺失，SGLang 侧早已补齐）——它们是这个页面能否
筛到实例的前提。

---

## 1. 文件

| 文件 | 作用 |
|---|---|
| `vllm.html` | 独立监控页（部署到 `/home/ll/deploy/vllm.html`，由 server.js 以 `/vllm.html` 伺服） |
| `patch-vllm-page-1003.py` | 补丁①：server.js 归因兜底 + `/vllm.html` 伺服 + gzip 白名单；index.html 新增标签页 |
| `patch-vllm-runtime2-1003.py` | 补丁②：`/v1/internal/stats` 的 vLLM 实例补 `runtime` 字段 |
| `test-vllm-page.js` | 渲染实测（DOM stub + 真实端点数据跑页面脚本，17 项断言） |

## 2. 部署（生产机，约 1 分钟）

```bash
# 1) 页面
cp vllm.html /home/ll/deploy/vllm.html && chmod 644 /home/ll/deploy/vllm.html

# 2) 打补丁（幂等，自动备份 *.bak-vllm-page-1003-*）
python3 patch-vllm-page-1003.py --check     # 先看锚点状态
python3 patch-vllm-page-1003.py --apply
python3 patch-vllm-runtime2-1003.py --apply
node --check /home/ll/deploy/server.js      # 语法门禁

# 3) 生效（server.js 改动必须重启；KillMode=process，不会误杀 vLLM 实例）
systemctl --user restart dsh-console
```

`index.html` 为每请求实时读盘 → 标签页改动**无需重启**即时生效；浏览器需强刷（`PAGE_VERSION`
已从 `r7` 递增到 `r8`，右上角版本戳可确认没跑旧缓存）。

## 3. 验证

```bash
# 页面与端点
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8889/vllm.html     # 期望 200
curl -s 'http://127.0.0.1:8889/v1/internal/metrics?port=18420' | head -3      # 期望 200 + vllm: 族

# 归因（补丁①②的验收判据，两条都必须有值）
curl -s http://127.0.0.1:8889/v1/internal/model-manager \
  | python3 -c 'import sys,json;[print(i) for i in json.load(sys.stdin)["instances"]]'
#   期望：runtime='vllm'  gpu='0'  gpus=[0,1]  pid=<非空>
curl -s http://127.0.0.1:8889/v1/internal/stats \
  | python3 -c 'import sys,json;print([(i["port"],i.get("runtime")) for i in json.load(sys.stdin)["instances"]])'
#   期望：[('18420', 'vllm')]

# 前端渲染实测（17 项断言，含真实 KPI 出数）
node test-vllm-page.js      # 期望：通过 17/17
```

## 4. 回滚

```bash
cp /home/ll/deploy/server.js.bak-vllm-page-1003-<ts>  /home/ll/deploy/server.js
cp /home/ll/deploy/index.html.bak-vllm-page-1003-<ts> /home/ll/deploy/index.html
# 若已打过补丁②，再叠 server.js.bak-vllm-runtime2-1003-<ts>
rm -f /home/ll/deploy/vllm.html
systemctl --user restart dsh-console
```

补丁脚本幂等：重复 `--apply` 会跳过已应用锚点；锚点被外部回滚时会明确报
`FAIL: 必需锚点未找到` 并以非零码退出，不会写坏文件。

## 5. 两处修复的根因

**① 实例筛不到（model-manager `runtime=unknown` / `gpu=null` / `pid=null`）**

本栈 vLLM 由 `sudo`（root）启动 → `ll` 用户的 `lsof -ti:<port> -sTCP:LISTEN` 看不到监听端口；
`findVllmPidByPort` 的兜底 `pgrep -f "[v]llm.entrypoints"` 又匹配不到 0.30.0 实跑的
`vllm serve` CLI 形式（`/home/ll/vllm-env/bin/vllm serve ...`，不含 `vllm.entrypoints`）。
于是 pid=runtime=gpu 三空。10-03 给 SGLang 加过同款兜底（`listSglangInstances`），vLLM 侧漏了。
修法：两处（`findVllmPidByPort`、model-manager 实例发现）补 `listVllmInstances()` 兜底——
它走 `/proc/<pid>/cmdline` 扫描，root 进程同样可见，并回填 `gpu/gpus/pid`。

**② `/v1/internal/stats` 缺 `runtime`**

stats 的 SGLang 主/从实例对象都带 `runtime:'sglang'`（源码注释自陈「缺 runtime → 徽标误显示
vllm」），而 vLLM 主实例与从实例对象都漏了该字段 → 前端 `instances[0].runtime === 'sglang'`
之类的判据恒为 `undefined`（`index.html:2943` 的二级缓存口径分支、`:4023` 弹窗 runtime 默认值）。
修法：按 SGLang 侧对称补 `runtime: 'vllm'`。

## 6. 页面口径要点（vLLM 与 SGLang 的差异）

- **无 `pp_rank` 副本**：vLLM 指标只带 `engine=` / `model_name=`，是单引擎聚合值 → 直接取值。
  （SGLang 侧同族会按 PP rank 各注册一份同值副本，必须取 max 而非求和——两页口径不可照搬。）
- **多序列族按功能标签拆分**：`prompt_tokens_by_source_total{source=local_compute|local_cache_hit|external_kv_transfer}`、
  `request_success_total{finished_reason=...}`、`spec_decode_num_accepted_tokens_per_pos_total{position=N}`
  → 按标签过滤，只有同族同义分量才求和。
- **`cache_config_info` 是「标签承载配置」的 gauge**：`block_size` / `kv_cache_size_tokens` /
  `kv_cache_max_concurrency` / `enable_prefix_caching` / `cache_dtype` 全部从**标签字典**读，
  不是从样本值读。
- **页内独有看点**：MTP **逐位接受率**（position 0~3 的衰减曲线，判断草稿质量）、KV 池容量与
  可承载并发、CPU KV 二级缓存（`external_prefix_cache_*`，未启用时显式说明原因）。
- 容错沿用 SGLang 页的四条设计约束（双侧续跑 / 自建超时控制器 / 后台标签暂停 / 分段 safeRun），
  见 `vllm.html` 头部注释。

## 7. 已知边界

- 页面按 `runtime === 'vllm'` 筛实例；归因失败（unknown）的实例不进统计，页面会显式弹黄条提示端口。
- 首帧 Decode/Prefill 吞吐显示「空闲」属正常：两者是 counter 差值，需第二帧（2s 后）才有值。
- `/metrics` 非 200（未启 `--enable-metrics`）时页面显示明确的红色诊断卡，而不是空数据。
