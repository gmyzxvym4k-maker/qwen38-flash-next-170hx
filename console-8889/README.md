# console-8889 — vLLM 管理台（8889 控制台）全量现行快照

2026-10-14 入库。形态 = 部署机 `/home/ll/deploy` 的**现行生产版本**整体快照：
`server.js`（单文件 Node 服务，零 npm 依赖，只用标准库）+ 六个页面 + `static/` 自托管资产
+ 白名单提权脚本 + systemd 用户单元 + 配置样例。仪表盘能力：实例探测/并发请求/GPU 排队卡/
吞吐与缓存命中/每请求二级缓存命中列（依赖引擎 rt-patch #12）/硬件监视（CPU/内存带宽/存储
SMART/PCIe 拓扑/功耗墙）/崩溃记录页/快速启动预设/Strata 远端监控。

版本判据：`index.html` 的 `PAGE_VERSION = 20261014-r43`；`server.js` 含
docker 端口映射、栈路由哨兵（`vllm-0310/DISABLED`）、`close-127`、gpu-ctl 功耗持久化等全部现行端点。

## 1. 文件

| 文件 | 作用 |
|---|---|
| `server.js` | 主程序（默认端口 8889，`SERVER_PORT` 可改；`VLLM_MODEL_PORTS`/`MODEL_ALIASES`/`SCRIPT_MODELS` 注册表在头部） |
| `index.html` | 主仪表盘（每请求实时读盘伺服 → 改页面**无需重启**；server.js 改动必须重启服务） |
| `vllm.html` / `sglang.html` / `strata-monitor.html` / `bench.html` / `mobile.html` / `cpu.html` | 各标签页（iframe 内嵌；vLLM/SGLang 监控、Strata 远端、压测、移动端、CPU 详情） |
| `static/` | **必须随迁**：`tailwind.js`+`fonts/`+`fonts-local.css`（缺失=布局整体崩，历史实锤）、`chart.umd.min.js`、`crash.html`（崩溃记录页） |
| `crash-watch.mjs` | 崩溃守护进程（独立 systemd 单元，不占控制台事件循环；产出 `static/crash-records.json`） |
| `gpu-ctl` / `cpu-ctl` | 白名单提权脚本（装到 `/usr/local/bin`，sudoers 精确 NOPASSWD；功耗墙/锁频/睿频/上下线核） |
| `quickstart-presets.json` | 快启预设（含 18420 生产定版档「二级缓存160G·t0.6」） |
| `gpu-hotshield.json` / `power-config.json` / `energy-config.json` / `billing-config.json` / `flashnext-scheme-*.json` | 功能配置样例 |
| `systemd/dsh-console.service` | 控制台用户服务（**`KillMode=process` 必须保留**，否则重启服务会连坐杀掉它启动的推理实例） |
| `systemd/fnx-crash-watch.service` | 崩溃守护用户服务 |

## 2. 部署（裸机，约 2 分钟）

```bash
# ① node ≥ 20（本机实测 v22.23.2），纯标准库无需 npm install
mkdir -p /home/ll/deploy && cp server.js index.html *.html quickstart-presets.json *.json /home/ll/deploy/
cp -r static /home/ll/deploy/ && cp crash-watch.mjs /home/ll/deploy/
sudo install -m 755 gpu-ctl cpu-ctl /usr/local/bin/

# ② sudoers（按需，缺项只是对应卡片降级，不影响启动）
#   ll ALL=(root) NOPASSWD: /usr/sbin/smartctl, /usr/bin/lspci, /usr/sbin/dmidecode, /usr/bin/perf
#   ll ALL=(root) NOPASSWD: /usr/local/bin/gpu-ctl, /usr/local/bin/cpu-ctl
#   （含冒号参数必须转义为 \: ，改前先 visudo -cf）

# ③ 服务（用户级 + linger）
mkdir -p ~/.config/systemd/user && cp systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now dsh-console fnx-crash-watch
loginctl enable-linger ll

# ④ 验证
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8889/            # 200
curl -s http://127.0.0.1:8889/static/tailwind.js -o /dev/null -w '%{http_code}\n'  # 必须 200，404=布局会崩
```

仪表盘靠**同机 `/proc` 进程探测**发现 vLLM/SGLang 实例（含 docker 端口映射换算）；
「最近完成请求」数据源 = `dsh-logger-pkg` 插件的 `request-traces.jsonl`（可选，装后需重启引擎）；
「二级缓存命中」列 = 引擎 rt-patch #12 的 `vllm-ext-cache.jsonl`（可选，见 `stack-0310/`）。

## 3. 脱敏说明（与生产机逐字节差异仅此三类，其余一致）

1. Strata 远端监控的缺省内网 IP（原值=部署机网段内某主机及其广播址）→ RFC5737 文档段
   `192.0.2.38` / `192.0.2.255`，涉及 `server.js`（缺省值+注释）、`index.html`（注释）、
   `strata-monitor.html`（文案）。实际目标机由 `/home/ll/deploy/strata-remote.json` 热加载覆盖：
   `{"baseUrl":"http://<STRATA_HOST>:8080","mac":"<MAC>","sshHost":"user@<STRATA_HOST>","bat":"<启动bat>","broadcast":"<网段广播址>"}`
   （该文件含 MAC/内网地址，**不入库**）。
2. `console-auth.json`（控制台口令门禁，文件不存在=免口令）**不入库**。
3. 运行时状态/数据文件不入库：`*-state.json`、`metrics-snapshot.json`、`crash-records.json`、
   `request-traces.jsonl`、`vllm-live-*.jsonl`、`vllm-ext-cache.jsonl`、日志。

## 4. 与仓库其它目录的关系

- `console-ui/`：10-03「vLLM 标签页」的**补丁史**（vllm.html 首版 + 幂等补丁脚本），保留作沿革；
  本目录是此后持续演进到 r43 的**全量现行版**，部署取本目录即可（vllm.html 已含两轮补丁后的形态）。
- `stack-0310/`：控制台 `SCRIPT_MODELS` 启动的 18420 引擎侧（运行时补丁 + 启停件 + 看门狗）。
