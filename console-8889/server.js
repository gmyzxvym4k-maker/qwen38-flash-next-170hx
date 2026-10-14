#!/usr/bin/env node
/**
 * vLLM Management Console - Backend Service
 * Runs on port 8889, proxies API calls to vLLM (port 8000)
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');
const { spawn } = require('child_process');
const zlib = require('zlib');

// 09-20 性能优化：异步 sleep（替代 execSync('sleep N')——后者会冻结整个事件循环，
// 停止/清理模型期间所有轮询与页面请求排队 1~30 秒）。
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// execFile 的 Promise 版：长耗时脚本（停止脚本可达分钟级）不再用 execSync 阻塞主线程。
const execFileAsync = (file, args, opts) => new Promise((resolve, reject) => {
// [__close127_btn_1006__] ===== 关闭 127（18420 Flash-Next）按钮指令 =====
// 独立端点：POST /v1/internal/close-127 → 走 SCRIPT_MODELS 注册的宿主停止脚本
// （SIGTERM 优先、等退净才兜底，绝不无差别 SIGKILL 持 CUDA 上下文的进程）。
// __vllmver_1006__
// 与 model-manager 的脚本化停止分支同链路，供页面顶部按钮与外部 curl 指令共用。
async function close127Instance() {
  const port = 18420;
  let smStop = null;
  try { smStop = scriptModelForPort(port); } catch (e) {}
  if (!smStop) {
    try { smStop = SCRIPT_MODELS['qwen3.8-flash-next-w4a16'] ? Object.assign({ key: 'qwen3.8-flash-next-w4a16' }, SCRIPT_MODELS['qwen3.8-flash-next-w4a16']) : null; } catch (e) {}
  }
  if (!smStop || !smStop.stopScript) return { success: false, error: '未找到 18420 的脚本化模型注册或停止脚本' };
  let stopPath = null;
  try { stopPath = resolveStopScript(smStop); } catch (e) { stopPath = smStop.stopScript; }
  if (!stopPath || !fs.existsSync(stopPath)) return { success: false, error: '停止脚本不存在：' + (stopPath || '(未注册)') };
  let out = '';
  try {
    out = await execFileAsync('bash', [stopPath, String(port)], { encoding: 'utf8', timeout: 240000, maxBuffer: 4 * 1024 * 1024 });
  } catch (e) { out = String((e && e.stdout || '') + (e && e.stderr || '') || (e && e.message) || e); }
  try { Object.keys(VLLM_MODEL_PORTS).forEach(k => { if (VLLM_MODEL_PORTS[k] === port) delete VLLM_MODEL_PORTS[k]; }); } catch (e) {}
  try { global.__GPU_INSTANCES.delete(port); } catch (e) {}
  let stillAlive = null;
  try { stillAlive = scriptModelInstance(smStop); } catch (e) {}
  console.log(`[close-127] stop ${smStop.key} port ${port}${stillAlive ? ' (WARN: 仍有残留 pid=' + stillAlive.pid + ')' : ' (已彻底停止)'} via ${stopPath}`);
  return { success: !stillAlive, port, stoppedModel: smStop.key, script: stopPath,
    error: stillAlive ? '已执行停止脚本但仍有残留进程（pid=' + stillAlive.pid + '），详见输出' : undefined,
    output: String(out).trim().slice(-600) };
}
// ===== [__close127_btn_1006__] end =====

  require('child_process').execFile(file, args, opts, (err, stdout, stderr) => {
    if (err) { err.stdout = stdout; err.stderr = stderr; reject(err); } else resolve(stdout);
  });
});



// ====== 后端端口动态解析 ======
// VLLM_PORT 环境变量显式指定时优先（用户手动固定端口）；否则扫描运行中的
// vllm serve / sglang.launch_server 进程的 --port，跟随实际实例 —— 避免 vLLM
// 换了端口而控制台还指旧端口导致仪表盘没数据（08-24 两次实测踩坑）。
// 找不到进程时回落默认端口 8000。
// ====== Docker 端口映射（09-19）======
// vLLM 跑在 docker 容器里时（qwen-flash-sm80/serve.py 栈），宿主可见进程的 cmdline
// --port 是容器内端口（8000），宿主真实监听端口由 docker-proxy 经 -p 映射（18420）。
// 控制台按容器内端口注册路由/拉 metrics → 宿主 8000 无监听 → 仪表盘所有数据卡空转。
// 判据：/proc/<pid>/cgroup 含 docker 容器 id → docker port 查宿主端口。
// 非容器进程原样返回；结果缓存 10s，失败也缓存负结果，防 dockerd 挂起反复阻塞事件循环。
const __dockerHostPortCache = new Map();
function dockerCidForPid(pid) {
  try {
    const cg = fs.readFileSync(`/proc/${pid}/cgroup`, 'utf8');
    const m = cg.match(/\/docker\/([0-9a-f]{12,64})/) || cg.match(/docker-([0-9a-f]{12,64})\.scope/);
    return m ? m[1].slice(0, 12) : null;
  } catch (e) { return null; }
}
function mappedHostPort(pid, port) {
  const cid = dockerCidForPid(pid);
  if (!cid) return port;
  const key = cid + ':' + port;
  const now = Date.now();
  const hit = __dockerHostPortCache.get(key);
  if (hit && now - hit.ts < 10000) return hit.hp || port;
  let hp = null;
  try {
    const { execSync } = require('child_process');
    const out = execSync(`docker port ${cid} ${port}/tcp 2>/dev/null`, { encoding: 'utf8', timeout: 1500 }).toString();
    const m2 = out.match(/:(\d+)/);
    if (m2) { const n = parseInt(m2[1]); if (n > 0 && n < 65536) hp = n; }
  } catch (e) {}
  __dockerHostPortCache.set(key, { hp, ts: now });
  if (hp && hp !== port) console.log(`[docker-map] ${cid}:${port} -> host ${hp}`);
  return hp || port;
}

function resolveBackendPort() {
  const envPort = parseInt(process.env.VLLM_PORT);
  if (envPort) return envPort;
  // 直接扫 /proc（不走 pgrep/tr 子进程）：高负载下 execSync 偶发 2s 超时会让探测
  // 间歇性漏掉 vllm 实例、回落到 sglang 端口，造成默认端口 8000↔8001 横跳
  // （08-26 首次踩坑加了探测进程过滤，08-30 仍因超时横跳，且横跳连带把
  // VLLM_MODEL_PORTS 显式映射覆盖掉 → Qwen3.6-27B 404 事故）。
  // 保留探测类进程过滤（ssh/bash -c 里带 "vllm serve" 字样会误匹配）。
  const found = []; // {rank: 0=vllm 1=sglang, port}
  let entries = [];
  // 09-22：探不到任何实例时返回 null（调用方沿用当前端口），不再硬回落 8000。
  // 此前 vLLM 重启/加载窗口进程短暂消失 → resolve 返回 8000 → maybeReDetect 把
  // 主端口横跳到 8000（无监听）→ 仪表盘主 stats 拿不到数据整页停刷（日志里大量
  // 「端口变化 18420 -> 8000」即此）。
  try { entries = fs.readdirSync('/proc'); } catch (e) { return null; }
  for (const name of entries) {
    if (!/^\d+$/.test(name)) continue;
    let cmd;
    try { cmd = fs.readFileSync(`/proc/${name}/cmdline`, 'utf8').split('\0').join(' '); } catch (e) { continue; }
    if (!cmd) continue;
    if (/(^|\s)(ssh|bash|sh|expect)(\s|$)/.test(cmd)) continue;
    if (/\b(pgrep|grep|ps)\b/.test(cmd)) continue;
    const m = cmd.match(/--port\s+(\d+)/);
    if (!m) continue;
    if (isVllmProcCmd(cmd)) found.push({ rank: 0, port: mappedHostPort(parseInt(name), parseInt(m[1])) });
    else if (cmd.includes('sglang.launch_server')) found.push({ rank: 1, port: mappedHostPort(parseInt(name), parseInt(m[1])) });
  }
  if (!found.length) return null; // 09-22：无实例→null，不横跳到 8000
  found.sort((a, b) => a.rank - b.rank || a.port - b.port);
  return found[0].port;
}

// ====== 多实例探测（多卡多实例：每张卡独立跑一个 vllm serve）======
// 扫描所有运行中的 vllm serve 主进程，返回 [{port, pid, gpu, modelPath, servedName}]。
// gpu 取进程环境 CUDA_VISIBLE_DEVICES 的第一个值（vLLM 按该顺序映射物理卡）；
// 未设置时回落 0。3 秒缓存（ticker 每秒调用一次，避免频繁 pgrep/读 proc）。
// 判断进程 cmdline 是否为 vLLM 主进程（两种启动形式都要认）：
// ① vllm serve <model> ...（CLI 形式）
// ② python -m vllm.entrypoints.openai.api_server --model <model> ...（模块形式，
//    09-05 发现 GPU0 的 syv 实例用此形式，此前被 pgrep '[v]llm serve' 漏掉 →
//    8889 仪表盘 GPU0 无信息的根因）
function isVllmProcCmd(cmd) {
  return cmd.includes('vllm serve') ||
    (cmd.includes('vllm.entrypoints.openai.api_server') &&
     cmd.includes('vllm.entrypoints')) ||
    // vLLM 0.28 CLI 形式：python -m vllm.entrypoints.cli.main serve <model> ...
    // （chroot 内 Flash-Next 实例即此形式；09-14 实测旧规则探测不到 → 路由/仪表盘丢失）
    cmd.includes('vllm.entrypoints.cli.main');
}

let __vllmInstancesCache = { at: 0, list: [] };
function listVllmInstances(force) {
  const now = Date.now();
  if (!force && now - __vllmInstancesCache.at < 3000) return __vllmInstancesCache.list;
  const list = [];
  try {
    const { execSync } = require('child_process');
    // 用 [v]llm 括号技巧避免 pgrep 匹配到「自己 spawn 的 sh -c 包装进程」
    // （其 cmdline 含 'vllm serve' 字样，会自匹配后随即退出，导致读 /proc 报
    // "cannot open /proc/PID/cmdline" 噪音）。
    let pids = '';
    try {
      pids = execSync(`pgrep -f "[v]llm serve|vllm.entrypoints.openai.api_server|vllm.entrypoints.cli.main"`, { timeout: 2000 }).toString().trim();
    } catch (e) { pids = ''; }
    for (const pid of pids.split('\n').filter(Boolean)) {
      try {
        // 直接读 /proc（不经过 sh 子进程，避免 stderr 噪音）
        const cmd = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8').split('\0').join(' ');
        // 排除误匹配的探测进程（与 resolveBackendPort 同规则：pgrep -f 是子串匹配，
        // ssh/bash -c 里含 'vllm serve' 字样的探测命令也会命中）
        if (/(^|\s)(ssh|bash|sh|expect)(\s|$)/.test(cmd)) continue;
        if (/\b(pgrep|grep|ps)\b/.test(cmd)) continue;
        if (!isVllmProcCmd(cmd)) continue;
        const pm = cmd.match(/--port\s+(\d+)/);
        if (!pm) continue;
        // 容器内进程 → 宿主映射端口（09-19：docker 栈容器 8000 → 宿主 18420）
        const port = mappedHostPort(parseInt(pid), parseInt(pm[1]));
        // 模型路径：vllm serve <model> 或 -m ... --model <model> 两种形式
        const modelPath = (cmd.match(/vllm serve\s+(\S+)/) ||
          cmd.match(/-m vllm\.entrypoints\.cli\.main serve\s+(\S+)/) ||
          cmd.match(/--model[=\s](\S+)/) || [])[1] || '';
        const servedName = (cmd.match(/--served-model-name\s+(\S+)/) || [])[1] || '';
        // ---- GPU 归属（09-14 增强：单实例可跨多卡=TP/PP/DP）----
        // 旧版只取 CUDA_VISIBLE_DEVICES 的第一个值 → PP2/TP2 这种「一个实例占两张卡」
        // 的部署被错标成 GPU0，仪表盘 GPU1 整组没数据。现在给出完整 gpus 列表：
        //   · CUDA_VISIBLE_DEVICES 显式列出 → 用它（截到 worldSize）
        //   · 未设置 → vLLM 按 0..worldSize-1 连续占卡，worldSize = TP × PP × DP
        // gpus[0] 仍作为主卡（沿用旧字段 gpu 的语义，兼容既有前端/接口）。
        let gpus = [];
        try {
          const env = fs.readFileSync(`/proc/${pid}/environ`, 'utf8').split('\0');
          const cvdLine = env.find(l => l.startsWith('CUDA_VISIBLE_DEVICES=')) || '';
          const cvd = (cvdLine.split('=').slice(1).join('=')) || '';
          if (cvd.trim() !== '') {
            gpus = cvd.split(',').map(s => parseInt(s.trim())).filter(g => !isNaN(g));
          }
        } catch (e2) {}
        const sizeOf = (re) => {
          const mm = cmd.match(re);
          const n = mm ? parseInt(mm[1]) : 1;
          return (isNaN(n) || n < 1) ? 1 : n;
        };
        const worldSize = sizeOf(/--tensor-parallel-size[=\s]+(\d+)/)
          * sizeOf(/--pipeline-parallel-size[=\s]+(\d+)/)
          * sizeOf(/--data-parallel-size[=\s]+(\d+)/);
        if (gpus.length === 0) {
          for (let i = 0; i < worldSize; i++) gpus.push(i);
        } else if (gpus.length > worldSize) {
          gpus = gpus.slice(0, worldSize);
        }
        const gpu = gpus.length ? gpus[0] : 0;
        list.push({ port, pid: parseInt(pid), gpu, gpus, worldSize, modelPath, servedName });
      } catch (e2) { /* 进程已退出 */ }
    }
  } catch (e) {}
  // 按端口去重（同一端口只保留第一个主进程）
  const seen = new Set();
  const dedup = [];
  for (const it of list) {
    if (seen.has(it.port)) continue;
    seen.add(it.port);
    dedup.push(it);
  }
  __vllmInstancesCache = { at: now, list: dedup };
  return dedup;
}

// ---- vLLM 版本探测（[vllm-version-1006] 供「vLLM」标签页显示当前模型的引擎版本）----
// 判据链（与 ple_table 同款"日志是真值"）：
//   ① 引擎启动日志自报行 "Initializing a V1 LLM engine (<ver>)"——脚本化/chroot 实例
//      （引擎属 root：/proc/<pid>/exe、/root、/environ 对 ll 全权限不够；宿主 glob 到的
//      dist-info 又可能是另一个环境的）唯有此项可靠。126 实锤：18420=root chroot，
//      宿主 /home/ll/vllm-env 是 0.30.0，引擎实际 0.1.dev20073+g8e685d198。
//   ② 宿主 dist-info 目录名推导（venv 直起实例）：先按 cmdline 解释器路径锁定所属 venv，
//      再扫常见站点目录。
// 端口级 5 分钟缓存（负结果也缓存，避免前端 2s 轮询反复读大日志）。
let __vllmVerCache = new Map(); // port → { at, ver }
function vllmVerFromLog(logPath) {
  // 尾部 8MB 反向分块找最近一条引擎自报行（实例重启换版本时新行必然更靠近尾部）。
  try {
    const st = fs.statSync(logPath);
    const CHUNK = 1024 * 1024, MIN_POS = Math.max(0, st.size - 8 * 1024 * 1024);
    const fd = fs.openSync(logPath, 'r');
    try {
      let pos = st.size, carry = '';
      while (pos > MIN_POS) {
        const start2 = Math.max(MIN_POS, pos - CHUNK);
        const len = pos - start2;
        const buf = Buffer.alloc(len);
        fs.readSync(fd, buf, 0, len, start2);
        const parts = (buf.toString('utf8') + carry).split('\n');
        carry = parts.shift() || '';
        for (let i = parts.length - 1; i >= 0; i--) {
          const m = parts[i].match(/Initializing a V1 LLM engine \(([^)]+)\)/);
          if (m) return m[1].replace(/^v/, ''); // 归一化：日志自报行带 v 前缀，去掉（前端统一补 'v'）
        }
        pos = start2;
      }
    } finally { try { fs.closeSync(fd); } catch (e) {} }
  } catch (e) {}
  return null;
}
function vllmDistVersion(pkgRoot) {
  try {
    const infos = fs.readdirSync(pkgRoot).filter(n => /^vllm-[0-9][^-]*\.dist-info$/.test(n));
    if (infos.length) {
      const m = infos[infos.length - 1].match(/^vllm-([0-9][^-]*)\.dist-info$/);
      if (m) return m[1];
    }
  } catch (e) {}
  return null;
}
function detectVllmVersion(pid, port, cmdline) {
  if (port == null) return null;
  const hit = __vllmVerCache.get(port);
  if (hit && Date.now() - hit.at < 300000) return hit.ver;
  let ver = null;
  // ① 脚本化模型：日志引擎自报行（条目 log + altLogs；pickScriptModelLogFile 内容打分为首选）
  try {
    const sms = scriptModelsForPort(port, cmdline || []);
    for (const sm of sms) {
      const cands = [];
      try { const pf = pickScriptModelLogFile({ pid, port }, sm); if (pf) cands.push(pf); } catch (e) {}
      if (sm.log) cands.push(sm.log);
      for (const al of sm.altLogs || []) cands.push(al);
      for (const lf of cands) {
        ver = vllmVerFromLog(lf);
        if (ver) break;
      }
      if (ver) break;
    }
  } catch (e) {}
  // ② 宿主 dist-info：按 cmdline 解释器的 realpath 推出其 site/dist-packages 根
  if (!ver && cmdline && cmdline.length) {
    try {
      const py = String(cmdline[0] || '');
      if (/python/.test(py)) {
        let rp = py;
        try { rp = fs.realpathSync(py); } catch (e) {}
        const mm = rp.match(/^(.*)[\/\\]lib[\/\\]python[0-9.]+[\/\\](site|dist)-packages/);
        if (mm) ver = vllmDistVersion(mm[0].replace(/[\/\\](site|dist)-packages$/, ''));
        if (!ver) {
          const m2 = rp.match(/^(.*)[\/\\]bin[\/\\]python[^\/\\]*$/);
          if (m2) {
            for (const sp of ['lib', 'lib64']) {
              const dp = path.join(m2[1], sp);
              try {
                for (const d of fs.readdirSync(dp)) {
                  if (!/^python3\./.test(d)) continue;
                  ver = vllmDistVersion(path.join(dp, d, 'site-packages'))
                     || vllmDistVersion(path.join(dp, d, 'dist-packages'));
                  if (ver) break;
                }
              } catch (e) {}
              if (ver) break;
            }
          }
        }
      }
    } catch (e) {}
  }
  if (!ver) {
    for (const hp of ['/home/ll/vllm-env/lib/python3.11/site-packages',
                      '/usr/local/lib/python3.12/dist-packages',
                      '/usr/lib/python3.12/dist-packages']) {
      ver = vllmDistVersion(hp);
      if (ver) break;
    }
  }
  __vllmVerCache.set(port, { at: Date.now(), ver: ver || null });
  return ver || null;
}

// ---- trace 记录时间戳合理性校验（09-18）----
// 现场实锤：127 机系统时钟会异常跳到 2161-01-01（request-traces.jsonl 出现 t≈6.03e9
// 的记录、实例日志同现 "INFO 01-01"）。这类"未来时间"记录在 recent-requests 的
// t 倒序里永远顶在最前，把真实数据整段挤出 limit 窗口；写它们的实例进程又已死
// （pid 归因失败 → gpu=null），前端就堆出一整组「GPU ?」历史数据。
// 判据：非有限数 / 早于 2001-01-01（1e9，顺带抓到误用单调钟）/ 晚于当前+5min → 一律视为脏。
function traceTsPlausible(t) {
  if (typeof t !== 'number' || !Number.isFinite(t)) return false;
  return t >= 1e9 && t <= Date.now() / 1000 + 300;
}

// ---- pid → 实例解析（09-14：支持 EngineCore / Worker 子进程）----
// vLLM 的 stat logger 插件在 **EngineCore 子进程**里调用 os.getpid()，写进
// request-traces.jsonl 的 pid 是 EngineCore 的 pid；而 listVllmInstances() 扫到的是
// API server 主进程 pid（cgroup 里最外层 vllm serve）。两者不相等 → 旧版
// instByPid.get(rec.pid) 恒 miss → 所有 vLLM 完成记录 gpu=null（前端显示「—」），
// PP/TP 双卡实例更是整卡无数据。这里沿 /proc/<pid>/stat 的 ppid 链上溯 8 层找主进程。
// 解析成功的 pid 落 __pidInstSeen 长期缓存（实例重启换 pid 后，旧 trace 行的归因不丢，
// 不再因为 pid 消失而整列变「—」）。
let __pidInstSeen = new Map(); // pid → {port, gpu, gpus, at}
function __ppidOf(pid) {
  try {
    const st = fs.readFileSync(`/proc/${pid}/stat`, 'utf8');
    // 第 2 字段是 (comm)，可能含空格/括号 → 从最后一个 ')' 之后再切：state ppid ...
    const rest = st.slice(st.lastIndexOf(')') + 2).split(' ');
    return parseInt(rest[1]) || 0;
  } catch (e) { return 0; }
}
function resolveInstanceByPid(pid, insts) {
  let cur = parseInt(pid);
  if (!cur) return null;
  const chain = [];
  for (let hop = 0; hop < 8 && cur > 1; hop++) {
    chain.push(cur);
    const hit = insts.find(i => i.pid === cur);
    if (hit) {
      const snap = { port: hit.port, gpu: hit.gpu, gpus: hit.gpus, at: Date.now() };
      for (const p of chain) __pidInstSeen.set(p, snap);
      if (__pidInstSeen.size > 800) { // 防无限增长：只留最近 400 条
        const keep = [...__pidInstSeen.entries()].sort((a, b) => (b[1].at || 0) - (a[1].at || 0)).slice(0, 400);
        __pidInstSeen = new Map(keep);
      }
      return snap;
    }
    const cached = __pidInstSeen.get(cur);
    if (cached) return cached;
    const up = __ppidOf(cur);
    if (!up || up === cur) break;
    cur = up;
  }
  return __pidInstSeen.get(parseInt(pid)) || null;
}

// rid → pid 映射（从 vllm-live-prefill.jsonl 构建，2s 缓存）。
// vLLM 的 request-traces.jsonl 完成记录不带 pid，但 live-prefill 每条都带
// rid+pid，且 trace.request_id == live-prefill.rid。据此把「最近完成请求」
// 的每条记录关联到所属实例（pid）→ GPU。文件由插件封顶 2MB/2000 行，读取廉价。
let __ridPidCache = { at: 0, map: new Map() };
function buildRidPidMap() {
  const now = Date.now();
  if (now - __ridPidCache.at < 2000) return __ridPidCache.map;
  const map = new Map();
  try {
    const data = fs.readFileSync(path.join(__dirname, 'vllm-live-prefill.jsonl'), 'utf8');
    for (const l of data.split('\n')) {
      if (!l.trim()) continue;
      try {
        const r = JSON.parse(l);
        if (!r.rid || r.pid == null) continue;
        map.set(r.rid, r.pid);
        // vLLM 内部 rid 带 -<8hex> 后缀（如 chatcmpl-xxx-8b83f9ae），而 trace 的
        // request_id 是去后缀的（chatcmpl-xxx）。按前缀（去后缀）再建一条索引。
        const base = r.rid.replace(/-[0-9a-f]{8}$/, '');
        if (base !== r.rid) map.set(base, r.pid);
      } catch (e) { /* 半截行/坏行跳过 */ }
    }
  } catch (e) { /* 文件不存在 */ }
  __ridPidCache = { at: now, map };
  return map;
}

// [extcache-1007] 每请求二级缓存命中（rt-patch #12 → vllm-ext-cache.jsonl）。
// 插件行 {rid,t,ext,loc,cached,pt,consistent,port}：ext=CPU 二级缓存回载命中 token，
// loc=本级 GPU 前缀命中，cached=二者总和（= request-traces 的 cached_tokens 口径）。
// 按 rid 建索引（含 vLLM 内部 -<8hex> 后缀去尾索引，口径同 buildRidPidMap），
// 2s 缓存 + 只读尾部 512KB（文件自身 >4MiB 轮转，读量恒定廉价）。
// 缺文件/引擎未带补丁 → 空 Map，前端新列如实显示 --（绝不回落估算值）。
let __extCacheIndex = { at: 0, map: new Map() };
function extCacheLookup(rid) {
  if (!rid) return null;
  const now = Date.now();
  if (now - __extCacheIndex.at > 2000) {
    const map = new Map();
    try {
      const fp = path.join(__dirname, 'vllm-ext-cache.jsonl');
      const sz = fs.statSync(fp).size;
      const LEN = Math.min(512 * 1024, sz);
      const fd = fs.openSync(fp, 'r');
      let text = '';
      try {
        const buf = Buffer.alloc(LEN);
        fs.readSync(fd, buf, 0, LEN, sz - LEN);
        text = buf.toString('utf8');
      } finally { fs.closeSync(fd); }
      for (const l of text.split('\n')) {
        if (!l.trim()) continue;
        let r; try { r = JSON.parse(l); } catch (e) { continue; }
        if (!r || !r.rid) continue;
        map.set(String(r.rid), r);
        const base = String(r.rid).replace(/-[0-9a-f]{8}$/, '');
        if (base !== r.rid && !map.has(base)) map.set(base, r);
      }
    } catch (e) { /* 文件不存在/读失败 → 空索引 */ }
    __extCacheIndex = { at: now, map };
  }
  return __extCacheIndex.map.get(String(rid)) || null;
}
// 数据可用性探针（供端点回传，前端做透明化说明）：
//   none = 文件不存在（引擎未带 rt-patch #12，或从未有请求完成）
//   stale = 存在但长时间没写（引擎起了却没出数，值得提示）
//   ok    = 近期有写入
function extCacheState() {
  try {
    const fp = path.join(__dirname, 'vllm-ext-cache.jsonl');
    const st = fs.statSync(fp);
    const ageMin = (Date.now() - st.mtimeMs) / 60000;
    return { state: ageMin > 30 ? 'stale' : 'ok', ageMin: Math.round(ageMin * 10) / 10, size: st.size };
  } catch (e) { return { state: 'none', ageMin: null, size: 0 }; }
}
function extCacheAvailable() { const x = extCacheState(); return x.state === 'ok' || x.state === 'stale'; }
function extCacheSource() { return extCacheState().state; }

// 给一批 recent-requests 记录挂上 ext_cached_tokens（无数据不写字段，前端显示 --）。
function extAttachToRecs(recs) {
  for (const r of recs) {
    if (!r || !r.request_id) continue;
    const hit = extCacheLookup(r.request_id);
    if (hit && hit.ext != null) r.ext_cached_tokens = hit.ext;
  }
}

// 运行中的 SGLang 实例列表（port + pid + gpu），3s 缓存。
// sglang 的 request-metrics 记录不带 pid/端口，无法逐请求精确归属；
// 「最近完成请求」表格用「恰好 1 个实例在跑」的启发式归属 GPU（0 个或多个则显示 —）。
let __sglangInstCache = { at: 0, list: [] };
// sglang 逐请求 metrics 目录：默认 sglang-request-metrics/ + 自定义 --export-metrics-dir
// 指定的 sglang-metrics-<port>/（36 主机启动 8000 实例时用了该目录，其记录此前完全
// 进不了「最近完成请求」表 = 用户反馈的 GPU0 无数据根因）。port 从目录名推导，
// 作为 rec 无 port 字段时的回退（覆盖补丁前旧 exporter 写出的记录）。
function sglangMetricsDirs() {
  const out = [{ dir: path.join(__dirname, "sglang-request-metrics"), port: null }];
  try {
    for (const e of fs.readdirSync(__dirname, { withFileTypes: true })) {
      if (!e.isDirectory()) continue;
      const m = e.name.match(/^sglang-metrics-(\d+)$/);
      if (m) out.push({ dir: path.join(__dirname, e.name), port: parseInt(m[1], 10) });
    }
  } catch (err) { /* ignore */ }
  return out;
}

function listSglangInstances() {
  const now = Date.now();
  if (now - __sglangInstCache.at < 3000) return __sglangInstCache.list;
  const list = [];
  try {
    const { execSync } = require('child_process');
    // [s]glang 括号技巧避免 pgrep 自匹配（同 listVllmInstances 的说明）
    let pids = '';
    try {
      pids = execSync(`pgrep -f "[s]glang.launch_server"`, { timeout: 2000 }).toString().trim();
    } catch (e) { pids = ''; }
    for (const pid of pids.split('\n').filter(Boolean)) {
      try {
        const cmd = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8').split('\0').join(' ');
        if (/(^|\s)(ssh|bash|sh|expect)(\s|$)/.test(cmd)) continue;
        if (/\b(pgrep|grep|ps)\b/.test(cmd)) continue;
        const pm = cmd.match(/--port\s+(\d+)/);
        if (!pm) continue;
        let gpu = 0;
        try {
          const env = fs.readFileSync(`/proc/${pid}/environ`, 'utf8').split('\0');
          const cvdLine = env.find(l => l.startsWith('CUDA_VISIBLE_DEVICES=')) || '';
          const cvd = (cvdLine.split('=').slice(1).join('=')) || '';
          if (cvd.trim() !== '') {
            const g = parseInt(cvd.split(',')[0].trim());
            if (!isNaN(g)) gpu = g;
          }
        } catch (e2) {}
        // 真实模型文件路径：--model-path（SGLang 的 /v1/models root 字段填的是
        // served-model-name 而非文件路径，仪表盘「模型路径」需要这个真实路径）
        const modelPath = (cmd.match(/--model-path\s+(\S+)/) || [])[1]
          || (cmd.match(/--model\s+(\S+)/) || [])[1] || '';
        const servedName = (cmd.match(/--served-model-name\s+(\S+)/) || [])[1] || '';
        list.push({ port: parseInt(pm[1]), pid: parseInt(pid), gpu, modelPath, servedName });
      } catch (e2) { /* 进程已退出 */ }
    }
  } catch (e) {}
  __sglangInstCache = { at: now, list };
  return list;
}

// 解析 vllm/sglang 服务进程 cmdline → 参数对象（model-params 单端口模式与全实例模式共用）。
// 未显式传参的字段保留下方默认值（前端对 null/缺省显示 '--'）。
function parseServerParams(cmdline, runtime, port) {
  const params = { runtime, port, tensor_parallel_size: 1, gpu_memory_utilization: 0.9, max_num_seqs: 4, block_size: 32, attention_backend: 'FLASHINFER', enable_chunked_prefill: true, enable_prefix_caching: false, speculative_config: null, dflash: false, dflashMethod: null, dspark: false, dsparkMethod: null, thinking: null, thinkingEffort: 'medium', kv_cache_quant: 'auto', temperature: null, top_p: null, top_k: null, min_p: null, repetition_penalty: null };
  for (let i = 0; i < cmdline.length; i++) {
    const a = cmdline[i];
    const next = () => cmdline[i + 1] || '';
    if (a === '--port') params.port = parseInt(next()) || 8000;
    // vllm 参数
    else if (a === '--tensor-parallel-size') params.tensor_parallel_size = parseInt(next()) || 1;
    else if (a === '--pipeline-parallel-size' || a === '--pp' || a === '--pp-size') params.pipeline_parallel_size = parseInt(next()) || 1;
    else if (a === '--max-model-len' || a === '--context-length') params.max_model_len = parseInt(next()) || null;
    else if (a === '--gpu-memory-utilization') params.gpu_memory_utilization = parseFloat(next()) || 0.9;
    else if (a === '--max-num-seqs') params.max_num_seqs = parseInt(next()) || 4;
    else if (a === '--block-size') params.block_size = parseInt(next()) || 32;
    else if (a === '--attention-backend') params.attention_backend = next();
    else if (a === '--enable-chunked-prefill') params.enable_chunked_prefill = true;
    else if (a === '--no-enable-chunked-prefill') params.enable_chunked_prefill = false;
    else if (a === '--enable-prefix-caching') params.enable_prefix_caching = true;
    else if (a === '--speculative-config') {
      try {
        const sc = JSON.parse(next());
        params.speculative_config = sc;
        // DFlash 探测：method 或草稿模型路径含 dflash（初代 DFlash / DFlash2 均命中）
        if (sc && (String(sc.method || '').toLowerCase().includes('dflash') || String(sc.model || '').toLowerCase().includes('dflash'))) {
          params.dflash = true;
          params.dflashMethod = sc.method || null;
        }
        // DSpark 探测：method 或草稿模型路径含 dspark（DFlash 骨干 + Markov 头，vllm-env 0.27.1）
        else if (sc && (String(sc.method || '').toLowerCase().includes('dspark') || String(sc.model || '').toLowerCase().includes('dspark'))) {
          params.dspark = true;
          params.dsparkMethod = sc.method || null;
        }
      } catch (e) { params.speculative_config = null; }
    }
    else if (a === '--default-chat-template-kwargs') {
      try { const ck = JSON.parse(next()); params.thinking = ck.enable_thinking !== false; params.thinkingEffort = ck.reasoning_effort || 'xhigh'; } catch (e) {}
    }
    else if (a === '--override-generation-config') {
      try { const g = JSON.parse(next()); params.temperature = g.temperature; params.top_p = g.top_p; params.top_k = g.top_k; params.min_p = g.min_p; params.repetition_penalty = g.repetition_penalty; } catch (e) {}
    }
    else if (a === '--kv-cache-dtype') {
      const kv = next();
      if (kv === 'fp8') params.kv_cache_quant = 'fp8 (FP8 量化 KV 缓存)';
      else if (kv === 'int8') params.kv_cache_quant = 'int8 (INT8 量化 KV 缓存)';
      else params.kv_cache_quant = kv;
    }
    else if (a === '--quantization') {
      const q = next();
      if (q === 'fp8_kv') params.kv_cache_quant = 'fp8_kv (仅 KV 缓存)';
      else if (q === 'fp8') params.kv_cache_quant = 'fp8 (权重+KV)';
      else if (q === 'awq') params.kv_cache_quant = 'int8 (AWQ)';
      else params.kv_cache_quant = q;
    }
    // sglang 参数适配
    else if (a === '--tp' || a === '--tp-size') params.tensor_parallel_size = parseInt(next()) || 1;
    else if (a === '--mem-fraction-static') params.gpu_memory_utilization = parseFloat(next()) || 0.9;
    else if (a === '--max-running-requests') params.max_num_seqs = parseInt(next()) || 4;
    else if (a === '--page-size') params.block_size = parseInt(next()) || 1;
    else if (a === '--chunked-prefill-size') { const v = parseInt(next()); if (v > 0) params.enable_chunked_prefill = true; }
    else if (a === '--disable-radix-cache') params.enable_prefix_caching = false;
    else if (a === '--radix-cache-backend') { /* 启用前缀缓存（默认有 radix cache）*/ params.enable_prefix_caching = true; }
    else if (a === '--speculative-algorithm') {
      const algo = next();
      const tokens = (() => { for (let j = i + 1; j < cmdline.length; j++) { if (cmdline[j] === '--speculative-num-draft-tokens') return parseInt(cmdline[j + 1]); } return 0; })();
      params.speculative_config = { method: algo, num_speculative_tokens: tokens || 0 };
      // DFlash 探测：算法名或草稿模型路径含 dflash
      if (/dflash/i.test(algo) || cmdline.some((x, j) => j > i && cmdline[j - 1] === '--speculative-draft-model-url' && /dflash/i.test(x))) {
        params.dflash = true;
        params.dflashMethod = algo;
      }
    }
    else if (a === '--kv-cache-dtype') {
      const kv = next();
      if (kv === 'fp8_e4m3' || kv === 'fp8') params.kv_cache_quant = 'fp8 (FP8 量化 KV 缓存)';
      else if (kv === 'fp8_e5m2') params.kv_cache_quant = 'fp8_e5m2';
      else if (kv === 'mxfp8') params.kv_cache_quant = 'mxfp8';
      else if (kv === 'int8' || kv === 'int8_bf16') params.kv_cache_quant = 'int8';
      else params.kv_cache_quant = kv;
    }
    // sglang 采样参数
    else if (a === '--preferred-sampling-params') {
      try { const g = JSON.parse(next()); params.temperature = g.temperature; params.top_p = g.top_p; params.top_k = g.top_k; params.min_p = g.min_p; params.repetition_penalty = g.repetition_penalty; } catch (e) {}
    }
    else if (a === '--sampling-defaults') {
      // 未显式传 preferred-sampling-params 时，用 sampling_defaults 的默认值
      const sd = next();
      if (sd === 'openai') { params.temperature = 1.0; params.top_p = 1.0; params.top_k = 0; params.min_p = 0.0; params.repetition_penalty = 1.0; }
      // 'model' 不设置值（保留 null，让前端显示 '--'，表示使用模型配置）
    }
  }
  // 前缀缓存：sglang 默认启用 radix cache，除非显式 --disable-radix-cache
  if (runtime === 'sglang' && params.enable_prefix_caching === false) {
    if (!cmdline.includes('--disable-radix-cache')) params.enable_prefix_caching = true;
  }
  return params;
}

const config = {
  vllmHost: process.env.VLLM_HOST || '127.0.0.1',
  vllmPort: resolveBackendPort() || 8000, // 09-22：resolve 探不到返回 null，启动缺省仍 8000
  serverPort: parseInt(process.env.SERVER_PORT) || 8889,
};

let vllmBaseUrl = `http://${config.vllmHost}:${config.vllmPort}`;

// ticker 拉不到 metrics 时自动重新探测端口并跟随（限频 5s 一次），
// 让控制台在 vLLM 重启/换端口后自愈，无需手动改配置重启。
function maybeReDetectBackend() {
  const now = Date.now();
  if (global.__lastPortDetect && now - global.__lastPortDetect < 5000) return;
  global.__lastPortDetect = now;
  try {
    const p = resolveBackendPort();
    if (p && p !== config.vllmPort) {
      const old = config.vllmPort;
      console.error(`[backend] 端口变化 ${old} -> ${p}，自动跟随`);
      config.vllmPort = p;
      vllmBaseUrl = `http://${config.vllmHost}:${p}`;
      // 注意：这里【不再】同步覆盖 VLLM_MODEL_PORTS 条目。路由表里都是显式
      // 模型→端口映射（启动弹窗注册 / 环境变量 / SGLang 实例同步），跟随默认
      // 端口会把它们横跳覆盖掉（08-30 事故：Qwen3.6-27B→8000 被改成 8001 → 404）。
      // 未注册的模型走 vllmPortForModel() 动态回落 config.vllmPort，天然跟随。
    }
  } catch (e) {}
}

// [hw-move 1005] 无条件定时兜底：原自愈只挂在 ticker 的 req error 回调（1774）上，
// 只要那条链断裂（busy 滞留 / 首个请求挂起不报错 / 迁移期时序竞态），主端口就会
// 永久卡在启动缺省 8000——10-04 换装机实锤：引擎在跑、/metrics 直连 200，但
// stats 恒 {}、代理打 8000，且日志里出现过跟随却没保住。此处每 6s 主动探一次
// （maybeReDetectBackend 内部自带 5s 节流；resolveBackendPort 是无子进程的纯
// /proc 扫描，毫秒级；探不到返回 null 保持现状，不会再引入横跳）。
setInterval(() => { try { maybeReDetectBackend(); } catch (e) {} }, 6000).unref();

// ====== Multi-model support ======
// Map of served model name -> vLLM backend port. Each backend is its own
// `vllm serve` instance pinned to one GPU. Unknown model names fall back to
// the default backend (config.vllmPort，动态跟随自愈后的端口)。
// 注意：不要把默认模型硬编码成 config.vllmPort 的「拷贝值」——端口自愈后
// 拷贝值会过期导致代理仍打旧端口（08-24 实测踩坑：路由表 8000 vs 实际 8001）。
// 09-03：移除旧硬编码 'qwen3.6-35b-a3b-fp8': 8001——运行中 vLLM 实例的模型名
// 现由 syncVllmModelPortsFromProcs() 从进程 cmdline 动态注册（只增改不删）。
const VLLM_MODEL_PORTS = Object.assign(
  {},
  (() => { try { return JSON.parse(process.env.VLLM_MODEL_PORTS || '{}'); } catch (e) { return {}; } })()
);
// ====== 模型名别名（2026-09-14）======
// 客户端/模型页可能用别名请求模型：路由前先解析成真实 served-model-name 再查路由表。
// Flash-Next 实例 served 名 = qwen3.8-flash-next；惯用别名 = qwen3.8-flash-next-nvfp4
// （模型启动页展示名）与磁盘目录名 Qwen3.8-Flash-Next-NVFP4。
const MODEL_ALIASES = {
  'qwen3.8-flash-next-nvfp4': 'qwen3.8-flash-next',
  'Qwen3.8-Flash-Next-NVFP4': 'qwen3.8-flash-next',
  // 2026-09-16: twin-709 目录名与 served 名相同，显式自映射（代理转发/弹窗注册共用）
  'qwen3.8-27b-twin-709': 'qwen3.8-27b-twin-709',
};
function resolveModelAlias(model) { return MODEL_ALIASES[model] || model; }

// ====== 脚本化模型（2026-09-14）======
// 必须在容器镜像 chroot 内启动、无法走标准 vLLM 启动路径的模型（Flash-Next-NVFP4：
// PP2 + PLE-mmap，依赖 chroot 内补丁与镜像内环境）。模型启动页的「启动/停止」按钮
// 对这类模型直接调用宿主侧脚本。
// __dirname_distinct_1006__
const SCRIPT_MODELS = {
  'qwen3.8-flash-next-nvfp4': {
    dirNames: ['Qwen3.8-Flash-Next-NVFP4'],
    modelPath: '/media/ll/data/models/Qwen3.8-Flash-Next-NVFP4',
    script: '/home/ll/deploy/start-flash-next-18420.sh',
    inner: '/home/ll/deploy/flash-next-inner-param.sh',
    stopScript: '/home/ll/deploy/stop-flash-next-18420.sh',
    port: 18420,
    served: 'qwen3.8-flash-next',
    log: '/home/ll/deploy/vllm-flash-next-18420.log',
    note: '容器镜像 PP2 脚本启动，加载约 7 分钟',
    // 生产基准值（= flash-next-inner-param.sh 内置缺省）：弹窗值与之相同则不产生额外 flag，
    // 保证「不改任何参数 = 与现行生产命令逐字一致」。
    base: {
      maxModelLen: 262144, gpuMemUtil: 0.95, maxNumSeqs: 8, maxBatchedTokens: 8192,
      blockSize: 1632, temperature: 1.0, topP: 0.95, topK: 20, minP: 0.0,
      presencePenalty: 1.5, repetitionPenalty: 1.0, pp: 2, mtpTokens: 6,
    },
    // 该栈对未文档化参数零容忍（09-14 定版）：弹窗记忆里可能残留 27B 的参数，必须剔除
    bannedArgs: ['--mamba-ssm-cache-dtype', '--mamba-cache-mode', '--language-model-only',
                 '--enable-prompt-tokens-details', '--safetensors-load-strategy'],
    bannedEnv: ['QWEN_GDN_REPLAY', 'GDN_DIAG_DISABLE_JIT_MONITOR', 'CUDA_MODULE_LOADING',
                'PYTORCH_NVML_BASED_CUDA_CHECK', 'PYTORCH_CUDA_ALLOC_CONF', 'VLLM_SPEC_DECODE_ATTN',
                'VLLM_DFLASH2_LOOKUP', 'VLLM_V2_CUDAGRAPH_MEM_MIB'],
  },
};
SCRIPT_MODELS['qwen3.8-flash-next-w4a16'] = {
  dirNames: ['Qwen3.8-Flash-Next-W4A16-AutoRound', 'Qwen3.8-Flash-Next-Channel-INT8-w8a8'],  // [w4a16-restore-1006] 主映射=W4A16-AutoRound（重下完成+验收通过，10-06 切回；w8a8 目录保留兼容）
  modelPath: '/media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound',  // [w4a16-restore-1006]
  script: '/home/ll/deploy/start-flash-next-w4a16.sh',
  inner: '/home/ll/deploy/flash-next-w4a16-inner.sh',
  stopScript: '/home/ll/deploy/stop-flash-next-w4a16.sh',
  // 新栈（官方 0.30.0）在位时优先用新栈脚本对（resolve* 按 DISABLED 哨兵动态选，同看门狗）
  // [dsh-sglang-stack-1003] SGLang 栈脚本对与日志（ACTIVE 哨兵在位时 resolve* 优先选它们）
  scriptSglang: '/home/ll/deploy/sglang-18420/start-flash-next-sglang.sh',
  stopScriptSglang: '/home/ll/deploy/sglang-18420/stop-flash-next-sglang.sh',
  logSglang: '/home/ll/deploy/sglang-18420.log',
  scriptNew: '/home/ll/deploy/vllm-0310/start-flash-next-0310.sh',
  stopScriptNew: '/home/ll/deploy/vllm-0310/stop-flash-next-0310.sh',
  innerNew: '/home/ll/deploy/vllm-0310/bin/flash-next-0310-inner.sh',  // [entry-inner-1006] 0.31 引擎祖先链里的入口
  port: 18420,
  served: 'qwen3.8-flash-next',
  log: '/home/ll/deploy/vllm-flash-next-w4a16.log',
  // 官方 0.30.0 新栈的日志（start-flash-next-0300.sh 里 FN_LOG 缺省值）。祖先链 cmdline
  // 通常已能定位到它（见 ancestorLogFiles），这里兜底：cmdline 拿不到时仍能选中新日志。
  altLogs: ['/home/ll/deploy/vllm-flash-next-0310.log', '/home/ll/deploy/vllm-flash-next-0300.log'],
  note: '官方 vLLM 0.31.0 运行时补丁栈 TP1×PP2（现行生产=W4A16-AutoRound-1M，PLE=INT8 磁盘 mmap，内存二级缓存 96GiB），加载约 5~8 分钟；回滚：touch /home/ll/deploy/vllm-0310/DISABLED → 退回 chroot 旧栈',  // [stack-0310-1006]
  base: {
    maxModelLen: 262144, gpuMemUtil: 0.90, maxNumSeqs: 4, maxBatchedTokens: 8192,  // [seqs4-1006] 用户 10-06 指定 4 并发（FULL 图捕获 [1,2,4]）
    // [gen-loopfix 0929] 采样基准回到 09-21 反循环定档 t0.6 / p0.95 / k20 / minp0 / pp0.1 / rp1.0（用户 09-29 拍板：09-27 的 1/0/1 裸档导致思考模型 uct/duct token 级硬循环）。
    // base 同时是「弹窗默认值」与「是否下发 FN_GENCFG 的比较基准」，因此两套栈的 inner
    // GENCFG_DEFAULT 必须与此逐字段一致（flash-next-w4a16-inner.sh / vllm-0300/bin/flash-next-0300-inner.sh），
    // 否则会出现「弹窗显示 ≠ 引擎 cmdline 真值」。
    blockSize: 1616, temperature: 1.0, topP: 0.95, topK: 20, minP: 0.0,  // [gen-1010 用户定档] 官方默认档
    presencePenalty: 0.0, repetitionPenalty: 1.0, pp: 2, mtpTokens: 4,  // [w4a16-restore-1006] 本机在位 2×CMP 170HX（lspci 仅 03:00/04:00），PP 基准 3→2；三卡机勿套此值
    // [hw-move 1005] 换装 HUANANZHI H12D-8D + EPYC 7F52 + 3×CMP 170HX + 仅 32GB 内存：
    // ① PP 基准 2→3（三卡满配，与 10-04 实跑实例 FN_PP=3 一致）；
    // ② 二级缓存关（classic 96GiB / SimpleCPU 100GiB 在 64GB 机即实锤 OOM，32GB 更快炸）；
    // ③ PLE 唯一可行档 = INT8+disk（mmap 47.7GiB 可回收页缓存，10-04 实跑验证；
    //    INT8 heap 匿名堆 48.3GiB 与 BF16 95.4GiB 均不可回收、必 OOM）。
    // （存档：0929/0927 定版的 kvoff=simple 96GiB 与 pleLoc=heap 均以大内存机为前提，已随换装作废。）
    kvoff: 'simple', kvoffGiB: 160, pleInt8: '1', pleLoc: 'heap',  // [kvoff160 10-10] 251GB 机账：48.3 PLE匿名 + 160 pinned + ~12 引擎 ≈220/251，权重页缓存被挤（重启多付加载时长）；内存告警第一刀降 96。PLE 放内存=INT8 匿名堆不可回收。「放硬盘」档=mmap 可回收页缓存〔原生产档，随时可切回〕。 // [stack-0310-1006] 二级缓存=SimpleCPUOffloadConnector 96GiB（每 rank ≈48GiB 锁页）；PLE=INT8 产物磁盘 mmap（/media/ll/data/ple，47.7+0.6 GiB 可回收页缓存）。0.31 只有两档：mmap-INT8 / pinned-BF16——弹窗「精度=BF16」或「位置=heap」都会走官方锁页 BF16 95.4GiB
  },
  bannedArgs: ['--mamba-ssm-cache-dtype', '--mamba-cache-mode', '--language-model-only',
               '--enable-prompt-tokens-details', '--safetensors-load-strategy',
               '--block-size', '--pipeline-parallel-size'],
  bannedEnv: ['QWEN_GDN_REPLAY', 'GDN_DIAG_DISABLE_JIT_MONITOR', 'CUDA_MODULE_LOADING',
              'PYTORCH_NVML_BASED_CUDA_CHECK', 'PYTORCH_CUDA_ALLOC_CONF', 'VLLM_SPEC_DECODE_ATTN',
              'VLLM_DFLASH2_LOOKUP', 'VLLM_V2_CUDAGRAPH_MEM_MIB'],
  aliases: ['Qwen3.8-Flash-Next-Channel-INT8-w8a8', 'Qwen3.8-Flash-Next-W4A16-AutoRound', 'qwen3.8-flash-next-w4a16-autoround'],
  // ===== 1M 长上下文（2026-09-16）=====
  // 本 checkpoint 原生 text_config.max_position_embeddings=262144、rope_type=default：
  // 直接把 max_model_len 填 >262144 会被 vLLM 拒（VLLM_ALLOW_LONG_MAX_MODEL_LEN），
  // 即便放行，RoPE 未缩放时位置越过 262144 会 NaN/越界。故 1M 走 YaRN×4 配置副本：
  //   /media/ll/data/models-1m/Qwen3.8-Flash-Next-W4A16-AutoRound-1M
  //   （max_position_embeddings 1048576 + rope_parameters{yarn, factor 4.0, original 262144}，
  //    其余文件全部软链回原目录，原 checkpoint 零改动；生成脚本 make-1m-config.py）
  // 上限 maxModelLenLong：mrope 的 cos/sin 缓存 = 262144×4 = 1048576，再大就位置越界。
  // 长上下文档位（2026-09-16）：原生 256K / 512K(YaRN×2) / 1M(YaRN×4)。
  //   每档一个「只改 config.json、其余软链回原目录」的副本；factor = 目标长度 / 262144。
  //   mrope 缓存恒 = original(262144) × 4 = 1048576，故 512K、1M 都安全（≤ 缓存）。
  altModelPaths: ['/media/ll/data/models-1m/Qwen3.8-Flash-Next-W4A16-AutoRound-1M'],  // [w4a16-restore-1006] 1M=YaRN×4 W4A16 副本
  longCtxModelPath: '/media/ll/data/models-1m/Qwen3.8-Flash-Next-W4A16-AutoRound-1M',  // [w4a16-restore-1006]
  pleInt8Dir: '/media/ll/data/ple',  // [w4a16-restore-1006] W4A16 专属 INT8 表产物目录（不存在则 inner 回落 BF16 磁盘驻留）；禁止跨 checkpoint 复用（inner 缺省是 ple-w8a8）
  // [model-variant-1008] 一个脚本条目挂多个 checkpoint（dirNames 多目录）时的「按目录识别」
  // 映射表：点哪张卡/哪个预设，就按对应 variant 下发 FN_MODEL_PATH/FN_1M_MODEL_PATH/
  // FN_PLE_INT8_DIR，不再一律落到注册键的 sm.modelPath（旧 bug：点 Channel 启动 W4A16）。
  // pp/minGpus/kvoffCap 为该 checkpoint 的硬约束护栏（依据见各 variant 行内注释）。
  modelVariants: {
    'Qwen3.8-Flash-Next-W4A16-AutoRound': {
      modelPath: '/media/ll/data/models/Qwen3.8-Flash-Next-W4A16-AutoRound',
      longCtxModelPath: '/media/ll/data/models-1m/Qwen3.8-Flash-Next-W4A16-AutoRound-1M',
      pleInt8Dir: '/media/ll/data/ple',
    },
    'Qwen3.8-Flash-Next-Channel-INT8-w8a8': {
      modelPath: '/media/ll/data/models/Qwen3.8-Flash-Next-Channel-INT8-w8a8',
      longCtxModelPath: '/media/ll/data/models-1m/Qwen3.8-Flash-Next-Channel-INT8-w8a8-1M',
      pleInt8Dir: '/media/ll/data/ple-w8a8',  // w8a8 专属 INT8 表产物（ple_ngram_meta 完整校验通过 10-08）
      pp: 3, minGpus: 3,        // 权重 ~132GB：PP2 每段 66GB > 单卡 64GB 显存，必须 PP3（本机 3 卡在位）
      kvoff: 'simple', kvoffGiB: 160, kvoffCap: 160,  // [kvoff160 10-10 用户要求 96→160] 251GB 机账：48.3 PLE + 160 pinned + 引擎 ≈220，权重页缓存(可回收)压余量；内存告警第一刀降 96
    },
  },  // __modelvariant_1008__
  maxModelLenLong: 1048576,
};
MODEL_ALIASES['qwen3.8-flash-next-w4a16'] = 'qwen3.8-flash-next';
MODEL_ALIASES['Qwen3.8-Flash-Next-W4A16-AutoRound'] = 'qwen3.8-flash-next';
MODEL_ALIASES['Qwen3.8-Flash-Next-Channel-INT8-w8a8'] = 'qwen3.8-flash-next';  // [w8a8-1005]
MODEL_ALIASES['Qwen3.8-Flash-Next-Channel-INT8-w8a8-1M'] = 'qwen3.8-flash-next';

function scriptModelForName(name) {
  if (!name) return null;
  if (SCRIPT_MODELS[name]) return Object.assign({ key: name }, SCRIPT_MODELS[name]);
  for (const k of Object.keys(SCRIPT_MODELS)) {
    const v = SCRIPT_MODELS[k];
    if ((v.dirNames || []).indexOf(name) >= 0) return Object.assign({ key: k }, v);
  }
  return null;
}
// [script-entry-route-1006] 脚本化模型实例认领的共享工具。
// 背景（126 机 10-06 实锤）：旧认领=「进程 cmdline 含候选模型路径之一」——同注册键
// （SCRIPT_MODELS 条目）挂多个 dirNames/altModelPaths 时，任何一个在跑目录被命中，
// 该条目名下所有目录卡片都显示「运行中」（AutoRound 卡明明没跑也亮绿，且点它会因
// 「已在运行」被拒启动）。正解=按「启动入口脚本」认领：引擎进程的 wrapper 脚本
// （script/scriptNew/scriptSglang）与条目一致才算该条目在跑；卡片再按「自身目录
// 是否是引擎实际加载路径（或其 YaRN 副本基底）」细分显示。
function smEntryScripts(v) {
  // [entry-inner-1006] 追加 inner/innerNew：wrapper 是 `bash start-*.sh` → 内部
  //   `sudo sh -c 'exec setsid bash <inner> >> log'`，setsid 后 wrapper 本体脱离进程树，
  //   引擎祖先链 cmdline 只剩 <inner> 路径（10-06 实锤：停止按钮误判未运行→去启动→
  //   探到旧实例 health=200→返回 healthy 但根本没停）。只收 start-* 会全瞎。
  return [v.script, v.scriptNew, v.scriptSglang, v.inner, v.innerNew].filter(Boolean);
}
function readCmdlineArgv(pid) {
  try { return fs.readFileSync('/proc/' + pid + '/cmdline', 'utf8').split('\0').filter(Boolean); } catch (e) { return []; }
}
function procCmdline(pid) {
  try { return fs.readFileSync('/proc/' + pid + '/cmdline', 'utf8').split('\0').join(' '); } catch (e) { return ''; }
}
// 祖先链上（含自身之上、≤maxHops 层）是否出现过该 wrapper 脚本的 cmdline。
// wrapper 谱系：bash <script> → sudo -S setsid chroot … <inner> → python 引擎（各级
// cmdline 各不相同）；引擎被 kill 后 wrapper 若还活着（停止中）依然命中=仍在运行，语义正确。
function procHasAncestorScript(pid, scripts, maxHops) {
  if (!scripts || !scripts.length) return false;
  let cur = pid | 0;
  for (let hop = 0; hop <= (maxHops || 8) && cur > 1; hop++) {
    let ppid = 0;
    try {
      const st = fs.readFileSync('/proc/' + cur + '/stat', 'utf8');
      ppid = parseInt(st.slice(st.lastIndexOf(')') + 1).trim().split(/\s+/)[1], 10) || 0;
    } catch (e) { break; }
    if (!ppid || ppid === cur) break;
    cur = ppid;
    const cmd = procCmdline(cur);
    if (cmd && scripts.some((s) => cmd.includes(s))) return true;
  }
  return false;
}
// 引擎 cmdline 的模型路径（serve 后第一个非 flag / --model / --model-path）
function procModelPath(cmdOrArgv) {
  // 兼容两种入参：/proc cmdline 原文（\0 分隔）或已 split 的 argv 数组
  const argv = Array.isArray(cmdOrArgv) ? cmdOrArgv : String(cmdOrArgv).split('\0').filter(Boolean);
  for (let i = 0; i < argv.length; i++) {
    if ((argv[i] === '--model' || argv[i] === '--model-path') && argv[i + 1]) return argv[i + 1];
    if (argv[i] === 'serve' && argv[i + 1] && !String(argv[i + 1]).startsWith('-')) return argv[i + 1];
  }
  return '';
}
// 把一个模型目录解析成「启动条目目录」集合：自身 + 指向它的外部软链（如
// /home/ll/models → /media/ll/data/models）+ 其软链子目录（models-1m 的 YaRN 副本
// 目录：10-04 用 cp -as 创建，实体文件只有 config 等少数、其余全是指回基底的软链，
// 例如 <副本>/LICENSE → <基底>/LICENSE）。[dir-entry-map-1006]
function resolveEntryDirs(dirPath) {
  const out = new Set();
  const norm = (x) => String(x || '').replace(/\/+$/, '');
  if (!dirPath) return out;
  const a = norm(dirPath);
  out.add(a);
  try { out.add(norm(fs.realpathSync(a))); } catch (e) {}
  // 外部软链：同父目录下哪些条目 symlink 指向本目录
  try {
    const parent = a.slice(0, a.lastIndexOf('/'));
    for (const d of fs.readdirSync(parent)) {
      const full = parent + '/' + d;
      if (full === a) continue;
      try {
        if (fs.lstatSync(full).isSymbolicLink() && norm(fs.realpathSync(full)) === a) out.add(full);
      } catch (e) {}
    }
  } catch (e) {}
  // 软链子目录（YaRN 上下文副本判据：目录名以本目录名为前缀 + 其下有 ≥1 个
  // 回指本目录的软链文件，防误伤恰好同前缀的独立目录）。副本常在**别的父目录**
  // （如基底在 models/、副本在 models-1m/，10-04 定版），父目录集合=自身父级 +
  // MODELS_DIR 同级派生目录（models-1m/models-512k 等）。[dir-parent-cross-1006]
  try {
    const parent2 = a.slice(0, a.lastIndexOf('/'));
    const gp = parent2.slice(0, parent2.lastIndexOf('/'));
    const parents = new Set([parent2]);
    try {
      for (const d of fs.readdirSync(gp)) {
        const full = gp + '/' + d;
        try { if (fs.lstatSync(full).isDirectory() && !fs.lstatSync(full).isSymbolicLink() && /^models([-_]|$)/.test(d) && full !== parent2) parents.add(full); } catch (e) {}
      }
    } catch (e) {}
    const myName = a.slice(a.lastIndexOf('/') + 1);
    for (const par of parents) {
    let ds = [];
    try { ds = fs.readdirSync(par); } catch (e) { continue; }
    for (const d of ds) {
      const full = par + '/' + d;
      if (full === a || d.indexOf(myName) !== 0 || d.length <= myName.length) continue;
      let st = null;
      try { st = fs.lstatSync(full); } catch (e) { continue; }
      if (!st.isDirectory() || st.isSymbolicLink()) continue;
      let linkedBack = false;
      try {
        // 任一软链回指本目录即认定副本（YaRN 副本用 cp -as 创建、百余文件全是指回
        // 基底的软链，实体只有 config 等数个；扫描带上隐藏文件、不设上限截断）
        for (const f of fs.readdirSync(full, { withFileTypes: true })) {
          try {
            if (!f.isSymbolicLink()) continue;
            const tgt = norm(fs.readlinkSync(full + '/' + f.name));
            if (tgt === a || tgt.startsWith(a + '/')) { linkedBack = true; break; }
          } catch (e) {}
        }
      } catch (e) {}
      if (linkedBack) out.add(norm(full));
    }
    }
  } catch (e) {}
  return out;
}
// 某模型目录是否「就是」引擎加载路径（含软链别名与 YaRN 副本展开）。
function modelDirMatchesEnginePath(dirPath, enginePath) {
  if (!dirPath || !enginePath) return false;
  const b = String(enginePath).replace(/\/+$/, '');
  for (const d of resolveEntryDirs(dirPath)) if (d === b) return true;
  return false;
}

// __script_entry_route_1006__
// 脚本模型实例探测：chroot 内引擎属 root（ll 的 lsof 看不到监听端口），按
// /proc/<pid>/cmdline 里的模型路径认领，端口从 --port 动态解析（弹窗可改端口）。
// 1M 档会用 longCtxModelPath（altModelPaths），故这里要把所有候选路径都算上。
function scriptModelInstance(sm) {
  if (!sm) return null;
  const paths = [sm.modelPath].concat(sm.altModelPaths || []).filter(Boolean);
  try {
    const { execSync } = require('child_process');
    // 09-26：两种栈形态都要认——旧 chroot 栈 cmdline 是 `python -m vllm.entrypoints.cli.main serve`，
    // 新官方 0.30.0 栈是 `/…/bin/vllm serve …`（comm=vllm）。旧模式只认前者，导致新栈在跑
    // 而本函数恒返回 null（停止/状态判活全瞎）。括号技巧防匹配到发起 pgrep 的父 shell 自身。
    const out = execSync('pgrep -f "[v]llm.entrypoints|[v]llm serve|[s]glang.launch_server"   2>/dev/null || true', { encoding: 'utf8', timeout: 3000 }).trim();
    for (const pid of out.split('\n').filter(Boolean)) {
      try {
        const cmd = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8').split('\0').join(' ');
        if (/(^|\s)(ssh|bash|sh|expect)(\s|$)/.test(cmd)) continue;
        // [script-entry-route-1006] 认领按「启动入口脚本」而非模型路径：同注册键条目挂着
        // 多个 dirNames/altModelPaths 时互不串味——只有被该条目自己的 wrapper（script/
        // scriptNew/scriptSglang）拉起的引擎才算这条目在跑。无入口脚本的条目退回旧路径认领。
        const entryScripts = smEntryScripts(sm);
        if (entryScripts.length) {
          if (!procHasAncestorScript(parseInt(pid), entryScripts, 8)) continue;
        } else if (paths.length && !paths.some(pp => {
          for (const ed of resolveEntryDirs(pp)) if (cmd.includes(ed)) return true;
          return false;
        })) continue;
        const m = cmd.match(/--port\s+(\d+)/);
        if (m) return { port: parseInt(m[1]), pid: parseInt(pid), engineModelPath: procModelPath(readCmdlineArgv(pid)) };
      } catch (e) {}
    }
  } catch (e) {}
  return null;
}
function scriptModelAlive(sm) { return !!scriptModelInstance(sm); }
function scriptModelForPort(port) {
  const p = parseInt(port);
  if (!p) return null;
  // ① 先认「真有活进程」的条目（scriptModelInstance 已兼容新旧栈形态）
  for (const k of Object.keys(SCRIPT_MODELS)) {
    const v = SCRIPT_MODELS[k];
    const inst = scriptModelInstance(v);
    if (inst && inst.port === p) return Object.assign({ key: k }, v);
  }
  // ② 再按端口声明兜底，但跳过启停脚本都已不存在的退役条目
  //（18420 先后注册过 NVFP4+W4A16 两条目，盲取第一个会拿到退役栈不存在的
  //  stop-flash-next-18420.sh——09-26「停止按钮点了没任何动静」的直接根因）
  const hasScripts = (v) => { try { return fs.existsSync(v.stopScript || '') || fs.existsSync(v.script || ''); } catch (e) { return false; } };
  for (const k of Object.keys(SCRIPT_MODELS)) {
    const v = SCRIPT_MODELS[k];
    if (v.port === p && hasScripts(v)) return Object.assign({ key: k }, v);
  }
  return null;
}
// 栈感知停止脚本解析（09-26）：18420 现由看门狗托管在【官方 vLLM 0.30.0 新栈】，
// 其主进程 comm=vllm、cmdline 是 "vllm serve"——旧栈停止脚本的 find_pids（只认 VLLM:: 子进程
// 与 entrypoints.cli.main）抓不到主进程，点了等于没杀；且看门狗 60s 内会按 launch.env 拉起，
// 表现为「停止不成功」。规则与 fnx-18420-watchdog.sh 选栈同源：DISABLED 哨兵缺席=新栈。
// 新栈 stop 脚本同时兼容旧栈主进程判据，并在入口 touch 人工停止闩锁 fnx-manual-stop。
const STACK0300_DISABLED = '/home/ll/deploy/vllm-0300/DISABLED';
// [dsh-sglang-stack-1003] 第三栈：SGLang@18420。sglang-18420/ACTIVE 哨兵在位 = 现行生产走 sglang 脚本对，
// 优先级高于 vllm-0300/DISABLED 双栈判定。移除哨兵文件即回退旧路由。
const SGLANG_ACTIVE = '/home/ll/deploy/sglang-18420/ACTIVE';
function sglangActive() {
  try { return fs.existsSync(SGLANG_ACTIVE); } catch (e) { return false; }
}
function stack0300Active() {
  // [stack-0310-1006] 「新栈」已升到官方 vLLM 0.31.0（vllm-0310）：
  // 哨兵换名生效——vllm-0310/DISABLED 在位 = 退回 chroot 旧栈；不在位 = 走 0.31.0。
  // （0.30.0 已被 0.31.0 取代，其 DISABLED 不再参与判定；回滚 0.30 用本补丁 --revert。）
  try { return !fs.existsSync('/home/ll/deploy/vllm-0310/DISABLED'); } catch (e) { return false; }
}
function resolveStopScript(sm) {
  try {
    if (sm && sm.stopScriptSglang && sglangActive() && fs.existsSync(sm.stopScriptSglang)) return sm.stopScriptSglang;  // [dsh-sglang-stack-1003]
    if (sm && sm.stopScriptNew && stack0300Active() && fs.existsSync(sm.stopScriptNew)) return sm.stopScriptNew;
  } catch (e) {}
  return sm && sm.stopScript;
}
// 启动脚本同样栈感知：新栈在位时用 start-flash-next-0300.sh（其 FN_* 透传/落盘机制与旧栈
// wrapper 同构）。注意解析到新栈时不能再传 INNER=<旧栈 inner>——新栈 wrapper 的 INNER 缺省
// 指向自家 inner，被旧值覆盖会跑出「新栈环境 + 旧栈 inner」的杂交命令（09-26 评审发现）。
function resolveStartScript(sm) {
  try {
    if (sm && sm.scriptSglang && sglangActive() && fs.existsSync(sm.scriptSglang)) return sm.scriptSglang;  // [dsh-sglang-stack-1003]
    if (sm && sm.scriptNew && stack0300Active() && fs.existsSync(sm.scriptNew)) return sm.scriptNew;
  } catch (e) {}
  return sm && sm.script;
}
// 弹窗默认值（= 生产基准）：前端 startConfigDefaults 用它覆盖 localStorage 记忆，
// 避免别的模型（如 27B 的 --mamba-ssm-cache-dtype）参数串到该栈上。
function scriptModelDefaults(sm) {
  const b = sm.base || {};
  return {
    port: sm.port, servedName: sm.served, maxModelLen: String(b.maxModelLen), ctxLen: '',
    gpuId: 0, gpuCount: sm.variantPp || b.pp || 2, parallelMode: 'pp', pdMode: '0',  // [model-variant-1008]
    maxSeqs: b.maxNumSeqs, maxBatchedTokens: b.maxBatchedTokens,
    blockSize: b.blockSize, temperature: b.temperature, topP: b.topP, topK: b.topK,
    minP: b.minP, presencePenalty: b.presencePenalty, repetitionPenalty: b.repetitionPenalty,
    // [gen-default 0927] 思考深度缺省 xhigh（启动页默认；脚本模型经 FN_CHATKWARGS 真下发到引擎）
    thinking: '1', thinkingEffort: 'xhigh', kvCacheQuant: 'auto', retention: '',
    // [mtp-default 0923] 投机缺省=生产现状：W4A16 档 MTP4 实测稳定（09-19：接受长度 2.84、
    // decode 111 tok/s、无乱码；block-size 1616 下合法档 1~4 与 9~12，5~8 启动即崩），
    // 故弹窗默认选中「MTP」并带出生产档位 mtpTokens（base=4）。NVFP4 档保持关闭：
    // 本镜像 PP2 下 MTP 有连续长请求 device assert 残余 bug。
    mtp: sm.key === 'qwen3.8-flash-next-w4a16' ? '1' : '0', dflash: '0', dspark: '0', mtpTokens: b.mtpTokens || 6,
    // [sglang-adapt-1003] SGLang 栈（sglang-18420/ACTIVE 哨兵）在位时，脚本模型的现行生产
    // 引擎是 sglang：弹窗默认 runtime 跟随，否则用户看到 vLLM 面板、启动的却是 sglang 栈，
    // 且 sglang 的 mem-fraction-static 上限 0.88（给 decode 期动态内核留余量）必须显示为真值。
    runtime: (typeof sglangActive === 'function' && sglangActive()) ? 'sglang' : 'vllm',
    gpuMemUtil: (typeof sglangActive === 'function' && sglangActive())
      ? Math.min(b.gpuMemUtil || 0.88, 0.88) : b.gpuMemUtil,
    dtype: 'auto',
    // 1M 长上下文开关（'0' 原生 256K / '1' YaRN×4 → 1M）；maxModelLenLong=0 表示该脚本
    // 模型没提供 1M 档（如 NVFP4 栈），前端据此隐藏开关
    longCtx: '0', maxModelLenLong: sm.maxModelLenLong || 0, maxModelLen512: sm.maxModelLen512 || 0,
    // [pagefix-1005] PLE 驻留精度/位置与 CPU 二级缓存缺省跟随 base（10-05 换装定版已在 base
    // 写明 kvoff/pleInt8/pleLoc）。本机 32GB 内存下 INT8 heap/BF16 heap/大 GiB 二级缓存均必 OOM，
    // base 未声明时回落安全档：INT8+disk、二级缓存关。旧版前端把这三档写死导致启动页误炸。
    pleInt8: String(b.pleInt8 || '1'), pleLoc: String(b.pleLoc || 'disk'),
    kvoff: String(sm.variantKvoff || b.kvoff || '0'), kvoffGiB: sm.variantKvoffGiB || b.kvoffGiB,  // __pagefix_1005__ [model-variant-1008]
  };
}
// ====== Flash-Next 启动方案（2026-09-14）======
// 方案二的参数由上游仓库 config/vllm-args.json + config/runtime-env.json 提取到独立 JSON，
// 便于随上游更新而无需改动 server.js 代码。
const FLASHNEXT_SCHEME_FILE = '/home/ll/deploy/flashnext-scheme-170hx.json';
function loadFlashNextSchemeLocal() {
  try {
    return JSON.parse(fs.readFileSync('/home/ll/deploy/flashnext-scheme-170hx-local.json', 'utf8'));
  } catch (e) { return null; }
}

function loadFlashNextScheme170hx() {
  try {
    return JSON.parse(fs.readFileSync(FLASHNEXT_SCHEME_FILE, 'utf8'));
  } catch (e) { return null; }
}
// [mm-perf-1003] 模型 revision 探测（分片数 + index.json 的 metadata.total_size），带缓存。
// 为什么要专门优化这一步（10-03 实锤）：为了拿一个 total_size，旧代码把
// model.safetensors.index.json 整个读进来再 JSON.parse —— 本机该文件 34MB / 744 分片，
// 实测单次 267~390ms，而 schemeReadiness 的 tep2 与 script 两个分支各做一次，
// flashNextSchemes() 又对每个脚本模型卡片各调一遍 ⇒ /v1/internal/model-manager 单请求
// 1.5~6.2s，全程同步按住 node 事件循环，同进程其它接口跟着变慢；前端 vllm.html 的
// 8s 超时被吃掉后就 catch→null→instances=[]，把「这次没取到数据」渲染成「无运行中实例」。
// 分片清单与 total_size 只在换权重时才变：按目录 mtime + TTL 缓存，且只流式读文件头部。
const __modelRevisionProbeCache = new Map();   // mp -> { at, mtimeMs, shards, bytes }
const MODEL_REVISION_PROBE_TTL = 120000;       // 2 分钟；目录 mtime 变化立即失效
function probeModelRevision(mp) {
  let mtimeMs = 0;
  try { mtimeMs = fs.statSync(mp).mtimeMs; } catch (e) { return { shards: 0, bytes: null }; }
  const now = Date.now();
  const hit = __modelRevisionProbeCache.get(mp);
  if (hit && hit.mtimeMs === mtimeMs && now - hit.at < MODEL_REVISION_PROBE_TTL) {
    return { shards: hit.shards, bytes: hit.bytes };
  }
  let shards = 0;
  try {
    shards = fs.readdirSync(mp).filter(f => f.endsWith('.safetensors')).length;
  } catch (e) {}
  let bytes = null;
  try {
    // total_size 位于文件开头的 metadata 段：分块读、命中即停，不解析整个索引
    const fd = fs.openSync(mp + '/model.safetensors.index.json', 'r');
    try {
      let buf = '', pos = 0, guard = 0;
      while (guard++ < 64) {
        const chunk = Buffer.alloc(65536);
        const n = fs.readSync(fd, chunk, 0, chunk.length, pos);
        if (n <= 0) break;
        pos += n;
        buf += chunk.toString('utf8', 0, n);
        const m = /"total_size"\s*:\s*(\d+)/.exec(buf);
        if (m) { bytes = parseInt(m[1]); break; }
        if (buf.length > 4194304) break;   // 保险丝：4MB 内找不到就当无该字段
      }
    } finally { fs.closeSync(fd); }
  } catch (e2) {}
  __modelRevisionProbeCache.set(mp, { at: now, mtimeMs, shards, bytes });
  return { shards, bytes };
}
// 方案二前置条件探测（Docker / GDS / PLE GDS 数据 / 镜像 / 模型 revision）
function schemeReadiness(scheme) {
  const r = { docker: false, image: false, cufile: false, nvidia_fs: false, pleArtifact: false, model: false };
  // 本机可运行形态（launcher.mode==='script'）：不依赖 Docker/GDS，只校验脚本与模型
  if (scheme && scheme.launcher && scheme.launcher.mode === 'script') {
    try { r.wrapper = fs.existsSync(scheme.launcher.wrapper); } catch (e) { r.wrapper = false; }
    try { r.inner = fs.existsSync(scheme.launcher.inner); } catch (e) { r.inner = false; }
    try {
      const mp = scheme.launcher.model || '/media/ll/data/models/Qwen3.8-Flash-Next-NVFP4';
      const pv = probeModelRevision(mp);
      const want = scheme.launcher.modelBytes || 135195303851;
      r.model = pv.shards >= 200 && (pv.bytes === null || Math.abs(pv.bytes - want) / want < 0.01);
      r.modelShards = pv.shards; r.modelBytes = pv.bytes || null;
    } catch (e) { r.model = false; }
    return r;
  }
  try {
    const out = require('child_process').execSync('command -v docker 2>/dev/null || true', { encoding: 'utf8', timeout: 3000 }).trim();
    r.docker = !!out;
  } catch (e) {}
  if (r.docker) {
    try {
      const img = (scheme && scheme.launcher && scheme.launcher.image) || 'qwen-flash-sm80:0.1.4';
      require('child_process').execSync('docker image inspect --format "{{.Id}}" ' + JSON.stringify(img) + ' 2>/dev/null', { timeout: 5000 });
      r.image = true;
    } catch (e) { r.image = false; }
  }
  try {
    r.cufile = fs.existsSync('/usr/lib/x86_64-linux-gnu/libcufile.so.0') ||
               fs.existsSync('/usr/lib/x86_64-linux-gnu/libcufile.so') ||
               fs.existsSync('/usr/local/cuda/lib64/libcufile.so');
  } catch (e) {}
  try {
    r.nvidia_fs = fs.existsSync('/sys/module/nvidia_fs') ||
                  (fs.existsSync('/proc/modules') && fs.readFileSync('/proc/modules', 'utf8').indexOf('nvidia_fs') >= 0);
  } catch (e) {}
  try { r.pleArtifact = fs.existsSync('/media/ll/data/ple-gds/CURRENT'); } catch (e) {}
  // 模型：目录存在 + 分片数量 + index total_size 与上游 revision 期望值比对（容许 1% 差异）
  try {
    const pv = probeModelRevision('/media/ll/data/models/Qwen3.8-Flash-Next-NVFP4');
    r.modelShards = pv.shards;
    r.modelBytes = pv.bytes || null;
    const want = (scheme && scheme.launcher && scheme.launcher.modelBytes) || 135195303851;
    r.model = pv.shards >= 200 && (pv.bytes === null || Math.abs(pv.bytes - want) / want < 0.01);
  } catch (e) { r.model = false; }
  return r;
}
// [mm-perf-1003] 整套方案就绪度的短缓存：schemeReadiness 除模型 revision 外还要 spawn
// docker 两次（command -v + image inspect，实测 ~30ms），叠加后 flashNextSchemes() 单次
// ~300ms。这类前置条件（装了没装 Docker、镜像在不在、权重对不对）分钟级都不会变，
// 而 model-manager 是前端 2s 轮询的热点接口 ⇒ 缓存 5s，既保住「刚装好就能看见」的实时感，
// 又把重复开销摊薄到近乎为零。
const __schemesCache = { at: 0, list: null };
const SCHEMES_CACHE_TTL = 5000;
function flashNextSchemes() {
  const now = Date.now();
  if (__schemesCache.list && now - __schemesCache.at < SCHEMES_CACHE_TTL) return __schemesCache.list;
  const out = [{
    key: 'chroot-pp2',
    name: '当前生产（chroot 镜像 + PP2 + PLE mmap，无投机）',
    kind: 'chroot',
    ready: true,
    missing: [],
    note: '本机现用方案：PP2 双卡流水并行，PLE 走 mmap + CPU offload，MTP 关闭（该镜像下不稳）。弹窗参数化启动即走此方案。',
  }];
  const s170 = loadFlashNextScheme170hx();
  if (s170) {
    const rd = schemeReadiness(s170);
    const missing = (s170.requires || []).filter(x => !rd[x.key]).map(x => x.label);
    out.push({
      key: s170.key, name: s170.name, kind: 'docker', source: s170.source, note: s170.note,
      vllmArgs: s170.vllmArgs, env: s170.env, launcher: s170.launcher,
      readiness: rd, missing: missing, ready: missing.length === 0,
    });
  }
  const sLocal = loadFlashNextSchemeLocal();
  if (sLocal) {
    const rdL = schemeReadiness(sLocal);
    const missingL = (sLocal.requires || []).filter(x => !rdL[x.key]).map(x => x.label);
    out.push({
      key: sLocal.key, name: sLocal.name, kind: 'chroot', source: sLocal.source, note: sLocal.note,
      vllmArgs: sLocal.vllmArgs, env: sLocal.env, launcher: sLocal.launcher,
      readiness: rdL, missing: missingL, ready: missingL.length === 0,
    });
  }
  __schemesCache.list = out;
  __schemesCache.at = Date.now();
  return out;
}
// 弹窗参数 → chroot 启动环境变量（FN_*）；与基准相同的项不产生额外 flag。
// [model-variant-1008] 按弹窗/预设传入的目录名（或模型路径）解析出携带该 checkpoint
// 路径与护栏的 sm 浅拷贝。命中判据=目录名与 variant 键相等或为其前缀延伸
// （xxx-1M / xxx-512K 副本目录同样归位到母 variant）。未命中返回原条目=旧行为零漂移。
function scriptModelVariantResolve(sm, d) {
  const variants = sm && sm.modelVariants;
  if (!variants) return sm;
  const keys = Object.keys(variants);
  const cands = d ? [d.modelPath, d.dir_name, d.dirName, d.modelName, d.model] : [];
  let hit = null;
  for (const c of cands) {
    if (!c) continue;
    const base = String(c).trim().replace(/\/+$/, '').split('/').pop();
    if (!base) continue;
    for (const k of keys) {
      if (base === k || base.indexOf(k + '-') === 0) { hit = k; break; }
    }
    if (hit) break;
  }
  if (!hit) return sm;
  const v = variants[hit] || {};
  const out = Object.assign({}, sm);
  out.variantName = hit;
  if (v.modelPath) out.modelPath = v.modelPath;
  if (v.longCtxModelPath) { out.longCtxModelPath = v.longCtxModelPath; out.altModelPaths = [v.longCtxModelPath]; }
  if (v.pleInt8Dir !== undefined) out.pleInt8Dir = v.pleInt8Dir;
  if (v.pp) { out.variantPp = v.pp; out.variantMinGpus = v.minGpus || v.pp; }
  if (v.kvoff) out.variantKvoff = String(v.kvoff);
  if (v.kvoffGiB) out.variantKvoffGiB = v.kvoffGiB;
  if (v.kvoffCap !== undefined) out.variantKvoffCap = v.kvoffCap;
  return out;
}

function scriptModelLaunchPlan(smIn, d) {
  // [model-variant-1008] 先把「用户点的是哪个 checkpoint」解析成条目浅拷贝：
  // 旧版把 FN_MODEL_PATH 写死 sm.modelPath —— 点 Channel 卡启动的却是 W4A16（10-08 实锤）。
  const sm = scriptModelVariantResolve(smIn, d);  // __modelvariant_1008__
  const b = sm.base || {};
  const warnings = [];
  // 不写死 FN_LOG：wrapper 按端口命名为 vllm-flash-next-${PORT}.log
  const env = { FN_MODEL_PATH: sm.modelPath };
  if (sm.pleInt8Dir) env.FN_PLE_INT8_DIR = sm.pleInt8Dir;  // [w4a16-restore-1006] 防落 inner 缺省(w8a8 表)
  const num = (v, dflt) => { const n = parseFloat(v); return isNaN(n) ? dflt : n; };
  const int = (v, dflt) => { const n = parseInt(v, 10); return isNaN(n) ? dflt : n; };
  const port = int(d.port, sm.port);
  const served = String(d.servedName || sm.served).trim() || sm.served;
  env.FN_PORT = String(port);
  env.FN_SERVED = served;
  // 上下文：ctxLen 优先，其次 maxModelLen；空/auto → 基准
  let maxLen = String(d.ctxLen || '').trim() || String(d.maxModelLen || '').trim();
  if (!maxLen || maxLen === 'auto') maxLen = String(b.maxModelLen);
  // ===== 长上下文档位（2026-09-16）：原生 256K / 512K(YaRN×2) / 1M(YaRN×4) =====
  // 触发路径：① 弹窗「上下文扩展」显式选档（longCtx='512k' | '1m'，旧值 '1' 兼容为 1m）；
  //          ② 上下文数值本身 >262144 → 按数值就近自动选档（否则会像 09-16 18:28 那次
  //             填 512144 被 vLLM 直接拒启 ValueError；且 >262144 不缩放会 NaN）。
  // 选档后：模型切到对应 YaRN 副本目录（inner 脚本兜底再判一次），maxLen 硬钳到该档上限。
  // 上限依据：mrope cos/sin 缓存 = original(262144)×4 = 1048576，512K/1M 都在缓存内。
  const cap1m = int(sm.maxModelLenLong, 0);     // 0 = 该脚本模型未提供长上下文档
  const cap512 = int(sm.maxModelLen512, 0);
  const baseCap = int(b.maxModelLen, 262144);
  const nMaxLen = int(maxLen, NaN);
  let tier = '';                                 // '' = 原生；'512k' / '1m'
  const lcRaw = String(d.longCtx || '').trim().toLowerCase();
  if (lcRaw === '1' || lcRaw === '1m') tier = '1m';
  else if (lcRaw === '512k' || lcRaw === '512') tier = '512k';
  else if (cap512 > 0 && !isNaN(nMaxLen) && nMaxLen > baseCap && nMaxLen <= cap512) tier = '512k';
  else if (cap1m > baseCap && !isNaN(nMaxLen) && nMaxLen > baseCap) tier = '1m';
  // 该模型若没有 1M 档（只有 512K 或全无），按可用上限收敛 tier
  if (tier === '1m' && !(cap1m > baseCap)) tier = (cap512 > baseCap ? '512k' : '');
  if (tier === '512k' && !(cap512 > baseCap)) tier = (cap1m > baseCap ? '1m' : '');
  if (tier) {
    const tierCap = tier === '1m' ? cap1m : cap512;
    const tierPath = tier === '1m'
      ? (sm.longCtxModelPath || (sm.altModelPaths || [])[0] || '')
      : (sm.longCtx512ModelPath || '');
    const factor = tier === '1m' ? 4.0 : 2.0;
    if (isNaN(nMaxLen) || nMaxLen > tierCap) maxLen = String(tierCap);
    env.FN_LONGCTX = '1';
    env.FN_1M_MODEL_PATH = tierPath;             // inner 脚本据此选模型目录
    env.FN_YARN_FACTOR = String(factor);         // inner 脚本据此钳制上限（缓存恒为 1048576）
    warnings.push(`已启用 ${tier === '1m' ? '1M' : '512K'} 长上下文（YaRN×${factor === 4.0 ? '4' : '2'}，上限 ${tierCap} token）：`
      + `模型切到 ${tierPath}；KV 池必须装得下 ${maxLen} token，并发会下降（max-num-seqs 从 ${env.FN_SEQS || b.maxNumSeqs} 起算）`);
  }
  env.FN_MAXLEN = maxLen;
  env.FN_GPUMEM = String(num(d.gpuMemUtil, b.gpuMemUtil));
  env.FN_SEQS = String(int(d.maxNumSeqs, b.maxNumSeqs));
  env.FN_BLOCK = String(int(d.blockSize, b.blockSize));
  env.FN_MBTOKENS = String(int(d.maxBatchedTokens, b.maxBatchedTokens));
  // [tp-presets 0928] 并行模式可切：弹窗/预设的 parallelMode='tp' + gpuCount=N →
  // TP=N×PP=1；'pp' 或字段缺失 → 生产基准 PP=base.pp||2（TP1），与旧行为一致。
  // TP2 内存账：PLE 表官方实现是 ETP-sharded（95.4GiB 跨 rank 分片，总量不变，
  // 每 rank 常驻 ~47.7GiB）；SimpleCPU --kv-offloading-size 按 world_size 均分
  // （96→48/rank，总量不变）⇒ 与 PP2 同账，251GiB 内存成立。
  // 关键护栏：PP=1 时上游按 VLLM_PP_LAYER_PARTITION 的列表长度==pp_size 校验，
  // 缺省 "26,22" 会让 TP 档启动直接 ValueError —— 用 FN_PP_PARTITION='none'
  // 通知 inner 不设该变量（inner 侧配套改动，见 flash-next-0300-inner.sh）。
  const _parMode = String(d.parallelMode || '').toLowerCase();
  const _gpus = int(d.gpuCount, 0);
  if (_parMode === 'tp' && _gpus >= 1 && !(sm.variantMinGpus && _gpus < sm.variantMinGpus)) {
    env.FN_TP = String(_gpus);
    env.FN_PP = '1';
    env.FN_PP_PARTITION = 'none';
    if (_gpus !== 2) warnings.push('TP×' + _gpus + '：请确认与在位 GPU 数匹配（10-05 换装后本机 3 卡；TP 档内存/PLE 分片账另行评估）');
  } else {
    env.FN_TP = '1';
    // [model-variant-1008] 变体可强制并行下限（Channel-INT8 权重 132GB：PP2 每段 66GB 超显存）
    let _pp = _parMode === 'pp' && _gpus >= 1 ? _gpus : (sm.variantPp || b.pp || 2);
    if (sm.variantMinGpus && _pp < sm.variantMinGpus) {
      warnings.push(sm.variantName + ' 权重合计超出 PP' + _pp + ' 的单卡显存（每段 ≥66 GiB > 64 GiB），启动必 OOM——已自动抬升为 PP=' + sm.variantMinGpus + '（请确认 ' + sm.variantMinGpus + ' 张卡空闲）');
      _pp = sm.variantMinGpus;
    }
    env.FN_PP = String(_pp);
  }
  env.FN_PREFIX_CACHE = String(String(d.prefixCaching) === '0' ? 0 : 1);
  env.FN_CHUNKED = String(String(d.chunkedPrefill) === '0' ? 0 : 1);
  env.FN_ASYNC = String(String(d.asyncScheduling) === '0' ? 0 : 1);
  const dtype = String(d.dtype || 'auto');
  if (dtype && dtype !== 'auto') env.FN_DTYPE = dtype;
  if (String(d.enforceEager) === '1') env.FN_ENFORCE_EAGER = '1';
  const sp = String(d.schedPolicy || '');
  if (sp && sp !== 'fcfs') env.FN_SCHED_POLICY = sp;
  if (d.seed) env.FN_SEED = String(d.seed);
  if (String(d.noLogRequests) === '1') env.FN_NOLOG = '1';
  if (String(d.disableAllReduce) === '1') env.FN_DISABLE_ALLREDUCE = '1';
  const mm = String(d.limitMm || '');
  if (mm && mm !== '999') env.FN_LIMIT_MM = mm;
  if (d.maxScheduledTokens) env.FN_MAX_SCHED_TOKENS = String(d.maxScheduledTokens);
  if (d.cpuOffloadGb) env.FN_CPU_OFFLOAD_GB = String(d.cpuOffloadGb);
  const kv = String(d.kvCacheQuant || 'auto');
  if (kv && kv !== 'auto' && kv !== 'bfloat16') env.FN_KV_DTYPE = kv;
  // [kvoff-toggle 09-22][kvoff-off 0929] CPU KV 二级缓存：inner 脚本按 FN_KVOFF(缺省关)/FN_KVOFF_BYTES 决定。
  // 弹窗显式传 '0'/'1'；字段缺失 → 关闭（与 inner 缺省 :-0 一致；0929 A/B 定案后翻转）。
  // [kvoff-start-0927] 三态：simple=官方 SimpleCPUOffloadConnector（生产定版档，
  // --kv-offloading-size GiB，实测容量≈36.3万token/GiB、命中80~97%无损）；
  // 1=经典 OffloadingConnector（0929 退役，仅 A/B 取证）；0/缺省=关。
  // simple 与经典互斥（inner 同开直接拒启），下发前显式清对侧变量。
  const kvMode = String(d.kvoff || '0');
  if (kvMode === 'simple') {
    env.FN_KVOFF = '0';
    // [model-variant-1008] 变体可钳制 CPU 二级缓存上限（Channel: 权重页缓存+PLE mlock 已 ~192GB）
    let _sg = Math.max(8, Math.min(200, int(d.kvoffGiB, 160)));
    if (sm.variantKvoffCap !== undefined && _sg > sm.variantKvoffCap) {
      warnings.push(sm.variantName + '：权重页缓存(~132 GiB) + PLE 表 mlock(~48.3 GiB) + 引擎常驻后，二级缓存 ' + _sg + ' GiB 会顶穿物理内存（历史 96GiB 档秒崩同族）——已自动降到 ' + sm.variantKvoffCap + ' GiB；如内存吃紧可选「关」');
      _sg = Math.max(8, sm.variantKvoffCap);
    }
    env.FN_SIMPLE_OFFLOAD = String(_sg);
  } else if (kvMode === '1') {
    env.FN_KVOFF = '1';
    const koG = int(d.kvoffGiB, 160);
    if (koG > 0) env.FN_KVOFF_BYTES = String(koG * 1073741824);
  } else {
    env.FN_KVOFF = '0';
  }
  // [ple-field 0923] PLE n-gram 表加载精度：inner 缺省=1（INT8 磁盘驻留 47.7GiB，可回收页缓存）；
  // 弹窗显式传 '0' → BF16 原生 95.4GiB（匿名堆不可回收）。旧快启预设无本字段 → 落 '1'，
  // 与 inner 缺省一致，行为不漂移。判据：日志 [FN-PLE-INT8] 行 / [FN-PLE-DISK] dtype=。
  env.FN_PLE_INT8 = String(d.pleInt8) === '0' ? '0' : '1';
  // [ple-mem 0923] PLE 表位置：只决定放内存(heap)/放硬盘(disk)，精度由 pleInt8 决定，
  // 两者正交、四种组合都成立（INT8+内存 = 引擎侧 VLLM_PLE_INT8_MEMORY 匿名堆 48.3GiB，
  // 由 inner 依据「精度+位置」推导后下发，不需要额外 FN_ 变量）。旧预设无本字段 → 'disk'。
  env.FN_PLE_LOC = String(d.pleLoc) === 'heap' ? 'heap' : 'disk';
  // [pleclamp-1005] 低内存机防线：heap=48.3GiB(INT8)/95.4GiB(BF16) 匿名堆不可回收，
  // MemTotal <100GiB 的机器（本机 32GB）必被 oom-killer 杀 → EngineDead → 看门狗循环。
  // 旧缓存页面/手选/旧预设传来的 heap 一律在此强制降级 disk（inner 侧第二道兜底同判据）。
  if (env.FN_PLE_LOC === 'heap' && require('os').totalmem() < 100 * 1024 * 1024 * 1024) {
    env.FN_PLE_LOC = 'disk';
    warnings.push('本机物理内存 <100GiB，「PLE 表位置=放内存」会被 OOM 杀（历史四连 oom-kill）——已自动降级为「放硬盘」（mmap 可回收页缓存）');
  }
  // 采样：全部等于基准 → 不传（= 生产无 --override-generation-config）
  const gen = {
    temperature: num(d.temperature, b.temperature), top_p: num(d.topP, b.topP),
    top_k: int(d.topK, b.topK), min_p: num(d.minP, b.minP),
    presence_penalty: num(d.presencePenalty, b.presencePenalty),
    repetition_penalty: num(d.repetitionPenalty, b.repetitionPenalty),
  };
  const genSame = gen.temperature === b.temperature && gen.top_p === b.topP && gen.top_k === b.topK &&
    gen.min_p === b.minP && gen.presence_penalty === b.presencePenalty &&
    gen.repetition_penalty === b.repetitionPenalty;
  if (!genSame) env.FN_GENCFG = JSON.stringify(gen);
  // [gen-default 0927] 思考模式下发 reasoning_effort。此前 thinking='1' 什么都不传 →
  // 弹窗「思考深度」对脚本化模型静默失效（旧栈 inner 也不消费 FN_CHATKWARGS，已一并补）。
  // Flash-Next 的 chat_template.jinja 只接受 xhigh / medium / low（模板缺省=xhigh，见该文件
  // 47~50 行），传别的值会在推理时 raise_exception ⇒ 白名单外一律不下发该键并给出警告。
  const effortIn = String(d.thinkingEffort || '').trim().toLowerCase();
  if (String(d.thinking) === '0') {
    env.FN_CHATKWARGS = JSON.stringify({ enable_thinking: false });
  } else {
    const kw = { enable_thinking: true, preserve_thinking: true };
    if (effortIn === '' || effortIn === 'xhigh') {
      kw.reasoning_effort = 'xhigh';   // 启动页默认；与模板缺省同值，显式写出保证「显示即真值」
    } else if (effortIn === 'medium' || effortIn === 'low') {
      kw.reasoning_effort = effortIn;
    } else {
      warnings.push(`思考深度「${effortIn}」不是 Flash-Next 模板支持的档位（仅 xhigh / medium / low），`
        + '本次不下发 reasoning_effort，引擎按模板缺省 xhigh 运行。');
    }
    env.FN_CHATKWARGS = JSON.stringify(kw);
  }
  // 投机：MTP 可用但本镜像 PP2 下不稳；DFlash/DSpark 无对应草稿 ckpt
  if (String(d.mtp) === '1') {
    const n = int(d.mtpTokens, b.mtpTokens || 6);
    env.FN_SPEC = JSON.stringify({ method: 'mtp', num_speculative_tokens: n, use_local_argmax_reduction: false });
    // 09-19：W4A16 档的 MTP 已实测稳定（与 NVFP4 档相反），警告按档位区分，避免误导
    warnings.push(sm.key === 'qwen3.8-flash-next-w4a16'
      ? `已启用 MTP(x${n})：W4A16 档实测稳定（09-19：接受长度 2.84、decode 111 tok/s、输出无乱码）。注意草稿档位受 QSA ring capacity 整除约束——block-size 1616 下合法档为 1~4 与 9~12，5~8 会启动即崩。`
      : `已启用 MTP(x${n})：本镜像 PP2 下 MTP 有连续长请求 device assert 残余 bug，不稳定请改回「无投机」`);
  } else if (String(d.dflash) === '1' || String(d.dspark) === '1') {
    warnings.push('Flash-Next-NVFP4 无 DFlash/DSpark 草稿 ckpt，已按「无投机」启动');
  } else {
    // 未选投机：显式下发 none，避免脚本侧缺省值（W4A16 缺省=MTP4）被误用
    env.FN_SPEC = 'none';
  }
  // 附加环境变量 / 附加启动参数（危险项过滤）
  const bannedEnv = (sm.bannedEnv || []).map(x => x.toUpperCase());
  const keptEnv = [];
  String(d.vllmExtraEnv || '').split('\n').map(x => x.trim()).filter(x => x && !x.startsWith('#')).forEach(line => {
    const key = line.split('=')[0].trim().toUpperCase();
    if (bannedEnv.some(x => key.includes(x))) { warnings.push('已剔除危险环境变量：' + line.split('=')[0].trim()); return; }
    keptEnv.push(line);
  });
  if (keptEnv.length) env.FN_EXTRA_ENV = keptEnv.join('\n');
  const extra = String(d.vllmExtraArgs || '').trim();
  if (extra) {
    const toks = extra.split(/\s+/);
    const banned = sm.bannedArgs || [];
    const kept = [];
    for (let i = 0; i < toks.length; i++) {
      const t = toks[i];
      if (banned.some(x => t === x || t.startsWith(x + '='))) { warnings.push('已剔除危险参数：' + t); i++; continue; }
      kept.push(t);
    }
    if (kept.length) env.FN_EXTRA_ARGS = kept.join(' ');
  }
  const summary = [
    'serve ' + (env.FN_MODEL_PATH || sm.modelPath), '--served-model-name ' + served, '--port ' + port,
    '--max-model-len ' + maxLen, '--gpu-memory-utilization ' + env.FN_GPUMEM,
    '--max-num-seqs ' + env.FN_SEQS, '--max-num-batched-tokens ' + env.FN_MBTOKENS,
    '--block-size ' + env.FN_BLOCK, 'TP' + env.FN_TP + '×PP' + env.FN_PP,
    env.FN_SPEC ? ('spec=' + env.FN_SPEC) : '无投机',
    env.FN_GENCFG ? ('gen=' + env.FN_GENCFG) : '采样=模型默认',
    env.FN_LONGCTX === '1' ? '长上下文=YaRN×4/1M' : '',
  ].filter(Boolean).join(' ');
  return { env, port, served, warnings, summary };
}
// ====== SGLang 脚本栈（sglang-18420 脚本对）参数下发 ======
// [sglang-adapt-1003] 背景：18420 现行生产切到 SGLang 后，启动 wrapper
// start-flash-next-sglang.sh 只把 SG_* 环境变量落盘（compgen 扫 ^SG_），inner 也只消费
// SG_*。若沿用 vLLM 栈的 scriptModelLaunchPlan 下发 FN_*，弹窗/快启的所有参数一个都进
// 不了引擎（= 09-26「参数三跳」铁律的复发形态：静默失效、inner 缺省值冒充实跑真值）。
// 因此本栈单独建 plan：语义与 vLLM plan 同字段，落到 SG_*；SGLang 不支持的项显式警告，
// 不静默丢弃。sglang-18420/ACTIVE 哨兵在位时，两个启动入口（弹窗/快启）自动走这里。
function scriptModelLaunchPlanSglang(sm, d) {
  sm = scriptModelVariantResolve(sm, d);  // [model-variant-1008] sglang 栈同样按目录识别模型路径
  const b = sm.base || {};
  const warnings = [];
  const env = {};
  const num = (v, dflt) => { const n = parseFloat(v); return isNaN(n) ? dflt : n; };
  const int = (v, dflt) => { const n = parseInt(v, 10); return isNaN(n) ? dflt : n; };
  const port = int(d.port, sm.port);
  const served = String(d.servedName || sm.served).trim() || sm.served;
  env.SG_PORT = String(port);
  env.SG_SERVED = served;
  // 上下文档位：SGLang 无运行时 YaRN 缩放参数，1M/512K 靠加载已含 rope 缩放的 config 副本
  //（与 vLLM 栈同一批 models-1m/models-512k 目录，选择逻辑与 FN 版一致）。
  let maxLen = String(d.ctxLen || '').trim() || String(d.maxModelLen || '').trim();
  if (!maxLen || maxLen === 'auto') maxLen = String(b.maxModelLen);
  const cap1m = int(sm.maxModelLenLong, 0);
  const cap512 = int(sm.maxModelLen512, 0);
  const baseCap = int(b.maxModelLen, 262144);
  const nMaxLen = int(maxLen, NaN);
  let tier = '';
  const lcRaw = String(d.longCtx || '').trim().toLowerCase();
  if (lcRaw === '1' || lcRaw === '1m') tier = '1m';
  else if (lcRaw === '512k' || lcRaw === '512') tier = '512k';
  else if (cap512 > 0 && !isNaN(nMaxLen) && nMaxLen > baseCap && nMaxLen <= cap512) tier = '512k';
  else if (cap1m > baseCap && !isNaN(nMaxLen) && nMaxLen > baseCap) tier = '1m';
  if (tier === '1m' && cap1m <= 0) tier = cap512 > 0 ? '512k' : '';
  if (tier === '512k' && cap512 <= 0) tier = cap1m > 0 ? '1m' : '';
  let modelPath = sm.modelPath;
  if (tier === '1m') { maxLen = String(cap1m); modelPath = sm.longCtxModelPath || sm.modelPath; }
  else if (tier === '512k') { maxLen = String(cap512); modelPath = sm.longCtx512ModelPath || sm.modelPath; }
  else if (!isNaN(nMaxLen) && nMaxLen > baseCap) {
    maxLen = String(baseCap);
    warnings.push(`原生档上限 ${baseCap} token；SGLang 栈更长上下文需选 1M/512K YaRN 副本，本次已钳到 ${baseCap}。`);
  }
  env.SG_MODEL = modelPath;
  env.SG_CTX = String(maxLen);
  // 并行拓扑：默认 PP2（内层缺省 TP1×PP2、层切 26,22）；弹窗改 TP 时下发 TP/PP
  const gpuN = Math.max(1, int(d.gpuCount, b.pp || 2));
  const gpuId = Math.max(0, int(d.gpuId, 0));
  env.SG_CVD = Array.from({ length: gpuN }, (_, i) => gpuId + i).join(',');
  if (String(d.parallelMode) === 'tp' && gpuN > 1) { env.SG_TP = String(gpuN); env.SG_PP = '1'; }
  else { env.SG_TP = '1'; env.SG_PP = String(gpuN); }
  // 显存比例：sglang --mem-fraction-static。投机档上限 0.88（decode 期动态加载 GDN/
  // EAGLE 内核需余量，超了会 OOM 打坏 CUDA 上下文——09-30 弹窗同款保护）、无投机 0.92。
  const specOn = String(d.mtp) === '1';
  if (String(d.dflash) === '1' || String(d.dspark) === '1') {
    warnings.push('SGLang 脚本栈（18420）投机仅支持内置 NEXTN（MTP），DFlash/DSpark 外部草稿未接线，本次按投机开关处理。');
  }
  env.SG_SPEC = specOn ? 'nextn' : 'none';
  const memCap = specOn ? 0.88 : 0.92;
  const memRaw = num(d.gpuMemUtil, specOn ? Math.min(b.gpuMemUtil || 0.88, 0.88) : 0.88);
  const memFrac = Math.min(memRaw, memCap);
  if (memRaw > memCap) warnings.push(`显存比例 ${memRaw} 已钳到 ${memCap}（SGLang ${specOn ? '投机' : ''}档安全上限，防 decode 期动态内核加载 OOM）。`);
  env.SG_MEMFRAC = String(memFrac);
  env.SG_SEQS = String(int(d.maxSeqs, b.maxNumSeqs));
  env.SG_CHUNKED_PREFILL = String(int(d.maxBatchedTokens, b.maxBatchedTokens));
  // 采样参数：无条件下发。inner 内置缺省与弹窗基准(base)不同值，省略会「弹窗显示 ≠ 引擎真值」。
  const gen = {
    temperature: num(d.temperature, b.temperature), top_p: num(d.topP, b.topP),
    top_k: int(d.topK, b.topK), min_p: num(d.minP, b.minP),
    presence_penalty: num(d.presencePenalty, b.presencePenalty),
    repetition_penalty: num(d.repetitionPenalty, b.repetitionPenalty),
  };
  env.SG_GENCFG = JSON.stringify(gen);
  // 思考模式（同 FN_CHATKWARGS 白名单：模板只认 xhigh/medium/low）
  const effortIn = String(d.thinkingEffort || '').trim().toLowerCase();
  if (String(d.thinking) === '0') {
    env.SG_CT_KWARGS = JSON.stringify({ enable_thinking: false });
  } else {
    const kw = { enable_thinking: true, preserve_thinking: true };
    if (effortIn === '' || effortIn === 'xhigh') kw.reasoning_effort = 'xhigh';
    else if (effortIn === 'medium' || effortIn === 'low') kw.reasoning_effort = effortIn;
    else warnings.push(`思考深度「${effortIn}」不是 Flash-Next 模板支持的档位（仅 xhigh / medium / low），本次不下发 reasoning_effort，引擎按模板缺省 xhigh 运行。`);
    env.SG_CT_KWARGS = JSON.stringify(kw);
  }
  // vLLM 专属项：SGLang 栈无对应实现 → 显式警告（不静默丢）
  const vOnly = [];
  if (String(d.blockSize || '') !== '' && int(d.blockSize, 0) !== 0) vOnly.push('block-size（SGLang page_size 由内核自动选）');
  if (String(d.kvoff || '') !== '' && String(d.kvoff) !== '0') vOnly.push('CPU 二级缓存（SGLang 为显存 radix cache，无对应档）');
  if (String(d.pleInt8 || '') !== '' || String(d.pleLoc || '') !== '') vOnly.push('PLE 精度/位置（本栈固定 BF16 锁页）');
  if (String(d.enforceEager || '') === '1') vOnly.push('enforce-eager');
  if (String(d.kvCacheQuant || '') !== '' && String(d.kvCacheQuant) !== 'auto') vOnly.push('KV 量化');
  if (String(d.dtype || '') !== '' && String(d.dtype) !== 'auto' && String(d.dtype) !== 'bfloat16') vOnly.push('dtype（本栈固定 bfloat16）');
  if (String(d.pdMode || '') === '1') vOnly.push('PD 分离');
  if (vOnly.length) warnings.push('以下参数为 vLLM 栈专属，SGLang 脚本栈忽略：' + vOnly.join('、') + '。');
  // 附加环境变量：仅 SG_* 键能进 wrapper（sudo env_reset，落盘只扫 SG_*）；其余剔除并警告
  const keptEnv = [];
  String(d.vllmExtraEnv || '').split('\n').map(x => x.trim()).filter(x => x && !x.startsWith('#')).forEach(line => {
    const key = line.split('=')[0].trim().toUpperCase();
    if (/^SG_[A-Z0-9_]+$/.test(key)) keptEnv.push(line);
    else if (key) warnings.push('附加环境变量「' + key + '」非 SG_* 键：SGLang 脚本栈经 sudo env_reset 只透传 SG_*，已忽略该行。');
  });
  keptEnv.forEach(line => { const i = line.indexOf('='); if (i > 0) env[line.slice(0, i).trim()] = line.slice(i + 1).trim(); });
  const extra = String(d.vllmExtraArgs || '').trim() + '\n' + String(d.sglangExtraArgs || '').trim();
  if (extra.trim()) warnings.push('SGLang 脚本栈暂不透传附加命令行参数（inner 命令行固定）；如需请修改 sglang-18420/sglang-inner.sh。填写内容未生效：\n' + extra.trim());
  const summary = [
    'sglang serve ' + modelPath, '--served-model-name ' + served, '--port ' + port,
    '--context-length ' + maxLen, '--mem-fraction-static ' + env.SG_MEMFRAC,
    '--max-running-requests ' + env.SG_SEQS, '--chunked-prefill-size ' + env.SG_CHUNKED_PREFILL,
    'TP' + env.SG_TP + '×PP' + env.SG_PP,
    'spec=' + (specOn ? 'NEXTN(MTP)' : 'none'),
    'gen=' + env.SG_GENCFG,
    tier === '1m' ? '长上下文=YaRN×4/1M副本' : (tier === '512k' ? '长上下文=YaRN×2/512K副本' : ''),
  ].filter(Boolean).join(' ');
  return { env, port, served, warnings, summary, runtime: 'sglang', gpuCount: gpuN };
}
// 启动入口选择：ACTIVE 哨兵 + 脚本对在位 → sglang plan，否则 vLLM plan（两栈共用同一弹窗字段）。
function scriptPlanFor(sm, d) {
  try {
    if (sm && sm.scriptSglang && sglangActive() && fs.existsSync(sm.scriptSglang)) return scriptModelLaunchPlanSglang(sm, d);
  } catch (e) {}
  return scriptModelLaunchPlan(sm, d);
}
// [sglang-adapt-1003] 脚本模型的启动 wrapper 输出日志（≠ inner 引擎日志 logSglang，后者由
// wrapper 自己重定向并有祖先链判据）：sglang 栈在位时写 sglang-launch.log 语义的文件，
// 避免 8889 把 sglang 启动行追进 vLLM 老日志里造成「日志面板显示陈旧文件」误判。
function scriptLaunchLog(sm) {
  try {
    if (sm && sm.scriptSglang && sglangActive() && fs.existsSync(sm.scriptSglang)) return sm.logSglang || sm.log;
  } catch (e) {}
  return sm.log;
}
function vllmPortForModel(model) {
  if (!model) return null;
  const name = resolveModelAlias(model);
  // 显式路由（多模型映射）优先；否则默认后端动态跟随 config.vllmPort
  return (VLLM_MODEL_PORTS[name] !== undefined) ? VLLM_MODEL_PORTS[name] : config.vllmPort;
}

// ====== PD (prefill/decode) 两步转发支持 ======
// PD 模型 = 两个 vLLM 实例分工：prefill（kv_producer）只算 KV 缓存，decode（kv_consumer）
// 拿到 KV 后直接生成。客户端无感：代理自动拆两步——
//   1) 请求转发给 prefill 实例（强制非流式，注入 return_token_ids + do_remote_decode），
//      响应带回 prompt_token_ids 与 kv_transfer_params（remote_block_ids/engine_id/request_id 等）；
//   2) 把这两个字段注入原请求转发给 decode 实例（按客户端意愿流式/非流式），响应原样透传。
// 环境变量 PD_MODEL_PORTS（JSON）：{"模型名":{"prefill":端口,"decode":端口}}；
// 需要 vLLM 以 --kv-transfer-config 跑成 kv_producer/kv_consumer 一对实例。
const PD_MODEL_PORTS = Object.assign(
  // 08-28 修正：当前 8000 为单实例（非 PD）。旧硬编码 {prefill:8010, decode:8011}
  // 会让 qwen3.8-27b-fp8 请求在控制台重启后（动态同步探测不到 kv_producer/consumer
  // 配对）回退指向不存在的 8010/8011 → ECONNREFUSED 502。真正起 PD 实例时改用
  // 启动弹窗注册或 PD_MODEL_PORTS 环境变量注入。
  {},
  (() => { try { return JSON.parse(process.env.PD_MODEL_PORTS || '{}'); } catch (e) { return {}; } })()
);
function pdPortsForModel(model) {
  if (!model) return null;
  const p = PD_MODEL_PORTS[model];
  return (p && p.prefill && p.decode) ? { prefill: p.prefill, decode: p.decode } : null;
}

// ====== PD 路由动态同步 ======
// 控制台重启后，启动弹窗注册到 PD_MODEL_PORTS 的运行时条目会丢失，而实例进程（setsid
// 独立）仍在跑——路由表会退回硬编码默认值导致代理打错端口（实测 08-27：实例在 8001/8002
// 而路由指 8010 → ECONNREFUSED）。这里从运行中的 vllm serve 进程 cmdline 探测
// kv_producer/kv_consumer 配对，按 served-model-name 重建路由（覆盖硬编码）。
function syncPdModelPortsFromProcs() {
  try {
    const { execSync } = require('child_process');
    const out = execSync('pgrep -f "vllm serve" 2>/dev/null || true', { encoding: 'utf8', timeout: 3000 }).trim();
    if (!out) return;
    const producers = {}; // servedName -> prefill port
    const consumers = {}; // servedName -> decode port
    for (const pid of out.split('\n').filter(Boolean)) {
      let cmd;
      try { cmd = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8').split('\0').join(' '); } catch (e) { continue; }
      if (/(^|\s)(ssh|bash|sh|expect)(\s|$)/.test(cmd)) continue;
      if (/\b(pgrep|grep|ps)\b/.test(cmd)) continue;
      const portM = cmd.match(/--port\s+(\d+)/);
      const nameM = cmd.match(/--served-model-name\s+(\S+)/);
      if (!portM || !nameM) continue;
      if (cmd.includes('"kv_role":"kv_producer"')) producers[nameM[1]] = parseInt(portM[1]);
      else if (cmd.includes('"kv_role":"kv_consumer"')) consumers[nameM[1]] = parseInt(portM[1]);
    }
    for (const name of new Set([...Object.keys(producers), ...Object.keys(consumers)])) {
      if (producers[name] && consumers[name]) {
        const cur = PD_MODEL_PORTS[name];
        if (!cur || cur.prefill !== producers[name] || cur.decode !== consumers[name]) {
          PD_MODEL_PORTS[name] = { prefill: producers[name], decode: consumers[name] };
          console.log(`[pd-sync] ${name} -> prefill ${producers[name]} / decode ${consumers[name]}`);
        }
      }
    }
  } catch (e) { /* 探测失败保持现状 */ }
}
// 启动时同步一次 + 每 5s 跟随实例变化（新增/停止 PD 对）
try { syncPdModelPortsFromProcs(); } catch (e) {}
setInterval(syncPdModelPortsFromProcs, 5000).unref();

// SGLang 实例 模型名→端口 动态同步（同 PD 同步思路）：
// 启动弹窗注册只存内存，控制台重启即丢；脚本手动起的实例更是从未注册过。
// 这里扫 sglang.launch_server 进程的 --port/--served-model-name 注册进
// VLLM_MODEL_PORTS，8889 代理按模型名正确路由（08-30 事故：Qwen3.6-27B 未注册
// 回落默认端口 8001 → vLLM 报 404 model does not exist）。
// 只增改不删：探测瞬时失败（进程列表读空）不会清掉已有映射。
function syncSglangModelPortsFromProcs() {
  try {
    for (const inst of listSglangInstances()) {
      let name = '';
      try {
        const cmd = fs.readFileSync(`/proc/${inst.pid}/cmdline`, 'utf8').split('\0').join(' ');
        const nameM = cmd.match(/--served-model-name\s+(\S+)/);
        const pathM = cmd.match(/--model-path\s+(\S+)/);
        // SGLang 未指定 --served-model-name 时默认对外名 = 模型路径
        name = nameM ? nameM[1] : (pathM ? pathM[1] : '');
      } catch (e) { continue; }
      if (!name) continue;
      if (VLLM_MODEL_PORTS[name] !== inst.port) {
        VLLM_MODEL_PORTS[name] = inst.port;
        console.log(`[sglang-sync] ${name} -> ${inst.port}`);
      }
    }
  } catch (e) { /* 探测失败保持现状 */ }
}
try { syncSglangModelPortsFromProcs(); } catch (e) {}
setInterval(syncSglangModelPortsFromProcs, 5000).unref();

// vLLM 实例 模型名→端口 动态同步（09-03，同 sglang-sync 思路）：
// 多实例时（8000/8001 各 serve 不同 served-model-name），路由表此前只有启动弹窗/
// 环境变量的显式注册，未注册的模型回落 config.vllmPort（主实例端口）→ 次实例的
// 模型名请求打到主实例 404「The model ... does not exist」
// （09-03 事故：qwen3.8-27b-1 回落 8000 → 404）。
// 这里从运行中 vllm serve 进程 cmdline 注册 --served-model-name（缺省=模型路径）→ --port。
// 规则：只增改不删（探测瞬时失败不清已有映射）；PD 实例命中 pdPortsForModel 时走
// 两步转发优先（见 CATCH-ALL），此表仅兜底；与 SGLang 实例同名时让位（保持既有
// 行为：SGLang 动态注册优先）。
function syncVllmModelPortsFromProcs() {
  try {
    // 运行中 SGLang 实例的对外名集合（同名让位判断，口径与 sglang-sync 一致）
    const sgNames = new Set();
    for (const inst of listSglangInstances()) {
      try {
        const cmd = fs.readFileSync(`/proc/${inst.pid}/cmdline`, 'utf8').split('\0').join(' ');
        const nameM = cmd.match(/--served-model-name\s+(\S+)/);
        const pathM = cmd.match(/--model-path\s+(\S+)/);
        const n = nameM ? nameM[1] : (pathM ? pathM[1] : '');
        if (n) sgNames.add(n);
      } catch (e) {}
    }
    const insts = listVllmInstances().slice().sort((a, b) => a.port - b.port);
    for (const inst of insts) {
      const name = inst.servedName || inst.modelPath;
      if (!name) continue;
      if (sgNames.has(name)) continue; // SGLang 实例正 serve 同名，路由让给它
      if (VLLM_MODEL_PORTS[name] !== inst.port) {
        VLLM_MODEL_PORTS[name] = inst.port;
        console.log(`[vllm-sync] ${name} -> ${inst.port}`);
      }
    }
  } catch (e) { /* 探测失败保持现状 */ }
}
// 启动时同步一次 + 每 5s 跟随实例变化（新增/停止）
try { syncVllmModelPortsFromProcs(); } catch (e) {}
setInterval(syncVllmModelPortsFromProcs, 5000).unref();

// ====== SGLang 支持 ======
// sglang 环境（main 快照，含 DFLASH / DFlash2DraftModel）与 DFlash2 草稿模型路径
const SGLANG_VENV = '/home/ll/sglang-env';
const SGLANG_DRAFT_PATH = '/home/ll/models/qwen3.8-27b-dflash2';
const SGLANG_DSPARK_PATH = '/home/ll/models/qwen3.8-27b-dspark';
const SGLANG_LOG_PATH = path.join(__dirname, 'sglang.log');

// DSpark 草稿 ckpt 的 dspark_block_size（gamma）训练时定死，运行时不可改：
// vLLM 要求 num_speculative_tokens ≥ gamma（小了乱码/硬报错），SGLang 要求
// --speculative-num-draft-tokens 严格等于 gamma+1（verify 窗口）。两引擎的 DSpark
// 分支一律以 ckpt config 为准，UI「投机 token 数」输入不再参与。读取失败回退 7。
function readDsparkGamma(draftPath) {
  try {
    const cfg = JSON.parse(fs.readFileSync(path.join(draftPath, 'config.json'), 'utf8'));
    const g = parseInt(cfg && cfg.dspark_block_size, 10);
    return (Number.isFinite(g) && g > 0) ? g : null;
  } catch (e) {
    return null;
  }
}

// ====== GPU 占用登记表 ======
// 防止 vLLM 与 SGLang 实例占用同一 GPU 造成冲突。
// Map<port, {port, gpuId, gpuCount, runtime, model, startedAt}>
if (!global.__GPU_INSTANCES) global.__GPU_INSTANCES = new Map();

// 按 PID 识别运行时：'sglang' / 'vllm' / null
function detectRuntimeForPid(pid) {
  if (!pid) return null;
  try {
    const cmd = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8').split('\0').join(' ');
    if (cmd.includes('sglang.launch_server') || cmd.includes('python -m sglang')) return 'sglang';
    if (cmd.includes('vllm serve') || cmd.includes('vllm.entrypoints') || cmd.includes('python -m vllm')) return 'vllm';
  } catch (e) {}
  return null;
}

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
  '.txt': 'text/plain',
};

// ====== 1-second token ticker（按端口）======
// Samples each backend port's /metrics every second INDEPENDENTLY of client
// polling, so the dashboard can show true "tokens produced in the last 1
// second" per request.
// 多卡多实例：每个端口一个独立 ticker（global.__tokTickers[port]）；
// global.__tokTicker 始终指向主后端（config.vllmPort）的 ticker 别名，
// 既有单实例逻辑（last_second / 计费 / 预填充聚合）引用它不变。
if (!global.__tokTickers) global.__tokTickers = new Map();
function getTicker(port) {
  let tk = global.__tokTickers.get(port);
  if (!tk) {
    tk = { lastTotal: null, lastPromptTotal: null, lastTime: null, busy: false,
      lastSecond: { tokens: 0, speed: 0, running: 0, at: 0 } };
    global.__tokTickers.set(port, tk);
  }
  return tk;
}
if (!global.__tokTicker) global.__tokTicker = getTicker(config.vllmPort);

// 每 GPU 累计缓存命中率：读各实例 ticker 的累计计数器
// （vLLM: local_cache_hit/(hit+local_compute)；SGLang: cached_tokens_total/(cached+uncached_sum)）。
// 口径=自实例启动累计，与主卡「重置后」口径独立。
// 返回 [{port, gpu, model, cached, uncached, rate}]（rate 为百分数或 null）。
function perPortCacheStats() {
  const out = [];
  try {
    const seen = new Set();
    const all = [...listVllmInstances(), ...listSglangInstances()];
    for (const inst of all) {
      if (seen.has(inst.port)) continue;
      seen.add(inst.port);
      const tk = global.__tokTickers.get(inst.port);
      if (!tk || tk.lastCacheCached === undefined || tk.lastCacheUncached === undefined) continue;
      const cached = Math.max(0, Math.round(tk.lastCacheCached));
      const uncached = Math.max(0, Math.round(tk.lastCacheUncached));
      const total = cached + uncached;
      out.push({
        port: inst.port,
        gpu: inst.gpu,
        gpus: inst.gpus || (inst.gpu != null ? [inst.gpu] : []),
        model: inst.model,
        cached, uncached,
        rate: total > 0 ? parseFloat((cached / total * 100).toFixed(1)) : null,
      });
    }
  } catch (e) {}
  return out;
}

// [kvoff-display 09-22] 从实例 cmdline 提取 CPU KV 二级缓存容量（--kv-transfer-config 的
// cpu_bytes_to_use，单位字节）。chroot 内进程属 root，但 /proc/<pid>/cmdline 全局可读。
// 60s 缓存（实例重启后 cmdline 变化会在下一周期反映）。
function kvOffloadCapacityBytes(pid) {
  try {
    const now = Date.now();
    if (!global.__kvOffCap) global.__kvOffCap = new Map();
    const e = global.__kvOffCap.get(pid);
    if (e && now - e.t < 60000) return e.b;
    const cmd = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8').replace(/\0/g, ' ');
    const mm = cmd.match(/cpu_bytes_to_use["'\s:\\]+(\d+)/);
    let b = mm ? parseInt(mm[1], 10) : 0;
    // [kvoff-simple 0927] 官方 SimpleCPUOffloadConnector：容量走 --kv-offloading-size <GiB>
    if (!b) { const ms = cmd.match(/--kv-offloading-size\s+(\d+(?:\.\d+)?)/); if (ms) b = Math.round(parseFloat(ms[1]) * 1073741824); }
    global.__kvOffCap.set(pid, { t: now, b });
    return b;
  } catch (err) { return 0; }
}
// [kvoff-mem 09-27] CPU KV 二级缓存的「物理驻留」真值：本机 /dev/shm 只被 vLLM
// offload 使用（kvoff 共享区 mmap 建在 tmpfs 上，实测 68.9GB ≈ 配置 64GiB），
// statfs 直读 tmpfs 已用字节——无需 root、不 spawn 子进程、口径即取即真。
// 注意：这是整机 tmpfs 用量，多实例共享同一 /dev/shm 时不可按实例切分。
function kvOffloadShmUsage() {
  try {
    const st = fs.statfsSync('/dev/shm');
    const unit = st.bsize;
    const total = st.blocks * unit;
    const used = Math.max(0, (st.blocks - st.bavail) * unit);
    return {
      used_gb: +(used / 1073741824).toFixed(2),
      total_gb: +(total / 1073741824).toFixed(1),
      used_pct: total > 0 ? +(used / total * 100).toFixed(1) : 0,
    };
  } catch (e) { return null; }
}
// 汇总各 vLLM 实例的 CPU KV 二级缓存状态（数据来自 ticker 每秒 /metrics 采样）。
// [kvoff-live 09-27] 常驻口径：**每个受管实例都出条目**（含未启用项），前端才能把
// 「二级缓存·CPU」卡常驻显示；此前未启用 / 引擎刚起还没采到指标时条目为空 →
// 前端 kv_offload_ports=[] → 整卡 display:none 消失。
// 返回 [{port, gpu, gpus, model, enabled, metric_kind, capacity_gb, fill_gb, fill_pct,
//        write_pct, read_pct, stored_gb, loaded_gb, ext_tokens, load_count, store_count,
//        queries, hits, hit_rate}]。
function kvOffloadPortsInfo() {
  const out = [];
  try {
    const gib = 1073741824;
    for (const inst of listVllmInstances()) {
      const tk = global.__tokTickers && global.__tokTickers.get(inst.port);
      const kv = tk && tk.kvOffload;
      const capBytes = kvOffloadCapacityBytes(inst.pid);
      out.push({
        port: inst.port,
        gpu: inst.gpu,
        gpus: inst.gpus || (inst.gpu != null ? [inst.gpu] : []),
        model: inst.servedName || inst.modelPath || '',
        // 已启用判据：/metrics 已出现 kv_offload 指标，或 cmdline 已配 cpu_bytes_to_use
        // （引擎加载窗口内指标尚未暴露，但配置已定 → 前端显示「已启用 · 等待指标」）
        enabled: !!(kv || capBytes),
        // [kvoff-0300 0930] fill=旧栈真实驻留占比；usage=新栈「被在飞传输钉住」占比
        metric_kind: kv ? (kv.metricKind || 'fill') : null,
        capacity_gb: capBytes ? +(capBytes / gib).toFixed(2) : null,
        // usage 口径换算成 GiB 会冒充驻留量 → 只有 fill 口径才给 fill_gb
        fill_gb: (capBytes && kv && kv.metricKind !== 'usage') ? +(kv.fillPerc * capBytes / gib).toFixed(2) : null,
        fill_pct: kv ? +(kv.fillPerc * 100).toFixed(1) : 0,
        // 新栈另有写/读两侧的在飞占比（write=GPU→CPU 传输中，read=CPU→GPU 回载中）
        write_pct: (kv && kv.writePerc != null) ? +(kv.writePerc * 100).toFixed(1) : null,
        read_pct: (kv && kv.readPerc != null) ? +(kv.readPerc * 100).toFixed(1) : null,
        stored_gb: kv ? +(kv.storedBytes / gib).toFixed(2) : 0,
        loaded_gb: kv ? +(kv.loadedBytes / gib).toFixed(2) : 0,
        // [kvoff-hit 09-22] 命中真值（自实例启动累计）：回载 token / 回载与写入次数
        ext_tokens: kv ? Math.round(kv.extTokens || 0) : 0,
        load_count: kv ? Math.round(kv.loadCount || 0) : 0,
        store_count: kv ? Math.round(kv.storeCount || 0) : 0,
        // [kvoff-live 09-27] OffloadingConnector 二级缓存查询/命中（external_prefix_cache_*）：
        // 与 ext_tokens 互补——前者看「查了多少次、命中几次」，后者看「免重算 token 量」。
        queries: kv ? Math.round(kv.queries || 0) : 0,
        hits: kv ? Math.round(kv.hits || 0) : 0,
        hit_rate: (kv && kv.queries > 0) ? +(kv.hits / kv.queries * 100).toFixed(1) : null,
      });
    }
  } catch (e) {}
  return out;
}

// [kv-detail-1003] vLLM 标签页「CPU KV 二级缓存」详细数据：按端口聚合
//   ① 配置真值（引擎 cmdline / cache_config_info 标签）：connector 类型、容量、block、GPU 池
//   ② 驻留真值（/dev/shm statfs）：tmpfs 已用/上限（SimpleCPU 与经典档都落 tmpfs）
//   ③ 指标真值（ticker.kvOffload）：查询/命中/命中率、累计存/取字节与次数、回载 token
//   ④ 速率（本函数内 6s 滑窗差值）：查询/命中 token 每秒、字节存取每秒
//   ⑤ 系统内存压力（readMemInfo）：整机 used/avail，判断二级缓存是否挤压系统内存
// 只读、不 spawn、无特权需求；单端口 ?port=，缺省返回全部 vLLM 实例。
// 返回 Promise<rows[]>（GPU 池要读 /metrics，走 fetchMetricsCached 400ms 缓存+单飞，
// 与 ticker 每秒采样共享，不给上游加压）。
function kvDetailInfo(portFilter) {
  const gib = 1073741824;
  const out = [];
  const jobs = [];
  try {
    for (const inst of listVllmInstances()) {
      if (portFilter && inst.port !== portFilter) continue;
      const rec = {
        port: inst.port,
        gpu: inst.gpu,
        gpus: inst.gpus || (inst.gpu != null ? [inst.gpu] : []),
        model: inst.servedName || inst.modelPath || '',
        pid: inst.pid || null,
        gpu_pool: null,
      };
      // GPU 池（cache_config_info 标签，同 buildKvCacheInfo 口径）
      jobs.push(fetchMetricsCached(`http://${config.vllmHost}:${inst.port}/metrics`, 400, 2500).then((data) => {
        try {
          const m = parseMetrics(data);
          for (const k of Object.keys(m)) {
            if (k.indexOf('vllm:cache_config_info|') !== 0) continue;
            const lb = JSON.parse(k.substring(k.indexOf('|') + 1));
            rec.gpu_pool = {
              size_tokens: parseInt(lb.kv_cache_size_tokens) || null,
              num_gpu_blocks: parseInt(lb.num_gpu_blocks) || null,
              block_size: parseInt(lb.block_size) || null,
              kv_cache_dtype: lb.cache_dtype || null,
              gpu_memory_utilization: parseFloat(lb.gpu_memory_utilization) || null,
            };
            break;
          }
        } catch (e) {}
      }, () => {}));
      const tk = global.__tokTickers && global.__tokTickers.get(inst.port);
      const kv = tk && tk.kvOffload;
      const capBytes = kvOffloadCapacityBytes(inst.pid);
      // connector 类型：cmdline 有 --kv-offloading-size → SimpleCPU；有 cpu_bytes_to_use → 经典 OffloadingConnector
      let connector = null;
      try {
        const cmd = fs.readFileSync(`/proc/${inst.pid}/cmdline`, 'utf8').replace(/\0/g, ' ');
        if (/--kv-offloading-size/.test(cmd)) connector = 'SimpleCPUOffloadConnector';
        else if (/cpu_bytes_to_use/.test(cmd)) connector = 'OffloadingConnector';
      } catch (e) {}
      if (!connector && kv) connector = (kv.metricKind === 'simple') ? 'SimpleCPUOffloadConnector' : 'OffloadingConnector';
      // 驻留真值：/dev/shm（整机 tmpfs，单实例口径）
      const shm = kvOffloadShmUsage();
      // 速率：6s 滑窗差值（查询/命中 token、存/取字节）
      let rates = null;
      if (kv) {
        if (!global.__kvDetailRates) global.__kvDetailRates = new Map();
        const now = Date.now();
        let buf = global.__kvDetailRates.get(inst.port);
        if (!buf) { buf = []; global.__kvDetailRates.set(inst.port, buf); }
        buf.push({ t: now, q: kv.queries || 0, h: kv.hits || 0, sb: kv.storedBytes || 0, lb: kv.loadedBytes || 0, et: kv.extTokens || 0 });
        while (buf.length > 2 && now - buf[0].t > 6000) buf.shift();
        if (buf.length >= 2) {
          const a = buf[0], b = buf[buf.length - 1];
          const dt = (b.t - a.t) / 1000;
          if (dt > 1) {
            rates = {
              window_s: +dt.toFixed(1),
              queries_per_s: Math.max(0, (b.q - a.q) / dt),
              hits_per_s: Math.max(0, (b.h - a.h) / dt),
              store_bytes_per_s: Math.max(0, (b.sb - a.sb) / dt),
              load_bytes_per_s: Math.max(0, (b.lb - a.lb) / dt),
              ext_tokens_per_s: Math.max(0, (b.et - a.et) / dt),
            };
          }
        }
      }
      Object.assign(rec, {
        enabled: !!(kv || capBytes),
        connector,
        metric_kind: kv ? (kv.metricKind || 'fill') : null,
        capacity_gb: capBytes ? +(capBytes / gib).toFixed(2) : null,
        // 驻留：fill 口径有真实驻留 GiB；其余用 tmpfs 实测（单实例）
        resident_gb: (kv && kv.metricKind === 'fill' && capBytes) ? +(kv.fillPerc * capBytes / gib).toFixed(2) : (shm ? +shm.used_gb.toFixed(2) : null),
        resident_src: (kv && kv.metricKind === 'fill') ? 'metric' : (shm ? 'tmpfs' : null),
        shm_used_gb: shm ? shm.used_gb : null,
        shm_total_gb: shm ? shm.total_gb : null,
        shm_used_pct: shm ? shm.used_pct : null,
        // 指标累计
        queries: kv ? Math.round(kv.queries || 0) : 0,
        hits: kv ? Math.round(kv.hits || 0) : 0,
        hit_rate: (kv && kv.queries > 0) ? +(kv.hits / kv.queries * 100).toFixed(2) : null,
        ext_tokens: kv ? Math.round(kv.extTokens || 0) : 0,
        stored_gb: kv ? +(kv.storedBytes / gib).toFixed(2) : 0,
        loaded_gb: kv ? +(kv.loadedBytes / gib).toFixed(2) : 0,
        store_count: kv ? Math.round(kv.storeCount || 0) : 0,
        load_count: kv ? Math.round(kv.loadCount || 0) : 0,
        fill_pct: kv ? +(kv.fillPerc * 100).toFixed(1) : null,
        write_pct: (kv && kv.writePerc != null) ? +(kv.writePerc * 100).toFixed(1) : null,
        read_pct: (kv && kv.readPerc != null) ? +(kv.readPerc * 100).toFixed(1) : null,
        rates,
        system_mem: readMemInfo(),
      });
      out.push(rec);
    }
  } catch (e) {}
  return Promise.all(jobs).then(() => out, () => out);
}

// 09-01 SGLang「理论可命中率」（LCP×8192 网格折算，sglang-theory.service 每 60s 写 json）
function readSglangTheory(port) {
  try {
    const now = Date.now();
    if (!global.__theoryCache) global.__theoryCache = new Map();
    let e = global.__theoryCache.get(String(port));
    if (!e || now - e.t > 15000) {
      const raw = fs.readFileSync('/home/ll/deploy/sglang-theory-' + port + '.json', 'utf8');
      e = { t: now, d: JSON.parse(raw) };
      global.__theoryCache.set(String(port), e);
    }
    return e.d;
  } catch (err) { return null; }
}

// 采样指定端口的 /metrics 并更新对应 ticker（主/从端口共用）
function samplePortMetrics(port) {
  const tk = getTicker(port);
  // [hw-move 1005] busy 滞留防御：底层请求既不 resolve 也不 reject（半开连接等罕见态）
  // 会让 busy 永真、该端口 ticker 从此停摆——连带端口自愈的错误回调永不触发。
  // 超过 15s 未见结算即强制放行下一轮（正常一轮 <3s）。
  if (tk.busy) {
    if (!(tk.busySince && Date.now() - tk.busySince > 15000)) return;
    console.error(`[ticker:${port}] busy 滞留 >15s，强制放行（疑似半开连接）`);
  }
  tk.busy = true; tk.busySince = Date.now();
  const req = http.get(`http://${config.vllmHost}:${port}/metrics`, (proxyRes) => {
    let data = '';
    proxyRes.on('data', c => data += c);
    proxyRes.on('error', () => { tk.busy = false; });
    proxyRes.on('end', () => {
      tk.busy = false;
      // v2 每请求速度采样：复用本次 /metrics 响应（零额外请求），每秒维护
      try { const _sp = reqOutSamplerFor(port); _sp.lastTickerFeedAt = Date.now(); processReqOutSample(port, data, Date.now()); } catch (e) {}
      try {
        const m = parseMetrics(data);
        const ns = metricsNamespace(m);
        // vllm 与 sglang 的生成/输入 token 计数器同名（仅前缀不同）；running 名称不同
        const genBase = ns === 'sglang' ? 'sglang:generation_tokens_total' : 'vllm:generation_tokens_total';
        const promptBase = ns === 'sglang' ? 'sglang:prompt_tokens_total' : 'vllm:prompt_tokens_total';
        const runningBase = ns === 'sglang' ? 'sglang:num_running_reqs' : 'vllm:num_requests_running';
        const total = Math.round(counterTotal(m, genBase));
        const promptTotal = Math.round(counterTotal(m, promptBase));
        // 未缓存预填充 token（source=local_compute）：vLLM 每个迭代步都递增该
        // 计数器（见 vllm/v1/metrics/loggers.py record()），所以它的每秒增量就是
        // 「真实发生的预填充计算量」——精确的实时预填充吞吐（缓存命中不计入）。
        const uncachedTotal = ns === 'sglang'
          ? 0 : Math.round(counterBySource(m, 'vllm:prompt_tokens_by_source_total', 'local_compute'));
        // 每 GPU 累计缓存命中统计（独立字段，不动上面 uncachedTotal 路径，
        // 避免改变 PD 走势图 sglang 分支行为）：
        // vLLM = local_cache_hit / (hit + local_compute)；
        // SGLang = cached_tokens_total / (cached + uncached_prompt_tokens_histogram_sum)
        const cacheCached = ns === 'sglang'
          ? Math.round(counterTotal(m, 'sglang:cached_tokens_total'))
          : Math.round(counterBySource(m, 'vllm:prompt_tokens_by_source_total', 'local_cache_hit'));
        const cacheUncached = ns === 'sglang'
          ? Math.round(counterTotal(m, 'sglang:uncached_prompt_tokens_histogram_sum'))
          : uncachedTotal;
        tk.lastCacheCached = cacheCached;
        tk.lastCacheUncached = cacheUncached;
        // [kvoff-display 09-22] CPU KV 二级缓存（vLLM OffloadingConnector 才有这些指标）。
        // [kvoff-0300 0930] 双栈兼容：官方 0.30.0 新栈只暴露上游原生
        // cpu_cache_usage_perc（官方文档口径=「被在飞传输钉住」的比例，**不是**驻留占比），
        // 且写入/回载序列变成无标签单值（store_bytes_total / load_size_count）；
        // 旧栈（自研镜像 + kvfill 补丁）才是 cpu_cache_fill_perc（真实驻留占比）
        // + 带 transfer_type 标签的 total_bytes_total / size_count。
        // 两栈任一 gauge 在 = 二级缓存已启用，口径差异用 metricKind 透传给前端，
        // 绝不把 usage 当驻留显示（否则会把「100% 被钉住」误读成「缓存塞满」）。
        tk.kvOffload = null;
        if (ns !== 'sglang') {
          try {
            const fillKey = Object.keys(m).find(k => k.startsWith('vllm:kv_offload_cpu_cache_fill_perc|'));
            const usageKey = Object.keys(m).find(k => k.startsWith('vllm:kv_offload_cpu_cache_usage_perc|'));
            if (fillKey || usageKey) {
              // 无标签单值序列（新栈）优先，回落带 transfer_type 标签的旧栈序列
              const plain = (name) => {
                for (const k of Object.keys(m)) { if (k === name || k.startsWith(name + '|')) return m[k] || 0; }
                return 0;
              };
              const pick = (newName, oldName, tt) => plain(newName) || counterByLabel(m, oldName, 'transfer_type', tt);
              tk.kvOffload = {
                metricKind: fillKey ? 'fill' : 'usage',
                fillPerc: (fillKey ? m[fillKey] : m[usageKey]) || 0,
                // [kvoff-live 09-27] 新栈写/读两侧在飞占比 + 二级缓存查询/命中计数
                // （旧栈无这些族 → 0，前端按 null 处理不当真值显示）
                writePerc: plain('vllm:kv_offload_cpu_cache_write_usage_perc'),
                readPerc: plain('vllm:kv_offload_cpu_cache_read_usage_perc'),
                queries: counterTotal(m, 'vllm:external_prefix_cache_queries_total'),
                hits: counterTotal(m, 'vllm:external_prefix_cache_hits_total'),
                storedBytes: pick('vllm:kv_offload_store_bytes_total', 'vllm:kv_offload_total_bytes_total', 'GPU_to_CPU'),
                loadedBytes: pick('vllm:kv_offload_load_bytes_total', 'vllm:kv_offload_total_bytes_total', 'CPU_to_GPU'),
                // [kvoff-hit 09-22] 命中真值：connector 回载的 prompt token（免重算部分，
                // source=external_kv_transfer）+ 回载/写入次数（新栈 _size_count 无标签、
                // 旧栈 _size_count 带 transfer_type）
                extTokens: counterBySource(m, 'vllm:prompt_tokens_by_source_total', 'external_kv_transfer'),
                loadCount: pick('vllm:kv_offload_load_size_count', 'vllm:kv_offload_size_count', 'CPU_to_GPU'),
                storeCount: pick('vllm:kv_offload_store_size_count', 'vllm:kv_offload_size_count', 'GPU_to_CPU'),
              };
            } else {
              // [kvoff-simple 0927] SimpleCPUOffloadConnector 不暴露 kv_offload_* 族，
              // 只有 external_prefix_cache_*（查询/命中 token）→ 以 simple 口径出条目，
              // 存/在飞等字段如实为 null/0，前端按 metric_kind=simple 隐藏误导性小字。
              const q0 = counterTotal(m, 'vllm:external_prefix_cache_queries_total');
              const h0 = counterTotal(m, 'vllm:external_prefix_cache_hits_total');
              if (q0 || h0) {
                tk.kvOffload = {
                  metricKind: 'simple', fillPerc: 0, writePerc: null, readPerc: null,
                  queries: q0, hits: h0, storedBytes: 0, loadedBytes: 0,
                  extTokens: counterBySource(m, 'vllm:prompt_tokens_by_source_total', 'external_kv_transfer'),
                  loadCount: 0, storeCount: 0,
                };
              }
            }
          } catch (e) {}
        }
        const running = Math.round(
          ns === 'sglang' ? gaugeValue(m, runningBase)
                          : (m['vllm:num_requests_running' + buildModelLabelFromMetrics(m, 'vllm:num_requests_running{')] || 0)
        );
        const now = Date.now();
        if (tk.lastTotal !== null && now - tk.lastTime >= 500) {
          const dt = (now - tk.lastTime) / 1000;
          const genDelta = total - tk.lastTotal;
          const promptDelta = promptTotal - (tk.lastPromptTotal || promptTotal);
          const uncachedDelta = uncachedTotal - (tk.lastUncachedTotal || uncachedTotal);
          tk.lastSecond = {
            tokens: Math.max(0, genDelta),
            speed: dt > 0 ? parseFloat(Math.max(0, genDelta / dt).toFixed(1)) : 0,
            promptTokens: Math.max(0, promptDelta),
            promptSpeed: dt > 0 ? parseFloat(Math.max(0, promptDelta / dt).toFixed(1)) : 0,
            // 未缓存预填充吞吐（精确实时，bench_pp_tps 与每行预填充速率的数据源）
            uncachedTokens: Math.max(0, uncachedDelta),
            uncachedPPS: dt > 0 ? parseFloat(Math.max(0, uncachedDelta / dt).toFixed(1)) : 0,
            running: running,
            at: now,
          };
        } else if (tk.lastTotal === null) {
          tk.lastSecond = { tokens: 0, speed: 0, promptTokens: 0, promptSpeed: 0, uncachedTokens: 0, uncachedPPS: 0, running: running, at: now };
        }
        tk.lastTotal = total;
        tk.lastPromptTotal = promptTotal;
        tk.lastUncachedTotal = uncachedTotal;
        tk.lastTime = now;
        // 生成 token 计数历史（09-15 瞬时输出速度）：每秒 1 样本、滚动 ~8 样本，
        // 供 computeConcurrencyDetails 短窗差分算并发请求行的「近1s tok/s」。
        if (!tk.genHist) tk.genHist = [];
        tk.genHist.push({ t: now, g: total });
        if (tk.genHist.length > 8) tk.genHist.shift();

        // ====== PD 动态图时间序列（全局按秒聚合，多端口累加） ======
        // 常规模式：单端口同时贡献 prefill(uncached) 与 gen；PD 模式：prefill 端口贡献
        // uncached（request_prefill_kv_computed_tokens 口径，缓存命中不计）、decode 端口
        // 贡献 gen，按秒桶累加后天然对齐成「预填充 → 输出」一条走势。保留 2h。
        if (tk.lastSecond) {
          if (!global.__pdSeries) global.__pdSeries = { points: [] };
          const S = global.__pdSeries;
          const bucket = Math.floor(now / 1000);
          let cur = S.points.length ? S.points[S.points.length - 1] : null;
          if (!cur || cur.t !== bucket) {
            cur = { t: bucket, prefill: 0, gen: 0 };
            S.points.push(cur);
            if (S.points.length > 7200) S.points.splice(0, S.points.length - 7200);
          }
          cur.prefill += tk.lastSecond.uncachedTokens || 0;
          cur.gen += tk.lastSecond.tokens || 0;
        }
      } catch (e) { /* ignore malformed metrics */ }
    });
  });
  // 只有主端口拉取失败才触发端口自愈探测（从实例挂掉不代表主后端换端口）
  req.on('error', () => { tk.busy = false; if (port === config.vllmPort) maybeReDetectBackend(); });
  req.setTimeout(2000, () => { req.destroy(new Error('ticker timeout')); });
}

setInterval(() => {
  // 主后端 + 所有探测到的 vllm 实例（多卡多实例各自独立采样）
  const ports = new Set([config.vllmPort]);
  try { for (const inst of listVllmInstances()) ports.add(inst.port); } catch (e) {}
  // 08-30: SGLang 实例也进 ticker 采样（每 GPU 缓存命中率依赖 ticker 累计计数器）
  try { for (const inst of listSglangInstances()) ports.add(inst.port); } catch (e) {}
  for (const p of ports) { samplePortMetrics(p); sampleReqOutGauges(p); }
  // 主别名跟随自愈后的端口
  global.__tokTicker = getTicker(config.vllmPort);
}, 1000).unref();

// ====== 1-second whole-machine energy sampler (能耗/电费) ======
// 每秒采样「整机估算总功耗」= 全部 GPU 功率之和 + CPU(RAPL) + 补偿值，
// 按 W × Δt 积分累计 Wh（零阶保持）。口径与能耗统计页的整机估算总功耗一致。
// 与 vLLM 是否运行无关：待机功耗同样计入。持久化到 energy-state.json，
// 每 10 秒落盘一次（进程被杀最多丢 ~10 秒积分，误差可忽略）。
const ENERGY_STATE_PATH = path.join(__dirname, 'energy-state.json');
const ENERGY_CONFIG_PATH = path.join(__dirname, 'energy-config.json');

function loadEnergyState() {
  try {
    const s = JSON.parse(fs.readFileSync(ENERGY_STATE_PATH, 'utf8'));
    if (s && typeof s === 'object') {
      const days = {};
      for (const [k, v] of Object.entries(s.days || {})) {
        const wh = parseFloat(v);
        if (!isNaN(wh) && wh >= 0) days[k] = wh;
      }
      return { wh: parseFloat(s.wh) || 0, history_wh: parseFloat(s.history_wh) || 0, days };
    }
  } catch (e) {}
  return { wh: 0, history_wh: 0, days: {} };
}

function saveEnergyState() {
  const en = global.__energy;
  if (!en) return;
  try {
    fs.writeFileSync(ENERGY_STATE_PATH, JSON.stringify({ wh: en.wh, history_wh: en.history_wh, days: en.days }, null, 2));
    en.lastFlush = Date.now();
  } catch (e) {}
}

function loadEnergyConfig() {
  try {
    const c = JSON.parse(fs.readFileSync(ENERGY_CONFIG_PATH, 'utf8'));
    if (c && typeof c === 'object') return { price_per_kwh: parseFloat(c.price_per_kwh) || 0, unit: c.unit || '元' };
  } catch (e) {}
  return { price_per_kwh: 1.0, unit: '元' };
}

function saveEnergyConfig(c) {
  fs.writeFileSync(ENERGY_CONFIG_PATH, JSON.stringify(c, null, 2));
}

// 功耗补偿值：用户手动设置，代表无法直接测量的其它硬件（NVMe/Wi-Fi/主板/电源损耗等）
const POWER_CONFIG_PATH = path.join(__dirname, 'power-config.json');
function loadPowerConfig() {
  try {
    const c = JSON.parse(fs.readFileSync(POWER_CONFIG_PATH, 'utf8'));
    const out = { offsetW: Math.max(0, Math.min(1000, parseFloat(c.offsetW) || 0)) };
    // [gpu-ctl 1003] 控制台最近一次设置的 GPU 功耗上限（记录用途；生效值以 nvidia-smi 现查为准）
    if (c.gpuPlW != null) { const g = parseInt(c.gpuPlW, 10); if (g >= 50 && g <= 500) out.gpuPlW = g; }
    return out;
  } catch (e) {}
  return { offsetW: 0 };
}
function savePowerConfig(c) {
  const out = { offsetW: c.offsetW };
  if (c.gpuPlW != null) out.gpuPlW = c.gpuPlW;
  fs.writeFileSync(POWER_CONFIG_PATH, JSON.stringify(out, null, 2));
}

{
  const seeded = loadEnergyState();
  global.__energy = { wh: seeded.wh, history_wh: seeded.history_wh, days: seeded.days || {}, lastWatts: null, lastSampleAt: null, lastFlush: Date.now(), busy: false };
}
setInterval(() => {
  const en = global.__energy;
  if (en.busy) return;
  en.busy = true;
  // 10-07 测试修复：原 spawn + "只 SIGKILL 不结算"的定时器，对 D 状态 nvidia-smi 无效
  // （'close' 永不触发）→ en.busy 永久停在 true，能耗积分从此静默冻结，永不恢复。
  // 改走 gpuQuery：自建定时器放弃等待，必定 settle，且共享 GPU 熔断。
  gpuQuery(['--query-gpu=power.draw', '--format=csv,noheader,nounits'], GPU_Q_MS_LIVE).then((out) => {
    en.busy = false;
    // 全部 GPU 功率求和（单卡机即该卡功耗；双卡机兼容，防未来扩卡口径漂移）
    let gpuW = 0;
    for (const line of String(out || '').split('\n')) {
      const w = parseFloat(line.trim());
      if (!isNaN(w) && w > 0) gpuW += w;
    }
    if (gpuW <= 0) return; // nvidia-smi 失败：跳过本轮，避免用 0/负值污染积分
    // 整机估算总功耗 = 全部 GPU + CPU(RAPL，功耗监测模块维护) + 补偿值
    // 口径与能耗统计页的「整机估算总功耗」(adjustedTotalW) 完全一致
    const cpuW = (typeof cpuPowerW !== 'undefined' && cpuPowerW !== null) ? cpuPowerW : 0;
    const totalW = gpuW + cpuW + loadPowerConfig().offsetW;
    const now = Date.now();
    if (en.lastSampleAt !== null) {
      const dtSec = (now - en.lastSampleAt) / 1000;
      if (dtSec > 0 && dtSec <= 10) { // 间隔健康才积分，避免停机/挂起导致的时间洞被放大
        const dWh = en.lastWatts * dtSec / 3600;
        en.wh += dWh;
        en.history_wh += dWh;
        // 每日能耗桶（服务器本地日期，与每日费用明细对齐）
        if (!en.days) en.days = {};
        const dk = billingDayKey();
        en.days[dk] = (en.days[dk] || 0) + dWh;
        const dkKeys = Object.keys(en.days).sort();
        while (dkKeys.length > BILLING_MAX_DAYS) delete en.days[dkKeys.shift()];
      }
    }
    en.lastWatts = totalW;
    en.lastSampleAt = now;
    if (now - en.lastFlush > 10000) saveEnergyState();
  });
}, 1000).unref();

// 系统内存（/proc/meminfo）：2 秒缓存，避免 buildEnergyInfo 高频调用时重复读盘
let _lastMemInfo = null, _lastMemInfoAt = 0;
function readMemInfo() {
  const now = Date.now();
  if (_lastMemInfo && now - _lastMemInfoAt < 2000) return _lastMemInfo;
  try {
    const txt = fs.readFileSync('/proc/meminfo', 'utf8');
    const mi = {};
    for (const line of txt.split('\n')) {
      const mm = line.match(/^(\w+):\s+(\d+)\s*kB/);
      if (mm) mi[mm[1]] = parseInt(mm[2], 10);
    }
    if (mi.MemTotal) {
      const total = mi.MemTotal;
      const avail = mi.MemAvailable !== undefined ? mi.MemAvailable : (mi.MemFree || 0);
      const used = Math.max(0, total - avail);
      _lastMemInfo = {
        used_pct: parseFloat((used / total * 100).toFixed(1)),
        total_gb: Math.round(total / 1024 / 1024 * 10) / 10,
        used_gb: Math.round(used / 1024 / 1024 * 10) / 10,
      };
      _lastMemInfoAt = now;
    }
  } catch (e) {}
  return _lastMemInfo;
}

// ====== [gpu-ctl 1003] GPU 功耗/频率调节（硬件监视页）======
// 开机持久化联动：gpu-power-limit.service 开机按 drop-in 的 Environment=PL=<W> 设功耗上限
// （09-26 铁律：只敲 nvidia-smi -pl 不改 PL，重启必被打回）。控制台设完功耗后同步改写
// drop-in，让「控制台设的值 = 重启后的值」。drop-in 属 root，写入走 gpu-ctl 的
// persist-pl 子命令（脚本内部已是 root，见 syncGpuPlDropin）。
// 撤销联动：sudo rm -f /etc/systemd/system/gpu-power-limit.service.d/pl-console.conf && sudo systemctl daemon-reload
async function syncGpuPlDropin(watt) {
  // gpuCtlRun 永不 reject（内部消化错误），返回 {ok,output}
  const r = await gpuCtlRun(['persist-pl', String(watt)], 20000);
  if (!r.ok) return String(r.output || 'persist-pl 失败').slice(0, 300);
  return null;
}

function buildEnergyInfo() {
  const en = global.__energy || {};
  const cfg = loadEnergyConfig();
  const kwh = (en.wh || 0) / 1000;
  const hkwh = (en.history_wh || 0) / 1000;
  // 每日电费：按天累计的 Wh × 当前电价（电价调整后按新价实时显示）
  const days = {};
  for (const [dk, dwh] of Object.entries(en.days || {})) {
    const dkwh = dwh / 1000;
    days[dk] = { kwh: dkwh, cost: dkwh * cfg.price_per_kwh };
  }
  return {
    kwh,
    cost: kwh * cfg.price_per_kwh,
    history_kwh: hkwh,
    history_cost: hkwh * cfg.price_per_kwh,
    days,
    current_power_w: en.lastWatts || 0, // 整机估算总功耗（全部 GPU + CPU + 补偿值），仪表盘「总功耗」显示用
    price_per_kwh: cfg.price_per_kwh,
    unit: cfg.unit || '元',
    sampling_lag_s: en.lastSampleAt ? Math.round((Date.now() - en.lastSampleAt) / 1000) : -1,
    cpu_util_pct: (typeof cpuUtilPct !== 'undefined') ? cpuUtilPct : null, // 处理器 CPU 整机利用率 %（/proc/stat 2s 采样，全部核心平均）
    cpu_cores: (typeof CPU_CORES !== 'undefined') ? CPU_CORES : null,
    mem: readMemInfo(), // 系统内存 { used_pct, total_gb, used_gb }（/proc/meminfo，2s 缓存）
  };
}

function resetEnergyBucket(scope) {
  const en = global.__energy;
  if (!en) return;
  if (scope === 'current') en.wh = 0;
  else if (scope === 'history') en.history_wh = 0;
  saveEnergyState();
}

// ====== Live token-stream capture ======
// vLLM metrics carry no per-request generated text. Requests that pass
// through this console's /v1/(chat/)completions proxy are tee'd here: the
// raw bytes are forwarded untouched, while the delta text is accumulated
// per request so the dashboard can show each running request's token
// stream (red-box area of the "正在处理 N 个请求" card).
if (!global.__liveStreams) {
  global.__liveStreams = { seq: 0, map: new Map() };
}
const LIVE_TEXT_MAX = 4000;      // rolling window of text kept per request
const LIVE_DONE_KEEP_MS = 5000;  // keep finished entries this long for the UI
const LIVE_PATH_RE = /^\/v1\/(chat\/completions|completions|responses)$/

// ============================================================================
// 并发请求实时输出速度 v2（保留为「无 v3 流实例」的降级链）
// ----------------------------------------------------------------------------
// ⚠ 2026-10-06：主链路已换 v3（见 readLiveStream 模块注释）——引擎内部
// 每步每请求输出真值流，逐请求精确测速。本 v2 链只在 v3 不激活的实例上
// 运行：sglang / 远端实例 / 未装 rt-patch#11 的旧 vLLM。其产物各行同值
// （实例吞吐÷估计并发数），前端如实标「分摊」。
//
// v2 数据源（引擎 Prometheus /metrics，每秒采样）：
//  A. 每请求在途 gauge（带真实 rid 标签，仅定制导出该族的实例命中；
//     官方 0.30.0 无此族，实测恒不命中）：
//     vllm:request_generation_tokens{req_id} / request_decode_time_seconds
//     / time_to_first_token_seconds → 滑窗 Δgen/Δdecode_time。
//  B. 直方图驻留估计（当前实际生效路径）：decode 时长桶差分按 Little 定律
//     估在途 decode 数 N；速度 = 实例 generation_tokens 短窗吞吐 ÷ N。
//  C. 全无 → undefined（前端如实显示 --）。
//
// speed_src 取值：v3 / v3-exact（引擎流实测，主链路）；v2-exact /
// v2-nearest / v2-hist（v2 降级链）；tee / engine / residual（v1 残留兼容）。
// tee / engine / residual 仅作为「v2 全无信号」时旧逻辑回落值保留。
// ============================================================================
if (!global.__reqOutSamplers) global.__reqOutSamplers = new Map(); // port -> sampler
// rid 匹配（v2）：SSE 响应 id（chatcmpl-xxx）与引擎内部 rid（chatcmpl-xxx-yyy，
// 插件/gauge 用的带尾缀形式）互为前缀关系——精确匹配必须容忍该尾缀。
function ridMatches(a, b) {
  if (!a || !b) return false;
  if (a === b) return true;
  return String(b).indexOf(a + '-') === 0 || String(a).indexOf(b + '-') === 0;
}
// 在带 rid 键的 Map 里找与响应 id 匹配的真实键（gauge 序列 / live-prefill 记录）
function ridFind(map, rid) {
  if (!map || !rid) return null;
  if (map.has(rid)) return rid;
  const p = rid + '-';
  for (const k of map.keys()) { if (String(k).indexOf(p) === 0) return k; }
  return null;
}

// 代理捕获：clientReqId → { rid, at }（tee 首帧建立；行凭 taskId 反查认领 rid）
if (!global.__clientReqRid) global.__clientReqRid = new Map();
function linkClientReqRid(crid, rid) {
  if (!crid || !rid) return;
  const m = global.__clientReqRid;
  if (m.size >= 5000) { const k = m.keys().next().value; if (k !== undefined) m.delete(k); }
  if (!m.has(crid)) m.set(crid, { rid, at: Date.now() });
}

// 从 /metrics 文本抽每请求 gauge（带 req_id 标签）→ Map<rid,{gen,dec,ttft}>
function parseRequestGauges(text) {
  const out = new Map();
  if (!text) return out;
  const lines = text.split('\n');
  for (let li = 0; li < lines.length; li++) {
    const line = lines[li];
    if (!line || line.charCodeAt(0) === 35) continue; // '#' 注释/快速跳过空行
    const ridM = reqIdRe.exec(line);
    if (!ridM) continue;
    const sp = line.lastIndexOf(' ');
    const v = parseFloat(line.substring(sp + 1));
    if (!isFinite(v) || v < 0) continue;
    const name = line.substring(0, line.indexOf('{'));
    const rid = ridM[1];
    let r = out.get(rid);
    if (!r) { r = { gen: null, dec: null, ttft: null }; out.set(rid, r); }
    if (name.endsWith('request_generation_tokens')) { if (r.gen === null || v > r.gen) r.gen = v; }
    else if (name.endsWith('request_decode_time_seconds')) { if (r.dec === null || v > r.dec) r.dec = v; }
    else if (name.endsWith('time_to_first_token_seconds') && r.ttft === null) r.ttft = v;
  }
  return out;
}
const reqIdRe = /req_id="([^"]+)"/;

// 实例级直方图桶（decode 时长/首包等待两族）。用途：实例不导出带 req_id 的
// gauge 时（vLLM 默认配置），v2 引擎真值仍可用——桶计数按「请求在 bucket 内
// 驻留时间 ~ 桶上界」建模：
//   N_i(dt) = Σ_k C_k · min(1, dt/le_k)   （dt 秒未结束请求在桶 k 的期望计数）
//   平均驻留 = Σ dt·(N − N_prev)/Σ(N − N_prev) = 在途请求的 decode 平均驻留
// 数学性质：该估计【恒不高于】真实驻留（桶上界是驻留的硬下界，全请求封顶
// dt/le_k 使每个请求贡献 ≤1），且 dt 相对 le 分布较大时接近无偏。因此由此
// 得到的速度 = Δgen/驻留 是【保守下界】——宁可显示偏低也绝不虚高（用户痛点
// 是旧值虚高）。并发多请求共享桶差分时各请求分得同值（= 真实驻留分布的均值，
// 比旧版「总吞吐/行数」更接近个体真值）。
function parseBucketHist(text, metricBase) {
  // Prometheus 桶行名 = 指标名 + '_bucket'（调用方传基础名即可）。
  // 标签内可能有任意前缀（engine=... 等），le 两侧用 [^}]* 宽松匹配。
  const re = new RegExp('^' + metricBase + '_bucket\\{[^}]*le="([^"]+)"[^}]*\\}\\s+([0-9eE+.]+)\\s*$', 'gm');
  const out = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    const le = m[1] === '+Inf' ? Infinity : parseFloat(m[1]);
    const v = parseFloat(m[2]);
    if (isFinite(le) && isFinite(v)) out.push({ le, c: v });
  }
  out.sort((a, b) => a.le - b.le);
  return out;
}

// 实例采样器：每秒抓取（随 ticker 循环，busy 互斥）。cfgOk=false 会做低频探测
// （该实例是否支持 gauge，命中后自动升级），支持后无在途请求时降频省流量。
function reqOutSamplerFor(port) {
  let sp = global.__reqOutSamplers.get(port);
  if (!sp) {
    sp = {
      port, cfgOk: null, probeAt: 0, busy: false, lastSeenAt: 0,
      reqs: new Map(),       // rid -> {gen,dec,ttft,hist,stale}（gauge 模式：rid 精确序列）
      genHist: [],           // [{t,g}] 实例级 generation_tokens_total（总量守恒/顶栏兜底）
      lastGenTotal: null,
      occ: new Map(),        // rid -> 行 startedAt（最近邻粘性占用表，按端口隔离）
      // 直方图模式（实例无 req_id gauge 时的 v2 引擎真值回退）：
      decBk: [],             // decode 时长桶最新计数 [{le,c}]
      ttftBk: [],            // 首包等待桶最新计数
      prevDecBk: null,       // 上一秒快照（差分用）
      prevDecBkAt: 0,
      genAvgWin: undefined,  // 在途平均 decode 驻留（秒）；undefined=数据不足
      waitAvgWin: undefined, // 在途平均首包等待（prefill+queue 驻留）
      winN: 0,               // 驻留估计的样本数（本秒进入驻留的请求数）
      thrWin: undefined,     // 实例 gen 短窗吞吐（同 reqOutInstThroughput 口径，供驻留法分母）
      genCum: 0,             // 本会话累计输出（重启基线续算；直方图接管行累计用）
    };
    global.__reqOutSamplers.set(port, sp);
  }
  return sp;
}
// v2 采样处理核心（不发起请求）：解析每请求 gauge + 直方图桶 + 实例 gen 滑窗，
// 维护 sampler 状态。数据来自两处：主 ticker（samplePortMetrics，每秒，零额外
// 请求）或 sampleReqOutGauges 的兜底抓取（ticker 停摆时）。
function processReqOutSample(port, data, now) {
  const sp = reqOutSamplerFor(port);
  if (!data || data.indexOf('vllm:') === -1) return; // SGLang 等其它运行时：无对应指标，跳过
  try {
        const rg = parseRequestGauges(data);
        // genCumPost：在途请求「已完成部分」的累计（重启后新请求预填，行接管
        // 时扣除）。gauge 可见 = 至少一个在途请求：其 gen 之和即预填下界。
        if (rg.size > 0) {
          let postSum = 0;
          for (const g of rg.values()) if (g.gen) postSum += g.gen;
          sp.genCumPost = postSum;
        } else if (sp.genCumPost === undefined) sp.genCumPost = 0;
        if (rg.size > 0 || sp.cfgOk === true) {
          if (sp.cfgOk !== true) console.log(`[reqout] port ${port} per-request gauges active`);
        }
        // 直方图桶：无条件维护（gauge 模式实例也可能后续重启降级；且
        // 驻留估计同时为两种模式提供交叉校验基线）
        try {
          const decBk = parseBucketHist(data, 'vllm:request_decode_time_seconds');
          const ttftBk = parseBucketHist(data, 'vllm:time_to_first_token_seconds');
          if (decBk.length) {
            const bkAt = now;
            if (sp.prevDecBk && bkAt - sp.prevDecBkAt >= 700) {
              const dt = (bkAt - sp.prevDecBkAt) / 1000;
              // 桶是累计计数（le=上界）：dC = 本窗口「decode 完成」的请求数，
              // 其真实驻留 ∈ (le_{k-1}, le_k]。完成事件本身就是驻留样本——
              // W = 样本驻留均值（中点估计 sqrt(prev*cur)，单点桶退化为 le），
              // λ = 完成率。Little 定律：在途 decode 数 N = λ̄ × W（EWMA 平滑，
              // 突发完成摊平；λ 样本窗对齐 W 窗避免相位错配）。
              let dSum = 0, dWsum = 0;
              for (let k = 0; k < decBk.length && k < sp.prevDecBk.length; k++) {
                const dC = decBk[k].c - sp.prevDecBk[k].c;
                if (dC <= 0) continue;
                const lo = k > 0 ? sp.prevDecBk[k - 1].le : 0;
                const hi = decBk[k].le;
                const dur = (isFinite(lo) && lo > 0 && isFinite(hi))
                  ? Math.sqrt(lo * hi)
                  : (isFinite(hi) ? hi : (sp.winEw || 10));
                dSum += dC; dWsum += dC * dur;
              }
              const lamNew = dSum / dt;
              if (dSum > 0) {
                const wNew = dWsum / dSum;
                if (sp.winEw === undefined) { sp.winEw = wNew; sp.lamEw = lamNew; }
                else {
                  sp.winEw += 0.3 * (wNew - sp.winEw);
                  sp.lamEw += 0.3 * (lamNew - sp.lamEw);
                }
              } else if (sp.lamEw !== undefined) {
                // 空窗（无人完成 ≠ 无人 decode）：λ 保持、W 保持，N 估计不塌
                sp.lamEw *= Math.pow(0.85, dt / 1.5); // 3s 半衰向稳值缓降
                if (sp.lamEw < 0.001) sp.lamEw = 0.001;
              }
              if (sp.winEw !== undefined && sp.lamEw !== undefined) {
                const nNow = sp.lamEw * sp.winEw;
                sp.nEw = sp.nEw === undefined ? nNow : sp.nEw + 0.4 * (nNow - sp.nEw);
                sp.genAvgWin = sp.winEw;
                sp.winLastAt = now;
              }
            }
            sp.prevDecBk = decBk;
            sp.prevDecBkAt = bkAt;
          }
          if (ttftBk.length) {
            const bkAt2 = now;
            if (!sp.prevTtftBk) { sp.prevTtftBk = ttftBk; sp.prevTtftBkAt = now; }
            else {
              const dt2 = (bkAt2 - (sp.prevTtftBkAt || bkAt2)) / 1000;
              if (dt2 >= 0.7) {
                let dS = 0, dW2 = 0;
                for (let k = 0; k < ttftBk.length && k < sp.prevTtftBk.length; k++) {
                  const dC = ttftBk[k].c - sp.prevTtftBk[k].c;
                  if (dC > 0) { dS += dC; dW2 += dC * Math.min(1, dt2 / ttftBk[k].le); }
                }
                if (dS > 0) sp.waitAvgWin = Math.max(0.2, dt2 * dS / dW2);
              }
              sp.prevTtftBk = ttftBk;
              sp.prevTtftBkAt = bkAt2;
            }
          }
        } catch (e) { /* 桶解析失败不影响主路径 */ }
        // 实例级总吞吐（总量守恒 + 单请求兜底）
        try {
          const m = parseMetrics(data);
          const gt = counterTotal(m, 'vllm:generation_tokens_total');
          if (sp.lastGenTotal === null || gt < sp.lastGenTotal - 1000) sp.lastGenTotal = gt;
          const dG = Math.max(0, Math.round(gt - sp.lastGenTotal));
          sp.lastGenTotal = gt;
          if (dG > 0) { sp.genHist.push({ t: now, g: gt }); if (sp.genHist.length > 10) sp.genHist.shift(); }
          sp.thrWin = reqOutInstThroughput(sp, now); // 驻留法分母（直方图模式用）
          // 吞吐 EWMA：hist 速度显示用平滑值，消除 1.5s 窗的突发锯齿
          // （引擎 decode 有 chunk 节奏，瞬时吞吐天然抖 ±40%，用户感知为「跳」）
          if (sp.thrWin !== undefined) {
            sp.thrEw = sp.thrEw === undefined ? sp.thrWin : sp.thrEw + 0.35 * (sp.thrWin - sp.thrEw);
          } else if (sp.thrEw !== undefined) {
            // 滑窗空洞（gen 计数 1~2.5s 未更新）：EWMA 向 0 缓降，真停顿时
            // hist 速度如实回落，不会冻结在最后一个高值
            sp.thrEw *= 0.75;
            if (sp.thrEw < 0.5) sp.thrEw = undefined;
          }
          if (dG > 0) sp.genCum += dG;
        } catch (e) { /* metrics 解析失败忽略本轮 */ }
        if (rg.size === 0 && sp.cfgOk !== true) {
          // 可能该构建不带 req_id gauge，也可能只是空闲——空闲时无法区分，
          // 记 false 但保持 30s 探测（一有在途请求即升级）。
          // 重启基线捕获：无在途且从未建立累计 → genCumPre=当前 gen_total，
          // genCumPost=0（本轮起累计从引擎计数重新起算）。
          if (sp.genCumPre === undefined) {
            try {
              const mm = parseMetrics(data);
              sp.genCumPre = Math.round(counterTotal(mm, 'vllm:generation_tokens_total'));
              sp.genCumPost = 0;
            } catch (e) {}
          }
          sp.cfgOk = sp.cfgOk === true ? true : false;
          return;
        }
        sp.cfgOk = true;
        // 更新在途表。样本时间戳 = 请求到达 epoch + 累计 decode 时长（引擎
        // 真值，与抓取周期无关）：轮询被事件循环拖慢时样本时刻不会系统性
        // 提前/滞后，滑窗分母（decode 增量）严格等于窗内真实 GPU 生成时长。
        for (const [rid, g] of rg) {
          let r = sp.reqs.get(rid);
          if (!r) { r = { gen: 0, dec: 0, ttft: undefined, stale: 0, hist: [] }; sp.reqs.set(rid, r); }
          if (g.dec !== null) r.dec = g.dec;
          if (g.gen !== null) r.gen = g.gen;
          if (g.ttft !== null && r.ttft === undefined) r.ttft = g.ttft;
          r.stale = 0;
          if (r.dec > 0) {
            // 样本时刻按引擎真值折算成墙钟 ms（供与 nowMs 同域比较）
            const sampleT = ((r.ttft !== undefined ? r.ttft : (now / 1000 - r.dec)) + r.dec) * 1000;
            const lastS = r.hist.length ? r.hist[r.hist.length - 1] : null;
            if (!lastS || sampleT - lastS.t >= 800) {
              if (lastS && g.gen < r.lastGen) { r.hist = []; } // gauge 回退（重启/重采样）：清窗重建
              r.hist.push({ t: sampleT, gen: g.gen !== null ? g.gen : r.gen, dec: r.dec });
              if (r.hist.length > 10) r.hist.shift();
            }
            r.lastGen = g.gen !== null ? g.gen : r.gen;
          }
        }
        // 消失 = 请求已完成（Prometheus gauge 撤除）。保留对象一个短 TTL：
        // 响应 tee 首帧登记 rid 可能晚于最后一次 gauge 抓取，行需要凭 rid
        // 找到对象读出最终 gen。超 3 轮未见 → 删除。
        for (const [rid, r] of sp.reqs) {
          r.stale++;
          if (r.stale > 3) sp.reqs.delete(rid);
        }
  } catch (e) { /* ignore */ }
}

// 兜底抓取器：主 ticker 未覆盖该端口（>2.5s 没喂数据）时独立抓一次 /metrics。
function sampleReqOutGauges(port) {
  const sp = reqOutSamplerFor(port);
  if (sp.busy) return;
  const now = Date.now();
  if (now - (sp.lastTickerFeedAt || 0) < 2500) return;
  sp.busy = true;
  const req = http.get(`http://${config.vllmHost}:${port}/metrics`, (proxyRes) => {
    let data = '';
    let aborted = false;
    proxyRes.on('data', (c) => {
      data += c;
      if (data.length > 16 * 1024 * 1024) { aborted = true; try { proxyRes.destroy(); } catch (e) {} }
    });
    proxyRes.on('error', () => { sp.busy = false; });
    proxyRes.on('end', () => {
      sp.busy = false;
      if (aborted) return;
      processReqOutSample(port, data, Date.now());
    });
  });
  req.on('error', () => { sp.busy = false; });
  req.setTimeout(2500, () => { try { req.destroy(new Error('reqout timeout')); } catch (e) {} sp.busy = false; });
}

// 速度推导：给定某 rid 的采样对象 → { spd, gen, exact } 或 null。
// 滑窗基线 = hist 中「距现在 ∈[2s,6s] 最老的样本」；末样本距现在 ≤2.5s；
// 分母 = decode 时间增量（该请求 GPU 生成时长）。无合格窗口 → null（显示 --）。
function reqOutSpeed(r, nowMs) {
  if (!r || !Array.isArray(r.hist) || r.hist.length < 2) return null;
  const last = r.hist[r.hist.length - 1];
  if (!last || (nowMs - last.t) > 2500) return null;
  let base = null;
  for (let k = 0; k < r.hist.length - 1; k++) {
    const age = nowMs - r.hist[k].t;
    if (age < 2000) continue;
    if (age <= 6000) { base = r.hist[k]; break; }
    if (!base) base = r.hist[k];
  }
  if (!base) base = r.hist[0];
  const dDec = last.dec - base.dec;
  const dGen = last.gen - base.gen;
  if (dDec < 0.8 || dGen <= 0) return null;
  return { spd: dGen / dDec, gen: last.gen, win: dDec };
}

// 直方图驻留模式速度（实例无 req_id gauge 时的 v2 引擎真值回退）。
// 输入 sampler：sp.genAvgWin = 在途请求平均 decode 驻留（保守下界，见
// parseBucketHist 注释）；sp.thrWin = 实例 gen 短窗吞吐。
// 速度 = thrWin / N（N=在途 decode 请求数），再按驻留修正排队行：
// 返回 { spdPerDecodeReq, avgWin }；调用方只对 phase=decode 行发放。
// 驻留 2s 内无新桶样本沿用（稀疏流量下保持稳定），>2s 过期 → undefined。
function reqOutHistModeSpeed(sp, nowMs) {
  if (!sp || sp.nEw === undefined) return null;
  const thr = sp.thrEw !== undefined ? sp.thrEw : sp.thrWin;
  if (thr === undefined) return null;
  // nEw 过期（>4s 无桶样本）→ null：宁可回落 tee，不用陈旧并发数
  if (sp.winLastAt && nowMs - sp.winLastAt > 4000) return null;
  if (thr <= 0.5) return null;
  return { n: Math.max(1, sp.nEw), thr };
}

// 顶栏/总量口径：实例 genHist 短窗速率（最近样本 ≤2.5s）
function reqOutInstThroughput(sp, nowMs) {
  if (!sp || sp.genHist.length < 2) return undefined;
  const h = sp.genHist.slice(-4);
  const b = h[h.length - 1];
  if (!b || nowMs - b.t > 2500) return undefined;
  const a = h[0];
  const dt = (b.t - a.t) / 1000;
  if (dt < 0.5) return undefined;
  return Math.max(0, (b.g - a.g) / dt);
}

// 采样器回收：端口 60s 无任何访问（stats/ticker 都停了）→ 删除采样器，
// 防实例下线后 Map 泄漏。
setInterval(() => {
  try {
    const t = Date.now();
    for (const [p, sp] of global.__reqOutSamplers) {
      if (t - (sp.lastSeenAt || 0) > 60000) global.__reqOutSamplers.delete(p);
    }
  } catch (e) {}
}, 30000).unref();

function clientIp(req) {
  try {
    let a = (req.socket && req.socket.remoteAddress) || '';
    a = a.replace(/^::ffff:/, '');
    if (!a || a === '127.0.0.1' || a === '::1') return '—';
    return a;
  } catch (e) { return '—'; }
}

// 粗略估算请求的 prompt token 数（用于 SGLang 预填充进度条总量；vLLM 有引擎侧估算不用它）。
// 中文 tokenizer 约 1 字≈1 token，英文约 4 字符≈1 token；中英混合取 1 字符≈0.9 token 的
// 粗略值即可（前端标"(估)"，仅作进度参考）。
function estimatePromptTokens(b) {
  if (!b || !Array.isArray(b.messages)) return 0;
  let chars = 0;
  for (const m of b.messages) {
    const c = m && m.content;
    if (typeof c === 'string') chars += c.length;
    else if (Array.isArray(c)) for (const part of c) if (part && typeof part.text === 'string') chars += part.text.length;
  }
  return Math.max(1, Math.round(chars * 0.9));
}

// 异步用 sglang 的 /v1/messages/count_tokens 精确计算请求的 prompt token，
// 更新 live.promptTokens（预填充进度条总量用精确值而非字符估算）。
// 不阻塞转发：请求先照常发出，token 数异步回来覆盖估算值；失败/超时保持估算兜底。
function refreshPromptTokens(live, body, host, port) {
  try {
    if (!live || !body || !Array.isArray(body.messages) || !host) return;
    const payload = JSON.stringify({ model: body.model, messages: body.messages });
    const req = http.request({
      host, port: port || 8001, path: '/v1/messages/count_tokens', method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
      timeout: 5000,
    }, (res) => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        try {
          const j = JSON.parse(data);
          const n = Math.round(j.input_tokens || 0);
          if (n > 0) { live.promptTokens = n; live.promptTokensExact = true; }
        } catch (e) { /* 保持估算兜底 */ }
      });
    });
    req.on('error', () => { /* 保持估算兜底 */ });
    req.on('timeout', () => { try { req.destroy(); } catch (e) {} });
    req.end(payload);
  } catch (e) { /* ignore */ }
}

// ====== App-layer chat history trimming（prefix-cache 优化） ======
// 生产多轮聊天把全文历史（含陈旧轮次）每次全量发给 vLLM，单请求 prompt 可达 170K+ token，
// 一个请求就吃掉 KV 池的 60%+，多会话互斥导致 LRU 逐出、跨轮命中率 ≈0。
// 裁剪策略（冻结前缀 + 滑动尾部）：
//   1) 全部 system 消息 + 最早的非 system 消息（最长 KEEP_OLD_CHARS 字符）＝【冻结前缀】；
//      会话最早的内容永不变化 → 每轮请求 token[0..] 与该段恒定一致，1664-token 对齐稳定命中；
//   2) 固定文本占位消息插在冻结段之后（内容恒定 → 占位 token 稳定）；
//   3) 最新的非 system 消息（最长 KEEP_TAIL_CHARS 字符）＝【滑动尾部】，每轮被新消息替换。
// 仅当总量 > CAP_MAX_CHARS 才裁剪（低于阈值 byte 级原样转发，零重写开销）。
// 代价分析：被裁掉的中段历史只影响尾部（尾部在缓存里本来就是每轮新写的冷段），
// 冻结前缀每轮必命中 → 命中率 ≈ KEEP_OLD/(KEEP_OLD+KEEP_TAIL)（默认 ≈ 66%）。
// 参数（环境变量，单位＝字符；中文≈1 char/token，英文≈4-5 char/token；生产 monster 单请求 ~185K token ≈ 205K chars）：
//   VLLM_CHAT_TRIM=0 关闭；
//   VLLM_CHAT_CAP_MAX_CHARS（默认 80000≈63K token，低于此不裁）；
//   VLLM_CHAT_KEEP_OLD_CHARS（默认 40000≈31K token 冻结前缀）；
//   VLLM_CHAT_KEEP_TAIL_CHARS（默认 20000≈16K token 滑动尾部）。
// 裁剪后单请求 ≈ 47K token，KV 池（≈500K token）可容纳 10+ 个会话，LRU 不再互相逐出。
const CHAT_TRIM_PLACEHOLDER = '【较早对话已省略：为节省上下文，仅保留最早对话与最近对话内容。】';

function chatTrimEnabled() {
  const v = process.env.VLLM_CHAT_TRIM;
  if (v === undefined || v === null || v === '') return true;
  return !/^(0|false|off|no)$/i.test(String(v));
}

function chatTrimCapMaxChars() {
  const n = parseInt(process.env.VLLM_CHAT_CAP_MAX_CHARS || '80000', 10);
  return (n >= 2000 && n <= 2000000) ? n : 80000;
}

function chatTrimKeepOldChars() {
  const n = parseInt(process.env.VLLM_CHAT_KEEP_OLD_CHARS || '40000', 10);
  return (n >= 1000 && n <= 1000000) ? n : 40000;
}

function chatTrimKeepTailChars() {
  const n = parseInt(process.env.VLLM_CHAT_KEEP_TAIL_CHARS || '20000', 10);
  return (n >= 1000 && n <= 1000000) ? n : 20000;
}

// 消息内容字符数（兼容 content 字符串与多模态 parts 的 text 字段）
function msgChars(m) {
  const c = m && m.content;
  if (typeof c === 'string') return c.length;
  if (Array.isArray(c)) {
    let n = 0;
    for (const part of c) if (part && typeof part.text === 'string') n += part.text.length;
    return n;
  }
  return 0;
}

// 从头取最长前缀，使累计字符 ≤ maxChars；至少返回 1 条（保证不产生空段）
function takeMsgsByChars(msgs, maxChars) {
  const out = [];
  let chars = 0;
  for (const m of msgs) {
    const c = msgChars(m);
    if (out.length > 0 && chars + c > maxChars) break;
    out.push(m);
    chars += c;
  }
  return out;
}

// 返回裁剪统计；未发生裁剪返回 null（此时 body 原样转发，无任何重写开销）
function trimChatBody(body) {
  if (!body || !Array.isArray(body.messages) || body.messages.length === 0) return null;
  const capMax = chatTrimCapMaxChars();
  const msgs = body.messages;
  const beforeChars = msgs.reduce((s, m) => s + msgChars(m), 0);
  if (beforeChars <= capMax) return null; // 低于阈值：原样转发
  const stats = trimChatMessages(body, capMax, beforeChars);
  if (stats) return stats;
  // 消息级裁不掉（少量巨型消息，dropped<=0）：降级为消息内容内截断（Design I+）
  if (!chatTrimContentEnabled()) return null;
  const msgMax = chatTrimMsgMaxChars();
  let contentTruncated = 0;
  for (const m of msgs) {
    if (msgChars(m) > msgMax && truncateMsgContent(m, msgMax)) contentTruncated++;
  }
  if (contentTruncated === 0) return null; // 没有可截断的巨型消息，原样转发
  const afterChars = msgs.reduce((s, m) => s + msgChars(m), 0);
  const stats2 = trimChatMessages(body, capMax, afterChars);
  if (stats2) return stats2;
  return { before: msgs.length, after: msgs.length, dropped: 0, contentTruncated, beforeChars, afterChars };
}

// 消息级裁剪：冻结前缀(system+最早 non-system) + 固定占位符 + 滑动尾部；裁不掉返回 null
function trimChatMessages(body, capMax, beforeChars) {
  const keepOld = chatTrimKeepOldChars();
  const keepTail = chatTrimKeepTailChars();
  const msgs = body.messages;
  const systemMsgs = msgs.filter(m => m && m.role === 'system');
  const nonSystem = msgs.filter(m => !m || m.role !== 'system');
  if (nonSystem.length === 0) return null;
  const oldPart = takeMsgsByChars(nonSystem, keepOld);
  const tailPart = takeMsgsByChars(nonSystem.slice(oldPart.length).reverse(), keepTail).reverse();
  const trimmed = systemMsgs.concat(oldPart, [{ role: 'user', content: CHAT_TRIM_PLACEHOLDER }], tailPart);
  const dropped = msgs.length - trimmed.length;
  if (dropped <= 0) return null; // 裁不掉：交给上层（内容内截断）或原样转发
  body.messages = trimmed;
  return {
    before: msgs.length,
    after: trimmed.length,
    dropped,
    beforeChars,
    afterChars: trimmed.reduce((s, m) => s + msgChars(m), 0),
  };
}

const CHAT_TRIM_CONTENT_PLACEHOLDER = '【中间内容已省略：仅保留消息开头与结尾。】';

function chatTrimContentEnabled() {
  const v = process.env.VLLM_CHAT_TRIM_CONTENT;
  if (v === undefined || v === null || v === '') return true;
  return !/^(0|false|off|no)$/i.test(String(v));
}

function chatTrimMsgMaxChars() {
  const n = parseInt(process.env.VLLM_CHAT_KEEP_MSG_CHARS || '40000', 10);
  return (n >= 2000 && n <= 1000000) ? n : 40000;
}

// 单条消息内容原地截为 head+占位符+tail（总长 ≤ maxChars）；返回是否发生截断
function truncateMsgContent(m, maxChars) {
  const head = Math.floor(maxChars * 2 / 3);
  const tail = maxChars - head - CHAT_TRIM_CONTENT_PLACEHOLDER.length;
  if (!(tail >= 100 && head >= 100)) return false;
  const c = m && m.content;
  if (typeof c === 'string') {
    if (c.length <= maxChars) return false;
    m.content = c.slice(0, head) + CHAT_TRIM_CONTENT_PLACEHOLDER + c.slice(c.length - tail);
    return true;
  }
  if (Array.isArray(c)) {
    let changed = false;
    const parts = c.map((p) => {
      if (p && typeof p.text === 'string' && p.text.length > maxChars) {
        changed = true;
        return { ...p, text: p.text.slice(0, head) + CHAT_TRIM_CONTENT_PLACEHOLDER + p.text.slice(p.text.length - tail) };
      }
      return p;
    });
    if (changed) m.content = parts;
    return changed;
  }
  return false;
}

function startLiveReq(model, ip, stream, promptTokens, clientReqId) {  const ls = global.__liveStreams;
  const entry = {
    id: 'LIVE-' + (++ls.seq),
    model: model || '—',
    ip: ip || '—',
    stream: !!stream,
    startedAt: Date.now(),
    tokens: 0,
    text: '',
    done: false,
    doneAt: 0,
    finish: '',
    promptTokens: promptTokens || 0,  // 估算的 prompt token（SGLang 预填充进度条总量）
    tk: [],  // 逐秒采样 [{t, n=累计 token 估算}]：并发行「每行独立瞬时速度」数据源
    clientReqId: clientReqId || null, // 代理注入的 X-Client-Req-Id（v2 行↔rid 精确认领桥）
    rid: null,                        // tee 首帧解析出的引擎真实 rid（chatcmpl-xxx）
  };
  ls.map.set(entry.id, entry);
  return entry;
}

// 一段增量文本的 token 估算：CJK 字符≈0.7 tok/字，其余≈0.25 tok/字符
//（英文/代码约 4 字符/token）。09-16 实测校准：MTP 引擎一步一 delta 行携带
// 多个已接受 token（239 行 SSE = 600 真实 tokens），旧版按「行数」计数（每步
// +1）使齐步 decode 的并发各行数值完全相同（行速=步速），且单位虚低 ~2.5×。
function estimatePieceTokens(s) {
  let cjk = 0, other = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if ((c >= 0x3040 && c <= 0x9fff) || (c >= 0xac00 && c <= 0xd7af) ||
        (c >= 0xf900 && c <= 0xfaff) || (c >= 0xff00 && c <= 0xffef)) cjk++;
    else other++;
  }
  return cjk * 0.7 + other * 0.25;
}

function appendLiveText(live, s) {
  if (!s) return;
  if (!live.firstTokAt) live.firstTokAt = Date.now(); // 首 token 实时时刻（SGLang prefill 时长 = firstTokAt − startedAt，对齐 vLLM ttft 口径）
  live.tokens += estimatePieceTokens(s); // 09-16：按每行文本估 token 数（旧 +1/行 使并发各行速度齐步同值）
  live.text += s;
  if (live.text.length > LIVE_TEXT_MAX) live.text = live.text.slice(-LIVE_TEXT_MAX);
}

function finishLiveReq(live, finish) {
  if (!live || live.done) return;
  live.done = true;
  live.doneAt = Date.now();
  if (finish) live.finish = String(finish);
}

function liveStreamsSnapshot() {
  const now = Date.now();
  const ls = global.__liveStreams;
  const out = [];
  for (const [id, e] of ls.map) {
    if (e.done && now - e.doneAt > LIVE_DONE_KEEP_MS) { ls.map.delete(id); continue; }
    out.push({
      id: e.id,
      model: e.model,
      ip: e.ip,
      stream: e.stream,
      started_at: e.startedAt,
      elapsed_s: Math.round(((e.done ? e.doneAt : now) - e.startedAt) / 1000),
      tokens: e.tokens,
      done: e.done,
      finish: e.finish,
      text: e.text.length > 600 ? '…' + e.text.slice(-600) : e.text,
    });
  }
  out.sort((a, b) => a.started_at - b.started_at);
  return out;
}

// Sweep stale entries (finished long ago / hung in-flight > 30 min)
setInterval(() => {
  const now = Date.now();
  const ls = global.__liveStreams;
  for (const [id, e] of ls.map) {
    if ((e.done && now - e.doneAt > LIVE_DONE_KEEP_MS) ||
        (!e.done && now - e.startedAt > 30 * 60 * 1000)) {
      ls.map.delete(id);
    }
  }
}, 5000).unref();

// 逐秒采样 tee 条目的累计输出计数（滚动 ≤6 个样本）：并发请求每行独立
// 瞬时速度（computeConcurrencyDetails 里 REQ↔LIVE 绑定）的唯一数据源。
// 条目完成后清掉样本数组（行也随之消失，无绑定需求）。
setInterval(() => {
  const now = Date.now();
  const ls = global.__liveStreams;
  for (const [, e] of ls.map) {
    if (e.done) { if (e.tk) e.tk = null; continue; }
    if (!Array.isArray(e.tk)) e.tk = [];
    const last = e.tk[e.tk.length - 1];
    if (last && now - last.t < 500) continue; // 防同秒重复采样
    e.tk.push({ t: now, n: e.tokens });
    if (e.tk.length > 6) e.tk.shift();
  }
}, 1000).unref();

// Tee an SSE (streaming) response: forward raw bytes, parse delta text.
function teeSseStream(proxyRes, res, live, taskId) {
  let buf = '';
  proxyRes.on('data', (chunk) => {
    res.write(chunk); // transparent forward — client sees the original stream
    buf += chunk.toString('utf8');
    let nl;
    while ((nl = buf.indexOf('\n')) !== -1) {
      let line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      line = line.trim(); // tolerate \r\n
      if (!line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (!payload) continue;
      if (payload === '[DONE]') { finishLiveReq(live, live.finish || 'stop'); continue; }
      try {
        const obj = JSON.parse(payload);
        if (taskId && obj.id) linkRidTaskId(obj.id, taskId);
        if (obj.id && live && live.port) linkRidPort(obj.id, live.port);
        // v2 精确认领桥：首帧 rid（= vLLM req_id）→ clientReqId 登记，
        // REQ tracker 的 fill 行凭 taskId 反查 rid 直查引擎每请求序列。
        if (obj.id && live && live.clientReqId) { live.rid = obj.id; linkClientReqRid(live.clientReqId, obj.id); }
        if (obj.error) { finishLiveReq(live, 'error'); continue; }
        const ch = obj.choices && obj.choices[0];
        if (!ch) continue;
        const d = ch.delta || {};
        // this fork emits thinking text as delta.reasoning; upstream uses
        // delta.reasoning_content — accept both
        const piece = (d.content || '') + (d.reasoning_content || d.reasoning || '');
        if (piece) appendLiveText(live, piece);
        if (ch.finish_reason) live.finish = String(ch.finish_reason);
      } catch (e) { /* partial/garbled line — skip */ }
    }
  });
  proxyRes.on('end', () => {
    if (!live.done) finishLiveReq(live, live.finish || 'done');
    res.end();
  });
  proxyRes.on('error', () => {
    if (!live.done) finishLiveReq(live, 'error');
    try { res.end(); } catch (e) {}
  });
}

// Tee a non-streaming JSON response: forward bytes, capture final text.
function teeJsonStream(proxyRes, res, live, taskId) {
  let buf = '';
  let size = 0;
  proxyRes.on('data', (chunk) => {
    res.write(chunk);
    size += chunk.length;
    if (size < 8 * 1024 * 1024) buf += chunk.toString('utf8');
  });
  proxyRes.on('end', () => {
    res.end();
    if (buf) {
      try {
        const obj = JSON.parse(buf);
        if (taskId && obj.id) linkRidTaskId(obj.id, taskId);
        if (obj.id && live && live.port) linkRidPort(obj.id, live.port);
        if (obj.id && live && live.clientReqId) { live.rid = obj.id; linkClientReqRid(live.clientReqId, obj.id); }
        const ch = obj.choices && obj.choices[0];
        if (ch) {
          const piece = (ch.message && ch.message.content) || ch.text || '';
          if (piece) {
            live.text = piece.length > LIVE_TEXT_MAX ? piece.slice(-LIVE_TEXT_MAX) : piece;
            live.tokens = Math.max(live.tokens, Math.round(estimatePieceTokens(piece))); // 非流式：整段按文本估 token（09-16 与流式口径统一）
          }
          if (ch.finish_reason) live.finish = String(ch.finish_reason);
        }
      } catch (e) { /* not JSON — ignore */ }
    }
    if (!live.done) finishLiveReq(live, live.finish || 'done');
  });
  proxyRes.on('error', () => {
    if (!live.done) finishLiveReq(live, 'error');
    try { res.end(); } catch (e) {}
  });
}

// rid (vLLM chatcmpl id) → 控制台任务号（T 号）。代理 tee 响应流时记录，
// 供「最近完成请求」表格把每条记录关联到它的任务号。LRU 上限防堆积。
global.__ridTaskId = global.__ridTaskId || new Map();
function linkRidTaskId(rid, taskId) {
  if (!rid || !taskId) return;
  const m = global.__ridTaskId;
  if (m.size >= 5000) { const k = m.keys().next().value; if (k !== undefined) m.delete(k); }
  if (!m.has(rid)) m.set(rid, taskId); // 首写生效：rid 复用（概率可忽略）不覆盖既有认领
}
// 任务号 → clientReqId（行认领后凭 crid 查引擎 rid 的桥）。有界 FIFO。
if (!global.__taskCrid) global.__taskCrid = new Map();
function linkTaskCrid(taskId, crid) {
  if (!taskId || !crid) return;
  const m = global.__taskCrid;
  if (m.size >= 5000) { const k = m.keys().next().value; if (k !== undefined) m.delete(k); }
  if (!m.has(taskId)) m.set(taskId, crid);
}
// rid（响应 id）→ 实际转发端口（8889 代理 tee 层写入）。SGLang 多实例 metrics 共用同一
// 日志文件且记录不带 port，rid→port→gpu 是最近请求表唯一精确归因（vLLM 走 live-prefill rid→pid）
global.__ridPortMap = global.__ridPortMap || new Map();
function linkRidPort(rid, port) {
  if (!rid || !port) return;
  const m = global.__ridPortMap;
  if (m.size >= 20000) { const k = m.keys().next().value; if (k !== undefined) m.delete(k); }
  m.set(rid, port);
}
// port → 最近观察到的 GPU（stats 聚合实例时刷新；实例停止后该端口的记录仍可回溯归因）
global.__portGpuSeen = global.__portGpuSeen || new Map();
function rememberPortGpu(instList) {
  try {
    for (const i of instList || []) {
      if (i && i.port != null && i.gpu != null && i.gpu !== '—') global.__portGpuSeen.set(i.port, i.gpu);
    }
  } catch (e) {}
}

// ====== Core Proxy ======
function proxyToVllm(req, res, targetPath, targetPort, bufferedBody) {
  const base = targetPort ? `http://${config.vllmHost}:${targetPort}` : vllmBaseUrl;
  const targetUrl = new URL(targetPath, base);
  const options = {
    hostname: targetUrl.hostname,
    port: targetUrl.port,
    path: targetUrl.pathname + targetUrl.search,
    method: req.method,
    headers: { ...req.headers },
    timeout: 300000,
  };
  delete options.headers['host'];
  delete options.headers['content-length'];

  // Live-stream capture for generation endpoints (streaming & non-streaming)
  let live = null;
  let taskId = null;
  const fwdPort = parseInt(options.port) || config.vllmPort;
  if (req.method === 'POST' && bufferedBody &&
      LIVE_PATH_RE.test(targetPath.replace(/\?.*$/, ''))) {
    try {
      const b = JSON.parse(bufferedBody);
      live = startLiveReq(b.model, clientIp(req), !!b.stream, estimatePromptTokens(b),
        (b.headers && b.headers['X-Client-Req-Id']) || (b.headers && b.headers['x-client-req-id']) || null);
      // 异步用后端 /v1/messages/count_tokens 精确计算 prompt token（预填充进度总量用精确值）
      refreshPromptTokens(live, b, options.hostname, fwdPort);
    } catch (e) { live = null; }
    if (live) live.port = fwdPort; // rid→port 精确归因用（SGLang 多实例共用 metrics 日志，记录不带 port）
    // 分配控制台任务号（该请求在目标实例上 birth 时被 REQ tracker 认领）
    taskId = nextTaskId();
    taskForwardRegister(fwdPort, taskId, live ? live.clientReqId : null);
  }

  const proxyReq = http.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    if (live && live.stream) teeSseStream(proxyRes, res, live, taskId);
    else if (live) teeJsonStream(proxyRes, res, live, taskId);
    else proxyRes.pipe(res);
    // 响应结束 → 消费任务记录。快请求可能在两次 metrics 轮询之间跑完，
    // fill 循环来不及认领，不消费则记录滞留排队列表最多 120s（T35 类幽灵）。
    proxyRes.on('end', () => { if (taskId) taskForwardConsume(fwdPort, taskId); });
  });

  proxyReq.on('error', (err) => {
    if (live) finishLiveReq(live, 'proxy-error');
    if (err.code !== 'ECONNRESET') {
      console.error(`Proxy error: ${err.message}`);
    }
    if (!res.headersSent) {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Bad Gateway', message: err.message }));
    } else {
      // 上游在响应中途断开（SSE 流中断等）：客户端还在等数据，主动掐断连接，
      // 否则浏览器侧 fetch 会一直挂着直到其自身超时。
      try { res.destroy(); } catch (e) {}
    }
  });

  // options.timeout 只设置 socket 超时；不监听 timeout 事件的话，上游卡死
  // 5 分钟不响应时 socket 不会自动销毁，请求永久悬挂。超时后 destroy 触发 error 走上面分支。
  proxyReq.on('timeout', () => { try { proxyReq.destroy(); } catch (e) {} });

  res.on('close', () => {
    if (live && !live.done) {
      finishLiveReq(live, live.finish || 'client-closed');
      try { proxyReq.destroy(); } catch (e) {}
    }
    if (taskId) taskForwardConsume(fwdPort, taskId); // 客户端断开/异常收尾兜底
  });

  if (bufferedBody !== undefined) {
    proxyReq.end(bufferedBody);
  } else if (req.method === 'POST' || req.method === 'PUT' || req.method === 'PATCH') {
    req.pipe(proxyReq);
  } else {
    proxyReq.end();
  }
}

// ====== PD 两步转发 ======
// 辅助：向指定端口发 JSON POST，收集完整响应（非流式）
function httpJsonPost(host, port, path, bodyObj, timeoutMs) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify(bodyObj);
    const req = http.request({
      host, port, path, method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
      timeout: timeoutMs || 300000,
    }, (res) => {
      let data = '';
      res.on('data', (c) => { data += c; });
      res.on('end', () => resolve({ status: res.statusCode, body: data }));
      res.on('error', reject);
    });
    req.on('error', reject);
    req.on('timeout', () => { try { req.destroy(new Error('timeout')); } catch (e) {} });
    req.end(payload);
  });
}

// PD 两步转发入口：客户端 POST → STEP1 prefill（拿 token_ids + kv 参数）→ STEP2 decode（原样透传）
function proxyPdToVllm(req, res, targetPath, pd, bufferedBody, model) {
  const host = config.vllmHost;
  const pathOnly = targetPath.replace(/\?.*$/, '');
  const isGenPath = LIVE_PATH_RE.test(pathOnly);
  let body;
  try { body = JSON.parse(bufferedBody); } catch (e) {
    // body 解析失败：退回普通代理（打 decode 端口）
    return proxyToVllm(req, res, targetPath, pd.decode, bufferedBody);
  }
  const clientStream = !!body.stream;
  let live = null;
  let taskId = null;
  if (isGenPath) {
    try {
      live = startLiveReq(model, clientIp(req), clientStream, estimatePromptTokens(body),
        (body.headers && (body.headers['X-Client-Req-Id'] || body.headers['x-client-req-id'])) || null);
      refreshPromptTokens(live, body, host, pd.prefill); // 失败自动回落估算
    } catch (e) { live = null; }
    if (live) live.port = pd.decode; // PD 模式响应出自 decode 腿，GPU 归因记 decode 卡
    // prefill 腿注册任务号 → 两卡显示同 T 号；decode 腿延迟到 STEP2 真正转发时
    // 才注册（prefill 完成后），避免任务还没到 GPU1 就在那显示「排队」
    taskId = nextTaskId();
    taskForwardRegister(pd.prefill, taskId, live ? live.clientReqId : null);
  }
  const fail = (status, msg) => {
    if (live) finishLiveReq(live, 'pd-error');
    if (taskId) { taskForwardConsume(pd.prefill, taskId); taskForwardConsume(pd.decode, taskId); } // 失败兜底：双腿记录都清
    if (res.headersSent) { try { res.destroy(); } catch (e) {} return; }
    try {
      res.writeHead(status || 502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'PD proxy error', message: String(msg || '') }));
    } catch (e) {}
  };

  // STEP1: prefill 实例只做 prefill（强制非流式，注入 PD 参数）
  const prefillBody = {
    ...body,
    stream: false,
    max_tokens: Math.max(1, body.max_tokens || 1),
    return_token_ids: true,
    kv_transfer_params: { do_remote_decode: true },
  };
  // 流式专属字段在非流式 prefill 请求里会被 vLLM 拒绝（如 stream_options）
  delete prefillBody.stream_options;
  httpJsonPost(host, pd.prefill, pathOnly, prefillBody)
    .then((r1) => {
      if (r1.status !== 200) return fail(r1.status, (r1.body || '').slice(0, 500));
      let j1;
      try { j1 = JSON.parse(r1.body); } catch (e) { return fail(502, 'bad prefill response'); }
      const ids = j1.prompt_token_ids;
      const kvp = j1.kv_transfer_params || {};
      if (!Array.isArray(ids) || !kvp.remote_engine_id) {
        return fail(502, 'prefill response missing prompt_token_ids/kv_transfer_params');
      }
      // STEP2: decode 实例带透传参数直接生成，响应原样透传（含 SSE 流）
      const decodeBody = {
        ...body,
        stream: clientStream,
        kv_transfer_params: { ...kvp, do_remote_prefill: true, prompt_token_ids: ids },
      };
      const targetUrl = new URL(targetPath, `http://${host}:${pd.decode}`);
      const options = {
        hostname: targetUrl.hostname,
        port: targetUrl.port,
        path: targetUrl.pathname + targetUrl.search,
        method: 'POST',
        headers: { ...req.headers },
        timeout: 300000,
      };
      delete options.headers['host'];
      delete options.headers['content-length'];
      const proxyReq = http.request(options, (proxyRes) => {
        res.writeHead(proxyRes.statusCode, proxyRes.headers);
        if (live && live.stream) teeSseStream(proxyRes, res, live, taskId);
        else if (live) teeJsonStream(proxyRes, res, live, taskId);
        else proxyRes.pipe(res);
        // decode 腿响应结束 → 消费 decode 记录（快请求 fill 漏认领时防滞留）
        proxyRes.on('end', () => { if (taskId) taskForwardConsume(pd.decode, taskId); });
      });
      proxyReq.on('error', (err) => {
        if (live) finishLiveReq(live, 'proxy-error');
        if (!res.headersSent) {
          try { res.writeHead(502, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ error: 'Bad Gateway', message: err.message })); } catch (e) {}
        } else {
          try { res.destroy(); } catch (e) {}
        }
        if (taskId) { taskForwardConsume(pd.prefill, taskId); taskForwardConsume(pd.decode, taskId); }
      });
      proxyReq.on('timeout', () => { try { proxyReq.destroy(); } catch (e) {} });
      res.on('close', () => {
        if (live && !live.done) { finishLiveReq(live, live.finish || 'client-closed'); try { proxyReq.destroy(); } catch (e) {} }
        if (taskId) taskForwardConsume(pd.decode, taskId); // 客户端断开兜底
      });
      if (taskId) {
        taskForwardRegister(pd.decode, taskId, live ? live.clientReqId : null); // decode 腿：STEP2 真正转发时才注册
        taskForwardConsume(pd.prefill, taskId);  // prefill 腿已消费 → 任务从 GPU0 排队消失
      }
      proxyReq.end(JSON.stringify(decodeBody));
    })
    .catch((err) => fail(502, err && err.message));
}

// ====== 端口 → 真实模型文件路径（进程 cmdline 提取）======
// /v1/models 的 root 字段对 SGLang 是 served-model-name 而非文件路径，
// 仪表盘「模型路径」卡片需要真实 checkpoint 路径。这里从运行中实例的
// cmdline（vllm 取 serve 后第一参，sglang 取 --model-path）按端口查真实路径。
// listVllmInstances/listSglangInstances 各自带 3s 缓存，调用廉价。
function modelPathForPort(port) {
  try {
    for (const v of listVllmInstances()) {
      if (v.port === port && v.modelPath) return v.modelPath;
    }
    for (const s of listSglangInstances()) {
      if (s.port === port && s.modelPath) return s.modelPath;
    }
  } catch (e) {}
  return null;
}

// ====== Merged /v1/models across all vLLM backends ======
function handleMergedModels(res) {
  const ports = Array.from(new Set([config.vllmPort, ...Object.values(VLLM_MODEL_PORTS)]));
  let pending = ports.length;
  const all = [];
  let finished = false;
  const finish = () => {
    pending--;
    if (pending > 0) return;
    if (finished) return; // proxyRes 'end' 与 'error' 可能先后触发，防止重复收尾
    finished = true;
    if (res.writableEnded) return;
    // 模型名别名条目（2026-09-14）：客户端刷新模型列表时也能看到注册别名
    // （如 qwen3.8-flash-next-nvfp4）；请求侧会改写成真实 served 名再转发。
    const aliasItems = [];
    for (const a of Object.keys(MODEL_ALIASES)) {
      if (all.some(x => x.id === a)) continue;
      const src = all.find(x => x.id === MODEL_ALIASES[a]);
      if (src) aliasItems.push(Object.assign({}, src, { id: a, alias_of: MODEL_ALIASES[a] }));
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ object: 'list', data: all.concat(aliasItems) }));
  };
  ports.forEach((p) => {
    // 探测端口必须带超时：若该端口被 vLLM EngineCore 的内部 socket 占用（非 HTTP），
    // http.get 会连上但永远等不到响应，导致整个请求挂起。
    const req = http.get(`http://${config.vllmHost}:${p}/v1/models`, (proxyRes) => {
      let data = '';
      proxyRes.on('data', (c) => { data += c; });
      // 上游在响应中途断开：不监听会让 unhandled 'error' 打崩整个进程
      proxyRes.on('error', () => finish());
      proxyRes.on('end', () => {
        try {
          const j = JSON.parse(data);
          // 真实模型文件路径（vllm/sglang 进程 cmdline）。SGLang 的 /v1/models
          // root 字段填的是 served-model-name（如 qwen3.8-27b-0）而非文件路径，
          // 仪表盘「模型路径」据此补全为真实 checkpoint 路径。
          const realPath = modelPathForPort(p);
          (j.data || []).forEach((m) => {
            const item = { ...m, port: p };
            if (realPath) item.root = realPath;
            all.push(item);
          });
        } catch (e) {}
        finish();
      });
    });
    // destroy() 必须带错误参数：不带参数时不会 emit 'error'，下面的
    // req.on('error') 收尾逻辑永不执行 → 该端口的 pending 永不归零 → 整个
    // /v1/models 请求永久挂起（目标端口被 EngineCore 内部 socket 等非 HTTP
    // 进程占用时正是这种情况，实测挂起 >12s）。
    req.setTimeout(3000, () => req.destroy(new Error('probe timeout')));
    req.on('error', () => finish());
  });
}

// ====== Running model instances (multi-backend: vllm / sglang) ======
function findVllmPidByPort(port) {
  try {
    const { execSync } = require('child_process');
    // 关键修复：-sTCP:LISTEN 只取「监听该端口」的服务进程，并排除控制台自身。
    // 否则 lsof 会先命中控制台到该端口的 keep-alive 客户端 socket（本机进程），
    // 导致实例 PID 显示为控制台自身、GPU 显示 '?'、停止也停错对象。
    const out = execSync(`lsof -ti:${port} -sTCP:LISTEN 2>/dev/null || true`, { encoding: 'utf8', timeout: 3000 }).trim();
    const first = out.split('\n').find(l => parseInt(l.trim()) !== process.pid);
    if (first) return parseInt(first);
    // 兜底：chroot 内引擎进程属 root，ll 用户的 lsof 看不到其监听端口，
    // 改按 /proc/<pid>/cmdline 的 --port 匹配定位（cmdline 宿主可见）。
    // [sglang-adapt-1003] root 起的 sglang 同样看不见，且 vllm.entrypoints 兜底不认它 →
    // 先查 /proc 扫描的 sglang 实例表（listSglangInstances，3s 缓存）。
    try {
      const sgi = listSglangInstances().find(x => x.port === port);
      if (sgi && sgi.pid) return sgi.pid;
    } catch (e) {}
    // [vllm-page-1003] vLLM 侧同款兜底：本栈 vLLM 由 sudo(root) 启动 → ll 的 lsof 看不到
    // 监听端口；而下面的 pgrep 只认 vllm.entrypoints.*，匹配不到 0.30.0 实跑的
    // `vllm serve` CLI 形式 → pid=null → runtime=unknown / gpu=null（vLLM 标签页
    // 与模型管理页实例归因退化的根因）。listVllmInstances 走 /proc cmdline 扫描，
    // 与 SGLang 的 listSglangInstances 对称，root 进程同样可见。
    try {
      const vi = listVllmInstances().find(x => x.port === port);
      if (vi && vi.pid) return vi.pid;
    } catch (e) {}
    const pg = execSync('pgrep -f "[v]llm.entrypoints|[v]llm serve" 2>/dev/null || true', { encoding: 'utf8', timeout: 3000 }).trim();
    for (const pid of pg.split('\n').filter(Boolean)) {
      try {
        const cmd = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8').split('\0').join(' ');
        if (cmd.includes(`--port ${port}`)) return parseInt(pid);
        // 09-19：docker 容器进程按「映射后宿主端口」匹配（容器内 --port 8000 → 宿主 18420）
        const pm2 = cmd.match(/--port\s+(\d+)/);
        if (pm2 && mappedHostPort(parseInt(pid), parseInt(pm2[1])) === port) return parseInt(pid);
      } catch (e) {}
    }
    return null;
  } catch (e) { return null; }
}
function gpuIndexForPid(pid) {
  if (!pid) return null;
  try {
    const env = fs.readFileSync(`/proc/${pid}/environ`, 'utf8').split('\0');
    const v = env.find(e => e.startsWith('CUDA_VISIBLE_DEVICES='));
    return v ? v.split('=')[1] : null;
  } catch (e) { return null; }
}
function checkGpuConflict(gpuId, gpuCount, startPort) {
  // 检查请求的 GPU 区间 [gpuId, gpuId+gpuCount) 是否已被其他运行中实例占用
  // 返回冲突信息 {port, gpuId, gpuCount, runtime, model} 或 null
  const requested = [];
  for (let i = 0; i < gpuCount; i++) requested.push(gpuId + i);
  for (const [port, inst] of global.__GPU_INSTANCES) {
    if (port === startPort) continue;
    const occupied = [];
    for (let i = 0; i < inst.gpuCount; i++) occupied.push(inst.gpuId + i);
    for (const g of requested) {
      if (occupied.includes(g)) return inst;
    }
  }
  return null;
}
function getVllmInstances(ports, callback) {
  let pending = ports.length;
  const instances = [];
  let finished = false;
  const done = () => {
    pending--;
    if (pending > 0) return;
    if (finished) return; // 'end' 与 'error' 可能先后触发，防止 callback 被调用两次
    finished = true;
    callback(instances);
  };
  ports.forEach((p) => {
    // 同 handleMergedModels：探测必须带超时，防止 EngineCore 内部 socket 占端口时永久挂起
    const req = http.get(`http://${config.vllmHost}:${p}/v1/models`, (proxyRes) => {
      let data = '';
      proxyRes.on('data', (c) => { data += c; });
      proxyRes.on('error', () => done());
      proxyRes.on('end', () => {
        try {
          const j = JSON.parse(data);
          if (j.data && j.data.length) {
            let pid = findVllmPidByPort(p);
            let gpu = gpuIndexForPid(pid);
            let rt = detectRuntimeForPid(pid);
            let gpus = null;
            // [sglang-adapt-1003] root 起的 sglang：/proc/<pid>/environ 读不到（gpu=null）、
            // 或 pid 兜底未命中（runtime=unknown）→ 用 /proc cmdline 扫描的实例表补归因。
            if (!rt || gpu == null || !pid) {
              try {
                const sgi = listSglangInstances().find(x => x.port === p);
                if (sgi) {
                  rt = rt || 'sglang';
                  if (gpu == null) gpu = sgi.gpu != null ? String(sgi.gpu) : null;
                  if (!pid && sgi.pid) pid = sgi.pid;
                }
              } catch (e) {}
              // [vllm-page-1003] vLLM 侧兜底（对称 SGLang）：sudo(root) 启动 + `vllm serve`
              // CLI 形式让 pid/runtime/gpu 三空 → 「vLLM」标签页按 runtime==='vllm' 筛实例
              // 会一个都筛不到。listVllmInstances 从 /proc cmdline 拿 pid/gpu/gpus。
              try {
                const vi = listVllmInstances().find(x => x.port === p);
                if (vi) {
                  rt = rt || 'vllm';
                  if (gpu == null && vi.gpu != null) gpu = String(vi.gpu);
                  if (!pid && vi.pid) pid = vi.pid;
                  if (Array.isArray(vi.gpus) && vi.gpus.length) gpus = vi.gpus;
                }
              } catch (e) {}
            }
            instances.push({
              port: p,
              model: j.data[0].id,
              pid,
              gpu,
              gpus: gpus || (gpu != null && gpu !== '' ? [parseInt(gpu)] : null),
              runtime: rt || 'unknown',
              running: true,
            });
          }
        } catch (e) {}
        done();
      });
    });
    req.setTimeout(3000, () => req.destroy(new Error('probe timeout')));
    req.on('error', () => done());
  });
}
function portInUse(port) {
  try {
    const { execSync } = require('child_process');
    return !!execSync(`lsof -ti:${port} 2>/dev/null || true`, { encoding: 'utf8', timeout: 3000 }).trim();
  } catch (e) { return false; }
}

// ====== GPU Info ======
function getGpuInfo(callback) {
  // 10-07 测试修复：改走统一放弃式通道（gpuQuery）。原实现两处 spawn 的 5s 定时器只
  // SIGKILL 不结算——nvidia-smi 进 D 状态时 SIGKILL 无效、'close' 永不触发，done 到不了 2，
  // callback 永不触发 → stats 链条挂死，且每轮轮询都泄漏一个僵尸 nvidia-smi
  // （09-20 现场累积 43 个 D 状态进程正是这种泄漏的累积形态）。
  // gpuQuery 必定 settle（自建定时器放弃 + 连续失败熔断），callback 必然触发。
  const gpuInfo = { utilization: 0, total: 0, used: 0, power: 0, temperature: 0, temperature_mem: 0 };
  Promise.all([
    gpuQuery(['--query-compute-apps=pid,name,used_memory', '--format=csv'], GPU_Q_MS_LIVE),
    gpuQuery(['--query-gpu=index,utilization.gpu,memory.total,memory.used,power.draw,temperature.gpu,temperature.memory', '--format=csv'], GPU_Q_MS_LIVE),
  ]).then(([output, gpuOutput]) => {
    const processes = [];
    if (output && String(output).trim()) {
      String(output).trim().split('\n').forEach(line => {
        if (line.toLowerCase().includes('pid')) return;
        const parts = line.split(',').map(p => p.trim());
        if (parts.length >= 3) {
          const pid = parseInt(parts[0]);
          if (!isNaN(pid) && pid > 0) {
            processes.push({
              gpu_id: '0',
              pid: pid,
              name: (parts[1] || 'VLLM').replace(/[^\x20-\x7E]/g, ''),
              gpu_memory: parts[2] ? (parseInt(parts[2]) || 0) * 1024 * 1024 : 0,
            });
          }
        }
      });
    }
    let info = gpuInfo;
    if (gpuOutput && String(gpuOutput).trim()) {
      const lines = String(gpuOutput).trim().split('\n');
      let total = 0, used = 0, utilSum = 0, utilCount = 0, power = 0, temperature = 0, temperatureMem = 0;
      const perGpu = [];
      for (const line of lines) {
        if (line.toLowerCase().includes('index')) continue;
        const parts = line.split(',').map(p => p.replace(/[^\d.]/g, '').trim());
        if (parts.length >= 4) {
          total += parseInt(parts[2]) * 1024 * 1024 || 0;
          used += parseInt(parts[3]) * 1024 * 1024 || 0;
          utilSum += parseFloat(parts[1]) || 0;
          utilCount++;
          if (parts.length >= 5) power += parseFloat(parts[4]) || 0;
          if (parts.length >= 6) temperature = Math.max(temperature, parseFloat(parts[5]) || 0);
          if (parts.length >= 7) temperatureMem = Math.max(temperatureMem, parseFloat(parts[6]) || 0);
          perGpu.push({
            index: parts[0],
            utilization: parseFloat(parts[1]) || 0,
            total: (parseInt(parts[2]) || 0) * 1024 * 1024,
            used: (parseInt(parts[3]) || 0) * 1024 * 1024,
            power: parts.length >= 5 ? (parseFloat(parts[4]) || 0) : 0,
            temperature: parts.length >= 6 ? (parseFloat(parts[5]) || 0) : 0,
            // 09-29 显存温度（temperature.memory，GDDR/HBM；驱动不支持时为 0 → 前端隐藏）
            temperature_mem: parts.length >= 7 ? (parseFloat(parts[6]) || 0) : 0,
          });
        }
      }
      info = {
        utilization: utilCount ? Math.round(utilSum / utilCount) : 0,
        total, used, power, temperature,
        temperature_mem: temperatureMem,
        gpuCount: utilCount,
        perGpu,
      };
    }
    callback({ processes, gpu: info });
  }).catch(() => callback({ processes: [], gpu: gpuInfo }));
}

// ====== Metrics Parser ======
function parseMetrics(text) {
  const result = {};
  const lines = text.split('\n');
  for (const line of lines) {
    if (line.startsWith('#') || !line.trim()) continue;
    const idx = line.lastIndexOf(' ');
    if (idx === -1) continue;
    const metricPart = line.substring(0, idx).trim();
    const value = parseFloat(line.substring(idx + 1).trim());
    if (isNaN(value)) continue;
    let name = metricPart;
    let labels = {};
    const labelMatch = metricPart.match(/^(.+?)\{(.+)\}$/);
    if (labelMatch) {
      name = labelMatch[1];
      for (const kv of labelMatch[2].split(',')) {
        const eqIdx = kv.indexOf('=');
        if (eqIdx > 0) {
          const k = kv.substring(0, eqIdx);
          const v = kv.substring(eqIdx + 1).replace(/"/g, '');
          labels[k] = v;
        }
      }
    }
    const key = name + '|' + JSON.stringify(labels);
    result[key] = value;
  }
  return result;
}

// ====== Metrics namespace / 取值工具（vllm ↔ sglang 双运行时） ======
// parseMetrics 输出 key 形如 `metric_name|{labels}`。sglang 指标用 `sglang:` 前缀，
// vllm 用 `vllm:` 前缀，且同名指标的标签结构也略有差异。以下 helpers 用于按前缀读到值。
function metricsNamespace(m) {
  for (const k of Object.keys(m)) {
    if (k.startsWith('sglang:')) return 'sglang';
    if (k.startsWith('vllm:')) return 'vllm';
  }
  return 'vllm';
}
// Gauge/值类型：不限标签，取第一条 `base|` 开头的值（忽略标签结构）
function gaugeValue(m, base) {
  for (const k of Object.keys(m)) if (k.startsWith(base + '|')) return m[k];
  return 0;
}
// [sglang-adapt-1003] PP 双 stage 的 gauge 取值：sglang 每个 pp_rank 各注册一份 gauge，
// 且部分族（spec_accept_rate/spec_accept_length/spec_num_steps…）只登记在**末段 stage**
// （rank0 恒 0）——gaugeValue「取第一条」会拿到 rank0 的假 0，控制台投机命中率永远空。
// 正确口径：带 pp_rank 的序列取 max（副本同值不受影响，单段真值不再被 0 遮蔽）。
function gaugeValuePp(m, base) {
  let best = null;
  for (const k of Object.keys(m)) {
    if (k.indexOf(base + '|') !== 0) continue;
    let l = {};
    try { l = JSON.parse(k.substring(k.indexOf('|') + 1)); } catch (e) {}
    if (l && l.pp_rank != null) { if (best == null || m[k] > best) best = m[k]; continue; }
    return m[k]; // 无 pp_rank 维度：保持原语义（首条）
  }
  return best == null ? 0 : best;
}
// Counter 类：累加所有该指标（可能多标签）的值
function counterTotal(m, base) {
  let t = 0;
  for (const k of Object.keys(m)) if (k.startsWith(base + '|')) t += m[k];
  return t;
}
// 按标签值累加 counter（如 prompt_tokens_by_source_total 的 source 标签）：
// parseMetrics 把标签编码进 key（|{"engine":"0","model_name":"...","source":"..."}），
// 直接匹配 `"source":"<v>"` 子串即可精确过滤出该 source 的全部 label 组合。
function counterBySource(m, base, source) {
  const needle = '"source":"' + source + '"';
  let t = 0;
  for (const k of Object.keys(m)) {
    if (k.indexOf(base + '|') === 0 && k.indexOf(needle) !== -1) t += m[k];
  }
  return t;
}
// 按任意标签值累加 counter（parseMetrics 把标签编码进 key 的 JSON 里）
function counterByLabel(m, base, label, value) {
  const needle = '"' + label + '":"' + value + '"';
  let t = 0;
  for (const k of Object.keys(m)) {
    if (k.startsWith(base + '|') && k.indexOf(needle) !== -1) t += m[k];
  }
  return t;
}
// Histogram：取 _count / _sum（忽略 _bucket），按标签名独立累加
function histogramSumCount(m, base) {
  // 10-07 测试修复：原实现用 `=` 赋值，同一指标族有多个标签序列（多引擎 / PP 分段 /
  // 多 model_name）时只保留最后一条，sum/count 双双少算（单元测试实锤：两条序列
  // {100,10}+{70,7} 得 {70,7} 而非 {170,17}）。直方图的 _sum/_count 本就该跨标签累加，
  // 与同文件 counterTotal 的口径一致。当前线上每族恰好只有一条序列，属潜伏 bug。
  let sum = 0, count = 0;
  for (const k of Object.keys(m)) {
    if (k.indexOf(base + '_count') === 0) count += m[k];
    else if (k.indexOf(base + '_sum') === 0) sum += m[k];
  }
  return { sum, count };
}

// ====== SGLang 仪表盘统计 ======
// sglang 的 /metrics 指标命名/结构（sglang: 前缀）与 vLLM 差异较大，这里用独立构建器
// 产出与前端 `/v1/internal/stats` 兼容的结果结构（vllm 路径保持不变）。
function buildSglangStats(m, ticker) {
  // [sglang-adapt-1003] 改用 gaugeValuePp：PP2 下 spec_* 等 gauge 只在末段登记（rank0 恒 0），
  // 旧 gaugeValue 取首条 → 投机命中率等永远显示空/0。
  const g = (base) => gaugeValuePp(m, base);
  const hist = (base) => histogramSumCount(m, base);
  const cnt = (base) => counterTotal(m, base);

  const ttft = hist('sglang:time_to_first_token_seconds');
  const tpot = hist('sglang:inter_token_latency_seconds');
  const e2e = hist('sglang:e2e_request_latency_seconds');

  const genTokens = cnt('sglang:generation_tokens_total');
  const promptTokens = cnt('sglang:prompt_tokens_total');
  const cachedTokens = Math.round(counterByLabel(m, 'sglang:prefill_effective_tokens_total', 'mode', 'device_hit')) + Math.round(counterByLabel(m, 'sglang:prefill_effective_tokens_total', 'mode', 'host_hit'));
  const uncached = Math.max(0, promptTokens - cachedTokens);

  const running = Math.round(g('sglang:num_running_reqs'));
  const queue = Math.round(g('sglang:num_queue_reqs'));

  // 生成吞吐（tg TPS）：直接用引擎实时 gauge sglang:gen_throughput。
  // 注意不能用 generation_tokens_total 差值——sglang 该 counter 是请求完成时才批量
  // 计入（请求期间恒定、结束瞬间一次 +N），差值法会显示 0 或脉冲尖峰。
  const ls = (ticker && ticker.lastSecond) || { speed: 0, promptSpeed: 0, tokens: 0, running: 0, at: 0 };
  const genTps = g('sglang:gen_throughput');
  // 预填充吞吐：引擎实时 prefill_effective_tokens_total{mode=input} 的 3s 差值。
  // （prompt_tokens_total 是 prefill 完成时批量计入，ticker 差值法不可用）
  const ppSpeed3 = buildSglangPrefillSpeed3s(m);
  const ppTps = ppSpeed3 !== undefined ? ppSpeed3 : (ls.promptSpeed > 0 ? ls.promptSpeed : 0);

  // 投机解码命中率（DFLASH/MTP）：spec_accept_rate 为 0~1 比例
  const acceptRate = g('sglang:spec_accept_rate');

  // 峰值显存（KV + 权重）
  const kvGb = g('sglang:kv_cache_memory_usage_gb');
  const wGb = g('sglang:weight_memory_usage_gb');

  // KV 缓存占用（0~1）
  const tokenUsage = g('sglang:token_usage');

  const out = {
    cache_blocks: { used: Math.min(100, tokenUsage * 100), total_blocks: Math.max(1, Math.round(g('sglang:max_total_num_tokens') / Math.max(1, g('sglang:page_size')))), blocks: null },
    kv_cache: {
      usage_pct: g('sglang:max_total_num_tokens') > 0 ? Math.min(1, g('sglang:kv_used_tokens') / g('sglang:max_total_num_tokens')) : 0,
      max_tokens: Math.round(g('sglang:max_total_num_tokens')),
      used_tokens: Math.round(g('sglang:kv_used_tokens')),
      block_size: Math.round(g('sglang:page_size')) || 0,
      num_gpu_blocks: Math.max(1, Math.round(g('sglang:max_total_num_tokens') / Math.max(1, g('sglang:page_size')))),
    },
    num_running_seqs: running,
    num_queue_seqs: queue,
    running,
    total_completed_requests: Math.round(cnt('sglang:num_requests_total')),
    total_input_tokens: promptTokens,
    cached_input_tokens: cachedTokens,
    uncached_input_tokens: uncached,
    total_output_tokens: genTokens,
    cache_hit_rate: promptTokens > 0 ? ((cachedTokens / promptTokens) * 100).toFixed(1) : 0,
    avg_ttft_ms: ttft.count > 0 ? (ttft.sum / ttft.count * 1000) : 0,
    avg_tpot_ms: tpot.count > 0 ? (tpot.sum / tpot.count * 1000) : 0,
    bench_pp_tps: parseFloat(ppTps.toFixed(1)),
    bench_tg_tps: parseFloat(genTps.toFixed(1)),
    bench_e2e_ms: e2e.count > 0 ? (e2e.sum / e2e.count * 1000) : 0,
    bench_throughput: parseFloat((genTps + ppTps).toFixed(1)),
    bench_peak_mem_gb: parseFloat((kvGb + wGb).toFixed(2)),
    active_requests: [],
    total_speed: genTps,
    avg_speed_per_request: genTps,
    avg_prefill_tokens_per_request: 0,
    avg_prefill_time_seconds: 0,
    avg_decode_time_seconds: 0,
    avg_tokens_per_request: 0,
    mtp_hit_rate: acceptRate > 0 ? parseFloat((acceptRate * 100).toFixed(1)) : undefined,
    last_second: ls,
    energy: (typeof buildEnergyInfo === 'function') ? buildEnergyInfo() : undefined,
  };
  return out;
}

// ====== SGLang 日志 prefill 命中（每请求 cached token 真值，08-30 vLLM 口径对齐）======
// SGLang 日志 "Prefill batch, #new-seq, #new-token, #cached-token, ... #pending-token" 行的
// #cached-token = 该请求 prefill 的真实缓存命中数（= vLLM num_cached_tokens 口径，引擎
// 逐 chunk 处理时第一 chunk 行即给出）。引擎累计 counter（prefill_effective_tokens_total
// 的 device_hit/input）是进程级累计，用全局累计命中率摊算单请求命中对新请求误差很大
// （实测 4414-token 新请求真值 0 命中，全局 14.8% 估算会错减 662）。
// 日志发现：/proc/<pid>/fd/{1,2} readlink（30s TTL），只解析 Prefill batch 行，保留最近 400 行。
const __sglLogState = new Map(); // port -> {path, pos, lines:[{tMs,newSeq,newToken,cached,pending}], lastPathAt}

function readSglangPrefillLines(port, pid) {
  const now = Date.now();
  let st = __sglLogState.get(port);
  if (!st) { st = { path: null, pos: 0, lines: [], lastPathAt: 0 }; __sglLogState.set(port, st); }
  if (!st.path || now - st.lastPathAt > 30000) {
    st.lastPathAt = now;
    let p = null;
    if (pid) for (const fd of [1, 2]) {
      try {
        const q = fs.readlinkSync(`/proc/${pid}/fd/${fd}`);
        if (/\.log$/.test(q) && fs.existsSync(q)) { p = q; break; }
      } catch (e) {}
    }
    if (p !== st.path) { st.path = p; st.pos = 0; st.lines = []; }
  }
  if (!st.path) return st.lines;
  try {
    const size = fs.statSync(st.path).size;
    if (size < st.pos) st.pos = 0; // 日志被截断/轮转
    if (size > st.pos) {
      const fd = fs.openSync(st.path, 'r');
      const len = Math.min(size - st.pos, 2 * 1024 * 1024);
      const buf = Buffer.alloc(len);
      fs.readSync(fd, buf, 0, len, st.pos);
      fs.closeSync(fd);
      st.pos += len;
      for (const ln of buf.toString('utf8').split('\n')) {
        const m = ln.match(/^\[(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+),.*#pending-token: (\d+)/);
        if (!m) continue;
        // 日志时间戳=服务器本地时间；控制台同机运行，按本地时区解析
        st.lines.push({ tMs: new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]).getTime(), newSeq: +m[7], newToken: +m[8], cached: +m[9], pending: +m[10] });
      }
      if (st.lines.length > 400) st.lines.splice(0, st.lines.length - 400);
    }
  } catch (e) {}
  return st.lines;
}

// 查找某请求的每请求缓存命中真值（组首行 #cached-token），未匹配返回 undefined（回落全局估算）。
// 组首行 = 该请求 prefill 的第一行：前 3s 内没有其他 Prefill batch 行（并发交织时无法
// 区分组 → 宁可回退也不误配）。匹配依据：
//   ① 时间窗 [startedAt−8s, +30s]（代理+模板+调度开销在 8s 内）；
//   ② 组首行 (pending+cached) ≈ 请求全量 token（count_tokens 口径，±25%/512 容差——
//      组首行 pending 含在途 chunk 与排队请求，容差吸收模板开销与少量排队量）。
function sglangCachedForRequest(lines, e, promptTokens) {
  if (!promptTokens || !lines.length) return undefined;
  const start = e.startedAt || 0;
  for (let i = 0; i < lines.length; i++) {
    const L = lines[i];
    if (L.newSeq !== 1) continue;
    const prev = i > 0 ? lines[i - 1] : null;
    if (prev && (L.tMs - prev.tMs) < 3000) continue; // 非组首（交织）
    if (L.tMs < start - 8000 || L.tMs > start + 30000) continue;
    const fullEst = L.pending + L.cached;
    if (Math.abs(fullEst - promptTokens) <= Math.max(512, promptTokens * 0.25)) return L.cached;
  }
  return undefined;
}

// ===== SGLang 逐请求缓存命中环（09-01）：从实例日志解析 "Prefill batch" 行的 #cached-token，
// 为「0命中占比」徽标补上 SGLang 侧逐请求口径（request-traces.jsonl 由 vLLM 插件写入，
// 生产全切 SGLang 后停更，徽标此前对 SGLang 流量失明）。
// 分组口径与 sglangCachedForRequest 一致：#new-seq>=1 且距上一行 >=3s 的行=新请求起点
//（max-running-requests=1 时 chunked prefill 各 chunk 行 1s 间隔连续，不会误切）。
// 进行中的请求存 st.open 跨增量读保持（100k+ 长 prefill 跨多次扫描不断成两条）；
// 首次见到日志只 tail 2MB 作 1h 窗口种子，之后增量读；每 30s 重新探测 /proc/pid/fd。
const __sglTraceState = new Map(); // port -> {path,pos,open,lastMs,ring,}
function sglangTraceRecords() {
  let instances = [];
  try { instances = listSglangInstances(); } catch (e) {}
  const now = Date.now();
  const cutoffMs = now - 3600 * 1000;
  for (const inst of instances) {
    const port = inst.port, pid = inst.pid;
    let st = __sglTraceState.get(port);
    if (!st) { st = { path: null, pos: 0, open: null, lastMs: null, ring: [] }; __sglTraceState.set(port, st); }
    if (!st._pathAt || now - st._pathAt > 30000) {
      st._pathAt = now;
      let p = null;
      if (pid) for (const fd of [1, 2]) {
        try { const q = fs.readlinkSync(`/proc/${pid}/fd/${fd}`); if (/\.log$/.test(q) && fs.existsSync(q)) { p = q; break; } } catch (e) {}
      }
      if (p !== st.path) {
        st.path = p; st.open = null; st.lastMs = null; st.ring = [];
        try { st.pos = p ? Math.max(0, fs.statSync(p).size - 2 * 1024 * 1024) : 0; } catch (e) { st.pos = 0; } // tail 种子
      }
    }
    if (!st.path) continue;
    try {
      const size = fs.statSync(st.path).size;
      if (size < st.pos) { st.pos = Math.max(0, size - 2 * 1024 * 1024); st.open = null; st.lastMs = null; } // 日志截断/轮转
      if (size > st.pos) {
        const fd = fs.openSync(st.path, 'r');
        const len = Math.min(size - st.pos, 2 * 1024 * 1024);
        const buf = Buffer.alloc(len);
        fs.readSync(fd, buf, 0, len, st.pos);
        fs.closeSync(fd);
        st.pos += len;
        for (const ln of buf.toString('utf8').split('\n')) {
          // 行格式两种：新 build "#new-seq: 1, #new-token: 80, #cached-token: 0"；
          // 旧 build 中间多 "#full-token: 512"——做成可选组兼容两种
          const m = ln.match(/^\[(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\] Prefill batch, #new-seq: (\d+)(?:, #full-token: \d+)?, #new-token: (\d+), #cached-token: (\d+)/);
          if (!m) continue;
          const tMs = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]).getTime();
          if (tMs < cutoffMs) continue;
          const ns = +m[7], nt = +m[8], ct = +m[9];
          if (ns >= 1 && (st.lastMs === null || tMs - st.lastMs >= 3000)) {
            if (st.open) st.ring.push(st.open);
            st.open = { tMs, prompt: 0, cached: 0, port };
          }
          st.lastMs = tMs;
          if (st.open) { st.open.prompt += nt; st.open.cached += ct; }
        }
      }
    } catch (e) {}
    if (st.ring.length > 5000) st.ring.splice(0, st.ring.length - 5000);
  }
  const out = [];
  for (const st of __sglTraceState.values())
    for (const r of st.ring) if (r.tMs >= cutoffMs) out.push(r);
  return out;
}

// ====== SGLang 并发请求明细（active_requests） ======
// sglang 的 /metrics 只有聚合值，没有每请求明细；但走控制台 8889 代理的请求会被
// teeSseStream 捕获（global.__liveStreams），这里把「进行中的请求」转成与 vLLM
// active_requests 同构的行（速度/阶段/耗时/进度），前端「并发请求」卡片即可像
// vLLM 一样逐请求展示。直连后端端口（不经 8889）的请求没有 live 数据，会回落到
// 聚合显示（"正在处理 N 个请求"），不影响 vLLM 路径。
//
// 「输出 token/s」口径（2026-08-25 用户要求与 tg TPS 对齐）：单请求输出速度不再用
// tee 的 chunk 计数（MTP 时一个 chunk 多个 token 只算 1，偏低），改用引擎实时 gauge
// sglang:gen_throughput（与仪表盘「生成吞吐」同源，每步更新），按各 decode 请求的
// live token 占比分摊：单请求时占比=1，直接等于全局生成吞吐（准确）。
function buildSglangActiveRequests(sg, liveStreams, genSpeed1s, prefillSpeed3s, ppTotalNow, getCached) {
  const now = Date.now();
  const out = [];
  if (!liveStreams || !liveStreams.map) return out;
  // 第一遍：收集进行中请求，算上一秒速度（meta 行小字，live 计数口径），区分 decode/prefill，
  // 累计 decode 请求的 live token 作为吞吐分摊权重
  const rows = [];
  let decodeTokens = 0;
  for (const e of liveStreams.map.values()) {
    if (e.done) continue;                     // 只看进行中
    // 惰性初始化「上一秒 token 采样」基线（首次采样 tok_last_sec=0 → 前端显示 --）
    if (e._lastTok === undefined) { e._lastTok = e.tokens; e._lastAt = now; }
    // dt 单位是秒；毫秒差要先除 1000（Math.max(1, ms)/1000 会得到 0.001s 恒小于阈值）。
    // dt 不足（多个 stats 客户端 500ms 轮询同频共振）时不重置基线，delta 累积到下次采样；
    // 同时沿用上次有效速度，避免撞车时页面速度闪烁为 0。
    const dt = (now - e._lastAt) / 1000;
    let tokLastSec = 0;
    if (dt >= 0.3) {
      tokLastSec = Math.max(0, (e.tokens - e._lastTok) / dt);
      e._lastTok = e.tokens;
      e._lastAt = now;
      e._lastSpeed = tokLastSec;
    } else {
      tokLastSec = e._lastSpeed || 0;
    }
    const elapsed = Math.max(1, (now - e.startedAt) / 1000);
    const isDecode = (e.text || '').length > 0;  // 已有输出文本 → decode；否则还在预填充
    if (isDecode) decodeTokens += (e.tokens || 0);
    rows.push({ e, tokLastSec, elapsed, isDecode });
  }
  // 第二遍：构建输出行
  // 预填充数口径对齐 vLLM（fillActiveRequests 判定逻辑）：
  //  ① 总量 = 精确 prompt − 缓存命中。缓存命中不计入预填充工作量（vLLM 口径
  //     prompt−cached）；命中优先取每请求日志 "Prefill batch" 行 #cached-token 真值
  //     （getCached，vLLM num_cached_tokens 同源口径），未匹配回落引擎总命中率近似
  //     （cache_hit_rate = 命中/(命中+实算)）。
  //  ② 进行中已处理：单请求 prefill = 引擎计数增量（最准）；并发 prefill =
  //     vLLM 守恒分摊（引擎实时预填充吞吐/并发数 × 已耗时）——旧实现把全局增量
  //     同时记给每行，并发时双记虚高；总量旧实现=全量 prompt（未减命中）。
  //  ③ 进入 decode 后 prefill 已完成 → 总量/进度冻结为最终值（旧实现直接归 0，
  //     进度瞬间塌掉），prefill_s 用首 token 实时值（tee 捕获 firstTokAt），
  //     缺实时数据时回落采样精度（elapsed − decode 段时长）。
  const decodeCount = rows.filter(r => r.isDecode).length;
  const numPrefill = rows.length - decodeCount;
  for (const r of rows) {
    const { e, tokLastSec, elapsed, isDecode } = r;
    // 引擎实测输出速度：总生成吞吐 × 该请求生成占比（单请求占比=1 → 直接等于全局生成吞吐）
    let engineSpeed = undefined;
    if (isDecode) {
      const share = decodeTokens > 0
        ? ((e.tokens || 0) / decodeTokens)
        : (decodeCount > 0 ? 1 / decodeCount : 0);
      if (genSpeed1s !== undefined && genSpeed1s >= 0) engineSpeed = genSpeed1s * share;
    }
    // —— 预填充数（vLLM 口径）——
    // 08-30 修复：缓存命中优先用每请求日志 #cached-token 真值（= vLLM num_cached_tokens
    // 口径，Prefill batch 行第一 chunk 即给出）；未匹配回退全局累计命中率估算（保守上限）。
    // 旧实现一律用全局命中率摊算：新请求（真值 0 命中）会被错减估算值（实测 14.8%×4414=662），
    // 总量虚低、进度虚高——与 vLLM 路径（prompt_total − cached 实测）口径不一致。
    const prompt = e.promptTokens || 0;
    const cachedExact = (typeof getCached === 'function') ? getCached(e) : undefined;
    const hitRate = parseFloat(sg.cache_hit_rate) || 0;
    const cachedEst = prompt > 0 && hitRate > 0 ? Math.min(prompt, Math.round(prompt * hitRate / 100)) : 0;
    const cached = (cachedExact !== undefined && cachedExact >= 0) ? Math.min(cachedExact, prompt) : cachedEst;
    const totalNow = Math.max(0, prompt - cached);
    let ppTotal, ppDone, ppS = 0;
    if (isDecode) {
      // prefill 已完成：冻结（避免命中率波动抖动已展示的数；total 用首次观察值）
      if (e._ppTotal === undefined) e._ppTotal = totalNow;
      ppTotal = e._ppTotal;
      if (e._ppDone === undefined) e._ppDone = ppTotal;
      ppDone = e._ppDone;
      if (e._ppS === undefined) {
        // 首 token 实时值（ttft）；缺失时采样精度（vLLM 同款回落：elapsed − decode 段）
        const decSpeed = ((engineSpeed !== undefined && engineSpeed > 0) ? engineSpeed : (tokLastSec > 0 ? tokLastSec : 10));
        e._ppS = e.firstTokAt
          ? Math.max(0.1, (e.firstTokAt - e.startedAt) / 1000)
          : Math.max(0.5, elapsed - (e.tokens || 0) / decSpeed);
      }
      ppS = e._ppS;
    } else {
      ppTotal = totalNow;
      let d = 0;
      if (ppTotalNow !== undefined && numPrefill > 0) {
        // 并发数变化时重设引擎基线（增量只有在单请求状态下才可全部归属该请求）
        if (e._ppMode !== numPrefill || e._ppBase === undefined) {
          e._ppBase = ppTotalNow;
          e._ppMode = numPrefill;
        }
        if (numPrefill === 1) {
          // 单请求 prefill：全局增量 = 该请求真实预填充（引擎计数，最准）
          d = Math.max(0, ppTotalNow - (e._ppBase || 0));
        } else {
          // 并发 prefill：vLLM 守恒分摊（引擎 3s 预填充吞吐/并发数 × 已耗时）
          const perSec = (prefillSpeed3s !== undefined && prefillSpeed3s > 0) ? (prefillSpeed3s / numPrefill) : 0;
          d = perSec * elapsed;
        }
      }
      // 单调不减（重设基线/估算抖动时进度条不回退）
      ppDone = Math.min(ppTotal, Math.max(e._ppDone || 0, d));
      e._ppDone = ppDone;
    }
    // 每请求预填充速度（vLLM 口径）：引擎 3s 实时预填充吞吐按 prefill 并发数分摊
    const ppTps = (prefillSpeed3s !== undefined && prefillSpeed3s > 0)
      ? prefillSpeed3s / Math.max(1, numPrefill)
      : (sg.bench_pp_tps || 0);
    out.push({
      id: e.id.replace(/^LIVE-/, ''),
      ip: e.ip || '—',
      elapsed_s: Math.round(elapsed),
      tokens_generated: e.tokens,
      phase: isDecode ? 'decode' : 'prefill',
      // decode 行大数字 tok/s：引擎实测分摊值，回落上一秒/全程均值
      speed: isDecode
        ? (engineSpeed !== undefined ? parseFloat(engineSpeed.toFixed(1)) : (tokLastSec > 0 ? tokLastSec : parseFloat((e.tokens / elapsed).toFixed(1))))
        : 0,
      avg_speed: engineSpeed !== undefined ? parseFloat(engineSpeed.toFixed(1)) : parseFloat((e.tokens / elapsed).toFixed(1)),
      // 前端 decode 行大数字优先用此值（标签「近3s tok/s」）：引擎实测总生成吞吐分摊
      avg_speed_3s: engineSpeed !== undefined ? parseFloat(engineSpeed.toFixed(1)) : undefined,
      // sglang 无每请求实时数据源，本行值是实例吞吐按 token 占比分摊 → 如实标 share
      speed_src: 'share',
      tok_last_sec: Math.round(tokLastSec),
      // 预填充速度：引擎实时 prefill_effective_tokens_total{mode=input} 3s 差值，
      // 按 prefill 并发数分摊到单请求（prompt_tokens_total 批量计入不可用）
      prefill_tps: isDecode ? 0 : parseFloat(ppTps.toFixed(1)),
      // 预填充进度（口径与 vLLM 一致）：
      //  总量 = 精确 prompt token（count_tokens；未返回前字符估算）− 缓存命中
      //         （引擎总命中率近似，vLLM 同源口径）；进入 decode 后冻结
      //  已处理 = 单请求 prefill 期间引擎实时计数增量（最准）；并发 prefill 守恒
      //         分摊；进入 decode 后 = 总量（冻结）
      prefill_total_uncached: ppTotal,
      prefill_done_uncached: ppDone,
      prefill_exact: false,
      prefill_s: ppS,
    });
  }
  // 与前端 liveList 按 startedAt 升序 zip（同一条请求的 token 流对应同一行）；
  // 先开始的请求 elapsed 更大，故按 elapsed 降序 = startedAt 升序
  out.sort((a, b) => b.elapsed_s - a.elapsed_s);
  return out;
}

// 引擎实测总预填充吞吐（3s 平滑）：
// 数据源 = sglang:prefill_effective_tokens_total{mode="input"} —— 该 counter 是 prefill
// 每步实时增长的（实测每 2s +4k~7k），而 prompt_tokens_total 是 prefill 完成时才批量
// 计入（不可用于实时速度）。3s 窗口差值 ÷3 得实时预填充 tok/s（线性插值估算 3 秒前累计）。
function buildSglangPrefillSpeed3s(m, hist) {
  if (hist === undefined) hist = (hist = hist || []);
  const now = Date.now();
  const ppTotal = Math.round(counterByLabel(m, 'sglang:prefill_effective_tokens_total', 'mode', 'input'));
  hist.push({ t: now, pp: ppTotal });
  const cutoff = now - 3000;
  while (hist.length > 3 && hist[1].t <= cutoff) hist.shift();
  if (hist.length < 2) return undefined;
  const first = hist[0], last = hist[hist.length - 1];
  let base = first;
  if (first.t < cutoff && hist[1].t > cutoff) {
    const a = first, b = hist[1];
    const frac = (cutoff - a.t) / (b.t - a.t);
    base = { t: cutoff, pp: a.pp + (b.pp - a.pp) * frac };
  }
  const span = (last.t - base.t) / 1000;
  if (span < 0.5) return undefined;
  return Math.max(0, (last.pp - base.pp) / 3);
}

// 引擎实测总生成吞吐（1s 平滑，与 tg TPS 同源）：
// 数据源 = sglang:gen_throughput gauge（引擎实时生成 tok/s，每步更新）。
// 1 秒窗口内时间加权平均（不除以 3：窗口就是 1s，平均速率 ≈ 该秒的 token 数）。
function buildSglangGenSpeed1s(m, hist) {
  if (hist === undefined) hist = (hist = hist || []);
  const now = Date.now();
  const thr = gaugeValue(m, 'sglang:gen_throughput');
  hist.push({ t: now, thr });
  const cutoff = now - 1000;
  // 保留至少 3 个样本；[0] 保持为「跨界左点」（t ≤ cutoff）供插值
  while (hist.length > 3 && hist[1].t <= cutoff) hist.shift();
  if (hist.length < 2) return undefined;
  const last = hist[hist.length - 1];
  // 窗口起点：跨界线性插值出 cutoff（now-1s）时刻的吞吐值
  let base = hist[0];
  if (hist[0].t < cutoff && hist[1].t > cutoff) {
    const a = hist[0], b = hist[1];
    const frac = (cutoff - a.t) / (b.t - a.t);
    base = { t: cutoff, thr: a.thr + (b.thr - a.thr) * frac };
  }
  const spanS = (last.t - base.t) / 1000;
  if (spanS < 0.3) return undefined;
  // 时间加权（梯形积分）：每段的 token = 首尾吞吐均值 × 段时长，累计 = 窗口内总 token
  let totalTok = 0;
  let prev = base;
  for (const s of hist) {
    if (s.t <= prev.t) continue;
    totalTok += ((prev.thr + s.thr) / 2) * (s.t - prev.t) / 1000;
    prev = s;
  }
  // 窗口 ≈1s：平均速率 = 总 token ÷ 窗口时长（不再固定 ÷3）
  return Math.max(0, totalTok / spanS);
}

// ====== Request-level speed tracking ======
// vLLM only exposes aggregate metrics (num_requests_running,
// generation_tokens_total, histogram buckets). We estimate per-request
// phase (prefill vs decode) using cumulative tracking.
//
// Heuristic:
//   - If running increased since last poll: new requests are in prefill
//   - If totalSpeed > 0: some requests are producing tokens (decode)
//   - If totalSpeed === 0 but running > 0: all requests are still prefilling
//


// ====== Stats reset baseline ======
// vLLM metrics are cumulative counters / histograms for the model's whole
// lifetime. "Reset stats" captures their current values as a baseline, and
// the dashboard displays (raw - baseline), i.e. values since last reset.

// 直扫 /proc 找推理进程。不能用 pgrep -f：它会匹配到 ssh/bash/expect 等探测命令
// **自身**（命令行里恰好含 "vllm serve"/"sglang.launch_server" 字样），日志源会随
// 「谁在 ssh 上敲了什么」漂移。必须直扫并过滤探测类进程。
function scanInferenceProcs() {
  const out = { vllm: [], sglang: [] };
  let names = [];
  try { names = fs.readdirSync('/proc'); } catch (e) { return out; }
  for (const name of names) {
    if (!/^\d+$/.test(name)) continue;
    const pid = parseInt(name, 10);
    if (pid === process.pid) continue;
    let cmd = '';
    try {
      cmd = fs.readFileSync('/proc/' + name + '/cmdline', 'utf8').split('\0').join(' ').trim();
    } catch (e) { continue; }
    if (!cmd) continue;
    // 探测类进程排除（ssh 命令行、bash -c 包装、grep/pgrep/ps 自身）
    if (/(^|\s)(ssh|expect|grep|pgrep|ps|scp|rsync|tail|cat)(\s|$)/.test(cmd)) continue;
    if (/\b(?:bash|sh)\s+-c\b/.test(cmd)) continue;
    // 09-19：改用 isVllmProcCmd 统一判定——旧正则 `(^|\s)vllm\s+serve` 匹配不到
    // 容器栈的 `/usr/local/bin/vllm serve`（token 前是 `/` 非空白），导致 docker
    // 实例日志面板无源可回。
    if (isVllmProcCmd(cmd)) {
      out.vllm.push({ pid, cmd });
    } else if (/(^|\s)-m\s+sglang\.launch_server/.test(cmd) ||
               /(^|\s)sglang\.launch_server(\s|$)/.test(cmd)) {
      out.sglang.push({ pid, cmd });
    }
  }
  out.vllm.sort((a, b) => a.pid - b.pid);
  out.sglang.sort((a, b) => a.pid - b.pid);
  return out;
}

// 该日志文件是否属于某个实例：文件首/尾若干 KB 里出现实例 cmdline 的模型路径即认定归属。
// 为什么需要（2026-09-16 实锤）：同一端口换过启动栈（18420 先跑 NVFP4 栈、后换
// W4A16-1M 栈）后，旧栈按端口推导命名的日志文件仍残留在磁盘上；旧逻辑
// 「端口推导语径存在就返回」会一直显示旧栈的陈错误行，真实日志完全看不到。
function logFileMentions(file, needle, maxBytes) {
  if (!file || !needle) return false;
  const N = maxBytes || 262144;
  let fd;
  try {
    fd = fs.openSync(file, 'r');
    const sz = fs.fstatSync(fd).size;
    const len = Math.min(N, sz);
    if (!len) return false;
    const buf = Buffer.alloc(len);
    const chunks = [];
    fs.readSync(fd, buf, 0, len, 0);
    chunks.push(buf.toString('utf8'));
    if (sz > len) {
      fs.readSync(fd, buf, 0, len, sz - len);
      chunks.push(buf.toString('utf8'));
    }
    return chunks.some(c => c.indexOf(needle) >= 0);
  } catch (e) { return false; }
  finally { if (fd !== undefined) { try { fs.closeSync(fd); } catch (e2) {} } }
}

// 在「端口推导语径 + 注册的 sm.log（+ altLogs）」候选里，为在跑实例挑真正属于它的日志：
// ① 内容命中实例模型路径 +10（决定性）② 新鲜度（≤5min +4 / ≤60min +2 / ≤24h +1）
// ③ 等于注册名 sm.log +1（同分时优先，端口推导语径最易撞遗留文件）。
// 结果按 5s TTL 记忆，避免前端 2s 轮询反复读大文件。
// [ple-display 0923] 端口对应的全部脚本模型条目，按进程 cmdline 的模型路径命中优先排序。
// 同端口多栈（18420=NVFP4+W4A16 先后注册）时 scriptModelForPort 的「取第一个」会拿错。
function scriptModelsForPort(port, cmdlineArr) {
  const p = parseInt(port);
  if (!p) return [];
  const out = [];
  for (const k of Object.keys(SCRIPT_MODELS)) {
    const v = SCRIPT_MODELS[k];
    const inst0 = scriptModelInstance(v);
    if (v.port === p || (inst0 && inst0.port === p)) out.push(Object.assign({ key: k }, v));
  }
  if (out.length > 1 && cmdlineArr && cmdlineArr.length) {
    const cmd = cmdlineArr.join(' ');
    const score = (v) => {
      const paths = [v.modelPath, v.longCtxModelPath, v.longCtx512ModelPath].concat(v.altModelPaths || []);
      let sc = 0;
      for (const mp of paths) {
        if (!mp) continue;
        // cmdline 模型路径与条目路径互为前缀即命中（1M/512K 副本目录不同父同级）
        if (cmd.includes(mp) || (mp.includes('/media/ll/data/models') && cmd.includes('/media/ll/data/models'))) sc = Math.max(sc, cmd.includes(mp) ? 2 : 1);
      }
      return sc;
    };
    out.sort((a, b) => score(b) - score(a));
  }
  return out;
}
// 09-26 修复「vLLM 运行日志看不到当前输出」的根因：日志文件名跟着栈走，注册表跟不上。
// 官方 0.30.0 新栈的日志由 start-flash-next-0300.sh 里的 FN_LOG 决定
// （vllm-flash-next-0300.log），而 SCRIPT_MODELS.w4a16.log 记的还是旧栈的
// vllm-flash-next-w4a16.log；只按「端口推导语径 + sm.log + altLogs」打分挑选，
// 栈一切换就永久锁死在陈旧文件上——面板表现为「一堆老日志 + 全是 /metrics +
// 提示已 N 分钟没有新内容」，看起来就是"没有日志"。
// 权威来源＝进程祖先链 argv 里的 .log 路径：启动 wrapper 形态是
//   sudo -S sh -c 'exec setsid bash "$1" >> "$2" 2>&1 </dev/null' _ <inner> <logfile>
// 真实日志路径就写在它自己的 cmdline 里。root 进程的 /proc/<pid>/fd 普通用户读不到
// （chroot 栈的引擎正是 root），但 /proc/<pid>/cmdline 全局可读，比读 fd 更通用。
function ancestorLogFiles(pid, maxHops) {
  const HOP = maxHops || 4;
  const found = [];
  let cur = pid | 0;
  for (let hop = 0; hop <= HOP && cur > 1; hop++) {
    let toks = [];
    try { toks = fs.readFileSync('/proc/' + cur + '/cmdline', 'utf8').split('\0').map(s => s.trim()).filter(Boolean); } catch (e) {}
    for (let i = toks.length - 1; i >= 0; i--) {
      const t = toks[i];
      if (!/^\/\S+\.log$/.test(t) || found.some(f => f.file === t)) continue;
      try {
        if (fs.statSync(t).isFile()) { fs.accessSync(t, fs.constants.R_OK); found.push({ file: t, hop: hop }); }
      } catch (e) { /* 不存在或不可读：不是候选 */ }
    }
    let ppid = 0;
    try {
      const st = fs.readFileSync('/proc/' + cur + '/stat', 'utf8');
      ppid = parseInt(st.slice(st.lastIndexOf(')') + 1).trim().split(/\s+/)[1], 10) || 0;
    } catch (e) { break; }
    if (!ppid || ppid === cur) break;
    cur = ppid;
  }
  return found;
}
// 10-05 修复「vLLM 运行日志显示不出来」的根因：ancestorLogFiles 的判据太窄。
// 它能工作的唯一前提是「日志路径作为一个独立 argv token 出现在祖先链 cmdline 里」
// （旧 wrapper 形态 sudo -S sh -c 'exec setsid bash "$1" >> "$2"' _ <inner> <logfile>）。
// 现行启动链是 看门狗 setsid nohup bash start-flash-next-w4a16.sh >> LAUNCH_LOG
// → wrapper 最后一行 sudo_run setsid chroot …（shell 级重定向不进任何 cmdline）
// → sudo(root) → 引擎(root)。于是祖先链 argv 里没有任何 .log token（实测 hops=4 全空），
// ①′ 判据失明；② 候选打分链的 sm.log 注册路径也已与实际写入文件脱钩；普通进程读 root
// 进程的 /proc/<pid>/fd 又是 EACCES——三条路全断 → 一路跌到 ⑤ 兜底 ./vllm.log
// （10-01 的陈旧崩溃日志），面板表现为空白 + 「已 N 千分钟没有新内容」。
// 补救判据：祖先链上**本用户可读**的进程（看门狗/wrapper 属 ll），其 fd1/fd2 指向的
// 普通文件就是日志真身（bash 会把重定向继承给它 exec 的子进程）；再用内容交叉校验
// （文件里出现过引擎的 APIServer pid=N 或模型路径）防止拿错文件。
function ancestorFdLogFiles(pid, needles, maxHops) {
  const out = [];
  let cur = pid | 0;
  for (let hop = 1; hop <= (maxHops || 5) && cur > 1; hop++) {
    let ppid = 0;
    try {
      const st = fs.readFileSync('/proc/' + cur + '/stat', 'utf8');
      ppid = parseInt(st.slice(st.lastIndexOf(')') + 1).trim().split(/\s+/)[1], 10) || 0;
    } catch (e) { break; }
    if (!ppid || ppid === cur) break;
    cur = ppid;
    for (const fd of ['1', '2']) {
      let link = '';
      try { link = fs.readlinkSync('/proc/' + cur + '/fd/' + fd); } catch (e) { continue; } // root 进程 → EACCES，跳过
      if (!/^\/\S+\.log$/.test(link)) continue;
      let ok = false;
      try {
        if (!fs.statSync(link).isFile()) continue;
        fs.accessSync(link, fs.constants.R_OK);
        ok = !needles || !needles.length || needles.some((nd) => nd && logFileMentions(link, nd));
      } catch (e) { continue; }
      if (ok && !out.some((f) => f.file === link)) out.push({ file: link, hop: hop });
    }
  }
  return out;
}
// 祖先链 fd 判据的统一入口：先从引擎 cmdline 提取模型路径作校验 needle，再向上找。
function ancestorFdLogFile(enginePid, maxHops) {
  const needles = [];
  try {
    const cmd = fs.readFileSync('/proc/' + (enginePid | 0) + '/cmdline', 'utf8').split('\0').join(' ');
    const m = cmd.match(/(?:serve|--model(?:-path)?)\s+(\/\S+)/);
    if (m) needles.push(m[1]);
    needles.push('(APIServer pid=' + enginePid + ')');
  } catch (e) { /* 引擎已退出：无 needle，接受任意 .log 候选 */ }
  const hits = ancestorFdLogFiles(enginePid, needles, maxHops);
  if (!hits.length) return null;
  const mt = (f) => { try { return fs.statSync(f).mtimeMs; } catch (e) { return 0; } };
  hits.sort((a, b) => a.hop - b.hop || mt(b.file) - mt(a.file));
  return hits[0].file;
}
function pickScriptModelLogFile(inst, sm) {
  // [dsh-sglang-stack-1003] sglang 栈在位 → 直接读 sglang 日志（打分链候选不含该文件名）
  try {
    if (sm && sm.logSglang && sglangActive() && fs.existsSync(sm.logSglang)) return sm.logSglang;
  } catch (e) {}
  if (!inst) return null;
  const memo = (global.__scriptLogPick = global.__scriptLogPick || new Map());
  const memoKey = (inst.pid || 0) + ':' + (inst.port || 0);
  const hit = memo.get(memoKey);
  if (hit && Date.now() - hit.at < 5000) return hit.file;
  // ①′ 权威：祖先链 argv 里的日志（自动跟随栈切换 / FN_LOG 改名）；同层取最近写入的
  try {
    const anc = ancestorLogFiles(inst.pid);
    if (anc.length) {
      const mt = (f) => { try { return fs.statSync(f).mtimeMs; } catch (e) { return 0; } };
      anc.sort((a, b) => a.hop - b.hop || mt(b.file) - mt(a.file));
      if (mt(anc[0].file) > 0) {
        memo.set(memoKey, { file: anc[0].file, at: Date.now() });
        return anc[0].file;
      }
    }
  } catch (e) { /* 退回候选打分 */ }
  // ①″ 10-05 新增：argv 无 .log token（重定向发生在 wrapper shell 内部）时，
  //     沿祖先链找「本用户可读的 fd1/fd2 所指 .log」并以引擎身份内容校验。
  try {
    const f2 = ancestorFdLogFile(inst.pid, 5);
    if (f2) {
      memo.set(memoKey, { file: f2, at: Date.now() });
      return f2;
    }
  } catch (e) { /* 退回候选打分 */ }
  const cands = [];
  if (inst.port) cands.push(path.join(__dirname, 'vllm-flash-next-' + inst.port + '.log'));
  if (sm && sm.log) cands.push(sm.log);
  if (sm && Array.isArray(sm.altLogs)) sm.altLogs.forEach(f => f && cands.push(f));
  let needle = null;
  try {
    const cmd = fs.readFileSync('/proc/' + inst.pid + '/cmdline', 'utf8').split('\0').join(' ');
    const m = cmd.match(/(?:serve|--model(?:-path)?)\s+(\/\S+)/);
    if (m) needle = m[1];
  } catch (e) {}
  let best = null, bestScore = -1, bestMtime = -1;
  for (const f of cands) {
    if (!f) continue;
    let st;
    try { st = fs.statSync(f); if (!st.isFile()) continue; } catch (e) { continue; }
    const owned = needle && logFileMentions(f, needle) ? 10 : 0;
    const ageMin = (Date.now() - st.mtimeMs) / 60000;
    const fresh = ageMin <= 5 ? 4 : (ageMin <= 60 ? 2 : (ageMin <= 1440 ? 1 : 0));
    const reg = (sm && f === sm.log) ? 1 : 0;
    const score = owned + fresh + reg;
    if (score > bestScore || (score === bestScore && st.mtimeMs > bestMtime)) {
      bestScore = score; bestMtime = st.mtimeMs; best = f;
    }
  }
  if (memo.size > 32) memo.clear();
  memo.set(memoKey, { file: best, at: Date.now() });
  return best;
}

// Resolve the log file the running vLLM process actually writes to.
// Returns { file, tty }. When vLLM was started from a terminal, its stdout is
// a TTY (/dev/pts/N) — reading it with readFileSync would BLOCK THE EVENT
// LOOP FOREVER (a tty read waits for input/EOF that never comes). Mark it and
// let the endpoint answer with a friendly notice instead of reading it.
// 优先级：⓪ 在跑引擎进程祖先链 argv 里的 .log（通用，不依赖注册表）
//         ① 脚本化模型（chroot 内 root 进程，日志路径由 SCRIPT_MODELS 注册，最可靠）
//         ② 主实例端口(config.vllmPort)对应的 vLLM 进程 stdout
//         ③ 其它 vLLM 进程 stdout  ④ SGLang 日志  ⑤ ./vllm.log
//
// ⓪ 段是 09-29 新增的治本修复。面板语义 =「在跑引擎正在写的那份日志」，而真实路径
// 就写在引擎启动 wrapper 自己的 argv 里（形态见 ancestorLogFiles）。root 进程的
// /proc/<pid>/fd 普通用户读不到，但 cmdline 全局可读 —— 所以这条判据对任何栈、
// 任何新模型都成立，不需要有人先去 SCRIPT_MODELS 登记。
// 复发史（同一病三次）：09-26 旧栈 w4a16 → 官方 0.30.0、09-27 二栈并存、09-29 换
// uncensored 栈。每次都因「注册表跟不上换栈 → ① 段零候选 → 回落 ⑤ ./vllm.log」，
// 面板显示陈旧日志、stale_min 一路增长，看起来就是"日志没了"。
function pickLogFromProcs(procs) {
  const out = [];
  const mt = (x) => { try { return fs.statSync(x).mtimeMs; } catch (e) { return 0; } };
  for (const c of ((procs && procs.vllm) || [])) {
    let file = null;
    try {
      const anc = ancestorLogFiles(c.pid, 5);
      // 近祖先优先（hop 小者更接近引擎本身），同层取最近写入的
      anc.sort((a, b) => a.hop - b.hop || mt(b.file) - mt(a.file));
      if (anc.length && mt(anc[0].file) > 0) file = anc[0].file;
    } catch (e) { /* 读不到就换个进程 */ }
    if (!file) {
      // 10-05：argv 判据落空的启动链（重定向在 wrapper shell 内部做）→ 祖先链 fd1/fd2 判据
      try { file = ancestorFdLogFile(c.pid, 5); } catch (e2) { /* 继续换下一个进程 */ }
    }
    if (!file) continue;
    const pm = String(c.cmd || '').match(/--port\s+(\d+)/);
    out.push({ file: file, tty: false, pid: c.pid, port: pm ? parseInt(pm[1], 10) : null });
  }
  return out;
}
function getVllmLogSource() {
  const procs = scanInferenceProcs();
  // ⓪ 通用判据：任何在跑的 vLLM 进程，其祖先链 argv 里的 .log 即真实日志。
  //    主实例端口优先，其次取最近仍在写的；若候选全部停写超过 24h，说明只是残留
  //    进程（引擎其实没在跑），继续往下走，别拿陈年日志冒充"当前日志"。
  try {
    const hits = pickLogFromProcs(procs);
    if (hits.length) {
      // 命名避开 ②③ 段的 mainPort/isMain（同一函数块，const 不可重复声明）
      const mainPort0 = (typeof config !== 'undefined' && config && config.vllmPort) ? String(config.vllmPort) : '';
      const mtOf = (h) => { try { return fs.statSync(h.file).mtimeMs; } catch (e) { return 0; } };
      const onMain0 = (h) => !!(mainPort0 && String(h.port) === mainPort0);
      hits.sort((a, b) => (onMain0(b) - onMain0(a)) || (mtOf(b) - mtOf(a)));
      if (Date.now() - mtOf(hits[0]) < 24 * 3600 * 1000) return hits[0];
    }
  } catch (e) { /* 落回下面的旧逻辑 */ }
  // ① 脚本化模型（Flash-Next 等 chroot 镜像实例）：逐个条目定位「真正属于在跑实例」的
  //    日志文件（见 pickScriptModelLogFile），多实例时取最近写过的那个。
  try {
    const picks = [];
    for (const k of Object.keys(SCRIPT_MODELS)) {
      const sm = SCRIPT_MODELS[k];
      const inst = scriptModelInstance(sm);
      if (!inst) continue;
      const f = pickScriptModelLogFile(inst, sm);
      if (!f) continue;
      let mt = 0;
      try { mt = fs.statSync(f).mtimeMs; } catch (e) {}
      picks.push({ file: f, tty: false, pid: inst.pid, port: inst.port, mtime: mt });
    }
    if (picks.length) {
      picks.sort((a, b) => b.mtime - a.mtime);
      return { file: picks[0].file, tty: false, pid: picks[0].pid, port: picks[0].port };
    }
  } catch (e) { /* ignore */ }
  // ①½ docker 容器化 vLLM（09-19）：容器 json 日志属 root 不可直读，
  //     返回 cid 交给 handler 用 docker logs（ll 可经 daemon 读取）拉取。
  for (const c of procs.vllm) {
    const cid = dockerCidForPid(c.pid);
    if (cid) {
      const pm = c.cmd.match(/--port\s+(\d+)/);
      return { docker: cid, tty: false, pid: c.pid, port: pm ? parseInt(pm[1]) : null, file: 'docker:' + cid };
    }
  }
  // ②③ 普通 vLLM：读 stdout fd 目标（控制台启动=./vllm.log，脚本启动=各自日志）
  const mainPort = (typeof config !== 'undefined' && config && config.vllmPort) ? String(config.vllmPort) : '';
  const isMain = (c) => mainPort && new RegExp('--port\\s+' + mainPort + '(\\s|$)').test(c.cmd);
  const ordered = procs.vllm.slice().sort((a, b) => (isMain(a) ? 0 : 1) - (isMain(b) ? 0 : 1) || a.pid - b.pid);
  for (const c of ordered) {
    try {
      const link = fs.readlinkSync('/proc/' + c.pid + '/fd/1');
      if (link.startsWith('/') && fs.existsSync(link)) {
        try {
          if (fs.statSync(link).isFile()) return { file: link, tty: false, pid: c.pid };
        } catch (e2) {}
        return { file: link, tty: true, pid: c.pid };
      }
    } catch (e) { /* fd 不可读（root 进程）：继续找下一个 */ }
  }
  // ④ SGLang
  if (procs.sglang.length && fs.existsSync(SGLANG_LOG_PATH)) {
    return { file: SGLANG_LOG_PATH, tty: false, pid: procs.sglang[0].pid };
  }
  // ⑤ 兜底（fallback:true 供上层/前端判别：没能定位到在跑引擎的真实日志）
  return { file: path.join(__dirname, 'vllm.log'), tty: false, fallback: true };
}

function resolveVllmLogPath() {
  return getVllmLogSource().file;
}

// Detect the model label (|{...}) used by cumulative metrics, plus the
// source-label prefix used by prompt_tokens_by_source_total
function detectModelLabels(m) {
  let modelLabel = '';
  for (const k of Object.keys(m)) {
    const pipeIdx = k.indexOf('|');
    if (pipeIdx > 0 && k.substring(0, pipeIdx) === 'vllm:num_requests_running') {
      modelLabel = k.substring(pipeIdx);
      break;
    }
  }
  let sourceLabel = modelLabel;
  for (const k of Object.keys(m)) {
    if (k.startsWith('vllm:prompt_tokens_by_source_total|')) {
      const labelPart = k.substring(k.indexOf('|'));
      const sIdx = labelPart.indexOf('"source"');
      if (sIdx > 0) { sourceLabel = labelPart.substring(0, sIdx); break; }
    }
  }
  return { modelLabel, sourceLabel };
}

// Capture current raw values of all cumulative metrics (the reset baseline)
function captureResetBlock(m, modelLabel, sourceLabel) {
  const r = {};
  const add = (k) => { if (m[k] !== undefined) r[k] = m[k]; };
  add('vllm:generation_tokens_total' + modelLabel);
  add('vllm:prompt_tokens_total' + modelLabel);
  add('vllm:prompt_tokens_by_source_total' + sourceLabel + '"source":"local_cache_hit"}');
  add('vllm:prompt_tokens_by_source_total' + sourceLabel + '"source":"local_compute"}');
  add('vllm:request_generation_tokens_count' + modelLabel);
  add('vllm:request_generation_tokens_sum' + modelLabel);
  add('vllm:request_decode_time_seconds_count' + modelLabel);
  add('vllm:request_decode_time_seconds_sum' + modelLabel);
  add('vllm:request_prefill_time_seconds_count' + modelLabel);
  add('vllm:request_prefill_time_seconds_sum' + modelLabel);
  add('vllm:request_prompt_tokens_count' + modelLabel);
  add('vllm:request_prompt_tokens_sum' + modelLabel);
  add('vllm:time_to_first_token_seconds_count' + modelLabel);
  add('vllm:time_to_first_token_seconds_sum' + modelLabel);
  add('vllm:request_time_per_output_token_seconds_count' + modelLabel);
  add('vllm:request_time_per_output_token_seconds_sum' + modelLabel);
  return r;
}

// Return a copy of m with the reset baseline subtracted from cumulative metrics
function applyResetView(m, reset) {
  const mj = Object.assign({}, m);
  for (const k of Object.keys(reset || {})) {
    const v = (m[k] || 0) - (reset[k] || 0);
    mj[k] = v > 0 ? v : 0;
  }
  return mj;
}

// ====== Live per-request prefill progress (vLLM 插件实时回传) ======
// dsh_vllm_logger 插件（vllm.stat_logger_plugins 入口）把每个 prefill chunk 的
// 每请求数据写成 JSONL：{t, sid, pid, model, rid, arrival, prompt_total,
// computed, cached}（computed/cached 为请求级累计）。本函数 tail 读该文件并
// 按 sid 缓存成 Map(rid -> 最新记录)，供 computeConcurrencyDetails 匹配 REQ 行。
// sid 在插件初始化时随机生成：vLLM 重启 → sid 变化 → 控制台自动重置匹配。
const LIVE_PREFILL_PATH = path.join(__dirname, 'vllm-live-prefill.jsonl');
if (!global.__livePrefill) {
  global.__livePrefill = { sid: null, updatedAt: 0, byRid: new Map(), order: [], lastFileSize: 0, lastReadAt: 0 };
}
// pid 可选：多卡多实例共用同一 jsonl（每实例独立 sid+pid）。指定 pid 时只选
// 该实例的 sid（否则主实例的匹配会被更活跃的从实例 sid 抢走）；不指定时选
// 最近更新的 sid（旧行为）。
function readLivePrefill(pid) {
  const lp = global.__livePrefill;
  const now = Date.now();
  try {
    // 09-06 性能修复：原来每次调用都 statSync（即使限频命中也要 stat），
    // stats 端点每 0.5s 轮询 → 每秒 2 次同步 stat。改为 2s 限频 + 缓存 stat 结果。
    if (lp._lastStatAt && now - lp._lastStatAt < 400 && lp._lastStat) {
      const st = lp._lastStat;
      if (lp.byteOffset !== undefined && st.size === lp.byteOffset && lp.selectedPid === pid && now - (lp.lastReadAt || 0) < 500) return lp;
    } else {
      const st = fs.statSync(LIVE_PREFILL_PATH);
      lp._lastStat = st;
      lp._lastStatAt = now;
      if (lp.byteOffset === undefined || st.size < lp.byteOffset) {
        lp.byteOffset = Math.max(0, st.size - 4 * 1024 * 1024);
        lp.sidMaps = new Map();
        lp.orderMaps = new Map();
        lp.firstTMaps = new Map();
        lp.pending = '';
        lp.sid = null; lp.byRid = null; lp.order = null; lp.firstT = null;
      }
    }
    const st = lp._lastStat;
    const fresh = (now - st.mtimeMs) < 10000;
    if (lp.byteOffset !== undefined && st.size === lp.byteOffset && lp.selectedPid === pid && now - (lp.lastReadAt || 0) < 500) return lp;
    lp.lastReadAt = now;
    let text = lp.pending || '';
    if (st.size > lp.byteOffset) {
      const LEN = Math.min(4 * 1024 * 1024, st.size - lp.byteOffset);
      const buf = Buffer.alloc(LEN);
      const fd = fs.openSync(LIVE_PREFILL_PATH, 'r');
      try { fs.readSync(fd, buf, 0, LEN, lp.byteOffset); } finally { fs.closeSync(fd); }
      lp.byteOffset += LEN;
      lp._lastStat = null; // 文件已增长，下次重新 stat
      text += buf.toString('utf8');
    }
    // 单行消化：合并记录 cached 取历史最大值（完成记录可能被 decode 阶段
    // cached=0 快照覆盖，覆盖会丢命中数）；首块记录首个 prefill 时刻。
    const ingest = (r) => {
      if (!r || typeof r.rid !== 'string' || !r.sid) return;
      let m2 = lp.sidMaps.get(r.sid);
      if (!m2) {
        m2 = new Map();
        lp.sidMaps.set(r.sid, m2);
        lp.orderMaps.set(r.sid, []);
        lp.firstTMaps.set(r.sid, new Map());
      }
      if (!m2.has(r.rid)) {
        lp.orderMaps.get(r.sid).push(r.rid);
        lp.firstTMaps.get(r.sid).set(r.rid, r.t || 0);
        m2.set(r.rid, r);
      } else {
        const prev = m2.get(r.rid);
        m2.set(r.rid, {
          t: r.t, sid: r.sid, pid: r.pid, model: r.model,
          rid: r.rid, arrival: r.arrival,
          prompt_total: r.prompt_total, computed: r.computed,
          cached: Math.max(prev.cached || 0, r.cached || 0),
        });
      }
    };
    const parts = text.split('\n');
    lp.pending = parts.pop() || '';
    for (const l of parts) {
      if (!l.trim()) continue;
      try { ingest(JSON.parse(l)); } catch (e) { /* 坏行跳过 */ }
    }
    // 无换行结尾的完整末行：能解析则立即纳入，半截行等下一轮补齐
    if (lp.pending && lp.pending.trim()) {
      try { ingest(JSON.parse(lp.pending)); lp.pending = ''; } catch (e) { /* 继续积压 */ }
    }
    // 每个 sid 的最新记录时刻 + 归属 pid（多实例文件里 sid 交错，插入序 ≠ 时间序，
    // 不能再用「最后插入的 sid」当最新会话）
    const sidStats = new Map(); // sid -> {maxT, pid}
    for (const [sid, m2] of lp.sidMaps) {
      let maxT = 0, sp = null;
      for (const rec of m2.values()) {
        const t = rec.t || 0;
        if (t >= maxT) { maxT = t; sp = rec.pid != null ? rec.pid : null; }
      }
      sidStats.set(sid, { maxT, pid: sp });
    }
    // 只保留最近更新的 3 个 sid 的映射，防内存无限增长
    for (const sid of [...sidStats.keys()].sort((a, b) => sidStats.get(b).maxT - sidStats.get(a).maxT).slice(3)) {
      sidStats.delete(sid);
      lp.sidMaps.delete(sid);
      lp.orderMaps.delete(sid);
      lp.firstTMaps.delete(sid);
    }
    // 选当前会话 sid：pid 指定时只在该实例的 sid 里选；否则选最近更新者
    let bestSid = null, bestT = 0;
    for (const [sid, sst] of sidStats) {
      if (pid != null && sst.pid !== pid) continue;
      if (sst.maxT > bestT) { bestT = sst.maxT; bestSid = sid; }
    }
    if (bestSid) {
      lp.sid = bestSid;
      lp.byRid = lp.sidMaps.get(bestSid);
      lp.order = lp.orderMaps.get(bestSid) || [];
      lp.firstT = lp.firstTMaps.get(bestSid) || new Map();
      // 清理：超过 10 分钟没更新的请求记录丢弃（请求早该完成）
      const cutoff = now / 1000 - 600;
      for (const [rid, rec] of lp.byRid) {
        if ((rec.t || 0) < cutoff) { lp.byRid.delete(rid); lp.firstT.delete(rid); }
      }
      lp.updatedAt = now;
    }
    lp.selectedPid = pid;
    lp.fresh = fresh;
  } catch (e) { /* 文件不存在 → 无实时数据，走估算 */ lp.fresh = false; }
  return lp;
}

// ============================================================================
// 并发请求实时输出速度 v3（2026-10-06 全新设计——推翻 v2 的"从聚合指标反推
// 每请求速度 + 到达时间猜归属"架构）
// ----------------------------------------------------------------------------
// v2 的根因性不准：官方 vLLM 0.30.0 的 /metrics 里**不存在任何每请求实时数据**
// （request_* 族全是请求完成后才进桶的直方图），于是 v2 实际总是落到
// "直方图驻留 + Little 定律估计并发数 + 实例总吞吐 ÷ 并发数"——每个 decode
// 行显示同一个估计值、滞后数秒；行↔请求归属靠到达时间最近邻猜测，并发错绑
// 时整行数值都是别人的。聚合指标反推个体在数学上就测不出"这一个请求此刻的
// 速度"，修补参数没有意义。
//
// v3 把数据源换成引擎内部真值：
//   rt-patch #11（vllm-0300/patches-extra/dsh_stream_rt.py）包前端进程的
//   IterationStats.update_from_output——官方对每个产出 token 的引擎步都会调
//   它，参数里现成有 output.request_id（真实 rid）、req_stats
//   .num_generation_tokens（该请求累计输出，MTP 一步多 token 如实计入）、
//   .arrival_time（进引擎墙钟）、is_prefilling（首 token 步=True）。
//   dsh_vllm_logger 插件把搭车数据逐步骤写成 vllm-live-stream.jsonl：
//     进度行 {"t","sid","pid","port","rid","g","a","pf","n"}
//     完成行 {"t","sid","pid","port","rid","f":1,"g"}
//   本模块 tail 该文件，维护 rid→{累计g, 样本环, arrival, firstT, done}。
//
// 由此每一行的三个数全部是"它自己"的引擎实测，不存在摊分：
//   速度   = 该 rid 自身样本的滑窗差分 Δg/Δt（窗 [2s,6s]，末样本 >2.2s 未更新
//            → null 如实显示 --；引擎停顿/抢占/排队都不产出假值）
//   累计   = g（token 精确，非字符估算）
//   TTFT   = firstT − arrival（引擎侧真实首 token 时刻）
// 行↔rid 归属退化为纯标签问题（速度数值不依赖归属正确性）：走 8889 代理的
// 流量用 crid→SSE id→rid 精确认领；直连流量按 arrival 最近邻配行——即便配
// 错，每行显示的仍是某个真实请求的真实数值，且 Σ行 ≈ 实例总吞吐（天然守恒）。
// 相位同样变精确：有 rid 在流里 = 已过 TTFT = 输出中；running − 流内数 =
// 预填充行数，不再需要 nEw/bootstrap 估计。
//
// 降级链（如实标注，绝不编造）：
//   ① 流新鲜（本模块）           → speed_src 'v3' / 'v3-exact'（引擎实测）
//   ② 旧栈 live-prefill computed → speed_src 'eng'（同为每请求真值，保留）
//   ③ 全无（sglang/远端/无插件） → speed_src 'share'：实例短窗吞吐 ÷ decode
//      行数（守恒均摊，前端明标"估算"）；窗口未成熟行显示 --。
// 引擎重启（sid 变化）即清空重采；流文件缺失/陈旧 → fresh=false，走②③。
// ============================================================================
const LIVE_STREAM_PATH = process.env.DSH_LIVE_STREAM_FILE || path.join(__dirname, 'vllm-live-stream.jsonl');
if (!global.__liveStream) {
  global.__liveStream = {
    byteOffset: undefined, pending: '', lastStat: null, lastStatAt: 0, lastReadAt: 0,
    sids: new Map(),      // sid -> { byRid: Map<rid,st>, port, pid, maxT, lastWall }
    byPort: new Map(),    // port(String) -> 活跃 sid 的状态对象（消费视图）
  };
}
const LS_SID_KEEP = 3;         // 文件里混跑多实例/多次重启，只留最近 3 个会话
const LS_SAMPLE_GAP = 0.35;    // 样本环抽稀间隔（秒）：测速窗 2~6s，0.35s 粒度足够
const LS_WIN_KEEP = 12;        // 样本环保留窗口（秒）
const LS_STALE_DONE_S = 30;    // 有 rid 但 >30s 无新行且无完成行 → 视为已终止（abort/抢占丢弃）
const LS_ACTIVE_FRESH_S = 8;   // [ghost-done-1014] active 集新鲜度：>8s 无新行不算「活跃」（decode 停顿上限≈chunked prefill 单块时长，实测 <6s）
const LS_PORT_FRESH_MS = 6000; // 端口级新鲜度：该 port 会话 6s 内有行才算 v3 激活

// v3 测速窗参数（滑窗 Δg/Δt，全部该 rid 自身样本）
const V3_WIN_MIN = 1.5, V3_WIN_MAX = 6.0, V3_LAST_STALE = 2.2;

function lsIngestLine(r) {
  const S = global.__liveStream;
  if (!r || typeof r.rid !== 'string' || !r.sid) return;
  let m = S.sids.get(r.sid);
  if (!m) {
    m = { byRid: new Map(), port: String(r.port || ''), pid: r.pid || null, maxT: 0, lastWall: Date.now(), skewEw: null, guardN: 0 };
    S.sids.set(r.sid, m);
    // 只留最近 LS_SID_KEEP 个会话（按最近写入时刻淘汰）
    if (S.sids.size > LS_SID_KEEP) {
      let dead = null, deadT = Infinity;
      for (const [k, v] of S.sids) { if (v.lastWall < deadT) { deadT = v.lastWall; dead = k; } }
      if (dead && S.sids.get(dead) === m) dead = null; // 新会话即最小者时不淘汰自己
      if (dead) S.sids.delete(dead);
    }
  }
  m.lastWall = Date.now();
  const t = Number(r.t) || Date.now() / 1000;
  if (t > m.maxT) m.maxT = t;
  // 时钟护栏：引擎与本机同机部署时 t 与墙钟同域；偏差>600s 视为异常时钟，
  // 用本地接收时刻替换（样本时刻只影响窗口分母，替换后仍单调可用）
  const nowS = Date.now() / 1000;
  // 时钟偏差/链路延迟可观测（同机恒 ~tail+轮询延迟，正常 <1.5s；跨主机误配时
  // skew/guardN 会说话）。口径：只算年龄 <3s 的进度行，EWMA 滑动。
  if (!r.f && Math.abs(t - nowS) < 3) {
    const d = Math.abs(t - nowS);
    m.skewEw = m.skewEw === null ? d : m.skewEw + 0.25 * (d - m.skewEw);
  }
  // 异常时钟/积压行（偏差>600s）：整行丢弃。旧版"替换为接收时刻"会制造
  // 同一瞬间的假样本（回放场景撑爆 guard 计数、伪造 avg 尖峰的源头）。
  if (Math.abs(t - nowS) > 600) { m.guardN++; return; }
  const ts = t;
  let st = m.byRid.get(r.rid);
  if (!st) {
    st = {
      rid: r.rid, arrival: Number(r.a) || ts, g: 0, firstT: null,
      samples: [], done: false, doneAt: 0, lastLineT: ts, lastWall: Date.now(),
      obsT: nowS, g0: 0, gLastT: ts, // 观测基线：窗口从"控制台开始观测它"起算——首见行即便是一分钟前的旧行（回放/接管），avg 分母也如实是观测窗而非历史全程
    };
    m.byRid.set(r.rid, st);
  }
  st.lastLineT = ts;
  st.lastWall = Date.now();
  if (r.a && !st.arrival) st.arrival = Number(r.a);
  if (r.f) {
    st.done = true; st.doneAt = ts;
    // [ghost-done-1014] 引擎完成行携带裸 rid（无 -8hex 内部后缀），进度行携带后缀 rid——
    // 后缀序列永远等不到自己的 f 行，会以「幽灵活跃」滞留 LS_STALE_DONE_S(30s)，
    // 阶段④为幽灵建行 → 行数/running 虚高（实测 4 并发显示 14 行）。
    // 裸 rid 完成 = 同会话所有 "裸rid-*" 前缀的兄弟序列一并 done。
    const pfx = r.rid + '-';
    for (const [k2, st2] of m.byRid) {
      if (!st2.done && k2.length > pfx.length && k2.slice(0, pfx.length) === pfx) { st2.done = true; st2.doneAt = ts; }
    }
  }
  const g = Number(r.g) || 0;
  // gauge 只增不减；回退 = 引擎重启后 rid 复用（同 sid 内不会），忽略
  if (g > st.g) {
    if (st.g === 0 && g > 0 && !r.pf && st.firstT === null) st.g0 = g; // 中途首见：基线抬到观测值
    st.g = g;
    st.gLastT = ts;
  }
  // 首 token 判据维持 pf-only：151 序列实测 150 个首样本即 pf=1,g=1（精确真值）。
  // （曾加 g>0 兜底，统计证伪：pf=1 时兜底先把 firstT 提前到 max(a,t−0.35)，
  // 分母虚大反而稀释速度——已撤，勿再画蛇添足。）
  if (r.pf && st.firstT === null) st.firstT = ts; // 首 token 步
  if (st.g > 0 || r.f) {
    const last = st.samples[st.samples.length - 1];
    if (!last || g !== last.g || ts - last.t >= LS_SAMPLE_GAP) {
      if (!last || ts > last.t) st.samples.push({ t: ts, g });
      else last.g = g;
    } else if (g !== last.g) {
      last.g = g; // 同刻多行（批内合并）取最新累计
    }
    const cut = ts - LS_WIN_KEEP;
    while (st.samples.length > 2 && st.samples[0].t < cut) st.samples.shift();
  }
}

function readLiveStream() {
  const S = global.__liveStream;
  const now = Date.now();
  try {
    if (!S.lastStatAt || now - S.lastStatAt > 300) {
      const st = fs.statSync(LIVE_STREAM_PATH);
      // 偏移策略：只跟随尾部，从不倒读历史（2026-09-28 修复"开局上千"根因之一）。
      // 旧插件 ring-trim 是 truncate+全文件重写：轮询撞上清空窗口时 size 骤减，
      // 若倒着读 2MB 会把几分钟历史行全量回放——时钟护栏把这些老 t 替换成
      // "现在"，样本环被压扁到同一瞬间、st.g 直接顶到历史累计值 → avg 上千。
      // 代价：trim/重启后 1~2s 内新绑定行滑窗暂缺（如实 --），换零伪造数值。
      if (S.byteOffset === undefined || st.size < S.byteOffset) {
        S.byteOffset = st.size;
        S.pending = '';
      }
      S.lastStat = st; S.lastStatAt = now;
    }
    const stt = S.lastStat;
    if (!stt || stt.size <= S.byteOffset) { rebuildPortView(now); return S; }
    if (now - S.lastReadAt < 120) { rebuildPortView(now); return S; }
    S.lastReadAt = now;
    const LEN = Math.min(512 * 1024, stt.size - S.byteOffset);
    const buf = Buffer.alloc(LEN);
    const fd = fs.openSync(LIVE_STREAM_PATH, 'r');
    try { fs.readSync(fd, buf, 0, LEN, S.byteOffset); } finally { fs.closeSync(fd); }
    S.byteOffset += LEN;
    const text = S.pending + buf.toString('utf8');
    const parts = text.split('\n');
    S.pending = parts.pop() || '';
    for (const l of parts) {
      if (!l || !l.charCodeAt) continue;
      try { lsIngestLine(JSON.parse(l)); } catch (e) { /* 坏行跳过 */ }
    }
    if (S.pending) {
      try { const o = JSON.parse(S.pending); lsIngestLine(o); S.pending = ''; } catch (e) { /* 半行待补 */ }
    }
  } catch (e) { /* 文件不存在 → v3 不激活 */ }
  rebuildPortView(now);
  return S;
}

// 消费视图：port → 该端口最新活跃会话。归属优先级：
//   ① 行带 port 字段（新插件+inner 已 export DSH_ENGINE_PORT）→ 精确到端口；
//   ② 无 port 字段的会话（旧插件形态）只兜底服务主实例端口，且不抢①的地盘。
function rebuildPortView(now) {
  const S = global.__liveStream;
  const byPort = S.byPort;
  byPort.clear();
  const noPort = [];
  for (const [sid, m] of S.sids) {
    if (now - m.lastWall > 60000) { S.sids.delete(sid); continue; } // 死亡会话回收
    // 会话内逐 rid 清理：完成超 30s / 无行超 LS_STALE_DONE_S → 剔除
    for (const [rid, st] of m.byRid) {
      const ageS = now / 1000 - st.lastLineT;
      if ((st.done && now - st.doneAt * 1000 > 30000) || (!st.done && ageS > LS_STALE_DONE_S)) m.byRid.delete(rid);
    }
    if (m.port) {
      const cur = byPort.get(m.port);
      if (!cur || m.maxT > cur.maxT) byPort.set(m.port, m);
    } else noPort.push(m);
  }
  if (noPort.length) {
    const key = String(config.vllmPort || 8000);
    if (!byPort.has(key)) {
      let best = null;
      for (const m of noPort) { if (!best || m.maxT > best.maxT) best = m; }
      if (best) byPort.set(key, best);
    }
  }
}

// v3 消费入口：返回该端口活跃 rid 状态 + 新鲜度。
function v3StreamFor(port) {
  const S = global.__liveStream;
  const key = String(port || config.vllmPort || 8000);
  const m = S.byPort.get(key);
  const now = Date.now();
  if (!m) return { active: [], fresh: false };
  const active = [];
  const doneIds = new Set();
  const _nowS = Date.now() / 1000;
  for (const st of m.byRid.values()) {
    if (st.done) { if (st.g > 0) doneIds.add(st.rid); continue; }
    // [ghost-done-1014] 无新行超 LS_ACTIVE_FRESH_S 的未 done 序列（abort/丢完成行）不进活跃集，
    // 但保留在 byRid 里等 done 或 30s 回收——已绑它的行走 v3goneAt 语义自然退场。
    if (st.g > 0 && _nowS - st.lastLineT <= LS_ACTIVE_FRESH_S) active.push(st);
  }
  const fresh = (now - m.lastWall) < LS_PORT_FRESH_MS;
  return { active, doneIds, fresh: fresh && active.length > 0, sid: m ? [...S.sids.keys()].find(k => S.sids.get(k) === m) : null, session: m,
    skew: m && m.skewEw !== null && m.skewEw !== undefined ? Math.round(m.skewEw * 1000) / 1000 : 0, guardN: m ? m.guardN : 0 };
}

// 每 rid 滑窗测速（纯自身样本）：
//   末样本 t 距今 ≤V3_LAST_STALE 秒；基线 = 年龄∈[V3_WIN_MIN,V3_WIN_MAX] 的最老
//   样本；窗长上限钳制（span>V3_WIN_MAX → null）。
//   返回 {spd, gen, span}；无合格窗 → null（如实 --）。g 冻结（抢占/排队）
//   → Δg=0 → null，前端回落全程均值自然衰减，绝不发放 0 速假精确。
function v3Rate(st, nowMs) {
  const s = st.samples;
  if (!s || s.length < 2) return null;
  const last = s[s.length - 1];
  const nowS = nowMs / 1000;
  if (nowS - last.t > V3_LAST_STALE) return null;
  let base = null;
  for (let k = 0; k < s.length - 1; k++) {
    const age = nowS - s[k].t;
    if (age < V3_WIN_MIN) continue;
    if (age <= V3_WIN_MAX) { base = s[k]; break; }
    // 全部样本年龄都 >V3_WIN_MAX（长冻结后恢复）：绝不返回跨冻结期的稀释均速
    // ——那会把十几秒均值标成「实时」。如实 null，等窗内出现合格基线再出数。
    return null;
  }
  if (!base) base = s[0];
  const span = last.t - base.t;
  if (span < 1.0 || span > V3_WIN_MAX) return null;
  const d = last.g - base.g;
  if (d <= 0) return null;
  return { spd: d / span, gen: last.g, span };
}

// 上一秒真实产出（行 meta「上一秒 N」用）：窗末 g − 1s 前最近样本 g。
function v3LastSec(st, nowMs) {
  const s = st.samples;
  if (!s || s.length < 1) return undefined;
  const last = s[s.length - 1];
  const cut = last.t - 1.0;
  let b = null;
  for (let k = s.length - 1; k >= 0; k--) { if (s[k].t <= cut) { b = s[k]; break; } }
  if (!b) b = s[0];
  if (last.t - b.t < 0.6) return undefined; // 窗太短不外推 1s 口径（偏小失真）
  return Math.max(0, Math.round((last.g - b.g) * 1.0 / (last.t - b.t)));
}

// 行数值字段以 rid 真值接管：累计输出 = 引擎累计 g（精确，含 MTP；旧的
// 均摊积分/字符估算一经 v3 绑定立即作废）；decode 起点 = 引擎首 token 时刻
// （avg_speed 分母自此精确）。
function v3TouchRow(lv, st) {
  lv.tokens = st.g;
  // g 冻结 >2.5s（客户端断连/被抢占/引擎停顿，行未 done 但不再产出）：
  // 均值钉住在最后活跃值——否则分母随时间增长把显示稀释成假低速（实测 117→16）。
  // [v3-pin-stale-1006] g 停增但**新行仍在到达**（st.lastLineT 比 gLastT 新）
  // ≠ 输出冻结：多为客户端断连/抢占丢弃后引擎仍带该序列收尾，或观测间隙。
  // 只有"新行也停了"才算真冻结——否则分母钉在墙钟 now 上，行滞留期间每轮
  // 重算 den 随墙钟增长 → 显示被稀释成 0.2 这类假低速（10-06 并发截图实锤）。
  const gAge = Date.now() / 1000 - (st.lastLineT || st.gLastT || st.obsT || 0);
  if (gAge > 2.5) {
    // [v3avg-pin-1003] 钉住 = 把最后活跃时刻算出的均值固化为终值（v3AvgFrozen），
    // 此后不再重算。旧版只"跳过赋值"，留下两个数值缺陷：
    //  ① 冻结前最后一次赋值用的是冻结瞬间的分母，之后 g 已终、行滞留期间
    //     （gauge 滞后 + goneAt 冻结显示 ≤15s）任何再进本函数的路径都会让
    //     分母继续增长 → 显示被稀释成十几 tok/s 的假低速；
    //  ② 行被别的 rid 接管复用（bindRow 换绑）时旧 v3Avg 残留显示。
    // 缺 v3AvgFrozen（首轮即冻结：接管时 g 已停）→ 回落最后活跃时刻口径。
    // [v3-pin-stale-1006] 同活跃支路：钉住分母只到 gLastT 为止；观测窗未成熟
    // （接管即停/完成行 g 同刻覆盖样本基线，10-06 实测 REQ-22 0.3 形态）→ 不发
    // 稀释值，前端回落上一有效值/显示 --。
    if (lv.v3AvgFrozen === undefined) {
      const tEnd = Math.min(Date.now() / 1000, st.gLastT || st.obsT || (Date.now() / 1000));
      if (st.firstT) {
        const den = tEnd - st.firstT;
        lv.v3AvgFrozen = den >= 0.5 ? st.g / den : undefined;
      } else {
        const den = tEnd - st.obsT;
        lv.v3AvgFrozen = den >= 0.5 ? Math.max(0, st.g - st.g0) / den : undefined;
      }
    }
    lv.v3Avg = lv.v3AvgFrozen;
    lv.v3seenAt = Date.now();
    return;
  }
  if (st.firstT) {
    lv.decodeStart = Math.round(st.firstT * 1000);
    // [v3-pin-stale-1006] **活跃期全程均值的分母钉在 min(now, 最后产出)**：
    // 10-06 截图形态（g=8·el=34s·0.2 tok/s）的直接源头——行 g 已停/完成、rid
    // 暂离绑定（v3goneAt）后仍走本支路，分母随墙钟增长而分子是冻结终值，
    // 全程均值被稀释成零点几。停增/停观测期不再发放稀释值；产出恢复（g 推进
    // → gLastT 变新）时窗口自然重新成熟。
    const _anchorS = Math.max((st.gLastT || st.obsT || st.firstT) - 0.05, st.firstT || 0);
    const _denEnd = Math.min(Date.now() / 1000, _anchorS);
    const den = (_denEnd * 1000 - lv.decodeStart) / 1000;
    lv.v3Avg = den >= 0.5 ? st.g / den : undefined; // 全程真均值（首token起算）
  } else {
    // 中途接管（pf 行未见，真实首token时刻未知）：全程均值退化为「接管后
    // 观测窗均值」——增量/时长都有界，绝不再用引擎累计值除以观测零头秒。
    lv.decodeStart = Math.round(st.obsT * 1000);
    const den = (Date.now() / 1000 - st.obsT);
    lv.v3Avg = den >= 0.5 ? Math.max(0, st.g - st.g0) / den : undefined;
  }
  lv.v3AvgFrozen = undefined; // 活跃期清除钉住值，恢复实时重算
  lv.v3seenAt = Date.now();
}

// ====== 控制台任务号（跨 GPU 关联同一任务）======
// 每个经控制台代理转发的生成请求，到达时分配全局递增任务号 T<seq>。
// PD 两步转发：prefill 腿到达即注册、decode 腿在 STEP2 真正转发（prefill 完成）
// 时注册，两腿同任务号；常规单实例转发每请求一个。
// REQ 跟踪器按端口给请求生 REQ-n 时按 FIFO 认领任务号 → 同一逻辑请求在
// GPU0（预填充）与 GPU1（输出）两行显示相同的 T 号；前端按 T 号同色。
let __taskSeq = 0;
const __taskForward = new Map(); // port -> [{taskId, at, claimed, crid}]
function nextTaskId() { return 'T' + (++__taskSeq); }
function taskForwardRegister(port, taskId, crid) {
  const key = String(port || 8000);
  if (!__taskForward.has(key)) __taskForward.set(key, []);
  const q = __taskForward.get(key);
  if (q.length > 500) q.shift(); // 防异常堆积
  q.push({ taskId, at: Date.now(), claimed: false, crid: crid || null });
}
// 认领任务号；返回 {taskId, crid}。crid 供行凭 clientReqId→rid 精确认领引擎序列。
function taskForwardClaim2(port) {
  const q = __taskForward.get(String(port || 8000));
  if (!q || !q.length) return null;
  const now = Date.now();
  while (q.length && now - q[0].at > 120000) q.shift(); // 120s 未认领过期（客户端断开等）
  for (const rec of q) { if (!rec.claimed) { rec.claimed = true; return { taskId: rec.taskId, crid: rec.crid || null }; } }
  return null;
}
function taskForwardClaim(port) {
  const c = taskForwardClaim2(port);
  return c ? c.taskId : null;
}
// 消费指定任务号在指定端口的未认领记录：幂等。调用点：
// ① PD 交棒——prefill 完成后 STEP2 转发 decode 腿时消费 prefill 腿；
// ② 请求收尾——代理响应结束/客户端断开时消费该腿（快请求在两次 metrics
//    轮询间跑完、fill 没认领的记录，靠这里清掉，否则滞留排队列表最多 120s）
function taskForwardConsume(port, taskId) {
  const q = __taskForward.get(String(port || 8000));
  if (!q || !taskId) return;
  for (const rec of q) if (!rec.claimed && rec.taskId === taskId) { rec.claimed = true; return; }
}

// port：实例端口（请求身份 tracker 按端口独立，ss 抓对端 IP 用该端口）
// ticker：该端口的每秒 token ticker（多实例下各采各的，主实例传 global.__tokTicker）
function computeConcurrencyDetails(m, genTokensTotal, lastGenTokensTotal, elapsedMs, lastRunning, perReqTokens, livePrefill, port, ticker) {
  const keyPrefix = 'vllm:';
  // v2 采样兜底：stats 轮询时若该端口采样器 >2.5s 未更新（主 ticker 因
  // 事件循环阻塞/实例刚上线漏采样），就地补采一次。
  {
    const sp0 = reqOutSamplerFor(port || config.vllmPort);
    if (Date.now() - (sp0.lastSeenAt || 0) > 2500) sampleReqOutGauges(port || config.vllmPort);
  }
  // Build model label dynamically from available metrics keys
  // parseMetrics encodes as |{json}, so search for | not {
  let modelLabel = '';
  const allLabels = Object.keys(m);
  for (const k of allLabels) {
    const pipeIdx = k.indexOf('|');
    if (pipeIdx > 0 && k.substring(0, pipeIdx) === 'vllm:num_requests_running') {
      modelLabel = k.substring(pipeIdx);
      break;
    }
  }
  // For prompt_tokens_by_source_total, extract model label from any _by_source metric
  let sourceLabelPrefix = modelLabel;
  let foundSource = false;
  for (const k of allLabels) {
    if (k.startsWith('vllm:prompt_tokens_by_source_total|')) {
      const pipeIdx = k.indexOf('|');
      const labelPart = k.substring(pipeIdx); // |{...}
      // Find the source key portion
      const sourceIdx = labelPart.indexOf('"source"');
      if (sourceIdx > 0) {
        // Extract everything before "source"
        sourceLabelPrefix = labelPart.substring(0, sourceIdx);
        foundSource = true;
      }
      break;
    }
  }
  if (!foundSource) sourceLabelPrefix = modelLabel;
  let running = Math.round(m[keyPrefix + 'num_requests_running' + modelLabel] || 0); // [rid-stable-1014] v3 激活时行数权威切给注册行，需重赋值
  const queued = Math.round(m[keyPrefix + 'num_requests_waiting' + modelLabel] || 0);
  const reqGenCount = m[keyPrefix + 'request_generation_tokens_count' + modelLabel] || 0;
  const reqGenSum = m[keyPrefix + 'request_generation_tokens_sum' + modelLabel] || 0;
  const reqDecodeCount = m[keyPrefix + 'request_decode_time_seconds_count' + modelLabel] || 0;
  const reqDecodeSum = m[keyPrefix + 'request_decode_time_seconds_sum' + modelLabel] || 0;

  const keyPrefillCount = keyPrefix + 'request_prefill_time_seconds_count' + modelLabel;
  const keyPrefillSum = keyPrefix + 'request_prefill_time_seconds_sum' + modelLabel;
  const keyPromptTokensCount = keyPrefix + 'request_prompt_tokens_count' + modelLabel;
  const keyPromptTokensSum = keyPrefix + 'request_prompt_tokens_sum' + modelLabel;
  const reqPrefillCount = m[keyPrefillCount] || 0;
  const reqPrefillSum = m[keyPrefillSum] || 0;
  const reqPromptTokensCount = m[keyPromptTokensCount] || 0;
  const reqPromptTokensSum = m[keyPromptTokensSum] || 0;
  const avgPrefillTime = reqPrefillCount > 0 ? (reqPrefillSum / reqPrefillCount) : 0;
  const avgPrefillTokens = reqPromptTokensCount > 0 ? (reqPromptTokensSum / reqPromptTokensCount) : 0;

  const uncachedTotal = m[keyPrefix + 'prompt_tokens_by_source_total' + sourceLabelPrefix + '"source":"local_compute"}'] || 0;
  const cachedTotal   = m[keyPrefix + 'prompt_tokens_by_source_total' + sourceLabelPrefix + '"source":"local_cache_hit"}'] || 0;
  const grandTotal    = uncachedTotal + cachedTotal;
  const uncachedRatio = grandTotal > 0 ? (uncachedTotal / grandTotal) : 0;
  const avgUncachedTokens = reqPromptTokensCount > 0 ? (avgPrefillTokens * uncachedRatio) : 0;
  const avgCachedTokens = reqPromptTokensCount > 0 && grandTotal > 0 ? (avgPrefillTokens * (cachedTotal / grandTotal)) : 0;

  const avgTokensPerReq = reqGenCount > 0 ? (reqGenSum / reqGenCount) : 0;
  const avgDecodeTime = reqDecodeCount > 0 ? (reqDecodeSum / reqDecodeCount) : 0;

  let totalSpeed = 0;
  // 分母下限 300ms：主轮询 + 并发卡旁路 + 多标签页并存时 lastSnapshot.time
  // 被交替推进，会撞出几十毫秒的碎窗，delta/碎窗 放大成上千假峰值（实测 4473）
  if (elapsedMs >= 300 && genTokensTotal > lastGenTokensTotal) {
    const delta = genTokensTotal - lastGenTokensTotal;
    totalSpeed = (delta / elapsedMs) * 1000;
  }

  // Per-request speed estimation
  let perRequestDecodeSpeed = 0;
  let perRequestPrefillSpeed = 0;

  if (avgDecodeTime > 0 && avgTokensPerReq > 0) {
    perRequestDecodeSpeed = avgTokensPerReq / avgDecodeTime;
  } else if (running > 0 && totalSpeed > 0) {
    perRequestDecodeSpeed = totalSpeed / running;
  }

  if (avgPrefillTime > 0 && avgUncachedTokens > 0) {
    perRequestPrefillSpeed = avgUncachedTokens / avgPrefillTime;
  }

  // ====== Request identity tracker（按端口独立）======
  // vLLM metrics carry no per-request id. Track request births via the
  // cumulative vllm:num_requests_total counter and attribute each new request
  // to a newly-appeared TCP connection on the instance port (best-effort caller IP).
  // Ids (REQ-n) and per-request token totals persist across polls.
  // 多卡多实例：每个端口一个独立 tracker，两实例的 REQ 编号/live 列表/预填充
  // 基线互不串扰（共用一个 tracker 会让两实例的 num_requests_total 相加、
  // live 列表互相挤占，并发显示错乱）。
  if (!global.__reqTrackers) global.__reqTrackers = new Map();
  const rtKey = String(port || 8000);
  let rt = global.__reqTrackers.get(rtKey);
  if (!rt) {
    rt = { seq: 0, lastTotal: null, live: [], conns: new Set() };
    global.__reqTrackers.set(rtKey, rt);
  }
  let totalStarted = 0;
  for (const k of Object.keys(m)) {
    if (k.startsWith('vllm:num_requests_total')) totalStarted += (m[k] || 0);
  }
  // Established peer IPs on the instance port (server side)
  let peers = new Set();
  const ssPort = port || 8000;
  try {
    const ssOut = require('child_process').execSync(
      `ss -tHn state established '( sport = :${ssPort} )' 2>/dev/null || true`,
      { encoding: 'utf8', timeout: 4000 });
    ssOut.split('\n').forEach(line => {
      for (const c of line.trim().split(/\s+/)) {
        if (c && c.includes(':') && !c.endsWith(':' + ssPort) && /^[\d.:]+$/.test(c)) {
          peers.add(c.split(':')[0]);
          break;
        }
      }
    });
  } catch (e) { /* ss unavailable — identities degrade to '—' */ }

  if (rt.lastTotal === null) {
    rt.lastTotal = totalStarted; // first sight: baseline, no retroactive ids
  } else if (totalStarted < rt.lastTotal) {
    // Counter went backwards: vLLM restart or stats-reset rebaselining
    rt.lastTotal = totalStarted; rt.live = []; rt.conns = new Set();
  } else {
    const born = totalStarted - rt.lastTotal;
    const newPeers = [...peers].filter(ip => !rt.conns.has(ip));
    for (let i = 0; i < born; i++) {
      // Keep-alive reuse: no fresh connection, but if there is exactly one
      // connected client it is almost certainly the caller.
      const ip = newPeers[i]
        || (newPeers.length === 0 && peers.size === 1 ? [...peers][0] : '—');
      {
        const claim = taskForwardClaim2(port);
        if (claim && claim.crid) linkTaskCrid(claim.taskId, claim.crid);
        rt.live.push({ id: 'REQ-' + (++rt.seq), ip: ip, startedAt: Date.now(), tokens: 0, taskId: claim ? claim.taskId : null, crid: claim ? claim.crid : null });
      }
    }
  }
  rt.lastTotal = totalStarted;
  rt.conns = peers;

  // v3 流状态提前读取（readLiveStream 自带节流，幂等）：retire/fill 循环需要
  // 知道本轮 v3 是否激活——否则 v2→v3 切换首轮仍走 v2 retire 规则，会误丢
  // 正在流上活跃的刚出生行（REQ 号断一次）。
  readLiveStream();
  const v3s = v3StreamFor(port);
  const v3Active = v3s.fresh;
  // Align the live list with the running gauge. Completions are normally the
  // oldest requests (FCFS), but a request that was born AND completed between
  // two polls must not knock out a long-running one — retire young entries
  // (<2s old) first.
  // 排队请求保活：超过 running 的多余行，若「本轮刚出生」（仍排队未进 running
  // 批次）则移到 rt.hold 保留身份（REQ 号/任务号），调度后优先回归，保证
  // 排队→预填充→输出全程同一编号；1-2.5s 前出生的行视为出生+完成于两轮之间，
  // 直接丢弃；其余按 FCFS 视为完成。
  while (rt.live.length > running) {
    const nowTs = Date.now();
    if (v3Active || rt.v3Count) {
      // v3 模式（上轮流激活）：已绑 rid 的行 = 引擎活跃真值，只退未绑行
      //（预填充完成/排队取消的那批）；全绑满仍超（gauge 滞后瞬态）退最老
      // 绑定行并把身份按 rid 暂存，本轮 ②③④ 重新绑定时原样复活。
      let idx = rt.live.findIndex(r => !r.v3rid && nowTs - r.startedAt >= 1000);
      if (idx === -1) idx = rt.live.findIndex(r => !r.v3rid);
      if (idx !== -1) {
        const r = rt.live.splice(idx, 1)[0];
        if (r && !r.tokens && !r.decodeStart && !r.filled) {
          rt.hold = rt.hold || [];
          if (rt.hold.length < 300) rt.hold.push(r);
        }
        continue;
      }
      // [rid-stable-1014] 全绑仍超发（gauge 滞后瞬态）：只退「rid 已不在本轮活跃集」的绑定行，
      // 活跃 rid 的行一律保留——行数权威在 v3 流，不跟滞后 gauge 抖动（退绑→重绑正是
      // 实测「同一请求 rid 每拍换、完成 rid 被绑回」错位的根源）。
      const activeRids = new Set(v3s.active.map(st => st.rid));
      const sidx = rt.live.findIndex(r2 => r2 && r2.v3rid && !activeRids.has(r2.v3rid));
      if (sidx === -1) break;
      const r = rt.live.splice(sidx, 1)[0];
      if (r && r.v3rid) {
        rt.parked = rt.parked || new Map();
        if (rt.parked.size < 500) rt.parked.set(r.v3rid, { id: r.id, taskId: r.taskId, crid: r.crid, ip: r.ip, startedAt: r.startedAt, at: nowTs });
      }
      continue;
    }
    let idx = rt.live.findIndex(r => nowTs - r.startedAt >= 1000 && nowTs - r.startedAt < 2500);
    if (idx !== -1) { rt.live.splice(idx, 1); continue; }
    idx = rt.live.findIndex(r => nowTs - r.startedAt < 1000);
    if (idx !== -1) {
      const r = rt.live.splice(idx, 1)[0];
      // 仅「出生行」（fill 之前创建的排队行）可保活；fill 创建的行是 running
      // 请求（刚启动的 prefill tokens 仍为 0、decodeStart 未设，不能误判为排队）
      if (r && !r.tokens && !r.decodeStart && !r.filled) {
        rt.hold = rt.hold || [];
        if (rt.hold.length < 300) rt.hold.push(r); // 排队保活（上限防泄漏）
      }
      continue;
    }
    rt.live.shift();
  }
  // 排队保活行过期清理（客户端取消等，10 分钟未调度即丢弃）
  if (rt.hold && rt.hold.length) {
    const hc = Date.now() - 600000;
    rt.hold = rt.hold.filter(r => Date.now() - r.startedAt < hc);
  }
  // done 即时退场后 gauge 可能还差 1 轮才回落：本轮少建对应数量的补位行，
  // 防 ghost「预填充中」一闪（下轮 gauge 自然追上，抑制量自动归零）。
  const _dr = Math.max(0, Math.min(rt.doneRetired || 0, running));
  while (rt.live.length < running - _dr) {
    // 最老的排队请求优先调度（FCFS 近似）：回归原行，身份（REQ 号/任务号）不丢
    const held = (rt.hold && rt.hold.length) ? rt.hold.shift() : null;
    if (held) { held.filled = true; rt.live.push(held); }
    else {
      // 本环境 metrics 无 num_requests_total（vLLM 0.27.1），请求不会走 birth
      // 路径，REQ 行全靠这里创建 → 任务号也必须在此认领（FIFO 对应转发顺序）
      {
        const claim = taskForwardClaim2(port);
        if (claim && claim.crid) linkTaskCrid(claim.taskId, claim.crid);
        rt.live.push({ id: 'REQ-' + (++rt.seq), ip: '—', startedAt: Date.now(), tokens: 0, taskId: claim ? claim.taskId : null, crid: claim ? claim.crid : null, filled: true });
      }
    }
  }

  // ====== REQ 行 ↔ 引擎请求匹配（vLLM 实时 prefill 数据）======
  // vLLM 请求没有客户端可见的 id，控制台的 REQ-n 是合成的。这里按到达时间
  // （引擎侧 arrival）就近匹配，已匹配的行记住 ppRid 持续复用；vLLM 重启
  // （sid 变化）时清除全部匹配重新建立。
  // 容差说明：REQ 行的 startedAt 可能是「请求到达代理」或「请求进入 running
  // 批次」（排队请求在 max-num-seqs 满时等待后被调度，行才创建）。两者与引擎
  // arrival 的偏差可达排队时长（几十秒），故容差放宽到 120s；近距优先 + 已匹配
  // 复用保证并发请求不会互相抢错。
  if (livePrefill && livePrefill.sid && livePrefill.byRid && livePrefill.byRid.size > 0) {
    const lpm = livePrefill.byRid;
    const used = new Set();
    for (let i = 0; i < rt.live.length; i++) {
      const lv = rt.live[i];
      if (!lv) continue;
      if (rt.ppSid !== livePrefill.sid) {
        // 会话变化（vLLM 重启）：旧匹配作废
        lv.ppRid = null; lv.ppLastDone = undefined; lv.ppEstDone = undefined;
      }
      const target = lv.startedAt / 1000;
      // v2 精确认领（09-17）：代理 tee 首帧捕获的引擎 rid（taskId→crid→rid 链，
      // 见 __clientReqRid）若存在于本引擎流 → 直接绑定，跳过 arrival 最近邻。
      // 并发同秒启动时最近邻会错绑（arrival 几乎相同），错绑行会显示别人的
      // 输出序列——精确链把这类行全部救回。
      if (lv.taskId || lv.crid) {
        const _crid = lv.crid || (global.__taskCrid && global.__taskCrid.get(lv.taskId));
        const _ce = _crid && global.__clientReqRid ? global.__clientReqRid.get(_crid) : null;
        const _prid = _ce ? ridFind(lpm, _ce.rid) : null;
        if (_prid) {
          if (lv.ppRid !== _prid) {
            lv.ppRid = _prid; lv.ppLastDone = undefined; lv.ppEstDone = undefined;
            lv.cHist = []; // 换绑序列作废，防旧序列污染测速窗
          }
          lv.ppExact = true;
          used.add(lv.ppRid);
          continue;
        } else if (lv.ppExact) lv.ppExact = false; // rid 记录已被清理：回落最近邻
      }
      // 已进入 decode 的行放宽回绑：只允许绑定「prefill 已完成」的引擎记录
      // （computed+cached ≥ prompt_total，防止旧 decode 行把仍在 prefill 的
      // 新请求记录抢走），容差 300s（覆盖 decode 阶段引擎快照生失效的间隔）；
      // prefill/排队行保持 120s 就近匹配。
      const isDecodeRow = !!lv.decodeStart || lv.tokens > 0;
      // 09-06 修复：prefill 行绑定的引擎记录如果已完成（computed+cached >= prompt_total），
      // 说明匹配错误（并发流量下 arrival 时间碰撞），清除重绑到未完成的记录。
      if (lv.ppRid && lpm.has(lv.ppRid)) {
        const prevRec = lpm.get(lv.ppRid);
        const prevFinished = (prevRec.computed || 0) + (prevRec.cached || 0) >= Math.round(prevRec.prompt_total || 0);
        if (!isDecodeRow && prevFinished && !lv.ppExact) {
          // prefill 行绑到了已完成记录 → 匹配错误，清除重绑（exact 绑定的
          // 已完成记录是真的完成了，不清）
          lv.ppRid = null;
          lv.ppLastDone = undefined;
          lv.ppEstDone = undefined;
        } else {
          used.add(lv.ppRid);
          continue;
        }
      }
      let best = null, bestDist = isDecodeRow ? 300 : 120;
      for (const [rid, rec] of lpm) {
        if (used.has(rid)) continue;
        if (isDecodeRow) {
          const finished = (rec.computed || 0) + (rec.cached || 0) >= Math.round(rec.prompt_total || 0);
          if (!finished) continue;
        } else {
          // prefill 行：优先匹配未完成的记录（正在 prefill 的请求）
          const finished = (rec.computed || 0) + (rec.cached || 0) >= Math.round(rec.prompt_total || 0);
          if (finished) continue;
        }
        const d = Math.abs((rec.arrival || 0) - target);
        if (d < bestDist) { bestDist = d; best = rid; }
      }
      // 09-06：prefill 行第一轮只匹配未完成记录；如果全部已完成（无未完成记录），
      // 第二轮放宽到全部记录（兜底，防止无匹配）
      if (!best && !isDecodeRow) {
        for (const [rid, rec] of lpm) {
          if (used.has(rid)) continue;
          const d = Math.abs((rec.arrival || 0) - target);
          if (d < bestDist) { bestDist = d; best = rid; }
        }
      }
      lv.ppRid = best || null;
      if (best) used.add(best);
    }
    rt.ppSid = livePrefill.sid;
  } else if (rt.ppSid) {
    // 实时数据缺失/过期：清除匹配，回落估算逻辑
    rt.ppSid = null;
    for (const lv of rt.live) {
      if (lv) { lv.ppRid = null; lv.ppLastDone = undefined; }
    }
  }

  // ======================================================================
  // ====== v3：引擎每请求输出真值流 → 行绑定 + 精确相位（2026-10-06） ======
  // 流数据（readLiveStream）来自引擎内部逐步骤上报，rid 在流 = 该请求已过
  // TTFT 且在产出 = 精确「输出中」；running − 流内数 = 预填充行数。行↔rid
  // 绑定只决定标签（ip/任务号/REQ 号）挂到哪一行——数值永远跟着 rid 走，
  // 绑定错位不再产生错误速度（v2 顽疾的根）。
  // ======================================================================
  const v3StateById = new Map(); // rid -> st（本轮活跃）
  if (v3Active) {
    const nowV3 = Date.now();
    for (const st of v3s.active) v3StateById.set(st.rid, st);
    // 身份暂存表：行被 gauge 滞后误 retire 时按 rid 暂存身份，重新绑定时
    // 原样复活（REQ 号/任务号/ip/曲线 key 不闪断）
    if (!rt.parked) rt.parked = new Map();
    for (const [rid, pk] of rt.parked) { if (nowV3 - pk.at > 60000) rt.parked.delete(rid); }
    const claimed = new Set();
    const bindRow = (lv, st, exact) => {
      // 首次绑定优先复活暂存身份（REQ 号连续、sparkline 不reset）
      const pk = rt.parked.get(st.rid);
      if (pk) {
        lv.id = pk.id; lv.taskId = pk.taskId; lv.crid = pk.crid; lv.ip = pk.ip;
        rt.parked.delete(st.rid);
      }
      // [v3-pin-stale-1006] 复活暂存身份时同步清除 stale goneAt：行在 gauge
      // 滞后窗（≤5s）内按 rid 暂存、随后 bindRow 复活，若不重清，v3TouchRow
      // 活跃支路会把 goneAt 当 decodeStart 起点 → 新请求 avg 分母虚大被稀释。
      if (lv.v3goneAt && st.lastLineT) lv.v3goneAt = Math.round(st.lastLineT * 1000);
      lv.v3rid = st.rid; lv.v3Exact = !!exact;
      // 接管过的行（曾是别的已完成请求）复位旧身份钉与 decode 起点，再按
      // 新 rid 重钉——否则 elapsed 继承旧行虚高、avg_speed 分母错。
      // [v3avg-pin-1003] 换绑必须同时清钉住均值：否则旧请求冻结的 v3Avg 会
      // 在新 rid 首轮活跃赋值前被显示（接管瞬间闪现别人的速度）。
      if (lv.v3goneAt || (lv.v3born && lv.tokens === 0)) { lv.v3born = false; lv.decodeStart = null; lv.v3Avg = undefined; lv.v3AvgFrozen = undefined; }
      if (st.arrival > 0 && !lv.v3born) { lv.startedAt = Math.round(st.arrival * 1000); lv.v3born = true; } // 引擎真实进队时刻，钉一次
      // 断开 v2 时代残留的 tee 绑定（字符估算源），v3 行数值只认引擎真值
      lv.boundLive = null; lv.boundLiveId = null; lv.boundSpd = undefined;
      v3TouchRow(lv, st);
      claimed.add(st.rid);
    };
    // ① 粘性：上轮绑定且 rid 仍活跃 → 保持（REQ 号不跳）
    for (const lv of rt.live) {
      if (!lv || !lv.v3rid) continue;
      const st = v3StateById.get(lv.v3rid);
      if (st) { v3TouchRow(lv, st); claimed.add(lv.v3rid); }
      else if (v3s.doneIds && v3s.doneIds.has(lv.v3rid)) {
        // 流上完成行（f:1）= 引擎真实完成 → 标记即时退场（不等 gauge，实测
        // gauge 滞后会让完成行滞留数秒、avg_speed 被继续的时间分母稀释）
        lv.v3done = true;
      }
      else { lv.v3goneAt = nowV3; lv.v3rid = null; lv.v3Exact = false; lv.v3AvgFrozen = undefined; } // [v3avg-pin-1003] rid 消失≠请求完成：清钉住值，行若重新绑回同一 rid（流瞬断恢复）继续实时重算
    }
    // ② 精确认领（代理流量）：taskId→crid→SSE id→rid
    for (const lv of rt.live) {
      if (!lv || lv.v3rid) continue;
      const crid = lv.crid || (global.__taskCrid ? global.__taskCrid.get(lv.taskId) : null);
      if (!crid) continue;
      const e = global.__clientReqRid && global.__clientReqRid.get(crid);
      if (!e) continue;
      const rid = ridFind(v3StateById, e.rid);
      if (!rid || claimed.has(rid)) continue;
      bindRow(lv, v3StateById.get(rid), true);
    }
    // ③ 直连流量：剩余 rid × 未绑行 按引擎 arrival vs 行 startedAt 最近邻
    const freeRids = [];
    for (const [rid] of v3StateById) { if (!claimed.has(rid)) freeRids.push({ rid, arrive: v3StateById.get(rid).arrival * 1000 }); }
    if (freeRids.length) {
      const pairs = [];
      for (let i = 0; i < rt.live.length; i++) {
        const lv = rt.live[i];
        if (!lv || lv.v3rid) continue;
        // [rid-stable-1014] 只允许「从未绑过 rid」的新行（代理/排队行）走最近邻；
        // 曾绑定（v3born）或丢过 rid（v3goneAt）的行不得抢别的 rid——回归 rid
        // 由 ④+parked 复活原身份。容差 30s→10s（同机 arrival vs 行创建偏差 <1s）。
        if (lv.v3born || lv.v3goneAt) continue;
        for (const fr of freeRids) {
          if (claimed.has(fr.rid)) continue;
          const d = Math.abs(fr.arrive - lv.startedAt);
          if (d < 10000) pairs.push({ i, rid: fr.rid, d });
        }
      }
      pairs.sort((a, b) => a.d - b.d);
      for (const pr of pairs) {
        const lv = rt.live[pr.i];
        if (!lv || lv.v3rid || claimed.has(pr.rid)) continue;
        bindRow(lv, v3StateById.get(pr.rid), false);
      }
      // ④ 无行可领的 rid（gauge 滞后期刚进 decode）：建行接管，保行数=真值
      for (const fr of freeRids) {
        if (claimed.has(fr.rid)) continue;
        const lv2 = { id: 'REQ-' + (++rt.seq), ip: '—', startedAt: Date.now(), tokens: 0, taskId: null, crid: null, filled: true };
        rt.live.push(lv2);
        bindRow(lv2, v3StateById.get(fr.rid), false);
        claimed.add(fr.rid);
      }
    }
    rt.v3Count = claimed.size;
    // 完成确认行即时退场（本轮 rowV2 供数循环在其后，行序尚未被读取，安全）
    const before = rt.live.length;
    rt.live = rt.live.filter(lx => !(lx && lx.v3done));
    rt.doneRetired = before - rt.live.length;
  } else if (rt.v3Count) {
    rt.doneRetired = 0;
    // v3 掉线（引擎重启/插件停写）：清全部绑定，回落原链路
    rt.v3Count = 0;
    for (const lv of rt.live) { if (lv) { lv.v3rid = null; lv.v3Exact = false; } }
  }
  // [rid-stable-1014] v3 激活时行数权威=注册行：一个活跃 rid 一行，gauge 滞后窗内
  // 行数=真并发（行循环/响应 running 统一取 live 长度；queued 独立不受影响）。
  if (v3Active && rt.live.length > running) running = rt.live.length;

  // ====== Determine per-request phases & speeds ======
  let activeRequests = [];
  const savedPerReq = perReqTokens || {};
  const nowPerReq = {};

  // ====== 实时预填充速度/大小（直方图增量实测，无需归属猜测）======
  // vLLM 的 request_prefill_kv_computed_tokens 直方图在每次 prefill 完成时记录
  // 「实际计算的 token 数」（不含缓存命中）。count/sum 的增量 = 刚完成的 prefill
  // 的平均 uncached token 量；request_prefill_time_seconds 增量给出耗时 → 实测速度。
  // 存入 rt.lastPrefillUncached / rt.lastPrefillSpeed：
  // prefill 行用它估计「总需」，decode 行显示最近一次完成 prefill 的实测值，
  // 每次 prefill 完成都会刷新 —— 替代原来「显示一个值就不动」的静态累计平均值。
  // 注意：必须在 if(running>0) 之外执行 —— 空闲时也要持续更新基线，否则首个
  // 请求进来时增量检测被跳过（基线未初始化），只能回落累计平均值。
  const ppComputedCount = m[keyPrefix + 'request_prefill_kv_computed_tokens_count' + modelLabel] || 0;
  const ppComputedSum = m[keyPrefix + 'request_prefill_kv_computed_tokens_sum' + modelLabel] || 0;
  if (rt.lastPpComputedCount !== undefined && ppComputedCount > rt.lastPpComputedCount) {
    const dCount = ppComputedCount - rt.lastPpComputedCount;
    const dSum = ppComputedSum - rt.lastPpComputedSum;
    if (dCount > 0 && dSum > 0) rt.lastPrefillUncached = Math.round(dSum / dCount);
    if (reqPrefillCount > rt.lastPpTimeCount) {
      const dTime = reqPrefillSum - rt.lastPpTimeSum;
      if (dTime > 0.05) rt.lastPrefillSpeed = parseFloat((dSum / dTime).toFixed(1));
    }
  }
  rt.lastPpComputedCount = ppComputedCount;
  rt.lastPpComputedSum = ppComputedSum;
  rt.lastPpTimeCount = reqPrefillCount;
  rt.lastPpTimeSum = reqPrefillSum;

  if (running > 0) {
    // Capture previous-poll token counts (for prefill→decode transition detection)
    for (let i = 0; i < running; i++) {
      const lv = rt.live[i];
      if (lv) lv.lastTokens = lv.tokens;
    }
    // 本轮显示基准：优先用最近一次实测预填充速度，无实测时回落累计平均值
    const livePrefillSpeed = (rt.lastPrefillSpeed && rt.lastPrefillSpeed > 0)
      ? rt.lastPrefillSpeed : perRequestPrefillSpeed;

    // ====== Per-request phase: ground truth = has this request passed TTFT? ======
    // v3（流激活）：相位 = 该行是否绑定了流上活跃 rid（引擎已过 TTFT 且在
    // 产出）——真值，不再需要任何 bootstrap 估计。
    // 无流回落：A request is in PREFILL iff it has generated 0 tokens so far
    // (never passed TTFT); once it has >=1 token it is in DECODE. We read this
    // directly from the per-request cumulative token tracker (rt.live[i].tokens).
    const phaseOf = new Array(running).fill('prefill');
    const zeroTokenIdx = [];
    if (v3Active) {
      for (let i = 0; i < running; i++) {
        const lv = rt.live[i];
        if (!lv) { zeroTokenIdx.push(i); continue; }
        if (lv.v3rid && v3StateById.has(lv.v3rid)) phaseOf[i] = 'decode';
        else if (lv.v3goneAt && Date.now() - lv.v3goneAt < 5000) {
          // [v3-pin-stale-1006] 冻结滞留窗 15s→5s：旧窗内行显示"输出中"但数值
          // 已停（tokens 冻结），elapsed 继续随墙钟涨 → 并发高发期截图上"8 tok ·
          // 34s · 0.2 tok/s"形态的窗口本体会被缩到 5s（配合分母钉住双保险）。
          // 刚完成（gauge 滞后）：保持「输出中」避免闪回「预填充中」。
          phaseOf[i] = 'decode';
        } else zeroTokenIdx.push(i);
      }
    } else {
    for (let i = 0; i < running; i++) {
      const lv = rt.live[i];
      if (lv && lv.tokens > 0) phaseOf[i] = 'decode';
      else zeroTokenIdx.push(i);
    }
    // Bootstrap: when tokens ARE being generated (totalSpeed>0), a request that just
    // passed TTFT may not have been credited yet (0 tokens). Use the decode-count
    // estimate to mark the most-recent 0-token requests as decode so they start
    // accumulating. (totalSpeed==0: 0-token requests stay prefill — still prefilling.)
    if (totalSpeed > 0 && zeroTokenIdx.length > 0) {
      // 引擎并发数估计优先：hist 模式 nEw = λ̄×W（Little 定律，λ 完成率/W 驻留
      // 均来自引擎直方图实测）> 「总吞吐÷历史平均单请求速度」间接推断。后者在
      // 长短请求混跑时把 decodeCount 估歪——长 prompt 并发下 0-token 行被误留在
      // 预填充相位（整行速度恒 --，且 numDecode 变小让其余行速度全偏）。
      let decodeCount;
      const _sp0 = global.__reqOutSamplers && global.__reqOutSamplers.get(String(port || config.vllmPort));
      if (_sp0 && _sp0.nEw !== undefined && Date.now() - (_sp0.winLastAt || 0) < 4000) {
        decodeCount = Math.max(1, Math.min(Math.round(_sp0.nEw), running));
      } else {
        const avgSpeedPerRequest = avgDecodeTime > 0 && avgTokensPerReq > 0 ? (avgTokensPerReq / avgDecodeTime) : 0;
        const numActiveDecodes = avgSpeedPerRequest > 0 ? totalSpeed / avgSpeedPerRequest : 0;
        decodeCount = Math.max(1, Math.min(Math.round(numActiveDecodes), running));
      }
      const tokenDecodeCount = running - zeroTokenIdx.length;
      let bootstrap = Math.max(0, decodeCount - tokenDecodeCount);
      bootstrap = Math.min(bootstrap, zeroTokenIdx.length);
      for (let j = 0; j < bootstrap; j++) phaseOf[zeroTokenIdx[zeroTokenIdx.length - 1 - j]] = 'decode';
    }
    } // ====== v3/非 v3 相位分支收口 ======

    // 实际处于 decode 的行数（bootstrap 之后）。总生成速度（generation_tokens_total
    // 增量）只由 decode 行产生，prefill 行产出为 0 —— 所以单行速度必须按 decode
    // 行数分摊，而不是按 running（含 prefill 行）分摊。旧逻辑用 running 分摊，
    // 混跑时（如 2 个 prefill + 2 个 decode）每行只拿到真实速度的一半，偏差可达数倍。
    let numDecode = 0;
    for (let i = 0; i < running; i++) if (phaseOf[i] === 'decode') numDecode++;
    const numPrefill = running - numDecode; // 当前处于预填充的行数（守恒分摊用）
    const decodeShareSpeed = numDecode > 0 ? (totalSpeed / numDecode) : 0;

    // ======================================================================
    // ====== 行级瞬时速度供数（优先级：v3 引擎流 > v2 估计链 > 均摊） ======
    // v3（2026-10-06 重设计，见 readLiveStream 模块注释）：行绑定的 rid 在
    // 引擎输出真值流上有活跃样本 → 速度 = 自身样本滑窗差分（引擎实测、逐
    // 请求独立、零摊分），累计/相位/TTFT 全部随 rid 精确接管。
    // v2（保留为无流实例的降级链）：/metrics 每请求 gauge（官方 0.30 无此
    // 族，仅带 gauge 的定制实例命中）与直方图驻留 + Little 定律估计——
    // 各行同值、有滞后，前端标「分摊」。v3 激活时整段跳过。
    // ======================================================================
    const _nowV2 = Date.now();
    const sp = reqOutSamplerFor(port);
    sp.lastSeenAt = _nowV2; // 端口活跃标记（清理器据此回收死亡端口的采样器）
    let v2Used = false; // 本轮存在每请求实测行 → 后续回落估计路径整体跳过
    const rowV2 = new Array(running).fill(null); // {spd,gen,src} 行级瞬时速度
    if (v3Active) {
      const nowV3b = Date.now();
      for (let i = 0; i < running; i++) {
        const lv = rt.live[i];
        if (!lv || phaseOf[i] !== 'decode') continue;
        const st = lv.v3rid ? v3StateById.get(lv.v3rid) : null;
        if (st) {
          const rs = v3Rate(st, nowV3b);
          v3TouchRow(lv, st);
          rowV2[i] = { spd: rs ? rs.spd : undefined, gen: st.g, src: lv.v3Exact ? 'v3-exact' : 'v3' };
          lv.v3lastSec = v3LastSec(st, nowV3b);
          v2Used = true;
        } else if (lv.v3goneAt && nowV3b - lv.v3goneAt < 5000) {
          // [v3-pin-stale-1006] 与相位支路同步 15s→5s：刚完成（gauge 滞后）
          // 瞬时速度不出数，累计冻结在终值
          rowV2[i] = { spd: undefined, gen: lv.tokens, src: 'v3' };
          v2Used = true;
        }
      }
      rt.__v3 = { n: rt.v3Count || 0, rids: v3s.active.length };
      rt.__v2 = { n: 0, sum: 0, reqs: 0, cfgOk: sp.cfgOk, hist: false, v3: true };
    }
    if (!v3Active) {
      {
      // ---- 归属阶段（幂等：已绑定行保持，新行重配） ----
      const claimed = new Set(); // 本轮已被行占用的 rid（orphan/最近邻守卫）
      // ① 精确认领：taskId→crid→rid（代理流量）
      for (let i = 0; i < running; i++) {
        const lv = rt.live[i];
        if (!lv) continue;
        if (lv.roRidExact && sp.reqs.has(lv.roRidExact)) {
          claimed.add(lv.roRidExact);
          if (lv.roRid !== lv.roRidExact) { lv.roRid = lv.roRidExact; lv.roRidStickyAt = _nowV2; }
          continue;
        }
        if (lv.roRidExact && !sp.reqs.has(lv.roRidExact)) lv.roRidExact = null; // 引擎序列已消失
      }
      for (let i = 0; i < running; i++) {
        const lv = rt.live[i];
        if (!lv || lv.roRidExact || !lv.taskId) continue;
        const crid = lv.crid || (global.__taskCrid ? global.__taskCrid.get(lv.taskId) : null);
        if (!crid) continue;
        const e = global.__clientReqRid.get(crid);
        if (!e) continue;
        const rid = ridFind(sp.reqs, e.rid); // 容忍引擎内部 rid 的 -yyy 尾缀
        if (!rid || claimed.has(rid)) continue;
        if (sp.reqs.has(rid)) { claimed.add(rid); lv.roRidExact = rid; lv.roRid = rid; lv.roRidStickyAt = _nowV2; }
      }
      // ② 最近邻兜底（直连流量）：粘性优先 + 未绑定行 × 空闲 rid 按
      //    |(ttft − dec) − startedAt/1000| 全局最近邻一对一（≤30s）。
      //    引擎到达时刻 = ttft − dec（decode 前近似请求进引擎时刻）。
      //    粘性有效期 10s：超时（gauge 空窗/实例重启）则重配。
      const freeRids = [];
      for (const [rid, r] of sp.reqs) {
        if (claimed.has(rid)) continue;
        if (r.ttft === undefined || r.dec === undefined) continue;
        freeRids.push({ rid, arrive: r.ttft - (r.dec || 0) });
      }
      // 清理：被占用但行已消失/超时的绑定（占用表按端口隔离 = sp.occ）
      const occ = sp.occ;
      const occCutoff = _nowV2 - 20000;
      for (const [rid, t] of occ) { if (t < occCutoff) occ.delete(rid); }
      for (let i = 0; i < running; i++) {
        const lv = rt.live[i];
        if (!lv || lv.roRidExact || !lv.roRid) continue;
        const busy = occ.get(lv.roRid);
        const mine = busy === undefined || busy === lv.startedAt;
        const fresh = sp.reqs.has(lv.roRid) && lv.roRidStickyAt !== undefined && (_nowV2 - lv.roRidStickyAt) < 10000;
        if (mine && fresh) { claimed.add(lv.roRid); occ.set(lv.roRid, lv.startedAt); }
        else lv.roRid = null;
      }
      const pairs = [];
      for (let i = 0; i < running; i++) {
        const lv = rt.live[i];
        if (!lv || lv.roRidExact || lv.roRid) continue;
        const target = lv.startedAt / 1000;
        for (const fr of freeRids) {
          if (claimed.has(fr.rid)) continue;
          const d = Math.abs(fr.arrive - target);
          if (d < 30) pairs.push({ i, rid: fr.rid, d });
        }
      }
      pairs.sort((a, b) => a.d - b.d);
      for (const pr of pairs) {
        const lv = rt.live[pr.i];
        if (!lv || lv.roRid || lv.roRidExact || claimed.has(pr.rid)) continue;
        claimed.add(pr.rid); lv.roRid = pr.rid; lv.roRidStickyAt = _nowV2;
        occ.set(pr.rid, lv.startedAt);
      }
      // ---- 测速阶段 ----
      for (let i = 0; i < running; i++) {
        const lv = rt.live[i];
        if (!lv || phaseOf[i] !== 'decode' || !lv.roRid) continue;
        const r = sp.reqs.get(lv.roRid);
        if (!r) continue;
        const rs = reqOutSpeed(r, _nowV2);
        if (rs) {
          v2Used = true;
          rowV2[i] = { spd: rs.spd, gen: rs.gen, src: lv.roRidExact ? 'exact' : 'nearest' };
          // 输出真值直接接管累计（引擎值优先于估算；允许回落修正）
          if (rs.gen > lv.tokens) lv.tokens = rs.gen;
        } else if (r.hist && r.hist.length) {
          const lastS = r.hist[r.hist.length - 1];
          if (lastS.gen > lv.tokens) lv.tokens = lastS.gen; // 窗口未成熟：至少接管累计
        }
      }
      // ---- v2 直方图驻留兜底（实例无 req_id gauge 时的引擎真值回退） ----
      // 进入兜底前清除残留 rid 绑定（该实例本无 gauge；绑到上轮 rid = 错速错数）
      if (sp.cfgOk === false) {
        for (let i = 0; i < running; i++) {
          const lv = rt.live[i];
          if (lv && (lv.roRid || lv.roRidExact)) { lv.roRid = null; lv.roRidExact = null; }
        }
      }
      // Little 定律：速度 = 实例短窗吞吐 thrWin ÷ N（N=在途 decode 驻留
      // genAvgWin，桶差分保守下界）。驻留与吞吐都是引擎真值 → 每行分得
      // 同值（齐步 decode 真值本就相同；非齐步时各自窗口差更准）。
      if (!v2Used && sp.cfgOk === false) {
        const hm = reqOutHistModeSpeed(sp, _nowV2);
        if (hm && hm.n > 0 && hm.thr > 0.5 && numDecode > 0) {
          // 并发数取「引擎估计 nEw」与「行数 numDecode」的平衡：nEw 偏离行数
          // ≤1 视为一致→按行数均分（个体公平）；偏离大（phase 误判/行泄漏）
          // →按 nEw（引擎守恒优先，宁可每行偏高不少给）。
          const nEst = Math.abs(hm.n - numDecode) <= 1 ? numDecode : Math.max(1, Math.min(Math.round(hm.n), numDecode + 1));
          const v2HistSpd = hm.thr / nEst;
          for (let i = 0; i < running; i++) {
            if (phaseOf[i] !== 'decode') continue;
            rowV2[i] = { spd: v2HistSpd, gen: undefined, src: 'hist' }; // src 值 hist→前端映射 v2-hist
          }
          // 累计输出不做均分回填（hist 只提供速度）：每行 gen 仍由 tee 文本估
          // / v1 守恒积分负责。不置 v2Used——保留 tee 采集与 residual 兜底。
          rt.__v2Hist = { win: hm.win, thr: hm.thr, n: nEst };
        }
      }
      {
        // v2 命中统计（供守恒/诊断字段）
        let v2N = 0, v2Sum = 0;
        for (let i = 0; i < running; i++) if (rowV2[i]) { v2N++; v2Sum += rowV2[i].spd; }
        rt.__v2 = { n: v2N, sum: v2Sum, reqs: sp.reqs.size, cfgOk: sp.cfgOk, hist: !!rt.__v2Hist };
      }
    }
    } // ====== v2 降级链收口（v3 激活时整段不执行） ======
    // ====== 回落路径（v2 无数据时启用：实例级短窗吞吐 + tee/引擎滑窗）======
    let instTotalThroughput;
    // v2-hist 模式：直方图驻留已建立时优先用 sampler 同口径吞吐（避免与
    // ticker 滑窗窗口不一致导致 boundScale 偏离 1）
    if (rt.__v2Hist && sp.thrWin !== undefined && sp.thrWin > 0) {
      instTotalThroughput = sp.thrWin;
    } else if (numDecode > 0 && ticker && Array.isArray(ticker.genHist) && ticker.genHist.length >= 2) {
      const _hnow = Date.now();
      const _h = ticker.genHist.slice(-4);
      const _b = _h[_h.length - 1];
      if (_b && _hnow - _b.t <= 3500) {
        const _a = _h[0];
        const _dt = (_b.t - _a.t) / 1000;
        if (_dt >= 0.5) {
          instTotalThroughput = Math.max(0, (_b.g - _a.g) / _dt);
        }
      }
    }

    // ====== 行级实测速度（09-15c）：代理 tee 流 ↔ REQ 行绑定 ======
    // 引擎 /metrics 没有每请求计数器，齐步 decode 的各行历来同值（=总吞吐
    // ÷ decode 行数）。8889 代理层对每条流式响应做 tee（__liveStreams，
    // entry.tokens = 每行增量文本的 token 估算（09-16 起；旧版按行计数，
    // MTP 一步一行时齐步各行数值恒相同；直连流量无 tee）。
    // 绑定 = 同实例端口 + sticky（沿用上轮绑定）+ 全局最近邻（|行到达−流到达|
    // 最小优先，≤120s，一对一）。绑定行显示自己的实测速度；未绑定行的残差
    // =（引擎短窗总吞吐 − 已绑定实测之和）÷ 未绑定行数，守恒（各行之和≈引擎
    // 实测）；直连流量引擎侧本就没有每请求数据，只能拿残差分摊值，不编造。
    const _bnow = Date.now();
    const liveCands = [];
    const lsMap = v2Used ? null : ((global.__liveStreams && global.__liveStreams.map) || null);
    if (lsMap) {
      for (const [, e] of lsMap) {
        if (e.done || !Array.isArray(e.tk) || e.tk.length < 2) continue;
        if (e.port !== undefined && String(e.port) !== String(port)) continue;
        const b = e.tk[e.tk.length - 1];
        if (!b || _bnow - b.t > 3500) continue;
        const a = e.tk[Math.max(0, e.tk.length - 4)];
        const dt = (b.t - a.t) / 1000;
        liveCands.push({ e, spd: (dt >= 0.5 && b.n > a.n) ? (b.n - a.n) / dt : undefined });
      }
    }
    // ====== 引擎每请求实测（09-16，09-17 修订）：decode 阶段每请求输出增量 ======
    // 两个每请求数据源，都要求「窗口内确有新 token」且窗口 ≥1.0s：
    //  ① vLLM scheduler 每步流式回传的 num_computed_tokens（live-prefill jsonl，
    //     含直连流量，最准）；
    //  ② 代理 tee 流的输出 token 估算（走 8889 的流量）。
    // 关键修订（09-17，用户报「并发请求输出实时 tok/s 不准确」）：
    //  - 旧版只取窗口首个样本做基线（lv.cHist[0]）：decode 每行 8 个样本
    //    ≈13s 长窗 → 速度是十几秒的平均，瞬时变化完全跟不上；现按样本时间戳
    //    取「距现在最远且 ≥1s」的样本为基线（滑窗 ~1-3s）。
    //  - 旧版 token 零增量也出数：行刚进 decode、尚未产出任何 token 时，
    //    残差均摊立刻给满速（实测 REQ-86 显示 131 tok/s 时 gen=0）→ 每行
    //    头几秒系统性虚高。现无增量 → undefined（行显示 --，如实）。
    //  - 旧版 engine 源与 tee 源各自独立取窗：并发下若 ppRid 错配会显示
    //    别人的速度。现 tee 绑定行一律以自身流为准（每请求真值，与引擎
    //    boundScale 对齐后误差远小于错配风险）。
    const _spdNow = Date.now();
    const _SPD_WIN_MIN_MS = 1000;   // 速度窗口下限：低于此不出数（防噪声）
    const _SPD_WIN_MAX_MS = 3000;   // 速度窗口上限：瞬时性优先，不做长平均
    const _SPD_STALE_MS = 4000;     // 最后一个样本陈旧阈值：流卡死/断开则不出数
    // 通用滑窗测速：从 {t, c} 升序样本数组取「最远且 ≥winMin」的样本为基线
    // （受 winMax 约束：若无样本落在 [now-winMax, now-winMin]，取距现在 ≥winMin
    //  的最老样本）；末样本必须比基线新且有正增量。返回 {spd, span, idle}。
    const windowRate = (hist, nowMs) => {
      if (!Array.isArray(hist) || hist.length < 2) return null;
      const last = hist[hist.length - 1];
      if (!last || (nowMs - last.t) > _SPD_STALE_MS) return null;
      let base = null;
      for (let k = 0; k < hist.length - 1; k++) {
        const smp = hist[k];
        const age = nowMs - smp.t;
        if (age < _SPD_WIN_MIN_MS) continue;
        if (age <= _SPD_WIN_MAX_MS) { base = smp; break; }
        if (!base) base = smp; // 首个超窗样本：暂选，继续找更靠近窗内的
      }
      if (!base) { base = hist[0]; if ((nowMs - base.t) < _SPD_WIN_MIN_MS) return null; }
      const span = (last.t - base.t) / 1000;
      if (span < 1.0) return null;
      const d = (last.c || 0) - (base.c || 0);
      if (d <= 0) return { spd: 0, span: 0, idle: true };
      return { spd: d / span, span, idle: false };
    };
    const rowEngOut = new Array(running).fill(undefined);
    {
      const lpm2 = (!v3Active && livePrefill && livePrefill.byRid) || null;
      if (lpm2) {
        for (let i = 0; i < running; i++) {
          if (phaseOf[i] !== 'decode') continue;
          const lv = rt.live[i];
          if (!lv || !lv.ppRid) continue;
          const pr = lpm2.get(lv.ppRid);
          if (!pr || pr.computed === undefined) continue;
          if (!Array.isArray(lv.cHist)) lv.cHist = [];
          const cl = lv.cHist[lv.cHist.length - 1];
          if (!cl || cl.c !== pr.computed) {
            lv.cHist.push({ t: _spdNow, c: pr.computed });
            if (lv.cHist.length > 12) lv.cHist.shift();
          }
          if (pr.prompt_total) rowEngOut[i] = Math.max(0, Math.round(pr.computed - pr.prompt_total));
          // v2 引擎每请求流测速（实例不导出 Prometheus req_id gauge 时的每请求
          // 真值通道：dsh_vllm_logger 按 rid 流式上报 computed，computed−
          // prompt_total = 该请求已生成 token）。占用 rowV2 覆盖 hist 均摊行；
          // 窗口无增量=流冻结，保留 hist 聚合值（分层降级，均不出编造值）。
          const wrE = windowRate(lv.cHist, _spdNow);
          if (wrE && !wrE.idle) {
            let exactE = false;
            if (lv.taskId || lv.crid) {
              const _crid = lv.crid || global.__taskCrid.get(lv.taskId);
              const _e = _crid ? global.__clientReqRid.get(_crid) : null;
              if (_e && ridMatches(_e.rid, lv.ppRid)) exactE = true; // tee 首帧确认 rid
            }
            rowV2[i] = { spd: wrE.spd, gen: rowEngOut[i], src: exactE ? 'exact' : 'eng' };
          }
        }
      }
    }
    // 代理 tee 流 → 与 REQ 行绑定（sticky 沿用 + 全局最近邻一对一，≤120s）。
    // 09-17：绑定改用 windowRate 滑窗（旧版固定取 tk 前 4 样本 ≈3s 定窗）。
    const rowLiveSpd = new Array(running).fill(undefined);
    const rowLiveSpan = new Array(running).fill(undefined);
    const usedLiveIds = new Set();
    if (liveCands.length > 0) {
      for (let i = 0; i < running; i++) {
        if (phaseOf[i] !== 'decode') continue;
        const lv = rt.live[i];
        if (!lv || !lv.boundLiveId) continue;
        const c = liveCands.find(c2 => c2.e.id === lv.boundLiveId && !usedLiveIds.has(c2.e.id));
        if (c) {
          usedLiveIds.add(c.e.id);
          const wr = windowRate(c.e.tk.map(x => ({ t: x.t, c: x.n })), _bnow);
          if (wr) { rowLiveSpd[i] = wr.spd; rowLiveSpan[i] = wr.span; }
          lv.boundLive = c.e;
          lv.boundSpd = c.spd;
        } else { lv.boundLiveId = null; lv.boundLive = null; lv.boundSpd = undefined; }
      }
      const pairs = [];
      for (let i = 0; i < running; i++) {
        if (phaseOf[i] !== 'decode' || rowLiveSpd[i] !== undefined) continue;
        const lv = rt.live[i];
        if (!lv) continue;
        for (let j = 0; j < liveCands.length; j++) {
          const c = liveCands[j];
          if (usedLiveIds.has(c.e.id)) continue;
          const d = Math.abs((lv.startedAt || 0) - c.e.startedAt);
          if (d < 120000) pairs.push({ i, j, d });
        }
      }
      pairs.sort((a, b) => a.d - b.d);
      for (const p of pairs) {
        if (rowLiveSpd[p.i] !== undefined || usedLiveIds.has(liveCands[p.j].e.id)) continue;
        usedLiveIds.add(liveCands[p.j].e.id);
        const wr = windowRate(liveCands[p.j].e.tk.map(x => ({ t: x.t, c: x.n })), _bnow);
        if (wr) { rowLiveSpd[p.i] = wr.spd; rowLiveSpan[p.i] = wr.span; }
        rt.live[p.i].boundLiveId = liveCands[p.j].e.id;
        rt.live[p.i].boundLive = liveCands[p.j].e;
        rt.live[p.i].boundSpd = liveCands[p.j].spd;
      }
    }
    // ====== 行级速度仲裁 + 守恒分摊（09-17） ======
    // 每行候选值优先级：tee 绑定行实测 > 引擎每请求实测 > 残差。
    // boundScale 用「所有每请求候选值之和」对齐引擎短窗总吞吐（限 [0.5,3]），
    // 无每请求数据的行（直连流量）按缩放后的候选值之和分摊剩余吞吐——
    // 各行之和 ≈ 引擎实测，任何行都不再凭空满速。
    const rowCandSpd = new Array(running).fill(undefined); // 参与校准的每请求候选值（未缩放）
    const rowCandKind = new Array(running).fill(null);      // 'v2' | 'tee' | 'engine'
    for (let i = 0; i < running; i++) {
      if (phaseOf[i] !== 'decode') continue;
      // v2 每请求真值优先（decode 时间滑窗，精确到请求个体）；v1 tee/引擎作次选
      if (rowV2[i] !== null && rowV2[i].src !== 'hist') { rowCandSpd[i] = rowV2[i].spd; rowCandKind[i] = 'v2-' + rowV2[i].src; }
      else if (rowLiveSpd[i] !== undefined) { rowCandSpd[i] = rowLiveSpd[i]; rowCandKind[i] = 'tee'; }
    }
    let candSum = 0, candN = 0, unboundDecode = 0;
    for (let i = 0; i < running; i++) {
      if (phaseOf[i] !== 'decode') continue;
      if (rowCandSpd[i] !== undefined) { candSum += rowCandSpd[i]; candN++; }
      else unboundDecode++;
    }
    // 守恒校准（v2 修订）：行值整体缩放以对齐「引擎实例短窗总吞吐 ∪ Σv2」
    // （两者都实测；Σv2 在部分行窗口未成熟时可能低于总量，取大者保证不被
    // 低估）。缩放限 [0.5,3]：字符估算口径误差可双向；纯 v2 场景 Σ≈真值。
    const instThrEff = Math.max(instTotalThroughput !== undefined ? instTotalThroughput : 0, rt.__v2 ? rt.__v2.sum : 0);
    let boundScale = 1;
    if (candN > 0 && instThrEff > 0 && candSum > 0.5) {
      const estDenom = candSum + (candSum / candN) * unboundDecode;
      boundScale = Math.min(3, Math.max(0.5, instThrEff / estDenom));
    }
    let residualSpd;
    if (!v2Used && instThrEff > 0 && unboundDecode > 0) {
      residualSpd = Math.max(0, instThrEff - candSum * boundScale) / unboundDecode;
    }

    // Decode speed estimate from avg_decode_time
    // avg_decode_time = average time per token per request (seconds)
    // So 1/avg_decode_time = tokens per second per request
    let estimatedDecodeSpeed = 0;
    if (avgDecodeTime > 0) {
      estimatedDecodeSpeed = parseFloat((1 / avgDecodeTime).toFixed(1));
    } else if (perRequestDecodeSpeed > 0) {
      estimatedDecodeSpeed = perRequestDecodeSpeed;
    }

    for (let i = 0; i < running; i++) {
      const isPrefill = phaseOf[i] === 'prefill';
      const reqId = i + 1;
      const live = rt.live[i] || { id: 'REQ-' + reqId, ip: '—', startedAt: Date.now(), tokens: 0 };
      const lvV3st = live && live.v3rid ? v3StateById.get(live.v3rid) : null; // [v3-pin-stale-1006] elapsed 钉锚用
      let reqSpeed = 0;
      let reqPhase;
      let justEnteredDecode = false;

      if (isPrefill) {
        reqPhase = 'prefill';
        nowPerReq[reqId] = 0;
      } else {
        reqPhase = 'decode';
        // 首次进入 decode（越过 TTFT）记下时刻，作为「输出阶段」的起点：
        // avg_speed 只按 decode 时长求平均，不再把 prefill/排队/TTFT 等待算进
        // 分母（旧逻辑用整个生命周期做分母，长 prompt 请求显示值被严重拉低）。
        // 本轮不发放 token 积分（无法确定何时越过 TTFT，按整窗摊分会虚高），
        // 从下一轮起再开始累计。
        justEnteredDecode = !live.decodeStart;
        if (justEnteredDecode) live.decodeStart = Date.now();
        // 累计输出（tokens_generated / avg_speed 的数据源）优先每请求真值
        // （09-17）：旧版一律按 decodeShareSpeed 均摊积分，直连请求会吃掉
        // 代理请求的产出（两行同值）、并发下还会互相污染。每请求真值：
        //  ① 引擎 computed − prompt_total（rowEngOut）；② 代理 tee 流文本估
        //     （lv.boundLive.tokens，走 8889 的流量）。都没有才回落均摊积分。
        reqSpeed = 0;
        nowPerReq[reqId] = savedPerReq[reqId] || 0;
      }

      // 每请求真值累计回填（09-17，09-17b 修正）：每请求真值
      // （引擎 computed−prompt_total / 代理 tee 流文本估）可能因 rid 匹配
      // 漂移、字符估算口径而短暂低于已持有的累计值（含旧均摊积分，只高不低
      // 会误伤），故只有真值 ≥ 当前累计才覆盖；真值回落后又抬升时取抬升值。
      // 效果：单调不减，消除「gen 冻结 → 恢复跳变」的显示锯齿。
      if (reqPhase === 'decode') {
        let trueTok = -1;
        if (rowV2[i] !== null && rowV2[i].gen !== undefined) trueTok = Math.max(trueTok, rowV2[i].gen);
        if (rowEngOut[i] !== undefined) trueTok = Math.max(trueTok, rowEngOut[i]);
        // v3 行禁用 tee 字符估算（会顶掉刚钉准的引擎真值）；仅非 v3 行保留
        const _blv = rt.live[i] && rt.live[i].boundLive;
        if (_blv && !_blv.done && !live.v3rid) trueTok = Math.max(trueTok, Math.round(_blv.tokens || 0));
        if (trueTok >= 0 && trueTok >= live.tokens) live.tokens = trueTok;
      }
      // tokens_since_last: 本轮该行新增输出（优先每请求真值差，回落均摊）
      let tokensSinceLast = 0;
      {
        const prevTk = savedPerReq[reqId] || 0;
        if (reqPhase === 'decode' && (rowV2[i] !== null || (rt.live[i] && rt.live[i].boundLive))) {
          tokensSinceLast = Math.max(0, live.tokens - prevTk);
        } else if (reqPhase === 'decode' && elapsedMs > 0 && totalSpeed > 0) {
          tokensSinceLast = parseFloat((decodeShareSpeed * elapsedMs / 1000).toFixed(1));
        }
      }

      // A request born DURING this interval only existed part of it — crediting
      // the full interval share inflates its cumulative (and its average).
      const bornThisInterval = live.startedAt > (Date.now() - (elapsedMs || 0));
      if (reqPhase === 'decode' && tokensSinceLast > 0 && !bornThisInterval && !justEnteredDecode) {
        live.tokens += (rowV2[i] !== null || (rt.live[i] && rt.live[i].boundLive)) ? 0 : tokensSinceLast;
      }
      if (reqPhase === 'decode') nowPerReq[reqId] = Math.round(live.tokens);
      // 行级瞬时速度（v2 定版）：v2 真值（exact/nearest/hist）> 代理 tee
      // 实测 > 引擎实时实测（v1）> 残差分摊（仅 v2 全无时）。
      //  ① v2 hist 行由测速阶段统一发放（已按实例吞吐校准，scale=1）；
      //  ② v2 exact/nearest 与 tee/engine 候选同乘 boundScale：各行之和
      //     恒 ≈ 引擎短窗总吞吐（守恒）；
      //  ③ 刚进 decode 只用该行 tee 即时值，绝不用残差（旧版「刚输出就
      //     满速」的根因）；
      //  ④ residual 仅在 v2Used=false（无任何每请求真值）时才存在。
      let reqSpeed3s;
      let reqSpdSrc;
      if (reqPhase === 'decode') {
        if (rowV2[i] !== null && rowV2[i].src !== 'hist') {
          // v3/v2 每请求实测（引擎绝对值不缩放）。v3 窗未成熟/流冻结时 spd 为
          // undefined → 不发瞬时数（前端回落全程均值，同样精确），绝不编造。
          reqSpeed3s = rowV2[i].spd !== undefined ? parseFloat(rowV2[i].spd.toFixed(1)) : undefined;
          reqSpdSrc = (rowV2[i].src === 'v3' || rowV2[i].src === 'v3-exact') ? rowV2[i].src : 'v2-' + rowV2[i].src;
        } else if (rowV2[i] !== null && rowV2[i].src === 'hist') {
          // v2 直方图驻留（每行同值 = 实例吞吐÷引擎测得并发数 λ̄×W）。优先于
          // 逐行 tee：无每请求 gauge 的实例上 tee 依赖易漂移的 行↔流 配对，
          // 错配时各行显示互相错位的值（历史顽疾）；齐步 decode 下每行真值
          // 本就相同——引擎守恒值即真值，且天然稳定。
          reqSpeed3s = parseFloat(rowV2[i].spd.toFixed(1)); reqSpdSrc = 'v2-hist';
        } else if (rowCandSpd[i] !== undefined) {
          reqSpeed3s = parseFloat((rowCandSpd[i] * boundScale).toFixed(1));
          reqSpdSrc = rowCandKind[i];
        } else if (justEnteredDecode) {
          const _b0 = rt.live[i] && rt.live[i].boundSpd;
          if (_b0 !== undefined && _b0 > 0) {
            reqSpeed3s = parseFloat((_b0 * boundScale).toFixed(1)); reqSpdSrc = 'tee';
          }
        } else if (residualSpd !== undefined) {
          reqSpeed3s = parseFloat(residualSpd.toFixed(1)); reqSpdSrc = 'residual';
        }
      }
      // Per-request actual throughput = its own cumulative tokens / its own
      // DECODE-phase elapsed time (自 TTFT 之后起算)。这样「输出中」行显示的值
      // 就是它真实产出 token 的平均速度，prefill/TTFT 等待不再拉低分母；
      // 引擎真正停顿时（本轮增量 0、token 不再累计）均值会如实回落。
      // [v3-pin-stale-1006] v3 行 elapsed 钉在 min(now, lastOut+2s)：冻结滞留窗
      // 内卡片仍显示"输出中"，秒表随墙钟走会与钉住的 avg 互相矛盾（"8 tok ·
      // 34s · 0.2 tok/s"形态的显示层源头）；+2s 容差防行轮转时秒表倒跳。
      const _lastOutS = live.v3rid && (live.tokens || 0) > 0 && lvV3st && lvV3st.gLastT
        ? lvV3st.gLastT
        : (live.v3goneAt ? live.v3goneAt / 1000 : null);
      // [v3-pin-stale-1006] 秒表钉 min(now, 最后产出+2)：冻结滞留窗内 elapsed
      // 不再随墙钟增长（"8 tok·34s·0.2"形态的显示层源头）；行出生晚于最后
      // 产出（接管即停倒挂）钳 0，绝不发放 dilute 分母。
      const liveElapsed = _lastOutS !== null
        ? Math.max(0, Math.min(Date.now() / 1000, _lastOutS + 2) - live.startedAt / 1000)
        : (Date.now() - live.startedAt) / 1000;
      const decodeElapsed = live.v3rid ? liveElapsed : (Date.now() - (live.decodeStart || live.startedAt)) / 1000; // [v3-pin-stale-1006] v3 行分母同源（avg_speed 非 v3Avg 支路），冻结期不随墙钟稀释
      const avgDenom = decodeElapsed >= 0.5 ? decodeElapsed : liveElapsed;
      // Tokens THIS request produced in the last 1 second (server-side ticker):
      // the batch's measured last-second token total split across decode rows.
      // In lockstep decode every row gets the same count — that is the truth.
      const tk = (ticker && ticker.lastSecond) ? ticker.lastSecond : { tokens: 0 };
      // 「上一秒」：v3 行 = 该 rid 自身样本 1s 差分（每请求精确）；无 v3 沿用
      // 本批实测总产出的 decode 行分摊（齐步 decode 下各行真值本就相同）。
      const tokLastSec = reqPhase === 'decode' && rt.live[i] && rt.live[i].v3lastSec !== undefined
        ? rt.live[i].v3lastSec
        : (reqPhase === 'decode' && numDecode > 0
          ? Math.max(0, Math.round((tk.tokens || 0) / numDecode))
          : 0);

      // ====== 本行预填充统计：引擎实测 > 实测聚合守恒分摊 > 历史估算 ======
      // 1) 引擎实测（prefill_exact=true，vLLM 插件实时回传）：该请求的
      //    prompt_total / computed / cached 全为引擎值，进度与速率精确。
      // 2) 守恒分摊（无每请求数据时）：把 ticker 每秒实测的未缓存预填充吞吐
      //    （prompt_tokens_by_source local_compute 增量）按并发预填充行数均分
      //    —— 各行之和 = 实测聚合，可溯源；进度 = 分摊速率 × 耗时的累计。
      // 3) 兜底（无引擎实测也无实时聚合信号）：最近一次完成 prefill 的直方图
      //    实测值（原逻辑），前端标「估」。
      const ppRec = (livePrefill && livePrefill.byRid && live.ppRid)
        ? livePrefill.byRid.get(live.ppRid) : null;
      const ppTicker = (ticker && ticker.lastSecond) || {};
      const aggPP = ppTicker.uncachedPPS || 0; // 实测未缓存预填充吞吐（tok/s）
      const ppDt = elapsedMs > 0 ? elapsedMs / 1000 : 0;
      let prefillExact = false;
      let prefillS = 0;        // 预填充执行耗时（精确；完成时才有值）
      let prefillCached = 0;
      let prefillTotalUncached = rt.lastPrefillUncached || Math.round(avgUncachedTokens);
      let prefillDoneUncached = prefillTotalUncached;
      let rowPrefillTps = 0;
      if (ppRec) {
        prefillExact = true;
        prefillCached = Math.round(ppRec.cached || 0);
        prefillTotalUncached = Math.max(0, Math.round((ppRec.prompt_total || 0) - prefillCached));
        // 不含缓存命中口径：调度器 num_computed_tokens 包含缓存采纳的 token
        // （采纳瞬时计入，见 vllm scheduler num_computed_tokens 赋值），
        // 扣除 cached 即为真实计算量；完成判定 = 采纳 + 计算 ≥ 总 prompt。
        const ppComputed = Math.max(0, Math.round(ppRec.computed || 0));
        const ppFinished = (ppComputed + prefillCached) >= Math.round(ppRec.prompt_total || 0);
        prefillDoneUncached = ppFinished
          ? prefillTotalUncached
          : Math.min(prefillTotalUncached, Math.max(0, ppComputed - prefillCached));
        const ppDone = ppFinished || prefillDoneUncached >= prefillTotalUncached;
        // 本版 vLLM 的 prefill_stats 只在 prefill 完成（首个 token 输出）时回传
        // 一次，携带完整请求数据。完成时：执行耗时 = 完成时刻 − 引擎到达时刻
        // （含排队，并发≤2 时排队≈0）；速率 = 该请求 uncached 总量 / 耗时。
        if (ppDone && (ppRec.t || 0) > 0 && (ppRec.arrival || 0) > 0) {
          const dur = Math.max(0, ppRec.t - ppRec.arrival);
          prefillS = parseFloat(dur.toFixed(2));
          rowPrefillTps = dur > 0.05 && prefillDoneUncached > 0
            ? parseFloat((prefillDoneUncached / dur).toFixed(1)) : 0;
        } else if (!isPrefill) {
          // decode 行但引擎数据未标记完成（理论不应发生）：用累计均值兜底
          const ppAge = (ppRec.arrival || 0) > 0 ? (Date.now() / 1000 - ppRec.arrival) : 0;
          rowPrefillTps = ppAge > 0.5 && prefillDoneUncached > 0
            ? parseFloat((prefillDoneUncached / ppAge).toFixed(1)) : 0;
        } else {
          // 预填充中：速率 = 引擎两次进度更新间的平均速率。
          // 引擎 prefill 快照可能数秒才更新一次（DFlash2 实例每 chunk 计算
          // ~数秒），0.5s 轮询下相邻轮 done 常无变化 → 不能用「本轮窗口」算
          // 增量（会闪 0 或虚高）。改为：done 变化时用「自上次变化以来的真实
          // 时间」算速率并记住；引擎 8s 内未推进则保持该最近实测值，超过才归 0。
          const nowMs = Date.now();
          if (live.ppLastDone !== undefined) {
            const dDone = prefillDoneUncached - live.ppLastDone;
            if (dDone > 0) {
              const dT = (nowMs - (live.ppLastDoneAt || nowMs)) / 1000;
              if (dT > 0) {
                rowPrefillTps = parseFloat((dDone / dT).toFixed(1));
                live.ppRate = rowPrefillTps;
                live.ppRateAt = nowMs;
              }
              live.ppLastDoneAt = nowMs; // 仅进度变化时推进时间基准
            } else if (live.ppRate !== undefined && (nowMs - (live.ppRateAt || 0)) < 8000) {
              rowPrefillTps = live.ppRate; // 引擎未推进：保持最近实测速率
            } else {
              rowPrefillTps = 0;
            }
          } else {
            // 首轮（尚无历史）：到达时刻起算的累计均值兜底
            const ppAge = (ppRec.arrival || 0) > 0 ? (nowMs / 1000 - ppRec.arrival) : 0;
            rowPrefillTps = ppAge > 0.5 && prefillDoneUncached > 0
              ? parseFloat((prefillDoneUncached / ppAge).toFixed(1)) : 0;
            if (rowPrefillTps > 0) { live.ppRate = rowPrefillTps; live.ppRateAt = nowMs; }
            live.ppLastDoneAt = nowMs; // 首轮也建立时间基准
          }
          live.ppLastDone = prefillDoneUncached;
        }
      } else if (isPrefill && numPrefill > 0 && aggPP > 0) {
        // 守恒分摊：实测聚合按并发预填充行数均分（不再伪造行间抖动）
        rowPrefillTps = parseFloat((aggPP / numPrefill).toFixed(1));
        if (live.ppEstDone === undefined) live.ppEstDone = 0;
        if (ppDt > 0) live.ppEstDone += rowPrefillTps * ppDt;
        prefillDoneUncached = Math.min(prefillTotalUncached, Math.max(0, Math.round(live.ppEstDone)));
      } else if (isPrefill) {
        // 兜底估算（无引擎实测也无实时聚合信号）：最近完成 prefill 的实测值
        const estSpeed = (live.prefillSpeed && live.prefillSpeed > 0) ? live.prefillSpeed : livePrefillSpeed;
        rowPrefillTps = parseFloat(estSpeed.toFixed(1));
        prefillDoneUncached = Math.min(prefillTotalUncached, Math.max(0, Math.round(liveElapsed * estSpeed)));
      } else {
        // decode 行无引擎数据：填满总需（与旧逻辑一致），速度用最近完成实测值
        rowPrefillTps = parseFloat((livePrefillSpeed || 0).toFixed(1));
        prefillDoneUncached = prefillTotalUncached;
      }

      activeRequests.push({
        id: live.id,
        taskId: live.taskId || null, // 控制台任务号：PD 两腿同号，跨 GPU 关联
        ip: live.ip,
        started_at: live.startedAt,
        elapsed_s: Math.round(liveElapsed),
        tok_last_sec: tokLastSec,
        // v3 行：全程均值用 v3Avg（firstT 未知时自动退化为接管后窗口均值，
        // 分子分母同窗）；非 v3 行沿用 tokens/decodeElapsed 原式。
        avg_speed: (reqPhase === 'decode' && live.v3Avg !== undefined && live.v3Avg !== null)
          ? parseFloat(live.v3Avg.toFixed(1))
          : (liveElapsed > 0.5 ? parseFloat((live.tokens / Math.max(avgDenom, 0.5)).toFixed(1)) : 0),
        // 行级瞬时输出速度（09-15c）：绑定到代理 tee 流的行显示自己实测值，
        // 未绑定的 decode 行显示残差分摊值；前端既有契约优先显示此字段并标
        // 「近3s tok/s」，undefined 时回落 avg_speed 历史均值。
        avg_speed_3s: reqSpeed3s,
        // 瞬时速度来源（调试/审计用，前端不消费）：measured=该行代理流实测；
        // residual=无该流（直连流量）时的引擎残差分摊。
        speed_src: reqSpdSrc,
        // v2 诊断（绑定链可观测性）：pp=行绑定的引擎 rid 尾段，ex=是否 tee
        // 精确确认，cn=cHist 样本数（测速窗料），ck=taskId 是否认领到 crid。
        pp_rid: (live.v3rid || live.ppRid) ? String(live.v3rid || live.ppRid).slice(-8) : null,
        pp_exact: !!(live.v3Exact || live.ppExact),
        chist_n: Array.isArray(live.cHist) ? live.cHist.length : 0,
        has_crid: !!(live.crid || (live.taskId && global.__taskCrid && global.__taskCrid.get(live.taskId))),
        speed: reqPhase === 'prefill' ? 0 : reqSpeed,
        prefill_tps: rowPrefillTps,
        phase: reqPhase,
        tokens_generated: Math.round(live.tokens),
        tokens_prefill: Math.round(prefillTotalUncached),
        tokens_since_last: tokensSinceLast,
        prefill_total_uncached: prefillTotalUncached,
        prefill_done_uncached: prefillDoneUncached,
        prefill_cached: prefillCached,
        prefill_exact: prefillExact,
        prefill_s: prefillS,
      });
    }
  }

  if (running === 0 && reqGenCount > 0) {
    perRequestDecodeSpeed = avgTokensPerReq / avgDecodeTime;
  }
  // 排队中请求 = tracker 保活行（REQ+taskId）+ 代理已转发但尚未被认领的请求
  // （仍在等待调度，仅 taskId；认领发生在 fill 时 → 一进 running 即从排队
  // 消失、以同 T 号出现在 active）。1.5s 内的未认领记录视为刚转发即将运行，
  // 不显示；120s 未认领过期清理。
  // PD 幽灵清理：任务已在 decode 端口被认领（decode 腿已开跑）→ 其 prefill
  // 腿的未认领记录是已完成阶段的残留（prefill 太快没触发 fill 认领），不显示。
  const fwdQ = __taskForward.get(String(port || 8000)) || [];
  const fnow = Date.now();
  const decodePorts = new Set();
  for (const pm of Object.values(PD_MODEL_PORTS)) {
    if (pm && pm.decode) decodePorts.add(String(pm.decode));
  }
  const claimedOnDecode = new Set();
  for (const [pk, pq] of __taskForward) {
    if (!decodePorts.has(pk)) continue;
    for (const r2 of pq) if (r2.claimed && r2.taskId) claimedOnDecode.add(r2.taskId);
  }
  const fwdWaiting = fwdQ
    .filter(rec => !rec.claimed
      && !claimedOnDecode.has(rec.taskId)
      && (fnow - rec.at) > 1500
      && (fnow - rec.at) < 120000)
    .map(rec => ({ id: null, taskId: rec.taskId, ip: null, startedAt: rec.at }));
  const holdRows = (rt.hold || [])
    .filter(h => !h.taskId || !claimedOnDecode.has(h.taskId)) // PD 幽灵行同清
    .map(h => ({ id: h.id, taskId: h.taskId || null, ip: h.ip, startedAt: h.startedAt }));
  const allWaiting = holdRows.concat(fwdWaiting).slice(0, 20);

  return {
    // v3 诊断（前端不消费；curl /v1/internal/stats | jq .v3 排障用）：
    // active=v3 供数中 n=绑定行数 rids=流上活跃 rid skew=引擎↔本机时钟差(秒)
    // guard=时钟护栏触发行数（>0 说明流来自异机/异常时钟，数值已降级保守）
    _v3diag: { active: !!v3Active, n: (rt.v3Count || 0), rids: v3s.active ? v3s.active.length : 0, skew: v3s.skew || 0, guard: v3s.guardN || 0 },
    active_requests: activeRequests,
    waiting_requests: allWaiting,
    connected_clients: [...peers],
    running: running,
    queued: queued,
    total_completed_requests: Math.round(reqGenCount),
    avg_tokens_per_request: Math.round(avgTokensPerReq),
    avg_prefill_tokens_per_request: Math.round(avgUncachedTokens),
    avg_cached_tokens_per_request: Math.round(avgCachedTokens),
    avg_decode_time_seconds: parseFloat(avgDecodeTime.toFixed(3)),
    avg_prefill_time_seconds: parseFloat(avgPrefillTime.toFixed(3)),
    avg_speed_per_request: parseFloat(perRequestDecodeSpeed.toFixed(1)),
    total_generated_tokens_all_time: Math.round(reqGenSum),
    total_speed: totalSpeed,
    _perReqTokens: nowPerReq,
  };
}

// ====== 多卡多实例：从实例并发明细 ======
// 主实例的并发在 /v1/internal/stats 内联计算（快照落盘 metrics-snapshot.json）；
// 从实例（每张卡独立跑一个 vllm serve）用内存快照（控制台重启丢失，仅影响
// 速度类增量的首窗，可忽略）。每行 active_requests 打 port/gpu/model 标签，
// 前端按卡分组显示。
if (!global.__instSnapshots) global.__instSnapshots = new Map();

// KV Cache 池信息（主/从实例同口径）：usage_pct 取 vllm:kv_cache_usage_perc，
// 池容量取 vllm:cache_config_info 的 kv_cache_size_tokens（vLLM 0.27+ 提供），
// 按 model 标签对齐取值。
function buildKvCacheInfo(m) {
  const mlCachePct = 'vllm:kv_cache_usage_perc' + buildModelLabelFromMetrics(m, 'vllm:kv_cache_usage_perc{');
  const usage = m[mlCachePct] || 0;
  const out = { usage_pct: usage, max_tokens: 0, used_tokens: 0, block_size: 0, num_gpu_blocks: 0, prefix_cache_hits: 0 };
  const usageModel = mlCachePct ? ((mlCachePct.match(/"model_name":"([^"]*)"/) || [])[1] || '') : '';
  for (const k of Object.keys(m)) {
    if (k.indexOf('vllm:cache_config_info|') !== 0) continue;
    try {
      const labels = JSON.parse(k.substring(k.indexOf('|') + 1));
      if (usageModel && labels.model_name && labels.model_name !== usageModel) continue;
      const maxT = parseInt(labels.kv_cache_size_tokens);
      if (!isNaN(maxT) && maxT > 0) {
        out.max_tokens = maxT;
        out.block_size = parseInt(labels.block_size) || 0;
        out.num_gpu_blocks = parseInt(labels.num_gpu_blocks) || 0;
        break;
      }
    } catch (e) { /* 忽略无法解析的 info 行 */ }
  }
  // 已缓存 token 数 = 池容量 × 当前使用率（usage 为 0~1 的 block 级占用比例）
  out.used_tokens = out.max_tokens > 0 ? Math.round(out.max_tokens * usage) : 0;
  // 累计前缀缓存命中 token（vllm:prefix_cache_hits_total，counter，只增不减；按 model 标签对齐）
  const mlHits = 'vllm:prefix_cache_hits_total' + buildModelLabelFromMetrics(m, 'vllm:prefix_cache_hits_total{');
  out.prefix_cache_hits = Math.round(m[mlHits] || 0);
  return out;
}

function buildInstanceConcurrency(inst, data) {
  const m = parseMetrics(data);
  if (metricsNamespace(m) !== 'vllm') return null; // 只处理 vllm 实例
  const mlRunning = 'vllm:num_requests_running' + buildModelLabelFromMetrics(m, 'vllm:num_requests_running{');
  const running = Math.round(m[mlRunning] || 0);
  const genTokensTotal = Math.round(m['vllm:generation_tokens_total' + buildModelLabelFromMetrics(m, 'vllm:generation_tokens_total{')] || 0);
  let st = global.__instSnapshots.get(inst.port);
  if (!st) {
    st = { lastGen: genTokensTotal, lastTime: Date.now(), lastRunning: 0, perReqTokens: {} };
    global.__instSnapshots.set(inst.port, st);
  }
  // 计数器回退（实例重启）→ 重新定基线
  if (st.lastGen > genTokensTotal + 1000) {
    st.lastGen = genTokensTotal; st.lastTime = Date.now(); st.perReqTokens = {};
  }
  const elapsedMs = Date.now() - st.lastTime;
  const livePrefill = readLivePrefill(inst.pid);
  const ticker = getTicker(inst.port);
  const concurrency = computeConcurrencyDetails(
    m, genTokensTotal, st.lastGen, elapsedMs,
    st.lastRunning || 0, st.perReqTokens || {}, livePrefill, inst.port, ticker
  );
  st.lastGen = genTokensTotal;
  st.lastTime = Date.now();
  st.lastRunning = running;
  st.perReqTokens = concurrency._perReqTokens || {};
  const model = inst.servedName || inst.modelPath || '';
  for (const r of concurrency.active_requests) {
    r.port = inst.port; r.gpu = inst.gpu; r.model = model;
    r.gpus = (inst.gpus && inst.gpus.length) ? inst.gpus.slice() : (inst.gpu != null ? [inst.gpu] : null);
  }
  return {
    port: inst.port,
    gpu: inst.gpu,
    gpus: inst.gpus || (inst.gpu != null ? [inst.gpu] : []),
    model,
    running: concurrency.running,
    queued: concurrency.queued,
    active_requests: concurrency.active_requests,
    waiting_requests: concurrency.waiting_requests || [],
    last_second: ticker.lastSecond || null,
    kv: buildKvCacheInfo(m),
    primary: false,
    runtime: 'vllm',   // [vllm-page-1003] 对称 SGLang 从实例；前端运行时徽标/口径分支的判据
    gen_speed_1s: (ticker.lastSecond && ticker.lastSecond.speed) || 0,
  };
}

// ====== 09-20 性能优化：/metrics 缓存 + 单飞合并 ======
// 前端 stats 卡 500ms×2 轮询，此前每次请求都实时抓上游 vLLM /metrics（约 64KB，
// 序列化发生在 vLLM API 线程上），多标签页叠加会持续干扰推理服务。
// 同一 URL 在 TTL 内复用文本结果；并发请求共享同一个在途抓取（单飞）。
// 抓取失败时若有旧数据则回落旧数据（宁旧勿断供）。
const __metricsCache = new Map(); // url -> { at, text, inflight }
function fetchMetricsCached(url, ttlMs, timeoutMs) {
  ttlMs = ttlMs == null ? 400 : ttlMs;
  timeoutMs = timeoutMs == null ? 3000 : timeoutMs;
  const now = Date.now();
  let e = __metricsCache.get(url);
  if (!e) { e = { at: 0, text: null, inflight: null }; __metricsCache.set(url, e); }
  if (e.text != null && now - e.at < ttlMs) return Promise.resolve(e.text);
  if (e.inflight) return e.inflight;
  e.inflight = new Promise((resolve, reject) => {
    let settled = false;
    const done = (fn, arg) => { if (!settled) { settled = true; fn(arg); } };
    let req;
    try {
      req = http.get(url, { timeout: timeoutMs, headers: { 'User-Agent': 'dsh-console' } }, (proxyRes) => {
        let data = '';
        proxyRes.on('data', (c) => { data += c; if (data.length > 8e6) { try { req.destroy(); } catch (_) {} } });
        proxyRes.on('error', (err) => { e.inflight = null; done(reject, err); });
        proxyRes.on('end', () => {
          e.at = Date.now(); e.text = data; e.inflight = null;
          done(resolve, data);
        });
      });
      req.on('error', (err) => {
        e.inflight = null;
        if (e.text != null) done(resolve, e.text); // 失败回落旧数据
        else done(reject, err);
      });
      req.setTimeout(timeoutMs, () => { try { req.destroy(new Error('metrics timeout')); } catch (_) {} });
    } catch (err) {
      e.inflight = null;
      if (e.text != null) done(resolve, e.text); else done(reject, err);
    }
  });
  return e.inflight;
}

function fetchInstanceConcurrency(inst) {
  // 09-20：走 /metrics 缓存+单飞（400ms TTL），多标签页共享一次上游抓取
  return fetchMetricsCached(`http://${config.vllmHost}:${inst.port}/metrics`, 400, 2500)
    .then((data) => { try { return buildInstanceConcurrency(inst, data); } catch (e) { return null; } })
    .catch(() => null);
}

// ====== 08-30 跨运行时：SGLang 实例行构建（vLLM 主分支下 GPU1 的 sglang 也进并发卡片）======
// 从 SGLang 实例抓 /metrics 并构建并发卡片行（与 vLLM 从实例同形）
// 速度历史按端口隔离（独立数组），不污染主实例的 __sglPpHist/__sglGenThrHist
async function fetchSglangInstanceConcurrency(inst) {
  if (!inst || !inst.port) return null;
  const data = await new Promise((resolve) => {
    const req = http.get({ host: '127.0.0.1', port: inst.port, path: '/metrics', headers: { 'User-Agent': 'dsh-console' }, timeout: 2500 }, res => {
      let body = ''; res.setEncoding('utf8');
      res.on('data', d => { body += d; if (body.length > 4e6) req.destroy(); });
      res.on('end', () => resolve(body));
    });
    req.on('error', () => resolve(null));
    req.setTimeout(2500, () => { try { req.destroy(new Error('sg instance metrics timeout')); } catch (e) {} resolve(null); });
  });
  if (!data) return null;
  let m; try { m = parseMetrics(data); } catch (e) { return null; }
  if (metricsNamespace(m) !== 'sglang') return null;
  const sub = (global.__sglSubHist = global.__sglSubHist || new Map());
  let h = sub.get(inst.port);
  if (!h) { h = { pp: [], gen: [] }; sub.set(inst.port, h); }
  const sg2 = buildSglangStats(m, null);
  const genSpeed1s = buildSglangGenSpeed1s(m, h.gen);
  const prefillSpeed3s = buildSglangPrefillSpeed3s(m, h.pp);
  const ppTotalNow = Math.round(counterByLabel(m, 'sglang:prefill_effective_tokens_total', 'mode', 'input'));
  let modelName = inst.model || '';
  try {
    const mk = Object.keys(m).find(x => x.indexOf('sglang:num_running_reqs|') === 0);
    if (mk) {
      const mm = JSON.parse(mk.substring(mk.indexOf('|') + 1));
      if (mm && mm.model_name) modelName = mm.model_name;
    }
  } catch (e) {}
  // live 流：__liveStreams = {seq, map: Map}（共享，多模型并发时），按流的 model 字段过滤出本实例的
  const liveMap = new Map();
  if (global.__liveStreams && global.__liveStreams.map) {
    for (const [id, e] of global.__liveStreams.map) {
      if (e && !e.done && e.model === modelName) liveMap.set(id, e);
    }
  }
  // 08-30：每请求缓存命中真值 = 本实例日志 "Prefill batch" 行 #cached-token（vLLM 口径对齐）
  const logLines = readSglangPrefillLines(inst.port, inst.pid);
  const rows = buildSglangActiveRequests(sg2, { map: liveMap }, genSpeed1s, prefillSpeed3s, ppTotalNow,
    (e) => sglangCachedForRequest(logLines, e, e.promptTokens));
  return {
    port: inst.port,
    gpu: inst.gpu,
    gpus: inst.gpus || (inst.gpu != null ? [inst.gpu] : []),
    model: modelName,
    running: sg2.running || 0,
    queued: sg2.queued || 0,
    active_requests: rows,
    waiting_requests: [],
    last_second: null,
    kv: sg2.kv_cache || null,
    primary: false,
    runtime: 'sglang',
    gen_speed_1s: (genSpeed1s !== undefined && genSpeed1s >= 0) ? genSpeed1s : 0,
  };
}

// ====== Model Manager ======
const MODELS_DIR = '/media/ll/data/models';
// 额外模型根目录（2026-09-14）：Flash-Next-NVFP4（126G）在数据盘 /media/ll/data/models，
// chroot 内 /media/ll/data 与宿主为同一份挂载，一并参与「模型启动页」列表扫描。
const EXTRA_MODELS_DIRS = [];
const VLLM_DEPLOY_DIR = '/home/ll/vllm-deploy';
const VLLM_START_SH = path.join(VLLM_DEPLOY_DIR, 'start.sh');
const VLLM_ENV = '/home/ll/vllm-env';

function discoverModels(callback) {
  // 多根扫描（2026-09-14）：/home/ll/models 之外，数据盘上的模型
  // （如 Qwen3.8-Flash-Next-NVFP4 126G）也要出现在「模型启动页」列表里。
  const roots = [MODELS_DIR].concat(EXTRA_MODELS_DIRS);
  const results = [];
  let rootsPending = roots.length;
  let done = false;
  const finish = () => {
    if (done || rootsPending > 0) return;
    done = true;
    callback(results);
  };
  roots.forEach(root => {
    fs.readdir(root, (err, dirs) => {
      if (err) { rootsPending--; finish(); return; }
      const candidates = (dirs || []).filter(d => !d.startsWith('.'));
      let pending = candidates.length;
      if (pending === 0) { rootsPending--; finish(); return; }
      candidates.forEach(d => {
        const dirPath = path.join(root, d);
        const step = () => { if (--pending === 0) { rootsPending--; finish(); } };
        fs.stat(dirPath, (err1, stat) => {
          if (err1 || !stat.isDirectory()) return step();
          // Check for config.json to confirm it's a valid model
          fs.stat(path.join(dirPath, 'config.json'), (err2) => {
            if (err2) return step();
            if (!results.some(r => r.name === d)) {
              const totalSize = estimateDirSize(dirPath);
              results.push({
                name: d,
                path: dirPath,
                size_gb: totalSize > 0 ? (totalSize / 1024 / 1024 / 1024).toFixed(1) : '--',
                size_bytes: totalSize,
              });
            }
            step();
          });
        });
      });
    });
  });
}

function estimateDirSize(dirPath) {
  let total = 0;
  try {
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });
    const walk = (dir) => {
      try {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const e of entries) {
          const full = path.join(dir, e.name);
          if (e.isFile()) {
            try { total += fs.statSync(full).size; } catch (e) {}
          } else if (e.isDirectory()) {
            walk(full);
          }
        }
      } catch (e) {}
    };
    walk(dirPath);
  } catch (e) {}
  return total;
}

function isVllmRunning(callback) {
  const { execSync } = require('child_process');
  try {
    // Try to get the actual vllm process via ps first
    const output = execSync('ps aux | grep "[v]llm serve" || true', { encoding: 'utf8', timeout: 5000 });
    const model = getCurrentVllmModel();
    // Get PID from ps output
    let pid = null;
    const lines = output.trim().split('\n');
    for (const line of lines) {
      if (line) {
        const parts = line.trim().split(/\s+/);
        if (parts.length >= 2) {
          pid = parseInt(parts[1]);
          break;
        }
      }
    }
    if (pid && !isNaN(pid)) {
      callback({ running: true, pid: pid, model: model });
    } else {
      // Fallback to lsof
      const lsofPid = execSync('lsof -ti:8000 2>/dev/null || true', { encoding: 'utf8', timeout: 5000 }).trim();
      if (lsofPid) {
        callback({ running: true, pid: parseInt(lsofPid.split('\n')[0]), model: model });
      } else {
        callback({ running: false, pid: null, model: null });
      }
    }
  } catch (e) {
    callback({ running: false, pid: null, model: null });
  }
}

function getCurrentVllmModel() {
  // 5s 缓存：原实现每次调用都 execSync('ps aux | grep ...') spawn 子进程，
  // stats 端点每 0.5s 轮询一次 → 每分钟 120 次子进程 spawn，是 node CPU 57% 的主因之一
  const now = Date.now();
  if (__curModelCache.at && now - __curModelCache.at < 5000) return __curModelCache.val;
  let val = null;
  try {
    const vllmInsts = listVllmInstances();
    if (vllmInsts.length) {
      val = vllmInsts[0].modelPath || vllmInsts[0].servedName || null;
    }
    if (!val) {
      const sgInsts = listSglangInstances();
      if (sgInsts.length) {
        val = sgInsts[0].modelPath || sgInsts[0].servedName || null;
      }
    }
  } catch (e) {}
  __curModelCache = { at: now, val };
  return val;
}
let __curModelCache = { at: 0, val: null };

// ====== Billing (实时费用计算) ======
const BILLING_CONFIG_PATH = path.join(__dirname, 'billing-config.json');
const BILLING_PRICE_FIELDS = ['cached_input_price_per_1m', 'uncached_input_price_per_1m', 'output_price_per_1m'];

// 09-06 性能修复：billing-config.json 原来每次 computeBilling/accumulateBilling 都 readFileSync，
// stats 端点每 0.5s 轮询 → 每秒 2-4 次同步读。改为 10s 缓存（配置很少变）。
let __billingCfgCache = { at: 0, val: null };
function loadBillingConfig() {
  const now = Date.now();
  if (__billingCfgCache.val && now - __billingCfgCache.at < 10000) return __billingCfgCache.val;
  let cfg = { models: {}, currency: 'CNY', unit: '元' };
  try {
    const raw = JSON.parse(fs.readFileSync(BILLING_CONFIG_PATH, 'utf8'));
    if (raw && typeof raw === 'object') {
      if (!raw.models || typeof raw.models !== 'object') raw.models = {};
      cfg = raw;
    }
  } catch (e) {}
  __billingCfgCache = { at: now, val: cfg };
  return cfg;
}

function saveBillingConfig(cfg) {
  fs.writeFileSync(BILLING_CONFIG_PATH, JSON.stringify(cfg, null, 2));
  __billingCfgCache = { at: 0, val: null }; // 失效缓存
}

// Match running model path (e.g. /home/ll/models/qwen3.6-35b-a3b-fp8) to a
// billing config key (e.g. qwen3.6-35b-a3b-fp8). Fuzzy: basename first, then
// substring, then single-entry fallback.
function matchBillingModel(modelPath) {
  const cfg = loadBillingConfig();
  const keys = Object.keys(cfg.models || {});
  if (!keys.length || !modelPath) return null;
  const base = String(modelPath).replace(/\/+$/, '').split('/').pop();
  let hit = keys.find(k => k === base);
  if (!hit) hit = keys.find(k => base.indexOf(k) !== -1 || k.indexOf(base) !== -1);
  if (!hit) {
    const num = base.replace(/[^0-9.]/g, '');
    if (num) hit = keys.find(k => {
      const kn = k.replace(/[^0-9.]/g, '');
      return kn && (kn.indexOf(num) !== -1 || num.indexOf(kn) !== -1);
    });
  }
  if (!hit && keys.length === 1) hit = keys[0];
  return hit || null;
}

// Compute real-time cost for the token counters present in `result`
// (cached_input_tokens / uncached_input_tokens / total_output_tokens).
function computeBilling(result, modelPath) {
  try {
    const cfg = loadBillingConfig();
    const key = matchBillingModel(modelPath);
    const entry = key ? cfg.models[key] : null;
    const pricing = entry && entry.pricing ? entry.pricing : {};
    const pCached = parseFloat(pricing.cached_input_price_per_1m) || 0;
    const pInput = parseFloat(pricing.uncached_input_price_per_1m) || 0;
    const pOutput = parseFloat(pricing.output_price_per_1m) || 0;
    const cached = result.cached_input_tokens || 0;
    const uncached = result.uncached_input_tokens || 0;
    const output = result.total_output_tokens || 0;
    const cCost = (cached / 1e6) * pCached;
    const iCost = (uncached / 1e6) * pInput;
    const oCost = (output / 1e6) * pOutput;
    return {
      enabled: true,
      configured: !!entry,
      model: key || null,
      model_path: modelPath || null,
      currency: cfg.currency || 'CNY',
      unit: cfg.unit || '元',
      price_per_1m: {
        cached_input: pCached,
        uncached_input: pInput,
        output: pOutput
      },
      tokens: {
        cached_input: cached,
        uncached_input: uncached,
        output: output
      },
      cost: {
        cached_input: cCost,
        uncached_input: iCost,
        output: oCost,
        total: cCost + iCost + oCost
      },
      cost_display: {
        cached_input: cCost < 0.0001 ? cCost.toFixed(6) : cCost.toFixed(4),
        uncached_input: iCost < 0.0001 ? iCost.toFixed(6) : iCost.toFixed(4),
        output: oCost < 0.0001 ? oCost.toFixed(6) : oCost.toFixed(4),
        total: (cCost + iCost + oCost).toFixed(4)
      }
    };
  } catch (e) {
    return { enabled: false, error: e.message };
  }
}

// ====== Billing state persistence ======
// 新费用(current)/历史总费用(history) 两个桶持久化到 billing-state.json：
//  - 重启不丢（落盘）
//  - 与「重置统计」完全无关（独立状态文件，基于 vLLM 原始累计计数器算差值）
const BILLING_STATE_PATH = path.join(__dirname, 'billing-state.json');
const BILLING_ZERO_BUCKET = () => ({ cost: { cached_input: 0, uncached_input: 0, output: 0 }, tokens: { cached_input: 0, uncached_input: 0, output: 0 } });

// ====== 每日费用明细（持久化，可手动删除） ======
const BILLING_MAX_DAYS = 120; // 最多保留最近 120 个有费用的日期
const BILLING_DAY_FIELDS = ['cached_input', 'uncached_input', 'output'];

// 服务器本地日期 YYYY-MM-DD（与「每天」的直觉一致）
function billingDayKey(d) {
  const t = d || new Date();
  const p = n => String(n).padStart(2, '0');
  return t.getFullYear() + '-' + p(t.getMonth() + 1) + '-' + p(t.getDate());
}

function ensureBillingDay(st, date) {
  if (!st.days || typeof st.days !== 'object') st.days = {};
  if (!st.days[date] || typeof st.days[date].cost !== 'object') st.days[date] = BILLING_ZERO_BUCKET();
  return st.days[date];
}

// 输出用：附加 total，按日期倒序
function buildBillingDays(st) {
  const days = {};
  for (const [date, bk] of Object.entries((st && st.days) || {})) {
    const c = bk.cost || {}, tk = bk.tokens || {};
    days[date] = {
      cost: {
        cached_input: c.cached_input || 0,
        uncached_input: c.uncached_input || 0,
        output: c.output || 0,
        total: (c.cached_input || 0) + (c.uncached_input || 0) + (c.output || 0)
      },
      tokens: {
        cached_input: tk.cached_input || 0,
        uncached_input: tk.uncached_input || 0,
        output: tk.output || 0
      }
    };
  }
  return days;
}

// 09-06 性能修复：billing-state.json 原来每次 stats 请求都 readFileSync+writeFileSync
// （accumulateBilling 内 load+save），前端 0.5s 轮询 → 每秒 2 次同步磁盘读写阻塞事件循环。
// 改为内存缓存 + 5s 节流落盘。
let __billingStateMem = null;
function loadBillingState() {
  if (__billingStateMem) return __billingStateMem;
  try {
    const st = JSON.parse(fs.readFileSync(BILLING_STATE_PATH, 'utf8'));
    if (st && typeof st === 'object') {
      if (!st.current || typeof st.current.cost !== 'object') st.current = BILLING_ZERO_BUCKET();
      if (!st.history || typeof st.history.cost !== 'object') st.history = BILLING_ZERO_BUCKET();
      if (!st.lastRaw || typeof st.lastRaw !== 'object') st.lastRaw = {};
      if (!st.days || typeof st.days !== 'object') st.days = {};
      __billingStateMem = st;
      return st;
    }
  } catch (e) {}
  const st = { current: BILLING_ZERO_BUCKET(), history: BILLING_ZERO_BUCKET(), lastRaw: {} };
  st.days = {};
  __billingStateMem = st;
  return st;
}

function saveBillingState(st) {
  __billingStateMem = st;
  const now = Date.now();
  if (now - (saveBillingState._lastSave || 0) < 5000) return; // 5s 节流
  saveBillingState._lastSave = now;
  try { fs.writeFileSync(BILLING_STATE_PATH, JSON.stringify(st, null, 2)); } catch (e) {}
}

// 用 vLLM RAW 累计计数器（未减重置基线）算差值，按当前模型单价把增量费用
// 同时计入 current 与 history 两桶。计数器回退（vLLM 重启/换模型）时重新
// 定基线，不计负值。
function accumulateBilling(raw) {
  const st = loadBillingState();
  const delta = { cached_input: 0, uncached_input: 0, output: 0 };
  const pairs = [['cached_input', 'cached'], ['uncached_input', 'uncached'], ['output', 'gen']];
  for (const [field, rk] of pairs) {
    const r = Math.round(raw[rk] || 0);
    const prev = st.lastRaw[rk];
    if (prev === undefined || r < prev) { st.lastRaw[rk] = r; continue; } // 首次/回退：定基线
    if (r > prev) { delta[field] = r - prev; st.lastRaw[rk] = r; }
  }
  if (delta.cached_input + delta.uncached_input + delta.output > 0) {
    const cfg = loadBillingConfig();
    const key = matchBillingModel(getCurrentVllmModel());
    const entry = key ? cfg.models[key] : null;
    const pr = (entry && entry.pricing) || {};
    const p = {
      cached_input: parseFloat(pr.cached_input_price_per_1m) || 0,
      uncached_input: parseFloat(pr.uncached_input_price_per_1m) || 0,
      output: parseFloat(pr.output_price_per_1m) || 0,
    };
    for (const bucket of [st.current, st.history]) {
      bucket.cost.cached_input = (bucket.cost.cached_input || 0) + delta.cached_input / 1e6 * p.cached_input;
      bucket.cost.uncached_input = (bucket.cost.uncached_input || 0) + delta.uncached_input / 1e6 * p.uncached_input;
      bucket.cost.output = (bucket.cost.output || 0) + delta.output / 1e6 * p.output;
      bucket.tokens.cached_input = (bucket.tokens.cached_input || 0) + delta.cached_input;
      bucket.tokens.uncached_input = (bucket.tokens.uncached_input || 0) + delta.uncached_input;
      bucket.tokens.output = (bucket.tokens.output || 0) + delta.output;
    }
    // 每日明细：按服务器本地日期归桶（单价为入账时点的当前单价）
    const day = ensureBillingDay(st, billingDayKey());
    for (const f of BILLING_DAY_FIELDS) {
      day.cost[f] = (day.cost[f] || 0) + delta[f] / 1e6 * p[f];
      day.tokens[f] = (day.tokens[f] || 0) + delta[f];
    }
    // 只保留最近 BILLING_MAX_DAYS 个有费用的日期
    const dayKeys = Object.keys(st.days).sort();
    while (dayKeys.length > BILLING_MAX_DAYS) delete st.days[dayKeys.shift()];
  }
  saveBillingState(st);
  return st;
}

function resetBillingBucket(scope) {
  const st = loadBillingState();
  if (scope === 'current' || scope === 'history') st[scope] = BILLING_ZERO_BUCKET();
  saveBillingState._lastSave = 0; // 强制立即落盘（重置操作不能等 5s 节流）
  saveBillingState(st);
  return st;
}

function buildModelLabelFromMetrics(m, metricPrefix) {
  // parseMetrics encodes labels as |{json}, so key looks like:
  //   vllm:generation_tokens_total|{"engine":"0","model_name":"foo"}
  // metricPrefix is like 'vllm:generation_tokens_total{'
  // We need to find keys where the name part matches, then return the |{json} suffix.
  for (const k of Object.keys(m)) {
    // Key format: name|{json_labels}
    // metricPrefix ends with '{', so we need name part == metricPrefix without trailing '{'
    const nameWithoutBrace = metricPrefix.slice(0, -1); // e.g. 'vllm:generation_tokens_total'
    const pipeIdx = k.indexOf('|');
    if (pipeIdx > 0 && k.substring(0, pipeIdx) === nameWithoutBrace) {
      return k.substring(pipeIdx); // e.g. '|{"engine":"0","model_name":"foo"}'
    }
  }
  return '';
}

// 判断某 PID 是否仍是存活的 vLLM 相关进程（APIServer cmdline 含 "vllm serve"，
// EngineCore 的进程标题为 "VLLM::EngineCore"）。
// 关键修复：vLLM 0.27 的 EngineCore 在父 APIServer 被杀后会被重新挂到仍然存活的
// systemd --user（subreaper）名下（PPID=1293 之类），PPID 不再是 1、父进程也"活着"，
// 旧逻辑 `ppid==='1' || !parentAlive` 永远判不出它是孤儿 → EngineCore 不退出，
// GPU 显存一直不释放（CUDA out of memory）。正确判据：父进程是否仍是 vLLM 进程。
function parentIsVllmProcess(ppid) {
  if (!ppid || ppid === '1') return false;
  try {
    const cmd = fs.readFileSync(`/proc/${ppid}/cmdline`, 'utf8').split('\0').join(' ');
    // 识别全部 vLLM 主进程形式：CLI（vllm serve）、模块（vllm.entrypoints.*，含
    // 0.28 的 cli.main）与 EngineCore 自身；漏判会让活着的 EngineCore 被判孤儿后误杀
    // （09-14 chroot 内 Flash-Next 即 cli.main 形式）。
    return cmd.includes('vllm serve') || cmd.includes('vllm.entrypoints') || cmd.includes('EngineCore');
  } catch (e) { return false; }
}

// 收集「父进程已不是 vLLM 进程」的残留进程 PID：
//   - VLLM::EngineCore（持显存的引擎核心，父 APIServer 死后变孤儿）
//   - multiprocessing.resource_tracker（vLLM mp 的共享内存跟踪器，同样会孤儿化）
// 双实例模式：其他存活实例的 EngineCore/resource_tracker 其父进程仍是 vLLM，不会被误杀。
function collectOrphanVllmPids() {
  const { execSync } = require('child_process');
  const orphans = [];
  const scan = (pattern) => {
    try {
      const lines = execSync(pattern, { encoding: 'utf8', timeout: 5000 }).trim();
      if (!lines) return;
      lines.split('\n').forEach(line => {
        const parts = line.trim().split(/\s+/);
        if (parts.length < 2) return;
        const pid = parts[0];
        const ppid = parts[1];
        if (!parentIsVllmProcess(ppid)) orphans.push(parseInt(pid));
      });
    } catch (e) {}
  };
  scan("ps -eo pid,ppid,args | grep '[V]LLM::EngineCore' 2>/dev/null || true");
  scan("ps -eo pid,ppid,args | grep '[m]ultiprocessing.resource_tracker' 2>/dev/null || true");
  return orphans;
}

// 通用：杀掉指定 PID 列表并等待 GPU 显存释放（最多 ~30s）。
// 判据：总显存占用相比清理前减少 >= 被清理进程的显存（1024 MiB 容差）。
// 双卡场景：其他存活实例占用的显存不在差额里，不会被误判为"没释放"。
async function waitGpuMemRelease(pids) {
  if (!pids || pids.length === 0) return;
  // 10-07 测试修复：原实现 4 处 execSync(nvidia-smi) 违反 09-20 铁律——驱动进 D 状态时
  // timeout 无效，停止/清理流程会把整个控制台（含所有页面请求）一起永久按住。
  // 全部改走放弃式异步通道；显存不可探测时不再空转 30s 去"验证释放"。
  const appsOut = await gpuQuery(['--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], GPU_Q_MS_LIVE);
  let freedTargetMiB = 0;
  if (appsOut) String(appsOut).split('\n').forEach(l => {
    const [pid, mem] = l.split(',').map(s => s.trim());
    if (pid && pids.includes(parseInt(pid))) freedTargetMiB += parseInt(mem) || 0;
  });
  const readUsedMiB = async () => {
    const out = await gpuQuery(['--query-gpu=memory.used', '--format=csv,noheader,nounits'], GPU_Q_MS_LIVE);
    if (out == null) return null;
    let sum = 0;
    String(out).split('\n').forEach(l => { const v = parseInt(l.trim()); if (!isNaN(v)) sum += v; });
    return sum;
  };
  const usedBeforeMiB = await readUsedMiB();

  for (const pid of pids) {
    // 不再 execSync('kill -9 ' + pid)：省掉一次同步子进程，同时消除把 pid 拼进 shell 的注入面
    try { process.kill(parseInt(pid, 10), 'SIGKILL'); } catch (e2) {}
  }

  if (usedBeforeMiB == null) return; // 显存不可探测（驱动异常/熔断冷却）：无法验证，直接返回不空等
  const targetMiB = usedBeforeMiB - freedTargetMiB + 1024; // 1024 MiB 容差
  for (let i = 0; i < 15; i++) {
    const usedNowMiB = await readUsedMiB();
    if (usedNowMiB == null) return; // 中途探测失效同样不再空等
    if (usedNowMiB <= targetMiB) return;
    await sleep(2000);
  }
}

async function killOrphanEngineCores() {
  // vLLM 0.27 的 EngineCore 是独立子进程：主进程(APIServer)被杀后它不会退出，
  // 变成孤儿继续占满 GPU 显存，导致下次启动直接显存不足失败（CUDA out of memory）。
  // 停服务时必须一并清理并等待显存释放。
  const orphans = collectOrphanVllmPids();
  if (orphans.length === 0) return;
  await waitGpuMemRelease(orphans);
}

// 读 /proc/<pid>/cmdline（空格连接），失败返回 ''
function procCmdline(pid) {
  try { return fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8').split('\0').join(' '); } catch (e) { return ''; }
}
// 读 /proc/<pid>/stat 的 ppid（field 4，comm 括号之后第一个字段），失败返回 0
function procPpid(pid) {
  try {
    const s = fs.readFileSync(`/proc/${pid}/stat`, 'utf8');
    const rest = s.slice(s.lastIndexOf(')') + 2).split(' ');
    return parseInt(rest[1]) || 0;
  } catch (e) { return 0; }
}
// 沿进程树向上找实例端口：自身或任一祖先进程 cmdline 里第一个 --port N（锚定空白边界，
// 避免误匹配 --tokenizer-port 之类的参数）。找不到返回 null。
function instancePortOf(pid) {
  let cur = parseInt(pid, 10);
  for (let i = 0; i < 16 && cur > 1; i++) {
    const m = procCmdline(cur).match(/(?:^|\s)--port (\d+)/);
    if (m) return parseInt(m[1], 10);
    const pp = procPpid(cur);
    if (!pp || pp === cur) break;
    cur = pp;
  }
  return null;
}
// 收集 sglang 实例的残留进程（主进程 + scheduler/tokenizer/detokenizer 等工作进程）。
// 端口归属一律按进程树的 --port 判定：worker 进程 cmdline 形如 `sglang::scheduler`
// （setproctitle，无 --port），旧逻辑按 'scheduler'/'tokenizer' 子串绕过端口过滤，
// 导致启动/停止任一端口的实例时把所有 sglang 实例（含另一 GPU 的）的 worker 全部 kill -9，
// 对方实例 scheduler 被杀后整体自杀 —— 即"启动一个 GPU 的 sglang，另一个 GPU 的 sglang 被结束"。
function collectSglangPids(port) {
  const { execSync } = require('child_process');
  const pids = [];
  try {
    const raw = execSync(`pgrep -f "sglang" 2>/dev/null || true`, { encoding: 'utf8', timeout: 5000 }).trim().split('\n').filter(Boolean);
    for (const pid of raw) {
      try {
        const cmd = procCmdline(pid);
        const isSglangProc = cmd.includes('sglang.launch_server') || cmd.includes('sglang.srt') ||
          cmd.includes('sglang/') || cmd.includes('scheduler') || cmd.includes('tokenizer') || cmd.includes('detokenizer');
        if (!isSglangProc) continue;
        // 不杀控制台自身与 ssh/脚本壳
        if (cmd.includes('server.js') || cmd.includes('start_sglang')) continue;
        // 端口归属：自身或祖先主进程的 --port 必须等于目标端口，否则跳过（保护其他实例）
        if (port && instancePortOf(pid) !== port) continue;
        pids.push(parseInt(pid));
      } catch (e) {}
    }
  } catch (e) {}
  return pids;
}

// 清理目标端口上的残留实例（vllm 或 sglang：按 cmdline --port 匹配 + lsof 监听兜底），
// 再清孤儿 EngineCore / sglang 残留，等待显存释放。启动前调用，避免 CUDA out of memory。
async function killPortResidents(port) {
  const { execSync } = require('child_process');
  const toKill = [];
  try {
    const pids = execSync(`pgrep -f "vllm serve" 2>/dev/null; pgrep -f "sglang.launch_server" 2>/dev/null || true`, { encoding: 'utf8', timeout: 5000 }).trim().split('\n').filter(Boolean);
    for (const pid of pids) {
      try {
        const cmd = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8').split('\0').join(' ');
        if (cmd.includes(`--port ${port}`)) toKill.push(parseInt(pid));
      } catch (e) {}
    }
    const lp = execSync(`lsof -ti:${port} -sTCP:LISTEN 2>/dev/null || true`, { encoding: 'utf8', timeout: 3000 }).trim();
    if (lp) lp.split('\n').forEach(l => { const n = parseInt(l.trim()); if (!isNaN(n) && n !== process.pid) toKill.push(n); });
  } catch (e) {}
  if (toKill.length) await waitGpuMemRelease(Array.from(new Set(toKill)));
  await killOrphanEngineCores();
  const sg = collectSglangPids(port);
  if (sg.length) await waitGpuMemRelease(sg);
}

async function startVllmModel(modelName, params, callback) {
  const { execSync } = require('child_process');
  const { spawn } = require('child_process');
  const { port, maxModelLen, gpuId, gpuCount, parallelMode, pdMode, mtp, dflash, dspark, servedName, maxNumSeqs, gpuMemUtil, thinking, thinkingEffort, mtpTokens, temperature, topP, topK, minP, presencePenalty, repetitionPenalty, kvCacheQuant, kvoff, kvoffGiB, retention,
    maxBatchedTokens, maxScheduledTokens, schedPolicy, asyncScheduling, enforceEager, blockSize, cpuOffloadGb, prefixCaching, chunkedPrefill, seed, dtype, noLogRequests, limitMm, maxLoraRank, disableAllReduce, attentionBackend, ctxLen, languageModelOnly } = params;
  // reasoning_effort：允许 low/medium/high/xhigh 及手动输入（字母/数字/下划线/连字符，最长 32）
  // 例如 qwen3.8-27b 支持 xhigh；非法/空值回落 medium
  const rawEffort = String(thinkingEffort || '').trim();
  // [gen-default 0927] 空值=未填 → 用启动页默认 xhigh；填了但非法 → 保守回落 medium（不静默变最深档）
  const effort = rawEffort === '' ? 'xhigh' : (/^[a-zA-Z0-9_-]{1,32}$/.test(rawEffort) ? rawEffort : 'medium');
  // MTP speculative steps: clamp to 1-8, default 3
  const mtpN = Math.min(8, Math.max(1, parseInt(mtpTokens) || 5));
  // 启用显卡数量：从 gpuId 起连续取 N 张卡，N>1 时启用多卡张量并行
  const gpuN = Math.max(1, parseInt(gpuCount) || 1);
  const gpuDevices = Array.from({ length: gpuN }, (_, i) => gpuId + i).join(',');

  // 校验可用显卡数：请求卡数超过物理显卡时直接报错，避免杀掉现有服务后才发现启动失败
  // 10-07 测试修复：① 原 execSync(nvidia-smi) 违反 09-20 铁律；② 原兜底"探不到当 1 张卡"
  // 属"探测失败被当成真值"——驱动抖动时，三卡机上请求 2 卡会被误拒
  // （"超出当前可用显卡数(1)"）。现改为探不到就跳过校验，让引擎自己如实报错。
  let availableGpus = null;
  try {
    const gpuCountOut = await gpuQuery(['--query-gpu=index', '--format=csv,noheader'], GPU_Q_MS_LIVE);
    if (gpuCountOut != null) {
      const parsed = String(gpuCountOut).split('\n').filter((l) => l.trim().length).length;
      if (parsed > 0) availableGpus = parsed;
    }
  } catch (e) {}
  if (availableGpus != null && (gpuId < 0 || gpuId + gpuN > availableGpus)) {
    callback({ success: false, error: `启用显卡数量(${gpuN}，起始卡 ${gpuId}) 超出当前可用显卡数(${availableGpus})，请调整后重试` });
    return;
  }

  // 采样参数（SamplingParams）：NaN 回落默认值，越界钳制到合法范围
  const clampNum = (v, dflt, lo, hi) => {
    const n = parseFloat(v);
    return isNaN(n) ? dflt : Math.min(hi, Math.max(lo, n));
  };
  const samplingParams = {
    temperature: clampNum(temperature, 1.0, 0, 2),
    top_p: clampNum(topP, 0.95, 0.01, 1),
    top_k: Math.round(clampNum(topK, 20, -1, 100)),
    min_p: clampNum(minP, 0.0, 0, 1),
    presence_penalty: clampNum(presencePenalty, 0.0, -2, 2),
    repetition_penalty: clampNum(repetitionPenalty, 1.0, 0, 2),
  };
  const modelPath = path.join(MODELS_DIR, modelName);

  // 推测解码方式：DFlash2（block-diffusion 草稿模型，需 nightly venv）/ DSpark（DFlash 骨干 + Markov 头，
  // vllm-env 0.27.1 原生支持）/ MTP（内置多 token 预测）
  const isDflash = dflash === '1';
  const isDspark = dspark === '1';
  const dflashN = Math.min(8, Math.max(1, parseInt(mtpTokens) || 7));
  // DSpark：投机 token 数固定为草稿 ckpt 的 dspark_block_size（UI 输入不再当 gamma）
  const dsparkGamma = isDspark ? (readDsparkGamma(SGLANG_DSPARK_PATH) || 7) : 7;
  if (isDspark && parseInt(mtpTokens) && parseInt(mtpTokens) !== dsparkGamma) {
    console.log(`[vllm-start] DSpark 投机 token 数固定为草稿 ckpt 的 dspark_block_size=${dsparkGamma}，忽略 UI 输入 ${mtpTokens}`);
  }
  const dsparkN = dsparkGamma;
  const VENV = '/home/ll/vllm-env'; // 2026-09-05: vllm-env 现为 syv 0.28.0 环境（符号链接），原生支持 DFlash2；nightly 环境已清理
  if (isDflash) {
    if (!fs.existsSync(path.join(VENV, 'bin', 'vllm'))) {
      callback({ success: false, error: 'DFlash2 环境未就绪：未找到 /home/ll/vllm-env/bin/vllm（vllm-env 现为 0.28.0 syv 环境，原生支持 DFlash2）' });
      return;
    }
    if (!fs.existsSync('/home/ll/models/qwen3.8-27b-dflash2/config.json')) {
      callback({ success: false, error: 'DFlash2 草稿模型未下载：/home/ll/models/qwen3.8-27b-dflash2/config.json（需先下载 incoai/Qwen3.8-27B-DFlash2）' });
      return;
    }
    if (!String(modelName).includes('qwen3.8-27b')) {
      callback({ success: false, error: 'DFlash2 草稿模型目前仅支持 qwen3.8-27b 系列目标模型' });
      return;
    }
  }
  if (isDspark) {
    if (!fs.existsSync('/home/ll/models/qwen3.8-27b-dspark/config.json')) {
      callback({ success: false, error: 'DSpark 草稿模型未下载：/home/ll/models/qwen3.8-27b-dspark/config.json（需先下载 incoai/dspark-qwen3.8-27b）' });
      return;
    }
    if (!String(modelName).includes('qwen3.8-27b')) {
      callback({ success: false, error: 'DSpark 草稿模型目前仅支持 qwen3.8-27b 系列目标模型' });
      return;
    }
  }

  // 清理目标端口上的残留实例（vllm / sglang 均可）并等待 GPU 显存释放，
  // 避免新实例 CUDA out of memory。按 cmdline --port 匹配（加载中未监听也能杀）
  // + lsof 监听兜底，只动目标端口，不碰其他端口的实例。
  await killPortResidents(port);

  // Build command args
  // maxModelLen 来自前端输入，只允许 'auto' 或纯数字，防止把非法内容拼进启动参数
  const safeModelLen = /^auto$|^\d+$/.test(String(maxModelLen || '')) ? String(maxModelLen) : 'auto';
  // 上下文长度文本框（ctxLen）：非空且为纯数字时覆盖 safeModelLen；超过 262144 时自动加 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  const rawCtxLen = String(ctxLen || '').trim();
  let needAllowLong = false;
  if (/^\d+$/.test(rawCtxLen)) {
    const ctxNum = parseInt(rawCtxLen, 10);
    if (ctxNum > 0) {
      safeModelLen = rawCtxLen;
    }
  }
  // 2026-09-10 修复：超长上下文放行 env 不能只认 ctxLen 框。
  // 用户在「最大上下文长度」(maxModelLen) 或快速启动预设里填 >262144 时同样要注入，
  // 否则 vLLM 0.28.0 在 ModelConfig 校验阶段直接抛 ValueError 拒启（实测 09-10 08:35 / 08:48 两次）。
  if (/^\d+$/.test(safeModelLen) && parseInt(safeModelLen, 10) > 262144) needAllowLong = true;
  // 注意力后端：仅允许字母/数字/下划线（vLLM 后端名如 FLASHINFER / TRITON / FLASH_ATTN / MLA / FLASHMLA），
  // 非法/空值回落 FLASHINFER（与旧版硬编码一致）
  const rawAttn = String(attentionBackend || '').trim();
  const safeAttn = /^[A-Za-z0-9_]{1,32}$/.test(rawAttn) ? rawAttn : 'FLASHINFER';
  // 2026-09-05 w8a16+fp8 修复: vLLM 0.28.0 的 FLASH_ATTN 后端不支持 FP8 KV cache
  // (FP8 KV 仅 FA3/SM90 或 FA4/SM100 可用); 本机 CMP170HX=SM80, fp8 KV 只能走 FlashInfer。
  // fp8 类 KV 量化与 FLASH_ATTN 同时出现时强制改回 FLASHINFER, 防一键启动再次失败。
  if (safeAttn === 'FLASH_ATTN' && /^fp8/.test(String(kvCacheQuant || '').toLowerCase())) safeAttn = 'FLASHINFER';
  const args = [
    'serve', modelPath,
    '--port', port.toString(),
    '--max-model-len', safeModelLen,
    '--reasoning-parser', 'qwen3',
    '--enable-auto-tool-choice',
    '--tool-call-parser', 'qwen3_coder',
    '--trust-remote-code',
    '--served-model-name', servedName,
    '--enable-prompt-tokens-details',
    '--max-num-seqs', maxNumSeqs.toString(),
    '--gpu-memory-utilization', gpuMemUtil,
    '--attention-backend', safeAttn,
    '--default-chat-template-kwargs', thinking === '1'
      ? `{"enable_thinking": true, "reasoning_effort": "${effort}"}`
      : '{"enable_thinking": false}',
  ];

  // ---- 高级参数（可选；空/未勾选则不追加，保持命令干净）----
  // 正整数校验：非法/<=0 回落默认值（默认值与旧版硬编码一致，行为不变）
  const intParam = (v, dflt) => { const n = parseInt(v); return (isNaN(n) || n <= 0) ? dflt : n; };
  const pushArg = (flag, val) => { const s = String(val == null ? '' : val).trim(); if (s !== '') args.push(flag, s); };
  const pushFlag = (flag, on) => { if (on === '1' || on === true) args.push(flag); };
  // 批处理与调度
  pushArg('--max-num-batched-tokens', intParam(maxBatchedTokens, 8192));
  pushArg('--max-num-scheduled-tokens', maxScheduledTokens);
  pushArg('--scheduling-policy', schedPolicy);
  pushFlag('--async-scheduling', asyncScheduling);
  pushFlag('--enforce-eager', enforceEager);
  // 缓存与显存
  pushArg('--block-size', intParam(blockSize, 32));
  pushArg('--cpu-offload-gb', cpuOffloadGb);
  // 前缀缓存 / 分块预填充：默认开启（与旧版一致），显式关闭才传 --no-enable-*
  if (prefixCaching === '0') args.push('--no-enable-prefix-caching'); else args.push('--enable-prefix-caching');
  if (chunkedPrefill === '0') args.push('--no-enable-chunked-prefill'); else args.push('--enable-chunked-prefill');
  // 复现性与精度
  pushArg('--seed', seed);
  if (dtype && dtype !== 'auto') args.push('--dtype', dtype);
  // 日志
  pushFlag('--no-enable-log-requests', noLogRequests);
  // 多模态与 LoRA
  // 2026-09-08 修复: vLLM 0.28.0 要求 --limit-mm-per-prompt 为 dict 格式 {image:{count:N}}
  // (裸 int 触发 pydantic ValidationError 启动即退); 留空=不传该 flag, 0=禁止图片
  const limitMmN = parseInt(limitMm);
  if (limitMm !== "" && limitMm !== undefined && limitMm !== null && !isNaN(limitMmN)) args.push("--limit-mm-per-prompt", JSON.stringify({image: {count: limitMmN}}));
  pushArg('--max-lora-rank', maxLoraRank);
  pushFlag('--disable-custom-all-reduce', disableAllReduce);
  // 2026-09-16: twin-709 等 Qwen3_5ForConditionalGeneration（VL wrapper）模型需跳过视觉塔加载，
  // 否则控制台弹窗启动会去解析 image processor 失败。params.languageModelOnly==='1' 时透传。
  pushFlag('--language-model-only', languageModelOnly);

  // 草稿模型选择（2026-09-05）：前端 vllmDraftModel 非空则覆盖内置草稿路径
  // 2026-09-05 修复：短名（如 qwen3.8-27b-dflash2-w4a16）会被 vLLM 的 HF 解析器
  // 误判为 repo 名去远程解析，导致引擎初始化卡死/失败——短名自动展开为
  // /home/ll/models/<短名> 完整本地路径（与弹窗草稿下拉 DRAFT_OPTIONS 的 name 对应）。
  const draftSelRaw = String(params.vllmDraftModel || '').trim();
  const resolveDraft = (sel) => {
    if (!sel) return '';
    if (sel.startsWith('/') || sel.startsWith('hf://') || sel.includes('/')) return sel;
    return path.join(MODELS_DIR, sel);
  };
  const draftSel = resolveDraft(draftSelRaw);
  const pickDraft = (builtin) => (draftSel && draftSel !== '' ? draftSel : builtin);
  if (isDflash) {
    const _dp = pickDraft('/home/ll/models/qwen3.8-27b-dflash2-w4a16');
    // 2026-09-08 修复: 草稿模型是 DSpark 时 method 必须用 dspark
    // (method=dflash + Qwen3DSparkModel → EAGLEConfig 前缀成 DFlashQwen3DSparkModel → registry 不认)
    if (/dspark/i.test(_dp)) {
      args.push('--speculative-config', `{"method":"dspark","model":"${_dp}","num_speculative_tokens":${dsparkN}}`);
    } else {
      args.push('--speculative-config', `{"method":"dflash","model":"${_dp}","num_speculative_tokens":${dflashN},"draft_sample_method":"probabilistic"}`);
    }
    // syv 最优配置 JSON 参数硬编码（带正确引号，避开前端 shellTokenize 剥引号坑）
    args.push('--compilation-config', JSON.stringify({max_cudagraph_capture_size: 64, custom_ops: ['+rms_norm', '+silu_and_mul']}));
    args.push("--limit-mm-per-prompt", limitMm ? JSON.stringify({image: {count: parseInt(limitMm) || 1}}) : JSON.stringify({image: {count: 1}}));
    args.push('--mm-processor-kwargs', JSON.stringify({size: {shortest_edge: 65536, longest_edge: 2097152}}));
    // 2026-09-16: dflash 分支支持 FP8 在线量化（SM80 无 FP8 TC，走 Marlin W8A16）。
    // BF16 权重(~55GB)会吃掉 gpu-mem 0.90 的大部分，KV 池仅剩 ~5GB < 17.65GB 需求 → OOM。
    // FP8 量化后权重 ~28GB，KV 池 ~350K token，与 MTP 预设行为一致。
    if (kvCacheQuant === 'fp8') args.push('--quantization', 'fp8');
    // API 响应带 prompt token 明细（cached_tokens 等），零成本，利于观察缓存命中
    args.push('--enable-prompt-tokens-details');
  } else if (isDspark) {
    // DSpark：DFlash 骨干 + Markov 头，vllm-env 0.27.1 原生支持（method=dspark）
    args.push('--speculative-config', `{"method":"dspark","model":"${pickDraft('/home/ll/models/qwen3.8-27b-dspark')}","num_speculative_tokens":${dsparkN}}`);
  } else if (mtp === '1') {
    args.push('--speculative-config', `{"method":"qwen3_next_mtp","num_speculative_tokens":${mtpN}}`);
  }

  // 多卡启动：TP=张量并行（默认，每层切分到多卡）/ PP=流水线并行（按层分阶段）；
  // 并行度必须等于实际启用的显卡数
  if (gpuN > 1) {
    if (parallelMode === 'pp') {
      args.push('--pipeline-parallel-size', gpuN.toString());
    } else {
      args.push('--tensor-parallel-size', gpuN.toString());
    }
  }

  // [__ncclp2p_1001_arg__] P2P 损坏 ⇒ 多卡实例必须关 custom allreduce（它走 CUDA IPC 直读对端显存，
  // 不受 NCCL_P2P_DISABLE 管辖，会静默污染数据）。弹窗显式开启过时不重复添加。
  if (gpuN > 1 && !args.includes('--disable-custom-all-reduce')) {
    args.push('--disable-custom-all-reduce');
  }

  // KV 缓存量化：fp8_kv 权重不量化仅量化 KV；fp8/int8 同时量化权重+KV。
  // 注意：若模型自身 config.json 已声明量化（quantization_config，如 ornith 的
  // compressed-tensors、qwen fp8），再传 --quantization 会因不匹配直接启动失败
  // （pydantic ValidationError）。因此模型已声明量化时一律不传 --quantization，
  // 由 vLLM 从模型配置推断；int8 的 KV 缓存量化仍可单独应用。
  const modelConfigPath = path.join(modelPath, 'config.json');
  let modelQuant = null;
  try {
    const mc = JSON.parse(fs.readFileSync(modelConfigPath, 'utf8'));
    // 新版 transformers(5.x) 把量化配置放到 compression_config 键（如 hygon 的 W8A8 INT8），
    // 旧版仍是 quantization_config。两者都识别，避免误判"未量化"而叠加 --quantization 导致启动失败。
    const qc = mc.quantization_config || mc.compression_config;
    if (qc) modelQuant = qc.quant_method || qc.quantization_method || 'yes';
  } catch (e) {}
  if (modelQuant) {
    // 模型自身已量化（如 qwen3.8-27b-fp8）：不能再传 --quantization（会 pydantic 校验失败），
    // 但用户仍可单独对 KV cache 做量化（--kv-cache-dtype），不动权重。
    if (kvCacheQuant === 'int8') args.push('--kv-cache-dtype', 'int8');
    else if (kvCacheQuant === 'fp8' || kvCacheQuant === 'fp8_kv') args.push('--kv-cache-dtype', 'fp8');
    else if (kvCacheQuant === 'bfloat16' || kvCacheQuant === 'bf16') args.push('--kv-cache-dtype', 'bfloat16');
  } else if (kvCacheQuant === 'fp8_kv') {
    args.push('--quantization', 'fp8_kv');
  } else if (kvCacheQuant === 'fp8') {
    args.push('--quantization', 'fp8');
  } else if (kvCacheQuant === 'int8') {
    args.push('--quantization', 'awq');
    args.push('--kv-cache-dtype', 'int8');
  }

  // [kvoff-toggle 09-22] CPU KV 二级缓存（vLLM 原生 OffloadingConnector，纯内存档、不落盘）。
  // 弹窗缺省关闭 → 不传任何 flag，与旧行为逐字一致；PD 模式跳过（其自带 NixlConnector 配置）。
  // 注意：mamba 混合模型（qwen3.8-27b 系列）在本机 vLLM 0.29 上未验证过 connector，
  // 若开启后启动失败，把弹窗「二级缓存」切回关闭即可。
  if ((String(kvoff) === '1' || String(kvoff) === 'simple') && String(pdMode) !== '1') {
    const koGiB = Math.max(8, Math.min(256, parseInt(kvoffGiB, 10) || 64));
    args.push('--kv-transfer-config', JSON.stringify({
      kv_connector: 'OffloadingConnector',
      kv_role: 'kv_both',
      kv_connector_extra_config: { cpu_bytes_to_use: koGiB * 1073741824 },
    }));
  }

  // 采样参数：通过 override-generation-config 合并覆盖模型的默认采样配置
  // （此 vLLM fork 0.27.1 无 --temperature 等独立 CLI 参数，仅有 --override-generation-config JSON）
  args.push('--override-generation-config', JSON.stringify(samplingParams));

  // ====== vLLM 附加启动参数（前端「vLLM 高级参数」区块，仅 vllm 运行时传入）======
  // 附加命令行参数：shell 风格分词（支持引号 / $((...)) 算术），整体追加在基础命令之后。
  // 同名 flag 时附加参数胜出：先移除基础命令中的出现（flag 及其值），再追加附加参数。
  // 放在 PD 分派之前：PD 双实例会继承同一份附加参数（各自端口/并行度仍由 PD 逻辑改写）。
  {
    const extraArgsRaw = String(params.vllmExtraArgs || '').trim();
    if (extraArgsRaw) {
      const extraTokens = shellTokenize(extraArgsRaw);
      // 显示名称统一以弹窗「显示名称」字段为准：附加参数里的 --served-model-name（及其值）一律忽略（08-30），
      // 否则预设/附加参数会静默覆盖字段值，导致改名"没修改成功"。
      {
        let si = extraTokens.indexOf('--served-model-name');
        while (si !== -1) {
          const removeN = (si + 1 < extraTokens.length && !String(extraTokens[si + 1]).startsWith('--')) ? 2 : 1;
          extraTokens.splice(si, removeN);
          si = extraTokens.indexOf('--served-model-name', si);
        }
      }
      if (extraTokens.length) {
        for (const t of extraTokens) {
          if (!t.startsWith('--')) continue;
          const i = args.indexOf(t);
          if (i === -1) continue;
          // 若下一 token 是值（不以 -- 开头）则连同 flag 一起移除，避免残留旧值
          const removeN = (i + 1 < args.length && !String(args[i + 1]).startsWith('--')) ? 2 : 1;
          args.splice(i, removeN);
        }
        args.push(...extraTokens);
      }
    }
  }

  const env = {
    ...process.env,
    VLLM_USE_MODELSCOPE: 'False',
    VLLM_USE_FLASHINFER_SAMPLER: '0',
    VLLM_MOE_USE_DEEP_GEMM: '0',
    VLLM_SLEEP_WHEN_IDLE: '1',
    // ---- Hermes 修复：GCC10(c++20) + nvrtc LD_LIBRARY_PATH + humming 兜底 ----
    VLLM_HUMMING_USE_FALLBACK: '1',
    HUMMING_NO_JIT: '1',
    CC: 'gcc-10',
    CXX: 'g++-10',
    LD_LIBRARY_PATH: VENV + '/lib/python3.11/site-packages/nvidia/cu13/lib' + (process.env.LD_LIBRARY_PATH ? ':' + process.env.LD_LIBRARY_PATH : ''),
    CUDA_VISIBLE_DEVICES: gpuDevices,
    CUDA_HOME: VENV + '/lib/python3.11/site-packages/nvidia/cu13',
    PATH: VENV + '/lib/python3.11/site-packages/nvidia/cu13/bin:' + VENV + '/bin:' + process.env.PATH,
    FLASHINFER_EXTRA_CUDAFLAGS: '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK',
    FLASHINFER_DISABLE_VERSION_CHECK: '1',
    // 2026-09-16: 本地模型路径解析失败时禁止回退 HF hub 在线查询（否则报 Repo id 错误）
    HF_HUB_OFFLINE: '1',
    TRANSFORMERS_OFFLINE: '1',
    // [__ncclp2p_1001_env__] 本机 BAR1 P2P 补丁数据通路静默损坏（10-01 实锤：跨卡拷贝 data ok=False），
    // 任何多卡实例（TP/PP/PD）一律 NCCL 走 host SHM，否则首个 all-reduce 即死锁。
    NCCL_P2P_DISABLE: '1',
    NCCL_SHM_DISABLE: '0',
    NCCL_CUMEM_ENABLE: '0',
    NCCL_NET_GDR_LEVEL: '0',
  };

  // 上下文长度超过模型原生上限时自动加 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  if (needAllowLong) {
    env.VLLM_ALLOW_LONG_MAX_MODEL_LEN = '1';
  }

  // ====== vLLM 附加环境变量（前端「vLLM 高级参数」区块，仅 vllm 运行时传入）======
  // 每行一个 KEY=VALUE（# 开头为注释行）；KEY 必须为合法环境变量名，否则丢弃。
  // 用户变量在基础 env 之后合并，同名时用户值胜出（如覆盖 VLLM_LOGGING_LEVEL）。
  {
    const extraEnvRaw = String(params.vllmExtraEnv || '').trim();
    if (extraEnvRaw) {
      for (const line of extraEnvRaw.split('\n')) {
        const t = line.trim();
        if (!t || t.startsWith('#')) continue;
        const eq = t.indexOf('=');
        if (eq <= 0) continue;
        const k = t.slice(0, eq).trim();
        const val = t.slice(eq + 1).trim();
        if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(k)) env[k] = val;
      }
    }
  }

  let callbackCalled = false;
  function safeCallback(result) {
    if (callbackCalled) return;
    callbackCalled = true;
    callback(result);
  }

  const logFile = path.join(__dirname, 'vllm.log');
  // Must pass a real fd number to stdio: an unopened WriteStream (fd: null)
  // is rejected by spawn with '"value" is invalid for option "stdio"'.
  let logFd = null;
  try { logFd = fs.openSync(logFile, 'a'); } catch (e) { /* logging disabled */ }

  // PD 分离模式：拆成 prefill + decode 两个实例（见 startVllmPd）
  if (pdMode === '1') {
    return startVllmPd(modelName, params, { args, env, venv: VENV }, callback);
  }

  const vllm = spawn(VENV + '/bin/vllm', args, {
    env,
    detached: true,
    stdio: logFd === null ? ['ignore', 'ignore', 'ignore'] : ['ignore', logFd, logFd],
  });
  vllm.on('exit', () => {
    if (logFd !== null) { try { fs.closeSync(logFd); } catch (e) {} }
  });

  vllm.on('error', (err) => {
    safeCallback({ success: false, error: err.message });
  });

  // Wait a moment and check if it's still running
  setTimeout(() => {
    if (callbackCalled) return; // already handled by 'error'
    try {
      const alive = process.kill(vllm.pid, 0);
      if (alive) {
        safeCallback({ success: true, pid: vllm.pid, model: modelName, runtime: 'vllm' });
      } else {
        safeCallback({ success: false, error: 'vLLM process exited immediately' });
      }
    } catch (e) {
      safeCallback({ success: false, error: e.message });
    }
  }, 5000);
}

// ====== PD（prefill/decode）双实例启动 ======
// 由 startVllmModel 在 pdMode=1 时分派：同一模型起两个 vLLM 实例——
//   prefill（kv_producer，第 1 卡，端口 port）只算 KV 缓存
//   decode（kv_consumer，第 2 卡，端口 port+1）拿 KV 直接生成
// 必须的环境差异（对照 08-27 双卡实测结论）：
//   - nixl 依赖 libssl.so.3/libcrypto.so.3 → LD_LIBRARY_PATH 前置 /home/ll/libssl3
//   - hybrid mamba 模型 conv state 传输 → VLLM_SSM_CONV_STATE_LAYOUT=DS
//   - 同机两个实例侧信道端口必须不同 → prefill 5600 / decode 5601
// 启动成功后控制台代理（PD_MODEL_PORTS）自动走两步转发，客户端无感。
async function startVllmPd(modelName, params, base, callback) {
  const { spawn } = require('child_process');
  const { port, servedName, gpuId } = params;
  const decodePort = port + 1;
  const gpuN = Math.max(1, parseInt(params.gpuCount) || 1);
  if (gpuN < 2) {
    callback({ success: false, error: 'PD 分离模式需要至少 2 张显卡（一卡 prefill + 一卡 decode），请把「启用显卡数量」设为 2' });
    return;
  }
  if (portInUse(decodePort)) {
    callback({ success: false, error: `PD 模式第二个端口 ${decodePort} 已被占用，请换一个空闲起始端口` });
    return;
  }
  await killPortResidents(decodePort);

  // args 工具函数：去掉成对参数（--tensor-parallel-size N / --speculative-config JSON）
  const stripArgs = (args, names) => {
    const out = [];
    for (let i = 0; i < args.length; i++) {
      if (names.includes(args[i])) { i++; continue; }
      out.push(args[i]);
    }
    return out;
  };
  const replacePort = (args, p) => {
    const idx = args.indexOf('--port');
    if (idx >= 0 && idx + 1 < args.length) args[idx + 1] = String(p);
    return args;
  };

  const baseArgs = base.args.slice();

  // decode 端投机参数（DFlash2/MTP）单独取出；PD 下 prefill 端也配相同投机配置——
  // 让两侧 physical_blocks_per_logical 一致（实测：仅 decode 带 MTP 时 51 vs 49，
  // NIXL 握手拒绝 hybrid mamba 模型的前缀缓存传输；两侧同配后同为 51，握手通过，
  // MTP 只在 decode 端实际执行，prefill 端仅布局对齐）。08-27 实测验证。
  let specArg = null;
  const specIdx = baseArgs.indexOf('--speculative-config');
  if (specIdx >= 0 && specIdx + 1 < baseArgs.length) specArg = baseArgs.slice(specIdx, specIdx + 2);
  const baseArgsNoSpec = specArg ? stripArgs(baseArgs.slice(), ['--speculative-config']) : baseArgs.slice();

  const prefillArgs = replacePort(
    stripArgs(baseArgsNoSpec, ['--tensor-parallel-size', '--pipeline-parallel-size']),
    port);
  prefillArgs.push('--kv-transfer-config', JSON.stringify({ kv_connector: 'NixlConnector', kv_role: 'kv_producer', kv_rank: 0, kv_parallel_size: 1 }));
  if (specArg) prefillArgs.push(...specArg);
  const decodeArgs = replacePort(
    stripArgs(baseArgsNoSpec, ['--tensor-parallel-size', '--pipeline-parallel-size']),
    decodePort);
  decodeArgs.push('--kv-transfer-config', JSON.stringify({ kv_connector: 'NixlConnector', kv_role: 'kv_consumer', kv_rank: 1, kv_parallel_size: 1 }));
  if (specArg) decodeArgs.push(...specArg);

  const ssl3Path = '/home/ll/libssl3';
  // flashinfer JIT 编译必需：nvcc 在 pip cu13 包里（CUDA_HOME），且 0.6.16 的 cccl
  // 与 CUDA 13.3 的版本检查不兼容需跳过（FLASHINFER_EXTRA_CUDAFLAGS）。缺则 PD 实例
  // 启动触发新 kernel 编译时失败（"Could not find nvcc" / cccl incompatible）。
  const jitEnv = {
    CUDA_HOME: base.venv + '/lib/python3.11/site-packages/nvidia/cu13',
    FLASHINFER_EXTRA_CUDAFLAGS: '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK',
  };
  const prefillEnv = {
    ...base.env,
    ...jitEnv,
    CUDA_VISIBLE_DEVICES: String(gpuId),
    LD_LIBRARY_PATH: ssl3Path + (base.env.LD_LIBRARY_PATH ? ':' + base.env.LD_LIBRARY_PATH : ''),
    VLLM_SSM_CONV_STATE_LAYOUT: 'DS',
    VLLM_NIXL_SIDE_CHANNEL_PORT: '5600',
  };
  const decodeEnv = {
    ...base.env,
    ...jitEnv,
    CUDA_VISIBLE_DEVICES: String(gpuId + 1),
    LD_LIBRARY_PATH: ssl3Path + (base.env.LD_LIBRARY_PATH ? ':' + base.env.LD_LIBRARY_PATH : ''),
    VLLM_SSM_CONV_STATE_LAYOUT: 'DS',
    VLLM_NIXL_SIDE_CHANNEL_PORT: '5601',
  };

  let callbackCalled = false;
  const safeCallback = (r) => { if (callbackCalled) return; callbackCalled = true; callback(r); };
  const openLog = (name) => {
    try { return fs.openSync(path.join(__dirname, name), 'a'); } catch (e) { return null; }
  };
  const fdP = openLog('vllm-pd-prefill.log');
  const fdD = openLog('vllm-pd-decode.log');

  const proc1 = spawn(base.venv + '/bin/vllm', prefillArgs, {
    env: prefillEnv, detached: true,
    stdio: fdP === null ? ['ignore', 'ignore', 'ignore'] : ['ignore', fdP, fdP],
  });
  const proc2 = spawn(base.venv + '/bin/vllm', decodeArgs, {
    env: decodeEnv, detached: true,
    stdio: fdD === null ? ['ignore', 'ignore', 'ignore'] : ['ignore', fdD, fdD],
  });

  const killAll = () => {
    try { process.kill(proc1.pid, 'SIGKILL'); } catch (e) {}
    try { process.kill(proc2.pid, 'SIGKILL'); } catch (e) {}
  };

  setTimeout(() => {
    let a1 = false, a2 = false;
    try { a1 = process.kill(proc1.pid, 0); } catch (e) { a1 = false; }
    try { a2 = process.kill(proc2.pid, 0); } catch (e) { a2 = false; }
    if (a1 && a2) {
      safeCallback({ success: true, pid: proc1.pid, pidDecode: proc2.pid, model: modelName, runtime: 'vllm', pd: true, prefillPort: port, decodePort });
    } else if (!a1 && !a2) {
      safeCallback({ success: false, error: `PD 两个实例均启动失败，详见 vllm-pd-prefill.log / vllm-pd-decode.log（端口 ${port}/${decodePort}）` });
    } else if (!a1) {
      killAll();
      safeCallback({ success: false, error: `prefill 实例启动失败（端口 ${port}），已连带停止 decode；详见 vllm-pd-prefill.log` });
    } else {
      killAll();
      safeCallback({ success: false, error: `decode 实例启动失败（端口 ${decodePort}），已连带停止 prefill；详见 vllm-pd-decode.log` });
    }
  }, 5000);

  proc1.on('error', (err) => safeCallback({ success: false, error: 'prefill 启动错误: ' + err.message }));
  proc2.on('error', (err) => safeCallback({ success: false, error: 'decode 启动错误: ' + err.message }));
  proc1.on('exit', () => { try { if (fdP !== null) fs.closeSync(fdP); } catch (e) {} });
  proc2.on('exit', () => { try { if (fdD !== null) fs.closeSync(fdD); } catch (e) {} });
}

// ====== SGLang 实例启动 ======
// 参数与 vllm 分支同源（port / dflash / mtpTokens / gpuMemUtil / servedName / gpuId / gpuCount），
// 但只使用 sglang 支持的子集。DFlash2 投机解码与本地部署验证过的参数一致：
//   --disable-prefill-cuda-graph 必带（GDN 线性注意力 Triton 内核在 prefill CUDA 图捕获时
//   会非法内存访问，把 GPU 打成 NV_ERR_RESET_REQUIRED，只能重启机器恢复）。
// Shell 风格分词：处理空白 / 单引号 / 双引号 / 反斜杠转义，并求值 $((...)) 算术
// （如 $((256*1024)) → 262144）。仅允许纯数字与四则运算符进入求值，无代码执行风险。
function shellTokenize(input) {
  const s = String(input);
  const tokens = [];
  let cur = '', has = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === '\\' && i + 1 < s.length) { cur += s[++i]; has = true; continue; }
    if (c === "'" || c === '"') {
      let j = i + 1;
      while (j < s.length && s[j] !== c) j++;
      cur += s.slice(i + 1, j); has = true; i = j; continue;
    }
    if (c === '$' && s[i + 1] === '(' && s[i + 2] === '(') {
      // $(( expr ))：两个右括号都需消费，depth 从 2 起算
      let j = i + 3, depth = 2;
      while (j < s.length && depth > 0) {
        if (s[j] === '(') depth++;
        else if (s[j] === ')') depth--;
        j++;
      }
      const expr = s.slice(i + 3, Math.max(i + 3, j - 2));
      if (/\d/.test(expr) && /^[\d+\-*/%() ]+$/.test(expr)) {
        try { cur += String(Function('"use strict";return (' + expr + ')')()); has = true; i = j - 1; continue; } catch (e) { /* 求值失败按字面量处理 */ }
      }
      cur += c; has = true; continue;
    }
    if (/\s/.test(c)) { if (has) { tokens.push(cur); cur = ''; has = false; } continue; }
    cur += c; has = true;
  }
  if (has) tokens.push(cur);
  return tokens;
}

async function startSglangModel(modelName, params, callback) {
  const { execSync } = require('child_process');
  const { spawn } = require('child_process');
  const { port, gpuId, gpuCount, parallelMode, gpuMemUtil, servedName, mtpTokens, kvCacheQuant, thinking, thinkingEffort } = params;
  // reasoning_effort：允许 low/medium/high/xhigh 及手动输入
  const rawEffort = String(thinkingEffort || '').trim();
  // [gen-default 0927] 空值=未填 → 用启动页默认 xhigh；填了但非法 → 保守回落 medium（不静默变最深档）
  const effort = rawEffort === '' ? 'xhigh' : (/^[a-zA-Z0-9_-]{1,32}$/.test(rawEffort) ? rawEffort : 'medium');
  const VENV = SGLANG_VENV;

  if (!fs.existsSync(path.join(VENV, 'bin', 'python'))) {
    callback({ success: false, error: 'sglang 环境未就绪：未找到 /home/ll/sglang-env/bin/python，请先完成 sglang 安装' });
    return;
  }
  const isMtp = params.mtp === '1';
  const isDflash = params.dflash === '1';
  const isDspark = params.dspark === '1';
  if (isMtp) {
    // SGLang MTP：内置 NEXTN 算法，Qwen3.8-27B 走 Qwen3NextForCausalLMMTP
    // 草稿模型路径 = 目标模型路径（MTP 层从目标 ckpt 提取）
    if (!String(modelName).includes('qwen3.8-27b')) {
      callback({ success: false, error: 'SGLang MTP 目前仅支持 qwen3.8-27b 系列目标模型（Qwen3NextForCausalLMMTP）' });
      return;
    }
  }
  if (isDflash) {
    if (!fs.existsSync(path.join(SGLANG_DRAFT_PATH, 'config.json'))) {
      callback({ success: false, error: 'DFlash2 草稿模型未下载：/home/ll/models/qwen3.8-27b-dflash2/config.json（需先下载 incoai/Qwen3.8-27B-DFlash2）' });
      return;
    }
    if (!String(modelName).includes('qwen3.8-27b')) {
      callback({ success: false, error: 'DFlash2 草稿模型目前仅支持 qwen3.8-27b 系列目标模型' });
      return;
    }
  }
  if (isDspark) {
    if (!fs.existsSync(path.join(SGLANG_DSPARK_PATH, 'config.json'))) {
      callback({ success: false, error: 'DSpark 草稿模型未下载：/home/ll/models/qwen3.8-27b-dspark/config.json（需先下载 incoai/dspark-qwen3.8-27b）' });
      return;
    }
    if (!String(modelName).includes('qwen3.8-27b')) {
      callback({ success: false, error: 'DSpark 草稿模型目前仅支持 qwen3.8-27b 系列目标模型' });
      return;
    }
  }

  // 清理目标端口残留（vllm/sglang 均可）并等待显存释放
  await killPortResidents(port);

  const gpuN = Math.max(1, parseInt(gpuCount) || 1);
  const gpuDevices = Array.from({ length: gpuN }, (_, i) => (parseInt(gpuId) || 0) + i).join(',');

  // 显存占用上限保护（sglang 专用）：
  // CMP 170HX 无原生 FP8 → sglang 走 Marlin weight-only；DFLASH 的 GDN Triton 内核
  // 是 decode 时才动态加载的（chunk_gated_delta_rule_fwd_*，约 0.5~0.6 GB）。若
  // --mem-fraction-static 过高（如 0.95）会预占全部显存，decode 加载该内核时 OOM，
  // 损坏 CUDA 上下文 → illegal memory access / UVM fatal error，只能重启机器恢复。
  // 故 DFLASH 上限 0.88、普通 sglang 上限 0.92，给运行时 kernel 预留余量。
  const memFracRaw = parseFloat(gpuMemUtil) || 0.9;
  const memFracCap = (isMtp || isDflash || isDspark) ? 0.88 : 0.92;
  const memFracSafe = Math.min(memFracRaw, memFracCap);
  const memFracClamped = memFracRaw > memFracSafe;

  const args = [
    '-m', 'sglang.launch_server',
    '--model-path', path.join(MODELS_DIR, modelName),
    '--port', port.toString(),
    '--served-model-name', servedName,
    '--trust-remote-code',
    '--mem-fraction-static', String(memFracSafe),
    '--disable-prefill-cuda-graph',
    '--enable-metrics',  // 开启 Prometheus /metrics 端点（默认关闭），仪表盘统计依赖它
    '--reasoning-parser', 'qwen3',
    // 工具调用解析（与 vLLM 分支对齐）：qwen3_coder 解析器把模型的 <tool_call> 文本解析成
    // OpenAI 结构化的 tool_calls 数组；不带此参数时模型会输出原始工具文本、tool_calls=null，
    // 客户端工具调用会报错。模型模板会自动检测（日志确认 qwen3_coder 可用）。
    '--tool-call-parser', 'qwen3_coder',
    '--default-chat-template-kwargs', thinking === '1'
      ? '{"enable_thinking": true, "reasoning_effort": "' + effort + '"}'
      : '{"enable_thinking": false}',
    // 每请求实测数据导出（与 vLLM stat-logger 的 request-traces.jsonl 对应）：
    // sglang 内置 FileRequestMetricsExporter，每个请求完成后写一行 JSONL，
    // 前端「最近完成请求」表格据此显示 sglang 的真实请求数据。
    '--export-metrics-to-file',
    '--export-metrics-to-file-dir', path.join(__dirname, 'sglang-request-metrics'),
    // ====== 09-01 用户要求：所有 SGLang 启动统一附带以下两个 flag（startSglangModel 是唯一入口，
    // 弹窗/快速启动预设/预设启动 API 全部走此基础命令，一处生效全路径覆盖）======
    // --dp 1：数据并行度显式钉 1（sglang 默认即 1，写死防未来默认变化误改语义，单卡/TP 拓扑不变）。
    // --schedule-policy lpm：Longest Prefix Match 优先（sglang 默认 fcfs）。交织流量下优先调度共享
    // 最长公共前缀的请求，最大化 radix cache 复用、提高前缀缓存命中率（生产定版）。
    // 「SGLang 附加启动参数」里含同名 flag 时附加参数胜出（下方去重逻辑先移除基础命令中的同名项）。
    '--dp', '1',
    '--schedule-policy', 'lpm',
    // 090x：SGLang 默认 enable_cache_report=False，不往 OpenAI usage 上报 cached_tokens，
    // 导致 DSH/pi-ai 拿不到 cacheRead、前端不显示缓存命中。显式开启。
    '--enable-cache-report',
  ];
  // 多卡：TP=张量并行（默认）/ PP=流水线并行；「SGLang 附加启动参数」里显式指定 --tp/--pp 时附加参数胜出
  // （下方去重逻辑会移除基础命令中的同名 flag）
  if (gpuN > 1) {
    if (parallelMode === 'pp') args.push('--pp', String(gpuN));
    else args.push('--tp', String(gpuN));
  }
  if (isMtp) {
    // SGLang MTP：NEXTN（内部折叠为 EAGLE），草稿模型路径 = 目标模型路径
    // MTP 层从目标 ckpt 提取（Qwen3NextForCausalLMMTP）。
    // 约束：topk=1 时 SGLang 要求 num_draft_tokens = num_steps + 1，
    // 且 num_steps 与 topk 均不可缺省（缺省会触发参数解析断言失败）。
    // 用户「投机步数」= 每步投机 token 数 = num_draft_tokens，故 num_steps = N-1。
    const mtpN = Math.min(8, Math.max(1, parseInt(mtpTokens) || 5));
    const mtpSteps = Math.max(1, mtpN - 1);
    args.push(
      '--speculative-algorithm', 'NEXTN',
      '--speculative-draft-model-path', path.join(MODELS_DIR, modelName),
      '--speculative-eagle-topk', '1',
      '--speculative-num-steps', String(mtpSteps),
      '--speculative-num-draft-tokens', String(mtpSteps + 1)
    );
  } else if (isDflash) {
    args.push(
      '--speculative-algorithm', 'DFLASH',
      '--speculative-draft-model-path', SGLANG_DRAFT_PATH,
      '--speculative-num-draft-tokens', String(Math.min(8, Math.max(1, parseInt(mtpTokens) || 8)))
    );
  } else if (isDspark) {
    // DSPARK：gamma 由草稿 ckpt 的 dspark_block_size 定死（训练决定，运行时改小无效），
    // sglang 硬校验 --speculative-num-draft-tokens 必须严格等于 gamma+1（verify 窗口），
    // 故以 ckpt config 为准；UI「投机 token 数」不再当 gamma（旧逻辑 UI 填 5 → 6 被拒启动即退）
    const dsparkGamma = readDsparkGamma(SGLANG_DSPARK_PATH) || 7;
    if (parseInt(mtpTokens) && parseInt(mtpTokens) !== dsparkGamma) {
      console.log(`[sglang-start] DSPARK gamma 固定为草稿 ckpt 的 dspark_block_size=${dsparkGamma}，忽略 UI 输入 ${mtpTokens}`);
    }
    args.push(
      '--speculative-algorithm', 'DSPARK',
      '--speculative-draft-model-path', SGLANG_DSPARK_PATH,
      '--speculative-num-draft-tokens', String(dsparkGamma + 1)
    );
  }
  // KV 缓存量化（sglang 用 --kv-cache-dtype；仅 KV 生效、不动权重，故 fp8 与 fp8_kv 等价）。
  // sglang 支持 auto / fp8_e4m3 / fp8_e5m2 / mxfp8 / bf16 / nvfp4 / fp4_mx_block16，不支持 int8，
  // 故 int8 在 sglang 下回落 auto（不传）+ 前端已有「sglang 不支持 int8」提示。
  {
    const kv = String(kvCacheQuant || 'auto');
    if (kv === 'fp8' || kv === 'fp8_kv') {
      args.push('--kv-cache-dtype', 'fp8_e4m3');
    }
  }
  // 采样参数：sglang 用 --preferred-sampling-params（JSON 格式），与 vllm 的 --override-generation-config 对应
  {
    const p = {};
    if (params.temperature !== undefined && params.temperature !== '' && params.temperature !== null) p.temperature = parseFloat(params.temperature);
    if (params.topP !== undefined && params.topP !== '' && params.topP !== null) p.top_p = parseFloat(params.topP);
    if (params.topK !== undefined && params.topK !== '' && params.topK !== null) p.top_k = parseInt(params.topK);
    if (params.minP !== undefined && params.minP !== '' && params.minP !== null) p.min_p = parseFloat(params.minP);
    if (params.repetitionPenalty !== undefined && params.repetitionPenalty !== '' && params.repetitionPenalty !== null) p.repetition_penalty = parseFloat(params.repetitionPenalty);
    if (Object.keys(p).length > 0) {
      args.push('--preferred-sampling-params', JSON.stringify(p));
    }
  }

  // ====== SGLang 附加启动参数（前端「SGLang 附加启动参数」框，仅 sglang 运行时传入）======
  // 附加命令行参数：shell 风格分词（支持引号 / $((...)) 算术），整体追加在基础命令之后。
  // 同名 flag 时附加参数胜出：先移除基础命令中的出现（flag 及其值），再追加附加参数。
  {
    const extraArgsRaw = String(params.sglangExtraArgs || '').trim();
    if (extraArgsRaw) {
      const extraTokens = shellTokenize(extraArgsRaw);
      // 显示名称统一以弹窗「显示名称」字段为准：附加参数里的 --served-model-name（及其值）一律忽略（08-30），
      // 否则预设/附加参数会静默覆盖字段值，导致改名"没修改成功"。
      {
        let si = extraTokens.indexOf('--served-model-name');
        while (si !== -1) {
          const removeN = (si + 1 < extraTokens.length && !String(extraTokens[si + 1]).startsWith('--')) ? 2 : 1;
          extraTokens.splice(si, removeN);
          si = extraTokens.indexOf('--served-model-name', si);
        }
      }
      if (extraTokens.length) {
        for (const t of extraTokens) {
          if (!t.startsWith('--')) continue;
          const i = args.indexOf(t);
          if (i === -1) continue;
          // 若下一 token 是值（不以 -- 开头）则连同 flag 一起移除，避免残留旧值
          const removeN = (i + 1 < args.length && !String(args[i + 1]).startsWith('--')) ? 2 : 1;
          args.splice(i, removeN);
        }
        args.push(...extraTokens);
      }
    }
  }

  const env = {
    ...process.env,
    SGLANG_USE_MODELSCOPE: 'false',
    CUDA_VISIBLE_DEVICES: gpuDevices,
    CUDA_HOME: VENV + '/lib/python3.11/site-packages/nvidia/cu13',
    LD_LIBRARY_PATH: VENV + '/lib/python3.11/site-packages/nvidia/cu13/lib' + (process.env.LD_LIBRARY_PATH ? ':' + process.env.LD_LIBRARY_PATH : ''),
    // gcc-10 符号链接目录（sgl-kernel JIT 需要 C++20）+ sglang venv bin（ninja 等）
    PATH: '/home/ll/bin:' + VENV + '/bin:' + process.env.PATH,
    CC: 'gcc-10',
    CXX: 'g++-10',
    // pip 混装 nvcc 13.3 / cudart 13.0 时 flashinfer 运行时 JIT 的 CCCL 兼容性校验会失败
    FLASHINFER_EXTRA_CUDAFLAGS: '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK',
  };
  // 附加环境变量（前端每行一个 KEY=VALUE），覆盖同名基础 env（如 CUDA_VISIBLE_DEVICES）
  {
    const extraEnvRaw = String(params.sglangExtraEnv || '').trim();
    if (extraEnvRaw) {
      for (const line of extraEnvRaw.split('\n')) {
        const l = line.trim();
        if (!l || l.startsWith('#')) continue;
        const m = l.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/);
        if (m) env[m[1]] = m[2].trim().replace(/^["']|["']$/g, '');
      }
    }
  }

  let callbackCalled = false;
  function safeCallback(result) {
    if (callbackCalled) return;
    callbackCalled = true;
    callback(result);
  }

  const logFile = SGLANG_LOG_PATH;
  let logFd = null;
  try { logFd = fs.openSync(logFile, 'a'); } catch (e) { /* logging disabled */ }

  // 记录最终命令行（含附加参数），便于排查启动失败
  try { fs.appendFileSync(logFile, '# [' + new Date().toISOString() + '] cmd: ' + VENV + '/bin/python ' + args.join(' ') + '\n'); } catch (e) { /* ignore */ }

  const sgl = spawn(VENV + '/bin/python', args, {
    env,
    detached: true,
    stdio: logFd === null ? ['ignore', 'ignore', 'ignore'] : ['ignore', logFd, logFd],
  });
  sgl.on('exit', () => {
    if (logFd !== null) { try { fs.closeSync(logFd); } catch (e) {} }
  });
  sgl.on('error', (err) => {
    safeCallback({ success: false, error: err.message });
  });

  // 等 5 秒确认进程存活（sglang 启动加载需几分钟，此处只确认进程没立即退出）
  setTimeout(() => {
    if (callbackCalled) return;
    try {
      const alive = process.kill(sgl.pid, 0);
      if (alive) {
        const notice = memFracClamped
          ? '显存占用已自动限制为 ' + memFracSafe + '（您设置的是 ' + memFracRaw + '）：DFLASH 运行时需预留显存给动态加载的 Triton 内核，过高会 OOM 损坏 GPU 状态。'
          : undefined;
        safeCallback({ success: true, pid: sgl.pid, model: modelName, runtime: 'sglang', notice: notice });
      } else {
        safeCallback({ success: false, error: 'sglang process exited immediately（详见 ' + SGLANG_LOG_PATH + '）' });
      }
    } catch (e) {
      safeCallback({ success: false, error: e.message });
    }
  }, 5000);
}

async function stopVllm(callback, port) {
  // 通用停止：vllm（vllm serve + EngineCore）与 sglang（sglang.launch_server + 工作进程）都支持。
  port = port || config.vllmPort;
  try {
    const { execSync } = require('child_process');
    // 找该端口的实例主进程（cmdline 含 vllm serve / sglang.launch_server 且 --port 匹配），
    // 不动其他端口的实例。同样不能用 lsof：加载中的实例尚未监听端口，用 pgrep 匹配命令行更可靠。
    let apiPid = null;
    let runtime = null;
    const pids = execSync(`pgrep -f "vllm serve" 2>/dev/null; pgrep -f "sglang.launch_server" 2>/dev/null || true`, { encoding: 'utf8', timeout: 5000 }).trim().split('\n').filter(Boolean);
    for (const pid of pids) {
      try {
        const cmd = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8').split('\0').join(' ');
        if (cmd.includes(`--port ${port}`)) {
          apiPid = pid;
          runtime = cmd.includes('sglang') ? 'sglang' : 'vllm';
          break;
        }
      } catch (e) {}
    }
    // 兜底：按监听端口找（已就绪实例；-sTCP:LISTEN 排除控制台自身的客户端 socket）
    if (!apiPid) {
      try {
        const lp = execSync(`lsof -ti:${port} -sTCP:LISTEN 2>/dev/null || true`, { encoding: 'utf8', timeout: 3000 }).trim().split('\n').find(l => parseInt(l.trim()) !== process.pid);
        if (lp) {
          apiPid = lp.trim();
          runtime = detectRuntimeForPid(parseInt(apiPid)) || 'unknown';
        }
      } catch (e) {}
    }
    if (!apiPid) {
      // 主进程已不在（可能上次停止失败后残留 EngineCore/sglang 工作进程）：直接清理残留并等显存释放
      await killOrphanEngineCores();
      const sg = collectSglangPids(port);
      if (sg.length) await waitGpuMemRelease(sg);
      setTimeout(() => callback({ success: true, port }), 300);
      return;
    }
    // 1) 先优雅停止（SIGTERM）：让服务自行停掉子进程并释放显存
    try { execSync('kill -TERM ' + apiPid, { encoding: 'utf8', timeout: 3000 }); } catch (e2) {}
    let exited = false;
    for (let i = 0; i < 10; i++) {
      try {
        const stat = fs.readFileSync(`/proc/${apiPid}/stat`, 'utf8');
        const m = stat.match(/\)\s+([A-Z])/);
        const state = m ? m[1] : '?';
        if (state === 'Z' || state === 'X') { exited = true; break; } // 僵尸/已消失
      } catch (e) { exited = true; break; }
      await sleep(1000);
    }
    if (!exited) {
      // 2) 优雅超时：强杀主进程，残留子进程由下方兜底清理
      try { execSync('kill -9 ' + apiPid, { encoding: 'utf8', timeout: 3000 }); } catch (e2) {}
      await sleep(1000);
    }
    // 3) 清理残留：vllm → 孤儿 EngineCore/resource_tracker；sglang → 工作进程；并等待显存真正释放
    if (runtime === 'sglang') {
      const sg = collectSglangPids(port);
      if (sg.length) await waitGpuMemRelease(sg);
    } else {
      await killOrphanEngineCores();
    }
    setTimeout(() => callback({ success: true, port, runtime }), 500);
  } catch (e) {
    callback({ success: false, error: e.message });
  }
}

// ====== 功耗监测（Scaphandre 式：RAPL 总功率 + CPU 时间比例归因到进程 + 能量积分） ======
const osPower = require('os');
const CPU_CORES = osPower.cpus().length || 1;
let CPU_MODEL = 'CPU';
try {
  const _mi = fs.readFileSync('/proc/cpuinfo', 'utf8').match(/model name\s*:\s*(.+)/);
  if (_mi && _mi[1]) CPU_MODEL = _mi[1].trim();
} catch (e) {}
const RAPL_PKG_PATH = '/sys/class/powercap/intel-rapl:0/energy_uj';
let cpuPowerW = null;
let lastEnergyUj = null;
let lastEnergyTs = 0;
let gpuPowerW = null;      // 最近一次 nvidia-smi power.draw
let gpuUtil = null;        // GPU 利用率 %
let gpuProcs = [];         // [{pid, memMiB}] 正在使用 GPU 的进程
let lastProcSnap = null;   // { ts, pids: {pid: {t, name}} } 上一轮 /proc 快照
let lastProcDeltas = null; // { pid: {deltaJiffies, name, memMiB} } 本轮 CPU 时间增量
let procEnergyJ = new Map(); // pid -> 累计能量（焦耳），进程退出即清除
let lastCpuStat = null;
let cpuUtilPct = null;     // 整机 CPU 利用率 %
let cpuIdleBaselineW = null; // 空闲基线功率估算（利用率 ≤2% 时的缓慢 EMA）
let lastActiveCpuW = null;   // 本轮可归因的活跃 CPU 功率

function readProcSnap() {
  const now = Date.now();
  const pids = {};
  let names = [];
  try { names = fs.readdirSync('/proc'); } catch (e) { return null; }
  for (const n of names) {
    if (!/^\d+$/.test(n)) continue;
    try {
      const stat = fs.readFileSync('/proc/' + n + '/stat', 'utf8');
      const close = stat.lastIndexOf(')');
      if (close < 0) continue;
      const nm = (stat.match(/^\d+\s+\((.+)\)/) || [])[1] || '?';
      const f = stat.slice(close + 2).trim().split(/\s+/);
      // 括号后 f[0]=state(字段3) … utime=字段14→f[11]，stime=字段15→f[12]
      const t = (parseInt(f[11], 10) || 0) + (parseInt(f[12], 10) || 0);
      pids[n] = { t, name: nm };
    } catch (e) {}
  }
  return { ts: now, pids };
}

function readCpuUtilPct() {
  try {
    const line = fs.readFileSync('/proc/stat', 'utf8').split('\n')[0];
    const p = line.split(/\s+/).slice(1).map(x => parseInt(x, 10) || 0);
    const idle = (p[3] || 0) + (p[4] || 0);
    const total = p.reduce((a, b) => a + b, 0);
    if (lastCpuStat) {
      const dt = total - lastCpuStat.total;
      const di = idle - lastCpuStat.idle;
      if (dt > 0) cpuUtilPct = Math.max(0, Math.min(100, Math.round((1 - di / dt) * 100)));
    }
    lastCpuStat = { idle, total };
  } catch (e) {}
}

function samplePowerCycle() {
  // 10-07 测试修复（高危）：本函数每 2s 跑一次，原实现里有 3 处 execSync
  // （sudo cat RAPL + 两处 nvidia-smi，timeout 5s）。nvidia-smi 进 D 状态时 timeout 无效，
  // 每 2 秒就把 Node 事件循环按住一次且永不返回 → 全站接口无响应（09-20/09-29 实锤形态，
  // 也正是 10-03 立下的"2s 轮询链里做同步重 IO 即地雷"）。全部改为放弃式异步通道，
  // 并加防重入：一轮没结束不叠加下一轮。
  if (global.__powerCycleBusy) return;
  global.__powerCycleBusy = true;
  samplePowerCycleInner().catch(() => {}).then(() => { global.__powerCycleBusy = false; });
}
async function samplePowerCycleInner() {
  // 1) CPU 封装功率（RAPL 差值）
  let e = null;
  try {
    const out = await spawnCollect('sudo', ['-n', 'cat', RAPL_PKG_PATH], 3000, false);
    const v = parseInt(String(out || '').trim(), 10);
    if (!isNaN(v)) e = v;
  } catch (err) {}
  if (e !== null) {
    const now = Date.now();
    if (lastEnergyUj !== null && now > lastEnergyTs) {
      const dj = e - lastEnergyUj;
      const secs = (now - lastEnergyTs) / 1000;
      if (dj > 0 && secs > 0) cpuPowerW = Math.round((dj / 1e6 / secs) * 10) / 10;
    }
    lastEnergyUj = e;
    lastEnergyTs = now;
  }
  // 2) GPU 功率 + 使用 GPU 的进程（nvidia-smi，走放弃式通道 + 熔断）
  try {
    const g = await gpuQuery(['--query-gpu=power.draw,utilization.gpu', '--format=csv,noheader'], GPU_Q_MS_LIVE);
    const first = g ? String(g).trim().split('\n')[0] : '';
    if (first) {
      const p = first.split(',').map(s => s.trim());
      gpuPowerW = parseFloat(p[0]) || null;
      gpuUtil = parseInt(p[1]) || null;
    }
    const a = await gpuQuery(['--query-compute-apps=pid,used_memory', '--format=csv,noheader'], GPU_Q_MS_LIVE);
    gpuProcs = a ? String(a).trim().split('\n').map(l => {
      const [pid, mem] = l.split(',').map(s => s.trim());
      return { pid: parseInt(pid, 10) || 0, memMiB: parseInt(mem, 10) || 0 };
    }).filter(x => x.pid > 0) : [];
  } catch (err) {}
  // 3) CPU 利用率 + 空闲基线 + 进程 CPU 时间增量
  readCpuUtilPct();
  if (cpuPowerW !== null && cpuUtilPct !== null && cpuUtilPct <= 2) {
    // 近空闲：缓慢更新基线功率（EMA），用于把"空闲功耗"与"进程活跃功耗"分开
    cpuIdleBaselineW = cpuIdleBaselineW === null ? cpuPowerW : cpuIdleBaselineW * 0.9 + cpuPowerW * 0.1;
  }
  lastActiveCpuW = cpuPowerW !== null && cpuIdleBaselineW !== null
    ? Math.max(0, cpuPowerW - cpuIdleBaselineW)
    : cpuPowerW;
  const snap = readProcSnap();
  if (snap) {
    const dt = lastProcSnap ? (snap.ts - lastProcSnap.ts) / 1000 : 0;
    if (lastProcSnap && dt > 0.5) {
      const deltas = {};
      let totalDelta = 0;
      for (const pid of Object.keys(snap.pids)) {
        const prev = lastProcSnap.pids[pid];
        if (!prev) continue;
        const d = snap.pids[pid].t - prev.t;
        if (d > 0) { deltas[pid] = d; totalDelta += d; }
      }
      lastProcDeltas = deltas;
      // 4) 功率归因 + 能量积分（Scaphandre 比例归因法，仅归因活跃功率）
      if (totalDelta > 0 && lastActiveCpuW !== null) {
        const HZ = 100; // Linux USER_HZ
        const gpuMemSum = gpuProcs.reduce((s, g) => s + g.memMiB, 0);
        for (const pid of Object.keys(deltas)) {
          const cpuW = lastActiveCpuW * (deltas[pid] / totalDelta);
          let gpuW = 0;
          const gp = gpuProcs.find(g => g.pid === parseInt(pid, 10));
          if (gp && gpuPowerW !== null) gpuW = gpuPowerW * (gpuMemSum > 0 ? gp.memMiB / gpuMemSum : 1);
          const w = cpuW + gpuW;
          if (w > 0) procEnergyJ.set(parseInt(pid, 10), (procEnergyJ.get(parseInt(pid, 10)) || 0) + w * dt);
        }
      }
      for (const pid of Array.from(procEnergyJ.keys())) {
        if (!snap.pids[pid]) procEnergyJ.delete(pid);
      }
    }
    lastProcSnap = snap;
  }
}
samplePowerCycle();
setInterval(samplePowerCycle, 2000).unref();

// ====== 处理器（CPU）详情（8889 存储管理页 CPU 卡）======
// 静态规格：/proc/cpuinfo + /sys/.../topology + cache + cpufreq + dmidecode(免密可选) + vulnerabilities，5 分钟缓存；
// 动态采样：/proc/stat 逐核增量（1s 窗口）→ global.__cpuLive；整机利用率历史 → global.__cpuHistory（保留 30 分钟）。
function cpuRead(f) { try { return fs.readFileSync(f, 'utf8'); } catch (e) { return null; } }
function cpuNum(f) { const t = cpuRead(f); if (t == null) return null; const n = parseInt(t.trim(), 10); return Number.isFinite(n) ? n : null; }
function expandCpuList(s) {
  const out = [];
  for (const part of String(s || '').trim().split(',')) {
    const p2 = part.trim();
    if (!p2) continue;
    const m = p2.match(/^(\d+)(?:-(\d+))?$/);
    if (!m) continue;
    const a = +m[1], b = m[2] != null ? +m[2] : a;
    for (let i = a; i <= b && i - a < 1024; i++) out.push(i);
  }
  return out;
}
function buildCpuStatic() {
  const out = { supported: false };
  const txt = cpuRead('/proc/cpuinfo');
  if (!txt) return out;
  out.supported = true;
  const get = (k) => { const m = txt.match(new RegExp('^' + k + '[ \\t]*:[ \\t]*(.+)$', 'm')); return m ? m[1].trim() : null; };
  out.model = get('model name');
  out.vendor = get('vendor_id');
  out.family = get('cpu family');
  out.model_id = get('model');
  out.stepping = get('stepping');
  out.microcode = get('microcode');
  out.bugs = get('bugs');
  try { out.arch = osPower.arch(); out.platform = osPower.release(); } catch (e) {}
  // 指令集：只保留对推理/通用负载有意义的一组，按 cpuinfo flags 命中过滤
  const flags = new Set((get('flags') || '').toUpperCase().replace(/[._]/g, '_').split(/\s+/));
  const FEATS = ['SSE','SSE2','SSE3','SSSE3','SSE4.1','SSE4.2','AVX','AVX2','FMA','AVX512F','AVX512BW','AVX512VL','AVX512_VNNI','AES','VAES','GFNI','BMI1','BMI2','SHA_NI','AMX-BF16','AMX-TILE','AMX-INT8','RDRAND','RDSEED'];
  out.features = FEATS.filter((f) => flags.has(f.replace(/[._]/g, '_')));
  // 拓扑：逻辑核数 + 每核线程兄弟（>1=超线程核；混合架构下 1 线程核=E 核）
  const sysbase = '/sys/devices/system/cpu';
  let cpus = [];
  try { cpus = fs.readdirSync(sysbase).filter((n) => /^cpu\d+$/.test(n)).map((n) => +n.slice(3)).sort((a, b) => a - b); } catch (e) {}
  out.logical = cpus.length || (txt.match(/^processor[ \t]*:/gm) || []).length;
  const coreList = [];
  let htThreads = 0, singleThreads = 0;
  for (const i of cpus) {
    const tp = sysbase + '/cpu' + i + '/topology';
    const sib = expandCpuList(cpuRead(tp + '/thread_siblings_list') || String(i));
    coreList.push({ cpu: i, threads: sib.length, core_id: cpuNum(tp + '/core_id'), pkg: cpuNum(tp + '/physical_package_id') });
    if (sib.length > 1) htThreads++; else singleThreads++;
  }
  out.coreList = coreList;
  out.sockets = new Set(coreList.map((c) => c.pkg)).size || 1;
  const htCores = htThreads / 2;
  out.ht_core_count = htCores;                 // 超线程核数（Intel 混合架构 = P 核）
  out.single_thread_core_count = singleThreads; // 单线程核数（Intel 混合架构 = E 核）
  out.hybrid = htThreads > 0 && singleThreads > 0;
  out.physical_cores = htCores + singleThreads;
  // 缓存层级：扫描全部 CPU 的 index*，按「共享域」去重后聚合总量（10-05 EPYC 机实锤只读 cpu0 会把
  // 16 核×16MB=256MB 的 L3 显示成一个域的 16MB——服务器 CPU 每核/每 CCX 一个同级切片）。
  // 域判据：(level,type,size,min(shared_cpu)) 唯一化后计数；size 输出 Σ，slice/domains 留明细。
  const caches = [];
  const cacheAgg = new Map(); // key(level:type) -> {level,type,totalKB,slices:Set,sliceKB,line,repShared}
  const kbOf = (sz) => { const m = String(sz).match(/^(\d+)\s*([KMGT]?B?)$/i); if (!m) return 0; const n = +m[1]; const u = m[2].toUpperCase(); return n * (u.startsWith('G') ? 1048576 : u.startsWith('M') ? 1024 : 1); };
  const humanKB = (kb) => (kb >= 1048576 && kb % 1048576 === 0 ? (kb / 1048576) + 'G' : (kb % 1024 === 0 ? (kb / 1024) + 'M' : kb + 'K'));
  const _cacheDomSeen = new Set();
  for (const ci of cpus) {
    for (let idx = 0; idx < 10; idx++) {
      const dir = sysbase + '/cpu' + ci + '/cache/index' + idx;
      if (!fs.existsSync(dir)) continue;
      const level = cpuNum(dir + '/level'); if (level == null) continue;
      const type = (cpuRead(dir + '/type') || '').trim();
      const szRaw = (cpuRead(dir + '/size') || '').trim();
      const skb = kbOf(szRaw); if (!skb) continue;
      const sharedList = expandCpuList(cpuRead(dir + '/shared_cpu_list') || '');
      const domKey = level + ':' + type + ':' + szRaw + ':' + (sharedList[0] != null ? sharedList[0] : ci);
      if (!_cacheDomSeen.has(domKey)) {
        _cacheDomSeen.add(domKey);
        const ak = level + ':' + type;
        let a = cacheAgg.get(ak);
        if (!a) { a = { level, type, totalKB: 0, sliceKB: skb, domains: 0, line: cpuNum(dir + '/physical_line_size'), repShared: sharedList.length || 1 }; cacheAgg.set(ak, a); }
        a.totalKB += skb; a.domains += 1;
      }
    }
  }
  for (const a of Array.from(cacheAgg.values()).sort((x, y) => x.level - y.level)) {
    caches.push({ level: a.level, type: a.type, size: humanKB(a.totalKB), slice_size: humanKB(a.sliceKB), domains: a.domains, shared_cpus: a.repShared, line: a.line });
  }
  out.caches = caches;
  // 调频：驱动 / governor / EPP / 频率范围 / 睿频开关
  const cf = sysbase + '/cpu0/cpufreq';
  out.cpufreq_driver = (cpuRead(cf + '/scaling_driver') || '').trim() || null;
  out.governor = (cpuRead(cf + '/scaling_governor') || '').trim() || null;
  out.epp = (cpuRead(cf + '/energy_performance_preference') || '').trim() || null;
  const fmin = cpuNum(cf + '/cpuinfo_min_freq'), fmax = cpuNum(cf + '/cpuinfo_max_freq'), fbase = cpuNum(cf + '/base_frequency');
  out.freq_min_mhz = fmin != null ? Math.round(fmin / 1000) : null;
  out.freq_max_mhz = fmax != null ? Math.round(fmax / 1000) : null;
  out.freq_base_mhz = fbase != null ? Math.round(fbase / 1000) : null;
  const noTurbo = cpuNum(sysbase + '/intel_pstate/no_turbo');
  const boostGlob = cpuNum(sysbase + '/cpufreq/boost');
  if (noTurbo != null) out.boost = noTurbo === 0;
  else if (boostGlob != null) out.boost = boostGlob === 1;
  else out.boost = null;
  // 漏洞缓解状态
  out.vulnerabilities = {};
  try {
    for (const f of fs.readdirSync(sysbase + '/vulnerabilities')) {
      out.vulnerabilities[f] = (cpuRead(sysbase + '/vulnerabilities/' + f) || '').trim();
    }
  } catch (e) {}
  // dmidecode 硬件规格（需免密；失败静默降级）
  out.dmi = null;
  try {
    const { execSync } = require('child_process');
    const dm = execSync('sudo -n dmidecode -t processor 2>/dev/null', { timeout: 4000, encoding: 'utf8' });
    const pick = (k) => { const m = dm.match(new RegExp('^\\s*' + k + ':\\s*(.+)$', 'm')); return m ? m[1].trim() : null; };
    if (/Processor Information/i.test(dm)) {
      out.dmi = {
        socket: pick('Socket Designation'), manufacturer: pick('Manufacturer'), version: pick('Version'),
        current_speed: pick('Current Speed'), max_speed: pick('Max Speed'),
        core_count: pick('Core Count'), enabled_cores: pick('Enabled Core Count'),
        thread_count: pick('Thread Count'), l1: pick('L1 Cache'), l2: pick('L2 Cache'), l3: pick('L3 Cache'),
        stepping: pick('Stepping'), upgrade: pick('Upgrade'),
      };
    }
  } catch (e) {}
  return out;
}
function getCpuStatic() {
  const now = Date.now();
  if (!global.__cpuStatic || now - (global.__cpuStaticAt || 0) > 300000) {
    try { global.__cpuStatic = buildCpuStatic(); } catch (e) { global.__cpuStatic = { supported: false, error: e.message }; }
    global.__cpuStaticAt = now;
  }
  return global.__cpuStatic;
}
// [0918] 异步逐核频率刷新（1s 一轮；fs.readFile 走 libuv 线程池，不阻塞主循环）
function refreshCpuFreqs() {
  if (global.__cpuFreqBusy) return;
  global.__cpuFreqBusy = true;
  try {
    if (!global.__cpuIds) global.__cpuIds = fs.readdirSync('/sys/devices/system/cpu').filter((n) => /^cpu\d+$/.test(n)).map((n) => +n.slice(3));
  } catch (e) { global.__cpuIds = []; }
  const ids = global.__cpuIds;
  if (!ids.length) { global.__cpuFreqBusy = false; return; }
  const vals = new Array(ids.length).fill(null);
  let done = 0;
  const finish = () => {
    if (done < ids.length) return;
    let fSum = 0, fN = 0, fMax = null;
    for (const f of vals) { if (f != null) { fSum += f; fN++; if (fMax == null || f > fMax) fMax = f; } }
    global.__cpuFreq = { fSum, fN, fMax, ts: Date.now() };
    global.__cpuFreqBusy = false;
  };
  ids.forEach((i, idx) => {
    fs.readFile('/sys/devices/system/cpu/cpu' + i + '/cpufreq/scaling_cur_freq', 'utf8', (err, data) => {
      if (!err) { const n = parseInt(String(data).trim(), 10); if (Number.isFinite(n)) vals[idx] = n; }
      done++; finish();
    });
  });
}
if (!global.__cpuFreqTimer) {
  global.__cpuFreqTimer = setInterval(refreshCpuFreqs, 1000);
  if (global.__cpuFreqTimer.unref) global.__cpuFreqTimer.unref();
  refreshCpuFreqs();
}

function sampleCpuStats() {
  try {
    const txt = cpuRead('/proc/stat');
    if (!txt) return;
    const prev = global.__cpuPrev || {};
    const cores = [];
    let agg = null;
    for (const line of txt.split('\n')) {
      const m = line.match(/^(cpu\d*)\s/);
      if (!m) { if (cores.length || agg != null) break; else continue; }
      const f = line.trim().split(/\s+/).slice(1).map(Number);
      const idle = f[3] + (f[4] || 0);
      const total = f.reduce((a, b) => a + b, 0);
      const p = prev[m[1]];
      let pct = null;
      if (p && total > p.total) pct = Math.max(0, Math.min(100, (1 - (idle - p.idle) / (total - p.total)) * 100));
      prev[m[1]] = { idle, total };
      if (m[1] === 'cpu') agg = pct;
      else cores.push({ cpu: +m[1].slice(3), pct: pct != null ? Math.round(pct) : null });
    }
    global.__cpuPrev = prev;
    // 封装温度：优先 thermal_zone（x86_pkg_temp/k10temp/...），zone 里找不到偏好类型或缺 zone 时
    // 回落 hwmon（10-05 EPYC 板实锤：/sys/class/thermal 整个为空，k10temp 只在 hwmon2 挂着，
    // 页面封装温度恒 --）。hwmon 侧标签判据：Tctl/Package 优先（整机封装值），Tdie 次之，
    // 避开 k10temp 的逐 CCD 分量 Tccd*（拿它当封装温度是错的口径）。
    let temp = null, tempSrc = null;
    try {
      const prefer = ['x86_pkg_temp', 'k10temp', 'coretemp', 'soc_thermal', 'cpu_thermal', 'acpitz'];
      let best = null;
      for (const z of fs.readdirSync('/sys/class/thermal')) {
        if (!/^thermal_zone\d+$/.test(z)) continue;
        const t = (cpuRead('/sys/class/thermal/' + z + '/type') || '').trim();
        const v = cpuNum('/sys/class/thermal/' + z + '/temp');
        if (v == null) continue;
        const r = prefer.indexOf(t);
        const rank = r >= 0 ? r : 99;
        if (!best || rank < best.rank) best = { temp: v / 1000, type: t || z, rank };
      }
      if (best && best.rank < 99) { temp = +best.temp.toFixed(1); tempSrc = best.type; }
    } catch (e) {}
    if (temp == null) {
      try {
        const preferHw = ['k10temp', 'x86_pkg_temp', 'coretemp', 'zenpower', 'soc_thermal', 'cpu_thermal', 'acpitz'];
        let bh = null;
        for (const hd of fs.readdirSync('/sys/class/hwmon')) {
          if (!/^hwmon\d+$/.test(hd)) continue;
          const nm = (cpuRead('/sys/class/hwmon/' + hd + '/name') || '').trim();
          const rank = preferHw.indexOf(nm);
          if (rank < 0) continue;
          let tf;
          try { tf = fs.readdirSync('/sys/class/hwmon/' + hd).filter((f) => /^temp\d+_input$/.test(f)); } catch (e) { continue; }
          for (const f of tf) {
            const v = cpuNum('/sys/class/hwmon/' + hd + '/' + f);
            if (v == null) continue;
            const lab = (cpuRead('/sys/class/hwmon/' + hd + '/' + f.replace('_input', '_label')) || '').trim();
            let sc; // 同传感器族内择优：封装级 > 无名单传感器 > 其它
            if (/^(Tctl|Package|Physical)/i.test(lab)) sc = 0;
            else if (/^Tdie$/i.test(lab)) sc = 1;
            else if (!lab || /^temp$/i.test(lab)) sc = 2;
            else if (/^Core ?\d+$/i.test(lab)) sc = 3;      // coretemp 无 Package 标签时退逐核
            else if (/^Tccd/i.test(lab)) sc = 9;            // 逐 CCD 分量：垫底不作封装值
            else sc = 4;
            if (!bh || rank < bh.rank || (rank === bh.rank && sc < bh.sc)) bh = { v, nm, lab, rank, sc };
          }
        }
        if (bh) { temp = +(bh.v / 1000).toFixed(1); tempSrc = bh.nm + (bh.lab ? '(' + bh.lab + ')' : ''); }
      } catch (e) {}
    }
    // 各逻辑核当前频率（kHz→MHz，取均值与最大值；混合架构 P/E 频率不同）
    // 09-18 修复：同步读全部核 scaling_cur_freq 需 350~600ms（每核首次打开触发内核慢路径），
    // 在 1s 采样器里会阻塞事件循环、饿死 GPU 卡等所有秒级数据 → 频率改由
    // refreshCpuFreqs() 用 fs.readFile（libuv 线程池）异步刷新到 global.__cpuFreq，这里只取缓存。
    const cf = global.__cpuFreq || { fSum: 0, fN: 0, fMax: null };
    const fSum = cf.fSum, fN = cf.fN, fMax = cf.fMax;
    // 负载
    let load = null;
    try {
      const l = (cpuRead('/proc/loadavg') || '').trim().split(/\s+/);
      const rp = String(l[3] || '').split('/');
      load = { l1: +l[0], l5: +l[1], l15: +l[2], running: +rp[0], procs: +rp[1] };
    } catch (e) {}
    global.__cpuLive = {
      ts: Date.now(),
      util_pct: agg != null ? +agg.toFixed(1) : null,
      cores,
      temp_c: temp, temp_src: tempSrc,
      freq_avg_mhz: fN ? Math.round(fSum / fN / 1000) : null,
      freq_live_max_mhz: fMax != null ? Math.round(fMax / 1000) : null,
      load,
    };
    const H = global.__cpuHistory || (global.__cpuHistory = []);
    if (agg != null) {
      H.push({ t: Math.round(Date.now() / 1000), pct: +agg.toFixed(1) });
      if (H.length > 1800) H.splice(0, H.length - 1800); // 30 分钟
    }
  } catch (e) { /* 采样失败下一轮重试 */ }
}
if (!global.__cpuSampler) {
  global.__cpuSampler = setInterval(sampleCpuStats, 1000);
  if (global.__cpuSampler.unref) global.__cpuSampler.unref();
  sampleCpuStats();
}

// ====== GPU 详情（8889 硬件监视页 GPU 卡）======
// 数据源：nvidia-smi --query-gpu CSV（本机实测 ~33ms/次；驱动 610.43.03 支持 pcie.link.* 字段）。
// 静态规格 5min 缓存（global.__gpuStatic）；动态（利用率/显存/温度/功耗/频率/P状态/PCIe 当前链路）1s 采样（global.__gpuLive）。
// 需求：显示 GPU 当前工作在 PCIe x16 等链路信息，并与链路能力对比（宽度/代数降级检测）。
// ====== GPU 查询统一通道（10-07 测试补全，落实 09-20 铁律）======
// 铁律：绝不同步调 nvidia-smi。驱动 GSP/Falcon 异常时 nvidia-smi 会进 D 状态
// （不可中断睡眠），execSync/execFileSync 会把 Node 事件循环永久按住 → 8889 全站无响应
// （09-20 GSP 引导失败、09-29 开机 GSP 0x31 两次实锤）。
// 关键认知：execFile/execSync 的 timeout 选项靠"给子进程发信号"实现，对 D 状态进程
// 完全无效（SIGKILL 都不生效，回调也永不触发）。所以超时必须由**我们自己的定时器**驱动：
// 到点就 SIGKILL 并立刻放弃等待（哪怕 kill 没生效也不再拖住调用方）。
// 再加熔断：连续失败 N 次进入冷却，期间直接返回 null 不再往驱动上撞，
// 避免每次轮询都新起一个永不退出的 nvidia-smi（09-20 现场累积 43 个 D 状态进程）。
let __gpuQFails = 0;
let __gpuQBlockUntil = 0;
const GPU_Q_FAIL_MAX = 4;                 // 连续失败次数阈值
const GPU_Q_COOLDOWN_MS = 2 * 60 * 1000;  // 熔断冷却 2 分钟（必须明显大于各调用方重试间隔，否则重试永远撞边界）
// 10-07 线上实测口径（别再拍脑袋）：本机冷启动后**第一次** nvidia-smi（NVML 初始化）实测
// 就要 3.3s，load 10+ 时更久；热态仅 0.06~0.5s。原统一 4000ms 会在服务启动瞬间
// （采样器 + 静态预热 + 功耗链同时开火）连吃超时 → 熔断 → 硬件页整块空白。
const GPU_Q_MS_STATIC = 12000; // 静态规格（字段多 + 冷初始化）
const GPU_Q_MS_LIVE = 8000;    // 实时采样 / 功耗 / 显存校验
function gpuQueryAvailable() { return Date.now() >= __gpuQBlockUntil; }
function gpuQFail(why) {
  __gpuQFails++;
  if (__gpuQFails >= GPU_Q_FAIL_MAX) {
    __gpuQFails = 0;
    __gpuQBlockUntil = Date.now() + GPU_Q_COOLDOWN_MS;
    console.warn(`[gpu] nvidia-smi 连续 ${GPU_Q_FAIL_MAX} 次失败（${why}），熔断 ${GPU_Q_COOLDOWN_MS / 1000}s 内不再调用（GPU 采样/静态规格暂缺，其余功能不受影响；冷却后自动恢复）`);
  }
}
// 通用"放弃式"子进程采集：resolve(stdout) 或 resolve(null)（失败/超时/熔断），永不 reject。
// countFail=false 时不计入 GPU 熔断（用于 python3 这类"缺了只是少个字段"的辅助采集）。
function spawnCollect(file, args, ms, countFail) {
  return new Promise((resolve) => {
    let proc = null;
    try { proc = spawn(file, args, { stdio: ['ignore', 'pipe', 'ignore'] }); }
    catch (e) { resolve(null); return; }
    let out = '';
    let done = false;
    const finish = (v, why) => {
      if (done) return;
      done = true;
      clearTimeout(t);
      try { proc.stdout.destroy(); } catch (e) {}
      if (v === null && why && countFail !== false) gpuQFail(why);
      resolve(v);
    };
    const t = setTimeout(() => {
      try { proc.kill('SIGKILL'); } catch (e) {}
      finish(null, `超时 ${ms}ms（疑似 D 状态，已放弃等待）`);
    }, ms || 4000);
    proc.stdout.on('data', (d) => { if (out.length < 4 * 1024 * 1024) out += d; });
    proc.on('error', (e) => finish(null, (e && e.code) || 'spawn 失败'));
    proc.on('close', (code) => { if (code === 0) { __gpuQFails = 0; finish(out); } else finish(null, '退出码 ' + code); });
  });
}
// nvidia-smi 专用通道（带熔断判断）。
// 10-07 线上回归教训：并发调用会把"冷初始化 3.3s"叠加成超时（启动瞬间采样器/静态预热/
// 功耗链/前端 stats 同时开火），故同一时刻只允许一个 nvidia-smi 在飞，其余排队。
let __gpuQChain = Promise.resolve();
function gpuQuery(args, ms) {
  if (!gpuQueryAvailable()) return Promise.resolve(null);
  const run = () => (gpuQueryAvailable() ? spawnCollect('nvidia-smi', args, ms) : Promise.resolve(null));
  __gpuQChain = __gpuQChain.then(run, run);
  return __gpuQChain;
}
async function gpuSmi(fields) {
  const out = await gpuQuery(['--query-gpu=' + fields, '--format=csv,noheader,nounits'], GPU_Q_MS_STATIC);
  if (out == null) throw new Error('nvidia-smi 查询失败（超时 / 驱动不可用 / 熔断冷却中）');
  return out;
}
function gpuParseRows(out) {
  return String(out).split('\n').filter((l) => l.trim().length).map((l) => l.split(',').map((s) => s.trim()));
}
function gpuNum(v) {
  if (v == null) return null;
  const s = String(v);
  if (!/\d/.test(s)) return null; // "[N/A]" 等无数字输入归 null
  const n = parseFloat(s.replace(/[^0-9.\-]/g, ''));
  return Number.isFinite(n) ? n : null;
}
const GPU_STATIC_FIELDS = 'index,name,uuid,pci.bus_id,driver_version,vbios_version,compute_cap,pcie.link.gen.max,pcie.link.width.max,power.limit,power.min_limit,power.max_limit';
const GPU_LIVE_FIELDS = 'index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,temperature.memory,power.draw,pstate,clocks.sm,clocks.max.sm,pcie.link.gen.current,pcie.link.width.current,fan.speed';
// [sm-count 1006 rev3] SM 数量：唯一权威源 = CUDA Driver API cuDeviceGetAttribute(
// CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT=16)——直读驱动当前生效的 SM 拓扑，能反映
// cmpunlocker SM-RECONFIG 解锁（本机 10-03 实测：解锁后该属性=74）。
// 走过的弯路（勿再犯）：① rev1 把 NVML nvmlDeviceGetNumGpuCores 当 SM 数显示 → 它是 CUDA
// 核数（4736）；② rev2 按「cc 查表每 SM 核数」折算 → GA102 每 SM 实为 64 核（74×64=4736，
// 我误按 128 折成 37），且该字段是固件静态表、解锁前后纹丝不动，折算思路根本不成立。
// 采集用 python3 ctypes 直调 libcuda.so.1（不依赖 torch/pynvml），一次进程枚举全部卡输出
// JSON；失败/无 python3 → 返回 {}，前端显示 "--"，绝不让 GPU 静态信息整体构建失败。
// 结果并入 __gpuStatic 的 5min 缓存，无常离子进程负担。
// 注意：cuInit 会建 CUDA 上下文（轻量、毫秒级、不占显存池），只在静态缓存重建时发生（5min 一次）。
async function gpuPyCollect(py) {
  // 同样禁止同步：cuInit/nvmlInit 在驱动异常时也会进 D 状态（与 nvidia-smi 同病），
  // 故走放弃式采集。python3 缺失属环境差异，不计入 GPU 熔断计数。
  const out = await spawnCollect('python3', ['-c', py], GPU_Q_MS_LIVE, false);
  if (!out) return {};
  try {
    const j = JSON.parse(String(out).trim().split('\n').pop());
    return (j && typeof j === 'object') ? j : {};
  } catch (e) { return {}; }
}
async function gpuSmCounts() {
  return gpuPyCollect([
    'import ctypes,json',
    'cu=ctypes.CDLL("libcuda.so.1")',
    'cu.cuInit(0)',
    'n=ctypes.c_int()',
    'cu.cuDeviceGetCount(ctypes.byref(n))',
    'o={}',
    'for i in range(n.value):',
    '    v=ctypes.c_int()',
    '    try:',
    '        if cu.cuDeviceGetAttribute(ctypes.byref(v), 16, i)==0: o[str(i)]=v.value',
    '    except Exception: pass',
    'print(json.dumps(o))',
  ].join('\n'));
}
// NVML 固件静态表的 CUDA 核数（仅作规格参考，不随 SM-RECONFIG 解锁变化）
async function gpuCoresNVML() {
  return gpuPyCollect([
    'import ctypes,json',
    'm=ctypes.CDLL("libnvidia-ml.so.1")',
    'm.nvmlInit_v2()',
    'n=ctypes.c_uint()',
    'm.nvmlDeviceGetCount_v2(ctypes.byref(n))',
    'o={}',
    'for i in range(n.value):',
    '    h=ctypes.c_void_p(); c=ctypes.c_uint()',
    '    try:',
    '        m.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h))',
    '        if m.nvmlDeviceGetNumGpuCores(h, ctypes.byref(c))==0: o[str(i)]=c.value',
    '    except Exception: pass',
    'print(json.dumps(o))',
  ].join('\n'));
}
async function buildGpuStatic() {
  const out = { supported: false, gpus: [], driver_version: null };
  let rows;
  // 10-07 线上回归修复：这里原来 catch 后 return 一个错误帧，于是 gpuStaticRebuild 的
  // .then 会把"空帧"当正常结果覆盖上一帧好数据——一次抖动即整表清空且无自愈。
  // 现改为向上抛，由 gpuStaticRebuild 决定保留旧帧。
  rows = gpuParseRows(await gpuSmi(GPU_STATIC_FIELDS));
  const smMap = await gpuSmCounts(); // [sm-count 1006 rev3] CUDA 属性直读真实 SM 数
  const coreMap = await gpuCoresNVML(); // [sm-count 1006 rev3] NVML 静态 CUDA 核数表
  for (const r of rows) {
    // [index,name,uuid,bus_id,driver,vbios,cc,gen_max,width_max,pl,pl_min,pl_max]
    const g = {
      index: gpuNum(r[0]), name: r[1] || null, uuid: r[2] || null, pci_bus_id: r[3] || null,
      driver_version: r[4] || null, vbios: r[5] || null, compute_cap: r[6] || null,
      gen_max: gpuNum(r[7]), width_max: gpuNum(r[8]),
      power_limit: gpuNum(r[9]), power_min_limit: gpuNum(r[10]), power_max_limit: gpuNum(r[11]),
      // [sm-count 1006 rev3] sm_count = CUDA 属性直读（真实拓扑，含 SM-RECONFIG 解锁）；
      // cuda_cores = NVML 固件静态表（不随解锁变化，仅作规格参考）
      sm_count: smMap[String(gpuNum(r[0]))] != null ? gpuNum(smMap[String(gpuNum(r[0]))]) : null,
      cuda_cores: coreMap[String(gpuNum(r[0]))] != null ? gpuNum(coreMap[String(gpuNum(r[0]))]) : null,
    };
    if (g.index == null) continue;
    out.gpus.push(g);
    if (!out.driver_version) out.driver_version = g.driver_version;
  }
  out.supported = out.gpus.length > 0;
  return out;
}
// 静态规格缓存：调用方（请求处理链、PCIe 拓扑富化）都是同步上下文，绝不能在这里
// 同步等 nvidia-smi（09-20 铁律）。改为"过期即后台重建 + 先回上一帧"：
// 首帧可能为空（前端显示 --），最迟 1s 后的下一帧补上；重建期间单飞不叠加。
let __gpuStaticBuilding = false;
function gpuStaticRebuild() {
  if (__gpuStaticBuilding) return;
  __gpuStaticBuilding = true;
  buildGpuStatic()
    .then((st) => { global.__gpuStatic = st; global.__gpuStaticAt = Date.now(); })
    .catch((e) => {
      // 失败不覆盖上一帧（否则驱动抖动一次，硬件页整表清空），只推迟下次重试；
      // 已有好帧时把原因挂在 stale_error 上，前端可显示"数据可能偏旧"而不是整块消失。
      const msg = String((e && e.message) || e);
      if (global.__gpuStatic && global.__gpuStatic.gpus && global.__gpuStatic.gpus.length) {
        global.__gpuStatic = Object.assign({}, global.__gpuStatic, { stale_error: msg });
      } else {
        global.__gpuStatic = { supported: false, gpus: [], error: msg };
      }
      global.__gpuStaticAt = Date.now();
    })
    .finally(() => { __gpuStaticBuilding = false; });
}
function getGpuStatic() {
  const now = Date.now();
  // 重试间隔 60s，必须明显小于熔断冷却（GPU_Q_COOLDOWN_MS）：两者相等时重试永远落在
  // 冷却边界差几毫秒的位置，每次都撞墙 → 自我维持的永久空白（10-07 线上实锤）。
  if (!global.__gpuStatic || now - (global.__gpuStaticAt || 0) > 60000) gpuStaticRebuild();
  return global.__gpuStatic || { supported: false, gpus: [], driver_version: null, pending: true };
}
function sampleGpuStats() {
  // 09-18：execFileSync 同步阻塞主循环 → 改异步 + 防重入。
  // 10-07：再改走统一放弃式通道。原 execFile 的 timeout 靠发信号实现，对 D 状态进程无效，
  // 回调永不触发 → __gpuSampleBusy 永久停在 true，GPU 实时数据再也不会恢复（静默死掉）。
  // 现在 gpuQuery 必定 settle，且连续失败会进熔断冷却，不再反复往驱动上撞。
  if (global.__gpuSampleBusy) return;
  global.__gpuSampleBusy = true;
  gpuQuery(['--query-gpu=' + GPU_LIVE_FIELDS, '--format=csv,noheader,nounits'], GPU_Q_MS_LIVE).then((outStr) => {
    global.__gpuSampleBusy = false;
    if (!outStr) return; // nvidia-smi 失败保留上次采样，下一轮重试
    try {
      const rows = gpuParseRows(outStr);
    const gpus = [];
    for (const r of rows) {
      // [index,util_gpu,util_mem,mem_used,mem_total,temp,temp_mem,power,pstate,sm_clk,sm_clk_max,gen_cur,width_cur,fan]
      // 09-29 新增 temp_mem（nvidia-smi temperature.memory，HBM/GDDR 显存温度；驱动不支持时该列 N/A → null）
      const g = {
        index: gpuNum(r[0]), util_gpu: gpuNum(r[1]), util_mem: gpuNum(r[2]),
        mem_used: gpuNum(r[3]), mem_total: gpuNum(r[4]), temp: gpuNum(r[5]),
        temp_mem: gpuNum(r[6]),
        power_draw: gpuNum(r[7]), pstate: r[8] || null,
        sm_clock: gpuNum(r[9]), sm_clock_max: gpuNum(r[10]),
        gen_current: gpuNum(r[11]), width_current: gpuNum(r[12]), fan: gpuNum(r[13]),
      };
      if (g.index != null) gpus.push(g);
    }
      global.__gpuLive = { ts: Date.now(), gpus };
    } catch (e) { /* 解析失败保留上次采样 */ }
  });
}
if (!global.__gpuSampler) {
  global.__gpuSampler = setInterval(sampleGpuStats, 1000);
  if (global.__gpuSampler.unref) global.__gpuSampler.unref();
  sampleGpuStats();
  // 静态规格（5min 缓存）改为后台重建：启动即预热一次，避免首个 /v1/internal/gpu
  // 请求撞上空缓存（前端表现为硬件监视页首帧全 "--"）
  gpuStaticRebuild();
}

// —— PCIe 通道带宽占用采样：nvidia-smi dmon -s t（NVML rxpci/txpci，MB/s，本机实测可用）——
// dmon 自身阻塞 ~1s，用异步 execFile + 完成后再排下一轮（不占事件循环、不并发重入）；历史 1800 点 ≈ 36 分钟。
function sampleGpuPcie() {
  // 10-07 测试修复：改走放弃式通道。原 execFile 的 timeout 对 D 状态进程无效（SIGKILL 不生效、
  // 回调永不触发），而本函数靠"回调里续排"维持循环 → 一次驱动抖动就把 PCIe 带宽采样永久打死，
  // 且再也无法自愈（图表停在最后一帧）。gpuQuery 必定 settle，循环得以持续。
  gpuQuery(['dmon', '-s', 't', '-c', '1', '-d', '1'], GPU_Q_MS_LIVE).then((out) => {
    try {
      if (out) {
        const byIndex = {};
        for (const line of String(out).split('\n')) {
          const m = line.trim().match(/^(\d+)\s+(\d+)\s+(\d+)$/);
          if (m) byIndex[+m[1]] = { rx_mbs: +m[2], tx_mbs: +m[3] };
        }
        if (Object.keys(byIndex).length) {
          global.__gpuPcie = { ts: Date.now(), byIndex };
          const H = global.__gpuPcieHistory || (global.__gpuPcieHistory = []);
          H.push({ t: Math.round(Date.now() / 1000), g: byIndex });
          if (H.length > 1800) H.splice(0, H.length - 1800); // ~1.2s/点 ≈ 36 分钟
        }
      }
    } catch (e) { /* 失败保留上次数据，下轮重试 */ }
    const t = setTimeout(sampleGpuPcie, 0); // 显卡卡 1 秒刷新：dmon 自身 ~1s + 立即续排 ≈ 1.2s/轮
    if (t.unref) t.unref();
  });
}
if (!global.__gpuPcieStarted) { global.__gpuPcieStarted = true; sampleGpuPcie(); }


// ====== 内存带宽（8889 硬件监视页 RAM 带宽卡）======
// 数据源：Intel uncore IMC CAS 计数器（本机 E5-2686 v4 / BDW-EP，内核 PMU uncore_imc_0/1/4/5 原生
// 支持 cas_count_read / cas_count_write 别名，每 CAS 行 = 64 B）。perf 通配符 `uncore_imc/` 会
// fan-out 到全部 IMC box 并合并成单行（已实测聚合正确，机器无关，兼容 6-box 的 SKX 等）。
// 采集方式：常驻 `sudo -n perf stat -a -I 1000 -x, ...`（免密 /etc/sudoers.d/dsh-perf，同 dsh-lspci 先例），
// CSV 走 stderr；按 interval 分组结算 bytes/s，1s 一点保留 600 点。进程秒退（sudo 无免权等）自动退避重启。
// 金丝雀验证（2026-09-18）：空载合计 ~1.1GB/s，4 路 memcpy 加压 ~2.3GB/s，方向与量级正确。
// 峰值口径：dmidecode 理论带宽（RAM 卡同源，前端优先使用）；回落 76.8 GB/s = 4×DDR4-2400×8B。
const MEMBW_EVENTS = 'uncore_imc/cas_count_read/,uncore_imc/cas_count_write/';
const MEMBW_PEAK_FALLBACK = 76.8;
function membwUnitMul(u) {
  switch (String(u || '').trim()) {
    case 'KiB': return 1024;
    case 'MiB': return 1048576;
    case 'GiB': return 1073741824;
    case 'B': return 1;
    default: return 64; // 无单位 → 原始 CAS 行数 × 64B
  }
}
function membwHandleLine(line) {
  // perf stat -I 1000 -x, 行格式：elapsed秒,值,单位,事件名,运行ns,占比,,
  const f = line.split(',');
  if (f.length < 4) return;
  const el = parseFloat(f[0]);
  const val = parseFloat(f[1]);
  if (!Number.isFinite(el) || !Number.isFinite(val)) return; // '# started on...' 等杂行
  const ev = f[3] || '';
  const isR = ev.indexOf('cas_count_read') >= 0;
  const isW = ev.indexOf('cas_count_write') >= 0;
  if (!isR && !isW) return;
  const bytes = val * membwUnitMul(f[2]);
  const st = global.__membwSt || (global.__membwSt = { elapsed: null, read: 0, write: 0 });
  if (st.elapsed === null) st.elapsed = el;
  if (el > st.elapsed + 0.05) {
    // interval 切换：结算上一桶
    const dt = el - st.elapsed;
    const rb = st.read, wb = st.write;
    st.elapsed = el; st.read = 0; st.write = 0;
    if (dt > 0.2 && dt < 5) {
      const now = Date.now();
      const sample = { t: Math.round(now / 1000), r: Math.round(rb / dt), w: Math.round(wb / dt) }; // bytes/s
      global.__membw = { ts: now, read_bps: sample.r, write_bps: sample.w };
      const H = global.__membwHistory || (global.__membwHistory = []);
      H.push(sample);
      if (H.length > 600) H.splice(0, H.length - 600);
    }
  }
  if (isR) st.read += bytes; else st.write += bytes;
}
function membwScheduleRestart() {
  if (global.__membwRestartTimer || global.__membwProc) return;
  const delay = (global.__membwFailQuick || 0) >= 4 ? 60000 : 5000;
  global.__membwRestartTimer = setTimeout(() => { global.__membwRestartTimer = null; startMembwSampler(); }, delay);
  if (global.__membwRestartTimer.unref) global.__membwRestartTimer.unref();
}
function startMembwSampler() {
  if (global.__membwProc || global.__membwRestartTimer) return;
  const { spawn } = require('child_process');
  let child = null;
  try {
    child = spawn('sudo', ['-n', '/usr/bin/perf', 'stat', '-a', '-I', '1000', '-x,', '-e', MEMBW_EVENTS, '--', 'sleep', 'infinity'],
      { stdio: ['ignore', 'ignore', 'pipe'] });
  } catch (e) { child = null; }
  if (!child) { membwScheduleRestart(); return; }
  global.__membwProc = child;
  global.__membwSpawnAt = Date.now();
  let buf = '';
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk) => {
    buf += chunk;
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (line) { try { membwHandleLine(line); } catch (e) { /* 单行解析失败丢弃 */ } }
    }
  });
  child.on('error', () => { /* exit 事件会统一处理重启 */ });
  child.on('exit', () => {
    const quick = Date.now() - (global.__membwSpawnAt || 0) < 3000;
    global.__membwFailQuick = quick ? (global.__membwFailQuick || 0) + 1 : 0;
    global.__membwProc = null;
    global.__membwSt = null; // 残留未结算桶作废，重启后重新起算
    membwScheduleRestart();
  });
}
if (!global.__membwStarted) { global.__membwStarted = true; startMembwSampler(); }


// ====== PCIe 通道拓扑 / 占用（8889 硬件监视页 PCIe 卡）======
// 数据源：sudo -n lspci -vv（/etc/sudoers.d/dsh-lspci 免密；解析失败回落普通 lspci -vv）。
// 口径：每个 PCI 桥（根端口）的 LnkSta 当前宽度 = 该链路实际占用的 CPU/PCH 通道数；LnkCap = 板载能力。
// 拓扑变化慢且 lspci -vv 较重，结果缓存 60s（global.__pcieTopoCache）。
function buildPcieTopo() {
  const out = { supported: false, groups: [], error: null };
  const { execSync } = require('child_process');
  let txt;
  try {
    txt = execSync('sudo -n lspci -vv 2>/dev/null || lspci -vv 2>/dev/null', { timeout: 6000, encoding: 'utf8', maxBuffer: 8 * 1024 * 1024 });
  } catch (e) { out.error = 'lspci -vv 读取失败: ' + ((e && e.message) || e); return out; }
  // 1) 切设备块（行首 BDF：兼容带/不带 domain 两种格式）
  const blocks = [];
  let cur = null;
  for (const line of txt.split('\n')) {
    const m = line.match(/^([0-9a-fA-F]{2,4}:[0-9a-fA-F]{2}\.[0-9a-fA-F]) (.+)$/);
    if (m) { cur = { bdf: m[1], desc: m[2].replace(/\s*\(rev [0-9a-f]+\)\s*$/, ''), lines: [] }; blocks.push(cur); }
    else if (cur) cur.lines.push(line);
  }
  const genOf = (spd) => ({ '2.5': 1, '5': 2, '8': 3, '16': 4, '32': 5 }[String(spd).replace('GT/s', '')] || null);
  const busOf = (bdf) => { const p = bdf.split(':'); return p.length === 3 ? p[1] : p[0]; };
  // 2) 逐块抽取链路信息
  const devs = [];
  for (const b of blocks) {
    const body = b.lines.join('\n');
    const sta = body.match(/LnkSta:\s+Speed ([\d.]+GT\/s|\w+)(?: \((\w+)\))?, Width x(\d+)(?: \((\w+)\))?/);
    const cap = body.match(/LnkCap:[^\n]*Speed ([\d.]+GT\/s|\w+), Width x(\d+)/);
    if (!sta && !cap) continue;
    const bus = body.match(/Bus:\s*primary=([0-9a-f]+),\s*secondary=([0-9a-f]+),\s*subordinate=([0-9a-f]+)/);
    const slot = body.match(/Physical Slot:\s*(\S+)/);
    const port = body.match(/LnkCap:[^\n]*Port #(\d+)/);
    devs.push({
      bdf: b.bdf, desc: b.desc, isBridge: /PCI bridge/i.test(b.desc),
      busSec: bus ? bus[2].toLowerCase() : null, busSub: bus ? bus[3].toLowerCase() : null,
      slot: slot ? slot[1] : null, portNo: port ? +port[1] : null,
      spdCur: sta ? sta[1] : null, widCur: sta ? +sta[3] : null,
      spdCap: cap ? cap[1] : null, widCap: cap ? +cap[2] : null,
    });
  }
  // 3) 端点归桥（取总线区间内 secondary 最大的桥 = 最深一级）
  const bridges = devs.filter((d) => d.isBridge && (d.widCap != null || d.widCur != null));
  const endpoints = devs.filter((d) => !d.isBridge && d.bdf !== '00:00.0');
  for (const e of endpoints) {
    const eb = busOf(e.bdf).toLowerCase();
    let best = null;
    for (const br of bridges) {
      if (!br.busSec || !br.busSub) continue;
      if (eb >= br.busSec && eb <= br.busSub && (!best || br.busSec > best.busSec)) best = br;
    }
    if (best) best.device = e;
  }
  // 4) GPU bdf → nvidia-smi 型号名 富化（统一成 lspci 短格式 '03:00.0'，小写）
  const gpuNameByBus = {};
  try {
    for (const g of (getGpuStatic().gpus || [])) {
      if (!g.pci_bus_id || !g.name) continue;
      const parts = String(g.pci_bus_id).toLowerCase().split(':');
      const key = parts.length >= 3 ? parts[parts.length - 2] + ':' + parts[parts.length - 1] : parts.join(':');
      gpuNameByBus[key] = g.name;
    }
  } catch (err) {}
  // 5) 分组（桥 desc 含 chipset → 芯片组；否则 CPU 直连；再兜底 other）
  const gmap = {};
  for (const br of bridges) {
    const key = /chipset/i.test(br.desc) ? 'pch' : 'cpu';
    const grp = gmap[key] || (gmap[key] = { key, label: key === 'cpu' ? 'CPU 直连' : (key === 'pch' ? '芯片组（C610/PCH）' : '其他'), ports: [], used: 0, cap_total: 0 });
    const widUsed = br.widCur != null ? br.widCur : 0;
    const widCap = br.widCap || 0;
    grp.used += widUsed;
    grp.cap_total += widCap;
    const warns = [];
    const gc = genOf(br.spdCur), gcap = genOf(br.spdCap);
    const dev0 = br.device;
    // 有效能力 = min(端口能力, 设备自身能力)——降速到设备/固件上限（如矿卡 Gen1）不算降级
    const effWidCap = (dev0 && dev0.widCap != null && br.widCap != null) ? Math.min(br.widCap, dev0.widCap) : (br.widCap || 0);
    const devGenCap = dev0 && dev0.spdCap ? genOf(dev0.spdCap) : null;
    const effGenCap = Math.min(gcap == null ? 99 : gcap, devGenCap == null ? 99 : devGenCap);
    if (br.widCur != null && br.widCur > 0) {
      if (br.widCur < effWidCap) warns.push('宽度 x' + effWidCap + '→x' + br.widCur);
      else if (dev0 && dev0.widCap != null && br.widCap != null && dev0.widCap > br.widCap) warns.push('设备 x' + dev0.widCap + ' 受端口接线限至 x' + br.widCur);
      if (gc != null && effGenCap < 99 && gc < effGenCap) warns.push('速率 Gen' + effGenCap + '→Gen' + gc);
    }
    let devDesc = null, devBus = null, devSlot = null;
    if (br.device) {
      devBus = br.device.bdf;
      devDesc = br.device.desc;
      devSlot = br.device.slot || br.slot || null;
      const gname = gpuNameByBus[String(devBus).toLowerCase()];
      if (gname) devDesc += '（' + gname + '）';
      // 端点自身能力若高于当前协商/端口接线，附注（如 GPU 卡支持 x16 而实际 x8）
      if (br.device.widCap != null && br.device.widCap > Math.max(br.widCap || 0, widUsed)) devDesc += ' [设备能力 x' + br.device.widCap + ']';
      const dgc = genOf(br.device.spdCap);
      if (br.device.spdCap && dgc != null && gc != null && dgc > Math.max(gcap || 0, gc)) devDesc += ' [设备能力 Gen' + dgc + ']';
    }
    grp.ports.push({
      bdf: br.bdf, port: br.portNo,
      gen_cur: gc, gen_cap: gcap, wid_cur: widUsed, wid_cap: widCap,
      free: Math.max(0, widCap - widUsed),
      active: widUsed > 0,
      warns, device: devBus ? { bdf: devBus, desc: devDesc, slot: devSlot } : null,
    });
  }
  out.groups = ['cpu', 'pch', 'other'].filter((k) => gmap[k]).map((k) => {
    const g = gmap[k];
    g.ports.sort((a, b) => a.bdf.localeCompare(b.bdf));
    return g;
  });
  out.supported = out.groups.length > 0;
  return out;
}
function getPcieTopo() {
  const now = Date.now();
  if (!global.__pcieTopoCache || now - (global.__pcieTopoCacheAt || 0) > 60000) {
    try { global.__pcieTopoCache = buildPcieTopo(); } catch (e) { global.__pcieTopoCache = { supported: false, groups: [], error: e.message }; }
    global.__pcieTopoCacheAt = now;
  }
  return global.__pcieTopoCache;
}


// ====== Server ======

// ====== Server ======
// ====== 09-23 Bench Console（移植自 github.com/polyuij42-del/bench-console v2.2.1，MIT 协议）======
// LLM 推理基准测试台：三种独立模式（单流解码·13类 / 并发档位 / 预填充 TTFT）+ 测试期间每 1s
// 采样 /metrics 的实时监控 + 轮次独立（等排空/冲前缀缓存/salt 加盐）+ 结果落盘 JSON 与 A/B 对比。
// API 挂 /v1/internal/bench/*（POST 自动落入控制台口令拦截口径）；UI 页面 /bench.html，
// 以「基准测试」标签内嵌管理台。被测服务 = 受管实例（动态跟随 config.vllmPort 端口自愈）
// + 可选 bench-services.json（远端引擎 baseUrl / apiKey，schema 同上游 config.services）。
// 铁律合规：本模块零 execSync；所有上游调用均为 fetch + AbortController 带超时。
// 整块包 IIFE：外部已有 sleep/parseMetrics 等同名符号，闭包内自带一份互不冲突。
const BENCH = (function () {
  'use strict';
  const VERSION = '2.3.0';
  const MODES = ['single', 'conc', 'prefill'];
  // 预填充档位边界（10-05 扩档）：最高测到 1M tokens。引擎实际 max_model_len 更小时，
  // 超档位在开测前就跳过并写进事件流——绝不能让一次 400 把整轮测试打死。
  const PF_MAX_TOKENS = 1048576;
  const PF_MIN_TOKENS = 256;
  // 档位标签：与前端 fmtK 同口径（≥1M 记 M、≥1K 记 K，非整除留一位小数）
  function pfLabel(n) {
    n = +n || 0;
    if (n >= 1048576) { const m = n / 1048576; return (m % 1 ? m.toFixed(1) : m) + 'M'; }
    if (n >= 1024) { const k = n / 1024; return (k % 1 ? k.toFixed(1) : k) + 'K'; }
    return String(n);
  }
  const PROMPT_FILES = {
    '13': path.join(__dirname, 'prompts', 'prompts13.json'),
    '6': path.join(__dirname, 'prompts', 'prompts6.json'),
  };
  const RESULT_DIR = path.join(__dirname, 'bench-results');
  try { fs.mkdirSync(RESULT_DIR, { recursive: true }); } catch (e) {}

  // ---------- 被测服务：受管实例（动态）+ bench-services.json（可选扩展，热读） ----------
  function localSvc() {
    const port = config.vllmPort;
    return { id: 'local', port, name: '受管实例 · :' + port, desc: '控制台当前 vLLM 后端（跟随端口自愈）', baseUrl: 'http://127.0.0.1:' + port };
  }
  function services() {
    let extra = [];
    try {
      const j = JSON.parse(fs.readFileSync(path.join(__dirname, 'bench-services.json'), 'utf8'));
      if (Array.isArray(j)) extra = j;
    } catch (e) {}
    extra = extra
      .map((s) => ({ ...s, id: String(s.id != null ? s.id : s.port), port: Number(s.port), name: s.name || (s.port + ' · Engine') }))
      .filter((s) => Number.isFinite(s.port));
    const out = [localSvc()];
    for (const s of extra) if (!out.some((x) => x.id === s.id)) out.push(s);
    return out;
  }
  function svc(k) {
    const key = String(k);
    const list = services();
    return list.find((x) => x.id === key) || list.find((x) => String(x.port) === key);
  }
  function svcOf(s) { return (typeof s === 'string' || typeof s === 'number') ? svc(s) : s; }
  // 带鉴权服务统一由此产出 Authorization 头；apiKey 只在服务端使用，绝不下发浏览器
  function svcKey(s) { const x = svcOf(s); return (x && x.apiKey) ? { 'Authorization': 'Bearer ' + x.apiKey } : {}; }
  function baseUrl(s) { const x = svcOf(s); const b = (x && x.baseUrl) || ('http://127.0.0.1:' + (x ? x.port : '')); return String(b).replace(/\/+$/, ''); }

  // ---------- utils ----------
  function json(res, code, obj) {
    const body = JSON.stringify(obj);
    res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
    res.end(body);
  }
  async function readBody(req) {
    let d = '';
    for await (const c of req) d += c;
    try { return JSON.parse(d || '{}'); } catch (e) { return {}; }
  }
  async function fetchWithTimeout(url, ms, extraHeaders) {
    const ac = new AbortController();
    const t = setTimeout(() => ac.abort(), ms);
    try { return await fetch(url, { signal: ac.signal, headers: extraHeaders || undefined }); } finally { clearTimeout(t); }
  }
  function parseMetrics(text) {
    const out = {};
    for (const m of text.matchAll(/^((?:vllm|sglang):[a-z_0-9]+)\{[^}]*\}\s+([0-9.eE++-]+)$/gm)) {
      const k = m[1], v = parseFloat(m[2]);
      if (!Number.isNaN(v)) out[k] = (out[k] || 0) + v;
    }
    return out;
  }
  async function getMetrics(s) {
    try {
      const r = await fetchWithTimeout(baseUrl(s) + '/metrics', 4000, svcKey(s));
      if (!r.ok) return {};
      return parseMetrics(await r.text());
    } catch (e) { return {}; }
  }
  function metricsDelta(a, b) {
    const d = {};
    for (const k of new Set([...Object.keys(a || {}), ...Object.keys(b || {})])) d[k] = (b[k] || 0) - (a[k] || 0);
    return d;
  }
  function mean(a) { return a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0; }
  function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

  // ---------- 轮次隔离（v2.2）----------
  // 引擎完全排空判定（无在跑、无排队请求；指标缺失时视为空闲）。vLLM 新旧指标名 + SGLang 名都兼容。
  function engineIdle(m) {
    const running = m['vllm:num_requests_running'] ?? m['vllm:num_running_requests'] ?? m['sglang:num_running_requests'];
    const waiting = m['vllm:num_requests_waiting'] ?? m['vllm:num_waiting_requests'] ?? m['sglang:num_queue_reqs'];
    if (running == null && waiting == null) return true;
    return !running && !waiting;
  }
  // 探测缓存冲刷端点：vLLM /reset_prefix_cache、SGLang /flush_cache（带鉴权头）。每个 run 对每个服务只探一次。
  async function probeFlush(s, cap) {
    for (const p of ['/reset_prefix_cache', '/flush_cache']) {
      try {
        const r = await fetchWithTimeout(baseUrl(s) + p, 3000, svcKey(s));
        if (r.ok) { cap.svc = s; cap.path = p; return; } // 探测成功本身就完成了一次冲刷
      } catch (e) {}
    }
    cap.svc = s; cap.path = null;
  }
  // 每轮结束后的隔离动作：冲刷前缀缓存（若支持）→ 等引擎完全排空（最多 20s）→ 轮间静置
  async function roundIsolate(s, state, label) {
    if (!state.flushCap) state.flushCap = { svc: null, path: null };
    const cap = state.flushCap;
    if (cap.svc !== s) {
      await probeFlush(s, cap);
    } else if (cap.path) {
      try { await fetchWithTimeout(baseUrl(s) + cap.path, 3000, svcKey(s)); } catch (e) {}
    }
    state.iso = label + ' · 等引擎排空…';
    const t0 = Date.now();
    while (Date.now() - t0 < 20000) {
      if (state.abort) throw new Error('aborted');
      if (engineIdle(await getMetrics(s))) break;
      await sleep(500);
    }
    const n = (state.repSettle || 0) | 0;
    if (n > 0) {
      state.iso = label + ` · 轮间静置 ${n}s`;
      for (let i = 0; i < n; i++) {
        if (state.abort) throw new Error('aborted');
        await sleep(1000);
      }
    }
    state.iso = null;
  }

  // ---------- streaming chat ----------
  // cb(tokens, elapsedMs)：每 20 个 token 回调一次，供实时监控显示本轮进度
  async function streamChat(s, model, prompt, maxTokens, signal, cb) {
    const t0 = Date.now();
    let ttft = null, tokens = 0;
    const body = JSON.stringify({
      model, messages: [{ role: 'user', content: prompt }],
      max_tokens: maxTokens, temperature: 0, stream: true,
      stream_options: { include_usage: true },
    });
    const res = await fetch(baseUrl(s) + '/v1/chat/completions', {
      method: 'POST', headers: { 'Content-Type': 'application/json', ...svcKey(s) }, body, signal,
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    let usage = null;
    const dec = new TextDecoder();
    let buf = '';
    for await (const chunk of res.body) {
      buf += dec.decode(chunk, { stream: true });
      const lines = buf.split('\n'); buf = lines.pop();
      for (const line of lines) {
        const t = line.trim();
        if (!t.startsWith('data:')) continue;
        const data = t.slice(5).trim();
        if (data === '[DONE]') continue;
        try {
          const jj = JSON.parse(data);
          if (jj.usage) { usage = jj.usage; tokens = usage.completion_tokens || tokens; }
          const dl = jj.choices && jj.choices[0] && jj.choices[0].delta;
          // 思考模型走 delta.reasoning/reasoning_content，正文走 delta.content，都算生成 token（TTFT=首个 token 时间）
          const d = dl && (dl.content || dl.reasoning || dl.reasoning_content);
          if (d) {
            if (ttft === null) ttft = Date.now() - t0;
            tokens++;
            if (cb && tokens % 20 === 0) cb(tokens, Date.now() - t0);
          }
        } catch (e) {}
      }
    }
    const wall = (Date.now() - t0) / 1000;
    return { ttft, tokens, wall, tps: tokens / wall, promptTokens: (usage && usage.prompt_tokens) || 0 };
  }

  // ---------- prefill：走 /v1/completions（无 chat 模板/思考干扰），TTFT ≈ prefill 完成时间 ----------
  async function streamPrefill(s, model, prompt, signal) {
    const t0 = Date.now();
    let ttft = null;
    const body = JSON.stringify({
      model, prompt, max_tokens: 1, temperature: 0, stream: true,
      stream_options: { include_usage: true },
    });
    const res = await fetch(baseUrl(s) + '/v1/completions', {
      method: 'POST', headers: { 'Content-Type': 'application/json', ...svcKey(s) }, body, signal,
    });
    if (!res.ok) {
      // 带上引擎原文（截断）：超上下文 / 显存不足 / 引擎已死 的判因全在这句话里
      let detail = '';
      try { detail = (await res.text()).replace(/\s+/g, ' ').slice(0, 260); } catch (e) {}
      throw new Error('HTTP ' + res.status + (detail ? ' · ' + detail : ''));
    }
    let usage = null;
    const dec = new TextDecoder();
    let buf = '';
    for await (const chunk of res.body) {
      buf += dec.decode(chunk, { stream: true });
      const lines = buf.split('\n'); buf = lines.pop();
      for (const line of lines) {
        const t = line.trim();
        if (!t.startsWith('data:')) continue;
        const data = t.slice(5).trim();
        if (data === '[DONE]') continue;
        try {
          const jj = JSON.parse(data);
          if (jj.usage) usage = jj.usage;
          if (jj.choices && jj.choices.length && ttft === null) ttft = Date.now() - t0;
        } catch (e) {}
      }
    }
    return { ttft, promptTokens: (usage && usage.prompt_tokens) || 0 };
  }

  // ---------- prefill filler（多段不同文本轮换，避免前缀缓存命中干扰） ----------
  const FILLERS = [
    '数据中心机房内，成排的服务器指示灯规律地闪烁，冷却风扇发出低沉而持续的嗡鸣声，运维工程师正在巡检每一台机柜的运行状态并记录温度读数。',
    'The distributed tracing system collected spans from every microservice, revealing latency outliers in the payment pipeline during peak traffic hours.',
    '秋日的阳光穿过办公楼的落地窗，洒在键盘和显示器的边缘，工程师们一边讨论着架构图的细节，一边在白板上画出新的服务边界与调用关系。',
    'Benchmark methodology requires isolating variables: identical prompt sets, fixed token budgets, repeated rounds, and metric deltas sampled before and after each request.',
    '数据库慢查询日志显示，联合索引缺失导致的全表扫描在夜间批处理窗口反复出现，DBA 建议对订单表的时间列增加复合索引并重建统计信息。',
    'Kubernetes 集群的节点压力在流量高峰时段逼近阈值，水平扩缩容策略基于自定义指标触发，新的 Pod 在三十秒内完成调度并接入服务网格。',
    'Long-context inference shifts the bottleneck from decode bandwidth to prefill compute: attention over tens of thousands of tokens dominates time-to-first-token.',
    '缓存命中率的变化往往比吞吐量更早暴露问题：当热数据集超出容量时，逐出率上升，尾延迟随之抬升，告警应在命中率跌破阈值时触发。',
  ];
  // maxChars = 字符上限（按引擎 max_model_len 折算，防顶格档分词后超出被拒）
  function buildPrefillPrompt(targetTokens, variant, ratio, maxChars) {
    let chars = Math.max(Math.round(targetTokens * (ratio || 2.7)), 300);
    if (maxChars > 0 && chars > maxChars) chars = maxChars;
    const header = `请阅读以下材料，读完后输出 OK 即可。\n材料编号 V${variant}：\n`;
    // 1M 档约 280 万字符：原先逐段 while 拼接要三万多次（大字符串反复扩容），改成
    // 「8 段材料按 variant 轮转成一个 unit」再倍增复制，二十次内即到位；
    // 首段随 variant 变化 ⇒ 每轮/每档前缀都不同，前缀缓存照样命不中。
    const start = ((variant % FILLERS.length) + FILLERS.length) % FILLERS.length;
    let unit = '';
    for (let k = 0; k < FILLERS.length; k++) unit += FILLERS[(start + k) % FILLERS.length];
    let body = unit;
    while (body.length < chars) body += body;
    return header + body.slice(0, chars) + '\n（材料结束）';
  }

  // ---------- runner ----------
  let RUN = null; // current/last run
  let RUN_SEQ = 0;

  function addEvent(state, kind, title, text) {
    state.events.push({ ts: new Date().toTimeString().slice(0, 8), kind, title, text });
    if (state.events.length > 300) state.events.shift();
  }

  // 实时采样器（每 1s 采 /metrics，环形缓冲 300 点）
  const LIVE_CAP = 300;
  function liveInit(state) {
    state.live = { t: [], tg: [], pg: [], run: [], wait: [], kv: [], prevTs: 0, prevGen: 0, prevProm: 0, t0: Date.now() };
  }
  async function liveTick(state, s) {
    const L = state.live;
    if (!L) return;
    const now = Date.now();
    const m = await getMetrics(s);
    const gen = m['vllm:generation_tokens_total'] ?? m['sglang:generation_tokens_total'] ?? 0;
    const prom = m['vllm:prompt_tokens_total'] ?? m['sglang:prompt_tokens_total'] ?? 0;
    const kvRaw = m['vllm:kv_cache_usage_perc'] ?? m['sglang:token_usage'];
    const kv = (kvRaw == null ? (m['vllm:gpu_cache_usage_perc'] || 0) : kvRaw) * 100;
    if (!L.prevTs) { L.prevTs = now; L.prevGen = gen; L.prevProm = prom; }
    const dt = (now - L.prevTs) / 1000;
    if (dt >= 0.5) {
      let tg = 0, pg = 0;
      if (gen >= L.prevGen) tg = (gen - L.prevGen) / dt;
      if (prom >= L.prevProm) pg = (prom - L.prevProm) / dt;
      L.t.push(Math.round((now - L.t0) / 1000));
      L.tg.push(+tg.toFixed(1));
      L.pg.push(+pg.toFixed(1));
      L.run.push(m['vllm:num_requests_running'] ?? m['vllm:num_running_requests'] ?? m['sglang:num_running_requests'] ?? 0);
      L.wait.push(m['vllm:num_requests_waiting'] ?? m['vllm:num_waiting_requests'] ?? m['sglang:num_queue_reqs'] ?? 0);
      L.kv.push(+kv.toFixed(1));
      if (L.t.length > LIVE_CAP) { L.t.shift(); L.tg.shift(); L.pg.shift(); L.run.shift(); L.wait.shift(); L.kv.shift(); }
      L.prevTs = now; L.prevGen = gen; L.prevProm = prom;
    }
  }

  function buildFinal(mode, state, repsUsed) {
    if (mode === 'single') {
      const ids = state.order || Object.keys(state.single);
      const rows = ids.map((id) => state.single[id]).filter(Boolean)
        .map((r) => ({ name: r.name, tps: r.meanTps, ttft: r.meanTtft, accept: r.accept }));
      if (!rows.length) return null;
      const tpss = rows.map((r) => r.tps).filter(Boolean);
      if (!tpss.length) return null;
      const sorted = [...rows].sort((a, b) => b.tps - a.tps);
      const st = [...tpss].sort((a, b) => a - b);
      const median = st.length % 2 ? st[(st.length - 1) / 2] : (st[st.length / 2 - 1] + st[st.length / 2]) / 2;
      const accRows = rows.filter((r) => r.accept != null);
      return {
        mode, count: rows.length,
        avg: +mean(tpss).toFixed(1), median: +median.toFixed(1),
        best: sorted[0], worst: sorted[sorted.length - 1],
        spreadPct: +(100 * (sorted[0].tps - sorted[sorted.length - 1].tps) / mean(tpss)).toFixed(1),
        meanTtft: Math.round(mean(rows.map((r) => r.ttft).filter(Boolean))),
        meanAccept: accRows.length ? +mean(accRows.map((r) => r.accept)).toFixed(1) : null,
        repsUsed: repsUsed || null,
        top3: sorted.slice(0, 3), bottom3: sorted.slice(-3).reverse(),
        rows: sorted,
      };
    }
    if (mode === 'conc') {
      const cs = Object.keys(state.conc).map(Number).sort((a, b) => a - b);
      if (!cs.length) return null;
      const rows = cs.map((c) => ({ c, agg: state.conc[c].meanAgg, accept: state.conc[c].meanAccept, wall: +(mean(state.conc[c].reps.map((x) => x.wall))).toFixed(2) }));
      const peak = rows.reduce((a, b) => (b.agg > a.agg ? b : a));
      const base = rows.find((r) => r.c === 1) || rows[0];
      return {
        mode, rows, peak,
        baseC: base.c, baseAgg: base.agg,
        scale: base.agg ? +(peak.agg / base.agg).toFixed(2) : null,
        meanAccept: rows.some((r) => r.accept != null) ? +mean(rows.filter((r) => r.accept != null).map((r) => r.accept)).toFixed(1) : null,
      };
    }
    if (mode === 'prefill') {
      const ks = Object.keys(state.prefill).map(Number).sort((a, b) => a - b);
      if (!ks.length) return null;
      // 成功的档一律进表（哪怕 ptps 罕见地为 0），只有 error 档被摘出去点名——
      // 否则「跑了但没数」的档既不在 rows 也不在 failed，曲线上凭空少一格没人知道为什么
      const failed = ks.filter((k) => state.prefill[k].error);
      const ok = ks.filter((k) => !state.prefill[k].error);
      if (!ok.length) return null;
      const rows = ok.map((k) => ({ len: k, ptps: state.prefill[k].meanPtps, ttft: state.prefill[k].meanTtft, tokens: state.prefill[k].meanPromptTokens }));
      const best = rows.reduce((a, b) => ((b.ptps || 0) > (a.ptps || 0) ? b : a), rows[0]);
      // failed 原样回传：汇总里点名失败档，不静默丢档（否则曲线少两档没人知道为什么）
      return { mode, rows, best, total: ks.length, failed: failed.length ? failed : null };
    }
    return null;
  }

  async function runBench(params) {
    const { model, suite, reps, concLevels, maxTokens, settle, repSettle, tag, prefill } = params;
    const mode = MODES.includes(params.mode) ? params.mode : 'single';
    // 内部一律用服务 id 作身份标识（缺 sid 时回退到 port）
    const _target = svc(params.sid != null ? params.sid : params.port);
    const s = _target ? _target.id : params.port;
    const state = RUN;
    const runAc = new AbortController(); // /api/stop 时立刻中止所有在途请求
    state.runAc = runAc;
    // 轮次独立：默认开启；repSettle=轮间静置秒数（缺省 3）；salt 保证每轮提示词首部唯一
    const roundIso = params.roundIso !== false;
    const isoSettle = Number.isFinite(+repSettle) ? +repSettle : 3;
    state.repSettle = isoSettle;
    state.salt = 'bench ' + Math.random().toString(36).slice(2, 8) + ' ';
    state.flushCap = { svc: null, path: null };
    state.mode = mode;
    state.status = 'running';
    let liveTimer = null;
    try {
      // 1. health（最多等 3 分钟，避免引擎启动中误判就绪）
      state.stage = 'health';
      let healthy = false;
      for (let i = 0; i < 90; i++) {
        try {
          const r = await fetchWithTimeout(baseUrl(s) + '/health', 3000, svcKey(s));
          if (r.ok) { healthy = true; break; }
        } catch (e) {}
        if (state.abort) throw new Error('aborted');
        await sleep(2000);
      }
      if (!healthy) throw new Error('服务未就绪（health 检查未通过）');
      // 1.5 配置快照（09-29）：测试一开始抓 vLLM/模型配置，随结果落盘供后续比对
      state.stage = 'snapshot';
      state.stageNote = '采集配置快照…';
      try { state.env = await captureEnv(s, model); } catch (e) { state.env = { error: String(e.message || e) }; }
      try {
        const brief = envBriefText(state.env);
        addEvent(state, 'type', '📸 配置快照', (brief || '已采集（字段不全或服务远程）') + ' → 结果 JSON.env，历史详情/对比可看');
      } catch (e) {}
      // 2. load prompts（单流/并发需要；预填充不需要）
      let prompts = [];
      if (mode !== 'prefill') {
        try { prompts = JSON.parse(fs.readFileSync(PROMPT_FILES[suite] || PROMPT_FILES['13'], 'utf8')); } catch (e) {}
        if (!prompts.length) throw new Error(`prompt 文件加载失败或为空：${PROMPT_FILES[suite] || PROMPT_FILES['13']}`);
        state.order = prompts.map((p) => p.id);
      }
      // 3. settle
      state.stage = 'settle';
      state.stageNote = `静置 ${settle}s`;
      for (let i = 0; i < settle; i++) {
        if (state.abort) throw new Error('aborted');
        await sleep(1000);
      }
      // 4. warmup（失败自动重试 3 次，引擎刚就绪时可能瞬时拒绝连接）
      state.stage = 'warmup';
      state.stageNote = '预热请求';
      for (let w = 0; w < 3; w++) {
        try { await streamChat(s, model, '你好', 20, runAc.signal); break; } catch (e) {
          if (state.abort) throw new Error('aborted');
          if (w === 2) throw e;
          state.stageNote = `预热失败，重试 ${w + 2}/3…`;
          await sleep(5000);
        }
      }
      // 5. 启动实时采样
      liveInit(state);
      liveTick(state, s).catch(() => {});
      liveTimer = setInterval(() => { liveTick(state, s).catch(() => {}); }, 1000);

      const m0 = await getMetrics(s);

      if (roundIso) addEvent(state, 'type', '🧹 轮次独立模式',
        '每轮之间：等引擎完全排空 + 冲刷前缀缓存（若服务支持）+ 轮间静置 ' + isoSettle + 's；'
        + '每轮提示词加 salt 前缀，前缀/radix 缓存永不命中，各轮互不干扰');

      // 6a. 单流逐类型（仅此模式执行）
      if (mode === 'single') {
        state.single = {};
        for (let ti = 0; ti < prompts.length; ti++) {
          if (state.abort) throw new Error('aborted');
          const p = prompts[ti];
          state.stage = 'single';
          state.stageNote = p.name;
          state.progress = { phase: '单流', cur: ti + 1, total: prompts.length, rep: 0, reps };
          const rec = { name: p.name, reps: [], ttfts: [], running: true };
          state.single[p.id] = rec; // 先挂上，前端实时可见
          for (let r = 0; r < reps; r++) {
            if (state.abort) throw new Error('aborted');
            state.progress.rep = r + 1;
            state.cur = { phase: '单流', name: p.name, rep: r + 1, reps, tokens: 0, t0: Date.now() };
            const before = await getMetrics(s);
            const pmt = roundIso ? state.salt + 'r' + (r + 1) + '\n' + p.prompt : p.prompt;
            const out = await streamChat(s, model, pmt, maxTokens, runAc.signal,
              (tk) => { if (state.cur) state.cur.tokens = tk; });
            const after = await getMetrics(s);
            const d = metricsDelta(before, after);
            const acc = d['vllm:spec_decode_num_accepted_tokens_total'] || 0;
            const dft = d['vllm:spec_decode_num_draft_tokens_total'] || 0;
            rec.reps.push({ tps: +out.tps.toFixed(1), ttft: out.ttft, tokens: out.tokens, wall: +out.wall.toFixed(2), accept: dft ? +(100 * acc / dft).toFixed(1) : null });
            rec.ttfts.push(out.ttft);
            rec.meanTps = +mean(rec.reps.map((x) => x.tps)).toFixed(1);
            rec.meanTtft = Math.round(mean(rec.ttfts));
            if (roundIso) await roundIsolate(s, state, p.name + ' 第' + (r + 1) + '轮后');
          }
          rec.running = false;
          rec.accept = rec.reps.map((x) => x.accept).filter((x) => x !== null).length
            ? +mean(rec.reps.map((x) => x.accept).filter((x) => x !== null)).toFixed(1) : null;
          addEvent(state, 'type', '✔ ' + p.name,
            rec.meanTps + ' tok/s · TTFT ' + rec.meanTtft + 'ms' + (rec.accept != null ? ' · 接受率 ' + rec.accept + '%' : '')
            + ' · ' + reps + '轮 [' + rec.reps.map((x) => x.tps).join(' / ') + ']');
        }
        state.cur = null;
      }

      // 6b. 并发档位（仅此模式执行）
      if (mode === 'conc') {
        state.conc = {};
        const levels = concLevels.filter((c) => c >= 1 && c <= 32);
        for (let li = 0; li < levels.length; li++) {
          const c = levels[li];
          state.stage = 'conc';
          state.stageNote = `并发 c=${c}`;
          state.progress = { phase: '并发', cur: li + 1, total: levels.length, rep: 0, reps };
          const rec = { reps: [], running: true };
          state.conc[c] = rec;
          for (let r = 0; r < reps; r++) {
            if (state.abort) throw new Error('aborted');
            state.progress.rep = r + 1;
            state.cur = { phase: '并发', c, rep: r + 1, reps, tokens: 0, done: 0, total: c, t0: Date.now() };
            const before = await getMetrics(s);
            const t0 = Date.now();
            const ac = new AbortController();
            runAc.signal.addEventListener('abort', () => ac.abort(), { once: true }); // 随时停止联动
            const jobs = [];
            for (let i = 0; i < c; i++) {
              const p = prompts[(i + r * c) % prompts.length];
              const pmt = roundIso ? state.salt + 'r' + (r + 1) + 'j' + i + '\n' + p.prompt : p.prompt;
              jobs.push(streamChat(s, model, pmt, maxTokens, ac.signal,
                (tk) => { if (state.cur) state.cur.tokens = Math.max(state.cur.tokens, tk) + 0; })
                .then((o) => { if (state.cur) state.cur.done++; return o; })
                .catch((e) => { if (state.cur) state.cur.done++; return { err: String(e.message || e) }; }));
            }
            const outs = await Promise.all(jobs);
            const wall = (Date.now() - t0) / 1000;
            const ok = outs.filter((o) => !o.err);
            const totalTokens = ok.reduce((sum, o) => sum + o.tokens, 0);
            const after = await getMetrics(s);
            const d = metricsDelta(before, after);
            const acc = d['vllm:spec_decode_num_accepted_tokens_total'] || 0;
            const dft = d['vllm:spec_decode_num_draft_tokens_total'] || 0;
            rec.reps.push({
              aggTps: wall ? +(totalTokens / wall).toFixed(1) : 0,
              wall: +wall.toFixed(2), ok: ok.length, fail: outs.length - ok.length,
              accept: dft ? +(100 * acc / dft).toFixed(1) : null,
            });
            rec.meanAgg = +mean(rec.reps.map((x) => x.aggTps)).toFixed(1);
            if (roundIso) await roundIsolate(s, state, 'c=' + c + ' 第' + (r + 1) + '轮后');
          }
          rec.running = false;
          rec.meanAccept = rec.reps.map((x) => x.accept).filter((x) => x !== null).length
            ? +mean(rec.reps.map((x) => x.accept).filter((x) => x !== null)).toFixed(1) : null;
          addEvent(state, 'conc', '✔ 并发 c=' + c,
            '聚合 ' + rec.meanAgg + ' tok/s' + (rec.meanAccept != null ? ' · 接受率 ' + rec.meanAccept + '%' : '')
            + ' · ' + reps + '轮 [' + rec.reps.map((x) => x.aggTps).join(' / ') + ']');
        }
        state.cur = null;
      }

      // 6c. 预填充（仅此模式执行；/v1/completions，每轮换 filler 变体规避前缀缓存）
      if (mode === 'prefill' && prefill && prefill.enabled && Array.isArray(prefill.lengths) && prefill.lengths.length) {
        state.prefill = {};
        let ratio = 2.7;
        try {
          const cal = await streamPrefill(s, model, buildPrefillPrompt(1024, 997, ratio), runAc.signal);
          if (cal.promptTokens) ratio = Math.round(1024 * 2.7) / cal.promptTokens;
        } catch (e) {}
        // 档位区间 256 ~ 1M（PF_MIN/PF_MAX），再用配置快照里的引擎真实 max_model_len 收一道。
        // 字符上限留 1.5% 余量：target*ratio 与实际分词数有偏差，且 max_tokens=1 也要占位，
        // 顶格档一旦被判「prompt is too long」就白跑几百秒，宁可少测 1.5%。
        const ctxCap = +((state.env && state.env.engineParams && state.env.engineParams.max_model_len) || 0);
        const charCap = ctxCap > 0 ? Math.floor(ctxCap * 0.985 * ratio) : 0;
        const uniq = Array.from(new Set(prefill.lengths.map(Number)));
        const bad = uniq.filter((n) => !(Number.isFinite(n) && n >= PF_MIN_TOKENS && n <= PF_MAX_TOKENS));
        const want = uniq.filter((n) => Number.isFinite(n) && n >= PF_MIN_TOKENS && n <= PF_MAX_TOKENS).sort((a, b) => a - b);
        const lens = ctxCap > 0 ? want.filter((n) => n <= ctxCap) : want;
        const over = want.filter((n) => lens.indexOf(n) < 0);
        if (bad.length) {
          addEvent(state, 'pf', '⏭ 非法/超 1M 的档位已忽略',
            '档位须在 ' + PF_MIN_TOKENS + ' ~ ' + PF_MAX_TOKENS + '（1M）之间，已忽略：'
            + bad.map((n) => '~' + pfLabel(n)).join(' / ') + '（要测 1M 以上得先提高引擎 max_model_len）');
        }
        if (over.length) {
          addEvent(state, 'pf', '⏭ 超引擎上下文的档位已跳过',
            '引擎 max_model_len=' + ctxCap + ' tok（' + pfLabel(ctxCap) + '），这些档测不了：'
            + over.map((n) => '~' + pfLabel(n)).join(' / ')
            + '。要测更高档位先在启动页把上下文切到 1M（YaRN×4）再重测');
        }
        if (!lens.length) {
          throw new Error('无档位可测：所选 ' + uniq.map((n) => '~' + pfLabel(n)).join(' / ')
            + ' 全部非法（限 ' + PF_MIN_TOKENS + ' ~ ' + PF_MAX_TOKENS + '）或超出引擎 max_model_len（' + (ctxCap || '未知') + ' tok）');
        }
        for (let li = 0; li < lens.length; li++) {
          const target = lens[li];
          const label = pfLabel(target);
          state.stage = 'prefill';
          state.stageNote = `预填充 ~${label} tokens` + (ctxCap > 0 ? `（引擎上限 ${pfLabel(ctxCap)}）` : '');
          state.progress = { phase: '预填充', cur: li + 1, total: lens.length, rep: 0, reps };
          const rec = { reps: [], running: true, target };
          state.prefill[target] = rec;
          for (let r = 0; r < reps; r++) {
            if (state.abort) throw new Error('aborted');
            state.progress.rep = r + 1;
            state.cur = { phase: '预填充', len: target, rep: r + 1, reps, tokens: 0, t0: Date.now() };
            const prompt = buildPrefillPrompt(target, r * 7 + li, ratio, charCap);
            let out;
            try {
              out = await streamPrefill(s, model, prompt, runAc.signal);
            } catch (e) {
              if (state.abort) throw new Error('aborted');
              // 单档失败不再打死整轮：记原因 + 跳过该档剩余轮次，继续测后面的档位
              rec.error = String((e && e.message) || e).slice(0, 400);
              break;
            }
            const pt = out.promptTokens || Math.round(target * 0.9);
            rec.reps.push({
              promptTokens: pt,
              ttft: out.ttft,
              ptps: out.ttft ? +(pt / (out.ttft / 1000)).toFixed(0) : null,
            });
            rec.meanPtps = +(mean(rec.reps.map((x) => x.ptps).filter(Boolean))).toFixed(0);
            rec.meanTtft = Math.round(mean(rec.reps.map((x) => x.ttft).filter(Boolean)));
            rec.meanPromptTokens = Math.round(mean(rec.reps.map((x) => x.promptTokens)));
            if (roundIso) await roundIsolate(s, state, '~' + label + 'tok 第' + (r + 1) + '轮后');
          }
          rec.running = false;
          if (rec.error) {
            addEvent(state, 'pf', '✖ 预填充 ~' + label + ' tok 失败', rec.error + '（该档剩余轮次已跳过，继续后面的档位）');
            continue;
          }
          addEvent(state, 'pf', '✔ 预填充 ~' + label + ' tok',
            rec.meanPtps + ' tok/s · TTFT ' + rec.meanTtft + 'ms · ' + reps + '轮 [' + rec.reps.map((x) => x.ptps).join(' / ') + ']');
        }
        state.cur = null;
      }

      // 7. summary metrics
      const mEnd = await getMetrics(s);
      const dAll = metricsDelta(m0, mEnd);
      state.summary = {
        prefixHit: dAll['vllm:prefix_cache_queries_total'] > 0
          ? +(100 * (dAll['vllm:prefix_cache_hits_total'] || 0) / dAll['vllm:prefix_cache_queries_total']).toFixed(1) : null,
        accept: dAll['vllm:spec_decode_num_draft_tokens_total'] > 0
          ? +(100 * (dAll['vllm:spec_decode_num_accepted_tokens_total'] || 0) / dAll['vllm:spec_decode_num_draft_tokens_total']).toFixed(1) : null,
      };

      // 8. 最终汇总
      state.final = buildFinal(mode, state, reps);
      if (state.final) {
        let head = '';
        if (mode === 'single') head = state.final.count + ' 类均值 ' + state.final.avg + ' tok/s，最快「' + state.final.best.name + '」' + state.final.best.tps + '，最慢「' + state.final.worst.name + '」' + state.final.worst.tps;
        else if (mode === 'conc') head = '峰值 c=' + state.final.peak.c + ' 聚合 ' + state.final.peak.agg + ' tok/s' + (state.final.scale ? '（相对 c' + state.final.baseC + ' ×' + state.final.scale + '）' : '');
        else head = '峰值 ~' + pfLabel(state.final.best.len) + ' tok 档 ' + state.final.best.ptps + ' tok/s'
          + (state.final.failed ? ('（另有 ' + state.final.failed.map((n) => '~' + pfLabel(n)).join(' / ') + ' 档失败）') : '');
        addEvent(state, 'final', '🏁 测试完成', head);
      }

      // 9. save
      state.stage = 'save';
      const file = `${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}_${(tag || 'run')}_${mode}.json`;
      const record = {
        file, tag: tag || 'run', mode, timestamp: new Date().toISOString(),
        service: svc(s) || { port: s, name: String(s) },
        model, params: { suite, reps, concLevels, maxTokens, settle, repSettle: isoSettle, roundIso, prefill: prefill || null },
        single: state.single, conc: state.conc, prefill: state.prefill,
        summary: state.summary, final: state.final, events: state.events,
        order: state.order,
        env: state.env || null, // 09-29 配置快照：vLLM 引擎/模型/启动 env/GPU，比对用
      };
      fs.writeFileSync(path.join(RESULT_DIR, file), JSON.stringify(record, null, 1));
      state.status = 'done';
      state.stage = 'done';
      state.stageNote = '';
      state.file = file;
    } catch (e) {
      state.status = state.abort ? 'aborted' : 'error';
      state.error = String(e.message || e);
      state.stage = state.error;
    } finally {
      if (liveTimer) clearInterval(liveTimer);
      state.cur = null;
      state.progress = {};
      // 中断/出错时清掉「测试中」状态并结算已完成轮次的均值，避免前端永远显示 running
      for (const g of [state.single, state.conc, state.prefill]) {
        for (const k of Object.keys(g || {})) {
          const rec = g[k];
          if (rec && rec.running) {
            rec.running = false;
            if (rec.reps && rec.reps.length) {
              if (rec.meanTps === undefined && rec.reps[0].tps !== undefined) rec.meanTps = +mean(rec.reps.map((x) => x.tps)).toFixed(1);
              if (rec.meanAgg === undefined && rec.reps[0].aggTps !== undefined) rec.meanAgg = +mean(rec.reps.map((x) => x.aggTps)).toFixed(1);
              if (rec.meanPtps === undefined && rec.reps[0].ptps !== undefined) rec.meanPtps = +(mean(rec.reps.map((x) => x.ptps).filter(Boolean))).toFixed(0);
            }
          }
        }
      }
    }
  }

  // ---------- 配置快照（09-29）：每次测试开始抓 vLLM/模型配置，随结果落 bench-results 供后续比对 ----------
  // 数据源（单项失败只缺该项，绝不影响测试本身）：
  //   ① /v1/models + /metrics 原始文本（任意服务 HTTP 可达，含远程）→ 模型名 / vllm:info 版本号
  //   ② /proc/<pid>/cmdline（仅本机受管实例）→ 引擎真实 argv + parseServerParams 结构化参数
  //   ③ 模型目录 config.json / generation_config.json（宿主可见路径，与 model-params 采样兜底同源）
  //   ④ flash-next 启动 env 文件（start wrapper 每次启动落盘 = 弹窗参数权威源，09-26 铁律）
  //   ⑤ GPU 规格/功耗上限：只读 global.__gpuStatic 缓存，绝不现场调 nvidia-smi（09-20 铁律）
  function parseEnvFile(txt) {
    const out = {};
    for (const line of String(txt).split('\n')) {
      const t = line.trim();
      if (!t || t.startsWith('#')) continue;
      const i = t.indexOf('=');
      if (i < 1) continue;
      let v = t.slice(i + 1);
      // start wrapper 用 printf %q 写值：还原 \" \$ \' \\ 转义与 $'...' 包裹
      if (/^\$'/.test(v)) v = v.slice(2, -1);
      v = v.replace(/\\(["'$\\])/g, '$1');
      out[t.slice(0, i)] = v;
    }
    return out;
  }
  function modelPathFromArgv(argv) {
    for (let i = 0; i < argv.length; i++) {
      if ((argv[i] === '--model' || argv[i] === '--model-path') && argv[i + 1]) return argv[i + 1];
      if (argv[i] === 'serve' && argv[i + 1] && !String(argv[i + 1]).startsWith('-')) return argv[i + 1];
    }
    return null;
  }
  function readJsonCap(file, capBytes) {
    try {
      const st = fs.statSync(file);
      if (!st.isFile() || st.size > (capBytes || 512 * 1024)) return null;
      return JSON.parse(fs.readFileSync(file, 'utf8'));
    } catch (e) { return null; }
  }
  async function captureEnv(sid, model) {
    const env = { capturedAt: new Date().toISOString(), model: model || null };
    const x = svcOf(sid);
    // ① /v1/models
    try {
      const r = await fetchWithTimeout(baseUrl(sid) + '/v1/models', 3000, svcKey(sid));
      if (r.ok) {
        const jj = await r.json();
        env.httpModels = (jj.data || []).map((m) => ({ id: m.id, root: m.root || null }));
      }
    } catch (e) {}
    // ② /metrics 原始文本 → vllm:info 版本标签（旧版 vLLM 无该指标则缺省）
    try {
      const r = await fetchWithTimeout(baseUrl(sid) + '/metrics', 4000, svcKey(sid));
      if (r.ok) {
        const mi = (await r.text()).match(/^vllm:info\{([^}]*)\}/m);
        if (mi) {
          for (const kv of mi[1].matchAll(/(\w+)="([^"]*)"/g)) {
            if (kv[1] === 'version') env.vllmVersion = kv[2];
            else if (kv[1] === 'model_name') env.metricsModelName = kv[2];
          }
        }
      }
    } catch (e) {}
    // ③④ 本机受管实例：引擎进程 cmdline + 模型目录 + 启动 env 文件
    if (x && String(x.id) === 'local' && x.port) {
      const pid = findVllmPidByPort(x.port);
      if (pid) {
        try {
          const argv = fs.readFileSync('/proc/' + pid + '/cmdline', 'utf8').split('\0').filter(Boolean);
          const runtime = /sglang/.test(argv.join(' ')) ? 'sglang' : 'vllm';
          env.engine = { pid, runtime, argv };
          try { env.engineParams = parseServerParams(argv, runtime, x.port); } catch (e) {}
          const mp = modelPathFromArgv(argv) || modelPathForPort(x.port);
          if (mp) {
            env.modelPath = mp;
            const cfg = readJsonCap(mp + '/config.json');
            if (cfg) env.modelConfig = cfg;
            const gc = readJsonCap(mp + '/generation_config.json');
            if (gc) env.generationConfig = gc;
          }
          // flash-next 脚本链路：env 文件是该引擎弹窗参数的权威落盘（其它实例无此文件）
          if (/flash[-_]?next/i.test(argv.join(' ')) || /flash[-_]?next/i.test(mp || '')) {
            try {
              const ef = path.join(__dirname, 'flash-next-w4a16-launch.env');
              if (fs.existsSync(ef)) env.launchEnv = parseEnvFile(fs.readFileSync(ef, 'utf8'));
            } catch (e) {}
          }
        } catch (e) { env.engineError = String(e.message || e); }
      } else env.engineError = '未找到引擎进程（远程服务，或进程刚退出）';
    }
    // ⑤ GPU 规格/功耗上限（全局缓存，无子进程调用）
    try {
      const st = global.__gpuStatic;
      if (st && st.supported) {
        env.gpu = {
          driver: st.driver_version || null,
          cards: st.gpus.map((g) => ({ index: g.index, name: g.name, vbios: g.vbios || null, powerLimitW: g.power_limit != null ? g.power_limit : null })),
        };
      }
    } catch (e) {}
    return env;
  }
  // 事件流一行摘要（详情页/事件流都有迹可循，不用点开快照卡）
  function envBriefText(e) {
    if (!e) return '';
    const p = e.engineParams || {};
    const bits = [];
    if (e.vllmVersion) bits.push('vLLM ' + e.vllmVersion);
    if (p.runtime) bits.push(p.runtime);
    if (p.tensor_parallel_size) bits.push('TP' + p.tensor_parallel_size);
    if (p.pipeline_parallel_size > 1) bits.push('PP' + p.pipeline_parallel_size);
    if (p.max_model_len) bits.push('ctx ' + p.max_model_len);
    if (p.block_size) bits.push('block ' + p.block_size);
    if (p.max_num_seqs) bits.push('seqs ' + p.max_num_seqs);
    if (p.speculative_config && p.speculative_config.num_speculative_tokens != null) bits.push('MTP×' + p.speculative_config.num_speculative_tokens);
    if (e.gpu && e.gpu.cards && e.gpu.cards[0] && e.gpu.cards[0].powerLimitW) bits.push(e.gpu.cards[0].powerLimitW + 'W');
    return bits.join(' · ');
  }

  // ---------- API（pathname 前缀 /v1/internal/bench） ----------
  // 对外一律不暴露 apiKey（前端用不到它，鉴权全部在服务端完成）
  function publicSvc(x) { const { apiKey, ...rest } = x; return { ...rest, hasKey: !!apiKey }; }
  async function handleApi(req, res, url) {
    const sub = url.pathname.slice('/v1/internal/bench'.length);
    if (sub === '/config' && req.method === 'GET') {
      return json(res, 200, {
        version: VERSION, embedded: true,
        configFile: fs.existsSync(path.join(__dirname, 'bench-services.json')) ? path.join(__dirname, 'bench-services.json') : null,
        resultsDir: RESULT_DIR,
        promptFiles: PROMPT_FILES,
        services: services().length,
      });
    }
    if (sub === '/services' && req.method === 'GET') {
      const out = [];
      for (const x of services()) {
        let healthy = false, model = null;
        try {
          const r = await fetchWithTimeout(baseUrl(x.id) + '/v1/models', 2500, svcKey(x.id));
          if (r.ok) { const j = await r.json(); model = j.data && j.data[0] && j.data[0].id; healthy = true; }
        } catch (e) {}
        out.push({ ...publicSvc(x), healthy, model });
      }
      return json(res, 200, out);
    }
    if (sub === '/metrics' && req.method === 'GET') {
      const key = url.searchParams.get('sid') || url.searchParams.get('port') || 'local';
      const m = await getMetrics(key);
      const q = m['vllm:prefix_cache_queries_total'] || 0, h = m['vllm:prefix_cache_hits_total'] || 0;
      const dft = m['vllm:spec_decode_num_draft_tokens_total'] || 0, acc = m['vllm:spec_decode_num_accepted_tokens_total'] || 0;
      return json(res, 200, {
        prefixHit: q ? +(100 * h / q).toFixed(1) : null,
        accept: dft ? +(100 * acc / dft).toFixed(1) : null,
      });
    }
    if (sub === '/run' && req.method === 'POST') {
      if (RUN && RUN.status === 'running') return json(res, 409, { error: '已有测试在跑' });
      const b = await readBody(req);
      b.mode = MODES.includes(b.mode) ? b.mode : 'single';
      RUN = {
        runId: ++RUN_SEQ, status: 'init', stage: 'init', stageNote: '', mode: b.mode,
        params: b, abort: false, startedAt: new Date().toISOString(),
        single: {}, conc: {}, prefill: {}, progress: {}, events: [], cur: null, final: null,
      };
      runBench(b); // async
      return json(res, 200, { runId: RUN.runId, mode: b.mode });
    }
    if (sub === '/run' && req.method === 'GET') {
      if (!RUN) return json(res, 200, {});
      const st = { ...RUN };
      delete st.abort;
      delete st.params;
      delete st.runAc;
      // 1s 轮询瘦身：整份 config.json 只随结果文件/详情接口给，轮询只留摘要所需字段
      if (st.env) {
        const { modelConfig, generationConfig, ...lite } = st.env;
        if (modelConfig || generationConfig) lite.hasFullModelConfig = true;
        st.env = lite;
      }
      return json(res, 200, st);
    }
    if (sub === '/stop' && req.method === 'POST') {
      if (RUN) { RUN.abort = true; if (RUN.runAc) RUN.runAc.abort(); } // 随时停止：立刻掐断所有在途流
      return json(res, 200, { ok: true });
    }
    if (sub === '/history' && req.method === 'GET') {
      let files = [];
      try { files = fs.readdirSync(RESULT_DIR).filter((f) => f.endsWith('.json')).sort().reverse(); } catch (e) {}
      const list = [];
      for (const f of files) {
        try {
          const j = JSON.parse(fs.readFileSync(path.join(RESULT_DIR, f), 'utf8'));
          const singleIds = Object.keys(j.single || {});
          const orderIds = j.order && j.order.length ? j.order : singleIds;
          const mArr = singleIds.map((id) => (j.single[id] || {}).meanTps).filter((x) => x);
          const singleMean = mArr.length ? +mean(mArr).toFixed(1) : null;
          const tArr = orderIds.map((id) => (j.single[id] || {}).meanTtft).filter((x) => x != null && x);
          const aArr = orderIds.map((id) => (j.single[id] || {}).accept).filter((x) => x != null);
          const concMeans = {};
          for (const c of Object.keys(j.conc || {})) concMeans[c] = j.conc[c].meanAgg;
          const concPeak = j.final && j.final.mode === 'conc' && j.final.peak ? j.final.peak : null;
          const pfPeak = j.final && j.final.mode === 'prefill' && j.final.best ? j.final.best.ptps : null;
          list.push({
            file: f, tag: j.tag, mode: j.mode || 'full', timestamp: j.timestamp,
            service: j.service && j.service.name, model: j.model,
            hasEnv: !!(j.env && !j.env.error), // 09-29：该记录是否带配置快照
            reps: (j.params && j.params.reps) || null, types: singleIds.length || null,
            singleMean,
            singleTtft: tArr.length ? Math.round(mean(tArr)) : null,
            singleAccept: aArr.length ? +mean(aArr).toFixed(1) : null,
            concMeans, concPeak, pfPeak,
          });
        } catch (e) {}
      }
      return json(res, 200, list);
    }
    if (sub === '/history/read' && req.method === 'GET') {
      const f = url.searchParams.get('f') || '';
      if (!f.endsWith('.json') || f.includes('..')) return json(res, 400, { error: 'bad file' });
      try { return json(res, 200, JSON.parse(fs.readFileSync(path.join(RESULT_DIR, f), 'utf8'))); } catch (e) { return json(res, 404, { error: String(e) }); }
    }
    return json(res, 404, { error: 'not found' });
  }

  return { handleApi, VERSION };
})();

// ====== 09-20 控制台鉴权（可选启用）======
// /home/ll/deploy/console-auth.json 存在且含 token 时启用：所有 /v1/internal/ 的 POST
// （启动/停止模型、重置计费、改功耗等状态变更）必须带 X-Console-Token 头或 ?ct= 参数。
// 删除该文件即回到免鉴权现状。GET 只读接口与 /v1 OpenAI 代理不受影响（不破坏 DSH 等客户端）。
let __authCache = { at: 0, token: null };
function getConsoleToken() {
  const now = Date.now();
  if (now - __authCache.at < 5000) return __authCache.token;
  let tok = null;
  try {
    const j = JSON.parse(fs.readFileSync(path.join(__dirname, 'console-auth.json'), 'utf8'));
    if (j && typeof j.token === 'string' && j.token.length >= 4) tok = j.token;
  } catch (e) {}
  __authCache = { at: now, token: tok };
  return tok;
}

// ====== 09-20 性能优化：文本响应 gzip（仅内部 API/页面/静态，绝不影响代理流式响应）======
function installGzip(req, res) {
  if (!/\bgzip\b/.test(String(req.headers['accept-encoding'] || ''))) return;
  const origWriteHead = res.writeHead.bind(res);
  const origWrite = res.write.bind(res);
  const origEnd = res.end.bind(res);
  let status = 200, reason = null, headers = null;
  const chunks = [];
  res.writeHead = function (code, hOrR, h2) {
    status = code || 200;
    if (typeof hOrR === 'string') { reason = hOrR; headers = h2 || null; }
    else headers = hOrR || null;
    return res;
  };
  res.write = function (c, e, cb) {
    if (c != null && typeof c !== 'function') chunks.push(Buffer.isBuffer(c) ? c : Buffer.from(String(c), (typeof e === 'string' ? e : 'utf8')));
    if (typeof e === 'function') e(); else if (typeof cb === 'function') cb();
    return true;
  };
  res.end = function (c, e, cb) {
    if (typeof c === 'function') { cb = c; c = null; }
    if (c != null) chunks.push(Buffer.isBuffer(c) ? c : Buffer.from(String(c), (typeof e === 'string' ? e : 'utf8')));
    const body = Buffer.concat(chunks);
    const h = Object.assign({}, headers);
    const ct = String(h['Content-Type'] || h['content-type'] || '');
    const textual = /^(text\/|application\/json|application\/javascript|text\/javascript|image\/svg)/.test(ct);
    const send = (payload, extra) => {
      const hh = Object.assign({}, h, extra || {});
      hh['Content-Length'] = Buffer.byteLength(payload);
      try { if (reason) origWriteHead(status, reason, hh); else origWriteHead(status, hh); }
      catch (_) { try { origWriteHead(status); } catch (__) {} }
      origEnd(payload);
      if (typeof cb === 'function') cb();
    };
    if (textual && body.length > 1024) {
      zlib.gzip(body, { level: 5 }, (err, out) => {
        if (err || !out || out.length >= body.length) send(body);
        else send(out, { 'Content-Encoding': 'gzip', 'Vary': 'Accept-Encoding' });
      });
    } else send(body);
    return res;
  };
}

// 09-20：请求入口与采样器 async 化后，任何漏网异常会变成 unhandledRejection 打崩进程；
// 加全局兜底日志（记录但不退出），保证控制台常驻。
process.on('unhandledRejection', (e) => { try { console.error('[unhandledRejection]', (e && e.stack) || e); } catch (_) {} });
process.on('uncaughtException', (e) => { try { console.error('[uncaughtException]', (e && e.stack) || e); } catch (_) {} });

// ==== [strata-console] BEGIN module ====
// ====== Strata 推理引擎管理 API（console-backend，2026-09-30）======
// 契约文档：strata/console/STRATA-API.md（前缀 /v1/internal/strata）。
// 设计铁律（本项目事件循环教训）：
//  · 绝不 execSync 长耗时调用；nvidia-smi / 端口探测一律异步 execFile + 短缓存；
//  · 对引擎（默认 127.0.0.1:8080）的 http.get 一律 2s 超时，失败降级为 {ok:false} 字段，不抛穿；
//  · 进程存活判定走 /proc 直扫 cmdline（不依赖 pid 文件）；spawn 用 setsid + detached + unref，
//    使 Strata 进程组独立于 dsh-console 的 cgroup（控制台重启/停止绝不带崩引擎）。
// 资源门禁（产品要求，非可选项）：Strata 需 35-55GB 专家进 RAM + 大量空闲显存，与 18420
//  生产实例互斥——启动前每卡显存占用必须 < 8GB 且系统可用内存 > 80GB，否则 409 拒绝。
//  · force=true 跳过显存/内存两项（端口占用与已在运行两项永不跳过），并在日志记 OVERRIDE 一笔。
// 进程识别（实测引擎侧事实）：
//  · 主服务进程 cmdline = "<Strata>/.venv/bin/python <Strata>/serve/server.py --engine strata --config ... --port N"
//  · 引擎子进程 cmdline = "<Strata>/engine/strata --serve ..."（由 server.py Popen 拉起，同进程组）
//  · 启动器 setup.py 进程只在加载期短暂存在，不作运行判据。
// 停止 = SIGTERM 整个进程组（负 pgid），最多等 60s，超时不强杀。

// —— 路径常量（env 可覆盖，便于异机复现）——
const STRATA_ROOT   = process.env.STRATA_DIR      || '/media/ll/data/strata/Strata';
const STRATA_DATA   = process.env.STRATA_DATA_DIR || '/media/ll/data/strata/Strata-data';
const STRATA_PY     = process.env.STRATA_PY       || path.join(STRATA_ROOT, '.venv/bin/python');
const STRATA_SETUP  = process.env.STRATA_SETUP    || path.join(STRATA_ROOT, 'setup.py');
const STRATA_CONSOLE_LOG = process.env.STRATA_CONSOLE_LOG || '/media/ll/data/strata/strata-console.log';
const STRATA_ENGINE_BIN  = path.join(STRATA_ROOT, 'engine/strata');   // 引擎二进制名以 setup.py EXE 常量为准
const STRATA_BUILD_JSON  = path.join(STRATA_ROOT, 'engine/BUILD.json');
const STRATA_GATE_VRAM_MB = 8192;    // 每卡占用必须低于此值
const STRATA_GATE_RAM_GB  = 80;      // 系统可用内存必须高于此值
const STRATA_FETCH_TIMEOUT_MS = 2000;

// —— 小工具 ——
function strataFsExists(p) { try { return fs.existsSync(p); } catch (e) { return false; } }
function strataReadJson(p) {
  try { return JSON.parse(fs.readFileSync(p, 'utf8')); } catch (e) { return null; }
}

// 最近一次启动参数（config = Strata/strata-<tag>.json，installed_configs 按 mtime 倒序取第一份）
function strataLatestConfig() {
  try {
    const files = fs.readdirSync(STRATA_ROOT)
      .filter((f) => /^strata-.*\.json$/.test(f))
      .map((f) => { const p = path.join(STRATA_ROOT, f); let mt = 0; try { mt = fs.statSync(p).mtimeMs; } catch (e) {} return { p, mt }; })
      .sort((a, b) => b.mt - a.mt);
    for (const c of files) { const j = strataReadJson(c.p); if (j && j.exe) { j.__path = c.p; return j; } }
  } catch (e) { /* 目录不存在=未安装 */ }
  return null;
}

// —— 引擎 args 工具（flag 值读写；setup.py 快路径不吃命令行覆盖，参数调整全靠改写 cfg JSON）——
function strataArgVal(argsArr, flag) {
  const i = argsArr.indexOf(flag);
  return i >= 0 && i + 1 < argsArr.length ? argsArr[i + 1] : null;
}
// 设置 flag：已有则替换值；没有则插入到 anchorFlag 之前（无 anchor 则追加尾部）
function strataArgSet(argsArr, flag, val, anchorFlag) {
  const v = String(val);
  const i = argsArr.indexOf(flag);
  if (i >= 0 && i + 1 < argsArr.length) { argsArr[i + 1] = v; return argsArr; }
  const at = anchorFlag ? argsArr.indexOf(anchorFlag) : -1;
  if (at >= 0) argsArr.splice(at, 0, flag, v);
  else argsArr.push(flag, v);
  return argsArr;
}
function strataArgRemove(argsArr, flag) {
  const i = argsArr.indexOf(flag);
  if (i >= 0) argsArr.splice(i, flag === '--mmap-experts' ? 1 : 2);   // --mmap-experts 是无值开关
  return argsArr;
}

// 按 start 请求体改写并落盘 cfg（幂等：目标态写入；同文件备份只留最新一份 .bak-console）
// 返回 {changed, backup, effective} 或抛 Error（写失败=拒绝带着不确定参数启动）
function strataApplyTuning(cfgPath, cfg, t) {
  // [STRICT_8G_PATCH_V1] strict_8g（用户红线：只用 GPU0 的 8GB、GPU1 禁用）归一化。
  // 引擎语义实锤：--vram-reserve-mib 是"预留"（缓存吃 free−reserve，设大反而缓存更大）；
  // --expert-cache auto/0 都会吃满（0 实测被当 auto：24576 slots/33.02GiB）；只有显式正整数被尊重
  // （200 → 205 slots/0.28GiB 峰值 5716MiB；300 + 32K ctx → 6260MiB）。
  // 固定 reserve=1024、expert_cache=300、gpu=0 ⇒ serve 端 CUDA_VISIBLE_DEVICES=0 ⇒ GPU1 不可见。
  if (t && t.strict_8g === true) {
    t = Object.assign({}, t, { vram_mb: 1024, expert_cache: 300, gpu: 0 });
    if (t.kv_resident == null && t.max_context != null && t.max_context >= 65536) t.kv_resident = 20480;   // [KV_STREAM_PATCH_V1]
  }
  const before = JSON.stringify(cfg);
  const errs = [];
  const args = Array.isArray(cfg.args) ? cfg.args.slice() : [];
  if (t.vram_mb != null) {
    // 上限=每卡实际显存（170HX=65536MiB）；预留值不应超过整卡显存
    if (!(Number.isInteger(t.vram_mb) && t.vram_mb >= 256 && t.vram_mb <= 65535)) errs.push('vram_mb 需为 256~65535 的整数（MiB）');
    else strataArgSet(args, '--vram-reserve-mib', t.vram_mb, '--max-context');
  }
  if (t.expert_cache != null) {
    // [STRICT_8G_PATCH_V1] 0 必须拒绝：引擎 `--expert-cache 0` 实测被当 auto 吃满显存
    const ecOk = t.expert_cache === 'auto' || (Number.isInteger(t.expert_cache) && t.expert_cache >= 1);
    if (t.expert_cache === 0) errs.push('expert_cache 不能为 0（引擎会当 auto 吃满显存），严格模式请用 300');
    else if (!ecOk) errs.push('expert_cache 需为 "auto" 或 ≥1 的整数');
    else strataArgSet(args, '--expert-cache', t.expert_cache, '--max-context');
  }
  if (t.mmap_experts === true) { if (!args.includes('--mmap-experts')) args.push('--mmap-experts'); }
  else if (t.mmap_experts === false) { strataArgRemove(args, '--mmap-experts'); }
  if (t.max_context != null) {
    const mc = parseInt(t.max_context, 10);
    if (!isFinite(mc) || mc < 1024 || mc > 1048576) errs.push('max_context 需为 1024~1048576');
    else strataArgSet(args, '--max-context', mc);
  }
  if (t.kv_resident != null) {   // [KV_STREAM_PATCH_V1] KV 流式：整个 K/V 进 pinned RAM，VRAM 每层只留 N cells
    const kr = parseInt(t.kv_resident, 10);
    if (!isFinite(kr) || kr < 20480 || kr > 1048576) errs.push('kv_resident 需为 20480~1048576（引擎下限 20480）');
    else strataArgSet(args, '--kv-resident', kr, '--max-context');
  }
  if (t.gpu != null) {
    // [DUAL_170HX_PATCH_V1] gpu 可为 int（单卡）或 "0,1"/[0,1]（多卡分层 PP — 引擎 layer split，见 docs/MULTI_GPU.md）
    const raw = Array.isArray(t.gpu) ? t.gpu : String(t.gpu).split(',');
    const list = raw.map((x) => parseInt(x, 10)).filter((x) => Number.isInteger(x) && x >= 0 && x <= 15);
    if (!list.length) errs.push('gpus 需为合法卡号（单卡 0；双卡 "0,1"）');
    else if (list.length === 1) {
      cfg.gpu = list[0];             // 单卡=int；serve 端 CUDA_VISIBLE_DEVICES 按 gpu_list 收敛
      delete cfg.layer_split;        // 多卡遗留项
    } else {
      cfg.gpu = list;                // 多卡=列表；serve 端自动追加 --layer-split
      cfg.layer_split = t.layer_split || cfg.layer_split || 'auto';
    }
    cfg.gpus_asked = true;           // 保持 true：offer_together 见标志早退，--yes 下不会再把 cfg 改回双卡列表
  }
  if (t.prefill != null) {   // [DUAL_170HX_PATCH_V1] 8GB 卡双卡必需：提示路径缓冲随 chunk 线性增长（160+chunk*680/1024 MiB）
    const pf = String(t.prefill) === 'auto' ? 'auto' : parseInt(t.prefill, 10);
    if (pf !== 'auto' && (!isFinite(pf) || pf < 128 || pf > 8192)) errs.push('prefill 需为 auto 或 128~8192');
    else if (pf === 'auto') strataArgSet(args, '--prefill', 'auto', '--max-context');
    else strataArgSet(args, '--prefill', pf, '--max-context');
  }
  if (errs.length) { const e = new Error(errs.join('；')); e.userFacing = true; throw e; }
  cfg.args = args;
  const after = JSON.stringify(cfg);
  const changed = before !== after;   // 含 gpu/layer_split/gpus_asked 字段增删，杜绝「只删不改写回」漏网
  let backup = null;
  if (changed) {
    try {
      backup = cfgPath + '.bak-console';
      fs.copyFileSync(cfgPath, backup);   // 同文件只留最新一份（固定名覆盖，防堆积）
      const out = Object.assign({}, cfg);
      delete out.__path;
      fs.writeFileSync(cfgPath, JSON.stringify(out, null, 1), 'utf8');   // [STRICT_8G_PATCH_V1] 修 encoding= 赋值式误写
      console.log('[strata] cfg 已改写并备份：' + cfgPath + '（backup=' + backup + '）');
    } catch (e) {
      const err = new Error('改写配置失败：' + String((e && e.message) || e));
      throw err;
    }
  }
  return { changed, backup, effective: strataEffectiveArgs(cfg) };
}

// 生效参数摘要（status.effective_args / 前端展示）：从 args+cfg 提炼关键项
function strataEffectiveArgs(cfg) {
  if (!cfg) return null;
  const a = Array.isArray(cfg.args) ? cfg.args : [];
  return {
    vram_reserve_mib: strataArgVal(a, '--vram-reserve-mib') != null ? parseInt(strataArgVal(a, '--vram-reserve-mib'), 10) : null,
    expert_cache: strataArgVal(a, '--expert-cache'),
    mmap_experts: a.includes('--mmap-experts'),
    max_context: strataArgVal(a, '--max-context') != null ? parseInt(strataArgVal(a, '--max-context'), 10) : null,
    gpu: cfg.gpu === undefined ? null : cfg.gpu,
    kv: strataArgVal(a, '--kv'),
    kv_resident: strataArgVal(a, '--kv-resident') != null ? parseInt(strataArgVal(a, '--kv-resident'), 10) : null,   // [KV_STREAM_PATCH_V1]
    spec: strataArgVal(a, '--spec'),
    prefill: strataArgVal(a, '--prefill'),
  };
}

// 从配置提炼契约要求的 config 摘要（model/context/vision/host/port + 派生项）
function strataConfigBrief(cfg) {
  if (!cfg) return null;
  const a = Array.isArray(cfg.args) ? cfg.args : [];
  const argVal = (flag) => { const i = a.indexOf(flag); return i >= 0 && i + 1 < a.length ? a[i + 1] : null; };
  let model = null;
  try {
    const native = argVal('--native') || '';
    const m = native.match(/-(Q2_0|IQ2_XS|IQ3_XXS|IQ3_S|Q2_K|Q3_K|IQ2_M|IQ3_M)[-_]/i) || native.match(/\/([A-Za-z0-9_]+)\//);
    if (m) model = m[1].toUpperCase();
    if (!model) model = (String(cfg.model_name || '').split('-').pop() || '').toUpperCase() || null;
  } catch (e) {}
  return {
    model,
    model_name: cfg.model_name || null,
    context: parseInt(argVal('--max-context') || '0', 10) || null,
    vision: a.includes('--vision') ? 'gpu' : 'none',
    kv: argVal('--kv'),
    spec: argVal('--spec'),
    port: cfg.port || 8080,
    host: cfg.host || '127.0.0.1',
    api_key_set: !!cfg.api_key,
    exe: cfg.exe || null,
    log: cfg.log || null,
    config_path: cfg.__path || null,
    gpu: cfg.gpu === undefined ? null : cfg.gpu,
    effective: strataEffectiveArgs(cfg),
  };
}

// /proc 直扫找 Strata 主服务进程（serve/server.py --engine strata）。
// 排除 ssh/bash/pgrep/grep/ps 探测类进程（沿用本项目 resolveBackendPort 过滤规则）。
function strataScanMainProc() {
  let names = [];
  try { names = fs.readdirSync('/proc'); } catch (e) { return null; }
  const cands = [];
  for (const name of names) {
    if (!/^\d+$/.test(name)) continue;
    let cmd;
    try { cmd = fs.readFileSync(`/proc/${name}/cmdline`, 'utf8'); } catch (e) { continue; }
    if (!cmd) continue;
    const flat = cmd.split('\0').join(' ');
    if (!cmd.includes('serve/server.py') || !flat.includes('--engine strata')) continue;
    if (/(^|\s)(ssh|bash|sh|expect)(\s|$)/.test(flat)) continue;
    if (/\b(pgrep|grep|ps)\b/.test(flat)) continue;
    const pm = flat.match(/--port[\s=]+(\d+)/);
    let stat = '';
    try { stat = fs.readFileSync(`/proc/${name}/stat`, 'utf8'); } catch (e) {}
    const state = (stat.match(/\)\s+(\S)/) || [])[1] || '';
    if (state === 'Z') continue;   // 僵尸进程绝不算存活（本项目 stop 脚本铁律）
    let starttimeTicks = 0;
    // comm 可含空格：剥到最后一个 ')' 再切（fields[19]=starttime ticks）
    try { starttimeTicks = parseInt(stat.slice(stat.lastIndexOf(')') + 2).split(/\s+/)[19], 10) || 0; } catch (e) {}
    let uid = null;
    try { uid = fs.statSync(`/proc/${name}`).uid; } catch (e) {}
    // 多个 Strata 实例并存时取 pid 最大者（最后启动）
    cands.push({
      pid: parseInt(name, 10),
      port: pm ? parseInt(pm[1], 10) : null,
      state,
      starttime_ticks: starttimeTicks,
      uid,
      cmd: flat.slice(0, 400),
    });
  }
  if (!cands.length) return null;
  cands.sort((a, b) => b.pid - a.pid);
  return cands[0];
}

// /proc 扫引擎子进程（engine/strata --serve），仅作辅助展示/取证
function strataScanEngineProcs() {
  const out = [];
  let names = [];
  try { names = fs.readdirSync('/proc'); } catch (e) { return out; }
  for (const name of names) {
    if (!/^\d+$/.test(name)) continue;
    let cmd;
    try { cmd = fs.readFileSync(`/proc/${name}/cmdline`, 'utf8'); } catch (e) { continue; }
    if (!cmd || !cmd.includes(STRATA_ENGINE_BIN) || !cmd.split('\0').includes('--serve')) continue;
    out.push(parseInt(name, 10));
  }
  return out;
}

// 进程启动时刻（epoch ms）：btime + starttime/HZ，不依赖 pid 文件
function strataProcStartEpochMs(pid) {
  try {
    const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf8');
    // 铁律：comm 可含空格/括号，必须先剥到最后一个 ')' 再切字段（fields[19]=starttime，即整行第 22 字段）
    const fields = stat.slice(stat.lastIndexOf(')') + 2).split(/\s+/);
    const startTicks = parseInt(fields[19], 10);
    if (!isFinite(startTicks)) return null;
    const uptime = parseFloat(fs.readFileSync('/proc/uptime', 'utf8').split(' ')[0]);
    if (!isFinite(uptime)) return null;
    return Date.now() - Math.round(uptime * 1000) + Math.round((startTicks / 100) * 1000);
  } catch (e) { return null; }
}

// —— 引擎 HTTP 转发（http.get + 2s 超时，失败 resolve null，绝不抛穿）——
function strataHttpGetJson(port, pathName, timeoutMs) {
  return new Promise((resolve) => {
    let settled = false;
    const done = (v) => { if (!settled) { settled = true; resolve(v); } };
    let req;
    try {
      req = http.get({ host: '127.0.0.1', port, path: pathName, timeout: timeoutMs || STRATA_FETCH_TIMEOUT_MS }, (res) => {
        let body = '';
        res.setEncoding('utf8');
        res.on('data', (c) => { body += c; if (body.length > 4 * 1024 * 1024) { req.destroy(); } });
        res.on('end', () => {
          if (res.statusCode !== 200) return done(null);
          try { done(JSON.parse(body)); } catch (e) { done({ __raw: body.slice(0, 2048) }); }
        });
      });
    } catch (e) { return done(null); }
    req.on('timeout', () => { try { req.destroy(); } catch (e) {} done(null); });
    req.on('error', () => done(null));
  });
}

// —— 资源门禁：GPU 显存（异步 execFile，2s 缓存 + 单飞）。
// 竞态铁律（09-30 实锤）：过期时绝不可在 execFile 完成前 resolve(缓存)——首发调用会拿到
// null（调用方 .ok 直接炸 TypeError）。正解=每次都 await 本轮刷新 Promise，缓存只作回填源。
let __strataGpuCache = { at: 0, data: null };
let __strataGpuFly = null;
function strataGpuMemUsed() {
  const now = Date.now();
  if (__strataGpuCache.data && now - __strataGpuCache.at <= 2000) return Promise.resolve(__strataGpuCache.data);
  if (__strataGpuFly) return __strataGpuFly;
  __strataGpuFly = new Promise((resolve) => {
    // 10-07 测试修复：原 execFileAsync 的 timeout 靠发信号实现，对 D 状态 nvidia-smi 无效
    // → 该 Promise 永不 settle → 单飞句柄 __strataGpuFly 永久挂住，之后所有调用都 await
    // 同一个死 Promise，启动资源门禁彻底卡死。改走 gpuQuery（自建定时器放弃，必定 settle）。
    gpuQuery(['--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], GPU_Q_MS_LIVE)
      .then((out) => {
        const gpus = [];
        if (out) String(out).split('\n').forEach((line) => {
          const p = line.split(',').map((x) => x.trim());
          if (p.length >= 2 && /^\d+$/.test(p[0])) gpus.push({ index: parseInt(p[0], 10), used_mb: parseInt(p[1], 10) || 0 });
        });
        __strataGpuCache = { at: Date.now(), data: { ok: gpus.length > 0, gpus } };
      })
      .catch(() => { __strataGpuCache = { at: Date.now(), data: { ok: false, gpus: [] } }; })
      .then(() => { __strataGpuFly = null; resolve(__strataGpuCache.data); });
  });
  return __strataGpuFly;
}

// —— 资源门禁：系统可用内存（/proc/meminfo 直读，2s 缓存）——
let __strataRamCache = { at: 0, avail_kb: 0 };
function strataAvailMemKB() {
  const now = Date.now();
  if (now - __strataRamCache.at > 2000) {
    try {
      const txt = fs.readFileSync('/proc/meminfo', 'utf8');
      const m = txt.match(/^MemAvailable:\s+(\d+)\s*kB/m);
      __strataRamCache = { at: now, avail_kb: m ? parseInt(m[1], 10) : 0 };
    } catch (e) { __strataRamCache = { at: now, avail_kb: 0 }; }
  }
  return __strataRamCache.avail_kb;
}

// —— 资源门禁：端口占用探测（异步 execFile ss，60s 负缓存）。
// 缓存只缓存「空闲=false」：true（被占用）必须每次现查——Strata 刚停 60s 内重开是常态，
// 缓存忙碌态会误报「端口已被监听」。
const __strataPortCache = new Map();   // port -> {at, listening}
function strataPortListening(port) {
  return new Promise((resolve) => {
    const hit = __strataPortCache.get(port);
    if (hit && hit.listening === false && Date.now() - hit.at < 60000) return resolve(false);
    execFileAsync('ss', ['-ltnH', `( sport = :${port} )`], { timeout: 3000, encoding: 'utf8' })
      .then((out) => {
        const listening = String(out).split('\n').some((l) => l.trim() && l.includes(':' + port));
        if (!listening) __strataPortCache.set(port, { at: Date.now(), listening: false });
        resolve(listening);
      })
      .catch(() => resolve(false));   // ss 不可用：不缓存、不误拦（spawn 后端口真被占 setup.py 自会失败并落日志）
  });
}

// —— 安装/就绪度（status 与 options 共用）——
function strataInstalled() { return strataFsExists(STRATA_SETUP); }
function strataEngineReady() {
  const meta = strataReadJson(STRATA_BUILD_JSON);
  return { ready: !!meta && strataFsExists(STRATA_ENGINE_BIN), build: meta };
}
function strataModelDirs() {
  // 扫 Strata-data/models/*/：目录内所有 .gguf 都有同名 .done → ready；gb=分片实际字节和
  const out = [];
  let dirs = [];
  try { dirs = fs.readdirSync(path.join(STRATA_DATA, 'models')); } catch (e) { return out; }
  for (const d of dirs) {
    const dp = path.join(STRATA_DATA, 'models', d);
    let st;
    try { st = fs.statSync(dp); } catch (e) { continue; }
    if (!st.isDirectory()) continue;
    let files = [];
    try { files = fs.readdirSync(dp); } catch (e) { continue; }
    const shards = files.filter((f) => f.endsWith('.gguf'));
    let bytes = 0, doneCnt = 0;
    for (const s of shards) {
      try { bytes += fs.statSync(path.join(dp, s)).size; } catch (e) {}
      if (files.includes(s + '.done')) doneCnt++;
    }
    out.push({ name: d, gb: parseFloat((bytes / 1e9).toFixed(1)), shards: shards.length, ready: shards.length > 0 && doneCnt === shards.length });
  }
  return out;
}
function strataModelReady() { return strataModelDirs().some((m) => m.ready); }

// 上次启动失败摘要：控制台日志尾部若有 error/Traceback 迹象，取最后一行非空错误行。
// 注意 statSync 必须在 try 内——文件不存在（从未启动）时直接返回 null，不得抛穿。
function strataLastError() {
  let sz = 0;
  try { sz = fs.statSync(STRATA_CONSOLE_LOG).size; } catch (e) { return null; }
  try {
    if (sz === 0) return null;
    const fd = fs.openSync(STRATA_CONSOLE_LOG, 'r');
    try {
      const len = Math.min(sz, 64 * 1024);
      const buf = Buffer.alloc(len);
      fs.readSync(fd, buf, 0, len, sz - len);
      const lines = buf.toString('utf8').split('\n').map((l) => l.trim()).filter(Boolean);
      const bad = lines.slice(-80).reverse().find((l) => /error|Error|ERROR|Traceback|Exception|refused|failed/.test(l));
      return bad ? bad.slice(0, 300) : null;
    } finally { fs.closeSync(fd); }
  } catch (e) { return null; }
}

// —— status 短缓存（1s TTL + 单飞，防前端轮询叠发；引擎 fetch 只在缓存过期时真跑）——
let __strataStatusCache = null;
let __strataStatusFly = null;
async function strataStatusCompute() {
  const eng = strataEngineReady();
  const models = strataModelDirs();
  const cfg = strataLatestConfig();
  const proc = strataScanMainProc();
  const running = !!proc;
  const port = (proc && proc.port) || (cfg && cfg.port) || 8080;
  const host = (cfg && cfg.host) || '127.0.0.1';
  let health = { ok: false };
  let activity = { phase: running ? 'unknown' : 'idle' };
  if (running) {
    const [h, s] = await Promise.all([strataHttpGetJson(port, '/health'), strataHttpGetJson(port, '/status')]);
    if (h && !h.__raw) health = { ok: true, status: h.status, max_context: h.max_context, model: h.model, images: h.images, api_key: h.api_key };
    else if (h && h.__raw) health = { ok: true, raw: h.__raw.slice(0, 512) };
    if (s && !s.__raw) {
      const busy = !!s.busy;
      activity = {
        phase: busy ? (s.first_token ? 'generating' : 'reading') : 'idle',
        queued: s.queued || 0,
        elapsed_s: s.elapsed_s != null ? s.elapsed_s : null,
        tokens_per_s: s.tokens_per_s != null ? s.tokens_per_s : null,
        tokens_per_s_mean: s.tokens_per_s_mean != null ? s.tokens_per_s_mean : null,
        raw: s,
      };
    }
  }
  const startedAtMs = running ? strataProcStartEpochMs(proc.pid) : null;
  return {
    installed: strataInstalled(),
    engine_ready: eng.ready,
    model_ready: models.some((m) => m.ready),
    running,
    pid: running ? proc.pid : null,
    engine_pids: running ? strataScanEngineProcs() : [],
    port,
    host,
    api_key_set: !!(cfg && cfg.api_key),
    started_at: startedAtMs ? new Date(startedAtMs).toISOString() : null,
    uptime_s: startedAtMs ? Math.max(0, Math.round((Date.now() - startedAtMs) / 1000)) : null,
    health,
    activity,
    config: strataConfigBrief(cfg),
    effective_args: strataEffectiveArgs(cfg),   // 改写后 args 的关键项摘要（vram_reserve/expert_cache/mmap/gpu/context）
    last_error: strataLastError(),
    models,
    engine_build: eng.build,
    vram_watch: strataVramWatchView(),   // [STRICT_8G_PATCH_V1] 严格模式显存看护（limit/peak/current/violated）
  };
}
function strataStatus(force) {
  if (!force && __strataStatusCache && Date.now() - __strataStatusCache.t < 1000) {
    return Promise.resolve(__strataStatusCache.data);
  }
  if (__strataStatusFly) return __strataStatusFly;
  __strataStatusFly = strataStatusCompute()
    .then((data) => { __strataStatusFly = null; __strataStatusCache = { t: Date.now(), data }; return data; })
    .catch((e) => {
      __strataStatusFly = null;
      const data = { installed: strataInstalled(), engine_ready: false, model_ready: false, running: false,
        pid: null, port: 8080, host: '127.0.0.1', api_key_set: false, started_at: null, uptime_s: null,
        health: { ok: false }, activity: { phase: 'unknown' }, config: null,
        last_error: 'status 计算异常：' + String((e && e.message) || e).slice(0, 200), models: [], engine_build: null };
      __strataStatusCache = { t: Date.now(), data };
      return data;
    });
  return __strataStatusFly;
}

// —— start ——
function strataShellQuote(s) { return "'" + String(s).replace(/'/g, "'\\''") + "'"; }

async function strataStart(body) {
  const reply409 = (reason) => ({ code: 409, obj: { ok: false, reason } });
  const cfg = strataLatestConfig();
  const defPort = (cfg && cfg.port) || 8080;
  const defHost = (body && body.host) || (cfg && cfg.host) || '127.0.0.1';
  const port = Math.min(65535, Math.max(1, parseInt((body && body.port) || defPort, 10) || defPort));
  const host = (body && body.host) || defHost;
  const api_key = (body && body.api_key) || (cfg && cfg.api_key) || null;
  const force = !!(body && body.force);

  // —— 单卡/显存调优参数（返场任务 09-30）：vram_mb / expert_cache / mmap_experts / gpus / max_context ——
  // 语义：vram_mb = 传给引擎的 --vram-reserve-mib（预留显存，专家缓存只吃"显存-预留"；
  //       8GB 单卡方案=预留 6144~7168 → 专家缓存 ~1-2GB，其余全在 CPU 池）。
  // 实现：快路径不吃命令行 flag，全部通过改写 cfg JSON 生效（见 strataApplyTuning）；
  //       gpus（单卡卡号）走 setup.py start() 的真参数 --gpu N，同时落盘进 cfg["gpu"]。
  const tuning = {
    strict_8g: !!(body && body.strict_8g),   // [STRICT_8G_PATCH_V1] 严格模式：GPU0 only ≤8GB、GPU1 禁用
    kv_resident: (body && body.kv_resident != null && body.kv_resident !== '') ? parseInt(body.kv_resident, 10) : null,   // [KV_STREAM_PATCH_V1] KV 流式常驻 cells
    vram_mb: (body && body.vram_mb != null && body.vram_mb !== '') ? parseInt(body.vram_mb, 10) : null,
    expert_cache: (body && body.expert_cache != null && body.expert_cache !== '')
      ? (String(body.expert_cache) === 'auto' ? 'auto' : parseInt(body.expert_cache, 10)) : null,
    mmap_experts: (body && body.mmap_experts != null) ? !!body.mmap_experts : null,
    // [DUAL_170HX_PATCH_V1] gpus：0=单卡；"0,1"/[0,1]=多卡分层 PP（引擎 layer split）
    gpu: (body && body.gpus != null && body.gpus !== '')
      ? (Array.isArray(body.gpus)
          ? body.gpus.map((x) => parseInt(x, 10)).filter((x) => Number.isInteger(x))
          : (String(body.gpus).indexOf(',') >= 0
              ? String(body.gpus).split(',').map((s) => parseInt(s.trim(), 10)).filter((x) => Number.isInteger(x))
              : parseInt(body.gpus, 10)))
      : null,
    prefill: (body && body.prefill != null && body.prefill !== '')
      ? (String(body.prefill) === 'auto' ? 'auto' : parseInt(body.prefill, 10)) : null,   // [DUAL_170HX_PATCH_V1]
    layer_split: (body && body.layer_split != null && body.layer_split !== '') ? String(body.layer_split) : null,   // [DUAL_170HX_PATCH_V1]
    max_context: (body && body.max_context != null && body.max_context !== '') ? parseInt(body.max_context, 10) : null,
  };
  // [STRICT_8G_PATCH_V1] strict_8g 归一化前置——显存门禁也要按 GPU0 单卡判定
  if (tuning.strict_8g) {
    tuning.vram_mb = 1024; tuning.expert_cache = 300; tuning.gpu = 0;
    // [KV_STREAM_PATCH_V1] 长上下文必须把 KV 流到 pinned RAM：int8 KV ≈1056B/cell/层，
    // 256K 全驻显存要 ~3.6GB，加上权重/MTP 会顶穿 8GB（实测 32K 全驻已 6736MiB）。
    // 20480 = 引擎下限（layer.cpp:526），VRAM 侧仅 ~0.28GB。
    if (tuning.kv_resident == null && tuning.max_context != null && tuning.max_context >= 65536) tuning.kv_resident = 20480;
  }
  const hasTuning = tuning.strict_8g || tuning.kv_resident != null || tuning.vram_mb != null || tuning.expert_cache != null || tuning.mmap_experts != null || tuning.gpu != null || tuning.max_context != null;   // [KV_STREAM_PATCH_V1]
  if (hasTuning && !cfg) return reply409('尚无已安装配置（strata-*.json），调优参数仅对已安装模型有效；首次安装请用 model/context/vision 参数');
  // NaN 防御（parseInt 失败）
  if (tuning.vram_mb != null && !Number.isFinite(tuning.vram_mb)) return reply409('vram_mb 不是合法数值');
  if (tuning.expert_cache != null && !(tuning.expert_cache === 'auto' || Number.isInteger(tuning.expert_cache))) return reply409('expert_cache 需为 "auto" 或整数');
  if (tuning.expert_cache === 0) return reply409('expert_cache 不能为 0（引擎会当 auto 吃满显存），严格模式请用 300');   // [STRICT_8G_PATCH_V1]
  if (tuning.gpu != null && !Number.isFinite(tuning.gpu)) return reply409('gpus 不是合法数值');
  if (tuning.max_context != null && !Number.isFinite(tuning.max_context)) return reply409('max_context 不是合法数值');
  // 范围校验同样前置到资源门禁之前（09-30 lead 验收：非法参数必须报参数错误，不被显存拦截遮蔽）
  if (tuning.vram_mb != null && !(tuning.vram_mb >= 256 && tuning.vram_mb <= 65535)) return reply409('vram_mb 需为 256~65535 的整数（MiB）');
  if (tuning.expert_cache != null && tuning.expert_cache !== 'auto' && tuning.expert_cache < 1) return reply409('expert_cache 需为 "auto" 或 ≥1 的整数');   // [STRICT_8G_PATCH_V1]
  if (tuning.gpu != null && !(tuning.gpu >= 0 && tuning.gpu <= 15)) return reply409('gpus 需为合法卡号（0~15）');
  if (tuning.max_context != null && !(tuning.max_context >= 1024 && tuning.max_context <= 1048576)) return reply409('max_context 需为 1024~1048576');
  if (tuning.kv_resident != null && (!Number.isFinite(tuning.kv_resident) || tuning.kv_resident < 20480 || tuning.kv_resident > 1048576)) return reply409('kv_resident 需为 20480~1048576（引擎下限 20480）');   // [KV_STREAM_PATCH_V1]

  // 参数校验（先于就绪门禁：给用户明确的原因，不被泛化拦截遮蔽）
  const STRATA_MODELS_OK = { Q2_0: 1, IQ2_XS: 1, IQ3_XXS: 1, IQ3_S: 1, IQ1_M: 1 };   // = setup.py MODELS 档位表（IQ1_M 属 coder 家族）
  let model = (body && body.model) ? String(body.model).toUpperCase() : null;
  if (model && !STRATA_MODELS_OK[model]) return reply409('未知模型档位 ' + model + '（合法：' + Object.keys(STRATA_MODELS_OK).join('/') + '）');
  if (!model) {
    const tail = cfg && cfg.model_name ? String(cfg.model_name).split('-').pop().toUpperCase() : null;
    model = (tail && STRATA_MODELS_OK[tail]) ? tail : (strataModelDirs().find((m) => m.ready) || {}).name || null;
  }
  const context = parseInt((body && body.context) || (body && body.max_context) || (cfg && cfg.args && cfg.args[cfg.args.indexOf('--max-context') + 1]) || 32768, 10) || 32768;
  const vision = (body && body.vision) || (cfg && Array.isArray(cfg.args) && cfg.args.includes('--vision') ? 'gpu' : 'no');

  // 就绪门禁（安装/引擎/模型）——放在参数校验之后
  if (!strataInstalled()) return reply409('Strata 未安装（找不到 ' + STRATA_SETUP + '）');
  {
    const eng = strataEngineReady();
    if (!eng.ready) return reply409('引擎尚未编译完成（engine/strata 或 BUILD.json 缺失），请先完成构建');
  }
  if (!strataModelReady()) return reply409('模型分片未就绪（Strata-data/models/ 无完整 .done 档位）');

  // 门禁 4：已在运行（永不跳过）
  const proc = strataScanMainProc();
  if (proc) return reply409('Strata 已在运行 (pid ' + proc.pid + ', port ' + (proc.port || '?') + ')');
  // 门禁 3：端口占用（永不跳过）
  if (await strataPortListening(port)) return reply409('端口 ' + port + ' 已被监听，无法启动（可换 --port 或先停占用方）');

  // 门禁 1/2：显存 + 可用内存（force=true 跳过并记 OVERRIDE）
  // 单卡模式（gpus 指定）：显存门禁只查所选那张卡——其它卡即便被占也不影响本方案。
  if (!force) {
    const gpu = await strataGpuMemUsed();
    if (!gpu.ok) return reply409('nvidia-smi 查询失败，无法确认显存占用，拒绝启动（紧急情况可用 force）');
    // [DUAL_170HX_PATCH_V1] gpu 可能是单值或列表
    const wantGpus = tuning.gpu == null ? null : (Array.isArray(tuning.gpu) ? tuning.gpu : [tuning.gpu]);
    const scope = wantGpus ? gpu.gpus.filter((g) => wantGpus.indexOf(g.index) >= 0) : gpu.gpus;
    if (wantGpus && scope.length === 0) return reply409('所选 GPU ' + wantGpus.join(',') + ' 不存在（nvidia-smi 未见该卡）');
    const hot = scope.filter((g) => g.used_mb >= STRATA_GATE_VRAM_MB);
    if (hot.length) {
      const detail = hot.map((g) => 'GPU' + g.index + ' 占用 ' + (g.used_mb / 1024).toFixed(1) + 'GB').join('/');
      return reply409('GPU 占用过高：18420 生产实例仍在运行（' + detail + '，门禁 <8GB），请先停掉生产实例；紧急情况可带 force=true 跳过（风险自担）');
    }
    const availGB = strataAvailMemKB() / 1024 / 1024;
    if (availGB < STRATA_GATE_RAM_GB) {
      return reply409('可用内存不足（当前 ' + availGB.toFixed(1) + ' GB，需 ≥' + STRATA_GATE_RAM_GB + 'GB）：Strata 首次加载需 35-55GB 专家进 RAM');
    }
  } else {
    console.log('[strata] START OVERRIDE：force=true 跳过显存/内存门禁（' + new Date().toISOString() + '，body=' + JSON.stringify(body || {}).slice(0, 200) + '）');
  }

  // —— 改写 cfg JSON（仅在有调优参数时；备份 .bak-console 同文件只留最新一份）——
  // 注意顺序：门禁全过才动配置——拒绝启动时不留半成品改写。
  let tuningResult = null;
  if (hasTuning && cfg && cfg.__path) {
    try {
      tuningResult = strataApplyTuning(cfg.__path, cfg, tuning);
    } catch (e) {
      if (e.userFacing) return reply409(String(e.message));
      return { code: 500, obj: { ok: false, reason: '改写启动参数失败：' + String((e && e.message) || e) } };
    }
    __strataStatusCache = null;   // cfg 变了，status 立即反映 effective_args
  }

  // spawn：setsid 独立进程组 + detached + unref；日志追加到 strata-console.log。
  // 两种形态（与引擎侧实测一致）：
  //  ① 有已安装配置（Strata/strata-*.json）→ `python setup.py --yes [--port N] [--gpu N]`：走
  //     installed→start() 快路径（subprocess.call 前台常驻=进程组长），host/api-key 由配置文件承载
  //     （setup.py 启动路径不接收这两个参数）。**绝不可带 --model/--context**：那会落入完整安装
  //     流程（PC 检查/引擎更新/重生成配置），前台窗口退出即 SIGTERM 杀掉刚起的引擎。
  //     引擎 flag 类调优（vram_mb/expert_cache/mmap/max_context）不走命令行，靠上面改写 cfg。
  //  ② 无配置（首次安装）→ 完整安装参数；此时 body 的 model/context/vision 生效。
  // 注：setup.py 无 --no-browser 参数；--open 由 start() 追加，headless 无 DISPLAY 时
  //     webbrowser.open 静默失败，不影响服务。
  const haveCfg = !!cfg;
  let args;
  if (!haveCfg) {
    const m = (body && body.model) ? String(body.model).toUpperCase() : (model || null);
    if (!m || !STRATA_MODELS_OK[m]) return reply409('首次安装需指定合法模型档位（' + Object.keys(STRATA_MODELS_OK).join('/') + '）');
    args = [STRATA_SETUP, '--model', m, '--context', String(context), '--vision', String(vision), '--yes', '--port', String(port)];
    if (host && host !== '127.0.0.1' && host !== 'localhost') args.push('--host', String(host));
    if (api_key) args.push('--api-key', String(api_key));
  } else {
    if (body && body.model) console.log('[strata] 已有安装配置：忽略 model 变更请求（改档位请用停止后手动 setup.py --setup），走快路径启动');
    args = [STRATA_SETUP, '--yes', '--port', String(port)];
    if (tuning.gpu != null) args.push('--gpu', String(tuning.gpu));   // start() 真参数：cmd += --gpu N（cfg 里 int 亦已落盘）
    if (api_key && !cfg.api_key) args.push('--api-key', String(api_key));   // 仅首次落盘进配置（start 路径不读该参数，但安装流程会写）
  }
  const cmdLine = strataShellQuote(STRATA_PY) + ' ' + args.map(strataShellQuote).join(' ') +
    ' >> ' + strataShellQuote(STRATA_CONSOLE_LOG) + ' 2>&1 < /dev/null';
  let child;
  try {
    // [STRICT_8G_PATCH_V1] GPU 隔离双保险：serve/server.py 按 cfg["gpu"] 设 CUDA_VISIBLE_DEVICES，
    // 这里在派发层再钉一次 —— 单卡/严格模式下 GPU1 对整棵子进程树完全不可见。
    const spawnEnv = Object.assign({}, process.env);
    if (tuning.gpu != null) spawnEnv.CUDA_VISIBLE_DEVICES = String(tuning.gpu);
    child = spawn('setsid', ['-f', 'bash', '-c', cmdLine], { cwd: STRATA_ROOT, detached: true, stdio: 'ignore', env: spawnEnv });
  } catch (e) {
    return { code: 500, obj: { ok: false, reason: 'spawn 失败：' + String((e && e.message) || e) } };
  }
  child.on('error', (e) => { console.error('[strata] spawn error:', String((e && e.message) || e)); });
  child.unref();
  const spawnPid = child.pid || null;
  console.log('[strata] 启动已派发 pid=' + spawnPid + ' port=' + port + ' model=' + model + ' ctx=' + context
    + (hasTuning ? ' tuning=' + JSON.stringify(tuning) + ' cfg_changed=' + !!(tuningResult && tuningResult.changed) : '')
    + '（加载 35-55GB 专家需 1-3 分钟）');
  __strataStatusCache = null;   // 状态缓存作废，下次 status 立见新进程
  if (tuning.strict_8g) strataVramWatchStart();   // [STRICT_8G_PATCH_V1] 机器级兜底：GPU0 超限立即停实例
  return { code: 200, obj: {
    ok: true, pid: spawnPid, port,
    cfg_rewritten: !!(tuningResult && tuningResult.changed),
    effective_args: (tuningResult && tuningResult.effective) || strataEffectiveArgs(cfg),
    vram_watch: tuning.strict_8g ? strataVramWatchView() : null,
    note: '进程组已派发；模型加载中（1-3 分钟），/v1/internal/strata/status 可轮询进度',
  } };
}

// —— stop：SIGTERM 进程组，最多等 60s，超时不强杀 ——
async function strataStop() {
  const proc = strataScanMainProc();
  if (!proc) return { code: 200, obj: { ok: true, note: '未在运行，无需停止' } };
  let pgid = proc.pid;
  try {
    const stat = fs.readFileSync(`/proc/${proc.pid}/stat`, 'utf8');
    const fields = stat.slice(stat.lastIndexOf(')') + 2).split(/\s+/);
    const p = parseInt(fields[2], 10);   // 第 3 字段（跳过 comm 后）= pgrp
    if (isFinite(p) && p > 0) pgid = p;
  } catch (e) {}
  try {
    process.kill(-pgid, 'SIGTERM');
  } catch (e) {
    // 进程组信号失败（如权限/组不存在）→ 退回单进程 SIGTERM
    try { process.kill(proc.pid, 'SIGTERM'); } catch (e2) {
      return { code: 500, obj: { ok: false, reason: '发送 SIGTERM 失败：' + String((e && e.message) || e) } };
    }
  }
  const t0 = Date.now();
  while (Date.now() - t0 < 60000) {
    await sleep(1000);
    if (!strataScanMainProc()) {
      __strataStatusCache = null;
      strataVramWatchStop();   // [STRICT_8G_PATCH_V1]
      console.log('[strata] 已优雅退出（' + ((Date.now() - t0) / 1000).toFixed(0) + 's）');
      return { code: 200, obj: { ok: true } };
    }
  }
  return { code: 200, obj: { ok: false, reason: '优雅退出超时（60s），未强杀；请人工检查 pid ' + proc.pid } };
}

// —— logs：日志尾 N 行（默认 200，上限 2000；只读尾部窗口，绝不整文件读）——
function strataLogs(tail) {
  const n = Math.min(2000, Math.max(1, parseInt(tail, 10) || 200));
  const meta = { file: STRATA_CONSOLE_LOG, lines: [], mtime: null, size: 0, truncated: false };
  let st;
  try { st = fs.statSync(STRATA_CONSOLE_LOG); } catch (e) { return meta; }
  meta.mtime = st.mtimeMs; meta.size = st.size;
  if (st.size === 0) return meta;
  const window = Math.min(st.size, Math.max(n * 4096, 256 * 1024));   // 每行按 4KB 估，最少 256KB
  try {
    const fd = fs.openSync(STRATA_CONSOLE_LOG, 'r');
    try {
      const buf = Buffer.alloc(window);
      fs.readSync(fd, buf, 0, window, st.size - window);
      const lines = buf.toString('utf8').split('\n');
      if (st.size > window) { lines.shift(); meta.truncated = true; }   // 窗口首行可能不完整，丢弃
      meta.lines = lines.filter((l) => l.length > 0).slice(-n);
    } finally { fs.closeSync(fd); }
  } catch (e) { meta.error = String((e && e.message) || e); }
  return meta;
}

// —— metrics：透传引擎 /metrics（未跑 {ok:false}；解析失败回 raw 前 2KB）——
async function strataMetrics() {
  const proc = strataScanMainProc();
  if (!proc) return { code: 200, obj: { ok: false } };
  const j = await strataHttpGetJson(proc.port || 8080, '/metrics', 3000);
  if (!j) return { code: 200, obj: { ok: false, reason: '引擎 /metrics 无响应（加载中或已挂）' } };
  if (j.__raw) return { code: 200, obj: { ok: false, raw: j.__raw } };
  return { code: 200, obj: j };
}

// —— 严格 8GB 显存看护（[STRICT_8G_PATCH_V1]，用户红线：GPU0 ≤8GB、GPU1 完全禁用）——
// 参数只是第一道防线；这里是机器级兜底：每 2s 采样 GPU0，超限立即保护性停止并留下违规事实。
// 状态挂 global（本项目铁律：handler 作用域 const 会被 TDZ 吞掉，让定时器静默失效）。
function strataVramWatchState() {
  if (!global.__strataVramWatch) {
    global.__strataVramWatch = { active: false, limit_mib: STRATA_GATE_VRAM_MB, peak_mib: 0, current_mib: 0, violated: false, since: null, timer: null };
  }
  return global.__strataVramWatch;
}
function strataVramWatchView() {   // 只回标量：timer 不可进 JSON
  const st = strataVramWatchState();
  return { active: st.active, limit_mib: st.limit_mib, peak_mib: st.peak_mib, current_mib: st.current_mib, violated: st.violated, since: st.since };
}
async function strataVramWatchTick() {
  const st = strataVramWatchState();
  if (!st.active) return;
  const gpu = await strataGpuMemUsed();
  if (!gpu || !gpu.ok || !Array.isArray(gpu.gpus)) return;
  const g0 = gpu.gpus.find((g) => g.index === 0);
  if (!g0) return;
  st.current_mib = g0.used_mb;
  if (g0.used_mb > st.peak_mib) st.peak_mib = g0.used_mb;
  if (g0.used_mb > st.limit_mib) {
    st.violated = true;
    st.since = st.since || new Date().toISOString();
    console.error('[strata][vram-watch] GPU0 ' + g0.used_mb + ' MiB 超过上限 ' + st.limit_mib + ' MiB → 保护性停止实例');
    try { await strataStop(); } catch (e) { console.error('[strata][vram-watch] 停止失败：' + String((e && e.message) || e)); }
    strataVramWatchStop();
    __strataStatusCache = null;
  }
}
function strataVramWatchStart() {
  const st = strataVramWatchState();
  st.active = true; st.violated = false; st.peak_mib = 0; st.current_mib = 0; st.since = null;
  if (st.timer) { clearInterval(st.timer); st.timer = null; }
  st.timer = setInterval(() => { strataVramWatchTick().catch(() => {}); }, 2000);
  console.log('[strata][vram-watch] 已启动：GPU0 硬上限 ' + st.limit_mib + ' MiB，每 2s 采样，超限即停');
}
function strataVramWatchStop() {
  const st = strataVramWatchState();
  st.active = false;
  if (st.timer) { clearInterval(st.timer); st.timer = null; }
}

// —— options：可选档位 + 引擎构建信息 + 缺省值 ——
function strataOptions() {
  const eng = strataEngineReady();
  const cfg = strataLatestConfig();
  const brief = strataConfigBrief(cfg);
  return {
    models: strataModelDirs(),
    engine: eng.build ? { version: eng.build.version || null, archs: eng.build.archs || null, source: eng.build.source || null, ...eng.build } : null,
    engine_ready: eng.ready,
    defaults: { context: (brief && brief.context) || 32768, port: (brief && brief.port) || 8080, vision: 'no', host: '127.0.0.1' },
    gate: { vram_free_per_gpu_mb: STRATA_GATE_VRAM_MB, ram_avail_gb: STRATA_GATE_RAM_GB },
    // [STRICT_8G_PATCH_V1] 显存方案预设：strict8g = 用户红线方案（GPU0 only ≤8GB、GPU1 禁用），实测峰值 6260MiB。
    // 旧 single8g（vram_mb=7168）语义错误——--vram-reserve-mib 是"预留"，会让专家缓存自动吃满 33GiB。
    // [DEFAULT_FULL_VRAM_V1] 用户 09-30 拍板：默认=吃满整卡空闲显存（Strata 原生最优形态，专家缓存 auto）。
    // 每个 preset 同时带平铺字段与 fields{}（前端两版提取逻辑都兼容）；is_default 标记缺省选中项。
    vram_presets: [
      // [DUAL_170HX_PATCH_V1] 用户 09-30 拍板：默认档 = 双卡 170HX 分层 PP（8GB×2 实测 prefill 767~947 tok/s、
      // decode 66~88 tok/s）。prefill 512 与 kv 流式是 8GB 卡上双卡能起来的硬条件。
      { value: 'default', label: '默认（双卡 170HX 分层 PP，推荐）', is_default: true, vram_mb: null,
        gpus: '0,1', layer_split: 'auto', prefill: 512, kv_resident: 20480, max_context: 65536,
        fields: { gpus: '0,1', layer_split: 'auto', prefill: 512, kv_resident: 20480, max_context: 65536 } },
      { value: 'strict8g', label: '严格 GPU0 ≤8GB', strict_8g: true, gpus: 0, expert_cache: 300, vram_mb: 1024, max_context: 32768,
        fields: { strict_8g: true, gpus: 0, expert_cache: 300, vram_mb: 1024, max_context: 32768 } },
    ],
    tuning_fields: ['strict_8g', 'max_context', 'vram_mb', 'expert_cache', 'mmap_experts', 'gpus', 'prefill', 'layer_split', 'kv_resident'],   // [DUAL_170HX_PATCH_V1]
    strict_8g: { limit_mib: STRATA_GATE_VRAM_MB, expert_cache: 300, vram_reserve_mib: 1024, gpus: 0, kv_resident: 20480, kv_resident_from_context: 65536 },   // [KV_STREAM_PATCH_V1]
  };
}

// —— 路由入口（pathname 前缀 /v1/internal/strata；POST 已被上层口令拦截）——
async function strataHandle(req, res, urlObj) {
  const reply = (code, obj) => { res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8' }); res.end(JSON.stringify(obj)); };
  const sub = urlObj.pathname.slice('/v1/internal/strata'.length);
  // 兼容前端可能的无子路径 GET（等价 /status），避免 404
  if (req.method === 'GET' && (sub === '' || sub === '/')) { return reply(200, await strataStatus()); }
  if (req.method === 'GET' && sub === '/status') { return reply(200, await strataStatus()); }
  if (req.method === 'GET' && sub === '/logs')  { return reply(200, strataLogs(urlObj.searchParams.get('tail'))); }
  if (req.method === 'GET' && sub === '/metrics') { const r = await strataMetrics(); return reply(r.code, r.obj); }
  if (req.method === 'GET' && sub === '/options') { return reply(200, strataOptions()); }
  if (req.method === 'POST' && sub === '/start') {
    let body = {};
    try {
      const raw = await new Promise((resolve, reject) => {
        let s = ''; req.on('data', (c) => { s += c; if (s.length > 65536) req.destroy(); });
        req.on('end', () => resolve(s)); req.on('error', reject);
      });
      body = raw ? JSON.parse(raw) : {};
    } catch (e) { return reply(400, { ok: false, reason: '请求体不是合法 JSON' }); }
    const r = await strataStart(body);
    return reply(r.code, r.obj);
  }
  if (req.method === 'POST' && sub === '/stop') { const r = await strataStop(); return reply(r.code, r.obj); }
  return reply(404, { ok: false, reason: 'strata: 未知路由 ' + sub });
}
// ==== [strata-console] END module ====

// ---------------- 5) CPU 控制（跨平台版 v3：Intel intel_pstate / AMD·acpi-cpufreq boost 自适应，移植自 bench-console/cpu-control） ----------------
// 调 /usr/local/bin/cpu-ctl（脚本随本仓库部署；非 root 时自提权，需 sudoers.d 白名单，
// 见 ops/install-cpu-ctl-127.sh）。全部为运行时软控制 sysfs，重启回 BIOS/内核默认。
// 混合架构专属动作（小核簇/超线程/EPP）维持删除；睿频开关由 cpu-ctl 抽象
// （intel 走 no_turbo，AMD/acpi 走 /sys/devices/system/cpu/cpufreq/boost）。
// 10-05 换装：AMD EPYC 7F52 16C/32T（SMT 开），HUANANZHI H12D-8D。
// 接口: GET  /v1/internal/cpuctl      状态 JSON（0.5s TTL 缓存 + 单飞，防轮询叠发）
//       POST /v1/internal/cpuctl/cmd  action 白名单（走既有 /v1/internal/ POST 口令拦截）
// 铁律遵守：execFile + 自身 timeout，绝不 execSync——sysfs/脚本卡住不许拖垮事件循环。
const CPU_CTL = process.env.CPU_CTL || '/usr/local/bin/cpu-ctl';
const CPU_FREQ_RE = /^\d+(\.\d+)?\s*[GgMm]?$/;          // 3.7G / 2000M / 1200000
const CPU_SPEC_RE = /^(all|[0-9][0-9,\-]{0,31})$/;      // all | 0-7,16（同构核无 p/e）
const CPU_VAL_RE  = /^[a-z_]{1,40}$/;                    // governor 值

function cpuCtlRun(args, timeoutMs) {
  return new Promise((resolve) => {
    require('child_process').execFile(CPU_CTL, args.map(String), { timeout: timeoutMs || 30000, maxBuffer: 4 * 1024 * 1024 },
      (err, stdout, stderr) => {
        const out = String(stdout || '').trim();
        const errText = String(stderr || '').trim();
        resolve({
          ok: !err,
          output: out || errText || (err ? String(err.message || err) : '(无输出)'),
          killed: !!(err && err.killed),
        });
      });
  });
}

// 状态短缓存：UI 1s 轮询（多标签页会叠发），TTL 0.5s + 单飞去重，execFile（内含 sudo 提权 + python）
// 同刻只跑一份；TTL 必须小于前端轮询间隔，否则数字看着不刷新。
let __cpuStateCache = null; // { t, state }
let __cpuStateFly = null;   // 在途 Promise
function cpuCtlStatus(force) {
  if (!force && __cpuStateCache && Date.now() - __cpuStateCache.t < 500) {
    return Promise.resolve({ ok: true, state: __cpuStateCache.state });
  }
  if (__cpuStateFly) return __cpuStateFly;
  __cpuStateFly = cpuCtlRun(['status', '--json'], 20000).then((r) => {
    __cpuStateFly = null;
    try {
      const state = JSON.parse(r.output);
      __cpuStateCache = { t: Date.now(), state };
      return { ok: true, state };
    } catch (e) {
      return { ok: false, error: 'cpu-ctl status 输出无法解析：' + String(r.output).slice(0, 200) };
    }
  }).catch((e) => { __cpuStateFly = null; return { ok: false, error: String((e && e.message) || e) }; });
  return __cpuStateFly;
}

// action 白名单 → cpu-ctl 参数（返回 null = 非法；杜绝任意命令注入）
function cpuCtlAction(body) {
  const a = String(body.action || '');
  const spec = (s) => (CPU_SPEC_RE.test(String(s || 'all')) ? String(s || 'all') : null);
  switch (a) {
    case 'freq_max': case 'freq_min': {
      const f = String(body.freq || '').trim();
      if (!CPU_FREQ_RE.test(f) || !spec(body.spec)) return null;
      return { args: ['freq', a === 'freq_max' ? 'max' : 'min', f, spec(body.spec)], timeout: 25000 };
    }
    case 'freq_reset': return { args: ['freq', 'reset'], timeout: 25000 };
    case 'gov': {
      if (!CPU_VAL_RE.test(String(body.val || '')) || !spec(body.spec)) return null;
      return { args: ['gov', body.val, spec(body.spec)], timeout: 25000 };
    }
    case 'turbo_on':  return { args: ['turbo', 'on'],  timeout: 25000 };
    case 'turbo_off': return { args: ['turbo', 'off'], timeout: 25000 };
    case 'core_on': case 'core_off': {
      const n = body.cpu;
      if (!/^\d{1,3}$/.test(String(n))) return null;
      return { args: ['core', a === 'core_on' ? 'on' : 'off', String(n)], timeout: 30000 };
    }
    case 'all_on': return { args: ['core', 'all-on'], timeout: 90000 };
    case 'bench': {
      const s = parseInt(body.secs, 10);
      if (!isFinite(s) || s < 1 || s > 30) return null;
      return { args: ['bench', String(s)], timeout: s * 20000 + 180000, noState: true };
    }
    default: return null;
  }
}

function cpuCtlReadBody(req) {
  return new Promise((resolve) => {
    let body = '';
    req.on('data', (c) => { body += c; if (body.length > 65536) req.destroy(); });
    req.on('end', () => { try { resolve(JSON.parse(body || '{}')); } catch (_) { resolve(null); } });
    req.on('error', () => resolve(null));
  });
}

async function cpuCtlHandle(req, res, urlObj) {
  const reply = (code, obj) => { res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8' }); res.end(JSON.stringify(obj)); };
  const sub = urlObj.pathname.slice('/v1/internal/cpuctl'.length);
  if (req.method === 'GET' && sub === '') {
    return reply(200, await cpuCtlStatus());
  }
  if (req.method === 'POST' && sub === '/cmd') {
    const body = await cpuCtlReadBody(req);
    if (!body) return reply(400, { ok: false, msg: '请求体不是合法 JSON' });
    const act = cpuCtlAction(body);
    if (!act) return reply(400, { ok: false, msg: '非法参数或不在白名单内' });
    const r = await cpuCtlRun(act.args, act.timeout);
    if (act.noState) return reply(200, { ok: r.ok, msg: r.output, killed: r.killed });
    const st = await cpuCtlStatus(true); // 命令改过状态，强制绕缓存补拉
    return reply(200, { ok: r.ok, msg: r.output, state: st.ok ? st.state : null, error: st.ok ? null : st.error });
  }
  return reply(404, { ok: false, msg: 'cpuctl: 未知路由' });
}

// ==== [strata-remote-monitor] BEGIN module ====
// 用途：克隆 Strata Web UI（http://192.0.2.38:8080/#monitor）的 Monitor 页到控制台「Strata 监控」标签。
// 数据源 = 远端 Strata 自带的 /metrics + /health + /mcp（Strata 服务端不发 CORS 头，浏览器跨源直连会被拦，
// 由控制台服务端代理转发；远端机在 Windows 上，控制台机已实测可达）。
// 目标地址 = strata-remote.json（{"baseUrl":"http://192.0.2.38:8080","mac":"..","sshHost":"ll@192.0.2.38",
// "bat":"D:\\Strata-0.1.40\\start-strata-0140.bat","broadcast":"192.0.2.255"}），按 mtime 热加载；文件缺失用缺省值。
// 电源控制（POST /v1/internal/strata-remote/cmd，走 /v1/internal/ POST 口令拦截）：
//   wake      dgram 发魔术包（零外部依赖、无需 root；UDP 9 广播，.38 网卡已确认在 powercfg wake_armed 列内）
//   shutdown  ssh 远端 shutdown /s /t 0（.127 公钥已入 .38 administrators_authorized_keys，ll 为管理员）
//   startbat  ssh 远端 start "" /min <bat>（bat 末尾有 pause，必须 start 分离，否则 ssh 挂到超时）
// 铁律：fetch + AbortController 带超时（metrics 3s / health·mcp 4s）；800ms TTL 缓存 + 单飞防轮询叠发；
//       失败返回 {ok:false,msg}——前端保留上一帧 + 黄条，绝不伪造空 metrics 帧覆盖好数据（10-03 三态铁律）；
//       电源命令 execFile(ssh) + 自建定时器直接结算（10-07 铁律：定时器放弃等待，不依赖 'close'），
//       BatchMode=yes 防 ssh 挂起等密码；action 白名单 + MAC/路径正则，配置文件即信任边界。
const STRATA_REMOTE_DEFAULT_BASE = 'http://192.0.2.38:8080';
const STRATA_REMOTE_MAC_RE = /^([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$/;
const STRATA_REMOTE_BAT_RE = /^[A-Za-z0-9 :\\._\-]{1,200}$/;
let __strataRemoteCfg = { mtime: -1, baseUrl: STRATA_REMOTE_DEFAULT_BASE, mac: '', sshHost: 'll@192.0.2.38', bat: '', broadcast: '192.0.2.255' };
function strataRemoteCfgGet() {
  const p = path.join(__dirname, 'strata-remote.json');
  try {
    const st = fs.statSync(p);
    if (st.mtimeMs !== __strataRemoteCfg.mtime) {
      const j = JSON.parse(fs.readFileSync(p, 'utf8'));
      const b = String((j && j.baseUrl) || STRATA_REMOTE_DEFAULT_BASE).replace(/\/+$/, '');
      __strataRemoteCfg = {
        mtime: st.mtimeMs,
        baseUrl: b || STRATA_REMOTE_DEFAULT_BASE,
        mac: String((j && j.mac) || ''),
        sshHost: String((j && j.sshHost) || 'll@192.0.2.38'),
        bat: String((j && j.bat) || ''),
        broadcast: String((j && j.broadcast) || '192.0.2.255'),
      };
    }
  } catch (e) { /* 文件缺失/解析失败：保持上一次值（首跑即缺省 .38:8080） */ }
  return __strataRemoteCfg;
}
function strataRemoteBase() { return strataRemoteCfgGet().baseUrl; }
function strataRemoteGet(base, sub, timeoutMs) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);
  return fetch(base + sub, { signal: ac.signal, headers: { Accept: 'application/json' } })
    .then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .finally(() => clearTimeout(timer));
}
const __strataRemoteCache = { at: 0, key: '', data: null, inflight: null, inflightKey: '', health: null, healthAt: 0, mcp: null, mcpAt: 0 };
function strataRemoteSnapshot(all) {
  const key = all ? 'all' : 'top';
  const c = __strataRemoteCache;
  const now = Date.now();
  if (c.data && c.key === key && now - c.at < 800) return Promise.resolve(c.data);
  if (c.inflight && c.inflightKey === key) return c.inflight;
  const base = strataRemoteBase();
  const p = (async () => {
    const out = { ok: true, ts: Date.now(), base };
    try {
      out.metrics = await strataRemoteGet(base, all ? '/metrics?requests=all' : '/metrics', 3000);
    } catch (e) {
      out.ok = false;
      out.msg = '远端 Strata /metrics 不可达：' + String((e && e.message) || e);
    }
    if (!c.health || Date.now() - c.healthAt > 5000) {
      try { c.health = await strataRemoteGet(base, '/health', 4000); c.healthAt = Date.now(); } catch (e) { /* 保留旧值 */ }
    }
    out.health = c.health;
    if (!c.mcp || Date.now() - c.mcpAt > 10000) {
      try { c.mcp = await strataRemoteGet(base, '/mcp', 4000); c.mcpAt = Date.now(); } catch (e) { /* 保留旧值 */ }
    }
    out.mcp = c.mcp;
    return out;
  })();
  c.inflight = p; c.inflightKey = key;
  return p.then((d) => { c.data = d; c.key = key; c.at = Date.now(); return d; })
          .finally(() => { if (c.inflight === p) { c.inflight = null; c.inflightKey = ''; } });
}
// 魔术包：6×FF + MAC×16，UDP 9；向子网定向广播 + 全局广播各发一份（目标关机后单播无意义，广播是正解）。
// dgram.send 对广播地址不会挂；仍给 4s 兜底结算，绝不把调用方挂死。
function strataWake(mac, broadcast) {
  return new Promise((resolve) => {
    let sock;
    try { sock = require('dgram').createSocket('udp4'); } catch (e) { return resolve({ ok: false, msg: 'dgram 不可用：' + String((e && e.message) || e) }); }
    let settled = false;
    const done = (r) => { if (settled) return; settled = true; try { sock.close(); } catch (e) {} resolve(r); };
    setTimeout(() => done({ ok: false, msg: '唤醒包发送超时（放弃等待）' }), 4000).unref();
    // Node dgram 的 fd 是懒创建的：未 bind 就 setBroadcast 会 EBADF（10-08 实锤），先 bind(0) 再设选项。
    sock.on('error', (e) => done({ ok: false, msg: 'UDP 错误：' + String((e && e.message) || e) }));
    sock.bind(0, () => {
      try {
        sock.setBroadcast(true);
        const macBuf = Buffer.from(mac.replace(/[^0-9a-fA-F]/g, ''), 'hex');
        const parts = [Buffer.alloc(6, 0xff)];
        for (let i = 0; i < 16; i++) parts.push(macBuf);
        const pkt = Buffer.concat(parts);
        const targets = [];
        if (broadcast) targets.push(broadcast);
        targets.push('255.255.255.255');
        let sent = 0, pending = targets.length;
        targets.forEach((t) => sock.send(pkt, 9, t, (err) => {
          if (!err) sent++;
          if (--pending === 0) done({ ok: sent > 0, msg: sent ? '魔术包已发往 ' + targets.join(' / ') + '（MAC ' + mac + '）；开机到服务恢复通常 1~3 分钟，页面会自动转绿' : '魔术包发送失败' });
        }));
      } catch (e) { done({ ok: false, msg: String((e && e.message) || e) }); }
    });
  });
}
// ssh 固定目标执行（信任边界=strata-remote.json 的 sshHost；BatchMode 防交互挂起；
// 自建定时器直接结算——10-07 铁律，不依赖子进程 'close'）。
function strataSshExec(cmd, timeoutMs) {
  const cfg = strataRemoteCfgGet();
  return new Promise((resolve) => {
    let settled = false;
    const finish = (r) => { if (settled) return; settled = true; resolve(r); };
    let proc;
    try {
      proc = require('child_process').execFile('ssh', ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=accept-new', cfg.sshHost, cmd],
        (err, stdout, stderr) => finish({ ok: !err, msg: String(stderr || stdout || (err && err.message) || '').trim().slice(0, 300) }));
    } catch (e) { return finish({ ok: false, msg: String((e && e.message) || e) }); }
    if (proc && proc.on) proc.on('error', (e) => finish({ ok: false, msg: String((e && e.message) || e) }));
    setTimeout(() => {
      if (!settled) { try { proc.kill('SIGKILL'); } catch (e) {} finish({ ok: false, msg: 'SSH 执行超时（' + timeoutMs + 'ms 放弃等待；命令可能已在远端生效，以页面状态为准）' }); }
    }, timeoutMs).unref();
  });
}
async function strataRemoteCmd(req, res) {
  const reply = (code, obj) => { res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8' }); res.end(JSON.stringify(obj)); };
  const body = await cpuCtlReadBody(req);
  if (!body) return reply(400, { ok: false, msg: '请求体不是合法 JSON' });
  const cfg = strataRemoteCfgGet();
  const act = String(body.action || '');
  if (act === 'wake') {
    if (!STRATA_REMOTE_MAC_RE.test(cfg.mac)) return reply(200, { ok: false, msg: 'strata-remote.json 缺 mac 字段或格式非法（应为 aa:bb:cc:dd:ee:ff）' });
    const r = await strataWake(cfg.mac, cfg.broadcast);
    return reply(200, Object.assign({ action: act }, r));
  }
  if (act === 'shutdown') {
    // 10-08 实锤：.38 是精简 Win11，64 位与 WOW64 两份 shutdown.exe 都缺依赖（cmd 报"找不到文件"），
    // 必须走 PowerShell Stop-Computer（直调 advapi32 关机 API，不经 shutdown.exe）。-WhatIf 已实测可用。
    const r = await strataSshExec('powershell -NoProfile -Command "Stop-Computer -Force"', 12000);
    return reply(200, Object.assign({ action: act }, r, r.ok ? { msg: '关机指令已下发（远端立即开始关机，页面稍后转「远端不可达」属正常）' } : {}));
  }
  if (act === 'startbat') {
    if (!STRATA_REMOTE_BAT_RE.test(cfg.bat)) return reply(200, { ok: false, msg: 'strata-remote.json 的 bat 路径为空或含非法字符' });
    // 10-08 用户定：按钮语义=「用指定脚本重启」——先把占着 8080 的旧实例干掉（无论它叫
    // python.exe 还是 pythonw.exe、是脚本起的还是手动双击起的），释放端口与 srv_256k.log 独占锁，
    // 再跑 bat，保证跑起来的永远是 bat 指定的配置（256K）。
    // 10-08 二次实锤：现场旧实例是 pythonw.exe（手动起），只 taskkill python.exe 杀不到它 →
    // 端口一直被占、新实例绑不上、静默失败。故主判据改成「按 8080 端口拿 PID 精确杀」，
    // 镜像名 taskkill 仅作兜底。
    const occ = await strataSshExec('netstat -ano | findstr :8080 | findstr LISTENING', 10000);
    const pids = Array.from(new Set(((occ.ok && occ.msg) || '').split('\n').map((ln) => {
      const m = ln.trim().match(/(\d+)\s*$/); return m ? m[1] : '';
    }).filter((x) => /^\d+$/.test(x))));
    let killed = pids.length > 0;
    for (const pid of pids) await strataSshExec('taskkill /f /pid ' + pid, 8000);
    // 兜底：万一端口已松但进程树还有残留（vision 子进程等），按镜像名再扫一遍。
    await strataSshExec('taskkill /f /im python.exe', 8000);
    await strataSshExec('taskkill /f /im pythonw.exe', 8000);
    await strataSshExec('taskkill /f /im strata.exe', 8000);
    await strataSshExec('taskkill /f /im strata-vision.exe', 8000);
    if (killed) await new Promise((res2) => setTimeout(res2, 2000));
    // 10-08 三次实锤（决定性）：bat 内层那句 start "StrataQ3" /min cmd /c "<带空格/>重定向的长 python 命令>"
    // 经 ssh→cmd→cmd 三层引号剥壳后散架，start 返回 0 却没真撑起 python（这就是用户反复所见「bat 未触发」）。
    // 直连跑 bat 里的 python 命令能起（health 256K 正常）证明脚本本身没问题，坏的是那层间接。
    // 解法：用一次性计划任务在原生上下文里跑 bat（与 ssh 会话彻底解耦），bat 的内层 start 在原生 cmd 下正常，
    // 实测进程树 python×2+strata.exe+strata-vision.exe 齐全、加载完成后 8080=256K。任务用完即删。
    const TN = 'DSH_Strata_' + Date.now().toString(36);
    const crt = await strataSshExec('schtasks /create /tn ' + TN + ' /tr "cmd /c ' + cfg.bat + '" /sc once /st 00:00 /f', 12000);
    if (!crt.ok) return reply(200, { action: act, ok: false, msg: '创建一次性计划任务失败：' + (crt.msg || '') });
    const run = await strataSshExec('schtasks /run /tn ' + TN, 12000);
    await strataSshExec('schtasks /delete /tn ' + TN + ' /f', 10000); // 触发后即删（任务已入队执行，删定义不影响）
    if (!run.ok) return reply(200, { action: act, ok: false, msg: '触发计划任务失败：' + (run.msg || '') });
    // 自校验：schtasks 拉起后进程树应在数秒内出现（模型加载还要 2~6 分钟，此刻 8080 未必监听属正常，
    // 故以「python/strata 进程是否出现」为准，不以端口为准）。
    await new Promise((res2) => setTimeout(res2, 6000));
    const chk = await strataSshExec('tasklist /nh /fo csv', 10000);
    const hasProc = chk.ok && /pythonw?\.exe|strata(\.exe|-vision\.exe)/i.test(chk.msg || '');
    return reply(200, { action: act, ok: !!hasProc, msg: hasProc
      ? (killed ? '已停掉旧实例并经计划任务启动 ' : '已通过计划任务启动 ') + cfg.bat + '（进程树已起，模型加载 2~6 分钟后 8080 自动可用）'
      : '计划任务已触发，但 6 秒后未见 python/strata 进程——可能被杀软拦截或 bat 报错，请看远端桌面窗口/srv_256k.log' });
  }
  return reply(400, { ok: false, msg: 'action 必须是 wake | shutdown | startbat' });
}
function strataRemoteHandle(req, res, urlObj) {
  const reply = (code, obj) => { res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8' }); res.end(JSON.stringify(obj)); };
  if (req.method === 'POST' && urlObj.pathname === '/v1/internal/strata-remote/cmd') return strataRemoteCmd(req, res);
  const all = urlObj.searchParams.get('requests') === 'all';
  return strataRemoteSnapshot(all).then((d) => reply(200, d))
    .catch((e) => reply(200, { ok: false, msg: String((e && e.message) || e), ts: Date.now() }));
}
// ==== [strata-remote-monitor] END module ====

// ====== [gpu-ctl 1003] GPU 功耗/频率控制模块（硬件监视页，模式对齐 CPU CTL）======
// 提权边界：只 execFile 固定路径 /usr/local/bin/gpu-ctl（root:root 0755，sudoers 单命令白名单），
// 参数全部白名单 action + 正则数值校验，杜绝注入；nvidia-smi 本体不放行 sudo。
// 铁律（09-20）：execFile + 自身 timeout，绝不 execSync；gpu-ctl 内部所有 nvidia-smi
// 再包一层 timeout 8，双保险。连续失败不熔断（低频人工操作），但失败原样回显给用户。
const GPU_CTL = process.env.GPU_CTL || '/usr/local/bin/gpu-ctl';
const GPU_W_RE = /^([5-9][0-9]|[1-4][0-9][0-9]|500)$/;   // 50-500 整数瓦（真实区间再按驱动 min/max 收窄）
const GPU_MHZ_RE = /^([1-9][0-9]{2}|[1-3][0-9]{3}|4000)$/; // 100-4000 整数 MHz
const GPU_IDX_RE = /^(all|[0-7])$/;

function gpuCtlRun(args, timeoutMs) {
  return new Promise((resolve) => {
    require('child_process').execFile(GPU_CTL, args.map(String), { timeout: timeoutMs || 20000, maxBuffer: 2 * 1024 * 1024 },
      (err, stdout, stderr) => {
        const out = String(stdout || '').trim();
        const errText = String(stderr || '').trim();
        resolve({ ok: !err, output: out || errText || (err ? String(err.message || err) : '(无输出)'), killed: !!(err && err.killed) });
      });
  });
}

// 状态短缓存：UI 1s 轮询（多标签页叠发），TTL 1.5s + 单飞。sudo+python 一趟 ~200ms，
// 命令成功后调用方带 force=true 绕缓存拿即时回显。
let __gpuCtlCache = null; // { t, state }
let __gpuCtlFly = null;
function gpuCtlStatus(force) {
  if (!force && __gpuCtlCache && Date.now() - __gpuCtlCache.t < 1500) {
    return Promise.resolve({ ok: true, state: __gpuCtlCache.state });
  }
  if (__gpuCtlFly) return __gpuCtlFly;
  __gpuCtlFly = gpuCtlRun(['status', '--json'], 15000).then((r) => {
    __gpuCtlFly = null;
    if (!r.ok) return { ok: false, error: r.output };
    try {
      const state = JSON.parse(r.output);
      if (state && state.supported) __gpuCtlCache = { t: Date.now(), state };
      return { ok: true, state };
    } catch (e) {
      return { ok: false, error: 'gpu-ctl status 输出无法解析：' + String(r.output).slice(0, 200) };
    }
  }).catch((e) => { __gpuCtlFly = null; return { ok: false, error: String((e && e.message) || e) }; });
  return __gpuCtlFly;
}

// action 白名单 → gpu-ctl 参数（返回 null = 非法）
function gpuCtlAction(body) {
  const a = String(body.action || '');
  const gi = (s) => (GPU_IDX_RE.test(String(s == null ? 'all' : s)) ? String(s == null ? 'all' : s) : null);
  switch (a) {
    case 'pl': {
      const w = String(body.watt == null ? '' : body.watt);
      if (!GPU_W_RE.test(w)) return null;
      const g = gi(body.gpu);
      if (!g) return null;
      return { args: ['pl', w, g], timeout: 20000, watt: parseInt(w, 10) };
    }
    case 'lock': {
      const lo = String(body.lo == null ? '' : body.lo);
      const hi = String(body.hi == null ? '' : body.hi);
      if (!GPU_MHZ_RE.test(lo) || !GPU_MHZ_RE.test(hi)) return null;
      if (parseInt(lo, 10) > parseInt(hi, 10)) return null;
      const g = gi(body.gpu);
      if (!g) return null;
      return { args: ['lock', lo, hi, g], timeout: 20000 };
    }
    case 'unlock': {
      const g = gi(body.gpu);
      if (!g) return null;
      return { args: ['unlock', g], timeout: 20000 };
    }
    default: return null;
  }
}

async function gpuCtlHandle(req, res, urlObj) {
  const reply = (code, obj) => { res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8' }); res.end(JSON.stringify(obj)); };
  const sub = urlObj.pathname.slice('/v1/internal/gpu-ctl'.length);
  if (req.method === 'GET' && sub === '') {
    return reply(200, await gpuCtlStatus());
  }
  if (req.method === 'POST' && sub === '/cmd') {
    const body = await cpuCtlReadBody(req);
    if (!body) return reply(400, { ok: false, msg: '请求体不是合法 JSON' });
    const act = gpuCtlAction(body);
    if (!act) return reply(400, { ok: false, msg: '非法参数或不在白名单内（pl 50-500W / lock 100-4000MHz / gpu=all|0-7）' });
    const r = await gpuCtlRun(act.args, act.timeout);
    let persist = null;
    if (r.ok && act.watt) {
      // 功耗上限联动开机持久化（09-26 铁律）：gpu-ctl persist-pl 写 drop-in + daemon-reload。
      // 失败不回滚运行时值（已生效），只在 msg 里如实说明持久化状态。
      persist = await syncGpuPlDropin(act.watt);
      if (!persist) savePowerConfig(Object.assign(loadPowerConfig(), { gpuPlW: act.watt }));
    }
    const st = await gpuCtlStatus(true); // 命令改过状态，强制绕缓存补拉
    const note = act.watt ? (persist ? ('（运行时已生效；开机持久化失败：' + persist + '）') : '（运行时已生效，并已写入开机持久化 pl-console.conf）') : null;
    return reply(200, {
      ok: r.ok,
      msg: r.ok ? (r.output + (note || '')) : (r.output + (r.killed ? '（超时被放弃）' : '')),
      persisted: act.watt ? !persist : null,
      state: st.ok ? st.state : null,
      error: st.ok ? null : st.error,
    });
  }
  return reply(404, { ok: false, msg: 'gpu-ctl: 未知路由' });
}

// ====== [gpu-hotshield 1009] GPU 高温自动关机守护 ======
// 需求：任一张 GPU 温度 ≥ 阈值并持续一段时间 → 自动关闭当前电脑（控制台所在机）。
// 设计：
//  · 数据源复用 GPU 采样器 global.__gpuLive（1s 异步采样，09-20 铁律通道，不新增 nvidia-smi 压力）；
//    比较口径 = max(核心温度 temp, 显存温度 temp_mem)——显存热点往往是真正的损伤源。
//  · 双重防误触发：① 迟滞（降到 阈值-回差 以下才清空计时，抖动不反复重置）；
//    ② 持续时间 holdSec 达标才行动；触发后先进入 graceSec 倒计时（页面大字告警 + 每秒刷新剩余秒数），
//    倒计时内可在页面「取消关机」，也可随时降温自动取消；倒计时结束走 /usr/local/bin/gpu-ctl halt
//    （root 提权脚本内 sync 落盘 + systemctl poweroff，异步 execFile + 自建定时器结算，10-07 铁律）。
//  · 触发即停 vLLM/SGLang 推理实例（脚本化模型走各自 stop 脚本，其余 stopVllm 优雅停止，
//    KV 池随之释放；整段限时 45s，超时直接进入关机，不因停服卡死耽误保护动作），再关机。
//  · 配置热加载（mtime 判据，同 strata-remote.json 模式）：gpu-hotshield.json
//    {enabled, thresholdC(40-105), holdSec(5-3600), graceSec(0-600), cooldownMin(0-1440)}
//  · 状态落盘 gpu-hotshield-state.json（lastTriggeredTs/lastAction），重启后可追溯；日志 [hotshield] 前缀。
//  · 安全底线：评估循环整体 try/catch，任何一步异常只记录不动作；GPU 数据缺失/过期（>15s）不参与判定
//    （宁可不触发，绝不在盲态下关机）；cooldownMin 防"关机→开机→仍热→再关机"振荡。
const HOTSHIELD_CONFIG_PATH = path.join(__dirname, 'gpu-hotshield.json');
const HOTSHIELD_STATE_PATH = path.join(__dirname, 'gpu-hotshield-state.json');
const HS_DEFAULTS = { enabled: false, thresholdC: 92, holdSec: 30, graceSec: 60, cooldownMin: 30 };
let __hsCfg = Object.assign({ mtime: -1 }, HS_DEFAULTS);
function hotShieldCfg() {
  try {
    const st = fs.statSync(HOTSHIELD_CONFIG_PATH);
    if (st.mtimeMs !== __hsCfg.mtime) {
      const j = JSON.parse(fs.readFileSync(HOTSHIELD_CONFIG_PATH, 'utf8'));
      const clamp = (v, lo, hi, dft) => { const n = parseFloat(v); return Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : dft; };
      __hsCfg = {
        mtime: st.mtimeMs,
        enabled: !!j.enabled,
        thresholdC: clamp(j.thresholdC, 40, 105, HS_DEFAULTS.thresholdC),
        holdSec: clamp(j.holdSec, 5, 3600, HS_DEFAULTS.holdSec),
        graceSec: clamp(j.graceSec, 0, 600, HS_DEFAULTS.graceSec),
        cooldownMin: clamp(j.cooldownMin, 0, 1440, HS_DEFAULTS.cooldownMin),
      };
    }
  } catch (e) { /* 文件缺失/损坏：保持上一次值（首跑即缺省 disabled） */ }
  return __hsCfg;
}
function hotShieldSaveCfg(patch) {
  const cur = hotShieldCfg();
  const next = Object.assign({}, HS_DEFAULTS, cur, patch, { mtime: -1 });
  const out = { enabled: !!next.enabled, thresholdC: next.thresholdC, holdSec: next.holdSec, graceSec: next.graceSec, cooldownMin: next.cooldownMin };
  fs.writeFileSync(HOTSHIELD_CONFIG_PATH, JSON.stringify(out, null, 2));
  return hotShieldCfg(); // 立即重载（mtime 变了必然重读）
}
function hotShieldLoadState() {
  try { return JSON.parse(fs.readFileSync(HOTSHIELD_STATE_PATH, 'utf8')) || {}; } catch (e) { return {}; }
}
function hotShieldSaveState(s) {
  try { fs.writeFileSync(HOTSHIELD_STATE_PATH, JSON.stringify(s, null, 2)); } catch (e) {}
}
// 守护运行时状态（内存态，重启即清；lastTriggeredTs 从 state 文件恢复用于冷却判定）
const __hs = {
  aboveSince: 0, peakAbove: null, hottestIdx: null,
  countdownEndsAt: 0, countdownPeak: null, cancelPending: false,
  firing: false, firedOnce: false, lastLogAt: 0,
};
// 停服阶段的叫停闸门：关机倒计时进入尾声时 abort 掉仍在跑的停止脚本，优先保证 halt 按时发出
const __hsAbort = (() => { try { return new AbortController(); } catch (e) { return { signal: undefined, abort() {} }; } })();
function hsNowCooldownLeftMs() {
  const cfg = hotShieldCfg();
  const st = hotShieldLoadState();
  const until = (st.lastActionTs || 0) + cfg.cooldownMin * 60000;
  return Math.max(0, until - Date.now());
}
function hsEvaluate(frame, now) {
  // frame = global.__gpuLive（{ts, gpus:[{index,temp,temp_mem,...}]}）
  const cfg = hotShieldCfg();
  if (!cfg.enabled) { __hs.aboveSince = 0; __hs.countdownEndsAt = 0; __hs.firedOnce = false; return; }
  if (__hs.firing) return; // 正在执行停服+关机流程，不再重复判定
  const gpus = (frame && Array.isArray(frame.gpus)) ? frame.gpus.filter((g) => g && g.index != null) : [];
  const fresh = !!gpus.length && !!frame.ts && (now - frame.ts) < 15000;
  let hottest = null, hottestIdx = null;
  if (fresh) {
    for (const g of gpus) {
      const t = Math.max(g.temp == null ? -Infinity : g.temp, g.temp_mem == null ? -Infinity : g.temp_mem);
      if (t === -Infinity) continue;
      if (hottest == null || t > hottest) { hottest = t; hottestIdx = g.index; }
    }
  }
  __hs.hottestIdx = hottestIdx;
  __hs.peakAbove = hottest;
  const cdLeft = hsNowCooldownLeftMs();
  if (!fresh || hottest == null) {
    // 数据缺失/过期：保守清空计时（宁可漏触发，不在盲态积累触发）
    __hs.aboveSince = 0;
    if (__hs.countdownEndsAt) { hsCancelCountdown('GPU 温度数据缺失或过期（守护转入盲态，取消倒计时）'); }
    return;
  }
  if (hottest >= cfg.thresholdC) {
    if (!__hs.aboveSince) __hs.aboveSince = now;
  } else if (hottest <= cfg.thresholdC - 2) {
    __hs.aboveSince = 0; // 迟滞：降回阈值-2°C 才算脱离高温
    if (__hs.countdownEndsAt) hsCancelCountdown('温度已回落至 ' + Math.round(hottest) + '°C（低于阈值 ' + cfg.thresholdC + '°C），自动取消');
    __hs.firedOnce = false;
  }
  if (cdLeft > 0) return; // 冷却期内不再武装（防关机-开机振荡）
  const heldMs = __hs.aboveSince ? now - __hs.aboveSince : 0;
  if (!__hs.countdownEndsAt && !__hs.firedOnce && heldMs >= cfg.holdSec * 1000) {
    hsStartCountdown(cfg, hottest, hottestIdx, now);
  }
  if (__hs.countdownEndsAt && now >= __hs.countdownEndsAt) hsFire(now);
}
function hsStartCountdown(cfg, hottest, idx, now) {
  if (cfg.graceSec <= 0) { hsFire(now); return; }
  __hs.countdownEndsAt = now + cfg.graceSec * 1000;
  __hs.countdownPeak = hottest;
  console.warn('[hotshield] ⚠ GPU' + idx + ' 温度 ' + Math.round(hottest) + '°C ≥ 阈值 ' + cfg.thresholdC + '°C 持续 ' + Math.round(cfg.holdSec) + 's，' + cfg.graceSec + 's 后自动关机（可在页面取消）');
  hotShieldPushEvent('warn', 'GPU' + idx + ' 已达 ' + Math.round(hottest) + '°C（阈值 ' + cfg.thresholdC + '°C），' + cfg.graceSec + ' 秒后自动关机——请立即保存工作或点击「取消关机」');
}
function hsCancelCountdown(reason) {
  console.log('[hotshield] ✔ 已取消自动关机：' + reason);
  __hs.countdownEndsAt = 0;
  hotShieldPushEvent('info', '自动关机已取消：' + reason);
}
function hotShieldPushEvent(level, msg) {
  const st = hotShieldLoadState();
  const ev = { ts: Date.now(), level, msg: String(msg).slice(0, 300) };
  const hist = Array.isArray(st.events) ? st.events : [];
  hist.push(ev);
  while (hist.length > 50) hist.shift();
  st.events = hist;
  hotShieldSaveState(st);
}
async function hsFire(now) {
  if (__hs.firing) return;
  __hs.firing = true;
  __hs.countdownEndsAt = 0;
  __hs.firedOnce = true;
  const st0 = hotShieldLoadState();
  st0.lastTriggeredTs = now;
  hotShieldSaveState(st0);
  const cfg = hotShieldCfg();
  console.error('[hotshield] 🔥 触发高温保护：停止推理实例并关机（阈值 ' + cfg.thresholdC + '°C）');
  hotShieldPushEvent('fire', '高温保护触发：正在停止 vLLM/SGLang 实例并执行关机');
  // ① 尽力停服（限时 45s 总预算：高温场景下时间就是损失，超时直接关机）
  try {
    const stopAll = (async () => {
      const seen = new Set();
      const ports = [];
      for (const [, inst] of (global.__GPU_INSTANCES instanceof Map ? global.__GPU_INSTANCES : [])) {
        if (inst && inst.port != null && !seen.has(inst.port)) { seen.add(inst.port); ports.push({ port: inst.port, runtime: inst.runtime || 'vllm' }); }
      }
      for (const p of [...(global.__vllmModelPorts instanceof Set ? global.__vllmModelPorts : []), ...(global.__sglangModelPorts instanceof Set ? global.__sglangModelPorts : [])]) {
        if (p != null && !seen.has(p)) { seen.add(p); ports.push({ port: p, runtime: 'vllm' }); }
      }
      for (const { port, runtime } of ports) {
        try {
          const smStop = scriptModelForPort(port);
          if (smStop) {
            // 脚本化模型（chroot root 引擎）：只能走宿主停止脚本；abort signal 尽力中途叫停
            const stopPath = resolveStopScript(smStop);
            if (stopPath && fs.existsSync(stopPath)) {
              await execFileAsync('bash', [stopPath, String(smStop.port)], { encoding: 'utf8', timeout: 20000, maxBuffer: 1024 * 1024, signal: __hsAbort.signal }).catch(() => {});
            }
          } else {
            await new Promise((resolve) => {
              let done = false;
              const fin = () => { if (!done) { done = true; resolve(); } };
              try { stopVllm(fin, port); } catch (e) { fin(); }
              setTimeout(fin, 20000).unref();
            });
          }
          try { global.__GPU_INSTANCES && global.__GPU_INSTANCES.delete(port); } catch (e) {}
          console.warn('[hotshield] 停服完成：端口 ' + port);
        } catch (e) { console.warn('[hotshield] 停服端口 ' + port + ' 异常（不阻断关机）：' + String((e && e.message) || e)); }
      }
    })();
    const budget = new Promise((resolve) => setTimeout(resolve, 45000).unref());
    await Promise.race([stopAll, budget]);
  } catch (e) { console.warn('[hotshield] 停服阶段异常（不阻断关机）：' + String((e && e.message) || e)); }
  // ② 关机：走 gpu-ctl halt（root：sync 落盘 + systemctl poweroff）
  const r = await gpuCtlRun(['halt'], 15000);
  console.error('[hotshield] 关机指令结果：' + (r.ok ? '已下发' : '失败—' + String(r.output).slice(0, 200)));
  hotShieldPushEvent(r.ok ? 'fire' : 'error', r.ok ? '关机指令已下发（sync + systemctl poweroff）' : '关机指令失败：' + String(r.output).slice(0, 200));
  const st1 = hotShieldLoadState();
  st1.lastActionTs = Date.now();
  st1.lastAction = r.ok ? 'poweroff' : 'failed';
  hotShieldSaveState(st1);
  __hs.firing = false;
  __hs.aboveSince = 0;
}
// 评估循环：1s 一拍（与 __gpuLive 采样同频）；关机倒计时进行中额外把剩余秒数广播给 GET 状态端点
function hotShieldTick() {
  try { hsEvaluate(global.__gpuLive || null, Date.now()); } catch (e) { console.warn('[hotshield] 评估异常（忽略本拍）：' + String((e && e.message) || e)); }
}
if (!global.__hotShieldTimer) {
  global.__hotShieldTimer = setInterval(hotShieldTick, 1000);
  if (global.__hotShieldTimer.unref) global.__hotShieldTimer.unref();
}
function hotShieldStatus() {
  const cfg = hotShieldCfg();
  const now = Date.now();
  const armed = !!(cfg.enabled && !hsNowCooldownLeftMs());
  return {
    ok: true,
    config: { enabled: cfg.enabled, thresholdC: cfg.thresholdC, holdSec: cfg.holdSec, graceSec: cfg.graceSec, cooldownMin: cfg.cooldownMin },
    state: {
      armed,
      cooling_down: hsNowCooldownLeftMs() > 0,
      cooldown_left_s: Math.ceil(hsNowCooldownLeftMs() / 1000),
      holding_s: __hs.aboveSince ? Math.floor((now - __hs.aboveSince) / 1000) : 0,
      peak_above: __hs.peakAbove,
      hottest_idx: __hs.hottestIdx,
      counting_down: !!__hs.countdownEndsAt,
      seconds_left: __hs.countdownEndsAt ? Math.max(0, Math.ceil((__hs.countdownEndsAt - now) / 1000)) : 0,
      firing: __hs.firing,
      live_age_s: (global.__gpuLive && global.__gpuLive.ts) ? Math.round((now - global.__gpuLive.ts) / 1000) : null,
      last_triggered_ts: hotShieldLoadState().lastTriggeredTs || null,
      events: (hotShieldLoadState().events || []).slice(-10).reverse(),
    },
  };
}
async function hotShieldCmd(req, res) {
  const reply = (code, obj) => { res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8' }); res.end(JSON.stringify(obj)); };
  const body = await cpuCtlReadBody(req);
  if (!body) return reply(400, { ok: false, msg: '请求体不是合法 JSON' });
  const act = String(body.action || '');
  if (act === 'cancel') {
    if (__hs.countdownEndsAt) hsCancelCountdown('用户在页面手动取消（关机前 ' + Math.max(0, Math.ceil((__hs.countdownEndsAt - Date.now()) / 1000)) + ' 秒）');
    else hotShieldPushEvent('info', '收到取消请求（当前并无进行中的倒计时）');
    return reply(200, Object.assign({ action: act, ok: true, msg: __hs.countdownEndsAt ? '已取消' : '当前没有进行中的关机倒计时' }, { state: hotShieldStatus().state }));
  }
  if (act === 'save') {
    const patch = {};
    if (body.enabled != null) patch.enabled = !!body.enabled;
    if (body.thresholdC != null) { const v = parseFloat(body.thresholdC); if (!(v >= 40 && v <= 105)) return reply(400, { ok: false, msg: 'thresholdC 需在 40~105 之间' }); patch.thresholdC = v; }
    if (body.holdSec != null) { const v = parseInt(body.holdSec, 10); if (!(v >= 5 && v <= 3600)) return reply(400, { ok: false, msg: 'holdSec 需在 5~3600 之间' }); patch.holdSec = v; }
    if (body.graceSec != null) { const v = parseInt(body.graceSec, 10); if (!(v >= 0 && v <= 600)) return reply(400, { ok: false, msg: 'graceSec 需在 0~600 之间' }); patch.graceSec = v; }
    if (body.cooldownMin != null) { const v = parseInt(body.cooldownMin, 10); if (!(v >= 0 && v <= 1440)) return reply(400, { ok: false, msg: 'cooldownMin 需在 0~1440 之间' }); patch.cooldownMin = v; }
    hotShieldSaveCfg(patch);
    // 关闭开关时同步清空运行时计时，避免"关了开关还剩个倒计时"
    if (patch.enabled === false) { __hs.aboveSince = 0; __hs.countdownEndsAt = 0; __hs.firedOnce = false; }
    console.log('[hotshield] 配置已更新：' + JSON.stringify(hotShieldCfg()).replace(/"mtime":-?\d+,/, ''));
    return reply(200, Object.assign({ action: act, ok: true, msg: '设置已保存并即时生效' }, hotShieldStatus()));
  }
  return reply(400, { ok: false, msg: 'action 必须是 save | cancel' });
}
function hotShieldHandle(req, res, urlObj) {
  const reply = (code, obj) => { res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8' }); res.end(JSON.stringify(obj)); };
  const sub = urlObj.pathname.slice('/v1/internal/hotshield'.length);
  if (req.method === 'POST' && sub === '/cmd') return hotShieldCmd(req, res);
  return reply(200, hotShieldStatus());
}
// ==== [gpu-hotshield 1009] END module ====

const server = http.createServer(async (req, res) => {

  let urlObj;
  try {
    // 恶意/畸形 Host 头（含空格、缺失等）会让 new URL 抛 Invalid URL。
    // 若不加保护，异常会从请求回调里冒出去直接打崩整个进程（一次请求即可 DoS）。
    urlObj = new URL(req.url, `http://${req.headers.host}`);
  } catch (e) {
    try { urlObj = new URL(req.url, 'http://localhost'); } catch (e2) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Bad Request' }));
      return;
    }
  }
  const pathname = urlObj.pathname;

  // 09-20：内部 API/页面/静态资源启用 gzip；代理路径（chat/completions 流式）绝不压缩
  if (pathname === '/' || pathname === '/index.html' || pathname === '/m' || pathname === '/mobile.html' || pathname === '/bench.html' || pathname === '/cpu.html' || pathname === '/sglang.html' || pathname === '/vllm.html' || pathname === '/strata-monitor.html'
      || pathname.startsWith('/static/') || pathname.startsWith('/v1/internal/')) {
    installGzip(req, res);
  }

  // 09-20：管理口令启用时，拦截内部 API 的写操作（401 由前端弹窗补录口令重试）
  if (req.method === 'POST' && pathname.startsWith('/v1/internal/')) {
    const _tok = getConsoleToken();
    if (_tok) {
      const _given = String(req.headers['x-console-token'] || urlObj.searchParams.get('ct') || '');
      if (_given !== _tok) {
        res.writeHead(401, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: 'unauthorized', need_token: true }));
        return;
      }
    }
  }

  // === Bench Console API（09-23 移植，详见 BENCH 模块头注释）===
  if (pathname === '/v1/internal/bench' || pathname.startsWith('/v1/internal/bench/')) {
    return BENCH.handleApi(req, res, urlObj).catch((e) => {
      try { res.writeHead(500, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ error: String((e && e.message) || e) })); } catch (_) {}
    });
  }

  // === CPU 控制 API（移植自 bench-console/cpu-control，见 CPU CTL 模块头注释）===
  if (pathname === '/v1/internal/cpuctl' || pathname.startsWith('/v1/internal/cpuctl/')) {
    return cpuCtlHandle(req, res, urlObj).catch((e) => {
      try { res.writeHead(500, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ ok: false, error: String((e && e.message) || e) })); } catch (_) {}
    });
  }

  // === Strata 远端监控 API（克隆 .38:8080 Web UI Monitor 页，见 [strata-remote-monitor] 模块）===
  // 注册在 /v1/internal/strata 前缀路由之前；只匹配 strata-remote 与 strata-remote/cmd，不吞 strata/* 子路由。
  if (pathname === '/v1/internal/strata-remote' || pathname === '/v1/internal/strata-remote/cmd') {
    return strataRemoteHandle(req, res, urlObj).catch((e) => {
      try { res.writeHead(500, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ ok: false, error: String((e && e.message) || e) })); } catch (_) {}
    });
  }

  // ==== [strata-console] BEGIN route ====
  // === Strata 推理引擎管理 API（契约见 STRATA-API.md，模块实现见上方 strata-console 块）===
  if (pathname === '/v1/internal/strata' || pathname.startsWith('/v1/internal/strata/')) {
    return strataHandle(req, res, urlObj).catch((e) => {
      try { res.writeHead(500, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ ok: false, error: String((e && e.message) || e) })); } catch (_) {}
    });
  }
  // ==== [strata-console] END route ====
  // === Internal API: Model Running Parameters ===
  if (pathname === '/v1/internal/model-params') {
    // 采样参数回落：命令行未显式传采样参数时，读模型目录的 generation_config.json。
    // 脚本化启动实例（如 Flash-Next chroot 链路）按设计不下发 --override-generation-config，
    // 实际默认采样值由模型自带 generation_config 决定（vLLM 运行时同款来源），
    // 否则前端「采样参数」卡会全部显示 '--'。
    function applyGenConfigDefaults(p, cmdline) {
      try {
        let mp = null;
        for (let i = 0; i < cmdline.length; i++) {
          const a = cmdline[i];
          if (a === '--model' || a === '--model-path') { mp = cmdline[i + 1]; break; }
          if (a === 'serve' && cmdline[i + 1] && !String(cmdline[i + 1]).startsWith('-')) { mp = cmdline[i + 1]; break; }
        }
        if (!mp) return;
        if (!global.__genCfgCache) global.__genCfgCache = new Map();
        let hit = global.__genCfgCache.get(mp);
        if (!hit || Date.now() - hit.t > 10000) {
          let data = null;
          try { data = JSON.parse(fs.readFileSync(mp + '/generation_config.json', 'utf8')); } catch (e) {}
          hit = { t: Date.now(), data };
          global.__genCfgCache.set(mp, hit);
        }
        const g = hit.data;
        if (!g) return;
        let filled = false;
        if (p.temperature == null && typeof g.temperature === 'number') { p.temperature = g.temperature; filled = true; }
        if (p.top_p == null && typeof g.top_p === 'number') { p.top_p = g.top_p; filled = true; }
        if (p.top_k == null && typeof g.top_k === 'number') { p.top_k = g.top_k; filled = true; }
        // min_p / repetition_penalty：generation_config 没有时按 vLLM SamplingParams 全局默认
        if (p.min_p == null) { p.min_p = (typeof g.min_p === 'number') ? g.min_p : 0.0; filled = true; }
        if (p.repetition_penalty == null) { p.repetition_penalty = (typeof g.repetition_penalty === 'number') ? g.repetition_penalty : 1.0; filled = true; }
        if (filled) p.sampling_source = 'generation_config';
      } catch (e) {}
    }
    // [ple-display 0923→0930] PLE n-gram 表精度与驻留位置（Flash-Next 专属，两套栈都判）。
    // 数据源=日志里的加载判据行，**绝不回读弹窗/FN_* 请求值**：官方 0.30.0 新栈没有
    // INT8 与磁盘加载器（inner 只 export VLLM_PLE_CPU_OFFLOAD=1 ⇒ BF16 锁页一档），
    // 旧栈的 FN_PLE_INT8/FN_PLE_LOC 在新栈是 NOOP（见 flash-next-0300-inner.sh NOOP_NOTE），
    // 只有日志是"显示即真值"。引擎进程属 root（旧栈 chroot / 新栈 sudo），
    // /proc/<pid>/environ 读不了、cmdline 也不含该信息。反向扫尾 64MB，最近一条判据行赢
    // （= 本次启动的形态；重启进行中时新 echo 晚于上次引擎行，同样赢）。10s TTL 缓存，
    // 前端 500ms 轮询不至于反复读日志。
    // 精度：bf16 / int8 / fp16 / fp8；驻留四态：
    //   disk   = mmap 页缓存（可回收）
    //   heap   = 匿名堆（不可回收）
    //   pinned = 锁页主机内存（cuMemHostRegister，不可回收且不可换出）← 0.30.0 新栈唯一档
    //   gpu    = 显存驻留（VLLM_PLE_CPU_OFFLOAD=0，本机必 OOM，仅口径完备）
    const pleGib = (line) => { const m = /(\d+(?:\.\d+)?)\s*GiB/.exec(line); return m ? parseFloat(m[1]) : null; };
    const pleDtype = (s) => {
      const x = String(s || '').toLowerCase();
      if (/bfloat16|bf16/.test(x)) return 'bf16';
      if (/float16|fp16|half/.test(x)) return 'fp16';
      if (/int8/.test(x)) return 'int8';
      if (/float8|fp8/.test(x)) return 'fp8';
      return x.replace(/^torch\./, '') || null;
    };
    // 单行判据 → {dtype, loc, gib}；非判据行返回 null。顺序=信息量：引擎实建张量行
    // 最权威（含自校验回落后的真实形态），其次分配行，最后 inner echo（启动意图，
    // 覆盖"正在加载、引擎行还没写出来"的窗口）。
    function matchPleLine(line) {
      // —— [anon-ple 1010] 0.31 栈引擎实建权威行（本次启动真身，最先判）——
      if (line.indexOf('PLE anon-heap storage attached') >= 0) {
        return { dtype: 'int8', loc: 'heap', gib: pleGib(line), resident: true };
      }
      if (line.indexOf('PLE mmap storage attached') >= 0) {
        return { dtype: 'int8', loc: 'disk', gib: pleGib(line), mmap: true };
      }
      // inner echo（匿名堆档；覆盖"正在加载、引擎行还没写出来"的窗口）：
      if (line.indexOf('[FN-PLE] INT8 \u5185\u5b58\u9a7b\u7559') >= 0) {
        const gz = /^[^\uff08(]*[\uff08(](\d+(?:\.\d+)?)\+/.exec(line);
        return { dtype: 'int8', loc: 'heap', gib: gz ? parseFloat(gz[1]) : pleGib(line), resident: true };
      }
      // —— 新栈（官方 vLLM 0.30.0 + rt-patch）——
      // ① 引擎权威行：Initialized PLE embedding ... weight_dtype=torch.bfloat16,
      //    weight_device=cpu, pinned=True   （ngram_embedding.py）
      if (line.indexOf('Initialized PLE embedding') >= 0) {
        const dt = pleDtype((/weight_dtype=([A-Za-z0-9_.]+)/.exec(line) || [])[1]);
        const dev = (/weight_device=([A-Za-z0-9_.]+)/.exec(line) || [])[1] || 'cpu';
        const loc = /^(cuda|gpu)/.test(dev) ? 'gpu' : (/pinned=True/.test(line) ? 'pinned' : 'heap');
        return { dtype: dt || 'bf16', loc, gib: pleGib(line) };
      }
      // ② 分配行：[rt-patch] PLE pinned alloc: 95.368 GiB registered in 2 chunk(s)。
      //    必须含 registered——">60 GiB tables go through chunked" 那条是能力提示行，不是实建。
      if (line.indexOf('PLE pinned alloc:') >= 0 && line.indexOf('registered') >= 0) {
        return { dtype: 'bf16', loc: 'pinned', gib: pleGib(line) };
      }
      // ③ 新栈 inner echo：[FN-0300] PLE 表：官方 BF16 锁页（pinned CPU 95.4 GiB…）
      if (line.indexOf('[FN-0300] PLE') >= 0) {
        return { dtype: 'bf16', loc: line.indexOf('显存') >= 0 ? 'gpu' : 'pinned', gib: pleGib(line) };
      }
      // [ple-loc-0310-1006] 新栈（0.31.0）inner echo：带体积，可作为 gib 来源
      if (line.indexOf('[FN-PLE] INT8 \u78c1\u76d8 mmap') >= 0) {
        // echo 文案是「（47.7+0.6 GiB …）」：表体积是加号前那个数，pleGib 会取到 0.6，需专用提取
        const g0 = /^[^\uff08(]*[\uff08(](\d+(?:\.\d+)?)\+/.exec(line);  // [anon-ple 1010] 半/全角括号均可
        return { dtype: 'int8', loc: 'disk', gib: g0 ? parseFloat(g0[1]) : pleGib(line) };
      }
      if (line.indexOf('[FN-PLE] \u5b98\u65b9 pinned-host') >= 0) return { dtype: 'bf16', loc: 'pinned', gib: pleGib(line) };
      // —— 旧栈（自研镜像的 INT8 / 磁盘驻留加载器）——
      if (line.indexOf('[FN-PLE-') < 0) return null;
      // 引擎真值行（晚于 inner echo，反向先命中）
      if (line.indexOf('[FN-PLE-INT8]') >= 0 && line.indexOf('n-gram table attached') >= 0) {
        return { dtype: 'int8', loc: line.indexOf('anonymous heap') >= 0 ? 'heap' : 'disk', gib: pleGib(line) };
      }
      if (line.indexOf('[FN-PLE-INT8MEM]') >= 0) return { dtype: 'int8', loc: 'heap', gib: pleGib(line) };
      if (line.indexOf('[FN-PLE-DISK]') >= 0 && line.indexOf('n-gram table attached') >= 0) {
        return { dtype: pleDtype((/dtype=([A-Za-z0-9_.]+)/.exec(line) || [])[1]) || 'bf16', loc: 'disk', gib: pleGib(line) };
      }
      // inner echo 行（BF16 内存驻留无引擎挂载行，靠它判定；旧文案"回退 BF16 匿名堆"同归 heap）
      if (line.indexOf('INT8 磁盘驻留') >= 0) return { dtype: 'int8', loc: 'disk', gib: pleGib(line) };
      if (line.indexOf('INT8 内存驻留') >= 0) return { dtype: 'int8', loc: 'heap', gib: pleGib(line) };
      if (line.indexOf('BF16 磁盘驻留') >= 0) return { dtype: 'bf16', loc: 'disk', gib: pleGib(line) };
      if (line.indexOf('内存驻留') >= 0 || line.indexOf('匿名堆') >= 0) return { dtype: 'bf16', loc: 'heap', gib: pleGib(line) };
      return null;
    }
    function readPleStatusCached(logPath) {
      if (!logPath) return null;
      if (!global.__pleStatusCache) global.__pleStatusCache = new Map();
      const hit = global.__pleStatusCache.get(logPath);
      if (hit && Date.now() - hit.t < 10000) return hit.ple;
      let ple = null;
      // [ple-loc-0310-1006] 同一次启动里是否出现 DSH PLE mmap 证据行（0.31 栈的
      // weight_device=cpu + pinned=False 既可能是匿名堆也可能是文件页缓存 mmap，
      // 只有这行能区分；引擎初始化行本身不带这个信息）
      let mmapSeen = false;
      const fdArr = [];
      try {
        const st = fs.statSync(logPath);
        const CHUNK = 1024 * 1024, MIN_POS = Math.max(0, st.size - 64 * 1024 * 1024);
        const fd = fs.openSync(logPath, 'r');
        fdArr.push(fd);
        let pos = st.size, carry = '';
        scan:
        while (pos > MIN_POS) {                       // 反向分块：找到的第一条=最近一次启动的判据
          const start2 = Math.max(MIN_POS, pos - CHUNK);
          const len = pos - start2;
          const buf = Buffer.alloc(len);
          fs.readSync(fd, buf, 0, len, start2);
          const parts = (buf.toString('utf8') + carry).split('\n');
          carry = parts.shift() || '';                // 块首残句，并入下一（更早）块
          for (let i = parts.length - 1; i >= 0; i--) {
            const ln = parts[i];
            if (!mmapSeen && (ln.indexOf('PLE mmap mode:') >= 0 || ln.indexOf('PLE mmap storage attached') >= 0)) mmapSeen = true; // [anon-ple 1010] class installed 两模式都装，不作文件档判据
            const m = matchPleLine(ln);
            if (!m) {
              // 越过本次启动的分段边界（更早的启动不再参与 mmapSeen 判定）
              if (ln.indexOf('Initializing a V1 LLM engine') >= 0 && ple) break scan;
              continue;
            }
            if (!ple) {                               // 最新一条=主判据（定 dtype/loc）
              ple = m;
              if (ple.gib != null) break scan;        // 大小齐了，收工
              continue;
            }
            // 主判据只缺大小（新栈①行不带 GiB）：往回第一条补上即可，
            // 再往回就是上一次启动，不再跨启动合并。
            if (ple.gib == null && m.gib != null) { ple.gib = m.gib; if (m.resident) ple.resident = true; }
            break scan;
          }
          pos = start2;
        }
      } catch (e) {}
      for (const fd of fdArr) { try { fs.closeSync(fd); } catch (e) {} }
      if (ple && ple.loc === 'heap' && mmapSeen && !ple.resident) {    // [ple-loc-0310-1006] 0.31 mmap 档改判（[anon-ple 1010] 匿名堆档不翻转）
        ple = Object.assign({}, ple, { loc: 'disk', mmap: true });
        if (ple.gib == null) {
          const g = /\((\d+(?:\.\d+)\+)\+?[\d.]*\s*GiB/.exec('') || null;
          ple.gib = null;   // 留给下面的 mmapGiB 兜底
        }
      }
      if (ple && ple.mmap && ple.gib == null) {
        // inner 的 [FN-PLE] 行带体积（47.7+0.6 GiB），补一次扫描
        try {
          const st2 = fs.statSync(logPath);
          const fd2 = fs.openSync(logPath, 'r');
          fdArr.push(fd2);
          const len2 = Math.min(st2.size, 8 * 1024 * 1024);
          const b2 = Buffer.alloc(len2);
          fs.readSync(fd2, b2, 0, len2, Math.max(0, st2.size - len2));
          const ls = b2.toString('utf8').split('\n');
          for (let i = ls.length - 1; i >= 0; i--) {
            if (ls[i].indexOf('[FN-PLE]') < 0) continue;
            const mm0 = /^[^\uff08(]*[\uff08(](\d+(?:\.\d+)?)\+/.exec(ls[i]);   // [anon-ple 1010] 表体积=括号里加号前那个数（半/全角均可）
            const g = mm0 ? parseFloat(mm0[1]) : pleGib(ls[i]);
            if (g != null && g > 1) { ple.gib = g; break; }
          }
        } catch (e) {}
      }
      global.__pleStatusCache.set(logPath, { t: Date.now(), ple });
      return ple;
    }
    try {
      const { execSync } = require('child_process');
      const query = require('url').parse(req.url, true).query;
      // 不带 port = 全实例模式：一次返回所有运行实例的参数（带 GPU 标签），
      // 前端「运行参数/采样参数」卡片按卡分块显示；带 port = 单实例（旧行为）。
      if (!query.port) {
        const all = [];
        const pushInst = (inst, runtime) => {
          try {
            const cmdline = fs.readFileSync(`/proc/${inst.pid}/cmdline`, 'utf8').split('\0').filter(Boolean);
            const p = parseServerParams(cmdline, runtime, inst.port);
            if (runtime === 'sglang') p.attention_backend = null; // sglang 无此参数，避免显示 vLLM 默认值
            else if (!p.vllm_version) p.vllm_version = detectVllmVersion(inst.pid, inst.port, cmdline); // [vllm-version-1006]
            applyGenConfigDefaults(p, cmdline); // 未显式传采样参数时回落模型 generation_config
            // [ple-display 0923] Flash-Next 脚本化实例：附 PLE 表精度/驻留（日志判据）。
            // 注意：同端口可能注册着多个脚本模型条目（18420 先后有 NVFP4/W4A16 两栈，
            // scriptModelForPort 取先到者会拿错日志）——这里枚举该端口全部条目，用进程
            // cmdline 的模型路径命中排序，逐个尝试解析直到命中（日志文件经
            // pickScriptModelLogFile 内容打分选定，防同端口换栈后拿旧文件说谎）。
            const pleSm = scriptModelsForPort(inst.port, cmdline);
            for (const cand of pleSm) {
              const lf = pickScriptModelLogFile(inst, cand) || cand.log;
              const ple = readPleStatusCached(lf);
              if (ple) { p.ple_table = ple; break; }
            }
            all.push({ ...p, port: inst.port, gpu: inst.gpu,
              gpus: inst.gpus || (inst.gpu != null ? [inst.gpu] : []),
              runtime, model: inst.servedName || inst.modelPath || '' });
          } catch (e) { /* 进程已退出 */ }
        };
        for (const inst of listVllmInstances()) pushInst(inst, 'vllm');
        for (const inst of listSglangInstances()) pushInst(inst, 'sglang');
        all.sort((a, b) => (((a.gpu == null) ? 999 : a.gpu) - ((b.gpu == null) ? 999 : b.gpu)) || (a.port - b.port));
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ instances: all }));
        return;
      }
      const wantPort = parseInt(query.port) || config.vllmPort;
      // 找到监听指定端口的服务进程（vllm serve / sglang.launch_server）
      let out = null;
      let runtime = null;
      try {
        const lsofOut = execSync(`lsof -ti:${wantPort} -sTCP:LISTEN 2>/dev/null || true`, { encoding: 'utf8', timeout: 5000 }).trim();
        for (const pid of lsofOut.split('\n')) {
          try {
            const cmd = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8').split('\0').join(' ');
            if (cmd.includes('vllm serve') || cmd.includes('vllm.entrypoints') || cmd.includes('python -m vllm')) { out = pid; runtime = 'vllm'; break; }
            if (cmd.includes('sglang.launch_server')) { out = pid; runtime = 'sglang'; break; }
          } catch (e) {}
        }
      } catch (e) {}
      if (!out) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ runtime: null }));
        return;
      }
      const cmdline = fs.readFileSync(`/proc/${out}/cmdline`, 'utf8').split('\0').filter(Boolean);
      const params = parseServerParams(cmdline, runtime, wantPort);
      applyGenConfigDefaults(params, cmdline); // 未显式传采样参数时回落模型 generation_config
      if (runtime === 'vllm' && !params.vllm_version) params.vllm_version = detectVllmVersion(parseInt(out) || null, wantPort, cmdline); // [vllm-version-1006]
      // [ple-display 0923] 单端口模式同样附 PLE 表状态（脚本化模型端口才有效）
      const smPle1 = scriptModelForPort(wantPort);
      if (smPle1) {
        const ple1 = readPleStatusCached(pickScriptModelLogFile({ pid: out, port: wantPort }, smPle1) || smPle1.log);
        if (ple1) params.ple_table = ple1;
      }
      // 思考模式：sglang 从 cmdline --default-chat-template-kwargs 读取（同 vllm）
      // 未传参时保持 null，前端显示「✅ 默认（开）」（模板缺省 enable_thinking=true）
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(params));
    } catch (e) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({}));
    }
    return;
  }

  // === Internal API: GPU Info ===
  if (pathname === '/v1/internal/gpu_info') {
    getGpuInfo((info) => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(info));
    });
    return;
  }


  // === Internal API: 关闭 127（18420 Flash-Next）[__close127_btn_1006__] ===
  if (pathname === '/v1/internal/close-127' && req.method === 'POST') {
    close127Instance().then((r) => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(r));
    }).catch((e) => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ success: false, error: String((e && e.message) || e) }));
    });
    return;
  }
  // === Internal API: Model Manager ===
  if (pathname === '/v1/internal/model-manager') {
    if (req.method === 'GET') {
      // List available models + all running instances (multi-backend)
      const ports = Array.from(new Set([config.vllmPort, ...Object.values(VLLM_MODEL_PORTS)]));
      getVllmInstances(ports, (instances) => {
        const primary = instances.find(i => i.port === config.vllmPort) || null;
        discoverModels((models) => {
          // 判断模型目录是否正在运行：vLLM 实例的 model 字段是 served-model-name
          // （通常等于目录名），而 m.path 是完整目录路径（/home/ll/models/<name>）。
          // 旧写法 `i.model === m.path` 永远为 false → 运行中的模型卡片恒显示「未运行」。
          const isModelRunning = (m) => {
            const sm0 = scriptModelForName(m.name);
            if (sm0) {
              // [script-entry-route-1006] 脚本卡片：条目在跑还不够，须引擎实际加载的
              // 就是这个目录（或其 YaRN 上下文副本）才算「运行中」。
              const ins0 = scriptModelInstance(sm0);
              if (ins0) {
                if (!ins0.engineModelPath) {
                  // 拿不到引擎路径（极端形态）：退回端口级旧语义，宁多报不漏报运行态
                  return instances.some(i => i.port === sm0.port);
                }
                return modelDirMatchesEnginePath(m.path, ins0.engineModelPath);
              }
              return instances.some(i => i.port === sm0.port);
            }
            return !!instances.find(i =>
              i.model === m.name || i.model === m.path || m.path === i.model ||
              (m.path && i.model && m.path.endsWith('/' + i.model)));
          };
          // 脚本化模型：展示名用注册名（qwen3.8-flash-next-nvfp4），并带 port/script_model
          // 标记，前端据此渲染「▶ 启动（脚本）」/「■ 停止」按钮。
          const mapModel = (m) => {
            const sm1 = scriptModelForName(m.name);
            if (sm1) {
              const inst = scriptModelInstance(sm1);
              // [script-entry-route-1006] 卡片级运行判定细化：条目在跑但引擎加载的不是
              // 本卡片目录 → 本卡显示未运行（可启动），路径行提示实际在服务的目录。
              let smRunning = !!inst;
              let siblingNote = '';
              if (inst && inst.engineModelPath && !modelDirMatchesEnginePath(m.path, inst.engineModelPath)) {
                smRunning = false;
                siblingNote = '（本启动条目当前由 ' + String(inst.engineModelPath).replace(/^.*\//, '') + ' 在服务于端口 ' + inst.port + '）';
              }
              // [sglang-adapt-1003] SGLang 脚本栈在位 → 卡片注记显示真实引擎（否则写着
              // 「容器镜像 PP2 脚本启动」误导：现行 18420 是 sglang.launch_server）。
              let onSgStack = false;
              try { onSgStack = !!(sm1.scriptSglang && sglangActive() && require('fs').existsSync(sm1.scriptSglang)); } catch (e) {}
              return Object.assign({}, m, {
                // [dir-name-distinct-1006] 展示名=真实目录名：同注册键（dirNames 多目录）的
                // 卡片不再同名；name 同时作为 modelName 回传，后端 scriptModelForName 按
                // dirNames 正常解析路由（点哪张卡启动哪个目录的模型）。script_name=注册键，
                // 供前端 defaults（脚本模型弹窗基准参数）查找。
                name: m.name, script_name: sm1.key, dir_name: m.name, script_model: true,
                port: inst ? inst.port : sm1.port, pid: smRunning ? (inst ? inst.pid : null) : null, // [script-entry-route-1006] 未运行卡片不带 pid（防「关闭」按钮误现）
                served_name: sm1.served,
                note: onSgStack
                  ? 'SGLang 脚本启动（sglang-18420 脚本对 · NEXTN 投机 + PLE BF16 锁页 · 加载约 6~10 分钟）'
                  : (sm1.note || ''),
                engine: onSgStack ? 'sglang' : 'vllm',
                running: smRunning,
                defaults: scriptModelDefaults(scriptModelVariantResolve(sm1, { modelName: m.name })),  // [model-variant-1008] 弹窗默认值跟随卡片实际目录
                schemes: flashNextSchemes(),
                script_note_extra: siblingNote || null,
              });
            }
            return Object.assign({}, m, { running: isModelRunning(m) });
          };
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            models: models.map(mapModel),
            vllm: primary
              ? { running: true, pid: primary.pid, model: primary.model, port: primary.port, gpu: primary.gpu }
              : { running: false, pid: null, model: null },
            instances,
          }));
        });
      });
      return;
    }

    if (req.method === 'POST') {
      let body = '';
      req.on('data', chunk => body += chunk.toString());
      req.on('end', async () => {
        try {
          const data = JSON.parse(body);
          const action = data.action;

          if (action === 'start') {
            const startPort = parseInt(data.port) || 8000;
            // 脚本化模型（chroot 镜像内启动，如 Flash-Next-NVFP4）：弹窗参数映射为
            // FN_* 环境变量，由宿主 wrapper 落成 env 文件、chroot 内参数化脚本构造命令行。
            const smStart = scriptModelForName(data.modelName);
            if (smStart) {
              // 启动方案：默认 chroot-pp2（现有行为）；sm80-170hx 需 Docker+GDS 前置，
              // 未就绪时明确拒绝并列出缺失项，避免用不兼容参数启动。
              const wantScheme = String(data.scheme || 'chroot-pp2');
              if (wantScheme !== 'chroot-pp2') {
                const sc = flashNextSchemes().find(x => x.key === wantScheme);
                if (!sc) {
                  res.writeHead(200, { 'Content-Type': 'application/json' });
                  res.end(JSON.stringify({ success: false, error: '未知启动方案：' + wantScheme }));
                  return;
                }
                if (!sc.ready) {
                  res.writeHead(200, { 'Content-Type': 'application/json' });
                  res.end(JSON.stringify({
                    success: false, scheme: sc.key,
                    error: '方案「' + sc.name + '」前置条件未就绪，缺失：' + sc.missing.join('、'),
                    missing: sc.missing, readiness: sc.readiness,
                  }));
                  return;
                }
                // script 模式启动：走本机 wrapper（chroot + 170HX 参数集，无需 Docker/GDS）
                if (sc.launcher && sc.launcher.mode === 'script') {
                  const wrap = sc.launcher.wrapper;
                  const schemeLog = '/home/ll/deploy/vllm-flash-next-170hx.log';
                  try {
                    const ch = require('child_process').spawn('bash', ['-c', `setsid nohup bash ${wrap} > /dev/null 2>&1 & echo $!`], { detached: true, stdio: ['ignore', 'pipe', 'ignore'] });
                    let buf2 = '';
                    ch.stdout.on('data', (d) => { buf2 += d.toString(); });
                    ch.on('close', () => {
                      VLLM_MODEL_PORTS['qwen3.8-flash-next'] = sc.launcher.port || 18420;
                      console.log(`[script-model] start scheme=${sc.key} (script) pid=${parseInt(String(buf2).trim()) || null}`);
                      if (!res.headersSent) {
                        res.writeHead(200, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ success: true, script: true, scheme: sc.key, port: sc.launcher.port || 18420, log: schemeLog }));
                      }
                    });
                    ch.on('error', (e) => { if (!res.headersSent) { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ success: false, scheme: sc.key, error: String(e) })); } });
                  } catch (e) {
                    if (!res.headersSent) { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ success: false, scheme: sc.key, error: String(e) })); }
                  }
                  return;
                }
                const repo = (sc.launcher && sc.launcher.repo) || '/home/ll/deploy/qwen-flash-sm80-170hx';
                const schemeLog = '/home/ll/deploy/vllm-flash-next-170hx.log';
                try {
                  const child = require('child_process').spawn('bash', ['-c', `cd ${repo} && setsid python3 scripts/serve.py start --config config/local.json >> ${schemeLog} 2>&1 < /dev/null & echo $!`], {
                    detached: true, stdio: ['ignore', 'pipe', 'ignore'],
                  });
                  let buf = '';
                  child.stdout.on('data', (d) => { buf += d.toString(); });
                  child.on('close', () => {
                    VLLM_MODEL_PORTS['qwen3.8-flash-next'] = (sc.launcher && sc.launcher.port) || 18420;
                    console.log(`[script-model] start scheme=${sc.key} pid=${parseInt(String(buf).trim()) || null}`);
                    if (!res.headersSent) {
                      res.writeHead(200, { 'Content-Type': 'application/json' });
                      res.end(JSON.stringify({
                        success: true, script: true, scheme: sc.key,
                        notice: `已按「${sc.name}」提交启动（Docker 容器，日志 ${schemeLog}），就绪需等 /health 返回 200`,
                      }));
                    }
                  });
                } catch (e) {
                  res.writeHead(200, { 'Content-Type': 'application/json' });
                  res.end(JSON.stringify({ success: false, error: '方案启动异常: ' + e.message }));
                }
                return;
              }
              const inst0 = scriptModelInstance(smStart);
              if (inst0) {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ success: false, error: `模型 ${smStart.key} 已在运行（端口 ${inst0.port}，PID ${inst0.pid}），如需重启请先点「停止」` }));
                return;
              }
              const plan = scriptPlanFor(smStart, data); // [sglang-adapt-1003] sglang 栈在位时下发 SG_*
              try {
                const startScript = resolveStartScript(smStart);
                // INNER 只对旧栈 wrapper 有意义；解析到新栈时绝不带旧 inner 覆盖（见 resolveStartScript 注）
                const envPrefix = (startScript === smStart.script && smStart.inner) ? `INNER=${smStart.inner} ` : '';
                const child = require('child_process').spawn('bash', ['-c', `${envPrefix}setsid bash ${startScript} >> ${scriptLaunchLog(smStart)} 2>&1 < /dev/null & echo $!`], {
                  detached: true,
                  stdio: ['ignore', 'pipe', 'ignore'],
                  env: Object.assign({}, process.env, plan.env),
                });
                let outBuf = '';
                child.stdout.on('data', (d) => { outBuf += d.toString(); });
                child.on('error', (e) => {
                  if (!res.headersSent) { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ success: false, error: '脚本启动失败: ' + e.message })); }
                });
                child.on('close', () => {
                  const pid = parseInt(String(outBuf).trim()) || null;
                  // 路由与别名跟随弹窗实际填写的 served 名与端口
                  VLLM_MODEL_PORTS[plan.served] = plan.port;
                  VLLM_MODEL_PORTS[smStart.key] = plan.port;
                  MODEL_ALIASES[smStart.key] = plan.served;
                  (smStart.dirNames || []).forEach(dn => { MODEL_ALIASES[dn] = plan.served; });
                  if (global.__GPU_INSTANCES && typeof global.__GPU_INSTANCES.set === 'function') {
                    global.__GPU_INSTANCES.set(plan.port, { port: plan.port, gpuId: 0, gpuCount: plan.gpuCount || (smStart.base && smStart.base.pp) || 2, runtime: plan.runtime || 'vllm', model: smStart.key, startedAt: Date.now() });
                  }
                  console.log(`[script-model] start ${smStart.key} port=${plan.port} served=${plan.served} pid=${pid} :: ${plan.summary}`);
                  plan.warnings.forEach(w => console.warn(`[script-model] warn: ${w}`));
                  if (!res.headersSent) {
                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({
                      success: true, script: true, pid, port: plan.port, served: plan.served,
                      summary: plan.summary, warnings: plan.warnings,
                      notice: `${smStart.key} 已按弹窗参数启动（端口 ${plan.port}，${smStart.note || ''}），加载完成前请勿重复点击`,
                    }));
                  }
                });
              } catch (e) {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ success: false, error: '脚本启动异常: ' + e.message }));
              }
              return;
            }
            // PD 分离模式：prefill 占 startPort、decode 占 startPort+1，两个端口都必须空闲
            const isPd = data.pdMode === '1';
            // 双实例保护：目标端口已有 vLLM 运行则拒绝启动，避免误杀现有实例
            if (portInUse(startPort)) {
              res.writeHead(200, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ success: false, error: `端口 ${startPort} 已有模型在运行，请换一个空闲端口，或先停止该端口的实例` }));
              return;
            }
            if (isPd && portInUse(startPort + 1)) {
              res.writeHead(200, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ success: false, error: `PD 模式第二个端口 ${startPort + 1} 已被占用，请换一个空闲起始端口，或先停止该端口的实例` }));
              return;
            }
            // GPU 独占保护：检查请求的 GPU 是否已被其他 vLLM/SGLang 实例占用
            const wantGpuId = parseInt(data.gpuId) || 0;
            // PD 模式固定占 2 张卡（一卡 prefill + 一卡 decode）
            const wantGpuCount = Math.max(1, isPd ? 2 : (parseInt(data.gpuCount) || 1));
            const gpuConflict = checkGpuConflict(wantGpuId, wantGpuCount, startPort);
            if (gpuConflict) {
              const runtimeLabel = gpuConflict.runtime === 'sglang' ? 'SGLang' : 'vLLM';
              const gpuRange = gpuConflict.gpuCount > 1
                ? `GPU ${gpuConflict.gpuId}~${gpuConflict.gpuId + gpuConflict.gpuCount - 1}`
                : `GPU ${gpuConflict.gpuId}`;
              res.writeHead(200, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ success: false, error: `GPU ${wantGpuId} 已被 ${runtimeLabel} 实例（${gpuConflict.model}，端口 ${gpuConflict.port}，占用 ${gpuRange}）占用。请选择其他空闲显卡，或先停止该实例` }));
              return;
            }
            // 推测解码：DFlash2 优先（草稿模型 qwen3.8-27b-dflash2 + nightly 环境）；MTP 仅 35B (MoE) 默认开启。
            // PD 分离模式两种投机均可用（08-27 实测定版）：prefill 端也配相同 speculative-config
            // 对齐 KV 布局——MTP 时两侧 physical_blocks_per_logical 同为 51 握手通过；DFlash2 时
            // 草稿模型两端都前向、草稿层 KV 一起传输，hash 一致（无需跳过校验），输出正常。
            const rawDflash = data.dflash === '1' || data.mtp === 'dflash';
            const wantDflash = rawDflash;
            // DSpark（DFlash 骨干 + Markov 头，vllm-env 0.27.1 原生支持）：PD 分离模式未实测，暂不允许
            const wantDspark = data.dspark === '1' || data.mtp === 'dspark';
            if (isPd && wantDspark) {
              res.writeHead(200, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ success: false, error: 'DSpark 投机解码暂不支持 PD 分离模式（未实测），请改用 MTP 或常规模式' }));
              return;
            }
            let pdNotice = null;
            const params = {
              port: startPort,
              maxModelLen: data.maxModelLen || 'auto',
              gpuId: parseInt(data.gpuId) || 0,
              // 启用显卡数量：从 gpuId 起连续取 N 张卡，N>1 时多卡张量并行启动
              gpuCount: Math.max(1, parseInt(data.gpuCount) || 1),
              // 运行模式：0=常规单实例；1=PD 分离（prefill + decode 双实例，固定 2 卡）
              pdMode: isPd ? '1' : '0',
              mtp: (wantDflash || wantDspark) ? '0' : (data.mtp !== undefined ? (data.mtp === '1' ? '1' : '0') : (String(data.modelName).includes('35b') ? '1' : '0')),
              dflash: wantDflash ? '1' : '0',
              dspark: wantDspark ? '1' : '0',
              // 投机步数 num_speculative_tokens：1~8（MTP 默认 5 / DFlash2 推荐 7）
              mtpTokens: data.mtpTokens !== undefined ? data.mtpTokens : 5,
              servedName: data.servedName || data.modelName,
              maxNumSeqs: parseInt(data.maxNumSeqs) || 4,
              gpuMemUtil: parseFloat(data.gpuMemUtil) || 0.9,
              // 采样参数（SamplingParams），走 --override-generation-config 合并覆盖
              temperature: data.temperature !== undefined && data.temperature !== '' ? data.temperature : 1.0,
              topP: data.topP !== undefined && data.topP !== '' ? data.topP : 0.95,
              topK: data.topK !== undefined && data.topK !== '' ? data.topK : 20,
              minP: data.minP !== undefined && data.minP !== '' ? data.minP : 0.0,
              presencePenalty: data.presencePenalty !== undefined && data.presencePenalty !== '' ? data.presencePenalty : 0.0,
              repetitionPenalty: data.repetitionPenalty !== undefined && data.repetitionPenalty !== '' ? data.repetitionPenalty : 1.0,
              // 思考模式：thinking=1(默认) -> enable_thinking=true + reasoning_effort=<effort>
              // thinking=0 -> enable_thinking=false（非思考）
              thinking: data.thinking !== undefined ? data.thinking : '1',
              // 思考深度：low / medium / high / xhigh（缺省 xhigh，与启动页默认一致）
              thinkingEffort: data.thinkingEffort || 'xhigh',
              // KV 缓存量化：auto / fp8 / int8 / fp8_kv
              kvCacheQuant: data.kvCacheQuant || 'auto',
              // 高级参数（vLLM 0.27.1 实测存在的 flag；空值/未勾选不追加，行为与旧版一致）
              maxBatchedTokens: data.maxBatchedTokens,
              maxScheduledTokens: data.maxScheduledTokens,
              schedPolicy: data.schedPolicy,
              asyncScheduling: data.asyncScheduling,
              enforceEager: data.enforceEager,
              blockSize: data.blockSize,
              cpuOffloadGb: data.cpuOffloadGb,
              prefixCaching: data.prefixCaching,
              chunkedPrefill: data.chunkedPrefill,
              seed: data.seed,
              dtype: data.dtype,
              noLogRequests: data.noLogRequests,
              limitMm: data.limitMm,
              maxLoraRank: data.maxLoraRank,
              disableAllReduce: data.disableAllReduce,
              attentionBackend: data.attentionBackend,
              // vLLM 附加环境变量 / 附加启动参数（前端弹窗「vLLM 高级参数」区块，仅 vllm 运行时生效）
              vllmDraftModel: data.vllmDraftModel || '',
              vllmExtraEnv: data.vllmExtraEnv || '',
              vllmExtraArgs: data.vllmExtraArgs || '',
              // SGLang 附加启动参数（前端弹窗「SGLang 附加启动参数」框，仅 sglang 运行时生效）
              sglangExtraEnv: data.sglangExtraEnv || '',
              sglangExtraArgs: data.sglangExtraArgs || '',
            };
            // 启动方式（运行时）：vllm（默认）或 sglang
            const runtime = data.runtime === 'sglang' ? 'sglang' : 'vllm';
            const starter = runtime === 'sglang' ? startSglangModel : startVllmModel;
            starter(data.modelName, params, (result) => {
              if (result.success && params.servedName) {
                // 注册路由表：8889 会把该模型的请求转发到对应端口
                VLLM_MODEL_PORTS[params.servedName] = startPort;
                // 登记 GPU 占用，防止 vLLM/SGLang 冲突
                global.__GPU_INSTANCES.set(startPort, {
                  port: startPort,
                  gpuId: wantGpuId,
                  gpuCount: 1,
                  runtime: runtime,
                  model: data.modelName,
                  startedAt: Date.now(),
                  pdRole: isPd ? 'prefill' : undefined,
                });
                // PD 分离：注册两步转发路由 + 登记 decode 实例（第 2 卡）
                if (isPd && result.pd) {
                  PD_MODEL_PORTS[params.servedName] = { prefill: startPort, decode: startPort + 1 };
                  global.__GPU_INSTANCES.set(startPort + 1, {
                    port: startPort + 1,
                    gpuId: wantGpuId + 1,
                    gpuCount: 1,
                    runtime: runtime,
                    model: data.modelName,
                    startedAt: Date.now(),
                    pdRole: 'decode',
                  });
                }
              }
              if (pdNotice && !result.error) result.notice = pdNotice;
              res.writeHead(200, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify(result));
            });
          } else if (action === 'stop') {
            const stopPort = data.port ? parseInt(data.port) : null;
            // 脚本化模型：chroot 内引擎属 root（ll 的 lsof 看不到监听端口），
            // 走宿主侧停止脚本（内部按 --port 匹配后 sudo kill，不误杀其他实例）。
            const smStop = stopPort ? scriptModelForPort(stopPort) : null;
            if (smStop) {
              const stopPath = resolveStopScript(smStop);
              if (!stopPath || !fs.existsSync(stopPath)) {
                // 09-26：脚本不存在时过去会静默"success"（bash 报错也被吞），按钮点了没任何动静
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ success: false, error: '停止脚本不存在：' + (stopPath || '(未注册)') }));
                return;
              }
              let out = '';
              try {
                // 09-20：execSync(90s) 阻塞事件循环 → 异步 execFile
                // 09-26：超时 90s→240s——新栈停止脚本自身要等 90s 优雅退出 + 最多 120s 显存归零轮询
                out = await execFileAsync('bash', [stopPath, String(smStop.port)], { encoding: 'utf8', timeout: 240000, maxBuffer: 4 * 1024 * 1024 });
              } catch (e) { out = String((e && e.stdout || '') + (e && e.stderr || '') || (e && e.message) || e); }
              Object.keys(VLLM_MODEL_PORTS).forEach(k => { if (VLLM_MODEL_PORTS[k] === smStop.port) delete VLLM_MODEL_PORTS[k]; });
              global.__GPU_INSTANCES.delete(smStop.port);
              const stillAlive = scriptModelInstance(smStop);
              console.log(`[script-model] stop ${smStop.key} port ${smStop.port}${stillAlive ? ' (WARN: 仍有残留 pid=' + stillAlive.pid + ')' : ' (已彻底停止)'} via ${stopPath}`);
              res.writeHead(200, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ success: !stillAlive, script: true, port: smStop.port, stoppedModel: smStop.key,
                error: stillAlive ? '已执行停止脚本但仍有残留进程（pid=' + stillAlive.pid + '），详见输出' : undefined,
                output: String(out).trim().slice(-400) }));
              return;
            }
            // PD 配对：停止其中一个端口时，连带停止配对实例并清理 PD 两步转发路由
            let pdPair = null;
            if (stopPort) {
              for (const [m, pp] of Object.entries(PD_MODEL_PORTS)) {
                if (pp.prefill === stopPort || pp.decode === stopPort) {
                  pdPair = { model: m, ports: [pp.prefill, pp.decode] };
                  break;
                }
              }
            }
            Promise.resolve(stopVllm((result) => {
              if (result.success && stopPort) {
                // 移除路由表中指向该端口的模型
                Object.keys(VLLM_MODEL_PORTS).forEach(k => { if (VLLM_MODEL_PORTS[k] === stopPort) delete VLLM_MODEL_PORTS[k]; });
                // 移除 GPU 占用登记，释放该端口占用的显卡
                global.__GPU_INSTANCES.delete(stopPort);
                if (pdPair) {
                  delete PD_MODEL_PORTS[pdPair.model];
                  const other = pdPair.ports.find(p => p !== stopPort);
                  console.log('[pd-stop] stopPort=' + stopPort + ' pair=' + JSON.stringify(pdPair) + ' other=' + other);
                  if (other !== undefined) {
                    global.__GPU_INSTANCES.delete(other);
                    // 连带停止配对实例（后台执行，结果并入主响应）
                    Promise.resolve(stopVllm((r) => console.log('[pd-stop] sibling stop result: ' + JSON.stringify(r)), other)).catch(e => console.log('[pd-stop] sibling stop error: ' + (e && e.message || e)));
                  }
                }
              }
              res.writeHead(200, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify(result));
            }, stopPort)).catch(e => { console.log('[stop] stopVllm error: ' + (e && e.message || e)); try { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ success: false, error: String((e && e.message) || e) })); } catch (_) {} });
          } else {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ success: false, error: 'Unknown action: ' + action }));
          }
        } catch (e) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: false, error: e.message }));
        }
      });
      return;
    }
  }

  // === Internal API: per-request measured stats (vLLM stat-logger plugin) ===
  if (pathname === '/v1/internal/recent-requests') {
    try {
      const q = require('url').parse(req.url, true).query;
      const limit = Math.min(200, Math.max(1, parseInt(q.limit) || 8));
      const p = path.join(__dirname, 'request-traces.jsonl');
      const out = [];
      // rid → pid 映射（从 live-prefill 文件构建，2s 缓存）。vLLM 的 trace 记录
      // 本身带 pid（插件 09-05 补丁：os.getpid=EngineCore pid），但那是**子进程** pid，
      // 需经 resolveInstanceByPid 上溯到主进程；live-prefill 的 rid→pid 作为二级兜底。
      const ridPid = buildRidPidMap();
      const vllmInsts = listVllmInstances();
      const tagGpu = (rec) => {
        if (rec.runtime === 'sglang') {
          // 精确 GPU 归因（优先级）：① 记录自带 port 字段（sglang exporter 补丁+实例重启
          // 后写入，覆盖绕过 8889 的直连流量）② rid→port 代理 tee 注册（响应 id=rid，
          // 覆盖经代理流量）③ 唯一在跑实例启发式（旧行为兜底，多实例时不生效避免误导）
          const sgl = listSglangInstances();
          let port = rec.port ? parseInt(rec.port, 10) : null;
          if (!port || Number.isNaN(port)) {
            const mp = global.__ridPortMap;
            if (mp && rec.request_id && mp.has(rec.request_id)) port = mp.get(rec.request_id);
          }
          if (port && !Number.isNaN(port)) {
            rec.port = port;
            const hit = sgl.find(i => i.port === port);
            rec.gpu = (hit && hit.gpu != null) ? hit.gpu : (global.__portGpuSeen.get(port) || null);
          } else {
            // ⑤ 模型名 → 该模型最近实例端口（两实例 served 名不同：qwen3.8-27b-0/-1；
            //    直连流量不经代理、rid→port 无注册时靠它归因）
            let p = null;
            if (rec.model) {
              const lastSeen = global.__lastSglangModelByPort = global.__lastSglangModelByPort || new Map();
              for (const i of sgl) { if (i.model) lastSeen.set(i.model, i.port); }
              if (lastSeen.has(rec.model)) p = lastSeen.get(rec.model);
            }
            // ④ rid 前缀 → 端口（36 主机 rid 实测 f=8000 / h=8001，经验性兑底）
            if (p == null && rec.request_id) {
              if (rec.request_id.startsWith('h')) p = 8001;
              else if (rec.request_id.startsWith('f')) p = 8000;
            }
            if (p != null) {
              const hit = sgl.find(i => i.port === p);
              if (hit) { rec.gpu = hit.gpu; rec.port = hit.port; return; }
              rec.port = p; rec.gpu = global.__portGpuSeen.get(p) || null;
              return;
            }
            // ⑥ 唯一在跑实例（原启发式）
            if (sgl.length === 1) { rec.gpu = sgl[0].gpu; rec.port = sgl[0].port; }
            else { rec.gpu = null; rec.port = null; }
          }
          return;
        }
        // 优先用 trace 记录自带的 pid（插件 09-05 补丁：os.getpid=EngineCore pid），
        // 不依赖 live-prefill 的 rid→pid 映射（live-prefill 依赖 scheduler.py 补丁，
        // 换 vLLM 版本/venv 后补丁丢失时 live-prefill 停更→映射失效→gpu 全 null）。
        let inst = null;
        if (rec.pid != null) {
          // 09-14：pid 可能是 EngineCore/Worker 子进程 → 沿 ppid 链上溯到主进程
          inst = resolveInstanceByPid(rec.pid, vllmInsts);
        }
        if (!inst && ridPid.has(rec.request_id)) {
          inst = resolveInstanceByPid(ridPid.get(rec.request_id), vllmInsts);
        }
        rec.gpu = inst ? (inst.gpu != null ? inst.gpu : null) : null;
        // gpus = 该实例实际占用的卡列表（PP/TP 多卡实例：[0,1]）；前端据此把同一条
        // 记录同时归入 GPU0 与 GPU1 分组，并打「GPU 0+1」徽标。
        rec.gpus = (inst && Array.isArray(inst.gpus) && inst.gpus.length)
          ? inst.gpus.slice()
          : (rec.gpu != null ? [rec.gpu] : null);
        rec.port = inst ? inst.port : null;
        rec.model = (inst && (inst.servedName || inst.modelPath)) || rec.model || null;
      };
      try {
        const fd = fs.openSync(p, 'r');
        try {
          const sz = fs.fstatSync(fd).size;
          // 尾部窗口 256KB（≈700 行 @366B/行）：GPU0/GPU1 独立行数各可设到 200，
          // 64KB 只够 ~179 行，高行数设置时凑不满（09-05 扩窗）
          const LEN = Math.min(256 * 1024, sz);
          const buf = Buffer.alloc(LEN);
          fs.readSync(fd, buf, 0, LEN, sz - LEN);
          const lines = buf.toString('utf8').split('\n').filter(l => l.trim());
          // 09-18：先解析 + 剔除脏时间戳记录，再截最后 limit 条。
          // 旧版是「先截断后过滤」：脏时钟（2161 年）记录会占满窗口，真实数据整段消失。
          const parsed = [];
          for (const l of lines) {
            let rec; try { rec = JSON.parse(l); } catch (e2) { continue; } // 残行跳过
            if (!traceTsPlausible(rec && rec.t)) continue;
            parsed.push(rec);
          }
          parsed.slice(-limit).forEach(r => out.push(r));
        } finally { fs.closeSync(fd); }
      } catch (e) { /* file missing — no completed requests yet */ }

      // sglang 每请求实测数据（--export-metrics-to-file 写入）：
      // 记录形如 {"request_parameters":"...","prompt_tokens":N,"completion_tokens":N,
      // "cached_tokens":N,"finish_reason":{...},"e2e_latency":X,"queue_time":Y,
      // "spec_accept_rate":Z,"id":"rid", "request_received_ts":..., ...}
      // 转成与 vLLM trace 同构的字段，前端「最近完成请求」表格直接复用。
      {
        // 09-18 修复：旧版每请求全量 readFileSync+逐行 JSON.parse 每目录最后 3 个文件
        // （单文件实测最大 46MB、合计 ~130MB）同步阻塞主循环 ~2s/请求，饿死 GPU 卡等
        // 所有秒级采样 → 改「每文件只读尾部 1MB + 3s 结果缓存」。out 接缓存记录浅拷贝，
        // 后续 tagGpu 的字段改写不会污染缓存对象。
        let sglCached = global.__sglRecentCache;
        if (!sglCached || Date.now() - sglCached.at > 3000) {
          const recs = [];
          for (const { dir: sglDir, port: dirPort } of sglangMetricsDirs()) {
          try {
            const files = fs.readdirSync(sglDir).filter(f => /^sglang-request-metrics-.*\.log$/.test(f)).sort();
            for (const fn of files.slice(-3)) {
              let lines = [];
              try {
                const fp = path.join(sglDir, fn);
                const st = fs.statSync(fp);
                const TL = Math.min(1024 * 1024, st.size);
                const fd = fs.openSync(fp, 'r');
                try {
                  const buf = Buffer.alloc(TL);
                  fs.readSync(fd, buf, 0, TL, st.size - TL);
                  lines = buf.toString('utf8').split('\n').filter(l => l.trim());
                  if (TL < st.size && lines.length) lines.shift(); // 砍掉首行残半行
                } finally { fs.closeSync(fd); }
              } catch (eRead) { continue; }
            for (const l of lines) {
              try {
                const rec = JSON.parse(l);
                if (!rec || rec.prompt_tokens === undefined || rec.completion_tokens === undefined) continue;
                // e2e：优先用 sglang 直接给的 e2e_latency（秒），否则用请求时间戳差。
                const e2eS = (typeof rec.e2e_latency === 'number' && rec.e2e_latency > 0)
                  ? rec.e2e_latency
                  : (rec.request_finished_ts && rec.request_received_ts
                    ? Math.max(0, rec.request_finished_ts - rec.request_received_ts) : 0);
                // prefill：forward_entry_time（进入 scheduler）→ prefill_finished_time 为 epoch 秒
                const prefillS = (rec.prefill_finished_time && rec.forward_entry_time
                  && rec.prefill_finished_time > rec.forward_entry_time)
                  ? rec.prefill_finished_time - rec.forward_entry_time : 0;
                // decode：e2e 减去 prefill（剩余主要是 decode + 调度开销）
                const decodeS = e2eS > prefillS ? e2eS - prefillS : 0;
                const gen = parseInt(rec.completion_tokens) || 0;
                const pp = parseInt(rec.prompt_tokens) || 0;
                // finish_reason 可能是 {"type":"length",...} 对象或 "stop" 字符串
                let fr = rec.finish_reason;
                if (fr && typeof fr === 'object' && !Array.isArray(fr)) fr = fr.type || '';
                // DFLASH 命中率：sglang 的 spec_accept_rate（接受 draft 比例，0~1）→ 前端 MTP% 列
                let mtpHit = null;
                if (typeof rec.spec_accept_rate === 'number' && rec.spec_accept_rate > 0) {
                  mtpHit = parseFloat((rec.spec_accept_rate * 100).toFixed(1));
                }
                recs.push({
                  t: (rec.request_finished_ts || Date.now() / 1000),
                  request_id: rec.id || rec.request_id || '',
                  finish_reason: String(fr || ''),
                  prompt_tokens: pp,
                  gen_tokens: gen,
                  cached_tokens: parseInt(rec.cached_tokens) || 0,
                  // [extcache-1007] 二级缓存命中真值：SGLang 官方 exporter 在完成记录里
                  // 分开给 storage_backend（磁盘/CPU 二级回载）与 local（本级 radix）两分项，
                  // 缺字段（旧版 sglang）时不写该键 → 前端显示 --，不用总数冒充。
                  ext_cached_tokens: (rec.cached_tokens_storage_backend !== undefined)
                    ? (parseInt(rec.cached_tokens_storage_backend) || 0) : undefined,
                  loc_cached_tokens: (rec.cached_tokens_local !== undefined)
                    ? (parseInt(rec.cached_tokens_local) || 0) : undefined,
                  queued_s: typeof rec.queue_time === 'number' ? parseFloat(rec.queue_time.toFixed(3)) : 0,
                  prefill_s: parseFloat(prefillS.toFixed(3)),
                  decode_s: parseFloat(decodeS.toFixed(3)),
                  e2e_s: parseFloat(e2eS.toFixed(3)),
                  tpot_s: decodeS > 0 && gen > 1 ? parseFloat((decodeS / (gen - 1)).toFixed(5)) : 0,
                  decode_tps: decodeS > 0 ? parseFloat((gen / decodeS).toFixed(1)) : 0,
                  mtp_hit_rate: mtpHit,
                  mtp_drafted: rec.spec_proposed_drafts,
                  mtp_accepted: rec.spec_accepted_drafts,
                  mtp_exact: true,
                  runtime: 'sglang',
                  port: rec.port || dirPort || null,
                });
              } catch (e2) { /* skip bad line */ }
            }
            }
          } catch (e) { /* dir missing — no sglang metrics yet */ }
          }
          sglCached = global.__sglRecentCache = { at: Date.now(), recs };
        }
        for (const r of sglCached.recs) out.push(Object.assign({}, r));
      }

      // 关联 GPU（vLLM 记录经 rid→pid→实例；sglang 记录按「唯一在跑实例」启发式归属）
      for (const rec of out) tagGpu(rec);
      // [extcache-1007] 关联每请求二级缓存命中：vLLM 走 rt-patch #12 落盘文件按 rid join；
      // SGLang 官方 exporter 已在完成记录里直接给分项（cached_tokens_storage_backend），
      // 在上面 sglang 转换段已映射为 ext_cached_tokens，这里不覆盖已有的真值。
      extAttachToRecs(out);
      // 关联控制台任务号（T 号）：rid → taskId 映射由代理 tee 响应流时记录
      const ridTask = global.__ridTaskId;
      for (const rec of out) { if (rec.request_id && ridTask.has(rec.request_id)) rec.taskId = ridTask.get(rec.request_id); }
      // 合并 vllm + sglang，按完成时间倒序取 limit 条
      // 09-18：返回前统一剔除脏时间戳记录（含 sglang 侧），避免脏时钟霸占排序首位
      const plausible = out.filter(r => traceTsPlausible(r && r.t));
      plausible.sort((a, b) => (b.t || 0) - (a.t || 0));
      // [sglang-adapt-1003] 在跑 sglang 实例未带 --export-metrics-to-file 时，本表没有任何
      // 该实例的逐请求记录，只剩 vLLM 时代的历史残留行（时间是旧的、字段无 port）——用户
      // 视角就是「表格全是过期假数据」。显式回传缺导出旗标，前端据此显示黄条提示。
      let sgExportOff = [];
      try {
        for (const si of listSglangInstances()) {
          let cmd = '';
          try { cmd = fs.readFileSync(`/proc/${si.pid}/cmdline`, 'utf8').split('\0').join(' '); } catch (e) { continue; }
          if (cmd && !/--export-metrics-to-file(\s|$)/.test(cmd)) sgExportOff.push(si.port);
        }
      } catch (e) {}
      const hasSglRows = plausible.some(r => r.runtime === 'sglang');
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ requests: plausible.slice(0, limit),
        // [extcache-1007] 透明化二级缓存命中列的数据可用性，前端据此在表尾如实说明，
        // 不再让用户对着满屏 -- 猜是不是坏了。
        ext_cache_available: extCacheAvailable(),
        ext_cache_source: extCacheSource(),
        sglang_export_off: (sgExportOff.length && !hasSglRows) ? sgExportOff : undefined })); // newest first
    } catch (e) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: e.message }));
    }
    return;
  }

  // === Internal API: vLLM Logs ===
  if (pathname === '/v1/internal/vllm-logs') {
    // vLLM may be started externally (e.g. ~/vllm-deploy/start-*.sh) with its
    // stdout redirected elsewhere. Resolve the file the running process actually
    // writes to; fall back to ./vllm.log when vLLM was started from this console.
    const logSrc = getVllmLogSource();
    const logFile = logSrc.file;
    const stateFile = path.join(__dirname, 'log-clear-state.json');
    const query = require('url').parse(req.url, true).query;
    // 09-19：docker 容器化实例 → docker logs（异步子进程，不阻塞事件循环）
    if (logSrc.docker) {
      const dfile = 'docker:' + logSrc.docker;
      if (req.method === 'POST') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, file: dfile, note: '容器日志由 docker daemon 保管，清除仅对本次视图生效' }));
        return;
      }
      const tailN = Math.min(1000, parseInt(query.tailLines) || 100);
      require('child_process').execFile('docker', ['logs', '--tail', String(tailN), logSrc.docker],
        { maxBuffer: 16 * 1024 * 1024, timeout: 5000 }, (err, stdout, stderr) => {
          const raw = String(stdout || '') + String(stderr || '');
          const lines = raw.split('\n').filter(l => l.trim());
          // 增量打时间戳：只给本轮新出现的行盖观测时间（与文件路径逻辑同构）
          if (!global.__vllmLogTracker) global.__vllmLogTracker = { file: null, offset: 0, seen: false, times: new Map() };
          const t = global.__vllmLogTracker;
          if (t.file !== dfile) { t.file = dfile; t.times.clear(); t.seen = false; }
          const nowT = Date.now();
          const obs = new Set();
          for (let i = lines.length - 1; i >= 0; i--) {
            if (t.times.has(lines[i])) break; // 再往前都是上轮已见
            obs.add(i);
          }
          for (const i of obs) t.times.set(lines[i], nowT);
          if (t.times.size > 30000) { const ks = [...t.times.keys()].slice(-20000); t.times = new Map(ks.map(k => [k, t.times.get(k)])); }
          const out = lines.map((l, i) => (obs.has(i) && !/\d{2}-\d{2} \d{2}:\d{2}:\d{2}/.test(l))
            ? `[${new Date(nowT).toTimeString().slice(0, 8)}] ${l}` : l);
          t.seen = true;
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            logs: out.slice(-tailN).join('\n'), totalLines: out.length, file: dfile,
            mtime_ms: nowT, stale_min: 0, docker: true,
          }));
        });
      return;
    }
    if (logSrc.tty) {
      // stdout is a terminal: nothing to read/clear here
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        success: req.method === 'GET' ? true : false,
        file: logFile, tty: true,
        error: req.method === 'GET' ? undefined : 'vLLM 从终端启动，日志输出在 ' + logFile + '，无法在此清除',
        logs: req.method === 'GET' ? 'vLLM 从终端（' + logFile + '）启动，日志直接输出到该终端，控制台无法读取。' : null,
        totalLines: 0
      }));
      return;
    }
    if (req.method === 'POST') {
      // "Clear" = remember the current byte offset; never truncate a live log
      // (the writer keeps its file offset, truncation would leave NUL holes).
      try {
        let size = 0;
        try { size = fs.statSync(logFile).size; } catch (e) {}
        fs.writeFileSync(stateFile, JSON.stringify({ byteOffset: size, filePath: logFile, time: new Date().toISOString() }));
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, file: logFile, byteOffset: size }));
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: e.message }));
      }
      return;
    }
    const tailLines = parseInt(query.tailLines) || 100;
    try {
      // Defensive: only ever read regular files (a TTY/device read would block
      // the main thread forever). For very large logs read only the last 5MB.
      let content;
      let windowed = false;
      let byteBase = 0;   // content[0] 在整份文件里的绝对字节位置（窗口化读取时非 0）
      let winBuf = null;  // 原始字节窗口：把「清空标记」的绝对字节偏移精确换算成字符下标用
      let winSkip = 0;    // 窗口首部丢掉的半行字节数（byteBase 已含它）
      let fileSize = 0;   // 本次读取时整份文件的字节大小（判断标记点是否已在 EOF 之后）
      {
        const fst = fs.statSync(logFile);
        if (!fst.isFile()) {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ file: logFile, logs: null }));
          return;
        }
        const MAX_READ = 5 * 1024 * 1024;
        const fd = fs.openSync(logFile, 'r');
        try {
          const sz = fs.fstatSync(fd).size;
          fileSize = sz;
          if (sz <= MAX_READ) {
            winBuf = fs.readFileSync(fd);
            content = winBuf.toString('utf8');
          } else {
            windowed = true;
            const buf = Buffer.alloc(MAX_READ);
            const n = fs.readSync(fd, buf, 0, MAX_READ, sz - MAX_READ);
            const nl = buf.indexOf(0x0a, 0, n);
            winSkip = nl >= 0 ? nl + 1 : 0;
            byteBase = (sz - MAX_READ) + winSkip;
            winBuf = buf;
            content = buf.toString('utf8', winSkip, n);
          }
        } finally { fs.closeSync(fd); }
      }
      // Incremental timestamp tracking: remember the file offset seen on the
      // previous poll and stamp newly-appearing lines with the observation time.
      if (!global.__vllmLogTracker) global.__vllmLogTracker = { file: null, offset: 0, seen: false, times: new Map() };
      const t = global.__vllmLogTracker;
      const tKey = logFile + (windowed ? '#tail' : '');
      if (t.file !== tKey || !t.seen) {
        // First sight of this file (fresh boot or new file): do not stamp
        // historical lines with "now" — just start tracking from the end.
        t.file = logFile; t.offset = content.length; t.seen = true; t.times.clear();
      }
      if (content.length < t.offset) { t.offset = 0; t.times.clear(); } // file truncated/rotated
      const now = Date.now();
      if (content.length > t.offset) {
        let idx = t.offset;
        while (idx < content.length) {
          const nl = content.indexOf('\n', idx);
          const lineStart = idx;
          idx = nl === -1 ? content.length : nl + 1;
          if (idx > lineStart) t.times.set(lineStart, now);
        }
        if (t.times.size > 20000) {
          const keys = [...t.times.keys()].sort((a, b) => a - b);
          for (const k of keys.slice(0, keys.length - 10000)) t.times.delete(k);
        }
        t.offset = content.length;
      }
      let marker = 0;
      let cleared = false;
      try {
        const st = JSON.parse(fs.readFileSync(stateFile, 'utf8'));
        if (st && st.filePath === logFile) {
          // 09-26：state 里存的是「整份文件」的绝对字节偏移（清空时记下的 size）。旧代码把它
          // 直接当窗口/字符串下标用，两处错：① 大文件尾部窗口读取时原点不同；② UTF-8 多字节
          // 行导致「字节数 ≠ 字符数」——中文日志里偏移一路前移，标记点跑到 EOF 之后被整份显示，
          // 表现为「清空日志按了没反应」（实测 1.7MB 中文日志即中招）。
          // 现在用原始窗口字节精确换算成字符下标，再吸附到行首，避免切在半行上。
          const abs = Math.max(0, st.byteOffset || 0);
          if (abs > fileSize) {
            marker = 0;   // 日志被轮转/换文件：标记点已在 EOF 之后 → 全部显示
          } else if (abs > byteBase && winBuf) {
            const endByte = Math.min(winBuf.length, winSkip + (abs - byteBase));
            let m = winBuf.toString('utf8', winSkip, endByte).length;
            const nlM = content.indexOf('\n', m);
            m = nlM === -1 ? content.length : nlM + 1;
            marker = Math.min(m, content.length);
            cleared = true;
          }
        }
      } catch (e) {}
      const nowTs = Date.now();
      const fsz = fs.statSync(logFile).size;
      const fsMod = new Date(fs.statSync(logFile).mtimeMs).getTime();
      const out = [];
      let idx = marker;
      while (idx < content.length) {
        const nl = content.indexOf('\n', idx);
        const lineStart = idx;
        const line = content.substring(lineStart, nl === -1 ? content.length : nl);
        idx = nl === -1 ? content.length : nl + 1;
        if (!line.trim()) continue;
        let time = t.times.get(lineStart);
        // 历史行不再「按 mtime~now 均匀插值」编造时间（2026-09-16）：那会让一个几小时没写的
        // 文件里的陈旧行显示成刚刚发生，掩盖真实情况。现在只有本轮实际观测到的行才打时间戳，
        // 其余（含引擎自带 MM-DD HH:MM:SS 的行）原样输出，由前端解析原生时间。
        // Lines that already carry vLLM's own "08-19 11:03:33" timestamp get no extra prefix
        out.push(time && !/\d{2}-\d{2} \d{2}:\d{2}:\d{2}/.test(line) ? `[${new Date(time).toTimeString().slice(0, 8)}] ${line}` : line);
      }
      const tail = out.slice(-tailLines).join('\n');
      // 日志新鲜度：面板据此提示「日志已 N 分钟没有新内容」，避免把陈旧内容当成实时输出
      const staleMin = Math.max(0, Math.round((nowTs - fsMod) / 60000));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        logs: tail, totalLines: out.length, file: logFile,
        mtime_ms: fsMod, stale_min: staleMin, cleared: cleared,
        // 10-05 判障透传：日志源没能在跑引擎的祖先链上定位到（跌到 ⑤ 兜底 vllm.log）
        // 时明示，避免把陈年兜底文件当成"当前日志"却毫无提示。
        fallback: !!logSrc.fallback,
      }));
    } catch (e) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ logs: '', totalLines: 0, error: 'No vLLM log found', file: logFile }));
    }
    return;
  }

  // === Internal API: Stats ===
  // === Internal API: system memory (Linux /proc/meminfo) ===
  if (pathname === '/v1/internal/sysmem') {
    try {
      const mi = {};
      try {
        const txt = fs.readFileSync('/proc/meminfo', 'utf8');
        for (const line of txt.split('\n')) {
          const mm = line.match(/^(\w+):\s+(\d+)\s*kB/);
          if (mm) mi[mm[1]] = parseInt(mm[2], 10);
        }
      } catch (e) { /* non-Linux or unreadable */ }
      if (!mi.MemTotal) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ supported: false }));
        return;
      }
      const total = mi.MemTotal;
      const avail = mi.MemAvailable !== undefined ? mi.MemAvailable : (mi.MemFree || 0);
      const used = Math.max(0, total - avail);
      const buffCache = (mi.Buffers || 0) + (mi.Cached || 0) + (mi.SReclaimable || 0);
      const swapTotal = mi.SwapTotal || 0;
      const swapUsed = swapTotal > 0 ? Math.max(0, swapTotal - (mi.SwapFree || 0)) : 0;
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        supported: true,
        total_kb: total,
        used_kb: used,
        available_kb: avail,
        buff_cache_kb: buffCache,
        used_pct: total > 0 ? parseFloat((used / total * 100).toFixed(1)) : 0,
        swap_total_kb: swapTotal,
        swap_used_kb: swapUsed,
      }));
    } catch (e) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: e.message }));
    }
    return;
  }

  if (pathname === '/v1/internal/pd-series') {
    const S = global.__pdSeries || { points: [] };
    try {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ points: S.points }));
    } catch (e) {}
    return;
  }

  // ===== 0命中占比（近 1h 滚动窗口，20s 缓存）=====
  // 口径：完成请求中 cached_tokens==0 的条数占比；>16k 大 prompt 单列 big 子集
  //（大请求 0 命中意味着全额 prefill 重算，是拖低命中率与 TTFT 的主因）。
  // 数据源双轨：① vLLM request-traces.jsonl（dsh_vllm_logger 插件写入，runtime=sglang 记录不计）；
  // ② SGLang 实例日志 Prefill batch #cached-token 逐请求环（sglangTraceRecords，09-01 加——
  //    生产全切 SGLang 后 trace 文件停更，徽标此前对 SGLang 流量失明）。
  // 窗口内无任何记录时返回 null（前端保持 '--'），避免 0/0 显示成「0% 全命中」。
  let __zeroHitCache = { at: 0, data: null };
  function zeroHitStats() {
    const now = Date.now();
    if (__zeroHitCache.data && now - __zeroHitCache.at < 20000) return __zeroHitCache.data;
    let total = 0, zero = 0, bigTotal = 0, bigZero = 0;
    // ① vLLM：request-traces.jsonl
    try {
      const p = path.join(__dirname, 'request-traces.jsonl');
      const fd = fs.openSync(p, 'r');
      try {
        const sz = fs.fstatSync(fd).size;
        // 忙时 1h ~500 条 ≈ 200KB，tail 1MB 余量充足；末尾残行 JSON.parse 失败自动跳过
        const LEN = Math.min(1024 * 1024, sz);
        const buf = Buffer.alloc(LEN);
        fs.readSync(fd, buf, 0, LEN, sz - LEN);
        const cutoff = now / 1000 - 3600;
        for (const l of buf.toString('utf8').split('\n')) {
          if (!l.trim()) continue;
          let rec; try { rec = JSON.parse(l); } catch (e2) { continue; }
          if (!rec || rec.runtime === 'sglang' || typeof rec.t !== 'number' || rec.t < cutoff) continue;
          if (!traceTsPlausible(rec.t)) continue; // 09-18：脏时钟记录不计入 1h 窗口
          total++;
          const cached = rec.cached_tokens || 0;
          if (cached === 0) zero++;
          if ((rec.prompt_tokens || 0) > 16384) { bigTotal++; if (cached === 0) bigZero++; }
        }
      } finally { fs.closeSync(fd); }
    } catch (e) {} // 文件不存在/读失败 → 该源计 0，不拖累 SGLang 侧
    // ② SGLang：日志逐请求环
    let sgl; try { sgl = sglangTraceRecords(); } catch (e) { sgl = []; }
    for (const r of sgl) {
      total++;
      if (r.cached === 0) zero++;
      if (r.prompt > 16384) { bigTotal++; if (r.cached === 0) bigZero++; }
    }
    const data = total === 0 ? null : {
      window_min: 60, total, zero,
      pct: parseFloat((zero / total * 100).toFixed(1)),
      big: { total: bigTotal, zero: bigZero },
    };
    __zeroHitCache = { at: now, data };
    return data;
  }

  // ===== 快速启动（08-30 定版）：两卡一键拉起，每卡可选引擎规格 =====
  // 预设定义在 quickstart-presets.json（热加载，改 JSON 即生效，无需重启控制台）；
  // 脚本为实跑正典（cmdline+environ 重建）。已运行端口自动跳过（不杀生产实例）。
  function quickStartCards() {
    try { return JSON.parse(fs.readFileSync(path.join(__dirname, 'quickstart-presets.json'), 'utf8')); }
    catch (e) { return null; }
  }
  function qsCardStatus(cards, key) {
    const card = cards[key];
    if (!card) return null;
    let live = null, engine = null;
    try {
      // 卡上注册的脚本化模型端口优先（如 Flash-Next 18420：实例活着就永远算运行中，
      // 不依赖「默认预设恰好是脚本预设」）
      const smk = card.scriptModel && SCRIPT_MODELS[card.scriptModel];
      if (smk) {
        const inst = scriptModelInstance(Object.assign({ key: card.scriptModel }, SCRIPT_MODELS[card.scriptModel]));
        if (inst) { live = inst; engine = 'script'; }
      }
      if (!live) {
        const v = listVllmInstances(true).find(x => x.port === card.port);
        const s = listSglangInstances().find(x => x.port === card.port);
        if (v) { live = v; engine = 'vllm'; } else if (s) { live = s; engine = 'sglang'; }
      }
      if (!live) {
        // 兜底：该卡默认预设是脚本化模型时，按 /proc cmdline 认领 chroot 内实例
        const def = (card.presets || []).find(p => p.standard) || (card.presets || [])[0];
        if (def && def.engine === 'script') {
          const sm = scriptModelForName(def.modelName) || scriptModelForPort(def.port);
          const inst = sm ? scriptModelInstance(sm) : null;
          if (inst) { live = inst; engine = 'script'; }
        }
      }
    } catch (e) {}
    return { key, gpu: card.gpu, port: (live && live.port) || card.port, presets: card.presets, running: !!live, pid: live ? live.pid : null, engine };
  }
  // 脚本预设（Flash-Next 这类 chroot 脚本化模型）的运行态：chroot 内引擎属 root，
  // ll 的 lsof/ss 看不到监听端口，一律按 /proc/<pid>/cmdline 的 --port 匹配认领。
  function qsScriptStatus(preset) {
    const sm = scriptModelForName(preset.modelName) || scriptModelForPort(preset.port);
    const inst = sm ? scriptModelInstance(sm) : null;
    const port = (inst && inst.port) || preset.port;
    return { key: preset.cardKey || 'script', gpu: preset.gpu, port, presets: preset.cardPresets,
      running: !!inst, pid: inst ? inst.pid : null, engine: 'script',
      scriptModel: sm ? sm.key : (preset.modelName || null) };
  }
  // 脚本化预设的停止：走注册模型的宿主停止脚本（内部按 --port 匹配后 sudo kill，
  // SIGTERM 优先，绝不 SIGKILL 持 CUDA 上下文的进程）。等待上限须覆盖 W4A16 停止脚本
  // 自身的 90s 优雅退出窗口（SIGTERM → 等退净 → 才兜底 SIGKILL）。
  async function qsStopScript(smKey, port) {
    const sm = (smKey && SCRIPT_MODELS[smKey]) ? Object.assign({ key: smKey }, SCRIPT_MODELS[smKey]) : scriptModelForPort(port);
    if (!sm || !sm.stopScript) return '未注册停止脚本';
    const stopPath = resolveStopScript(sm);
    if (!stopPath || !fs.existsSync(stopPath)) return '停止脚本不存在：' + (stopPath || '(未注册)');
    let out = '';
    try {
      // 09-20：execSync(180s) 会冻结事件循环（停止期间整个控制台无响应），改异步 execFile
      // 09-26：栈感知（resolveStopScript）+ 超时 240s，覆盖新栈脚本 90s 优雅退出 + 120s 显存轮询
      out = await execFileAsync('bash', [stopPath, String(port)], { encoding: 'utf8', timeout: 240000, maxBuffer: 4 * 1024 * 1024 });
    } catch (e) { out = String((e && e.stdout || '') + (e && e.stderr || '') || (e && e.message) || e); }
    Object.keys(VLLM_MODEL_PORTS).forEach(k => { if (VLLM_MODEL_PORTS[k] === port) delete VLLM_MODEL_PORTS[k]; });
    try { global.__GPU_INSTANCES.delete(port); } catch (e) {}
    const still = scriptModelInstance(sm);
    console.log(`[quickstart-script] stop ${sm.key} port ${port}${still ? ' (WARN: 仍有残留 pid=' + still.pid + ')' : ' (已彻底停止)'}`);
    return String(out).trim().slice(-400);
  }
  // 按指定端口判定运行态（多端口预设：跳过/健康检测跟所选预设的端口走，不跟 card.port 走）
  function qsPortStatus(cards, key, port) {
    const card = cards[key];
    if (!card) return null;
    let live = null, engine = null;
    try {
      const v = listVllmInstances(true).find(x => x.port === port);
      const s = listSglangInstances().find(x => x.port === port);
      if (v) { live = v; engine = 'vllm'; } else if (s) { live = s; engine = 'sglang'; }
    } catch (e) {}
    return { key, gpu: card.gpu, port, presets: card.presets, running: !!live, pid: live ? live.pid : null, engine };
  }
  // 参数预设启动：走「启动模型」弹窗「启动」按钮完全相同的 startVllmModel/startSglangModel，
  // 与弹窗路径零漂移（校验/清理/命令构造/环境全部同源）
  function qsLaunchParams(preset) {
    return new Promise(resolve => {
      const cb = r => resolve(r || { success: false, error: '启动无返回' });
      try {
        const pr = preset.params || {};
        const _startP = preset.engine === 'sglang' ? startSglangModel(pr.modelName, pr, cb) : startVllmModel(pr.modelName, pr, cb);
        Promise.resolve(_startP).catch(e => { console.log('[quickstart] launch error: ' + (e && e.message || e)); try { cb({ success: false, error: String((e && e.message) || e) }); } catch (_) {} });
      } catch (e) { resolve({ success: false, error: String((e && e.message) || e) }); }
    });
  }
  // 脚本化模型预设（如 Qwen3.8-Flash-Next W4A16：chroot 镜像内启动，无法走标准 vLLM 路径）：
  // 与「模型卡片 → 启动（脚本）」按钮同一条链路——scriptModelLaunchPlan 把弹窗参数映射成
  // FN_* 环境变量 → 宿主 wrapper 落成 env 文件（sudo 会清环境，chroot 内靠文件）→ chroot 内
  // 参数化脚本构造 argv。端口/served 注册也与弹窗路径一致。
  function qsLaunchScript(preset) {
    return new Promise(resolve => {
      const done = r => resolve(r || { success: false, error: '脚本启动无返回' });
      try {
        const sm = scriptModelForName(preset.modelName);
        if (!sm) { done({ success: false, error: '未注册的脚本化模型：' + preset.modelName }); return; }
        const inst0 = scriptModelInstance(sm);
        if (inst0) { done({ success: false, alreadyRunning: true, error: `已在运行（端口 ${inst0.port}，PID ${inst0.pid}）` }); return; }
        const plan = scriptPlanFor(sm, Object.assign({ modelName: preset.modelName }, preset.params || {})); // [sglang-adapt-1003] + [model-variant-1008] 预设 modelName 参与变体识别
        const startScript = resolveStartScript(sm);
        // INNER 只对旧栈 wrapper 有意义；新栈 wrapper 的 INNER 缺省即自家 inner
        const envPrefix = (startScript === sm.script && sm.inner) ? `INNER=${sm.inner} ` : '';
        // 上次启动落盘的 FN_*/SG_* 若不清掉，本次「没写到的变量」会继承旧值（sudo 会清环境，
        // chroot 内的 inner 只能靠这个文件）。09-15 实跑就被残留的 FN_SPEC=none 影响过。
        // 09-26：清哪个 env 文件跟解析后的栈走——新栈的 launch.env 在 vllm-0300/ 下。
        // [sglang-adapt-1003] SGLang 脚本栈：launch.env 在 sglang-18420/ 下（wrapper 每次覆写，
        // 但本次 plan 未写到的 SG_* 若残留在文件里仍会被 source 出上次的值 → 必须先清）。
        const envFile = preset.envFile || (sm.scriptSglang && startScript === sm.scriptSglang
                        ? path.join(path.dirname(sm.scriptSglang), 'launch.env')
                        : (startScript.match(/0310/) ? '/home/ll/deploy/vllm-0310/launch.env'  // [model-variant-1008] 0310 栈 envFile 此前算错→清错文件
                        : (startScript.match(/0300/) ? '/home/ll/deploy/vllm-0300/launch.env'
                        : ((sm.script || '').match(/w4a16/) ? '/home/ll/deploy/flash-next-launch-w4a16.env' : ''))));
        if (envFile) { try { fs.writeFileSync(envFile, ''); } catch (e) {} }
        const child = require('child_process').spawn('bash', ['-c', `${envPrefix}setsid bash ${startScript} >> ${scriptLaunchLog(sm)} 2>&1 < /dev/null & echo $!`], {
          detached: true,
          stdio: ['ignore', 'pipe', 'ignore'],
          env: Object.assign({}, process.env, plan.env),
        });
        let outBuf = '';
        child.stdout.on('data', d => { outBuf += d.toString(); });
        child.on('error', e => done({ success: false, error: '脚本启动失败: ' + e.message }));
        child.on('close', () => {
          const pid = parseInt(String(outBuf).trim()) || null;
          VLLM_MODEL_PORTS[plan.served] = plan.port;
          VLLM_MODEL_PORTS[sm.key] = plan.port;
          MODEL_ALIASES[sm.key] = plan.served;
          (sm.dirNames || []).forEach(dn => { MODEL_ALIASES[dn] = plan.served; });
          if (global.__GPU_INSTANCES && typeof global.__GPU_INSTANCES.set === 'function') {
            global.__GPU_INSTANCES.set(plan.port, { port: plan.port, gpuId: 0, gpuCount: plan.gpuCount || (sm.base && sm.base.pp) || 2, runtime: plan.runtime || 'vllm', model: sm.key, startedAt: Date.now() });
          }
          console.log(`[quickstart-script] ${sm.key} port=${plan.port} served=${plan.served} pid=${pid} :: ${plan.summary}`);
          done({ success: true, script: true, pid, port: plan.port, served: plan.served, summary: plan.summary, warnings: plan.warnings });
        });
      } catch (e) { done({ success: false, error: String((e && e.message) || e) }); }
    });
  }
  // ===== 快速启动预设保存（「启动模型」弹窗 →「保存到快速启动」，08-30）======
  // 把弹窗完整参数集存为指定卡的 custom 预设（params 预设，启动时走与弹窗同源的启动函数），
  // 替换该卡已有 custom 预设、撤销其它 standard 标记、卡端口跟随保存值。
  if (pathname === '/v1/internal/quickstart/save' && req.method === 'POST') {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', () => {
      let data = {};
      try { data = JSON.parse(body || '{}'); } catch (e) {}
      const target = String(data.target || '').trim();
      const params = (data.params && typeof data.params === 'object') ? data.params : null;
      const cards = quickStartCards() || {};
      if (!cards[target] || !params) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ ok: false, error: '参数缺失（target/params）' }));
      }
      if (!params.modelName || !fs.existsSync(path.join(MODELS_DIR, params.modelName))) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ ok: false, error: '模型目录不存在: ' + params.modelName }));
      }
      const portN = parseInt(params.port);
      if (!portN || portN < 1024 || portN > 65535) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ ok: false, error: '端口非法' }));
      }
      const eng = String(params.runtime || 'vllm') === 'sglang' ? 'sglang' : 'vllm';
      const card = cards[target];
      card.port = portN;
      const tk = parseInt(params.mtpTokens);
      const spec = (params.dflash === '1') ? 'DFlash2×' + Math.min(8, Math.max(1, tk || 7))
        : (params.dspark === '1') ? 'DSPARK×' + (tk || 8)
        : (eng === 'vllm' && params.mtp === '1') ? 'MTP×' + Math.min(8, Math.max(1, tk || 5))
        : '基线';
      const name = (eng === 'sglang' ? 'SGLang' : 'vLLM') + ' · ' + spec + '（弹窗保存）';
      card.presets = (card.presets || []).filter(x => x.key !== 'custom');
      (card.presets || []).forEach(x => { x.standard = false; });
      card.presets.push({ key: 'custom', name, engine: eng, standard: true, port: portN, params });
      try {
        fs.writeFileSync(path.join(__dirname, 'quickstart-presets.json'), JSON.stringify(cards, null, 2));
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ ok: false, error: '写入失败: ' + e.message }));
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, target, name, port: portN }));
    });
    return;
  }
  // ===== 快速启动预设删除（08-30）：每卡至少保留一条；删默认则剩余第一条提为默认 =====
  if (pathname === '/v1/internal/quickstart/delete' && req.method === 'POST') {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', () => {
      let data = {};
      try { data = JSON.parse(body || '{}'); } catch (e) {}
      const target = String(data.target || '').trim();
      const key = String(data.key || '').trim();
      const cards = quickStartCards() || {};
      const card = cards[target];
      if (!card || !key || !(card.presets || []).some(x => x.key === key)) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ ok: false, error: '预设不存在（target/key）' }));
      }
      card.presets = card.presets.filter(x => x.key !== key);
      if (!card.presets.length) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ ok: false, error: '每卡至少保留一条预设' }));
      }
      if (!card.presets.some(x => x.standard)) card.presets[0].standard = true;
      const restPort = (card.presets.find(x => x.port) || {}).port;
      if (restPort) card.port = restPort;
      try {
        fs.writeFileSync(path.join(__dirname, 'quickstart-presets.json'), JSON.stringify(cards, null, 2));
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ ok: false, error: '写入失败: ' + e.message }));
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, target, deleted: key, standard: (card.presets.find(x => x.standard) || {}).key || null }));
    });
    return;
  }
  if (pathname === '/v1/internal/quickstart' && req.method === 'GET') {
    const cards = quickStartCards() || {};
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true, cards: Object.keys(cards).map(k => qsCardStatus(cards, k)) }));
    return;
  }
  if (pathname === '/v1/internal/quickstart' && req.method === 'POST') {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', async () => {
      const cards = quickStartCards() || {};
      let target = 'both', want = {}, restart = false;
      let actionAllowed = false;
      try { const b = JSON.parse(body || '{}'); target = b.target || 'both'; want = b.presets || {}; restart = !!b.restart; actionAllowed = String(b.confirm || '') === 'restart'; } catch (e) {}
      const keys = target === 'both' ? Object.keys(cards) : [target];
      const results = [];
      (async () => {
        for (const k of keys) {
          const card = cards[k];
          if (!card) { results.push({ key: k, status: 'unknown' }); continue; }
          const preset = card.presets.find(p => p.key === want[k]) || card.presets.find(p => p.standard) || card.presets[0];
          const pPort = preset.port || card.port;
          const isScriptPreset = preset.engine === 'script';
          // 脚本化预设（Flash-Next）与脚本模型端口：状态判定/健康检测都走 /proc cmdline 认领
          const st = isScriptPreset ? qsScriptStatus(Object.assign({ cardKey: k, cardPresets: card.presets }, preset))
            : (card.scriptModel && SCRIPT_MODELS[card.scriptModel] && scriptModelInstance(Object.assign({ key: card.scriptModel }, SCRIPT_MODELS[card.scriptModel]))
                ? qsScriptStatus({ cardKey: k, cardPresets: card.presets, port: pPort, gpu: card.gpu, modelName: card.scriptModel })
                : qsPortStatus(cards, k, pPort));
          if (st.running) {
            // 脚本化预设：行内「停止」可用。默认不自动停（保护生产实例），
            // 仅当请求显式带 restart=1 时先停再起。
            if (restart) {
              if (!actionAllowed) {
                results.push({ ...st, preset: preset.key, presetName: preset.name, status: 'refused', error: '需要确认口令 restart（快速启动默认不动运行中的卡）' });
                continue;
              }
              if (isScriptPreset || scriptModelForPort(pPort)) {
                const info = await qsStopScript(isScriptPreset ? preset.modelName : null, pPort);
                results.push({ ...st, preset: preset.key, presetName: preset.name, status: 'stopped', stopOutput: info });
                continue;
              }
              results.push({ ...st, preset: preset.key, presetName: preset.name, status: 'running', error: '标准实例请到「运行中」卡片点停止' });
              continue;
            }
            results.push({ ...st, preset: preset.key, presetName: preset.name, status: 'running' });
            continue;
          }
          const t0 = Date.now();
          let pid = null, healthy = false;
          try {
            const hp = preset.port || card.port; // 参数预设可能用与卡默认不同的端口
            let launched;
            if (isScriptPreset) {
              launched = await qsLaunchScript(preset);
            } else if (preset.script) {
              // 脚本预设：setsid 后台，独立于控制台进程
              const { spawn } = require('child_process');
              const child = spawn('bash', ['-c', `setsid bash ${preset.script} >> ${preset.log} 2>&1 < /dev/null & echo $!`], { detached: true, stdio: ['ignore', 'pipe', 'ignore'] });
              child.stdout.on('data', d => { pid = parseInt(d.toString().trim().split(/\s+/)[0]) || null; });
              await new Promise(r => { child.on('close', r); setTimeout(r, 3000); });
              launched = { success: true };
            } else {
              // 参数预设：与「启动模型」弹窗同一条启动链路
              launched = await qsLaunchParams(preset);
            }
            if (!launched || !launched.success) throw new Error((launched && launched.error) || '启动失败（进程未存活）');
            for (let i = 0; i < (isScriptPreset ? 150 : 30) && !healthy; i++) {
              await new Promise(r => setTimeout(r, 5000));
              healthy = await new Promise(ok => {
                const hreq = http.get(`http://127.0.0.1:${hp}/health`, r2 => { ok(r2.statusCode === 200); r2.resume(); });
                hreq.on('error', () => ok(false));
                hreq.setTimeout(2000, () => { hreq.destroy(); ok(false); });
              });
            }
            results.push({ ...(isScriptPreset ? qsScriptStatus(Object.assign({ cardKey: k, cardPresets: card.presets }, preset)) : qsPortStatus(cards, k, pPort)), preset: preset.key, presetName: preset.name, status: healthy ? 'healthy' : 'starting', pid: pid || (launched.pid || null), waited_s: Math.round((Date.now() - t0) / 1000), log: preset.log || (preset.engine === 'sglang' ? SGLANG_LOG_PATH : path.join(__dirname, 'vllm.log')) });
          } catch (e) {
            results.push({ key: k, preset: preset.key, status: 'error', error: String((e && e.message) || e) });
          }
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: true, results }));
      })();
    });
    return;
  }
  if (pathname === '/v1/internal/stats') {
    const startTime = Date.now();

    // [stats-cache 1006 r18] 350ms 结果缓存：本页有主链 + 并发卡两路 500ms 轮询、多标签页
    // 再乘一遍，同一窗口内 stats 会被重复完整计算（全量 parseMetrics + 并发明细 + 计费聚合）。
    // 高负载期（该机 load avg ~5.5，实测 stats p95≈1.9s、峰值 4s）重复计算挤兑事件循环、
    // 进一步拉长每一轮响应——是"仪表盘经常停刷"的后端放大器。命中缓存直接回上次序列化的
    // JSON；未命中走完整计算并回填（含 {} 空响应同样缓存，引擎重启窗口不重算）。
    {
      const _sc = global.__statsCache;
      if (_sc && _sc.body && Date.now() - _sc.at < 350) {
        try {
          res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
          res.end(_sc.body);
        } catch (e) {}
        return;
      }
    }
    // 09-20：/metrics 走缓存+单飞（400ms TTL）——此前每次 stats 都实时抓 64KB 上游
    fetchMetricsCached(`${vllmBaseUrl}/metrics`, 400, 3000).then(async (data) => {
        // [hw-move 1005] 主端口拿不到 metrics 也是端口漂移的直接证据，顺手触发一次
        // 再探测（内部 5s 节流；探不到保持现状）。防止 stats 长期 {} 而无人纠偏。
        if (!data) { try { maybeReDetectBackend(); } catch (e) {} }
        try {
          const m = parseMetrics(data);
          // SGLang 运行时：指标命名/结构不同，走独立构建器（vllm 路径保持不变）
          if (metricsNamespace(m) === 'sglang') {
            const sg = buildSglangStats(m, global.__tokTicker);
            // 引擎实测总生成吞吐（3s 窗口，gen_throughput gauge 平均，与 tg TPS 同源）
            const genSpeed1s = buildSglangGenSpeed1s(m);
            // 引擎实测预填充吞吐（3s 窗口，prefill_effective_tokens_total{mode=input} 差值）
            const prefillSpeed3s = buildSglangPrefillSpeed3s(m);
            // 引擎预填充累计（mode=input），用于单请求预填充进度（首次观察差值）
            const ppTotalNow = Math.round(counterByLabel(m, 'sglang:prefill_effective_tokens_total', 'mode', 'input'));
            // 并发请求明细：与 vLLM 对齐，用进行中的 live 流构建每请求行（无 live 数据时回落聚合显示）
            // 与实例 ticker 路径一致：按 model 名过滤出本实例的 live 流——主路径原先不过滤，
            // 会把其他运行时/实例（如 vLLM 8001）的进行中请求混进 SGLang 卡片
            {
              let _primModel = '';
              try {
                const _mk = Object.keys(m).find(x => x.indexOf('sglang:num_running_reqs|') === 0);
                if (_mk) _primModel = (JSON.parse(_mk.substring(_mk.indexOf('|') + 1)).model_name) || '';
              } catch (e) {}
              const _lsPrim = new Map();
              for (const [id, e] of global.__liveStreams.map) {
                if (e && !e.done && (_primModel === '' || e.model === _primModel)) _lsPrim.set(id, e);
              }
              // 08-30：每请求缓存命中真值（日志 #cached-token，vLLM 口径对齐）
              const _primSg = listSglangInstances().find(x => x.port === sg.port) || {};
              const _primLines = readSglangPrefillLines(sg.port, _primSg.pid);
              sg.active_requests = buildSglangActiveRequests(sg, { map: _lsPrim }, genSpeed1s, prefillSpeed3s, ppTotalNow,
                (e) => sglangCachedForRequest(_primLines, e, e.promptTokens));
            }
            // 实时 token 流（tee 捕获），前端与 active_requests 按顺序 zip 展示
            sg.live_streams = liveStreamsSnapshot();
            // ====== Real-time billing（与 vLLM 分支同构：持久化累加，重启不丢）======
            // sglang 无 source 标签区分缓存命中，用 cached_tokens_total 作为缓存命中输入，
            // prompt_tokens_total − cached_tokens_total 作为未缓存输入（与 buildSglangStats 口径一致）
            {
              const cachedRaw = Math.round(counterByLabel(m, 'sglang:prefill_effective_tokens_total', 'mode', 'device_hit')) + Math.round(counterByLabel(m, 'sglang:prefill_effective_tokens_total', 'mode', 'host_hit'));
              const promptRaw = Math.round(counterTotal(m, 'sglang:prompt_tokens_total'));
              const genRaw = Math.round(counterTotal(m, 'sglang:generation_tokens_total'));
              const st = accumulateBilling({ cached: cachedRaw, uncached: Math.max(0, promptRaw - cachedRaw), gen: genRaw });
              const b = computeBilling(sg, getCurrentVllmModel()); // 保留单价/币种/模型匹配等元信息
              const totalOf = (bk) => (bk.cost.cached_input || 0) + (bk.cost.uncached_input || 0) + (bk.cost.output || 0);
              b.current = { cost: { ...st.current.cost, total: totalOf(st.current) }, tokens: st.current.tokens };
              b.history = { cost: { ...st.history.cost, total: totalOf(st.history) }, tokens: st.history.tokens };
              b.days = buildBillingDays(st); // 每日费用明细（持久化，可手动删除）
              sg.billing = b;
              // 08-30 跨运行时聚合（与 vLLM 分支对称）：SGLang 主实例 + 所有 vLLM 实例
              // 各自成组进并发卡片——否则 vLLM 独占的卡（如 GPU0 的 vllm 8001）整组消失
              const _sgModelK = Object.keys(m).find(x => x.indexOf("sglang:num_running_reqs|") === 0);
              let _sgModel = "";
              try { if (_sgModelK) _sgModel = (JSON.parse(_sgModelK.substring(_sgModelK.indexOf("|") + 1)).model_name || ""); } catch (e) {}
              // 09-01: 主实例 GPU 从进程列表解析（原写死 null → 主行无 GPU 徽标、无法按卡分组；缺 runtime → 徽标误显示 vllm）
              const _sglSelf = listSglangInstances().find(x => x.port === config.vllmPort) || null;
              const _sglPrimaryInst = {
                port: config.vllmPort,
                gpu: _sglSelf ? _sglSelf.gpu : null, model: _sgModel,
                running: sg.running || 0, queued: sg.queued || 0,
                active_requests: sg.active_requests || [], waiting_requests: [],
                last_second: sg.last_second || null, kv: sg.kv_cache,
                primary: true,
                runtime: 'sglang',
                gen_speed_1s: (genSpeed1s !== undefined && genSpeed1s >= 0) ? genSpeed1s : 0,
              };
              let _sglInstList = [_sglPrimaryInst];
              try {
                const _allVllm = listVllmInstances();
                for (const _vi of _allVllm.filter(i => i.port !== config.vllmPort)) {
                  const _vres = await fetchInstanceConcurrency(_vi);
                  if (_vres) _sglInstList.push(_vres);
                }
                // tg TPS = 全实例（全部 GPU）生成吞吐总和
                let _allGenSpeed = 0;
                for (const _i of _sglInstList) _allGenSpeed += (_i.gen_speed_1s || 0);
                if (_allGenSpeed > 0) sg.bench_tg_tps = parseFloat(_allGenSpeed.toFixed(1));
                // 09-01 修复：SGLang 主实例下 SGLang 从实例从未聚合（如 8000/GPU0 的 DFLASH 实例）→
                // 该卡整组（并发请求/模型信息）从仪表盘消失。与 vLLM 主分支对称补聚合。
                for (const _sgi of listSglangInstances().filter(i => i.port !== config.vllmPort)) {
                  const _sres = await fetchSglangInstanceConcurrency(_sgi);
                  if (_sres) _sglInstList.push(_sres);
                }
                _sglInstList.sort((a, b) =>
                  (((a.gpu == null) ? 999 : a.gpu) - ((b.gpu == null) ? 999 : b.gpu)) || (a.port - b.port));
              } catch (e) { /* 从实例聚合失败不影响主实例 */ }
              sg.instances = _sglInstList;
              rememberPortGpu(_sglInstList); // 最近请求表 GPU 归因回溯
            }
            sg.cache_per_port = perPortCacheStats();
            // [kvoff-display 09-22] SGLang 主分支同样输出（18420 这类 vLLM 从实例也要显示）
            sg.kv_offload_ports = kvOffloadPortsInfo();
            // [kvoff-live 09-27] 二级缓存物理驻留（/dev/shm tmpfs 已用），随 stats 常驻下发
            sg.kv_offload_mem = kvOffloadShmUsage();
            // 0命中占比（vLLM+SGLang 双源，09-01 修复：此前仅 vLLM 分支调用，SGLang 主实例下恒缺失）
            try { const _zh = zeroHitStats(); if (_zh) sg.zero_hit = _zh; } catch (e) {}
            try {
              sg.sglang_theory = {};
              for (const _ti of listSglangInstances()) { const _tt = readSglangTheory(_ti.port); if (_tt) sg.sglang_theory[_ti.port] = _tt; }
            } catch (e) {}
            try { if (res && !res.headersSent) { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(sg)); } } catch (e) {}
            return;
          }
          // Build model label dynamically from metrics
          // Derive model labels dynamically
          const mlGen = 'vllm:generation_tokens_total' + buildModelLabelFromMetrics(m, 'vllm:generation_tokens_total{');
          const mlPrompt = 'vllm:prompt_tokens_total' + buildModelLabelFromMetrics(m, 'vllm:prompt_tokens_total{');
          const mlCachePct = 'vllm:kv_cache_usage_perc' + buildModelLabelFromMetrics(m, 'vllm:kv_cache_usage_perc{');
          const mlRunning = 'vllm:num_requests_running' + buildModelLabelFromMetrics(m, 'vllm:num_requests_running{');
          const mlReqGen = 'vllm:request_generation_tokens_count' + buildModelLabelFromMetrics(m, 'vllm:request_generation_tokens_count{');
          // For metrics with _count/_sum suffixes, extract label from _count key
          // and use it for both _count and _sum lookups
          const mlTTFT = 'vllm:time_to_first_token_seconds_count' + buildModelLabelFromMetrics(m, 'vllm:time_to_first_token_seconds_count{');
          const mlTpot = 'vllm:request_time_per_output_token_seconds_count' + buildModelLabelFromMetrics(m, 'vllm:request_time_per_output_token_seconds_count{');
          const mlReqDecode = 'vllm:request_decode_time_seconds_count' + buildModelLabelFromMetrics(m, 'vllm:request_decode_time_seconds_count{');
          const mlReqPrompt = 'vllm:request_prompt_tokens' + buildModelLabelFromMetrics(m, 'vllm:request_prompt_tokens{');
          const mlReqPrefill = 'vllm:request_prefill_time_seconds_count' + buildModelLabelFromMetrics(m, 'vllm:request_prefill_time_seconds_count{');

          // For prompt_tokens_by_source_total, find each source separately
          // parseMetrics keys are: vllm:prompt_tokens_by_source_total|{"engine":"0","model_name":"foo","source":"bar"}
          let mlCachePrefix = '';
          for (const k of Object.keys(m)) {
            if (k.startsWith('vllm:prompt_tokens_by_source_total|')) {
              mlCachePrefix = k.substring(0, k.indexOf('"source"'));
              break;
            }
          }
          const mlCacheHit = mlCachePrefix ? mlCachePrefix + '"source":"local_cache_hit"}' : '';
          const mlCacheLocal = mlCachePrefix ? mlCachePrefix + '"source":"local_compute"}' : '';

          const genTokensTotal = Math.round(m[mlGen] || 0);
          const kvUsage = m[mlCachePct] || 0;

          // ====== KV Cache 池容量（主实例，与从实例同口径 buildKvCacheInfo）======
          const kvCache = buildKvCacheInfo(m);

          const result = {
            cache_blocks: {
              used: Math.round(kvUsage * 100),
              total_blocks: kvCache.num_gpu_blocks || 100,
            },
            kv_cache: kvCache,
            num_running_seqs: 0,
            num_queue_seqs: 0,
            num_prompt_tokens: 0,
            num_generation_tokens: 0,
            total_completed_requests: 0,
            total_generated_tokens_all_time: 0,
            avg_tokens_per_request: 0,
            avg_prefill_tokens_per_request: 0,
            avg_decode_time_seconds: 0,
            avg_speed_per_request: 0,
            avg_prefill_time_seconds: 0,
            active_requests: [],
            total_speed: 0,
          };

          // Fetch previous snapshot for delta calculation
          // 09-06 性能修复：原实现每次 stats 请求都 readFileSync+writeFileSync（同步 I/O 阻塞事件循环），
          // 前端 0.5s 轮询 → 每秒 2 次同步磁盘读写。改为内存缓存 + 5s 节流落盘。
          if (!global.__metricsSnapshot) global.__metricsSnapshot = { data: null, dirty: false, lastSave: 0 };
          const snapStore = global.__metricsSnapshot;
          let lastSnapshot = snapStore.data;
          if (!lastSnapshot) {
            lastSnapshot = { offsetTokens: 0, offsetInput: 0, offsetCached: 0, offsetUncached: 0, offsetReqCount: 0, lastRunning: 0, perReqTokens: {}, time: 0 };
            try {
              const snap = fs.readFileSync(path.join(__dirname, 'metrics-snapshot.json'), 'utf8');
              lastSnapshot = JSON.parse(snap);
            } catch (e) {}
          } else {
            // Validate that saved offsets match current model's metric labels
            const needsReset =
              (lastSnapshot.offsetInput > 0 && lastSnapshot.offsetInput > (m[mlPrompt] || 0) * 2 + 1000000) ||
              (lastSnapshot.offsetCached > 0 && lastSnapshot.offsetCached > (m[mlPrompt] || 0) * 2) ||
              (lastSnapshot.offsetTokens > 0 && lastSnapshot.offsetTokens > genTokensTotal * 2 + 1000);
            if (needsReset) {
              lastSnapshot.offsetTokens = genTokensTotal;
              lastSnapshot.offsetInput = Math.round(m[mlPrompt] || 0);
              lastSnapshot.offsetCached = Math.round(m[mlCacheHit] || 0);
              lastSnapshot.offsetUncached = Math.round(m[mlCacheLocal] || 0);
              lastSnapshot.offsetReqCount = Math.round(m[mlReqGen] || 0);
              lastSnapshot.perReqTokens = {};
              lastSnapshot.time = Date.now();
              lastSnapshot.reset = null;
            }
          }

          // Reset baseline: display shows values since last "reset stats".
          // Re-capture when missing (first run) or when the model switched (new label)
          const lbl = detectModelLabels(m);
          const genKey = 'vllm:generation_tokens_total' + lbl.modelLabel;
          if (!lastSnapshot.reset || typeof lastSnapshot.reset !== 'object' || lastSnapshot.reset[genKey] === undefined) {
            lastSnapshot.reset = captureResetBlock(m, lbl.modelLabel, lbl.sourceLabel);
          }
          const mj = applyResetView(m, lastSnapshot.reset);

          const elapsedMs = Date.now() - lastSnapshot.time;
          const rawGenTokensBefore = lastSnapshot.offsetTokens || genTokensTotal;

          // ====== 多卡多实例：探测全部 vllm serve 实例（port/pid/gpu/model）======
          // 主实例 = config.vllmPort（端口自愈跟随的那个）；其余为从实例。
          const vllmInstances = listVllmInstances();
          const primaryInst = vllmInstances.find(i => i.port === config.vllmPort)
            || { port: config.vllmPort, pid: null, gpu: null, modelPath: '', servedName: '' };

          // vLLM 插件实时回传的每请求预填充进度（无文件/未打补丁时为 null）。
          // 多实例共用同一 jsonl：按主实例 pid 选 sid，避免匹配被从实例抢走。
          const livePrefill = readLivePrefill(primaryInst.pid);

          const concurrency = computeConcurrencyDetails(
            mj,
            genTokensTotal,
            rawGenTokensBefore,
            elapsedMs,
            lastSnapshot.lastRunning || 0,
            lastSnapshot.perReqTokens || {},
            livePrefill,
            primaryInst.port,
            global.__tokTicker
          );
          const curRunning = Math.round(m[mlRunning] || 0);
          lastSnapshot.lastRunning = curRunning;
          lastSnapshot.offsetTokens = genTokensTotal;
          if (concurrency._perReqTokens) {
            lastSnapshot.perReqTokens = concurrency._perReqTokens;
          }
          lastSnapshot.time = Date.now();
          // Persist snapshot so the NEXT poll can compute a real delta
          // (without this the file never exists → delta is always 0 → total_speed always 0)
          // 09-06：内存缓存 + 5s 节流落盘（原实现每次请求都 writeFileSync 阻塞事件循环）
          snapStore.data = lastSnapshot;
          const _nowMs = Date.now();
          if (_nowMs - snapStore.lastSave > 5000) {
            snapStore.lastSave = _nowMs;
            try { fs.writeFileSync(path.join(__dirname, 'metrics-snapshot.json'), JSON.stringify(lastSnapshot)); } catch (e) {}
          }
          Object.assign(result, concurrency);
          if (concurrency._v3diag) result.v3 = concurrency._v3diag; // 顶层排障字段：curl stats | jq .v3
          delete result._perReqTokens;
          // Remove debug
          delete result._debug;
          // True measured "last 1 second" numbers (server-side 1s ticker)
          result.last_second = global.__tokTicker ? global.__tokTicker.lastSecond : { tokens: 0, speed: 0, running: 0, at: 0 };
          result.energy = buildEnergyInfo(); // 能耗/电费（1 秒级功耗积分，持久化）

          // ====== Token Counts (since last reset) ======
          const totalInputRaw = Math.round(mj[mlPrompt] || 0);
          const cachedRaw = Math.round(mj[mlCacheHit] || 0);
          const uncachedRaw = Math.round(mj[mlCacheLocal] || 0);
          result.total_input_tokens = totalInputRaw;
          result.total_output_tokens = Math.round(mj[mlGen] || 0);
          result.cached_input_tokens = cachedRaw;
          result.uncached_input_tokens = uncachedRaw;

          // Cache hit rate: cached / total * 100
          const totalInput = cachedRaw + uncachedRaw;
          result.cache_hit_rate = totalInput > 0 ? (cachedRaw / totalInput * 100).toFixed(1) : 0;

          // 0命中占比：近 1h 完成请求中 cached_tokens==0 的占比（request-traces 真值，20s 缓存）
          const zh = zeroHitStats();
          if (zh) result.zero_hit = zh;

          // 每 GPU 累计缓存命中率（各实例 ticker 累计计数器，独立于重置基线）
          result.cache_per_port = perPortCacheStats();
          // [kvoff-display 09-22] 各 vLLM 实例 CPU KV 二级缓存状态（含未启用条目，前端常驻显示）
          result.kv_offload_ports = kvOffloadPortsInfo();
          // [kvoff-live 09-27] 二级缓存物理驻留（/dev/shm tmpfs 已用）
          result.kv_offload_mem = kvOffloadShmUsage();

          // ====== Performance Benchmark Metrics (since last reset) ======
          // pp TPS：优先用 ticker 每秒实测的「未缓存预填充吞吐」（vLLM 每迭代
          // 递增 prompt_tokens_by_source local_compute，增量即真实发生的预填充
          // 计算量，缓存命中不计入）—— 实时精确，随负载即时变化。
          // 无 ticker 数据时回落累计平均：uncached_prompt_tokens / avg_prefill_time
          const tickerPP = (global.__tokTicker && global.__tokTicker.lastSecond && global.__tokTicker.lastSecond.uncachedPPS) || 0;
          const uncachedPP_TPS = (concurrency.avg_prefill_time_seconds > 0 && result.uncached_input_tokens > 0)
            ? (concurrency.avg_prefill_tokens_per_request / concurrency.avg_prefill_time_seconds)
            : 0;
          result.bench_pp_tps = tickerPP > 0
            ? parseFloat(Math.min(tickerPP, 50000).toFixed(1))
            : parseFloat(Math.min(uncachedPP_TPS, 10000).toFixed(1));
          // 诊断/透明：实时 prefill 数据源状态（sid 变化=vLLM 重启；byRid 为
          // 引擎记录的请求数；matched 为本次匹配成功的行数）
          result.live_prefill = {
            sid: livePrefill && livePrefill.sid,
            by_rid: livePrefill && livePrefill.byRid ? livePrefill.byRid.size : 0,
            updated_at: livePrefill && livePrefill.updatedAt,
            matched: (concurrency.active_requests || []).filter(r => r.prefill_exact).length,
          };

          // TTFT: _count and _sum share the same model label
          const ttftCountKey = mlTTFT; // already includes _count suffix
          const ttftSumKey = mlTTFT.replace('_count|', '_sum|');
          const ttftCount = mj[ttftCountKey] || 0;
          const ttftSum = mj[ttftSumKey] || 0;
          result.avg_ttft_ms = ttftCount > 0 ? (ttftSum / ttftCount * 1000) : 0;

          // TPOT
          const tpotCountKey = mlTpot;
          const tpotSumKey = mlTpot.replace('_count|', '_sum|');
          const tpotCount = mj[tpotCountKey] || 0;
          const tpotSum = mj[tpotSumKey] || 0;
          result.avg_tpot_ms = tpotCount > 0 ? (tpotSum / tpotCount * 1000) : 0;

          // Avg decode time
          const decodeCountKey = mlReqDecode;
          const decodeSumKey = mlReqDecode.replace('_count|', '_sum|');
          const reqDecodeCount = mj[decodeCountKey] || 0;
          const reqDecodeSum = mj[decodeSumKey] || 0;
          const avgDecodeTime = reqDecodeCount > 0 ? (reqDecodeSum / reqDecodeCount) : 0;

          // Avg prefill time
          const prefillCountKey = mlReqPrefill;
          const prefillSumKey = mlReqPrefill.replace('_count|', '_sum|');
          const reqPrefillCount = mj[prefillCountKey] || 0;
          const reqPrefillSum = mj[prefillSumKey] || 0;
          const avgPrefillTime = reqPrefillCount > 0 ? (reqPrefillSum / reqPrefillCount) : 0;

          // tg TPS：所有实例（全部 GPU）的 1s 生成吞吐总和（各端口 ticker 独立采样后累加，
          // 多卡多实例 = 全局总生成速度）。主分支此处先算一个初值（仅主实例），
          // instances 聚合完成后用全实例总和覆盖（见下方 result.bench_tg_tps = _allGenSpeed）。
          result.bench_tg_tps = parseFloat((result.last_second && result.last_second.speed) || 0);

          // E2E latency: avg_prefill_time + avg_tokens_per_request * avg_decode_time / avg_tokens_per_request = avg_prefill_time + avg_decode_time
          // Simplified: e2e ≈ avg_prefill_time + avg_tokens_per_request * tpot
          const avgTPOT_seconds = tpotCount > 0 ? (tpotSum / tpotCount) : 0;
          const e2e_ms = concurrency.avg_prefill_time_seconds * 1000 + concurrency.avg_tokens_per_request * avgTPOT_seconds;
          result.bench_e2e_ms = parseFloat(e2e_ms.toFixed(0));

          // Throughput: pp_tps + tg_tps (combined)
          result.bench_throughput = parseFloat((result.bench_pp_tps + (result.bench_tg_tps || 0)).toFixed(1));

          // GPU memory: use model size estimate (~2x params for FP16 weights + KV cache + overhead)
          // qwen3.6-35b is ~35B params, FP8 uses ~0.5 bytes/param = ~17GB weights
          // + KV cache + activation overhead ≈ 25-30GB total
          const modelParams = 35; // 35B
          const memGB = parseFloat((modelParams * 0.5 + 10).toFixed(2)); // weights + KV cache overhead
          result.bench_peak_mem_gb = memGB;

          // ====== MTP Hit Rate (推测解码命中率: accepted / drafted) ======
          // CRITICAL: must read the cumulative counters (..._total), NOT the
          // prometheus "..._created" gauge. The _created value is the metric's
          // creation TIMESTAMP (epoch seconds); both are created in the same
          // millisecond, so accepted/drafted ≈ 1.0 → a constant, meaningless
          // 100%. This matches vLLM's own draft_acceptance_rate =
          // num_accepted_tokens / num_draft_tokens (see SpecDecodingLogging.log).
          {
            let specLabel = '';
            for (const k of Object.keys(m)) {
              if (k.startsWith('vllm:spec_decode_num_draft_tokens_total|')) {
                specLabel = k.substring(k.indexOf('|'));
                break;
              }
            }
            if (specLabel) {
              const drafted  = Math.round(m['vllm:spec_decode_num_draft_tokens_total' + specLabel] || 0);
              const accepted = Math.round(m['vllm:spec_decode_num_accepted_tokens_total' + specLabel] || 0);
              const drafts   = Math.round(m['vllm:spec_decode_num_drafts_total' + specLabel] || 0);
              if (drafted > 0) {
                // Per-token acceptance rate (primary "MTP 命中率"), vLLM convention
                result.mtp_hit_rate = parseFloat((accepted / drafted * 100).toFixed(1));
                result.mtp_draft_tokens = drafted;
                result.mtp_accepted_tokens = accepted;
                // Mean accepted tokens per draft step (incl. the bonus token)
                result.mtp_accept_len = drafts > 0 ? parseFloat((1 + accepted / drafts).toFixed(2)) : 0;
                // Per-position acceptance rates (accepted_at_pos / num_drafts)
                const posRates = [];
                for (let p = 0; p < 16; p++) {
                  const pk = 'vllm:spec_decode_num_accepted_tokens_per_pos_total' + specLabel.replace(/}\s*$/, '') + ',"position":"' + p + '"}';
                  const pv = m[pk];
                  if (pv === undefined) break;
                  posRates.push(drafts > 0 ? parseFloat((pv / drafts * 100).toFixed(1)) : 0);
                }
                if (posRates.length) result.mtp_pos_rates = posRates;
              }
            }
          }


          // ====== Concurrency simulation ======
          const running = concurrency.running || 0;
          const tgTPS = result.bench_tg_tps || 0;
          const basePP_TPS = result.bench_pp_tps || 0;
          const baseTTFT = result.avg_ttft_ms || 0;
          const baseE2E = e2e_ms || 0;

          // ====== Real-time billing (持久化累加：重启不丢、不受重置统计影响) ======
          {
            const st = accumulateBilling({ cached: m[mlCacheHit] || 0, uncached: m[mlCacheLocal] || 0, gen: m[mlGen] || 0 });
            const b = computeBilling(result, getCurrentVllmModel()); // 保留单价/币种/模型匹配等元信息
            const totalOf = (bk) => (bk.cost.cached_input || 0) + (bk.cost.uncached_input || 0) + (bk.cost.output || 0);
            b.current = { cost: { ...st.current.cost, total: totalOf(st.current) }, tokens: st.current.tokens };
            b.history = { cost: { ...st.history.cost, total: totalOf(st.history) }, tokens: st.history.tokens };
            b.days = buildBillingDays(st); // 每日费用明细（持久化，可手动删除）
            result.billing = b;
          }

          // Per-request live token streams (requests routed through this console)
          result.live_streams = liveStreamsSnapshot();

          // ====== 多卡多实例：聚合从实例并发，全部行打 GPU 标签 ======
          // result.active_requests = 主实例行（打 port/gpu/model）；
          // result.instances = 每实例明细 [{port, gpu, model, running, queued,
          //   active_requests, last_second, primary}]，前端按卡分组显示。
          try {
            for (const r of (result.active_requests || [])) {
              r.port = primaryInst.port;
              r.gpu = primaryInst.gpu;
              r.gpus = (primaryInst.gpus && primaryInst.gpus.length) ? primaryInst.gpus.slice() : (primaryInst.gpu != null ? [primaryInst.gpu] : null);
              r.model = primaryInst.servedName || primaryInst.modelPath || '';
            }
            const instList = [{
              port: primaryInst.port,
              gpu: primaryInst.gpu,
              gpus: primaryInst.gpus || (primaryInst.gpu != null ? [primaryInst.gpu] : []),
              world_size: primaryInst.worldSize || 1,
              model: primaryInst.servedName || primaryInst.modelPath || '',
              running: result.running || 0,
              queued: result.queued || 0,
              active_requests: result.active_requests || [],
              waiting_requests: result.waiting_requests || [],
              last_second: result.last_second || null,
              kv: kvCache,
              primary: true,
              runtime: 'vllm',   // [vllm-page-1003] 对称 _sglPrimaryInst；缺该字段时前端徽标判据拿不到运行时
              gen_speed_1s: (result.last_second && result.last_second.speed) || 0,
            }];
            const otherInsts = vllmInstances.filter(i => i.port !== primaryInst.port);
            if (otherInsts.length) {
              const secResults = await Promise.all(otherInsts.map(inst => fetchInstanceConcurrency(inst)));
              for (const s of secResults) if (s) instList.push(s);
            }
            // 08-30 跨运行时聚合：SGLang 实例（如 GPU1 的 sglang serve）也进并发卡片，
            // 与 listVllmInstances 的 GPU 分组口径一致（否则 sglang 独占的卡整组消失）
            try {
              for (const sgi of listSglangInstances()) {
                if (sgi.port === primaryInst.port) continue;
                const sres = await fetchSglangInstanceConcurrency(sgi);
                if (sres) instList.push(sres);
              }
            } catch (e) { /* 跨运行时聚合失败不影响主实例 */ }
            // 按 GPU 号排序（未知排最后），再按端口，显示顺序稳定
            instList.sort((a, b) =>
              (((a.gpu == null) ? 999 : a.gpu) - ((b.gpu == null) ? 999 : b.gpu)) || (a.port - b.port));
            result.instances = instList;
            // tg TPS = 全实例（全部 GPU）1s 生成吞吐总和
            let _allGenSpeed = 0;
            for (const _i of instList) _allGenSpeed += (_i.gen_speed_1s || 0);
            if (_allGenSpeed > 0) result.bench_tg_tps = parseFloat(_allGenSpeed.toFixed(1));
            rememberPortGpu(instList); // 最近请求表 GPU 归因回溯
          } catch (e) { /* 多实例聚合失败不影响主实例统计 */ }

          res.writeHead(200, { 'Content-Type': 'application/json' });
          const _sb = JSON.stringify(result);
          global.__statsCache = { at: Date.now(), body: _sb }; // [stats-cache 1006 r18] 回填缓存
          res.end(_sb);
        } catch (e) {
          console.error('Stats error:', e.message, e.stack);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end('{}');
        }
    }).catch(() => {
      // 上游不可达/超时：返回空对象（与原行为一致；vLLM 挂起时不悬挂）
      global.__statsCache = { at: Date.now(), body: '{}' }; // [stats-cache 1006 r18] 空响应同样短路，引擎重启窗口不重算
      try { if (!res.headersSent) res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({})); } catch (e) {}
    });
    return;
  }

  // === Internal API: Billing reset (新费用/历史总费用 独立重置，与重置统计无关) ===
  if (pathname === '/v1/internal/billing/reset' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); if (body.length > 1e4) req.destroy(); });
    req.on('end', () => {
      try {
        const data = JSON.parse(body || '{}');
        const scope = data.scope === 'history' ? 'history' : 'current';
        const st = resetBillingBucket(scope);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, scope, state: st }));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: e.message }));
      }
    });
    return;
  }

  // === Internal API: Power (各电子元件 + 进程级实时功耗，Scaphandre 式归因) ===
  if (pathname === '/v1/internal/power') {
    let gpu = { name: null, powerW: null, limitW: null, utilization: null, memUsedMiB: null, cardCount: 0 };
    // 09-22：改用 1s 异步采样的全局缓存（sampleGpuStats → global.__gpuLive / __gpuStatic）。
    // 此前这里是 execSync nvidia-smi（timeout 5s）——本机制卡驱动异常时 nvidia-smi 会进
    // D 状态，execSync 的 timeout 对其无效，会把整个 Node 事件循环卡死 → 全站接口
    // （含仪表盘 stats）成片停摆。违反 09-20 铁律「nvidia-smi 必须异步+真超时」。
    try {
      const live = global.__gpuLive;
      const st = global.__gpuStatic;
      if (live && Array.isArray(live.gpus) && live.gpus.length) {
        let power = 0, limit = 0, mem = 0, utilMax = 0, name = null;
        for (const g of live.gpus) {
          power += g.power_draw || 0;
          mem += g.mem_used || 0;
          if ((g.util_gpu || 0) > utilMax) utilMax = g.util_gpu || 0;
        }
        if (st && Array.isArray(st.gpus)) {
          for (const s2 of st.gpus) { name = s2.name || name; limit += s2.power_limit || 0; }
        }
        gpu = {
          name,
          powerW: Math.round(power * 10) / 10,
          limitW: limit ? Math.round(limit * 10) / 10 : null,
          utilization: utilMax,
          memUsedMiB: mem,
          cardCount: live.gpus.length,
        };
      }
    } catch (e) {}
    // 进程级功耗：CPU 活跃功率按 CPU 时间增量比例归因，GPU 功率按显存占用比例归因
    const HZ = 100;
    const snapTs = lastProcSnap ? lastProcSnap.ts : 0;
    const dtSec = lastProcSnap && lastProcDeltas ? Math.max(0.1, (Date.now() - snapTs) / 1000) : 2;
    const deltas = lastProcDeltas || {};
    const totalDelta = Object.values(deltas).reduce((s, d) => s + d, 0);
    const activeCpuW = lastActiveCpuW;
    // 本轮活跃的进程 ∪ 有累计能耗的进程（瞬时空闲的进程也保留显示）
    const pidsToShow = new Set(Object.keys(deltas));
    for (const pid of procEnergyJ.keys()) {
      if ((procEnergyJ.get(pid) || 0) >= 50) pidsToShow.add(String(pid));
    }
    const procList = [];
    for (const pid of pidsToShow) {
      const pidN = parseInt(pid, 10);
      const name = (lastProcSnap && lastProcSnap.pids && lastProcSnap.pids[pid] && lastProcSnap.pids[pid].name) || '?';
      const d = deltas[pid] || 0;
      const cpuW = (totalDelta > 0 && activeCpuW !== null) ? activeCpuW * (d / totalDelta) : 0;
      const cpuPct = d > 0 ? Math.round(((d / HZ) / dtSec / CPU_CORES) * 1000) / 10 : 0;
      const gp = gpuProcs.find(g => g.pid === pidN);
      const gpuMemSum = gpuProcs.reduce((s, g) => s + g.memMiB, 0);
      const gpuW = (gp && gpu.powerW !== null) ? gpu.powerW * (gpuMemSum > 0 ? gp.memMiB / gpuMemSum : 1) : 0;
      const totalW = Math.round((cpuW + gpuW) * 10) / 10;
      if (totalW < 0.1 && (procEnergyJ.get(pidN) || 0) < 50) continue;
      procList.push({
        pid: pidN,
        name,
        cpuPct,
        cpuW: Math.round(cpuW * 10) / 10,
        gpuW: Math.round(gpuW * 10) / 10,
        totalW,
        gpuMemMiB: gp ? gp.memMiB : 0,
        energyWh: Math.round(((procEnergyJ.get(pidN) || 0) / 3600) * 1000) / 1000,
      });
    }
    procList.sort((a, b) => b.totalW - a.totalW || b.energyWh - a.energyWh);
    const procs = procList.slice(0, 10);
    const comps = [
      {
        id: 'gpu',
        name: 'GPU · ' + (gpu.name || 'NVIDIA'),
        measured: gpu.powerW !== null,
        powerW: gpu.powerW,
        limitW: gpu.limitW,
        extra: [gpu.cardCount > 1 ? gpu.cardCount + ' 卡合计' : null,
                gpu.utilization !== null ? '利用率 ' + gpu.utilization + '%' : null,
                gpu.memUsedMiB ? '显存 ' + gpu.memUsedMiB + ' MiB' : null].filter(Boolean).join(' · '),
      },
      { id: 'cpu', name: 'CPU · ' + CPU_MODEL, measured: cpuPowerW !== null, powerW: cpuPowerW,
        extra: [cpuUtilPct !== null ? '利用率 ' + cpuUtilPct + '%' : null, CPU_CORES + ' 核',
                cpuIdleBaselineW !== null ? '空闲基线 ~' + Math.round(cpuIdleBaselineW * 10) / 10 + ' W' : null]
          .filter(Boolean).join(' · ') + ' · RAPL 实时采样' },
      { id: 'nvme', name: 'NVMe SSD', measured: false, powerW: null, note: '硬件未暴露功耗传感器' },
      { id: 'wifi', name: 'Wi-Fi 模块', measured: false, powerW: null, note: '硬件未暴露功耗传感器' },
      { id: 'platform', name: '主板 / 电源等其它', measured: false, powerW: null, note: '本机无 IPMI，整机功耗无法直接测量' },
    ];
    const meas = comps.filter(c => c.measured && c.powerW !== null);
    const totalW = Math.round(meas.reduce((s, c) => s + c.powerW, 0) * 10) / 10;
    const offsetW = loadPowerConfig().offsetW;
    const adjustedTotalW = Math.round((totalW + offsetW) * 10) / 10;
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ time: Date.now(), totalW, offsetW, adjustedTotalW, measuredCount: meas.length, components: comps,
      cpu: { utilization: cpuUtilPct, cores: CPU_CORES, powerW: cpuPowerW, idleBaselineW: cpuIdleBaselineW !== null ? Math.round(cpuIdleBaselineW * 10) / 10 : null, activeW: activeCpuW !== null ? Math.round(activeCpuW * 10) / 10 : null },
      gpu: { powerW: gpu.powerW, limitW: gpu.limitW, utilization: gpu.utilization, memUsedMiB: gpu.memUsedMiB, name: gpu.name },
      processes: procs,
      procEnergyTotalWh: Math.round((Array.from(procEnergyJ.values()).reduce((s, v) => s + v, 0) / 3600) * 1000) / 1000,
    }));
    return;
  }
  if (pathname === '/v1/internal/power/offset' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); if (body.length > 1e4) req.destroy(); });
    req.on('end', () => {
      try {
        const data = JSON.parse(body || '{}');
        const offsetW = parseFloat(data.offsetW);
        if (isNaN(offsetW) || offsetW < 0 || offsetW > 1000) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: false, error: '补偿值需为 0-1000 之间的数字（瓦特）' }));
          return;
        }
        savePowerConfig({ offsetW: Math.round(offsetW * 10) / 10, gpuPlW: loadPowerConfig().gpuPlW });
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, offsetW: Math.round(offsetW * 10) / 10 }));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: e.message }));
      }
    });
    return;
  }

  // === Internal API: Energy (能耗/电费：单价设置 + 重置) ===
  if (pathname === '/v1/internal/energy' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(buildEnergyInfo()));
    return;
  }
  if (pathname === '/v1/internal/energy' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); if (body.length > 1e4) req.destroy(); });
    req.on('end', () => {
      try {
        const data = JSON.parse(body || '{}');
        const price = parseFloat(data.price_per_kwh);
        if (isNaN(price) || price < 0 || price > 100) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: false, error: '电价需为 0-100 之间的数字（元/kWh）' }));
          return;
        }
        const cfg = { price_per_kwh: price, unit: data.unit ? String(data.unit) : (loadEnergyConfig().unit || '元') };
        saveEnergyConfig(cfg);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, ...buildEnergyInfo() }));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: e.message }));
      }
    });
    return;
  }
  if (pathname === '/v1/internal/energy/reset' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); if (body.length > 1e4) req.destroy(); });
    req.on('end', () => {
      try {
        const data = JSON.parse(body || '{}');
        const scope = data.scope === 'history' ? 'history' : 'current';
        resetEnergyBucket(scope);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, scope, energy: buildEnergyInfo() }));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: e.message }));
      }
    });
    return;
  }

  // === Internal API: Billing days delete (手动删除每日费用明细) ===
  if (pathname === '/v1/internal/billing/days/delete' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); if (body.length > 1e4) req.destroy(); });
    req.on('end', () => {
      try {
        const data = JSON.parse(body || '{}');
        const st = loadBillingState();
        let deleted = null;
        if (data.all === true) {
          st.days = {};
          deleted = 'all';
        } else if (/^\d{4}-\d{2}-\d{2}$/.test(data.date || '')) {
          if (st.days[data.date]) { delete st.days[data.date]; deleted = data.date; }
        } else {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: false, error: '参数错误：需要 {date: "YYYY-MM-DD"} 或 {all: true}' }));
          return;
        }
        saveBillingState(st);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, deleted, days: buildBillingDays(st) }));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: e.message }));
      }
    });
    return;
  }

  // === Internal API: Billing config (手动设置计费单价) ===
  if (pathname === '/v1/internal/billing') {
    if (req.method === 'GET') {
      const cfg = loadBillingConfig();
      cfg.current_model = getCurrentVllmModel();
      // 合并服务器上实际存在的模型目录（/home/ll/models 下带 config.json 的目录）：
      // 保证计费弹窗能对每个模型设置单价，新加的模型目录也会自动出现。
      // 只合并到本次响应、不落盘 —— 未定价模型不会污染 billing-config.json；
      // 用户在弹窗里保存时，后端会按提交的模型逐个持久化。
      discoverModels((models) => {
        const cfgModels = cfg.models || {};
        for (const m of models) {
          if (cfgModels[m.name]) continue; // 已有配置（含手动保存过的）以配置为准
          cfgModels[m.name] = { name: m.name, enabled: true, auto: true, pricing: {} };
        }
        cfg.models = cfgModels;
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(cfg));
      });
      return;
    }
    if (req.method === 'POST') {
      let body = '';
      req.on('data', chunk => { body += chunk.toString(); if (body.length > 1e6) req.destroy(); });
      req.on('end', () => {
        try {
          const data = JSON.parse(body || '{}');
          const cfg = loadBillingConfig();
          if (data.currency !== undefined) cfg.currency = String(data.currency);
          if (data.unit !== undefined) cfg.unit = String(data.unit);
          let models = null;
          if (data.models && typeof data.models === 'object') {
            models = data.models;
          } else if (data.pricing && typeof data.pricing === 'object') {
            models = { [data.model || 'default']: { pricing: data.pricing } };
          }
          if (models) {
            cfg.models = cfg.models || {};
            for (const name of Object.keys(models)) {
              const m = models[name];
              if (!m || typeof m !== 'object') continue;
              const cur = cfg.models[name] || { name: name };
              if (m.name !== undefined) cur.name = String(m.name);
              if (m.enabled !== undefined) cur.enabled = !!m.enabled;
              cur.pricing = cur.pricing || {};
              const p = m.pricing && typeof m.pricing === 'object' ? m.pricing : m;
              for (const f of BILLING_PRICE_FIELDS) {
                const v = parseFloat(p[f]);
                if (!isNaN(v) && v >= 0) cur.pricing[f] = v;
              }
              cfg.models[name] = cur;
            }
          }
          saveBillingConfig(cfg);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: true, config: cfg }));
        } catch (e) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: false, error: e.message }));
        }
      });
      return;
    }
    res.writeHead(405, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ success: false, error: 'Method not allowed' }));
    return;
  }

  // === Internal API: Raw metrics ===
  if (pathname === '/v1/internal/metrics') {
    // [sglang-adapt-1003] ?port= 支持按实例抓 /metrics（SGLang 独立页 /sglang.html 需要非主端口实例；
    // 缺省主端口，端口自愈后仍跟随 config.vllmPort 而非启动时快照）
    const mpM = (req.url.split('?')[1] || '').match(/(?:^|&)port=(\d+)/);
    const mq = mpM ? parseInt(mpM[1], 10) : NaN;
    const mPort = (!isNaN(mq) && mq > 0) ? mq : config.vllmPort;
    const rawReq = http.get(`http://${config.vllmHost}:${mPort}/metrics`, (proxyRes) => {
      let data = '';
      proxyRes.on('data', chunk => data += chunk);
      proxyRes.on('error', () => {
        try {
          if (!res.headersSent) res.writeHead(502);
          res.end('Error fetching metrics');
        } catch (e) {}
      });
      proxyRes.on('end', () => {
        // [sglang-adapt-1003] 透传上游状态码：sglang 不带 --enable-metrics 时 /metrics 是
        // 404，旧版一律转 200 → 前端把「指标端点不存在」误当「指标全 0」。现在把 404/5xx 原样带出。
        res.writeHead(proxyRes.statusCode === 200 ? 200 : (proxyRes.statusCode || 502), { 'Content-Type': 'text/plain' });
        res.end(data);
      });
    }).on('error', () => {
      if (!res.headersSent) res.writeHead(502);
      res.end('Error fetching metrics');
    });
    rawReq.setTimeout(3000, () => rawReq.destroy(new Error('metrics timeout')));
    return;
  }

  // === Internal API: CPU KV 二级缓存详细数据（[kv-detail-1003] vLLM 标签页专属）===
  // ?port= 单实例；缺省全部 vLLM 实例。只读，无鉴权（GET 只读口径同 stats/metrics）。
  if (pathname === '/v1/internal/kv-detail') {
    const dpM = (req.url.split('?')[1] || '').match(/(?:^|&)port=(\d+)/);
    const dp = dpM ? parseInt(dpM[1], 10) : null;
    kvDetailInfo(dp).then((rows) => {
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
      res.end(JSON.stringify({ instances: rows }));
    }).catch(() => {
      if (!res.headersSent) res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end('{"error":"kv-detail failed"}');
    });
    return;
  }

  // === Internal API: Reset stats snapshot ===
  if (pathname === '/v1/internal/reset-stats' && req.method === 'POST') {
    const snapshotPath = path.join(__dirname, 'metrics-snapshot.json');
    const resetReq = http.get(`${vllmBaseUrl}/metrics`, (proxyRes) => {
      let data = '';
      proxyRes.on('data', chunk => data += chunk);
      proxyRes.on('error', () => {
        try {
          if (!res.headersSent) res.writeHead(502, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ status: 'error', message: 'Cannot reach vLLM metrics' }));
        } catch (e) {}
      });
      proxyRes.on('end', () => {
        try {
          const m = parseMetrics(data);
          const lbl = detectModelLabels(m);
          const reset = captureResetBlock(m, lbl.modelLabel, lbl.sourceLabel);
          const snap = {
            offsetTokens: Math.round(m['vllm:generation_tokens_total' + lbl.modelLabel] || 0),
            offsetInput: Math.round(m['vllm:prompt_tokens_total' + lbl.modelLabel] || 0),
            offsetCached: Math.round(m['vllm:prompt_tokens_by_source_total' + lbl.sourceLabel + '"source":"local_cache_hit"}'] || 0),
            offsetUncached: Math.round(m['vllm:prompt_tokens_by_source_total' + lbl.sourceLabel + '"source":"local_compute"}'] || 0),
            offsetReqCount: Math.round(m['vllm:request_generation_tokens_count' + lbl.modelLabel] || 0),
            lastRunning: 0,
            perReqTokens: {},
            time: Date.now(),
            reset: reset,
          };
          fs.writeFileSync(snapshotPath, JSON.stringify(snap));
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ status: 'ok', message: 'Stats reset', baselineKeys: Object.keys(reset).length }));
        } catch (e) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ status: 'error', message: e.message }));
        }
      });
    }).on('error', () => {
      if (!res.headersSent) res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'error', message: 'Cannot reach vLLM metrics' }));
    });
    resetReq.setTimeout(3000, () => resetReq.destroy(new Error('reset-stats timeout')));
    return;
  }

  // === Management API: Server status ===
  if (pathname === '/admin/api/server-status') {
    const statusReq = http.get(`${vllmBaseUrl}/health`, (proxyRes) => {
      proxyRes.on('error', () => {
        try {
          if (!res.headersSent) res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ status: 'disconnected', vllm_url: vllmBaseUrl }));
        } catch (e) {}
      });
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        status: proxyRes.statusCode === 200 ? 'connected' : 'disconnected',
        vllm_url: vllmBaseUrl,
      }));
    }).on('error', () => {
      if (!res.headersSent) res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'disconnected', vllm_url: vllmBaseUrl }));
    });
    statusReq.setTimeout(3000, () => statusReq.destroy(new Error('status timeout')));
    return;
  }

  // === Internal API: CPU 详情（静态规格 + 各核实时利用率/频率/温度/负载/近 3 分钟走势）===
  if (pathname === '/v1/internal/cpu') {
    let st = null;
    try { st = getCpuStatic(); } catch (e) { st = { supported: false, error: String(e && e.message || e) }; }
    const live = global.__cpuLive || {};
    const hist = (global.__cpuHistory || []).slice(-180); // 近 3 分钟
    res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
    res.end(JSON.stringify(Object.assign({}, st, { live, history: hist })));
    return;
  }


  // === Internal API: GPU 详情（硬件监视页 GPU 卡：PCIe 链路 x16 等 / 带宽占用 / 显存 / 温度 / 功耗 / 频率）===
  if (pathname === '/v1/internal/gpu') {
    let st = null;
    try { st = getGpuStatic(); } catch (e) { st = { supported: false, error: String(e && e.message || e) }; }
    const liveArr = (global.__gpuLive && global.__gpuLive.gpus) || [];
    const liveMap = {};
    for (const g of liveArr) liveMap[g.index] = g;
    const pcieBy = (global.__gpuPcie && global.__gpuPcie.byIndex) || {};
    const gpus = (st.gpus || []).map((s) => {
      const g = Object.assign({}, s, liveMap[s.index] || {});
      const p = pcieBy[s.index];
      if (p) {
        g.rx_mbs = p.rx_mbs; g.tx_mbs = p.tx_mbs;
        // 理论带宽 = 每通道有效速率（按当前代数）× 当前宽度（MB/s）：Gen1=250、Gen2=500、Gen3≈985、Gen4≈1969、Gen5≈3938
        const perLane = { 1: 250, 2: 500, 3: 985, 4: 1969, 5: 3938 }[g.gen_current] || null;
        g.pcie_theo_mbs = (perLane != null && g.width_current) ? perLane * g.width_current : null;
        g.rx_pct = (g.pcie_theo_mbs && p.rx_mbs != null) ? Math.min(100, Math.round(p.rx_mbs / g.pcie_theo_mbs * 1000) / 10) : null;
        g.tx_pct = (g.pcie_theo_mbs && p.tx_mbs != null) ? Math.min(100, Math.round(p.tx_mbs / g.pcie_theo_mbs * 1000) / 10) : null;
      }
      return g;
    });
    res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
    res.end(JSON.stringify(Object.assign({}, st, {
      gpus,
      live_ts: (global.__gpuLive && global.__gpuLive.ts) || null,
      pcie_ts: (global.__gpuPcie && global.__gpuPcie.ts) || null,
      history: (global.__gpuPcieHistory || []).slice(-180), // ~1.2s/点 ≈ 3.6 分钟窗口，前端截近 3 分钟（150 点）
    })));
    return;
  }

  // === Internal API: GPU 控制（[gpu-ctl 1003] 硬件监视页：功耗上限 / SM 频率锁）===
  // GET  /v1/internal/gpu-ctl          状态（gpu-ctl status --json，1.5s TTL 缓存+单飞）
  // POST /v1/internal/gpu-ctl/cmd      {action:'pl'|'lock'|'unlock', watt?, lo?, hi?, gpu?}
  //   白名单 + 正则校验（同 cpuctl 铁律）；nvidia-smi 需 root → 走 /usr/local/bin/gpu-ctl
  //   （root:root 0755 + sudoers.d/ll-gpu-ctl 单命令白名单，ops/install-gpu-ctl-127.sh 安装）。
  //   pl 成功后同步改写 gpu-power-limit.service 的 drop-in（09-26 铁律：不改 PL 重启必被打回）。
  if (pathname === '/v1/internal/gpu-ctl' || pathname.startsWith('/v1/internal/gpu-ctl/')) {
    return gpuCtlHandle(req, res, urlObj).catch((e) => {
      try { res.writeHead(500, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ ok: false, error: String((e && e.message) || e) })); } catch (_) {}
    });
  }

  // === GPU 高温自动关机守护 API（[gpu-hotshield 1009]：GET 状态 / POST cmd {action:save|cancel}）===
  if (pathname === '/v1/internal/hotshield' || pathname.startsWith('/v1/internal/hotshield/')) {
    return Promise.resolve(hotShieldHandle(req, res, urlObj)).catch((e) => {
      try { res.writeHead(500, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ ok: false, error: String((e && e.message) || e) })); } catch (_) {}
    });
  }

  // === Internal API: 内存带宽（硬件监视页 RAM 带宽卡；perf uncore IMC CAS 计数，1s 采样）===
  if (pathname === '/v1/internal/membw') {
    const live = global.__membw || null;
    const ageMs = live ? Date.now() - live.ts : null;
    const fresh = !!(live && ageMs != null && ageMs < 8000);
    const gb = (x) => Math.round(x / 1073741824 * 100) / 100;
    res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
    res.end(JSON.stringify({
      supported: fresh,
      stale: !!live && !fresh,
      proc_alive: !!global.__membwProc,
      age_ms: ageMs,
      peak_gbs_fallback: MEMBW_PEAK_FALLBACK,
      peak_note: '4×DDR4-2400 理论峰值（BDW-EP 四通道）；前端优先用 dmidecode 理论带宽',
      live: live ? {
        ts: live.ts,
        read_gbs: gb(live.read_bps), write_gbs: gb(live.write_bps),
        total_gbs: gb(live.read_bps + live.write_bps),
      } : null,
      history: (global.__membwHistory || []).slice(-180), // ~1s/点 ≈ 3 分钟窗口
    }));
    return;
  }

  // === Internal API: PCIe 通道拓扑/占用（硬件监视页 PCIe 卡，60s 缓存）===
  if (pathname === '/v1/internal/pcie-topo') {
    let d = null;
    try { d = getPcieTopo(); } catch (e) { d = { supported: false, groups: [], error: String((e && e.message) || e) }; }
    res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
    res.end(JSON.stringify(Object.assign({ ts: Date.now() }, d)));
    return;
  }

  // === Internal API: 网络速度（/proc/net/dev 每秒采样，窗口增量算速率）===
  // 采样器必须在 handler 之前启动（handler 末尾会 return，放在后面永不执行）
  if (!global.__netSampler) {
    global.__netSampler = setInterval(sampleNetDev, 1000);
    if (global.__netSampler.unref) global.__netSampler.unref();
    sampleNetDev();
  }
  if (pathname === '/v1/internal/net') {
    let snap = { ts: new Date().toISOString(), interfaces: [], totals: {} };
    try {
      const net = buildNetSnapshot();
      snap.interfaces = net.interfaces;
      snap.totals = net.totals;
    } catch (e) {
      snap.error = String((e && e.message) || e);
    }
    res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
    res.end(JSON.stringify(snap));
    return;
  }

  // === Internal API: Storage（容量 / SMART 健康 / 实时读写速率 / 累计读写量）===
  // 速率采样器必须在 handler 之前启动（handler 末尾会 return，放在后面永不执行）
  if (!global.__diskSampler) {
    global.__diskSampler = setInterval(sampleDiskStats, 1000);
    if (global.__diskSampler.unref) global.__diskSampler.unref();
    sampleDiskStats();
  }
  if (pathname === '/v1/internal/storage') {
    let snap;
    try {
      snap = buildStorageSnapshot();
    } catch (e) {
      snap = { ts: new Date().toISOString(), error: String(e && e.message || e), disks: [], partitions: [], ioStats: {}, totals: {} };
    }
    res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
    res.end(JSON.stringify(snap));
    return;
  }

  // === Internal API: 磁盘读写速度走势（纯内存历史，不调 lsblk/smartctl，可 2 秒轮询）===
  if (pathname === '/v1/internal/disk-trend') {
    let win = 180;
    try { win = parseInt(urlObj.searchParams.get('win'), 10) || 180; } catch (e) {}
    win = Math.min(1800, Math.max(30, win));
    let body;
    try {
      const s = diskTrendSeries(win);
      const labels = global.__diskLabels || [];
      const names = labels.map((x) => x.name);
      const live = {};
      for (const n of names) {
        const r = diskRateFor(n);
        if (r) live[n] = { read_bps: r.read_bps, write_bps: r.write_bps, window_s: r.window_s };
      }
      body = {
        ts: new Date().toISOString(), win_s: win,
        sample_s: s.sample_s, raw_points: s.raw_points,
        points: s.points, devices: labels, live,
      };
    } catch (e) {
      body = { ts: new Date().toISOString(), win_s: win, points: [], devices: [], live: {}, error: String(e && e.message || e) };
    }
    res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
    res.end(JSON.stringify(body));
    return;
  }

  // ---------- 存储采集辅助函数 ----------
  function fmtBytes(v) {
    const n = Number(v);
    if (!isFinite(n)) return '--';
    const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
    let i = 0, x = Math.abs(n);
    while (x >= 1024 && i < units.length - 1) { x /= 1024; i++; }
    const s = i === 0 ? String(Math.round(x)) : (x >= 100 ? x.toFixed(0) : (x >= 10 ? x.toFixed(1) : x.toFixed(2)));
    return (n < 0 ? '-' : '') + s + ' ' + units[i];
  }
  function fmtRate(v) { return v == null ? '--' : fmtBytes(v) + '/s'; }

  // 设备类型：loop/snap、软 RAID/映射、整盘、分区
  function diskDevKind(name) {
    if (/^(loop|ram|zram)\d+$/.test(name)) return 'loop';
    if (/^(dm-|md)\d+$/.test(name)) return 'virtual';
    if (/^nvme\d+n\d+$/.test(name)) return 'disk';
    if (/^nvme\d+n\d+p\d+$/.test(name)) return 'part';
    if (/^(sd|vd|hd)[a-z]+$/.test(name)) return 'disk';
    if (/^(sd|vd|hd)[a-z]+\d+$/.test(name)) return 'part';
    return 'other';
  }

  // /proc/diskstats 全字段（顺序见内核 Documentation/admin-guide/iostats.rst）
  function readDiskStatsRaw() {
    const out = {};
    try {
      for (const line of fs.readFileSync('/proc/diskstats', 'utf8').split('\n')) {
        const p = line.trim().split(/\s+/);
        if (p.length < 14) continue;
        const n = (i) => parseInt(p[i], 10) || 0;
        out[p[2]] = {
          reads: n(3), reads_merged: n(4), read_sectors: n(5), read_ms: n(6),
          writes: n(7), writes_merged: n(8), write_sectors: n(9), write_ms: n(10),
          in_flight: n(11), io_ms: n(12), weighted_io_ms: n(13),
          discards: n(14), discard_sectors: n(16),
          flushes: n(18), flush_ms: n(19),
        };
      }
    } catch (e) {}
    return out;
  }

  // 每秒采样一次、保留最近 32 个样本；速率取「≥4 秒前的最新样本」为基线，
  // 与前端 5s 轮询对齐，多个页面同时打开也能得到稳定读数。
  // ---------- 网络速度采集辅助函数 ----------
  // /proc/net/dev 全字段（每行：iface: rx_bytes rx_packets rx_errs rx_drop rx_fifo rx_frame rx_compressed rx_multicast tx_bytes ...）
  function readNetDevRaw() {
    const out = {};
    try {
      const lines = fs.readFileSync('/proc/net/dev', 'utf8').split('\n');
      for (const line of lines) {
        const idx = line.indexOf(':');
        if (idx < 0) continue;
        const name = line.slice(0, idx).trim();
        const p = line.slice(idx + 1).trim().split(/\s+/).map((x) => parseInt(x, 10) || 0);
        if (p.length < 16) continue;
        out[name] = {
          rx_bytes: p[0], rx_packets: p[1], rx_errs: p[2], rx_drop: p[3],
          tx_bytes: p[8], tx_packets: p[9], tx_errs: p[10], tx_drop: p[11],
        };
      }
    } catch (e) {}
    return out;
  }
  function sampleNetDev() {
    if (!global.__netSamples) global.__netSamples = [];
    const arr = global.__netSamples;
    const now = Date.now();
    if (arr.length && now - arr[arr.length - 1].ts < 900) return;
    arr.push({ ts: now, dev: readNetDevRaw() });
    while (arr.length > 32) arr.shift();
  }
  // 窗口增量速率：找与当前样本间隔 >=4s 的最近基线（与磁盘口径一致）
  function netRateFor(name) {
    const arr = global.__netSamples || [];
    if (arr.length < 2) return null;
    const cur = arr[arr.length - 1];
    let base = null;
    for (let i = arr.length - 1; i >= 0; i--) {
      if (cur.ts - arr[i].ts >= 4000) { base = arr[i]; break; }
    }
    if (!base) base = arr[0];
    const a = base.dev[name], b = cur.dev[name];
    if (!a || !b) return null;
    const dt = (cur.ts - base.ts) / 1000;
    if (dt <= 0.5) return null;
    const d = (x, y) => Math.max(0, y - x);
    return {
      window_s: Math.round(dt * 10) / 10,
      rx_bps: Math.round(d(a.rx_bytes, b.rx_bytes) / dt),
      tx_bps: Math.round(d(a.tx_bytes, b.tx_bytes) / dt),
      rx_pps: Math.round(d(a.rx_packets, b.rx_packets) / dt),
      tx_pps: Math.round(d(a.tx_packets, b.tx_packets) / dt),
      rx_errs: d(a.rx_errs, b.rx_errs),
      rx_drop: d(a.rx_drop, b.rx_drop),
      tx_errs: d(a.tx_errs, b.tx_errs),
      tx_drop: d(a.tx_drop, b.tx_drop),
    };
  }
  // 网卡速率快照（供 /v1/internal/net 与 /v1/internal/storage.network 复用）
  function buildNetSnapshot() {
    const arr = global.__netSamples || [];
    const cur = arr.length ? arr[arr.length - 1] : null;
    const dev = cur ? cur.dev : readNetDevRaw();
    let uptimeS = null;
    try { uptimeS = Math.round(parseFloat(fs.readFileSync('/proc/uptime', 'utf8').split(' ')[0])); } catch (e) {}
    const interfaces = [];
    let tRx = 0, tTx = 0, tRxCum = 0, tTxCum = 0, tRxPkt = 0, tTxPkt = 0;
    let tRxErr = 0, tTxDrop = 0;
    let windowS = null;
    for (const [name, s] of Object.entries(dev)) {
      if (name === 'lo') continue;
      const rate = netRateFor(name);
      if (rate) windowS = windowS == null ? rate.window_s : Math.min(windowS, rate.window_s);
      const rx_bps = rate ? rate.rx_bps : null;
      const tx_bps = rate ? rate.tx_bps : null;
      interfaces.push({
        name,
        // 实时速率（约 5 秒窗口均值）
        rx_bps, tx_bps,
        rx_pps: rate ? rate.rx_pps : null,
        tx_pps: rate ? rate.tx_pps : null,
        rx_rate_h: rate ? fmtRate(rate.rx_bps) : '--',
        tx_rate_h: rate ? fmtRate(rate.tx_bps) : '--',
        window_s: rate ? rate.window_s : null,
        // 本次开机累计（/proc/net/dev，重启归零）
        rx_bytes: s.rx_bytes, tx_bytes: s.tx_bytes,
        rx_packets: s.rx_packets, tx_packets: s.tx_packets,
        rx_cum_h: fmtBytes(s.rx_bytes), tx_cum_h: fmtBytes(s.tx_bytes),
        // 窗口内错误/丢包增量
        rx_errs_delta: rate ? rate.rx_errs : null,
        rx_drop_delta: rate ? rate.rx_drop : null,
        tx_errs_delta: rate ? rate.tx_errs : null,
        tx_drop_delta: rate ? rate.tx_drop : null,
      });
      tRx += rx_bps || 0;
      tTx += tx_bps || 0;
      tRxCum += s.rx_bytes;
      tTxCum += s.tx_bytes;
      tRxPkt += s.rx_packets;
      tTxPkt += s.tx_packets;
      if (rate) { tRxErr += rate.rx_errs + rate.rx_drop; tTxDrop += rate.tx_errs + rate.tx_drop; }
    }
    interfaces.sort((a, b) => (b.rx_bps || 0) + (b.tx_bps || 0) - ((a.rx_bps || 0) + (a.tx_bps || 0)));
    return {
      ts: new Date().toISOString(),
      interfaces,
      totals: {
        rx_bps: tRx, tx_bps: tTx,
        rx_rate_h: fmtRate(tRx), tx_rate_h: fmtRate(tTx),
        rx_bytes: tRxCum, tx_bytes: tTxCum,
        rx_cum_h: fmtBytes(tRxCum), tx_cum_h: fmtBytes(tTxCum),
        rx_packets: tRxPkt, tx_packets: tTxPkt,
        window_s: windowS,
        uptime_s: uptimeS,
        iface_count: interfaces.length,
        errs_drops_window: tRxErr + tTxDrop,
      },
    };
  }

  function sampleDiskStats() {
    if (!global.__diskSamples) global.__diskSamples = [];
    const arr = global.__diskSamples;
    const now = Date.now();
    if (arr.length && now - arr[arr.length - 1].ts < 900) return;
    arr.push({ ts: now, dev: readDiskStatsRaw() });
    while (arr.length > 32) arr.shift();
    // 逐秒速率历史（磁盘读写速度走势，接口 /v1/internal/disk-trend）
    try { appendDiskHistory(arr[arr.length - 2], arr[arr.length - 1]); } catch (e) {}
  }
  function diskRateFor(name) {
    const arr = global.__diskSamples || [];
    if (arr.length < 2) return null;
    const cur = arr[arr.length - 1];
    let base = null;
    for (let i = arr.length - 1; i >= 0; i--) {
      if (cur.ts - arr[i].ts >= 4000) { base = arr[i]; break; }
    }
    if (!base) base = arr[0];
    const a = base.dev[name], b = cur.dev[name];
    if (!a || !b) return null;
    const dt = (cur.ts - base.ts) / 1000;
    if (dt <= 0.5) return null;
    const d = (x, y) => Math.max(0, y - x);
    const dReads = d(a.reads, b.reads), dWrites = d(a.writes, b.writes);
    const dReadBytes = d(a.read_sectors, b.read_sectors) * 512;
    const dWriteBytes = d(a.write_sectors, b.write_sectors) * 512;
    const dIoMs = d(a.io_ms, b.io_ms), dWIoMs = d(a.weighted_io_ms, b.weighted_io_ms);
    const r3 = (v) => Math.round(v * 1000) / 1000;
    return {
      window_s: Math.round(dt * 10) / 10,
      read_bps: Math.round(dReadBytes / dt),
      write_bps: Math.round(dWriteBytes / dt),
      read_iops: r3(dReads / dt),
      write_iops: r3(dWrites / dt),
      busy_pct: Math.min(100, Math.round((dIoMs / (dt * 1000)) * 1000) / 10),
      read_latency_ms: dReads > 0 ? r3(d(a.read_ms, b.read_ms) / dReads) : null,
      write_latency_ms: dWrites > 0 ? r3(d(a.write_ms, b.write_ms) / dWrites) : null,
      avg_queue: dIoMs > 0 ? r3(dWIoMs / dIoMs) : 0,
    };
  }

  // ---- 磁盘读写速度历史（走势曲线，接口 /v1/internal/disk-trend）----
  // 注意：本段与 sampleDiskStats 同在 request handler 作用域内，且 setInterval 捕获的是
  // 首个请求的那份闭包——若那个请求在下方某条路由提前 return，handler 作用域里的
  // const/let 会永远停在 TDZ（ReferenceError）。故这里只用函数声明 + 函数内字面量，
  // 不在 handler 作用域放 const。
  function appendDiskHistory(prev, cur) {
    if (!prev || !cur || !prev.dev || !cur.dev) return;
    const devRe = /^(nvme\d+n\d+|sd[a-z]+|vd[a-z]+|hd[a-z]+)$/; // 只算物理整盘：分区/dm/loop 与整盘重复计数
    const KEEP = 1800;                                            // 1 秒/点 → 保留 30 分钟
    const dt = (cur.ts - prev.ts) / 1000;
    if (!(dt > 0.2 && dt < 5)) return; // 间隔异常（事件循环卡顿、计数器重置）不出点，避免假尖峰
    const per = {};
    let tr = 0, tw = 0;
    for (const name of Object.keys(cur.dev)) {
      if (!devRe.test(name)) continue;
      const a = prev.dev[name], b = cur.dev[name];
      if (!a || !b) continue;
      const r = Math.max(0, b.read_sectors - a.read_sectors) * 512 / dt;
      const w = Math.max(0, b.write_sectors - a.write_sectors) * 512 / dt;
      per[name] = [Math.round(r), Math.round(w)];
      tr += r; tw += w;
    }
    if (!Object.keys(per).length) return;
    const H = global.__diskHistory || (global.__diskHistory = []);
    H.push({ t: Math.round(cur.ts / 1000), r: Math.round(tr), w: Math.round(tw), d: per });
    if (H.length > KEEP) H.splice(0, H.length - KEEP);
  }
  // 取窗口内走势并按桶抽稀到 ≤300 点（桶内均值）：30 分钟窗口也只有 300 点，响应不膨胀
  function diskTrendSeries(winS) {
    const H = global.__diskHistory || [];
    const from = Math.round(Date.now() / 1000) - winS;
    let i = 0;
    while (i < H.length && H[i].t < from) i++;
    const pts = H.slice(i);
    const bucket = Math.max(1, Math.ceil(pts.length / 300));
    const out = [];
    for (let k = 0; k < pts.length; k += bucket) {
      const seg = pts.slice(k, k + bucket);
      if (!seg.length) continue;
      const acc = { t: seg[seg.length - 1].t, r: 0, w: 0, d: {} };
      for (const p of seg) {
        acc.r += p.r; acc.w += p.w;
        for (const n of Object.keys(p.d)) {
          const a = acc.d[n] || (acc.d[n] = [0, 0]);
          a[0] += p.d[n][0]; a[1] += p.d[n][1];
        }
      }
      acc.r = Math.round(acc.r / seg.length);
      acc.w = Math.round(acc.w / seg.length);
      for (const n of Object.keys(acc.d)) {
        acc.d[n] = [Math.round(acc.d[n][0] / seg.length), Math.round(acc.d[n][1] / seg.length)];
      }
      out.push(acc);
    }
    return { points: out, sample_s: bucket, raw_points: pts.length };
  }

  // lsblk --json：型号/序列号/固件/容量等（比按空格切列可靠得多）
  function readLsblkJson() {
    try {
      const raw = require('child_process').execSync(
        'lsblk -J -b -o NAME,MODEL,SERIAL,SIZE,TYPE,ROTA,TRAN,REV,FSTYPE,MOUNTPOINT,PHY-SEC,LOG-SEC 2>/dev/null || true',
        { encoding: 'utf8', timeout: 5000, maxBuffer: 8 * 1024 * 1024 }
      );
      const j = JSON.parse(raw);
      return Array.isArray(j.blockdevices) ? j.blockdevices : [];
    } catch (e) { return []; }
  }

  // SMART（smartctl -j -a）：NVMe 与 ATA 通吃；30 秒缓存避免 5 秒轮询频繁起进程
  function readSmartInfo(devPath) {
    if (!global.__smartCache) global.__smartCache = new Map();
    const cached = global.__smartCache.get(devPath);
    if (cached && Date.now() - cached.ts < 30000) return cached.data;
    let data = null;
    try {
      const raw = require('child_process').execSync(
        `sudo -n smartctl -j -a -n standby ${devPath} 2>/dev/null || true`,
        { encoding: 'utf8', timeout: 8000, maxBuffer: 4 * 1024 * 1024 }
      );
      const j = JSON.parse(raw);
      if (!j || !j.smart_status) {
        const msgs = (j && j.smartctl && Array.isArray(j.smartctl.messages)) ? j.smartctl.messages : [];
        const first = msgs.find((m) => m && m.string);
        data = { available: false, reason: first ? first.string : (j && j.device && j.device.type ? 'SMART 不可用' : '无 SMART 数据') };
      } else {
        data = {
          available: true,
          health: j.smart_status.passed === true ? 'PASS' : (j.smart_status.passed === false ? 'FAIL' : 'UNKNOWN'),
          model: j.model_name || '', serial: j.serial_number || '', firmware: j.firmware_version || '',
          protocol: (j.device && j.device.protocol) || '',
          capacity_bytes: (j.user_capacity && j.user_capacity.bytes) || null,
          temperature: (j.temperature && j.temperature.current != null) ? j.temperature.current : null,
          power_on_hours: (j.power_on_time && j.power_on_time.hours != null) ? j.power_on_time.hours : null,
          power_cycles: j.power_cycle_count != null ? j.power_cycle_count : null,
        };
        const n = j.nvme_smart_health_information_log;
        if (n) {
          data.kind = 'nvme';
          data.critical_warning = n.critical_warning;
          data.available_spare = n.available_spare;
          data.available_spare_threshold = n.available_spare_threshold;
          data.write_life_pct = n.percentage_used != null ? n.percentage_used : null;
          data.data_units_read = n.data_units_read != null ? n.data_units_read : null;
          data.data_units_written = n.data_units_written != null ? n.data_units_written : null;
          // NVMe 规范：1 data unit = 1000 × 512 B
          data.bytes_read = n.data_units_read != null ? n.data_units_read * 512000 : null;
          data.bytes_written = n.data_units_written != null ? n.data_units_written * 512000 : null;
          data.host_reads = n.host_reads != null ? n.host_reads : null;
          data.host_writes = n.host_writes != null ? n.host_writes : null;
          data.controller_busy_min = n.controller_busy_time != null ? n.controller_busy_time : null;
          data.unsafe_shutdowns = n.unsafe_shutdowns != null ? n.unsafe_shutdowns : null;
          data.media_errors = n.media_errors != null ? n.media_errors : null;
          data.error_log_entries = n.num_err_log_entries != null ? n.num_err_log_entries : null;
          data.temp_sensors = Array.isArray(n.temperature_sensors) ? n.temperature_sensors : [];
          if (data.power_on_hours == null && n.power_on_hours != null) data.power_on_hours = n.power_on_hours;
          if (data.power_cycles == null && n.power_cycles != null) data.power_cycles = n.power_cycles;
        } else if (j.ata_smart_attributes && Array.isArray(j.ata_smart_attributes.table)) {
          data.kind = 'ata';
          const byId = {}, byName = {};
          for (const a of j.ata_smart_attributes.table) {
            byId[a.id] = a;
            if (a.name) byName[String(a.name).toLowerCase()] = a;
          }
          const rawOf = (id, nm) => {
            const a = byId[id] || (nm ? byName[nm] : null);
            return (a && a.raw && a.raw.value != null) ? a.raw.value : null;
          };
          const lbasWritten = rawOf(241, 'total_lbas_written');
          const lbasRead = rawOf(242, 'total_lbas_read');
          data.bytes_written = lbasWritten != null ? lbasWritten * 512 : null;
          data.bytes_read = lbasRead != null ? lbasRead * 512 : null;
          const wear = byId[177] || byName['wear_leveling_count'];
          data.write_life_pct = wear && wear.value != null ? Math.max(0, 100 - wear.value) : null;
          data.media_errors = rawOf(5, 'reallocated_sector_ct');
          data.pending_sectors = rawOf(197, 'current_pending_sector');
          data.unsafe_shutdowns = rawOf(192, 'unsafe_shutdown_count') != null ? rawOf(192, 'unsafe_shutdown_count') : rawOf(174, 'unexpected_power_loss');
          if (data.temperature == null) {
            const t = rawOf(194, 'temperature_celsius') != null ? rawOf(194, 'temperature_celsius') : rawOf(190, 'airflow_temperature_cel');
            data.temperature = t;
          }
          if (data.power_on_hours == null) {
            data.power_on_hours = rawOf(9, 'power_on_hours') != null ? rawOf(9, 'power_on_hours') : rawOf(240, 'head_flying_hours');
          }
        } else {
          data.kind = 'other';
        }
      }
    } catch (e) {
      data = { available: false, reason: 'smartctl 调用失败：' + String(e && e.message || e) };
    }
    global.__smartCache.set(devPath, { ts: Date.now(), data });
    return data;
  }

  // ---------- 内存（RAM）：实时用量 + 硬件规格（容量/频率/型号/插槽）----------
  // dmidecode 读 SMBIOS（需 root，已配 NOPASSWD）；硬件信息静态，缓存 60 秒
  function dmiSizeToBytes(txt) {
    if (!txt || /No Module Installed/i.test(txt)) return 0;
    const m = /([\d.]+)\s*(GB|MB|TB)/i.exec(txt);
    if (!m) return 0;
    const n = parseFloat(m[1]) || 0;
    const unit = m[2].toUpperCase();
    return Math.round(n * (unit === 'TB' ? 1099511627776 : unit === 'GB' ? 1073741824 : 1048576));
  }

  function readMemoryInfo() {
    const mem = {
      available: false, modules: [], slots_total: 0, populated: 0,
      total_bytes: 0, total_h: '--', speed_mts: null, type: null,
      ecc: null, max_capacity_bytes: null, max_capacity_h: null,
      channels: null, channel_text: null, bandwidth_gbs: null,
      form_factor: null, rank: null, live: null, dmi_error: null,
    };

    // 1) 实时用量（/proc/meminfo）
    try {
      const kv = {};
      for (const line of fs.readFileSync('/proc/meminfo', 'utf8').split('\n')) {
        const m = /^(\w+):\s+(\d+)\s*kB/.exec(line);
        if (m) kv[m[1]] = parseInt(m[2], 10) * 1024;
      }
      if (kv.MemTotal) {
        const total = kv.MemTotal;
        const avail = kv.MemAvailable != null ? kv.MemAvailable : (kv.MemFree || 0);
        const used = Math.max(0, total - avail);
        const swapTotal = kv.SwapTotal || 0, swapFree = kv.SwapFree || 0;
        mem.live = {
          total_bytes: total,
          used_bytes: used,
          avail_bytes: avail,
          free_bytes: kv.MemFree || 0,
          buffers_bytes: kv.Buffers || 0,
          cached_bytes: kv.Cached || 0,
          shared_bytes: kv.Shmem || 0,
          dirty_bytes: kv.Dirty || 0,
          slab_bytes: kv.Slab || 0,
          swap_total_bytes: swapTotal,
          swap_used_bytes: Math.max(0, swapTotal - swapFree),
          swap_use_pct: swapTotal > 0 ? Math.round((swapTotal - swapFree) / swapTotal * 100) : 0,
          use_pct: Math.round(used / total * 100),
          total_h: fmtBytes(total),
          used_h: fmtBytes(used),
          avail_h: fmtBytes(avail),
          cached_h: fmtBytes((kv.Buffers || 0) + (kv.Cached || 0)),
          swap_total_h: fmtBytes(swapTotal),
          swap_used_h: fmtBytes(Math.max(0, swapTotal - swapFree)),
        };
      }
    } catch (e) {}

    // 2) 硬件规格（dmidecode -t memory，60 秒缓存）
    if (!global.__memHwCache) global.__memHwCache = { ts: 0, data: null };
    let hw = null;
    if (global.__memHwCache.data && Date.now() - global.__memHwCache.ts < 60000) {
      hw = global.__memHwCache.data;
    } else {
      let raw = '';
      try {
        raw = require('child_process').execSync(
          'sudo -n dmidecode -t memory 2>/dev/null || true',
          { encoding: 'utf8', timeout: 8000, maxBuffer: 4 * 1024 * 1024 }
        );
      } catch (e) { raw = ''; }
      hw = { modules: [], slots_total: 0, ecc: null, max_capacity_bytes: null, dmi_error: null };
      if (!raw || !/DMI type 17/.test(raw)) {
        hw.dmi_error = '无法读取 SMBIOS（需要 sudo dmidecode 免密权限）';
      } else {
        let type = null, cur = null;
        const dmi = [];
        const flush = () => { if (cur && type === '17') dmi.push(cur); cur = null; };
        for (const line of raw.split('\n')) {
          const hm = /^Handle 0x[0-9A-Fa-f]+, DMI type (\d+)/.exec(line);
          if (hm) { flush(); type = hm[1]; if (type === '17') cur = {}; continue; }
          const fm = /^\t([^:]+):\s*(.*)$/.exec(line);
          if (!fm) continue;
          const k = fm[1].trim(), v = fm[2].trim();
          if (type === '16') {
            if (k === 'Error Correction Type') hw.ecc = /No Error|^None$/i.test(v) ? 'none' : v;
            if (k === 'Maximum Capacity') hw.max_capacity_bytes = dmiSizeToBytes(v);
            if (k === 'Number Of Devices') hw.slots_total = parseInt(v, 10) || 0;
          } else if (type === '17' && cur) {
            cur[k] = v;
          }
        }
        flush();

        const norm = (s) => (!s || /Not Specified|Unknown|^None$/i.test(s)) ? null : String(s).trim();
        const speedNum = (s) => { const m = /([\d.]+)\s*(MT\/s|MHz)/i.exec(s || ''); return m ? parseFloat(m[1]) : null; };
        hw.modules = dmi.map((m) => {
          const sizeTxt = m['Size'] || '';
          const installed = !/No Module Installed/i.test(sizeTxt);
          const sz = installed ? dmiSizeToBytes(sizeTxt) : 0;
          return {
            locator: m['Locator'] || '',
            bank: m['Bank Locator'] || '',
            installed,
            size_bytes: sz,
            size_h: installed ? fmtBytes(sz) : '空槽',
            type: norm(m['Type']),
            form_factor: norm(m['Form Factor']),
            speed_mts: speedNum(m['Speed']),
            configured_mts: speedNum(m['Configured Memory Speed']),
            voltage_configured: norm(m['Configured Voltage']),
            voltage_min: norm(m['Minimum Voltage']),
            voltage_max: norm(m['Maximum Voltage']),
            manufacturer: norm(m['Manufacturer']),
            part_number: norm(m['Part Number']),
            serial: norm(m['Serial Number']),
            rank: norm(m['Rank']),
            total_width: norm(m['Total Width']),
            data_width: norm(m['Data Width']),
            technology: norm(m['Memory Technology']),
          };
        });
      }
      global.__memHwCache = { ts: Date.now(), data: hw };
    }

    mem.modules = hw.modules || [];
    mem.slots_total = hw.slots_total || mem.modules.length;
    mem.ecc = hw.ecc;
    mem.dmi_error = hw.dmi_error;
    if (hw.max_capacity_bytes) {
      mem.max_capacity_bytes = hw.max_capacity_bytes;
      mem.max_capacity_h = fmtBytes(hw.max_capacity_bytes);
    }

    const inst = mem.modules.filter((m) => m.installed);
    mem.populated = inst.length;
    mem.available = inst.length > 0;
    mem.total_bytes = inst.reduce((a, m) => a + (m.size_bytes || 0), 0);
    if (mem.total_bytes > 0) mem.total_h = fmtBytes(mem.total_bytes);

    const speeds = inst.map((m) => m.configured_mts || m.speed_mts).filter((v) => v);
    if (speeds.length) {
      mem.speed_mts = Math.max.apply(null, speeds);
      mem.speed_min_mts = Math.min.apply(null, speeds);
      mem.mixed_speed = mem.speed_mts !== mem.speed_min_mts;
    }
    const types = inst.map((m) => m.type).filter(Boolean);
    if (types.length) mem.type = types[0];
    const ffs = inst.map((m) => m.form_factor).filter(Boolean);
    if (ffs.length) mem.form_factor = ffs[0];
    const ranks = inst.map((m) => m.rank).filter(Boolean);
    if (ranks.length) mem.rank = ranks[0];

    // 通道数：按 Locator 的“通道前缀”去重，两条规则：
    // ① Locator 含 Channel 字样（常见 BIOS：Controller0-ChannelA-DIMM0）→ 直接取该前缀；
    // ② 否则剥掉槽位尾号（DIMM_A1/DIMM_D2 → DIMM_A/DIMM_D）。
    // 【坑】绝不能用 Bank Locator 兜底：E5 v4 这类平台 Bank=NODE 是内存控制器封装
    //（每颗含 2 通道），两颗控制器会被误判成双通道（127 机实锤：A1/B1+C1/D1/D2
    //  实为四通道，曾误显“双通道 33.6GB/s”，STREAM 实测 50.3GB/s）。
    const chs = new Set(inst.map((m) => {
      const loc = (m.locator || '').trim();
      if (!loc) return '';
      const mm = /^(.*?Channel[A-Za-z0-9]*)/i.exec(loc);
      if (mm) return mm[1].toLowerCase();
      return loc.replace(/[\s_-]*\d+\s*$/i, '').toLowerCase();
    }).filter(Boolean));
    let chCount = chs.size || (inst.length || null);
    if (chCount && inst.length) chCount = Math.min(chCount, inst.length); // 前缀去重不得超过实插条数
    mem.channels = chCount;
    const chNames = ['单通道', '双通道', '三通道', '四通道'];
    mem.channel_text = mem.channels ? (chNames[mem.channels - 1] || (mem.channels + ' 通道')) : null;

    // 理论带宽 ≈ 每通道有效数据位(≤64) × 频率(MT/s) × 通道数 / 8
    // 【坑】ECC RDIMM 位宽应 Data Width=64/Total Width=72，但白牌 BIOS 常把 Data Width
    //  也虚报 72（127 机如此），ECC 校验位不占有效带宽，一律钳到 64。
    const rawWidthBits = inst.length ? (parseFloat(inst[0].data_width) || 64) : 64;
    const widthBits = Math.min(rawWidthBits, 64);
    if (mem.speed_mts && mem.channels) {
      mem.bandwidth_gbs = Math.round(mem.speed_mts * (widthBits / 8) * mem.channels / 1000 * 10) / 10;
    }

    return mem;
  }

  function buildStorageSnapshot() {
    const result = { ts: new Date().toISOString(), disks: [], partitions: [], ioStats: {}, totals: {} };

    // 1) 文件系统用量（df -B1，排除 snap 循环设备与内存文件系统）
    const dfBySource = {};
    try {
      const dfOut = require('child_process').execSync(
        'df -B1 -x squashfs -x tmpfs -x devtmpfs -x overlay --output=source,fstype,size,used,avail,pcent,target 2>/dev/null || true',
        { encoding: 'utf8', timeout: 5000, maxBuffer: 4 * 1024 * 1024 }
      );
      for (const line of dfOut.trim().split('\n').slice(1)) {
        const p = line.trim().split(/\s+/);
        if (p.length < 7) continue;
        const rec = {
          source: p[0], fstype: p[1],
          size_bytes: parseInt(p[2], 10) || 0,
          used_bytes: parseInt(p[3], 10) || 0,
          avail_bytes: parseInt(p[4], 10) || 0,
          size_gb: ((parseInt(p[2], 10) || 0) / 1073741824).toFixed(1),
          used_gb: ((parseInt(p[3], 10) || 0) / 1073741824).toFixed(1),
          avail_gb: ((parseInt(p[4], 10) || 0) / 1073741824).toFixed(1),
          use_pct: parseInt(p[5], 10) || 0,
          mount: p[6],
        };
        result.partitions.push(rec);
        dfBySource[rec.source] = rec;
        dfBySource[p[0].replace('/dev/', '')] = rec;
      }
    } catch (e) {}

    let uptimeS = null;
    try { uptimeS = Math.round(parseFloat(fs.readFileSync('/proc/uptime', 'utf8').split(' ')[0])); } catch (e) {}

    const stats = readDiskStatsRaw();

    // 2) 块设备树（lsblk JSON）→ 整盘条目 + 分区明细
    const flat = [];
    const walkAll = (nodes) => {
      for (const n of (nodes || [])) { flat.push(n); if (Array.isArray(n.children)) walkAll(n.children); }
    };
    walkAll(readLsblkJson());

    const rateWindow = (() => {
      const arr = global.__diskSamples || [];
      if (arr.length < 2) return null;
      const cur = arr[arr.length - 1];
      let base = null;
      for (let i = arr.length - 1; i >= 0; i--) {
        if (cur.ts - arr[i].ts >= 4000) { base = arr[i]; break; }
      }
      if (!base) base = arr[0];
      return Math.round((cur.ts - base.ts) / 100) / 10;
    })();

    for (const node of flat) {
      const kind = diskDevKind(node.name);
      if (kind !== 'disk' && kind !== 'virtual') continue;

      const name = node.name;
      const st = stats[name] || null;
      const mounts = [];
      const partList = [];
      const collectParts = (nodes) => {
        for (const c of (nodes || [])) {
          if (c.mountpoint) mounts.push(c.mountpoint);
          const df = dfBySource[c.name] || dfBySource['/dev/' + c.name];
          const sz = Number(c.size) || 0;
          partList.push({
            name: c.name,
            size_bytes: sz,
            size_gb: (sz / 1073741824).toFixed(1),
            fstype: c.fstype || '',
            mount: c.mountpoint || '',
            used_bytes: df ? df.used_bytes : null,
            avail_bytes: df ? df.avail_bytes : null,
            use_pct: df ? df.use_pct : null,
            used_gb: df ? df.used_gb : null,
            avail_gb: df ? df.avail_gb : null,
          });
          if (Array.isArray(c.children)) collectParts(c.children);
        }
      };
      if (node.mountpoint) mounts.push(node.mountpoint);
      collectParts(node.children);

      const readBytes = st ? st.read_sectors * 512 : null;
      const writeBytes = st ? st.write_sectors * 512 : null;
      const rate = diskRateFor(name);
      const sizeBytes = Number(node.size) || 0;

      const io = {
        // 本次开机累计（/proc/diskstats，重启归零）
        reads: st ? st.reads : null,
        writes: st ? st.writes : null,
        read_bytes: readBytes,
        write_bytes: writeBytes,
        read_h: readBytes != null ? fmtBytes(readBytes) : '--',
        write_h: writeBytes != null ? fmtBytes(writeBytes) : '--',
        read_gb: readBytes != null ? Math.round(readBytes / 1073741824 * 100) / 100 : null,
        write_gb: writeBytes != null ? Math.round(writeBytes / 1073741824 * 100) / 100 : null,
        discard_bytes: st ? st.discard_sectors * 512 : null,
        discards: st ? st.discards : null,
        flushes: st ? st.flushes : null,
        in_flight: st ? st.in_flight : null,
        since_boot_s: uptimeS,
        // 实时速率（约 5 秒窗口均值）
        read_bps: rate ? rate.read_bps : null,
        write_bps: rate ? rate.write_bps : null,
        read_rate_h: rate ? fmtRate(rate.read_bps) : '--',
        write_rate_h: rate ? fmtRate(rate.write_bps) : '--',
        read_iops: rate ? rate.read_iops : null,
        write_iops: rate ? rate.write_iops : null,
        busy_pct: rate ? rate.busy_pct : null,
        read_latency_ms: rate ? rate.read_latency_ms : null,
        write_latency_ms: rate ? rate.write_latency_ms : null,
        avg_queue: rate ? rate.avg_queue : null,
        window_s: rate ? rate.window_s : null,
      };

      let smart = null;
      if (kind === 'disk' && /^(nvme\d+n\d+|sd[a-z]+|vd[a-z]+|hd[a-z]+)$/.test(name)) {
        smart = readSmartInfo('/dev/' + name);
      }

      result.disks.push({
        name,
        path: '/dev/' + name,
        kind,
        type: node.type || '',
        tran: node.tran || '',
        model: (node.model || '').trim() || (smart && smart.model) || '',
        serial: (node.serial || '').trim() || (smart && smart.serial) || '',
        firmware: (node.rev || '').trim() || (smart && smart.firmware) || '',
        rota: node.rota === true || node.rota === '1' || node.rota === 1,
        size_bytes: sizeBytes,
        size_gb: (sizeBytes / 1073741824).toFixed(1),
        size_h: fmtBytes(sizeBytes),
        fstype: node.fstype || '',
        mountpoints: mounts,
        mount_display: mounts.length ? mounts.join(', ') : '未挂载',
        partitions: partList,
        io,
        smart,
      });
    }

    // 按容量倒序（系统盘/数据盘在前，虚拟设备在后）
    result.disks.sort((a, b) => (b.size_bytes || 0) - (a.size_bytes || 0));

    // 设备标签留给走势卡图例复用，省得 /v1/internal/disk-trend 再跑一次 lsblk。
    // mount 取「图例友好」的那一个：lsblk 的 MOUNTPOINT 单列只报一个挂载点，本机数据盘
    // 报的是 chroot bind 的 <真实路径>/rootfs/<真实路径>（如
    // /media/ll/data/vllm-image/rootfs/media/ll/data），故先按 /rootfs/ 剥回真实路径，
    // 再优先 /，最后取最短。表格用的 mount_display 保持原样不动。
    try {
      global.__diskLabels = result.disks.filter((d) => d.kind === 'disk').map((d) => {
        const norm = (m) => { const s = String(m), k = s.indexOf('/rootfs/'); if (k < 0) return s; const rest = s.slice(k + 8); return rest ? (rest.charAt(0) === '/' ? rest : '/' + rest) : '/'; };
        const pool = Array.from(new Set((Array.isArray(d.mountpoints) ? d.mountpoints : []).map(norm)));
        const mount = pool.indexOf('/') >= 0 ? '/' : (pool.length ? pool.sort((a, b) => a.length - b.length)[0] : '未挂载');
        return { name: d.name, model: d.model || '', mount, mount_display: d.mount_display || '未挂载', size_gb: d.size_gb };
      });
    } catch (e) {}

    // 3) 累计 I/O（兼容旧前端字段名：reads/readSectors/writes/writeSectors）
    for (const [name, s] of Object.entries(stats)) {
      result.ioStats[name] = {
        reads: s.reads, readSectors: s.read_sectors, read_bytes: s.read_sectors * 512,
        writes: s.writes, writeSectors: s.write_sectors, write_bytes: s.write_sectors * 512,
        in_flight: s.in_flight, busy_ms: s.io_ms,
      };
    }

    // 4) 汇总
    let tSize = 0, tUsed = 0, tAvail = 0, tRead = 0, tWrite = 0, tReadRate = 0, tWriteRate = 0;
    for (const p of result.partitions) { tSize += p.size_bytes; tUsed += p.used_bytes; tAvail += p.avail_bytes; }
    for (const d of result.disks) {
      if (d.kind !== 'disk') continue;
      tRead += d.io.read_bytes || 0;
      tWrite += d.io.write_bytes || 0;
      tReadRate += d.io.read_bps || 0;
      tWriteRate += d.io.write_bps || 0;
    }
    result.totals = {
      size_bytes: tSize, used_bytes: tUsed, avail_bytes: tAvail,
      use_pct: tSize > 0 ? Math.round(tUsed / tSize * 100) : 0,
      size_h: fmtBytes(tSize), used_h: fmtBytes(tUsed), avail_h: fmtBytes(tAvail),
      read_bytes: tRead, write_bytes: tWrite,
      read_h: fmtBytes(tRead), write_h: fmtBytes(tWrite),
      read_bps: tReadRate, write_bps: tWriteRate,
      read_rate_h: fmtRate(tReadRate), write_rate_h: fmtRate(tWriteRate),
      window_s: rateWindow,
      uptime_s: uptimeS,
      disk_count: result.disks.filter((d) => d.kind === 'disk').length,
      loop_count: flat.filter((n) => diskDevKind(n.name) === 'loop').length,
    };

    // 5) 内存（RAM）：容量 / 实时用量 / 工作频率 / 型号 / 插槽
    try {
      result.memory = readMemoryInfo();
    } catch (e) {
      result.memory = { available: false, modules: [], dmi_error: String((e && e.message) || e) };
    }

    // 6) 网络速度（硬件监视页「网络」卡片）
    try {
      result.network = buildNetSnapshot();
    } catch (e) {
      result.network = { ts: new Date().toISOString(), interfaces: [], totals: {}, error: String((e && e.message) || e) };
    }

    return result;
  }

  // === CATCH-ALL: All /v1/* paths proxy to vLLM ===
  if (pathname.startsWith('/v1/')) {
    if (req.method === 'GET' && pathname === '/v1/models') {
      return handleMergedModels(res);
    }
    // Multi-model routing: requests carrying a "model" field go to that
    // model's own vLLM backend (each pinned to one GPU).
    if (req.method === 'POST' || req.method === 'PUT' || req.method === 'PATCH') {
      let body = '';
      req.on('data', (c) => { body += c; });
      req.on('end', () => {
        let parsed = null;
        let model = null;
        try { parsed = JSON.parse(body); model = parsed.model; } catch (e) {}
        // 模型名别名改写（2026-09-14）：别名（如 qwen3.8-flash-next-nvfp4）在转发前
        // 替换成后端真实 served-model-name，否则 vLLM 按 model 字段校验直接报 404。
        const realModel = resolveModelAlias(model);
        if (parsed && typeof model === 'string' && realModel !== model) {
          parsed.model = realModel;
          body = JSON.stringify(parsed);
        }
        // 客户端请求关联头（2026-09-17 行级测速 v2）：生成类请求注入唯一 id，
        // 代理 tee 捕获响应流首个 SSE data 帧里的引擎真实 rid（chatcmpl-xxx）后
        // 登记 clientReqId→rid，REQ 行据此【精确认领】引擎每请求输出序列。
        // 未知 header vLLM 忽略；SGLang 实例不产出该 header 也无害。
        if (parsed && typeof parsed === 'object' && LIVE_PATH_RE.test(pathname)) {
          const _h = parsed.headers && typeof parsed.headers === 'object' ? parsed.headers : (parsed.headers = {});
          if (!_h['X-Client-Req-Id']) _h['X-Client-Req-Id'] = 'dsh-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 10);
          body = JSON.stringify(parsed);
        }
        let forward = body;
        if (parsed && req.method === 'POST' && pathname === '/v1/chat/completions' && chatTrimEnabled()) {
          try {
            const stats = trimChatBody(parsed);
            if (stats) {
              forward = JSON.stringify(parsed);  // 转发裁剪后的 body：live 指标/估算/count_tokens 与上游一致
              const extra = stats.contentTruncated ? ` content=${stats.contentTruncated}` : '';
              console.log(`[chat-trim] ${clientIp(req)} msgs ${stats.before}->${stats.after} (dropped ${stats.dropped})${extra}`);
            }
          } catch (e) {
            console.error(`[chat-trim] error: ${e && e.message}`);
          }
        }
        // PD 两步转发：命中 PD 模型的生成端点时，自动拆 prefill→decode 两步
        // （prefill 只算 KV，decode 生成；客户端无感。工具调用暂不支持——PD 实例未开 tool parser）
        const pd = pdPortsForModel(realModel);
        if (pd && req.method === 'POST' && LIVE_PATH_RE.test(pathname)) {
          return proxyPdToVllm(req, res, req.url, pd, forward, model);
        }
        proxyToVllm(req, res, req.url, vllmPortForModel(model), forward);
      });
      req.on('error', () => { try { if (!res.headersSent) res.destroy(); } catch (e) {} });
      return;
    }
    return proxyToVllm(req, res, req.url);
  }

  // === Serve admin UI ===
  if (pathname === '/' || pathname === '/index.html') {
    const indexPath = path.join(__dirname, 'index.html');
    try {
      const content = fs.readFileSync(indexPath, 'utf8');
      // 09-20：no-store → no-cache + ETag：内容未变时 304 空响应（省 461KB 重传），改了立即生效
      const etag = 'W/"' + Buffer.byteLength(content) + '-' + fs.statSync(indexPath).mtimeMs.toString(36) + '"';
      if (req.headers['if-none-match'] === etag) { res.writeHead(304); res.end(); return; }
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-cache', 'ETag': etag });
      res.end(content);
    } catch (e) {
      res.writeHead(404);
      res.end('Not found');
    }
    return;
  }

  // === Serve mobile UI ===
  if (pathname === '/m' || pathname === '/mobile.html') {
    const mobilePath = path.join(__dirname, 'mobile.html');
    try {
      const content = fs.readFileSync(mobilePath, 'utf8');
      const etag = 'W/"' + Buffer.byteLength(content) + '-' + fs.statSync(mobilePath).mtimeMs.toString(36) + '"';
      if (req.headers['if-none-match'] === etag) { res.writeHead(304); res.end(); return; }
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-cache', 'ETag': etag });
      res.end(content);
    } catch (e) {
      res.writeHead(404);
      res.end('Not found');
    }
    return;
  }

  // === Serve bench console UI（09-23：「基准测试」标签的内嵌页）===
  if (pathname === '/bench.html' || pathname === '/bench') {
    const benchPath = path.join(__dirname, 'bench.html');
    try {
      const content = fs.readFileSync(benchPath, 'utf8');
      const etag = 'W/"' + Buffer.byteLength(content) + '-' + fs.statSync(benchPath).mtimeMs.toString(36) + '"';
      if (req.headers['if-none-match'] === etag) { res.writeHead(304); res.end(); return; }
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-cache', 'ETag': etag });
      res.end(content);
    } catch (e) {
      res.writeHead(404);
      res.end('bench.html not found');
    }
    return;
  }

  // === Serve CPU control UI（移植 bench-console/cpu-control：「CPU 控制」标签的内嵌页）===
  if (pathname === '/cpu.html' || pathname === '/cpu') {
    const cpuPath = path.join(__dirname, 'cpu.html');
    try {
      const content = fs.readFileSync(cpuPath, 'utf8');
      const etag = 'W/"' + Buffer.byteLength(content) + '-' + fs.statSync(cpuPath).mtimeMs.toString(36) + '"';
      if (req.headers['if-none-match'] === etag) { res.writeHead(304); res.end(); return; }
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-cache', 'ETag': etag });
      res.end(content);
    } catch (e) {
      res.writeHead(404);
      res.end('cpu.html not found');
    }
    return;
  }

  // === Serve SGLang monitor UI（[sglang-adapt-1003]「SGLang」标签的内嵌页，纯只读监控）===
  if (pathname === '/sglang.html' || pathname === '/sglang') {
    const sglPath = path.join(__dirname, 'sglang.html');
    try {
      const content = fs.readFileSync(sglPath, 'utf8');
      const etag = 'W/"' + Buffer.byteLength(content) + '-' + fs.statSync(sglPath).mtimeMs.toString(36) + '"';
      if (req.headers['if-none-match'] === etag) { res.writeHead(304); res.end(); return; }
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-cache', 'ETag': etag });
      res.end(content);
    } catch (e) {
      res.writeHead(404);
      res.end('sglang.html not found');
    }
    return;
  }

  // === Serve vLLM monitor UI（[vllm-page-1003]「vLLM」标签的内嵌页，纯只读监控）===
  if (pathname === '/vllm.html' || pathname === '/vllm') {
    const vlPath = path.join(__dirname, 'vllm.html');
    try {
      const content = fs.readFileSync(vlPath, 'utf8');
      const etag = 'W/"' + Buffer.byteLength(content) + '-' + fs.statSync(vlPath).mtimeMs.toString(36) + '"';
      if (req.headers['if-none-match'] === etag) { res.writeHead(304); res.end(); return; }
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-cache', 'ETag': etag });
      res.end(content);
    } catch (e) {
      res.writeHead(404);
      res.end('vllm.html not found');
    }
    return;
  }

  // === Serve Strata 远端监控 UI（「Strata 监控」标签的内嵌页，克隆 .38:8080 Web UI Monitor，纯只读）===
  if (pathname === '/strata-monitor.html' || pathname === '/strata-monitor') {
    const smPath = path.join(__dirname, 'strata-monitor.html');
    try {
      const content = fs.readFileSync(smPath, 'utf8');
      const etag = 'W/"' + Buffer.byteLength(content) + '-' + fs.statSync(smPath).mtimeMs.toString(36) + '"';
      if (req.headers['if-none-match'] === etag) { res.writeHead(304); res.end(); return; }
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-cache', 'ETag': etag });
      res.end(content);
    } catch (e) {
      res.writeHead(404);
      res.end('strata-monitor.html not found');
    }
    return;
  }

  // === Serve static files ===
  if (pathname.startsWith('/static/')) {
    const filePath = path.join(__dirname, pathname);
    try {
      const st = fs.statSync(filePath);
      const content = fs.readFileSync(filePath);
      const ext = path.extname(filePath);
      // 09-20：静态资源加 ETag/304（tailwind.js 407KB 命中 304 即零传输）；
      // 缓存 10 分钟平衡更新时效（改动后最多 10 分钟生效，强刷立即生效）
      const etag = '"' + st.size.toString(36) + '-' + st.mtimeMs.toString(36) + '"';
      if (req.headers['if-none-match'] === etag) { res.writeHead(304); res.end(); return; }
      res.writeHead(200, { 'Content-Type': MIME_TYPES[ext] || 'application/octet-stream', 'Cache-Control': 'public, max-age=600', 'ETag': etag });
      res.end(content);
    } catch (e) {
      res.writeHead(404);
      res.end('Not found');
    }
    return;
  }

  // === Fallback ===
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Not Found', path: pathname }));
});

// 启动时自动探测运行中的 vLLM 实例并注册进路由表（控制台重启后不丢多模型路由）
getVllmInstances([config.vllmPort, 8001, 8002, 8003], (insts) => {
  insts.forEach(i => {
    if (i.model && i.port) VLLM_MODEL_PORTS[i.model] = i.port;
    // 重建 GPU 占用登记表（控制台重启后 GPU 独占保护仍生效）
    if (i.port && i.gpu) {
      const gpuParts = String(i.gpu).split(',').map(s => parseInt(s.trim()) || 0);
      global.__GPU_INSTANCES.set(i.port, {
        port: i.port,
        gpuId: gpuParts[0] || 0,
        gpuCount: Math.max(1, gpuParts.length),
        runtime: i.runtime || 'vllm',
        model: i.model,
        startedAt: Date.now(),
      });
    }
  });
});

server.listen(config.serverPort, '0.0.0.0', () => {
  console.log('='.repeat(50));
  console.log('  vLLM Management Console');
  console.log('='.repeat(50));
  console.log(`  Admin URL: http://localhost:${config.serverPort}/`);
  console.log(`  vLLM backend: ${vllmBaseUrl}`);
  console.log('='.repeat(50));
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`Port ${config.serverPort} already in use!`);
    process.exit(1);
  } else {
    // 其它 listen 失败（权限不足等）：也打日志并退出，避免进程活着但不监听任何端口
    console.error('Server error:', err.message || err);
    process.exit(1);
  }
});

process.on('SIGTERM', () => { server.close(); process.exit(0); });
process.on('SIGINT', () => { server.close(); process.exit(0); });
