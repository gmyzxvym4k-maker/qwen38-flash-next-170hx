#!/usr/bin/env node
/* ============================================================
 * crash-watch.mjs — vLLM/SGLang 崩溃记录守护进程（10-10 上线）
 *
 * 定位：8889 控制台「崩溃记录」标签的数据源。**不改动 server.js、不重启
 * dsh-console**（vLLM 在其 cgroup 内，重启连坐引擎），而是以独立 node 进程
 * （systemd --user: fnx-crash-watch.service，ll 身份运行）做三件事：
 *   ① 实时监测：轮询控制台 /v1/internal/model-manager 的实例快照，端口连续
 *      两轮消失且 /proc/<pid> 不存在 → 判定引擎异常退出（崩溃）；同时增量
 *      tail 看门狗日志 fnx-*-watchdog.log 的「确认离线」行（带精确时间戳，
 *      是 18420 最权威的离线信号）与 server.log 的启停事件（人工停止/控制台
 *      重启 → 不算崩溃，抑制记录）。
 *   ② 原因分析：崩溃时读该实例日志尾部（日志路径经「祖先链 cmdline 的 .log
 *      token」判据解析，root 进程 fd 不可读但 cmdline 全局可读，口径同
 *      server.js ancestorLogFiles），从尾部反向匹配错误签名（EngineDead/
 *      CUDA OOM/NCCL 超时/段错误/断言/磁盘满/非法内存…）取最深具体签名为根因，
 *      附带原文证据；再查 dmesg 窗口内 OOM Killer / NVRM Xid 补充硬件侧原因。
 *   ③ 历史回填：首跑扫描 fnx-*-watchdog.log、vllm*.log、dmesg 里既往崩溃
 *      事件生成 backfill 记录（此后不再重复扫描，state.backfillDone）。
 * 输出：/home/ll/deploy/crash-records.json（真源，上限 1000 条）
 *       /home/ll/deploy/static/crash-records.json（前端消费；/static/ 路由
 *       每请求读盘，控制台免重启即可取到最新）。
 * 纪律：全程不碰 nvidia-smi（09-20 铁律）；所有子进程带超时自建结算；任何
 * 单轮异常都不允许打死循环（tick 外层 catch + setTimeout 续排）。
 * ============================================================ */
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { execFile } from 'node:child_process';

const BASE = process.env.CRASH_WATCH_BASE || '/home/ll/deploy';
const API = process.env.CRASH_WATCH_API || 'http://127.0.0.1:8889/v1/internal/model-manager';
const RECORDS_FILE = path.join(BASE, 'crash-records.json');
const OUT_FILE = path.join(BASE, 'static', 'crash-records.json');
const STATE_FILE = path.join(BASE, 'crash-watch-state.json');
const SELFLOG = path.join(BASE, 'fnx-crash-watch.log');
const SERVER_LOG = path.join(BASE, 'server.log');

const TICK_MS = 5000;                 // 主循环周期
const MAX_RECORDS = 1000;             // 记录上限（超出丢最旧）
const MANUAL_SUPPRESS_S = 600;        // server.log 出现停止指令后 N 秒内的离线不算崩溃
const CONSOLE_RESTART_SUPPRESS_S = 300; // 控制台刚重启（横幅）后 N 秒内实例消失不算崩溃
const MERGE_WINDOW_MS = 180 * 1000;   // 同端口 N 分钟内只留一条（watchdog/gap/backfill 去重）
const CAUSE_WINDOW_MS = 10 * 60 * 1000; // 日志尾部找签名时只认事件时间前 N 分钟内的行

// ---------- 崩溃签名（specific=优先作为根因） ----------
const SIGS = [
  { code: 'CUDA_OOM',     label: '显存不足（CUDA out of memory）',            re: /CUDA out of memory|torch\.OutOfMemoryError/, specific: true },
  { code: 'ILLEGAL_MEM',  label: 'CUDA 非法内存访问（illegal memory access）', re: /illegal memory access/, specific: true },
  { code: 'NCCL_TIMEOUT', label: '多卡通信超时（NCCL watchdog / collective）',  re: /Watchdog caught collective operation timeout|NCCL error|ncclInternalError|ncclUnhandledCudaError/i, specific: true },
  { code: 'SEGFAULT',     label: '段错误（SIGSEGV / core dumped）',            re: /Segmentation fault|core dumped|SIGSEGV/, specific: true },
  { code: 'ASSERT',       label: '引擎断言失败（Assertion failed）',           re: /Assertion.{0,160}failed|AssertionError/i, specific: true },
  { code: 'DISK_FULL',    label: '磁盘写满（No space left on device）',        re: /No space left on device/, specific: true },
  { code: 'CUDA_DRV',     label: 'CUDA 驱动/上下文错误',                       re: /CUDA error:|CUBLAS_STATUS|cudnn.*error|no CUDA-capable device/i, specific: true },
  { code: 'ENGINE_DEAD',  label: 'EngineCore 崩溃（引擎核心进程死亡）',         re: /EngineDeadError|EngineCore encounter|EngineCore process (?:died|terminated|exited)|Engine process .* terminated/i, specific: false },
  { code: 'WORKER_DIED',  label: 'GPU Worker 进程异常退出',                    re: /Worker.*?\b(?:died|exited unexpectedly|terminated)\b/i, specific: false },
  { code: 'PY_FATAL',     label: 'Python 致命异常导致引擎退出',                 re: /(?:^|\]\s)(?:[A-Za-z_][\w.]*)(?:Error|Exception|FatalError)(?::|\s)/, specific: false },
  { code: 'SIGNAL_TERM',  label: '进程收到终止信号退出（外部 kill / 系统 OOM / 关机）', re: /received signal (?:SIGTERM|SIGINT|SIGQUIT|SIGHUP|SIGKILL)|Signal \d+ received|terminated with signal|killed with signal/i, specific: false },
];
// 干净停机特征（回填时用于把「人为停止」从崩溃里剔除）
const CLEAN_STOP = /Shutdown complete|Waiting for application shutdown|Shutting down|SIGTERM signal received|Received signal SIGTERM|Received signal SIGINT|Runtime cleanup successful|engine client stopped|优雅停止|cleanly/i;

// ---------- 小工具 ----------
function selflog(msg) {
  try {
    if (fs.existsSync(SELFLOG) && fs.statSync(SELFLOG).size > 2 * 1024 * 1024) fs.writeFileSync(SELFLOG, '');
    fs.appendFileSync(SELFLOG, `[${fmtTs(Date.now())}] ${msg}\n`);
  } catch (e) { /* 自监控日志失败静默 */ }
}
function fmtTs(ms) {
  const d = new Date(ms);
  const p = (n, w = 2) => String(n).padStart(w, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}
function clip(s, n = 300) { s = String(s || ''); return s.length > n ? s.slice(0, n) + '…' : s; }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
async function writeAtomic(fp, text) {
  const tmp = fp + '.tmp';
  await fsp.writeFile(tmp, text);
  await fsp.rename(tmp, fp);
}
// 带自建超时结算的 execFile（铁律：定时器直接结算，不依赖 close 事件）
function execCollect(file, args, ms, maxBuf = 32 * 1024 * 1024) {
  return new Promise((resolve) => {
    let done = false; let proc;
    const finish = (v) => { if (!done) { done = true; try { if (proc) proc.kill('SIGKILL'); } catch (e) {} resolve(v); } };
    const timer = setTimeout(() => finish(null), ms);
    try {
      proc = execFile(file, args, { maxBuffer: maxBuf, timeout: ms }, (err, stdout, stderr) => {
        clearTimeout(timer);
        finish({ stdout: String(stdout || ''), stderr: String(stderr || ''), code: err ? 1 : 0 });
      });
    } catch (e) { clearTimeout(timer); finish(null); }
  });
}

// ---------- 时间戳解析 ----------
// vLLM 日志行内时间：`(APIServer pid=1) ERROR 10-10 18:35:34 [xx.py:9] ...` → MM-DD HH:MM:SS
// 年份用 hint（文件 mtime 年份，跨回滚判定）补全；watchdog 行是完整 [YYYY-MM-DD HH:MM:SS]。
function parseLogTs(line, yearHint) {
  const m = /(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})/.exec(line);
  if (!m) return null;
  const y = yearHint || new Date().getFullYear();
  const d = new Date(y, +m[1] - 1, +m[2], +m[3], +m[4], +m[5]);
  if (isNaN(d.getTime())) return null;
  if (d.getTime() > Date.now() + 3 * 86400000) d.setFullYear(y - 1); // 未来 3 天视为跨年旧日志
  return d.getTime();
}
function parseAbsTs(line) {
  const m = /\[(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\]/.exec(line);
  if (!m) return null;
  return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]).getTime();
}

// ---------- 增量读文件（inode/截断/半行 carry 处理） ----------
async function readNewLines(fp, st) {
  // st: {ino, offset, carry} → { lines, st }
  let s;
  try { s = await fsp.stat(fp); } catch (e) { return { lines: [], st: null }; }
  let out = [];
  if (!st || st.ino !== s.ino) st = { ino: s.ino, offset: Math.max(0, s.size - 256 * 1024), carry: '' }; // 首见：只回看尾部 256KB
  if (s.size < st.offset) { st.offset = 0; st.carry = ''; } // 截断
  if (s.size === st.offset) return { lines: out, st };
  try {
    const fh = await fsp.open(fp, 'r');
    try {
      const len = Math.min(s.size - st.offset, 2 * 1024 * 1024);
      const buf = Buffer.alloc(len);
      await fh.read(buf, 0, len, st.offset);
      st.offset += len;
      const text = st.carry + buf.toString('utf8');
      const parts = text.split('\n');
      st.carry = text.endsWith('\n') ? '' : parts.pop();
      out = parts;
    } finally { await fh.close(); }
  } catch (e) { return { lines: [], st }; }
  return { lines: out, st };
}
// ---------- 崩溃日志解析：读尾部并按需扩展窗口（行 + 事件 ts） ----------
async function tailEvents(fp) {
  try {
    let size = (await fsp.stat(fp)).size;
    for (const cap of [1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]) {
      const from = Math.max(0, size - cap);
      const fh = await fsp.open(fp, 'r');
      let text = '';
      try {
        const buf = Buffer.alloc(size - from);
        await fh.read(buf, 0, buf.length, from);
        text = buf.toString('utf8');
      } finally { await fh.close(); }
      if (from > 0) text = text.slice(text.indexOf('\n') + 1);
      const lines = text.split('\n');
      let lastTs = null;
      const rows = [];
      for (const ln of lines) {
        if (!ln.trim()) continue;
        const t = parseLogTs(ln);
        if (t != null) lastTs = t;
        rows.push({ line: ln, ts: lastTs });
      }
      if (rows.length && rows[rows.length - 1].ts != null) return rows; // 尾部带得出时间 → 够定位
      if (from === 0) return rows; // 整文件都读了
    }
    return [];
  } catch (e) { return []; }
}

// ---------- 日志路径判据：祖先链 cmdline 的 .log token（cmdline 全局可读） ----------
function resolveLogFile(pid) {
  let cur = pid;
  const found = [];
  for (let hop = 0; hop < 8 && cur && cur > 1; hop++) {
    let cmd = '';
    let ppid = 0;
    try {
      cmd = fs.readFileSync(`/proc/${cur}/cmdline`, 'utf8').replace(/\0/g, ' ');
      const st = fs.readFileSync(`/proc/${cur}/status`, 'utf8');
      const m = /^PPid:\s*(\d+)/m.exec(st);
      ppid = m ? +m[1] : 0;
    } catch (e) { break; }
    for (const tok of cmd.split(/\s+/)) {
      if (/\.log(\.\d+)?$/.test(tok) && tok.startsWith('/') && !found.includes(tok)) found.push(tok);
    }
    cur = ppid;
  }
  return found[0] || null; // hop 近者优先
}
async function pickFallbackLog() {
  // 兜底：deploy 目录下 mtime 最新且在 10 分钟内写过的 vllm*.log
  try {
    const files = (await fsp.readdir(BASE)).filter(f => /^vllm.*\.log$/.test(f));
    let best = null, bestM = 0;
    for (const f of files) {
      try {
        const s = await fsp.stat(path.join(BASE, f));
        if (Date.now() - s.mtimeMs < 10 * 60 * 1000 && s.mtimeMs > bestM) { best = path.join(BASE, f); bestM = s.mtimeMs; }
      } catch (e) {}
    }
    return best;
  } catch (e) { return null; }
}

// 看门狗日志 → 端口映射：fnx-18420-watchdog.log 直读；裸名 fnx-watchdog.log 是 18420 守护
// （fnx-18420-watchdog.sh 的 LOG 变量即此名，systemd 单元名才带端口段）。
function wdLogFilePort(f) {
  const mp = /fnx-(\d+)-watchdog\.log/.exec(f);
  if (mp) return +mp[1];
  if (/^fnx-watchdog\.log/.test(f)) return 18420;
  return null;
}
// ---------- 状态与记录存储 ----------
async function loadJson(fp, dflt) { try { return JSON.parse(await fsp.readFile(fp, 'utf8')); } catch (e) { return dflt; } }
let records = [];
let state = {
  backfillDone: false, tails: {}, insts: {}, manualStopAt: {}, consoleRestartAt: 0, startedAt: Date.now(),
};

async function saveRecords() {
  records.sort((a, b) => b.ts - a.ts);
  if (records.length > MAX_RECORDS) records = records.slice(0, MAX_RECORDS);
  const payload = JSON.stringify({ generator: 'crash-watch', updatedAt: new Date().toISOString(), total: records.length, records }, null, 1);
  await writeAtomic(RECORDS_FILE, payload);
  try { fs.mkdirSync(path.dirname(OUT_FILE), { recursive: true }); } catch (e) {}
  await writeAtomic(OUT_FILE, payload);
}
async function saveState() { await writeAtomic(STATE_FILE, JSON.stringify(state)); }

function dedupPush(rec) {
  // 同端口 ±MERGE_WINDOW 内已有 → 不重复记录；若旧记录无根因细节而新的有，则替换
  for (let i = 0; i < records.length; i++) {
    const r = records[i];
    if (r.port === rec.port && Math.abs(r.ts - rec.ts) <= MERGE_WINDOW_MS) {
      const better = (rec.evidence && rec.evidence.length > 0) && !(r.evidence && r.evidence.length > 0);
      if (better) records[i] = { ...r, ...rec, id: r.id };
      return false;
    }
  }
  records.push(rec);
  return true;
}

// ---------- 原因分析 ----------
function analyzeCause(rows, eventTs) {
  // rows: [{line, ts}]；取窗口内（事件前 CAUSE_WINDOW_MS ~ 事件后 1 分钟）的签名行。
  // 根因选择：有具体签名（specific，如 CUDA_OOM/NCCL/断言）→ 取最后一个具体签名
  // （EngineCore 侧真凶通常在尾部错误块的最深处）；否则取最后一个泛化签名
  // （EngineDead/Worker 死/异常行——它们多是 API 侧对已死核心的复读，标为兜底原因）。
  const lo = eventTs != null ? eventTs - CAUSE_WINDOW_MS : -Infinity;
  const hi = eventTs != null ? eventTs + 60000 : Infinity;
  let lastSpecific = null;
  let lastGeneric = null;
  let cleanStopSeen = false;
  for (let i = 0; i < rows.length; i++) {
    const { line, ts } = rows[i];
    if (ts != null && (ts < lo || ts > hi)) continue;
    if (CLEAN_STOP.test(line)) cleanStopSeen = true;
    for (const sg of SIGS) {
      if (!sg.re.test(line)) continue;
      const cand = { sg, idx: i, line, ts };
      if (sg.specific) lastSpecific = cand; else lastGeneric = cand;
      break;
    }
  }
  const best = lastSpecific || lastGeneric;
  if (!best) {
    if (cleanStopSeen) return { causeCode: 'CLEAN_STOP', cause: '干净停机（收到停止指令，非崩溃）', evidence: [] };
    return { causeCode: 'UNKNOWN', cause: '未知（日志尾部未匹配到已知错误签名）', evidence: rows.slice(-6).map(r => clip(r.line)).filter(Boolean) };
  }
  const ctx = rows.slice(Math.max(0, best.idx - 4), best.idx + 11).map(r => clip(r.line)).filter(Boolean).slice(0, 15);
  return { causeCode: best.sg.code, cause: best.sg.label, evidence: ctx };
}

async function dmesgContext(eventTs) {
  // 事件前后 6 分钟内与推理进程相关的 OOM Killer / NVRM Xid 行
  try {
    const r = await execCollect('dmesg', [], 8000);
    if (!r) return null;
    let bootMs = Date.now() - parseFloat(fs.readFileSync('/proc/uptime', 'utf8').split(' ')[0]) * 1000;
    const lo = eventTs - 6 * 60000, hi = eventTs + 6 * 60000;
    const oom = [], xid = [];
    for (const ln of r.stdout.split('\n')) {
      const m = /^\[\s*(\d+\.\d+)\]/.exec(ln);
      if (!m) continue;
      const wall = bootMs + parseFloat(m[1]) * 1000;
      if (wall < lo || wall > hi) continue;
      if (/Killed process|oom-kill|Out of memory/i.test(ln) && /VLLM|vllm|sglang|python|EngineCore/i.test(ln)) oom.push(clip(ln, 260));
      else if (/NVRM: Xid/i.test(ln)) xid.push(clip(ln, 260));
    }
    if (oom.length) return { causeCode: 'CPU_OOM', cause: '系统内存耗尽，被内核 OOM Killer 终止', evidence: oom.slice(0, 6) };
    if (xid.length) return { causeCode: 'GPU_XID', cause: 'GPU Xid 硬件/驱动错误', evidence: xid.slice(0, 6) };
    return null;
  } catch (e) { return null; }
}

// ---------- 崩溃记录主流程 ----------
async function recordCrash({ port, model, runtime, tsEpoch, tsExact, source, pid }) {
  const key = `${port}:${Math.round(tsEpoch / 60000)}`;
  if (recordedKeys.has(key)) return; // 单轮内快速去重
  const info = state.insts[String(port)] || {};
  const logFile = info.log || await pickFallbackLog();
  let cause = { causeCode: 'UNKNOWN', cause: `日志不可得（${logFile || '未定位到日志文件'}），无法判定具体原因`, evidence: [] };
  if (logFile) {
    const rows = await tailEvents(logFile);
    const a = analyzeCause(rows, tsEpoch);
    if (a.causeCode === 'CLEAN_STOP') { selflog(`抑制停机误报: port=${port} ${fmtTs(tsEpoch)}（日志尾部为优雅停机特征，非崩溃）`); return; }
    cause = a;
  }
  // 硬件/OOM 侧证据优先补充
  const hw = await dmesgContext(tsEpoch);
  if (hw && (cause.causeCode === 'UNKNOWN' || cause.causeCode === 'SIGNAL_TERM' || cause.causeCode === 'ENGINE_DEAD')) {
    cause = { ...hw, specific: true };
  }
  const rec = {
    id: 'r' + tsEpoch + ':' + port + ':' + Math.floor(Math.random() * 1e4),
    ts: tsEpoch, tsText: fmtTs(tsEpoch), port, model: model || info.model || null,
    runtime: runtime || info.runtime || 'vllm', pid: pid || info.pid || null,
    causeCode: cause.causeCode, cause: cause.cause,
    evidence: cause.evidence || [], logFile: logFile || null,
    source, // live=进程消失监测 / watchdog=看门狗确认离线 / backfill=历史回填
    tsExact: !!tsExact, note: null,
  };
  if (dedupPush(rec)) {
    recordedKeys.add(key);
    await saveRecords();
    selflog(`记录崩溃: port=${port} ${rec.tsText} [${rec.causeCode}] ${rec.cause}${rec.source !== 'backfill' ? '（' + rec.source + '）' : ''}`);
  }
}
const recordedKeys = new Set();

// ---------- 每轮 tick ----------
let tickBusy = false;
async function tick() {
  if (tickBusy) return;
  tickBusy = true;
  try {
    if (!state.backfillDone) { await backfill(); state.backfillDone = true; await saveState(); }

    // ① server.log 增量：启停事件
    const sl = await readNewLines(SERVER_LOG, state.tails['__server__']);
    state.tails['__server__'] = sl.st;
    for (const line of sl.lines) {
      let m = /\[(?:script-model|quickstart-script)\] stop \S+ port (\d+)/.exec(line) || /\[pd-stop\] stopPort=(\d+)/.exec(line);
      if (m) { state.manualStopAt[m[1]] = Date.now(); state.manualStopAt['*'] = Date.now(); }
      if (/\[hotshield\]|\[stop\]|\[strata\].*停止/.test(line)) state.manualStopAt['*'] = Date.now();
      if (/^\s*vLLM Management Console\s*$/.test(line)) state.consoleRestartAt = Date.now();
    }

    // ② 看门狗日志增量：「确认离线」是权威崩溃信号（带精确时间戳）
    for (const f of fs.readdirSync(BASE).filter(x => /^fnx-.*watchdog\.log$/.test(x))) {
      const fp = path.join(BASE, f);
      const tr = await readNewLines(fp, state.tails[f]);
      state.tails[f] = tr.st;
      const wport = wdLogFilePort(f);
      for (const line of tr.lines) {
        if (!/确认离线|拉起/.test(line)) continue;
        const ts = parseAbsTs(line);
        // offset 已持久化，正常只会读到新行；state 被清空重跑时跳过回填已覆盖的陈旧行
        if (ts == null || ts < state.startedAt - 120000) continue;
        // 「拉起」行 → 给最近的离线记录补备注（看门狗是否自动恢复、是否降级为安全模式）
        if (/拉起/.test(line) && state.lastWdOffline && ts >= state.lastWdOffline.ts && ts - state.lastWdOffline.ts <= 180000) {
          const hit = records.find(r => r.port === state.lastWdOffline.port && Math.abs(r.ts - state.lastWdOffline.ts) <= MERGE_WINDOW_MS);
          if (hit && !hit.note) {
            hit.note = /安全模式/.test(line) ? '看门狗已拉起（安全模式：已剥离内存二级缓存档）' : '看门狗已自动拉起';
            await saveRecords();
          }
          state.lastWdOffline = null;
          continue;
        }
        if (!/确认离线/.test(line)) continue;
        const port = wport == null ? 0 : wport;
        const stopAt = Math.max(state.manualStopAt[wport == null ? '0' : String(wport)] || 0, state.manualStopAt['*'] || 0);
        if (ts - stopAt < MANUAL_SUPPRESS_S * 1000) continue;           // 人工停止在前 → 不算崩溃
        if (ts - state.consoleRestartAt < CONSOLE_RESTART_SUPPRESS_S * 1000) continue; // 控制台重启连坐 → 不算崩溃
        state.lastWdOffline = { port, ts };
        await recordCrash({ port, tsEpoch: ts, tsExact: true, source: 'watchdog',
          model: state.insts[String(port)] && state.insts[String(port)].model });
      }
    }

    // ③ model-manager 实例快照：端口消失 + /proc 校验 = 异常退出
    let data = null;
    try {
      const ac = new AbortController();
      const to = setTimeout(() => ac.abort(), 9000);
      try {
        const resp = await fetch(API, { signal: ac.signal });
        if (resp.ok) data = await resp.json();
      } finally { clearTimeout(to); }
    } catch (e) { data = null; }
    if (data && Array.isArray(data.instances)) {
      const now = new Map();
      for (const it of data.instances) {
        if (!it || !it.port) continue;
        const k = String(it.port);
        const prev = state.insts[k] || {};
        let log = prev.log;
        if (!log || prev.pid !== it.pid) log = resolveLogFile(it.pid);
        now.set(k, { pid: it.pid, model: it.model || null, runtime: it.runtime || 'vllm', log: log || null, missing: 0 });
        // 同端口 pid 变了且旧 pid 进程已死 → 也是异常退出（重启太快 API 没断档）
        if (prev.pid && it.pid && prev.pid !== it.pid) {
          if (!fs.existsSync(`/proc/${prev.pid}`) && Date.now() - (state.manualStopAt[k] || 0) > MANUAL_SUPPRESS_S * 1000) {
            await recordCrash({ port: +k, tsEpoch: Date.now(), tsExact: false, source: 'live', pid: prev.pid, model: prev.model, runtime: prev.runtime });
          }
        }
      }
      for (const [k, prev] of Object.entries(state.insts)) {
        if (now.has(k)) continue;
        prev.missing = (prev.missing || 0) + 1;
        if (prev.missing >= 2 && prev.pid && !fs.existsSync(`/proc/${prev.pid}`)) {
          const stopAt = Math.max(state.manualStopAt[k] || 0, state.manualStopAt['*'] || 0);
          if (Date.now() - stopAt > MANUAL_SUPPRESS_S * 1000 && Date.now() - state.consoleRestartAt > CONSOLE_RESTART_SUPPRESS_S * 1000) {
            await recordCrash({ port: +k, tsEpoch: Date.now(), tsExact: false, source: 'live', pid: prev.pid, model: prev.model, runtime: prev.runtime });
          }
          delete state.insts[k];
        } else { now.set(k, prev); } // 保留观察一轮
      }
      state.insts = Object.fromEntries(now);
      state.apiDown = false;
    } else {
      // 控制台接口不可达：保留旧快照，不做消失判定（防误报）
      state.apiDown = true;
    }
    await saveState();
  } catch (e) {
    selflog('tick 异常: ' + (e && e.stack ? e.stack.split('\n')[0] : String(e)));
  } finally {
    tickBusy = false;
  }
}

// ---------- 历史回填 ----------
// 本机部署惯例：文件名端口直读（vllm-xxx-18430.log），flash-next/0310/0300/w4a16 系列属 18420。
const FILE_HINTS = [
  { re: /-((?:1[0-9]{3,4}))\.log/, port: null, model: null },           // 文件名尾端口
  { re: /uncensored/, port: 18430, model: 'qwen3.8-flash-next-uncensored' },
  { re: /flash-?next|0310|0300|w4a16/, port: 18420, model: 'qwen3.8-flash-next' },
];
function fileHint(f) {
  const pm = /-((?:1[0-9]{3,4}))\.log/.exec(f);
  for (const h of FILE_HINTS) {
    if (h.re !== undefined && h.re.test(f) && !pm && (h.port || h.model)) return h;
  }
  if (pm) return { port: +pm[1], model: null };
  for (const h of FILE_HINTS) if (h.port && h.re.test(f)) return h;
  return null;
}
async function backfill() {
  const t0 = Date.now();
  const events = []; // 统一事件池：watchdog 离线 / 日志签名 / dmesg OOM
  let wdCount = 0, sigCount = 0, dmCount = 0;

  // A. 看门狗日志全量：每一行「确认离线」是一次权威离线；后续「拉起」行做成备注
  for (const f of fs.readdirSync(BASE).filter(x => /^fnx-.*watchdog\.log$/.test(x))) {
    const port = wdLogFilePort(f);
    let lines = [];
    try { lines = (await fsp.readFile(path.join(BASE, f), 'utf8')).split('\n'); } catch (e) { continue; }
    let lastOffline = null;
    for (const line of lines) {
      const ts = parseAbsTs(line);
      if (ts == null) continue;
      if (/确认离线/.test(line)) {
        const ev = { port, ts, kind: 'watchdog', causeCode: 'PENDING', cause: '', evidence: [], logFile: null, note: null, _buf: [] };
        events.push(ev); lastOffline = ev; wdCount++;
      } else if (lastOffline && ts - lastOffline.ts <= 180000 && /拉起/.test(line)) {
        lastOffline.note = /安全模式/.test(line) ? '看门狗已拉起（安全模式：已剥离内存二级缓存档）' : '看门狗已自动拉起';
        lastOffline = null;
      }
    }
  }
  const wdSorted = events.filter(e => e.kind === 'watchdog').sort((a, b) => a.ts - b.ts);
  const wdLo = (ts) => { // 第一个 ev.ts >= ts 的下标
    let lo = 0, hi = wdSorted.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (wdSorted[m].ts < ts) lo = m + 1; else hi = m; }
    return lo;
  };

  // B. vllm 日志签名扫描 + 为看门狗事件按窗口收集错误行（时间两路归并）
  const vlogFiles = (await fsp.readdir(BASE)).filter(x => /^vllm.*\.log$/.test(x));
  for (const f of vlogFiles) {
    const fp = path.join(BASE, f);
    let s; try { s = await fsp.stat(fp); } catch (e) { continue; }
    if (s.size === 0 || s.size > 300 * 1024 * 1024) continue;
    const hint = fileHint(f);
    const ring = [];
    let lastTs = null, prevSigTs = 0;
    const fh = await fsp.open(fp, 'r');
    let pos = 0, carry = '';
    const yearHint = new Date(s.mtimeMs).getFullYear();
    try {
      while (pos < s.size) {
        const len = Math.min(4 * 1024 * 1024, s.size - pos);
        const buf = Buffer.alloc(len);
        await fh.read(buf, 0, len, pos);
        pos += len;
        const parts = (carry + buf.toString('utf8')).split('\n');
        carry = parts.pop();
        for (const line of parts) {
          if (!line.trim()) continue;
          const t = parseLogTs(line, yearHint);
          if (t != null) lastTs = t;
          ring.push(line.length > 400 ? line.slice(0, 400) : line);
          if (ring.length > 25) ring.shift();
          if (lastTs == null) continue;
          // (a) 签名事件（SIGNAL_TERM 不算崩溃事件，只作 analyzeCause 的停机特征）
          for (const sg of SIGS) {
            if (sg.code === 'SIGNAL_TERM') { if (sg.re.test(line)) break; continue; }
            if (!sg.re.test(line)) continue;
            if (lastTs - prevSigTs >= 120000) {
              prevSigTs = lastTs;
              events.push({ port: hint ? hint.port : null, model: hint ? hint.model : null, ts: lastTs, kind: 'log',
                causeCode: sg.code, cause: sg.label, evidence: ring.slice(-15).map(l => clip(l)), logFile: fp, note: null });
              sigCount++;
            }
            break;
          }
          // (b) 落入看门狗事件窗口的错误行进缓冲（窗口 = 事件前 CAUSE_WINDOW_MS ~ 事件后 60s）
          if (/ERROR|Traceback|Error|Fatal|CRITICAL|died|Killed|signal|CUDA|NCCL|[Aa]ssert|memory|Timeout/i.test(line)) {
            // 收集窗口与 analyzeCause 对齐：ev.ts ∈ [lastTs-60s, lastTs+CAUSE_WINDOW] ⇔ lastTs ∈ [ev.ts-CW, ev.ts+60s]
            const i0 = wdLo(lastTs - 60000), n = wdSorted.length;
            for (let i = i0; i < n; i++) {
              const ev = wdSorted[i];
              if (ev.ts > lastTs + CAUSE_WINDOW_MS) break;
              if (ev._buf.length < 300) ev._buf.push({ line: line.length > 400 ? line.slice(0, 400) : line, ts: lastTs });
            }
          }
        }
      }
    } finally { await fh.close(); }
  }

  // C. dmesg 里推理进程被 OOM Killer 杀（内核缓冲区保留期内）
  try {
    const r = await execCollect('dmesg', [], 8000);
    if (r) {
      const bootMs = Date.now() - parseFloat(fs.readFileSync('/proc/uptime', 'utf8').split(' ')[0]) * 1000;
      for (const line of r.stdout.split('\n')) {
        const m = /^\[\s*(\d+\.\d+)\].*Killed process (\d+) \(([^)]*)\)/.exec(line);
        if (!m) continue;
        if (!/VLLM|vllm|sglang|EngineCore|python/i.test(m[3])) continue;
        events.push({ port: null, ts: bootMs + parseFloat(m[1]) * 1000, kind: 'dmesg', causeCode: 'CPU_OOM',
          cause: '系统内存耗尽，被内核 OOM Killer 终止', evidence: [clip(line, 300)], logFile: 'dmesg', note: null });
        dmCount++;
      }
    }
  } catch (e) {}

  // D. 合并：按 ts 升序；端口相同或一侧未知（null）且 ±3 分钟 → 同一事件。
  //    取更早 ts（崩溃时刻先于看门狗确认时刻）；根因按具体度择优（PENDING < generic < specific）。
  events.sort((a, b) => a.ts - b.ts);
  const SPECIFIC = new Set(SIGS.filter(s => s.specific).map(s => s.code));
  SPECIFIC.add('CPU_OOM'); SPECIFIC.add('GPU_XID');
  const rank = (e) => e.causeCode === 'PENDING' ? 0 : (SPECIFIC.has(e.causeCode) ? 2 : 1);
  const merged = [];
  for (const ev of events) {
    let hit = null;
    for (let i = merged.length - 1; i >= 0 && ev.ts - merged[i].ts <= MERGE_WINDOW_MS; i--) {
      const r = merged[i];
      if (r.port != null && ev.port != null && r.port !== ev.port) continue;
      hit = r; break;
    }
    if (!hit) { merged.push(ev); continue; }
    if (hit.port == null && ev.port != null) hit.port = ev.port;
    if (!hit.model && ev.model) hit.model = ev.model;
    if (!hit.note && ev.note) hit.note = ev.note;
    if (ev.ts < hit.ts) { hit.ts = ev.ts; }
    if (rank(ev) > rank(hit)) {
      hit.causeCode = ev.causeCode; hit.cause = ev.cause; hit.evidence = ev.evidence; hit.logFile = ev.logFile;
    } else if ((!hit.evidence || !hit.evidence.length) && ev.evidence && ev.evidence.length) {
      hit.evidence = ev.evidence; if (!hit.logFile) hit.logFile = ev.logFile;
    }
  }

  // E. 仍是 PENDING 的看门狗事件 → 用窗口缓冲行分析根因
  for (const ev of merged) {
    if (ev.causeCode !== 'PENDING') continue;
    if (ev._buf && ev._buf.length) {
      const a = analyzeCause(ev._buf, ev.ts);
      if (a.causeCode === 'CLEAN_STOP') { ev.skip = true; continue; }
      ev.causeCode = a.causeCode; ev.cause = a.cause; ev.evidence = a.evidence;
    }
    if (ev.causeCode === 'PENDING') { ev.causeCode = 'UNKNOWN'; ev.cause = '进程离线被看门狗确认（当时的引擎日志未留存或无匹配签名：可能整机断电重启 / 被外部 kill / 日志已滚动清理）'; }
  }

  // F. 入库
  let added = 0;
  for (const ev of merged) {
    if (ev.skip || ev.causeCode === 'SIGNAL_TERM') continue;
    if (dedupPush({
      id: 'b' + ev.ts + ':' + (ev.port || 0), ts: ev.ts, tsText: fmtTs(ev.ts), port: ev.port,
      model: ev.model || null, runtime: 'vllm', pid: null,
      causeCode: ev.causeCode, cause: ev.cause, evidence: ev.evidence || [], logFile: ev.logFile || null,
      source: 'backfill', tsExact: true, note: ev.note || null,
    })) added++;
  }
  await saveRecords();
  selflog(`历史回填完成：watchdog 离线 ${wdCount} / 日志签名 ${sigCount} / dmesg OOM ${dmCount} → 合并入库 ${added} 条，耗时 ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}

// ---------- 入口 ----------
async function main() {
  state = { ...state, ...(await loadJson(STATE_FILE, {})) };
  state.startedAt = Date.now();
  records = (await loadJson(RECORDS_FILE, { records: [] })).records || [];
  selflog(`crash-watch 启动（pid=${process.pid}，base=${BASE}，已有记录 ${records.length} 条，backfill=${state.backfillDone ? 'done' : 'pending'}）`);
  await saveRecords(); // 先把 static 输出立起来（即使 records 为空）
  for (;;) {
    await tick();
    await sleep(TICK_MS);
  }
}
process.on('uncaughtException', (e) => selflog('uncaught: ' + (e && e.stack || e)));
process.on('unhandledRejection', (e) => selflog('unhandledRejection: ' + (e && e.stack || e)));
main().catch(e => selflog('main 崩溃: ' + (e && e.stack || e)));
