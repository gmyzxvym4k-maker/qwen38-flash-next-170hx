#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vLLM 管理页对应修复 + 新增「vLLM」标签页（2026-10-03）

A. server.js（vLLM 实例归因修复，对称 10-03 给 SGLang 加的兜底）
   A1 findVllmPidByPort：加 listVllmInstances 兜底（root 启动 + `vllm serve` CLI 形式探测不到）
   A2 model-manager 实例发现：加 vLLM 兜底，回填 pid/gpu/gpus/runtime
   A3 新增 /vllm.html 静态伺服 handler（仿 /sglang.html，带 ETag）
   A4 gzip 白名单加 /vllm.html
B. index.html
   B1 nav-tab 新增「vLLM」
   B2 tab 内容块（iframe 内嵌 /vllm.html）
   B3 switchTab 映射加 vllm:9 + 懒加载调用
   B4 initVllmTab / reloadVllmFrame（新函数名不与既有冲突）
   B5 PAGE_VERSION 递增（r7 → r8）
幂等：重复执行会跳过已应用的锚点。
"""
import os
import re
import shutil
import sys
import time

DEPLOY = "/home/ll/deploy"
SERVER = os.path.join(DEPLOY, "server.js")
INDEX = os.path.join(DEPLOY, "index.html")

MARK = "vllm-page-1003"


def backup(path):
    b = "%s.bak-%s-%s" % (path, MARK, time.strftime("%m%d-%H%M%S"))
    shutil.copy2(path, b)
    return b


def apply_edits(path, edits, required_edit_names=None):
    """edits: list of (name, old, new)。返回 True 表示有改动。"""
    required_edit_names = required_edit_names or set()
    src = open(path, encoding="utf-8").read()
    orig = src
    report = []
    for name, old, new in edits:
        if new in src and old not in src:
            report.append("%-46s SKIP(已是新版)" % name)
            continue
        cnt = src.count(old)
        if cnt == 0:
            if name in required_edit_names:
                print("FAIL[%s]: 必需锚点未找到（文件可能被回滚/改动）" % name)
                return None
            report.append("%-46s SKIP(锚点不存在)" % name)
            continue
        if cnt > 1:
            print("FAIL[%s]: 锚点不唯一（%d 处），拒绝改" % (name, cnt))
            return None
        src = src.replace(old, new, 1)
        report.append("%-46s OK" % name)
    if src == orig:
        print("[%s] 无改动（全部已应用）" % os.path.basename(path))
        for r in report:
            print("   " + r)
        return False
    b = backup(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("[%s] 已写入；备份 %s" % (os.path.basename(path), b))
    for r in report:
        print("   " + r)
    return True


# ============ server.js ============
S1_FINDPID_OLD = """    try {
      const sgi = listSglangInstances().find(x => x.port === port);
      if (sgi && sgi.pid) return sgi.pid;
    } catch (e) {}
    const pg = execSync('pgrep -f "[v]llm.entrypoints" 2>/dev/null || true', { encoding: 'utf8', timeout: 3000 }).trim();"""

S1_FINDPID_NEW = """    try {
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
    const pg = execSync('pgrep -f "[v]llm.entrypoints|[v]llm serve" 2>/dev/null || true', { encoding: 'utf8', timeout: 3000 }).trim();"""

S2_MM_OLD = """            const pid = findVllmPidByPort(p);
            let gpu = gpuIndexForPid(pid);
            let rt = detectRuntimeForPid(pid);
            // [sglang-adapt-1003] root 起的 sglang：/proc/<pid>/environ 读不到（gpu=null）、
            // 或 pid 兜底未命中（runtime=unknown）→ 用 /proc cmdline 扫描的实例表补归因。
            if (!rt || gpu == null) {
              try {
                const sgi = listSglangInstances().find(x => x.port === p);
                if (sgi) {
                  rt = rt || 'sglang';
                  if (gpu == null) gpu = sgi.gpu != null ? String(sgi.gpu) : null;
                }
              } catch (e) {}
            }
            instances.push({
              port: p,
              model: j.data[0].id,
              pid,
              gpu,
              runtime: rt || 'unknown',
              running: true,
            });"""

S2_MM_NEW = """            let pid = findVllmPidByPort(p);
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
            });"""

S3_SERVE_OLD = """      res.end('sglang.html not found');
    }
    return;
  }

  // === Serve static files ==="""

S3_SERVE_NEW = """      res.end('sglang.html not found');
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

  // === Serve static files ==="""

S4_GZIP_OLD = "|| pathname === '/sglang.html'\n      || pathname.startsWith('/static/')"
S4_GZIP_NEW = "|| pathname === '/sglang.html' || pathname === '/vllm.html'\n      || pathname.startsWith('/static/')"

SERVER_EDITS = [
    ("A1 findVllmPidByPort vLLM 兜底", S1_FINDPID_OLD, S1_FINDPID_NEW),
    ("A2 model-manager 实例归因兜底", S2_MM_OLD, S2_MM_NEW),
    ("A3 /vllm.html 静态伺服 handler", S3_SERVE_OLD, S3_SERVE_NEW),
    ("A4 gzip 白名单", S4_GZIP_OLD, S4_GZIP_NEW),
]

# ============ index.html ============
I1_NAV_OLD = """            <div class="nav-tab" onclick="switchTab('sglang')">SGLang</div>"""
I1_NAV_NEW = """            <div class="nav-tab" onclick="switchTab('sglang')">SGLang</div>
            <div class="nav-tab" onclick="switchTab('vllm')">vLLM</div>"""

I2_TAB_OLD = """        <!-- ==================== Strata 引擎管理（后端见 server.js STRATA 模块 · 契约 STRATA-API.md） ==================== -->"""
I2_TAB_NEW = """        <!-- ==================== vLLM 引擎监控（[vllm-page-1003] 内嵌 /vllm.html，纯只读） ==================== -->
        <div id="tab-vllm" class="tab-content hidden">
            <div class="dash-header">
                <div class="dash-title">vLLM</div>
                <div class="dash-meta">
                    <span style="font-size:12px;color:var(--text-tertiary)">vLLM 实例专属监控 · MTP 逐位接受率 / KV 池与 block 配置 / 前缀缓存 / CPU 二级缓存 / PP 双卡 · 每 2s 自动刷新</span>
                    <button class="btn" onclick="reloadVllmFrame()">重载页面</button>
                </div>
            </div>
            <iframe id="vllmFrame" title="vLLM Monitor" style="width:100%;height:calc(100vh - 200px);min-height:640px;border:1px solid var(--border);border-radius:14px;background:var(--bg-card)"></iframe>
        </div>

        <!-- ==================== Strata 引擎管理（后端见 server.js STRATA 模块 · 契约 STRATA-API.md） ==================== -->"""

I3_MAP_OLD = """    const map = { dashboard: 0, models: 1, energy: 2, storage: 3, usage: 4, bench: 5, cpuctl: 6, strata: 7, sglang: 8 };"""
I3_MAP_NEW = """    const map = { dashboard: 0, models: 1, energy: 2, storage: 3, usage: 4, bench: 5, cpuctl: 6, strata: 7, sglang: 8, vllm: 9 };"""

I4_INIT_OLD = """    if (tab === 'sglang') initSglangTab();"""
I4_INIT_NEW = """    if (tab === 'sglang') initSglangTab();
    if (tab === 'vllm') initVllmTab();"""

I5_FUNC_OLD = """function reloadSglangFrame() {
    const f = document.getElementById('sglangFrame');
    if (f) f.setAttribute('src', '/sglang.html?r=' + Date.now());
}"""
I5_FUNC_NEW = """function reloadSglangFrame() {
    const f = document.getElementById('sglangFrame');
    if (f) f.setAttribute('src', '/sglang.html?r=' + Date.now());
}

// ====== vLLM 引擎监控标签（[vllm-page-1003] 内嵌 /vllm.html，后端 metrics?port= 按实例取数） ======
// 懒加载口径同 bench/cpu/sglang：首次激活才拉页；页内 2s 轮询自带后台门控与看门狗，切标签不重复起。
// 命名带 Vllm 域前缀，避免 index.html 同名函数静默覆盖（09-15 CPU 卡事故铁律）。
function initVllmTab() {
    const f = document.getElementById('vllmFrame');
    if (f && !f.getAttribute('src')) f.setAttribute('src', '/vllm.html');
}
function reloadVllmFrame() {
    const f = document.getElementById('vllmFrame');
    if (f) f.setAttribute('src', '/vllm.html?r=' + Date.now());
}"""

INDEX_EDITS = [
    ("B1 nav-tab 新增 vLLM", I1_NAV_OLD, I1_NAV_NEW),
    ("B2 tab 内容块(iframe)", I2_TAB_OLD, I2_TAB_NEW),
    ("B3 switchTab 索引映射", I3_MAP_OLD, I3_MAP_NEW),
    ("B3b switchTab 懒加载调用", I4_INIT_OLD, I4_INIT_NEW),
    ("B4 initVllmTab/reloadVllmFrame", I5_FUNC_OLD, I5_FUNC_NEW),
]


def bump_page_version():
    src = open(INDEX, encoding="utf-8").read()
    m = re.search(r"const PAGE_VERSION = '([^']*)';", src)
    if not m:
        print("   B5 PAGE_VERSION: SKIP(未找到)")
        return False
    if "-r8" in m.group(1):
        print("   B5 PAGE_VERSION: SKIP(已是 r8)")
        return False
    new = "const PAGE_VERSION = '20261006-r8';"
    src = src[:m.start()] + new + src[m.end():]
    with open(INDEX, "w", encoding="utf-8") as f:
        f.write(src)
    print("   B5 PAGE_VERSION: OK（%s → r8）" % m.group(1))
    return True


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--apply"
    if mode == "--check":
        s = open(SERVER, encoding="utf-8").read()
        i = open(INDEX, encoding="utf-8").read()
        print("server.js: A1=%s A2=%s A3=%s A4=%s" % tuple(
            ("done" if new in s else "todo") for _, _, new in SERVER_EDITS))
        print("index.html: B1=%s B2=%s B3=%s B3b=%s B4=%s" % tuple(
            ("done" if new in i else "todo") for _, _, new in INDEX_EDITS))
        return
    for p in (SERVER, INDEX):
        if not os.path.exists(p):
            print("FAIL: %s 不存在" % p)
            sys.exit(2)
    r1 = apply_edits(SERVER, SERVER_EDITS, required_edit_names={"A1 findVllmPidByPort vLLM 兜底",
                                                               "A2 model-manager 实例归因兜底",
                                                               "A3 /vllm.html 静态伺服 handler"})
    if r1 is None:
        sys.exit(3)
    r2 = apply_edits(INDEX, INDEX_EDITS, required_edit_names={"B2 tab 内容块(iframe)",
                                                             "B4 initVllmTab/reloadVllmFrame"})
    if r2 is None:
        sys.exit(4)
    if r2:
        bump_page_version()
    print("DONE")


main()
