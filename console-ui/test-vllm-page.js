#!/usr/bin/env node
// vllm.html 渲染实测：DOM stub + 真实端点数据，跑页面自身脚本，检查卡片是否真出数。
// 口径同项目「vm 沙箱测前端函数」方法论：不靠肉眼看页面，用真实 /metrics 文本驱动渲染。
const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync('/home/ll/deploy/vllm.html', 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) { console.log('FAIL: 未找到内联脚本'); process.exit(2); }
const script = m[1];

const els = {};
function mkEl(id) {
  return {
    id: id, innerHTML: '', textContent: '',
    _attrs: {},
    setAttribute(k, v) { this._attrs[k] = v; },
    getAttribute(k) { return this._attrs[k] === undefined ? null : this._attrs[k]; },
  };
}
const sandbox = {
  console: console,
  document: {
    hidden: false,
    getElementById(id) { return els[id] || (els[id] = mkEl(id)); },
    addEventListener() {},
  },
  setTimeout: () => 0,        // 阻断轮询续跑，只留首帧
  clearTimeout: () => {},
  setInterval: () => 0,
  Date: Date,
  Math: Math,
  Promise: Promise,
  AbortController: AbortController,
  JSON: JSON,
  Number: Number,
  String: String,
  parseFloat: parseFloat,
  fetch: async (url) => {
    const full = url.startsWith('http') ? url : 'http://127.0.0.1:8889' + url;
    const r = await fetch(full);
    const text = await r.text();
    return {
      ok: r.ok, status: r.status,
      text: async () => text,
      json: async () => JSON.parse(text),
    };
  },
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(script, sandbox, { filename: 'vllm-page.js' });

function strip(h) { return String(h).replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim(); }

setTimeout(() => {
  const body = els['body'] ? els['body'].innerHTML : '';
  const bar = els['instBar'] ? els['instBar'].innerHTML : '';
  const notes = els['notes'] ? els['notes'].innerHTML : '';
  const stamp = els['stamp'] ? els['stamp'].textContent : '';
  const txt = strip(body);

  const checks = [
    ['实例徽标含 vllm :18420', /vllm\s*:18420/.test(strip(bar))],
    ['实例徽标含 GPU0+1', /GPU0\+1/.test(strip(bar))],
    ['非空态（不是"没有运行中的 vLLM 实例"）', body.length > 500 && !/当前没有运行中/.test(txt)],
    ['含 KV 池卡', /KV 池/.test(txt)],
    ['KV 池容量取自 cache_config_info', /1,118,584/.test(txt)],
    ['含 block 1616', /1,616/.test(txt)],
    ['含 MTP 投机卡', /MTP 投机解码/.test(txt)],
    ['接受率已出数（%)', /接受率/.test(txt) && /\d+\.\d%/.test(txt)],
    ['逐位接受率条存在', /pos 0/.test(txt) && /pos 1/.test(txt)],
    ['含累计吞吐卡', /累计吞吐/.test(txt)],
    ['含运行时参数卡（PP2/TP1）', /运行时参数/.test(txt)],
    ['并行显示 PP2', /PP2/.test(txt)],
    ['思考深度 xhigh', /xhigh/.test(txt)],
    ['含二级缓存卡', /CPU KV 二级缓存/.test(txt)],
    ['时间戳已更新', /更新于/.test(stamp)],
    ['无实例发现异常告警', !/实例发现接口异常/.test(strip(notes))],
    ['无 unknown 归因告警', !/归因失败/.test(strip(notes))],
  ];
  let pass = 0;
  console.log('=== vLLM 页渲染实测（真实数据驱动）===');
  for (const [name, ok] of checks) {
    console.log((ok ? '  PASS  ' : '  FAIL  ') + name);
    if (ok) pass++;
  }
  console.log('\n通过 %d/%d', pass, checks.length);
  console.log('\n--- 渲染正文摘要（前 700 字）---');
  console.log(txt.slice(0, 700));
  console.log('\n--- 告警区 ---');
  console.log(strip(notes) || '(空)');
  process.exit(pass === checks.length ? 0 : 1);
}, 2500);
