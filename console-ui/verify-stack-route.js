#!/usr/bin/env node
// 栈路由验证：把 server.js 里真实的 sglangActive/stack0300Active/resolveStartScript/
// resolveStopScript 提取到隔离沙箱执行，确认控制台启动/停止实际会调哪个脚本。
// 只读：不改任何生产文件；哨兵对照用临时文件 + 常量注入模拟，不触碰真实哨兵。
const fs = require('fs');

const src = fs.readFileSync('/home/ll/deploy/server.js', 'utf8');
function grab(re, label) {
  const m = src.match(re);
  if (!m) throw new Error('提取失败: ' + label);
  return m[0];
}

const SGLANG_ACTIVE = '/home/ll/deploy/sglang-18420/ACTIVE';
const STACK_DISABLED = '/home/ll/deploy/vllm-0300/DISABLED';

// 提取真实实现，注入两个常量（等价于源码里的同名常量）
const code = [
  grab(/function sglangActive\(\)[\s\S]*?\n\}/, 'sglangActive'),
  grab(/function stack0300Active\(\)[\s\S]*?\n\}/, 'stack0300Active'),
  grab(/function resolveStartScript\(sm\)[\s\S]*?\n\}/, 'resolveStartScript'),
  grab(/function resolveStopScript\(sm\)[\s\S]*?\n\}/, 'resolveStopScript'),
].join('\n');

const api = new Function(
  'fs', 'SGLANG_ACTIVE', 'STACK0300_DISABLED',
  code + '\nreturn { sglangActive, stack0300Active, resolveStartScript, resolveStopScript };'
)(fs, SGLANG_ACTIVE, STACK_DISABLED);

const sm = {
  key: 'qwen3.8-flash-next-w4a16',
  script: '/home/ll/deploy/start-flash-next-w4a16.sh',
  stopScript: '/home/ll/deploy/stop-flash-next-w4a16.sh',
  scriptSglang: '/home/ll/deploy/sglang-18420/start-flash-next-sglang.sh',
  stopScriptSglang: '/home/ll/deploy/sglang-18420/stop-flash-next-sglang.sh',
  scriptNew: '/home/ll/deploy/vllm-0300/start-flash-next-0300.sh',
  stopScriptNew: '/home/ll/deploy/vllm-0300/stop-flash-next-0300.sh',
};

console.log('哨兵实测: SGLang ACTIVE=%s  0.30.0 DISABLED=%s',
  fs.existsSync(SGLANG_ACTIVE), fs.existsSync(STACK_DISABLED));
console.log('sglangActive()=%s  stack0300Active()=%s', api.sglangActive(), api.stack0300Active());
const start = api.resolveStartScript(sm);
const stop = api.resolveStopScript(sm);
console.log('resolveStartScript ->', start);
console.log('resolveStopScript  ->', stop);

function tag(p) {
  if (/sglang-18420/.test(p)) return 'SGLang栈';
  if (/vllm-0300/.test(p)) return 'vLLM0.30.0栈';
  if (/w4a16/.test(p)) return '旧chroot栈';
  return '?';
}
const ok = tag(start) === 'vLLM0.30.0栈' && tag(stop) === 'vLLM0.30.0栈';
console.log('\n启动脚本栈=%s  停止脚本栈=%s', tag(start), tag(stop));
console.log(ok ? 'ROUTE_OK：控制台启动/停止均指向官方 vLLM 0.30.0 栈'
               : 'ROUTE_WRONG：路由未回到 vLLM 栈');
process.exit(ok ? 0 : 1);
