// 校验 [stack-0310-1006] 后：控制台解析到的启动/停止脚本、以及预设生成的 FN_*
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync('/home/ll/deploy/server.js', 'utf8');

function sliceFn(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('未找到函数 ' + name);
  let depth = 0, k = src.indexOf('{', i), end = -1;
  for (; k < src.length; k++) {
    const c = src[k];
    if (c === '{') depth++;
    else if (c === '}') { depth--; if (depth === 0) { end = k + 1; break; } }
  }
  return src.slice(i, end);
}

const skey = "SCRIPT_MODELS['qwen3.8-flash-next-w4a16'] = {";
const i0 = src.indexOf(skey);
const i1 = src.indexOf('\n};', i0) + 3;

const ctx = { require, console, fs, JSON, Math, String, Number, parseInt, parseFloat, Object, Array, process };
vm.createContext(ctx);
vm.runInContext('var SCRIPT_MODELS = {};', ctx);
vm.runInContext(src.slice(i0, i1).replace("SCRIPT_MODELS['qwen3.8-flash-next-w4a16'] = {", "SCRIPT_MODELS['k'] = {"), ctx);
['sglangActive', 'stack0300Active', 'resolveStartScript', 'resolveStopScript', 'scriptModelDefaults', 'scriptModelLaunchPlan']
  .forEach(n => vm.runInContext(sliceFn(n), ctx));

const out = (label, v) => console.log(label.padEnd(34), '=', v);
out('新栈哨兵在位?', vm.runInContext('fs.existsSync("/home/ll/deploy/vllm-0310/DISABLED")', ctx));
out('stack0300Active()', vm.runInContext('stack0300Active()', ctx));
out('resolveStartScript', vm.runInContext('resolveStartScript(SCRIPT_MODELS.k)', ctx));
out('resolveStopScript', vm.runInContext('resolveStopScript(SCRIPT_MODELS.k)', ctx));
const d = vm.runInContext('scriptModelDefaults(SCRIPT_MODELS.k)', ctx);
console.log('\n弹窗默认（scriptModelDefaults）:');
['gpuCount', 'parallelMode', 'maxSeqs', 'blockSize', 'kvoff', 'kvoffGiB', 'pleInt8', 'pleLoc', 'temperature', 'presencePenalty', 'repetitionPenalty']
  .forEach(k => out('  ' + k, d[k]));

const presets = JSON.parse(fs.readFileSync('/home/ll/deploy/quickstart-presets.json', 'utf8'));
for (const p of presets.flashnext.presets) {
  const r = vm.runInContext('scriptModelLaunchPlan(SCRIPT_MODELS.k, ' + JSON.stringify(p.params) + ')', ctx);
  const e = r.env;
  console.log('\n预设 ' + p.key + ' → ' + p.name);
  ['FN_MODEL_PATH', 'FN_1M_MODEL_PATH', 'FN_PP', 'FN_MAXLEN', 'FN_BLOCK', 'FN_SEQS', 'FN_GPUMEM',
    'FN_SPEC', 'FN_SIMPLE_OFFLOAD', 'FN_KVOFF', 'FN_PLE_INT8', 'FN_PLE_LOC', 'FN_PLE_INT8_DIR', 'FN_GENCFG']
    .forEach(k => out('  ' + k, e[k] === undefined ? '(未下发)' : String(e[k]).slice(0, 78)));
}
