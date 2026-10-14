"""dsh_stream_rt —— rt-patch #11：前端每步每请求输出真值流（2026-09-28）。

背景（并发卡「实时输出 tok/s 不准」的重建基础）：
官方 vLLM 0.30.0 的 /metrics 没有任何每请求实时数据（request_* 族都是
完成后才进桶的直方图）；旧栈的 scheduler_stats.prefill_progress 是自研镜像
补丁字段，官方栈不存在（getattr 静默 None）。控制台因此只能拿实例级聚合 +
直方图驻留去估计每个请求的速度——各行同值、滞后数秒，这就是不准确的根。

本补丁一行不改上游文件：只包 IterationStats.update_from_output——官方前端
进程（output_processor.process_outputs，v1/engine/output_processor.py:856
位置传参）对**每个产出 token 的 EngineCoreOutput 每步都会调它**，参数里
现成就有全部真值：

  * ``output.request_id``  引擎真实 rid（SSE 响应 id 同串，内部重试时带
    ``-xxxxxxxx`` 尾缀——控制台 ridFind 前缀容忍）
  * ``req_stats.num_generation_tokens``  该请求累计输出 token（官方在本函数
    里逐 token 累加，MTP 一步多 token 也如实计入）
  * ``req_stats.arrival_time``  请求进引擎的墙钟时刻（epoch 秒）
  * ``is_prefilling``  True=首 token 那一步（天然 TTFT 标记）

包装只把这几个已在手里的值搭车挂到 iteration_stats 实例的 ``_dsh_reqs``
字典（rid -> [gen_cum, arrival, is_first_step, new_tokens_sum]），由
dsh_vllm_logger 插件在 record() 里逐步落盘 vllm-live-stream.jsonl，控制台
tail 后按 rid 滑窗差分 → 每请求速度 = 引擎自报真值，零猜测零估计。

开销：每 output 一次 dict 赋值；插件不装载时无人消费、零写盘。
紧急总闸：DSH_STREAM_RT_DISABLE=1 ⇒ 不挂钩。
PP2/async-scheduling 注意：outputs 由最后一级 stage 回传、前端统一消费，
包装点在 API server 主进程内，与 worker 进程无关；时间戳取
iteration_timestamp（前端 time.time()），与 arrival_time 同域。
"""
from __future__ import annotations

import os
import sys


def _log(msg: str) -> None:
    sys.stderr.write(f"[rt-patch-stream] {msg}\n")
    sys.stderr.flush()


def _patch_stats(module) -> None:
    if os.environ.get("DSH_STREAM_RT_DISABLE", "") == "1":
        _log("DSH_STREAM_RT_DISABLE=1 ⇒ #11 不动作")
        return
    cls = getattr(module, "IterationStats", None)
    if cls is None or not hasattr(cls, "update_from_output"):
        _log("没找到 IterationStats.update_from_output ⇒ #11 跳过（上游结构变了）")
        return
    if getattr(cls, "_dsh_stream_patched", False):
        return

    orig = cls.update_from_output

    def update_from_output(
        self, output, engine_core_timestamp, is_prefilling,
        req_stats, lora_states, lora_name,
    ):
        ret = orig(
            self, output, engine_core_timestamp, is_prefilling,
            req_stats, lora_states, lora_name,
        )
        # 搭车记录：任何异常都不能影响官方链路
        try:
            d = getattr(self, "_dsh_reqs", None)
            if d is None:
                d = self._dsh_reqs = {}
            rid = output.request_id
            ng = len(output.new_token_ids) if output.new_token_ids else 0
            g = int(req_stats.num_generation_tokens)
            ent = d.get(rid)
            if ent is None:
                # [累计gen, arrival墙钟, 首token步标记, 本步新token合计]
                d[rid] = [g, float(req_stats.arrival_time or 0.0), 1 if is_prefilling else 0, ng]
            else:
                ent[0] = g
                if is_prefilling:
                    ent[2] = 1
                if ng:
                    ent[3] += ng
        except Exception:
            pass
        return ret

    cls.update_from_output = update_from_output
    cls._dsh_stream_patched = True
    _log("IterationStats.update_from_output 已搭车每请求输出真值（rt-patch #11）")


PATCHES = {
    "vllm.v1.metrics.stats": _patch_stats,
}
