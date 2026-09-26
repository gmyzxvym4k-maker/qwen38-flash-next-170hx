#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rt-patch #9（KV 二级缓存移植）离线自检：不启引擎、不占显存。

用法（127 机）：
  PYTHONPATH=/home/ll/deploy/vllm-0300/patches:/home/ll/deploy/vllm-0300/patches-extra \
    /home/ll/vllm-env/bin/python selftest_kvoff_rt.py

判据：全部 [OK]、exit 0。任一 [FAIL] 即移植未生效/漂移。
"""
import os
import sys
import types
from collections import deque

FAIL = []


def check(name, cond, detail=""):
    tag = "OK " if cond else "FAIL"
    if not cond:
        FAIL.append(name)
    print(f"[{tag}] {name}" + (f"  <- {detail}" if detail and not cond else ""))


# 触发 vllm 的 offloading 模块导入 = 挂载钩子
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (  # noqa: E402
    config as off_cfg,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (  # noqa: E402
    common as off_common,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (  # noqa: E402
    worker as off_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (  # noqa: E402
    scheduler as off_sched,
)
from vllm.v1.kv_offload.cpu import gpu_worker as cpu_gw  # noqa: E402
from vllm.v1.kv_offload.cpu import spec as cpu_spec  # noqa: E402
from vllm.v1.kv_offload.cpu import swap_blocks_triton as sbt  # noqa: E402

# ---------------------------------------------------------------- c1
check("c1 get_offloading_group_ids 已包裹",
      getattr(off_cfg.get_offloading_group_ids, "_dsh_c1", False))


class _Spec:
    def __init__(self, pc):
        self.prefix_cacheable = pc


def _grp(pc, layers):
    return types.SimpleNamespace(kv_cache_spec=_Spec(pc), layer_names=layers)


fake_cfg = types.SimpleNamespace(
    hisparse_host_num_blocks=None,
    kv_cache_groups=[_grp(True, ["a"]), _grp(True, ["b"]), _grp(False, ["ring"]),
                     _grp(True, []), _grp(True, ["e"])],
)
kept = off_cfg.get_offloading_group_ids(fake_cfg)
check("c1 剔除不可缓存/空层分组，保留全局 id", kept == (0, 1, 4), str(kept))

# ---------------------------------------------------------------- c2
check("c2 _uses_shared_region 已包裹",
      getattr(cpu_spec.CPUOffloadingSpec._uses_shared_region, "_dsh_c2", False))
fake_spec_obj = types.SimpleNamespace(
    config=types.SimpleNamespace(parallel=types.SimpleNamespace(pp_size=2)))
check("c2 PP2 -> False（私有 pinned）",
      cpu_spec.CPUOffloadingSpec._uses_shared_region(fake_spec_obj) is False)
fake_spec_obj1 = types.SimpleNamespace(
    config=types.SimpleNamespace(parallel=types.SimpleNamespace(pp_size=1)))
r1 = cpu_spec.CPUOffloadingSpec._uses_shared_region(fake_spec_obj1)
check("c2 PP1 -> 原行为（True on CUDA）", isinstance(r1, bool), str(r1))

# ---------------------------------------------------------------- c7
check("c7 swap_blocks_triton.MIN_N=0", getattr(sbt, "MIN_N", None) == 0)
check("c7 _select_swap_blocks_fn 已包裹",
      getattr(cpu_gw._select_swap_blocks_fn, "_dsh_c7", False))
refs = [[types.SimpleNamespace(page_size_bytes=1616 * 2048)]]
f = cpu_gw._select_swap_blocks_fn(refs, gpu_to_cpu=True)
check("c7 store 方向也走 Triton（不再 ops.swap_blocks_batch）",
      getattr(f, "func", None) is sbt.swap_blocks_batch, str(f))

# ---------------------------------------------------------------- c6 metadata
M = off_common.OffloadingWorkerMetadata
m0 = M()
check("c6 meta.failed_jobs/fuse 字段可用",
      isinstance(m0.failed_jobs, dict) and m0.fuse_tripped is False)
m1 = M(completed_jobs={1: 1}, failed_jobs={2: 1}, fuse_tripped=True)
m2 = M(completed_jobs={3: 1}, failed_jobs={2: 1})
agg = m1.aggregate(m2)
check("c6 aggregate 合并 failed", agg.failed_jobs == {2: 2}, str(agg.failed_jobs))
check("c6 aggregate OR fuse", agg.fuse_tripped is True)
check("c6 实例隔离（m0 仍空）", m0.failed_jobs == {} and m0.fuse_tripped is False)

# ---------------------------------------------------------------- c6 handler wait 有界
H = cpu_gw.SingleDirectionOffloadingHandler
check("c6 handler 已包裹", getattr(H, "_dsh_c6", False))


class _NeverEvent:
    def query(self):
        return False


hs = types.SimpleNamespace(
    gpu_to_cpu=True, _dsh_stalled=False,
    _transfer_events={7: _NeverEvent()}, _transfers=deque([]),
)
os.environ["FN_KVOFF_WAIT_TIMEOUT"] = "0.2"
inc = H.wait(hs, {7})
check("c6 wait 超时返回未完成集合", inc == {7}, str(inc))
check("c6 超时后 store 方向熔断", hs._dsh_stalled is True)
check("c6 熔断后 wait 直接返回全部", H.wait(hs, set()) == {7})
os.environ.pop("FN_KVOFF_WAIT_TIMEOUT")

# ---------------------------------------------------------------- c6 connector worker
C = off_worker.OffloadingConnectorWorker
check("c6 worker 方法已替换",
      all(getattr(C, "_dsh_c6", False) for _ in [0])
      and getattr(C.handle_preemptions, "__name__", "") == "handle_preemptions")


class _FailWorker:
    def submit_store(self, *a):
        return False

    def submit_load(self, *a):
        return True

    def wait(self, job_ids):
        return set(job_ids)

    def get_finished(self):
        return []


class _Meta(types.SimpleNamespace):
    pass


cw = C.__new__(C)  # 手工装配（不跑真 __init__，避免 spec 依赖）
cw.worker = _FailWorker()
cw._is_store_writer = True
cw._load_jobs = {}
cw._unsubmitted_store_jobs = []
cw._connector_worker_meta = M()
cw._dsh_failed_jobs = set()
cw._dsh_store_fused = False
meta_in = _Meta(
    jobs_to_flush={1},
    store_jobs={},
    load_jobs={},
)
cw._unsubmitted_store_jobs.append((1, types.SimpleNamespace(), types.SimpleNamespace()))
C.handle_preemptions(cw, meta_in)
check("c6 submit 失败 -> failed ack", cw._connector_worker_meta.failed_jobs.get(1) == 1,
      str(cw._connector_worker_meta.failed_jobs))
check("c6 wait 超时 -> fuse", cw._dsh_store_fused and cw._connector_worker_meta.fuse_tripped)
wm = C.build_connector_worker_meta(cw)
check("c6 build_meta 失败也上报", wm is not None and wm.failed_jobs.get(1) == 1)
# 熔断后 prepare_store_kv：writer 失败 ack / 非 writer completed
cw2 = C.__new__(C)
cw2.worker = _FailWorker()
cw2._is_store_writer = True
cw2._load_jobs = {}
cw2._unsubmitted_store_jobs = []
cw2._connector_worker_meta = M()
cw2._dsh_failed_jobs = set()
cw2._dsh_store_fused = True
C.prepare_store_kv(cw2, _Meta(store_jobs={5: None}, load_jobs={}))
check("c6 熔断后新 store 直接失败 ack", 5 in cw2._connector_worker_meta.failed_jobs)

# ---------------------------------------------------------------- c6 scheduler
S = off_sched.OffloadingConnectorScheduler
check("c6 scheduler update_connector_output 已替换", getattr(S, "_dsh_c6", False))


class _Mgr:
    def __init__(self):
        self.calls = []

    def complete_store(self, keys, ctx, success=True):
        self.calls.append(("store", tuple(keys), success))


def _mkjob(pending, is_store=True):
    return types.SimpleNamespace(
        pending_count=pending, is_store=is_store, keys={9}, req_id="r",
        fenced_block_ids=None, deferred_fence_block_ids=None)


fs = types.SimpleNamespace()
fs._jobs = {11: _mkjob(1)}  # 单 worker 结算形态：一次失败 ack 即到 0
fs._req_status = {"r": types.SimpleNamespace(
    req_context=None, transfer_jobs={11}, finished_signaled=False,
    req=types.SimpleNamespace(is_finished=lambda: False))}
fs._stale_job_threshold = 0
fs._chunks_being_loaded = set()
fs._block_id_to_pending_jobs = {}
fs._connector_stats = types.SimpleNamespace(
    is_empty=lambda: True, aggregate=lambda x: None)
fs.manager = _Mgr()
fs._remove_pending_job = lambda j, b: None
out = types.SimpleNamespace(kv_connector_worker_meta=M(
    completed_jobs={}, failed_jobs={11: 1}, fuse_tripped=True))
S.update_connector_output(fs, out)
check("c6 失败 job 到 0 -> complete_store(success=False)",
      fs.manager.calls == [("store", (9,), False)], str(fs.manager.calls))
check("c6 失败后 _jobs 清理", not fs._jobs)
check("c6 fuse -> 停建 store", fs._dsh_stores_disabled is True)

# 混合场景：rankA 失败 + rankB 成功（不同步到达）
fs2 = types.SimpleNamespace()
fs2._jobs = {12: _mkjob(2)}
fs2._req_status = {"r": types.SimpleNamespace(
    req_context=None, transfer_jobs={12}, finished_signaled=False,
    req=types.SimpleNamespace(is_finished=lambda: False))}
fs2._stale_job_threshold = 0
fs2._chunks_being_loaded = set()
fs2._block_id_to_pending_jobs = {}
fs2._connector_stats = types.SimpleNamespace(is_empty=lambda: True, aggregate=lambda x: None)
fs2.manager = _Mgr()
fs2._remove_pending_job = lambda j, b: None
S.update_connector_output(fs2, types.SimpleNamespace(
    kv_connector_worker_meta=M(completed_jobs={}, failed_jobs={12: 1})))
check("c6 先失败计数（未到 0）不结算", fs2.manager.calls == [] and 12 in fs2._jobs)
S.update_connector_output(fs2, types.SimpleNamespace(
    kv_connector_worker_meta=M(completed_jobs={12: 1})))
check("c6 后到 completed 仍按失败结算（一 rank 缺数据）",
      fs2.manager.calls == [("store", (9,), False)], str(fs2.manager.calls))

# happy path 不受影响
fs3 = types.SimpleNamespace()
fs3._jobs = {13: _mkjob(2)}
fs3._req_status = {"r": types.SimpleNamespace(
    req_context=None, transfer_jobs={13}, finished_signaled=False,
    req=types.SimpleNamespace(is_finished=lambda: False))}
fs3._stale_job_threshold = 0
fs3._chunks_being_loaded = set()
fs3._block_id_to_pending_jobs = {}
fs3._connector_stats = types.SimpleNamespace(is_empty=lambda: True, aggregate=lambda x: None)
fs3.manager = _Mgr()
fs3._remove_pending_job = lambda j, b: None
S.update_connector_output(fs3, types.SimpleNamespace(
    kv_connector_worker_meta=M(completed_jobs={13: 2})))
check("c6 正常完成 -> success=True 登记",
      fs3.manager.calls == [("store", (9,), True)], str(fs3.manager.calls))

# ---------------------------------------------------------------- 汇总
print()
if FAIL:
    print(f"SELFTEST FAIL: {len(FAIL)} 项: {FAIL}")
    sys.exit(1)
print("SELFTEST PASS：rt-patch #9 (c1/c2/c6/c7) 全部就位")
