#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rt-patch #9（KV 二级缓存移植）离线自检：不启引擎、不占显存。
覆盖 c1/c2/c6/c7 与 c8（PP>1 公共区布局协商 + 前缀和偏移视图）。

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
# 2026-09-27 修正：只剔除 prefix_cacheable=False 的组；**空层分组必须保留**
# （PP>1 时各 rank 只带自己的层名，空组是合法占位；剔除会让 worker 的
#  group 数与 scheduler 不一致 → store 断言/越界）。
check("c1 只剔除不可缓存分组、保留空层分组与全局 id",
      kept == (0, 1, 3, 4), str(kept))

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
f_store = cpu_gw._select_swap_blocks_fn(refs, gpu_to_cpu=True)
check("c7 store 方向回落到上游 C++ DMA（Triton 会 MMU fault）",
      f_store is cpu_gw.ops.swap_blocks_batch, str(f_store))
f_load = cpu_gw._select_swap_blocks_fn(refs, gpu_to_cpu=False)
check("c7 load 方向仍强制 Triton swap 内核",
      getattr(f_load, "func", None) is sbt.swap_blocks_batch, str(f_load))

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

# ---------------------------------------------------------------- c8：公共区
import dsh_kvoff_rt as kvrt

check("c8 spec.SharedOffloadRegion -> DshPpSharedRegion",
      getattr(cpu_spec.SharedOffloadRegion, "__name__", "") == "DshPpSharedRegion")
check("c8 create_worker 已包裹",
      bool(getattr(cpu_spec.CPUOffloadingSpec.create_worker, "_dsh_c8", False)))
check("c8 get_manager 已包裹",
      bool(getattr(cpu_spec.CPUOffloadingSpec.get_manager, "_dsh_c8", False)))
check("c8 _uses_shared_region 认 _dsh_c8",
      bool(getattr(cpu_spec.CPUOffloadingSpec._uses_shared_region, "_dsh_c2", False)))

_PAR = dict(tp_size=1, pp_size=2, world_size=2, rank=0, data_parallel_size=1)


def _mk_spec(eid, cpu_bytes, own, rank=0, pp=2, tp=1, blocks_per_chunk=1,
             tpb=1616, replicated=False, canonical=False, dp=1):
    par = types.SimpleNamespace(**dict(_PAR, pp_size=pp, tp_size=tp, rank=rank,
                                       world_size=pp * tp * dp,
                                       data_parallel_size=dp))
    cfg = types.SimpleNamespace(parallel=par, engine_id=eid,
                                canonical_layout=canonical)
    sp = types.SimpleNamespace(config=cfg, extra_config={"cpu_bytes_to_use": cpu_bytes},
                               cpu_page_size_per_worker=own,
                               blocks_per_chunk=blocks_per_chunk,
                               replicated_layout=replicated,
                               tokens_per_block=(tpb,),
                               kv_bytes_per_chunk=own, num_chunks=0)
    return sp


def _pub(eid, rank, slot, world=2, decision=None, row=0, nch=0, rows=None):
    tag = kvrt._engine_tag(eid)
    payload = {"engine_id": eid, "world_size": world, "rank": rank,
               "slot_bytes": slot, "blocks_per_chunk": 1,
               "pid": os.getpid(), "starttime": kvrt._proc_start_ticks(os.getpid())}
    if decision:
        payload.update(decision=decision, row_stride=row, num_chunks=nch,
                       rows=slot and (rows if rows is not None else nch))
    return kvrt._publish(tag, rank, payload)


os.environ["FN_KVOFF_LAYOUT_TIMEOUT"] = "1"
os.environ["FN_KVOFF_LAYOUT_WINDOW"] = "900"

# --- 布局数学：rank0(25.5MB) + rank1(21.6MB) 的紧凑前缀和 ---
EID = "selftest-c8-%d" % os.getpid()
S0, S1 = 25_500_000, 21_600_000
CPU = 200 * 1024 * 1024  # 200 MiB（必须 > row_stride≈47 MB，否则 nchunks=0 正确地退回私有）
_pub(EID, 0, S0)
_pub(EID, 1, S1)
sp0 = _mk_spec(EID, CPU, S0, rank=0)
ok0 = kvrt._c8_resolve(sp0, publish=True, own_required=S0)
row = kvrt._round_up(S0) + kvrt._round_up(S1)
exp_n = CPU // row
check("c8 worker rank0 协商成功", ok0 is True)
check("c8 row_stride = Σ align(slot_i)", sp0.kv_bytes_per_chunk == row,
      "%d vs %d" % (sp0.kv_bytes_per_chunk, row))
check("c8 num_chunks = cpu_bytes // row_stride", sp0.num_chunks == exp_n,
      "%d vs %d" % (sp0.num_chunks, exp_n))
check("c8 rank0 偏移=0", sp0._dsh_c8["offset"] == 0)
check("c8 rank0 slot 已对齐", sp0._dsh_c8["slot"] == kvrt._round_up(S0))

sp1 = _mk_spec(EID, CPU, S1, rank=1)
ok1 = kvrt._c8_resolve(sp1, publish=True, own_required=S1)
check("c8 worker rank1 协商成功", ok1 is True)
check("c8 rank1 偏移 = align(slot0)（前缀和，区段不相交）",
      sp1._dsh_c8["offset"] == kvrt._round_up(S0))
check("c8 两侧 num_chunks 一致", sp1.num_chunks == sp0.num_chunks)
check("c8 区段不越行尾",
      sp1._dsh_c8["offset"] + sp1._dsh_c8["slot"] <= row)

# 调度器侧采纳同一口径（这是防越界的关键）
sps = _mk_spec(EID, CPU, S0, rank=0)
ok_s = kvrt._c8_resolve(sps, publish=False)
check("c8 调度器侧采纳公共区", ok_s is True)
check("c8 调度器 num_chunks == worker num_chunks", sps.num_chunks == sp0.num_chunks,
      "%d vs %d" % (sps.num_chunks, sp0.num_chunks))

# --- 门控：不适用的拓扑一律退回私有 ---
check("c8 pp=1 不启用", kvrt._c8_resolve(
    _mk_spec(EID + ".pp1", CPU, S0, pp=1), publish=True) is False)
check("c8 tp>1 不启用", kvrt._c8_resolve(
    _mk_spec(EID + ".tp", CPU, S0, tp=2), publish=True) is False)
check("c8 canonical_layout 不启用", kvrt._c8_resolve(
    _mk_spec(EID + ".cn", CPU, S0, canonical=True), publish=True) is False)
os.environ["FN_KVOFF_SHARED"] = "0"
check("c8 FN_KVOFF_SHARED=0 不启用", kvrt._c8_resolve(
    _mk_spec(EID, CPU, S0), publish=True) is False)
os.environ.pop("FN_KVOFF_SHARED")

# --- 容量不足一个 chunk ⇒ 退回私有 ---
tiny = _mk_spec(EID + ".tiny", 10 * 1024 * 1024, S0)
check("c8 cpu_bytes < row_stride -> 私有",
      kvrt._c8_resolve(tiny, publish=True, own_required=S0) is False)

# --- /dev/shm 不够 ⇒ 退回私有（公共区是 tmpfs 文件） ---
_orig_statvfs = kvrt._shm_free_bytes
kvrt._shm_free_bytes = lambda path="/dev/shm": 1024
nosm = _mk_spec(EID + ".shm", CPU, S0)
check("c8 /dev/shm 不足 -> 私有",
      kvrt._c8_resolve(nosm, publish=True, own_required=S0) is False)
kvrt._shm_free_bytes = _orig_statvfs
# --- 本 rank 实际需求超过自己那格 ⇒ 退回私有 ---
big = _mk_spec(EID + ".big", CPU, S0)
check("c8 own_required > slot -> 私有",
      kvrt._c8_resolve(big, publish=True, own_required=10 * S0) is False)

# --- 对端缺席（超时）⇒ 私有 ---
os.environ["FN_KVOFF_LAYOUT_TIMEOUT"] = "0.5"
lonely = _mk_spec("selftest-c8-lonely-%d" % os.getpid(), CPU, S0, rank=0)
check("c8 对端缺席超时 -> 私有",
      kvrt._c8_resolve(lonely, publish=True, own_required=S0) is False)
os.environ["FN_KVOFF_LAYOUT_TIMEOUT"] = "1"

# --- 陈旧发布（写者进程已死）必须被忽略 ---
EID2 = "selftest-c8-stale-%d" % os.getpid()
tag2 = kvrt._engine_tag(EID2)
dead = {"engine_id": EID2, "world_size": 2, "rank": 1, "slot_bytes": S1,
        "blocks_per_chunk": 1, "pid": 2_000_000_000 % 4_194_304,
        "starttime": 1}
kvrt._publish(tag2, 1, dead)
_pub(EID2, 0, S0)
os.environ["FN_KVOFF_LAYOUT_TIMEOUT"] = "0.5"
sp_dead = _mk_spec(EID2, CPU, S0, rank=0)
check("c8 忽略死进程留下的陈旧 slot 文件",
      kvrt._c8_resolve(sp_dead, publish=True, own_required=S0) is False)
os.environ["FN_KVOFF_LAYOUT_TIMEOUT"] = "1"

# --- 调度器侧：worker 未决 ⇒ CPU 档行数置 0（关 store，不炸启动，也不越界） ---
EID3 = "selftest-c8-nosub-%d" % os.getpid()
_pub(EID3, 0, S0)
_pub(EID3, 1, S1)  # 只有 slot、无 decision
os.environ["FN_KVOFF_LAYOUT_TIMEOUT"] = "0.5"
sp3 = _mk_spec(EID3, CPU, S0, rank=0)
sp3.num_chunks = CPU // kvrt._round_up(S0)  # 模拟上游口径（偏大，越界风险来源）
check("c8 调度器侧未收齐决策 -> num_chunks 归零（禁 store 而非带错几何跑）",
      kvrt._c8_resolve(sp3, publish=False) is False and sp3.num_chunks == 0,
      "num_chunks=%d" % sp3.num_chunks)
os.environ["FN_KVOFF_LAYOUT_TIMEOUT"] = "1"

# --- 调度器侧：全私有 ⇒ 取各 rank 行数下界（不越任何 rank 的界） ---
EID4 = "selftest-c8-priv-%d" % os.getpid()
r0 = CPU // kvrt._round_up(S0)
r1 = CPU // kvrt._round_up(S1)
_pub(EID4, 0, S0, decision="private", nch=r0, rows=r0)
_pub(EID4, 1, S1, decision="private", nch=r1, rows=r1)
sp4 = _mk_spec(EID4, CPU, S0, rank=0)
check("c8 全私有决策 -> 采纳 min(rows)（=上游 rank0 口径，rank0 块更大）",
      kvrt._c8_resolve(sp4, publish=False) is False and sp4.num_chunks == min(r0, r1),
      "%d vs min(%d,%d)" % (sp4.num_chunks, r0, r1))

# --- 后段 rank 块更大时，上游口径会越界；min 规则必须收紧 ---
EID6 = "selftest-c8-big-%d" % os.getpid()
SA, SB = 20_000_000, 30_000_000  # rank0 小、rank1 大
ra = CPU // kvrt._round_up(SA)
rb = CPU // kvrt._round_up(SB)
_pub(EID6, 0, SA, decision="private", nch=ra, rows=ra)
_pub(EID6, 1, SB, decision="private", nch=rb, rows=rb)
sp6 = _mk_spec(EID6, CPU, SA, rank=0)
check("c8 rank1 块更大 -> 采纳下界，堵住上游口径的越界（rows=%d<%d）" % (rb, ra),
      kvrt._c8_resolve(sp6, publish=False) is False and sp6.num_chunks == rb,
      "%d vs %d" % (sp6.num_chunks, rb))

# --- 混合决策（一侧公共区、一侧私有）⇒ 仍取 min，绝不越界 ---
EID7 = "selftest-c8-mixed-%d" % os.getpid()
_pub(EID7, 0, S0, decision="shared", row=kvrt._round_up(S0) + kvrt._round_up(S1),
     nch=CPU // (kvrt._round_up(S0) + kvrt._round_up(S1)))
_pub(EID7, 1, S1, decision="private", nch=r1, rows=r1)
sp7 = _mk_spec(EID7, CPU, S0, rank=0)
check("c8 混合决策 -> min(rows) 且不宣称公共区",
      kvrt._c8_resolve(sp7, publish=False) is False
      and sp7.num_chunks == min(CPU // (kvrt._round_up(S0) + kvrt._round_up(S1)), r1))

# --- 真实共享区：两个 rank 的前缀和视图落在同一块内存且区段不相交 ---
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion as _BASE  # noqa: E402

EID5 = "selftest-c8-region-%d" % os.getpid()
ROW = 4096 + 8192
CTX0 = {"rank": 0, "offset": 0, "slot": 4096, "row_stride": ROW,
        "num_chunks": 3, "slots": [4096, 8192]}
CTX1 = {"rank": 1, "offset": 4096, "slot": 8192, "row_stride": ROW,
        "num_chunks": 3, "slots": [4096, 8192]}
holder = {}
kvrt._LAYOUT_CTX[0] = CTX0
reg0 = cpu_spec.SharedOffloadRegion(
    engine_id=EID5, num_chunks=3, rank=0, kv_bytes_per_chunk=ROW,
    cpu_page_size=4096,
    barrier=lambda: holder.setdefault(
        "b", (kvrt._LAYOUT_CTX.__setitem__(0, CTX1),
              cpu_spec.SharedOffloadRegion(
                  engine_id=EID5, num_chunks=3, rank=1, kv_bytes_per_chunk=ROW,
                  cpu_page_size=8192),
              kvrt._LAYOUT_CTX.__setitem__(0, CTX0))[1]))
kvrt._LAYOUT_CTX[0] = None
reg1 = holder.get("b")
check("c8 双 rank 打开同一 region（joiner 成功）", reg1 is not None)
if reg1 is not None:
    check("c8 两侧 total_size 一致", reg0.total_size_bytes == reg1.total_size_bytes == ROW * 3)
    check("c8 rank0 区段 [0,4096)", (reg0._worker_offset, reg0._worker_area_end) == (0, 4096))
    check("c8 rank1 区段 [4096,12288)", (reg1._worker_offset, reg1._worker_area_end) == (4096, 12288))
    v0 = reg0.create_next_worker_view(4096)
    v1 = reg1.create_next_worker_view(8192)
    check("c8 视图跨步行长 = row_stride", v0.stride(0) == ROW and v1.stride(0) == ROW)
    v0[:, :] = 7
    check("c8 rank0 写入对 rank1 可见（同一物理区）",
          int(reg1.base_tensor[0].item()) == 7)
    check("c8 rank1 自己的区段未被串写",
          int(reg1.base_tensor[4096].item()) == 0)
    v1[:, 4096:] = 9  # rank1 视图的后半格
    check("c8 rank1 写入不越到自己区段外",
          int(reg0.base_tensor[4095].item()) == 7 and int(reg1.base_tensor[12287].item()) == 9)
    # --- 指针算术：公共区视图喂给上游 compute_sub_block_ptrs ---
    # （c8 改的只是"每 rank 的区段起点/行长"，越界与否全看这段地址算术）
    import numpy as np

    from vllm.v1.kv_offload.cpu.gpu_worker import compute_sub_block_ptrs as _csbp

    # 注意：两个 rank 各自 mmap 同一个文件 ⇒ **虚拟基址不同**，只能各按自己的
    # 基址算区内偏移（跨 rank 的等价性由上面的字节可见性用例证明）。
    addr0 = reg0.base_tensor.data_ptr()
    addr1 = reg1.base_tensor.data_ptr()
    blk = np.array([0, 2], dtype=np.int64)  # 两个 chunk id
    out = np.empty(2, dtype=np.uint64)
    _csbp(blk, 1, out, v0)
    got = [int(x) - addr0 for x in out]
    check("c8 rank0 指针 = 区段起点 + chunk×row_stride", got == [0, 2 * ROW], str(got))
    out1 = np.empty(2, dtype=np.uint64)
    _csbp(blk, 1, out1, v1)
    got1 = [int(x) - addr1 for x in out1]
    check("c8 rank1 指针 = 自己区段起点 + chunk×row_stride（不与 rank0 重叠）",
          got1 == [4096 + int(b) * ROW for b in blk], str(got1))
    check("c8 rank1 同 chunk 偏移恰比 rank0 大一个区段起点",
          [g - o for g, o in zip(got1, got)] == [4096, 4096], str(got1))

    overflow = False
    try:
        reg1.create_next_worker_view(8192)  # 再要一格必然越界
    except AssertionError:
        overflow = True
    check("c8 区段溢出会被断言拦住", overflow)
    del v0, v1
    reg0.cleanup()
    reg1.cleanup()

import glob as _glob

for _f in _glob.glob("/dev/shm/vllm_kvoff_slot.selftest-c8-*"):
    try:
        os.unlink(_f)
    except OSError:
        pass



# ---------------------------------------------------------------- 汇总
print()
if FAIL:
    print(f"SELFTEST FAIL: {len(FAIL)} 项: {FAIL}")
    sys.exit(1)
print("SELFTEST PASS：rt-patch #9 (c1/c2/c6/c7/c8) 全部就位")
