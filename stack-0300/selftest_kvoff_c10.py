#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selftest_kvoff_c10 —— KV 二级缓存 c10（悬挂 pending 自愈）离线自检（不启引擎、不用 GPU）。

跑法（部署机 ll 用户）：
    PYTHONPATH=/home/ll/deploy/vllm-0300/patches:/home/ll/deploy/vllm-0300/patches-extra \
    FN_KVOFF_PENDING_TTL=1 FN_KVOFF_JOB_TTL=1 \
    /home/ll/vllm-env/bin/python /home/ll/deploy/vllm-0300/selftest_kvoff_c10.py

判据（全部 PASS 才算通过）：
  A. 补丁挂上：CPUOffloadingManager._dsh_c10 == True
  B. 未完成 store 的 chunk：lookup 先 HIT_PENDING
  C. TTL 到期后同一 key 的 lookup 变 MISS（自愈摘除），pending 计数归零、chunk 回池
  D. 正常完成的 store 不受影响：TTL 过后仍 HIT（绝不会把 ready 的块摘掉）
  E. 另一 key 仍能正常 store/lookup（自愈没有破坏别的条目）
  F. scheduler 侧 c10b：超时未收尾的 store job 被强制 complete_store(success=False) 并摘除
"""
import os
import sys
import time

os.environ.setdefault("FN_KVOFF_PENDING_TTL", "1")
os.environ.setdefault("FN_KVOFF_JOB_TTL", "1")

FAIL = 0


def chk(name, ok, detail=""):
    global FAIL
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL += 1


def main():
    from vllm.v1.kv_offload.base import ReqContext, LookupResult, make_offload_key
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

    chk("A. c10 管理器补丁已挂载", getattr(CPUOffloadingManager, "_dsh_c10", False))

    mgr = CPUOffloadingManager(num_chunks=8, cache_policy="lru", enable_events=False)
    ctx = ReqContext(req_id="selftest-c10")
    k_unfinished = make_offload_key(b"\x11" * 32, 0)
    k_done = make_offload_key(b"\x22" * 32, 0)

    # --- 未完成 store：prepare_store 后不 complete ---
    out = mgr.prepare_store([k_unfinished], ctx)
    assert out is not None and out.keys_to_store == [k_unfinished]
    chk("B. 未完成 store 的 lookup = HIT_PENDING", mgr.lookup(k_unfinished, ctx) is LookupResult.HIT_PENDING)
    chk("B2. 未完成 store 计入 write_pending", mgr._num_write_pending_chunks == 1,
        "pending=%d" % mgr._num_write_pending_chunks)

    # --- 正常完成 store ---
    out2 = mgr.prepare_store([k_done], ctx)
    assert out2 is not None and out2.keys_to_store == [k_done]
    mgr.complete_store([k_done], ctx, success=True)
    chk("D0. 完成的 store lookup = HIT", mgr.lookup(k_done, ctx) is LookupResult.HIT)

    free_before = len(mgr._free_list)
    time.sleep(1.3)  # 越过 TTL=1s

    # --- TTL 到期：未完成的被自愈摘除（lookup 触发） ---
    res = mgr.lookup(k_unfinished, ctx)
    chk("C. TTL 后未完成 key 的 lookup 变 MISS", res is LookupResult.MISS, "got=%s" % res)
    chk("C2. pending 计数归零", mgr._num_write_pending_chunks == 0,
        "pending=%d" % mgr._num_write_pending_chunks)
    chk("C3. chunk 已归还池", len(mgr._free_list) == free_before + 1,
        "free %d -> %d" % (free_before, len(mgr._free_list)))

    # --- ready 的块绝不被自愈摘掉 ---
    chk("D. TTL 后已完成的 key 仍是 HIT（ready 块不动）",
        mgr.lookup(k_done, ctx) is LookupResult.HIT)

    # --- 自愈后同 key 可重新 store ---
    out3 = mgr.prepare_store([k_unfinished], ctx)
    chk("E. 自愈后同 key 可重新 store", out3 is not None and out3.keys_to_store == [k_unfinished])
    mgr.complete_store([k_unfinished], ctx, success=True)
    chk("E2. 重新 store 后 lookup = HIT", mgr.lookup(k_unfinished, ctx) is LookupResult.HIT)

    # --- c10b：scheduler 侧超时 job 强制收尾 ---
    from dsh_kvoff_rt import _dsh_reap_stale_jobs

    class _JobStatus:
        def __init__(self):
            self.req_id = "req-1"
            self.is_store = True
            self.keys = [make_offload_key(b"\x33" * 32, 0)]
            self.fenced_block_ids = []
            self.deferred_fence_block_ids = []

    class _ReqStatus:
        def __init__(self):
            self.req_context = ReqContext(req_id="req-1")
            self.transfer_jobs = {77}

    class _FakeSched:
        def __init__(self, mgr):
            self.manager = mgr
            self._jobs = {77: _JobStatus()}
            self._req_status = {"req-1": _ReqStatus()}
            self._chunks_being_loaded = set()
            self._failed_seen = set()

        def _remove_pending_job(self, job_id, block_ids):
            return None

    fs = _FakeSched(mgr)
    # 造一个 pending chunk 给这个 job
    out4 = mgr.prepare_store(fs._jobs[77].keys, fs._req_status["req-1"].req_context)
    assert out4 is not None
    chk("F0. 造出 pending chunk（job 键）", mgr._num_write_pending_chunks == 1)
    # 生产语义：reaper 每步都会跑，job 出生即被登记（first-seen≈创建时刻）。
    # 这里先调一次登记，再越过 TTL 调第二次。
    _dsh_reap_stale_jobs(fs)
    time.sleep(1.3)
    n = _dsh_reap_stale_jobs(fs)
    chk("F. 超时 job 被强制收尾", n == 1 and not fs._jobs, "reaped=%d jobs=%d" % (n, len(fs._jobs)))
    chk("F2. 该 job 的 pending chunk 撤登记", mgr._num_write_pending_chunks == 0,
        "pending=%d" % mgr._num_write_pending_chunks)
    chk("F3. 请求的 transfer_jobs 已清空", not fs._req_status["req-1"].transfer_jobs)

    # --- c11：worker meta 聚合必须把 other.completed_jobs 求和（否则 PP2 永不完成） ---
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingWorkerMetadata,
    )

    a = OffloadingWorkerMetadata()
    b = OffloadingWorkerMetadata()
    a.mark_completed(7)
    b.mark_completed(7)
    b.mark_completed(8)
    agg = a.aggregate(b)
    chk("G. 聚合后 completed_jobs 求和（PP2 两 rank 各 1 次 ⇒ 2）",
        agg.completed_jobs.get(7) == 2, "got=%r" % (agg.completed_jobs,))
    chk("G2. 另一 job 的完成数保留", agg.completed_jobs.get(8) == 1)
    chk("G3. 聚合不吞自身完成数", agg.completed_jobs.get(7) == a.completed_jobs[7] + b.completed_jobs[7])

    print("\n%s（FAIL=%d）" % ("全绿" if FAIL == 0 else "有失败项", FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
