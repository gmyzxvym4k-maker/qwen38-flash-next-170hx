#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kvoff-c6 (2026-09-24)：KVOFF store 路径「有界等待 + 熔断降级」补丁。

背景（实锤链）：
  * 09-23 抢占 flush 路径 submit_store → cuMemcpyBatchAsync error1 → Worker 死 → EngineDead；
  * 09-24 常规流量三次卡死：worker 主线程在 handle_preemptions → worker.wait(jobs_to_flush)
    → event.synchronize() 无限阻塞；flush store 的流链 wait_stream(compute)，而 compute 流
    含 PP2 跨 rank NCCL 中继，对端主线程同样阻塞在自己的 wait() → 互等成环 → 永久冻结
    （shm_broadcast×4 → RPC sample_tokens 超时 → EngineDead）。96GiB/64GiB 容量档均复现，
    与容量无关。

修复设计（全部落在 store 方向，LOAD 正确性零妥协）：
  1. gpu_worker.py  wait() 改有界轮询（默认 15s，env FN_KVOFF_WAIT_TIMEOUT 可调）；
     超时 → 本方向 _stalled=True（熔断）+ 取证日志 + 返回未完成 job 集合。
     transfer_async 在熔断后直接返回 False（不入队）。有界等待本身就拆掉了死锁环：
     主线程不再无限阻塞 → 对端中继得以推进 → 多数情况下 store 会正常完成、熔断不触发。
  2. common.py  OffloadingWorkerMetadata 增 failed_jobs / fuse_tripped 字段 + aggregate 合并。
  3. offloading/worker.py  提交循环 try/except：submit 抛错或返回 False → 失败 ack；
     wait 返回未完成集 → 熔断；prepare_store_kv 熔断后直接失败 ack；
     get_finished 对已失败 job 去重（迟到完成不双计）。
  4. scheduler.py  failed_jobs 处理（store → manager.complete_store(success=False)，
     释放 CPU 块且不登记有效）；fuse_tripped → _stores_disabled=True，
     两个 store 任务构建函数短路返回 {}（顺带清理 finished req_status 防泄漏）。
     已有缓存条目继续提供命中（lookup/load 不受影响）——即「只读降级」。

安全性论证：
  * store 是 GPU→CPU 纯读：跳过/失败只丢缓存条目，不污染 GPU 状态；
  * 失败 ack 走 manager 现成的 complete_store(success=False)：撤销登记并释放 CPU 块；
  * 熔断后不再有新 store → 被释放的 CPU 块不会被复用写入（无新分配者），
    卡死的迟到写入落在无人引用的块上，不构成数据竞争；
  * jobs_to_flush 恒为 store（scheduler 侧 assert is_store 把关），load 方向永不熔断。

用法：python3 patch-kvoff-c6.py --check | --apply | --revert
"""
import argparse
import os
import py_compile
import shutil
import sys

ROOT = "/media/ll/data/vllm-image/rootfs/usr/local/lib/python3.12/dist-packages/vllm"
GW = os.path.join(ROOT, "v1/kv_offload/cpu/gpu_worker.py")
COMMON = os.path.join(ROOT, "distributed/kv_transfer/kv_connector/v1/offloading/common.py")
CW = os.path.join(ROOT, "distributed/kv_transfer/kv_connector/v1/offloading/worker.py")
SCH = os.path.join(ROOT, "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py")
MARK = "kvoff-c6"
BAK = ".bak-kvoffc6-0924"

EDITS = []  # (file, old, new, count)


def E(path, old, new, count=1):
    EDITS.append((path, old, new, count))


# ---------------- gpu_worker.py ----------------
E(GW,
  "import functools\nimport time\n",
  "import functools\nimport os\nimport time\n")

E(GW,
  "        # job_id -> event\n        self._transfer_events: dict[int, torch.Event] = {}",
  "        # [local-patch kvoff-c6] store 方向熔断标志（见 wait/transfer_async）\n"
  "        self._stalled = False\n"
  "        # job_id -> event\n        self._transfer_events: dict[int, torch.Event] = {}")

E(GW,
  """    def transfer_async(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)""",
  """    def transfer_async(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        if self._stalled:
            # [local-patch kvoff-c6] 熔断后不再入队，由 connector 走失败 ack
            return False
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)""")

E(GW,
  """    def wait(self, job_ids: set[int]):
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()
""",
  '''    def wait(self, job_ids: set[int]) -> set[int]:
        """[local-patch kvoff-c6] 有界等待，返回超时未完成的 job 集合。

        原版 event.synchronize() 无限阻塞主线程：PP2 下 flush store 的流链
        wait_stream(compute)，compute 流含跨 rank NCCL 中继，对端主线程同样
        阻塞在自己的 wait() → 互等成环（09-24 三次卡死实锤）。有界轮询拆开
        环路；超时则熔断本方向并交由 connector 失败降级。"""
        if self._stalled:
            return set(self._transfer_events.keys())
        timeout = float(os.environ.get("FN_KVOFF_WAIT_TIMEOUT", "15"))
        deadline = time.monotonic() + timeout
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is None:
                continue
            while not event.query():
                if time.monotonic() >= deadline:
                    self._fuse_with_forensics(job_id, timeout)
                    return set(self._transfer_events.keys())
                time.sleep(0.002)
        return set()

    def _fuse_with_forensics(self, stuck_job_id: int, timeout: float) -> None:
        """[local-patch kvoff-c6] 熔断 + 现场取证（链深/队头/字节数）。"""
        self._stalled = True
        try:
            head = self._transfers[0] if self._transfers else None
            logger.error(
                "[kvoff-c6] FUSE: %s job=%s 未在 %.1fs 内完成 -> 本方向熔断；"
                "chain_len=%d head_job=%s head_bytes=%d pending=%d",
                "GPU->CPU" if self.gpu_to_cpu else "CPU->GPU",
                stuck_job_id,
                timeout,
                len(self._transfers),
                head.job_id if head else None,
                head.num_bytes if head else 0,
                len(self._transfer_events),
            )
        except Exception:  # noqa: BLE001
            logger.exception("[kvoff-c6] fuse forensics failed")
''')

E(GW,
  """    def wait(self, job_ids: set[int]) -> None:
        self._store_handler.wait(job_ids)
        self._load_handler.wait(job_ids)
""",
  """    def wait(self, job_ids: set[int]) -> set[int]:
        # [local-patch kvoff-c6] jobs_to_flush 恒为 store（scheduler assert is_store），
        # load 侧 wait 找不到对应事件、天然 no-op。
        unfinished = self._store_handler.wait(job_ids)
        self._load_handler.wait(job_ids)
        return unfinished

    def inflight_store_job_ids(self) -> set[int]:
        \"\"\"[local-patch kvoff-c6] 当前在 store 链上未完成的 job 集合。\"\"\"
        return set(self._store_handler._transfer_events.keys())
""")

# ---------------- common.py ----------------
E(COMMON,
  """    completed_jobs: dict[int, int] = field(default_factory=dict)
    transfer_stats: TransferStats = field(default_factory=TransferStats)

    def mark_completed(self, job_id: int) -> None:
        \"\"\"Record a transfer job completion from this worker.\"\"\"
        self.completed_jobs[job_id] = 1
""",
  """    completed_jobs: dict[int, int] = field(default_factory=dict)
    # [local-patch kvoff-c6] 失败 ack（store 熔断/提交异常）与熔断标志
    failed_jobs: dict[int, int] = field(default_factory=dict)
    fuse_tripped: bool = False
    transfer_stats: TransferStats = field(default_factory=TransferStats)

    def mark_completed(self, job_id: int) -> None:
        \"\"\"Record a transfer job completion from this worker.\"\"\"
        self.completed_jobs[job_id] = 1

    def mark_failed(self, job_id: int) -> None:
        \"\"\"[local-patch kvoff-c6] Record a failed transfer job from this worker.\"\"\"
        self.failed_jobs[job_id] = 1
""")

E(COMMON,
  """        merged = dict(self.completed_jobs)
        for job_id, v in other.completed_jobs.items():
            merged[job_id] = merged.get(job_id, 0) + v

        return OffloadingWorkerMetadata(
            completed_jobs=merged,
            transfer_stats=self.transfer_stats.aggregate(other.transfer_stats),
        )""",
  """        merged = dict(self.completed_jobs)
        for job_id, v in other.completed_jobs.items():
            merged[job_id] = merged.get(job_id, 0) + v

        # [local-patch kvoff-c6]
        merged_failed = dict(self.failed_jobs)
        for job_id, v in other.failed_jobs.items():
            merged_failed[job_id] = merged_failed.get(job_id, 0) + v

        return OffloadingWorkerMetadata(
            completed_jobs=merged,
            failed_jobs=merged_failed,
            fuse_tripped=self.fuse_tripped or other.fuse_tripped,
            transfer_stats=self.transfer_stats.aggregate(other.transfer_stats),
        )""")

# ---------------- offloading/worker.py ----------------
E(CW,
  """        self._unsubmitted_store_jobs: list[
            tuple[int, GPULoadStoreSpec, LoadStoreSpec]
        ] = []
        self._connector_worker_meta = OffloadingWorkerMetadata()""",
  """        self._unsubmitted_store_jobs: list[
            tuple[int, GPULoadStoreSpec, LoadStoreSpec]
        ] = []
        self._connector_worker_meta = OffloadingWorkerMetadata()
        # [local-patch kvoff-c6] store 熔断状态与失败去重集合
        self._store_fused = False
        self._failed_jobs: set[int] = set()""")

E(CW,
  """        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            assert isinstance(src_spec, GPULoadStoreSpec)
            success = self.worker.submit_store(job_id, src_spec, dst_spec)
            assert success
        self._unsubmitted_store_jobs.clear()""",
  """        self._submit_stores()  # [local-patch kvoff-c6]""")

E(CW,
  """        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            success = self.worker.submit_store(job_id, src_spec, dst_spec)
            assert success
        self._unsubmitted_store_jobs.clear()""",
  """        self._submit_stores()  # [local-patch kvoff-c6]""")

E(CW,
  """        if kv_connector_metadata.jobs_to_flush:
            self.worker.wait(kv_connector_metadata.jobs_to_flush)""",
  """        if kv_connector_metadata.jobs_to_flush:
            # [local-patch kvoff-c6] 有界等待：超时集合走失败 ack + 熔断
            unfinished = self.worker.wait(kv_connector_metadata.jobs_to_flush)
            if unfinished:
                self._trip_store_fuse(extra_failed=set(unfinished))""")

E(CW,
  """    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        for job_id, entry in metadata.store_jobs.items():
            if not self._is_store_writer:""",
  """    # ------------------------------------------------------------------ #
    # [local-patch kvoff-c6] store 提交与熔断降级
    # ------------------------------------------------------------------ #
    def _submit_stores(self) -> None:
        if not self._unsubmitted_store_jobs:
            return
        pending = self._unsubmitted_store_jobs
        self._unsubmitted_store_jobs = []
        for job_id, src_spec, dst_spec in pending:
            assert isinstance(src_spec, GPULoadStoreSpec)
            if self._store_fused:
                self._fail_store_job(job_id)
                continue
            try:
                success = self.worker.submit_store(job_id, src_spec, dst_spec)
            except Exception:  # noqa: BLE001
                # 09-23 实锤：cuMemcpyBatchAsync 入队即抛 error1 → 原版直接
                # 炸穿 execute_model 杀死 worker；现改为熔断降级。
                logger.exception("[kvoff-c6] submit_store raised, job=%s", job_id)
                self._trip_store_fuse(extra_failed={job_id})
                continue
            if not success:
                self._fail_store_job(job_id)

    def _fail_store_job(self, job_id: int) -> None:
        if job_id in self._failed_jobs:
            return
        self._failed_jobs.add(job_id)
        self._connector_worker_meta.mark_failed(job_id)

    def _trip_store_fuse(self, extra_failed: set[int] | None = None) -> None:
        if not self._store_fused:
            self._store_fused = True
            logger.error(
                "[kvoff-c6] KVOFF store 熔断触发：CPU 二级缓存转入只读降级"
                "（不再存新块，已有条目仍可命中；重启实例可恢复写入）"
            )
        if self.worker is not None:
            getter = getattr(self.worker, "inflight_store_job_ids", None)
            if getter is not None:
                for jid in getter():
                    self._fail_store_job(jid)
        for jid, _s, _d in self._unsubmitted_store_jobs:
            self._fail_store_job(jid)
        self._unsubmitted_store_jobs = []
        if extra_failed:
            for jid in extra_failed:
                self._fail_store_job(jid)
        self._connector_worker_meta.fuse_tripped = True

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        for job_id, entry in metadata.store_jobs.items():
            if self._store_fused:
                # [local-patch kvoff-c6] 熔断后到达的新任务直接失败 ack
                self._fail_store_job(job_id)
                continue
            if not self._is_store_writer:""")

E(CW,
  """        for transfer_result in self.worker.get_finished():
            # we currently do not support job failures
            job_id = transfer_result.job_id
            assert transfer_result.success""",
  """        for transfer_result in self.worker.get_finished():
            job_id = transfer_result.job_id
            if job_id in self._failed_jobs:
                # [local-patch kvoff-c6] 熔断前入队的传输迟到完成：
                # 失败已 ack，跳过避免 scheduler 双计（pending_count 变负）。
                self._failed_jobs.discard(job_id)
                continue
            assert transfer_result.success""")

E(CW,
  """    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        \"\"\"Return completed transfer job IDs since the last call.\"\"\"
        if not self._connector_worker_meta.completed_jobs:
            return None""",
  """    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        \"\"\"Return completed transfer job IDs since the last call.\"\"\"
        m = self._connector_worker_meta
        # [local-patch kvoff-c6] failed/fuse 也要及时回传
        if not (m.completed_jobs or m.failed_jobs or m.fuse_tripped):
            return None""")

# ---------------- scheduler.py ----------------
E(SCH,
  "        self._current_batch_jobs_to_flush: set[int] = set()",
  "        self._current_batch_jobs_to_flush: set[int] = set()\n"
  "        # [local-patch kvoff-c6] worker 侧 store 熔断后置 True：\n"
  "        # 不再创建新 store 任务；已有 CPU 条目继续提供命中（只读降级）。\n"
  "        self._stores_disabled = False")

E(SCH,
  """        if not self.config.supports_partial_tail or not handoffs:
            return {}""",
  """        if self._stores_disabled:
            # [local-patch kvoff-c6] 熔断降级：不再生成 store 任务
            return {}
        if not self.config.supports_partial_tail or not handoffs:
            return {}""")

E(SCH,
  """    def _build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[int, TransferJob]:
        blocks_per_chunk = self.config.blocks_per_chunk""",
  """    def _build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[int, TransferJob]:
        if self._stores_disabled:
            # [local-patch kvoff-c6] 熔断降级：不再生成 store 任务；
            # 顺带清理已终结且无在途任务的 req_status，防跳过正常
            # 完成路径造成字典泄漏。
            for rid in list(self._req_status.keys()):
                rs = self._req_status[rid]
                if rs.finished_signaled and not rs.transfer_jobs:
                    del self._req_status[rid]
            return {}
        blocks_per_chunk = self.config.blocks_per_chunk""")

E(SCH,
  """            del self._jobs[job_id]
            req_status.transfer_jobs.remove(job_id)
            if req_status.finished_signaled and not req_status.transfer_jobs:
                del self._req_status[job_status.req_id]

    def get_stats(self) -> OffloadingConnectorStats | None:""",
  """            del self._jobs[job_id]
            req_status.transfer_jobs.remove(job_id)
            if req_status.finished_signaled and not req_status.transfer_jobs:
                del self._req_status[job_status.req_id]

        # [local-patch kvoff-c6] 失败 ack：store → 撤销登记并释放 CPU 块；
        # 与 completed 分支同构，但不调用 success=True 的 complete_store。
        for job_id, count in meta.failed_jobs.items():
            assert count > 0
            if job_id < self._stale_job_threshold:
                continue
            job_status = self._jobs[job_id]
            job_status.pending_count -= count
            if job_status.pending_count > 0:
                continue
            assert job_status.pending_count == 0
            req_status = self._req_status[job_status.req_id]
            if job_status.is_store:
                self.manager.complete_store(
                    job_status.keys, req_status.req_context, success=False
                )
            else:
                logger.error(
                    "[kvoff-c6] unexpected failed LOAD job %s", job_id
                )
                self.manager.complete_load(
                    job_status.keys, req_status.req_context
                )
            if self._block_id_to_pending_jobs:
                self._remove_pending_job(job_id, job_status.fenced_block_ids)
                if req_status.req.is_finished():
                    self._remove_pending_job(
                        job_id, job_status.deferred_fence_block_ids
                    )
            del self._jobs[job_id]
            req_status.transfer_jobs.remove(job_id)
            if req_status.finished_signaled and not req_status.transfer_jobs:
                del self._req_status[job_status.req_id]

        if meta.fuse_tripped and not self._stores_disabled:
            self._stores_disabled = True
            logger.error(
                "[kvoff-c6] worker 侧 store 熔断：本实例 CPU 二级缓存转入"
                "只读降级（停止存新块，保留既有命中），重启实例可恢复"
            )

    def get_stats(self) -> OffloadingConnectorStats | None:""")


# ------------------------------------------------------------------ #
def load(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if args.revert:
        ok = True
        for path in (GW, COMMON, CW, SCH):
            bak = path + BAK
            if os.path.exists(bak):
                shutil.copy2(bak, path)
                print(f"reverted {os.path.basename(path)}")
            elif MARK not in load(path):
                print(f"{os.path.basename(path)} 本就未打补丁")
            else:
                ok = False
                print(f"!! {os.path.basename(path)} 已打补丁但缺备份 {bak}")
        sys.exit(0 if ok else 2)

    srcs = {p: load(p) for p in (GW, COMMON, CW, SCH)}
    done_files = {p for p, t in srcs.items() if MARK in t}
    problems = []
    for path, old, new, count in EDITS:
        if path in done_files:
            continue  # [idempotent] 该文件已应用过，跳过其锚点校验
        n = srcs[path].count(old)
        if n != count:
            problems.append(
                f"{os.path.basename(path)}: 锚点命中 {n} 次（期望 {count}）: {old[:60]!r}"
            )
    if args.apply and len(done_files) == 4:
        print("已应用（幂等跳过）")
        sys.exit(0)
    if args.apply and problems:
        print("!! 锚点校验失败，未做任何修改：")
        for p in problems:
            print("  -", p)
        sys.exit(2)

    if args.apply:
        for path in (GW, COMMON, CW, SCH):
            if path in done_files:
                continue
            if not os.path.exists(path + BAK):
                shutil.copy2(path, path + BAK)
        for path, old, new, count in EDITS:
            if path in done_files:
                continue
            srcs[path] = srcs[path].replace(old, new)
        for path, text in srcs.items():
            if path in done_files:
                continue
            tmp = path + ".c6tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        # 语法校验交给 chroot 内 py3.12（宿主 py3.8 无权也无版本保障）
        print(f"已应用 c6：新改 {4 - len(done_files)} 文件"
              f"（跳过已完成 {len(done_files)}），共 {len(EDITS)} 处编辑")
        # 清 __pycache__（09-15 铁律：改 py 不删 pyc 可能不生效）
        import glob as _g
        for path in srcs:
            d = os.path.join(os.path.dirname(path), "__pycache__")
            stem = os.path.splitext(os.path.basename(path))[0]
            for pyc in _g.glob(os.path.join(d, f"{stem}*.pyc")):
                try:
                    os.remove(pyc)
                except OSError as e:
                    print(f"  警告: 无法删除 {pyc}: {e}")
    else:
        print("check：锚点", "全部命中 ✓" if not problems else "有失配 ✗")
        for p in problems:
            print("  -", p)
        sys.exit(0 if not problems else 2)


if __name__ == "__main__":
    main()
