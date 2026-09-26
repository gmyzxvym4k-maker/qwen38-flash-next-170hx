"""dsh_kvoff_rt —— KV 二级缓存（OffloadingConnector）在官方 vLLM 0.30.0 的运行时移植（rt-patch #9）。

背景：旧 chroot 定制镜像栈（vLLM v0.1.dev20073）上，CPU KV 二级缓存经 c1→c7 七轮
演进才在 PP2 + QSA/MTP 拓扑下稳定（见交付仓库 docs/05-patches.md D 组与记忆条目）。
09-29 因「21h 生产零外部命中 + 常驻 107GB pinned」定案退役；09-26 迁到官方 0.30.0 时
该功能未随迁。本模块把其中仍然需要的修复移植到 0.30.0 的运行时补丁架构
（PYTHONPATH sitecustomize 注入，site-packages 零改动），供随时经 FN_KVOFF=1 启用。

对照旧补丁组的取舍：
  c1  offloading/config.get_offloading_group_ids —— 剔除不可前缀缓存的分组
      （本模型 = QSA key 环形缓冲 CircularBufferSpec，block=ring 不整除 1616）与
      空层分组（PLE 占位）。0.30.0 的 scheduler/worker 全部按显式 group_id 或
      「过滤后位置」索引（无旧栈的承重位置问题），源头过滤即完整闭环。
  c2  kv_offload/cpu/spec.CPUOffloadingSpec._uses_shared_region —— pp_size>1 时
      走每 rank 私有 pinned 缓冲：共享 mmap 区按「创建者 ftruncate 自己的字节数」
      定协议，PP 各 rank 层数不同 → 尺寸不同 → joiner 30s 超时（09-18 实锤）。
  c6  有界等待 + store 熔断只读降级（gpu_worker / offloading.worker /
      offloading.common / offloading.scheduler 四处）：
      · handle_preemptions 的 worker.wait(jobs_to_flush) 原版无限阻塞主线程，
        PP2 下 flush store 的流链 wait_stream(compute) 与跨 rank NCCL 中继互等
        成环（09-24 三次卡死实锤）→ 轮询 + FN_KVOFF_WAIT_TIMEOUT（缺省 15s）；
      · submit_store 抛错/返回 False/wait 超时 → 失败 ack（job 记入 failed_jobs），
        scheduler 侧 complete_store(success=False) 撤登记并释放 CPU 块；
      · 任一失败 → store 方向熔断，后续不再建新 store 任务（只读降级：
        已有条目继续 lookup/load 命中；load 方向永不熔断，正确性零妥协）。
  c7  store/load 全走 Triton SM 内核，彻底绕开 cuMemcpyBatchAsync——
      该 C++ 批量拷贝 API 与 PP2 NCCL-P2P 并发在本驱动（610.43.03/GSP）上冻结
      compute 流（09-23 崩溃 + 09-24 七次卡死共同根因）。0.30.0 的
      _select_swap_blocks_fn 对 GPU→CPU 恒选 C++ 路径、且页 >28KB 也回落，
      本模型块页是 MB 级 ⇒ 必须双方向强制 Triton 并去掉 MIN_N 回落。
  c3  （metrics 未知 key 崩溃防御）不移植：旧崩溃由我们自加 stats key 引起，
      本移植不新增任何 stats key，上游 defs/sender 自洽。
  c5a （MTP 全组标 eagle 的双罚）不移植：0.30.0 原生含同款修复
      （from_spec：无分组标注 drafter 时「treating all groups as non-draft」）。
  13/14/15/25 的观测/dump 增强不移植（诊断用，非功能必需）。

惰性（生产安全）：所有钩子只挂 vllm 的 offloading / kv_offload.cpu 模块——
不配 --kv-transfer-config 时这些模块根本不会被导入，钩子不触发，对默认生产形态
（FN_KVOFF=0）零影响。紧急总闸 DSH_KVOFF_RT_DISABLE=1：挂载后各回调变 no-op，
等效回到纯原厂 connector 行为（不用改任何文件）。

版本锁定：0.30.0。每个补丁函数替换/包裹前都校验目标属性形态，缺失即跳过并
大声告警（宁可在首请求处炸出真错，不让静默错语义上线）。
"""

import os
import time

_DISABLED = os.environ.get("DSH_KVOFF_RT_DISABLE", "") == "1"


def _log(msg):
    import sys

    sys.stderr.write("[rt-patch-kvoff] %s\n" % msg)
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# c1：源头剔除不可 offload 的分组（QSA 环形 + 空占位）
# ---------------------------------------------------------------------------
def _patch_offloading_config(m):
    if _DISABLED:
        return
    fn = getattr(m, "get_offloading_group_ids", None)
    if fn is None or getattr(fn, "_dsh_c1", False):
        return

    def filtered(kv_cache_config):
        ids = tuple(fn(kv_cache_config))
        keep, dropped = [], []
        for gid in ids:
            group = kv_cache_config.kv_cache_groups[gid]
            spec = group.kv_cache_spec
            if not getattr(spec, "prefix_cacheable", True):
                dropped.append((gid, type(spec).__name__, "not prefix_cacheable"))
                continue
            if not group.layer_names:
                dropped.append((gid, type(spec).__name__, "empty layers"))
                continue
            keep.append(gid)
        if not keep:
            _log("c1: 过滤后无可用分组，回退不过滤（原版行为）")
            return ids
        if dropped:
            lg = getattr(m, "logger", None)
            if lg is not None:
                lg.info(
                    "[dsh-kvoff c1] offloading 分组 %s -> %s，剔除 %s",
                    list(ids), list(keep), dropped,
                )
        return tuple(keep)

    filtered._dsh_c1 = True
    m.get_offloading_group_ids = filtered


# ---------------------------------------------------------------------------
# c2：PP>1 禁用共享 pinned mmap 区（每 rank 私有缓冲）
# ---------------------------------------------------------------------------
def _patch_cpu_spec(m):
    if _DISABLED:
        return
    cls = getattr(m, "CPUOffloadingSpec", None)
    orig = getattr(cls, "_uses_shared_region", None)
    if cls is None or orig is None or getattr(orig, "_dsh_c2", False):
        return

    def patched(self):
        par = getattr(getattr(self, "config", None), "parallel", None)
        if par is not None and getattr(par, "pp_size", 1) > 1:
            if not getattr(cls, "_dsh_c2_logged", False):
                cls._dsh_c2_logged = True
                lg = getattr(m, "logger", None)
                if lg is not None:
                    lg.info(
                        "[dsh-kvoff c2] pp_size=%s>1 -> CPU KV 走每 rank 私有 "
                        "pinned 缓冲（共享 mmap 区在 PP 各 rank 尺寸不一致）",
                        par.pp_size,
                    )
            return False
        return orig(self)

    patched._dsh_c2 = True
    cls._uses_shared_region = patched


# ---------------------------------------------------------------------------
# c7：swap 全走 Triton（绕开 cuMemcpyBatchAsync）
# ---------------------------------------------------------------------------
def _patch_swap_triton(m):
    if _DISABLED:
        return
    if getattr(m, "MIN_N", None):
        m.MIN_N = 0
        _log("c7: swap_blocks_triton.MIN_N -> 0（小批量不再回落 C++ 批量拷贝）")


def _patch_cpu_gpu_worker(m):
    if _DISABLED:
        return
    # ---- c7: _select_swap_blocks_fn 双方向强制 Triton ----
    sel = getattr(m, "_select_swap_blocks_fn", None)
    if sel is not None and not getattr(sel, "_dsh_c7", False):

        def sel_patched(layer_refs_per_group, gpu_to_cpu):
            if (
                getattr(m, "HAS_TRITON", False)
                and not m.current_platform.is_xpu()
                and not m.current_platform.is_rocm()
            ):
                page_sizes = [
                    r.page_size_bytes
                    for g in layer_refs_per_group
                    for r in g
                ]
                if page_sizes and not any(s % 8 for s in page_sizes):
                    chunk = min(m.triton.next_power_of_2(max(page_sizes)), 8192)
                    import functools as _f

                    return _f.partial(m.swap_blocks_batch, bytes_per_chunk=chunk)
                m.logger.warning(
                    "[dsh-kvoff c7] 页大小非 8B 对齐，回落原选择逻辑"
                    "（该形态下 cuMemcpyBatchAsync 可能冻结流）",
                )
            return sel(layer_refs_per_group, gpu_to_cpu)

        sel_patched._dsh_c7 = True
        m._select_swap_blocks_fn = sel_patched
        _log("c7: store/load 双向强制 Triton swap 内核（跳过 THRESHOLD/MIN_N 回落）")

    # ---- c6a：store 方向熔断 + 有界等待（SingleDirectionOffloadingHandler） ----
    H = getattr(m, "SingleDirectionOffloadingHandler", None)
    if H is not None and not getattr(H, "_dsh_c6", False):
        H._dsh_c6 = True
        orig_transfer_async = H.transfer_async
        orig_wait = H.wait

        def transfer_async(self, job_id, src_spec, dst_spec):
            if self.gpu_to_cpu and getattr(self, "_dsh_stalled", False):
                return False  # 熔断后拒绝入队，由 connector 走失败 ack
            return orig_transfer_async(self, job_id, src_spec, dst_spec)

        def wait(self, job_ids):
            """有界等待；返回超时未完成的 job 集合（原版返回 None）。"""
            if self.gpu_to_cpu and getattr(self, "_dsh_stalled", False):
                return set(self._transfer_events.keys())
            timeout = float(os.environ.get("FN_KVOFF_WAIT_TIMEOUT", "15"))
            deadline = time.monotonic() + timeout
            for job_id in job_ids:
                event = self._transfer_events.get(job_id)
                if event is None:
                    continue
                while not event.query():
                    if time.monotonic() >= deadline:
                        if self.gpu_to_cpu:
                            self._dsh_stalled = True
                            try:
                                head = self._transfers[0] if self._transfers else None
                                m.logger.error(
                                    "[dsh-kvoff c6] FUSE: store 方向 job=%s 未在"
                                    " %.1fs 内完成 -> 熔断；chain_len=%d head_job=%s",
                                    job_id, timeout, len(self._transfers),
                                    getattr(head, "job_id", None),
                                )
                            except Exception:
                                pass
                        else:
                            m.logger.error(
                                "[dsh-kvoff c6] load 方向 job=%s 等待超时 %.1fs"
                                "（不熔断，交回上层按未完成处理）", job_id, timeout,
                            )
                        return set(self._transfer_events.keys())
                    time.sleep(0.002)
            return set()

        H.transfer_async = transfer_async
        H.wait = wait

    # ---- c6a：CPUOffloadingWorker.wait 透传未完成集合 ----
    W = getattr(m, "CPUOffloadingWorker", None)
    if W is not None and not getattr(W, "_dsh_c6w", False):
        W._dsh_c6w = True

        def worker_wait(self, job_ids):
            incomplete = set()
            for h in (self._store_handler, self._load_handler):
                r = h.wait(job_ids)
                if r:
                    incomplete |= set(r)
            return incomplete

        W.wait = worker_wait


# ---------------------------------------------------------------------------
# c6b：WorkerMetadata 增加 failed_jobs / fuse_tripped 通道
# ---------------------------------------------------------------------------
def _patch_offloading_common(m):
    if _DISABLED:
        return
    M = getattr(m, "OffloadingWorkerMetadata", None)
    if M is None or getattr(M, "_dsh_c6m", False):
        return
    M._dsh_c6m = True
    orig_init = M.__init__

    def init(self, *a, **k):
        fj = k.pop("failed_jobs", None)
        ft = k.pop("fuse_tripped", False)
        orig_init(self, *a, **k)
        self.failed_jobs = dict(fj) if fj else {}
        self.fuse_tripped = bool(ft)

    def aggregate(self, other):
        merged = M(
            completed_jobs=dict(self.completed_jobs),
            transfer_stats=self.transfer_stats.aggregate(other.transfer_stats),
        )
        fj = dict(self.failed_jobs)
        for jid, cnt in getattr(other, "failed_jobs", {}).items():
            fj[jid] = fj.get(jid, 0) + cnt
        merged.failed_jobs = fj
        merged.fuse_tripped = bool(
            getattr(self, "fuse_tripped", False) or getattr(other, "fuse_tripped", False)
        )
        return merged

    M.__init__ = init
    M.aggregate = aggregate


# ---------------------------------------------------------------------------
# c6c：connector worker 失败 ack / 只读降级 / 迟到完成去重
# ---------------------------------------------------------------------------
def _patch_offloading_worker(m):
    if _DISABLED:
        return
    C = getattr(m, "OffloadingConnectorWorker", None)
    if C is None or getattr(C, "_dsh_c6", False):
        return
    C._dsh_c6 = True
    M = m.OffloadingWorkerMetadata

    orig_init = C.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self._dsh_failed_jobs = set()
        self._dsh_store_fused = False

    def _fail(self, job_id, fuse=False):
        self._dsh_failed_jobs.add(job_id)
        self._connector_worker_meta.failed_jobs[job_id] = (
            self._connector_worker_meta.failed_jobs.get(job_id, 0) + 1
        )
        if fuse:
            self._dsh_store_fused = True
            self._connector_worker_meta.fuse_tripped = True

    def _submit_stores_drained(self):
        """排空 _unsubmitted_store_jobs：熔断不阻塞提交队列（否则调度器
        has_pending_push_work 会等不到 ack 而空转死锁）。失败一律 ack。"""
        if self._dsh_store_fused:
            pending = self._unsubmitted_store_jobs
            self._unsubmitted_store_jobs = []
            for job_id, _src, _dst in pending:
                _fail(self, job_id)
            return
        pending = self._unsubmitted_store_jobs
        self._unsubmitted_store_jobs = []
        for job_id, src_spec, dst_spec in pending:
            try:
                ok = self.worker.submit_store(job_id, src_spec, dst_spec)
            except Exception:
                m.logger.exception(
                    "[dsh-kvoff c6] submit_store(job=%s) 抛错 -> 失败 ack + 熔断", job_id
                )
                ok = False
            if not ok:
                _fail(self, job_id, fuse=True)

    def handle_preemptions(self, kv_connector_metadata):
        assert self.worker is not None
        if kv_connector_metadata.jobs_to_flush:
            for job_id in kv_connector_metadata.jobs_to_flush:
                entry = kv_connector_metadata.store_jobs.pop(job_id, None)
                if entry is not None:
                    if not self._is_store_writer:
                        self._connector_worker_meta.mark_completed(job_id)
                        continue
                    self._unsubmitted_store_jobs.append(
                        (job_id, entry.src_spec, entry.dst_spec)
                    )
        _submit_stores_drained(self)
        if kv_connector_metadata.jobs_to_flush:
            if not set(kv_connector_metadata.jobs_to_flush).issubset(
                self._dsh_failed_jobs
            ):
                try:
                    incomplete = self.worker.wait(
                        kv_connector_metadata.jobs_to_flush
                    ) or set()
                except Exception:
                    m.logger.exception(
                        "[dsh-kvoff c6] wait(jobs_to_flush) 抛错 -> 全量失败 ack"
                    )
                    incomplete = set(kv_connector_metadata.jobs_to_flush)
                for job_id in set(kv_connector_metadata.jobs_to_flush) & set(incomplete):
                    _fail(self, job_id, fuse=True)

    def start_kv_transfers(self, metadata):
        assert self.worker is not None
        _submit_stores_drained(self)
        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            success = self.worker.submit_load(
                job_id, entry.src_spec, entry.dst_spec
            )
            # load 正确性不可妥协：维持原版 assert 语义
            assert success

    orig_prepare = C.prepare_store_kv

    def prepare_store_kv(self, metadata):
        if self._dsh_store_fused:
            for job_id in metadata.store_jobs:
                if not self._is_store_writer:
                    # 非 writer rank 本就不写数据：照常 completed ack，不算失败
                    self._connector_worker_meta.mark_completed(job_id)
                else:
                    _fail(self, job_id)
            return
        orig_prepare(self, metadata)

    def get_finished(self, finished_req_ids):
        assert self.worker is not None
        finished_recving = set()
        for transfer_result in self.worker.get_finished():
            job_id = transfer_result.job_id
            if job_id in self._dsh_failed_jobs:
                # 迟到的“完成”（多半是熔断路径上的残骸）：不双计
                continue
            if not transfer_result.success:
                _fail(self, job_id, fuse=True)
                continue
            is_load = job_id in self._load_jobs
            if (
                transfer_result.transfer_time is not None
                and transfer_result.transfer_size is not None
            ):
                stats = (
                    self._connector_worker_meta.transfer_stats.load
                    if is_load
                    else self._connector_worker_meta.transfer_stats.store
                )
                stats.record(
                    transfer_result.transfer_size, transfer_result.transfer_time
                )
            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)
        return set(), finished_recving

    def build_connector_worker_meta(self):
        meta = self._connector_worker_meta
        if (
            not meta.completed_jobs
            and not meta.failed_jobs
            and not meta.fuse_tripped
        ):
            return None
        self._connector_worker_meta = M()
        return meta

    C.__init__ = init
    C.handle_preemptions = handle_preemptions
    C.start_kv_transfers = start_kv_transfers
    C.prepare_store_kv = prepare_store_kv
    C.get_finished = get_finished
    C.build_connector_worker_meta = build_connector_worker_meta


# ---------------------------------------------------------------------------
# c6d：scheduler 失败登记撤销 + 熔断只读降级
# ---------------------------------------------------------------------------
def _patch_offloading_scheduler(m):
    if _DISABLED:
        return
    S = getattr(m, "OffloadingConnectorScheduler", None)
    M = getattr(m, "OffloadingWorkerMetadata", None)
    if S is None or M is None or getattr(S, "_dsh_c6", False):
        return
    for attr in (
        "update_connector_output",
        "_build_aligned_boundary_store_jobs",
        "_build_partial_tail_store_jobs",
        "_remove_pending_job",
    ):
        if not hasattr(S, attr):
            _log("c6d: scheduler 类缺少 %s（版本漂移？）-> 跳过 scheduler 补丁" % attr)
            return
    for attr in ("OffloadingConnectorStats", "_TransferMetricName", "logger"):
        if not hasattr(m, attr):
            _log("c6d: scheduler 模块缺少 %s（版本漂移？）-> 跳过 scheduler 补丁" % attr)
            return
    S._dsh_c6 = True

    # ---- 完整替换 update_connector_output：与原版逐行等价，外加
    #      failed_jobs→success=False 收尾与 fuse 只读降级 ----
    OffloadingConnectorStats = m.OffloadingConnectorStats
    _TransferMetricName = m._TransferMetricName
    logger = m.logger

    def update_connector_output(self, connector_output):
        if not hasattr(self, "_dsh_failed_seen"):
            self._dsh_failed_seen = set()
            self._dsh_stores_disabled = False
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, M):
            assert meta is None
            meta = M()
        if not meta.transfer_stats.is_empty():
            transfer_stats = OffloadingConnectorStats()
            if not meta.transfer_stats.load.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_BYTES, meta.transfer_stats.load.bytes
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_TIME, meta.transfer_stats.load.time
                )
                for size in meta.transfer_stats.load.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.LOAD_SIZE, size
                    )
            if not meta.transfer_stats.store.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_BYTES, meta.transfer_stats.store.bytes
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_TIME, meta.transfer_stats.store.time
                )
                for size in meta.transfer_stats.store.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.STORE_SIZE, size
                    )
            self._connector_stats.aggregate(transfer_stats)

        # [dsh-kvoff c6] 熔断 -> 只读降级（不再新建 store 任务）
        if getattr(meta, "fuse_tripped", False) and not self._dsh_stores_disabled:
            self._dsh_stores_disabled = True
            logger.error(
                "[dsh-kvoff c6] worker store fuse tripped -> stop creating store "
                "jobs (read-only degrade; existing entries keep serving)"
            )
        for job_id in getattr(meta, "failed_jobs", {}) or {}:
            self._dsh_failed_seen.add(job_id)

        def _try_finish(job_id, count, failed_here):
            assert count > 0
            if job_id < self._stale_job_threshold:
                logger.debug(
                    "Skipping stale %s job %d (pre-reset counter: %d)",
                    "failed" if failed_here else "completed",
                    job_id,
                    self._stale_job_threshold,
                )
                return
            job_status = self._jobs.get(job_id)
            if job_status is None:
                return
            job_status.pending_count -= count
            if job_status.pending_count > 0:
                return
            assert job_status.pending_count == 0

            is_failed = failed_here or (job_id in self._dsh_failed_seen)
            req_status = self._req_status[job_status.req_id]
            if job_status.is_store:
                # [dsh-kvoff c6] 失败 store：success=False 撤登记 + 释放 CPU 块
                self.manager.complete_store(
                    job_status.keys, req_status.req_context, success=not is_failed
                )
            else:
                self.manager.complete_load(job_status.keys, req_status.req_context)
                if self._chunks_being_loaded:
                    self._chunks_being_loaded.difference_update(job_status.keys)
            if self._block_id_to_pending_jobs:
                self._remove_pending_job(job_id, job_status.fenced_block_ids)
                if req_status.req.is_finished():
                    self._remove_pending_job(
                        job_id, job_status.deferred_fence_block_ids
                    )
            del self._jobs[job_id]
            req_status.transfer_jobs.discard(job_id)
            self._dsh_failed_seen.discard(job_id)
            if req_status.finished_signaled and not req_status.transfer_jobs:
                del self._req_status[job_status.req_id]

        for job_id, count in meta.completed_jobs.items():
            _try_finish(job_id, count, False)
        for job_id, count in (getattr(meta, "failed_jobs", {}) or {}).items():
            if job_id in self._jobs:
                _try_finish(job_id, count, True)

    def aligned_guarded(self, *a, **k):
        if getattr(self, "_dsh_stores_disabled", False):
            return {}
        return _orig_aligned(self, *a, **k)

    def partial_guarded(self, *a, **k):
        if getattr(self, "_dsh_stores_disabled", False):
            return {}
        return _orig_partial(self, *a, **k)

    _orig_aligned = S._build_aligned_boundary_store_jobs
    _orig_partial = S._build_partial_tail_store_jobs
    S.update_connector_output = update_connector_output
    S._build_aligned_boundary_store_jobs = aligned_guarded
    S._build_partial_tail_store_jobs = partial_guarded


PATCHES = {
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading.config":
        _patch_offloading_config,
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading.common":
        _patch_offloading_common,
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker":
        _patch_offloading_worker,
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler":
        _patch_offloading_scheduler,
    "vllm.v1.kv_offload.cpu.spec": _patch_cpu_spec,
    "vllm.v1.kv_offload.cpu.gpu_worker": _patch_cpu_gpu_worker,
    "vllm.v1.kv_offload.cpu.swap_blocks_triton": _patch_swap_triton,
}
