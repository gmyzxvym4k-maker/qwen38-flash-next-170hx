"""rt-patch #13 —— SimpleCPUOffload PP2 CPU 档行数握手对齐 + 越界守卫（2026-09-28 定案）。

【segfault 完整根因链（两个 core dump + 源码实锤，见 docs/08 P61）】
  · 5+ 次 segfault 全在 Worker_PP1：ctypes 调 cuMemcpy* 后 libcuda 内部 near-NULL
    解引用（si_addr=0xcbeff，libcuda+0x1988 同一指令；08:03 与 12:16 两个 core 的
    r11 恰=该实例 PP1 的 CPU 行数 2075/2157）；
  · 上游 generate_scheduler_kv_cache_config() 直接 deepcopy(kv_cache_configs[0])
    —— PP2 下 scheduler 只见 rank0 的张量尺寸（块 22.06MB）；
  · SimpleCPUOffloadScheduler._derive_cpu_config 由此推 num_cpu_blocks=2224，
    cpu_block_pool 发号 0..2223；
  · 但 PP1 含 MTP draft 的 attention 层 → 每块 23.23MB → 自身只配 2157 行。
    id ∈ [2157,2223] 在 PP1 上 base+id*bpb 越过 cudaHostRegister 注册区
    → 本机驱动（610.43.03 + CMP 定制固件）查表 miss 分支缺 NULL 校验 → 原生
    segfault。批量/逐块 API 同崩：坏的是地址，不是 API。

【修复 1（主）：握手 clamp】
  worker._init_cpu_mode 完成后把 {cfg,pid,rows,bpb} 发布到
  /dev/shm/vllm_simple_offload_rows.<cfg_hash>.<pid>.json；
  manager 在 SimpleCPUOffloadScheduler.__init__ 前收集同 cfg 存活 worker 的行数，
  把派生的 cpu_kv_cache_config.num_blocks 钳到 min(rows)。
  启动顺序保证：EngineCore._initialize_kv_caches 中 initialize_from_config
  （worker connector 构建）先于 Scheduler 构建 ⇒ manager 时文件必已存在
  （另有 10s 有界等待 + "文件数<world_size" 告警兜底）。
  无握手文件（TP/单 worker/disk 后端）⇒ 与上游行为完全一致。
  保守性：多收（跨实例误合并）只会更小档 = 更安全，绝不越界。

【修复 2（防漂移）：copy_blocks 越界守卫】
  build_params 按 stream 登记两端行数与方向（src tensor device=cpu ⇒ load，
  cuda ⇒ store）；copy_blocks 逐块校验：
  · store 越界 → 跳过该块（不入档，无害）+ 限频 WARN；
  · load  越界 → 越界端重定向行 0（确定性错数据但绝不越界）+ 限频 ERROR。
  守卫只在主修失效时触发，职责是"不崩 + 可观测"。

【保留】#11 的逐块 cuMemcpyAsync DMA 实现（与根因无关，无害）。

【开关】
  DSH_SIMPLE_OFFLOAD_UPSTREAM=1 ⇒ 全部不钩（上游行为，会复崩，仅取证）
  DSH_SIMPLE_HANDSHAKE=0        ⇒ 关握手 clamp（会复崩，仅取证）
  DSH_SIMPLE_GUARD=0            ⇒ 关越界守卫
  DSH_SIMPLE_BATCH=1            ⇒ 拷贝退回批量 API（不推荐）

挂载点：patches-extra/sitecustomize.py 动态 import 本模块的 PATCHES（无需改 sitecustomize）。
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import json
import os
import socket
import sys
import time

_MARKER13 = "_dsh_simple_rt_v13"
_ROWS_PREFIX = "/dev/shm/vllm_simple_offload_rows."

_WARN_STATE = {"last_ts": 0.0, "count": 0}
_HANDSHAKE_TIMEOUT_S = 10.0


def _log(msg: str) -> None:
    sys.stderr.write(f"[dsh-simple-rt] {msg}\n")
    sys.stderr.flush()


def _warn(kind: str, direction: str, i: int, n: int, blk: int, rows: int) -> None:
    now = time.monotonic()
    _WARN_STATE["count"] += 1
    if now - _WARN_STATE["last_ts"] >= 60.0:
        _WARN_STATE["last_ts"] = now
        _log(
            f"{kind} 越界({direction})：job 第 {i}/{n} 块 id={blk} >= 行数 {rows}"
            f"（累计 {_WARN_STATE['count']} 次）——握手 clamp 疑似失效，"
            f"请核对 worker 发布文件与 manager Allocating 数字"
        )


def _cfg_fingerprint(vllm_config) -> str:
    """同 engine 的 scheduler/worker 进程算出相同指纹；跨模型天然隔离。"""
    try:
        ktc = vllm_config.kv_transfer_config
        payload = json.dumps(
            getattr(ktc, "kv_transfer_config", None) or {},
            sort_keys=True,
            default=str,
        )
    except Exception:
        payload = "?"
    try:
        model = str(vllm_config.model_config.model)
    except Exception:
        model = "?"
    raw = socket.gethostname() + "|" + model + "|" + payload
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 握手发布（worker 侧）
# ---------------------------------------------------------------------------
def _publish_rows(vllm_config, rows: int, bpb: int) -> None:
    try:
        cfg = _cfg_fingerprint(vllm_config)
        # 顺手清理同指纹下 pid 已死的陈旧文件（/dev/shm 不累积）
        for old in glob.glob(f"{_ROWS_PREFIX}{cfg}.*.json"):
            try:
                with open(old) as f:
                    d = json.load(f)
                opid = int(d.get("pid", -1))
                if d.get("cfg") != cfg or not os.path.exists(f"/proc/{opid}"):
                    os.remove(old)
            except Exception:
                pass
        path = f"{_ROWS_PREFIX}{cfg}.{os.getpid()}.json"
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"cfg": cfg, "pid": os.getpid(), "rows": int(rows), "bpb": int(bpb)}, f)
        os.replace(tmp, path)
        _log(f"worker 发布 CPU 档行数：rows={rows} bpb={bpb/2**20:.2f}MiB -> {os.path.basename(path)}")
    except Exception as e:
        _log(f"worker 发布行数失败（manager 将退回上游口径）：{e!r}")


# ---------------------------------------------------------------------------
# 握手收集 + clamp（manager 侧）
# ---------------------------------------------------------------------------
def _collect_rows_min(cfg: str, world_size: int, wait_s: float = _HANDSHAKE_TIMEOUT_S):
    """按 cfg 指纹收集存活 worker 行数 → min；无有效文件 None。"""
    deadline = time.monotonic() + wait_s
    rows: list[int] = []
    while True:
        rows = []
        for p in glob.glob(f"{_ROWS_PREFIX}{cfg}.*.json"):
            try:
                with open(p) as f:
                    d = json.load(f)
                if d.get("cfg") != cfg:
                    continue
                pid = int(d.get("pid", -1))
                if pid >= 0 and not os.path.exists(f"/proc/{pid}"):
                    continue  # 陈旧文件（进程已死）
                rows.append(int(d["rows"]))
            except Exception:
                continue
        if len(rows) >= max(1, world_size) or time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    if not rows:
        return None
    if world_size and len(rows) < world_size:
        _log(
            f"握手警告：只收到 {len(rows)}/{world_size} 个 worker 行数 {rows}"
            f"（先 clamp 到已知 min={min(rows)}；缺失 rank 若更小由守卫兜底）"
        )
    return min(rows)


def _clamp_cpu_config(cpu_config, n_min: int):
    from dataclasses import replace as _replace

    n_old = int(cpu_config.num_blocks)
    if n_min >= n_old:
        return cpu_config
    new_tensors = [_replace(t, size=t.size // n_old * n_min) for t in cpu_config.kv_cache_tensors]
    _log(f"握手 clamp：调度器 CPU 块数 {n_old} -> {n_min}（对齐最窄 worker，防 PP 越界）")
    return _replace(cpu_config, num_blocks=n_min, kv_cache_tensors=new_tensors)


def _patch_manager(module) -> None:
    cls = getattr(module, "SimpleCPUOffloadScheduler", None)
    if cls is None or getattr(cls, _MARKER13, False):
        return
    if os.environ.get("DSH_SIMPLE_OFFLOAD_UPSTREAM", "") == "1":
        _log("DSH_SIMPLE_OFFLOAD_UPSTREAM=1 ⇒ manager 不 clamp")
        return
    handshake_on = os.environ.get("DSH_SIMPLE_HANDSHAKE", "1") != "0"

    orig_init = cls.__init__
    _raw_derive = cls.__dict__.get("_derive_cpu_config")
    orig_derive = (
        _raw_derive.__func__
        if isinstance(_raw_derive, staticmethod)
        else _raw_derive
    )

    def __init__(self, vllm_config, kv_cache_config, cpu_capacity_bytes, *a, **kw):
        if handshake_on:
            n_min = None
            try:
                world = int(vllm_config.parallel_config.world_size)
                n_min = _collect_rows_min(_cfg_fingerprint(vllm_config), world)
            except Exception as e:
                _log(f"握手收集失败（沿用上游口径）：{e!r}")
            if n_min is not None:

                def _derive_clamped(gpu_config, cap, _od=orig_derive, _n=n_min):
                    cfg = _od(gpu_config, cap)
                    try:
                        return _clamp_cpu_config(cfg, _n)
                    except Exception as e:
                        _log(f"clamp 失败（沿用上游口径）：{e!r}")
                        return cfg

                # 实例属性遮蔽类 staticmethod：__init__ 里
                # self._derive_cpu_config(gpu_cfg, cap) 会命中本函数。
                self._derive_cpu_config = _derive_clamped
        orig_init(self, vllm_config, kv_cache_config, cpu_capacity_bytes, *a, **kw)

    cls.__init__ = __init__
    setattr(cls, _MARKER13, True)
    _log("manager：#13 握手 clamp 钩子已挂（SimpleCPUOffloadScheduler.__init__）")


def _patch_worker(module) -> None:
    worker_cls = getattr(module, "SimpleCPUOffloadWorker", None)
    if worker_cls is None or getattr(worker_cls, _MARKER13, False):
        return
    if os.environ.get("DSH_SIMPLE_OFFLOAD_UPSTREAM", "") == "1":
        return

    _orig_init_cpu = worker_cls._init_cpu_mode

    def _init_cpu_mode_v13(self, unique_gpu_caches, total_bytes_per_block, device):
        ret = _orig_init_cpu(self, unique_gpu_caches, total_bytes_per_block, device)
        _publish_rows(self.vllm_config, self.num_cpu_blocks, total_bytes_per_block)
        return ret

    worker_cls._init_cpu_mode = _init_cpu_mode_v13
    setattr(worker_cls, _MARKER13, True)
    _log("worker._init_cpu_mode：#13 行数发布钩子已挂")


# ---------------------------------------------------------------------------
# 拷贝实现（#11 逐块 cuMemcpyAsync）+ 越界守卫
# ---------------------------------------------------------------------------
def _get_memcpy_async():
    test = globals().get("_DSH_TEST_MEMCPY")  # 离线自检注入点
    if test is not None:
        return test
    lib = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    fn = lib.cuMemcpyAsync
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    return fn


def _patch_cuda_mem_ops(module) -> None:
    if os.environ.get("DSH_SIMPLE_OFFLOAD_UPSTREAM", "") == "1":
        _log("DSH_SIMPLE_OFFLOAD_UPSTREAM=1 ⇒ cuda_mem_ops 不动作")
        return
    if getattr(module, _MARKER13, False):
        return
    if not hasattr(module, "copy_blocks") or not hasattr(module, "build_params"):
        _log("cuda_mem_ops 结构不认识 ⇒ #13 跳过")
        return

    import numpy as np

    if not hasattr(module, "_dsh_rows_by_stream"):
        module._dsh_rows_by_stream = {}
    rows_reg = module._dsh_rows_by_stream
    _orig_build_params = module.build_params

    def build_params_guarded(src_caches, dst_caches, stream, src_access_order=None):
        if src_access_order is None:
            params = _orig_build_params(src_caches, dst_caches, stream)
        else:
            params = _orig_build_params(
                src_caches, dst_caches, stream, src_access_order=src_access_order
            )
        try:
            # 方向：store 的 src 是 GPU 张量；load 的 src 是 CPU 张量
            first = next(iter(src_caches.values()))
            dev = str(getattr(getattr(first, "device", "cpu"), "type", None) or getattr(first, "device", "cpu"))
            direction = "load" if dev == "cpu" else "store"
            rows_reg[params.stream_handle] = (
                np.array([int(t.size(0)) for t in src_caches.values()], dtype=np.int64),
                np.array([int(t.size(0)) for t in dst_caches.values()], dtype=np.int64),
                direction,
            )
        except Exception as e:
            _log(f"行数登记失败（守卫失去作用，功能不受影响）：{e!r}")
        return params

    module.build_params = build_params_guarded

    use_batch = os.environ.get("DSH_SIMPLE_BATCH", "") == "1"

    if use_batch:

        def copy_blocks(src_block_ids, dst_block_ids, params):
            if getattr(module, "_batch_fallback", None) is None:
                module._batch_fallback = module._resolve_batch_memcpy()
            fn, _num_attrs = module._batch_fallback
            n = len(src_block_ids)
            if n == 0:
                return
            src_ids = np.asarray(src_block_ids, dtype=np.uint64)
            dst_ids = np.asarray(dst_block_ids, dtype=np.uint64)
            src_all = (
                params.src_bases[:, None] + src_ids[None, :] * params.bpb[:, None]
            ).ravel()
            dst_all = (
                params.dst_bases[:, None] + dst_ids[None, :] * params.bpb[:, None]
            ).ravel()
            sz_all = np.repeat(params.bpb, n)
            total = n * params.num_layers
            max_desc = module._resolve_max_batch_descriptors()
            step = total if max_desc <= 0 else max_desc
            for off in range(0, total, step):
                cnt = min(step, total - off)
                attr_idxs = np.zeros(cnt, dtype=np.uint64)
                err = fn(
                    dst_all[off : off + cnt].ctypes.data,
                    src_all[off : off + cnt].ctypes.data,
                    sz_all[off : off + cnt].ctypes.data,
                    cnt,
                    ctypes.addressof(params.attrs),
                    attr_idxs.ctypes.data,
                    params.num_attrs,
                    ctypes.byref(params.fail_idx),
                    params.stream_handle,
                )
                if err != 0:
                    raise RuntimeError(
                        f"batch memcpy failed: err={err} failIdx={params.fail_idx.value}"
                    )

    else:
        try:
            memcpy_async = _get_memcpy_async()
        except (OSError, AttributeError) as e:
            _log(f"cuMemcpyAsync 解析失败（{e}）⇒ 不钩 copy_blocks（保留上游）")
            setattr(module, _MARKER13, True)
            return

        guard_on = os.environ.get("DSH_SIMPLE_GUARD", "1") != "0"

        def copy_blocks(src_block_ids, dst_block_ids, params):
            n = len(src_block_ids)
            if n == 0:
                return
            if len(dst_block_ids) != n:
                raise ValueError(
                    f"[dsh-simple-rt] copy_blocks: src({n})/dst({len(dst_block_ids)}) 块数不等"
                )
            src_bases = params.src_bases
            dst_bases = params.dst_bases
            bpb = params.bpb
            stream = params.stream_handle
            nl = params.num_layers
            sids = np.asarray(src_block_ids, dtype=np.int64).tolist()
            dids = np.asarray(dst_block_ids, dtype=np.int64).tolist()
            if (sids and min(sids) < 0) or (dids and min(dids) < 0):
                raise ValueError("[dsh-simple-rt] copy_blocks: 块 id 出现负数")
            reg = rows_reg.get(stream) if guard_on else None
            if reg is not None:
                srows, drows, direction = reg
                is_store = direction == "store"
            for li in range(nl):
                sb = int(src_bases[li])
                db = int(dst_bases[li])
                step = int(bpb[li])
                for i in range(n):
                    s_id = sids[i]
                    d_id = dids[i]
                    if reg is not None:
                        srow = int(srows[li])
                        drow = int(drows[li])
                        if s_id >= srow:
                            if is_store:
                                _warn("SKIP", "store-src", i, n, s_id, srow)
                                continue
                            _warn("REDIR", "load-src", i, n, s_id, srow)
                            s_id = 0
                        if d_id >= drow:
                            if is_store:
                                _warn("SKIP", "store-dst", i, n, d_id, drow)
                                continue
                            _warn("REDIR", "load-dst", i, n, d_id, drow)
                            d_id = 0
                    err = memcpy_async(db + d_id * step, sb + s_id * step, step, stream)
                    if err:
                        raise RuntimeError(
                            f"cuMemcpyAsync failed: err={err} (layer={li}/{nl}, i={i}/{n})"
                        )

    module.copy_blocks = copy_blocks
    setattr(module, _MARKER13, True)
    _log("cuda_mem_ops：逐块 cuMemcpyAsync + 越界守卫（#13）")


PATCHES = {
    "vllm.v1.simple_kv_offload.cuda_mem_ops": _patch_cuda_mem_ops,
    "vllm.v1.simple_kv_offload.worker": _patch_worker,
    "vllm.v1.simple_kv_offload.manager": _patch_manager,
}
