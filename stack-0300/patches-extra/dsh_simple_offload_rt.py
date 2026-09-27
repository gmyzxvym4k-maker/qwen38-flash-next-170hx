"""运行时补丁 · rt-patch #11 —— SimpleCPUOffload 拷贝路径改逐块 cuMemcpyAsync（2026-09-27 深夜崩溃根因修正）。

【#10 的结论被推翻（2026-09-27 实测复现）】
  #10 曾判定根因 = attrIdxs 传标量越界（对齐上游 issue #53860 的解读）。但按
  cuMemcpyBatchAsync 官方契约，attrsIdxs 是 **numAttrs 长度** 的「属性起始块索引」
  数组（docs: "Both attrs and attrsIdxs must be of the same length as specified by
  numAttrs"），上游传 byref(c_size_t(0))（1 个元素，numAttrs=1）本来就合法 ⇒ #10
  实为无操作。实锤：打满 #10 的修复实例（21:06 就绪）21:51 仍同签名 segfault
  （Worker_PP1、栈全在 libcuda 的 cuda-EvtHandlr 驱动线程、无 Xid/MCE/Python 帧），
  与 17:38 / 20:38 两次崩溃同一时间尺度（带二级缓存运行 26~59 分钟）。当日全部
  三次崩溃都发生在 FN_SIMPLE_OFFLOAD>0 的实例上，无 offload 实例稳定 ⇒ 定罪
  cuMemcpyBatchAsync 这条 API 路径本身在本机驱动（610.43.03 + CMP 170HX 定制
  固件）上的稳定性，而非某个入参写法。

【#11 修复】
  copy_blocks 改为逐块 cuMemcpyAsync（经典流有序 DMA API）循环：
  · 不存在任何宿主端「描述符数组」，彻底绕开批量 API 的数组/属性/完成回收机制；
  · 同一 params.stream_handle 上入队，与原有「compute-done 事件 → DMA 流 →
    完成事件」的排程契约完全一致，store/load 语义、线程模型、连接层零改动 ⇒
    内存二级缓存功能原样保留；
  · 单次 launch 开销 ≈1-2µs，本模型 num_layers=1、每 job 数十~数百块，后台
    DMA 线程上不可观测；离线压测 2000 轮 ×16 块 ×64KB（33.5GB）零错误，
    pinned H2D 吞吐 3975MB/s（达标）。

【A/B 回退开关】
  DSH_SIMPLE_BATCH=1      ⇒ 退回批量 API + 零索引数组实现（= #10 行为，仅供取证）。
  DSH_SIMPLE_OFFLOAD_UPSTREAM=1 ⇒ 完全不打钩（上游原码行为）。

挂载点：patches-extra/sitecustomize.py（rt-patch #10 同名文件整版升级）。
"""

from __future__ import annotations

import ctypes
import os
import sys

_MARKER = "_dsh_copy_blocks_v11"
_BATCH_MARKER = "_dsh_attridxs_fixed"  # 兼容 #10 哨兵，避免旧 pyc/旧挂载双钩


def _log(msg: str) -> None:
    sys.stderr.write(f"[dsh-simple-rt] {msg}\n")
    sys.stderr.flush()


def _get_memcpy_async():
    """解析 libcuda 的 cuMemcpyAsync（逐块 DMA，线程安全，流有序）。"""
    lib = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    fn = lib.cuMemcpyAsync
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    return fn


def _make_batch_copy(module, np):
    """#10 旧实现（批量 API + count 长零索引数组），仅供 DSH_SIMPLE_BATCH=1 取证。"""

    def copy_blocks_batch(src_block_ids, dst_block_ids, params):
        n = len(src_block_ids)
        if n == 0:
            return
        if len(dst_block_ids) != n:
            raise ValueError(
                f"[dsh-simple-rt] copy_blocks: src({n})/dst({len(dst_block_ids)}) 块数不等"
            )
        if min(min(src_block_ids), min(dst_block_ids)) < 0:
            raise ValueError("[dsh-simple-rt] copy_blocks: 块 id 出现负数")
        if getattr(module, "_batch_memcpy", None) is None:
            module._batch_memcpy = module._resolve_batch_memcpy()
        fn, _num_attrs = module._batch_memcpy
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

    return copy_blocks_batch


def _patch_cuda_mem_ops(module) -> None:
    if os.environ.get("DSH_SIMPLE_OFFLOAD_UPSTREAM", "") == "1":
        _log("DSH_SIMPLE_OFFLOAD_UPSTREAM=1 ⇒ #11 no-op（保留上游批量 API，仅供取证）")
        return
    if getattr(module, _MARKER, False) or getattr(module, _BATCH_MARKER, False):
        return
    if not hasattr(module, "copy_blocks"):
        _log("cuda_mem_ops 结构不认识（copy_blocks 缺失）⇒ #11 跳过")
        return

    import numpy as np

    use_batch = os.environ.get("DSH_SIMPLE_BATCH", "") == "1"

    if use_batch:
        impl = _make_batch_copy(module, np)
        impl.__name__ = "copy_blocks"
        module.copy_blocks = impl
        setattr(module, _MARKER, True)
        setattr(module, _BATCH_MARKER, True)
        _log("DSH_SIMPLE_BATCH=1 ⇒ 保留批量 API 路径（#10 行为，已知会 segfault，勿用于生产）")
        return

    try:
        memcpy_async = _get_memcpy_async()
    except (OSError, AttributeError) as e:
        _log(f"cuMemcpyAsync 解析失败（{e}）⇒ 回落 #10 批量路径")
        impl = _make_batch_copy(module, np)
        impl.__name__ = "copy_blocks"
        module.copy_blocks = impl
        setattr(module, _MARKER, True)
        setattr(module, _BATCH_MARKER, True)
        return

    def copy_blocks_loop(src_block_ids, dst_block_ids, params):
        """逐块 cuMemcpyAsync 版 copy_blocks（#11 生产实现）。

        与批量版逐字节等价的地址算式：
          addr(layer li, block i) = base[li] ± 0，块内偏移 ids[i] * bpb[li]
        唯一差异 = 一次一个描述符入队（流有序），不再使用 cuMemcpyBatchAsync。
        """
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
        # numpy 预取到 python int，避免循环里反复索引 numpy 标量
        sids = np.asarray(src_block_ids, dtype=np.int64).tolist()
        dids = np.asarray(dst_block_ids, dtype=np.int64).tolist()
        if (sids and min(sids) < 0) or (dids and min(dids) < 0):
            raise ValueError("[dsh-simple-rt] copy_blocks: 块 id 出现负数")
        if nl == 1:
            sb = int(src_bases[0])
            db = int(dst_bases[0])
            step = int(bpb[0])
            for i in range(n):
                err = memcpy_async(db + dids[i] * step, sb + sids[i] * step, step, stream)
                if err:
                    raise RuntimeError(f"cuMemcpyAsync failed: err={err} (i={i}/{n})")
            return
        for li in range(nl):
            sb = int(src_bases[li])
            db = int(dst_bases[li])
            step = int(bpb[li])
            for i in range(n):
                err = memcpy_async(db + dids[i] * step, sb + sids[i] * step, step, stream)
                if err:
                    raise RuntimeError(
                        f"cuMemcpyAsync failed: err={err} (layer={li}/{nl}, i={i}/{n})"
                    )

    copy_blocks_loop.__name__ = "copy_blocks"
    module.copy_blocks = copy_blocks_loop
    setattr(module, _MARKER, True)
    setattr(module, _BATCH_MARKER, True)
    _log("cuda_mem_ops.copy_blocks：批量 cuMemcpyBatchAsync → 逐块 cuMemcpyAsync（#11，绕开批量 API 段错误）")


PATCHES = {
    "vllm.v1.simple_kv_offload.cuda_mem_ops": _patch_cuda_mem_ops,
}
