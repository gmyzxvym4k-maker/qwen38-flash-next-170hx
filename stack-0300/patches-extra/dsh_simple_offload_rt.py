"""运行时补丁 · rt-patch #10 —— SimpleCPUOffload attrIdxs 越界 UB 修复（2026-09-27 崩溃根因）。

根因（上游 github.com/vllm-project/vllm issue #53860，本机代码逐字命中）：
  vllm/v1/simple_kv_offload/cuda_mem_ops.py 的 copy_blocks() 把
  ctypes.byref(params.attrs_idx) —— 单个 c_size_t 标量 —— 当作 attrIdxs 传给
  cuMemcpyBatchAsync。该 API 契约要求 attrIdxs 是 **count 个元素** 的 size_t 数组
  （每个拷贝描述符一个属性索引，值 < numAttrs）。驱动会读 attrIdxs[0..cnt-1]，
  越过 8 字节标量之后全是堆上相邻随机字节；命中非零值 → attrs[垃圾索引] 越界取
  CUmemcpyAttributes → 间歇性原生 segfault。
  与本机事故完全吻合：09-27 两次崩溃（17:38:35 / 20:38:30）都是**开了二级缓存的
  实例**、都死在 PP1 worker、崩溃栈只有 glibc pthread 帧（`Segfault encountered`
  → VllmWorker-1 died unexpectedly, exit code None）、无 GPU Xid 前导、无 MCE。
  CUDA 侧 num_attrs 恒为 1（见 _resolve_batch_memcpy），attrIdxs 必然被消费，
  本崩溃面在 CUDA 上同样成立（上游原文确认 "the out-of-bounds read is live there too"）。

修复（与上游 #53860 PR 同语义，最小改动）：
  每次调用显式分配 np.zeros(cnt, uint64) 索引数组 —— 每描述符仍用 attrs[0]，
  与原实现"期望值"完全一致，只是不再越界读。拷贝对象、流、事件排程、线程模型
  一概不动 ⇒ **内存二级缓存功能原样保留**。

加固（本地新增，非上游）：
  ① src/dst 块数不等直接 ValueError（原实现会让驱动拿着 count 读更短的 dst 数组，
    同样是越界读）；块 id 负数经 uint64 环绕会撞出天文地址，一并拦截。

回滚开关：DSH_SIMPLE_OFFLOAD_UPSTREAM=1 ⇒ 不打钩（回到上游原行为，仅供 A/B 取证）。

挂载点：patches-extra/sitecustomize.py 把 PATCHES 并进 _PATCHES；
  钩子在 cuda_mem_ops 模块 exec 完成后、copy_backend 的 from-import 绑定之前替换
  module.copy_blocks ⇒ DmaCopyBackend 后台线程拿到的就是修复版。
"""

from __future__ import annotations

import ctypes
import os
import sys

_MARKER = "_dsh_attridxs_fixed"


def _log(msg: str) -> None:
    sys.stderr.write(f"[dsh-simple-rt] {msg}\n")
    sys.stderr.flush()


def _patch_cuda_mem_ops(module) -> None:
    if os.environ.get("DSH_SIMPLE_OFFLOAD_UPSTREAM", "") == "1":
        _log("DSH_SIMPLE_OFFLOAD_UPSTREAM=1 ⇒ #10 no-op（保留上游越界 UB，仅供取证）")
        return
    if getattr(module, _MARKER, False):
        return

    import numpy as np

    if not hasattr(module, "copy_blocks") or not hasattr(module, "_resolve_batch_memcpy"):
        _log("cuda_mem_ops 结构不认识（copy_blocks/_resolve_batch_memcpy 缺失）⇒ #10 跳过")
        return

    def copy_blocks_fixed(src_block_ids, dst_block_ids, params):
        """与上游 copy_blocks 同行为；差异仅在 attrIdxs 传 count 个元素的零数组。"""
        n = len(src_block_ids)
        if n == 0:
            return
        # ---- 本地加固：入参合法性（防御性，正常调度永不会触发）----
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
            # ★ 修复核心：契约要求 count 个 size_t；全零 = 每描述符用 attrs[0]，
            #   与原代码唯一意图一致，且不再越界读标量后的堆字节。
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

    copy_blocks_fixed.__name__ = "copy_blocks"
    module.copy_blocks = copy_blocks_fixed
    setattr(module, _MARKER, True)
    _log("cuda_mem_ops.copy_blocks：attrIdxs 单标量→count 元素零数组（修复 #53860 越界 UB）")


PATCHES = {
    "vllm.v1.simple_kv_offload.cuda_mem_ops": _patch_cuda_mem_ops,
}
