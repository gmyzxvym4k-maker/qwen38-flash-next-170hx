#!/usr/bin/env python3
"""selftest_simple_rt.py（v11 版）—— rt-patch #11（逐块 cuMemcpyAsync 拷贝路径）离线自检。

纯 CPU / mock：不建 CUDA 上下文。校验三件事：
  G1  钩子可挂载（结构识别 + 哨兵幂等）；
  G2  copy_blocks 的地址算式与 #10 批量版逐字节等价（用假 memcpy 记录入队序列）；
  G3  入参防御（块数不等 / 负 id）；
  G4  DSH_SIMPLE_BATCH=1 退回批量实现；DSH_SIMPLE_OFFLOAD_UPSTREAM=1 不打钩。
用法：PYTHONPATH=patches:patches-extra /home/ll/vllm-env/bin/python selftest_simple_rt.py
真机端到端另见会话记录（双向+乱序+2000 轮浸泡，需 GPU，挂载态直测）。
"""
import ctypes
import importlib
import os
import sys

import numpy as np

FAILS = []


def check(name, cond, detail=""):
    print(("PASS" if cond else "FAIL"), name, detail)
    if not cond:
        FAILS.append(name)


class FakeParams:
    def __init__(self, src_bases, dst_bases, bpb, stream=0x1234):
        self.src_bases = np.asarray(src_bases, dtype=np.uint64)
        self.dst_bases = np.asarray(dst_bases, dtype=np.uint64)
        self.bpb = np.asarray(bpb, dtype=np.uint64)
        self.num_layers = len(self.src_bases)
        self.stream_handle = stream
        self.attrs = ctypes.c_uint(3)
        self.attrs_idx = ctypes.c_size_t(0)
        self.num_attrs = 1
        self.fail_idx = ctypes.c_size_t(0)


class FakeModule:
    """模仿 cuda_mem_ops 的可挂载面（copy_blocks 原函数 + _resolve_batch_memcpy）。"""

    def __init__(self):
        self._batch_memcpy = None

    def copy_blocks(self, *a):  # 原版占位（不应被调用）
        raise AssertionError("原版 copy_blocks 被调用 ⇒ 钩子未生效")

    def _resolve_batch_memcpy(self):
        def fake_fn(*args):
            FAKE_BATCH_CALLS.append(args)
            return 0
        fake_fn.attrs_entry = object()
        return fake_fn, 1

    def _resolve_max_batch_descriptors(self):
        return 0


FAKE_BATCH_CALLS = []


def fresh_rt():
    import dsh_simple_offload_rt as rt
    return importlib.reload(rt)


def ref_descriptors(params, src_ids, dst_ids):
    """#10 批量版的地址算式（参照真值）：逐层 × 逐块展开。"""
    s, d, z = [], [], []
    for li in range(params.num_layers):
        base_s = int(params.src_bases[li])
        base_d = int(params.dst_bases[li])
        step = int(params.bpb[li])
        for i in range(len(src_ids)):
            s.append(base_s + src_ids[i] * step)
            d.append(base_d + dst_ids[i] * step)
            z.append(step)
    return s, d, z


def g1_mount():
    rt = fresh_rt()
    mod = FakeModule()
    rt._patch_cuda_mem_ops(mod)
    check("G1 挂载哨兵", getattr(mod, "_dsh_copy_blocks_v11", False) is True)
    n_before = mod.copy_blocks
    rt._patch_cuda_mem_ops(mod)  # 幂等
    check("G1 幂等（二次挂载不换实现）", mod.copy_blocks is n_before)


def g2_addr_equivalence():
    rt = fresh_rt()
    mod = FakeModule()
    # 把 libcuda 换成假实现：记录 (dst, src, size, stream) 序列
    recorded = []

    class FakeLib:
        def __getattr__(self, name):
            assert name == "cuMemcpyAsync"
            def fn(dst, src, size, stream):
                recorded.append((dst, src, size, stream))
                return 0
            return fn

    orig_cdll = ctypes.CDLL
    ctypes.CDLL = lambda *a, **k: FakeLib()
    try:
        rt._patch_cuda_mem_ops(mod)
    finally:
        ctypes.CDLL = orig_cdll
    # 单层
    p1 = FakeParams([0x10000000], [0x40000000], [4096])
    sids = [0, 5, 9]; dids = [3, 1, 7]
    recorded.clear(); mod.copy_blocks(sids, dids, p1)
    s, d, z = ref_descriptors(p1, sids, dids)
    got = [(dd, ss, sz, p1.stream_handle) for dd, ss, sz, _ in recorded]
    check("G2 单层地址算式与批量版等价",
          got == list(zip(d, s, z, [p1.stream_handle]*len(z))),
          f"n={len(got)}")
    # 多层
    p2 = FakeParams([0x1000, 0x20000], [0x90000, 0xA0000], [8192, 16384])
    recorded.clear(); mod.copy_blocks([1, 2], [5, 6], p2)
    s, d, z = ref_descriptors(p2, [1, 2], [5, 6])
    check("G2 多层算式等价", recorded == list(zip(d, s, z, [p2.stream_handle]*len(z))))
    # 空输入
    recorded.clear(); mod.copy_blocks([], [], p2)
    check("G2 空输入零调用", not recorded)


def g3_guards():
    rt = fresh_rt()
    mod = FakeModule()
    class FakeLib2:
        def __getattr__(self, name):
            def fn(*a): return 0
            return fn
    orig = ctypes.CDLL; ctypes.CDLL = lambda *a, **k: FakeLib2()
    try:
        rt._patch_cuda_mem_ops(mod)
    finally:
        ctypes.CDLL = orig
    p = FakeParams([1 << 20], [1 << 30], [4096])
    try:
        mod.copy_blocks([0, 1], [0], p); ok1 = False
    except ValueError:
        ok1 = True
    try:
        mod.copy_blocks([-1], [0], p); ok2 = False
    except ValueError:
        ok2 = True
    check("G3 块数不等 ValueError", ok1)
    check("G3 负 id ValueError", ok2)


def g4_switches():
    mod = FakeModule()
    os.environ["DSH_SIMPLE_OFFLOAD_UPSTREAM"] = "1"
    try:
        rt = fresh_rt(); rt._patch_cuda_mem_ops(mod)
        check("G4 UPSTREAM=1 不打钩", mod.copy_blocks.__name__ == "copy_blocks" and not getattr(mod, "_dsh_copy_blocks_v11", False))
    finally:
        del os.environ["DSH_SIMPLE_OFFLOAD_UPSTREAM"]
    mod2 = FakeModule()
    os.environ["DSH_SIMPLE_BATCH"] = "1"
    try:
        rt = fresh_rt()
        # 批量路径需要 module._batch_memcpy 惰性解析——用 fake 顶替真实 libcuda 依赖
        rt._patch_cuda_mem_ops(mod2)
        p = FakeParams([1024], [2048], [16])
        FAKE_BATCH_CALLS.clear()
        mod2.copy_blocks([0, 1], [2, 3], p)
        check("G4 BATCH=1 走批量实现（零索引数组长度=count）",
              len(FAKE_BATCH_CALLS) == 1 and FAKE_BATCH_CALLS[0][3] == 2)
    finally:
        del os.environ["DSH_SIMPLE_BATCH"]


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "patches-extra"))
    g1_mount(); g2_addr_equivalence(); g3_guards(); g4_switches()
    print("=" * 40)
    if FAILS:
        print("SELFTEST FAIL:", FAILS); sys.exit(1)
    print("SELFTEST PASS（v11 全部用例）")
