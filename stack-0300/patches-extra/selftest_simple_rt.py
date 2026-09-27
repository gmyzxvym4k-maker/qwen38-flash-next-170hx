"""rt-patch #10 离线自检（无 GPU 依赖）：
  G1  cuda_mem_ops.copy_blocks 为修复版（导入链或手动挂钩后，标记位+docstring 双判据）；
  G2  假 fn 捕获参数：attrIdxs 是 count 个可读的 uint64、全零；
      描述符地址/长度数组与上游公式逐值一致；
  G4  本地加固：src/dst 长度不等、负块 id → ValueError；空列表短路；
  G5  回滚开关 DSH_SIMPLE_OFFLOAD_UPSTREAM=1 → reload 后不动作；
  G6  copy_backend 绑定传导：模块重载后 from-import 拿到的仍是修复版。
运行：PYTHONPATH=patches:patches-extra /home/ll/vllm-env/bin/python selftest_simple_rt.py
（注意：PYTHONPATH 链路上 sitecustomize 已把钩子挂进真实导入，G1 允许"已修复"态。）
"""
import ctypes
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.pop("DSH_SIMPLE_OFFLOAD_UPSTREAM", None)

import numpy as np  # noqa: E402

import vllm.v1.simple_kv_offload.cuda_mem_ops as m  # noqa: E402
import dsh_simple_offload_rt  # noqa: E402

FIX_MARK = "attrIdxs 传 count 个元素的零数组"


def is_fixed(fn):
    return FIX_MARK in (getattr(fn, "__doc__", "") or "")


# ---- G1 ----
assert getattr(m, "_dsh_attridxs_fixed", False) and is_fixed(m.copy_blocks), (
    "G1 失败：导入链上钩子未生效（检查 sitecustomize 接入与 _PATCHES 键名）"
)
dsh_simple_offload_rt._patch_cuda_mem_ops(m)  # 幂等复跑不应出错
assert getattr(m, "_dsh_attridxs_fixed", False), "G1 幂等失败"


class P:
    pass


def _mk_params():
    p = P()
    p.src_bases = np.array([1000], dtype=np.uint64)
    p.dst_bases = np.array([5_000_000], dtype=np.uint64)
    p.bpb = np.array([4096], dtype=np.uint64)
    p.num_layers = 1
    p.attrs = m._CUmemcpyAttributes(srcAccessOrder=3)
    p.attrs_idx = ctypes.c_size_t(0)
    p.num_attrs = 1
    p.fail_idx = ctypes.c_size_t(0)
    p.stream_handle = 0
    return p


calls = []


def fake_fn(dsts, srcs, sizes, count, attrs, attr_idx, num_attrs, fail_idx, stream):
    arr = (ctypes.c_uint64 * count).from_address(attr_idx)
    assert all(arr[i] == 0 for i in range(count)), "attrIdxs 存在非零索引"
    d = (ctypes.c_uint64 * count).from_address(dsts)
    s = (ctypes.c_uint64 * count).from_address(srcs)
    z = (ctypes.c_uint64 * count).from_address(sizes)
    calls.append({"count": count, "num_attrs": num_attrs, "d": list(d), "s": list(s), "z": list(z)})
    return 0


# ---- G2 ----
p = _mk_params()
m._batch_memcpy = (fake_fn, 1)
m.copy_blocks([0, 1, 2], [7, 8, 9], p)
assert len(calls) == 1, calls
c = calls[0]
assert c["count"] == 3 and c["num_attrs"] == 1
assert c["s"] == [1000, 1000 + 4096, 1000 + 8192], c["s"]
assert c["d"] == [5_000_000 + 7 * 4096, 5_000_000 + 8 * 4096, 5_000_000 + 9 * 4096], c["d"]
assert c["z"] == [4096, 4096, 4096]

# ---- G4 ----
try:
    m.copy_blocks([0, 1], [7], p)
    raise SystemExit("FAIL G4: 长度不等未拦截")
except ValueError:
    pass
try:
    m.copy_blocks([0, -1], [7, 8], p)
    raise SystemExit("FAIL G4: 负 id 未拦截")
except ValueError:
    pass
n_before = len(calls)
m.copy_blocks([], [], p)
assert len(calls) == n_before, "FAIL G4: 空列表未短路"

# ---- G5/G6：必须新进程验证（reload 不清命名空间，测不出 fresh-import）----
import subprocess

CHILD = (
    "import importlib, vllm.v1.simple_kv_offload.cuda_mem_ops as m, "
    "vllm.v1.simple_kv_offload.copy_backend as cb; "
    "print('MARK=%s FIXED=%s BIND=%s' % ("
    "int(getattr(m, '_dsh_attridxs_fixed', False)), "
    "int('attrIdxs 传 count 个元素的零数组' in (m.copy_blocks.__doc__ or '')), "
    "int(cb.copy_blocks is m.copy_blocks)))"
)

env_on = dict(os.environ, DSH_SIMPLE_OFFLOAD_UPSTREAM="1")
r5 = subprocess.run([sys.executable, "-c", CHILD], env=env_on, capture_output=True, text=True, timeout=300)
out5 = r5.stdout.strip().splitlines()[-1] if r5.stdout else ""
assert "MARK=0 FIXED=0" in out5, f"FAIL G5 回退开关失效: {out5} {r5.stderr[-300:]}"

env_off = {k: v for k, v in os.environ.items() if k != "DSH_SIMPLE_OFFLOAD_UPSTREAM"}
r6 = subprocess.run([sys.executable, "-c", CHILD], env=env_off, capture_output=True, text=True, timeout=300)
out6 = r6.stdout.strip().splitlines()[-1] if r6.stdout else ""
assert "MARK=1 FIXED=1 BIND=1" in out6, f"FAIL G6 绑定传导异常: {out6} {r6.stderr[-300:]}"

print("SELFTEST OK: G1/G2/G4 进程内 + G5 回退开关 + G6 绑定传导（fresh-import 子进程）")
