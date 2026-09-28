"""rt-patch #13 离线自检（纯 CPU mock，不触 CUDA；握手文件用唯一 cfg 名并清理）。

用例：
  G1 握手收集/陈旧剔除/clamp（含金丝雀：无文件=不动作、n_min>=n_old=不动作）
  G2 manager __init__ 集成：派生 2224 → clamp 到 worker 最窄 2157
  G3 worker _init_cpu_mode 发布文件字段正确
  G4 copy 地址等价 + load 越界重定向 + store 越界跳过（方向按 device 判定）
  G5 UPSTREAM=1 全直通
"""

import dataclasses
import glob
import importlib.util
import json
import os
import sys
import types
from types import SimpleNamespace

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def load_patch(path):
    key = "dsh_simple_offload_rt_v13_t"
    sys.modules.pop(key, None)
    spec = importlib.util.spec_from_file_location(key, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@dataclasses.dataclass
class DTensor:
    size: int
    layers: list = dataclasses.field(default_factory=list)
    layer_stride: int = 0
    block_stride: int = 0
    offset: int = 0


@dataclasses.dataclass
class DConfig:
    num_blocks: int
    kv_cache_tensors: list
    kv_cache_groups: list = dataclasses.field(default_factory=list)


class Device:
    def __init__(self, t):
        self.type = t

    def __str__(self):
        return self.type


class FakeTensor:
    def __init__(self, rows, base, bpb, dev="cpu"):
        self._rows = rows
        self.base = base
        self._bpb = bpb
        self.device = Device(dev)

    def size(self, d):
        return self._rows

    def data_ptr(self):
        return self.base

    def stride(self, d):
        return self._bpb

    def element_size(self):
        return 1


class FakeParams:
    def __init__(self, src_bases, dst_bases, bpb, num_layers, stream_handle):
        import numpy as np

        self.src_bases = np.array(src_bases, dtype=np.uint64)
        self.dst_bases = np.array(dst_bases, dtype=np.uint64)
        self.bpb = np.array(bpb, dtype=np.uint64)
        self.num_layers = num_layers
        self.stream_handle = stream_handle


UNIQ = "unittest-%d" % os.getpid()


def vcfg(cap_dict=None):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_transfer_config=cap_dict or {"m": UNIQ}),
        model_config=SimpleNamespace(model=f"/models/{UNIQ}"),
        parallel_config=SimpleNamespace(world_size=2),
    )


def cleanup_files(p):
    cfg = p._cfg_fingerprint(vcfg())
    for f in glob.glob(f"{p._ROWS_PREFIX}{cfg}.*.json"):
        os.remove(f)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        here, "dsh_simple_offload_rt_v13.py"
    )
    p = load_patch(target)
    cap = 48 * 2**30
    size0 = 16 * 2**30 + 3 * 2**29  # 16.75GiB
    N0 = 776 * cap // size0          # manager 派生块数（rank0 口径）
    N1 = N0 - 67                     # 模拟最窄 rank（PP1）

    # ---------------- G1 ----------------
    print("[G1] 握手收集 / 陈旧剔除 / clamp")
    cfg = p._cfg_fingerprint(vcfg())
    alive = os.getpid()
    for name, pid, rows in (
        (f"{cfg}.e1.json", alive, N0),
        (f"{cfg}.e2.json", 2147483647, 5),    # 不可能存在的 pid=陈旧，必须被忽略
        (f"{cfg}.e3.json", alive, N1),
    ):
        with open(f"{p._ROWS_PREFIX}{name}", "w") as f:
            json.dump({"cfg": cfg, "pid": pid, "rows": rows, "bpb": 1}, f)
    got = p._collect_rows_min(cfg, 2, wait_s=0.5)
    check(f"存活条目 min={N1}（陈旧 pid 被忽略）", got == N1, f"got {got}")
    missing = p._collect_rows_min("nonexistent-hash", 1, wait_s=0.5)
    check("金丝雀：无文件 → None", missing is None)
    clamped = p._clamp_cpu_config(
        DConfig(num_blocks=N0, kv_cache_tensors=[DTensor(size=N0 * 100)]), N1
    )
    check(
        "clamp 块数与张量同步缩",
        clamped.num_blocks == N1 and clamped.kv_cache_tensors[0].size == N1 * 100,
    )
    same = p._clamp_cpu_config(DConfig(num_blocks=N0, kv_cache_tensors=[DTensor(size=100)]), N0 + 800)
    check("金丝雀：n_min>=n_old 不动作", same.num_blocks == N0)

    # ---------------- G2 manager 集成 ----------------
    print("[G2] manager __init__ 集成 clamp")
    p_orig_collect = p._collect_rows_min
    p._collect_rows_min = lambda c, w: p_orig_collect(c, w, wait_s=0.5)

    def make_sched_cls():
        class FakeScheduler:
            def __init__(self, vllm_config, kv_cache_config, cpu_capacity_bytes, **kw):
                self.cpu_kv_cache_config = self._derive_cpu_config(
                    kv_cache_config, cpu_capacity_bytes
                )
                self.num_cpu_blocks = self.cpu_kv_cache_config.num_blocks

            @staticmethod
            def _derive_cpu_config(gpu_config, cap):
                n = max(1, gpu_config.num_blocks * cap // gpu_config.kv_cache_tensors[0].size)
                return dataclasses.replace(
                    gpu_config,
                    num_blocks=n,
                    kv_cache_tensors=[
                        DTensor(size=t.size // gpu_config.num_blocks * n)
                        for t in gpu_config.kv_cache_tensors
                    ],
                )

        return FakeScheduler

    mgrmod = types.ModuleType("mgr")
    FakeScheduler = make_sched_cls()
    mgrmod.SimpleCPUOffloadScheduler = FakeScheduler
    p._patch_manager(mgrmod)
    gpu_cfg = DConfig(num_blocks=776, kv_cache_tensors=[DTensor(size=size0)])
    s = FakeScheduler(vcfg(), gpu_cfg, cap)
    check(f"握手后调度器块数={N1}（原会得 {N0}）", s.num_cpu_blocks == N1, f"got {s.num_cpu_blocks}")
    # 金丝雀：关握手（全新未打钩类）
    os.environ["DSH_SIMPLE_HANDSHAKE"] = "0"
    p2 = load_patch(target)
    p2._collect_rows_min = lambda c, w: p._collect_rows_min(c, w, wait_s=0.5)
    mgr2 = types.ModuleType("mgr2")
    FakeScheduler2 = make_sched_cls()
    mgr2.SimpleCPUOffloadScheduler = FakeScheduler2
    p2._patch_manager(mgr2)
    s2 = FakeScheduler2(vcfg(), gpu_cfg, cap)
    check(f"金丝雀：HANDSHAKE=0 → 保持 {N0}", s2.num_cpu_blocks == N0, f"got {s2.num_cpu_blocks}")
    os.environ.pop("DSH_SIMPLE_HANDSHAKE")

    # ---------------- G3 worker 发布 ----------------
    print("[G3] worker 发布")
    wmod = types.ModuleType("wmod")

    class FakeWorker:
        def __init__(self):
            self.vllm_config = vcfg()
            self.num_cpu_blocks = N1

        def _init_cpu_mode(self, caches, total_bpb, device):
            self._done = True

    wmod.SimpleCPUOffloadWorker = FakeWorker
    p._patch_worker(wmod)
    w = FakeWorker()
    w._init_cpu_mode({}, 23 * 2**20, None)
    files = glob.glob(f"{p._ROWS_PREFIX}{cfg}.*.json")
    pub = [json.load(open(f)) for f in files if json.load(open(f))["bpb"] == 23 * 2**20]
    check("发布文件字段正确", any(d["rows"] == N1 for d in pub), f"files={files}")

    # ---------------- G4 拷贝等价 + 守卫方向 ----------------
    print("[G4] copy 等价 / load 重定向 / store 跳过")
    calls = []

    def fake_memcpy(dst, src, size, stream):
        calls.append((dst, src, size, stream))
        return 0

    p._DSH_TEST_MEMCPY = fake_memcpy
    cmod = types.ModuleType("cm")

    def build_params(src_caches, dst_caches, stream, src_access_order=3):
        return FakeParams(
            [t.data_ptr() for t in src_caches.values()],
            [t.data_ptr() for t in dst_caches.values()],
            [t.stride(0) for t in src_caches.values()],
            len(src_caches),
            id(stream),
        )

    cmod.build_params = build_params
    cmod.copy_blocks = lambda *a: None
    cmod._resolve_batch_memcpy = lambda: (None, 1)
    cmod._resolve_max_batch_descriptors = lambda: 0
    p._patch_cuda_mem_ops(cmod)

    st_load = object()
    cpu = {"a": FakeTensor(2157, 0x1000_0000, 23_230_000, "cpu")}
    gpu = {"a": FakeTensor(776, 0x7000_0000, 23_230_000, "cuda")}
    pl = cmod.build_params(cpu, gpu, st_load)  # load: src=cpu
    calls.clear()
    cmod.copy_blocks([5, 2157], [9, 10], pl)
    bpb = 23_230_000
    ok = (
        len(calls) == 2
        and calls[0] == (0x7000_0000 + 9 * bpb, 0x1000_0000 + 5 * bpb, bpb, pl.stream_handle)
        and calls[1][1] == 0x1000_0000  # 越界 2157 → 行 0，仍入队（load 不跳）
        and calls[1][0] == 0x7000_0000 + 10 * bpb
    )
    check("load 方向：地址等价 + 越界重定向不丢拷贝", ok, f"calls={calls}")

    st_store = object()
    pl2 = cmod.build_params(gpu, cpu, st_store)  # store: src=gpu
    calls.clear()
    cmod.copy_blocks([1, 2], [2157, 7], pl2)  # dst cpu id 2157 越界 → 跳过
    check(
        "store 方向：越界 dst 块被跳过",
        len(calls) == 1 and calls[0][0] == 0x1000_0000 + 7 * bpb,
        f"calls={calls}",
    )
    try:
        cmod.copy_blocks([1, 2], [1], pl2)
        check("块数不等抛错", False)
    except ValueError:
        check("块数不等抛错", True)

    # ---------------- G5 直通 ----------------
    print("[G5] UPSTREAM=1 直通")
    os.environ["DSH_SIMPLE_OFFLOAD_UPSTREAM"] = "1"
    p5 = load_patch(target)
    c5 = types.ModuleType("c5")
    cp5 = build_params
    c5.build_params = cp5
    orig_cb = lambda *a: None
    c5.copy_blocks = orig_cb
    p5._patch_cuda_mem_ops(c5)
    check("cuda_mem_ops 未打钩", c5.copy_blocks is orig_cb and c5.build_params is cp5)
    w5 = types.ModuleType("w5")

    class FW5:
        def _init_cpu_mode(self, a, b, c):
            pass

    w5.SimpleCPUOffloadWorker = FW5
    before = FW5._init_cpu_mode
    p5._patch_worker(w5)
    check("worker 未打钩", FW5._init_cpu_mode is before)
    m5 = types.ModuleType("m5")

    class FS5:
        def __init__(self, *a, **k):
            pass

    m5.SimpleCPUOffloadScheduler = FS5
    bi = FS5.__init__
    p5._patch_manager(m5)
    check("manager 未打钩", FS5.__init__ is bi)
    os.environ.pop("DSH_SIMPLE_OFFLOAD_UPSTREAM")

    cleanup_files(p)
    print(f"\n结果：PASS={PASS} FAIL={FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
