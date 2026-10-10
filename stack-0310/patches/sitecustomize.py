"""Runtime-only patches for the stock vLLM 0.31.0 install (site-packages
files are never modified).  Derived from the production vllm-0300 patch set.

Hooks (see docs/design note in repo for full rationale):
  1 models.config            PP>1 relax IF the stock verify ever inspects PP
                             again (0.31.0's is PP-neutral -> arms inertly)
  2 {nvidia,amd}.model_state PP>1 + PLE gate (still fires in 0.31.0)
  3 ngram_embedding          PLE quant resolution: INC-unquantized carryover
                             + NEW compressed-tensors int8 per-row-scale method
  4 weight_utils             safe_open reroute for FakeTensorMode poison
  5 pinned alloc             >60 GiB tables via chunked cuMemHostRegister
  6 {nvidia,amd}.mtp         draft local-invocation first-rank view
  7 kv_cache_utils           PP tensor trimming + projection diagnostics
  8 common.qsa_cache         NEW: ring capacity 12->8 shrink (0.31.0 widened
                             the ring; legacy block 1616 asserts.  Extra ring
                             rows are never read: slot=logical%capacity, a
                             group closes after ratio tokens so the newest
                             token of a closed group sits at slot<8.  Shrink
                             only when stock!=legacy AND legacy|block.)

All wrappers are fail-safe: any doubt -> original code path untouched.
"""

import os
import sys

_TAG = "[rt-patch-0310]"


def _log(msg: str) -> None:
    print(f"{_TAG} {msg}", file=sys.stderr, flush=True)


_SINGLE_PIN_CAP = 60 * (1 << 30)
_PIN_CHUNK = 60 * (1 << 30)
_PIN_PAGE = 4096
_PIN_FLAGS = 0x3  # PORTABLE | DEVICEMAP


def _chain_stock_sitecustomize() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        base = os.path.abspath(entry or os.getcwd())
        if base == here:
            continue
        candidate = os.path.join(base, "sitecustomize.py")
        if not os.path.isfile(candidate):
            continue
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "_qwen38_chain_sitecustomize", candidate
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception:
            pass
        break


def _ple_layers_in_first_stage(vllm_config) -> bool:
    try:
        text_config = vllm_config.model_config.hf_text_config
        ple_ids = list(getattr(text_config, "ple_layer_ids", None) or [])
        pp = vllm_config.parallel_config.pipeline_parallel_size
        if pp <= 1 or not ple_ids:
            return True
        from vllm.distributed.utils import get_pp_indices

        start, end = get_pp_indices(text_config.num_hidden_layers, 0, pp)
        return all(start <= (int(layer_id) - 1) < end for layer_id in ple_ids)
    except Exception:
        return False


def _with_pp_presented_as_one(vllm_config, call):
    pc = vllm_config.parallel_config
    saved = pc.pipeline_parallel_size
    pc.pipeline_parallel_size = 1
    try:
        call()
    finally:
        pc.pipeline_parallel_size = saved


# --------------------------------------------------------------- hook 1
def _patch_model_config(module) -> None:
    cls = getattr(module, "Qwen4ExpForConditionalGenerationConfig", None)
    if cls is None:
        return
    raw = cls.__dict__.get("verify_and_update_config")
    if raw is None:
        return
    orig = raw.__func__ if isinstance(raw, staticmethod) else raw
    if getattr(orig, "_qwen38_rt", False):
        return
    try:
        import inspect

        src = inspect.getsource(orig)
    except Exception:
        src = None
    if src is not None and "pipeline_parallel_size" not in src:
        _log("hook1: stock config verify is PP-neutral in 0.31.0 -> idle")
        return

    def verify_and_update_config(vllm_config) -> None:
        pc = vllm_config.parallel_config
        if pc.pipeline_parallel_size > 1 and _ple_layers_in_first_stage(vllm_config):
            _with_pp_presented_as_one(vllm_config, lambda: orig(vllm_config))
        else:
            orig(vllm_config)

    verify_and_update_config._qwen38_rt = True
    cls.verify_and_update_config = staticmethod(verify_and_update_config)
    _log("hook1 ARMED: config verify PP relax")


# --------------------------------------------------------------- hook 2
def _patch_model_state(module) -> None:
    cls = getattr(module, "Qwen4ExpModelState", None)
    if cls is None:
        return
    orig = cls.__dict__.get("__init__")
    if orig is None or getattr(orig, "_qwen38_rt", False):
        return

    def __init__(self, vllm_config, model, encoder_cache, device, **kw) -> None:
        pc = vllm_config.parallel_config
        if pc.pipeline_parallel_size > 1 and _ple_layers_in_first_stage(vllm_config):
            _log("hook2: PP presented as 1 for model-state gate")
            _with_pp_presented_as_one(
                vllm_config,
                lambda: orig(self, vllm_config, model, encoder_cache, device, **kw),
            )
        else:
            orig(self, vllm_config, model, encoder_cache, device, **kw)

    __init__._qwen38_rt = True
    cls.__init__ = __init__
    _log(f"hook2 ARMED ({cls.__module__})")


# --------------------------------------------------------------- hook 8
def _patch_qsa_ring(module) -> None:
    cls = getattr(module, "QSAKeyStateCache", None)
    if cls is None:
        return
    orig = cls.__dict__.get("get_kv_cache_spec")
    if orig is None or getattr(orig, "_qwen38_rt", False):
        return

    def get_kv_cache_spec(self, vllm_config):
        try:
            ratio = int(self.compress_ratio)
            spec_n = int(vllm_config.num_speculative_tokens or 0)
            blk = int(self.cache_config.block_size)

            def dUp(a, b):
                return -(-a // b)

            if ratio <= 1:
                stock = legacy = 1 + spec_n
            else:
                stock = ratio * (1 + dUp(spec_n, ratio))
                legacy = ratio * dUp(ratio + spec_n, ratio)
            if stock != legacy and legacy > 0 and blk % legacy == 0:
                from vllm.v1.kv_cache_interface import CircularBufferSpec

                _log(f"hook8 SHRANK ring {stock}->{legacy} "
                     f"(ratio={ratio} spec={spec_n} block={blk})")
                return CircularBufferSpec(
                    block_size=legacy,
                    num_kv_heads=1,
                    head_size=self.head_size,
                    head_size_v=0,
                    dtype=self.dtype,
                )
        except Exception as exc:
            _log(f"hook8 passthrough: {exc!r}")
        return orig(self, vllm_config)

    get_kv_cache_spec._qwen38_rt = True
    cls.get_kv_cache_spec = get_kv_cache_spec
    _log("hook8 ARMED (QSA ring shrink when legacy value divides block)")


# ------------------------------------------------------- hooks 3 (+8b)
_INT8_METHOD_CACHE: dict = {}

_DSH_PLE_MMAP = os.environ.get("DSH_PLE_MMAP", "0") == "1"
_PLE_MMAP_CACHE: dict = {}


def _dsh_mlock(addr: int, length: int) -> str:
    """Best-effort mlock of an mmap range (root + unlimited memlock).

    Returns a short status string for logging; never raises. Without this the
    table lives in reclaimable page cache: fast while RAM is free, but the
    kernel may silently drop those pages under memory pressure (then n-gram
    lookups start hitting NVMe again). Locking makes residency guaranteed, and
    as a side effect the pages stop counting as "cache" and start counting as
    used memory -- that is the honest accounting for "PLE 表在内存里".
    """
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.mlock.restype = ctypes.c_int
        libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        rc = libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(length))
        if rc == 0:
            return "locked"
        err = ctypes.get_errno()
        return f"mlock rc={rc} errno={err} (EPERM=1 需要 CAP_IPC_LOCK/memlock 不限; ENOMEM=12 地址区间未映射或超出 RLIMIT)"
    except Exception as exc:  # never break loading over a nicety
        return f"mlock skipped: {exc!r}"


def _ple_mmap_tensor(path: str, rows: int, cols: int, torch_dtype):
    """Read-only zero-copy mmap view [rows, cols] of a flat binary."""
    import numpy as np

    itemsize = {np.dtype(np.int8): 1}.get(np.dtype("int8"), 1)
    lock = os.environ.get("DSH_PLE_MMAP_LOCK", "0") == "1"
    if torch_dtype == __import__("torch").int8:
        arr = np.memmap(path, dtype=np.int8, mode="r", shape=(rows, cols))
        if lock:
            _st = _dsh_mlock(int(arr.ctypes.data), int(arr.nbytes))
            _log(f"PLE mmap mlock int8 {arr.nbytes / 2**30:.2f} GiB -> {_st}")
    else:  # bfloat16 via uint16 raw view
        arr = np.memmap(path, dtype=np.uint16, mode="r", shape=(rows,))
        if lock:
            _st = _dsh_mlock(int(arr.ctypes.data), int(arr.nbytes))
            _log(f"PLE mmap mlock bf16-raw {arr.nbytes / 2**30:.2f} GiB -> {_st}")
        t = __import__("torch").from_numpy(arr)
        return t.view(__import__("torch").bfloat16)
    return __import__("torch").from_numpy(np.asarray(arr))


def _dsh_ple_anon_enabled() -> bool:
    return os.environ.get("DSH_PLE_MEM_RESIDENT", "0") == "1"


def _dsh_read_into_anon(path, np_dtype, shape):
    """[anon-ple 1010] Load a flat artifact into ANONYMOUS resident memory.

    Unlike np.memmap the bytes live in private anonymous pages: they count
    towards process RSS / "used" memory and are never reclaimed by the kernel
    (even mlock()'d file pages stay in the "cache" bucket). Peak RAM ==
    artifact size; reading is parallel sequential pread straight into slices
    of the destination array (no extra buffering)."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    arr = np.empty(shape, dtype=np_dtype)
    # [anon-ple-fix-1010] memoryview(ndarray.data) 继承数组形状，2-D 时
    # mv[a:b] = bytes 抛 NotImplementedError（CPython 只允许 ndim=1）。
    # cast("B") 取同一缓冲的 1-D 字节视图，写入语义等价、无额外拷贝。
    mv = memoryview(arr).cast("B")
    nbytes = arr.nbytes
    CH = 128 << 20
    fd = os.open(path, os.O_RDONLY)
    try:
        offs = list(range(0, nbytes, CH))

        def _rd(o):
            end = min(o + CH, nbytes)
            pos = o
            while pos < end:
                b = os.pread(fd, end - pos, pos)
                if not b:
                    raise RuntimeError(f"short read at {pos} of {path}")
                mv[pos:pos + len(b)] = b
                pos += len(b)
        if len(offs) > 1:
            with ThreadPoolExecutor(max_workers=6) as ex:
                list(ex.map(_rd, offs))
        elif offs:
            _rd(0)
    finally:
        os.close(fd)
    return arr



def _build_int8_method_class(module):
    import torch
    import torch.nn as nn

    from vllm.model_executor.utils import set_weight_attrs

    Base = module.Qwen4ExpPLEEmbeddingMethod

    class DshPLEInt8EmbeddingMethod(Base):
        """int8 PLE table + per-row scale (compressed-tensors int-quantized
        channel strategy, force-unignored by hook3c), dequantized on lookup."""

        requires_device_loading = False

        def create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        ):
            del input_size, output_size
            weight_loader = extra_weight_attrs.get("weight_loader")
            vocab = sum(output_partition_sizes)
            hidden = input_size_per_partition
            if params_dtype is None:
                params_dtype = torch.bfloat16
            weight = nn.Parameter(
                layer.allocate_embedding_weight(vocab, hidden, torch.int8),
                requires_grad=False,
            )
            set_weight_attrs(
                weight,
                {"input_dim": 1, "output_dim": 0, "weight_loader": weight_loader},
            )
            layer.register_parameter("weight", weight)
            if os.environ.get("DSH_PLE_MMAP", "0") == "1":
                # Scale arrives from the flat artifact via the mmap embedding
                # class; no GPU/RAM copy, nothing to load.
                layer._dsh_scale_mmap_only = True
                return
            scale = nn.Parameter(
                torch.ones((vocab, 1), dtype=params_dtype), requires_grad=False
            )
            # Same row-slicing loader as the weight (PLEVocabParallelEmbedding
            # ._load_row_slice handles the trailing singleton dim).
            set_weight_attrs(scale, {"output_dim": 0, "weight_loader": weight_loader})
            layer.register_parameter("weight_scale", scale)

        def process_weights_after_loading(self, layer: nn.Module) -> None:
            if getattr(layer, "_dsh_scale_mmap_only", False):
                _log("PLE int8 method (mmap): storage attached by embedding class")
                return
            ws = getattr(layer, "weight_scale", None)
            if ws is None:
                raise RuntimeError("[rt-patch] PLE int8 checkpoint missing scales")
            if float(ws.data.abs().max()) == 0.0:
                raise RuntimeError("[rt-patch] PLE int8 scales all zero")
            _log(f"PLE int8 method ready: weight={tuple(layer.weight.shape)} "
                 f"{layer.weight.dtype} scale={tuple(ws.data.shape)} "
                 f"mean={float(ws.data.float().mean()):.3e}")

        def dequantize(self, layer, embeddings, output_dtype):
            if getattr(layer, "_dsh_scale_mmap_only", False):
                # CPU-side lookup already applied per-row scales.
                return embeddings.to(output_dtype)
            scale = layer.weight_scale
            if scale.device != embeddings.device:
                raise RuntimeError(
                    "PLE int8 scale/device mismatch: "
                    f"{scale.device} vs {embeddings.device}"
                )
            return embeddings.to(output_dtype) * scale.to(output_dtype)

    return DshPLEInt8EmbeddingMethod


def _defines(cls, name) -> bool:
    return name in vars(cls)


def _patch_ngram_embedding(module) -> None:
    cls = getattr(module, "Qwen4ExpPLEEmbeddingMethod", None)
    if cls is None or not _defines(cls, "from_quant_config"):
        return
    raw = cls.__dict__.get("from_quant_config")

    orig = raw.__func__ if isinstance(raw, staticmethod) else raw
    if getattr(cls.from_quant_config, "_qwen38_rt", False):
        return

    def _dsh_ple_dtype_fallback():
        try:
            import json as _json

            from vllm.config import get_current_vllm_config

            mc = get_current_vllm_config().model_config
            path = os.path.join(str(mc.model), "config.json")
            with open(path) as fh:
                raw_cfg = _json.load(fh)
            return (raw_cfg.get("text_config") or raw_cfg).get("ple_embedding_dtype")
        except Exception:
            return None

    def from_quant_config(quant_config, prefix, embedding_dtype=None):
        if not embedding_dtype:
            embedding_dtype = _dsh_ple_dtype_fallback()
            if embedding_dtype:
                _log(f"PLE embedding_dtype from model config.json: {embedding_dtype}")
        try:
            return orig(quant_config, prefix, embedding_dtype)
        except NotImplementedError:
            # (a) INC / auto-round: honour the per-layer resolver (0.30 carryover)
            if quant_config is not None and hasattr(quant_config, "config_parser"):
                try:
                    import torch.nn as nn

                    resolved = quant_config.config_parser.resolve(nn.Module(), prefix)
                    if not resolved.quantized:
                        _log(f"PLE INC unquantized accepted (bits={resolved.bits}): {prefix}")
                        return module.Qwen4ExpPLEUnquantizedEmbeddingMethod()
                except Exception:
                    pass
                raise
            # (b) compressed-tensors int-quantized: int8 + per-row scale method
            wq = _ct_int8_weight_args(prefix, quant_config)
            if wq is not None:
                _log(f"PLE compressed-tensors int8 (strategy={wq.strategy}) "
                     f"-> DshPLEInt8EmbeddingMethod: {prefix}")
                klass = _INT8_METHOD_CACHE.get(id(module))
                if klass is None:
                    klass = _build_int8_method_class(module)
                    _INT8_METHOD_CACHE[id(module)] = klass
                return klass()
            raise

    from_quant_config._qwen38_rt = True
    setattr(cls, "from_quant_config", from_quant_config)
    _log("hook3 ARMED (INC unquantized + compressed-tensors int8 PLE methods)")
    _patch_pinned_alloc(module)


def _ct_int8_weight_args(prefix, quant_config):
    """Return the static int8 QuantizationArgs covering `prefix` per the
    checkpoint's OWN compressed-tensors rules — bypassing the ignore-list
    entry that (in this checkpoint) excludes the entire .ple. subtree.
    That exclusion governs LINEAR layers; the PLE embedding's actual on-disk
    format is declared by the explicit `re:.*ngram_embedding.*` rule
    (int8 channel weights + bf16 per-row scales), which is exactly what the
    legacy production stack consumed.  Non-CT configs return None untouched."""
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.\
            compressed_tensors import CompressedTensorsConfig
        from compressed_tensors.config import CompressionFormat
        from compressed_tensors.quantization import QuantizationType
    except Exception:
        return None
    if not isinstance(quant_config, CompressedTensorsConfig):
        return None
    if getattr(quant_config, "quant_format", None) != CompressionFormat.int_quantized.value:
        return None
    try:
        import torch.nn as nn

        scheme_dict = quant_config.get_scheme_dict(nn.Module(), prefix)
    except Exception:
        scheme_dict = None
    if scheme_dict is None:
        # Ignored by the coarse .ple rule: retry with the ignore list
        # restricted to patterns that DON'T positively declare this prefix
        # as quantized (i.e. drop only the patterns whose own target rule
        # covers it).  Safe simplification: temporarily clear the list,
        # match the target rule, and judge from the matched weights args.
        saved = quant_config.ignore
        try:
            quant_config.ignore = []
            scheme_dict = quant_config.get_scheme_dict(nn.Module(), prefix)
        except Exception:
            scheme_dict = None
        finally:
            quant_config.ignore = saved
    wq = scheme_dict.get("weights") if scheme_dict else None
    if wq is None or wq.dynamic:
        return None
    if int(wq.num_bits) != 8 or wq.type != QuantizationType.INT:
        return None
    return wq


def _patch_ngram_loader_scale(module) -> None:
    """Route '<leaf>.weight_scale' tensors of the int8 PLE table through the
    same row-sliced loader as the weight itself (shard_k.weight_scale ->
    ngram_embedding.weight_scale with checkpoint_start=k*shard_size)."""
    cls = getattr(module, "Qwen4ExpNGramEmbedding", None)
    if cls is None:
        return
    orig = cls.__dict__.get("load_weights")
    if orig is None or getattr(orig, "_qwen38_rt", False):
        return

    def load_weights(self, weights):
        _mmap_skipped = [0]

        def gen():
            mmap_mode = os.environ.get("DSH_PLE_MMAP", "0") == "1"
            for name, wt in weights:
                if mmap_mode and name.startswith("ngram_embedding.shard_"):
                    # Flat artifact IS the table: never stream 51 GiB
                    # through the loader; claim loaded, skip copy.
                    _mmap_skipped[0] += 1
                    continue
                if (
                    name.startswith("ngram_embedding.shard_")
                    and name.endswith(".weight_scale")
                ):
                    shard_text = name[len("ngram_embedding.shard_"): -len(".weight_scale")]
                    if shard_text.isdigit():
                        yield (
                            "ngram_embedding.weight_scale",
                            wt,
                            int(shard_text),
                        )
                        continue
                yield (name, wt, None)

        out = set()
        pending = []
        for name, wt, shard_index in gen():
            if name == "ngram_embedding.weight_scale":
                emb = self.ngram_embedding
                scale_param = getattr(emb, "weight_scale", None)
                if scale_param is None:
                    # method didn't create one (unquantized/fp8 path): drop
                    continue
                if shard_index is None:
                    scale_param.weight_loader(scale_param, wt)
                    out.add("ngram_embedding.weight_scale")
                    continue
                shard_size = (
                    emb.org_vocab_size + self.split_ngram_parts - 1
                ) // self.split_ngram_parts
                checkpoint_start = shard_index * shard_size
                try:
                    scale_param.weight_loader(
                        scale_param, wt, checkpoint_start=checkpoint_start
                    )
                except TypeError:
                    scale_param.weight_loader(scale_param, wt, checkpoint_start)
                out.add("ngram_embedding.weight_scale")
            else:
                pending.append((name, wt))
        if _mmap_skipped[0]:
            out.add("ngram_embedding.weight")
            out.add("ngram_embedding.weight_scale")
            _log(f"PLE mmap mode: skipped {_mmap_skipped[0]} checkpoint "
                 f"shard tensors (flat artifact is authoritative)")
        if pending:
            out.update(orig(self, iter(pending)))
        return out

    load_weights._qwen38_rt = True
    cls.load_weights = load_weights
    _log("hook3c ARMED (PLE int8 weight_scale row-sliced routing)")




# ------------------------------------------------- DSH PLE mmap storage
_DSH_PLE_REGISTRY: dict = {}


def _dsh_spin_kernel(ptr, want, output_ptr, ids_dummy, BLOCK: int, TIMEOUT: int):
    """(triton) spin until *ptr >= want with timeout, then copy BLOCK bf16.
    Defined via triton lazily inside _dsh_finish to keep import cheap."""


def _make_dsh_join_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _dsh_join_kernel(ids_flag_ptr, done_flag_ptr,
                         TIMEOUT_ITERS: tl.constexpr):
        """Capture/replay-safe join: GPU spins on pinned host counters until
        the gather worker echoes done >= ids. Both sides are LIVE memory reads
        (no frozen scalar args), so graph replay re-evaluates correctly."""
        ids = tl.load(ids_flag_ptr, volatile=True)
        done = tl.load(done_flag_ptr, volatile=True)
        i = 0
        while done < ids and i < TIMEOUT_ITERS:
            i += 1
            done = tl.load(done_flag_ptr, volatile=True)

    return _dsh_join_kernel


def _build_mmap_embed_class(common_mod):
    """Disk-backed PLE table + threaded gather + CAPTURE-SAFE joins.

    No CPU-side events/waits/synchronize anywhere in the forward path:
      * ids D2H  : async copy into pinned ring (capturable, replays)
      * ids flag : GPU writes slot tag into pinned int64 (capturable)
      * worker   : CPU thread polls ids flag, gathers, writes staging,
                   then bumps staging flag (pure CPU)
      * join     : triton kernel spinning on the pinned staging flag
                   (GPU-side wait = capture/replay legal), then H2D copy
    This mirrors the legacy PleOffloadWorker contract on a smaller scale and
    satisfies Model Runner V2's all-ops-capturable requirement."""
    import torch
    import torch.nn as nn

    Pinned = common_mod.Qwen4ExpPLEPinnedHostEmbedding
    Base = common_mod.Qwen4ExpPLEEmbedding
    _dsh_join_kernel = _make_dsh_join_kernel()

    class DshPLEMmapEmbedding(Pinned):
        supports_prefetch = True

        # ---- init -------------------------------------------------
        def __init__(self, num_embeddings, embedding_dim, **kw):
            d = os.environ["DSH_PLE_INT8_DIR"]
            w_path = os.path.join(d, "ple_ngram_int8.bin")
            s_path = os.path.join(d, "ple_ngram_scale.bin")
            os.environ["_DSH_PLE_W_PATH"] = w_path
            Base.__init__(self, num_embeddings, embedding_dim, **kw)
            rows_table = self.weight.shape[0]
            hidden = self.weight.shape[1]
            org_total = int(self.shard_indices.org_vocab_end_index
                            - self.shard_indices.org_vocab_start_index)
            wbytes = os.path.getsize(w_path)
            sbytes = os.path.getsize(s_path)
            art_rows = wbytes // hidden
            if wbytes != art_rows * hidden or sbytes != art_rows * 2:
                raise RuntimeError(f"PLE mmap artifacts inconsistent "
                                   f"({wbytes},{sbytes})")
            if art_rows != rows_table:
                raise RuntimeError(
                    f"PLE mmap artifact rows {art_rows} != table "
                    f"{rows_table} (org {org_total})")
            import numpy as _np
            if _dsh_ple_anon_enabled():
                self._dsh_scale_np = _dsh_read_into_anon(
                    s_path, _np.uint16, (art_rows,))
            else:
                self._dsh_scale_np = _np.memmap(s_path, dtype=_np.uint16,
                                                mode="r", shape=(art_rows,))
            self._dsh_np_w = None
            heads = int(kw.get("num_ngram_heads", 1) or 1)
            self._dsh_heads = heads
            self._output_dim = heads * hidden
            max_rows = int(kw.get("max_total_tokens", 0) or 0)
            max_rows = max(max_rows * self.etp_data_parallel_size, 1)
            dev = torch.device(f"cuda:{torch.cuda.current_device()}")
            self._dsh_lo = self.shard_indices.org_vocab_start_index
            self._dsh_hi = min(self.shard_indices.org_vocab_end_index,
                               art_rows)
            NBUF = 4
            self._dsh_nbuf = NBUF
            self._dsh_max_rows = max_rows
            # pinned rings
            self._dsh_ids_ring = [
                torch.empty(max_rows * heads, dtype=torch.int64,
                            device="cpu", pin_memory=True)
                for _ in range(NBUF)]
            self._dsh_staging = [
                torch.empty(max_rows * heads, hidden,
                            dtype=torch.bfloat16, device="cpu",
                            pin_memory=True)
                for _ in range(NBUF)]
            # per-slot pinned counters: worker must echo done=ids before the
            # join kernel releases; GPU increments ids via UVA copy
            self._dsh_ids_host = torch.zeros(NBUF, dtype=torch.int64,
                                             device="cpu", pin_memory=True)
            self._dsh_done_host = torch.zeros(NBUF, dtype=torch.int64,
                                              device="cpu", pin_memory=True)
            self._dsh_ctr_gpu = torch.zeros(NBUF, dtype=torch.int64,
                                            device=dev)
            self._dsh_n_gpu = torch.zeros(NBUF, dtype=torch.int64,
                                          device=dev)
            self._dsh_n_host = torch.zeros(NBUF, dtype=torch.int64,
                                           device="cpu", pin_memory=True)
            from vllm.utils.torch_utils import (
                get_accelerator_view_from_cpu_tensor)
            self._dsh_ids_uva = get_accelerator_view_from_cpu_tensor(
                self._dsh_ids_host)
            self._dsh_done_uva = get_accelerator_view_from_cpu_tensor(
                self._dsh_done_host)
            self._dsh_pending_slot = 0
            self._dsh_tslot = 0
            self._dsh_cslot = 0
            self._dsh_worker = None
            self._dsh_q = __import__("queue").SimpleQueue()
            _DSH_PLE_REGISTRY[id(self)] = self
            _mode_tag = ("PLE anon-heap storage attached"
                        if _dsh_ple_anon_enabled()
                        else "PLE mmap storage attached")
            _log(f"{_mode_tag}: {wbytes/(1<<30):.1f} GiB from "
                 f"{d}; capture-safe rings {NBUF}x{max_rows*heads} rows"
                 + (" (resident, non-reclaimable)"
                    if _dsh_ple_anon_enabled() else ""))

        def allocate_embedding_weight(self, num_embeddings, embedding_dim,
                                      dtype):
            import numpy as np

            path = os.environ["_DSH_PLE_W_PATH"]
            want = num_embeddings * embedding_dim
            if os.path.getsize(path) != want:
                raise RuntimeError(
                    f"PLE mmap artifact {os.path.getsize(path)} B != "
                    f"table {want} B ({path})")
            if _dsh_ple_anon_enabled():
                art = _dsh_read_into_anon(path, np.int8,
                                          (num_embeddings, embedding_dim))
                return torch.from_numpy(art)
            art = np.memmap(path, dtype=np.int8, mode="r",
                            shape=(num_embeddings, embedding_dim))
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return torch.from_numpy(np.asarray(art))

        # ---- worker lifecycle -------------------------------------
        def _dsh_ensure_worker(self):
            if self._dsh_worker is None:
                import threading

                self._dsh_worker = threading.Thread(
                    target=self._dsh_worker_loop, daemon=True,
                    name="dsh-ple-worker")
                self._dsh_worker.start()

        def _dsh_worker_loop(self):
            import numpy as np
            import time as _time

            ids_f = self._dsh_ids_host
            done_f = self._dsh_done_host
            n_h = self._dsh_n_host
            while True:
                progressed = False
                for slot in range(self._dsh_nbuf):
                    ids = int(ids_f[slot].item())
                    if ids <= int(done_f[slot].item()):
                        continue
                    n = int(n_h[slot].item())
                    progressed = True
                    try:
                        idn = self._dsh_ids_ring[slot][:n].numpy()
                        lo, hi = self._dsh_lo, self._dsh_hi
                        valid = (idn >= lo) & (idn < hi)
                        idx = np.where(valid, idn - lo, 0)
                        prod = self._dsh_gather_bf16(idx, valid)
                        self._dsh_staging[slot][:n].copy_(prod)
                    except Exception as exc:
                        _log(f"PLE worker error: {exc!r}")
                    done_f[slot] = ids
                if not progressed:
                    _time.sleep(0.00004)

        def _dsh_gather_bf16(self, idx, valid):
            import numpy as np
            import torch as _t

            if self._dsh_np_w is None:
                self._dsh_np_w = self.weight.numpy()
            w_rows = self._dsh_np_w[idx]
            s_rows = self._dsh_scale_np[idx]
            prod = _t.from_numpy(np.ascontiguousarray(w_rows)).to(_t.bfloat16)
            prod *= _t.from_numpy(np.ascontiguousarray(s_rows)).view(
                _t.bfloat16).unsqueeze(1)
            if not valid.all():
                prod = _t.where(_t.from_numpy(valid).unsqueeze(1), prod,
                                _t.zeros((), dtype=_t.bfloat16))
            return prod

        # ---- forward-path pieces (all capture-safe) ----------------
        def start_prefetch(self, hidden_states, ngram_ids):
            import torch

            self._dsh_ensure_worker()
            slot_size, _ = self._get_dp_gather_slot(ngram_ids.shape[0])
            gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
            n = gathered_ids.numel()
            slot = self._dsh_tslot
            self._dsh_tslot = (slot + 1) % self._dsh_nbuf

            # Everything below is capturable GPU work; replay is fully
            # self-contained (the worker polls pinned flags, no queue):
            #   ids -> pinned ring -> n & seq published to pinned host
            self._dsh_ids_ring[slot][:n].copy_(
                gathered_ids.reshape(-1), non_blocking=True)
            self._dsh_n_gpu[slot].fill_(n)
            self._dsh_n_host[slot:slot + 1].copy_(
                self._dsh_n_gpu[slot:slot + 1], non_blocking=True)
            self._dsh_ctr_gpu[slot].add_(1)
            self._dsh_ids_uva[slot:slot + 1].copy_(
                self._dsh_ctr_gpu[slot:slot + 1], non_blocking=True)
            self._dsh_pending_slot = slot

        def forward(self, hidden_states):
            import torch

            out = torch.empty(
                hidden_states.shape[0], self._output_dim,
                dtype=torch.bfloat16, device=hidden_states.device)
            slot = self._dsh_pending_slot
            # GPU-side join on pinned counters (capture/replay legal)
            _dsh_join_kernel[(1,)](
                self._dsh_ids_uva[slot:slot + 1],
                self._dsh_done_uva[slot:slot + 1],
                TIMEOUT_ITERS=6_000_000, num_warps=1)
            out.view(-1, self.embedding_dim).copy_(
                self._dsh_staging[slot][:out.shape[0] * self._dsh_heads],
                non_blocking=True)
            return out

        # eager fallback (not used by prefetch path)
        @torch.compiler.disable
        def _lookup(self, input_ids, output=None):
            import numpy as np
            import torch as _t

            expect_shape = (*input_ids.shape, self.embedding_dim)
            rows = input_ids.numel()
            if output is None:
                output = _t.empty(expect_shape, dtype=_t.bfloat16,
                                  device=input_ids.device)
            ids_cpu = input_ids.reshape(-1).to("cpu")
            lo, hi = self._dsh_lo, self._dsh_hi
            idn = ids_cpu.numpy()
            valid = (idn >= lo) & (idn < hi)
            idx = np.where(valid, idn - lo, 0)
            prod = self._dsh_gather_bf16(idx, valid)
            output.view(rows, self.embedding_dim).copy_(prod)
            return output

    return DshPLEMmapEmbedding


def _patch_ple_embed_class(module) -> None:
    if os.environ.get("DSH_PLE_MMAP", "0") != "1":
        return
    target = getattr(module, "Qwen4ExpPLEPinnedHostEmbedding", None)
    if target is None or getattr(target, "_dsh_mmap_class", False):
        return
    common_mod = __import__(
        "vllm.models.qwen4_exp.common.ngram_embedding", fromlist=["x"])
    cls = _build_mmap_embed_class(common_mod)
    cls._dsh_mmap_class = True
    module.Qwen4ExpPLEPinnedHostEmbedding = cls
    _log("DSH PLE mmap class installed (capture-safe threaded storage)")

# --------------------------------------------------------------- hook 5
def _patch_pinned_alloc(module) -> None:
    cls = getattr(module, "Qwen4ExpPLEPinnedHostEmbedding", None)
    if cls is None:
        return
    orig = cls.__dict__.get("allocate_embedding_weight")
    if orig is None or getattr(orig, "_qwen38_rt", False):
        return

    def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
        nbytes = num_embeddings * embedding_dim * dtype.itemsize
        if nbytes <= _SINGLE_PIN_CAP:
            return orig(self, num_embeddings, embedding_dim, dtype)

        import ctypes
        import time

        import torch

        if not torch.cuda.is_initialized():
            torch.cuda.init()
        torch.cuda.set_device(torch.cuda.current_device())

        t0 = time.time()
        weight = torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="cpu")
        ptr = weight.data_ptr()
        start = ptr & ~(_PIN_PAGE - 1)
        end = (ptr + nbytes + _PIN_PAGE - 1) & ~(_PIN_PAGE - 1)

        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuMemHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        lib.cuMemHostRegister.restype = ctypes.c_int
        lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        lib.cuGetErrorString.restype = ctypes.c_int

        off = 0
        chunks = 0
        while off < end - start:
            n = min(_PIN_CHUNK, end - start - off)
            rc = lib.cuMemHostRegister(start + off, n, _PIN_FLAGS)
            if rc != 0:
                p = ctypes.c_char_p()
                lib.cuGetErrorString(rc, ctypes.byref(p))
                raise RuntimeError(
                    f"cuMemHostRegister failed @{off}+{n}: rc={rc} "
                    f"{p.value.decode() if p.value else '?'}"
                )
            off += n
            chunks += 1

        if not weight.is_pinned():
            raise RuntimeError("chunked registration done but is_pinned()==False")
        _log(f"PLE pinned {nbytes / (1<<30):.3f} GiB in {chunks} chunk(s), "
             f"{time.time()-t0:.1f}s")
        return weight

    allocate_embedding_weight._qwen38_rt = True
    setattr(cls, "allocate_embedding_weight", allocate_embedding_weight)
    _log("hook5 ARMED (>60 GiB pinned via chunked cuMemHostRegister)")


# --------------------------------------------------------------- hook 4
_FTM_POISON_MSG = (
    "could not determine the shape of object type 'torch.storage.UntypedStorage'"
)


def _patch_weight_utils(module) -> None:
    if getattr(module, "_qwen38_rt_sw", False):
        return
    module._qwen38_rt_sw = True

    orig_safe_open = getattr(module, "safe_open", None)
    if orig_safe_open is None or getattr(orig_safe_open, "_qwen38_rt", False):
        return

    import contextlib

    def _pop_active_func_modes() -> int:
        n = 0
        try:
            import torch._C as C

            while C._pop_functional_mode():
                n += 1
        except Exception:
            pass
        return n

    class _GuardedSafeOpen:
        """safe_open whose get_tensor retries once with any active torch
        functional/fake modes popped (pytorch#106732 poison).  Proxy object
        (the real safetensors handle forbids attribute monkey-patching)."""

        _qwen38_rt = True

        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs
            self._inner = None

        def __enter__(self):
            outer = self

            class _Proxy:
                def __init__(self, inner):
                    object.__setattr__(self, "_inner", inner)

                def get_tensor(self, name):
                    try:
                        return self._inner.get_tensor(name)
                    except Exception as exc:
                        if _FTM_POISON_MSG not in str(exc):
                            raise
                        popped = _pop_active_func_modes()
                        outer._log_pop(popped, name)
                        if popped == 0:
                            raise
                        return self._inner.get_tensor(name)

                def __getattr__(self, item):
                    return getattr(object.__getattribute__(self, "_inner"), item)

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return self._inner.__exit__(*exc)

            self._proxy = _Proxy(orig_safe_open(*self._args, **self._kwargs))
            return self._proxy

        def __exit__(self, *exc):
            return False

        @staticmethod
        def _log_pop(n, name):
            _log(f"hook4: get_tensor('{name}') poison; dropped {n} mode(s), retried")

    module.safe_open = _GuardedSafeOpen
    _log("hook4 ARMED (weight_utils.safe_open poison retry)")


# --------------------------------------------------------------- hook 6
def _patch_mtp_forward(module) -> None:
    cls = getattr(module, "Qwen4ExpMultiTokenPredictor", None)
    if cls is None:
        return
    orig = cls.__dict__.get("forward")
    if orig is None or getattr(orig, "_qwen38_rt", False):
        return

    import inspect

    _sig = inspect.signature(orig)
    _names = list(_sig.parameters)  # self, input_ids, positions, ...

    class _FirstRankView:
        __slots__ = ("_real",)

        def __init__(self, real):
            self._real = real

        @property
        def is_first_rank(self):
            return True

        def __getattr__(self, name):
            return getattr(self._real, name)

    def forward(self, *args, **kwargs):
        bound = {}
        try:
            bound = dict(zip(_names[1:], args))
            bound.update(kwargs)
        except Exception:
            pass
        local_invocation = (
            bound.get("intermediate_tensors") is None
            and bound.get("hidden_states") is not None
        )
        ns = module.__dict__
        saved = ns["get_pp_group"]
        if local_invocation and not saved().is_first_rank:
            real_group = saved()
            ns["get_pp_group"] = lambda _r=real_group: _FirstRankView(_r)
            try:
                return orig(self, *args, **kwargs)
            finally:
                ns["get_pp_group"] = saved
        return orig(self, *args, **kwargs)

    forward._qwen38_rt = True
    cls.forward = forward
    _log("hook6 ARMED (MTP draft local-invocation first-rank view)")


# --------------------------------------------------------------- hook 7
def _patch_kv_cache_utils(module) -> None:
    import dataclasses as _dc

    orig_build = getattr(module, "get_kv_cache_config_from_groups", None)
    if orig_build is not None and not getattr(orig_build, "_qwen38_rt", False):

        def build(vllm_config, kv_cache_groups, available_memory):
            out = orig_build(vllm_config, kv_cache_groups, available_memory)
            try:
                owned = set()
                for g in kv_cache_groups:
                    owned.update(g.layer_names)
                kept = []
                changed = False
                for t in out.kv_cache_tensors:
                    subs = [n for n in t.layers if n in owned]
                    if len(subs) != len(t.layers):
                        changed = True
                    if subs:
                        kept.append(_dc.replace(t, layers=subs) if len(subs) != len(t.layers) else t)
                    else:
                        changed = True
                if changed:
                    out = _dc.replace(out, kv_cache_tensors=kept)
                    _log(f"hook7: kv tensors trimmed -> {len(kept)} entries")
            except Exception as exc:
                _log(f"hook7 trim skipped: {exc!r}")
            return out

        build._qwen38_rt = True
        module.get_kv_cache_config_from_groups = build
        _log("hook7 ARMED (kv-cache tensor trimming)")

    orig_project = getattr(module, "_project_kv_cache_groups_to_worker", None)
    if orig_project is not None and not getattr(orig_project, "_qwen38_rt", False):

        def project(global_kv_cache_groups, worker_spec):
            out = orig_project(global_kv_cache_groups, worker_spec)
            empties = [i for i, p in enumerate(out) if not p.layer_names]
            if empties:
                _log(f"hook7: projection {len(out)} groups, empty placeholders {empties}")
            return out

        project._qwen38_rt = True
        module._project_kv_cache_groups_to_worker = project




# --------------------------------------------------------------- hook 9
def _patch_speculator_hc(module) -> None:
    """Draft configs spelling the HC widening factor `hc_count` (Qwen4Exp
    checkpoints) instead of `hc_mult`: seed hc_mult before BaseSpeculator
    reads it, else drafter buffers are hc_count x too narrow (09-15 bug)."""
    cls = getattr(module, "BaseSpeculator", None)
    if cls is None:
        return
    orig = cls.__dict__.get("__init__")
    if orig is None or getattr(orig, "_qwen38_rt", False):
        return

    def __init__(self, vllm_config, device) -> None:
        try:
            sc = getattr(vllm_config, "speculative_config", None)
            dmc = getattr(sc, "draft_model_config", None) if sc else None
            hc = getattr(dmc, "hf_config", None) if dmc else None
            if hc is not None and not hasattr(hc, "hc_mult"):
                cnt = int(getattr(hc, "hc_count", 1) or 1)
                if cnt > 1:
                    hc.hc_mult = cnt
                    _log(f"hook9: seeded draft hc_mult={cnt} from hc_count")
        except Exception as exc:
            _log(f"hook9 passthrough: {exc!r}")
        orig(self, vllm_config, device)

    __init__._qwen38_rt = True
    cls.__init__ = __init__
    _log("hook9 ARMED (speculator hc_count fallback)")




# --------------------------------------------------------------- hook 10
def _patch_dist_utils(module) -> None:
    """VLLM_PP_LAYER_PARTITION guard (legacy-image parity): when the env
    partition does not describe THIS model stage (e.g. the MTP draft is a
    pp_size=1 model while the env targets the 48-layer backbone), fall back
    to stock even splitting instead of raising."""
    orig = module.__dict__.get("get_pp_indices")
    if orig is None or getattr(orig, "_qwen38_rt", False):
        return

    def get_pp_indices(num_hidden_layers, pp_rank, pp_size):
        try:
            import vllm.envs as V
            part_str = V.VLLM_PP_LAYER_PARTITION
            if part_str:
                parts = [int(x) for x in part_str.split(",")]
                if len(parts) == pp_size and sum(parts) == num_hidden_layers:
                    start = sum(parts[:pp_rank])
                    return start, start + parts[pp_rank]
                # Mismatch for this model: hide env during the stock call
                # so it uses its own even-split logic instead of raising.
                V.VLLM_PP_LAYER_PARTITION = None
                try:
                    return orig(num_hidden_layers, pp_rank, pp_size)
                finally:
                    V.VLLM_PP_LAYER_PARTITION = part_str
        except Exception as exc:
            _log(f"hook10 passthrough: {exc!r}")
        return orig(num_hidden_layers, pp_rank, pp_size)

    get_pp_indices._qwen38_rt = True
    module.get_pp_indices = get_pp_indices
    _log("hook10 ARMED (pp partition env guard)")




# --------------------------------------------------------------- hook 11
def _patch_worker_kv_alloc(module) -> None:
    """PP empty-placeholder guard at the worker allocation entry.

    Scheduler-side config keeps empty projected groups (ids must align);
    a KVCacheTensor may still name a layer that lives on ANOTHER rank's
    group, and the stock `next(... if layer_name in group.layer_names)`
    then raises StopIteration (0.30-era patch #7 reworked for 0.31.0's
    worker entry).  Drop / trim tensors naming unknown layers before the
    stock allocator runs."""
    fn = module.__dict__.get("allocate_kv_cache")
    if fn is None or getattr(fn, "_qwen38_rt", False):
        return

    def allocate_kv_cache(kv_cache_config, *args, **kwargs):
        try:
            allowed = set()
            for g in kv_cache_config.kv_cache_groups:
                allowed.update(g.layer_names)
            kept = []
            changed = 0
            for t in kv_cache_config.kv_cache_tensors:
                subs = [n for n in t.layers if n in allowed]
                if not subs:
                    changed += 1
                    continue
                if len(subs) != len(t.layers):
                    import dataclasses

                    t = dataclasses.replace(t, layers=subs)
                    changed += 1
                kept.append(t)
            if changed:
                import dataclasses

                kv_cache_config = dataclasses.replace(
                    kv_cache_config, kv_cache_tensors=kept)
                _log(f"hook11: dropped/trimmed {changed} kv-cache tensors "
                     f"naming non-resident layers ({len(kept)} kept)")
        except Exception as exc:
            _log(f"hook11 passthrough: {exc!r}")
        return fn(kv_cache_config, *args, **kwargs)

    allocate_kv_cache._qwen38_rt = True
    module.allocate_kv_cache = allocate_kv_cache
    _log("hook11 ARMED (worker kv-alloc placeholder trim)")




# -------------------------------------------------------------- hook 12
def _patch_splitting_config(module) -> None:
    """Append the DSH PLE ops to the piecewise splitting list after the
    stock default expansion, so Dynamo keeps both ops eager-segmented."""
    if os.environ.get("DSH_PLE_MMAP", "0") != "1":
        return
    cls = getattr(module, "CompilationConfig", None)
    if cls is None:
        return
    orig = cls.__dict__.get("set_splitting_ops_for_v1")
    if orig is None or getattr(orig, "_qwen38_rt", False):
        return

    def wrapped(self, *a, **k):
        r = orig(self, *a, **k)
        try:
            ops = ("vllm::dsh_ple_prefetch", "vllm::dsh_ple_finish")
            if self.splitting_ops:
                added = [o for o in ops if o not in self.splitting_ops]
                if added:
                    self.splitting_ops = list(self.splitting_ops) + added
                    _log(f"hook12: splitting_ops += {added}")
        except Exception as exc:
            _log(f"hook12 passthrough: {exc!r}")
        return r

    wrapped._qwen38_rt = True
    cls.set_splitting_ops_for_v1 = wrapped
    _log("hook12 ARMED (PLE ops -> splitting_ops)")


# --------------------------------------------------------------- assembly
class _PatchFinder:
    def __init__(self, patches):
        self._patches = patches

    def find_spec(self, fullname, path=None, target=None):
        callback = self._patches.get(fullname)
        if callback is None:
            return None
        callbacks = callback if isinstance(callback, tuple) else (callback,)
        from importlib.machinery import PathFinder

        spec = PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=orig_exec, _cbs=callbacks,
                        _n=fullname):
            _orig(module)
            for _cb in _cbs:
                try:
                    _cb(module)
                except Exception as exc:
                    _log(f"FATAL applying patch to {_n}: {exc!r}")
                    raise

        spec.loader.exec_module = exec_module
        return spec


_chain_stock_sitecustomize()

sys.meta_path.insert(
    0,
    _PatchFinder(
        {
            "vllm.model_executor.models.config": _patch_model_config,
            "vllm.models.qwen4_exp.nvidia.model_state": _patch_model_state,
            "vllm.models.qwen4_exp.amd.model_state": _patch_model_state,
            "vllm.models.qwen4_exp.common.qsa_cache": _patch_qsa_ring,
            "vllm.models.qwen4_exp.common.ngram_embedding": _patch_ngram_embedding,
            "vllm.models.qwen4_exp.nvidia.ngram_embedding": (
                _patch_ngram_loader_scale, _patch_ple_embed_class),
            "vllm.v1.worker.gpu.spec_decode.speculator": _patch_speculator_hc,
            "vllm.distributed.utils": _patch_dist_utils,
            "vllm.v1.worker.utils": _patch_worker_kv_alloc,
            "vllm.config.compilation": _patch_splitting_config,
            "vllm.model_executor.model_loader.weight_utils": _patch_weight_utils,
            "vllm.models.qwen4_exp.nvidia.mtp": _patch_mtp_forward,
            "vllm.models.qwen4_exp.amd.mtp": _patch_mtp_forward,
            "vllm.v1.core.kv_cache_utils": _patch_kv_cache_utils,
        }
    ),
)
