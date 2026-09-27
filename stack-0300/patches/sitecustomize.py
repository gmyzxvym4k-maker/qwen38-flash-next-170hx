"""Runtime-only patches for the stock vLLM 0.30.0 install (site-packages
files are never modified).

Mechanism
---------
run/serve.sh prepends this directory to PYTHONPATH; CPython auto-imports
`sitecustomize` at interpreter start-up, so every process launched from
serve.sh — API server, engine core, all workers (they inherit the
environment) — loads this module.

Design notes
------------
* Patches are applied from a sys.meta_path hook at the moment the target
  module is imported: vLLM/torch are NOT imported from here, so processes
  that never import vLLM are unaffected, and the hook is a no-op for them.

* What is patched — the seven places hit while bringing up PP>1 on the stock
  0.30.0 install:

    1. Qwen4ExpForConditionalGenerationConfig.verify_and_update_config
       (vllm/model_executor/models/config.py:879) — rejects PP>1 because
       "non-first pipeline ranks do not receive the raw input_ids" the
       N-gram PLE lookup needs.
    2. Qwen4ExpModelState.__init__
       (vllm/models/qwen4_exp/{nvidia,amd}/model_state.py:35) — same check.
    3. Qwen4ExpPLEEmbeddingMethod.from_quant_config
       (vllm/models/qwen4_exp/nvidia/ngram_embedding.py:186) — rejects the
       INC (auto-round) quantization config of this checkpoint with
       NotImplementedError, although the checkpoint's own quant config marks
       the whole PLE subsystem as 16-bit float (extra_config ".*ple.*"
       bits=16, PLE tensors are BF16 on disk).  The PLE linears already get
       UnquantizedLinearMethod through INCConfig.get_quant_method's per-layer
       resolver (inc.py:336); only this embedding entry point lacks it.
       Fixed by asking the very same resolver and returning the unquantized
       method iff it says quantized=False; anything else keeps the original
       error.
    4. weight_utils.safe_open (vllm/model_executor/model_loader/
       weight_utils.py) — under an active FakeTensorMode (leaked or not),
       safetensors get_tensor raises
       "could not determine the shape of object type 'torch.storage.
       UntypedStorage'" (pytorch/pytorch#106732).  Diagnostic wrapper: on
       that exact error it dumps the dispatch/function mode stacks plus the
       recorded FakeTensorMode.__enter__ stacks (identifying the leaker),
       then drops the active modes and retries once.  If no mode is active
       the original error is re-raised untouched.
    5. Qwen4ExpPLEPinnedHostEmbedding.allocate_embedding_weight
       (vllm/models/qwen4_exp/nvidia/ngram_embedding.py:430) — the stock
       code pins the whole 95.4 GiB PLE table with one
       torch.empty(pin_memory=True), but this machine's CUDA driver refuses
       SINGLE host allocations above ~64 GiB (measured: 64 GiB OK, 66 GiB
       and 95.4 GiB both cudaErrorMemoryAllocation, while CUMULATIVE pinned
       memory is unbounded — 64+32 GiB in one process succeeded).  Patch:
       allocations <= 60 GiB keep the stock path untouched; larger ones
       allocate one plain CPU tensor and register it with the driver via
       cuMemHostRegister in page-aligned <= 60 GiB chunks
       (PORTABLE|DEVICEMAP), yielding a single contiguous storage with every
       page locked and mapped — which is all is_pinned() and the UVA view
       need.  Verified end-to-end: 95.4 GiB, 2 chunks, is_pinned() True,
       GPU-side round-trip at head/tail/mid/chunk-boundary all correct.
    6. Qwen4ExpMultiTokenPredictor.forward
       (vllm/models/qwen4_exp/{nvidia,amd}/mtp.py) — the MTP draft's
       forward branches on the GLOBAL pipeline group: the first-rank
       branch builds everything from the local hidden_states input, the
       other branch pops hidden_states out of intermediate_tensors that
       some earlier rank should have routed.  In stock 0.30.0 the
       speculator — and with it this draft module, whose layer list is
       unfiltered because the draft parallel config is
       pipeline_parallel_size=1 — exists ONLY on the last pipeline rank
       (model_runner.__init__ creates the speculator under
       is_last_pp_rank), propose() runs only there (_dummy_run returns
       early on non-last ranks; the real path returns right after the
       target forward), and nothing anywhere builds or routes
       intermediate_tensors for the draft.  So on the last rank the
       stock forward always dies at
       "assert intermediate_tensors is not None".  Fix: for a local
       invocation (hidden_states given, intermediate_tensors None) on a
       non-first rank, the module's get_pp_group() is presented as
       first-rank for the duration of that forward call (this module
       consults it in exactly three places: the two branches of this
       forward and Qwen4ExpMTP.__init__ during construction, which runs
       before any forward) while is_last_rank and everything else
       forwards to the real group unchanged, keeping the last-rank
       finalize path stock.  The original binding is restored in a
       finally block.
    7. get_kv_cache_config_from_groups
       (vllm/v1/core/kv_cache_utils.py:1648) — builds KVCacheTensor entries
       for a Uniform group from the group SPEC's dict keys
       (kv_cache_utils:1769-1771) while allocate_kv_cache looks each
       tensor's first layer up in group.layer_names (worker/utils:426).
       Under PP, _project_kv_cache_groups_to_worker:2599 deliberately keeps
       an EMPTY projected group — a positionally aligned placeholder for a
       group whose layers all live on the other rank — but skips projecting
       its UniformTypeKVCacheSpecs, so the full un-projected layer dict
       survives; the builder then emits tensors naming layers this worker
       does not have and the membership lookup raises StopIteration
       (observed on the last rank; the projection keeps empty groups
       because group ids must stay aligned with the scheduler config,
       which is a deep copy of rank 0's config).  Everything before
       allocate tolerates the full dict — first_spec, block sizing,
       init_attn_backend, mamba init and set_attn all completed before the
       observed crash — the pure membership test is the only failure.
       Fix: wrap the builder — after the stock build, trim every tensor to
       the layers the config's own groups claim and drop tensors left
       claiming nothing, logging any change; group placeholders, spec
       dicts, strides, sizes and num_blocks are untouched, so a config
       with no such tensors (e.g. rank 0's) is returned unchanged.  A
       companion wrapper logs empty projected groups for diagnosis.

* Relaxation rule (deliberately NOT a copy of any pre-0.30.0 patch): the
  original checks run unmodified except that pipeline_parallel_size is
  presented as 1 for their duration — audited against 0.30.0, no other code
  in those two call trees reads the field, so this suppresses exactly the
  single raise.  This is safe for our topology because every PLE layer
  (config ple_layer_ids is 1-based -> layer index 1) must fall inside the
  FIRST pipeline stage, the only rank that receives raw input_ids and the
  only rank that builds a PLE layer (later stages build their layers with
  ple=None and return before touching input_ids).  The stage range is
  evaluated with vllm.distributed.utils.get_pp_indices, i.e. it honours the
  real VLLM_PP_LAYER_PARTITION.  If any PLE layer would land outside the
  first stage, the wrapper steps aside and the original error is raised
  unchanged.
"""

import os
import sys

# --------------------------------------------------------------------------
# 1. Preserve the interpreter's stock sitecustomize (e.g. /usr/lib/python3.12/
#    sitecustomize.py installs the apport hook).  Our file shadows it on
#    sys.path, so chain-load the first sitecustomize.py found elsewhere.
# --------------------------------------------------------------------------
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
            pass  # stock hook is best-effort; never break start-up
        break


# --------------------------------------------------------------------------
# 2. Safety predicate: are ALL PLE layers inside the first PP stage?
# --------------------------------------------------------------------------
def _ple_layers_in_first_stage(vllm_config) -> bool:
    """True iff every 1-based ple_layer_ids entry maps into stage 0.

    Conservative: any unexpected state -> False -> stock check (original
    error) runs unchanged.
    """
    try:
        text_config = vllm_config.model_config.hf_text_config
        ple_ids = list(getattr(text_config, "ple_layer_ids", None) or [])
        pp = vllm_config.parallel_config.pipeline_parallel_size
        if pp <= 1 or not ple_ids:
            return True  # stock checks do not fire anyway
        from vllm.distributed.utils import get_pp_indices

        start, end = get_pp_indices(text_config.num_hidden_layers, 0, pp)
        return all(start <= (int(layer_id) - 1) < end for layer_id in ple_ids)
    except Exception:
        return False


def _with_pp_presented_as_one(vllm_config, call):
    """Run `call()` with parallel_config.pipeline_parallel_size shown as 1.

    Restores the real value in all cases.  Used only when
    _ple_layers_in_first_stage() has confirmed the relaxation is sound.
    """
    pc = vllm_config.parallel_config
    saved = pc.pipeline_parallel_size
    pc.pipeline_parallel_size = 1
    try:
        call()
    finally:
        pc.pipeline_parallel_size = saved


# --------------------------------------------------------------------------
# 3. Individual patches (each receives the freshly imported target module)
# --------------------------------------------------------------------------
def _patch_model_config(module) -> None:
    cls = module.Qwen4ExpForConditionalGenerationConfig
    raw = cls.__dict__.get("verify_and_update_config")
    orig = raw.__func__ if isinstance(raw, staticmethod) else raw
    if getattr(orig, "_qwen38_rt", False):
        return

    def verify_and_update_config(vllm_config) -> None:
        pc = vllm_config.parallel_config
        if pc.pipeline_parallel_size > 1 and _ple_layers_in_first_stage(
            vllm_config
        ):
            _with_pp_presented_as_one(vllm_config, lambda: orig(vllm_config))
        else:
            orig(vllm_config)

    verify_and_update_config._qwen38_rt = True
    # Subclasses (Qwen4ExpForCausalLMConfig, Qwen4ExpMTPConfig) and the
    # MODELS_CONFIG_MAP dispatch both resolve this attribute at call time.
    cls.verify_and_update_config = staticmethod(verify_and_update_config)
    print(
        "[rt-patch] Qwen4Exp PP>1 PLE check relaxed (config verify)",
        file=sys.stderr,
        flush=True,
    )


def _patch_model_state(module) -> None:
    cls = module.Qwen4ExpModelState
    orig = cls.__dict__.get("__init__")
    if orig is None or getattr(orig, "_qwen38_rt", False):
        return

    def __init__(self, vllm_config, model, encoder_cache, device) -> None:
        pc = vllm_config.parallel_config
        if pc.pipeline_parallel_size > 1 and _ple_layers_in_first_stage(
            vllm_config
        ):
            _with_pp_presented_as_one(
                vllm_config,
                lambda: orig(self, vllm_config, model, encoder_cache, device),
            )
        else:
            orig(self, vllm_config, model, encoder_cache, device)

    __init__._qwen38_rt = True
    cls.__init__ = __init__
    print(
        f"[rt-patch] Qwen4Exp PP>1 PLE check relaxed (model state: "
        f"{cls.__module__})",
        file=sys.stderr,
        flush=True,
    )


def _patch_ngram_embedding(module) -> None:
    """INC/auto-round: let the PLE embedding resolve like the PLE linears do.

    The stock function only whitelists None / ModelOpt / Fp8 configs and
    raises NotImplementedError for INCConfig even when the checkpoint says
    the PLE tensors are plain 16-bit floats.  We call the original first; it
    runs completely unmodified for every case it supports.  Only when it
    raises, and only for configs exposing INCConfig's per-layer resolver
    (attribute `config_parser`, present solely on INCConfig), do we ask that
    resolver about this prefix — exactly what INCConfig.get_quant_method does
    for the PLE linears — and return the unquantized method when it reports
    quantized=False.  A resolver failure or a genuinely quantized prefix
    re-raises the original NotImplementedError unchanged.
    """
    cls = module.Qwen4ExpPLEEmbeddingMethod
    raw = cls.__dict__.get("from_quant_config")
    orig = raw.__func__ if isinstance(raw, staticmethod) else raw
    if getattr(orig, "_qwen38_rt", False):
        return

    def _dsh_ple_dtype_fallback():
        """配置对象读不到 ple_embedding_dtype 时回退到模型目录的 config.json。

        2026-09-27 实测：引擎在 ngram_embedding.py:698 读
        `getattr(config, "ple_embedding_dtype", None)`，而该字段被 HF 配置类
        （Qwen4ExpTextConfig，Pydantic）解析时丢掉 ⇒ 永远是 None ⇒ 即使 checkpoint
        是 FP8 PLE 也会落到 Unquantized 方法（表按未量化加载、数值错）。
        """
        try:
            import json as _json
            import os as _os

            from vllm.config import get_current_vllm_config

            mc = get_current_vllm_config().model_config
            path = _os.path.join(str(mc.model), "config.json")
            raw = _json.load(open(path))
            return (raw.get("text_config") or raw).get("ple_embedding_dtype")
        except Exception:
            return None

    def from_quant_config(quant_config, prefix, embedding_dtype=None):
        if not embedding_dtype:
            embedding_dtype = _dsh_ple_dtype_fallback()
            if embedding_dtype:
                print(
                    "[rt-patch] PLE embedding_dtype from model config.json: %s"
                    % embedding_dtype,
                    file=sys.stderr,
                    flush=True,
                )
        try:
            return orig(quant_config, prefix, embedding_dtype)
        except NotImplementedError as exc:
            if quant_config is None or not hasattr(quant_config, "config_parser"):
                raise
            try:
                import torch.nn as nn

                resolved = quant_config.config_parser.resolve(nn.Module(), prefix)
            except Exception:
                raise exc from None
            if resolved.quantized:
                raise
            print(
                f"[rt-patch] INC PLE embedding accepted as unquantized "
                f"(bits={resolved.bits}): {prefix}",
                file=sys.stderr,
                flush=True,
            )
            return module.Qwen4ExpPLEUnquantizedEmbeddingMethod()

    from_quant_config._qwen38_rt = True
    cls.from_quant_config = staticmethod(from_quant_config)
    print(
        "[rt-patch] Qwen4Exp PLE embedding: INC config resolved per-layer "
        "instead of rejected",
        file=sys.stderr,
        flush=True,
    )
    _patch_pinned_alloc(module)


# Driver limit measured on this host (see docstring): a single
# cudaHostAlloc/cuMemHostRegister-backed allocation fails above ~64 GiB
# while cumulative pinned memory is unbounded.  Chunk size stays safely
# below the threshold (60 GiB verified OK, 60+35 GiB registered fine).
_SINGLE_PIN_CAP = 60 * (1 << 30)
_PIN_CHUNK = 60 * (1 << 30)
_PIN_PAGE = 4096
# CU_MEMHOSTREGISTER_PORTABLE (0x1) | CU_MEMHOSTREGISTER_DEVICEMAP (0x2):
# portable across contexts (we have 2 GPUs) and mapped for UVA access.
_PIN_FLAGS = 0x3


def _patch_pinned_alloc(module) -> None:
    cls = module.Qwen4ExpPLEPinnedHostEmbedding
    orig = cls.__dict__.get("allocate_embedding_weight")
    if orig is None or getattr(orig, "_qwen38_rt", False):
        return

    def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
        nbytes = num_embeddings * embedding_dim * dtype.itemsize
        if nbytes <= _SINGLE_PIN_CAP:
            # Small table: stock torch.empty(pin_memory=True) is provably
            # fine below the driver cap — behaviour unchanged.
            return orig(self, num_embeddings, embedding_dim, dtype)

        import ctypes
        import time

        import torch

        # cuMemHostRegister needs an initialized driver and a current
        # context; the worker always has both at model-construction time.
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        torch.cuda.set_device(torch.cuda.current_device())

        t0 = time.time()
        weight = torch.empty(
            num_embeddings, embedding_dim, dtype=dtype, device="cpu"
        )
        ptr = weight.data_ptr()
        total = nbytes
        start = ptr & ~(_PIN_PAGE - 1)
        end = (ptr + total + _PIN_PAGE - 1) & ~(_PIN_PAGE - 1)

        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuMemHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        lib.cuMemHostRegister.restype = ctypes.c_int
        lib.cuGetErrorName.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        lib.cuGetErrorName.restype = ctypes.c_int

        chunks = 0
        off = 0
        while off < end - start:
            n = min(_PIN_CHUNK, end - start - off)
            rc = lib.cuMemHostRegister(start + off, n, _PIN_FLAGS)
            if rc != 0:
                p = ctypes.c_char_p()
                lib.cuGetErrorName(rc, ctypes.byref(p))
                name = p.value.decode() if p.value else "?"
                raise RuntimeError(
                    f"[rt-patch] cuMemHostRegister failed at offset "
                    f"{off}+{n} bytes: {rc} {name}"
                )
            off += n
            chunks += 1

        if not weight.is_pinned():
            raise RuntimeError(
                "[rt-patch] chunked registration finished but "
                "tensor.is_pinned() is False — refusing to continue with "
                "unregistered memory (the UVA lookup would fault)"
            )
        print(
            f"[rt-patch] PLE pinned alloc: {total / (1 << 30):.3f} GiB "
            f"registered in {chunks} chunk(s) via cuMemHostRegister in "
            f"{time.time() - t0:.1f}s (single-alloc >~64 GiB unsupported "
            f"by driver)",
            file=sys.stderr,
            flush=True,
        )
        return weight

    allocate_embedding_weight._qwen38_rt = True
    cls.allocate_embedding_weight = allocate_embedding_weight
    print(
        "[rt-patch] Qwen4Exp PLE pinned alloc: >60 GiB tables go through "
        "chunked cuMemHostRegister",
        file=sys.stderr,
        flush=True,
    )


def _patch_weight_utils(module) -> None:
    """Diagnose + survive safetensors get_tensor under an active torch mode.

    An active FakeTensorMode makes safetensors' get_tensor raise
    "could not determine the shape of object type 'torch.storage.
    UntypedStorage'" (reproduced exactly; pytorch/pytorch#106732).  Stock
    0.30.0 wraps model creation AND weight loading in one context, so a mode
    leaked during model construction would hit every get_tensor call.

    Three pieces:
      (a) record the enter-stack of every FakeTensorMode activation
          (bounded deque) so a leak can be attributed to its entering code;
      (b) wrap weight_utils.safe_open — the name the loader calls — and on
          the known error dump the live dispatch/function mode stacks plus
          the recorded enter stacks;
      (c) if modes are actually active, drop them (they are not expected
          anywhere in the load path) and retry get_tensor once.  If nothing
          is active the original error propagates untouched.
    """
    if getattr(module, "_qwen38_rt", False):
        return
    module._qwen38_rt = True

    import traceback as _tb
    from collections import deque

    # (a) instrument FakeTensorMode.__enter__ -----------------------------
    recent_enters = deque(maxlen=16)
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode as _FTM
    except Exception:
        _FTM = None
    if _FTM is not None and not getattr(_FTM.__enter__, "_qwen38_rt", False):
        _orig_ftm_enter = _FTM.__enter__

        def _ftm_enter(self, _orig=_orig_ftm_enter, _rec=recent_enters):
            _rec.append("".join(_tb.format_stack(limit=14)))
            return _orig(self)

        _ftm_enter._qwen38_rt = True
        _FTM.__enter__ = _ftm_enter

    # (b)/(c) wrap safe_open ------------------------------------------------
    orig_safe_open = module.safe_open
    if getattr(orig_safe_open, "_qwen38_rt", False):
        return

    class _Handle:
        __slots__ = ("_inner",)

        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def get_tensor(self, name):
            try:
                return self._inner.get_tensor(name)
            except ValueError as exc:
                if "could not determine the shape of object type" not in str(exc):
                    raise
                import torch.overrides as _ov
                import torch.utils._python_dispatch as _pd

                dstack = list(_pd._get_current_dispatch_mode_stack())
                fstack = list(_ov._get_current_function_mode_stack())
                print(
                    f"[rt-patch] get_tensor({name!r}) hit the mode error; "
                    f"dispatch stack={[type(m).__name__ for m in dstack]} "
                    f"function stack={[type(m).__name__ for m in fstack]}",
                    file=sys.stderr,
                    flush=True,
                )
                for i, stack in enumerate(recent_enters):
                    print(
                        f"[rt-patch] FakeTensorMode enter #{i}:\n{stack}",
                        file=sys.stderr,
                        flush=True,
                    )
                if not dstack and not fstack:
                    raise  # not a mode problem; keep the original error
                while _ov._get_current_function_mode_stack():
                    _ov._pop_mode()
                while _pd._get_current_dispatch_mode_stack():
                    _pd._pop_mode()
                print(
                    "[rt-patch] dropped active modes; retrying get_tensor",
                    file=sys.stderr,
                    flush=True,
                )
                return self._inner.get_tensor(name)

    def safe_open(*args, **kwargs):
        return _Handle(orig_safe_open(*args, **kwargs))

    safe_open._qwen38_rt = True
    module.safe_open = safe_open
    print(
        "[rt-patch] weight_utils.get_tensor mode diagnostics armed",
        file=sys.stderr,
        flush=True,
    )


def _patch_mtp_forward(module) -> None:
    """MTP draft: local last-rank invocations take the first-rank branch.

    (Docstring item 6 has the full rationale.)  Stock
    Qwen4ExpMultiTokenPredictor.forward decides via the global pp group
    whether the hidden_states argument was produced locally (first-rank
    branch) or must be read out of routed intermediate_tensors (other
    branch).  Under 0.30.0 + PP the speculator and this draft module exist
    only on the LAST pipeline rank, propose() only runs there with local
    hidden_states, and the engine has no code to route draft intermediate
    tensors between ranks — so is_first_rank is False and the stock
    forward unconditionally trips the intermediate_tensors assert.

    The wrapper detects a local invocation (intermediate_tensors is None
    and hidden_states is not None) on a non-first rank and, only for the
    duration of that call, swaps the module-global get_pp_group() for a
    view whose is_first_rank is True; is_last_rank and all other
    attributes forward to the real group, so the last-rank finalize keeps
    stock semantics.  get_pp_group is consulted in exactly three places
    in this module (the two forward branches and Qwen4ExpMTP.__init__ at
    construction time), which bounds the swap's effect to the branch
    decision.  Restored in a finally even on error.
    """
    cls = module.Qwen4ExpMultiTokenPredictor
    orig = cls.__dict__.get("forward")
    if orig is None or getattr(orig, "_qwen38_rt", False):
        return

    class _FirstRankView:
        __slots__ = ("_real",)

        def __init__(self, real):
            self._real = real

        @property
        def is_first_rank(self):
            return True

        def __getattr__(self, name):
            return getattr(self._real, name)

    def forward(
        self,
        input_ids,
        positions,
        hidden_states=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        spec_step_idx=0,
    ):
        ns = module.__dict__
        saved_get_pp_group = ns["get_pp_group"]
        local_invocation = (
            intermediate_tensors is None and hidden_states is not None
        )
        if local_invocation and not saved_get_pp_group().is_first_rank:
            real_group = saved_get_pp_group()
            ns["get_pp_group"] = (
                lambda _real=real_group: _FirstRankView(_real)
            )
            try:
                return orig(
                    self,
                    input_ids,
                    positions,
                    hidden_states,
                    intermediate_tensors,
                    inputs_embeds,
                    spec_step_idx=spec_step_idx,
                )
            finally:
                ns["get_pp_group"] = saved_get_pp_group
        return orig(
            self,
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    forward._qwen38_rt = True
    cls.forward = forward
    print(
        "[rt-patch] Qwen4Exp MTP draft forward: local invocation on a "
        "non-first PP rank takes the first-rank branch",
        file=sys.stderr,
        flush=True,
    )


def _rt_repair_kv_cache_tensors(kv_cache_config) -> list[str]:
    """Trim KV-cache tensors to the layers this config's groups claim.

    (Docstring item 7 has the full rationale.)  Returns one line per
    change, empty when nothing changed; the config is only mutated when
    something actually had to go, so an already-consistent config keeps
    its tensor list identity.  Second call is a no-op.
    """
    import dataclasses

    claimed: set[str] = set()
    for group in kv_cache_config.kv_cache_groups:
        claimed.update(group.layer_names)
    kept = []
    report: list[str] = []
    for tensor in kv_cache_config.kv_cache_tensors:
        covered = [n for n in tensor.layers if n in claimed]
        if len(covered) == len(tensor.layers):
            kept.append(tensor)
            continue
        if not covered:
            report.append(f"drop {len(tensor.layers)}: {tensor.layers[:3]}")
            continue
        gone = [n for n in tensor.layers if n not in claimed]
        report.append(f"trim {len(gone)}: {gone[:3]}")
        kept.append(dataclasses.replace(tensor, layers=covered))
    if report:
        kv_cache_config.kv_cache_tensors = kept
    return report


def _patch_kv_cache_utils(module) -> None:
    """PP: never emit KV-cache tensors for layers this worker does not own.

    (Docstring item 7.)  Wraps the config builder so the returned config
    satisfies allocate_kv_cache's invariant — every tensor layer must be
    in some group's layer_names — and wraps the per-worker projection to
    report empty placeholder groups, the stock construction that hands
    the builder an un-projected Uniform spec dict.  Both wrappers carry
    the _qwen38_rt flag, so a repeat call is a no-op.
    """
    orig_build = module.get_kv_cache_config_from_groups
    if not getattr(orig_build, "_qwen38_rt", False):

        def build(vllm_config, kv_cache_groups, available_memory):
            cfg = orig_build(vllm_config, kv_cache_groups, available_memory)
            report = _rt_repair_kv_cache_tensors(cfg)
            if report:
                print(
                    f"[rt-patch] kv-cache tensors repaired ({len(report)}): "
                    f"{report}",
                    file=sys.stderr,
                    flush=True,
                )
            return cfg

        build._qwen38_rt = True
        build._qwen38_orig = orig_build
        module.get_kv_cache_config_from_groups = build
        print(
            "[rt-patch] kv-cache config builder: tensors trimmed to group members",
            file=sys.stderr,
            flush=True,
        )

    orig_project = module._project_kv_cache_groups_to_worker
    if not getattr(orig_project, "_qwen38_rt", False):

        def project(global_kv_cache_groups, worker_spec):
            out = orig_project(global_kv_cache_groups, worker_spec)
            empty = [
                (
                    i,
                    len(src.layer_names),
                    (
                        len(src.kv_cache_spec.kv_cache_specs)
                        if isinstance(
                            src.kv_cache_spec, module.UniformTypeKVCacheSpecs
                        )
                        else -1
                    ),
                    src.layer_names[:4],
                )
                for i, (proj, src) in enumerate(zip(out, global_kv_cache_groups))
                if not proj.layer_names
            ]
            if empty:
                print(
                    f"[rt-patch] kv-cache projection: {len(worker_spec)} layers -> "
                    f"{len(out)} groups, {len(empty)} empty placeholders "
                    f"(idx, global_layers, uniform_spec_size, head)={empty}",
                    file=sys.stderr,
                    flush=True,
                )
            return out

        project._qwen38_rt = True
        project._qwen38_orig = orig_project
        module._project_kv_cache_groups_to_worker = project
        print(
            "[rt-patch] kv-cache group projection armed",
            file=sys.stderr,
            flush=True,
        )


# --------------------------------------------------------------------------
# 4. Import hook: wrap the loader of exactly the modules we patch, so each
#    patch runs right after that module executes and nowhere else.
# --------------------------------------------------------------------------
class _PatchFinder:
    def __init__(self, patches):
        self._patches = patches

    def find_spec(self, fullname, path=None, target=None):
        callback = self._patches.get(fullname)
        if callback is None:
            return None
        from importlib.machinery import PathFinder

        spec = PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=orig_exec, _cb=callback, _n=fullname):
            _orig(module)
            print(f"[rt-patch] patching {_n}", file=sys.stderr, flush=True)
            _cb(module)

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
            "vllm.models.qwen4_exp.nvidia.ngram_embedding": _patch_ngram_embedding,
            "vllm.model_executor.model_loader.weight_utils": _patch_weight_utils,
            "vllm.models.qwen4_exp.nvidia.mtp": _patch_mtp_forward,
            "vllm.models.qwen4_exp.amd.mtp": _patch_mtp_forward,
            "vllm.v1.core.kv_cache_utils": _patch_kv_cache_utils,
        }
    ),
)
