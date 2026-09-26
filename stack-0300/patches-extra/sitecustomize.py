"""运行时补丁·本地扩展（rt-patch #8）—— 不改上游 patches/sitecustomize.py 一个字。

加载方式：PYTHONPATH 里本目录排在上游补丁目录**之后**：

    PYTHONPATH=/home/ll/deploy/vllm-0300/patches:/home/ll/deploy/vllm-0300/patches-extra

上游 sitecustomize.py 在装它自己的 8 个钩子之前会调用 _chain_stock_sitecustomize()：按
sys.path 顺序取「除自己目录以外」的第一个 sitecustomize.py 并执行 —— 就是我们的本文件。
所以这是上游设计内的扩展点，上游文件保持与参考仓库逐字节一致（可用 sha256 自证：
9b84700fee6dbe3da9da9b6f2f071bcb6049926a9acc14199927d15185de29b7）。

两个必须留意的机制细节：
1. 防回环：上游会链进本文件，本文件若照抄"链到下一个 sitecustomize"就会又指回上游目录，
   形成无限递归。⇒ 续链时跳过整棵 vllm-0300 补丁树，并用环境变量哨兵做幂等保护。
2. 今天这条链上其实没有第三方 sitecustomize.py（09-30 扫描 sys.path 全部条目：无），
   续链纯属防御 —— 将来 pip 包（如 nvidia_cutlass_dsl）往 site-packages 放一个时不会被
   我们的 PYTHONPATH 静默吃掉。

补丁内容（对照旧 chroot 定制镜像栈的同名能力）：
  #8 vllm.renderers.base.BaseRenderer._warmup_mm_processor
     —— VLLM_SKIP_MM_WARMUP=1 时直接返回，跳过本模型用不到的多模态处理器 warmup。
     旧栈里这是镜像内补丁 11-renderers__base.py（+import os + 早退），实测省 25~33 s/启动。
     官方 0.30.0 里该 warmup 还可能被 start_mm_warmup_in_background 放到后台线程、
     在首个请求前 join（v1/engine/async_llm.py:167）⇒ 省下的时间从"启动"变成"首请求"，
     两种形态下跳过都是净收益。代价：启动后第一次图片请求要现场初始化 mm 处理器
     （18420 只跑文本/工具调用，无图片流量）。
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys

_MARKER = "DSH_RT_EXTRA_LOADED"
_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
# 整棵补丁树都不可回头续链（上游目录是本目录的同级）
_PATCHES_ROOT = os.path.dirname(_SELF_DIR)

def _log(msg: str) -> None:
    sys.stderr.write(f"[rt-patch-extra] {msg}\n")
    sys.stderr.flush()


def _chain_external_sitecustomize() -> None:
    """把链子接下去：加载补丁树以外的第一个 sitecustomize.py（若存在）。"""
    for entry in sys.path:
        try:
            base = os.path.abspath(entry or os.getcwd())
            if base == _SELF_DIR or base.startswith(_PATCHES_ROOT + os.sep):
                continue
            candidate = os.path.join(base, "sitecustomize.py")
            if not os.path.isfile(candidate):
                continue
            spec = importlib.util.spec_from_file_location(
                "_dsh_chain_sitecustomize", candidate
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _log(f"已续链加载外部 sitecustomize：{candidate}")
        except Exception as exc:  # 防御性续链，坏了不能拖垮启动
            _log(f"续链失败（忽略）：{type(exc).__name__}: {exc}")
        break


# --------------------------------------------------------------------------
# rt-patch #8：跳过多模态 processor warmup
# --------------------------------------------------------------------------
def _patch_renderers_base(module) -> None:
    if os.environ.get("VLLM_SKIP_MM_WARMUP", "") != "1":
        _log("VLLM_SKIP_MM_WARMUP 未置 1 ⇒ #8 不动作（多模态 warmup 照常）")
        return
    cls = getattr(module, "BaseRenderer", None)
    if cls is None or not hasattr(cls, "_warmup_mm_processor"):
        _log("renderers.base：没找到 BaseRenderer._warmup_mm_processor ⇒ #8 跳过")
        return
    if getattr(cls, "_dsh_mm_warmup_skipped", False):
        return

    def _warmup_mm_processor(self, processor, *, log_prefix: str = "Multi-modal") -> None:
        # 与定制镜像补丁 11 同语义：不构造 dummy mm inputs，直接返回。
        return

    cls._warmup_mm_processor = _warmup_mm_processor
    cls._dsh_mm_warmup_skipped = True
    _log("多模态 warmup：VLLM_SKIP_MM_WARMUP=1 ⇒ _warmup_mm_processor 早退（省 25~33 s）")


class _PatchFinder(importlib.abc.MetaPathFinder):
    """与上游同一套机制，但只处理本文件的模块名；上游 8 个钩子不受影响。"""

    def __init__(self, patches):
        self._patches = patches

    def find_spec(self, fullname, path=None, target=None):
        callback = self._patches.get(fullname)
        if callback is None or target is not None:
            return None
        from importlib.machinery import PathFinder

        spec = PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        origin_exec = spec.loader.exec_module

        def exec_module(module, _orig=origin_exec, _cb=callback, _n=fullname):
            _orig(module)
            try:
                _cb(module)
            except Exception as exc:  # 补丁失败绝不炸掉导入链路
                _log(f"{_n} 打补丁失败：{type(exc).__name__}: {exc}")

        spec.loader.exec_module = exec_module
        return spec


_PATCHES = {
    "vllm.renderers.base": _patch_renderers_base,
}

# --------------------------------------------------------------------------
# rt-patch #9：KV 二级缓存（OffloadingConnector）PP2 移植钩子（2026-10-05）
# dsh_kvoff_rt.PATCHES 只挂 vllm 的 offloading / kv_offload.cpu 模块——不配
# --kv-transfer-config（FN_KVOFF=0 缺省）时这些模块不被导入，钩子天然惰性，
# 对默认生产零影响。移植语义对照旧栈 c1/c2/c6/c7（c3/c5a 上游已吸收，见模块
# docstring）。紧急总闸：DSH_KVOFF_RT_DISABLE=1 使各回调 no-op。
# --------------------------------------------------------------------------
try:
    import dsh_kvoff_rt

    _PATCHES.update(dsh_kvoff_rt.PATCHES)
except Exception as _kvexc:  # 加载失败绝不拖垮启动，但必须出声
    _log(f"dsh_kvoff_rt 加载失败（KV 二级缓存钩子未激活）：{_kvexc}")

# 幂等哨兵：用 sys 模块属性，**不要用环境变量** —— 环境变量会被 vLLM 的 mp 子进程
# 继承，子进程会误判"已加载"而跳过挂载（09-30 启动日志实测子进程打 "哨兵命中"）。
# sys 属性每进程独立：父进程挂一次、每个子进程各挂一次，语义正确（将来若加 worker
# 侧钩子，子进程也必须真的挂上）。
if getattr(sys, "_dsh_rt_extra_installed", False):
    _log("本进程已挂载过（sys 哨兵命中），跳过重复安装")
else:
    sys._dsh_rt_extra_installed = True
    _chain_external_sitecustomize()
    sys.meta_path.insert(0, _PatchFinder(_PATCHES))
    _log(f"本地扩展钩子已挂载：{', '.join(_PATCHES)}")
