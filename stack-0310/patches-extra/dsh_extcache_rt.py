"""dsh_extcache_rt —— rt-patch #12：每请求「二级缓存（CPU KV offload）命中 token」真值（2026-10-07）。

要给控制台「最近完成请求」表新增 二级缓存命中 列，需要**逐请求**的缓存命中拆分。
官方 vLLM 0.30.0 的调度器在首个 prefill 步就把拆分算好了
（v1/core/sched/scheduler.py:1034-1040 的 request.prefill_stats.set(...)）：

    num_local_cached_tokens     = 本级 GPU 前缀缓存命中
    num_external_cached_tokens  = 外部 KV 传输命中（CPU 二级缓存回载，免重算）
    num_cached_tokens           = local + external        ← 现「缓存命中」列的口径

但该对象随即被 ``request.take_prefill_stats()``（scheduler.py:2149）摘走、
finalize 后挂到 **EngineCoreOutput.prefill_stats**（scheduler.py:2216）发往前端。
前端 output_processor 只把它并进**整批聚合**的 prompt_token_stats（跨请求求和），
逐请求对象的两个分项就此丢失——FinishedRequestStats 只留 num_cached_tokens 总数。
⇒ 唯二的真值取回点都在**引擎核进程内**：

  甲) SchedulerInterface.update_from_output 的返回 dict（core.py:654 调用）——
     里面就是本步新建的 EngineCoreOutput，prefill_stats 尚未 finalize。
  乙) Request.take_prefill_stats（v1/request.py:350）——官方唯一的摘取点，
     一定在甲之前发生，是最稳的一道；还能顺手把 scheduler 实例交给兜底扫描。

本补丁一行不改上游文件，两道钩子都挂（互为冗余，任一生效即出数）：

  #12-A ``vllm.Request.take_prefill_stats``
        返回前记下 (ext, loc, cached, pt)；cached 此刻是 set() 写的 local+external，
        官方在同处有 assert 背书，一致性最强。顺带 register_scheduler(self._sched)。
  #12-B ``Scheduler.update_from_output``（vllm.v1.core.sched.scheduler）
        遍历返回的 EngineCoreOutputs.outputs，读 prefill_stats 分项，
        并在 output.finish_reason 非空时落盘该请求（主完成信号）。
        ⚠ 必须挂在这个**具体类**上：官方 Scheduler 自己定义了同名方法
        （scheduler.py:1988 起）遮蔽了接口那份，只包 SchedulerInterface 的话
        真实引擎里钩子根本不会执行（10-07 生产 venv 上实测复现：打到接口的
        包装一次都没进）。interface 那道仍保留，纯粹防未来换成别的调度器实现。

落盘：vllm-ext-cache.jsonl，一行一个 rid：
    {"rid","t","ext","loc","cached","pt","consistent","port"}
控制台 server.js readExtCache()/extApplyToRecs() 读尾部窗口按 rid join 到
request-traces 行（同串 chatcmpl-*），新列即显示 ext；join 不上如实显示 --。
SGLang 侧不需要本补丁：官方 exporter 已在完成记录里直接给
cached_tokens_storage_backend（二级命中）与 cached_tokens_local（本级命中），
server.js 侧映射即可。

保守性设计（宁可少显示，绝不显示假拆分）：
  · 只信 take_prefill_stats 那一道的分项；update_from_output 道只做完成信号与补全。
  · 写盘前校验 ext+loc == cached（官方 PrefillStats.set 的恒等式）。不相等
    （例如被 finalize 改写过）时 consistent:false 且 ext/loc 置 null。
  · 登记表 TTL(1800s)+LRU(8192) 双修剪；文件 >4MiB 轮转留一份 .1。
  · 任何异常都被吞掉并只在 stderr 出声一次，绝不影响推理链路。
紧急总闸：DSH_EXTCACHE_RT_DISABLE=1 ⇒ 完全不挂钩（控制台读不到文件 → 新列显示 --）。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time


def _log(msg: str) -> None:
    sys.stderr.write(f"[rt-patch-extcache] {msg}\n")
    sys.stderr.flush()


_OUT_PATH = os.environ.get(
    "DSH_EXT_CACHE_FILE", "/home/ll/deploy/vllm-ext-cache.jsonl")
_PORT = os.environ.get("DSH_ENGINE_PORT", "") or ""

_LOCK = threading.Lock()
_TRACK: dict = {}        # rid -> {ext, loc, cached, pt, seen, taken}
_ORDER: list = []        # rid 插入序（LRU 修剪）
_WRITTEN: set = set()    # 已落盘 rid（幂等去重）
_SCHED_REFS: list = []   # [(id, scheduler)] 兜底扫描用

MAX_TRACK = 8192
TTL_S = 1800.0
WRITTEN_MAX = 20000
ROTATE_BYTES = 4 * 1024 * 1024

_state = {"fd": None, "broken": False, "logged_once": False}


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
def _ensure_fd():
    st = _state
    if st["broken"]:
        return None
    if st["fd"] is not None:
        return st["fd"]
    try:
        d = os.path.dirname(_OUT_PATH)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        st["fd"] = os.open(_OUT_PATH, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        if not st["logged_once"]:
            st["logged_once"] = True
            _log(f"输出文件：{_OUT_PATH}（port={_PORT or '-'}）")
    except Exception as exc:
        st["broken"] = True
        _log(f"打开 {_OUT_PATH} 失败，本补丁转为静默无害：{type(exc).__name__}: {exc}")
        st["fd"] = None
    return st["fd"]


def _rotate_if_needed() -> None:
    st = _state
    try:
        if st["fd"] is None or not os.path.exists(_OUT_PATH):
            return
        if os.path.getsize(_OUT_PATH) <= ROTATE_BYTES:
            return
        os.close(st["fd"])
        st["fd"] = None
        bak = _OUT_PATH + ".1"
        try:
            if os.path.exists(bak):
                os.remove(bak)
            os.replace(_OUT_PATH, bak)
        except Exception:
            pass
        _log("已达体积上限，换新文件（旧文件留 .1）")
    except Exception:
        pass


def _write_line(obj: dict) -> None:
    fd = _ensure_fd()
    if fd is None:
        return
    try:
        data = (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        os.write(fd, data)
    except Exception:
        return
    _rotate_counter[0] -= 1
    if _rotate_counter[0] <= 0:
        _rotate_if_needed()
        _rotate_counter[0] = 400


_rotate_counter = [400]


# ---------------------------------------------------------------------------
# 登记表
# ---------------------------------------------------------------------------
def _prune_locked(now: float) -> None:
    dead = [r for r, v in _TRACK.items()
            if v.get("taken") or (now - v.get("seen", now)) > TTL_S]
    for r in dead:
        _TRACK.pop(r, None)
    for r in list(_ORDER):
        if r not in _TRACK:
            _ORDER.remove(r)
    while len(_ORDER) > MAX_TRACK:
        _TRACK.pop(_ORDER.pop(0), None)
    while len(_WRITTEN) > WRITTEN_MAX:
        _WRITTEN.pop()


def _note_take(rid, ps) -> None:
    """#12-A：官方摘取 PrefillStats 的瞬间记下分项（唯一有 assert 背书的口径）。"""
    if not rid or ps is None:
        return
    try:
        ext = int(ps.num_external_cached_tokens)
        loc = int(ps.num_local_cached_tokens)
        cached = int(ps.num_cached_tokens)
        pt = int(ps.num_prompt_tokens or 0)
    except Exception:
        return
    now = time.time()
    with _LOCK:
        rec = _TRACK.get(rid)
        if rec is None:
            rec = {"ext": None, "loc": None, "cached": None, "pt": pt,
                   "seen": now, "taken": False}
            _TRACK[rid] = rec
            _ORDER.append(rid)
        rec["ext"], rec["loc"], rec["cached"] = ext, loc, cached
        if pt:
            rec["pt"] = pt
        rec["seen"] = now
        rec["taken"] = True
        if len(_ORDER) > MAX_TRACK:
            _prune_locked(now)


def _flush(rid, pt_hint=None) -> None:
    """请求完成：校验并落盘一次（幂等；可在 _LOCK 内被调用）。

    取数与置位在同一把锁内原子完成，写盘延后到锁外（IO 不阻塞另一个钩子）。
    """
    if not rid:
        return
    with _LOCK:
        if rid in _WRITTEN:
            return
        rec = _TRACK.get(rid)
        if rec is None or not rec.get("taken"):
            # 从没被 take 过（纯解码/极早取消/preempt 重来）→ 无可信拆分，不写假数据
            return
        if pt_hint and not rec.get("pt"):
            rec["pt"] = int(pt_hint)
        _WRITTEN.add(rid)
        ext, loc, cached, pt = rec["ext"], rec["loc"], rec["cached"], rec["pt"]
        # take 之后官方还会 finalize()（只增 num_cache_creation_tokens，不动这两个
        # 分项），故这里读到的仍是 set() 写入的原值；恒等式只是纵深校验。
        payload = {
            "rid": rid,
            "t": round(time.time(), 3),
            "ext": ext,
            "loc": loc,
            "cached": cached,
            "pt": pt,
            "consistent": (ext is not None and loc is not None and cached is not None
                           and (ext + loc) == cached),
            "port": _PORT,
        }
        _TRACK.pop(rid, None)
        if rid in _ORDER:
            _ORDER.remove(rid)
        _prune_locked(time.time())
    if not payload["consistent"]:
        payload["ext"] = None
        payload["loc"] = None
    _write_line(payload)


# ---------------------------------------------------------------------------
# 钩子 #12-A：Request.take_prefill_stats（vllm.v1.request）
# ---------------------------------------------------------------------------
def _patch_request(module) -> None:
    cls = getattr(module, "Request", None)
    if cls is None or not hasattr(cls, "take_prefill_stats"):
        _log("没找到 Request.take_prefill_stats ⇒ #12-A 跳过（上游结构变了）")
        return
    if getattr(cls, "_dsh_extcache_a", False):
        return
    orig = cls.take_prefill_stats

    def take_prefill_stats(self):
        ps = orig(self)
        try:
            _note_take(getattr(self, "request_id", None), ps)
        except Exception:
            pass
        return ps

    cls.take_prefill_stats = take_prefill_stats
    cls._dsh_extcache_a = True
    _log("Request.take_prefill_stats 已挂钩（rt-patch #12 主口径）")


# ---------------------------------------------------------------------------
# 钩子 #12-B：<Class>.update_from_output（完成信号 + 冗余补全）
# 同一个包装挂到多处：具体类 Scheduler（真实生效点）+ 抽象 SchedulerInterface
# （防将来换调度器实现）。挂到已被具体类遮蔽的抽象方法上没有副作用，留着只是保险。
# ---------------------------------------------------------------------------
def _make_iface_hook(class_attr):
    def hook(module):
        cls = getattr(module, class_attr, None)
        if cls is None or not hasattr(cls, "update_from_output"):
            _log(f"没找到 {module.__name__}.{class_attr}.update_from_output ⇒ #12-B 跳过（上游结构变了）")
            return
        # ★ 幂等判据必须看**类自己的 __dict__**，不能用 getattr 的继承查找：
        #   Scheduler 继承自 SchedulerInterface，若先挂了接口那份，getattr(cls,"标记位")
        #   会从基类读到 True，导致针对具体类 Scheduler 的这一钩子直接 return、
        #   官方实现原地不动 ⇒ 真实引擎里永不落盘（10-07 第二轮实测抓到）。
        if "_dsh_extcache_b" in cls.__dict__:
            return
        if getattr(cls, "_dsh_extcache_b", False):
            _log(f"{class_attr} 的标记位来自基类继承，仍需为本类单独挂钩")
        # 取本类自己的实现（Scheduler 自定义了同名方法，会遮蔽接口那份）；
        # 若本类没有则退回继承所得（例如某个未覆写的调度器实现）。
        orig = cls.__dict__.get("update_from_output", None) or cls.update_from_output

        def update_from_output(self, scheduler_output, model_output):
            ret = orig(self, scheduler_output, model_output)
            try:
                register_scheduler(self)
                eos = ret.values() if isinstance(ret, dict) else []
                for eo in eos:
                    for o in getattr(eo, "outputs", ()) or ():
                        ps = getattr(o, "prefill_stats", None)
                        if ps is not None:
                            # 冗余一道：take 钩子若因上游改名失效，这里仍能拿到同一对象
                            _note_take(getattr(o, "request_id", None), ps)
                        if getattr(o, "finish_reason", None) is not None:
                            _flush(o.request_id, getattr(o, "num_prefill_tokens", None))
            except Exception as exc:
                if not _state.get("warn_b"):
                    _state["warn_b"] = True
                    _log(f"#12-B 遍历输出异常（忽略）：{type(exc).__name__}: {exc}")
            return ret

        cls.update_from_output = update_from_output
        cls._dsh_extcache_b = True
        _log(f"{class_attr}.update_from_output 已挂钩（rt-patch #12 完成信号）")
    return hook


# ---------------------------------------------------------------------------
# 兜底：abort / preempt 等不走正常完成路径的请求，靠 TTL 清理，不留悬挂
# ---------------------------------------------------------------------------
def register_scheduler(sched) -> None:
    try:
        key = id(sched)
        for k, _ in _SCHED_REFS:
            if k == key:
                return
        _SCHED_REFS.append((key, sched))
        if len(_SCHED_REFS) > 4:
            del _SCHED_REFS[:-4]
    except Exception:
        pass


def sweep() -> None:
    """供插件每步（低频）搭车：修剪 TTL/LRU，防长跑内存缓慢增长。"""
    try:
        with _LOCK:
            _prune_locked(time.time())
    except Exception:
        pass


PATCH_DISABLED_KEY = "DSH_EXTCACHE_RT_DISABLE"


def build_patches() -> dict:
    if os.environ.get(PATCH_DISABLED_KEY, "") == "1":
        _log(f"{PATCH_DISABLED_KEY}=1 ⇒ #12 不动作（控制台二级缓存命中列显示 --）")
        return {}
    # 顺序有意义：先挂具体类 Scheduler（真实生效点），再挂抽象 Interface（保险）。
    # 两处都用 cls.__dict__ 判据，互不干扰；sitecustomize 是按本 dict 的键逐个
    # 注册 meta-path 回调，实际触发时机取决于模块导入次序，故判据不能依赖顺序。
    return {
        "vllm.v1.request": _patch_request,
        "vllm.v1.core.sched.scheduler": _make_iface_hook("Scheduler"),
        "vllm.v1.core.sched.interface": _make_iface_hook("SchedulerInterface"),
    }


PATCHES = build_patches()
