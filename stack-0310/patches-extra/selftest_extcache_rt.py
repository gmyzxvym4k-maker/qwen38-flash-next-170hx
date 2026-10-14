"""rt-patch #12 离线自检（无 GPU 依赖，可在生产 venv 上安全跑）：
  T1 三处钩子装载判据：Request.take_prefill_stats / Scheduler.update_from_output
     / SchedulerInterface.update_from_output 都被包装（标记位双判据）；
  T2 ★回归护栏：钩子 B 必须挂在**具体类 Scheduler** 上——官方 Scheduler 自定义了
     同名方法遮蔽接口那份，只包 interface 时真实引擎里钩子永不执行（10-07 实锤）；
  T3 数值精确：注入 (pt, local, external) → 落盘 ext/loc/cached/pt 逐项相等，
     且过官方 PrefillStats.set 的 local+external<=pt 断言（假数据会被当场拒绝）；
  T4 ext=0 与「无数据」语义分离：真 0 落盘成 0，未 take 的请求不落盘；
  T5 重复完成信号幂等：同一 rid 多次 finish 只写一行；
  T6 紧急总闸 DSH_EXTCACHE_RT_DISABLE=1 → reload 后 PATCHES 为空；
  T7 一致性护栏：人为破坏恒等式 → consistent=false 且 ext/loc 置 null（不外发假拆分）。

运行（线上）：
  PYTHONPATH=/home/ll/deploy/vllm-0300/patches:/home/ll/deploy/vllm-0300/patches-extra \\
    DSH_EXT_CACHE_FILE=/tmp/ext-selftest.jsonl \\
    /media/ll/data/vllm-0310-env/bin/python /home/ll/deploy/vllm-0300/patches-extra/selftest_extcache_rt.py
注意：PYTHONPATH 链路上 sitecustomize 已把钩子挂进真实导入，T1 允许「已挂钩」态。
"""
from __future__ import annotations

import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("DSH_EXT_CACHE_FILE", "/tmp/ext-selftest.jsonl")
TRACE = os.environ["DSH_EXT_CACHE_FILE"]
if os.path.exists(TRACE):
    os.remove(TRACE)

fails: list[str] = []


def ck(cond: bool, name: str, detail: str = "") -> None:
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  :: " + detail) if detail and not cond else ""))
    if not cond:
        fails.append(name)


import dsh_extcache_rt as rt  # noqa: E402
from vllm.v1.core.sched.interface import SchedulerInterface  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.metrics.stats import PrefillStats  # noqa: E402
from vllm.v1.request import Request  # noqa: E402

import inspect as _insp

def _own_hooked(cls, attr):
    """真判据：本类 __dict__ 里的那个函数是我们的包装，而不是继承来的标记位。"""
    f = cls.__dict__.get(attr)
    if f is None:
        return False, "本类 __dict__ 无该方法（仅有继承实现）"
    src = _insp.getsourcefile(f) or "?"
    return ("dsh_extcache_rt" in src), src.split("/")[-1]


print("[T1/T2] 钩子装载（判据一律看类自己的 __dict__，不信继承的标记位）")
ok_a, det_a = _own_hooked(Request, "take_prefill_stats")
ck(ok_a, "A 主口径已挂 Request.take_prefill_stats（本类实现）", det_a)
ok_s, det_s = _own_hooked(Scheduler, "update_from_output")
ck(ok_s, "★B 完成信号已挂具体类 Scheduler 的本类实现（防遮蔽/防继承误判）",
   f"Scheduler.__dict__ 里仍是官方实现（文件={det_s}）⇒ 引擎里永不落盘")
ok_i, det_i = _own_hooked(SchedulerInterface, "update_from_output")
ck(ok_i, "B 保险也已挂 SchedulerInterface", det_i)
ck("vllm.v1.core.sched.scheduler" in rt.PATCHES,
   "PATCHES 含 vllm.v1.core.sched.scheduler", str(sorted(rt.PATCHES)))

print("[T3/T4/T5] 数值与语义")


def mk(rid, pt, loc, ext, fin_est):
    r = Request.__new__(Request)
    r.request_id = rid
    r.prefill_stats = PrefillStats()
    r.prefill_stats.set(num_prompt_tokens=pt, num_local_cached_tokens=loc,
                        num_external_cached_tokens=ext)
    ps = r.take_prefill_stats()
    ps.finalize(fin_est)
    return ps


PA = mk("chatcmpl-SA", 487665, 351800, 133000, 484800)   # 二级命中显著
PB = mk("chatcmpl-SB", 100000, 90000, 0, 90000)          # 真 0（纯本级命中）


class _O:
    def __init__(self, rid, ps, fr):
        self.request_id = rid
        self.prefill_stats = ps
        self.finish_reason = fr
        self.num_prefill_tokens = 0


class _EO:
    outputs: list = []


# 用一个不与 SchedulerInterface 相干的最小宿主，按 #12-B 的真实语义走一遍
# （Scheduler.update_from_output 已被包装，但它会真去调上游实现、依赖大量实例
#  状态，这里不宜伪造整机调度器；故直接驱动模块内部的记账/落盘函数，
#  等价检验钩子 B 的后半段——前半段「进入包装」由 T2 的标记位担保。）
for rid, ps, fr in [("chatcmpl-SA", PA, "STOP"), ("chatcmpl-SA", None, "STOP"),
                    ("chatcmpl-SB", PB, "LENGTH")]:
    if ps is not None:
        rt._note_take(rid, ps)
    if fr is not None:
        rt._flush(rid)
rt._flush("chatcmpl-SC")   # 从未 take → 不应落盘

lines = [json.loads(l) for l in open(TRACE)] if os.path.exists(TRACE) else []
by = {l["rid"]: l for l in lines}
ck(len(lines) == 2, "落盘 2 行（重复 finish 幂等、未 take 不落盘）", f"实得 {len(lines)} 行: {lines}")
a = by.get("chatcmpl-SA", {})
ck((a.get("ext"), a.get("loc"), a.get("cached"), a.get("pt")) == (133000, 351800, 484800, 487665),
   "T3 数值逐项精确", json.dumps(a, ensure_ascii=False))
ck(a.get("consistent") is True, "T3 恒等式为真", json.dumps(a, ensure_ascii=False))
b = by.get("chatcmpl-SB", {})
ck(b.get("ext") == 0, "T4 真 0 如实落盘（区别于缺字段）", json.dumps(b, ensure_ascii=False))
ck("chatcmpl-SC" not in by, "T4 未 take 的请求不落盘")

print("[T7] 一致性护栏")
BAD = {"ext": 5, "loc": 7, "cached": 99, "pt": 100, "taken": True}
rt._TRACK.clear(); rt._ORDER.clear(); rt._WRITTEN.clear()
rt._TRACK["chatcmpl-X"] = BAD; rt._ORDER.append("chatcmpl-X")
rt._flush("chatcmpl-X")
xl = [json.loads(l) for l in open(TRACE)][-1]
ck(xl["consistent"] is False and xl["ext"] is None and xl["loc"] is None,
   "破坏恒等式 → ext/loc 置 null 不外发假拆分", json.dumps(xl, ensure_ascii=False))

print("[T8] 装载顺序独立性（继承标记位陷阱）")
# 再造一对父子类：先给基类挂钩，再给子类挂钩；子类即使继承了基类标记位也必须被挂上
class _Base:
    def update_from_output(self, so, mo): return {}
class _Child(_Base):
    def update_from_output(self, so, mo): return {}
fake_mod_b = type("M", (), {"__name__": "fake.base"})
setattr(fake_mod_b, "K", _Base)
fake_mod_c = type("M", (), {"__name__": "fake.child"})
setattr(fake_mod_c, "K", _Child)
rt._make_iface_hook("K")(fake_mod_b)     # 先挂基类 ⇒ _Child 会继承到标记位
rt._make_iface_hook("K")(fake_mod_c)     # 后挂子类：必须无视继承来的标记位照样挂钩
ok_c, det_c = _own_hooked(_Child, "update_from_output")
ck(ok_c, "子类在有基类标记位的情况下仍被独立挂钩", det_c)

print("[T6] 紧急总闸")
os.environ["DSH_EXTCACHE_RT_DISABLE"] = "1"
try:
    rt2 = importlib.reload(rt)
    ck(rt2.PATCHES == {}, "DSH_EXTCACHE_RT_DISABLE=1 ⇒ 不挂钩", str(rt2.PATCHES))
finally:
    os.environ.pop("DSH_EXTCACHE_RT_DISABLE", None)

print()
if fails:
    print(f"❌ {len(fails)} 项失败：{fails}")
    sys.exit(1)
print("✅ rt-patch #12 自检全部通过")
