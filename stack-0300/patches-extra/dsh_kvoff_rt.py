"""dsh_kvoff_rt —— KV 二级缓存（OffloadingConnector）在官方 vLLM 0.30.0 的运行时移植（rt-patch #9）。

背景：旧 chroot 定制镜像栈（vLLM v0.1.dev20073）上，CPU KV 二级缓存经 c1→c7 七轮
演进才在 PP2 + QSA/MTP 拓扑下稳定（见交付仓库 docs/05-patches.md D 组与记忆条目）。
09-29 因「21h 生产零外部命中 + 常驻 107GB pinned」定案退役；09-26 迁到官方 0.30.0 时
该功能未随迁。本模块把其中仍然需要的修复移植到 0.30.0 的运行时补丁架构
（PYTHONPATH sitecustomize 注入，site-packages 零改动），供随时经 FN_KVOFF=1 启用。

对照旧补丁组的取舍：
  c1  offloading/config.get_offloading_group_ids —— 只剔除「不可前缀缓存」的分组
      （本模型 = QSA key 环形缓冲 CircularBufferSpec，block=ring=8 不整除 1616）。
      【09-27 实机验证修正】**不得**再剔除「空 layer_names」分组：worker 进程的
      kv_cache_config 只含本 PP rank 的层名（上游 generate_scheduler_kv_cache_config
      注释："All workers have the same kv_cache_config except layer names"），本模型
      的 PLE 占位分组在 PP0 有层、在 PP1 为空 —— 若按空层剔除，PP1 的
      group_data_refs 就会比 scheduler 的 kv_group_configs 少一项，store 路径
      gpu_worker.transfer_async 的 len(group_sizes)==len(layer_refs_per_group)
      断言必炸（实测 PP1 首个 store job 即 AssertionError → c6 熔断只读降级，
      CPU_to_GPU 恒 0）。上游对空组的正解是「保留位置、refs 为空列表」。
  c2  kv_offload/cpu/spec.CPUOffloadingSpec._uses_shared_region —— pp_size>1 时
      走每 rank 私有缓冲：共享区行布局的 slot 起点 = rank × cpu_page_size，
      而该值是 per-rank 量（TP 下各 rank 层相同才成立）；PP 各 rank 层子集不同
      ⇒ 行 stride 分歧 + total_size 取整各异（09-18 实测 34.32 vs 34.3566 GB）
      ⇒ 创建者与 joiner 尺寸不一致 → 30s 超时。注意：该私有路径本身是**真
      pinned**（else 分支 pin_memory=PIN_MEMORY，本栈实测 True），它不是 store
      崩溃的根因——09-27 的相反定性已于 09-30 复核推翻，详见下方 c2 段。
  c6  有界等待 + store 熔断只读降级（gpu_worker / offloading.worker /
      offloading.common / offloading.scheduler 四处）：
      · handle_preemptions 的 worker.wait(jobs_to_flush) 原版无限阻塞主线程，
        PP2 下 flush store 的流链 wait_stream(compute) 与跨 rank NCCL 中继互等
        成环（09-24 三次卡死实锤）→ 轮询 + FN_KVOFF_WAIT_TIMEOUT（缺省 15s）；
      · submit_store 抛错/返回 False/wait 超时 → 失败 ack（job 记入 failed_jobs），
        scheduler 侧 complete_store(success=False) 撤登记并释放 CPU 块；
      · 任一失败 → store 方向熔断，后续不再建新 store 任务（只读降级：
        已有条目继续 lookup/load 命中；load 方向永不熔断，正确性零妥协）。
  c7  **仅 load(CPU→GPU)** 方向强制 Triton SM 内核并去掉 MIN_N 小批量回落：
      0.30.0 的 _select_swap_blocks_fn 对 >28KB 的页会回落到 C++ 批量拷贝
      （cuMemcpyBatchAsync），本模型块页是 MB 级 ⇒ 该方向必走那个 API，而它与
      PP2 NCCL-P2P 并发在本驱动（610.43.03/GSP）上有冻结 compute 流的实证
      （09-23 崩溃 + 09-24 七次卡死）。**store(GPU→CPU) 一律保持上游 C++ DMA**：
      上游该方向显式选拷贝引擎（"GPU->CPU is bandwidth-bound"），强开 Triton 会
      让 SM 内核解引用 host 指针 → MMU Fault VIRT_WRITE → Xid31（09-27 实锤）。
      ⚠️ c7 只覆盖 load 方向；**store 方向的崩溃（C++ DMA error 1 / Triton Xid31）
      不在其覆盖范围内**，那才是本栈 KVOFF 不可用的直接原因。
  c8  **PP>1 公共区**：跨 PP rank 统一布局的单一共享 pinned mmap 区。
      上游共享区的 slot 起点写死为 rank×cpu_page_size（各 rank 等宽，只在 TP 成立），
      本补丁换成「前缀和偏移」——row_stride = Σ align4096(各 rank 自己的每块字节)，
      num_chunks = cpu_bytes_to_use // row_stride，于是 **物理钉住 = 配置值**（c2 私有
      路径是 1.56~2× 配置值，本机 09-22 实测 64GiB→107GB，在有 MCE 硬挂史的机器上
      是不可接受的隐性风险）。布局经 /dev/shm 文件在 worker 间协商、再由调度占侮采纳
      （不一致就拒绝启动，绝不带着错几何跑）。开关 FN_KVOFF_SHARED（缺省 1，
      0=退回 c2 私有缓冲）；仅限 pp>1 + tp==1 + 非 replicated/canonical 布局。
  c3  （metrics 未知 key 崩溃防御）不移植：旧崩溃由我们自加 stats key 引起，
      本移植不新增任何 stats key，上游 defs/sender 自洽。
  c5a （MTP 全组标 eagle 的双罚）不移植：0.30.0 原生含同款修复
      （from_spec：无分组标注 drafter 时「treating all groups as non-draft」）。
  13/14/15/25 的观测/dump 增强不移植（诊断用，非功能必需）。

惰性（生产安全）：所有钩子只挂 vllm 的 offloading / kv_offload.cpu 模块——
不配 --kv-transfer-config 时这些模块根本不会被导入，钩子不触发，对默认生产形态
（FN_KVOFF=0）零影响。紧急总闸 DSH_KVOFF_RT_DISABLE=1：挂载后各回调变 no-op，
等效回到纯原厂 connector 行为（不用改任何文件）。

版本锁定：0.30.0。每个补丁函数替换/包裹前都校验目标属性形态，缺失即跳过并
大声告警（宁可在首请求处炸出真错，不让静默错语义上线）。
"""

import os
import time

_DISABLED = os.environ.get("DSH_KVOFF_RT_DISABLE", "") == "1"


def _log(msg):
    import sys

    sys.stderr.write("[rt-patch-kvoff] %s\n" % msg)
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# c1：源头剔除不可 offload 的分组（QSA 环形 + 空占位）
# ---------------------------------------------------------------------------
def _patch_offloading_config(m):
    if _DISABLED:
        return
    fn = getattr(m, "get_offloading_group_ids", None)
    if fn is None or getattr(fn, "_dsh_c1", False):
        return

    def filtered(kv_cache_config):
        ids = tuple(fn(kv_cache_config))
        keep, dropped = [], []
        for gid in ids:
            group = kv_cache_config.kv_cache_groups[gid]
            spec = group.kv_cache_spec
            if not getattr(spec, "prefix_cacheable", True):
                dropped.append((gid, type(spec).__name__, "not prefix_cacheable"))
                continue
            keep.append(gid)
        if not keep:
            _log("c1: 过滤后无可用分组，回退不过滤（原版行为）")
            return ids
        # 常驻诊断走 stderr：config.py 模块**没有** logger，原先的 lg.info 恒静默
        # （09-27 实机验证时因此完全看不到剔除痕迹，误判成「未剔除任何分组」）。
        _log(
            "c1: kv_cache_groups=%d -> offload 分组 %s（原 %s，剔除 %s）pid=%d"
            % (
                len(kv_cache_config.kv_cache_groups),
                list(keep),
                list(ids),
                dropped,
                os.getpid(),
            )
        )
        return tuple(keep)

    filtered._dsh_c1 = True
    m.get_offloading_group_ids = filtered


# ---------------------------------------------------------------------------
# c8：PP>1「公共区」——跨 PP rank 统一布局的单一共享 pinned mmap 区
#
# 为什么需要（这是 KVOFF 在本机可用的前提，不只是优化）：
#   c2 的每 rank 私有 pinned 缓冲，容量口径是"每 rank 各配一份"：
#       rank_i 行数 = cpu_bytes_to_use // align(own_slot_i)
#   而调度器侧 manager 的 chunk id 空间 = cpu_bytes_to_use // align(slot_0)
#   （configs[0] 就是 PP rank0 的配置），两个量级叠加 ⇒ 配置 64 GiB 实测钉住
#   107~133 GiB 物理内存（09-22/09-24 smaps 实测 1.56×）。本机 09-22 起有
#   内存硬件不稳（MCE/硬挂）史，"配置值 ≠ 物理值"是不可接受的隐性风险。
#
# 公共区做法（配置即物理）：
#   把一行（chunk）定义成"整个模型一个 block 的字节"，按 rank 顺序紧密排布：
#       slot_i     = align4096(rank_i 自己的每块字节)      ← 各 rank 不同，允许
#       offsets_i  = Σ_{j<i} slot_j                        ← 前缀和，天然不相交
#       row_stride = Σ_i slot_i                            ← 全体一致
#       num_chunks = cpu_bytes_to_use // row_stride        ← 全体一致
#   ⇒ 物理占用恰为 cpu_bytes_to_use（不再乘 world_size），容量 = 配置/全局每块
#   字节，且每个 rank 只在自己的 [offsets_i, offsets_i+slot_i) 内读写，
#   与上游 create_next_worker_view 的 (row_stride, 1) 跨步视图完全兼容。
#
# 与上游共享区的差别：上游假设各 rank slot 相同（TP 成立），slot 起点写成
#   rank × cpu_page_size；PP 各 rank 层子集不同 ⇒ 起点/步长分歧 ⇒ 创建者与
#   joiner 尺寸不一致 → 30 s 超时（09-18 实锤）。本补丁用"前缀和偏移"替掉
#   那个乘法，就一条差异，其余（O_EXCL 创建、ftruncate、barrier 后 unlink、
#   cudaHostRegister 整区）全部沿用上游实现。
#
# 布局协商（无中心、无死锁、可判陈旧）：
#   每个 worker 在 /dev/shm 发布自己的 slot 字节数，收齐 world_size 份后各自
#   算出同一套 (row_stride, num_chunks)，再发布第二次（带 decision）。调度器侧
#   在 get_manager() 里读这些文件并采纳同一个 num_chunks——**必须一致**，否则
#   manager 发出的 chunk id 会越过 worker 的行数上界，写越界就是 device-side
#   assert/Xid31。调度器只在全体 decision 一致时才继续；不一致或超时 ⇒ 抛错
#   停机（宁可起不来，也不要静默错语义），关掉方式 FN_KVOFF_SHARED=0。
#   陈旧文件判据：写者 pid 必须存活，且其 /proc/<pid>/stat starttime 与本进程
#   相差 ≤ FN_KVOFF_LAYOUT_WINDOW 秒（默认 900）。用 starttime 而不是墙上时间，
#   是因为本机 RTC 会跳到 2161 年（09-18 定案），mtime 比较不可信。
#
# 适用范围（超出即自动退回 c2 私有缓冲，行为与补丁前完全一致）：
#   pp_size>1 且 tp_size==1 且 非 replicated_layout 且 非 canonical_layout，
#   且 single-node mp 后端。本机生产正好落在这个范围（PP2/TP1）。
# ---------------------------------------------------------------------------
_PAGE_ALIGN = 4096
_LAYOUT_CTX = [None]  # create_worker 调用期间临时存放本 rank 的布局
_CLK_TCK = None


def _clk_tck():
    global _CLK_TCK
    if _CLK_TCK is None:
        try:
            _CLK_TCK = int(os.sysconf("SC_CLK_TCK")) or 100
        except Exception:
            _CLK_TCK = 100
    return _CLK_TCK


def _round_up(v, m=_PAGE_ALIGN):
    v = int(v)
    return ((v + m - 1) // m) * m


def _proc_start_ticks(pid):
    """进程启动时刻（开机以来 jiffies）。进程不在则抛异常。"""
    with open("/proc/%d/stat" % pid, "rb") as f:
        raw = f.read()
    # comm 字段可能含空格/括号，从最后一个 ')' 之后开始切
    rest = raw[raw.rfind(b")") + 2:].split()
    return int(rest[19])  # state 之后第 19 个 = 全局第 22 字段 starttime


def _shm_free_bytes(path="/dev/shm"):
    """/dev/shm 可用字节（公共区是 tmpfs 文件，放不下就别硬上）。"""
    try:
        st = os.statvfs(path)
        return int(st.f_bavail) * int(st.f_frsize)
    except Exception:
        return None


def _engine_tag(engine_id):
    import re

    return re.sub(r"[^0-9A-Za-z_.-]", "_", str(engine_id))[:80]


def _slot_file(tag, rank):
    return "/dev/shm/vllm_kvoff_slot.%s.r%d.json" % (tag, rank)


def _publish(tag, rank, payload):
    """原子发布（tmp + os.replace），失败返回 False（调用方退回私有路径）。"""
    import json

    path = _slot_file(tag, rank)
    tmp = "%s.tmp%d" % (path, os.getpid())
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return True
    except Exception as exc:
        _log("c8: 发布 %s 失败：%s（本 rank 退回私有缓冲）" % (path, exc))
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


def _read_payload(path, engine_id, world_size):
    """读一份发布；不存在/不合法/属于上一次运行的陈旧文件 ⇒ None。"""
    import json

    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        # 半截 JSON（对端正在写）当作暂未就绪，稍后重试
        return None
    if not isinstance(data, dict):
        return None
    if data.get("engine_id") != engine_id or data.get("world_size") != world_size:
        return None
    pid = data.get("pid")
    st = data.get("starttime")
    if pid is None or st is None:
        return None
    try:
        cur = _proc_start_ticks(int(pid))
    except Exception:
        _cleanup_stale(path)
        return None
    if cur != int(st):
        # pid 被复用/是上一次运行的残留
        _cleanup_stale(path)
        return None
    return data


def _cleanup_stale(path):
    try:
        os.unlink(path)
        _log("c8: 清理陈旧布局文件 %s" % path)
    except OSError:
        pass


def _collect(tag, engine_id, world_size, need_decision, timeout, window_s):
    """收齐 world_size 份发布。返回按 rank 排序的 payload 列表，超时返回 None。"""
    my_ticks = None
    deadline = time.monotonic() + timeout
    while True:
        if my_ticks is None:
            try:
                my_ticks = _proc_start_ticks(os.getpid())
            except Exception:
                return None
        out, ok = [], True
        for r in range(world_size):
            data = _read_payload(_slot_file(tag, r), engine_id, world_size)
            if data is None:
                ok = False
                break
            try:
                if abs(int(data["starttime"]) - my_ticks) > window_s * _clk_tck():
                    ok = False  # 不是同一次启动窗口（对端是上个运行的残留）
                    break
            except Exception:
                ok = False
                break
            if need_decision and data.get("decision") is None:
                ok = False
                break
            out.append(data)
        if ok:
            return out
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.2)


def _c8_supported_reason(config, spec):
    par = config.parallel
    if par.pp_size <= 1:
        return "pp_size<=1（上游共享区本就成立）"
    if par.tp_size != 1:
        return "tp_size=%s!=1（slot 归属需按设备号扩展，未验证）" % par.tp_size
    if getattr(spec, "replicated_layout", False):
        return "replicated_layout（MLA 单副本布局）"
    if getattr(config, "canonical_layout", False):
        return "canonical_layout"
    if int(par.data_parallel_size or 1) > 1:
        return "data_parallel_size>1（engine_id 隔离未实测）"
    if os.environ.get("FN_KVOFF_SHARED", "1").strip().lower() in (
        "0",
        "false",
        "off",
        "no",
    ):
        return "FN_KVOFF_SHARED=0"
    return None


def _c8_resolve(spec, publish, own_required=0):
    """协商公共区布局。成功：改写 spec 的三个尺寸字段并返回 True。

    publish=True  —— worker 侧（先发布自己的 slot，再收齐，再发布 decision）
    publish=False —— 调度器侧（只读，采纳 worker 已达成一致的布局）
    """
    config = spec.config
    par = config.parallel
    cpu_bytes = int(spec.extra_config.get("cpu_bytes_to_use") or 0)
    own = int(getattr(spec, "cpu_page_size_per_worker", 0) or 0)
    timeout = float(os.environ.get("FN_KVOFF_LAYOUT_TIMEOUT", "120"))
    window_s = float(os.environ.get("FN_KVOFF_LAYOUT_WINDOW", "900"))

    def bail(why):
        _log("c8: 不启用公共区（%s）⇒ 维持每 rank 私有 pinned 缓冲" % why)
        return False

    reason = _c8_supported_reason(config, spec)
    if reason:
        return bail(reason)
    if cpu_bytes <= 0 or own <= 0:
        return bail("cpu_bytes/own_slot 非法 cpu_bytes=%s own=%s" % (cpu_bytes, own))

    tag = _engine_tag(config.engine_id)
    world = int(par.world_size)
    rank = int(par.rank)
    base = {
        "engine_id": config.engine_id,
        "world_size": world,
        "rank": rank,
        "slot_bytes": own,
        "blocks_per_chunk": int(spec.blocks_per_chunk),
        "pid": os.getpid(),
        "starttime": _proc_start_ticks(os.getpid()),
    }

    if not publish:
        # ---- 调度器侧：采纳 worker 们公布的行数下界 ----
        # 安全性论证：manager 发出的 chunk id 必须落在**每一个** rank 的 CPU 缓冲
        # 行数以内，否则 store 越界写（device-side assert / Xid31）。worker 各自在
        # 发布里带上自己真实的行数 rows，调度器取 min ⇒ 恒安全；全为 shared 时
        # min 就等于公共区 num_chunks（各 rank 同一个值）。
        # 收不齐（对端没起来 / /dev/shm 不可写）时**不能**沿用上游口径（那是按
        # rank0 每块字节算的最大值，可能大于某些 rank 的实际行数）⇒ 置 0 关闭
        # store，只让引擎活着（CPUOffloadingManager(num_chunks=0) 的 prepare_store
        # 恒返回 None = 不 offload，无崩溃路径）。
        def _disable(reason):
            spec.num_chunks = 0
            spec._dsh_c8 = None
            _log(
                "c8[调度器侧]: %s ⇒ CPU 档行数置 0（本次启动不做 offload，引擎照常服务；"
                "要彻底关二级缓存用 FN_KVOFF=0）" % reason
            )
            return False

        got = _collect(tag, config.engine_id, world, True, timeout, window_s)
        if got is None:
            return _disable(
                "%.0fs 内未收齐 %d 个 worker 的布局决策（/dev/shm/vllm_kvoff_slot.%s.r*.json）"
                % (timeout, world, tag)
            )
        by_rank = sorted(got, key=lambda d: int(d.get("rank", 0)))
        rows_list = []
        for g in by_rank:
            try:
                rows_list.append(int(g["rows"]))
            except Exception:
                return _disable("worker 发布缺 rows 字段：%s" % (g,))
        if len(rows_list) != world or min(rows_list) < 0:
            return _disable("worker 发布不完整 rows=%s" % rows_list)
        n = min(rows_list)
        if n < 1:
            return _disable("可用 CPU 档行数不足 rows=%s" % rows_list)
        decisions = {g.get("decision") for g in by_rank}
        slots_ = [int(g["slot_bytes"]) for g in by_rank]
        shared = decisions == {"shared"} and len(by_rank) == world
        row_stride = int(by_rank[0].get("row_stride", 0)) if shared else 0
        if shared and row_stride != sum(_round_up(x) for x in slots_):
            _log(
                "c8[调度器侧]: row_stride 自校验不符（公布=%d 重算=%d）⇒ 按行数下界继续"
                % (row_stride, sum(_round_up(x) for x in slots_))
            )
        upstream = spec.num_chunks
        spec.num_chunks = n
        if shared:
            spec.kv_bytes_per_chunk = row_stride
            spec.cpu_page_size_per_worker = _round_up(slots_[0])
            spec._dsh_c8 = {
                "rank": rank,
                "offset": 0,
                "slot": _round_up(slots_[0]),
                "row_stride": row_stride,
                "num_chunks": n,
                "slots": slots_,
            }
        _log(
            "c8[调度器侧]: decision=%s 各 rank 行数=%s ⇒ 采纳 num_chunks=%d"
            "（上游口径 %d，公共区 row_stride=%.2f MB，容量≈%.2f M token，"
            "物理钉住=%.2f GiB / 配置=%.2f GiB）"
            % (
                sorted(decisions),
                rows_list,
                n,
                upstream,
                row_stride / 1e6,
                n * int(spec.tokens_per_block[0]) / 1e6,
                n * (row_stride or _round_up(slots_[0])) / 2**30,
                cpu_bytes / 2**30,
            )
        )
        return bool(shared)

    # ---- worker 侧 ----
    if not _publish(tag, rank, base):
        return False
    got = _collect(tag, config.engine_id, world, False, timeout, window_s)
    if got is None:
        _publish(
            tag,
            rank,
            dict(base, decision="private", row_stride=0,
                 num_chunks=cpu_bytes // _round_up(own),
                 rows=cpu_bytes // _round_up(own)),
        )
        return bail(
            "%.0fs 内未收齐 %d 份 slot 发布（对端 worker 未起来？）" % (timeout, world)
        )
    slots = [int(g["slot_bytes"]) for g in sorted(got, key=lambda d: d["rank"])]
    aligned = [_round_up(s) for s in slots]
    row = sum(aligned)
    offs = [sum(aligned[:i]) for i in range(world)]
    nchunks = cpu_bytes // row
    need = max(int(own_required), own)
    priv_rows = cpu_bytes // _round_up(own)
    free = _shm_free_bytes()
    if free is not None and nchunks * row > free:
        _publish(
            tag,
            rank,
            dict(base, decision="private", row_stride=0, num_chunks=priv_rows,
                 rows=priv_rows),
        )
        return bail(
            "/dev/shm 放不下公共区：需要 %.2f GiB、可用 %.2f GiB"
            % (nchunks * row / 2**30, free / 2**30)
        )
    if nchunks < 1 or aligned[rank] < need:
        _publish(
            tag,
            rank,
            dict(base, decision="private", row_stride=0, num_chunks=priv_rows,
                 rows=priv_rows),
        )
        return bail(
            "容量/容量对齐不成立 row=%d nchunks=%d slot=%d need=%d"
            % (row, nchunks, aligned[rank], need)
        )
    _publish(
        tag,
        rank,
        dict(base, decision="shared", row_stride=row, num_chunks=nchunks,
             rows=nchunks),
    )
    spec._dsh_c8 = {
        "rank": rank,
        "offset": offs[rank],
        "slot": aligned[rank],
        "row_stride": row,
        "num_chunks": nchunks,
        "slots": slots,
    }
    spec.kv_bytes_per_chunk = row
    spec.cpu_page_size_per_worker = aligned[rank]
    spec.num_chunks = nchunks
    _log(
        "c8[worker rank%d pid%d]: 公共区已协商 —— row_stride=%.2f MB（slots=%s）"
        " 本 rank 区段 [%s, %s) num_chunks=%d ⇒ 物理钉住 %.2f GiB（配置 %.2f GiB，"
        "不再 ×world_size）"
        % (
            rank,
            os.getpid(),
            row / 1e6,
            ["%.1fMB" % (s / 1e6) for s in slots],
            offs[rank],
            offs[rank] + aligned[rank],
            nchunks,
            nchunks * row / 2**30,
            cpu_bytes / 2**30,
        )
    )
    return True


def _c8_region_cls(base_cls):
    """包一层：用「前缀和偏移」替代上游 rank×cpu_page_size 的等宽假设。"""

    class DshPpSharedRegion(base_cls):
        def __init__(
            self,
            engine_id,
            num_chunks,
            rank,
            kv_bytes_per_chunk,
            cpu_page_size,
            barrier=None,
            **kw
        ):
            ctx = _LAYOUT_CTX[0]
            if ctx is None:
                # 非公共区场景（TP 等）：完全按上游语义
                base_cls.__init__(
                    self,
                    engine_id,
                    num_chunks,
                    rank,
                    kv_bytes_per_chunk,
                    cpu_page_size,
                    barrier=barrier,
                    **kw
                )
                return
            # rank=None ⇒ 基类不做 per-worker populate；创建者整区预填、
            # joiner 跳过（同一批物理页，第二次填充是空操作）。
            # （基类要求 populate_only_on_creator 必须配 barrier，无 barrier 时不开）
            base_cls.__init__(
                self,
                engine_id,
                num_chunks,
                None,
                kv_bytes_per_chunk,
                ctx["slot"],
                barrier=barrier,
                populate_only_on_creator=barrier is not None,
                **kw
            )
            self.rank = ctx["rank"]
            self._worker_offset = ctx["offset"]
            self._worker_area_end = ctx["offset"] + ctx["slot"]

    DshPpSharedRegion.__name__ = "DshPpSharedRegion"
    return DshPpSharedRegion


# ---------------------------------------------------------------------------
# c2：PP>1 禁用共享 pinned mmap 区（每 rank 私有缓冲）
#
# 必要性（结构性，不是可选退路）：共享区把 region 切成
#     |--- W0-C0---|--- W1-C0---| ... |
# 的行布局，slot 起点 = rank × cpu_page_size，而
#     cpu_page_size_per_worker = worker_kv_bytes_per_block × blocks_per_chunk
# 是 per-rank 量。TP 下各 rank 层相同故成立；**PP>1 各 rank 层子集不同** →
# 行 stride 与 slot 划分整体分歧，且 total_size = num_chunks ×
# aligned_kv_bytes_per_chunk 又因整除取整各 rank 不同（09-18 实测
# 34.32GB vs 34.3566GB）→ 创建者 ftruncate 自己的尺寸、joiner 却在等自己的
# 预期尺寸 → _wait_for_file_size 30s 超时（09-18 实锤）。故 PP>1 必须回退
# 私有缓冲；**"统一尺寸后恢复共享区"不是可行修法**（slot 布局仍分歧）。
#
# ⚠️ 纠正（2026-09-30 源码 + 实测复核，推翻 09-27 写的"无 pin 退化路径"定性）：
#    私有路径**本来就是 pinned 的**——gpu_worker.py 该分支为
#    torch.zeros((num_chunks, page), dtype=int8, pin_memory=PIN_MEMORY)，
#    而 PIN_MEMORY = is_pin_memory_available() 在本栈实测 = True（CUDA 平台），
#    内存由 cudaHostAlloc 提供（09-22 smaps 实测该路径为 /dev/zero 映射，
#    与 cudaHostAlloc 特征吻合，是独立佐证）。所以"host 缓冲不经
#    cudaHostRegister、device 侧无法访问"**不成立**，c2 也不是 store 崩溃的根因。
#
# 崩溃的真正形态是**版本相关**（尚未定论）：旧 chroot 栈
#    (vLLM 0.1.dev20073 / torch 2.13.0+cu130) 在同一条 c2 私有 pinned 路径上
#    store 累计 99.3GB、零 Xid（09-18 P6 深测；且实例能启动本身就证明 c2 生效，
#    否则共享区在 PP>1 必然 30s 超时）；而 0.30.0 同路径、同 CUDA 崩：
#    C++ DMA(cuMemcpyBatchAsync) → error 1 / CUDA_ERROR_INVALID_VALUE（09-23），
#    Triton SM 内核 → MMU Fault VIRT_WRITE → Xid31（09-27）。两栈的
#    _custom_ops.swap_blocks_batch 逐字相同、torch/CUDA 版本相同 → 差异收敛在
#    各自编译的 C++ kernel（csrc/cache_kernels.cu 的 swap_blocks_batch）或其
#    调用参数。定论前保持 FN_KVOFF=0：本机负载下它收益本就为 0
#    （0929 实测 21.5h 零外部回载、GPU 池自扛 ~90% 命中，代价却是 107GB pinned）。
# ---------------------------------------------------------------------------
def _patch_cpu_spec(m):
    if _DISABLED:
        return
    cls = getattr(m, "CPUOffloadingSpec", None)
    orig = getattr(cls, "_uses_shared_region", None)
    if cls is None or orig is None or getattr(orig, "_dsh_c2", False):
        return

    def patched(self):
        # [c8] 公共区协商成功 ⇒ 走共享 mmap 区（偏移由 _LAYOUT_CTX 决定）
        if getattr(self, "_dsh_c8", None) is not None:
            return True
        par = getattr(getattr(self, "config", None), "parallel", None)
        if par is not None and getattr(par, "pp_size", 1) > 1:
            if not getattr(cls, "_dsh_c2_logged", False):
                cls._dsh_c2_logged = True
                _log(
                    "[dsh-kvoff c2] pp_size=%s>1 -> CPU KV 走每 rank 私有 pinned 缓冲"
                    "（上游共享区行布局要求各 rank row stride 相同，PP 下不成立；"
                    "c8 公共区可用时不会走到这里）" % (par.pp_size,)
                )
            return False
        return orig(self)

    patched._dsh_c2 = True
    cls._uses_shared_region = patched

    # ---- [c8] 区域类换成「前缀和偏移」版 ----
    region = getattr(m, "SharedOffloadRegion", None)
    if region is not None and region.__name__ != "DshPpSharedRegion":
        m.SharedOffloadRegion = _c8_region_cls(region)
        _log("c8: spec.SharedOffloadRegion -> DshPpSharedRegion（前缀和偏移）")

    # ---- [c8] worker 侧：create_worker 之前协商布局 ----
    orig_cw = cls.create_worker
    if not getattr(orig_cw, "_dsh_c8", False):

        def create_worker(self, kv_caches):
            if getattr(self, "_dsh_c8", None) is None and not getattr(
                self, "_dsh_c8_tried", False
            ):
                self._dsh_c8_tried = True
                try:
                    own_required = sum(
                        int(t.page_size_bytes) for t in kv_caches.tensors
                    ) * int(self.blocks_per_chunk)
                except Exception:
                    own_required = 0
                try:
                    _c8_resolve(self, publish=True, own_required=own_required)
                except Exception as exc:
                    _log("c8: 协商异常 -> 退回私有缓冲：%s" % exc)
            ctx = getattr(self, "_dsh_c8", None)
            if ctx is None:
                return orig_cw(self, kv_caches)
            _LAYOUT_CTX[0] = ctx
            try:
                return orig_cw(self, kv_caches)
            finally:
                _LAYOUT_CTX[0] = None

        create_worker._dsh_c8 = True
        cls.create_worker = create_worker

    # ---- [c8] 调度器侧：get_manager 之前采纳同一套 num_chunks ----
    orig_gm = cls.get_manager
    if not getattr(orig_gm, "_dsh_c8", False):

        def get_manager(self):
            if getattr(self, "_dsh_c8", None) is None and not getattr(
                self, "_dsh_c8_tried", False
            ):
                self._dsh_c8_tried = True
                # 这里**允许**抛错：调度器侧协商不成而 worker 侧已建公共区时，
                # chunk id 空间会与 worker 行数不一致（越界写 -> Xid31），
                # 必须拒绝启动而不是带着错几何跑。
                _c8_resolve(self, publish=False)
            return orig_gm(self)

        get_manager._dsh_c8 = True
        cls.get_manager = get_manager


# ---------------------------------------------------------------------------
# c7：swap 全走 Triton（绕开 cuMemcpyBatchAsync）
# ---------------------------------------------------------------------------
def _patch_swap_triton(m):
    if _DISABLED:
        return
    if getattr(m, "MIN_N", None):
        m.MIN_N = 0
        _log("c7: swap_blocks_triton.MIN_N -> 0（小批量不再回落 C++ 批量拷贝）")


def _patch_cpu_gpu_worker(m):
    if _DISABLED:
        return
    # ---- c7: _select_swap_blocks_fn 仅 load 方向强制 Triton ----
    sel = getattr(m, "_select_swap_blocks_fn", None)
    if sel is not None and not getattr(sel, "_dsh_c7", False):

        def sel_patched(layer_refs_per_group, gpu_to_cpu):
            # ⚠️ 铁律：GPU→CPU（store）绝不能走 Triton。上游 gpu_worker.py:41-43
            # 显式让该方向走拷贝引擎；2026-09-27 实测强行切 Triton 会在
            # store 时让 SM 内核解引用 host 指针 → MMU Fault VIRT_WRITE → Xid31
            # → EngineDead（比 C++ 的 error 1 危险得多：Xid31 可致 GPU 降级）。
            if gpu_to_cpu:
                return sel(layer_refs_per_group, gpu_to_cpu)
            if (
                getattr(m, "HAS_TRITON", False)
                and not m.current_platform.is_xpu()
                and not m.current_platform.is_rocm()
            ):
                page_sizes = [
                    r.page_size_bytes
                    for g in layer_refs_per_group
                    for r in g
                ]
                if page_sizes and not any(s % 8 for s in page_sizes):
                    chunk = min(m.triton.next_power_of_2(max(page_sizes)), 8192)
                    import functools as _f

                    return _f.partial(m.swap_blocks_batch, bytes_per_chunk=chunk)
                m.logger.warning(
                    "[dsh-kvoff c7] 页大小非 8B 对齐，回落原选择逻辑"
                    "（该形态下 cuMemcpyBatchAsync 可能冻结流）",
                )
            return sel(layer_refs_per_group, gpu_to_cpu)

        sel_patched._dsh_c7 = True
        m._select_swap_blocks_fn = sel_patched
        _log("c7: 仅 CPU->GPU(load) 方向强制 Triton swap 内核；"
             "store 方向保持上游 C++ DMA（Triton 会 MMU fault）")

    # ---- c6a：store 方向熔断 + 有界等待（SingleDirectionOffloadingHandler） ----
    H = getattr(m, "SingleDirectionOffloadingHandler", None)
    if H is not None and not getattr(H, "_dsh_c6", False):
        H._dsh_c6 = True
        orig_transfer_async = H.transfer_async
        orig_wait = H.wait

        def transfer_async(self, job_id, src_spec, dst_spec):
            if self.gpu_to_cpu and getattr(self, "_dsh_stalled", False):
                return False  # 熔断后拒绝入队，由 connector 走失败 ack
            return orig_transfer_async(self, job_id, src_spec, dst_spec)

        def wait(self, job_ids):
            """有界等待；返回超时未完成的 job 集合（原版返回 None）。"""
            if self.gpu_to_cpu and getattr(self, "_dsh_stalled", False):
                return set(self._transfer_events.keys())
            timeout = float(os.environ.get("FN_KVOFF_WAIT_TIMEOUT", "15"))
            deadline = time.monotonic() + timeout
            for job_id in job_ids:
                event = self._transfer_events.get(job_id)
                if event is None:
                    continue
                while not event.query():
                    if time.monotonic() >= deadline:
                        if self.gpu_to_cpu:
                            self._dsh_stalled = True
                            try:
                                head = self._transfers[0] if self._transfers else None
                                m.logger.error(
                                    "[dsh-kvoff c6] FUSE: store 方向 job=%s 未在"
                                    " %.1fs 内完成 -> 熔断；chain_len=%d head_job=%s",
                                    job_id, timeout, len(self._transfers),
                                    getattr(head, "job_id", None),
                                )
                            except Exception:
                                pass
                        else:
                            m.logger.error(
                                "[dsh-kvoff c6] load 方向 job=%s 等待超时 %.1fs"
                                "（不熔断，交回上层按未完成处理）", job_id, timeout,
                            )
                        return set(self._transfer_events.keys())
                    time.sleep(0.002)
            return set()

        H.transfer_async = transfer_async
        H.wait = wait

    # ---- c6a：CPUOffloadingWorker.wait 透传未完成集合 ----
    W = getattr(m, "CPUOffloadingWorker", None)
    if W is not None and not getattr(W, "_dsh_c6w", False):
        W._dsh_c6w = True

        def worker_wait(self, job_ids):
            incomplete = set()
            for h in (self._store_handler, self._load_handler):
                r = h.wait(job_ids)
                if r:
                    incomplete |= set(r)
            return incomplete

        W.wait = worker_wait


# ---------------------------------------------------------------------------
# c6b：WorkerMetadata 增加 failed_jobs / fuse_tripped 通道
# ---------------------------------------------------------------------------
def _patch_offloading_common(m):
    if _DISABLED:
        return
    M = getattr(m, "OffloadingWorkerMetadata", None)
    if M is None or getattr(M, "_dsh_c6m", False):
        return
    M._dsh_c6m = True
    orig_init = M.__init__

    def init(self, *a, **k):
        fj = k.pop("failed_jobs", None)
        ft = k.pop("fuse_tripped", False)
        orig_init(self, *a, **k)
        self.failed_jobs = dict(fj) if fj else {}
        self.fuse_tripped = bool(ft)

    def aggregate(self, other):
        merged = M(
            completed_jobs=dict(self.completed_jobs),
            transfer_stats=self.transfer_stats.aggregate(other.transfer_stats),
        )
        fj = dict(self.failed_jobs)
        for jid, cnt in getattr(other, "failed_jobs", {}).items():
            fj[jid] = fj.get(jid, 0) + cnt
        merged.failed_jobs = fj
        merged.fuse_tripped = bool(
            getattr(self, "fuse_tripped", False) or getattr(other, "fuse_tripped", False)
        )
        return merged

    M.__init__ = init
    M.aggregate = aggregate


# ---------------------------------------------------------------------------
# c6c：connector worker 失败 ack / 只读降级 / 迟到完成去重
# ---------------------------------------------------------------------------
def _patch_offloading_worker(m):
    if _DISABLED:
        return
    C = getattr(m, "OffloadingConnectorWorker", None)
    if C is None or getattr(C, "_dsh_c6", False):
        return
    C._dsh_c6 = True
    M = m.OffloadingWorkerMetadata

    orig_init = C.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self._dsh_failed_jobs = set()
        self._dsh_store_fused = False

    def _fail(self, job_id, fuse=False):
        self._dsh_failed_jobs.add(job_id)
        self._connector_worker_meta.failed_jobs[job_id] = (
            self._connector_worker_meta.failed_jobs.get(job_id, 0) + 1
        )
        if fuse:
            self._dsh_store_fused = True
            self._connector_worker_meta.fuse_tripped = True

    def _submit_stores_drained(self):
        """排空 _unsubmitted_store_jobs：熔断不阻塞提交队列（否则调度器
        has_pending_push_work 会等不到 ack 而空转死锁）。失败一律 ack。"""
        if self._dsh_store_fused:
            pending = self._unsubmitted_store_jobs
            self._unsubmitted_store_jobs = []
            for job_id, _src, _dst in pending:
                _fail(self, job_id)
            return
        pending = self._unsubmitted_store_jobs
        self._unsubmitted_store_jobs = []
        for job_id, src_spec, dst_spec in pending:
            try:
                ok = self.worker.submit_store(job_id, src_spec, dst_spec)
            except Exception:
                m.logger.exception(
                    "[dsh-kvoff c6] submit_store(job=%s) 抛错 -> 失败 ack + 熔断", job_id
                )
                ok = False
            if not ok:
                _fail(self, job_id, fuse=True)

    def handle_preemptions(self, kv_connector_metadata):
        assert self.worker is not None
        if kv_connector_metadata.jobs_to_flush:
            for job_id in kv_connector_metadata.jobs_to_flush:
                entry = kv_connector_metadata.store_jobs.pop(job_id, None)
                if entry is not None:
                    if not self._is_store_writer:
                        self._connector_worker_meta.mark_completed(job_id)
                        continue
                    self._unsubmitted_store_jobs.append(
                        (job_id, entry.src_spec, entry.dst_spec)
                    )
        _submit_stores_drained(self)
        if kv_connector_metadata.jobs_to_flush:
            if not set(kv_connector_metadata.jobs_to_flush).issubset(
                self._dsh_failed_jobs
            ):
                try:
                    incomplete = self.worker.wait(
                        kv_connector_metadata.jobs_to_flush
                    ) or set()
                except Exception:
                    m.logger.exception(
                        "[dsh-kvoff c6] wait(jobs_to_flush) 抛错 -> 全量失败 ack"
                    )
                    incomplete = set(kv_connector_metadata.jobs_to_flush)
                for job_id in set(kv_connector_metadata.jobs_to_flush) & set(incomplete):
                    _fail(self, job_id, fuse=True)

    def start_kv_transfers(self, metadata):
        assert self.worker is not None
        _submit_stores_drained(self)
        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            success = self.worker.submit_load(
                job_id, entry.src_spec, entry.dst_spec
            )
            # load 正确性不可妥协：维持原版 assert 语义
            assert success

    orig_prepare = C.prepare_store_kv

    def prepare_store_kv(self, metadata):
        if self._dsh_store_fused:
            for job_id in metadata.store_jobs:
                if not self._is_store_writer:
                    # 非 writer rank 本就不写数据：照常 completed ack，不算失败
                    self._connector_worker_meta.mark_completed(job_id)
                else:
                    _fail(self, job_id)
            return
        orig_prepare(self, metadata)

    def get_finished(self, finished_req_ids):
        assert self.worker is not None
        finished_recving = set()
        for transfer_result in self.worker.get_finished():
            job_id = transfer_result.job_id
            if job_id in self._dsh_failed_jobs:
                # 迟到的“完成”（多半是熔断路径上的残骸）：不双计
                continue
            if not transfer_result.success:
                _fail(self, job_id, fuse=True)
                continue
            is_load = job_id in self._load_jobs
            if (
                transfer_result.transfer_time is not None
                and transfer_result.transfer_size is not None
            ):
                stats = (
                    self._connector_worker_meta.transfer_stats.load
                    if is_load
                    else self._connector_worker_meta.transfer_stats.store
                )
                stats.record(
                    transfer_result.transfer_size, transfer_result.transfer_time
                )
            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)
        return set(), finished_recving

    def build_connector_worker_meta(self):
        meta = self._connector_worker_meta
        if (
            not meta.completed_jobs
            and not meta.failed_jobs
            and not meta.fuse_tripped
        ):
            return None
        self._connector_worker_meta = M()
        return meta

    C.__init__ = init
    C.handle_preemptions = handle_preemptions
    C.start_kv_transfers = start_kv_transfers
    C.prepare_store_kv = prepare_store_kv
    C.get_finished = get_finished
    C.build_connector_worker_meta = build_connector_worker_meta


# ---------------------------------------------------------------------------
# c6d：scheduler 失败登记撤销 + 熔断只读降级
# ---------------------------------------------------------------------------
def _patch_offloading_scheduler(m):
    if _DISABLED:
        return
    S = getattr(m, "OffloadingConnectorScheduler", None)
    M = getattr(m, "OffloadingWorkerMetadata", None)
    if S is None or M is None or getattr(S, "_dsh_c6", False):
        return
    for attr in (
        "update_connector_output",
        "_build_aligned_boundary_store_jobs",
        "_build_partial_tail_store_jobs",
        "_remove_pending_job",
    ):
        if not hasattr(S, attr):
            _log("c6d: scheduler 类缺少 %s（版本漂移？）-> 跳过 scheduler 补丁" % attr)
            return
    for attr in ("OffloadingConnectorStats", "_TransferMetricName", "logger"):
        if not hasattr(m, attr):
            _log("c6d: scheduler 模块缺少 %s（版本漂移？）-> 跳过 scheduler 补丁" % attr)
            return
    S._dsh_c6 = True

    # ---- 完整替换 update_connector_output：与原版逐行等价，外加
    #      failed_jobs→success=False 收尾与 fuse 只读降级 ----
    OffloadingConnectorStats = m.OffloadingConnectorStats
    _TransferMetricName = m._TransferMetricName
    logger = m.logger

    def update_connector_output(self, connector_output):
        if not hasattr(self, "_dsh_failed_seen"):
            self._dsh_failed_seen = set()
            self._dsh_stores_disabled = False
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, M):
            assert meta is None
            meta = M()
        if not meta.transfer_stats.is_empty():
            transfer_stats = OffloadingConnectorStats()
            if not meta.transfer_stats.load.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_BYTES, meta.transfer_stats.load.bytes
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_TIME, meta.transfer_stats.load.time
                )
                for size in meta.transfer_stats.load.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.LOAD_SIZE, size
                    )
            if not meta.transfer_stats.store.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_BYTES, meta.transfer_stats.store.bytes
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_TIME, meta.transfer_stats.store.time
                )
                for size in meta.transfer_stats.store.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.STORE_SIZE, size
                    )
            self._connector_stats.aggregate(transfer_stats)

        # [dsh-kvoff c6] 熔断 -> 只读降级（不再新建 store 任务）
        if getattr(meta, "fuse_tripped", False) and not self._dsh_stores_disabled:
            self._dsh_stores_disabled = True
            logger.error(
                "[dsh-kvoff c6] worker store fuse tripped -> stop creating store "
                "jobs (read-only degrade; existing entries keep serving)"
            )
        for job_id in getattr(meta, "failed_jobs", {}) or {}:
            self._dsh_failed_seen.add(job_id)

        def _try_finish(job_id, count, failed_here):
            assert count > 0
            if job_id < self._stale_job_threshold:
                logger.debug(
                    "Skipping stale %s job %d (pre-reset counter: %d)",
                    "failed" if failed_here else "completed",
                    job_id,
                    self._stale_job_threshold,
                )
                return
            job_status = self._jobs.get(job_id)
            if job_status is None:
                return
            job_status.pending_count -= count
            if job_status.pending_count > 0:
                return
            assert job_status.pending_count == 0

            is_failed = failed_here or (job_id in self._dsh_failed_seen)
            req_status = self._req_status[job_status.req_id]
            if job_status.is_store:
                # [dsh-kvoff c6] 失败 store：success=False 撤登记 + 释放 CPU 块
                self.manager.complete_store(
                    job_status.keys, req_status.req_context, success=not is_failed
                )
            else:
                self.manager.complete_load(job_status.keys, req_status.req_context)
                if self._chunks_being_loaded:
                    self._chunks_being_loaded.difference_update(job_status.keys)
            if self._block_id_to_pending_jobs:
                self._remove_pending_job(job_id, job_status.fenced_block_ids)
                if req_status.req.is_finished():
                    self._remove_pending_job(
                        job_id, job_status.deferred_fence_block_ids
                    )
            del self._jobs[job_id]
            req_status.transfer_jobs.discard(job_id)
            self._dsh_failed_seen.discard(job_id)
            if req_status.finished_signaled and not req_status.transfer_jobs:
                del self._req_status[job_status.req_id]

        for job_id, count in meta.completed_jobs.items():
            _try_finish(job_id, count, False)
        for job_id, count in (getattr(meta, "failed_jobs", {}) or {}).items():
            if job_id in self._jobs:
                _try_finish(job_id, count, True)

    def aligned_guarded(self, *a, **k):
        if getattr(self, "_dsh_stores_disabled", False):
            return {}
        return _orig_aligned(self, *a, **k)

    def partial_guarded(self, *a, **k):
        if getattr(self, "_dsh_stores_disabled", False):
            return {}
        return _orig_partial(self, *a, **k)

    _orig_aligned = S._build_aligned_boundary_store_jobs
    _orig_partial = S._build_partial_tail_store_jobs
    S.update_connector_output = update_connector_output
    S._build_aligned_boundary_store_jobs = aligned_guarded
    S._build_partial_tail_store_jobs = partial_guarded


PATCHES = {
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading.config":
        _patch_offloading_config,
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading.common":
        _patch_offloading_common,
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker":
        _patch_offloading_worker,
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler":
        _patch_offloading_scheduler,
    "vllm.v1.kv_offload.cpu.spec": _patch_cpu_spec,
    "vllm.v1.kv_offload.cpu.gpu_worker": _patch_cpu_gpu_worker,
    "vllm.v1.kv_offload.cpu.swap_blocks_triton": _patch_swap_triton,
}
