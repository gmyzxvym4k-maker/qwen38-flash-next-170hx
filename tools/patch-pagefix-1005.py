#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动页缺省参数修复（126 机 32GB 内存红线，2026-10-05，逐行锚定版）
根因：
  1) index.html 弹窗脚本模型覆盖块把 pleLoc 写死 'heap'、kvOff 缺省 'simple'——
     本机 32GB 内存下 INT8 heap(48.3GiB 匿名堆) 必被 OOM 杀 → PleOffloadWorker 死
     → Engine core init failed → 看门狗循环重拉（本次故障签名）。
  2) server.js scriptModelDefaults() 没把 base 里已定版的 kvoff/pleInt8/pleLoc
     （10-05 hw-move：'0'/'1'/'disk'）透传给前端，前端只能写死。
修法：后端 defaults 透传 + 前端跟随 smd（缺省回落安全档），单一真源=SCRIPT_MODELS.base。
幂等标记：__pagefix_1005__
"""
import os, shutil, subprocess

MARK = "__pagefix_1005__"
SVR = "/home/ll/deploy/server.js"
IDX = "/home/ll/deploy/index.html"


def sub_once(path, old, new, check_js=False):
    s = open(path, encoding="utf-8").read()
    if old not in s and new in s:
        print("[skip] %s 该处已打过" % path)
        return
    n = s.count(old)
    assert n == 1, "锚点在 %s 出现 %d 次（需恰好 1 次）:\n%r" % (path, n, old[:120])
    if not os.path.exists(path + ".bak-pagefix-1005"):
        shutil.copy2(path, path + ".bak-pagefix-1005")
    s2 = s.replace(old, new)
    tmp = path + ".tmp-pagefix.js" if check_js else path + ".tmp-pagefix"  # node --check 只认 .js 扩展名
    open(tmp, "w", encoding="utf-8").write(s2)
    if check_js:
        r = subprocess.run(["node", "--check", tmp])
        assert r.returncode == 0, "node --check 失败"
    os.replace(tmp, path)
    print("[ok] patched %s" % path)


# ── 1) server.js：scriptModelDefaults() 透传 base 的三档 ──
SVR_OLD = "    longCtx: '0', maxModelLenLong: sm.maxModelLenLong || 0, maxModelLen512: sm.maxModelLen512 || 0,\n  };"
SVR_NEW = ("    longCtx: '0', maxModelLenLong: sm.maxModelLenLong || 0, maxModelLen512: sm.maxModelLen512 || 0,\n"
           "    // [pagefix-1005] PLE 驻留精度/位置与 CPU 二级缓存缺省跟随 base（10-05 换装定版已在 base\n"
           "    // 写明 kvoff/pleInt8/pleLoc）。本机 32GB 内存下 INT8 heap/BF16 heap/大 GiB 二级缓存均必 OOM，\n"
           "    // base 未声明时回落安全档：INT8+disk、二级缓存关。旧版前端把这三档写死导致启动页误炸。\n"
           "    pleInt8: String(b.pleInt8 || '1'), pleLoc: String(b.pleLoc || 'disk'),\n"
           "    kvoff: String(b.kvoff || '0'), kvoffGiB: b.kvoffGiB,  // %s\n"
           "  };") % MARK
sub_once(SVR, SVR_OLD, SVR_NEW, check_js=True)

# ── 2) index.html：smd 覆盖块三行改为跟随 smd ──
IDX_OLD_1 = "            kvOff: String(smd.kvoff || 'simple'), kvOffGiB: Number(smd.kvoffGiB || smd.kvOffGiB) || 96, // 0927 生产定版=SimpleCPU 96GiB"
IDX_NEW_1 = "            kvOff: String(smd.kvoff == null ? '0' : smd.kvoff), kvOffGiB: Number(smd.kvoffGiB || smd.kvOffGiB) || 48, // [pagefix-1005] 本机 32GB：二级缓存缺省关（开档仅手动，48GiB 参考）"
sub_once(IDX, IDX_OLD_1, IDX_NEW_1)

IDX_OLD_2 = "            pleInt8: '1',"
IDX_NEW_2 = "            pleInt8: String(smd.pleInt8 == null ? '1' : smd.pleInt8), // [pagefix-1005]"
sub_once(IDX, IDX_OLD_2, IDX_NEW_2)

IDX_OLD_3 = "            pleLoc: 'heap',"
IDX_NEW_3 = "            pleLoc: String(smd.pleLoc == null ? 'disk' : smd.pleLoc), // [pagefix-1005] 写死 heap 曾在 32GB 机直接 OOM（48.3GiB 匿名堆），改随 base=disk"
sub_once(IDX, IDX_OLD_3, IDX_NEW_3)

print("DONE")
