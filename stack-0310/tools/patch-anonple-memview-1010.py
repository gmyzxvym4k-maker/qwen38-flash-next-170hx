#!/usr/bin/env python3
# [anon-ple-fix-1010] 修复匿名堆加载函数在 2-D 形状上的崩溃。
#
# 现场：10-10 08:17 启动（FN_PLE_LOC=heap → DSH_PLE_MEM_RESIDENT=1）在
#   sitecustomize.py:_dsh_read_into_anon 抛
#   NotImplementedError: memoryview slice assignments are currently restricted to ndim = 1
#   ⇒ Worker_PP0 起不来 → EngineCore failed to start → 18420 离线。
# 原因：memoryview(ndarray.data) 继承数组形状；调用点 allocate_embedding_weight 传的是
#   2-D shape=(num_embeddings, embedding_dim)，故 mv[pos:pos+n] = bytes 被 CPython 拒绝。
#   （scale 那处传 1-D 所以从没炸过。）
# 修法：memoryview(arr).cast("B") —— 同一缓冲的 1-D 字节视图，写入语义等价。
#
# 幂等：已打过则直接退出；--revert 还原 .bak-anonplefix-1010。
import ast
import glob
import os
import shutil
import sys

F = "/home/ll/deploy/vllm-0310/patches/sitecustomize.py"
BAK = F + ".bak-anonplefix-1010"
PYC = "/home/ll/deploy/vllm-0310/patches/__pycache__"

OLD = '    mv = memoryview(arr.data)\n'
NEW = ('    # [anon-ple-fix-1010] memoryview(ndarray.data) 继承数组形状，2-D 时\n'
       '    # mv[a:b] = bytes 抛 NotImplementedError（CPython 只允许 ndim=1）。\n'
       '    # cast("B") 取同一缓冲的 1-D 字节视图，写入语义等价、无额外拷贝。\n'
       '    mv = memoryview(arr).cast("B")\n')


def revert():
    if not os.path.exists(BAK):
        print("no backup, nothing to revert")
        return 1
    shutil.copy2(BAK, F)
    print("reverted from", BAK)
    clean_pyc()
    return 0


def clean_pyc():
    for p in glob.glob(os.path.join(PYC, "*")):
        try:
            os.remove(p)
            print("rm", p)
        except OSError:
            pass


def selftest():
    """把补丁文件里的 _dsh_read_into_anon 单独取出来真跑一遍（2-D 形状 + 跨 128MiB 分块）。"""
    import numpy as np
    src = open(F, encoding="utf-8").read()
    tree = ast.parse(src)
    fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_dsh_read_into_anon":
            fn = node
            break
    if fn is None:
        print("FAIL: 找不到 _dsh_read_into_anon")
        return 1
    mod = ast.Module(body=[fn], type_ignores=[])
    g = {"os": os}
    exec(compile(mod, F, "exec"), g)
    f = g["_dsh_read_into_anon"]

    rows, cols = 1_000_000, 160          # 160,000,000 B ≈ 152.6 MiB ⇒ 跨 2 个 128MiB 分块（走线程池）
    data = os.urandom(rows * cols)
    tmp = "/tmp/anonple-selftest.bin"
    with open(tmp, "wb") as fh:
        fh.write(data)
    try:
        arr = f(tmp, np.int8, (rows, cols))
        ref = np.frombuffer(data, dtype=np.int8).reshape(rows, cols)
        if not np.array_equal(arr, ref):
            print("FAIL: 字节不一致")
            return 1
        if arr.base is not None:
            print("FAIL: arr.base 不为 None（不是自有匿名缓冲）")
            return 1
        # 1-D 形状（scale 档）也要成立
        s = f(tmp, np.uint16, (rows // 2,))
        if not np.array_equal(s, np.frombuffer(data, dtype=np.uint16)[:rows // 2]):
            print("FAIL: 1-D 档字节不一致")
            return 1
        print(f"PASS: 2-D {rows}x{cols} 逐字节一致 + 匿名缓冲 + 1-D 档一致")
        return 0
    finally:
        os.remove(tmp)


def main():
    if "--revert" in sys.argv:
        return revert()
    src = open(F, encoding="utf-8").read()
    if 'memoryview(arr).cast("B")' in src:
        print("already patched")
    else:
        n = src.count(OLD)
        if n != 1:
            print(f"anchor count={n}，中止（不猜测）")
            return 1
        if not os.path.exists(BAK):
            shutil.copy2(F, BAK)
            print("backup ->", BAK)
        open(F, "w", encoding="utf-8").write(src.replace(OLD, NEW))
        print("patched")
    clean_pyc()
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
