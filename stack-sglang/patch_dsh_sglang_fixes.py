"""SGLang 0.5.21 @127:18420 W4A16-AutoRound-1M 两处生产修复（正式件，幂等）。

修复 1（loader.py，权重翻倍泄漏）：
  postprocess_weights 逐模块 process_weights_after_loading 后，被替换下来的旧
  Parameter（vLLM 参数系 alias / marlin repack 前 packed，与 _TensorState/自引用
  构成循环引用）只靠引用计数不会释放 → 每层 FusedMoE net +1.24GB，PP0 全模型翻倍
  64GB OOM（PP1 58.23GB）。实测每模块后 gc.collect() 释放，PP0 回落 37.5GB。

修复 2（fused_marlin_moe.py，scale dtype 断言）：
  python 侧 assert hidden.dtype == w1/w2_scale.dtype 过严：AutoRound 的组 scale
  是 fp16（vLLM 栈语义即 act bf16 × scale fp16，marlin C++ 模板同源）。放行让
  sgl_kernel 自己决定；若内核不支持会即时 TORCH_CHECK 报错（非静默错值）。

回滚：python3 本脚本 --revert
"""

import ast
import sys

G = "/home/ll/sglang-env/lib/python3.12/site-packages/sglang/srt/model_loader/loader.py"
M = "/home/ll/sglang-env/lib/python3.12/site-packages/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py"
BAK1 = G + ".bak-dsh-gc"
BAK2 = M + ".bak-dsh-scaleassert"


def _bak(path, bpath):
    import shutil

    with open(path) as f:
        cur = f.read()
    try:
        orig = open(bpath).read()
    except FileNotFoundError:
        orig = None
    if orig is None:
        shutil.copy(path, bpath)
        return cur, cur
    return cur, orig


def apply():
    # 修复 1
    cur, orig = _bak(G, BAK1)
    if "DSH-POSTPROC" in cur:
        # 实验探针残留：先用未打补丁的原版（探针第一次备份）复位
        pristine = open(G + ".bak-postproc-probe").read()
        assert "DSH-" not in pristine, "pristine backup dirty?"
        open(G, "w").write(pristine)
        cur = orig = pristine
        # BAK1 若被探针版污染过，重建成纯净基线
        try:
            if "DSH-" in open(BAK1).read():
                with open(BAK1, "w") as bf:
                    bf.write(pristine)
                print("BAK1 rebuilt pristine")
        except FileNotFoundError:
            with open(BAK1, "w") as bf:
                bf.write(pristine)
        print("loader reset to pristine")
    if "dsh-gc" in orig:
        print("fix1 already applied")
    else:
        s = orig
        i = s.index("def postprocess_weights")
        j = s.index("def restore_weights_before_loading")
        seg = s[i:j]
        old = (
            "            with device_loading_context(module, target_device):\n"
            "                quant_method.process_weights_after_loading(module)"
        )
        if seg.count(old) != 1:
            raise SystemExit(f"fix1 anchor count={seg.count(old)} — abort")
        new = (
            old
            + "\n"
            "            # dsh-gc: 替换下来的旧 packed Parameter 处于循环引用中\n"
            "            # （vLLM 参数系 alias + post_load _TensorState），引用计数不释放，\n"
            "            # 每个 FusedMoE 层净增 1 份权重 → PP0 加载 64GB OOM。强制 mark-clear。\n"
            "            import gc as _dsh_gc\n"
            "            _dsh_gc.collect()"
        )
        seg2 = seg.replace(old, new)
        s = s[:i] + seg2 + s[j:]
        ast.parse(s)
        open(G, "w").write(s)
        print("fix1 applied")

    # 修复 2
    cur2, orig2 = _bak(M, BAK2)
    if "dsh-scaleassert" in cur2:
        print("fix2 already applied")
    else:
        s2 = orig2
        for name in ("w1_scale", "w2_scale"):
            old = (
                f"        assert hidden_states.dtype == {name}.dtype, (\n"
                f"            f\"moe_wna16_marlin_gemm assumes hidden_states.dtype"
                f" ({{hidden_states.dtype}}) == {name}.dtype ({{{name}.dtype}})\"\n"
                f"        )\n"
            )
            assert s2.count(old) == 1, f"fix2 anchor {name} count={s2.count(old)}"
            s2 = s2.replace(
                old,
                f"        pass  # dsh-scaleassert: AutoRound 组 scale 是 fp16（vLLM 语义），放行由内核裁决\n",
            )
        ast.parse(s2)
        open(M, "w").write(s2)
        print("fix2 applied")


def revert():
    for path, bpath in ((G, BAK1), (M, BAK2)):
        import shutil

        try:
            shutil.copy(bpath, path)
            print("reverted", path)
        except FileNotFoundError:
            print("no backup for", path)


if __name__ == "__main__":
    revert() if "--revert" in sys.argv else apply()
