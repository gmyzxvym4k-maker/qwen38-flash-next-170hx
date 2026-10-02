"""sglang @127:18420 修正：GPTQ marlin MoE scale dtype 对齐 bf16（取代 dsh-scaleassert 放行）。

背景：sglang fused_marlin_moe 的 assert（act.dtype == scale.dtype）反映其 C++ 模板
配对（bf16×bf16 / fp16×fp16）。AutoRound 组 scale 是 fp16 → 被 bf16 模板误读 →
MoE 输出垃圾 → 首 token 即 EOS（matched_stop=248046）。

做法：还原 fused_marlin_moe.py 断言（撤 dsh-scaleassert），在
GPTQMarlinMoEKernel.process_weights_after_loading 的 w13/w2 scales replace 处把
permute 后的 scale 转成 bfloat16（与 act 同配对；scale 相对精度损失 ~2^-8，
数值影响 ≪ vLLM 栈 bf16→fp16 转换方向，可接受）。

幂等；--revert 恢复两文件到 .bak-dsh-scalebf16 / .bak-dsh-scaleassert。
"""

import ast
import shutil
import sys

G = "/home/ll/sglang-env/lib/python3.12/site-packages/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py"
M = "/home/ll/sglang-env/lib/python3.12/site-packages/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py"
BAK1 = G + ".bak-dsh-scalebf16"
BAK2 = M + ".bak-dsh-scaleassert"


def apply():
    # 1) 还原断言（撤销放行的 pass）
    cur2 = open(M).read()
    if "dsh-scaleassert" in cur2:
        shutil.copy(BAK2, M)
        print("assertions restored")

    # 2) scales 转 bf16
    s = open(G).read()
    if "dsh-scalebf16" in s:
        print("scales->bf16 already applied")
        return
    shutil.copy(G, BAK1)
    src = s
    for name in ("marlin_w13_scales", "marlin_w2_scales"):
        old = f'replace_parameter(layer, "{name.replace("marlin_", "")}", {name})'
        assert src.count(old) == 1, (name, src.count(old))
        new = (
            f"{name} = {name}.to(torch.bfloat16)  # dsh-scalebf16: 与 bf16 act 同配对\n        "
            + old
        )
        src = src.replace(old, new)
    ast.parse(src)
    open(G, "w").write(src)
    print("scales->bf16 applied")


def revert():
    for path, bpath in ((G, BAK1), (M, BAK2)):
        try:
            shutil.copy(bpath, path)
            print("reverted", path)
        except FileNotFoundError:
            print("no backup for", path)


if __name__ == "__main__":
    revert() if "--revert" in sys.argv else apply()
