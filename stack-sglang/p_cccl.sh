set -e
V=/home/ll/sglang-env/lib/python3.12/site-packages
F=$V/tilelang/jit/adapter/libgen.py
BAK=$F.bak-dsh-cccl
[ -f "$BAK" ] || cp $F $BAK
if grep -q "CCCL_DISABLE_CTK_COMPATIBILITY_CHECK" $F; then
  echo "cccl patch already applied"
else
  python3 - "$F" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
old='''                "-std=c++20",
                "-w",'''
assert s.count(old)==1, s.count(old)
new='''                "-std=c++20",
                # dsh-cccl: pip cu13 与系统 /usr/local/cuda(12.9) 混装时 nvcc 默认 include
                # 会命中 12.9 的 cuda.h，cccl 的 CTK 版本自检误报 incompatible（实测代码
                # 生成完全正常）。等价 vLLM 栈 FLASHINFER_EXTRA_CUDAFLAGS 同款旁路。
                "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
                "-w",'''
s=s.replace(old,new)
import ast; compile(s,p,"exec")
open(p,"w").write(s)
PY
  echo "cccl patch applied"
fi
/home/ll/sglang-env/bin/python -c "import tilelang.jit.adapter.libgen" && echo IMPORT_OK
F2=/home/ll/deploy/sglang-18420/sglang-inner.sh
sed -i "/export TL_DISABLE_FAST_MATH=1/d" $F2
bash -n $F2 && echo INNER_OK
rm -rf /root/.tilelang/cache 2>/dev/null || true
