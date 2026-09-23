#!/bin/bash
# fetch-image-rootfs.sh —— 不用 docker，直接从镜像仓库把 vLLM 定制镜像解成 rootfs。
#
# 为什么不用 docker：生产机走 chroot（见 docs/02），装 docker 反而多一套 daemon；
# 且本机历史上无 docker（`nvidia-smi topo`/GDS 路线才需要 docker，本方案不需要）。
#
# 镜像（务必按 digest 锁定，tag 会被上游重推）：
#   vllm/vllm-openai:qwen38-flash-next
#   多架构 index digest = sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8
#   其中 linux/amd64 manifest = sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3
#   （2026-09-23 用 `curl -D - .../manifests/qwen38-flash-next` 复核 docker-content-digest 一致）
#
# 用法：
#   bash fetch-image-rootfs.sh [目标目录]
#   默认目标 /media/ll/data/vllm-image
#
# 产物：<目标>/layers.txt（层序）、<目标>/blobs/sha256:<hex>（gzip 层）、<目标>/rootfs/（解出的根）
# 断点续传：已存在且大小一致的 blob 跳过；解包可重复执行（后层覆盖前层，whiteout 已处理）。
set -uo pipefail

DEST=${1:-/media/ll/data/vllm-image}
REG=registry-1.docker.io
REPO=vllm/vllm-openai
TAG=${TAG:-qwen38-flash-next}
ARCH_MANIFEST_DIGEST=${ARCH_MANIFEST_DIGEST:-sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3}
LOG=$DEST/fetch.log
mkdir -p "$DEST/blobs" "$DEST/rootfs"
say(){ echo "$(date +%T) $*" | tee -a "$LOG"; }

TOK=$(curl -fsSL "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$REPO:pull" \
      | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
[ -n "$TOK" ] || { echo "取 token 失败"; exit 1; }

# 1) 取 amd64 manifest（已按 digest 固定，避免 tag 漂移）
DIG=$(printf '%s' "$ARCH_MANIFEST_DIGEST" | tr -d ':')
if [ ! -s "$DEST/amd64.json" ]; then
  curl -fsSL -H "Authorization: Bearer $TOK" \
    -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
    "https://$REG/v2/$REPO/manifests/$ARCH_MANIFEST_DIGEST" -o "$DEST/amd64.json"
fi
python3 - "$DEST/amd64.json" "$DEST/layers.txt" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert m["mediaType"].endswith("manifest.v2+json"), "不是单架构 manifest"
open(sys.argv[2],"w").write("".join(l["digest"]+"\n" for l in m["layers"]))
print(f"manifest ok, {len(m['layers'])} layers, config={m['config']['digest']}")
PY
say "$(python3 -c "import json;m=json.load(open('$DEST/amd64.json'));print(len(m['layers']),'layers')") ready"

# 2) 逐层下载（带 .done 标记 + 大小校验）
n=0; total=$(wc -l < "$DEST/layers.txt")
while read -r d; do
  n=$((n+1))
  f="$DEST/blobs/$d"
  sz=$(python3 -c "
import json;m=json.load(open('$DEST/amd64.json'))
print(next(l['size'] for l in m['layers'] if l['digest']=='$d'))")
  if [ -f "$f.done" ] && [ "$(stat -c %s "$f" 2>/dev/null)" = "$sz" ]; then
    say "layer $n/$total cached ($sz)"; continue
  fi
  say "layer $n/$total downloading"
  curl -fsSL -H "Authorization: Bearer $TOK" -o "$f" \
    "https://$REG/v2/$REPO/blobs/$d" || { say "layer $n/$total FAILED"; exit 1; }
  [ "$(stat -c %s "$f")" = "$sz" ] || { say "layer $n/$total size mismatch"; exit 1; }
  touch "$f.done"; say "layer $n/$total ready ($sz)"
done < "$DEST/layers.txt"

# 3) 按层序解包到 rootfs（后层覆盖前层；处理 .wh. whiteout）
n=0
while read -r d; do
  n=$((n+1))
  tar -xzf "$DEST/blobs/$d" -C "$DEST/rootfs" 2>/dev/null || true
  find "$DEST/rootfs" -name '.wh.*' 2>/dev/null | while read -r w; do
    b=$(basename "$w"); dd=$(dirname "$w")
    if [ "$b" = ".wh..wh..opq" ]; then
      find "$dd" -mindepth 1 -maxdepth 1 ! -name '.wh.*' -exec rm -rf {} + 2>/dev/null
    else
      rm -rf "$dd/${b#.wh.}"
    fi
    rm -f "$w"
  done
  [ $((n % 8)) = 0 ] && say "extracted $n/$total"
done < "$DEST/layers.txt"
touch "$DEST/rootfs.done"
say "EXTRACTION DONE -> $DEST/rootfs"
say "下一步：bash host-prep.sh 之类的主机准备 + python3 scripts/apply-patches.py --target $DEST/rootfs/usr/local/lib/python3.12/dist-packages/vllm --apply"
