#!/bin/bash
# ---- sudo 凭据（脱敏版）----
# 生产机上 chroot 内进程属 root，宿主脚本需要 sudo。两种给法二选一：
#   A) 推荐：/etc/sudoers.d/ 配 NOPASSWD（见 docs/01-host-prep.md），无需本变量
#   B) 环境变量 SUDO_PASS 传入密码（不落盘、不写进脚本）
if [ -n "${SUDO_PASS:-}" ]; then
  SUDO() { printf '%s\n' "$SUDO_PASS" | sudo -S -p '' "$@"; }
else
  SUDO() { sudo -n "$@"; }
fi

# 给 chroot 环境挂载必要文件系统 + 校验 NVIDIA 用户态库（幂等，可反复执行）
# 重建于 2026-09-19（系统重装后 /etc 与 /usr/local 丢失，chroot 副本库仍在数据盘）
set -u
R=/media/ll/data/vllm-image/rootfs
LOG=/var/log/setup-chroot.log
VER=610.43.03

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

[ -d "$R" ] || { log "ERROR: rootfs 不存在: $R"; exit 1; }
[ -x "$R/usr/bin/python3.12" ] || { log "ERROR: chroot 内没有 python3.12"; exit 1; }

log "===== setup-chroot 开始 ====="

# --- 挂载点准备 ---
mkdir -p "$R/proc" "$R/sys" "$R/dev" "$R/dev/pts" "$R/dev/shm" "$R/run" "$R/tmp"

# --- 幂等挂载 ---
mnt_bind(){
  local src=$1 dst=$2
  if mountpoint -q "$dst" 2>/dev/null; then log "跳过(已挂载): $dst"; return 0; fi
  if mount --bind "$src" "$dst" 2>/dev/null; then log "挂载成功(bind): $src -> $dst"; else log "ERROR 挂载失败: $src -> $dst"; fi
}
mnt_fs(){
  local type=$1 dst=$2
  if mountpoint -q "$dst" 2>/dev/null; then log "跳过(已挂载): $dst"; return 0; fi
  if mount -t "$type" "$type" "$dst" 2>/dev/null; then log "挂载成功($type): $dst"; else log "ERROR 挂载失败: $dst"; fi
}

mnt_fs proc  "$R/proc"
mnt_fs sysfs "$R/sys"
mnt_bind /dev     "$R/dev"
mnt_bind /dev/pts "$R/dev/pts"
mnt_bind /dev/shm "$R/dev/shm"
mnt_bind /run     "$R/run"
mnt_bind /tmp     "$R/tmp"

  # 【2026-09-19 补齐】数据盘与 deploy：缺失会导致 chroot 内看不到模型与启动脚本
  mnt_bind /media/ll/data "$R/media/ll/data"
  mnt_bind /home/ll/deploy "$R/home/ll/deploy"

# --- NVIDIA 设备节点可见性 ---
for n in nvidiactl nvidia-uvm nvidia-uvm-tools nvidia0; do
  if [ -e "$R/dev/$n" ]; then log "设备节点 OK: /dev/$n"; else log "WARN 设备节点缺失: /dev/$n（GPU 可能未就绪）"; fi
done

# --- NVIDIA 用户态库：bind-mount 主机真实库到 chroot 内的 0 字节占位文件 ---
# （原设计如此：占位文件是挂载点，这样库永远跟随主机驱动版本，无需拷贝 300MB）
LIBDIR="$R/usr/lib/x86_64-linux-gnu"
bound=0; skipped=0
for f in "$LIBDIR"/libcuda.so.* "$LIBDIR"/libnvidia-*.so.* ; do
  [ -f "$f" ] || continue
  base=$(basename "$f")
  src="/usr/lib/x86_64-linux-gnu/$base"
  [ -s "$src" ] || continue
  if mountpoint -q "$f" 2>/dev/null; then skipped=$((skipped+1)); continue; fi
  if [ -s "$f" ]; then skipped=$((skipped+1)); continue; fi
  if mount --bind "$src" "$f" 2>/dev/null; then bound=$((bound+1)); else log "WARN bind 失败: $base"; fi
done
log "NVIDIA 库 bind-mount: 新挂载=$bound 已就绪=$skipped"

# --- 验证 libcuda 真的可加载（空壳检测：文件大小为 0 即为未挂载）---
if [ ! -s "$LIBDIR/libcuda.so.1" ] && [ ! -L "$LIBDIR/libcuda.so.1" ]; then
  log "ERROR libcuda.so.1 不可用（size 0 且非软链）"
fi
CUDA_REAL=$(readlink -f "$LIBDIR/libcuda.so.1" 2>/dev/null)
if [ -n "$CUDA_REAL" ] && [ -s "$CUDA_REAL" ]; then
  log "libcuda 解析成功: $CUDA_REAL ($(stat -Lc %s "$CUDA_REAL") 字节)"
else
  log "ERROR libcuda.so.1 解析后为空壳，chroot 内 CUDA 将不可用"
fi

# --- /usr/local/cuda 软链（部分脚本按此路径找 CUDA）---
if [ -d "$R/usr/local/cuda-13.0" ] && [ ! -e "$R/usr/local/cuda" ]; then
  ln -sfn /usr/local/cuda-13.0 "$R/usr/local/cuda" && log "创建 /usr/local/cuda -> cuda-13.0"
fi

# --- chroot 内自检 ---
if chroot "$R" /usr/bin/python3.12 -c "import torch;print('torch',torch.__version__,'cuda_dev',torch.cuda.device_count())" 2>/dev/null; then
  log "chroot 自检通过：torch 可导入且能看到 GPU"
else
  log "WARN chroot 自检未通过（GPU 数量为 0 或 torch 导入失败，详见上方日志）"
fi

log "===== setup-chroot 结束 ====="
exit 0
