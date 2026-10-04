#!/usr/bin/env python3
# [FN-PLE-FADV] 给 chroot 镜像里 PLE 磁盘驻留加载器的三个 mmap fd 关闭预读放大。
# 根因：本机 31 GiB RAM 装不下 48 GiB n-gram 表，服务期是随机访问；全局
# read_ahead_kb=128 让每个缺页投机读 128 KB（约 800× 放大），把有用工作集
# 从页缓存里冲出去，decode/prefill 都在等盘，严重时 worker 卡死→RPC 超时→EngineDead。
# 正解：POSIX_FADV_RANDOM 置该 fd 的 file->f_ra.ra_pages=0（内核 open 快照语义下
# 对已持有 fd 生效——fadvise 改的就是那份 file->f_ra），缺页只读所需页。
# 用法：sudo python3 patch_ple_fadvise.py [--revert]
import sys

F = "/media/ll/data/vllm-image/rootfs/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/worker.py"
MARK = "[FN-PLE-FADV]"

EDITS = [
    ("        handle = open(path, \"rb\")\n", "handle", 8),
    ("            i8_handle = open(i8_path, \"rb\")\n", "i8_handle", 12),
    ("            sc_handle = open(sc_path, \"rb\")\n", "sc_handle", 12),
]


def block(fd: str, indent: int) -> str:
    p = " " * indent
    return (
        f"{p}# {MARK} 随机查表禁预读：160 B/行的随机访问按全局 read_ahead_kb=128\n"
        f"{p}# 预读会 ~800x 放大读量并冲刷页缓存；FADV_RANDOM 把该 fd 的\n"
        f"{p}# file->f_ra.ra_pages 置 0，随 fd 生命周期生效（权重文件不受影响）。\n"
        f"{p}import os as _os_fadv\n"
        f"{p}_os_fadv.posix_fadvise({fd}.fileno(), 0, 0, _os_fadv.POSIX_FADV_RANDOM)\n"
    )


def revert(src: str) -> str:
    lines = src.splitlines(keepends=True)
    out, i, removed = [], 0, 0
    while i < len(lines):
        if MARK in lines[i]:
            j = i
            while j < len(lines) and "_os_fadv.posix_fadvise" not in lines[j]:
                j += 1
            i = j + 1
            removed += 1
            continue
        out.append(lines[i])
        i += 1
    print(f"revert: 移除 {removed} 块")
    return "".join(out)


def apply(src: str) -> str:
    if MARK in src:
        print("already patched")
        return src
    for anchor, fd, indent in EDITS:
        if src.count(anchor) != 1:
            raise SystemExit(f"锚点非唯一（{src.count(anchor)}）：{anchor!r}")
        src = src.replace(anchor, anchor + block(fd, indent), 1)
    return src


def main() -> None:
    src = open(F, encoding="utf-8").read()
    out = revert(src) if "--revert" in sys.argv else apply(src)
    if out != src:
        open(F, "w", encoding="utf-8").write(out)
        print("written", F)
    import py_compile
    py_compile.compile(F, doraise=True)
    print("py_compile OK")


if __name__ == "__main__":
    main()
