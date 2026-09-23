#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply-patches.py —— 给 vLLM 定制镜像打本仓库的全部补丁（幂等 / 可校验 / 可回滚）。

目标树（TARGET）是镜像解出的 rootfs 里的 vllm 包目录，例如：
    /media/ll/data/vllm-image/rootfs/usr/local/lib/python3.12/dist-packages/vllm

三种模式：
    --check   只体检：逐文件判定 pristine / patched / 其它（默认）
    --apply   打补丁：pristine 文件打补丁；已是 patched 的跳过；未知状态报错
    --revert  回滚：把 patched 文件还原成 pristine（用打补丁前的 sha256 校验）

补丁按 patches/MANIFEST.tsv 的顺序应用。每条补丁是 `diff -u a/<rel> b/<rel>` 格式，
用系统 `patch` 命令应用，应用后立即用 sha256 与 MANIFEST 里的 patched_sha256 比对——
**哈希不一致即视为失败并自动回滚该文件**，杜绝"打了但没打上/打错版本"。

改过 .py 必须删对应 __pycache__（本脚本自动处理，见坑 #P11）。

用法：
    python3 apply-patches.py --target <vllm 目录> [--apply|--revert|--check] [--root <repo>]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys

REL_ROOT_MARK = "vllm"  # TARGET 本身就是 vllm 目录


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_manifest(root: str):
    rows = []
    p = os.path.join(root, "patches", "MANIFEST.tsv")
    with open(p, encoding="utf-8") as f:
        head = f.readline()
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            rows.append(dict(zip(head.rstrip("\n").split("\t"), parts)))
    return rows


def pyclean(target: str, rel: str) -> None:
    """删除受影响文件所在目录的 __pycache__ 条目（改 .py 不删缓存 = 改动可能不生效）。"""
    d = os.path.join(target, os.path.dirname(rel))
    cache = os.path.join(d, "__pycache__")
    if not os.path.isdir(cache):
        return
    stem = os.path.basename(rel).replace(".py", "")
    for fn in os.listdir(cache):
        if fn.startswith(stem + ".") and fn.endswith(".pyc"):
            try:
                os.unlink(os.path.join(cache, fn))
            except OSError:
                pass


def run_patch(target: str, patchfile: str, reverse: bool = False) -> tuple[int, str]:
    cmd = ["patch", "-p1", "--silent", "--no-backup-if-mismatch"]
    if reverse:
        cmd.append("-R")
    cmd += ["-i", patchfile]
    r = subprocess.run(cmd, cwd=target, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="镜像内 vllm 包目录的绝对路径")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="仓库根（含 patches/），默认取脚本所在目录的上一级")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="只体检（默认）")
    g.add_argument("--apply", action="store_true", help="打补丁")
    g.add_argument("--revert", action="store_true", help="回滚到上游原版")
    ap.add_argument("--only", default="", help="只处理 id 列表，逗号分隔，如 06,19")
    args = ap.parse_args()

    target = os.path.abspath(args.target)
    if not os.path.isdir(target):
        print(f"[FATAL] 目标目录不存在：{target}", file=sys.stderr)
        return 2
    if not os.access(target, os.W_OK) and (args.apply or args.revert):
        print(f"[FATAL] 目标目录不可写（chroot 内的树属 root，请用 sudo 运行）：{target}", file=sys.stderr)
        return 2

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    rows = [r for r in load_manifest(args.root) if not only or r["id"] in only]
    state = {"pristine": 0, "patched": 0, "other": 0}
    failed: list[str] = []

    for r in rows:
        rel, pid = r["path"], r["id"]
        cur = os.path.join(target, rel)
        pf = os.path.join(args.root, "patches", r["patch_file"])
        if not os.path.exists(cur):
            print(f"  [{pid}] ✗ 缺文件 {rel}")
            failed.append(rel)
            continue
        h = sha256(cur)
        if h == r["patched_sha256"]:
            state["patched"] += 1
            tag = "patched "
        elif h == r["pristine_sha256"]:
            tag = "pristine "
            state["pristine"] += 1
        else:
            tag = "UNKNOWN "
            state["other"] += 1

        if args.apply and tag.startswith("pristine"):
            rc, out = run_patch(target, pf)
            if rc != 0:
                print(f"  [{pid}] ✗ patch 失败 rc={rc} {rel}\n{out}")
                failed.append(rel)
                continue
            h2 = sha256(cur)
            if h2 != r["patched_sha256"]:
                print(f"  [{pid}] ✗ 应用后哈希不符（镜像版本不对？）{rel}\n"
                      f"        期望 {r['patched_sha256']}\n        实得 {h2}")
                run_patch(target, pf, reverse=True)
                failed.append(rel)
                continue
            pyclean(target, rel)
            tag = "APPLIED   "
        elif args.revert and tag.startswith("patched"):
            rc, out = run_patch(target, pf, reverse=True)
            if rc != 0 or sha256(cur) != r["pristine_sha256"]:
                print(f"  [{pid}] ✗ 回滚失败 {rel} rc={rc}\n{out}")
                failed.append(rel)
                continue
            pyclean(target, rel)
            tag = "REVERTED  "

        if args.check or tag not in ("patched ",):
            print(f"  [{pid}] {tag}{rel}")

    mode = "apply" if args.apply else ("revert" if args.revert else "check")
    print(f"\n[{mode}] 共 {len(rows)} 个补丁：pristine={state['pristine']} "
          f"patched={state['patched']} unknown={state['other']} 失败={len(failed)}")
    if failed:
        print("[FATAL] 以下文件未达期望状态：\n  " + "\n  ".join(failed), file=sys.stderr)
        return 1
    if args.apply and state["patched"] + state["pristine"] != len(rows):
        return 1
    print("[OK] 补丁状态一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
