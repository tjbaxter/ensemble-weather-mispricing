#!/usr/bin/env python3
"""Sync repo code into a VM workdir without touching runtime state.

Managed code is copied from a staged repo snapshot into the remote workdir.
Runtime-managed state is preserved in place:

- `logs/`
- `venv/` and `.venv/`
- top-level `.env`
- all non-Python files under `data/`

This lets setup/redeploy refresh code safely without wiping paper/live datasets.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PRESERVE_TOP_LEVEL = {".env", ".venv", "venv", "logs"}


def remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.copy2(src, dst)


def sync_tree(src: Path, dst: Path) -> None:
    if src.is_symlink() or src.is_file():
        copy_file(src, dst)
        return

    if dst.exists() and not dst.is_dir():
        remove_path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    src_names = {child.name for child in src.iterdir()}
    for child in list(dst.iterdir()):
        if child.name not in src_names:
            remove_path(child)

    for child in src.iterdir():
        sync_tree(child, dst / child.name)


def prune_empty_dirs(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def sync_data_python_only(src_data: Path, dst_data: Path) -> None:
    dst_data.mkdir(parents=True, exist_ok=True)

    src_py = {path.relative_to(src_data) for path in src_data.rglob("*.py")}
    dst_py = {path.relative_to(dst_data) for path in dst_data.rglob("*.py")} if dst_data.exists() else set()

    for rel_path in sorted(dst_py - src_py):
        remove_path(dst_data / rel_path)

    for rel_path in sorted(src_py):
        copy_file(src_data / rel_path, dst_data / rel_path)

    prune_empty_dirs(dst_data)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: safe_remote_sync.py <staging_dir> <workdir>", file=sys.stderr)
        return 2

    staging = Path(argv[1]).resolve()
    workdir = Path(argv[2]).resolve()
    if not staging.is_dir():
        raise SystemExit(f"staging dir does not exist: {staging}")

    workdir.mkdir(parents=True, exist_ok=True)

    src_names = {child.name for child in staging.iterdir()}

    for src_item in staging.iterdir():
        if src_item.name == "data":
            sync_data_python_only(src_item, workdir / "data")
            continue
        sync_tree(src_item, workdir / src_item.name)

    for dst_item in list(workdir.iterdir()):
        if dst_item.name in PRESERVE_TOP_LEVEL:
            continue
        if dst_item.name == "data":
            continue
        if dst_item.name not in src_names:
            remove_path(dst_item)

    print(
        "safe_remote_sync complete:",
        "preserved=logs,venv,.venv,.env,data(non-py)",
        f"source={staging}",
        f"dest={workdir}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
