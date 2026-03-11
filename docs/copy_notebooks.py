#!/usr/bin/env python3
"""Copy example notebooks into docs/examples/ for Sphinx rendering.

This script is idempotent — safe to run multiple times. It copies notebooks
from the project's examples/ directory into docs/examples/ subdirectories
so that myst_nb can render them during the Sphinx build.

Usage:
    python docs/copy_notebooks.py
"""
import shutil
from pathlib import Path

# Resolve paths relative to this script's location
DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent
EXAMPLES_SRC = PROJECT_ROOT / "examples"
EXAMPLES_DST = DOCS_DIR / "examples"

# Mapping: source subdirectory -> destination subdirectory
NOTEBOOK_DIRS = {
    "cross-thread": "checkpoints",
    "functional": "checkpoints",
    "subgraph": "checkpoints",
    "human_in_the_loop": "human_in_the_loop",
    "memory": "memory",
    "middleware": "middleware",
    "react-agent": "react_agent",
}


def copy_notebooks() -> None:
    """Copy all notebooks from examples/ into docs/examples/."""
    copied = 0

    for src_subdir, dst_subdir in NOTEBOOK_DIRS.items():
        src_path = EXAMPLES_SRC / src_subdir
        dst_path = EXAMPLES_DST / dst_subdir
        dst_path.mkdir(parents=True, exist_ok=True)

        if not src_path.exists():
            print(f"  SKIP {src_subdir}/ (not found)")
            continue

        for notebook in sorted(src_path.glob("*.ipynb")):
            dst_file = dst_path / notebook.name
            shutil.copy2(notebook, dst_file)
            copied += 1
            print(f"  {notebook.relative_to(PROJECT_ROOT)} -> {dst_file.relative_to(DOCS_DIR)}")

    print(f"\nCopied {copied} notebooks.")


if __name__ == "__main__":
    print("Copying example notebooks into docs/examples/...\n")
    copy_notebooks()
