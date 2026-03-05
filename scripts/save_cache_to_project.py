#!/usr/bin/env python3
"""
Copy the current HuggingFace datasets cache into the project directory.

The **running** build/pretrain/finetune process is using the default cache
(e.g. ~/.cache/huggingface/datasets). This script runs in a **separate terminal**
and copies that cache into the project (data/hf_cache) so the files are saved
there. It does NOT exit or disturb the running process — it only reads from
the cache and writes to the project.

Run in a separate terminal:
  cd /teamspace/studios/this_studio/Cross-Doc-Jepa
  python scripts/save_cache_to_project.py
"""
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEST_DIR = os.path.join(PROJECT_ROOT, "data", "hf_cache")


def get_default_cache_dir():
    """Where the running process is likely writing (same logic as 'datasets' lib)."""
    if os.environ.get("HF_DATASETS_CACHE"):
        return os.environ["HF_DATASETS_CACHE"]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "datasets")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")


def main():
    src = get_default_cache_dir()
    if not os.path.isdir(src):
        print(f"Source cache not found (nothing to copy yet): {src}")
        print("The running process may still be downloading. Run this script again later.")
        return 1

    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"Copying cache from: {src}")
    print(f"             to:   {DEST_DIR}")
    print("(Running process is unchanged; only copying files.)\n")

    try:
        for name in os.listdir(src):
            src_path = os.path.join(src, name)
            dest_path = os.path.join(DEST_DIR, name)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dest_path)
        print("Done. Cache copied to project.")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
