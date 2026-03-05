#!/usr/bin/env python3
"""
Save HuggingFace datasets to project data directory (not default cache).

Run this in a **separate terminal** so it does not disturb any running
training/build process. Datasets are written to data/hf_cache/ so they
persist in the project and are reused by training scripts.

Usage:
  cd /teamspace/studios/this_studio/Cross-Doc-Jepa
  python scripts/save_datasets_to_disk.py
"""
import os
import sys

# Project root and cache dir (must match data/dataset.py and data/build_wiki_clusters.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "hf_cache")

# Force HuggingFace to use project dir (for any subprocess or library default)
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = HF_CACHE_DIR

def main():
    from datasets import load_dataset
    print(f"Saving datasets to: {HF_CACHE_DIR}")
    print("(This does not disturb any running build/pretrain/finetune process.)\n")

    # Multi-News (used for finetuning and leakage filter)
    print("Loading Multi-News...")
    load_dataset("multi_news", cache_dir=HF_CACHE_DIR)
    print("  Multi-News saved.\n")

    # Wikipedia 20220301.en (used for pretraining clusters)
    print("Loading Wikipedia 20220301.en (this is large, ~20GB)...")
    load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True, cache_dir=HF_CACHE_DIR)
    print("  Wikipedia saved.\n")

    print("Done. Datasets are in:", HF_CACHE_DIR)
    return 0

if __name__ == "__main__":
    sys.exit(main())
