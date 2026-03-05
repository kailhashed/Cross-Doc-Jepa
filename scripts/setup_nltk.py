#!/usr/bin/env python3
"""
Pre-download all NLTK assets required by CrossDoc-JEPA.
Run this ONCE on your compute node before training:

  python scripts/setup_nltk.py

In air-gapped HPC environments, run this on a node with internet access,
then copy ~/.nltk_data to your compute nodes or set NLTK_DATA env var to
a shared filesystem path.

  NLTK_DATA=/shared/nltk_data python scripts/setup_nltk.py
  # then on compute nodes:
  export NLTK_DATA=/shared/nltk_data
"""

import os
import sys


def main():
    try:
        import nltk
    except ImportError:
        print("ERROR: nltk is not installed. Run: pip install nltk")
        sys.exit(1)

    # Allow override via env var (for shared HPC filesystems)
    nltk_data_dir = os.environ.get("NLTK_DATA", None)
    if nltk_data_dir:
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.insert(0, nltk_data_dir)
        download_dir = nltk_data_dir
    else:
        download_dir = None  # NLTK default (~/.nltk_data)

    assets = [
        ("tokenizers/punkt",     "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),   # NLTK >= 3.9
    ]

    all_ok = True
    for data_path, package_id in assets:
        try:
            nltk.data.find(data_path)
            print(f"  ✓ {package_id} already present")
        except LookupError:
            print(f"  Downloading {package_id}...")
            success = nltk.download(package_id, download_dir=download_dir, quiet=False)
            if not success:
                print(f"  ✗ Failed to download {package_id}")
                all_ok = False
            else:
                print(f"  ✓ {package_id} downloaded")

    if all_ok:
        print("\nAll NLTK assets ready. You can now run training.")
    else:
        print("\nSome assets failed. Check network access or set NLTK_DATA to a writable path.")
        sys.exit(1)

    # Verify the tokenizer actually works
    try:
        from nltk.tokenize import sent_tokenize
        test = sent_tokenize("Dr. Smith went to Washington. He arrived on Jan. 1st.")
        assert len(test) == 2, f"Expected 2 sentences, got {len(test)}: {test}"
        print(f"  ✓ Sentence tokenizer verified: {test}")
    except Exception as e:
        print(f"  ✗ Tokenizer verification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
