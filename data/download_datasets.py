#!/usr/bin/env python3
"""
download_datasets.py
--------------------
Download CIFAR-100, Tiny ImageNet, and MNIST datasets from Hugging Face Hub
into the current working directory.

Datasets:
- cifar100        -> uoft-cs/cifar100
- tiny-imagenet   -> zh-plus/tiny-imagenet
- mnist           -> ylecun/mnist

Usage:
  python download_datasets.py          # download all (skips existing)
  python download_datasets.py --force  # re-download all
  python download_datasets.py --list   # list available datasets
"""

import sys
import subprocess
from pathlib import Path
import argparse
from typing import Dict, Tuple, Optional

# ------------------------------------------
# Auto-install Hugging Face `datasets`
# ------------------------------------------
def ensure_datasets_installed():
    try:
        import datasets  # noqa: F401
    except ImportError:
        print("📦 Installing required package: datasets ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        print("✅ Installed successfully.\n")

ensure_datasets_installed()
from datasets import load_dataset  # noqa: E402

# ------------------------------------------
# Dataset map
# ------------------------------------------
DATASET_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    "cifar100": ("uoft-cs/cifar100", None),
    "tiny-imagenet": ("zh-plus/tiny-imagenet", None),
    "mnist": ("ylecun/mnist", None),
}

# ------------------------------------------
# Helpers
# ------------------------------------------
def print_available():
    print("Available datasets:\n")
    for name, (repo, subset) in DATASET_MAP.items():
        subtxt = f" (subset={subset})" if subset else ""
        print(f"  - {name:13s} -> {repo}{subtxt}")
    print()

def already_downloaded(path: Path) -> bool:
    """Rudimentary check whether dataset already exists."""
    if not path.exists():
        return False
    return (path / "dataset_dict.json").exists() or any(path.glob("**/*.arrow"))

def download_and_save(hf_repo: str, out_dir: Path, subset: Optional[str] = None):
    print(f"⏳ Downloading {hf_repo} ...")
    try:
        ds = load_dataset(hf_repo, subset) if subset else load_dataset(hf_repo)
        out_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(out_dir))
        print(f"✅ Saved {hf_repo} to {out_dir}\n")
    except Exception as e:
        print(f"❌ Failed to download {hf_repo}: {e}\n")

# ------------------------------------------
# Main
# ------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download CIFAR-100, Tiny ImageNet, and MNIST datasets.")
    parser.add_argument("--list", action="store_true", help="List available datasets and exit.")
    parser.add_argument("--force", action="store_true", help="Re-download even if datasets already exist.")
    args = parser.parse_args()

    if args.list:
        print_available()
        return

    base_dir = Path(".")

    for key, (repo, subset) in DATASET_MAP.items():
        target = base_dir / key
        if already_downloaded(target) and not args.force:
            print(f"⏭️  Skipping {key}: already exists at {target} (use --force to re-download).")
            continue
        download_and_save(repo, target, subset=subset)

    print("🎉 All downloads complete.")

if __name__ == "__main__":
    main()
