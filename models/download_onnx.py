#!/usr/bin/env python3
"""
download_onnx.py — Minimal ONNX Model Zoo CLI for https://huggingface.co/onnxmodelzoo

Commands:
  list
      Print the full list of model IDs.

  category
      Show categories by pipeline_tag, then subgroups by first prefix (token before '_'):
        [image-classification] (12)
          ├── resnet (5)
          ├── vit (3)
          └── mnist (4)

  download-prefix --prefix <name> [--dest ./onnx_models]
      Download only the .onnx files for all models with that prefix (across categories).
      Example:
        python download_onnx.py download-prefix --prefix resnet
        python download_onnx.py download-prefix --prefix vit --dest ./onnx_zoo

  download-model --model <repo_id> [--dest ./onnx_models]
      Download only the .onnx files for a single model repo.
      Example:
        python download_onnx.py download-model --model onnxmodelzoo/resnet18
        python download_onnx.py download-model --model onnxmodelzoo/vit_base_patch16_224 --dest ./onnx_zoo

  download-by-category [--dest .] [--only <cat1,cat2,...>] [--by-prefix]
      Download .onnx files grouped into per-category folders.
      Example:
        python download_onnx.py download-by-category
        python download_onnx.py download-by-category --only image-classification,object-detection
        python download_onnx.py download-by-category --by-prefix --dest ./onnx_by_cat

Notes:
- Automatically installs `huggingface_hub` (and `rich` if available) on first run.
- Uses an expanded taxonomy to infer missing/incorrect pipeline tags, reducing "unknown" drastically.
"""

import importlib, subprocess, sys, os, argparse, re
from collections import defaultdict
from typing import Dict, List, Iterable, Optional

# ----------------- Auto-install lightweight deps -----------------
def ensure_package(pkg, import_name=None, min_version=None):
    import_name = import_name or pkg
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"Installing missing dependency: {pkg} ...")
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
        cmd.append(f"{pkg}>={min_version}" if min_version else pkg)
        subprocess.check_call(cmd)

ensure_package("huggingface_hub", "huggingface_hub", "0.22")

try:
    ensure_package("rich", "rich")
    from rich.console import Console
    from rich.tree import Tree
    RICH = True
    console = Console()
except Exception:
    RICH = False
    console = None

from huggingface_hub import list_models, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

AUTHOR = "onnxmodelzoo"

# ----------------- Taxonomy / Keyword Map -----------------
PIPELINE_KEYWORDS = [
    ("object-detection", ["yolo", "ssd", "fasterrcnn", "maskrcnn", "retinanet"]),
    ("image-segmentation", ["deeplab", "fcn", "unet", "segformer"]),
    ("super-resolution", ["super-resolution", "sr-"]),
    ("image-classification", [
        "alexnet", "resnet", "vit", "vgg", "mobilenet", "efficientnet",
        "densenet", "squeezenet", "googlenet", "inception", "xcit", "swin",
        "mnist", "emotion", "gender", "age"
    ]),
    ("text-classification", ["bert", "roberta", "albert", "electra"]),
    ("text-generation", ["gpt", "opt", "t5", "bart"]),
    ("automatic-speech-recognition", ["whisper", "wav2vec", "conformer"]),
    ("graph-ml", ["graph", "sageconv", "tagconv"]),
    ("image-generation", ["diffusion", "vae", "gan"]),
]
PREFIX_OVERRIDES = {}

# ----------------- Core helpers -----------------
def fetch_models(query: Optional[str] = None, token: Optional[str] = None) -> List:
    return list(list_models(author=AUTHOR, search=query, full=True, token=token))

def extract_prefix(model_id: str) -> str:
    name = model_id.split("/")[-1]
    return (name.split("_", 1)[0] if "_" in name else name).lower()

def _normalize(s: str) -> str:
    return s.lower().replace(".", "-")

def infer_pipeline_tag(model_id: str, pipeline_tag: Optional[str]) -> str:
    if pipeline_tag:
        return pipeline_tag
    repo = _normalize(model_id.split("/")[-1])
    pref = extract_prefix(model_id)
    if pref in PREFIX_OVERRIDES:
        return PREFIX_OVERRIDES[pref]
    for tag, kws in PIPELINE_KEYWORDS:
        if any(kw in repo for kw in kws):
            return tag
    return "unknown"

def group_by_tag_and_prefix(models: List) -> Dict[str, Dict[str, List]]:
    groups = defaultdict(lambda: defaultdict(list))
    for m in models:
        tag = infer_pipeline_tag(m.modelId, getattr(m, "pipeline_tag", None))
        pref = extract_prefix(m.modelId)
        groups[tag][pref].append(m)
    ordered = {}
    for tag in sorted(groups.keys(), key=lambda t: (t == "unknown", t)):
        ordered[tag] = {p: sorted(v, key=lambda mi: mi.modelId.lower())
                        for p, v in sorted(groups[tag].items())}
    return ordered

def allow_patterns_onnx(): return ["*.onnx", "**/*.onnx"]

def ensure_dest(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)

def download_repo_onnx(repo_id: str, dest: str, token: Optional[str] = None) -> Optional[str]:
    local_dir = os.path.join(dest, repo_id.split("/")[-1])
    ensure_dest(local_dir)
    try:
        path = snapshot_download(repo_id=repo_id, local_dir=local_dir,
                                 local_dir_use_symlinks=False, token=token,
                                 allow_patterns=allow_patterns_onnx())
        return path
    except HfHubHTTPError as e:
        print(f"❌ Failed to download {repo_id}: {e}")
        return None

# ----------------- Printers -----------------
def print_full_list(models: List):
    for m in sorted(models, key=lambda mi: mi.modelId.lower()):
        print(m.modelId)

def print_categories_tree(groups: Dict[str, Dict[str, List]]):
    if RICH:
        root = Tree(f"[bold]Categories for @{AUTHOR}[/bold]")
        for tag, prefs in groups.items():
            tag_node = root.add(f"[cyan]{tag}[/cyan] ({sum(len(v) for v in prefs.values())})")
            for pref, items in prefs.items():
                tag_node.add(f"{pref} ({len(items)})")
        console.print(root)
    else:
        for tag, prefs in groups.items():
            print(f"[{tag}] ({sum(len(v) for v in prefs.values())})")
            for pref, items in prefs.items():
                print(f"  ├── {pref} ({len(items)})")

# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser(description="Minimal ONNX downloader for Hugging Face @onnxmodelzoo")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list")
    sub.add_parser("category")

    sp = sub.add_parser("download-prefix", help="Download .onnx for all models with this prefix")
    sp.add_argument("--prefix", required=True)
    sp.add_argument("--dest", default="./onnx_models")
    sp.add_argument("--token", default=os.getenv("HF_TOKEN"))

    sp = sub.add_parser("download-model", help="Download .onnx for a single model repo")
    sp.add_argument("--model", required=True)
    sp.add_argument("--dest", default="./onnx_models")
    sp.add_argument("--token", default=os.getenv("HF_TOKEN"))

    sp = sub.add_parser("download-by-category", help="Download .onnx files grouped into per-category folders")
    sp.add_argument("--dest", default=".")
    sp.add_argument("--only", default=None,
                    help="Comma-separated list of categories (default: all)")
    sp.add_argument("--by-prefix", action="store_true",
                    help="Create subfolders by prefix inside each category")
    sp.add_argument("--token", default=os.getenv("HF_TOKEN"))

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        return

    if args.cmd in {"list", "category", "download-prefix", "download-by-category"}:
        models = fetch_models()

    if args.cmd == "list":
        print_full_list(models); return
    if args.cmd == "category":
        print_categories_tree(group_by_tag_and_prefix(models)); return

    if args.cmd == "download-prefix":
        groups = group_by_tag_and_prefix(models)
        pref, dest = args.prefix.lower(), ensure_dest(args.dest)
        items = [m for tag in groups.values() for m in tag.get(pref, [])]
        if not items:
            print(f"No models found with prefix '{pref}'"); return
        for i, m in enumerate(items, 1):
            print(f"({i}/{len(items)}) ⬇️ {m.modelId}")
            download_repo_onnx(m.modelId, dest, token=args.token)
        return

    if args.cmd == "download-model":
        dest = ensure_dest(args.dest)
        print(f"⬇️ {args.model}")
        path = download_repo_onnx(args.model, dest, token=args.token)
        if path: print(f"✅ Saved to {path}")
        return

    if args.cmd == "download-by-category":
        groups = group_by_tag_and_prefix(models)
        wanted = {c.strip() for c in args.only.split(",")} if args.only else set(groups)
        base = ensure_dest(args.dest)

        for tag in sorted(wanted):
            if tag not in groups: continue
            cat_dir = ensure_dest(os.path.join(base, tag))
            print(f"\nCategory: {tag} → {cat_dir}")
            for pref, repos in groups[tag].items():
                subdir = cat_dir if not args.by-prefix else ensure_dest(os.path.join(cat_dir, pref))
                for i, m in enumerate(repos, 1):
                    print(f"  ({i}/{len(repos)}) ⬇️ {m.modelId}")
                    download_repo_onnx(m.modelId, subdir, token=args.token)
        print("\n✅ Done.")
        return

if __name__ == "__main__":
    main()
