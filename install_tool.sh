#!/bin/bash
# VNN-COMP 2026 install_tool.sh for ACT.
# Arg: $1 = version string "v1". Runs once on the AWS instance; creates the
# act-py312 conda environment (installing Miniconda first if absent).
set -e

VERSION_STRING="v1"
if [ "$1" != "$VERSION_STRING" ]; then
    echo "install_tool.sh: expected first argument '$VERSION_STRING', got '$1'"
    exit 1
fi

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# machine info (licensing / debugging), mirrors the CORA example toolkit
ip link show || true
echo "user: $USER"
nvidia-smi || true

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found; installing Miniconda..."
    MC=/tmp/miniconda.sh
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$MC"
    bash "$MC" -b -p "$HOME/miniconda3"
    export PATH="$HOME/miniconda3/bin:$PATH"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -qE '/act-py312$'; then
    echo "creating act-py312 environment from environment.yml..."
    conda env create -f "$REPO_DIR/environment.yml"
fi

conda run -n act-py312 python -c "import torch, act; print('ACT import OK; torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "install_tool.sh: done"
