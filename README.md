# Supplementary Material for CLIMB: Core Lifting for Tensor Branch-and-Bound Neural Network Verification

Anonymous artifact accompanying the paper. CLIMB is implemented on top of the ACT neural-network verification framework (Python package `act/`).

## Contents

| Path | Description |
|---|---|
| `act/` | Source code: the ACT framework including CLIMB (CLIMB core in `act/back_end/bab/`, presets in `act/config/backend.yaml`) |
| `FSE27_paper_data/` | All result data reported in the paper and the scripts that regenerate its tables and figures (`make_rq*.py`) |
| `environment.yml` | Conda environment specification |

## Setup

```bash
conda env create -f environment.yml
conda activate act-py312
python -m nltk.downloader -d "$CONDA_PREFIX/share/nltk_data" punkt punkt_tab
chmod -R go-w "$CONDA_PREFIX/share/nltk_data"
```

The NLTK tokenizer data is used by the SST/Yelp loaders; NLTK 3.10 rejects group- or other-writable data directories, hence the `chmod`.

Gurobi is optional. If you have a license, place it at `modules/gurobi/gurobi.lic`.

## Benchmarks

The evaluation uses three benchmark suites:

1. **VNN-COMP 2026 benchmarks** (repository `github.com/VNN-COMP/vnncomp2026_benchmarks`, pinned commit `303f2327`):
   - `cifar100_2024`: ResNet image classification models on CIFAR-100.
   - `tinyimagenet_2024`: ResNet image classification models on TinyImageNet.
   - `cora_2024`: Fully connected ReLU networks for image classification.
   - `safenlp_2024`: Sentence embedding classifiers.

2. **VNN-COMP 2025 benchmarks** (repository `github.com/VNN-COMP/vnncomp2025_benchmarks`, pinned commit `8b7b811`):
   - `sat_relu_2025`: Hard satisfiability problems encoded as ReLU neural networks.

3. **Transformer sentiment classification models**:
   - SST and Yelp binary sentiment classification benchmarks under L1 (`p1`) and L-infinity (`pinf`) perturbations.
   - The compact BERT architectures and checkpoints originate from prior work on Transformer verification: Shi et al. (*Robustness Verification for Transformers*, ICLR 2020) and the BUFFET framework (*Precise Verification of Transformers through ReLU-Catalyzed Abstraction Refinement*, CAV 2026). 

### Obtaining Data and Expected Layout

To set up the VNN-COMP benchmarks, clone or sparse-checkout each repository at its pinned commit and uncompress any `.gz` archives:

```bash
# VNN-COMP 2026
git clone https://github.com/VNN-COMP/vnncomp2026_benchmarks.git data/vnncomp2026_repo
cd data/vnncomp2026_repo && git checkout 303f2327 && find . -name "*.gz" -exec gunzip {} + && cd ../..

# VNN-COMP 2025
git clone https://github.com/VNN-COMP/vnncomp2025_benchmarks.git data/vnncomp2025_repo
cd data/vnncomp2025_repo && git checkout 8b7b811 && find . -name "*.gz" -exec gunzip {} + && cd ../..
```

Place or symlink the benchmark files under `data/` according to the following layout:

```text
data/
├── vnncomp2026/
│   ├── cifar100_2024/2.0/
│   │   ├── onnx/
│   │   └── vnnlib/
│   ├── tinyimagenet_2024/2.0/
│   │   ├── onnx/
│   │   └── vnnlib/
│   ├── cora_2024/2.0/
│   │   ├── onnx/
│   │   └── vnnlib/
│   └── safenlp_2024/2.0/
│       ├── onnx/
│       └── vnnlib/
├── vnncomp2025/
│   └── sat_relu_2025/
│       ├── onnx/
│       └── vnnlib/
├── sst/
│   ├── test.txt
│   └── train-nodes.tsv
├── yelp/
│   └── test.csv
└── buffet/
    └── models/
        ├── model_sst_1/ckpt-1/{config.json,pytorch_model.bin,vocab.txt}
        ├── model_sst_2/ckpt-1/...
        ├── model_sst_3/ckpt-1/...
        ├── model_sst_6/ckpt-1/...
        └── model_yelp_*/ckpt-1/...
```

## Running CLIMB

The verification back-end executes via `python -m act.back_end`. Ensure `PYTHONPATH=.` is set when running from the repository root.

A generic verification command template:

```bash
python -m act.back_end --verify \
  --onnx data/vnncomp2026/<dataset>/2.0/onnx/<model>.onnx \
  --vnnlib data/vnncomp2026/<dataset>/2.0/vnnlib/<property>.vnnlib \
  --solver dual --bab \
  --bab-preset <preset> \
  --bab-branching fsb \
  --bab-max-batch-size <batch_size> \
  --device cuda --dtype float32 \
  --timeout <seconds>
```

A concrete example verifying a TinyImageNet instance with CLIMB:

```bash
python -m act.back_end --verify \
  --onnx data/vnncomp2026/tinyimagenet_2024/2.0/onnx/TinyImageNet_resnet_medium.onnx \
  --vnnlib data/vnncomp2026/tinyimagenet_2024/2.0/vnnlib/TinyImageNet_resnet_medium_prop_idx_9796_sidx_1847_eps_0.0039.vnnlib \
  --solver dual --bab \
  --bab-preset climb_split_lin \
  --bab-branching fsb \
  --bab-max-batch-size 256 \
  --device cuda --dtype float32 \
  --timeout 100.0
```

For SST and Yelp queries, pass the query index CSV and query row identifier:

```bash
python -m act.back_end --verify \
  --query-index queries.csv --query-id 0 \
  --solver dual --bab \
  --bab-preset climb_split_lin \
  --device cuda --dtype float32 \
  --timeout 30.0
```

Key configuration presets in `act/config/backend.yaml` include:
- `climb_split_lin`: CLIMB with split-aware linear interval refresh (primary configuration used in the paper).
- `climb_split`: CLIMB with split-derived interval refresh.
- `climb_refined`: CLIMB with root-level pre-activation bound refinement.
- `climb`: Base CLIMB configuration with exact zero-cost coarsening.

GPU acceleration (CUDA) is strongly recommended for batched dual tensor bounding. Gurobi is optional and used for exact LP/MILP solving when configured.

## Reproducing the paper's tables and figures

All tables and figures from the paper can be regenerated from the bundled result data using the scripts in `FSE27_paper_data/`.

### Requirements and Execution

- **Working Directory**: Run all scripts from inside the `FSE27_paper_data/` directory, or pass `--data <path>` pointing to it.
- **Python Packages**: Table generation requires only the Python standard library (`csv`, `math`, `statistics`, `argparse`, `pathlib`).
- **Plotting Packages**: Figure generation scripts require `matplotlib` and a LaTeX installation containing the `libertine`, `newtx`, and `zi4` font packages.

### Script Summary

| Script | Produces | CLI Options |
|---|---|---|
| `make_rq1_main_table.py` | RQ1 main table (cross-tool comparison on VNN-COMP image and NLP benchmarks) | `--out <file>` (default: stdout) |
| `make_rq1_text_table.py` | RQ1 text-model table (Transformer sentiment models on SST and Yelp) | `--out <file>` (default: stdout) |
| `make_rq2_table.py` | RQ2 table (search reduction comparing CLIMB enabled vs. disabled) | `--out <file>` (default: stdout) |
| `make_rq2_scatter.py` | RQ2 scatter figure (`exp_rq2_scatter_nodes.pdf`, `exp_rq2_scatter_time.pdf`) | `--out-dir <dir>` (default: `.`) |
| `make_rq2_trace.py` | RQ2 trace figure (`exp_trace_nodes.pdf`, `exp_trace_frontier.pdf`, `exp_trace_frontier_gap.pdf`) | `--out-dir <dir>` (default: `.`) |
| `make_rq3_table.py` | RQ3 ablation table (component ablations on coarsening, propagation, merging, and terminal cores) | `--out <file>` (default: stdout) |
| `make_rq4_table.py` | RQ4 batch-size table (batch size sensitivity at B in {64, 256, 1024}) | `--out <file>` (default: stdout) |

Results of the compared third-party verifiers (alpha-beta-CROWN/BICCOS, NeuralSAT, nnenum) are provided as data only; no instructions are included for running these external tools.

## License

AGPL-3.0; see `LICENSE`.
