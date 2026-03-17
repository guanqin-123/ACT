#!/usr/bin/env bash
# ===========================================================================
# Run ALL Experiments (RQ1 + RQ2 + RQ3) Overnight
#
# RQ1: 5 benchmarks × 4 methods × 3 runs = 60 configs (~5.5h)
# RQ2: Uses RQ1 data (figures + unique violation analysis)
# RQ3: 2 benchmarks × 2 methods × 6 scales × 3 runs = 72 configs (~6.6h)
# Total: ~12h
#
# Usage:
#   bash act/pipeline/script/run_all_experiments.sh
#   nohup bash act/pipeline/script/run_all_experiments.sh > experiments/overnight.log 2>&1 &
#
# Copyright (C) 2025 SVF-tools/ACT
# ===========================================================================

set -uo pipefail

PYTHON=/home/guanqinzhang/guanqin/miniconda3/envs/act-py312/bin/python
TIMEOUT=60
RUNS=3
DEVICE=cuda

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SD="$SCRIPT_DIR"

cd "$PROJECT_ROOT"

RQ1_DIR="$PROJECT_ROOT/experiments/rq1"
RQ3_DIR="$PROJECT_ROOT/experiments/rq3"
FIGURES_DIR="$PROJECT_ROOT/figures"

mkdir -p "$RQ1_DIR" "$RQ3_DIR" "$FIGURES_DIR"

START_TIME=$(date +%s)

echo "=================================================================="
echo "  ACT Overnight Experiment Suite"
echo "  Started: $(date)"
echo "  Device: $DEVICE | Timeout: ${TIMEOUT}s | Runs: $RUNS"
echo "  RQ1: 60 configs | RQ3: 72 configs | Total: ~12h"
echo "=================================================================="
echo ""

# ---------------------------------------------------------------------------
# PHASE 1: RQ1 — 5 benchmarks × 4 methods × 3 runs = 60 configs
# ---------------------------------------------------------------------------
echo "### PHASE 1/3: RQ1 ($(date +%H:%M:%S)) ###"

RQ1_OK=0; RQ1_SKIP=0; RQ1_FAIL=0

for bm in acasxu cifar100 tinyimagenet mnist trafficsigns; do
  for method in batch_aniso batch_iso seq_aniso seq_rand; do
    for ((run_id=0; run_id<RUNS; run_id++)); do
      RESULT="$RQ1_DIR/$bm/$method/run_${run_id}.json"
      if [[ -f "$RESULT" ]]; then
        echo "SKIP  RQ1  $bm/$method/run_$run_id"
        RQ1_SKIP=$((RQ1_SKIP + 1))
        continue
      fi
      echo "$(date +%H:%M:%S)  RUN  RQ1  $bm/$method/run_$run_id"
      if $PYTHON "$SD/rq1_runner.py" \
          --benchmark "$bm" --method "$method" \
          --runs 1 --start-run "$run_id" \
          --timeout "$TIMEOUT" --device "$DEVICE" \
          --output-dir "$RQ1_DIR" 2>&1; then
        RQ1_OK=$((RQ1_OK + 1))
      else
        RQ1_FAIL=$((RQ1_FAIL + 1))
        echo "  FAILED: RQ1 $bm/$method/run_$run_id"
      fi
    done
  done
done

echo ""
echo "RQ1 done: ok=$RQ1_OK skip=$RQ1_SKIP fail=$RQ1_FAIL ($(date +%H:%M:%S))"
echo ""

# ---------------------------------------------------------------------------
# PHASE 2: RQ2 — Analysis from RQ1 data
# ---------------------------------------------------------------------------
echo "### PHASE 2/3: RQ2 Analysis ($(date +%H:%M:%S)) ###"

echo "  Generating Table 1 (RQ1 aggregator)..."
$PYTHON "$SD/results_aggregator.py" "$RQ1_DIR" 2>&1 || echo "  WARN: results_aggregator failed"

echo "  Generating Figure 2 (single run)..."
$PYTHON "$SD/rq2_figures.py" "$RQ1_DIR" \
  --output "$FIGURES_DIR/viol_time" 2>&1 || echo "  WARN: rq2_figures single failed"

echo "  Generating Figure 2 (mean+std)..."
$PYTHON "$SD/rq2_figures.py" "$RQ1_DIR" \
  --mode mean --output "$FIGURES_DIR/viol_time_mean" 2>&1 || echo "  WARN: rq2_figures mean failed"

echo "  Generating Table 2 (unique violations)..."
$PYTHON "$SD/rq2_unique_violations.py" "$RQ1_DIR" \
  --method-a batch_aniso --method-b batch_iso \
  --output "$RQ1_DIR" 2>&1 || echo "  WARN: rq2_unique_violations failed"

echo ""
echo "RQ2 done ($(date +%H:%M:%S))"
echo ""

# ---------------------------------------------------------------------------
# PHASE 3: RQ3 — 2 benchmarks × 2 methods × 6 scales × 3 runs = 72 configs
# ---------------------------------------------------------------------------
echo "### PHASE 3/3: RQ3 ($(date +%H:%M:%S)) ###"

RQ3_OK=0; RQ3_SKIP=0; RQ3_FAIL=0

for bm in cifar100 tinyimagenet; do
  for scale in 0.01 0.05 0.1 0.2 0.3 0.5; do
    for method in batch_aniso batch_iso; do
      for ((run_id=0; run_id<RUNS; run_id++)); do
        SDIR=$(printf "%.2f" "$scale")
        RESULT="$RQ3_DIR/$SDIR/$bm/$method/run_${run_id}.json"
        if [[ -f "$RESULT" ]]; then
          echo "SKIP  RQ3  $bm/$method/s=$scale/run_$run_id"
          RQ3_SKIP=$((RQ3_SKIP + 1))
          continue
        fi
        echo "$(date +%H:%M:%S)  RUN  RQ3  $bm/$method/s=$scale/run_$run_id"
        if $PYTHON "$SD/rq3_runner.py" \
            --benchmark "$bm" --method "$method" --scale "$scale" \
            --runs 1 --start-run "$run_id" \
            --timeout "$TIMEOUT" --device "$DEVICE" \
            --output-dir "$RQ3_DIR" 2>&1; then
          RQ3_OK=$((RQ3_OK + 1))
        else
          RQ3_FAIL=$((RQ3_FAIL + 1))
          echo "  FAILED: RQ3 $bm/$method/s=$scale/run_$run_id"
        fi
      done
    done
  done
done

echo "  Generating Table 3 (RQ3 aggregator)..."
$PYTHON "$SD/rq3_aggregator.py" "$RQ3_DIR" 2>&1 || echo "  WARN: rq3_aggregator failed"

echo ""
echo "RQ3 done: ok=$RQ3_OK skip=$RQ3_SKIP fail=$RQ3_FAIL ($(date +%H:%M:%S))"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
END_TIME=$(date +%s)
TOTAL_MIN=$(( (END_TIME - START_TIME) / 60 ))

echo "=================================================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Finished: $(date)"
echo "  Total time: ${TOTAL_MIN} minutes"
echo "------------------------------------------------------------------"
echo "  RQ1: ok=$RQ1_OK skip=$RQ1_SKIP fail=$RQ1_FAIL"
echo "  RQ3: ok=$RQ3_OK skip=$RQ3_SKIP fail=$RQ3_FAIL"
echo "------------------------------------------------------------------"
echo "  Outputs:"
echo "    Table 1:   $RQ1_DIR/table1_latex.tex"
echo "    Figure 2:  $FIGURES_DIR/viol_time_*.pdf"
echo "    Table 2:   $RQ1_DIR/table2_unique_viol.tex"
echo "    Table 3:   $RQ3_DIR/table3_rq3_scale.tex"
echo "=================================================================="
