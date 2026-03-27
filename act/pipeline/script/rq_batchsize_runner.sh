#!/usr/bin/env bash
# ===========================================================================
# Batch-Size Experiment Runner
#
# Runs 50 batch-size experiments for TinyImageNet benchmark:
# 5 B values × 2 methods × 5 runs = 50 configs
#
# B values: 1, 10, 50, 100, 199
# Methods: batch_aniso, batch_iso
# Runs: 5 per configuration
#
# Usage:
#   bash act/pipeline/script/rq_batchsize_runner.sh [--dry-run] [--device cuda|cpu]
#
# Examples:
#   bash act/pipeline/script/rq_batchsize_runner.sh --dry-run
#   bash act/pipeline/script/rq_batchsize_runner.sh --device cpu
#   bash act/pipeline/script/rq_batchsize_runner.sh
#
# ===========================================================================

set -uo pipefail

DEVICE="cuda"
DRY_RUN=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BASE_DIR="$PROJECT_ROOT/experiments/rq_batchsize"

PYTHON="python"

# Per-benchmark batch sizes
declare -A BENCHMARK_B_VALUES
BENCHMARK_B_VALUES[tinyimagenet]="1 10 50 100 199"
BENCHMARK_B_VALUES[cifar100]="1 10 50 99"

METHODS=("batch_aniso" "batch_iso")
RUNS=5

# Statistics
OK=0
SKIP=0
FAIL=0
START_TIME=$(date +%s)

echo "=================================================================="
echo "  Batch-Size Experiment Runner"
echo "  Started: $(date)"
echo "  Device: $DEVICE | Dry-run: $DRY_RUN"
echo "  Benchmarks: tinyimagenet cifar100"
echo "  Methods: ${METHODS[*]}"
echo "  Runs per config: $RUNS"
echo "  Total experiments: 90 (50 TinyImageNet + 40 CIFAR-100)"
echo "=================================================================="
echo ""

# Run experiments
for benchmark in tinyimagenet cifar100; do
  IFS=' ' read -ra B_VALS <<< "${BENCHMARK_B_VALUES[$benchmark]}"
  
  for B in "${B_VALS[@]}"; do
    B_DIR=$(printf "B_%03d" "$B")
    
    for method in "${METHODS[@]}"; do
      for ((run_id=0; run_id<RUNS; run_id++)); do
        RESULT="$BASE_DIR/$B_DIR/$benchmark/$method/run_${run_id}.json"
        
        # Check if result already exists (resume logic)
        if [[ -f "$RESULT" ]]; then
          echo "SKIP  B=$B  $benchmark/$method/run_$run_id"
          SKIP=$((SKIP+1))
          continue
        fi
        
        # Build command
        CMD="$PYTHON -m act.pipeline.script.rq1_runner \
          --benchmark $benchmark \
          --method $method \
          --max-instances $B \
          --runs 1 --start-run $run_id \
          --timeout 60 --device $DEVICE \
          --output-dir $BASE_DIR/$B_DIR"
        
        if $DRY_RUN; then
          # Dry-run: just print the command
          echo "RUN  B=$B  $benchmark/$method/run_$run_id  |  $CMD"
          OK=$((OK+1))
        else
          # Actual run
          echo "$(date +%H:%M:%S)  RUN  B=$B  $benchmark/$method/run_$run_id"
          if eval "$CMD" 2>&1; then
            OK=$((OK+1))
          else
            FAIL=$((FAIL+1))
            echo "  FAILED: B=$B $benchmark/$method/run_$run_id"
          fi
        fi
      done
    done
  done
done

END_TIME=$(date +%s)
ELAPSED_MIN=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "=================================================================="
echo "  Batch-Size Experiments Complete"
echo "  Finished: $(date)"
echo "  Elapsed time: ${ELAPSED_MIN} minutes"
echo "------------------------------------------------------------------"
echo "  Results: ok=$OK skip=$SKIP fail=$FAIL"
echo "  Output directory: $BASE_DIR"
echo "=================================================================="

if [[ "$DRY_RUN" == "false" && $OK -gt 0 ]]; then
    echo ""
    echo "Running batch-size analysis..."
    python "$SCRIPT_DIR/rq4_batchsize_analysis.py" "$BASE_DIR" --merged || true
fi
