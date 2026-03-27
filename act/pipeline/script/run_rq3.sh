#!/usr/bin/env bash
# ===========================================================================
# RQ3 Scale Sensitivity Experiment Runner
#
# Usage:
#   bash act/pipeline/script/run_rq3.sh [OPTIONS]
#
# Options:
#   --device DEVICE          cpu or cuda (default: cuda)
#   --timeout SECONDS        Per-run timeout (default: 60)
#   --runs N                 Runs per config (default: 5)
#   --output-dir DIR         Results directory (default: experiments/rq3)
#   --dry-run                Print what would be run without executing
#   --benchmark NAME         Run only this benchmark (default: all)
#   --scale VALUE            Run only this scale factor (default: all)
#   --max-instances N        Limit instances per run (default: all)
#
# ===========================================================================

set -euo pipefail

# Defaults
DEVICE="cuda"
TIMEOUT=60
RUNS=5
OUTPUT_DIR="experiments/rq3"
DRY_RUN=false
FILTER_BENCHMARK=""
FILTER_SCALE=""
MAX_INSTANCES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)           DEVICE="$2"; shift 2 ;;
        --timeout)          TIMEOUT="$2"; shift 2 ;;
        --runs)             RUNS="$2"; shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run)          DRY_RUN=true; shift ;;
        --benchmark)        FILTER_BENCHMARK="$2"; shift 2 ;;
        --scale)            FILTER_SCALE="$2"; shift 2 ;;
        --max-instances)    MAX_INSTANCES="$2"; shift 2 ;;
        *)                  echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Benchmarks and scale factors (must match rq3_config.py)
BENCHMARKS=("cifar100" "tinyimagenet")
METHODS=("batch_aniso" "batch_iso")
SCALES=("0.01" "0.05" "0.1" "0.2" "0.3" "0.5")

# Script path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUNNER="$SCRIPT_DIR/rq3_runner.py"

# Log file
LOG_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/experiment.log"

# Counters
TOTAL=0
SKIPPED=0
EXECUTED=0
FAILED=0

# Scale string formatting (match rq3_runner.py _scale_str)
scale_str() {
    printf "%.2f" "$1"
}

echo "=================================================================="
echo "RQ3 Experiment: Scale Factor Sensitivity"
echo "=================================================================="
echo "Device:     $DEVICE"
echo "Timeout:    ${TIMEOUT}s per run"
echo "Runs:       $RUNS per config"
echo "Output:     $OUTPUT_DIR"
echo "Dry run:    $DRY_RUN"
echo "Log:        $LOG_FILE"
if [[ -n "$FILTER_BENCHMARK" ]]; then echo "Benchmark:  $FILTER_BENCHMARK"; fi
if [[ -n "$FILTER_SCALE" ]]; then echo "Scale:      $FILTER_SCALE"; fi
echo "=================================================================="
echo ""

for benchmark in "${BENCHMARKS[@]}"; do
    if [[ -n "$FILTER_BENCHMARK" && "$benchmark" != "$FILTER_BENCHMARK" ]]; then
        continue
    fi

    for scale in "${SCALES[@]}"; do
        if [[ -n "$FILTER_SCALE" && "$scale" != "$FILTER_SCALE" ]]; then
            continue
        fi

        SDIR=$(scale_str "$scale")

        for method in "${METHODS[@]}"; do
            for ((run_id=0; run_id<RUNS; run_id++)); do
                TOTAL=$((TOTAL + 1))
                RESULT_FILE="$PROJECT_ROOT/$OUTPUT_DIR/$SDIR/$benchmark/$method/run_${run_id}.json"

                if [[ -f "$RESULT_FILE" ]]; then
                    echo "SKIP  $benchmark  $method  scale=$scale  run_$run_id (exists)"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi

                if [[ "$DRY_RUN" == "true" ]]; then
                    echo "RUN   $benchmark  $method  scale=$scale  run_$run_id"
                    continue
                fi

                echo "--------------------------------------------------------------"
                echo "RUN   $benchmark  $method  scale=$scale  run_$run_id  ($(date '+%H:%M:%S'))"
                echo "--------------------------------------------------------------"

                MAX_INST_ARG=""
                [[ -n "$MAX_INSTANCES" ]] && MAX_INST_ARG="--max-instances $MAX_INSTANCES"
                if python "$RUNNER" \
                    --benchmark "$benchmark" \
                    --method "$method" \
                    --scale "$scale" \
                    --runs 1 \
                    --start-run "$run_id" \
                    --timeout "$TIMEOUT" \
                    --device "$DEVICE" \
                    --output-dir "$PROJECT_ROOT/$OUTPUT_DIR" \
                    $MAX_INST_ARG \
                    2>&1 | tee -a "$LOG_FILE"; then
                    EXECUTED=$((EXECUTED + 1))
                    echo "  ✓ Done"
                else
                    FAILED=$((FAILED + 1))
                    echo "  ✗ FAILED (see log)" | tee -a "$LOG_FILE"
                fi
                echo ""
            done
        done
    done
done

echo ""
echo "=================================================================="
echo "RQ3 Experiment Complete"
echo "=================================================================="
echo "Total:    $TOTAL"
echo "Executed: $EXECUTED"
echo "Skipped:  $SKIPPED"
echo "Failed:   $FAILED"
echo "=================================================================="

# Run aggregator if not dry-run and any results exist
if [[ "$DRY_RUN" == "false" && $EXECUTED -gt 0 ]]; then
    echo ""
    echo "Running results aggregator..."
    python "$SCRIPT_DIR/rq3_aggregator.py" "$PROJECT_ROOT/$OUTPUT_DIR" || true
fi
