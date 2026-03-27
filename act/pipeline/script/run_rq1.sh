#!/usr/bin/env bash
# ===========================================================================
# RQ1 Full Experiment Runner
#
# Usage:
#   bash act/pipeline/script/run_rq1.sh [OPTIONS]
#
# Options:
#   --device DEVICE          cpu or cuda (default: cuda)
#   --timeout SECONDS        Per-run timeout (default: 60)
#   --runs N                 Runs per config (default: 5)
#   --output-dir DIR         Results directory (default: experiments/rq1)
#   --dry-run                Print what would be run without executing
#   --benchmark NAME         Run only this benchmark (default: all)
#   --method NAME            Run only this method (default: all)
#   --max-instances N        Limit instances per run (default: all)
#
# ===========================================================================

set -euo pipefail

# Defaults
DEVICE="cuda"
TIMEOUT=60
RUNS=5
OUTPUT_DIR="experiments/rq1"
DRY_RUN=false
FILTER_BENCHMARK=""
FILTER_METHOD=""
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
        --method)           FILTER_METHOD="$2"; shift 2 ;;
        --max-instances)    MAX_INSTANCES="$2"; shift 2 ;;
        *)                  echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Benchmarks and methods (must match rq1_config.py)
BENCHMARKS=("trafficsigns" "cifar100" "tinyimagenet")
METHODS=("batch_aniso" "batch_iso" "seq_fix")

# Script path (relative to project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUNNER="$SCRIPT_DIR/rq1_runner.py"

# Log file
LOG_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/experiment.log"

# Counters
TOTAL=0
SKIPPED=0
EXECUTED=0
FAILED=0

echo "=================================================================="
echo "RQ1 Experiment: Batched Fuzzing Speedup"
echo "=================================================================="
echo "Device:     $DEVICE"
echo "Timeout:    ${TIMEOUT}s per run"
echo "Runs:       $RUNS per config"
echo "Output:     $OUTPUT_DIR"
echo "Dry run:    $DRY_RUN"
echo "Log:        $LOG_FILE"
if [[ -n "$FILTER_BENCHMARK" ]]; then echo "Benchmark:  $FILTER_BENCHMARK"; fi
if [[ -n "$FILTER_METHOD" ]]; then echo "Method:     $FILTER_METHOD"; fi
echo "=================================================================="
echo ""

for benchmark in "${BENCHMARKS[@]}"; do
    # Apply benchmark filter
    if [[ -n "$FILTER_BENCHMARK" && "$benchmark" != "$FILTER_BENCHMARK" ]]; then
        continue
    fi

    for method in "${METHODS[@]}"; do
        # Apply method filter
        if [[ -n "$FILTER_METHOD" && "$method" != "$FILTER_METHOD" ]]; then
            continue
        fi

        for ((run_id=0; run_id<RUNS; run_id++)); do
            TOTAL=$((TOTAL + 1))
            RESULT_FILE="$PROJECT_ROOT/$OUTPUT_DIR/$benchmark/$method/run_${run_id}.json"

            # Resume logic: skip if result already exists
            if [[ -f "$RESULT_FILE" ]]; then
                echo "SKIP  $benchmark/$method/run_$run_id (exists)"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            if [[ "$DRY_RUN" == "true" ]]; then
                echo "RUN   $benchmark  $method  run_$run_id"
                continue
            fi

            echo "--------------------------------------------------------------"
            echo "RUN   $benchmark  $method  run_$run_id  ($(date '+%H:%M:%S'))"
            echo "--------------------------------------------------------------"

            # Execute runner
            MAX_INST_ARG=""
            [[ -n "$MAX_INSTANCES" ]] && MAX_INST_ARG="--max-instances $MAX_INSTANCES"
            if python "$RUNNER" \
                --benchmark "$benchmark" \
                --method "$method" \
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

echo ""
echo "=================================================================="
echo "RQ1 Experiment Complete"
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
    python "$SCRIPT_DIR/results_aggregator.py" "$PROJECT_ROOT/$OUTPUT_DIR" || true
fi
