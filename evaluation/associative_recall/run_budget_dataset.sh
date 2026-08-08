#!/bin/bash
# Run all architectures on a single dataset (sequentially).
# Usage: run_budget_dataset.sh <dataset_key> <log_suffix>
set -u
cd "$(dirname "$0")"

DS="$1"
SUFFIX="$2"
LOG="/tmp/claude/budget_${SUFFIX}.log"
mkdir -p /tmp/claude
: > "$LOG"

ARCHES=(
  baseline_20 v15_tight_20 v2f_tight_20 pure_cue_20 single_cue_20
  baseline_50 v15_tight_50 v2f_tight_50 wide_cue_50 gencheck_50
  baseline_100 v2f_100
)

for arch in "${ARCHES[@]}"; do
  echo "=== $(date +%H:%M:%S) :: $arch on $DS ===" >> "$LOG"
  uv run python budget_aware_eval.py --arch "$arch" --dataset "$DS" \
    >> "$LOG" 2>&1
done

echo "=== ALL DONE $DS $(date +%H:%M:%S) ===" >> "$LOG"
