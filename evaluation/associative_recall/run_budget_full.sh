#!/bin/bash
# Serial driver for full budget-aware eval.
# Runs every (arch, dataset) combination sequentially, appending to a
# single log file so no cache collisions.

set -u
cd "$(dirname "$0")"
LOG=/tmp/claude/budget_full.log
mkdir -p /tmp/claude
> "$LOG"

ARCHES_K20=(baseline_20 v15_tight_20 v2f_tight_20 pure_cue_20 single_cue_20)
ARCHES_K50=(baseline_50 v15_tight_50 v2f_tight_50 wide_cue_50 gencheck_50)
ARCHES_K100=(baseline_100 v2f_100)
ALL_ARCHES=("${ARCHES_K20[@]}" "${ARCHES_K50[@]}" "${ARCHES_K100[@]}")

DATASETS=(synthetic_19q puzzle_16q advanced_23q locomo_30q)

for arch in "${ALL_ARCHES[@]}"; do
  for ds in "${DATASETS[@]}"; do
    echo "=== $(date +%H:%M:%S) :: $arch on $ds ===" | tee -a "$LOG"
    uv run python budget_aware_eval.py --arch "$arch" --dataset "$ds" \
      2>&1 | tee -a "$LOG" | tail -5
    echo "" | tee -a "$LOG"
  done
done

echo "=== ALL DONE $(date +%H:%M:%S) ===" | tee -a "$LOG"

# Run final summary
uv run python budget_aware_eval.py 2>&1 | tee -a "$LOG"
