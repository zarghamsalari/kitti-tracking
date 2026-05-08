#!/usr/bin/env bash
# Eval harness: compute metrics for any tracker whose track files exist.
#
# Detect/track steps are NOT here — they are CPU-bound (minutes to hours)
# and run separately:
#   make detect                      (T3b output, ~30 min on CPU)
#   make track TRACKER=bytetrack     (T5 output, ~5 min)
# This script is intentionally fast: reads existing track files, computes metrics.

set -euo pipefail

GT_DIR="data/kitti_tracking/training/label_02"

if [[ ! -d "$GT_DIR" ]]; then
    echo "GT directory not found: $GT_DIR"
    echo "Run 'make download' first."
    exit 1
fi

results=()
for tracker in bytetrack botsort; do
    track_dir="runs/track/${tracker}"
    if [[ ! -d "$track_dir" ]]; then
        echo "==> Skipping $tracker (no track files at $track_dir)"
        continue
    fi

    echo "==> Eval $tracker"
    tracking eval \
        --gt "$GT_DIR" \
        --pred "$track_dir" \
        --out "docs/results_${tracker}.md"
    results+=("docs/results_${tracker}.md")
done

if [[ ${#results[@]} -eq 0 ]]; then
    echo "No trackers evaluated. Run 'make track' first."
    exit 1
fi

{
    echo "# Results"
    echo
    for f in "${results[@]}"; do
        cat "$f"
        echo
    done
} > docs/results.md

echo "==> Wrote docs/results.md"
