#!/usr/bin/env bash
# Full pipeline: detect -> track (both trackers) -> eval -> markdown
# Assumes the detector has already been fine-tuned. If not, run `make detect` first.

set -euo pipefail

GT_DIR="data/kitti_tracking/training/label_02"

if [[ ! -d "$GT_DIR" ]]; then
    echo "GT directory not found: $GT_DIR"
    echo "Run 'make download' first."
    exit 1
fi

echo "==> Detection"
tracking detect --config configs/detector_yolov8.yaml

for tracker in bytetrack botsort; do
    echo "==> Tracking with $tracker"
    tracking track --config "configs/tracker_${tracker}.yaml"

    echo "==> Eval $tracker"
    tracking eval \
        --gt "$GT_DIR" \
        --pred "runs/track/${tracker}" \
        --out "docs/results_${tracker}.md"
done

# Concatenate per-tracker results into the main results file
{
    echo "# Results"
    echo
    cat docs/results_bytetrack.md
    echo
    cat docs/results_botsort.md
} > docs/results.md

echo "==> Wrote docs/results.md"
