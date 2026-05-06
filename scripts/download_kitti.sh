#!/usr/bin/env bash
# Download KITTI tracking benchmark — left color images + GT labels (training).
# Total ~15 GB. Requires `wget` and `unzip`.
#
# Usage:
#   bash scripts/download_kitti.sh [target_dir]
# Default target: data/kitti_tracking

set -euo pipefail

TARGET_DIR="${1:-data/kitti_tracking}"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

BASE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti"

declare -A FILES=(
    ["data_tracking_image_2.zip"]="left color images (training + testing), ~15 GB"
    ["data_tracking_label_2.zip"]="GT labels (training only), ~9 MB"
)

for fname in "${!FILES[@]}"; do
    desc="${FILES[$fname]}"
    if [[ -f "$fname" ]]; then
        echo "[skip] $fname already present ($desc)"
        continue
    fi
    echo "[get ] $fname — $desc"
    wget -q --show-progress "$BASE_URL/$fname"
    echo "[unzip] $fname"
    unzip -q "$fname"
done

echo "[done] KITTI tracking dataset in $(pwd)"
echo
echo "Layout:"
find . -maxdepth 3 -type d | head -20
