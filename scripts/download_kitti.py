#!/usr/bin/env python
"""Download the KITTI 2D MOT training dataset (left color images + GT labels).

Usage:
    python scripts/download_kitti.py [target_dir]

Default target: ``data/kitti_tracking``. Total ~15 GB.

Stdlib only — no ``wget``/``unzip`` required, runs identically on Windows,
Linux, and macOS. Idempotent: re-running on an already-extracted dataset
is a no-op (no re-download).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from tracking.data.download import DEFAULT_TARGET_DIR, download_kitti


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    target = Path(argv[1]) if len(argv) > 1 else DEFAULT_TARGET_DIR
    download_kitti(target)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
