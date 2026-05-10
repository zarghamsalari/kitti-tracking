"""Pre-render demo videos for all 20 cells of the 4-cell ablation.

Usage:
    python scripts/render_demo.py

Renders {detector}_{tracker}_{seq}.mp4 for every combination of:
    detectors: zeroshot, finetuned
    trackers:  bytetrack, botsort
    sequences: 0001, 0006, 0013, 0017, 0019

Requires raw KITTI frames under data/kitti_tracking/ and tracker
outputs under runs/track/. Output goes to streamlit_app/static/videos/.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_ROOT = Path("data/kitti_tracking")
TRACKS_ROOT = Path("runs/track")
OUT_DIR = Path("streamlit_app/static/videos")

VAL_SEQUENCES = ["0001", "0006", "0013", "0017", "0019"]
DETECTORS = ["zeroshot", "finetuned"]
TRACKERS = ["bytetrack", "botsort"]


def get_track_dir(detector: str, tracker: str) -> Path:
    suffix = "_finetuned" if detector == "finetuned" else ""
    return TRACKS_ROOT / f"{tracker}{suffix}"


def main() -> None:
    from tracking.trackers.mot16_io import read_mot16_v2_tracks
    from tracking.viz.overlay import write_video

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cells = [
        (det, trk, seq)
        for det in DETECTORS
        for trk in TRACKERS
        for seq in VAL_SEQUENCES
    ]

    logger.info("Rendering %d demo videos to %s", len(cells), OUT_DIR)
    rendered = 0
    skipped = 0
    errors = 0

    for det, trk, seq in cells:
        out_path = OUT_DIR / f"{det}_{trk}_{seq}.mp4"
        if out_path.exists():
            logger.info("SKIP (exists): %s", out_path.name)
            skipped += 1
            continue

        seq_dir = DATA_ROOT / "training" / "image_02" / seq
        track_file = get_track_dir(det, trk) / f"{seq}.txt"

        if not seq_dir.exists():
            logger.warning("MISSING seq dir: %s", seq_dir)
            errors += 1
            continue
        if not track_file.exists():
            logger.warning("MISSING track file: %s", track_file)
            errors += 1
            continue

        tracks = read_mot16_v2_tracks(track_file)
        write_video(seq_dir, tracks, out_path)
        rendered += 1

    logger.info(
        "Done: %d rendered, %d skipped, %d errors (total cells: %d)",
        rendered, skipped, errors, len(cells),
    )
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
