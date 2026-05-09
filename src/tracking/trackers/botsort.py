"""BoT-SORT tracker — task T7.

Runs BoT-SORT (boxmot 18.0.0) over saved v2 MOT16 detections, sequence by
sequence, and writes v2 MOT16 track files to runs/track/botsort/.

Class isolation design
----------------------
boxmot's BaseTracker stores `lost_stracks` at the instance level regardless of
the `per_class` flag. This means a Pedestrian detection can match a recently-
lost Car track (same IoU pool). To prevent cross-class ID continuation we
instantiate one BotSort per class and route each detection to its class-
specific tracker. Lost tracks in tracker_0 (Car) are never visible to
tracker_1 (Pedestrian).

Key parameter — frame_rate=10
------------------------------
boxmot computes buffer_size = int(frame_rate / 30.0 * track_buffer).
At the library default frame_rate=30 this gives buffer_size=30 frames = 3.0 s
at KITTI's 10 FPS — 3x too long. frame_rate=10 gives buffer_size=10 frames
(~1.0 s), matching the paper's intended tolerance.

with_reid=False (headline baseline)
------------------------------------
Row 2 of the ablation table: CMC-only, no ReID model download, CI-safe.
reid_weights positional arg is required syntactically but unused when
with_reid=False — pass Path("none") as a dummy.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from pydantic import BaseModel

from tracking.data.kitti import load_sequence
from tracking.trackers.mot16_io import read_mot16_v2, write_mot16_v2
from tracking.utils.seed import set_seed

logger = logging.getLogger(__name__)


class BoTSortConfig(BaseModel):
    name: str = "botsort"
    detections_dir: Path
    tracks_dir: Path
    data_root: Path
    val_sequences: list[str]
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    cmc_method: str = "sof"
    frame_rate: int = 10  # KITTI FPS — must not change; see module docstring
    with_reid: bool = False  # headline baseline; follow-up PR enables True
    nr_classes: int = 2  # number of class-specific tracker instances to create
    seed: int = 42


def _make_tracker(cfg: BoTSortConfig) -> Any:
    """Construct one BotSort instance with the config parameters."""
    from boxmot.trackers import BotSort

    return BotSort(
        reid_weights=Path("none"),  # unused when with_reid=False
        device=torch.device("cpu"),
        half=False,
        track_high_thresh=cfg.track_high_thresh,
        track_low_thresh=cfg.track_low_thresh,
        new_track_thresh=cfg.new_track_thresh,
        track_buffer=cfg.track_buffer,
        match_thresh=cfg.match_thresh,
        proximity_thresh=cfg.proximity_thresh,
        appearance_thresh=cfg.appearance_thresh,
        cmc_method=cfg.cmc_method,
        frame_rate=cfg.frame_rate,
        with_reid=cfg.with_reid,
    )


def run_botsort(config_path: Path) -> None:
    """Per-sequence BoT-SORT on saved v2 MOT16 detections.

    Uses one BotSort instance per KITTI class so that lost Car tracks are
    never matched against Pedestrian detections (and vice versa).
    """
    cfg = BoTSortConfig(**yaml.safe_load(config_path.read_text()))
    set_seed(cfg.seed)
    cfg.tracks_dir.mkdir(parents=True, exist_ok=True)

    det_hashes: dict[str, str] = {}
    total_rows = 0

    for seq_name in cfg.val_sequences:
        det_path = cfg.detections_dir / f"{seq_name}.txt"
        logger.info("Sequence %s: loading detections from %s", seq_name, det_path)

        dets_by_frame = read_mot16_v2(det_path)
        det_hashes[seq_name] = _sha256(det_path)

        # One tracker per class — prevents cross-class ID inheritance via lost_stracks.
        trackers = {cls_id: _make_tracker(cfg) for cls_id in range(cfg.nr_classes)}

        seq = load_sequence(cfg.data_root, seq_name)
        tracks_by_frame: dict[int, np.ndarray] = {}

        for frame_idx, img_path in enumerate(seq.frames()):
            img = cv2.imread(str(img_path))
            all_dets = dets_by_frame.get(frame_idx, np.empty((0, 6), dtype=np.float32))

            frame_results: list[np.ndarray] = []
            for cls_id, tracker in trackers.items():
                if len(all_dets) > 0:
                    cls_dets = all_dets[all_dets[:, 5] == cls_id]
                else:
                    cls_dets = np.empty((0, 6), dtype=np.float32)
                result = tracker.update(cls_dets, img)
                if result.shape[0] > 0:
                    frame_results.append(result)

            tracks_by_frame[frame_idx] = (
                np.vstack(frame_results) if frame_results else np.empty((0, 8), dtype=np.float32)
            )

        out_path = cfg.tracks_dir / f"{seq_name}.txt"
        n_rows = write_mot16_v2(tracks_by_frame, out_path)
        total_rows += n_rows
        logger.info("Sequence %s: wrote %d track rows to %s", seq_name, n_rows, out_path)

    _write_run_meta(config_path, cfg, det_hashes, total_rows)
    logger.info("BoT-SORT complete. Total track rows: %d", total_rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run_meta(
    config_path: Path,
    cfg: BoTSortConfig,
    det_hashes: dict[str, str],
    total_rows: int,
) -> None:
    meta = {
        "format_version": "mot16-kitti-v2",
        "detection_format_version_consumed": "mot16-kitti-v2",
        "tracker_name": cfg.name,
        "config_path": str(config_path),
        "config_hash": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "detection_input_hashes": det_hashes,
        "total_track_rows": total_rows,
        "seed": cfg.seed,
        "with_reid": cfg.with_reid,
        "cmc_method": cfg.cmc_method,
        "boxmot_version": importlib.metadata.version("boxmot"),
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }
    out = cfg.tracks_dir / "run_meta.json"
    out.write_text(json.dumps(meta, indent=2))
    logger.info("Wrote run_meta.json to %s", out)
