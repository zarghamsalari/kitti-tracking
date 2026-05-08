"""MOT16 v2 reader and writer for the KITTI tracking pipeline.

Format (11 columns, comma-separated):
    frame, track_id, x, y, w, h, conf, cls, -1, -1, -1

where x,y,w,h are in pixel coords (xywh on disk, xyxy in memory).
cls uses KITTI integer ids: Car=0, Pedestrian=1.

The reader enforces format_version == "mot16-kitti-v2" via the adjacent
run_meta.json. Consuming a v1 file (no cls column) silently produces
all-(-1) class ids, which corrupts per-class HOTA without a runtime error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_EXPECTED_FORMAT_VERSION = "mot16-kitti-v2"


def read_mot16_v2(path: Path) -> dict[int, np.ndarray]:
    """Read an 11-column v2 MOT16 detection file.

    Returns dict mapping frame_idx -> (N, 6) float32 [x1, y1, x2, y2, conf, cls]
    in xyxy coords. Empty frames are absent from the dict (caller supplies zeros).

    Raises ValueError if the adjacent run_meta.json format_version != mot16-kitti-v2.
    """
    meta_path = path.parent / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        fv = meta.get("format_version", "mot16-kitti-v1")
        if fv != _EXPECTED_FORMAT_VERSION:
            raise ValueError(
                f"Expected format_version '{_EXPECTED_FORMAT_VERSION}' in {meta_path}, "
                f"got '{fv}'. Re-run `tracking detect` to regenerate v2 detection files."
            )

    by_frame: dict[int, list[list[float]]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        frame = int(parts[0])
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])
        conf = float(parts[6])
        cls = float(parts[7])
        by_frame.setdefault(frame, []).append([x, y, x + w, y + h, conf, cls])

    return {frame: np.array(rows, dtype=np.float32) for frame, rows in by_frame.items()}


def read_mot16_v2_tracks(path: Path) -> dict[int, np.ndarray]:
    """Read an 11-column v2 MOT16 track file (tracker output, not detections).

    Returns dict mapping frame_idx -> (N, 7) float32
    [x1, y1, x2, y2, track_id, conf, cls] in xyxy coords.

    Unlike read_mot16_v2 (which discards track_id for detection use),
    this reader preserves track_id at column 4 for evaluation. Discarding
    it (as the detection reader does) produces all-zeros association and HOTA=0.
    Applies the same format_version check as read_mot16_v2.
    """
    meta_path = path.parent / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        fv = meta.get("format_version", "mot16-kitti-v1")
        if fv != _EXPECTED_FORMAT_VERSION:
            raise ValueError(
                f"Expected format_version '{_EXPECTED_FORMAT_VERSION}' in {meta_path}, "
                f"got '{fv}'. Re-run `tracking track` to regenerate v2 track files."
            )

    by_frame: dict[int, list[list[float]]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        frame = int(parts[0])
        track_id = float(parts[1])
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])
        conf = float(parts[6])
        cls = float(parts[7])
        by_frame.setdefault(frame, []).append([x, y, x + w, y + h, track_id, conf, cls])

    return {frame: np.array(rows, dtype=np.float32) for frame, rows in by_frame.items()}


def write_mot16_v2(tracks_by_frame: dict[int, np.ndarray], path: Path) -> int:
    """Write tracker output to an 11-column v2 MOT16 file. Returns row count.

    Input arrays are boxmot BYTETracker output: (M, 8) float32
    [x1, y1, x2, y2, track_id, conf, cls, det_ind]. Converts xyxy -> xywh on write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for frame_idx in sorted(tracks_by_frame.keys()):
        tracks = tracks_by_frame[frame_idx]
        if tracks.shape[0] == 0:
            continue
        for track in tracks:
            x1, y1, x2, y2 = float(track[0]), float(track[1]), float(track[2]), float(track[3])
            track_id = int(track[4])
            conf = float(track[5])
            cls = int(track[6])
            rows.append(
                f"{frame_idx},{track_id},{x1:.2f},{y1:.2f},"
                f"{x2 - x1:.2f},{y2 - y1:.2f},{conf:.4f},{cls},-1,-1,-1"
            )
    path.write_text("\n".join(rows) + ("\n" if rows else ""))
    return len(rows)
