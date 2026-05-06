"""Bounding box + track ID overlays on KITTI sequences. Task T8.

Plan:
- draw_tracks(image, tracks) — annotate one frame with bboxes coloured by ID.
- write_video(seq, tracks, out_path) — render a sequence to mp4 for the demo.
- Use OpenCV for drawing, ffmpeg via cv2.VideoWriter for mp4.
- Stable ID-to-color mapping: hash track_id -> HSV color, convert to BGR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def id_to_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic color for a track ID. Returns BGR for OpenCV."""
    import cv2

    rng = np.random.default_rng(seed=track_id)
    h = int(rng.integers(0, 180))
    hsv = np.array([[[h, 200, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_tracks(image: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    """Draw track bboxes + IDs on a frame.

    Args:
        image: HxWx3 BGR uint8.
        tracks: Nx6 array — frame, id, x, y, w, h.

    Returns:
        Annotated image (copy).
    """
    raise NotImplementedError("Task T8.")


def write_video(seq_dir: Path, tracks_path: Path, out_path: Path, fps: int = 10) -> None:
    """Render an annotated mp4 from a KITTI sequence + tracker output."""
    raise NotImplementedError("Task T8.")
