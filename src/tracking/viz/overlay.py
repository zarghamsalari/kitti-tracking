"""Bounding box + track ID overlays on KITTI sequences. Task T8.

- id_to_color: deterministic ID -> BGR color via HSV hash.
- draw_tracks: annotate one frame with bboxes coloured by track ID.
- write_video: render a full sequence to mp4 using imageio (vendored ffmpeg).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import imageio
import numpy as np

logger = logging.getLogger(__name__)


def id_to_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic color for a track ID. Returns BGR for OpenCV."""
    rng = np.random.default_rng(seed=track_id)
    h = int(rng.integers(0, 180))
    hsv = np.array([[[h, 200, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_tracks(image: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    """Draw track bboxes + IDs on a frame.

    Args:
        image: HxWx3 BGR uint8.
        tracks: (N, 7) float32 [x1, y1, x2, y2, track_id, conf, cls]
                as returned by read_mot16_v2_tracks for a single frame.
                Empty (0, 7) is valid — returns unmodified copy.

    Returns:
        Annotated image (copy).
    """
    out = image.copy()
    if tracks.shape[0] == 0:
        return out

    for row in tracks:
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        track_id = int(row[4])
        color = id_to_color(track_id)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = str(track_id)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return out


def write_video(
    seq_dir: Path,
    tracks_by_frame: dict[int, np.ndarray],
    out_path: Path,
    fps: int = 10,
) -> Path:
    """Render an annotated mp4 from a KITTI sequence + tracker output.

    Args:
        seq_dir: Path to image_02/<seq>/ directory with .png frames.
        tracks_by_frame: {frame_idx: (N, 7) array} from read_mot16_v2_tracks.
        out_path: Where to write the .mp4 file.
        fps: Frames per second (KITTI = 10).

    Returns:
        out_path for convenience.
    """
    frames = sorted(seq_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No .png frames in {seq_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)

    h, w = 0, 0
    for idx, frame_path in enumerate(frames):
        img = cv2.imread(str(frame_path))
        if img is None:
            raise FileNotFoundError(f"Could not read frame: {frame_path}")
        if idx == 0:
            h, w = img.shape[:2]
        tracks = tracks_by_frame.get(idx, np.empty((0, 7), dtype=np.float32))
        annotated = draw_tracks(img, tracks)
        # imageio expects RGB, cv2 gives BGR
        writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    writer.close()

    logger.info("Wrote %d frames to %s (%dx%d @ %d fps)", len(frames), out_path, w, h, fps)
    return out_path
