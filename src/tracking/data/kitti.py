"""KITTI 2D MOT dataset loader.

KITTI tracking benchmark layout (training):
    data/kitti_tracking/
        training/
            image_02/<seq>/<frame>.png   # left color images
            label_02/<seq>.txt           # GT labels (one file per sequence)

Label format (space-separated, per row):
    frame, track_id, type, truncated, occluded, alpha,
    x1, y1, x2, y2, h, w, l, x, y, z, ry

We use only frame, track_id, type, x1, y1, x2, y2 here. Bbox is converted
to MOT16 (frame, id, x, y, w, h) on demand.

This module is part of task T0 (scaffold). Tests use a tiny in-memory
fixture to verify parsing + sequence grouping; no real KITTI required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# KITTI classes we care about. KITTI uses "Pedestrian" (capital P) etc.
KITTI_CLASSES: tuple[str, ...] = ("Car", "Pedestrian", "Cyclist")


@dataclass(frozen=True)
class KittiAnnotation:
    """One annotated bounding box on one frame."""

    frame: int
    track_id: int
    obj_class: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox_xywh(self) -> tuple[float, float, float, float]:
        """Return MOT-style (x, y, w, h) — top-left + width/height."""
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)


@dataclass
class KittiSequence:
    """A single KITTI tracking sequence (e.g. '0013')."""

    name: str
    image_dir: Path
    annotations: list[KittiAnnotation] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(sorted(self.image_dir.glob("*.png")))

    def frames(self) -> list[Path]:
        """Sorted list of image paths."""
        return sorted(self.image_dir.glob("*.png"))

    def annotations_for_frame(self, frame: int) -> list[KittiAnnotation]:
        return [a for a in self.annotations if a.frame == frame]


def parse_label_file(path: Path, classes: tuple[str, ...] = KITTI_CLASSES) -> list[KittiAnnotation]:
    """Parse a KITTI tracking label_02/<seq>.txt file.

    Args:
        path: Path to a single sequence's label file.
        classes: Subset of object classes to keep. Others are filtered out.
                 DontCare entries are always filtered.

    Returns:
        List of KittiAnnotation, one per (frame, object).
    """
    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")

    annotations: list[KittiAnnotation] = []
    with path.open() as f:
        for raw in f:
            row = raw.strip().split()
            if not row:
                continue
            obj_class = row[2]
            if obj_class not in classes:
                continue
            track_id = int(row[1])
            if track_id < 0:  # DontCare and similar
                continue
            try:
                annotations.append(
                    KittiAnnotation(
                        frame=int(row[0]),
                        track_id=track_id,
                        obj_class=obj_class,
                        x1=float(row[6]),
                        y1=float(row[7]),
                        x2=float(row[8]),
                        y2=float(row[9]),
                    )
                )
            except (IndexError, ValueError) as exc:
                logger.warning("Skipping malformed row in %s: %r (%s)", path, raw, exc)

    return annotations


def load_sequence(
    root: Path,
    seq_name: str,
    classes: tuple[str, ...] = KITTI_CLASSES,
    split: str = "training",
) -> KittiSequence:
    """Load one KITTI sequence's images + annotations.

    Args:
        root: Path to data/kitti_tracking
        seq_name: Sequence id, e.g. "0013"
        classes: Object classes to keep
        split: "training" or "testing"
    """
    image_dir = root / split / "image_02" / seq_name
    label_path = root / split / "label_02" / f"{seq_name}.txt"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    annotations = parse_label_file(label_path, classes) if label_path.is_file() else []
    return KittiSequence(name=seq_name, image_dir=image_dir, annotations=annotations)


@dataclass
class KittiTrackingDataset:
    """Multi-sequence wrapper. Used for split-aware iteration."""

    root: Path
    sequences: list[KittiSequence]

    @classmethod
    def from_split(
        cls,
        root: Path,
        seq_names: list[str],
        classes: tuple[str, ...] = KITTI_CLASSES,
        split: str = "training",
    ) -> KittiTrackingDataset:
        seqs = [load_sequence(root, n, classes=classes, split=split) for n in seq_names]
        return cls(root=root, sequences=seqs)

    def total_frames(self) -> int:
        return sum(s.num_frames for s in self.sequences)

    def total_annotations(self) -> int:
        return sum(len(s.annotations) for s in self.sequences)

    def class_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.sequences:
            for a in s.annotations:
                counts[a.obj_class] = counts.get(a.obj_class, 0) + 1
        return counts


def annotations_to_mot16(annotations: list[KittiAnnotation]) -> np.ndarray:
    """Convert a list of annotations to a MOT16-formatted array.

    Returns:
        (N, 10) array: frame, id, x, y, w, h, conf=1, -1, -1, -1
    """
    rows = []
    for a in annotations:
        x, y, w, h = a.bbox_xywh
        rows.append([a.frame, a.track_id, x, y, w, h, 1.0, -1, -1, -1])
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 10), dtype=np.float32)
