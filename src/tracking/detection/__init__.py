"""YOLOv8 detection wrapper. Tasks T3 (zero-shot) and T4 (fine-tune)."""

from __future__ import annotations

from tracking.detection.run_meta import (
    EvalSummary,
    RunMeta,
    file_sha256,
    git_dirty,
    git_sha,
    read_run_meta,
    write_run_meta,
)
from tracking.detection.yolo import (
    COCO_TO_KITTI,
    DetectConfig,
    DetectorConfig,
    coco_to_kitti,
    detection_to_mot16_row,
    run_detection,
)

__all__ = [
    "COCO_TO_KITTI",
    "DetectConfig",
    "DetectorConfig",
    "EvalSummary",
    "RunMeta",
    "coco_to_kitti",
    "detection_to_mot16_row",
    "file_sha256",
    "git_dirty",
    "git_sha",
    "read_run_meta",
    "run_detection",
    "write_run_meta",
]
