"""Tests for tracking.data.yolo_format."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tracking.data.kitti import KittiAnnotation, KittiSequence
from tracking.data.yolo_format import (
    KITTI_IMAGE_SIZE,
    KITTI_TO_YOLO_ZEROSHOT,
    annotation_to_yolo_line,
    bbox_to_yolo,
    frame_yolo_labels,
    write_yolo_labels,
)

IMG_W, IMG_H = KITTI_IMAGE_SIZE  # (1242, 375)


def _ann(
    obj_class: str, x1: float, y1: float, x2: float, y2: float, frame: int = 0
) -> KittiAnnotation:
    return KittiAnnotation(frame=frame, track_id=0, obj_class=obj_class, x1=x1, y1=y1, x2=x2, y2=y2)


def test_bbox_to_yolo_centre_and_size() -> None:
    # 100x100 box centred at (200, 100) in a 1000x500 image.
    cx, cy, w, h = bbox_to_yolo(150, 50, 250, 150, 1000, 500)
    assert math.isclose(cx, 0.2)
    assert math.isclose(cy, 0.2)
    assert math.isclose(w, 0.1)
    assert math.isclose(h, 0.2)


def test_bbox_to_yolo_clamps_out_of_bounds() -> None:
    # Box partly off the right and bottom edges — clamp output to [0, 1].
    cx, cy, w, h = bbox_to_yolo(900, 400, 1100, 600, 1000, 500)
    for value in (cx, cy, w, h):
        assert 0.0 <= value <= 1.0


def test_bbox_to_yolo_rejects_zero_dimensions() -> None:
    with pytest.raises(ValueError):
        bbox_to_yolo(0, 0, 10, 10, 0, 100)
    with pytest.raises(ValueError):
        bbox_to_yolo(0, 0, 10, 10, 100, 0)


def test_annotation_to_yolo_line_mapped_class() -> None:
    ann = _ann("Car", 100, 200, 200, 280)
    line = annotation_to_yolo_line(ann, KITTI_TO_YOLO_ZEROSHOT, IMG_W, IMG_H)
    assert line is not None
    parts = line.split()
    assert len(parts) == 5
    assert parts[0] == "0"  # Car -> 0
    # All four normalised coords parse as float in [0, 1].
    for value in parts[1:]:
        f = float(value)
        assert 0.0 <= f <= 1.0


def test_annotation_to_yolo_line_pedestrian_maps_to_one() -> None:
    ann = _ann("Pedestrian", 50, 150, 80, 230)
    line = annotation_to_yolo_line(ann, KITTI_TO_YOLO_ZEROSHOT, IMG_W, IMG_H)
    assert line is not None
    assert line.startswith("1 ")


def test_annotation_to_yolo_line_unmapped_class_returns_none() -> None:
    # Cyclist is intentionally absent from the zero-shot map.
    ann = _ann("Cyclist", 100, 200, 150, 280)
    assert annotation_to_yolo_line(ann, KITTI_TO_YOLO_ZEROSHOT, IMG_W, IMG_H) is None


def test_frame_yolo_labels_filters_unmapped_classes(tmp_path: Path) -> None:
    seq = KittiSequence(
        name="0000",
        image_dir=tmp_path,
        annotations=[
            _ann("Car", 100, 200, 200, 280, frame=0),
            _ann("Cyclist", 300, 200, 350, 280, frame=0),  # filtered
            _ann("Pedestrian", 50, 150, 80, 230, frame=0),
            _ann("Car", 100, 200, 200, 280, frame=1),  # different frame, ignored
        ],
    )
    lines = frame_yolo_labels(
        seq, frame=0, class_map=KITTI_TO_YOLO_ZEROSHOT, img_w=IMG_W, img_h=IMG_H
    )
    assert len(lines) == 2
    classes = {line.split()[0] for line in lines}
    assert classes == {"0", "1"}  # Car and Pedestrian only


def test_frame_yolo_labels_empty_frame(tmp_path: Path) -> None:
    seq = KittiSequence(name="0000", image_dir=tmp_path, annotations=[])
    assert (
        frame_yolo_labels(seq, frame=0, class_map=KITTI_TO_YOLO_ZEROSHOT, img_w=IMG_W, img_h=IMG_H)
        == []
    )


def test_write_yolo_labels_creates_one_file_per_frame(tmp_path: Path) -> None:
    # Lay out a fake KITTI image dir with 3 PNGs so num_frames returns 3.
    img_dir = tmp_path / "images" / "0000"
    img_dir.mkdir(parents=True)
    for i in range(3):
        (img_dir / f"{i:06d}.png").write_bytes(b"")

    seq = KittiSequence(
        name="0000",
        image_dir=img_dir,
        annotations=[
            _ann("Car", 100, 200, 200, 280, frame=0),
            _ann("Cyclist", 300, 200, 350, 280, frame=1),  # filtered -> empty file
            _ann("Pedestrian", 50, 150, 80, 230, frame=2),
        ],
    )

    out_dir = tmp_path / "yolo_labels"
    total = write_yolo_labels([seq], out_dir)

    # One Car + one Pedestrian written; Cyclist filtered.
    assert total == 2

    files = sorted((out_dir / "0000").glob("*.txt"))
    assert [f.name for f in files] == ["000000.txt", "000001.txt", "000002.txt"]

    assert (out_dir / "0000" / "000000.txt").read_text().startswith("0 ")  # Car
    assert (out_dir / "0000" / "000001.txt").read_text() == ""  # filtered Cyclist
    assert (out_dir / "0000" / "000002.txt").read_text().startswith("1 ")  # Pedestrian
