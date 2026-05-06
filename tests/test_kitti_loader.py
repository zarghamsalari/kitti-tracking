"""Tests for tracking.data.kitti.

These verify that the loader correctly parses KITTI labels, filters DontCare
and unwanted classes, and converts to MOT16 format.

No real KITTI data is required — fixtures are tiny in-memory text files.
"""

from __future__ import annotations

from pathlib import Path

from tracking.data.kitti import (
    KittiAnnotation,
    annotations_to_mot16,
    parse_label_file,
)


def test_parse_label_file_basic(tiny_label_file: Path) -> None:
    annotations = parse_label_file(tiny_label_file)
    # 2 cars + 1 pedestrian; DontCare filtered
    assert len(annotations) == 3
    assert all(isinstance(a, KittiAnnotation) for a in annotations)


def test_parse_label_file_filters_dontcare(tiny_label_file: Path) -> None:
    annotations = parse_label_file(tiny_label_file)
    assert all(a.obj_class in {"Car", "Pedestrian", "Cyclist"} for a in annotations)
    assert all(a.track_id >= 0 for a in annotations)


def test_parse_label_file_class_filter(tiny_label_file: Path) -> None:
    annotations = parse_label_file(tiny_label_file, classes=("Car",))
    assert len(annotations) == 2
    assert {a.obj_class for a in annotations} == {"Car"}


def test_bbox_xywh_conversion() -> None:
    a = KittiAnnotation(
        frame=0, track_id=0, obj_class="Car", x1=100.0, y1=200.0, x2=200.0, y2=280.0
    )
    x, y, w, h = a.bbox_xywh
    assert (x, y, w, h) == (100.0, 200.0, 100.0, 80.0)


def test_annotations_to_mot16(tiny_label_file: Path) -> None:
    annotations = parse_label_file(tiny_label_file)
    arr = annotations_to_mot16(annotations)
    assert arr.shape == (3, 10)
    # MOT16 columns: frame, id, x, y, w, h, conf, -1, -1, -1
    assert (arr[:, -3:] == -1).all()
    assert (arr[:, 6] == 1.0).all()


def test_annotations_to_mot16_empty() -> None:
    arr = annotations_to_mot16([])
    assert arr.shape == (0, 10)


def test_parse_label_file_missing(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        parse_label_file(tmp_path / "nope.txt")
