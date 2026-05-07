"""Tests for tracking.detection.yolo and tracking.detection.run_meta.

Pure logic only — no model loading, no network. The actual ultralytics
inference is exercised by manual verification (`tracking detect ...`)
post-merge, since model weights and torch are too heavy for CI.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from tracking.detection.run_meta import (
    EvalSummary,
    RunMeta,
    file_sha256,
    read_run_meta,
    write_run_meta,
)
from tracking.detection.yolo import (
    COCO_TO_KITTI,
    coco_to_kitti,
    detection_to_mot16_row,
)

# --- COCO -> KITTI class remap ----------------------------------------


def test_coco_to_kitti_person_maps_to_pedestrian() -> None:
    assert coco_to_kitti(0) == 1


def test_coco_to_kitti_car_maps_to_car() -> None:
    assert coco_to_kitti(2) == 0


@pytest.mark.parametrize("coco_class", [1, 3, 16, 80, -1])
def test_coco_to_kitti_other_classes_filtered(coco_class: int) -> None:
    """Bicycle (1), motorcycle (3), dog (16), out-of-range — all filtered."""
    assert coco_to_kitti(coco_class) is None


def test_coco_to_kitti_table_is_minimal() -> None:
    """Guard against accidentally adding classes to the zero-shot map."""
    assert COCO_TO_KITTI == {0: 1, 2: 0}


# --- MOT16 row format -------------------------------------------------


def test_mot16_row_basic_format() -> None:
    row = detection_to_mot16_row(frame=42, x=10.0, y=20.0, w=100.0, h=50.0, conf=0.85, cls=0)
    parts = row.split(",")
    assert len(parts) == 11
    assert parts[0] == "42"
    assert parts[1] == "-1"  # default track_id
    assert parts[2:6] == ["10.00", "20.00", "100.00", "50.00"]
    assert parts[6] == "0.8500"
    assert parts[7] == "0"  # cls = Car (0-indexed parts[7] = doc "Column 8")
    assert parts[8:] == ["-1", "-1", "-1"]


def test_mot16_row_custom_track_id() -> None:
    row = detection_to_mot16_row(frame=0, x=0.0, y=0.0, w=1.0, h=1.0, conf=0.5, cls=1, track_id=7)
    assert row.split(",")[1] == "7"


def test_mot16_row_high_precision_conf() -> None:
    """Confidence should keep 4 decimal places — important for ByteTrack tuning."""
    row = detection_to_mot16_row(frame=0, x=0, y=0, w=1, h=1, conf=0.10056, cls=0)
    assert row.split(",")[6] == "0.1006"


def test_mot16_row_default_track_id_is_minus_one() -> None:
    """parts[1] must be '-1' when track_id is not passed.

    Regression guard: cls is now injected between conf and track_id;
    a future parameter reorder must not silently break the default.
    """
    row = detection_to_mot16_row(frame=0, x=0, y=0, w=1, h=1, conf=0.5, cls=0)
    assert row.split(",")[1] == "-1"


def test_mot16_row_class_column_is_valid_kitti_id() -> None:
    """Class slot (parts[7], doc 'Column 8') must be Car=0 or Pedestrian=1."""
    for kitti_cls in (0, 1):
        row = detection_to_mot16_row(frame=0, x=0, y=0, w=1, h=1, conf=0.5, cls=kitti_cls)
        parts = row.split(",")
        assert len(parts) == 11
        assert parts[7] in {"0", "1"}


# --- run_meta schema --------------------------------------------------


def _example_meta() -> RunMeta:
    return RunMeta(
        timestamp="2026-05-07T20:00:00+00:00",
        git_sha="abc123",
        git_dirty=False,
        config_path="/path/to/config.yaml",
        config_hash="cafebabe" * 8,
        weights_path="/path/to/yolov8m.pt",
        weights_checksum="deadbeef" * 8,
        seed=42,
        imgsz=1280,
        val_conf=0.001,
        dump_conf=0.1,
        iou=0.5,
        classes=[0, 2],
        python_version="3.11.15",
        torch_version="2.2.0",
        ultralytics_version="8.2.0",
        eval=EvalSummary(map50_95=0.42, map50=0.71, per_class={"Car": 0.55, "Pedestrian": 0.30}),
    )


def test_run_meta_roundtrip(tmp_path: Path) -> None:
    """Write meta, read it back, every field round-trips losslessly."""
    original = _example_meta()
    out = tmp_path / "run_meta.json"
    write_run_meta(original, out)
    loaded = read_run_meta(out)

    assert loaded == original


def test_run_meta_schema_includes_all_required_fields(tmp_path: Path) -> None:
    """Schema sanity: every required field must appear in the JSON output."""
    out = tmp_path / "run_meta.json"
    write_run_meta(_example_meta(), out)

    data = json.loads(out.read_text())

    required = {
        "timestamp",
        "git_sha",
        "git_dirty",
        "config_path",
        "config_hash",
        "weights_path",
        "weights_checksum",
        "seed",
        "imgsz",
        "val_conf",
        "dump_conf",
        "iou",
        "classes",
        "python_version",
        "torch_version",
        "ultralytics_version",
        "eval",
        "format_version",
    }
    missing = required - set(data.keys())
    assert not missing, f"run_meta.json missing required fields: {missing}"

    eval_required = {"map50_95", "map50", "per_class"}
    eval_missing = eval_required - set(data["eval"].keys())
    assert not eval_missing, f"eval block missing fields: {eval_missing}"


def test_run_meta_eval_can_be_none(tmp_path: Path) -> None:
    """A run with no eval (e.g., dump-only mode) writes eval: null cleanly."""
    meta = _example_meta()
    meta = dataclasses.replace(meta, eval=None)
    out = tmp_path / "run_meta.json"
    write_run_meta(meta, out)

    loaded = read_run_meta(out)
    assert loaded.eval is None


def test_read_run_meta_handles_v1_files_without_format_version(tmp_path: Path) -> None:
    """Old run_meta.json files (pre-v2, no format_version key) must load cleanly.

    The field's default ('mot16-kitti-v1') is the correct historical label —
    this test makes that guarantee rather than just a comment.
    """
    legacy = {
        "timestamp": "2026-05-07T14:31:20+00:00",
        "git_sha": "abc123",
        "git_dirty": False,
        "config_path": "/path/to/config.yaml",
        "config_hash": "cafebabe" * 8,
        "weights_path": "/path/to/yolov8m.pt",
        "weights_checksum": "deadbeef" * 8,
        "seed": 42,
        "imgsz": 1280,
        "val_conf": 0.001,
        "dump_conf": 0.1,
        "iou": 0.5,
        "classes": [0, 2],
        "python_version": "3.12.10",
        "torch_version": "2.11.0+cpu",
        "ultralytics_version": "8.4.47",
        "eval": None,
        # deliberately no "format_version" key
    }
    path = tmp_path / "run_meta.json"
    path.write_text(json.dumps(legacy))

    meta = read_run_meta(path)
    assert meta.format_version == "mot16-kitti-v1"


# --- file checksums ---------------------------------------------------


def test_file_sha256_known_value(tmp_path: Path) -> None:
    """Sanity check against a known-content file."""
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    # echo -n "hello world" | sha256sum
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert file_sha256(p) == expected
