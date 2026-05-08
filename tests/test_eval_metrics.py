"""Tests for tracking.eval.metrics — data-shaping + TrackEval integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tracking.eval.metrics import EvalResults, TrackerMetrics


def test_empty_results_renders_table_header() -> None:
    md = EvalResults().to_markdown()
    assert "| Tracker | HOTA |" in md
    assert "| MOTA |" in md
    assert "| HOTA |" in md


def test_results_renders_row() -> None:
    results = EvalResults(
        trackers=[
            TrackerMetrics(
                name="bytetrack",
                hota=58.4,
                mota=67.1,
                idf1=66.2,
                assa=55.3,
                deta=61.8,
                id_sw=42,
                frag=88,
                mt=120,
                ml=15,
            ),
        ]
    )
    md = results.to_markdown()
    assert "bytetrack" in md
    assert "58.40" in md
    assert "67.10" in md


def test_results_renders_multiple_rows() -> None:
    results = EvalResults(
        trackers=[
            TrackerMetrics("bytetrack", 58.4, 67.1, 66.2, 55.3, 61.8, 42, 88, 120, 15),
            TrackerMetrics("botsort", 60.1, 67.5, 68.0, 58.0, 62.2, 31, 75, 124, 12),
        ]
    )
    md = results.to_markdown()
    assert "bytetrack" in md
    assert "botsort" in md


def test_to_markdown_includes_aggregation_note() -> None:
    """The markdown note explaining macro vs weighted must be present."""
    md = EvalResults().to_markdown()
    assert "macro" in md
    assert "weighted" in md


# ---------------------------------------------------------------------------
# IoU matrix
# ---------------------------------------------------------------------------


def test_iou_matrix_identical_boxes() -> None:
    """Identical boxes must have IoU=1.0."""
    from tracking.eval.metrics import _iou_matrix

    boxes = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    np.testing.assert_allclose(_iou_matrix(boxes, boxes)[0, 0], 1.0, atol=1e-5)


def test_iou_matrix_non_overlapping() -> None:
    from tracking.eval.metrics import _iou_matrix

    a = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    b = np.array([[20.0, 20.0, 30.0, 30.0]], dtype=np.float32)
    np.testing.assert_allclose(_iou_matrix(a, b)[0, 0], 0.0, atol=1e-5)


def test_iou_matrix_empty_pred_returns_n_by_0() -> None:
    """Empty predictions must produce (N, 0) — not a crash or wrong shape."""
    from tracking.eval.metrics import _iou_matrix

    gt = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    pred = np.empty((0, 4), dtype=np.float32)
    result = _iou_matrix(gt, pred)
    assert result.shape == (1, 0)


def test_iou_matrix_empty_gt_returns_0_by_m() -> None:
    from tracking.eval.metrics import _iou_matrix

    gt = np.empty((0, 4), dtype=np.float32)
    pred = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    result = _iou_matrix(gt, pred)
    assert result.shape == (0, 1)


# ---------------------------------------------------------------------------
# _build_sequence_data
# ---------------------------------------------------------------------------


def test_build_sequence_data_shapes(tiny_label_file: Path) -> None:
    """data_dict list lengths and array shapes must match num_frames."""
    from tracking.data.kitti import parse_label_file
    from tracking.eval.metrics import _build_sequence_data

    # tiny_label_file: frame-0 car at [100,200,200,280], frame-1 car at [110,200,210,280]
    gt_annots = parse_label_file(tiny_label_file, classes=("Car",))
    pred_by_frame = {
        0: np.array([[100.0, 200.0, 200.0, 280.0, 1.0, 0.9, 0.0]], dtype=np.float32),
    }
    data = _build_sequence_data(gt_annots, pred_by_frame, num_frames=2, cls_id=0)
    assert data["num_timesteps"] == 2
    assert len(data["gt_ids"]) == 2
    assert len(data["tracker_ids"]) == 2
    assert data["similarity_scores"][0].shape == (1, 1)  # frame 0: 1 GT, 1 pred
    assert data["similarity_scores"][1].shape == (1, 0)  # frame 1: 1 GT, 0 pred


def test_build_sequence_data_empty_predictions_does_not_crash_hota(
    tiny_label_file: Path,
) -> None:
    """A (N, 0) similarity matrix at a frame with no predictions must not crash
    HOTA.eval_sequence(). This is the load-bearing empty-frame check — shape
    alone is not enough.

    If this test fails with KeyError or IndexError, the bug is most likely in
    _build_sequence_data not handling missing frames in pred_by_frame correctly.
    Fix: ensure each frame uses pred_by_frame.get(t) with an empty fallback,
    never assuming all frame keys are present.
    """
    from trackeval.metrics import HOTA

    from tracking.data.kitti import parse_label_file
    from tracking.eval.metrics import _build_sequence_data

    gt_annots = parse_label_file(tiny_label_file, classes=("Car",))
    # No predictions at all — every frame gets a (N, 0) similarity matrix
    data = _build_sequence_data(gt_annots, pred_by_frame={}, num_frames=2, cls_id=0)
    hota = HOTA()
    result = hota.eval_sequence(data)
    assert "HOTA" in result


def test_build_sequence_data_gt_count(tiny_label_file: Path) -> None:
    """num_gt_dets must equal the number of GT annotations for the queried class."""
    from tracking.data.kitti import parse_label_file
    from tracking.eval.metrics import _build_sequence_data

    gt_annots = parse_label_file(tiny_label_file, classes=("Car",))
    data = _build_sequence_data(gt_annots, pred_by_frame={}, num_frames=2, cls_id=0)
    # tiny_label_file has 2 car annotations (frames 0 and 1)
    assert data["num_gt_dets"] == 2
    assert data["num_tracker_dets"] == 0
