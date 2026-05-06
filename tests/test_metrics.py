"""Tests for tracking.eval.metrics — only the data-shaping parts (no TrackEval needed)."""

from __future__ import annotations

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
