"""Tracking metrics — task T6.

Primary metric: HOTA. Decomposed into AssA and DetA. MOTA + IDF1 alongside
for backward compat. IDSw and Frag for narrative.

Implementation plan:
1. Use TrackEval (https://github.com/JonathonLuiten/TrackEval, MIT) — installed
   via pip git+ if not on PyPI, or vendored under src/tracking/eval/_trackeval/.
2. Fallback: motmetrics for MOTA/IDF1 only — TrackEval is preferred because
   it implements HOTA correctly.
3. Convert KITTI GT (label_02/<seq>.txt) and our predicted MOT16 tracks
   into TrackEval's MOT format. KITTI conversion utility lives here.
4. Run TrackEval per sequence, aggregate, return EvalResults.
5. Render markdown table to docs/results.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrackerMetrics:
    """Per-tracker aggregated metrics."""

    name: str
    hota: float
    mota: float
    idf1: float
    assa: float
    deta: float
    id_sw: int
    frag: int
    mt: int  # mostly tracked
    ml: int  # mostly lost


@dataclass
class EvalResults:
    """Container for one or more trackers' metrics."""

    trackers: list[TrackerMetrics] = field(default_factory=list)

    def to_markdown(self) -> str:
        header = (
            "| Tracker | HOTA | MOTA | IDF1 | AssA | DetA | IDSw | Frag | MT | ML |\n"
            "|---------|------|------|------|------|------|------|------|----|----|\n"
        )
        rows = "\n".join(
            f"| {t.name} | {t.hota:.2f} | {t.mota:.2f} | {t.idf1:.2f} | "
            f"{t.assa:.2f} | {t.deta:.2f} | {t.id_sw} | {t.frag} | {t.mt} | {t.ml} |"
            for t in self.trackers
        )
        return f"# Tracking Results\n\n{header}{rows}\n"


def evaluate(gt_dir: Path, pred_dir: Path) -> EvalResults:
    """Run TrackEval on predictions vs KITTI GT.

    Args:
        gt_dir: Path to data/kitti_tracking/training/label_02
        pred_dir: Path to runs/track/<tracker>/

    Returns:
        EvalResults with one TrackerMetrics row.
    """
    raise NotImplementedError(
        "Task T6. See module docstring. Use TrackEval; fallback to motmetrics for MOTA/IDF1."
    )
