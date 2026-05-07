"""Run metadata for detector runs.

Captures the full reproducibility envelope: code state, config, weights,
runtime versions, eval results. Written to ``run_meta.json`` next to each
detector run's outputs.

Fields are split into two categories:

* **Static**: timestamp, git SHA, config + weights checksums, seeds,
  hyperparameters, library versions. Lets a future reader reconstruct
  the exact code+config+weights state without re-running.
* **Dynamic**: ``EvalSummary`` with mAP numbers from
  ``ultralytics.YOLO.val()``. ``None`` if eval was skipped.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalSummary:
    """Per-class mAP from ultralytics' val output."""

    map50_95: float
    map50: float
    per_class: dict[str, float] = field(default_factory=dict)


@dataclass
class RunMeta:
    """Frozen snapshot of one detector run."""

    timestamp: str
    git_sha: str
    git_dirty: bool
    config_path: str
    config_hash: str
    weights_path: str
    weights_checksum: str
    seed: int
    imgsz: int
    val_conf: float
    dump_conf: float
    iou: float
    classes: list[int]
    python_version: str
    torch_version: str
    ultralytics_version: str
    eval: EvalSummary | None = None
    format_version: str = "mot16-kitti-v1"


def git_sha() -> str:
    """Current HEAD SHA, or ``'unknown'`` if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_dirty() -> bool:
    """``True`` if working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def file_sha256(path: Path) -> str:
    """sha256 hex digest of a file, streamed in 1 MB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_meta(meta: RunMeta, out_path: Path) -> None:
    """Serialise a :class:`RunMeta` to JSON at ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataclasses.asdict(meta), indent=2))


def read_run_meta(path: Path) -> RunMeta:
    """Load and parse a ``run_meta.json``. Inverse of :func:`write_run_meta`."""
    data = json.loads(path.read_text())
    eval_data = data.pop("eval", None)
    eval_summary = EvalSummary(**eval_data) if eval_data is not None else None
    return RunMeta(eval=eval_summary, **data)
