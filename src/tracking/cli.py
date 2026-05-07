"""Command-line interface.

Usage:
    tracking detect --config configs/detector_yolov8.yaml
    tracking track  --config configs/tracker_bytetrack.yaml
    tracking eval   --gt data/kitti_tracking/training/label_02 \\
                    --pred runs/track/bytetrack
    tracking demo
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console

# Configure logging at CLI import so module-level ``logger.info`` calls in
# tracking.detection.yolo (and other long-running phases) reach the terminal.
# Without this, phase 2 of ``tracking detect`` runs silently for ~25 minutes,
# making the process look hung and tempting users to Ctrl+C a still-working
# inference job mid-flight.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()


@app.command()
def detect(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="Detector YAML config"),
) -> None:
    """Run YOLOv8 detection on KITTI val sequences and dump MOT16 detection files."""
    from tracking.detection.yolo import run_detection

    console.print(f"[bold green]Running detection[/] with config: {config}")
    run_detection(config)


@app.command()
def track(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="Tracker YAML config"),
) -> None:
    """Run a tracker over precomputed detections."""
    from tracking.trackers import run_tracker

    console.print(f"[bold green]Running tracker[/] with config: {config}")
    run_tracker(config)


@app.command()
def eval(
    gt: Path = typer.Option(..., "--gt", exists=True, help="KITTI GT label_02 directory"),
    pred: Path = typer.Option(..., "--pred", exists=True, help="Tracker output directory"),
    out: Path = typer.Option(Path("docs/results.md"), "--out", help="Markdown output path"),
) -> None:
    """Compute HOTA / MOTA / IDF1 / IDSw and render markdown table."""
    from tracking.eval.metrics import evaluate

    console.print(f"[bold green]Evaluating[/] {pred} vs {gt}")
    results = evaluate(gt, pred)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(results.to_markdown())
    console.print(f"[bold cyan]Wrote[/] {out}")


@app.command()
def demo() -> None:
    """Launch the Streamlit demo locally."""
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app/app.py"], check=True)


if __name__ == "__main__":
    app()
