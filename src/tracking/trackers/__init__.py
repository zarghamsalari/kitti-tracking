"""Tracker implementations: ByteTrack (T5) and BoT-SORT (T7)."""

from __future__ import annotations

from pathlib import Path

import yaml


def run_tracker(config_path: Path) -> None:
    """Dispatch to the configured tracker by name."""
    config = yaml.safe_load(config_path.read_text())
    name = config["tracker"]["name"]

    if name == "bytetrack":
        from tracking.trackers.bytetrack import run_bytetrack

        run_bytetrack(config)
    elif name == "botsort":
        from tracking.trackers.botsort import run_botsort

        run_botsort(config)
    else:
        raise ValueError(f"Unknown tracker: {name!r}. Expected 'bytetrack' or 'botsort'.")
