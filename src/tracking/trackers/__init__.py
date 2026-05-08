"""Tracker implementations: ByteTrack (T5) and BoT-SORT (T7)."""

from __future__ import annotations

from pathlib import Path

import yaml


def run_tracker(config_path: Path) -> None:
    """Dispatch to the configured tracker by name field in the YAML."""
    config = yaml.safe_load(config_path.read_text())
    name = config.get("name") or config.get("tracker", {}).get("name")

    if name == "bytetrack":
        from tracking.trackers.bytetrack import run_bytetrack

        run_bytetrack(config_path)
    elif name == "botsort":
        from tracking.trackers.botsort import run_botsort

        run_botsort(config_path)
    else:
        raise ValueError(f"Unknown tracker: {name!r}. Expected 'bytetrack' or 'botsort'.")
