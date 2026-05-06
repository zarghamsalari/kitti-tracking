"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tiny_label_file(tmp_path: Path) -> Path:
    """A 4-row KITTI label_02 file: 2 frames, 2 cars, 1 pedestrian, 1 DontCare."""
    content = (
        "0 0 Car 0 0 -1.5 100.0 200.0 200.0 280.0 1.5 1.6 4.0 0 0 0 0\n"
        "0 1 Pedestrian 0 0 -1.5 50.0 150.0 80.0 230.0 1.7 0.5 0.5 0 0 0 0\n"
        "0 -1 DontCare -1 -1 -10 0 0 50 50 -1 -1 -1 -10 -1 -1 -10\n"
        "1 0 Car 0 0 -1.5 110.0 200.0 210.0 280.0 1.5 1.6 4.0 0 0 0 0\n"
    )
    path = tmp_path / "0000.txt"
    path.write_text(content)
    return path
