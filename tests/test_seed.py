"""Tests for tracking.utils.seed."""

from __future__ import annotations

import random

from tracking.utils.seed import set_seed


def test_set_seed_python_random() -> None:
    set_seed(42)
    a = [random.random() for _ in range(5)]
    set_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_numpy() -> None:
    import numpy as np

    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    assert (a == b).all()
