"""Smoke tests for tracking.cli — minimal coverage focused on import-time
side effects (logging configuration), not Typer command behaviour.
"""

from __future__ import annotations

import logging


def test_logging_configured_after_cli_import() -> None:
    """After importing tracking.cli, the root logger must have at least one
    handler attached so module-level ``logger.info`` calls in
    :mod:`tracking.detection.yolo` reach the terminal during long runs.

    Regression guard: phase 2 of ``tracking detect`` previously logged
    silently because no handler was attached, leading the user to Ctrl+C a
    still-working inference job mid-flight. Wiring ``logging.basicConfig``
    at CLI import keeps the terminal alive with per-sequence progress.
    """
    import tracking.cli  # noqa: F401  (import-only side effect)

    handlers = logging.getLogger().handlers
    assert len(handlers) >= 1, (
        "tracking.cli must configure logging on import; otherwise "
        "logger.info calls in run_detection emit nothing during the "
        "20+ minute MOT16 dump phase."
    )
