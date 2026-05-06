"""ByteTrack tracker — task T5.

Plan:
1. Load detections from runs/det/yolov8m_finetuned/<seq>.txt (MOT16 format,
   one file per sequence).
2. For each sequence, run ByteTrack frame-by-frame:
   - Two-stage matching (high-conf, then low-conf detections)
   - Kalman filter for motion prediction
   - IoU-based association
3. Write tracks to runs/track/bytetrack/<seq>.txt (MOT16 format).
4. Write run_meta.json.

Implementation notes for Claude Code:
- Easiest path: install `bytetrack-pip` or vendor the official ByteTracker
  class from https://github.com/ifzhang/ByteTrack/blob/main/yolox/tracker/byte_tracker.py
  License is MIT — vendor with attribution under src/tracking/trackers/_bytetrack/.
- Alternative: ultralytics ships a ByteTrack implementation; we can call it
  via ultralytics.trackers.byte_tracker.BYTETracker. Less code to maintain.
- Pick the ultralytics path first for speed. Document the choice in commit msg.
- Tune track_thresh, match_thresh, track_buffer in `configs/tracker_bytetrack.yaml`.
  Cap tuning at 5 trials on val (per CLAUDE.md metrics discipline).
"""

from __future__ import annotations

from typing import Any


def run_bytetrack(config: dict[str, Any]) -> None:
    raise NotImplementedError(
        "Task T5. See module docstring for the implementation plan."
    )
