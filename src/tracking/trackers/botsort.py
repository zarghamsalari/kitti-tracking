"""BoT-SORT tracker — task T7.

Plan:
1. Same input/output shape as bytetrack.py — consumes the same detection files,
   produces tracks under runs/track/botsort/<seq>.txt.
2. Adds:
   - Camera motion compensation (CMC) via sparse optical flow
   - ReID embeddings (OSNet x0_25, MSMT17 weights — small, fast)
   - Combined IoU + appearance distance for association

Implementation notes for Claude Code:
- Vendor from official BoT-SORT repo: https://github.com/NirAharon/BoT-SORT (MIT)
  Or use `boxmot` package which exposes BoT-SORT cleanly.
- `boxmot` is the pragmatic choice — avoids vendoring a large repo.
- Same detections as ByteTrack — this is the whole point of the ablation.
  Verify by hashing the detection files in run_meta.json.
"""

from __future__ import annotations

from pathlib import Path


def run_botsort(config_path: Path) -> None:
    raise NotImplementedError("Task T7. See module docstring for the implementation plan.")
