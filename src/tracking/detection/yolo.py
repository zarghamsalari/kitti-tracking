"""YOLOv8 detection — tasks T3 (zero-shot) and T4 (fine-tune).

Plan:
1. Load YOLOv8m COCO-pretrained weights.
2. Zero-shot evaluation on KITTI val (map COCO car->Car, person->Pedestrian).
3. Convert KITTI labels to YOLO txt format respecting train/val sequence split.
4. Fine-tune YOLOv8m for 30-50 epochs at imgsz=1280.
5. Run fine-tuned detector on val sequences, dump per-sequence MOT16 detection
   files at runs/det/yolov8m_finetuned/<seq>.txt.

Implementation notes for Claude Code:
- Use ultralytics.YOLO directly. Don't rebuild the trainer.
- KITTI image size is 1242x375. Use imgsz=1280 (multiple of 32). Letterboxing
  is handled by ultralytics.
- For zero-shot, COCO classes 2 (car) and 0 (person) map to KITTI Car and
  Pedestrian. Drop everything else.
- Save run metadata (git SHA, config hash, ultralytics version, seed) to
  runs/det/yolov8m_finetuned/run_meta.json. See tracking.utils.seed.
- Sequence-level holdout is non-negotiable. Generate the YOLO data.yaml so
  train/val image lists are mutually exclusive at the sequence level.
"""

from __future__ import annotations

from pathlib import Path


def run_detection(config_path: Path) -> None:
    """Run the detection pipeline.

    Steps:
        1. Load YAML config
        2. Optionally fine-tune (if finetuned_weights missing)
        3. Run detection on val sequences
        4. Dump MOT16-format detections per sequence
        5. Write run_meta.json
    """
    raise NotImplementedError(
        "Tasks T3 + T4. See module docstring for the implementation plan. "
        "Open this file in Claude Code and ask: 'Implement run_detection per the docstring plan.'"
    )
