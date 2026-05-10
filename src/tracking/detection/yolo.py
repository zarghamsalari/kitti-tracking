"""YOLOv8 detection — task T3 (zero-shot) and T4 (fine-tuned).

Two-phase pipeline driven by ``configs/detector_yolov8.yaml``:

1. **mAP eval** — ``ultralytics.YOLO.val()`` against YOLO-format labels
   for the configured val split. Confidence threshold is set very low
   (``val_conf``, default 0.001) so the full PR curve feeds into mAP.
   Skippable via ``skip_eval: true`` in config (e.g. for fine-tuned runs
   where training already reported val mAP).
2. **MOT16 dump** — per-frame inference at ``dump_conf`` (default 0.1)
   on raw KITTI val frames, converting boxes to MOT16 rows under
   ``output.det_dir/<seq>.txt``. ``dump_conf`` is deliberately low to
   preserve ByteTrack's two-stage matching differentiator.

Class filter is applied at the model level (``classes=[0, 2]``) — the
COCO-pretrained model returns only ``person`` (0) and ``car`` (2),
which are then remapped to KITTI's ``Pedestrian`` (1) and ``Car`` (0)
in :func:`coco_to_kitti`. Cyclist is intentionally not handled here
(see :mod:`tracking.data.yolo_format` for the rationale).

Dual class-space design
-----------------------
There are TWO different KITTI -> integer mappings in play, and they
serve different phases:

* **Eval phase (``model.val()``)** uses
  :data:`tracking.data.yolo_format.KITTI_TO_COCO_ZEROSHOT` so GT labels
  live in COCO80 index space — the same space the COCO-pretrained model
  predicts in. Without this, ``model.val()`` matches predictions and GT
  by raw integer and most "matches" are coincidental IoU overlaps with
  semantically different classes (the mAP ≈ 0 collapse seen on the
  first run before this design was correct).
* **MOT16 dump phase** uses a class remap dict to convert model output
  class indices into KITTI tracker input space (Car=0, Pedestrian=1).
  For zero-shot: ``{0: 1, 2: 0}`` (COCO person=0, car=2 → KITTI).
  For fine-tuned: ``{0: 0, 3: 1}`` (5-class head Car=0, Pedestrian=3).
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from tracking.data.kitti import KittiSequence, KittiTrackingDataset
from tracking.data.yolo_format import (
    COCO80_NAMES,
    KITTI_TO_COCO_ZEROSHOT,
    prepare_yolo_eval_dataset,
)
from tracking.detection.run_meta import (
    EvalSummary,
    RunMeta,
    file_sha256,
    git_dirty,
    git_sha,
    write_run_meta,
)
from tracking.utils.seed import set_seed

logger = logging.getLogger(__name__)


# COCO -> KITTI class id mapping for zero-shot detection.
# COCO 'person' (0)  -> KITTI Pedestrian (1)
# COCO 'car'    (2)  -> KITTI Car (0)
# Anything else: filtered out (returns None).
COCO_TO_KITTI: dict[int, int] = {0: 1, 2: 0}


def coco_to_kitti(coco_class: int) -> int | None:
    """Map a COCO class id to KITTI class id, or ``None`` if filtered out."""
    return COCO_TO_KITTI.get(coco_class)


def detection_to_mot16_row(
    frame: int,
    x: float,
    y: float,
    w: float,
    h: float,
    conf: float,
    cls: int,
    track_id: int = -1,
) -> str:
    """One MOT16 v2 row: ``frame,id,x,y,w,h,conf,cls,-1,-1,-1`` (11 columns).

    Coordinates are top-left ``(x, y)`` with width/height in pixels.
    ``cls`` is the KITTI class id (Car=0, Pedestrian=1).
    ``track_id`` is ``-1`` for raw detections (set by tracker downstream).
    """
    return f"{frame},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},{cls},-1,-1,-1"


class DetectorConfig(BaseModel):
    weights: str
    imgsz: int = 1280
    val_conf: float = 0.001
    dump_conf: float = 0.1
    iou: float = 0.5
    classes: list[int] = Field(default_factory=lambda: [0, 2])
    device: str = ""


class DatasetConfig(BaseModel):
    root: Path
    val_sequences: list[str]
    yolo_eval_dir: Path = Path("data/yolo_kitti")


class OutputConfig(BaseModel):
    det_dir: Path
    format: str = "mot16"


class DetectConfig(BaseModel):
    detector: DetectorConfig
    dataset: DatasetConfig
    output: OutputConfig
    seed: int = 42
    class_remap: dict[int, int] | None = None  # model cls -> KITTI cls; None = coco_to_kitti
    skip_eval: bool = False  # skip Phase 1 mAP eval (e.g. fine-tuned runs)


def _load_config(path: Path) -> DetectConfig:
    raw = yaml.safe_load(path.read_text())
    return DetectConfig.model_validate(raw)


def _run_eval_phase(model: Any, data_yaml: Path, cfg: DetectorConfig) -> EvalSummary:
    """Phase 1: ``model.val()`` for mAP."""
    logger.info("Phase 1: running ultralytics val (mAP)...")
    metrics = model.val(
        data=str(data_yaml),
        imgsz=cfg.imgsz,
        conf=cfg.val_conf,
        iou=cfg.iou,
        classes=cfg.classes,
        device=cfg.device or None,
        verbose=False,
    )
    # In zero-shot mode the model has 80 COCO classes; only indices 0 (person
    # = KITTI Pedestrian) and 2 (car = KITTI Car) carry any GT. Pull the mAP
    # for those slots specifically and label them with their KITTI names.
    maps = metrics.box.maps
    per_class: dict[str, float] = {}
    if len(maps) >= 3:
        per_class["Pedestrian"] = float(maps[0])
        per_class["Car"] = float(maps[2])
    else:
        # Fine-tune mode (nc=2): contiguous indices Car=0, Pedestrian=1.
        per_class["Car"] = float(maps[0]) if len(maps) > 0 else 0.0
        per_class["Pedestrian"] = float(maps[1]) if len(maps) > 1 else 0.0
    return EvalSummary(
        map50_95=float(metrics.box.map),
        map50=float(metrics.box.map50),
        per_class=per_class,
    )


def _dump_mot16_for_sequence(
    model: Any,
    seq: KittiSequence,
    out_path: Path,
    cfg: DetectorConfig,
    class_remap: dict[int, int] | None = None,
) -> int:
    """Run inference frame-by-frame, write MOT16 rows. Returns row count.

    Args:
        class_remap: Model output class -> KITTI tracker class.
            If ``None``, falls back to :func:`coco_to_kitti` (zero-shot).
            Keys not in the dict are filtered out.
    """
    logger.info("Phase 2 [%s]: %d frames, dumping...", seq.name, seq.num_frames)
    rows: list[str] = []
    for frame in range(seq.num_frames):
        img_path = seq.image_dir / f"{frame:06d}.png"
        if not img_path.exists():
            continue
        results = model(
            str(img_path),
            imgsz=cfg.imgsz,
            conf=cfg.dump_conf,
            iou=cfg.iou,
            classes=cfg.classes,
            device=cfg.device or None,
            verbose=False,
        )
        for r in results:
            for box in r.boxes:
                model_class = int(box.cls)
                if class_remap is not None:
                    kitti_cls = class_remap.get(model_class)
                else:
                    kitti_cls = coco_to_kitti(model_class)
                if kitti_cls is None:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                rows.append(
                    detection_to_mot16_row(
                        frame, x1, y1, x2 - x1, y2 - y1, float(box.conf), kitti_cls
                    )
                )
    out_path.write_text("\n".join(rows) + ("\n" if rows else ""))
    return len(rows)


def run_detection(config_path: Path) -> None:
    """End-to-end detection pipeline (zero-shot or fine-tuned).

    Steps:
        1. Load + validate config (pydantic)
        2. Seed for determinism
        3. (Optional) Phase 1: mAP eval via ``model.val()``
        4. Phase 2: MOT16 dump via per-frame ``model(image)``
        5. Write ``run_meta.json``
    """
    cfg = _load_config(config_path)
    set_seed(cfg.seed)

    val_ds = KittiTrackingDataset.from_split(cfg.dataset.root, cfg.dataset.val_sequences)

    # Heavy imports deferred so test environments without these can import the module.
    from ultralytics import YOLO  # type: ignore[import-untyped]

    model = YOLO(cfg.detector.weights)

    eval_summary: EvalSummary | None = None
    if cfg.skip_eval:
        logger.info("Skipping Phase 1 mAP eval (skip_eval=True)")
    else:
        # Zero-shot eval: labels MUST be in COCO80 index space so that
        # COCO-pretrained predictions (class 0=person, 2=car) match GT by class.
        data_yaml = prepare_yolo_eval_dataset(
            val_ds.sequences,
            cfg.dataset.yolo_eval_dir,
            class_map=KITTI_TO_COCO_ZEROSHOT,
            class_names=COCO80_NAMES,
            split="val",
        )
        eval_summary = _run_eval_phase(model, data_yaml, cfg.detector)

    cfg.output.det_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Phase 2: dumping MOT16 detections...")
    for seq in val_ds.sequences:
        out_path = cfg.output.det_dir / f"{seq.name}.txt"
        n = _dump_mot16_for_sequence(model, seq, out_path, cfg.detector, cfg.class_remap)
        logger.info("[%s] %d detections -> %s", seq.name, n, out_path)

    weights_path = Path(cfg.detector.weights)
    if not weights_path.is_absolute():
        candidate = (Path.cwd() / weights_path).resolve()
        if candidate.exists():
            weights_path = candidate

    import torch  # type: ignore[import-untyped]
    import ultralytics  # type: ignore[import-untyped]

    meta = RunMeta(
        timestamp=dt.datetime.now(dt.UTC).isoformat(),
        git_sha=git_sha(),
        git_dirty=git_dirty(),
        config_path=str(config_path.resolve()),
        config_hash=file_sha256(config_path),
        weights_path=str(weights_path),
        weights_checksum=file_sha256(weights_path) if weights_path.is_file() else "missing",
        seed=cfg.seed,
        imgsz=cfg.detector.imgsz,
        val_conf=cfg.detector.val_conf,
        dump_conf=cfg.detector.dump_conf,
        iou=cfg.detector.iou,
        classes=cfg.detector.classes,
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        ultralytics_version=ultralytics.__version__,
        eval=eval_summary,
        format_version="mot16-kitti-v2",
    )
    write_run_meta(meta, cfg.output.det_dir / "run_meta.json")
    logger.info("Done. Outputs under %s", cfg.output.det_dir)
