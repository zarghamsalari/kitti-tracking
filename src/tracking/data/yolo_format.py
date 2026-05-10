"""KITTI -> YOLO label format conversion.

YOLO expects one ``.txt`` file per image, with each row::

    <class_id> <cx> <cy> <w> <h>

where the four box coordinates are normalised to ``[0, 1]`` against image
dimensions. Frames with no labelled objects get an empty ``.txt`` (this
keeps ``ultralytics``' eval pipeline from complaining about missing
ground-truth files).

Class mapping for zero-shot evaluation
--------------------------------------
For the zero-shot stage (T3) we evaluate **Car** and **Pedestrian** only and
deliberately drop **Cyclist**. COCO has ``person`` (class 0) and ``bicycle``
(class 1) as separate categories, while KITTI's ``Cyclist`` is a single
annotation covering "person on a bicycle". A YOLOv8m COCO checkpoint will
emit two detections (a person and a bicycle) for one KITTI cyclist, so any
zero-shot remap conflicts with the ``person -> Pedestrian`` mapping. We
re-introduce Cyclist after fine-tuning (T4), where the model learns the
merged concept directly.

Class mapping for fine-tune (T4)
--------------------------------
5-class mapping: Car=0, Van=1, Truck=2, Pedestrian=3, Cyclist=4.
Excludes Person_sitting, Tram, Misc, DontCare entirely.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import cv2
import yaml

from tracking.data.kitti import KittiAnnotation, KittiSequence, load_sequence

logger = logging.getLogger(__name__)

# KITTI -> 2-class YOLO mapping for fine-tune (T4). The output head will be
# rebuilt with nc=2 during fine-tuning, so we use a contiguous [0, 1] space.
KITTI_TO_YOLO_FINETUNE: dict[str, int] = {
    "Car": 0,
    "Pedestrian": 1,
}

# KITTI -> COCO80 mapping for zero-shot eval (T3b). The COCO-pretrained model
# emits predictions in COCO index space; for ``model.val()`` to match
# predictions to ground truth by class, GT labels must use the SAME indices.
KITTI_TO_COCO_ZEROSHOT: dict[str, int] = {
    "Car": 2,  # COCO 'car'
    "Pedestrian": 0,  # COCO 'person'
}

# Back-compat alias so external code that imported KITTI_TO_YOLO_ZEROSHOT
# (from T3a, before the zero-shot vs fine-tune distinction was clear) still
# resolves. Prefer the explicit names above for new code.
KITTI_TO_YOLO_ZEROSHOT = KITTI_TO_YOLO_FINETUNE

# 5-class YOLO mapping for T4 fine-tune. Excludes Person_sitting, Tram, Misc,
# DontCare entirely — these are not detection targets.
KITTI_TO_YOLO_5CLASS: dict[str, int] = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Cyclist": 4,
}

# Canonical KITTI image size — actual dims are read per-sequence in
# write_yolo_labels (see _read_image_size). Kept here for fixtures, docs,
# and the optional override path of write_yolo_labels.
KITTI_IMAGE_SIZE: tuple[int, int] = (1242, 375)

# Standard COCO80 class names — emitted in the order the COCO-pretrained
# YOLOv8 model uses internally. Required in data.yaml when running
# zero-shot eval, so ultralytics knows what each class index means.
COCO80_NAMES: list[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def bbox_to_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Convert pixel ``(x1, y1, x2, y2)`` to normalised YOLO ``(cx, cy, w, h)``.

    All outputs are clamped to ``[0, 1]`` so a bbox extending slightly
    outside the image (rare in KITTI but possible) doesn't break the YOLO
    parser. Coordinates are computed before clamping so a partly off-image
    box still has the correct centre.
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Image dimensions must be positive, got ({img_w}, {img_h})")

    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    )


def annotation_to_yolo_line(
    ann: KittiAnnotation,
    class_map: dict[str, int],
    img_w: int,
    img_h: int,
) -> str | None:
    """Render one ``KittiAnnotation`` as a YOLO label line.

    Returns ``None`` if ``ann.obj_class`` is not in ``class_map`` (e.g.
    Cyclist under :data:`KITTI_TO_YOLO_ZEROSHOT`) — the caller is
    responsible for skipping ``None`` lines.
    """
    if ann.obj_class not in class_map:
        return None
    class_id = class_map[ann.obj_class]
    cx, cy, w, h = bbox_to_yolo(ann.x1, ann.y1, ann.x2, ann.y2, img_w, img_h)
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def frame_yolo_labels(
    seq: KittiSequence,
    frame: int,
    class_map: dict[str, int],
    img_w: int,
    img_h: int,
) -> list[str]:
    """All YOLO label lines for a single frame, with unmapped classes filtered out."""
    lines: list[str] = []
    for ann in seq.annotations_for_frame(frame):
        line = annotation_to_yolo_line(ann, class_map, img_w, img_h)
        if line is not None:
            lines.append(line)
    return lines


def _read_image_size(seq: KittiSequence) -> tuple[int, int]:
    """Read ``(width, height)`` from the first frame and verify consistency.

    KITTI's documented invariant is that image dimensions are constant
    within a sequence. We verify this on a sample of frames (first,
    middle, last) to fail loudly if the invariant is ever violated,
    rather than silently producing wrong labels for the affected frames.
    """
    frames = seq.frames()
    if not frames:
        raise ValueError(f"Sequence {seq.name} has no frames in {seq.image_dir}")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise ValueError(f"Could not read first frame: {frames[0]}")
    h, w = first.shape[:2]

    # Spot-check middle and last frame; skip duplicates (e.g. 1- or 2-frame seqs).
    check_indices = {len(frames) // 2, len(frames) - 1} - {0}
    for idx in check_indices:
        check_img = cv2.imread(str(frames[idx]))
        if check_img is None:
            raise ValueError(f"Could not read frame: {frames[idx]}")
        ch, cw = check_img.shape[:2]
        if (cw, ch) != (w, h):
            raise ValueError(
                f"Sequence {seq.name} has inconsistent image dimensions: "
                f"frame 0 is {w}x{h} but frame {idx} is {cw}x{ch}. "
                f"This violates the KITTI single-resolution-per-sequence "
                f"invariant. Investigate before proceeding."
            )

    return (w, h)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink ``src`` to ``dst`` if possible, else copy.

    Hardlinks are zero-cost on disk and work on Windows NTFS without
    admin/dev mode (unlike symlinks). Cross-volume hardlinks fail, in
    which case we fall back to a regular copy.
    """
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_yolo_labels(
    sequences: list[KittiSequence],
    out_dir: Path,
    class_map: dict[str, int] = KITTI_TO_YOLO_ZEROSHOT,
    image_size: tuple[int, int] | None = None,
) -> int:
    """Write per-frame YOLO ``.txt`` files under ``out_dir/<seq>/<frame>.txt``.

    Every frame in every sequence gets a file (empty when no objects pass
    the class filter). Returns the total number of label lines written
    across all frames — useful for sanity-checking against
    :meth:`KittiTrackingDataset.total_annotations`.

    When ``image_size`` is ``None`` (production), dimensions are read
    from each sequence's frames on disk via :func:`_read_image_size`,
    which also enforces the within-sequence consistency invariant.
    Pass an explicit ``(w, h)`` to skip the disk read — useful for
    fixture-only tests where frames are placeholders rather than real
    PNGs.
    """
    total_lines = 0
    for seq in sequences:
        img_w, img_h = image_size if image_size is not None else _read_image_size(seq)
        seq_out = out_dir / seq.name
        seq_out.mkdir(parents=True, exist_ok=True)
        for frame in range(seq.num_frames):
            lines = frame_yolo_labels(seq, frame, class_map, img_w, img_h)
            (seq_out / f"{frame:06d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            total_lines += len(lines)
    return total_lines


def prepare_yolo_eval_dataset(
    sequences: list[KittiSequence],
    out_dir: Path,
    class_map: dict[str, int] = KITTI_TO_COCO_ZEROSHOT,
    class_names: list[str] | None = None,
    split: str = "val",
) -> Path:
    """Materialise sequences as a YOLOv8 dataset for ``ultralytics`` eval.

    Layout produced::

        out_dir/
        |- data.yaml
        |- <split>.txt              # absolute image paths, one per line
        |- images/<split>/<seq>_<frame>.png   # hardlinks (or copies)
        |- labels/<split>/<seq>_<frame>.txt   # YOLO-format labels

    Image dimensions are read per-sequence via :func:`_read_image_size`,
    which fails loudly if the within-sequence consistency invariant is
    ever violated.

    Returns the path to ``data.yaml``, which is what ``ultralytics`` wants.

    Examples:
        Zero-shot eval against a COCO-pretrained model (T3b)::

            prepare_yolo_eval_dataset(sequences, Path("data/yolo_eval"))

        The COCO preset is auto-recognised — no need to pass ``class_names``.

        Fine-tune eval against a KITTI-trained 2-class model (T4)::

            prepare_yolo_eval_dataset(
                sequences, Path("data/yolo_finetune_eval"),
                class_map=KITTI_TO_YOLO_FINETUNE,
            )

        Contiguous indices auto-derive their names from ``class_map`` keys.

        Explicit class space (override path)::

            prepare_yolo_eval_dataset(
                sequences, Path("data/custom_eval"),
                class_map={"Truck": 0, "Car": 1},
                class_names=["Truck", "Car"],
            )

        Sparse custom map without ``class_names`` raises ``ValueError`` to
        prevent silent namespace drift in ``ultralytics``' eval pipeline.
    """
    out_dir = out_dir.resolve()
    images_dir = out_dir / "images" / split
    labels_dir = out_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    for seq in sequences:
        img_w, img_h = _read_image_size(seq)
        for frame in range(seq.num_frames):
            src = seq.image_dir / f"{frame:06d}.png"
            if not src.exists():
                continue

            stem = f"{seq.name}_{frame:06d}"
            target_img = images_dir / f"{stem}.png"
            _link_or_copy(src, target_img)
            image_paths.append(target_img)

            lines = frame_yolo_labels(seq, frame, class_map, img_w, img_h)
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    split_txt = out_dir / f"{split}.txt"
    split_txt.write_text("\n".join(str(p.resolve()) for p in image_paths) + "\n")

    if class_names is None:
        # Recognise the COCO zero-shot preset by identity (the imported
        # constant or its default-arg value) so bare-default calls work
        # end-to-end. Identity-only — anyone reconstructing the dict
        # literally has explicitly chosen non-default behaviour and should
        # pass class_names themselves.
        if class_map is KITTI_TO_COCO_ZEROSHOT:
            class_names = COCO80_NAMES
        else:
            # Auto-derive a contiguous names list from class_map. Works
            # only when indices are contiguous starting at 0 (e.g., Car=0,
            # Pedestrian=1 for fine-tune). Sparse custom maps must pass
            # class_names explicitly.
            id_to_name = {v: k for k, v in class_map.items()}
            max_id = max(id_to_name)
            if set(id_to_name.keys()) != set(range(max_id + 1)):
                raise ValueError(
                    f"class_map indices {sorted(id_to_name)} are not "
                    f"contiguous starting at 0; pass class_names explicitly. "
                    f"For zero-shot eval against a COCO-pretrained model, "
                    f"use the KITTI_TO_COCO_ZEROSHOT preset (auto-recognised) "
                    f"or pass COCO80_NAMES directly."
                )
            class_names = [id_to_name[i] for i in range(max_id + 1)]

    # ultralytics' check_det_dataset requires BOTH 'train' and 'val' keys in
    # data.yaml regardless of which mode is being run — see
    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/utils.py
    # We always emit both; the inactive key points at the same image list as a
    # no-op (ultralytics only reads the key matching the current mode). T4 will
    # extend this to generate distinct train.txt and val.txt when fine-tuning.
    split_path = f"{split}.txt"
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir),
                "train": split_path,
                "val": split_path,
                "nc": len(class_names),
                "names": class_names,
            },
            sort_keys=False,
        )
    )

    return data_yaml


# ---------------------------------------------------------------------------
# T4 fine-tune dataset preparation (5-class)
# ---------------------------------------------------------------------------

# Sequence-level train/val split — matches the tracker val set from CLAUDE.md.
# Never split frames within a sequence; frame leakage inflates HOTA by 5-10 pts.
_DEFAULT_VAL_SEQUENCES: list[str] = ["0001", "0006", "0013", "0017", "0019"]
_DEFAULT_TRAIN_SEQUENCES: list[str] = [
    "0000",
    "0002",
    "0003",
    "0004",
    "0005",
    "0007",
    "0008",
    "0009",
    "0010",
    "0011",
    "0012",
    "0014",
    "0015",
    "0016",
    "0018",
    "0020",
]


def _clip_and_validate_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float] | None:
    """Clip bbox to image bounds. Returns None if degenerate (w or h <= 1px)."""
    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))
    if (x2 - x1) <= 1.0 or (y2 - y1) <= 1.0:
        return None
    return x1, y1, x2, y2


def prepare_kitti_yolo_finetune(
    data_root: Path,
    out_dir: Path,
    class_map: dict[str, int] = KITTI_TO_YOLO_5CLASS,
    val_sequences: list[str] | None = None,
    train_sequences: list[str] | None = None,
) -> dict:
    """Prepare KITTI MOT data in YOLO format for fine-tuning.

    Walks all sequences, splits into train/val by sequence name, writes::

        out_dir/
        ├── images/{train,val}/<seq>_<frame>.png
        ├── labels/{train,val}/<seq>_<frame>.txt
        └── data.yaml

    Images are hardlinked (or copied on cross-volume) via :func:`_link_or_copy`.
    Labels use the standard YOLO format: ``class_id cx cy w h`` normalised
    to [0, 1]. Bboxes are clipped to image bounds; degenerate boxes (width
    or height ≤ 1px after clipping) are silently skipped. Frames with no
    surviving labels produce empty .txt files (negative examples for YOLO).

    Args:
        data_root: Path to ``data/kitti_tracking``.
        out_dir: Output directory (e.g. ``data/kitti_yolo``).
        class_map: KITTI class name -> YOLO class id. Only classes present
                   in this map are emitted; all others are excluded.
        val_sequences: Sequence names for val split. Defaults to project
                       val set (0001, 0006, 0013, 0017, 0019).
        train_sequences: Sequence names for train split. Defaults to all
                         other 16 sequences.

    Returns:
        Stats dict with keys: ``train_images``, ``val_images``,
        ``train_labels``, ``val_labels``, ``train_classes`` (per-class counts),
        ``val_classes`` (per-class counts), ``skipped_degenerate``.
    """
    if val_sequences is None:
        val_sequences = _DEFAULT_VAL_SEQUENCES
    if train_sequences is None:
        train_sequences = _DEFAULT_TRAIN_SEQUENCES

    val_set = set(val_sequences)
    train_set = set(train_sequences)
    if val_set & train_set:
        raise ValueError(f"Train/val overlap: {val_set & train_set}")

    all_classes = tuple(class_map.keys())
    class_names = [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])]

    stats: dict = {
        "train_images": 0,
        "val_images": 0,
        "train_labels": 0,
        "val_labels": 0,
        "train_classes": {name: 0 for name in class_names},
        "val_classes": {name: 0 for name in class_names},
        "skipped_degenerate": 0,
    }

    for split_name, seq_names in [("train", train_sequences), ("val", val_sequences)]:
        images_dir = out_dir / "images" / split_name
        labels_dir = out_dir / "labels" / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for seq_name in seq_names:
            seq = load_sequence(data_root, seq_name, classes=all_classes)
            img_w, img_h = _read_image_size(seq)
            logger.info(
                "Processing %s/%s (%dx%d, %d frames)",
                split_name,
                seq_name,
                img_w,
                img_h,
                seq.num_frames,
            )

            for frame_idx, frame_path in enumerate(seq.frames()):
                stem = f"{seq_name}_{frame_idx:06d}"

                # Link/copy image
                target_img = images_dir / f"{stem}.png"
                _link_or_copy(frame_path, target_img)
                stats[f"{split_name}_images"] += 1

                # Build labels for this frame
                anns = seq.annotations_for_frame(frame_idx)
                lines: list[str] = []
                for ann in anns:
                    if ann.obj_class not in class_map:
                        continue
                    clipped = _clip_and_validate_bbox(
                        ann.x1,
                        ann.y1,
                        ann.x2,
                        ann.y2,
                        img_w,
                        img_h,
                    )
                    if clipped is None:
                        stats["skipped_degenerate"] += 1
                        continue
                    cx, cy, w, h = bbox_to_yolo(*clipped, img_w, img_h)
                    class_id = class_map[ann.obj_class]
                    lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    stats[f"{split_name}_classes"][ann.obj_class] += 1

                # Write label file (empty for negative examples)
                label_path = labels_dir / f"{stem}.txt"
                label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
                stats[f"{split_name}_labels"] += len(lines)

        logger.info(
            "%s split: %d images, %d label lines",
            split_name,
            stats[f"{split_name}_images"],
            stats[f"{split_name}_labels"],
        )

    # Write data.yaml
    data_yaml_path = out_dir / "data.yaml"
    data_yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "nc": len(class_names),
                "names": class_names,
            },
            sort_keys=False,
        )
    )
    logger.info("Wrote %s", data_yaml_path)

    return stats
