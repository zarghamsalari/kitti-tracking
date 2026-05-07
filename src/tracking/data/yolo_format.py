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
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import yaml

from tracking.data.kitti import KittiAnnotation, KittiSequence

KITTI_TO_YOLO_ZEROSHOT: dict[str, int] = {
    "Car": 0,
    "Pedestrian": 1,
}

# Canonical KITTI image size — actual dims are read per-sequence in
# write_yolo_labels (see _read_image_size). Kept here for fixtures, docs,
# and the optional override path of write_yolo_labels.
KITTI_IMAGE_SIZE: tuple[int, int] = (1242, 375)


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
    class_map: dict[str, int] = KITTI_TO_YOLO_ZEROSHOT,
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
        id_to_name = {v: k for k, v in class_map.items()}
        class_names = [id_to_name[i] for i in sorted(id_to_name)]

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
