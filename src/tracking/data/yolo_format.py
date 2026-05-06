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

from pathlib import Path

from tracking.data.kitti import KittiAnnotation, KittiSequence

KITTI_TO_YOLO_ZEROSHOT: dict[str, int] = {
    "Car": 0,
    "Pedestrian": 1,
}

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


def write_yolo_labels(
    sequences: list[KittiSequence],
    out_dir: Path,
    class_map: dict[str, int] = KITTI_TO_YOLO_ZEROSHOT,
    image_size: tuple[int, int] = KITTI_IMAGE_SIZE,
) -> int:
    """Write per-frame YOLO ``.txt`` files under ``out_dir/<seq>/<frame>.txt``.

    Every frame in every sequence gets a file (empty when no objects pass
    the class filter). Returns the total number of label lines written
    across all frames — useful for sanity-checking against
    :meth:`KittiTrackingDataset.total_annotations`.
    """
    img_w, img_h = image_size
    total_lines = 0
    for seq in sequences:
        seq_out = out_dir / seq.name
        seq_out.mkdir(parents=True, exist_ok=True)
        for frame in range(seq.num_frames):
            lines = frame_yolo_labels(seq, frame, class_map, img_w, img_h)
            (seq_out / f"{frame:06d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            total_lines += len(lines)
    return total_lines
