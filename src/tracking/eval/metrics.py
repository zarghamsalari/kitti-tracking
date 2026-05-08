"""Tracking metrics — task T6.

Primary metric: HOTA. Decomposed into AssA and DetA. MOTA + IDF1 alongside.
IDSw and Frag for narrative. All computed via TrackEval 1.3.0.

TrackEval metric keys verified against source on 2026-05-08:
  HOTA.eval_sequence / combine_sequences:
    Arrays (19,) over alpha levels: HOTA, DetA, AssA, DetRe, DetPr, AssRe, AssPr, LocA, OWTA
    Scalars: HOTA(0), LocA(0), HOTALocA(0)
    Input data keys: num_tracker_dets, num_gt_dets, num_gt_ids, num_tracker_ids,
                     gt_ids, tracker_ids, similarity_scores

  CLEAR.eval_sequence / combine_sequences:
    Scalars: MOTA, MOTP, MODA, CLR_Re, CLR_Pr, MTR, PTR, MLR, sMOTA, CLR_F1,
             FP_per_frame, MOTAL, MOTP_sum (post-combine: without MOTP_sum)
    Integers: CLR_TP, CLR_FN, CLR_FP, IDSW, MT, PT, ML, Frag, CLR_Frames
    Input data keys: num_tracker_dets, num_gt_dets, num_gt_ids, num_timesteps,
                     gt_ids, tracker_ids, similarity_scores
    Note: CLEAR is the only metric requiring num_timesteps in the data dict.

  Identity.eval_sequence / combine_sequences:
    Scalars: IDF1, IDR, IDP
    Integers: IDTP, IDFN, IDFP
    Input data keys: num_tracker_dets, num_gt_dets, num_gt_ids, num_tracker_ids,
                     gt_ids, tracker_ids, similarity_scores

Aggregation convention:
  rate-like metrics (HOTA, MOTA, IDF1, AssA, DetA): reported twice
    - macro: equal weight per class
    - weighted: weighted by GT instance count per class (matches published KITTI baselines)
  count-like metrics (IDSw, Frag, MT, ML): summed across classes (no weighting choice)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# (class_name, integer_cls_id) pairs — must match KITTI label strings and our cls encoding
_KITTI_EVAL_CLASSES: tuple[tuple[str, int], ...] = (("Car", 0), ("Pedestrian", 1))

# TrackEval key names verified against trackeval==1.3.0 source (2026-05-08).
# Update this comment if trackeval is upgraded and keys change.
_HOTA_ALPHA_KEYS = ("HOTA", "AssA", "DetA")   # np.ndarray shape (19,) over alpha
_CLEAR_RATE_KEYS = ("MOTA",)                   # scalar float, macro-averaged
_CLEAR_COUNT_KEYS = ("IDSW", "Frag", "MT", "ML")  # int, summed across classes
_IDENTITY_RATE_KEYS = ("IDF1",)                # scalar float, macro-averaged


@dataclass
class TrackerMetrics:
    """Per-tracker aggregated metrics."""

    name: str
    hota: float
    mota: float
    idf1: float
    assa: float
    deta: float
    id_sw: int
    frag: int
    mt: int  # mostly tracked
    ml: int  # mostly lost


@dataclass
class EvalResults:
    """Container for one or more trackers' metrics."""

    trackers: list[TrackerMetrics] = field(default_factory=list)

    def to_markdown(self) -> str:
        note = (
            "<!-- Rate metrics (*-macro): equal class weights. "
            "(*-weighted): GT-instance weights (matches published KITTI baselines). "
            "Count metrics (IDSw/Frag/MT/ML): summed across classes. -->\n\n"
        )
        header = (
            "| Tracker | HOTA | MOTA | IDF1 | AssA | DetA | IDSw | Frag | MT | ML |\n"
            "|---------|------|------|------|------|------|------|------|----|----|\n"
        )
        rows = "\n".join(
            f"| {t.name} | {t.hota:.2f} | {t.mota:.2f} | {t.idf1:.2f} | "
            f"{t.assa:.2f} | {t.deta:.2f} | {t.id_sw} | {t.frag} | {t.mt} | {t.ml} |"
            for t in self.trackers
        )
        return f"# Tracking Results\n\n{note}{header}{rows}\n"


def _iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """Compute N×M IoU matrix for xyxy boxes. Pure numpy, no GPU dependency.

    Args:
        gt_boxes:   (N, 4) float32 xyxy
        pred_boxes: (M, 4) float32 xyxy

    Returns:
        (N, M) float32 IoU values. Returns shape (N, 0) or (0, M) when
        either input is empty — TrackEval handles zero-dimension arrays.
    """
    n = gt_boxes.shape[0]
    m = pred_boxes.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    # Broadcast: gt (N,1,4), pred (1,M,4)
    gt = gt_boxes[:, np.newaxis, :]    # (N, 1, 4)
    pred = pred_boxes[np.newaxis, :, :]  # (1, M, 4)

    inter_x1 = np.maximum(gt[..., 0], pred[..., 0])
    inter_y1 = np.maximum(gt[..., 1], pred[..., 1])
    inter_x2 = np.minimum(gt[..., 2], pred[..., 2])
    inter_y2 = np.minimum(gt[..., 3], pred[..., 3])

    inter_w = np.maximum(inter_x2 - inter_x1, 0.0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h  # (N, M)

    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])

    union_area = gt_area[:, np.newaxis] + pred_area[np.newaxis, :] - inter_area
    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou.astype(np.float32)


def _build_sequence_data(
    gt_annots: list[Any],  # list[KittiAnnotation] — typed as Any to avoid circular import
    pred_by_frame: dict[int, np.ndarray],
    num_frames: int,
    cls_id: int,
) -> dict[str, Any]:
    """Build a TrackEval data_dict for one (sequence, class) pair.

    Args:
        gt_annots:      All KittiAnnotation objects for this sequence (any class).
                        Filtered here to cls_id.
        pred_by_frame:  {frame: (M, 7) [x1,y1,x2,y2,track_id,conf,cls]} from
                        read_mot16_v2_tracks. Missing frames mean no predictions.
                        If pred_by_frame.get(t) is missing, treated as 0 predictions.
        num_frames:     Total number of frames in the sequence.
        cls_id:         Integer class id (Car=0, Pedestrian=1).

    Returns:
        data dict compatible with HOTA, CLEAR, and Identity metric classes.
    """
    # Filter GT to this class
    cls_gt = [a for a in gt_annots if _kitti_class_to_id(a.obj_class) == cls_id]

    # Build per-frame lookup from GT
    gt_by_frame: dict[int, list[Any]] = {}
    for a in cls_gt:
        gt_by_frame.setdefault(a.frame, []).append(a)

    gt_ids_list: list[np.ndarray] = []
    tracker_ids_list: list[np.ndarray] = []
    similarity_scores_list: list[np.ndarray] = []
    num_gt_dets = 0
    num_tracker_dets = 0
    all_gt_ids: set[int] = set()
    all_tracker_ids: set[int] = set()

    for t in range(num_frames):
        # GT for this frame
        frame_gt = gt_by_frame.get(t, [])
        gt_ids_t = np.array([a.track_id for a in frame_gt], dtype=np.int32)
        gt_boxes_t = np.array(
            [[a.x1, a.y1, a.x2, a.y2] for a in frame_gt], dtype=np.float32
        ).reshape(-1, 4)

        # Predictions for this frame
        frame_pred = pred_by_frame.get(t)
        if frame_pred is not None and frame_pred.shape[0] > 0:
            cls_mask = frame_pred[:, 6] == cls_id
            frame_pred_cls = frame_pred[cls_mask]
        else:
            frame_pred_cls = np.empty((0, 7), dtype=np.float32)

        tracker_ids_t = frame_pred_cls[:, 4].astype(np.int32)
        pred_boxes_t = frame_pred_cls[:, :4]

        # Accumulate
        all_gt_ids.update(gt_ids_t.tolist())
        all_tracker_ids.update(tracker_ids_t.tolist())
        num_gt_dets += len(gt_ids_t)
        num_tracker_dets += len(tracker_ids_t)

        gt_ids_list.append(gt_ids_t)
        tracker_ids_list.append(tracker_ids_t)
        similarity_scores_list.append(_iou_matrix(gt_boxes_t, pred_boxes_t))

    return {
        "num_timesteps": num_frames,
        "num_gt_ids": len(all_gt_ids),
        "num_tracker_ids": len(all_tracker_ids),
        "num_gt_dets": num_gt_dets,
        "num_tracker_dets": num_tracker_dets,
        "gt_ids": gt_ids_list,
        "tracker_ids": tracker_ids_list,
        "similarity_scores": similarity_scores_list,
    }


def _kitti_class_to_id(obj_class: str) -> int:
    """Map KITTI string class to integer id. Returns -1 for unknown classes."""
    mapping = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
    return mapping.get(obj_class, -1)


def evaluate(gt_dir: Path, pred_dir: Path) -> EvalResults:
    """Compute HOTA / MOTA / IDF1 / IDSw across val sequences.

    Reads val_sequences and tracker_name from pred_dir/run_meta.json.
    Evaluates per class (Car, Pedestrian), then produces two TrackerMetrics rows:
      - <name>-macro:    equal class weight (balances Car vs Pedestrian quality)
      - <name>-weighted: instance-weighted (matches published KITTI baselines)

    Args:
        gt_dir:   Path to data/kitti_tracking/training/label_02
        pred_dir: Path to runs/track/<tracker>/
    """
    from trackeval.metrics import CLEAR, HOTA, Identity

    from tracking.data.kitti import parse_label_file
    from tracking.trackers.mot16_io import read_mot16_v2_tracks

    meta = json.loads((pred_dir / "run_meta.json").read_text())
    sequences: list[str] = meta["val_sequences"]
    tracker_name: str = meta.get("tracker_name") or pred_dir.name

    # Per-class results: {cls_name: {seq: result_dict}}
    class_results: dict[str, dict] = {}
    class_gt_counts: dict[str, int] = {}

    for cls_name, cls_id in _KITTI_EVAL_CLASSES:
        hota_metric = HOTA()
        clear_metric = CLEAR()
        id_metric = Identity()
        seq_hota: dict[str, Any] = {}
        seq_clear: dict[str, Any] = {}
        seq_id: dict[str, Any] = {}
        total_gt = 0

        for seq in sequences:
            logger.info("Evaluating %s / %s ...", cls_name, seq)
            gt_annots = parse_label_file(gt_dir / f"{seq}.txt", classes=(cls_name,))
            pred_path = pred_dir / f"{seq}.txt"
            pred_by_frame = read_mot16_v2_tracks(pred_path)

            all_frames = [a.frame for a in gt_annots]
            num_frames = (max(all_frames) + 1) if all_frames else 1
            total_gt += len(gt_annots)

            data = _build_sequence_data(gt_annots, pred_by_frame, num_frames, cls_id)
            seq_hota[seq] = hota_metric.eval_sequence(data)
            seq_clear[seq] = clear_metric.eval_sequence(data)
            seq_id[seq] = id_metric.eval_sequence(data)

        class_results[cls_name] = {
            "hota": hota_metric.combine_sequences(seq_hota),
            "clear": clear_metric.combine_sequences(seq_clear),
            "id": id_metric.combine_sequences(seq_id),
        }
        class_gt_counts[cls_name] = total_gt

    total_gt_all = sum(class_gt_counts.values())
    classes = [c for c, _ in _KITTI_EVAL_CLASSES]

    def _macro_mean_alpha(key: str, sub: str) -> float:
        return float(np.mean([np.mean(class_results[c][sub][key]) for c in classes]))

    def _weighted_mean_alpha(key: str, sub: str) -> float:
        w = np.array([class_gt_counts[c] for c in classes], dtype=float)
        w /= w.sum()
        return float(sum(w[i] * float(np.mean(class_results[c][sub][key]))
                         for i, c in enumerate(classes)))

    def _macro_mean_scalar(key: str, sub: str) -> float:
        return float(np.mean([class_results[c][sub][key] for c in classes]))

    def _weighted_mean_scalar(key: str, sub: str) -> float:
        w = np.array([class_gt_counts[c] for c in classes], dtype=float)
        w /= w.sum()
        return float(sum(w[i] * float(class_results[c][sub][key])
                         for i, c in enumerate(classes)))

    def _sum_counts(key: str, sub: str) -> int:
        return int(sum(class_results[c][sub][key] for c in classes))

    shared_counts = {
        "id_sw": _sum_counts("IDSW", "clear"),
        "frag": _sum_counts("Frag", "clear"),
        "mt": _sum_counts("MT", "clear"),
        "ml": _sum_counts("ML", "clear"),
    }

    tm_macro = TrackerMetrics(
        name=f"{tracker_name}-macro",
        hota=_macro_mean_alpha("HOTA", "hota"),
        mota=_macro_mean_scalar("MOTA", "clear"),
        idf1=_macro_mean_scalar("IDF1", "id"),
        assa=_macro_mean_alpha("AssA", "hota"),
        deta=_macro_mean_alpha("DetA", "hota"),
        **shared_counts,
    )
    tm_weighted = TrackerMetrics(
        name=f"{tracker_name}-weighted",
        hota=_weighted_mean_alpha("HOTA", "hota"),
        mota=_weighted_mean_scalar("MOTA", "clear"),
        idf1=_weighted_mean_scalar("IDF1", "id"),
        assa=_weighted_mean_alpha("AssA", "hota"),
        deta=_weighted_mean_alpha("DetA", "hota"),
        **shared_counts,
    )

    logger.info(
        "%s macro:    HOTA=%.2f  MOTA=%.2f  IDF1=%.2f",
        tracker_name, tm_macro.hota, tm_macro.mota, tm_macro.idf1,
    )
    logger.info(
        "%s weighted: HOTA=%.2f  MOTA=%.2f  IDF1=%.2f",
        tracker_name, tm_weighted.hota, tm_weighted.mota, tm_weighted.idf1,
    )
    return EvalResults(trackers=[tm_macro, tm_weighted])
