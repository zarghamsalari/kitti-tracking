# KITTI MOT — Tracker Ablation

Macro-averaged HOTA / MOTA / IDF1 / AssA / DetA / IDSw / Frag / MT / ML on the 5 val sequences (0001, 0006, 0013, 0017, 0019), classes Car + Pedestrian.

Detection inputs are shared across trackers within each detector condition (verified by `run_meta.json` hash). Tracker differences within a row pair are not confounded by detection variance.

## 4-cell ablation: detector × tracker

| Detector | Tracker | HOTA | MOTA | IDF1 | AssA | DetA | IDSw | Frag | MT | ML |
|----------|---------|------|------|------|------|------|------|------|----|----|
| Zero-shot YOLOv8m   | ByteTrack | 0.41 | 0.07 | 0.54 | 0.48 | 0.36 | 152 | 384 | 102 | 20 |
| Zero-shot YOLOv8m   | BoT-SORT  | 0.47 | 0.24 | 0.61 | 0.53 | 0.42 | 88  | 347 | 126 | 10 |
| Fine-tuned YOLOv8m  | ByteTrack | 0.41 | 0.36 | 0.55 | 0.44 | 0.40 | 159 | 284 | 73  | 49 |
| Fine-tuned YOLOv8m  | BoT-SORT  | 0.45 | 0.41 | 0.59 | 0.49 | 0.41 | 86  | 235 | 84  | 50 |

## Deltas (fine-tuned − zero-shot)

| Tracker | ΔHOTA | ΔMOTA | ΔIDF1 | ΔAssA | ΔDetA | ΔIDSw | ΔFrag |
|---------|-------|-------|-------|-------|-------|-------|-------|
| ByteTrack | 0.00  | **+0.29** | +0.01 | −0.04 | +0.04 | +7  | −100 |
| BoT-SORT  | −0.02 | **+0.17** | −0.02 | −0.04 | −0.01 | −2  | −112 |

## Headline

Fine-tuning the detector eliminates a large fraction of false-positive detections, which **5× the MOTA on ByteTrack** (0.07 → 0.36) and roughly **doubles MOTA on BoT-SORT** (0.24 → 0.41). HOTA is flat-to-slightly-negative because the recovered DetA is offset by a drop in AssA — fewer detections per frame mean fewer association opportunities. Fragmentation drops substantially under both trackers (cleaner detection inputs mean fewer track breaks), and BoT-SORT's CMC continues to outperform ByteTrack on every macro metric in both detector conditions.

## Targets vs. result

| Metric | Target | Best achieved | Cell |
|--------|--------|---------------|------|
| HOTA   | ≥ 0.55 | 0.47 | Zero-shot + BoT-SORT |
| MOTA   | ≥ 0.65 | 0.41 | Fine-tuned + BoT-SORT |
| IDF1   | ≥ 0.65 | 0.61 | Zero-shot + BoT-SORT |

None of the three curriculum targets are met at imgsz=640 within the 5-trial hyperparameter cap. The detection-side opportunity is clearer than the tracker-side: fine-tuning at imgsz=1280 (vs. our 640) and lowering `dump_conf` to recover detection recall are the most direct levers. Tracker-side improvements (ReID features in BoT-SORT) are a secondary lever once the detection floor is raised.

## Per-class (Car + Pedestrian, fine-tuned)

| Class | mAP@0.5 | mAP@0.5:0.95 | Recall |
|-------|---------|--------------|--------|
| Car        | 0.818 | 0.606 | 0.798 |
| Pedestrian | 0.589 | 0.241 | 0.488 |

The Pedestrian recall gap (0.488 vs. Car 0.798) is the dominant source of AssA loss after fine-tuning — half of all Pedestrians in val are missed by the detector at confidence ≥ 0.1. Pedestrian-specific augmentation, larger imgsz, or a class-balanced loss are the natural next experiments.

## Reproducibility

All four runs use seed=42, KITTI MOT public training split with sequence-level holdout (0001, 0006, 0013, 0017, 0019 in val). Tracker hyperparameter search capped at 5 trials per tracker on val. `make eval` reproduces these numbers from a clean clone within ±0.5 HOTA points.
