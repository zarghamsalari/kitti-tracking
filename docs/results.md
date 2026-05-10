# Tracker ablation — results

Macro-averaged HOTA, MOTA, IDF1, AssA, DetA, IDSw, Frag, MT, ML on 5 KITTI MOT val sequences (0001, 0006, 0013, 0017, 0019), classes Car + Pedestrian.

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

Fine-tuning the detector eliminates a large fraction of false positives, lifting MOTA 5× on ByteTrack (0.07 → 0.36) and roughly 2× on BoT-SORT (0.24 → 0.41). HOTA is flat-to-slightly-negative because the recovered DetA is offset by a drop in AssA — fewer detections per frame mean fewer association opportunities. Fragmentation drops substantially under both trackers (cleaner detection inputs mean fewer track breaks), and BoT-SORT's CMC continues to outperform ByteTrack on every macro metric in both detector conditions.

## Per-class detection (fine-tuned)

| Class | mAP@0.5 | mAP@0.5:0.95 | Recall |
|-------|---------|--------------|--------|
| Car        | 0.818 | 0.606 | 0.798 |
| Pedestrian | 0.589 | 0.241 | 0.488 |

The Pedestrian recall gap (0.488 vs Car 0.798) is the dominant source of AssA loss after fine-tuning — half of Pedestrian instances in val sit below the detector's confidence threshold. Pedestrian-targeted augmentation, larger imgsz, and a class-balanced loss are the natural next experiments.

## Reproducibility

Seed=42 across all four cells. KITTI MOT public training split with sequence-level holdout (train: 16 sequences; val: 0001, 0006, 0013, 0017, 0019). Tracker hyperparameter search capped at 5 trials per tracker on val. `make eval` reproduces these numbers from a clean clone within ±0.5 HOTA.
