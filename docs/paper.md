# Multi-Object Tracking on KITTI

> Public-dataset portfolio project. Code: github.com/USER/kitti-tracking. Demo: TBD.

## 1. Problem

Multi-object tracking (MOT) on driving scenes — given a video stream, assign a stable identity to every detected object across frames. KITTI 2D MOT is the canonical benchmark for autonomous-driving tracking research.

## 2. Dataset

KITTI tracking benchmark, training split only (21 sequences with public GT). Sequence-level holdout:

- **Train (16):** 0000, 0002–0005, 0007–0012, 0014–0016, 0018, 0020
- **Val (5):** 0001, 0006, 0013, 0017, 0019

Classes: Car, Pedestrian (Cyclist tracked but not headline-reported — too few instances for a stable HOTA estimate).

## 3. Method

Two-stage pipeline:

1. **Detection.** YOLOv8m, COCO pretrained, fine-tuned on KITTI training sequences for 50 epochs at imgsz=1280.
2. **Association.** Two trackers consume the same detection file:
   - **ByteTrack** — IoU + Kalman filter, two-stage matching (high-conf then low-conf).
   - **BoT-SORT** — adds camera motion compensation + OSNet ReID embeddings.

Detector held constant across the ablation. The only variable is association quality.

## 4. Results

(Auto-rendered in `docs/results.md` by `make eval`. Headline numbers below.)

| Tracker   | HOTA | MOTA | IDF1 | AssA | DetA | IDSw |
|-----------|------|------|------|------|------|------|
| ByteTrack | TBD  | TBD  | TBD  | TBD  | TBD  | TBD  |
| BoT-SORT  | TBD  | TBD  | TBD  | TBD  | TBD  | TBD  |

## 5. Discussion

### Where each tracker breaks
*(fill after T7)*

- Heavy occlusion (sequence 0017, around frame …)
- Crowded pedestrian scenes (sequence 0006)
- Fast camera motion (sequence 0013)

### Det vs Ass decomposition
HOTA = √(DetA × AssA). Most of the gap between trackers shows up in AssA — DetA is fixed by the shared detector. Quantify the AssA delta and read it as the headline of the experiment.

### Honest limitations
- Validation split is 5 sequences from KITTI training data — small. Numbers should be read with ±1 HOTA uncertainty.
- Only ≤5 hyperparameter trials per tracker on val. No grid search.
- Did not run on the held-back KITTI test server.

## 6. Reproducibility

```bash
git clone <repo>
cd kitti-tracking
make install
make download
make eval
```

Run metadata (git SHA, config hash, package versions, seed) is dumped to `runs/*/run_meta.json` for each stage.

## 7. References

- ByteTrack: Zhang et al., ECCV 2022.
- BoT-SORT: Aharon et al., 2022.
- HOTA: Luiten et al., IJCV 2021.
- KITTI: Geiger et al., CVPR 2012.
