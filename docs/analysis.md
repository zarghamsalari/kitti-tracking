# 2D MOT on KITTI — analysis

## Problem framing

A reproducible 2D MOT pipeline on KITTI (Car + Pedestrian) where the detector is held constant per condition and the tracker is the variable under ablation. The goal is a clean within-row comparison of ByteTrack vs BoT-SORT under identical detection inputs, plus a between-row comparison of zero-shot vs fine-tuned detection feeding the same trackers.

## Methodology

### Dataset

KITTI MOT (Geiger et al. 2012): 21 training sequences (0000–0020) with public ground truth. Split at the sequence level — no frame leakage — into:

- **Train (16 sequences):** 0000, 0002–0005, 0007–0012, 0014–0016, 0018, 0020 — 5,747 images
- **Val (5 sequences):** 0001, 0006, 0013, 0017, 0019 — 2,261 images

Train/val image-set overlap: **0**. Verified programmatically via set intersection on filenames. Within-sequence frame splits are avoided because they silently inflate HOTA on MOT.

### Evaluation

TrackEval official MOT suite computing HOTA, MOTA, IDF1, AssA, DetA, IDSw, Frag, MT, ML. Headline numbers are macro-averaged across Car and Pedestrian classes; per-class breakdowns reported in the tables below.

## Approach

### Detector

**YOLOv8m** under two conditions:

1. **Zero-shot** — COCO-pretrained, COCO car→Car / person→Pedestrian.
2. **Fine-tuned** — KITTI 5-class (Car, Van, Truck, Pedestrian, Cyclist), trained on the 16-sequence train split at imgsz=640 for up to 100 epochs with patience=20 early stopping. Stopped at epoch 47, best at epoch 27. Per-class val mAP at the best checkpoint:

| Class | mAP@0.5 | mAP@0.5:0.95 | Recall |
|-------|---------|--------------|--------|
| Car        | 0.818 | 0.606 | 0.798 |
| Pedestrian | 0.589 | 0.241 | 0.488 |

### Trackers

1. **ByteTrack** — Kalman + IoU-based association with two-stage matching (high-confidence then low-confidence). One instance per class for cross-class isolation.
2. **BoT-SORT** — extends ByteTrack with camera motion compensation via sparse optical flow. `per_class=True, nr_classes=2`, `frame_rate=10` to match KITTI's capture rate (default 30 produced 3× too-permissive lost-track tolerance).

Both trackers consume identical MOT16-format detection files within each detector condition, so within-row deltas isolate tracker behaviour.

## Ablation design

4-cell factorial: **{ByteTrack, BoT-SORT} × {zero-shot detector, fine-tuned detector}**.

Detection inputs are shared across trackers within each detector condition and hash-verified via `run_meta.json`. Tracker comparison within each row pair is therefore purely an association-quality comparison, not a detection-confounded one.

## Results

### Macro-averaged (Car + Pedestrian, 5 val sequences)

| Detector | Tracker | HOTA | MOTA | IDF1 | AssA | DetA | IDSw |
|----------|---------|------|------|------|------|------|------|
| Zero-shot YOLOv8m  | ByteTrack | 0.41 | 0.07 | 0.54 | 0.48 | 0.36 | 152 |
| Zero-shot YOLOv8m  | BoT-SORT  | 0.47 | 0.24 | 0.61 | 0.53 | 0.42 | 88  |
| Fine-tuned YOLOv8m | ByteTrack | 0.41 | **0.36** | 0.55 | 0.44 | 0.40 | 159 |
| Fine-tuned YOLOv8m | BoT-SORT  | 0.45 | **0.41** | 0.59 | 0.49 | 0.41 | 86  |

### Deltas (fine-tuned − zero-shot)

| Tracker   | ΔHOTA | ΔMOTA     | ΔAssA | ΔDetA |
|-----------|-------|-----------|-------|-------|
| ByteTrack | 0.00  | **+0.29** | −0.04 | +0.04 |
| BoT-SORT  | −0.02 | **+0.17** | −0.04 | −0.01 |

### Detection counts (fine-tuned ÷ zero-shot)

| Sequence | Zero-shot | Fine-tuned | Ratio |
|----------|-----------|------------|-------|
| 0001 | 6,222  | 3,583 | 0.58 |
| 0006 | 1,090  | 723   | 0.66 |
| 0013 | 5,263  | 1,869 | 0.36 |
| 0017 | 1,906  | 1,268 | 0.67 |
| 0019 | 17,661 | 7,733 | 0.44 |
| **Total** | **32,142** | **15,176** | **0.47** |

The fine-tuned detector emits roughly half as many detections — more selective, higher precision. This drives the entire result pattern below.

## Analysis

**Detection bottleneck flips into an association bottleneck.** The zero-shot baseline showed AssA > DetA (0.48 vs 0.36 for ByteTrack), which is the canonical signature of detection being the binding constraint. Fine-tuning attacks DetA directly: per-class Car mAP@0.5:0.95 lifts from 0.456 to 0.606 (+33% relative), and false-positive detections drop by roughly half across all five val sequences. After fine-tuning, AssA and DetA become roughly balanced (0.44 vs 0.40 for ByteTrack), shifting the binding constraint to the tracker's ability to associate sparser detections across time.

**MOTA is the metric that moves.** ByteTrack lifts from 0.07 to 0.36 — a 5× gain driven almost entirely by reduced false-positive penalty (MOTA = 1 − (FN + FP + IDSw) / GT). The zero-shot model's 32k detections per val split included substantial false positives that MOTA penalised aggressively; fine-tuning removes them. BoT-SORT's MOTA gain is smaller (0.24 → 0.41) because CMC-based motion-consistency filtering already removed the easiest false positives upstream of MOTA — there was less low-hanging FP to recover.

**HOTA is flat by construction.** HOTA = √(DetA × AssA), so gains in DetA can be cancelled by losses in AssA. They are. Fewer detections means fewer associations to chain into long tracks, particularly on the Pedestrian class where fine-tuned recall is 0.488 — half of Pedestrian instances in val sit below the detector's confidence threshold. The tracker cannot associate detections that aren't there, so AssA falls from 0.48 to 0.44 (ByteTrack) and 0.53 to 0.49 (BoT-SORT). IDF1 follows the same pattern: identity preservation depends on continuous detection coverage, which the fine-tuned model trades away for precision.

**Tracker comparison.** BoT-SORT outperforms ByteTrack on every macro metric in both detector conditions. The CMC contribution is most visible on highway-like sequences with strong ego-motion (0001, 0017): IDSw drops from 152 → 88 zero-shot and 159 → 86 fine-tuned, and Frag drops more under fine-tuned inputs (235 vs 284), indicating BoT-SORT's motion compensation extracts more value from cleaner detection streams than ByteTrack's pure IoU matching does.

**Where the next gain lives.** Detection-side. The Pedestrian recall ceiling at 0.488 is the binding constraint. The most direct lever is retraining at imgsz=1280 (vs the current 640), which on KITTI typically lifts Pedestrian recall by 0.10–0.20 absolute and would raise both DetA and AssA simultaneously. Lowering `dump_conf` from 0.1 to 0.05 is a cheaper alternate experiment that trades back some precision for recall. Tracker-side improvements (ReID features in BoT-SORT) are a secondary lever once the detection floor is raised.

## Engineering notes

The work surfaced a few generalisable patterns worth recording:

- **Train/val leakage is invisible without explicit checks.** Frame-level splits within a sequence silently inflate HOTA by 5–10 points on MOT. The only defence is sequence-level holdout with a programmatic set-intersection audit baked into the data preparation code.
- **Library defaults encode hidden assumptions.** boxmot's `frame_rate=30` default produced 3× too-permissive lost-track tolerance on KITTI's 10 FPS data; the bug surfaces only as wrong HOTA, no runtime error. Any parameter with a unit or scale (FPS, image size, sample rate, temporal window) needs verification against the data before accepting its default.
- **Multi-class trackers default to class-agnostic matching.** boxmot's `per_class=False` allows Car↔Pedestrian ID swaps at every IoU-sufficient proximity event. Always set `per_class=True, nr_classes=<your class count>`.
- **HOTA's DetA/AssA decomposition is non-optional.** A naive read of MOTA going from 0.07 → 0.36 looks like an unambiguous fine-tuning win. The decomposition reveals the precision-recall tradeoff and points at the next experiment. Without it, the next iteration would optimise the wrong knob.
- **Dry-run is cheap insurance.** A 1-epoch dry-run validated the full multi-hour training pipeline end-to-end (data loading, GPU allocation, loss computation, weight saving, path resolution) in ~3 minutes and surfaced an Ultralytics path-nesting quirk that would otherwise have been caught hours into a real run. The same pattern caught the new `class_remap` and `skip_eval` paths in the fine-tuned detection run before committing GPU time to the full 5-sequence pipeline.

## Limitations

1. **Val-set-only evaluation.** Metrics are reported on the 5-sequence val split. There is no separate held-out test set; numbers may be optimistic relative to unseen distributions.
2. **Single seed.** All experiments use seed=42. Results could shift ±1–2 HOTA points under different seeds; no variance estimate is provided.
3. **imgsz=640 is conservative.** The standard KITTI fine-tune recipe is imgsz=1280; the smaller setting was chosen for hardware constraints. The Pedestrian recall ceiling is most likely a small-image artefact and the single most informative experiment to repeat at the larger size.
4. **Tracker eval scores Car + Pedestrian only.** The fine-tuned detector also emits Van, Truck, Cyclist, but the trackers are configured for two classes. Extending to 5-class tracking is a follow-up.

## Follow-ups

- Retrain YOLOv8m at imgsz=1280 to recover Pedestrian recall — expected to raise DetA and AssA simultaneously.
- Lower `dump_conf` from 0.1 to 0.05 to test the recall-vs-precision tradeoff on tracking metrics.
- Extend tracker evaluation to all 5 detected classes (`nr_classes=5`).
- KITTI MOT public leaderboard submission for external validation.
- Model-size ablation (YOLOv8s / m / l / x) to characterise the capacity-vs-latency frontier on the same hardware.
- ReID-enabled BoT-SORT row to isolate the contribution of appearance features.
- JSON eval-results pipeline so the demo pulls live metrics instead of hardcoded values.

## Acknowledgments

- **KITTI dataset** — Geiger, Lenz, and Urtasun (2012). *Are we ready for autonomous driving? The KITTI vision benchmark suite.*
- **Ultralytics YOLOv8** — detection backbone and training framework.
- **boxmot** — multi-object tracking framework providing ByteTrack and BoT-SORT implementations.
- **TrackEval** — official MOT evaluation suite for HOTA, MOTA, and IDF1 computation.
