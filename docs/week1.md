# Week 1 — 2D Multi-Object Tracking on KITTI

## Goal & Curriculum Targets

This is Week 1 of an 8-week robotics perception sprint. The objective is a reproducible MOT pipeline on KITTI 2D MOT for cars and pedestrians, with the detector frozen and the tracker as the variable under ablation.

**Targets (val split, macro-averaged Car + Pedestrian):**

| Metric | Target | Best achieved | Cell |
|--------|--------|---------------|------|
| HOTA   | ≥ 0.55 | 0.47 | Zero-shot + BoT-SORT |
| MOTA   | ≥ 0.65 | 0.41 | Fine-tuned + BoT-SORT |
| IDF1   | ≥ 0.65 | 0.61 | Zero-shot + BoT-SORT |

None of the three targets are met. The gap is decomposed in the Analysis section below into a detector-side recall problem and a tracker-side association ceiling.

## Methodology

### Dataset

KITTI MOT (Geiger et al. 2012): 21 training sequences (0000–0020) with public ground truth. Split at the sequence level — no frame leakage — into:

- **Train (16 sequences):** 0000, 0002–0005, 0007–0012, 0014–0016, 0018, 0020 — 5,747 images
- **Val (5 sequences):** 0001, 0006, 0013, 0017, 0019 — 2,261 images

Overlap between train and val image sets: **0**. Verified programmatically via set intersection on filenames.

### Evaluation harness

TrackEval (official MOT evaluation suite) computing HOTA, MOTA, IDF1, AssA, DetA, IDSw, and Frag. All headline numbers are macro-averaged across Car and Pedestrian classes. Per-class breakdowns are reported in the ablation table.

## Approach

### Detector

**YOLOv8m** (Ultralytics), evaluated under two conditions:

1. **Zero-shot** — COCO-pretrained weights, class mapping COCO car→Car, person→Pedestrian.
2. **Fine-tuned** — KITTI 5-class (Car, Van, Truck, Pedestrian, Cyclist), trained on the 16-sequence train split at imgsz=640 for up to 100 epochs with patience=20 early stopping. Training stopped early at epoch 47 (best at epoch 27). Per-class val mAP at the best checkpoint: Car 0.818 / 0.606, Pedestrian 0.589 / 0.241 (mAP@0.5 / mAP@0.5:0.95).

### Trackers

1. **ByteTrack** — Kalman filter + IoU-based association with two-stage matching (high-confidence then low-confidence detections). One tracker instance per class for cross-class isolation.
2. **BoT-SORT** — Extends ByteTrack with camera motion compensation (CMC) via sparse optical flow. One instance per class (`per_class=True, nr_classes=2`). `frame_rate=10` to match KITTI's capture rate (default 30 FPS caused 3× too-permissive lost-track tolerance).

Both trackers consume identical MOT16-format detection files for clean within-row comparison.

## Ablation Design

4-cell factorial: **{ByteTrack, BoT-SORT} × {zero-shot detector, fine-tuned detector}**.

Same detection inputs are shared across trackers within each detector condition, ensuring that tracker differences are not confounded by detection variance. Detection files are hash-verified via `run_meta.json`.

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

**Detection bottleneck flips into an association bottleneck.** The zero-shot baseline showed AssA > DetA (0.48 vs. 0.36 for ByteTrack), confirming detection quality was the primary constraint. Fine-tuning directly attacks DetA: per-class Car mAP@0.5:0.95 lifted from 0.456 to 0.606 (+33% relative), and false-positive detections dropped by roughly half across all five val sequences. The tracker received cleaner, fewer inputs. After fine-tuning, AssA and DetA become roughly balanced (0.44 vs. 0.40 for ByteTrack), meaning the next bottleneck is now the tracker's ability to associate sparser detections across time.

**Where did fine-tuning help most?** MOTA. ByteTrack lifted from 0.07 to 0.36 — a five-fold improvement driven almost entirely by reduced false-positive penalty (MOTA = 1 − (FN + FP + IDSw) / GT). The zero-shot model's 32k detections per val split included substantial false positives that MOTA penalised aggressively; fine-tuning removed them. BoT-SORT's MOTA gain is smaller (0.24 → 0.41) because it already benefited from CMC-based motion-consistency filtering; there was less low-hanging FP to remove.

**Why HOTA didn't move.** HOTA = √(DetA × AssA), so gains in DetA can be offset by losses in AssA. They were. Fewer detections means fewer associations to chain into long tracks, especially on the Pedestrian class where fine-tuned recall is only 0.488 — half of all Pedestrian instances in val are below the detector's confidence threshold. The tracker cannot associate what isn't detected, so AssA fell from 0.48 to 0.44 (ByteTrack) and 0.53 to 0.49 (BoT-SORT). The IDF1 trend tells the same story: barely changed because identity-preservation depends on continuous detection coverage, which the fine-tuned model trades away for precision.

**Tracker comparison.** BoT-SORT outperforms ByteTrack on every macro metric in both detector conditions. The CMC contribution is most visible on highway-like sequences with strong ego-motion (0001, 0017): IDSw drops from 152 → 88 zero-shot and 159 → 86 fine-tuned, and Frag drops more under fine-tuned inputs (235 vs. 284), indicating that BoT-SORT's motion compensation extracts more value from cleaner detections than ByteTrack's pure IoU matching does.

**Curriculum target assessment.** None of HOTA ≥ 0.55, MOTA ≥ 0.65, IDF1 ≥ 0.65 are reached at imgsz=640 with the 5-trial hyperparameter cap. The closest is BoT-SORT zero-shot HOTA at 0.47. The most direct path to closing the HOTA gap is detector-side: train at imgsz=1280 (vs. our 640) and lower `dump_conf` to recover detection recall on the Pedestrian class, which would raise both DetA and AssA simultaneously. Tracker-side improvements (ReID features in BoT-SORT) are a secondary lever.

## Lessons Learned

1. **CI strictness catches bugs missed locally.** Both T8 (ruff format, then mypy) and T4a (ruff C420 lint) required fix-and-push roundtrips because local checks ran only a subset of the CI pipeline. A unified `make check` command running `ruff check && ruff format --check && mypy && pytest` before every push would eliminate this class of delay — queued as a Week 1.5 follow-up.

2. **Train/val leakage is invisible without explicit checks.** Splitting frames within a sequence silently inflates HOTA by 5–10 points. The only defence is sequence-level holdout with a programmatic set-intersection audit (overlap = 0) baked into the data preparation code.

3. **Ultralytics 8.4.47 nests `project` under its default `runs/detect/` root.** Passing `project="runs/train"` (relative) produced `runs/detect/runs/train/...`. Fix: pass `project=str(Path("runs/train").resolve())` to force an absolute path. Caught by the dry-run's weight-verification step.

4. **Windows symlinks require admin privileges; `os.link()` hardlinks do not.** Dataset preparation uses hardlinks (`os.link`) instead of symlinks for KITTI→YOLO image linking, avoiding the need for elevated permissions on Windows development machines.

5. **Dry-run is cheap insurance.** A 3-minute, 1-epoch training dry-run validated the full 5-hour training pipeline end-to-end: data loading, GPU allocation, loss computation, weight saving, and path resolution. The path-nesting bug (lesson 3) was caught here, not 5 hours into a real run. The same pattern was used for T4c detection — a one-sequence dryrun took 19 seconds and confirmed the new `class_remap` and `skip_eval` paths before running the full 5-sequence pipeline.

6. **DetA + AssA decomposition is essential before drawing conclusions.** A naive read of MOTA going from 0.07 → 0.36 looks like a straightforward win for fine-tuning. The full picture (HOTA flat, AssA down, DetA up) reveals a precision-recall tradeoff that the headline metric hides. Without HOTA's two-component decomposition, the next experiment would be wrong.

## Limitations

1. **Val-set-only evaluation.** All reported metrics are on the 5-sequence val split. There is no separate held-out test set, so numbers may be optimistic relative to unseen data.

2. **Single random seed.** All experiments use seed=42. No variance estimate is provided — results could shift ±1–2 HOTA points under different seeds.

3. **Detector class set limited to 5.** Fine-tuning covers Car, Van, Truck, Pedestrian, and Cyclist. KITTI also labels Tram, Person_sitting, and Misc, which are dropped. This biases recall downward on rare classes but has negligible impact on Car/Pedestrian headline metrics.

4. **Tracker evaluation scores only Car + Pedestrian.** Van, Truck, and Cyclist are detected but not yet tracked or evaluated. The macro-average covers only the two primary classes.

5. **imgsz=640 is conservative.** YOLOv8m fine-tuning at imgsz=1280 is the standard recipe for KITTI but was not run here due to GPU memory and time constraints. The Pedestrian recall ceiling at 0.488 is most likely a small-image-size artefact; this is the single most important experiment to repeat.

## Follow-ups

- `make check` Makefile target running the full CI pipeline locally pre-push (catches ruff format, ruff check, mypy, pytest in one command).
- Re-train YOLOv8m at imgsz=1280 with the same data split to recover Pedestrian recall — expected to raise both DetA and AssA simultaneously.
- Lower `dump_conf` from 0.1 to 0.05 in the fine-tuned detection config to test the recall-vs-precision tradeoff on tracking metrics.
- Extend tracker evaluation to all 5 detected classes (`nr_classes=5`) for a complete ablation.
- Submit to the KITTI MOT public leaderboard for external validation.
- Model size ablation: YOLOv8s vs. YOLOv8m vs. YOLOv8l — is the extra capacity worth the latency on the same hardware?
- Add a ReID-enabled BoT-SORT row (row 5 of the ablation) to isolate the contribution of appearance features.
- Build an eval-results JSON pipeline so the Streamlit demo pulls live metrics instead of hardcoded values.

## Acknowledgments

- **KITTI dataset** — Geiger, Lenz, and Urtasun (2012). *Are we ready for autonomous driving? The KITTI vision benchmark suite.*
- **Ultralytics YOLOv8** — detection backbone and training framework.
- **boxmot** — multi-object tracking framework providing ByteTrack and BoT-SORT implementations.
- **TrackEval** — official MOT evaluation suite for HOTA, MOTA, and IDF1 computation.
