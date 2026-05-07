# T3b plan — zero-shot YOLOv8m on KITTI val

> Drafted 2026-05-06. Awaiting review before implementation.

## Context
First real ML inference of the project. Run YOLOv8m COCO-pretrained on the 5 val sequences. Two artifacts: per-class mAP (writeup material), and MOT16-format detections (T5/T7 tracker input — the *fixed* input that makes the ablation clean). No fine-tuning yet — that's T4.

## Locked design decisions
1. `ultralytics` weights download to **cwd** (`./yolov8m.pt`), not `%APPDATA%\Ultralytics\` as previously assumed. Recent ultralytics versions (≥8.0) write to the working directory. Already gitignored via the `*.pt` rule — no repo bloat.
2. `imgsz=1280` everywhere — KITTI's small distant objects need it; speed cost is acceptable
3. Two pipelines from one model: `model.val()` for mAP, separate `model(image)` loop for MOT16 dumps
4. `conf=0.1` for MOT dumps (NOT 0.5) — preserves ByteTrack's low-conf differentiator. `conf=0.001` for `model.val()` (mAP needs the full PR curve)

## Note for T4 (and any future YOLO label generation)

KITTI image dimensions are **not uniform** across sequences. Empirically verified across all 21 training-split sequences (16 train + 5 val):

| Resolution | Sequences | Count |
|------------|-----------|-------|
| 1242×375   | 0000–0013 (excl. 0014–0016) | 14 |
| 1224×370   | 0014, 0015, 0016, 0017 | 4 |
| 1238×374   | 0019 | 1 |
| 1241×376   | 0020 | 1 |

**7 of 21 sequences (33%) deviate** from canonical 1242×375; 0020 is one pixel off in each axis (adversarially close to canonical).

**Implication for T4:** training-sequence YOLO label generation MUST call `write_yolo_labels(image_size=None)` so dimensions are read per-sequence via `_read_image_size()`. The hardcoded default would silently mis-normalise labels by 1–1.5% on a third of the training data, corrupting fine-tune targets. Same rule applies to T3b val labels.

## Dual class-space design (post-implementation correction)

The first manual run produced mAP ≈ 0.0002 — class-space mismatch between predictions (COCO indices) and GT labels (KITTI indices). The fix introduces two distinct mappings, each used in its right phase:

| Mapping | Index space | Used by | Semantics |
|---------|-------------|---------|-----------|
| `KITTI_TO_COCO_ZEROSHOT` | COCO80 (Car=2, Ped=0) | T3b eval phase (`model.val()`) | GT labels match COCO-pretrained predictions for correct mAP |
| `KITTI_TO_YOLO_FINETUNE` (a.k.a. `KITTI_TO_YOLO_ZEROSHOT`, alias kept) | KITTI 2-class (Car=0, Ped=1) | T4 fine-tune | Custom output head trained directly on KITTI |
| `COCO_TO_KITTI` (in `tracking.detection.yolo`) | COCO → KITTI | T3b dump phase | Remap inference output to MOT16 rows for trackers |

Eval `data.yaml` declares `nc: 80` with the full COCO names list (`COCO80_NAMES`) so ultralytics interprets indices correctly. Bug surfaced on first manual run; documented here so T4 doesn't redo the same mistake when it switches `data.yaml` to `nc: 2`.

## Implementation scope

### A. YOLO eval data layout
ultralytics needs a YOLOv8 dataset structure:
```
data/yolo_kitti/
├── data.yaml                       # paths + class names
├── images/val/<seq>_<frame>.png    # symlinked from data/kitti_tracking/training/image_02/
└── labels/val/<seq>_<frame>.txt    # written via existing write_yolo_labels (T3a)
```
New helper in `src/tracking/data/yolo_format.py` (extends T3a): `prepare_yolo_eval_dataset(sequences, root_in, root_out)` — symlinks images and writes labels with `<seq>_<frame>` flattened naming (ultralytics doesn't traverse subdirs by default).

### B. Detection runner — fill in `src/tracking/detection/yolo.py`
Currently `NotImplementedError`. Replace with:
- Load config (pydantic model)
- `set_seed(seed)` for determinism
- Build YOLO model from config (default `yolov8m.pt`)
- **Phase 1 — eval:** `model.val(data=data_yaml, imgsz=1280, conf=0.001, iou=0.5)` → keep `metrics.box.map`, `metrics.box.maps[c]` per class
- **Phase 2 — MOT16 dump:** for each val sequence, for each frame, `model(image_path, imgsz=1280, conf=0.1, classes=[0, 2])` (COCO person + car only, drops everything else at the model level). Convert results to MOT16 rows applying class remap `{2: 0 (Car), 0: 1 (Pedestrian)}`. Write `runs/det/yolov8m_zeroshot/<seq>.txt`
- Write `runs/det/yolov8m_zeroshot/run_meta.json` with: git SHA, config hash, ultralytics version, torch version, model checksum, seed, timestamp, eval mAP per class

### C. Config — `configs/detector_yolov8.yaml`
Already exists; need to read and extend if needed. Expected shape:
```yaml
detector:
  name: yolov8m
  weights: yolov8m.pt        # ultralytics will download
  imgsz: 1280
  val_conf: 0.001
  dump_conf: 0.1
  iou: 0.5
  classes: [0, 2]            # COCO person, car
data:
  root: data/kitti_tracking
  val_seqs: ["0001", "0006", "0013", "0017", "0019"]
  yolo_eval_dir: data/yolo_kitti
output:
  det_dir: runs/det/yolov8m_zeroshot
seed: 42
```
Pydantic model in `src/tracking/detection/yolo.py` to validate this.

### D. Tests (CI-runnable, no model download)
- `test_coco_to_kitti_remap`: COCO indices 0/2/16/3 → expected outputs (only 0,2 kept; remapped to 1,0)
- `test_mot16_row_format`: one detection → 10-column space-separated row matches MOT16 spec
- `test_dump_writes_per_seq_files`: with a fake "model" that returns canned boxes, confirm one `.txt` per sequence with right content
- The actual end-to-end `model.val()` and inference runs are excluded — too heavy for CI. Mark heavy tests `@pytest.mark.slow`.

### E. Manual verification (after merge)
```powershell
.\.venv\Scripts\tracking.exe detect --config configs/detector_yolov8.yaml
```
Expected:
- ~50 MB weights download on first run only
- `runs/det/yolov8m_zeroshot/{0001,0006,0013,0017,0019}.txt` exist, each with detection rows
- `run_meta.json` shows car mAP roughly in the 0.45–0.60 range
- pedestrian mAP lower (KITTI pedestrians are tiny — this is the headline finding that motivates fine-tuning in T4)

## Critical files
- `src/tracking/detection/yolo.py` — fill in (currently stub)
- `src/tracking/data/yolo_format.py` — extend with `prepare_yolo_eval_dataset`
- `configs/detector_yolov8.yaml` — read existing, adjust
- `tests/test_detection_yolo.py` — new
- `pyproject.toml` — verify `ultralytics>=8.1` is there

## Risks / things to watch
- **Image dimensions vary across KITTI seqs** — most are 1242×375 but a few sequences are slightly different. Read `cv2.imread().shape` per sequence's first frame; update `write_yolo_labels` signature to take per-sequence sizes
- **Pedestrian mAP will be embarrassing** — that's not a bug, that's the writeup material
- **First run is slow** — weights download + model warm-up
- **`runs/` is in `.gitignore`** — already verified

## Estimated effort
1.5–2 hours of focused work. Larger than T3a because of the ultralytics eval-data plumbing and the dual-pipeline design.

## Branch + PR
- Branch: `t3b-zeroshot-yolov8m`
- PR title: `feat(detect): YOLOv8m zero-shot inference + mAP eval (T3b)`

## Open iteration points
- Should the COCO class filter happen at the model level (`classes=[0, 2]`) or post-hoc? Lean model-level — faster, simpler
- The image-size-per-sequence change will touch T3a code — fine but worth flagging
- Anything to exclude from this PR (e.g., split mAP and MOT dump into T3b1/T3b2)?
