# Multi-Object Tracking on KITTI

A reproducible 2D MOT pipeline:

> YOLOv8 detector → ByteTrack / BoT-SORT → HOTA / MOTA / IDF1 evaluation on KITTI.

[![CI](https://github.com/zarghamsalari/kitti-tracking/actions/workflows/ci.yml/badge.svg)](https://github.com/zarghamsalari/kitti-tracking/actions/workflows/ci.yml)
[![Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://kitti-mot-demo.onrender.com)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Live demo

**[kitti-mot-demo.onrender.com](https://kitti-mot-demo.onrender.com)** — interactive 4-cell ablation: pick a detector (zero-shot vs fine-tuned YOLOv8m), pick a tracker (ByteTrack vs BoT-SORT), see metrics update live.

## Headline finding

Fine-tuning the detector on KITTI lifts MOTA dramatically (5× on ByteTrack, 2× on BoT-SORT) by removing false positives. **HOTA is flat** — the gained DetA is offset by lost AssA, because the fine-tuned model trades recall for precision (Pedestrian recall 0.80 → 0.49) and the trackers cannot associate detections that aren't there.

This is the precision-recall tradeoff that HOTA's DetA/AssA decomposition exists to expose. The decomposition identifies the limiting factor and the next experiment (retrain at imgsz=1280 to recover Pedestrian recall).

Full analysis: [`docs/analysis.md`](docs/analysis.md). Numbers: [`docs/results.md`](docs/results.md).

## 4-cell ablation

Macro-averaged across Car + Pedestrian on 5 val sequences (0001, 0006, 0013, 0017, 0019):

| Detector              | Tracker    | HOTA | MOTA     | IDF1 | AssA | DetA | IDSw |
|-----------------------|------------|------|----------|------|------|------|------|
| Zero-shot YOLOv8m     | ByteTrack  | 0.41 | 0.07     | 0.54 | 0.48 | 0.36 | 152  |
| Zero-shot YOLOv8m     | BoT-SORT   | 0.47 | 0.24     | 0.61 | 0.53 | 0.42 | 88   |
| Fine-tuned YOLOv8m    | ByteTrack  | 0.41 | **0.36** | 0.55 | 0.44 | 0.40 | 159  |
| Fine-tuned YOLOv8m    | BoT-SORT   | 0.45 | **0.41** | 0.59 | 0.49 | 0.41 | 86   |

Detection inputs are shared across trackers within each detector condition (verified via `run_meta.json` hash). Tracker differences within a detector pair are not confounded by detection variance.

## Quick start

```bash
git clone https://github.com/zarghamsalari/kitti-tracking.git
cd kitti-tracking
make install
make download          # ~15 GB raw KITTI
make eval              # detect → track → metrics
make demo              # streamlit on http://localhost:8501
```

## Method

1. **Detection.** YOLOv8m, evaluated zero-shot (COCO-pretrained, COCO car→Car / person→Pedestrian) and fine-tuned (KITTI 5-class: Car, Van, Truck, Pedestrian, Cyclist; trained on 16 sequences at imgsz=640, AdamW, early stopping with patience=20).
2. **Tracking.** ByteTrack (Kalman + IoU two-stage matching) and BoT-SORT (adds CMC via sparse optical flow). Both run with `per_class=True`, `frame_rate=10` to match KITTI's capture rate.
3. **Detection dump.** Per-frame inference at `dump_conf=0.1` to MOT16 format. Same files feed both trackers within each detector condition.
4. **Eval.** TrackEval official MOT suite. HOTA primary; MOTA, IDF1, AssA, DetA, IDSw, Frag, MT, ML reported alongside.

## Reproducibility

Seed=42 across all runs. Sequence-level train/val split with set-intersection audit (overlap = 0 verified). Each run dumps `run_meta.json` with git SHA, config hash, weights checksum, and library versions. `make eval` reproduces the headline numbers from a clean clone within ±0.5 HOTA.

## Repository layout

```
src/tracking/      pipeline modules (detection, trackers, eval, viz)
configs/           YAML configs per detector / tracker
streamlit_app/     interactive 4-cell demo
docs/              results.md, analysis.md
tests/             unit + integration tests (pytest)
scripts/           training script for fine-tune
```

## License

MIT. KITTI dataset is subject to [its own license](https://www.cvlibs.net/datasets/kitti/eval_tracking.php).
