# Object Tracking on KITTI MOT

## Owner
Zargham — AI Project Manager

## Goal
Reproducible MOT pipeline on KITTI 2D MOT (cars + pedestrians). Detector frozen, tracker is the variable being ablated.

### Success criteria
- HOTA ≥ 55, MOTA ≥ 65, IDF1 ≥ 65 on the KITTI MOT validation split (cars).
- ≥ 2 trackers benchmarked on identical detections (clean ablation).
- One-command reproduction: `make eval` runs detection → tracking → metrics → markdown table.
- CI green at all times.
- Live Streamlit demo.
- 1-page paper-style writeup in `docs/paper.md`.

## Stack
- **Python 3.11**
- Detection: `ultralytics` (YOLOv8)
- Trackers: ByteTrack and BoT-SORT (vendored or pip-installable)
- Vision: `opencv-python`, `torch`, `torchvision`
- Eval: `TrackEval` (preferred) or `motmetrics` (fallback)
- CLI: `typer`
- Configs: `pydantic` + YAML
- Demo: `streamlit`
- Packaging: `pyproject.toml` + `hatchling`
- Lint/format: `ruff`
- Types: `mypy` (lenient — strict on `src/tracking/eval/` and `src/tracking/data/`)
- Tests: `pytest` + `pytest-cov`
- Pre-commit: `ruff`, `ruff-format`, `mypy` (selective)
- CI: GitHub Actions
- Container: Dockerfile (slim-bookworm + cpu wheels for the demo image)
- Deploy: Render (Streamlit demo)

## Repo layout

## KITTI MOT split 
KITTI tracking has 21 training sequences (0000–0020) with public GT. We use:
- **Train:** 0000, 0002, 0003, 0004, 0005, 0007, 0008, 0009, 0010, 0011, 0012, 0014, 0015, 0016, 0018, 0020
- **Val:** 0001, 0006, 0013, 0017, 0019

Rule: never split frames within a sequence. Sequence-level holdout only. Frame leakage in MOT silently inflates HOTA by 5–10 points.

## Tasks

Pick them off in roughly this order — each builds on the previous. No time pressure; the project resumes wherever the last `[x]` is.

- [x] **T0 — Scaffold.** Repo structure, packaging, CI, configs, KITTI loader + tests, stubs.
- [x] **T1 — Push to GitHub + verify CI.** Init git, push to a new GitHub repo, branch protection on `main`, confirm Actions runs green on the existing test suite.
- [x] **T2 — KITTI download + smoke notebook.** `make download` works end-to-end. `notebooks/01_explore_kitti.ipynb` loads each sequence, prints class distribution, renders a few GT-overlaid frames.
- [x] **T3 — Detection: zero-shot.** Run YOLOv8m COCO-pretrained on KITTI val (COCO car→Car, person→Pedestrian). Log per-class mAP. Save MOT16 detections under `runs/det/yolov8m_zeroshot/`. Numbers: Car mAP@0.5:0.95 = 0.456, Pedestrian = 0.232, all-mAP@0.5 = 0.707. 32k detections across 5 val sequences.
- [ ] **T4 — Detection: fine-tune.** Convert KITTI labels to YOLO format (sequence-respecting train/val split), fine-tune YOLOv8m at imgsz=1280 for 30–50 epochs. Save MOT16 detections under `runs/det/yolov8m_finetuned/`.
- [x] **T5 — Tracker #1: ByteTrack.** Wire ByteTrack on top of saved detections. Tune `track_thresh`, `match_thresh`, `track_buffer` (≤5 trials on val). Output `runs/track/bytetrack/`. 17,146 track rows across 5 val sequences. Per-instance class isolation (boxmot 18.0.0 per_class=True has incomplete isolation via shared lost_stracks).
- [x] **T6 — Eval harness.** TrackEval integration. `make eval` produces `docs/results.md` with HOTA/MOTA/IDF1/AssA/DetA/IDSw/Frag/MT/ML. **Baseline (zero-shot YOLOv8m + ByteTrack, 5 val sequences):** bytetrack-macro HOTA=0.41, MOTA=0.07, IDF1=0.54, AssA=0.48, DetA=0.36, IDSw=152, Frag=384, MT=102, ML=20. AssA > DetA confirms detection is the bottleneck — T4 (fine-tune detector) directly attacks DetA. Three contract bugs found and fixed during integration: (1) `val_sequences` key absent from run_meta — derive from `detection_input_hashes` keys; (2) TrackEval requires dense 0-indexed gt/tracker IDs but KITTI ids are sparse — added remap step in `_build_sequence_data`; (3) integration test gap (no end-to-end test calling `evaluate()` against real run_meta schema) — followup PR pending.
- [ ] **T7 — Tracker #2: BoT-SORT + ablation.** Same detections, swap tracker. Side-by-side comparison in `docs/results.md`. Verify shared detection input by hashing detection files in `run_meta.json`.
- [ ] **T8 — Streamlit demo.** Sequence picker → tracker picker → annotated mp4 + metrics. Dockerize. Test container locally.
- [ ] **T9 — Deploy demo.** Push to Render. Public URL in README badge.
- [ ] **T10 — Writeup + tag.** Fill `docs/paper.md` with method, real numbers, decomposition (DetA vs AssA), failure-case analysis (occlusion / crowded / fast motion), honest limitations. Tag `v0.1.0`.

Update the boxes above as work progresses. Each task → one feature branch → one squash-merge to `main`.

## Conventions

### Code
- Type hints on every public function. `from __future__ import annotations` at top of every module.
- Public API in each module re-exported from `__init__.py`.
- No `print` in library code — use `logging.getLogger(__name__)`. `print` only allowed in `cli.py` and `streamlit_app/app.py`.
- Configs: pydantic `BaseModel` per config type, loaded from YAML.
- No global state. Pass configs/loggers explicitly.

### Git
- Branch per task: `t1-push-github`, `t2-download`, `t3-detection-zeroshot`, etc. Squash-merge to `main`.
- Conventional commits: `feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`.
- **Do NOT add `Co-authored-by:` trailers attributing Claude or Anthropic.** Author commits as the user only.
- Tag end of project as `v0.1.0`.

### Testing
- Every module added needs a test. CI must stay green.
- Tests must NOT require GPU, network, or KITTI download. Mock or use 1–2 frame fixtures.
- `pytest -m slow` for any test that needs real data — excluded from CI.
- Coverage target: 70% on `src/tracking/{data,eval}/` (the parts where bugs cause silent metric inflation).
- **Test the default call path.** When a function has defaulted arguments, write at least one test that calls it with no kwargs (`fn(required_arg)`) and asserts success. Tests that always pass kwargs miss bugs in the default values themselves.
- **Integration tests for cross-component contracts.** When function X reads a file produced by function Y (e.g., `evaluate()` reading `run_meta.json` written by `bytetrack.py`), write an end-to-end test that runs both. Unit tests of helpers in isolation miss contract mismatches; integration tests catch them.

### Metrics discipline
- Primary metric: **HOTA**. Report MOTA and IDF1 alongside but never as headline.
- Always report DetA and AssA decomposition — that is what tells you whether the detector or the tracker is the bottleneck.
- Report IDSw and Frag for tracker-quality narrative.
- Cap tracker hyperparameter trials at 5 on val. Note this in `docs/paper.md` honestly.

### Reproducibility
- Seed everywhere via `tracking.utils.seed.set_seed(42)`.
- Every run dumps `run_meta.json` with: git SHA, config hash, python version, torch version, CUDA version, seed, timestamp, input file hashes.
- `make eval` must reproduce headline numbers from a clean clone within ±0.5 HOTA.

## Commands
```bash
make install        # editable install + dev deps + pre-commit
make test           # pytest with coverage
make lint           # ruff + mypy
make download       # ./scripts/download_kitti.sh
make detect         # zero-shot + fine-tuned YOLOv8 on KITTI val
make track TRACKER=bytetrack    # run tracker on saved detections
make eval           # full pipeline + render results table
make demo           # streamlit run streamlit_app/app.py
make docker         # build container
```

## What Claude Code should do first
1. Read this file and `README.md`.
2. Read `pyproject.toml` to confirm deps.
3. Look at the task list above. Identify the first unchecked task.
4. Ask the user whether to start that task, or jump elsewhere.
5. When working on a task: implement on a feature branch, add tests, run `make lint` + `make test` locally, summarise files changed and how to verify, then update the checkbox in this file.

## Lessons learned (add to as the project progresses)

- **Library defaults may encode implicit frame-of-reference assumptions.** Any parameter with a unit or scale (FPS, image size, sample rate, temporal window) must be verified against your data before accepting its default. Discovered: boxmot's `frame_rate=30` default caused 3× too-permissive lost-track tolerance on KITTI 10 FPS data (`buffer_size = int(frame_rate / 30 * track_buffer)`). Always set `frame_rate=10` for KITTI.

- **Multi-class trackers default to class-agnostic matching — verify `per_class` in source.** boxmot's `per_class=False` default allows Car↔Pedestrian ID swaps at every IoU-sufficient proximity event. The bug is silent: no runtime error, just wrong HOTA. Always set `per_class=True, nr_classes=<your class count>` for class-preserving tracking.

- **Verify cross-component contracts before integration.** Three contract bugs surfaced in T6 alone — (1) `evaluate()` expected a `val_sequences` key in `run_meta.json` that the tracker writer never emitted; (2) TrackEval requires dense 0-indexed track IDs but KITTI's are sparse (gaps like 0,1,2,…,89,95,100); (3) no end-to-end test ever called `evaluate()` against real upstream artifacts, so unit tests passed while integration broke. For every future integration: read both producer and consumer source, list keys/types/index conventions, write an integration test that runs both ends with realistic synthetic data. The pattern from T5's boxmot pre-flight (`per_class`, `frame_rate`) generalizes — always run a "PR 0 research" step before writing integration code.

- **Tooling failure recognition.** When an agent's file-edit tool fails repeatedly with the same error (e.g., `EEXIST` on Windows file paths), do not escalate to shell workarounds (heredocs, `python -c`, `sed`). Each workaround invites a new escaping or encoding failure mode. Stop, restart the agent, or do the edit manually in a real editor. Time spent fighting tooling far exceeds time spent doing the edit by hand.

## Anti-goals (do not do these)
- Do not write a custom tracker from scratch. Use ByteTrack and BoT-SORT as published.
- Do not use private datasets, AISUS data, or any non-public asset imagery.
- Do not optimise on the test split. KITTI's official test set is held back; we use 5 sequences from train as val.
- Do not add heavyweight deps without updating `pyproject.toml` + this file.
- Do not skip tests "just for now". CI green is the bar for every merge.