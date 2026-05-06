# Claude Code — Setup & Workflow

This file is for the human (you), not for Claude Code itself. Claude Code reads `CLAUDE.md` automatically.

## One-time desktop setup

```bash
# 1. Clone (after pushing this scaffold to GitHub)
git clone git@github.com:USER/kitti-tracking.git
cd kitti-tracking

# 2. Create env (use uv, pyenv, or conda — your call)
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 3. Install
make install

# 4. Verify the scaffold tests pass
make test                          # KITTI loader + metrics + seed tests
make lint                          # ruff + mypy clean
```

## Open in Claude Code

```bash
# From the repo root:
claude
```

Claude Code will read `CLAUDE.md` automatically and understand the project. Your first prompt should be:

> Read CLAUDE.md and confirm you understand the task list. Tell me what's already scaffolded vs what's still unchecked. Don't write code yet.

## Per-task prompts

The task list lives in `CLAUDE.md`. Pick whichever task is next (or jump around — they're loosely ordered, not strictly sequential after T2). One focused prompt per task:

### T1 — Push to GitHub + verify CI
> Task T1: I want to push this scaffold to a new GitHub repo. Walk me through git init, remote setup, and the first push. Then confirm the existing CI workflow runs green on Actions. If anything in the scaffold is broken, fix it before we push.

### T2 — KITTI download + smoke notebook
> Task T2: Run `make download` and verify it works. Then fill in `notebooks/01_explore_kitti.ipynb` using `tracking.data.kitti` — load every sequence, print class distribution, render 3 GT-overlaid frames per split. Commit on a `t2-explore-kitti` branch.

### T3 — Detection: zero-shot
> Task T3: Implement zero-shot YOLOv8m on KITTI val per the docstring plan in `src/tracking/detection/yolo.py`. Just the COCO->KITTI mapping, run inference, save MOT16 detections, log per-class mAP. No fine-tuning yet. Add unit tests that don't need GPU.

### T4 — Detection: fine-tune
> Task T4: Add the fine-tuning path to `src/tracking/detection/yolo.py`. Convert KITTI labels to YOLO format respecting sequence-level train/val split. Fine-tune for 30-50 epochs at imgsz=1280. Save MOT16 detections from the fine-tuned model. Document choices in the commit message.

### T5 — ByteTrack
> Task T5: Implement the ByteTrack runner per `src/tracking/trackers/bytetrack.py` docstring. Use ultralytics' built-in ByteTracker. Output MOT16 tracks under `runs/track/bytetrack/`. Add tests. Update CLAUDE.md.

### T6 — Eval harness
> Task T6: Implement TrackEval-based metrics in `src/tracking/eval/metrics.py`. Render results to `docs/results.md`. Add tests for the conversion from KITTI GT to TrackEval's MOT format. Update CLAUDE.md.

### T7 — BoT-SORT + ablation
> Task T7: Implement BoT-SORT runner using `boxmot`. Run head-to-head against ByteTrack on identical detections (verify by hashing the detection files in run_meta.json). Update `docs/results.md` with both rows side-by-side. Update CLAUDE.md.

### T8 — Streamlit demo
> Task T8: Build the Streamlit demo in `streamlit_app/app.py`. Pre-render annotated mp4s for each val sequence × tracker combination. Dockerize. Test the container locally on port 8501. Update CLAUDE.md.

### T9 — Deploy demo
> Task T9: Deploy the Docker image to Render free tier. Walk me through the Render config. Add a public URL badge to README. Update CLAUDE.md.

### T10 — Writeup + tag
> Task T10: Complete `docs/paper.md` with real numbers, failure-case analysis, and reproducibility notes. Tag `v0.1.0`. Draft a 200-word LinkedIn post and a 150-word Instagram caption. Update CLAUDE.md — all boxes ticked.

## Tips

- **Use `/cost` and `/context`** in Claude Code to keep token usage in check.
- **One PR per task.** Use Claude Code to write the PR description as a CLAUDE.md status diff.
- **If Claude Code goes off-plan**, paste the relevant CLAUDE.md section and say: "stay within this scope".
- **Don't merge without `make lint` and `make test` green.** Pre-commit hooks enforce this on commit.

## Common issues

- **Ultralytics first run downloads weights** to `~/.config/Ultralytics/`. ~150 MB, one-time.
- **GPU not used:** ultralytics auto-picks CUDA if available. Confirm with `python -c "import torch; print(torch.cuda.is_available())"`.
- **Pre-commit fails on first commit:** run `pre-commit run --all-files` once, fix what it flags, commit again.
- **Render free-tier OOM on demo:** disable ReID in BoT-SORT for the deployed demo, keep it locally.
