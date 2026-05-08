# T5 plan — Tracker #1: ByteTrack

> Drafted 2026-05-07. Awaiting review before implementation.

## Context

Wire ByteTrack on top of the saved v2 MOT16 detections. The fixed detection
input (`runs/det/yolov8m_zeroshot/`) is the ablation anchor — identical input
to ByteTrack and (later) BoT-SORT means any metric difference is purely
tracker-attributable. The data contract at every boundary (detection format →
tracker input → tracker output → eval input) is fully verified in
`docs/eval-format.md` as of Steps 0a and 0b.

Two prior misconfigurations caught in pre-flight (Steps 0a and 0b) — both
would have produced silently-wrong HOTA:
1. **Missing class column** — 10-column v1 format had no class; ByteTrack
   would have used a sentinel `-1` as the class id. Fixed in Step 0a.
2. **per_class=False default** — cross-class ID swaps at every
   Car/Pedestrian proximity event. Must explicitly set `per_class=True`.

---

## Locked design decisions

### ByteTracker construction (all values verified from boxmot 18.0.0 source)

```python
BYTETracker(
    track_thresh=0.45,   # library default — first-pass baseline; tune later
    match_thresh=0.8,    # library default — matches original paper
    track_buffer=25,     # library default — "25 frames @ 30 FPS = ~0.83 s"
    frame_rate=10,       # KITTI FPS — MANDATORY: scales buffer_size correctly
    min_conf=0.1,        # library default — aligns with dump_conf=0.1
    per_class=True,      # MANDATORY: prevents Car↔Pedestrian ID swaps
    nr_classes=2,        # Car (0) + Pedestrian (1)
)
```

**MANDATORY parameters — must not be removed or defaulted:**

- `per_class=True`: boxmot's default `per_class=False` passes all detections
  into a single class-agnostic pool before matching. A Pedestrian detection
  whose IoU with an active Car track exceeds `match_thresh` will be assigned
  the car's track ID. On KITTI urban sequences — pedestrians crossing in front
  of moving cars, pedestrians standing adjacent to parked cars — this happens
  constantly. The corruption is silent: no exception, `track_id` column looks
  normal, but the class label on that ID silently changes mid-sequence.
  TrackEval's per-class HOTA then penalises every frame where the ID's
  class disagrees with GT, tanking AssA for both Car and Pedestrian with no
  obvious pointer to the cause.
- `frame_rate=10`: boxmot computes `buffer_size = int(frame_rate / 30.0 * track_buffer)`.
  With the library default `frame_rate=30`, `buffer_size = int(30/30 * 25) = 25`
  frames. At KITTI's 10 FPS that is 2.5 s of lost-track tolerance — 3× the
  intended ~0.83 s. Tracks that should be declared lost after a brief occlusion
  persist for 2.5 s, accumulate false re-associations, and inflate AssA. With
  `frame_rate=10`, `buffer_size = int(10/30 * 25) = 8` frames (0.8 s), which
  matches the original paper's intent.

**Alignment note:** `min_conf=0.1` matches `dump_conf=0.1` from T3b. This is
intentional — the dump threshold sets the floor on what enters the pipeline;
the tracker's `min_conf` is the floor on what it processes. Equal values mean
no detections are silently filtered at the tracker boundary.

### Library and version

Pin `boxmot==18.0.0` in `pyproject.toml`. This is the version the contract in
`docs/eval-format.md` was verified against. A future upgrade requires
re-verifying Boundaries 2 and 3 (especially `per_class` semantics and the
`frame_rate` formula, which are internal implementation details that could
change between major versions without a changelog entry).

### Ablation hygiene

T5 (ByteTrack) and T7 (BoT-SORT) both use library defaults for
`track_thresh`, `match_thresh`, and `track_buffer`. This is intentional — any
hyperparameter tuning applied to only one tracker would contaminate the
ablation; the metric difference would no longer be purely tracker-attributable.
Tune only **after** both trackers have a default-parameter baseline on record,
and apply the same trials to both.

### Detection input

Use `runs/det/yolov8m_zeroshot/` (v2 format, 11 columns, class at parts[7]).
T4 fine-tuned detections will replace this input for the T7 ablation. The
config path in `configs/tracker_bytetrack.yaml` must be updated when T4 lands;
until then, point at the zeroshot directory.

### Output format

Same 11-column v2 MOT16 shape as the detection input, with `track_id`
(column 2) filled in by the tracker. Written to
`runs/track/bytetrack/<seq>.txt`. Enables clean diff between detection and
tracking output (same format, same reader).

### Frame-counter invariant

Call `tracker.update()` for **every** frame, even frames with zero detections.
The tracker advances `self.frame_count` inside `update()`. Skipping a
zero-detection frame desyncs the internal frame counter from the data's frame
numbering, corrupting Kalman state for subsequent frames. Pattern:

```python
for frame_idx in range(seq.num_frames):
    frame_dets = detections_by_frame.get(frame_idx, np.empty((0, 6), dtype=np.float32))
    tracks = tracker.update(frame_dets, img)
    # write tracks even if tracks.shape[0] == 0
```

### One tracker instance per sequence

Reset (instantiate a new `BYTETracker`) at the start of each sequence. Track
IDs restart at 1 per sequence — cross-sequence ID collisions are expected and
handled correctly by TrackEval's per-sequence evaluation.

---

## Implementation scope

### A. Dependency

Add to `pyproject.toml` under `[project.dependencies]`:
```
"boxmot==18.0.0",
```
Run `make install` to pick it up.

### B. MOT16 reader (`src/tracking/trackers/mot16_io.py`, new file)

Pure-data module — no tracker imports, CI-runnable.

```python
def read_mot16_v2(path: Path) -> dict[int, np.ndarray]:
    """Read an 11-column v2 MOT16 file.

    Returns dict mapping frame_idx → (N, 6) float32 array
    [x1, y1, x2, y2, conf, cls] in xyxy coords (converted from xywh on read).
    Raises ValueError if format_version in adjacent run_meta.json != v2.
    """
```

The reader must:
1. Check `run_meta.json` `format_version == "mot16-kitti-v2"` and raise
   loudly if not — prevents silently consuming v1 files if re-dump is ever
   skipped.
2. Parse columns: `frame(0), id(1), x(2), y(3), w(4), h(5), conf(6), cls(7)`.
3. Convert xywh → xyxy: `x2 = x + w`, `y2 = y + h`.
4. Group by frame index, return dict.

### C. MOT16 writer (`src/tracking/trackers/mot16_io.py`, same file)

```python
def write_mot16_v2(tracks_by_frame: dict[int, np.ndarray], path: Path) -> int:
    """Write tracker output to 11-column v2 MOT16 file. Returns row count."""
```

Converts xyxy → xywh on write (tracker outputs xyxy from Boundary 3).

### D. ByteTrack runner (`src/tracking/trackers/bytetrack.py`)

Replace the current `raise NotImplementedError` stub with:

```python
@dataclass
class ByteTrackConfig(BaseModel):
    detections_dir: Path
    tracks_dir: Path
    val_sequences: list[str]
    track_thresh: float = 0.45
    match_thresh: float = 0.8
    track_buffer: int = 25
    frame_rate: int = 10     # must be KITTI FPS
    min_conf: float = 0.1
    per_class: bool = True   # must be True; see locked decisions
    nr_classes: int = 2
    seed: int = 42

def run_bytetrack(config_path: Path) -> None:
    """Per-sequence ByteTrack on saved v2 MOT16 detections."""
```

For each sequence:
1. Load detections via `read_mot16_v2`.
2. Instantiate fresh `BYTETracker(**config)`.
3. Load images via `KittiSequence.frames()`.
4. Loop frames 0..N-1, calling `tracker.update(dets, img)`.
5. Write tracks via `write_mot16_v2`.
6. Accumulate row counts for `run_meta.json`.

### E. Config update (`configs/tracker_bytetrack.yaml`)

The stub config was written speculatively and points at
`runs/det/yolov8m_finetuned/` — T4 detections that do not exist yet. For T5,
update `detections_dir` to `runs/det/yolov8m_zeroshot`. When T4 lands (T7
ablation), this will be a one-line config swap with no code changes. Add
`per_class: true` and `nr_classes: 2` explicitly (do not rely on defaults).

### F. CLI (`src/tracking/cli.py`)

Wire `run_bytetrack` into the existing `tracking track` command. Currently a
stub — implement to load the YAML config and call `run_bytetrack(config_path)`.

### G. run_meta.json

Write `runs/track/bytetrack/run_meta.json` with: git SHA, config hash,
detection input hashes (sha256 of each `<seq>.txt`), seed, timestamp, boxmot
version, detection `format_version` consumed, output `format_version` written.
Re-use `tracking.detection.run_meta.RunMeta` if fields align; otherwise a
lightweight dataclass is fine.

---

## Tests (CI-runnable, no GPU, no KITTI data)

Test names describe the contract being verified, not the setup, so CI failure
messages are self-explanatory six months from now.

### Stable-ID smoke test

```python
def test_bytetrack_assigns_stable_ids_across_frames() -> None:
    """Track IDs must be consistent across consecutive frames for the same object.

    Non-overlapping Car and Pedestrian boxes — verifies basic association
    stability. Cross-class separation is verified separately in
    test_per_class_prevents_cross_class_id_continuation.
    """
    from boxmot import BYTETracker
    tracker = BYTETracker(per_class=True, nr_classes=2, frame_rate=10)
    car  = np.array([[100, 100, 200, 200, 0.90, 0.0]], dtype=np.float32)
    ped  = np.array([[400, 100, 440, 200, 0.85, 1.0]], dtype=np.float32)
    dets = np.vstack([car, ped])
    img  = np.zeros((375, 1242, 3), dtype=np.uint8)

    ids = [set(tracker.update(dets, img)[:, 4].astype(int)) for _ in range(3)]
    assert ids[0] == ids[1] == ids[2], "IDs changed between frames"
    assert len(ids[0]) == 2, "Expected exactly 2 active tracks"
```

### Per-class isolation test (most important)

```python
def test_per_class_prevents_cross_class_id_continuation() -> None:
    """With per_class=True, a track of one class must not be continued
    by a detection of a different class, even if their boxes overlap perfectly.

    This is the specific failure mode of per_class=False on urban KITTI:
    a Pedestrian detection whose IoU with a Car track exceeds match_thresh
    gets the car's track ID. The boxes here overlap 100%, so without
    per_class filtering the match would be made.
    """
    from boxmot import BYTETracker
    tracker = BYTETracker(per_class=True, nr_classes=2, frame_rate=10)
    img = np.zeros((375, 1242, 3), dtype=np.uint8)

    # Frame 0: a Car at a specific location.
    car = np.array([[100, 100, 200, 200, 0.90, 0.0]], dtype=np.float32)
    out_0 = tracker.update(car, img)
    car_id = int(out_0[0, 4])

    # Frame 1: a Pedestrian at the SAME location. With per_class=False this
    # detection would be matched to the car track (IoU = 1.0 > match_thresh).
    # With per_class=True it must not — a different class must never extend
    # an existing track.
    ped = np.array([[100, 100, 200, 200, 0.90, 1.0]], dtype=np.float32)
    out_1 = tracker.update(ped, img)

    if len(out_1) > 0:
        ped_id = int(out_1[0, 4])
        assert ped_id != car_id, (
            f"Cross-class ID continuation: pedestrian got the car's track ID "
            f"({car_id}). per_class is not actually filtering by class."
        )
```

### Empty-frame / frame-counter test

```python
def test_empty_frame_advances_internal_frame_counter() -> None:
    """Empty-frame update() calls must advance the internal frame counter,
    so a track that disappears for longer than the buffer is correctly lost.

    With frame_rate=10 and track_buffer=25:
        buffer_size = int(10 / 30 * 25) = 8 frames
    10 consecutive empty frames is past the buffer — the original track must
    be declared lost and the next detection of the same object gets a new ID.
    If empty calls did NOT advance the counter, the tracker would never age
    out the stale track, and IDs would persist indefinitely through occlusions.
    """
    from boxmot import BYTETracker
    tracker = BYTETracker(
        per_class=True, nr_classes=2, frame_rate=10, track_buffer=25
    )
    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    car_box = np.array([[100, 100, 200, 200, 0.90, 0.0]], dtype=np.float32)

    # Frame 0: establish a Car track, record its ID.
    out_0 = tracker.update(car_box, img)
    original_id = int(out_0[0, 4])

    # Frames 1–10: empty. 10 > buffer_size (8), so the track must be lost.
    for _ in range(10):
        out_empty = tracker.update(np.empty((0, 6), dtype=np.float32), img)
        assert out_empty.shape == (0, 8), (
            f"Empty-frame output should be (0, 8), got {out_empty.shape}"
        )

    # Frame 11: same Car at same location. If the frame counter advanced
    # correctly, the original track is gone and this gets a NEW ID.
    out_11 = tracker.update(car_box, img)
    assert len(out_11) > 0, "Expected detection to produce a track"
    new_id = int(out_11[0, 4])
    assert new_id != original_id, (
        f"Frame counter did not advance during empty calls: "
        f"detection after 10 empty frames reused track ID {original_id}, "
        f"but it should have been declared lost (buffer is 8 frames)."
    )
```

### MOT16 I/O round-trip test

Write a known (frame, track_id, xywh, conf, cls) array → v2 file, read it
back, assert xyxy conversion round-trips losslessly.

### Format version guard test

`read_mot16_v2` on a file whose adjacent `run_meta.json` has
`format_version: "mot16-kitti-v1"` must raise `ValueError` with a message
that includes `"mot16-kitti-v1"` — so the error tells you exactly which file
triggered it. Silent parse is the failure mode being guarded: a v1 file
(missing class column) consumed as v2 would produce all-`-1` class IDs, and
HOTA would silently compute on the wrong class assignments.

```python
def test_read_mot16_v2_refuses_v1_format_version(tmp_path: Path) -> None:
    # write a v1 run_meta.json adjacent to a dummy detection file
    ...
    with pytest.raises(ValueError, match="mot16-kitti-v1"):
        read_mot16_v2(det_file)
```

---

## Manual verification (after merge)

```powershell
.\.venv\Scripts\tracking.exe track --config configs/tracker_bytetrack.yaml
```

Checklist:
- [ ] 5 track files exist: `runs/track/bytetrack/{0001,0006,0013,0017,0019}.txt`
- [ ] No file is all `-1` in the track-id column (column 2)
- [ ] `head -5` of each file shows sensible increasing frame indices
- [ ] Per-sequence track ID count is plausible (not 32k unique IDs)
- [ ] `run_meta.json` populated with detection input hashes
- [ ] `python -c "import json; d=json.load(open('runs/track/bytetrack/run_meta.json')); print(d['format_version'])"` → `mot16-kitti-v2`

---

## Critical files

- `src/tracking/trackers/bytetrack.py` — replace stub
- `src/tracking/trackers/mot16_io.py` — new file (reader + writer)
- `configs/tracker_bytetrack.yaml` — update detections_dir + add per_class/nr_classes
- `src/tracking/cli.py` — wire `tracking track` command
- `tests/test_bytetrack.py` — new: `test_bytetrack_assigns_stable_ids_across_frames`, `test_per_class_prevents_cross_class_id_continuation`, `test_empty_frame_advances_internal_frame_counter`, `test_read_mot16_v2_refuses_v1_format_version`, round-trip
- `pyproject.toml` — add `boxmot==18.0.0`

---

## Risks / things to watch

- **boxmot install on CPU-only machine** — verify `pip install boxmot==18.0.0`
  resolves without torch GPU requirement. If it pulls in a heavy GPU build, pin
  torch separately in pyproject.toml.
- **per_class=True with nr_classes=2 vs nr_classes=80** — boxmot allocates a
  track buffer per class. With `nr_classes=80` and `per_class=True`, it
  allocates 80 buffers for a 2-class problem. Use `nr_classes=2`.
- **Sequence 0019 is 1059 frames** — longest sequence; if per-frame inference
  is needed (image loading), it will take time. Tracker itself (no model) should
  be fast.
- **`runs/` is gitignored** — track files are local only, consistent with
  detection files. run_meta.json is the only artifact that should be checked in
  (it isn't — `runs/` is fully gitignored — but its content should be logged).

---

## Hyperparameter tuning (post-baseline)

Cap at 5 trials on val (per CLAUDE.md). Tune only after the baseline pass with
library defaults is committed and HOTA is on record. Parameters to consider:
`track_thresh` (most impactful), `match_thresh`, `track_buffer`. Do not tune
`frame_rate` — it's calibration, not a hyperparameter. Document each trial and
the final choice in `docs/paper.md`.

---

## Estimated effort

3–4 hours. Main work is the MOT16 reader/writer and the bytetrack runner.
CLI wiring and config update are straightforward. Tests add ~1 hour. No GPU
needed — tracker runs on CPU in milliseconds per frame.

---

## Branch + PR

- Branch: `t5-bytetrack`
- PR title: `feat(track): ByteTrack on v2 MOT16 detections (T5)`
