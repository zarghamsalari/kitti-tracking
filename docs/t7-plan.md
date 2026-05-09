# T7 — BoT-SORT runner + ablation plan

## Goal
Add BoT-SORT as the second tracker, consuming the **same** saved detections as ByteTrack (T5). This enables a clean ablation: detector fixed, tracker varied.

## Ablation table (3 rows, built incrementally)

| Row | Tracker   | ReID | CMC | PR     |
|-----|-----------|------|-----|--------|
| 1   | ByteTrack | No   | No  | T5 (#16) — already merged |
| 2   | BoT-SORT  | No   | Yes | **this PR** (`with_reid=False`) |
| 3   | BoT-SORT  | Yes  | Yes | follow-up PR (`with_reid=True`, OSNet x0_25) |

Row 2 is the headline baseline: CMC-only, no ReID model download, CI-safe.
Row 3 is a follow-up once row 2 is green and evaluated.

## boxmot 18.0.0 BotSort constructor (verified 2026-05-09)

```python
# .venv/Lib/site-packages/boxmot/trackers/botsort/botsort.py:61-108
class BotSort(BaseTracker):
    def __init__(
        self,
        reid_weights: Path,        # positional, required
        device: torch.device,      # positional, required
        half: bool,                # positional, required
        # keyword args with defaults:
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        cmc_method: str = "sof",         # sparse optical flow
        frame_rate: int = 30,
        fuse_first_associate: bool = False,
        with_reid: bool = True,
        **kwargs,                         # forwarded to BaseTracker
    ):
```

### Key observations

1. **`with_reid=False`** skips ReID model loading entirely — no weights file needed, no download. The `reid_weights` positional arg is still required syntactically but unused when `with_reid=False`. Pass `Path("none")` as a dummy.

2. **`frame_rate=10`** is MANDATORY for KITTI (10 FPS). Library default 30 gives `buffer_size = int(30/30*30) = 30` frames = 3.0 s tolerance — 3x too long for KITTI. With `frame_rate=10`: `buffer_size = int(10/30*30) = 10` frames = 1.0 s.

3. **`track_buffer` asymmetry is intentional.** ByteTrack uses 25 (published default), BoT-SORT uses 30 (published default). At `frame_rate=10`, ByteTrack buffer = 8 frames (~0.8 s), BoT-SORT buffer = 10 frames (~1.0 s). Each tracker's published default is preserved.

4. **Code-vs-YAML default mismatch (lesson).** BotSort.__init__ has `track_high_thresh=0.5`, `new_track_thresh=0.6`, but `boxmot/configs/botsort.yaml` has 0.6 and 0.7 (search-space bounds, not runtime defaults). Our YAML uses the **code defaults** from `__init__`, not the config-search YAML.

## Per-class isolation (same pattern as ByteTrack)

boxmot's `BaseTracker.per_class_decorator` swaps `self.active_tracks` per class but `self.lost_stracks` stays instance-level. A lost Car track can match a Pedestrian detection. Workaround: **one BotSort instance per class** (identical to ByteTrack T5 pattern). Each instance's `lost_stracks` is invisible to the other class.

## ECC CMC failure path (verified 2026-05-09)

`boxmot/motion/cmc/ecc.py:58-78`: `cv2.findTransformECC` is wrapped in try/except. On `cv2.error` with `StsNoConv`, logs WARNING and returns identity matrix. Other cv2 errors re-raise. **No try/except needed in the runner** — boxmot handles it.

## Files to create / modify

| # | File | Action |
|---|------|--------|
| 1 | `configs/tracker_botsort.yaml` | Create — flat YAML, mirrors bytetrack config structure |
| 2 | `src/tracking/trackers/botsort.py` | Replace stub — BoTSortConfig + `_make_tracker` + `run_botsort` |
| 3 | `src/tracking/cli.py` | No change needed — dispatch already handled in `__init__.py` |
| 4 | `tests/test_botsort.py` | Create — 5 tests, all `with_reid=False`, CI-safe |

### Files NOT touched
- `src/tracking/trackers/__init__.py` — already dispatches `"botsort"` to `run_botsort`
- `src/tracking/trackers/bytetrack.py` — no changes
- `src/tracking/eval/metrics.py` — no changes (ID remap fix landed in PR #21)
- `src/tracking/trackers/mot16_io.py` — no changes

## Config design (`configs/tracker_botsort.yaml`)

```yaml
name: botsort
detections_dir: runs/det/yolov8m_zeroshot
tracks_dir: runs/track/botsort
data_root: data/kitti_tracking
val_sequences: ["0001", "0006", "0013", "0017", "0019"]

track_high_thresh: 0.5       # code default (not botsort.yaml's 0.6)
track_low_thresh: 0.1
new_track_thresh: 0.6        # code default (not botsort.yaml's 0.7)
track_buffer: 30             # BoT-SORT published default (ByteTrack uses 25)
match_thresh: 0.8
proximity_thresh: 0.5
appearance_thresh: 0.25
cmc_method: sof
frame_rate: 10               # MANDATORY for KITTI 10 FPS
with_reid: false             # headline baseline — no model download
nr_classes: 2
seed: 42
```

## Test plan (`tests/test_botsort.py`)

All tests use `with_reid=False`. No GPU, no network, no KITTI data.

1. **`test_botsort_assigns_stable_ids_across_frames`** — same pattern as ByteTrack test
2. **`test_per_class_prevents_cross_class_id_continuation`** — separate instances, overlapping boxes
3. **`test_empty_frame_advances_internal_frame_counter`** — 10 empty frames > buffer_size=10, track must be lost
4. **`test_with_reid_false_does_not_load_model`** — construct BotSort with `with_reid=False`, assert no model attribute populated
5. **`test_botsort_config_round_trip`** — load YAML, validate with BoTSortConfig pydantic model

## Evaluation

After implementation PR is green:
```bash
python -m tracking.cli track configs/tracker_botsort.yaml
python -m tracking.cli eval data/kitti_tracking/training/label_02 runs/track/botsort
```

PR description will include the metric numbers from this eval run.

## Follow-up (not this PR)
- Row 3: `with_reid=True` + OSNet x0_25 MSMT17 weights
- `docs/eval-format.md` — standardise eval output format
- `docs/results.md` — side-by-side ablation table with all 3 rows
