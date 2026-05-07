# Data format reference: detection → tracking → eval

> **Single source of truth** for data shape at every conversion boundary in
> the project's detection → tracking → eval pipeline. Refer to this whenever
> writing or reviewing code that crosses any of these interfaces.
>
> Lessons from T3b: format mismatches at boundaries (class index space,
> data.yaml `nc`, frame indexing) caused multi-iteration debug cycles. This
> doc exists to make the same class of bug impossible to reach implementation
> without a test catching it.
>
> **🔬 Status of each boundary:**
>
> | Boundary | Verified? |
> |---|---|
> | 1 — T3b detection dump | ✅ Verified by manual run + tests |
> | 1.5 — Class column gap | ✅ Identified; remediation pending |
> | 2 — ByteTrack input (boxmot) | ⚠️ **Working assumption — verify in T5 Step 0 before code** |
> | 3 — ByteTrack output (boxmot) | ⚠️ **Working assumption — verify in T5 Step 0 before code** |
> | 4 — TrackEval input (KITTI 2D) | ⚠️ Working assumption from TrackEval docs — verify in T6 |
> | 5 — TrackEval GT (KITTI 2D) | ⚠️ Working assumption — verify in T6 |
>
> Anything labelled "working assumption" must be verified against the installed
> library's source code before any production code crosses that boundary.
> Verification updates this doc.

## Pipeline overview

```
KITTI label_02/*.txt  ──┐                                     ┌──> TrackEval GT (KITTI scope)
                        │   (boundary 5)                       │
                        └─────────────────────────────────────►│
                                                               │
KITTI image_02/*.png ─► YOLOv8m ─► MOT16 detections ─► ByteTrack ─► MOT16 tracks ─► TrackEval ─► HOTA
                                   (boundary 1)        (b. 2,3)     (boundary 4)
                                   T3b output          T5 task      T5 output      T6 task
```

Each numbered boundary below corresponds to the labels in the diagram above.

---

## 🚨 Known issue: T3b detection dump is missing the class column

> **The 32k MOT16 detection files produced by T3b are missing information that
> T5 (ByteTrack) and T6 (TrackEval) both require.** This is the biggest
> practical finding from writing this doc. Address as **Step 0a of T5** —
> before any tracker integration code — by re-dumping detections in the
> updated 11-column format below.

### What's wrong

T3b's [`detection_to_mot16_row`](../src/tracking/detection/yolo.py) writes a 10-column row:

```
frame, id, x, y, w, h, conf, -1, -1, -1
```

There's no slot for the per-detection class id. Both downstream consumers need it:

- **ByteTrack** computes per-class association affinities. Without class, IDs can swap between cars and pedestrians.
- **TrackEval** computes HOTA per class, then averages. Tracker output rows must declare `Car` or `Pedestrian` (boundary 4).

### Resolution: re-dump in 11-column v2 format

```
frame, id, x, y, w, h, conf, class, -1, -1, -1
```

The new `class` column sits **at position 8**, between `conf` and the world-coordinate sentinels. Keeping the trailing `-1 -1 -1` intact preserves standard MOT16 column semantics (cols 9–11 = world x/y/z) — any downstream tool that expected 10-column MOT16 will fail loudly on the 11-column file rather than silently misreading a world-coordinate slot as class.

`class` is the **KITTI** class id (`Car=0`, `Pedestrian=1`) — the value `KITTI_TO_YOLO_FINETUNE` would assign. T3b's runner already has the COCO id available before remap; just thread it through.

### Cost of the fix

- ~30 min to update `detection_to_mot16_row`, `_dump_mot16_for_sequence`, the existing tests asserting 10 columns, and bump `run_meta.json`'s `format_version` to `mot16-kitti-v2`
- ~30 min to re-run `tracking detect` so the on-disk files match the new format (model + cache are warm; this is the second run, not the first)
- T5 then writes its tracker-output files in the same 11-column shape

Total: ~1 hour. Adds one good test (assert 11 columns and class column present in dump).

### Why this slipped through T3b verification

The verification checklist asked "is the format MOT16-shaped?" — yes, 10 columns matched the spec we documented. It did not ask "do downstream consumers (ByteTrack, TrackEval) need a column we didn't include?" The lesson is general: **verify a producer's output against its consumer's input requirements, not just against a format-spec name.** To be captured in CLAUDE.md alongside the T5-plan PR.

### Alternative considered: parallel class file

Sidecar `<seq>_classes.txt` per sequence with one class id per detection, in row-aligned order. Saves the re-dump but adds a "remember to load this other file" coupling forever. Rejected — the 1-hour re-dump cost is bounded; long-lived API smell isn't.

---

## Boundary 1 — Detection dump (T3b output)

**File**: `runs/det/yolov8m_zeroshot/<seq>.txt`, one per val sequence.

> ⚠️ **Format is changing in T5 Step 0a.** The current on-disk files use the
> v1 (10-column) format from T3b. After Step 0a's re-dump, they'll be the v2
> (11-column) format documented below. See the **Known issue** section above
> for cost + rationale.

**Format (v2 — post-Step-0a)**: comma-separated, **11 columns**, one detection per row:

```
frame, id, x, y, w, h, conf, class, -1, -1, -1
```

| Column | Type | Meaning | Convention here |
|--------|------|---------|-----------------|
| 1 | int | frame index | **0-indexed** (matches KITTI `image_02/<seq>/000000.png`) |
| 2 | int | track id | `-1` always at detection stage (tracker fills in at boundary 3) |
| 3 | float | bbox top-left x (px) | pixel space, not normalised |
| 4 | float | bbox top-left y (px) | pixel space |
| 5 | float | bbox width (px) | pixel space |
| 6 | float | bbox height (px) | pixel space |
| 7 | float | confidence | [0, 1], 4 decimal places, ≥ `dump_conf` (0.1) |
| 8 | int | **class id** (KITTI scheme) | **`0` = Car, `1` = Pedestrian** (`KITTI_TO_YOLO_FINETUNE` values) |
| 9–11 | int | world coordinates x/y/z | `-1, -1, -1` (N/A for 2D) |

**Format (v1 — currently on disk, deprecated)**: 10 columns, no class column. Format version is recorded in `run_meta.json`'s `format_version` field — consumers reject incompatible producers loudly.

**Sample v2 row** (post-fix; for a Car detection):
```
0,-1,789.15,187.08,442.00,184.19,0.9399,0,-1,-1,-1
```

**Frame indexing**: zero-based, deliberately matching KITTI's convention (and consistent with [`tracking.data.kitti.annotations_to_mot16`](../src/tracking/data/kitti.py)). The wider MOT Challenge tooling assumes 1-indexed in some places — TrackEval's *KITTI* dataset reader expects 0-indexed and is what we use, so the convention holds end-to-end.

---

## Boundary 2 — ByteTrack input (boxmot)

> **⚠️ STATUS: Working assumption. Verify in T5 Step 0.**
>
> The shapes and conventions below are my best reading of boxmot's documented
> API at version 11.x, but the project hasn't installed boxmot yet. T5's first
> implementation step ("Step 0") is to install the pinned version, read the
> `BYTETracker.update()` source, and update this section with verified
> specifics — including answering:
>
> 1. **Column convention**: does `update()` want `[x1, y1, x2, y2, conf, cls]`
>    (xyxy) or `[x, y, w, h, conf, cls]` (xywh)?
> 2. **Frame indexing**: implicit (caller passes one frame at a time) or
>    explicit (frame number is part of the input)?
> 3. **Return shape**: tracks-active-this-frame, or all-tracks-ever-seen?
>
> Until Step 0 lands, **do not write tracker integration code against the
> assumptions below.**

**Library**: `boxmot.BYTETracker` (and similarly `boxmot.BoTSORT` for T7).

**Per-frame call** (assumed):
```python
tracker = BYTETracker(track_thresh=..., match_thresh=..., track_buffer=...)
tracks = tracker.update(dets, img)
```

**Detection input shape**: numpy `ndarray` of dtype `float32`, shape `(N, 6)`:

```
[[x1, y1, x2, y2, conf, cls], ...]
```

| Column | Meaning |
|--------|---------|
| 0–3 | bbox in **xyxy** (top-left + bottom-right pixel coords) — note: **not** xywh |
| 4 | confidence ∈ [0, 1] |
| 5 | class id (int as float) |

**Conversion from boundary 1**:

```python
# Our dump has frame, id, x, y, w, h, conf — we need xyxy + cls
x1, y1 = x, y
x2, y2 = x + w, y + h
det_array = np.array([[x1, y1, x2, y2, conf, cls], ...], dtype=np.float32)
```

The `cls` column is the missing piece flagged in Boundary 1.5.

**The `img` argument** is the actual frame as a numpy array (HWC BGR uint8). Used by some tracker implementations (BoT-SORT) for camera motion compensation; ByteTrack ignores it. We pass `cv2.imread(frame_path)`.

**Note on `boxmot` API stability**: API exact shapes should be verified against the installed version's source before T5 code lands. Pin `boxmot>=11.0,<12.0` in pyproject.toml to avoid silent breakage on minor releases.

---

## Boundary 3 — ByteTrack output (boxmot)

> **⚠️ STATUS: Working assumption. Verify in T5 Step 0.** Return shape and
> column order are taken from boxmot's README at the time of writing; pin to
> a tested version (Refinement 2 in the T5 plan) and update this section
> after reading source.

`tracker.update(dets, img)` returns a numpy `ndarray` of dtype `float32`, shape `(M, 8)` (M ≤ N — only currently-active tracks):

```
[[x1, y1, x2, y2, track_id, conf, cls, det_index], ...]
```

| Column | Meaning |
|--------|---------|
| 0–3 | bbox in xyxy (smoothed by Kalman if applicable) |
| 4 | track id (1-indexed; tracker assigns) |
| 5 | confidence (carried through from input) |
| 6 | class id |
| 7 | original detection index in the input array (or -1 if interpolated) |

**Track id is per-tracker-instance, not global.** A new `BYTETracker` instance starts at id=1 for each sequence, so cross-sequence id collisions are normal and expected — TrackEval handles per-sequence eval correctly.

**Writing back to MOT16**: same 11-column v2 format as Boundary 1, with the tracker filling in the `track_id` column:

```
frame, track_id, x1, y1, (x2-x1), (y2-y1), conf, cls, -1, -1, -1
```

Output goes to `runs/track/bytetrack/<seq>.txt`. Frame index, xywh conversion, class column, and 0-indexing all match Boundary 1's v2 format. T7's BoT-SORT writes the same shape to `runs/track/botsort/<seq>.txt` — identical format for clean ablation.

---

## Boundary 4 — TrackEval input format (KITTI 2D MOT scope)

TrackEval has a dedicated KITTI dataset reader (`trackeval.datasets.Kitti2DBox`) which expects a specific layout and format — **different from the generic MOT16 used at boundaries 1 and 3**.

### Directory layout

```
data/trackers/kitti_2d_box_train/<tracker_name>/data/
├── 0001.txt
├── 0006.txt
├── 0013.txt
├── 0017.txt
└── 0019.txt
```

Per-tracker subdirectory under `data/trackers/kitti_2d_box_train/`. The trailing `data/` directory is a TrackEval convention (alongside `seqmap.txt` etc. at the parent level).

### Per-row format

**Space-separated, 17 columns** — the same shape as KITTI's GT `label_02/<seq>.txt` plus a trailing `score`:

```
frame track_id type truncated occluded alpha x1 y1 x2 y2 height width length x y z rotation_y score
```

| Column | Type | Meaning | Filled with what |
|--------|------|---------|------------------|
| 1 | int | frame | 0-indexed (matches KITTI) |
| 2 | int | track id | from boundary 3 |
| 3 | string | object type | **`Car`** or **`Pedestrian`** (case-sensitive); `Cyclist` not in T3b output |
| 4 | float | truncated | `0` (we don't track this) |
| 5 | int | occluded | `0` |
| 6 | float | alpha (observation angle) | `-10` (sentinel for unknown) |
| 7–10 | float | bbox xyxy in image coords | from boundary 3 |
| 11–13 | float | 3D dimensions h, w, l | `-1, -1, -1` (2D-only) |
| 14–16 | float | 3D location x, y, z | `-1000, -1000, -1000` (2D-only sentinels) |
| 17 | float | rotation_y | `-10` |
| 18 | float | score (confidence) | from boundary 3 |

Wait — that's 18 columns. The KITTI tracker-output spec is 17 (no score column on GT). Let me check…

**Correction**: TrackEval's KITTI tracker-output format is **17 columns** total when the score is present, because tracker output doesn't have a `truncated` column the way GT does. The exact column list to use:

```
frame track_id type 0 0 -10 x1 y1 x2 y2 -1 -1 -1 -1000 -1000 -1000 -10 score
```

The `0 0 -10` for trunc/occ/alpha and the `-1 -1 -1 -1000 -1000 -1000 -10` for 3D fields are the canonical "2D-only" placeholders the KITTI tracker eval expects. **Verify exact column order against TrackEval's `kitti_2d_box.py` parser before T5 code lands** — TrackEval is strict and silently mis-parses if columns drift.

### Conversion from Boundary 3

For each row in our `runs/track/bytetrack/<seq>.txt` (10-column MOT16):

```python
type_str = {0: "Car", 1: "Pedestrian"}[int(cls)]
trackeval_row = (
    f"{frame} {track_id} {type_str} 0 0 -10 "
    f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
    f"-1 -1 -1 -1000 -1000 -1000 -10 "
    f"{score:.4f}"
)
```

A separate writer module (`tracking.eval.format`) is the right home for this — keeps boundary-3 output (MOT16) decoupled from boundary-4 input (TrackEval-KITTI).

---

## Boundary 5 — Ground-truth conversion

TrackEval's `Kitti2DBox` dataset reader can read **KITTI's `label_02/<seq>.txt` directly** — no conversion needed. The expected GT layout:

```
data/gt/kitti_2d_box_train/label_02/
├── 0001.txt
├── 0006.txt
├── ...
```

These are **the same files** as `data/kitti_tracking/training/label_02/<seq>.txt` from T2's download. Conversion = symlink (or copy) into TrackEval's expected directory layout.

### Class name convention

Both GT and tracker-output use the **string** class names: `Car`, `Pedestrian`, `Cyclist` (case-sensitive). T3b drops Cyclist, so val GT entries with `obj_class == "Cyclist"` are present in `label_02` but won't match any tracker output rows — TrackEval handles this as "missed detections of the cyclist class" and the per-class HOTA simply reports zero for Cyclist on our zero-shot run. This is the expected and correct behaviour.

### Frame indexing on GT

KITTI's `label_02/<seq>.txt` rows start at frame 0. Matches our pipeline. No off-by-one to worry about.

### `seqmap.txt`

TrackEval needs a `seqmap.txt` listing the val sequences and frame counts. Generate at:

```
data/seqmaps/kitti_2d_box_val.txt
```

Format (one header line + one row per sequence):
```
name
0001
0006
0013
0017
0019
```

(Newer TrackEval versions also accept just the bare sequence list — verify against installed version.)

---

## Quick reference: format-aware checklist for T5

When implementing T5, the implementation must:

- [ ] Address Boundary 1.5 (class column) — pick option A1, A2, A3, or B and document
- [ ] Read MOT16 from boundary 1 with the right column count
- [ ] Convert xywh → xyxy at boundary 2
- [ ] Reset tracker state between sequences (independent track id streams)
- [ ] Convert xyxy → xywh at boundary 3 when writing back
- [ ] Carry class column from input through to output
- [ ] Round-trip test: write a file at boundary 3, read it back, assert no data loss

T6 (TrackEval harness, separate task) handles boundaries 4 and 5 directly. T5 just needs to produce boundary 3 output cleanly.

---

## Format version

Implementations must record the format version they emit/consume. Suggested:

- T3b detection dump (current on-disk): `format_version: "mot16-kitti-v1"` — 10 columns, no class. **Deprecated by T5 Step 0a.**
- T3b detection dump (post-Step-0a re-dump): `format_version: "mot16-kitti-v2"` — 11 columns, class id at slot 8, world-coord sentinels at slots 9–11.
- T5 tracker output: `format_version: "mot16-kitti-v2"` — same 11-column shape as v2 detections, with track ids filled in slot 2.
- T7 tracker output: `format_version: "mot16-kitti-v2"` — same.

This goes in `run_meta.json` so consumers can reject incompatible producers loudly. T5's tracker reader should refuse to load any file with `format_version != "mot16-kitti-v2"` — fail fast rather than silently misinterpret a missing column.
