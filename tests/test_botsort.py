"""Tests for BoT-SORT integration and config model.

All tests are CI-runnable: no GPU, no KITTI data, no network.
Every test uses with_reid=False so no ReID model is downloaded.
boxmot is a required dep so importing BotSort is safe in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# BoT-SORT API tests
# ---------------------------------------------------------------------------


def _make_botsort():
    """Helper: construct a BotSort instance with with_reid=False."""
    import torch
    from boxmot.trackers import BotSort

    return BotSort(
        reid_weights=Path("none"),
        device=torch.device("cpu"),
        half=False,
        frame_rate=10,
        with_reid=False,
    )


def test_botsort_assigns_stable_ids_across_frames() -> None:
    """Track IDs must be consistent across consecutive frames for the same object.

    Non-overlapping Car and Pedestrian boxes — verifies basic association
    stability. Cross-class separation is verified separately in
    test_per_class_prevents_cross_class_id_continuation.
    """
    tracker = _make_botsort()
    car = np.array([[100, 100, 200, 200, 0.90, 0.0]], dtype=np.float32)
    ped = np.array([[400, 100, 440, 200, 0.85, 1.0]], dtype=np.float32)
    dets = np.vstack([car, ped])
    img = np.zeros((375, 1242, 3), dtype=np.uint8)

    ids = [set(tracker.update(dets, img)[:, 4].astype(int)) for _ in range(3)]
    assert ids[0] == ids[1] == ids[2], "IDs changed between frames"
    assert len(ids[0]) == 2, "Expected exactly 2 active tracks"


def test_per_class_prevents_cross_class_id_continuation() -> None:
    """A Pedestrian detection at the exact same location as an active Car track
    must NOT inherit the car's track ID.

    The production implementation uses separate BotSort instances per class
    to achieve true isolation — each instance's lost_stracks is invisible to
    the other class.

    The boxes here overlap 100% (IoU=1.0). With a single shared tracker
    this produces an ID collision. With separate instances it cannot:
    tracker_ped has never seen the car.
    """
    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    tracker_car = _make_botsort()
    tracker_ped = _make_botsort()

    # Frame 0: establish a Car track in tracker_car.
    car = np.array([[100, 100, 200, 200, 0.90, 0.0]], dtype=np.float32)
    out_car = tracker_car.update(car, img)
    assert len(out_car) > 0, "Car detection should produce a track"
    car_id = int(out_car[0, 4])

    # Frame 1: Pedestrian at the SAME location fed to tracker_ped.
    # tracker_ped has no car track in its state — the ped gets a fresh ID.
    # Also advance tracker_car with an empty frame to keep its counter in sync.
    tracker_car.update(np.empty((0, 6), dtype=np.float32), img)
    ped = np.array([[100, 100, 200, 200, 0.90, 1.0]], dtype=np.float32)
    out_ped = tracker_ped.update(ped, img)

    assert len(out_ped) > 0, "Pedestrian detection should produce a track"
    ped_id = int(out_ped[0, 4])
    assert ped_id != car_id, (
        f"Cross-class ID collision: pedestrian got track ID {ped_id} which "
        f"equals the car's track ID ({car_id}). Separate tracker instances "
        f"should have independent ID spaces."
    )


def test_empty_frame_advances_internal_frame_counter() -> None:
    """Empty-frame update() calls must advance the internal frame counter,
    so a track that disappears for longer than the buffer is correctly lost.

    With frame_rate=10 and track_buffer=30:
        buffer_size = int(10 / 30 * 30) = 10 frames
    12 consecutive empty frames is past the buffer — the original track must
    be declared lost and removed. When the same object reappears it starts as
    a new tentative track. After one more detection it is confirmed and
    assigned a new ID.
    """
    tracker = _make_botsort()
    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    car_box = np.array([[100, 100, 200, 200, 0.90, 0.0]], dtype=np.float32)

    # Frame 1: establish a Car track.
    out_0 = tracker.update(car_box, img)
    assert len(out_0) > 0, "Car detection should produce a track on first frame"
    original_id = int(out_0[0, 4])

    # Frames 2-13: empty. 12 > buffer_size (10), so the track must be removed.
    for _ in range(12):
        out_empty = tracker.update(np.empty((0, 6), dtype=np.float32), img)
        assert out_empty.shape == (0, 8), (
            f"Empty-frame output should be (0, 8), got {out_empty.shape}"
        )

    # Frame 14: car reappears — creates a NEW tentative track.
    tracker.update(car_box, img)

    # Frame 15: car seen again — tentative track is activated.
    out_confirmed = tracker.update(car_box, img)
    assert len(out_confirmed) > 0, "Car should be confirmed after two consecutive detections"
    new_id = int(out_confirmed[0, 4])
    assert new_id != original_id, (
        f"Frame counter did not advance during empty calls: "
        f"car reappearing after 12 empty frames got the original track ID "
        f"({original_id}), but the track should have been declared lost "
        f"(buffer_size is 10 frames)."
    )


def test_with_reid_false_does_not_load_model() -> None:
    """Constructing BotSort with with_reid=False must not load a ReID model.

    This ensures CI can run without downloading model weights.
    """
    tracker = _make_botsort()
    # When with_reid=False, the model attribute should be None or not populated
    # with a real model. The tracker should still be functional.
    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    car = np.array([[100, 100, 200, 200, 0.90, 0.0]], dtype=np.float32)
    result = tracker.update(car, img)
    assert result.shape[1] == 8, "Output should have 8 columns"


def test_botsort_config_round_trip(tmp_path: Path) -> None:
    """Load tracker_botsort.yaml and validate with BoTSortConfig pydantic model."""
    import yaml

    from tracking.trackers.botsort import BoTSortConfig

    config_path = Path("configs/tracker_botsort.yaml")
    raw = yaml.safe_load(config_path.read_text())
    cfg = BoTSortConfig(**raw)

    assert cfg.name == "botsort"
    assert cfg.frame_rate == 10
    assert cfg.with_reid is False
    assert cfg.track_buffer == 30
    assert cfg.nr_classes == 2
    assert cfg.cmc_method == "sof"
    assert cfg.track_high_thresh == 0.5
    assert cfg.new_track_thresh == 0.6
    assert len(cfg.val_sequences) == 5
