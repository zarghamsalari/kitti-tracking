"""Tests for ByteTrack integration and MOT16 v2 I/O.

All tests are CI-runnable: no GPU, no KITTI data, no network.
boxmot is a required dep so importing ByteTrack is safe in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# ByteTrack API tests
# ---------------------------------------------------------------------------


def test_bytetrack_assigns_stable_ids_across_frames() -> None:
    """Track IDs must be consistent across consecutive frames for the same object.

    Non-overlapping Car and Pedestrian boxes — verifies basic association
    stability. Cross-class separation is verified separately in
    test_per_class_prevents_cross_class_id_continuation.
    """
    from boxmot.trackers import ByteTrack

    tracker = ByteTrack(frame_rate=10)
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

    boxmot 18.0.0's per_class=True flag has incomplete isolation: lost_stracks
    is shared across class pools, so a ped can match a recently-lost car track.
    The production implementation uses separate ByteTrack instances per class
    to achieve true isolation — each instance's lost_stracks is invisible to
    the other class.

    The boxes here overlap 100% (IoU=1.0). With a single shared tracker and
    per_class=False this produces an ID collision. With separate instances
    it cannot: tracker_ped has never seen the car.
    """
    from boxmot.trackers import ByteTrack

    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    tracker_car = ByteTrack(frame_rate=10)
    tracker_ped = ByteTrack(frame_rate=10)

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

    With frame_rate=10 and track_buffer=25:
        buffer_size = int(10 / 30 * 25) = 8 frames
    10 consecutive empty frames is past the buffer — the original track must
    be declared lost and removed. When the same object reappears it starts as
    a new tentative track. After one more detection it is confirmed (ByteTrack
    requires two consecutive detections to activate a track beyond frame 1)
    and assigned a new ID.

    If empty calls did NOT advance the counter, the original track would never
    age out and the reappearing object would reuse the same ID — which would
    mean the tracker silently ignores long occlusions.
    """
    from boxmot.trackers import ByteTrack

    tracker = ByteTrack(frame_rate=10, track_buffer=25)
    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    car_box = np.array([[100, 100, 200, 200, 0.90, 0.0]], dtype=np.float32)

    # Frame 1 (frame_count=1 inside update): establish a Car track.
    # ByteTrack sets is_activated=True only when frame_id==1, so this is
    # the only frame where a brand-new track is output immediately.
    out_0 = tracker.update(car_box, img)
    assert len(out_0) > 0, "Car detection should produce a track on first frame"
    original_id = int(out_0[0, 4])

    # Frames 2-11: empty. 10 > buffer_size (8), so the track must be removed.
    for _ in range(10):
        out_empty = tracker.update(np.empty((0, 6), dtype=np.float32), img)
        assert out_empty.shape == (0, 8), (
            f"Empty-frame output should be (0, 8), got {out_empty.shape}"
        )

    # Frame 12: car reappears — creates a NEW tentative track (is_activated=False
    # because frame_id != 1). Not output yet.
    tracker.update(car_box, img)

    # Frame 13: car seen again — tentative track is activated via STrack.update()
    # which sets is_activated=True. Now it appears in output with a new ID.
    out_confirmed = tracker.update(car_box, img)
    assert len(out_confirmed) > 0, "Car should be confirmed after two consecutive detections"
    new_id = int(out_confirmed[0, 4])
    assert new_id != original_id, (
        f"Frame counter did not advance during empty calls: "
        f"car reappearing after 10 empty frames got the original track ID "
        f"({original_id}), but the track should have been declared lost "
        f"(buffer_size is 8 frames)."
    )


# ---------------------------------------------------------------------------
# MOT16 I/O tests
# ---------------------------------------------------------------------------


def test_mot16_v2_round_trip(tmp_path: Path) -> None:
    """Write a known (frame, track_id, xywh, conf, cls) array to v2 file,
    read it back, and assert the xyxy conversion round-trips losslessly.
    """
    from tracking.trackers.mot16_io import read_mot16_v2, write_mot16_v2

    # Write a run_meta.json so the reader's format-version check passes.
    (tmp_path / "run_meta.json").write_text(json.dumps({"format_version": "mot16-kitti-v2"}))

    # Build fake tracker output: (M, 8) [x1, y1, x2, y2, track_id, conf, cls, det_ind]
    tracks: dict[int, np.ndarray] = {
        0: np.array([[10.0, 20.0, 110.0, 120.0, 1.0, 0.91, 0.0, 0.0]], dtype=np.float32),
        1: np.array([[15.0, 25.0, 115.0, 125.0, 1.0, 0.88, 1.0, 0.0]], dtype=np.float32),
    }

    out_path = tmp_path / "0001.txt"
    n_rows = write_mot16_v2(tracks, out_path)
    assert n_rows == 2

    result = read_mot16_v2(out_path)
    assert set(result.keys()) == {0, 1}

    # Frame 0: x1=10, y1=20, x2=110, y2=120, conf=0.91, cls=0
    np.testing.assert_allclose(result[0][0], [10.0, 20.0, 110.0, 120.0, 0.91, 0.0], atol=1e-2)
    # Frame 1: x1=15, y1=25, x2=115, y2=125, conf=0.88, cls=1
    np.testing.assert_allclose(result[1][0], [15.0, 25.0, 115.0, 125.0, 0.88, 1.0], atol=1e-2)


def test_read_mot16_v2_tracks_preserves_track_id(tmp_path: Path) -> None:
    """track_id must appear at column 4 of the (N,7) output array.

    read_mot16_v2 (the detection reader) discards track_id because detections
    always have track_id=-1. The tracks reader must NOT discard it — evaluating
    HOTA with all-zero IDs gives HOTA=0 with no runtime error.
    """
    from tracking.trackers.mot16_io import read_mot16_v2_tracks

    (tmp_path / "run_meta.json").write_text(json.dumps({"format_version": "mot16-kitti-v2"}))
    (tmp_path / "0001.txt").write_text("0,7,10.00,20.00,100.00,50.00,0.9000,0,-1,-1,-1\n")
    result = read_mot16_v2_tracks(tmp_path / "0001.txt")
    assert 0 in result
    row = result[0][0]
    assert row.shape == (7,)
    assert row[4] == 7.0, "track_id must be preserved at column 4"
    np.testing.assert_allclose(row[0], 10.0, atol=1e-2)  # x1
    np.testing.assert_allclose(row[2], 110.0, atol=1e-2)  # x2 = x + w


def test_read_mot16_v2_refuses_v1_format_version(tmp_path: Path) -> None:
    """read_mot16_v2 must raise ValueError when the adjacent run_meta.json
    reports format_version 'mot16-kitti-v1', not silently parse stale files.

    Silent parse of a v1 file (no cls column at parts[7]) produces all-(-1)
    class ids, which corrupts per-class HOTA without a runtime error.
    """
    from tracking.trackers.mot16_io import read_mot16_v2

    (tmp_path / "run_meta.json").write_text(json.dumps({"format_version": "mot16-kitti-v1"}))
    det_file = tmp_path / "0001.txt"
    det_file.write_text("0,-1,100.00,200.00,50.00,80.00,0.9000,-1,-1,-1\n")

    with pytest.raises(ValueError, match="mot16-kitti-v1"):
        read_mot16_v2(det_file)
