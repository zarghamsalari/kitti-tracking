"""KITTI MOT Tracking Demo -- Streamlit app.

4-cell ablation: {ByteTrack, BoT-SORT} x {zero-shot, fine-tuned} YOLOv8m.
Sequence picker -> detector picker -> tracker picker -> annotated video + metrics.

In production (default): serves pre-rendered MP4s from
streamlit_app/static/videos/. No raw KITTI frames required.

For local dev with raw frames: set DEMO_LIVE_RENDER=1 to fall back
to on-the-fly rendering via overlay.py when a cached video is missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATIC_VIDEOS = Path(__file__).parent / "static" / "videos"
DATA_ROOT = Path("data/kitti_tracking")
TRACKS_ROOT = Path("runs/track")

LIVE_RENDER = os.environ.get("DEMO_LIVE_RENDER", "0") == "1"

VAL_SEQUENCES = ["0001", "0006", "0013", "0017", "0019"]
TRACKERS = {"ByteTrack": "bytetrack", "BoT-SORT": "botsort"}

# Pre-computed metrics from eval runs (macro averages, equal class weights).
METRICS: dict[tuple[str, str], dict[str, float | int]] = {
    ("zeroshot", "bytetrack"): {
        "HOTA": 0.41,
        "MOTA": 0.07,
        "IDF1": 0.54,
        "AssA": 0.48,
        "DetA": 0.36,
        "IDSw": 152,
        "Frag": 384,
    },
    ("zeroshot", "botsort"): {
        "HOTA": 0.47,
        "MOTA": 0.24,
        "IDF1": 0.61,
        "AssA": 0.53,
        "DetA": 0.42,
        "IDSw": 88,
        "Frag": 347,
    },
    ("finetuned", "bytetrack"): {
        "HOTA": 0.41,
        "MOTA": 0.36,
        "IDF1": 0.55,
        "AssA": 0.44,
        "DetA": 0.40,
        "IDSw": 159,
        "Frag": 284,
    },
    ("finetuned", "botsort"): {
        "HOTA": 0.45,
        "MOTA": 0.41,
        "IDF1": 0.59,
        "AssA": 0.49,
        "DetA": 0.41,
        "IDSw": 86,
        "Frag": 235,
    },
}


def get_track_dir(detector: str, tracker: str) -> Path:
    """Resolve tracker output directory for a given detector condition."""
    suffix = "_finetuned" if detector == "finetuned" else ""
    return TRACKS_ROOT / f"{tracker}{suffix}"


# ---------------------------------------------------------------------------
# Video resolution
# ---------------------------------------------------------------------------


def get_video_path(detector: str, tracker: str, seq: str) -> str | None:
    """Return path to a demo video, or None if unavailable.

    1. Check pre-rendered static videos (always available in Docker).
    2. If DEMO_LIVE_RENDER=1, fall back to on-the-fly rendering from
       raw KITTI frames + tracker output.
    3. Otherwise return None (production: clean error shown to user).
    """
    # 1. Pre-rendered cache
    static = STATIC_VIDEOS / f"{detector}_{tracker}_{seq}.mp4"
    if static.exists():
        return str(static)

    # 2. Live rendering (local dev only)
    if not LIVE_RENDER:
        return None

    from tracking.trackers.mot16_io import read_mot16_v2_tracks
    from tracking.viz.overlay import write_video

    seq_dir = DATA_ROOT / "training" / "image_02" / seq
    track_file = get_track_dir(detector, tracker) / f"{seq}.txt"

    if not seq_dir.exists() or not track_file.exists():
        return None

    out_path = STATIC_VIDEOS / f"{detector}_{tracker}_{seq}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tracks = read_mot16_v2_tracks(track_file)
    write_video(seq_dir, tracks, out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="KITTI MOT Demo", layout="wide")
    st.title("KITTI Multi-Object Tracking Demo")
    st.caption(
        "4-cell ablation: {ByteTrack, BoT-SORT} x {zero-shot, fine-tuned} YOLOv8m. "
        "Same detection inputs within each detector condition. "
        "Headline: fine-tuning lifts MOTA 5x on ByteTrack but HOTA stays flat "
        "-- see results.md."
    )

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        detector = st.radio(
            "Detector",
            ["zeroshot", "finetuned"],
            format_func=lambda x: "Zero-shot YOLOv8m" if x == "zeroshot" else "Fine-tuned YOLOv8m",
        )
        tracker_label = st.selectbox("Tracker", list(TRACKERS.keys()))
        seq = st.selectbox("Sequence", VAL_SEQUENCES)
        run = st.button("Show video", type="primary")

    tracker = TRACKERS[tracker_label]

    # Metrics panel
    col_video, col_metrics = st.columns([3, 1])

    with col_metrics:
        st.subheader("Metrics")
        m = METRICS.get((detector, tracker), {})
        if m:
            st.metric("HOTA", f"{m['HOTA']:.2f}")
            st.metric("MOTA", f"{m['MOTA']:.2f}")
            st.metric("IDF1", f"{m['IDF1']:.2f}")
            st.metric("AssA", f"{m['AssA']:.2f}")
            st.metric("DetA", f"{m['DetA']:.2f}")
            st.divider()
            st.metric("ID Switches", m["IDSw"])
            st.metric("Fragmentations", m["Frag"])
        else:
            st.warning("No metrics available for this configuration.")

        st.divider()
        st.caption(
            "Rate metrics: macro average (equal class weights). "
            "Counts: summed across Car + Pedestrian."
        )

    # Video panel
    with col_video:
        if run:
            with st.spinner("Loading video..."):
                video_path = get_video_path(detector, tracker, seq)
            if video_path:
                st.video(video_path)
            else:
                st.error(
                    "Video unavailable for this configuration. "
                    "Pre-rendered videos may not be bundled in this deployment."
                )
        else:
            st.info("Select a detector, tracker, and sequence, then click **Show video**.")


if __name__ == "__main__":
    main()
