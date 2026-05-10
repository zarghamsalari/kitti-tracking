"""KITTI MOT Tracking Demo — Streamlit app (Task T8/T9).

4-cell ablation: {ByteTrack, BoT-SORT} x {zero-shot, fine-tuned} YOLOv8m.
Sequence picker -> detector picker -> tracker picker -> annotated video + metrics.
Consumes existing track files from runs/track/<tracker>/<seq>.txt.
Does NOT re-run trackers or detectors.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data/kitti_tracking")
TRACKS_ROOT = Path("runs/track")
DEMO_CACHE = Path("runs/demo")

VAL_SEQUENCES = ["0001", "0006", "0013", "0017", "0019"]
TRACKERS = {"ByteTrack": "bytetrack", "BoT-SORT": "botsort"}

# Pre-computed metrics from eval runs (macro averages, equal class weights).
# Hardcoded to avoid re-running TrackEval on every demo load.
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
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_tracks(detector: str, tracker: str, seq: str) -> dict[int, list[list[float]]]:
    """Load track file and return as serialisable dict (for st.cache_data)."""
    from tracking.trackers.mot16_io import read_mot16_v2_tracks

    path = get_track_dir(detector, tracker) / f"{seq}.txt"
    if not path.exists():
        st.error(f"Track file not found: {path}")
        return {}

    tracks_np = read_mot16_v2_tracks(path)
    return {k: v.tolist() for k, v in tracks_np.items()}


def get_video_path(detector: str, tracker: str, seq: str) -> str | None:
    """Render annotated mp4 if not cached, return path string."""
    import numpy as np

    from tracking.viz.overlay import write_video

    out_path = DEMO_CACHE / f"{detector}_{tracker}_{seq}.mp4"
    if out_path.exists():
        return str(out_path)

    seq_dir = DATA_ROOT / "training" / "image_02" / seq
    if not seq_dir.exists():
        return None

    tracks_raw = load_tracks(detector, tracker, seq)
    if not tracks_raw:
        return None

    # Convert back to numpy for write_video
    tracks_np = {k: np.array(v, dtype=np.float32) for k, v in tracks_raw.items()}
    write_video(seq_dir, tracks_np, out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="KITTI MOT Demo", layout="wide")
    st.title("KITTI Multi-Object Tracking Demo")
    st.caption(
        "4-cell ablation: {ByteTrack, BoT-SORT} \u00d7 {zero-shot, fine-tuned} YOLOv8m. "
        "Same detection inputs within each detector condition. "
        "Headline: fine-tuning lifts MOTA 5\u00d7 on ByteTrack but HOTA stays flat "
        "\u2014 see results.md."
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
        run = st.button("Render video", type="primary")

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
            video_path = get_video_path(detector, tracker, seq)
            if video_path:
                st.video(video_path)
            else:
                st.error(
                    f"Could not render video. Check that KITTI data exists at "
                    f"{DATA_ROOT / 'training' / 'image_02' / seq}"
                )
        else:
            st.info("Select a detector, tracker, and sequence, then click **Render video**.")


if __name__ == "__main__":
    main()
