"""Streamlit demo — task T8.

Plan:
1. Sidebar: pick KITTI val sequence (0001, 0006, 0013, 0017, 0019),
   pick tracker (bytetrack | botsort).
2. Main panel:
   - Annotated mp4 (precomputed by viz.write_video, cached on disk).
   - Metrics card: HOTA / MOTA / IDF1 from docs/results.md.
   - Per-class breakdown.
3. About section: link to GitHub, paper, model card.

Implementation notes:
- Don't run inference on demand — too slow on Render free tier. Pre-render
  videos at build time and ship them in the container, or cache to /tmp on
  first request.
- Use st.video for mp4 playback.
- Render free tier has a 512 MB RAM limit. Skip ReID/embeddings here; the
  demo is read-only.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="KITTI MOT Tracking", page_icon="🚗", layout="wide")

st.title("Multi-Object Tracking on KITTI")
st.caption("ByteTrack vs BoT-SORT on YOLOv8 detections. Portfolio project by @ai.industrial.")

st.warning(
    "Demo not yet implemented — task T8. See `streamlit_app/app.py` for the plan, "
    "or run locally with `make demo` once T3–T7 land."
)

with st.sidebar:
    st.header("Coming soon")
    st.markdown(
        "- Sequence picker\n"
        "- Tracker picker (ByteTrack / BoT-SORT)\n"
        "- Annotated video playback\n"
        "- Live metrics panel"
    )
