import os
import tempfile
from pathlib import Path

import streamlit as st

from analyzer import analyze_video, make_excel
from youtube_utils import download_youtube_video, is_youtube_url

st.set_page_config(page_title="Video Content Analyzer", page_icon="🎬", layout="wide")

st.title("🎬 Video Content Analyzer")
st.caption("Research-oriented audiovisual content analysis — local video or YouTube URL")

with st.sidebar:
    st.header("Detection settings")
    threshold = st.slider(
        "Cut sensitivity threshold",
        min_value=0.20,
        max_value=0.90,
        value=0.48,
        step=0.02,
        help="Lower = more cuts detected; higher = only stronger visual changes count as cuts.",
    )
    min_shot = st.slider(
        "Minimum shot duration (seconds)",
        min_value=0.20,
        max_value=2.00,
        value=0.45,
        step=0.05,
    )
    st.info(
        "For thesis use, keep the same settings across your full sample "
        "after calibrating them on a small validation subset."
    )

source_mode = st.radio(
    "Video source",
    ["Upload Video", "YouTube URL"],
    horizontal=True,
)

uploaded = None
youtube_url = ""

if source_mode == "Upload Video":
    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "m4v", "avi", "mkv", "webm"],
    )
    if uploaded is not None:
        st.video(uploaded)
else:
    youtube_url = st.text_input(
        "Paste a YouTube link",
        placeholder="https://www.youtube.com/watch?v=…  or  https://youtu.be/…",
    ).strip()
    if youtube_url:
        if is_youtube_url(youtube_url):
            st.caption("The video will be downloaded temporarily for analysis and deleted afterward.")
        else:
            st.warning("That does not look like a YouTube link.")

can_analyze = (source_mode == "Upload Video" and uploaded is not None) or (
    source_mode == "YouTube URL" and bool(youtube_url) and is_youtube_url(youtube_url)
)

if can_analyze and st.button("Analyze video", type="primary", use_container_width=True):
    progress_bar = st.progress(0.0)
    status = st.empty()

    def update(p, text):
        progress_bar.progress(float(max(0.0, min(1.0, p))))
        status.write(text)

    tmp_path = None
    tmp_dir_obj = None
    display_name = "video"
    source_metadata = None

    try:
        if source_mode == "Upload Video":
            suffix = Path(uploaded.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            display_name = uploaded.name

            analysis_progress = update
        else:
            tmp_dir_obj = tempfile.TemporaryDirectory(prefix="video_content_analyzer_")
            tmp_path, source_metadata = download_youtube_video(
                youtube_url,
                tmp_dir_obj.name,
                progress=update,
            )
            display_name = f"{source_metadata.get('title', 'youtube_video')}{Path(tmp_path).suffix}"

            def analysis_progress(p, text):
                # Reserve the first quarter of the progress bar for downloading.
                update(0.25 + (0.75 * p), text)

        summary_df, shots_df = analyze_video(
            tmp_path,
            threshold=threshold,
            min_shot_sec=min_shot,
            progress=analysis_progress,
        )

        st.success("Analysis complete.")

        if source_metadata:
            st.subheader("YouTube source")
            m1, m2 = st.columns(2)
            m1.write(f"**Title:** {source_metadata.get('title', '')}")
            m2.write(f"**Channel:** {source_metadata.get('uploader', '') or '—'}")

        if not shots_df.empty:
            c1, c2, c3, c4 = st.columns(4)
            summary_map = dict(zip(summary_df["metric"], summary_df["value"]))
            c1.metric("Detected shots", int(summary_map.get("shot_count", 0)))
            c2.metric("Avg. shot length", f'{summary_map.get("average_shot_length_sec", 0):.2f} s')
            c3.metric("Video duration", f'{summary_map.get("video_duration_sec", 0):.1f} s')
            c4.metric("Mean faces/shot", f'{summary_map.get("mean_faces_per_shot_est", 0):.2f}')

        st.subheader("Video summary")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.subheader("Shot-level data")
        st.dataframe(shots_df, use_container_width=True, hide_index=True)

        excel_bytes = make_excel(summary_df, shots_df, display_name)
        stem = Path(display_name).stem
        st.download_button(
            "Download Excel report",
            data=excel_bytes,
            file_name=f"{stem}_content_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
    finally:
        if source_mode == "Upload Video" and tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if tmp_dir_obj is not None:
            tmp_dir_obj.cleanup()
