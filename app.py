import os
import tempfile
from pathlib import Path

import streamlit as st

from analyzer import (
    analyze_video,
    make_excel,
    SHOT_ANALYSIS,
    WARM_COLOR,
    SATURATION,
    CONTRAST,
)
from youtube_utils import download_youtube_video, is_youtube_url

st.set_page_config(page_title="Video Content Analyzer", page_icon="🎬", layout="wide")

st.title("🎬 Video Content Analyzer")
st.caption("Research-oriented audiovisual content analysis — local video or YouTube URL")

ANALYSIS_OPTIONS = {
    "analysis_shot": ("Shot / cut analysis", SHOT_ANALYSIS),
    "analysis_warm": ("Warm-color palette", WARM_COLOR),
    "analysis_saturation": ("Color saturation", SATURATION),
    "analysis_contrast": ("Contrast", CONTRAST),
}

# Initialize selection state once.
if "analysis_select_all" not in st.session_state:
    st.session_state.analysis_select_all = True
for key in ANALYSIS_OPTIONS:
    if key not in st.session_state:
        st.session_state[key] = True


def _select_all_changed():
    value = bool(st.session_state.analysis_select_all)
    for key in ANALYSIS_OPTIONS:
        st.session_state[key] = value


def _individual_changed():
    st.session_state.analysis_select_all = all(
        bool(st.session_state[key]) for key in ANALYSIS_OPTIONS
    )


with st.sidebar:
    st.header("Analyses to run")
    st.checkbox(
        "Select all",
        key="analysis_select_all",
        on_change=_select_all_changed,
    )
    st.divider()
    for key, (label, _) in ANALYSIS_OPTIONS.items():
        st.checkbox(label, key=key, on_change=_individual_changed)

    selected_analyses = {
        analysis_id
        for key, (_, analysis_id) in ANALYSIS_OPTIONS.items()
        if st.session_state[key]
    }

    threshold = 0.48
    min_shot = 0.45
    if SHOT_ANALYSIS in selected_analyses:
        st.divider()
        st.header("Detection settings")
        threshold = st.slider(
            "Shot-change threshold",
            min_value=0.20,
            max_value=0.90,
            value=0.48,
            step=0.02,
            help="Unitless detection threshold. Lower values detect more visual changes as cuts; higher values detect only stronger changes.",
        )
        min_shot = st.slider(
            "Minimum shot duration (seconds)",
            min_value=0.20,
            max_value=2.00,
            value=0.45,
            step=0.05,
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

if not selected_analyses:
    st.warning("Select at least one analysis from the sidebar.")

has_source = (source_mode == "Upload Video" and uploaded is not None) or (
    source_mode == "YouTube URL" and bool(youtube_url) and is_youtube_url(youtube_url)
)
can_analyze = has_source and bool(selected_analyses)

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
                update(0.25 + (0.75 * p), text)

        summary_df, shots_df, analysis_metadata = analyze_video(
            tmp_path,
            analyses=selected_analyses,
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

        summary_map = dict(zip(summary_df["Measure"], summary_df["Value"]))

        # Show only top metrics that correspond to selected analyses.
        metric_items = []
        if SHOT_ANALYSIS in selected_analyses:
            metric_items.append(
                ("Detected shots", str(int(summary_map.get("Detected shots", 0) or 0)))
            )
            metric_items.append(
                (
                    "Average shot length",
                    f'{float(summary_map.get("Average shot length (sec)", 0) or 0):.2f} s',
                )
            )
        if WARM_COLOR in selected_analyses:
            metric_items.append(
                ("Warm-color ratio", f'{float(summary_map.get("Warm-color ratio (0–1)", 0) or 0):.3f}')
            )
        if SATURATION in selected_analyses:
            metric_items.append(
                ("Color saturation", f'{float(summary_map.get("Color saturation score (0–1)", 0) or 0):.3f}')
            )
        if CONTRAST in selected_analyses:
            metric_items.append(
                ("Contrast", f'{float(summary_map.get("Contrast score (0–1)", 0) or 0):.3f}')
            )

        if metric_items:
            columns = st.columns(min(4, len(metric_items)))
            for i, (label, value) in enumerate(metric_items):
                columns[i % len(columns)].metric(label, value)

        st.subheader("Research measures")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        if SHOT_ANALYSIS in selected_analyses and not shots_df.empty:
            st.subheader("Shot-level data")
            st.dataframe(shots_df, use_container_width=True, hide_index=True)

        excel_bytes = make_excel(
            summary_df,
            shots_df,
            display_name,
            metadata=analysis_metadata,
        )
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
