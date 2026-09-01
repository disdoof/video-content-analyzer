import os
import tempfile
from pathlib import Path

import streamlit as st

from hybrid_analyzer import (
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
st.caption("Hybrid Cut Detector v2.0 — initial calibration build")

ANALYSIS_OPTIONS = {
    "analysis_shot": ("Shot / cut analysis", SHOT_ANALYSIS),
    "analysis_warm": ("Warm-color palette", WARM_COLOR),
    "analysis_saturation": ("Color saturation", SATURATION),
    "analysis_contrast": ("Contrast", CONTRAST),
}
if "analysis_select_all" not in st.session_state:
    st.session_state.analysis_select_all = True
for key in ANALYSIS_OPTIONS:
    if key not in st.session_state:
        st.session_state[key] = True


def select_all_changed():
    for key in ANALYSIS_OPTIONS:
        st.session_state[key] = bool(st.session_state.analysis_select_all)


def individual_changed():
    st.session_state.analysis_select_all = all(bool(st.session_state[k]) for k in ANALYSIS_OPTIONS)


with st.sidebar:
    st.header("Analyses to run")
    st.checkbox("Select all", key="analysis_select_all", on_change=select_all_changed)
    st.divider()
    for key, (label, _) in ANALYSIS_OPTIONS.items():
        st.checkbox(label, key=key, on_change=individual_changed)
    selected = {aid for key, (_, aid) in ANALYSIS_OPTIONS.items() if st.session_state[key]}

    threshold = 0.48
    min_shot = 0.20
    show_rejected = False
    if SHOT_ANALYSIS in selected:
        st.divider()
        st.header("Hybrid detection settings")
        threshold = st.slider(
            "Hybrid cut threshold",
            min_value=0.30,
            max_value=0.75,
            value=0.48,
            step=0.01,
            help="Higher = stricter. This score combines structural, edge, local-grid color and global color evidence.",
        )
        min_shot = st.slider(
            "Minimum shot duration (seconds)",
            min_value=0.05,
            max_value=1.00,
            value=0.20,
            step=0.05,
        )
        show_rejected = st.checkbox("Show strong rejected candidates", value=False)

source_mode = st.radio("Video source", ["Upload Video", "YouTube URL"], horizontal=True)
uploaded = None; youtube_url = ""
if source_mode == "Upload Video":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "m4v", "avi", "mkv", "webm"])
    if uploaded is not None:
        st.video(uploaded)
else:
    youtube_url = st.text_input("Paste a YouTube link", placeholder="https://www.youtube.com/watch?v=…").strip()
    if youtube_url and not is_youtube_url(youtube_url):
        st.warning("That does not look like a YouTube link.")

if not selected:
    st.warning("Select at least one analysis.")
has_source = (source_mode == "Upload Video" and uploaded is not None) or (source_mode == "YouTube URL" and youtube_url and is_youtube_url(youtube_url))

if has_source and selected and st.button("Analyze video", type="primary", use_container_width=True):
    bar = st.progress(0.0); status = st.empty()
    def update(p, text):
        bar.progress(float(max(0.0, min(1.0, p)))); status.write(text)

    tmp_path = None; tmp_dir = None; display_name = "video"
    try:
        if source_mode == "Upload Video":
            suffix = Path(uploaded.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer()); tmp_path = tmp.name
            display_name = uploaded.name; ap = update
        else:
            tmp_dir = tempfile.TemporaryDirectory(prefix="vca_v2_")
            tmp_path, source_meta = download_youtube_video(youtube_url, tmp_dir.name, progress=update)
            display_name = f"{source_meta.get('title','youtube')}{Path(tmp_path).suffix}"
            def ap(p, text): update(0.25 + 0.75 * p, text)

        summary, shots, cuts, candidates, events, meta = analyze_video(
            tmp_path, selected, threshold=threshold, min_shot_sec=min_shot, progress=ap
        )
        st.success("Analysis complete.")
        smap = dict(zip(summary["Measure"], summary["Value"]))

        if SHOT_ANALYSIS in selected:
            a,b,c = st.columns(3)
            a.metric("Detected cuts", int(smap.get("Detected cuts",0) or 0))
            b.metric("Detected shots", int(smap.get("Detected shots",0) or 0))
            c.metric("Average shot length", f'{float(smap.get("Average shot length (sec)",0) or 0):.2f} s')
            st.caption(f"Timecode: MM:SS:FF  |  Video FPS: {meta.get('fps')}  |  Detector: v2.0 Hybrid")

        st.subheader("Research measures")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        if SHOT_ANALYSIS in selected and not cuts.empty:
            st.subheader("Detected cut boundaries")
            cut_cols = ["cut_number","timecode","hybrid_score","global_hist","grid_hist","structural_diff","edge_change","prominence"]
            st.dataframe(cuts[cut_cols], use_container_width=True, hide_index=True)

            st.subheader("Before / after each detected cut")
            for rec in cuts.to_dict("records"):
                with st.expander(f"Cut {rec['cut_number']} · {rec['timecode']} · score {rec['hybrid_score']:.3f}"):
                    x,y = st.columns(2)
                    with x:
                        st.caption("Frame before cut")
                        if rec.get("before_frame_bytes"): st.image(rec["before_frame_bytes"], use_container_width=True)
                    with y:
                        st.caption("Frame after cut")
                        if rec.get("after_frame_bytes"): st.image(rec["after_frame_bytes"], use_container_width=True)

        if SHOT_ANALYSIS in selected and not shots.empty:
            st.subheader("Detected shots")
            st.dataframe(shots[["shot_number","start_timecode","end_timecode","duration_sec"]], use_container_width=True, hide_index=True)

        if SHOT_ANALYSIS in selected and show_rejected and not events.empty:
            rej = events[~events["accepted_cut"]].copy()
            if not rej.empty:
                st.subheader("Strong rejected candidates")
                cols = ["timecode","hybrid_score","structural_diff","edge_change","grid_hist","global_hist","prominence","decision"]
                st.dataframe(rej[cols], use_container_width=True, hide_index=True)
                for rec in rej.head(30).to_dict("records"):
                    with st.expander(f"Rejected · {rec['timecode']} · score {rec['hybrid_score']:.3f} · {rec['decision']}"):
                        x,y = st.columns(2)
                        with x:
                            st.caption("Frame before")
                            if rec.get("before_frame_bytes"): st.image(rec["before_frame_bytes"], use_container_width=True)
                        with y:
                            st.caption("Frame after")
                            if rec.get("after_frame_bytes"): st.image(rec["after_frame_bytes"], use_container_width=True)

        excel = make_excel(summary, shots, cuts, candidates, events, display_name, meta)
        st.download_button(
            "Download Excel report",
            excel,
            file_name=f"{Path(display_name).stem}_hybrid_v2.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
    finally:
        if source_mode == "Upload Video" and tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if tmp_dir is not None:
            tmp_dir.cleanup()
