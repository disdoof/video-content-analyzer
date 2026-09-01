import os
import tempfile
from pathlib import Path

import streamlit as st

from analyzer import analyze_video, make_excel, SHOT_ANALYSIS, WARM_COLOR, SATURATION, CONTRAST
from youtube_utils import download_youtube_video, is_youtube_url

st.set_page_config(page_title="Video Content Analyzer", page_icon="🎬", layout="wide")
st.title("🎬 Video Content Analyzer")
st.caption("Research-oriented audiovisual content analysis — Diagnostic cut-validation build")

ANALYSIS_OPTIONS = {
    "analysis_shot": ("Shot / cut analysis", SHOT_ANALYSIS),
    "analysis_warm": ("Warm-color palette", WARM_COLOR),
    "analysis_saturation": ("Color saturation", SATURATION),
    "analysis_contrast": ("Contrast", CONTRAST),
}
if "analysis_select_all" not in st.session_state: st.session_state.analysis_select_all = True
for key in ANALYSIS_OPTIONS:
    if key not in st.session_state: st.session_state[key] = True

def _select_all_changed():
    for key in ANALYSIS_OPTIONS: st.session_state[key] = bool(st.session_state.analysis_select_all)

def _individual_changed():
    st.session_state.analysis_select_all = all(bool(st.session_state[k]) for k in ANALYSIS_OPTIONS)

with st.sidebar:
    st.header("Analyses to run")
    st.checkbox("Select all", key="analysis_select_all", on_change=_select_all_changed)
    st.divider()
    for key,(label,_) in ANALYSIS_OPTIONS.items(): st.checkbox(label,key=key,on_change=_individual_changed)
    selected_analyses={aid for key,(_,aid) in ANALYSIS_OPTIONS.items() if st.session_state[key]}
    threshold=0.48; min_shot=0.45
    if SHOT_ANALYSIS in selected_analyses:
        st.divider(); st.header("Detection settings")
        threshold=st.slider("Shot-change threshold",0.20,0.90,0.48,0.02,
            help="Current detector threshold. This diagnostic build does not change the detector logic.")
        min_shot=st.slider("Minimum shot duration (seconds)",0.20,2.00,0.45,0.05)

source_mode=st.radio("Video source",["Upload Video","YouTube URL"],horizontal=True)
uploaded=None; youtube_url=""
if source_mode=="Upload Video":
    uploaded=st.file_uploader("Upload a video",type=["mp4","mov","m4v","avi","mkv","webm"])
    if uploaded is not None: st.video(uploaded)
else:
    youtube_url=st.text_input("Paste a YouTube link",placeholder="https://www.youtube.com/watch?v=…").strip()
    if youtube_url and not is_youtube_url(youtube_url): st.warning("That does not look like a YouTube link.")

if not selected_analyses: st.warning("Select at least one analysis from the sidebar.")
has_source=(source_mode=="Upload Video" and uploaded is not None) or (source_mode=="YouTube URL" and youtube_url and is_youtube_url(youtube_url))

if has_source and selected_analyses and st.button("Analyze video",type="primary",use_container_width=True):
    bar=st.progress(0.0); status=st.empty()
    def update(p,text): bar.progress(float(max(0,min(1,p)))); status.write(text)
    tmp_path=None; tmp_dir=None; display_name="video"; source_metadata=None
    try:
        if source_mode=="Upload Video":
            suffix=Path(uploaded.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer()); tmp_path=tmp.name
            display_name=uploaded.name; ap=update
        else:
            tmp_dir=tempfile.TemporaryDirectory(prefix="vca_")
            tmp_path,source_metadata=download_youtube_video(youtube_url,tmp_dir.name,progress=update)
            display_name=f"{source_metadata.get('title','youtube')}{Path(tmp_path).suffix}"
            def ap(p,text): update(0.25+0.75*p,text)

        summary,shots,diagnostics,events,meta=analyze_video(
            tmp_path,selected_analyses,threshold=threshold,min_shot_sec=min_shot,progress=ap
        )
        st.success("Analysis complete.")
        smap=dict(zip(summary["Measure"],summary["Value"]))
        if SHOT_ANALYSIS in selected_analyses:
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Detected cuts",int(smap.get("Detected cuts",0) or 0))
            c2.metric("Detected shots",int(smap.get("Detected shots",0) or 0))
            c3.metric("Average shot length",f'{float(smap.get("Average shot length (sec)",0) or 0):.2f} s')
            c4.metric("Blocked by min duration",int(smap.get("Threshold-passing frames blocked by minimum duration",0) or 0))

        st.subheader("Research measures"); st.dataframe(summary,use_container_width=True,hide_index=True)

        if SHOT_ANALYSIS in selected_analyses:
            st.caption(f"Timecode: MM:SS:FF  |  Video FPS: {meta.get('fps')}  |  Diagnostic v1.6")
            if not shots.empty:
                st.subheader("Detected shots")
                cols=["shot_number","start_timecode","end_timecode","duration_sec"]
                st.dataframe(shots[cols],use_container_width=True,hide_index=True)

            if not events.empty:
                st.subheader("Diagnostic events")
                st.caption("All threshold-passing frame transitions plus the strongest local peaks below threshold. These are for diagnosis only; the detector itself has not been changed.")
                table_cols=["timecode","histogram_distance","threshold_pass","seconds_since_last_accepted_cut","minimum_duration_pass","brightness_delta_signed_0_1","decision","diagnostic_role"]
                st.dataframe(events[table_cols],use_container_width=True,hide_index=True)

                st.subheader("Before / after frames for diagnostic events")
                for rec in events.to_dict("records"):
                    with st.expander(f"{rec['timecode']}  ·  {rec['diagnostic_role']}  ·  histogram {rec['histogram_distance']:.4f}"):
                        st.write(
                            f"**Decision:** {rec['decision']}  |  **Threshold pass:** {rec['threshold_pass']}  |  "
                            f"**Min-duration pass:** {rec['minimum_duration_pass']}  |  "
                            f"**Since last accepted cut:** {rec['seconds_since_last_accepted_cut']:.3f}s  |  "
                            f"**Brightness Δ:** {rec['brightness_delta_signed_0_1']:+.4f}"
                        )
                        a,b=st.columns(2)
                        with a:
                            st.caption("Frame before")
                            if rec.get("before_frame_bytes"): st.image(rec["before_frame_bytes"],use_container_width=True)
                        with b:
                            st.caption("Frame after")
                            if rec.get("after_frame_bytes"): st.image(rec["after_frame_bytes"],use_container_width=True)

        excel=make_excel(summary,shots,diagnostics,events,display_name,meta)
        st.download_button("Download diagnostic Excel report",excel,
            file_name=f"{Path(display_name).stem}_cut_diagnostics.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",use_container_width=True)
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
    finally:
        if source_mode=="Upload Video" and tmp_path and os.path.exists(tmp_path): os.unlink(tmp_path)
        if tmp_dir is not None: tmp_dir.cleanup()
