from __future__ import annotations

import cv2
from typing import Callable, Optional, Set

import numpy as np
import pandas as pd


SHOT_ANALYSIS = "shot"
WARM_COLOR = "warm_color"
SATURATION = "saturation"
CONTRAST = "contrast"
VALID_ANALYSES = {SHOT_ANALYSIS, WARM_COLOR, SATURATION, CONTRAST}


def _histogram(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def _selected_frame_metrics(frame_bgr: np.ndarray, analyses: Set[str]) -> dict:
    result = {}
    needs_hsv = WARM_COLOR in analyses or SATURATION in analyses
    if needs_hsv:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        if WARM_COLOR in analyses:
            h = hsv[:, :, 0]
            s = hsv[:, :, 1]
            result["warm_color_ratio_0_1"] = float(
                (((h <= 35) | (h >= 170)) & (s >= 40)).mean()
            )
        if SATURATION in analyses:
            result["color_saturation_score_0_1"] = float(hsv[:, :, 1].mean()) / 255.0
    if CONTRAST in analyses:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        result["contrast_score_0_1"] = min(1.0, float(gray.std()) / 127.5)
    return result


def _video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video could not be opened. Try MP4/H.264 if possible.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()
    return fps, total_frames, duration


def _frame_to_timecode(frame_idx: int, fps: float) -> str:
    """MM:SS:FF using the video's actual FPS (non-drop-frame research display)."""
    fps = float(fps) if fps and fps > 0 else 25.0
    t = float(max(0, frame_idx)) / fps
    whole_seconds = int(np.floor(t + 1e-9))
    minutes = whole_seconds // 60
    seconds = whole_seconds % 60
    nominal_fps = max(1, int(round(fps)))
    frame_in_second = int(round((t - whole_seconds) * fps))
    frame_in_second = min(nominal_fps - 1, max(0, frame_in_second))
    return f"{minutes:02d}:{seconds:02d}:{frame_in_second:02d}"


def _encode_thumbnail(frame_bgr: np.ndarray, max_width: int = 420) -> bytes:
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        new_h = int(h * max_width / w)
        frame_bgr = cv2.resize(frame_bgr, (max_width, new_h), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buffer.tobytes() if ok else b""


def detect_cuts_with_diagnostics(
    video_path: str,
    threshold: float = 0.48,
    min_shot_sec: float = 0.45,
    progress: Optional[Callable[[float, str], None]] = None,
):
    """Run the current histogram detector unchanged in decision logic, while logging every frame transition."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video could not be opened. Try MP4/H.264 if possible.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    min_gap_frames = max(1, int(round(min_shot_sec * fps)))

    prev_hist = None
    prev_brightness = None
    cut_frames = [0]
    last_cut = 0
    frame_idx = 0
    diagnostic_rows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        work = frame
        if w > 640:
            work = cv2.resize(work, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)

        hist = _histogram(work)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())

        if prev_hist is not None:
            similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            distance = float(1.0 - similarity)
            frames_since_last_cut = int(frame_idx - last_cut)
            min_duration_pass = frames_since_last_cut >= min_gap_frames
            threshold_pass = distance >= threshold

            if threshold_pass and min_duration_pass:
                decision = "Accepted cut"
                cut_frames.append(frame_idx)
                last_cut = frame_idx
            elif threshold_pass and not min_duration_pass:
                decision = "Rejected: minimum shot duration"
            else:
                decision = "Below threshold"

            signed_brightness = 0.0 if prev_brightness is None else (brightness - prev_brightness) / 255.0
            diagnostic_rows.append({
                "boundary_frame": int(frame_idx),
                "timecode": _frame_to_timecode(frame_idx, fps),
                "histogram_distance": round(distance, 6),
                "threshold": round(float(threshold), 4),
                "threshold_pass": bool(threshold_pass),
                "frames_since_last_accepted_cut": frames_since_last_cut,
                "seconds_since_last_accepted_cut": round(frames_since_last_cut / fps, 4),
                "minimum_duration_pass": bool(min_duration_pass),
                "brightness_prev_0_255": round(float(prev_brightness or 0.0), 3),
                "brightness_current_0_255": round(brightness, 3),
                "brightness_delta_signed_0_1": round(float(signed_brightness), 6),
                "brightness_change_abs_0_1": round(abs(float(signed_brightness)), 6),
                "decision": decision,
            })

        prev_hist = hist
        prev_brightness = brightness
        frame_idx += 1
        if progress and total_frames and frame_idx % max(1, int(fps)) == 0:
            progress(min(1.0, frame_idx / total_frames), "Running cut diagnostics…")

    cap.release()

    if total_frames > 0 and cut_frames[-1] != total_frames:
        cut_frames.append(total_frames)
    elif len(cut_frames) == 1:
        cut_frames.append(frame_idx)

    cut_frames = sorted(set(cut_frames))
    shots = [(a, b) for a, b in zip(cut_frames[:-1], cut_frames[1:]) if b > a]

    diagnostics_df = pd.DataFrame(diagnostic_rows)
    if not diagnostics_df.empty:
        vals = diagnostics_df["histogram_distance"].to_numpy(dtype=float)
        local_peak = np.ones(len(vals), dtype=bool)
        if len(vals) >= 3:
            local_peak[:] = False
            local_peak[0] = vals[0] >= vals[1]
            local_peak[-1] = vals[-1] >= vals[-2]
            local_peak[1:-1] = (vals[1:-1] >= vals[:-2]) & (vals[1:-1] >= vals[2:])
        diagnostics_df["local_peak"] = local_peak

        # Diagnostic-interest events: all threshold passes, plus the strongest local peaks below threshold.
        must_keep = diagnostics_df["threshold_pass"]
        below_peaks = diagnostics_df[(~diagnostics_df["threshold_pass"]) & diagnostics_df["local_peak"] & (diagnostics_df["histogram_distance"] > 0.001)].copy()
        below_peaks = below_peaks.nlargest(80, "histogram_distance")
        event_indices = sorted(set(diagnostics_df[must_keep].index.tolist() + below_peaks.index.tolist()))
        events_df = diagnostics_df.loc[event_indices].copy().reset_index(drop=True)
        events_df["diagnostic_role"] = np.where(
            events_df["decision"] == "Accepted cut",
            "Accepted by current detector",
            np.where(
                events_df["decision"] == "Rejected: minimum shot duration",
                "Would pass threshold; blocked by minimum duration",
                "Strong local peak below threshold",
            ),
        )
    else:
        diagnostics_df = pd.DataFrame()
        events_df = pd.DataFrame()

    return shots, fps, duration, diagnostics_df, events_df


def _extract_event_images(video_path: str, events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return events_df
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        events_df = events_df.copy()
        events_df["before_frame_bytes"] = b""
        events_df["after_frame_bytes"] = b""
        return events_df

    before_images, after_images = [], []
    for boundary_frame in events_df["boundary_frame"].astype(int):
        before_b, after_b = b"", b""
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, boundary_frame - 1))
        ok, frame = cap.read()
        if ok:
            before_b = _encode_thumbnail(frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, boundary_frame)
        ok, frame = cap.read()
        if ok:
            after_b = _encode_thumbnail(frame)
        before_images.append(before_b)
        after_images.append(after_b)
    cap.release()

    out = events_df.copy()
    out["before_frame_bytes"] = before_images
    out["after_frame_bytes"] = after_images
    return out


def _extract_shot_boundary_images(video_path: str, shots: list[tuple[int, int]]) -> list[tuple[bytes, bytes]]:
    if not shots:
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [(b"", b"") for _ in shots]
    images = []
    for start_f, end_f in shots:
        first_b, last_b = b"", b""
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_f))
        ok, frame = cap.read()
        if ok:
            first_b = _encode_thumbnail(frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(start_f, end_f - 1)))
        ok, frame = cap.read()
        if ok:
            last_b = _encode_thumbnail(frame)
        images.append((first_b, last_b))
    cap.release()
    return images


def _sample_visual_measures(video_path: str, analyses: Set[str], fps: float, total_frames: int, progress=None):
    visual = analyses & {WARM_COLOR, SATURATION, CONTRAST}
    if not visual or total_frames <= 0:
        return {}, 0
    duration = total_frames / fps if fps > 0 else 0.0
    sample_count = min(300, max(1, int(np.ceil(duration))))
    frame_indices = np.unique(np.linspace(0, total_frames - 1, sample_count, dtype=int))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video could not be opened for visual analysis.")
    collected = {k: [] for k in ["warm_color_ratio_0_1", "color_saturation_score_0_1", "contrast_score_0_1"]}
    for i, frame_idx in enumerate(frame_indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > 960:
            frame = cv2.resize(frame, (960, int(h * 960 / w)), interpolation=cv2.INTER_AREA)
        values = _selected_frame_metrics(frame, visual)
        for key, value in values.items():
            collected[key].append(value)
        if progress:
            progress(i / max(1, len(frame_indices)), f"Analyzing visual sample {i} of {len(frame_indices)}…")
    cap.release()
    return {k: float(np.mean(v)) for k, v in collected.items() if v}, len(frame_indices)


def analyze_video(video_path: str, analyses, threshold: float = 0.48, min_shot_sec: float = 0.45, progress=None):
    analyses = set(analyses) & VALID_ANALYSES
    if not analyses:
        raise ValueError("Select at least one analysis before running the video.")

    fps, total_frames, duration = _video_info(video_path)
    wants_shots = SHOT_ANALYSIS in analyses
    wants_visual = bool(analyses & {WARM_COLOR, SATURATION, CONTRAST})

    shots_df = pd.DataFrame()
    diagnostics_df = pd.DataFrame()
    events_df = pd.DataFrame()
    shots = []

    if wants_shots:
        def dp(p, text):
            if progress:
                progress((0.60 if wants_visual else 0.90) * p, text)
        shots, fps, duration, diagnostics_df, events_df = detect_cuts_with_diagnostics(
            video_path, threshold=threshold, min_shot_sec=min_shot_sec, progress=dp
        )
        boundaries = _extract_shot_boundary_images(video_path, shots)
        rows = []
        for i, ((start_f, end_f), (first_b, last_b)) in enumerate(zip(shots, boundaries), start=1):
            rows.append({
                "shot_number": i,
                "start_timecode": _frame_to_timecode(start_f, fps),
                "end_timecode": _frame_to_timecode(max(start_f, end_f - 1), fps),
                "duration_sec": round((end_f - start_f) / fps, 3),
                "first_frame_bytes": first_b,
                "last_frame_bytes": last_b,
            })
        shots_df = pd.DataFrame(rows)
        events_df = _extract_event_images(video_path, events_df)

    visual_means, sample_count = {}, 0
    if wants_visual:
        def vp(p, text):
            if progress:
                progress((0.65 if wants_shots else 0.0) + (0.30 if wants_shots else 0.95) * p, text)
        visual_means, sample_count = _sample_visual_measures(video_path, analyses, fps, total_frames, vp)

    summary_rows = [("Video duration (sec)", round(duration, 3))]
    if wants_shots:
        summary_rows += [
            ("Detected cuts", max(0, len(shots_df) - 1)),
            ("Detected shots", len(shots_df)),
            ("Average shot length (sec)", round(float(shots_df["duration_sec"].mean()), 3) if not shots_df.empty else None),
            ("Threshold-passing frames blocked by minimum duration", int((diagnostics_df["decision"] == "Rejected: minimum shot duration").sum()) if not diagnostics_df.empty else 0),
        ]
    if WARM_COLOR in analyses:
        v=visual_means.get("warm_color_ratio_0_1"); summary_rows.append(("Warm-color ratio (0–1)", round(v,4) if v is not None else None))
    if SATURATION in analyses:
        v=visual_means.get("color_saturation_score_0_1"); summary_rows.append(("Color saturation score (0–1)", round(v,4) if v is not None else None))
    if CONTRAST in analyses:
        v=visual_means.get("contrast_score_0_1"); summary_rows.append(("Contrast score (0–1)", round(v,4) if v is not None else None))

    if progress:
        progress(1.0, "Done")

    metadata = {
        "selected_analyses": sorted(analyses),
        "visual_sample_count": sample_count,
        "threshold": threshold,
        "min_shot_sec": min_shot_sec,
        "fps": round(fps, 3),
        "diagnostic_version": "1.6",
    }
    return pd.DataFrame(summary_rows, columns=["Measure", "Value"]), shots_df, diagnostics_df, events_df, metadata


def make_excel(summary_df, shots_df, diagnostics_df, events_df, video_name: str, metadata: Optional[dict] = None) -> bytes:
    from io import BytesIO
    metadata = metadata or {}
    selected = set(metadata.get("selected_analyses", []))
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        excel_shots = shots_df.drop(columns=["first_frame_bytes", "last_frame_bytes"], errors="ignore")
        if not excel_shots.empty:
            excel_shots.to_excel(writer, sheet_name="Shots", index=False)
        if not diagnostics_df.empty:
            diagnostics_df.to_excel(writer, sheet_name="Frame diagnostics", index=False)
        excel_events = events_df.drop(columns=["before_frame_bytes", "after_frame_bytes"], errors="ignore")
        if not excel_events.empty:
            excel_events.to_excel(writer, sheet_name="Diagnostic events", index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
        title_fmt = workbook.add_format({"bold": True, "font_size": 14})
        note_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
        for sheet_name in ["Summary", "Shots", "Frame diagnostics", "Diagnostic events"]:
            if sheet_name in writer.sheets:
                ws=writer.sheets[sheet_name]
                ws.freeze_panes(1,0)
                ws.set_column(0, 30, 20)
                df = {"Summary":summary_df,"Shots":excel_shots,"Frame diagnostics":diagnostics_df,"Diagnostic events":excel_events}[sheet_name]
                for c,n in enumerate(df.columns): ws.write(0,c,n,header_fmt)

        notes = workbook.add_worksheet("Method")
        notes.set_column("A:A", 34); notes.set_column("B:B", 105)
        notes.write("A1", "Video", title_fmt); notes.write("B1", video_name)
        method_rows = [
            ("Version", "Diagnostic v1.6 — current detector logic is preserved; diagnostic measurements are added for validation."),
            ("Shot-change threshold", str(metadata.get("threshold", ""))),
            ("Minimum shot duration", f'{metadata.get("min_shot_sec", "")} seconds'),
            ("Timecode", "MM:SS:FF (minute:second:frame), based on the video FPS."),
            ("Video FPS", str(metadata.get("fps", ""))),
            ("Histogram distance", "1 minus HSV histogram correlation between the previous frame and current frame. Higher values indicate a larger global color-distribution change."),
            ("Threshold pass", "True when histogram distance is greater than or equal to the selected shot-change threshold."),
            ("Minimum-duration pass", "True when enough frames have elapsed since the last accepted cut to satisfy the selected minimum shot duration."),
            ("Brightness delta", "Signed change in mean grayscale brightness from previous to current frame, normalized by 255. Negative values indicate darkening; positive values indicate brightening."),
            ("Frame diagnostics", "Contains one row for every transition between consecutive video frames."),
            ("Diagnostic events", "Contains all threshold-passing transitions plus up to 80 strongest local histogram peaks below threshold, for easier inspection of missed cuts and false positives."),
        ]
        for r,(k,v) in enumerate(method_rows,start=3):
            notes.write(r,0,k,header_fmt); notes.write(r,1,v,note_fmt)
    return output.getvalue()
