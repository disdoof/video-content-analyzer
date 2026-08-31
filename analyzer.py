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
    fps_rounded = max(1, int(round(fps)))
    total_seconds = int(frame_idx // fps_rounded)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    frames = int(frame_idx % fps_rounded)
    return f"{minutes:02d}:{seconds:02d}:{frames:02d}"


def _encode_thumbnail(frame_bgr: np.ndarray, max_width: int = 220) -> bytes:
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        new_h = int(h * max_width / w)
        frame_bgr = cv2.resize(frame_bgr, (max_width, new_h), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return b""
    return buffer.tobytes()


def detect_cuts(
    video_path: str,
    threshold: float = 0.48,
    min_shot_sec: float = 0.45,
    progress: Optional[Callable[[float, str], None]] = None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video could not be opened. Try MP4/H.264 if possible.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    min_gap_frames = max(1, int(min_shot_sec * fps))

    prev_hist = None
    cut_frames = [0]
    frame_idx = 0
    last_cut = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(
                frame,
                (640, int(h * 640 / w)),
                interpolation=cv2.INTER_AREA,
            )

        hist = _histogram(frame)
        if prev_hist is not None and frame_idx - last_cut >= min_gap_frames:
            similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            distance = 1.0 - similarity
            if distance >= threshold:
                cut_frames.append(frame_idx)
                last_cut = frame_idx
        prev_hist = hist

        frame_idx += 1
        if progress and total_frames and frame_idx % max(1, int(fps)) == 0:
            progress(min(1.0, frame_idx / total_frames), "Detecting shot boundaries…")

    cap.release()

    if total_frames > 0 and cut_frames[-1] != total_frames:
        cut_frames.append(total_frames)
    elif len(cut_frames) == 1:
        cut_frames.append(frame_idx)

    cut_frames = sorted(set(cut_frames))
    shots = [(a, b) for a, b in zip(cut_frames[:-1], cut_frames[1:]) if b > a]
    return shots, fps, duration


def _extract_shot_thumbnails(video_path: str, shots: list[tuple[int, int]], progress=None) -> list[bytes]:
    if not shots:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [b"" for _ in shots]

    thumbs = []
    total = len(shots)
    for i, (start_f, end_f) in enumerate(shots, start=1):
        thumb_frame = start_f + max(0, (end_f - start_f) // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(thumb_frame))
        ok, frame = cap.read()
        if ok:
            thumbs.append(_encode_thumbnail(frame))
        else:
            thumbs.append(b"")

        if progress:
            progress(i / max(1, total), f"Preparing shot thumbnails {i} of {total}…")

    cap.release()
    return thumbs


def _sample_visual_measures(
    video_path: str,
    analyses: Set[str],
    fps: float,
    total_frames: int,
    progress: Optional[Callable[[float, str], None]] = None,
):
    visual = analyses & {WARM_COLOR, SATURATION, CONTRAST}
    if not visual:
        return {}, 0

    if total_frames <= 0:
        return {}, 0

    duration = total_frames / fps if fps > 0 else 0.0
    sample_count = min(300, max(1, int(np.ceil(duration))))
    frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
    frame_indices = np.unique(frame_indices)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video could not be opened for visual analysis.")

    collected = {key: [] for key in [
        "warm_color_ratio_0_1",
        "color_saturation_score_0_1",
        "contrast_score_0_1",
    ]}

    total_samples = len(frame_indices)
    for i, frame_idx in enumerate(frame_indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]
        if w > 960:
            frame = cv2.resize(
                frame,
                (960, int(h * 960 / w)),
                interpolation=cv2.INTER_AREA,
            )

        values = _selected_frame_metrics(frame, visual)
        for key, value in values.items():
            collected[key].append(value)

        if progress:
            progress(i / max(1, total_samples), f"Analyzing visual sample {i} of {total_samples}…")

    cap.release()

    means = {}
    for key, values in collected.items():
        if values:
            means[key] = float(np.mean(values))

    return means, total_samples


def analyze_video(
    video_path: str,
    analyses,
    threshold: float = 0.48,
    min_shot_sec: float = 0.45,
    progress: Optional[Callable[[float, str], None]] = None,
):
    analyses = set(analyses) & VALID_ANALYSES
    if not analyses:
        raise ValueError("Select at least one analysis before running the video.")

    fps, total_frames, duration = _video_info(video_path)
    wants_shots = SHOT_ANALYSIS in analyses
    wants_visual = bool(analyses & {WARM_COLOR, SATURATION, CONTRAST})

    shots = []
    shots_df = pd.DataFrame()

    if wants_shots:
        def cut_progress(p, text):
            if progress:
                weight = 0.50 if wants_visual else 0.70
                progress(weight * p, text)

        shots, fps, duration = detect_cuts(
            video_path,
            threshold=threshold,
            min_shot_sec=min_shot_sec,
            progress=cut_progress,
        )

        thumb_bytes = []
        def thumb_progress(p, text):
            if progress:
                start = 0.50 if wants_visual else 0.70
                width = 0.15 if wants_visual else 0.25
                progress(start + width * p, text)

        thumb_bytes = _extract_shot_thumbnails(video_path, shots, progress=thumb_progress)

        rows = []
        for i, ((start_f, end_f), thumb) in enumerate(zip(shots, thumb_bytes), start=1):
            start_sec = start_f / fps
            end_sec = end_f / fps
            rows.append({
                "shot_number": i,
                "start_timecode": _frame_to_timecode(start_f, fps),
                "end_timecode": _frame_to_timecode(max(start_f, end_f - 1), fps),
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "duration_sec": round(end_sec - start_sec, 3),
                "thumbnail_bytes": thumb,
            })
        shots_df = pd.DataFrame(rows)

    visual_means = {}
    sample_count = 0
    if wants_visual:
        def visual_progress(p, text):
            if progress:
                if wants_shots:
                    progress(0.65 + 0.30 * p, text)
                else:
                    progress(0.95 * p, text)

        visual_means, sample_count = _sample_visual_measures(
            video_path,
            analyses,
            fps=fps,
            total_frames=total_frames,
            progress=visual_progress,
        )

    summary_rows = [("Video duration (sec)", round(duration, 3))]

    if wants_shots:
        summary_rows.append(("Detected shots", int(len(shots_df))))
        avg_shot = round(float(shots_df["duration_sec"].mean()), 3) if not shots_df.empty else None
        summary_rows.append(("Average shot length (sec)", avg_shot))

    if WARM_COLOR in analyses:
        value = visual_means.get("warm_color_ratio_0_1")
        summary_rows.append(("Warm-color ratio (0–1)", round(value, 4) if value is not None else None))

    if SATURATION in analyses:
        value = visual_means.get("color_saturation_score_0_1")
        summary_rows.append(("Color saturation score (0–1)", round(value, 4) if value is not None else None))

    if CONTRAST in analyses:
        value = visual_means.get("contrast_score_0_1")
        summary_rows.append(("Contrast score (0–1)", round(value, 4) if value is not None else None))

    summary_df = pd.DataFrame(summary_rows, columns=["Measure", "Value"])

    if progress:
        progress(1.0, "Done")

    metadata = {
        "selected_analyses": sorted(analyses),
        "visual_sample_count": sample_count,
        "threshold": threshold,
        "min_shot_sec": min_shot_sec,
        "fps": round(fps, 3),
    }
    return summary_df, shots_df, metadata


def make_excel(
    summary_df: pd.DataFrame,
    shots_df: pd.DataFrame,
    video_name: str,
    metadata: Optional[dict] = None,
) -> bytes:
    from io import BytesIO

    metadata = metadata or {}
    selected = set(metadata.get("selected_analyses", []))

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        excel_shots_df = shots_df.drop(columns=["thumbnail_bytes"], errors="ignore")
        if not excel_shots_df.empty:
            excel_shots_df.to_excel(writer, sheet_name="Shots", index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
        title_fmt = workbook.add_format({"bold": True, "font_size": 14})
        note_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})

        ws1 = writer.sheets["Summary"]
        ws1.set_column("A:A", 38)
        ws1.set_column("B:B", 18)
        for col, name in enumerate(summary_df.columns):
            ws1.write(0, col, name, header_fmt)

        if not excel_shots_df.empty:
            ws2 = writer.sheets["Shots"]
            ws2.freeze_panes(1, 0)
            ws2.autofilter(0, 0, max(1, len(excel_shots_df)), max(0, len(excel_shots_df.columns) - 1))
            ws2.set_column(0, max(0, len(excel_shots_df.columns) - 1), 18)
            for col, name in enumerate(excel_shots_df.columns):
                ws2.write(0, col, name, header_fmt)

        notes = workbook.add_worksheet("Method")
        notes.set_column("A:A", 31)
        notes.set_column("B:B", 100)
        notes.write("A1", "Video", title_fmt)
        notes.write("B1", video_name)

        label_map = {
            SHOT_ANALYSIS: "Shot / cut analysis",
            WARM_COLOR: "Warm-color palette",
            SATURATION: "Color saturation",
            CONTRAST: "Contrast",
        }
        selected_labels = [label_map[x] for x in [SHOT_ANALYSIS, WARM_COLOR, SATURATION, CONTRAST] if x in selected]

        method_rows = [("Selected analyses", ", ".join(selected_labels))]

        if SHOT_ANALYSIS in selected:
            method_rows.extend([
                (
                    "Shot boundary detection",
                    "Abrupt shot changes are detected from frame-to-frame HSV histogram change using the selected shot-change threshold and minimum shot duration.",
                ),
                ("Shot-change threshold", str(metadata.get("threshold", ""))),
                ("Minimum shot duration", f'{metadata.get("min_shot_sec", "")} seconds'),
                ("Validation timecode format", "MM:SS:FF (minute:second:frame)"),
                ("Video FPS", str(metadata.get("fps", ""))),
                ("Shot thumbnail", "Each shot thumbnail is taken from the approximate middle frame of the detected shot and shown in the app for validation."),
            ])

        if selected & {WARM_COLOR, SATURATION, CONTRAST}:
            method_rows.append(
                (
                    "Visual sampling",
                    f'Visual measures are calculated independently of shot detection from approximately one uniformly spaced frame per second, capped at 300 samples. Frames successfully sampled in this run: {metadata.get("visual_sample_count", 0)}.',
                )
            )

        if WARM_COLOR in selected:
            method_rows.append(("Warm-color ratio (0–1)", "Mean proportion of sufficiently saturated pixels falling in the approximate red/orange/yellow hue ranges. 0 = none; 1 = all sampled pixels meet the warm-color rule."))
        if SATURATION in selected:
            method_rows.append(("Color saturation score (0–1)", "Mean HSV saturation normalized from OpenCV's 0–255 scale to 0–1."))
        if CONTRAST in selected:
            method_rows.append(("Contrast score (0–1)", "Mean grayscale standard deviation normalized by the theoretical 8-bit maximum standard deviation (127.5), then clipped to 0–1."))

        if selected & {WARM_COLOR, SATURATION, CONTRAST}:
            method_rows.append(("Category thresholds", "This version does not convert the numeric scores into warm/cool or low/medium/high categories. Category thresholds can be fixed after calibration and validation."))

        for r, (k, v) in enumerate(method_rows, start=3):
            notes.write(r, 0, k, header_fmt)
            notes.write(r, 1, v, note_fmt)

    return output.getvalue()
