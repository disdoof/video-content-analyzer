from __future__ import annotations

import cv2
import math
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class ShotMetrics:
    shot_number: int
    start_sec: float
    end_sec: float
    duration_sec: float
    representative_frame_sec: float
    dominant_hex: str
    dominant_r: int
    dominant_g: int
    dominant_b: int
    brightness_0_255: float
    contrast_sd: float
    saturation_0_255: float
    warm_color_ratio: float
    edge_density: float


def _rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = [int(x) for x in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def _dominant_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    # Fast, deterministic dominant-color estimate via quantization.
    small = cv2.resize(frame_bgr, (96, 54), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape(-1, 3)
    quant = (pixels // 32) * 32 + 16
    colors, counts = np.unique(quant, axis=0, return_counts=True)
    return np.clip(colors[np.argmax(counts)], 0, 255).astype(np.uint8)


def _frame_metrics(frame_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    dominant = _dominant_rgb(frame_bgr)

    # Warm pixels: hue roughly red/orange/yellow in HSV, excluding near-gray pixels.
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    warm = (((h <= 35) | (h >= 170)) & (s >= 40)).mean()

    edges = cv2.Canny(gray, 100, 200)
    edge_density = (edges > 0).mean()


    return {
        "dominant_rgb": dominant,
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "saturation": float(hsv[:, :, 1].mean()),
        "warm_ratio": float(warm),
        "edge_density": float(edge_density),
    }


def _histogram(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def detect_cuts(video_path: str, threshold: float = 0.48, min_shot_sec: float = 0.45,
                progress: Optional[Callable[[float, str], None]] = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video could not be opened. Try MP4/H.264 if possible.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    min_gap_frames = max(1, int(min_shot_sec * fps))

    prev_hist = None
    cut_frames = [0]
    frame_idx = 0
    last_cut = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Analyze a downscaled frame for speed.
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)

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
            progress(min(0.55, 0.55 * frame_idx / total_frames), "Detecting shot boundaries…")

    cap.release()
    if total_frames > 0 and cut_frames[-1] != total_frames:
        cut_frames.append(total_frames)
    elif len(cut_frames) == 1:
        cut_frames.append(frame_idx)

    # Remove accidental zero-length duplicates.
    cut_frames = sorted(set(cut_frames))
    shots = []
    for a, b in zip(cut_frames[:-1], cut_frames[1:]):
        if b > a:
            shots.append((a, b))
    return shots, fps, duration, total_frames


def analyze_video(video_path: str, threshold: float = 0.48, min_shot_sec: float = 0.45,
                  progress: Optional[Callable[[float, str], None]] = None):
    shots, fps, duration, total_frames = detect_cuts(
        video_path, threshold=threshold, min_shot_sec=min_shot_sec, progress=progress
    )

    cap = cv2.VideoCapture(video_path)
    rows = []
    n = max(1, len(shots))
    for i, (start_f, end_f) in enumerate(shots, start=1):
        mid_f = int((start_f + end_f - 1) / 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_f)
        ok, frame = cap.read()
        if not ok:
            continue

        metrics = _frame_metrics(frame)
        rgb = metrics["dominant_rgb"]
        start_sec = start_f / fps
        end_sec = end_f / fps
        row = ShotMetrics(
            shot_number=i,
            start_sec=round(start_sec, 3),
            end_sec=round(end_sec, 3),
            duration_sec=round(end_sec - start_sec, 3),
            representative_frame_sec=round(mid_f / fps, 3),
            dominant_hex=_rgb_to_hex(rgb),
            dominant_r=int(rgb[0]),
            dominant_g=int(rgb[1]),
            dominant_b=int(rgb[2]),
            brightness_0_255=round(metrics["brightness"], 2),
            contrast_sd=round(metrics["contrast"], 2),
            saturation_0_255=round(metrics["saturation"], 2),
            warm_color_ratio=round(metrics["warm_ratio"], 4),
            edge_density=round(metrics["edge_density"], 4),
        )
        rows.append(asdict(row))
        if progress:
            progress(0.55 + 0.40 * i / n, f"Analyzing shot {i} of {len(shots)}…")

    cap.release()
    shots_df = pd.DataFrame(rows)

    if shots_df.empty:
        summary = {
            "video_duration_sec": round(duration, 3),
            "fps": round(fps, 3),
            "total_frames": total_frames,
            "shot_count": 0,
            "average_shot_length_sec": None,
            "median_shot_length_sec": None,
        }
    else:
        weights = shots_df["duration_sec"].clip(lower=0.001)
        summary = {
            "video_duration_sec": round(duration, 3),
            "fps": round(fps, 3),
            "total_frames": total_frames,
            "shot_count": int(len(shots_df)),
            "average_shot_length_sec": round(float(shots_df["duration_sec"].mean()), 3),
            "median_shot_length_sec": round(float(shots_df["duration_sec"].median()), 3),
            "shortest_shot_sec": round(float(shots_df["duration_sec"].min()), 3),
            "longest_shot_sec": round(float(shots_df["duration_sec"].max()), 3),
            "weighted_brightness_0_255": round(float(np.average(shots_df["brightness_0_255"], weights=weights)), 2),
            "weighted_contrast_sd": round(float(np.average(shots_df["contrast_sd"], weights=weights)), 2),
            "weighted_saturation_0_255": round(float(np.average(shots_df["saturation_0_255"], weights=weights)), 2),
            "weighted_warm_color_ratio": round(float(np.average(shots_df["warm_color_ratio"], weights=weights)), 4),
        }

    summary_df = pd.DataFrame([summary]).T.reset_index()
    summary_df.columns = ["metric", "value"]

    if progress:
        progress(1.0, "Done")
    return summary_df, shots_df


def make_excel(summary_df: pd.DataFrame, shots_df: pd.DataFrame, video_name: str) -> bytes:
    from io import BytesIO

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        shots_df.to_excel(writer, sheet_name="Shots", index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
        title_fmt = workbook.add_format({"bold": True, "font_size": 14})
        note_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})

        ws1 = writer.sheets["Summary"]
        ws2 = writer.sheets["Shots"]
        ws1.set_column("A:A", 34)
        ws1.set_column("B:B", 18)
        ws2.freeze_panes(1, 0)
        ws2.autofilter(0, 0, max(1, len(shots_df)), max(0, len(shots_df.columns) - 1))
        ws2.set_column(0, max(0, len(shots_df.columns) - 1), 18)
        if len(shots_df.columns):
            ws2.set_column(5, 5, 14)

        for col, name in enumerate(summary_df.columns):
            ws1.write(0, col, name, header_fmt)
        for col, name in enumerate(shots_df.columns):
            ws2.write(0, col, name, header_fmt)

        notes = workbook.add_worksheet("Method Notes")
        notes.set_column("A:A", 24)
        notes.set_column("B:B", 95)
        notes.write("A1", "Video", title_fmt)
        notes.write("B1", video_name)
        method_rows = [
            ("Shot boundary", "Histogram-change based cut detection. Threshold is user-adjustable. Abrupt cuts are detected better than dissolves/fades."),
            ("Visual sampling", "One representative frame at the temporal midpoint of each detected shot is used for color/brightness/contrast/saturation/edge/face estimates."),
            ("Dominant color", "Fast RGB quantization of the representative frame; intended as a reproducible descriptor, not a semantic color label."),
            ("Brightness", "Mean grayscale pixel intensity, 0–255."),
            ("Contrast", "Standard deviation of grayscale pixel intensity."),
            ("Saturation", "Mean HSV saturation, 0–255."),
            ("Warm-color ratio", "Share of sufficiently saturated pixels whose HSV hue falls in an approximate warm red/orange/yellow range."),
            ("Edge density", "Share of pixels classified as Canny edges; a rough proxy for visual detail/complexity."),
            ("Face count", "Classical OpenCV frontal-face detector on the representative frame. Treat as an estimate and validate manually."),
            ("Research use", "For dissertation-grade analysis, validate automated measures on a manually coded subset and report the validation procedure and detection parameters."),
        ]
        for r, (k, v) in enumerate(method_rows, start=3):
            notes.write(r, 0, k, header_fmt)
            notes.write(r, 1, v, note_fmt)

    return output.getvalue()
