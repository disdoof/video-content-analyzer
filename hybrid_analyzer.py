from __future__ import annotations

from io import BytesIO
from typing import Callable, Optional, Set

import cv2
import numpy as np
import pandas as pd

SHOT_ANALYSIS = "shot"
WARM_COLOR = "warm_color"
SATURATION = "saturation"
CONTRAST = "contrast"
VALID_ANALYSES = {SHOT_ANALYSIS, WARM_COLOR, SATURATION, CONTRAST}

# Initial v2 hybrid weights. These are intentionally transparent and should be
# calibrated/validated before final thesis use.
W_GLOBAL = 0.05
W_GRID = 0.15
W_STRUCT = 0.50
W_EDGE = 0.30
GRID_ROWS = 3
GRID_COLS = 3
PROCESS_WIDTH = 480
PEAK_RADIUS = 2
MIN_PROMINENCE = 0.07
VERY_STRONG_SCORE = 0.76


def _resize(frame: np.ndarray, width: int = PROCESS_WIDTH) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    nh = max(1, int(round(h * width / w)))
    return cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)


def _histogram(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [40, 48], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def _hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    corr = float(cv2.compareHist(_histogram(a), _histogram(b), cv2.HISTCMP_CORREL))
    return float(np.clip(1.0 - corr, 0.0, 1.0))


def _grid_hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    h, w = a.shape[:2]
    values = []
    for r in range(GRID_ROWS):
        y0, y1 = r * h // GRID_ROWS, (r + 1) * h // GRID_ROWS
        for c in range(GRID_COLS):
            x0, x1 = c * w // GRID_COLS, (c + 1) * w // GRID_COLS
            values.append(_hist_distance(a[y0:y1, x0:x1], b[y0:y1, x0:x1]))
    return float(np.mean(values)) if values else 0.0


def _structure_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Illumination-normalized structural difference, 0=same, 1=very different.

    Global mean/std normalization makes this deliberately less sensitive to a
    simple fade/dimming of an otherwise unchanged frame.
    """
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    na = (ga - ga.mean()) / (ga.std() + 1e-6)
    nb = (gb - gb.mean()) / (gb.std() + 1e-6)
    corr = float(np.mean(na * nb))
    return float(np.clip(1.0 - corr, 0.0, 1.0))


def _edge_change(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ea = cv2.Canny(ga, 60, 160)
    eb = cv2.Canny(gb, 60, 160)
    kernel = np.ones((3, 3), np.uint8)
    ea_d = cv2.dilate(ea, kernel, iterations=1)
    eb_d = cv2.dilate(eb, kernel, iterations=1)
    ma, mb = ea > 0, eb > 0
    unmatched_a = np.logical_and(ma, eb_d == 0).sum()
    unmatched_b = np.logical_and(mb, ea_d == 0).sum()
    denom = ma.sum() + mb.sum() + 1e-6
    return float(np.clip((unmatched_a + unmatched_b) / denom, 0.0, 1.0))


def _brightness_delta(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float((float(gb.mean()) - float(ga.mean())) / 255.0)


def _pair_features(prev_frame: np.ndarray, cur_frame: np.ndarray) -> dict:
    a = _resize(prev_frame)
    b = _resize(cur_frame)
    global_hist = _hist_distance(a, b)
    grid_hist = _grid_hist_distance(a, b)
    structural = _structure_difference(a, b)
    edge = _edge_change(a, b)
    brightness = _brightness_delta(a, b)
    score = (
        W_GLOBAL * global_hist
        + W_GRID * grid_hist
        + W_STRUCT * structural
        + W_EDGE * edge
    )
    return {
        "global_hist": global_hist,
        "grid_hist": grid_hist,
        "structural_diff": structural,
        "edge_change": edge,
        "brightness_delta": brightness,
        "hybrid_score": float(np.clip(score, 0.0, 1.0)),
    }


def _frame_to_timecode(frame_idx: int, fps: float) -> str:
    fps = float(fps) if fps and fps > 0 else 25.0
    t = max(0.0, float(frame_idx) / fps)
    whole_seconds = int(np.floor(t + 1e-9))
    minutes = whole_seconds // 60
    seconds = whole_seconds % 60
    nominal_fps = max(1, int(round(fps)))
    frame_in_second = int(np.floor((t - whole_seconds) * fps + 1e-6))
    frame_in_second = min(nominal_fps - 1, max(0, frame_in_second))
    return f"{minutes:02d}:{seconds:02d}:{frame_in_second:02d}"


def _video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video could not be opened. Try MP4/H.264 if possible.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()
    return fps, total_frames, duration


def _encode_thumbnail(frame_bgr: np.ndarray, max_width: int = 480) -> bytes:
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        nh = max(1, int(round(h * max_width / w)))
        frame_bgr = cv2.resize(frame_bgr, (max_width, nh), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 84])
    return buf.tobytes() if ok else b""


def _local_peak_and_prominence(scores: np.ndarray, radius: int = PEAK_RADIUS):
    n = len(scores)
    local_peak = np.zeros(n, dtype=bool)
    prominence = np.zeros(n, dtype=float)
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        neighborhood = scores[lo:hi]
        local_peak[i] = scores[i] >= np.max(neighborhood) - 1e-12
        others = np.concatenate([scores[lo:i], scores[i + 1:hi]])
        baseline = float(np.median(others)) if len(others) else 0.0
        prominence[i] = float(scores[i] - baseline)
    return local_peak, prominence


def detect_hybrid_cuts(
    video_path: str,
    threshold: float = 0.48,
    min_shot_sec: float = 0.20,
    progress: Optional[Callable[[float, str], None]] = None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video could not be opened. Try MP4/H.264 if possible.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    rows = []
    ok, prev = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Video contains no readable frames.")

    frame_idx = 1
    while True:
        ok, cur = cap.read()
        if not ok:
            break
        feat = _pair_features(prev, cur)
        rows.append({
            "boundary_frame": frame_idx,
            "timecode": _frame_to_timecode(frame_idx, fps),
            **feat,
        })
        prev = cur
        frame_idx += 1
        if progress and total_frames and frame_idx % max(1, int(fps)) == 0:
            progress(min(0.82, 0.82 * frame_idx / total_frames), "Running hybrid cut detection…")
    cap.release()

    cand = pd.DataFrame(rows)
    if cand.empty:
        return [(0, max(1, total_frames))], pd.DataFrame(), pd.DataFrame(), fps, duration

    scores = cand["hybrid_score"].to_numpy(float)
    peaks, prominence = _local_peak_and_prominence(scores)
    cand["local_peak"] = peaks
    cand["prominence"] = prominence

    # Fade/illumination guard: if geometry is still stable, a large global-color
    # jump is not enough to call a cut. This directly targets fade-like false positives.
    cand["fade_like_structure"] = (
        (cand["structural_diff"] < 0.36)
        & (cand["edge_change"] < 0.62)
    )
    cand["score_pass"] = cand["hybrid_score"] >= float(threshold)
    cand["temporal_pass"] = (
        (cand["prominence"] >= MIN_PROMINENCE)
        | (cand["hybrid_score"] >= VERY_STRONG_SCORE)
    )
    cand["preliminary_candidate"] = (
        cand["score_pass"]
        & cand["local_peak"]
        & cand["temporal_pass"]
        & (~cand["fade_like_structure"])
    )

    # Post-process close candidates. Unlike v1, raw evidence is computed first;
    # the minimum-duration rule no longer blinds the detector before scoring.
    min_gap = max(1, int(round(float(min_shot_sec) * fps)))
    prelim = cand[cand["preliminary_candidate"]].copy()
    selected_indices = []
    for idx, rec in prelim.iterrows():
        frame = int(rec["boundary_frame"])
        if not selected_indices:
            selected_indices.append(idx)
            continue
        last_idx = selected_indices[-1]
        last_frame = int(cand.loc[last_idx, "boundary_frame"])
        if frame - last_frame >= min_gap:
            selected_indices.append(idx)
        else:
            # Keep the stronger boundary within a very short cluster.
            if float(rec["hybrid_score"]) > float(cand.loc[last_idx, "hybrid_score"]):
                selected_indices[-1] = idx

    cand["accepted_cut"] = False
    if selected_indices:
        cand.loc[selected_indices, "accepted_cut"] = True

    def reason(rec):
        if bool(rec["accepted_cut"]):
            return "Accepted cut"
        if not bool(rec["score_pass"]):
            return "Rejected: below hybrid threshold"
        if bool(rec["fade_like_structure"]):
            return "Rejected: fade/illumination-like change"
        if not bool(rec["local_peak"]):
            return "Rejected: not a local peak"
        if not bool(rec["temporal_pass"]):
            return "Rejected: weak temporal prominence"
        if bool(rec["preliminary_candidate"]):
            return "Rejected: too close to stronger cut"
        return "Rejected"

    cand["decision"] = cand.apply(reason, axis=1)

    cut_frames = cand.loc[cand["accepted_cut"], "boundary_frame"].astype(int).tolist()
    boundaries = [0] + cut_frames + [total_frames if total_frames > 0 else frame_idx]
    boundaries = sorted(set(boundaries))
    shots = [(a, b) for a, b in zip(boundaries[:-1], boundaries[1:]) if b > a]

    # Events for refinement: all accepted cuts + strongest rejected local peaks.
    event_mask = cand["accepted_cut"]
    rejected = cand[(~cand["accepted_cut"]) & cand["local_peak"]].nlargest(60, "hybrid_score")
    event_indices = sorted(set(cand[event_mask].index.tolist() + rejected.index.tolist()))
    events = cand.loc[event_indices].copy().reset_index(drop=True)

    if progress:
        progress(0.86, "Preparing validation frames…")
    return shots, cand, events, fps, duration


def _extract_cut_images(video_path: str, cut_events: pd.DataFrame) -> pd.DataFrame:
    if cut_events.empty:
        return cut_events
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return cut_events
    before, after = [], []
    for boundary in cut_events["boundary_frame"].astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, boundary - 1))
        ok, f = cap.read(); before.append(_encode_thumbnail(f) if ok else b"")
        cap.set(cv2.CAP_PROP_POS_FRAMES, boundary)
        ok, f = cap.read(); after.append(_encode_thumbnail(f) if ok else b"")
    cap.release()
    out = cut_events.copy()
    out["before_frame_bytes"] = before
    out["after_frame_bytes"] = after
    return out


def _extract_shot_images(video_path: str, shots: list[tuple[int, int]]):
    cap = cv2.VideoCapture(video_path)
    result = []
    if not cap.isOpened():
        return [(b"", b"") for _ in shots]
    for start_f, end_f in shots:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_f))
        ok, f = cap.read(); first = _encode_thumbnail(f) if ok else b""
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(start_f, end_f - 1)))
        ok, f = cap.read(); last = _encode_thumbnail(f) if ok else b""
        result.append((first, last))
    cap.release()
    return result


def _selected_frame_metrics(frame_bgr: np.ndarray, analyses: Set[str]) -> dict:
    result = {}
    if WARM_COLOR in analyses or SATURATION in analyses:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        if WARM_COLOR in analyses:
            h, s = hsv[:, :, 0], hsv[:, :, 1]
            result["warm_color_ratio_0_1"] = float((((h <= 35) | (h >= 170)) & (s >= 40)).mean())
        if SATURATION in analyses:
            result["color_saturation_score_0_1"] = float(hsv[:, :, 1].mean()) / 255.0
    if CONTRAST in analyses:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        result["contrast_score_0_1"] = min(1.0, float(gray.std()) / 127.5)
    return result


def _sample_visual_measures(video_path, analyses, fps, total_frames, progress=None):
    visual = set(analyses) & {WARM_COLOR, SATURATION, CONTRAST}
    if not visual or total_frames <= 0:
        return {}, 0
    duration = total_frames / fps if fps > 0 else 0
    count = min(300, max(1, int(np.ceil(duration))))
    indices = np.unique(np.linspace(0, total_frames - 1, count, dtype=int))
    cap = cv2.VideoCapture(video_path)
    collected = {k: [] for k in ["warm_color_ratio_0_1", "color_saturation_score_0_1", "contrast_score_0_1"]}
    for i, idx in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        vals = _selected_frame_metrics(frame, visual)
        for k, v in vals.items():
            collected[k].append(v)
        if progress:
            progress(i / max(1, len(indices)), f"Analyzing visual sample {i} of {len(indices)}…")
    cap.release()
    return {k: float(np.mean(v)) for k, v in collected.items() if v}, len(indices)


def analyze_video(
    video_path: str,
    analyses,
    threshold: float = 0.48,
    min_shot_sec: float = 0.20,
    progress=None,
):
    analyses = set(analyses) & VALID_ANALYSES
    if not analyses:
        raise ValueError("Select at least one analysis before running the video.")
    fps, total_frames, duration = _video_info(video_path)
    wants_shots = SHOT_ANALYSIS in analyses
    wants_visual = bool(analyses & {WARM_COLOR, SATURATION, CONTRAST})

    shots_df = pd.DataFrame(); candidates = pd.DataFrame(); events = pd.DataFrame(); cuts = pd.DataFrame(); shots = []
    if wants_shots:
        def hp(p, text):
            if progress:
                progress((0.72 if wants_visual else 0.94) * p, text)
        shots, candidates, events, fps, duration = detect_hybrid_cuts(
            video_path, threshold=threshold, min_shot_sec=min_shot_sec, progress=hp
        )
        accepted = candidates[candidates["accepted_cut"]].copy().reset_index(drop=True)
        accepted.insert(0, "cut_number", np.arange(1, len(accepted) + 1))
        cuts = _extract_cut_images(video_path, accepted)
        shot_imgs = _extract_shot_images(video_path, shots)
        shot_rows = []
        for i, ((start_f, end_f), (first_b, last_b)) in enumerate(zip(shots, shot_imgs), start=1):
            shot_rows.append({
                "shot_number": i,
                "start_timecode": _frame_to_timecode(start_f, fps),
                "end_timecode": _frame_to_timecode(max(start_f, end_f - 1), fps),
                "duration_sec": round((end_f - start_f) / fps, 3),
                "first_frame_bytes": first_b,
                "last_frame_bytes": last_b,
            })
        shots_df = pd.DataFrame(shot_rows)
        events = _extract_cut_images(video_path, events)

    visual_means = {}; sample_count = 0
    if wants_visual:
        def vp(p, text):
            if progress:
                progress((0.72 if wants_shots else 0.0) + (0.24 if wants_shots else 0.95) * p, text)
        visual_means, sample_count = _sample_visual_measures(video_path, analyses, fps, total_frames, vp)

    summary = [("Video duration (sec)", round(duration, 3))]
    if wants_shots:
        summary.extend([
            ("Detected cuts", len(cuts)),
            ("Detected shots", len(shots_df)),
            ("Average shot length (sec)", round(float(shots_df["duration_sec"].mean()), 3) if not shots_df.empty else None),
        ])
    if WARM_COLOR in analyses:
        v = visual_means.get("warm_color_ratio_0_1"); summary.append(("Warm-color ratio (0–1)", round(v,4) if v is not None else None))
    if SATURATION in analyses:
        v = visual_means.get("color_saturation_score_0_1"); summary.append(("Color saturation score (0–1)", round(v,4) if v is not None else None))
    if CONTRAST in analyses:
        v = visual_means.get("contrast_score_0_1"); summary.append(("Contrast score (0–1)", round(v,4) if v is not None else None))

    meta = {
        "selected_analyses": sorted(analyses),
        "fps": round(fps, 3),
        "threshold": threshold,
        "min_shot_sec": min_shot_sec,
        "visual_sample_count": sample_count,
        "detector_version": "2.0 Hybrid",
        "weights": f"global={W_GLOBAL}, grid={W_GRID}, structural={W_STRUCT}, edge={W_EDGE}",
    }
    if progress:
        progress(1.0, "Done")
    return pd.DataFrame(summary, columns=["Measure", "Value"]), shots_df, cuts, candidates, events, meta


def make_excel(summary_df, shots_df, cuts_df, candidates_df, events_df, video_name: str, metadata: Optional[dict] = None) -> bytes:
    metadata = metadata or {}
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        s = shots_df.drop(columns=["first_frame_bytes", "last_frame_bytes"], errors="ignore")
        c = cuts_df.drop(columns=["before_frame_bytes", "after_frame_bytes"], errors="ignore")
        e = events_df.drop(columns=["before_frame_bytes", "after_frame_bytes"], errors="ignore")
        if not s.empty: s.to_excel(writer, sheet_name="Shots", index=False)
        if not c.empty: c.to_excel(writer, sheet_name="Detected cuts", index=False)
        if not candidates_df.empty: candidates_df.to_excel(writer, sheet_name="Hybrid diagnostics", index=False)
        if not e.empty: e.to_excel(writer, sheet_name="Top events", index=False)

        wb = writer.book
        header = wb.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
        wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        for name, ws in writer.sheets.items():
            ws.freeze_panes(1, 0)
            ws.set_column(0, 30, 19)
            if name == "Summary":
                ws.set_column("A:A", 38); ws.set_column("B:B", 22)
            df = {"Summary": summary_df, "Shots": s, "Detected cuts": c, "Hybrid diagnostics": candidates_df, "Top events": e}.get(name)
            if df is not None:
                for col, label in enumerate(df.columns): ws.write(0, col, label, header)

        method = wb.add_worksheet("Method")
        method.set_column("A:A", 32); method.set_column("B:B", 105)
        rows = [
            ("Video", video_name),
            ("Detector", "Hybrid Cut Detector v2.0"),
            ("Hybrid threshold", str(metadata.get("threshold", ""))),
            ("Minimum shot duration", f'{metadata.get("min_shot_sec", "")} seconds'),
            ("Video FPS", str(metadata.get("fps", ""))),
            ("Timecode", "MM:SS:FF (minute:second:frame), based on the video's actual FPS."),
            ("Hybrid weights", metadata.get("weights", "")),
            ("Global histogram", "Whole-frame HSV hue/saturation histogram change. Kept at low weight because it can react poorly to fade/dimming and can miss same-palette composition cuts."),
            ("Grid histogram", "The frame is split into a 3x3 grid; local HSV histogram changes are averaged to capture spatial/color rearrangement."),
            ("Structural difference", "Grayscale frames are normalized for mean and contrast, then compared by correlation. This emphasizes composition/layout change while reducing sensitivity to simple dimming."),
            ("Edge change", "Canny edge maps are compared with small spatial tolerance, targeting shape/contour changes in flat animation."),
            ("Temporal guard", "A cut candidate must be a local peak with sufficient prominence, unless the hybrid score is exceptionally strong."),
            ("Fade guard", "Large color changes are rejected when structural and edge evidence remain stable, reducing false cuts during fades or illumination changes."),
            ("Calibration status", "The v2 weights and thresholds are initial calibration values. They must be validated against manually coded ground truth before final research use."),
        ]
        for r, (a, b) in enumerate(rows):
            method.write(r, 0, a, header); method.write(r, 1, b, wrap)
    return out.getvalue()
