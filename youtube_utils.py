from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse


ProgressFn = Optional[Callable[[float, str], None]]


def is_youtube_url(url: str) -> bool:
    """Return True for ordinary YouTube/youtu.be URLs."""
    try:
        host = (urlparse(url.strip()).hostname or "").lower()
    except Exception:
        return False

    return (
        host == "youtu.be"
        or host.endswith(".youtu.be")
        or host == "youtube.com"
        or host.endswith(".youtube.com")
        or host == "youtube-nocookie.com"
        or host.endswith(".youtube-nocookie.com")
    )


def download_youtube_video(url: str, output_dir: str | Path, progress: ProgressFn = None):
    """
    Download one YouTube video to output_dir using yt-dlp.

    Returns (video_path, metadata_dict), where metadata_dict contains title,
    uploader, video_id, webpage_url, and duration when available.
    """
    if not is_youtube_url(url):
        raise ValueError("Please enter a valid YouTube or youtu.be link.")

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "YouTube support is not installed. Run setup again so yt-dlp is installed."
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        progress(0.01, "Preparing YouTube download…")

    def hook(data):
        if not progress:
            return
        status = data.get("status")
        if status == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            downloaded = data.get("downloaded_bytes") or 0
            if total:
                fraction = min(max(downloaded / total, 0.0), 1.0)
                progress(0.02 + 0.23 * fraction, f"Downloading YouTube video… {fraction:.0%}")
            else:
                progress(0.08, "Downloading YouTube video…")
        elif status == "finished":
            progress(0.25, "YouTube download complete. Preparing analysis…")

    ydl_opts = {
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "format": "bestvideo[vcodec^=avc1][ext=mp4][height<=720]/best[vcodec^=avc1][ext=mp4][height<=720]/bestvideo[ext=mp4][height<=720]/best[ext=mp4][height<=720]/bestvideo[height<=720]/best[height<=720]/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url.strip(), download=True)
            if info is None:
                raise RuntimeError("YouTube did not return video information.")

            requested = info.get("requested_downloads") or []
            candidates = []
            for item in requested:
                filepath = item.get("filepath")
                if filepath:
                    candidates.append(Path(filepath))

            prepared = ydl.prepare_filename(info)
            if prepared:
                candidates.append(Path(prepared))

            video_id = info.get("id")
            if video_id:
                candidates.extend(output_dir.glob(f"{video_id}.*"))

            video_path = next((p for p in candidates if p.exists() and p.is_file()), None)
            if video_path is None:
                raise RuntimeError("The video was downloaded, but the local file could not be located.")

            metadata = {
                "title": info.get("title") or video_id or "youtube_video",
                "uploader": info.get("uploader") or info.get("channel") or "",
                "video_id": video_id or "",
                "webpage_url": info.get("webpage_url") or url.strip(),
                "duration": info.get("duration"),
            }
            return str(video_path), metadata

    except Exception as exc:
        message = str(exc)
        if "Sign in" in message or "bot" in message.lower() or "cookies" in message.lower():
            raise RuntimeError(
                "YouTube blocked automated access for this video. Update yt-dlp and try again, "
                "or download the video yourself and use Upload Video."
            ) from exc
        raise RuntimeError(f"Could not download this YouTube video: {message}") from exc
