from __future__ import annotations
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime

VIDEO_EXTENSIONS = {".mov", ".mp4", ".mkv", ".avi", ".webm", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".amr"}
ALL_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

# Formats libsndfile (soundfile) decodes directly. Diarization loads audio via
# soundfile to bypass torchaudio's FFmpeg backend, so anything NOT in this set
# must be transcoded to WAV via ffmpeg first (see pipeline extraction gate).
# AMR in particular has no libsndfile support; mp3/m4a/aac are unreliable too.
SOUNDFILE_NATIVE = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def check_ffmpeg() -> tuple[bool, bool]:
    """Returns (ffmpeg_ok, ffprobe_ok)."""
    results = []
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([tool, "-version"], capture_output=True, check=True)
            results.append(True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            results.append(False)
    return tuple(results)


def get_source_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    # Unknown extension: ask ffprobe what's actually inside, so formats like
    # .opus, .wma, .m4b, .3gp, etc. work without being hardcoded. Anything with
    # a decodable audio track is accepted; the pipeline transcodes it to WAV.
    has_video, has_audio = probe_av_streams(path)
    if has_video:
        return "video"
    if has_audio:
        return "audio"
    raise ValueError(f"Unsupported or undecodable file: {path}")


def probe_av_streams(path: str) -> tuple[bool, bool]:
    """Return (has_real_video, has_audio) via ffprobe.

    Embedded cover art (album/poster images) shows up as a video stream tagged
    `attached_pic`; it's ignored here so an audio file with artwork is still
    classified as audio. Returns (False, False) if the file can't be read.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", path],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(result.stdout).get("streams", [])
    except Exception:
        return (False, False)
    has_video = any(
        s.get("codec_type") == "video"
        and not s.get("disposition", {}).get("attached_pic")
        for s in streams
    )
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return (has_video, has_audio)


def get_media_info(path: str) -> dict:
    """Returns duration_sec and file_created_at from ffprobe + os.stat."""
    duration_sec = None
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                path,
            ],
            capture_output=True, text=True, check=True,
        )
        info = json.loads(result.stdout)
        duration_sec = float(info.get("format", {}).get("duration", 0)) or None
    except Exception:
        pass

    file_created_at = None
    try:
        mtime = os.path.getmtime(path)
        file_created_at = datetime.fromtimestamp(mtime).isoformat()
    except Exception:
        pass

    return {"duration_sec": duration_sec, "file_created_at": file_created_at}


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
