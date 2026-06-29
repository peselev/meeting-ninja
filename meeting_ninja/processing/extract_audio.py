from __future__ import annotations
import subprocess
from pathlib import Path


def extract_audio(source_path: str, home_folder: str, offset_sec: float = 0.0,
                  out_stem: str | None = None) -> str:
    """
    Extract audio from a video OR audio file to 16kHz mono WAV.
    offset_sec: skip this many seconds from the start (dead air at the beginning).
    out_stem: name the output WAV after this stem instead of the source filename.
    Returns the path to the extracted .wav file.
    Raises subprocess.CalledProcessError on ffmpeg failure.
    """
    audio_dir = Path(home_folder) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    stem = out_stem or Path(source_path).stem
    out_path = audio_dir / f"{stem}.wav"

    cmd = ["ffmpeg", "-y"]
    # -ss before -i is fast (keyframe seek) and accurate enough for trimming dead air
    if offset_sec and offset_sec > 0:
        cmd += ["-ss", str(offset_sec)]
    cmd += [
        "-i", source_path,
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(out_path),
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    return str(out_path)
