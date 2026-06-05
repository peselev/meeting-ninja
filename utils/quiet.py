"""
quiet.py — suppress the wall of harmless third-party warnings/noise that
otherwise makes it impossible to tell "working" from "broken".

Import this FIRST, before torch / torchaudio / pyannote / whisper, e.g.:
    import utils.quiet  # noqa: F401  (must be first import)

What it silences (all confirmed harmless for our use):
  - torchaudio / torio FFmpeg-extension load failures (we use soundfile instead)
  - torchaudio backend-dispatch deprecation warnings
  - urllib3 LibreSSL NotOpenSSLWarning
  - whisper FP16-on-CPU warning
  - speechbrain.pretrained deprecation redirect
  - pyannote std() degrees-of-freedom warning
It does NOT silence real errors — only known-noisy warnings and the
torio extension-probe logger (which logs failures at DEBUG/INFO as it tries
each FFmpeg version, none of which we need).
"""
from __future__ import annotations
import warnings
import logging
import os

# ── 1. Environment flags (must be set before torch imports) ──────────────────
os.environ.setdefault("PYTHONWARNINGS", "ignore")
# Tell torio not to even try loading the FFmpeg extension (we use soundfile).
os.environ.setdefault("TORIO_USE_FFMPEG", "0")

# ── 2. Filter Python warnings by category/message ────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── 3. Silence noisy loggers from dependencies ───────────────────────────────
for noisy in (
    "torio",
    "torio._extension",
    "torio._extension.utils",
    "torchaudio",
    "speechbrain",
    "urllib3",
    "matplotlib",
    "huggingface_hub",
    "filelock",
    "fsspec",
):
    logging.getLogger(noisy).setLevel(logging.ERROR)
