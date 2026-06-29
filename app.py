from __future__ import annotations
import meeting_ninja.utils.quiet  # noqa: F401  — MUST be imported before torch/pyannote/whisper
import streamlit as st
from meeting_ninja.db.client import init_db
from meeting_ninja.utils.media_info import check_ffmpeg

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Transcriber",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DB init ───────────────────────────────────────────────────────────────────
init_db()

# ── ffmpeg check (warn once, don't block) ────────────────────────────────────
ffmpeg_ok, ffprobe_ok = check_ffmpeg()
if not ffmpeg_ok or not ffprobe_ok:
    st.warning(
        "⚠️ **ffmpeg / ffprobe not found.** "
        "Install with `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Linux). "
        "The app will not be able to process video files until this is resolved."
    )

st.title("🎙️ Meeting Transcriber")

# ── Settings live in the sidebar (collapsible) ───────────────────────────────
from ui.settings import render_sidebar
render_sidebar()

# ── Main area: either the file manager, or the speaker-labeling view ──────────
# Labeling is an ACTION triggered from a file row, not a nav tab.
if st.session_state.get("label_file_id"):
    from ui.speaker_labeling import render
    render()
else:
    from ui.file_manager import render
    render()
