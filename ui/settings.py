from __future__ import annotations
import sys
import subprocess
import streamlit as st
from pathlib import Path
from db import client as db
from utils.media_info import check_ffmpeg
from processing.transcribe import WHISPER_MODELS
from processing.diarize import is_pyannote_available


def _pick_folder(current: str) -> str:
    """
    Open a native folder picker. Cross-platform:
      - macOS:   osascript (built in)
      - Windows: PowerShell FolderBrowserDialog (built in)
      - Linux:   zenity or kdialog if available
    Returns chosen path, or `current` if cancelled/unavailable.
    """
    start = current.strip() if current.strip() else str(Path.home())

    try:
        if sys.platform == "darwin":
            script = (
                f'tell application "Finder" to set f to choose folder '
                f'with prompt "Select folder" default location POSIX file "{start}"\n'
                f'return POSIX path of f'
            )
            r = subprocess.run(["osascript", "-e", script],
                               capture_output=True, text=True, timeout=120)
            chosen = r.stdout.strip().rstrip("/")
            return chosen if chosen else current

        elif sys.platform.startswith("win"):
            # PowerShell folder browser dialog
            ps = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
                f"$f.SelectedPath = '{start}'; "
                "$null = $f.ShowDialog(); "
                "Write-Output $f.SelectedPath"
            )
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True, text=True, timeout=120,
            )
            chosen = r.stdout.strip()
            return chosen if chosen else current

        else:  # Linux / other
            for tool, args in (
                ("zenity", ["zenity", "--file-selection", "--directory",
                            f"--filename={start}/"]),
                ("kdialog", ["kdialog", "--getexistingdirectory", start]),
            ):
                try:
                    r = subprocess.run(args, capture_output=True, text=True, timeout=120)
                    chosen = r.stdout.strip()
                    if chosen:
                        return chosen
                except FileNotFoundError:
                    continue
            return current

    except Exception:
        return current


def render_sidebar():
    """Render settings in the sidebar (collapsible by default via the sidebar toggle)."""
    with st.sidebar:
        st.header("⚙️ Settings")

        # ── Environment status ────────────────────────────────────────────────
        with st.expander("Environment", expanded=False):
            ffmpeg_ok, ffprobe_ok = check_ffmpeg()
            st.write("ffmpeg:", "✅" if ffmpeg_ok else "❌ missing")
            st.write("ffprobe:", "✅" if ffprobe_ok else "❌ missing")
            st.write("pyannote:", "✅" if is_pyannote_available()
                     else "⚠️ not installed (diarization disabled)")

        # Backing state for folder pickers
        if "home_folder_val" not in st.session_state:
            st.session_state["home_folder_val"] = db.get_setting("home_folder", "")
        if "dest_folder_val" not in st.session_state:
            st.session_state["dest_folder_val"] = db.get_setting("destination_folder", "")

        # ── Home folder ───────────────────────────────────────────────────────
        st.subheader("Folders")
        home_folder = st.text_input(
            "Home folder",
            value=st.session_state["home_folder_val"],
            placeholder="/path/to/Recordings",
            help="Where recordings live. Audio and transcripts are saved here.",
        )
        st.session_state["home_folder_val"] = home_folder
        if st.button("📂 Browse…", key="browse_home", use_container_width=True):
            chosen = _pick_folder(st.session_state["home_folder_val"])
            st.session_state["home_folder_val"] = chosen
            st.rerun()

        # ── Destination folder (was "jobs root") ───────────────────────────────
        dest_folder = st.text_input(
            "Destination folder",
            value=st.session_state["dest_folder_val"],
            placeholder="(optional)",
            help="Optional. When a file has a Tag/ID, its transcript is copied to "
                 "{destination}/{tag}/transcript-{name}.txt",
        )
        st.session_state["dest_folder_val"] = dest_folder
        if st.button("📂 Browse…", key="browse_dest", use_container_width=True):
            chosen = _pick_folder(st.session_state["dest_folder_val"])
            st.session_state["dest_folder_val"] = chosen
            st.rerun()

        # ── Token + defaults ───────────────────────────────────────────────────
        st.subheader("Diarization")
        hf_token = st.text_input(
            "HuggingFace token",
            value=db.get_setting("hf_token", ""),
            type="password",
            help="For pyannote speaker diarization. huggingface.co/settings/tokens",
        )

        default_model = db.get_setting("default_whisper_model", "base")
        idx = WHISPER_MODELS.index(default_model) if default_model in WHISPER_MODELS else 1
        selected_model = st.selectbox("Default Whisper model", WHISPER_MODELS, index=idx)

        st.divider()
        if st.button("💾 Save settings", type="primary", use_container_width=True):
            home = st.session_state["home_folder_val"].strip()
            dest = st.session_state["dest_folder_val"].strip()
            if not home:
                st.error("Home folder is required.")
            else:
                db.set_setting("home_folder", home)
                db.set_setting("destination_folder", dest)
                db.set_setting("hf_token", hf_token.strip())
                db.set_setting("default_whisper_model", selected_model)
                st.success("Saved.")
                st.rerun()
