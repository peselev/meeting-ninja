from __future__ import annotations
import streamlit as st
from datetime import datetime
from pathlib import Path
import time

from meeting_ninja.db import client as db
from meeting_ninja.utils.media_info import (
    get_source_type, get_media_info, format_duration, ALL_EXTENSIONS
)
from meeting_ninja.processing.pipeline import run_pipeline, get_progress_queue
from meeting_ninja.processing.transcribe import WHISPER_MODELS
from meeting_ninja.processing.scanner import scan_home_folder


STATUS_EMOJI = {
    "pending":      "⏳ Pending",
    "extracting":   "🔊 Extracting audio",
    "transcribing": "✍️ Transcribing",
    "diarizing":    "🗣️ Diarizing",
    "transcribed":  "📝 Transcribed",
    "labeled":      "🏷️ Labeled",
    "done":         "✅ Done",
    "error":        "❌ Error",
}


def _drain_progress():
    q = get_progress_queue()
    while not q.empty():
        try:
            msg = q.get_nowait()
            fid = msg["file_id"]
            if msg.get("done"):
                st.session_state.processing_ids.discard(fid)
        except Exception:
            pass


# Rough progress mapping per pipeline stage (for the progress bar)
_STAGE_FRACTIONS = {
    "pending":      0.05,
    "extracting":   0.20,
    "transcribing": 0.50,
    "diarizing":    0.80,
    "transcribed":  0.60,
    "labeled":      0.95,
    "done":         1.00,
    "error":        1.00,
}


def _stage_fraction(stage: str) -> float:
    return _STAGE_FRACTIONS.get(stage, 0.05)


def render():
    home_folder = db.get_setting("home_folder", "")

    # ── Guard ────────────────────────────────────────────────────────────────
    if not home_folder:
        st.warning("Set a home folder to get started.")
        st.session_state["active_tab"] = "Settings"
        st.rerun()
        return

    if "processing_ids" not in st.session_state:
        st.session_state.processing_ids = set()

    # ── Auto-scan on first load of this session ──────────────────────────────
    if "did_initial_scan" not in st.session_state:
        n = scan_home_folder(home_folder)
        st.session_state["did_initial_scan"] = True
        if n:
            st.session_state["scan_message"] = f"Auto-detected {n} new file(s) in home folder."

    # Drain progress and auto-refresh while processing
    if st.session_state.processing_ids:
        _drain_progress()

    # ── Active progress panel ────────────────────────────────────────────────
    if st.session_state.processing_ids:
        active_files = [f for f in db.get_all_files() if f["id"] in st.session_state.processing_ids]
        with st.container(border=True):
            st.markdown("**Processing…**")
            for f in active_files:
                stage = f["status"]
                label = STATUS_EMOJI.get(stage, stage)
                st.write(f"{f['filename']} — {label}")
                st.progress(_stage_fraction(stage))
            st.caption("This can take a while for large files or large Whisper models. The page refreshes automatically.")
        time.sleep(1.5)
        st.rerun()

    # ── Toolbar row 1: scan + empty ──────────────────────────────────────────
    col_scan, col_empty, col_spacer = st.columns([1, 1, 3])
    with col_scan:
        if st.button("🔄 Rescan folder", use_container_width=True):
            n = scan_home_folder(home_folder)
            st.session_state["scan_message"] = (
                f"Found {n} new file(s)." if n else "No new files found."
            )
            st.rerun()
    with col_empty:
        if st.button("🗑️ Empty list", use_container_width=True):
            st.session_state["confirm_empty"] = True

    # Empty-list confirmation
    if st.session_state.get("confirm_empty"):
        st.warning("Remove ALL files from the list? (Your recordings and transcripts on disk are kept.)")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, empty the list", type="primary"):
                for f in db.get_all_files():
                    db.delete_file(f["id"])
                st.session_state["confirm_empty"] = False
                st.session_state.processing_ids = set()
                st.rerun()
        with c2:
            if st.button("Cancel"):
                st.session_state["confirm_empty"] = False
                st.rerun()

    if st.session_state.get("scan_message"):
        st.caption(st.session_state["scan_message"])

    # ── Toolbar row 2: upload + model + language + process ───────────────────
    col_add, col_model, col_lang, col_process = st.columns([2, 2, 1, 1])

    with col_add:
        uploaded = st.file_uploader(
            "Add files manually",
            type=[e.lstrip(".") for e in ALL_EXTENSIONS],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    with col_model:
        default_model = db.get_setting("default_whisper_model", "base")
        model_idx = WHISPER_MODELS.index(default_model) if default_model in WHISPER_MODELS else 1
        model_name = st.selectbox(
            "Whisper model",
            WHISPER_MODELS,
            index=model_idx,
            help="Larger models are more accurate but slower.",
        )

    with col_lang:
        lang_options = ["auto", "en", "ru", "he", "es", "fr", "de"]
        language = st.selectbox(
            "Language",
            lang_options,
            index=1,  # default to English
            help="Force a language, or 'auto' to detect. Auto can misfire on quiet intros.",
        )

    with col_process:
        st.write("")
        process_clicked = st.button("▶ Process", type="primary", use_container_width=True)

    # ── Handle manual uploads (dedup by resolved path) ───────────────────────
    if uploaded:
        existing_paths = {str(Path(f["source_path"]).resolve()) for f in db.get_all_files()}
        incoming_dir = Path(home_folder) / "incoming"
        incoming_dir.mkdir(parents=True, exist_ok=True)
        added = 0
        for uf in uploaded:
            dest = (incoming_dir / uf.name).resolve()
            if str(dest) in existing_paths:
                continue  # already tracked — skip silently
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            try:
                source_type = get_source_type(str(dest))
            except ValueError:
                st.warning(f"Skipped {uf.name}: unsupported format.")
                continue
            info = get_media_info(str(dest))
            db.add_file({
                "source_path":     str(dest),
                "filename":        uf.name,
                "source_type":     source_type,
                "duration_sec":    info["duration_sec"],
                "file_created_at": info["file_created_at"],
                "added_at":        datetime.now().isoformat(),
                "status":          "pending",
            })
            existing_paths.add(str(dest))
            added += 1
        if added:
            st.rerun()

    # ── Handle Process ───────────────────────────────────────────────────────
    if process_clicked:
        files = db.get_all_files()
        pending_ids = [f["id"] for f in files if f["status"] in ("pending", "error")]
        if not pending_ids:
            st.info("No pending files to process.")
        else:
            hf_token  = db.get_setting("hf_token") or None
            jobs_root = (db.get_setting("destination_folder")
                         or db.get_setting("jobs_root_folder") or None)
            for fid in pending_ids:
                st.session_state.processing_ids.add(fid)
            run_pipeline(pending_ids, model_name, home_folder, hf_token, jobs_root, language)
            st.rerun()

    # ── Files table ──────────────────────────────────────────────────────────
    files = db.get_all_files()

    if not files:
        st.info("No files yet. Drop recordings in your home folder and click Rescan, or upload manually above.")
        return

    st.divider()
    st.caption(f"{len(files)} file(s)")

    for f in files:
        fid = f["id"]
        is_processing = fid in st.session_state.processing_ids
        status_str = STATUS_EMOJI.get(f["status"], f["status"])

        with st.container(border=True):
            r1a, r1b, r1c, r1d = st.columns([4, 2, 1, 1])
            with r1a:
                st.markdown(f"**{f['filename']}**")
                meta = (
                    f"{f['source_type'].capitalize()} · "
                    f"{format_duration(f['duration_sec'])} · "
                    f"Created {(f['file_created_at'] or '')[:10]}"
                )
                st.caption(meta)
            with r1b:
                if is_processing:
                    st.markdown(f"⏳ {f['status']}…")
                else:
                    st.markdown(status_str)
                if f.get("error_message"):
                    st.caption(f"⚠️ {f['error_message']}")
            with r1c:
                if f["status"] in ("labeled", "done") and f.get("transcript_txt_path"):
                    if st.button("🏷️ Label", key=f"label_{fid}"):
                        st.session_state["label_file_id"] = fid
                        st.rerun()
            with r1d:
                if st.button("🗑️", key=f"del_{fid}", help="Remove from list"):
                    db.delete_file(fid)
                    st.rerun()

            r2a, r2b, r2c = st.columns([2, 2, 1])
            with r2a:
                new_job_id = st.text_input(
                    "Tag / ID",
                    value=f.get("job_id") or "",
                    key=f"job_{fid}",
                    placeholder="e.g. project-x (used for copy to destination)",
                )
            with r2b:
                new_desc = st.text_input(
                    "Description",
                    value=f.get("description") or "",
                    key=f"desc_{fid}",
                    placeholder="e.g. Final round with McKay",
                )
            with r2c:
                new_offset = st.number_input(
                    "Skip start (s)",
                    min_value=0,
                    value=int(f.get("offset_sec") or 0),
                    step=5,
                    key=f"offset_{fid}",
                    help="Seconds of dead air to skip at the start.",
                )

            changed = (
                new_job_id != (f.get("job_id") or "")
                or new_desc != (f.get("description") or "")
                or float(new_offset) != float(f.get("offset_sec") or 0)
            )
            if changed:
                db.update_file(
                    fid,
                    job_id=new_job_id or None,
                    description=new_desc or None,
                    offset_sec=float(new_offset),
                )

            if f.get("audio_path") or f.get("transcript_txt_path"):
                with st.expander("Derived files"):
                    if f.get("audio_path"):
                        st.caption(f"Audio: `{f['audio_path']}`")
                    if f.get("transcript_txt_path"):
                        col_path, col_copy = st.columns([4, 1])
                        with col_path:
                            st.caption(f"Transcript: `{f['transcript_txt_path']}`")
                        with col_copy:
                            if st.button("📋 Copy", key=f"copy_{fid}"):
                                try:
                                    with open(f["transcript_txt_path"], encoding="utf-8") as fp:
                                        content = fp.read()
                                    st.session_state[f"clipboard_{fid}"] = content
                                except Exception as e:
                                    st.error(str(e))
                    clip_key = f"clipboard_{fid}"
                    if clip_key in st.session_state:
                        st.text_area(
                            "Paste this into Claude.ai →",
                            value=st.session_state[clip_key],
                            height=200,
                            key=f"cliparea_{fid}",
                        )
