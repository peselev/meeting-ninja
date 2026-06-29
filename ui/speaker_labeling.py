from __future__ import annotations
import json
import streamlit as st
from pathlib import Path

from meeting_ninja.db import client as db
from meeting_ninja.processing.job_router import route_to_job_folder
from meeting_ninja.processing.labeler import write_labeled_transcript
from meeting_ninja.utils.media_info import format_duration

VIDEO_EXTS = {".mov", ".mp4", ".mkv", ".avi", ".webm", ".m4v"}


def _ts(sec: float) -> str:
    return format_duration(sec)


def render():
    file_id = st.session_state.get("label_file_id")
    record = db.get_file(file_id) if file_id else None
    if not record:
        st.session_state.pop("label_file_id", None)
        st.info("No file selected. Click 🏷️ Label on a processed file.")
        return

    # Header with a back button
    col_back, col_title = st.columns([1, 6])
    with col_back:
        if st.button("← Back"):
            st.session_state.pop("label_file_id", None)
            st.rerun()
    with col_title:
        st.subheader(f"Label speakers — {record['filename']}")

    speakers = db.get_speakers(file_id)
    if not speakers:
        st.warning("No speakers found. The file may not have been processed yet.")
        return

    # Show current saved state so the user can confirm a prior save worked.
    saved = {sp["diarization_label"]: sp.get("display_name") for sp in speakers}
    if any(saved.values()):
        st.caption("Currently saved: " + ", ".join(
            f"{k} → **{v}**" if v else f"{k} → _(unnamed)_" for k, v in saved.items()
        ))

    st.caption("Review each speaker's excerpts, type a name (blank = leave unnamed), "
               "then click Save. Use the preview to check the video if unsure.")

    source_path = record["source_path"]
    is_video = Path(source_path).suffix.lower() in VIDEO_EXTS

    # Video previews live OUTSIDE the form (forms can't contain buttons that
    # trigger reruns mid-edit). Render any requested preview first.
    for sp in speakers:
        sid = sp["id"]
        pkey = f"preview_open_{sid}"
        if is_video and st.session_state.get(pkey) is not None:
            st.video(source_path, start_time=int(st.session_state[pkey]))

    # ── The form: typing here does NOT trigger reruns until Save is clicked ──
    with st.form(key=f"label_form_{file_id}"):
        name_inputs: dict[int, str] = {}
        for sp in speakers:
            sid = sp["id"]
            label = sp["diarization_label"]
            st.markdown(f"**{label}**")

            samples = []
            if sp.get("sample_segments_json"):
                try:
                    samples = json.loads(sp["sample_segments_json"])
                except Exception:
                    pass
            for seg in samples:
                st.markdown(f"> [{_ts(seg['start_sec'])}] {seg['text']}")

            name_inputs[sid] = st.text_input(
                f"Name for {label}",
                value=sp.get("display_name") or "",
                key=f"form_name_{sid}",
                placeholder="e.g. Konstantin, Interviewer …",
            )
            st.divider()

        submitted = st.form_submit_button("💾 Save labels", type="primary")

    # Preview buttons (outside the form)
    if is_video:
        st.caption("Jump to a moment in the video:")
        cols = st.columns(len(speakers))
        for col, sp in zip(cols, speakers):
            sid = sp["id"]
            samples = []
            if sp.get("sample_segments_json"):
                try:
                    samples = json.loads(sp["sample_segments_json"])
                except Exception:
                    pass
            if samples:
                with col:
                    if st.button(f"▶ {sp['diarization_label']} @ {_ts(samples[0]['start_sec'])}",
                                 key=f"prevbtn_{sid}"):
                        st.session_state[f"preview_open_{sid}"] = samples[0]["start_sec"]
                        st.rerun()

    # ── Handle save ──────────────────────────────────────────────────────────
    if submitted:
        for sid, name in name_inputs.items():
            db.update_speaker_name(sid, name.strip() or None)

        db.update_file(file_id, status="done")
        write_labeled_transcript(file_id)

        # Copy to destination folder if a tag/id is set
        rec = db.get_file(file_id)
        tag = rec.get("job_id") or ""
        dest = (db.get_setting("destination_folder")
                or db.get_setting("jobs_root_folder") or "")
        txt_path = rec.get("transcript_txt_path") or ""
        stem = Path(rec["source_path"]).stem

        msgs = ["✅ Speaker labels saved."]
        if tag.strip() and dest.strip() and txt_path:
            ok, msg = route_to_job_folder(txt_path, stem, tag, dest)
            msgs.append(f"📁 Copied to `{msg}`" if ok else f"⚠️ {msg}")
        elif tag.strip() and not dest.strip():
            msgs.append("ℹ️ Tag is set but no Destination folder configured in Settings.")

        st.session_state["label_saved_msg"] = "\n\n".join(msgs)
        st.rerun()

    # Show post-save confirmation (survives the rerun)
    if st.session_state.get("label_saved_msg"):
        st.success(st.session_state.pop("label_saved_msg"))
