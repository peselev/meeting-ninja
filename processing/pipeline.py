"""
pipeline.py — orchestrates the full processing pipeline for a single file.
Runs in a background thread; communicates progress via a queue.
"""
from __future__ import annotations
import threading
import queue
import traceback
from pathlib import Path

from db import client as db
from processing.extract_audio import extract_audio
from processing.transcribe import transcribe, load_segments_from_json
from processing.diarize import (
    is_pyannote_available, diarize,
    merge_diarization_with_segments, collect_speaker_samples,
)
from processing.job_router import route_to_job_folder
from utils.media_info import get_source_type


# Each message: {"file_id": int, "stage": str, "done": bool, "error": str|None}
_progress_queue: queue.Queue = queue.Queue()


def get_progress_queue() -> queue.Queue:
    return _progress_queue


def _emit(file_id: int, stage: str, done: bool = False, error: str = None):
    _progress_queue.put({"file_id": file_id, "stage": stage, "done": done, "error": error})


def _process_file(file_id: int, model_name: str, home_folder: str,
                  hf_token: str | None, jobs_root: str | None,
                  language: str | None = None):
    record = db.get_file(file_id)
    if not record:
        return

    source_path = record["source_path"]
    stem = Path(source_path).stem
    offset_sec = float(record.get("offset_sec") or 0.0)

    try:
        # ── Step 1: Audio extraction ────────────────────────────────────────
        source_type = record["source_type"]
        # Extract if it's video, OR if an offset is set (so audio gets trimmed too)
        if source_type == "video" or (offset_sec and offset_sec > 0):
            _emit(file_id, "extracting")
            db.update_file(file_id, status="extracting")
            audio_path = extract_audio(source_path, home_folder, offset_sec)
            db.update_file(file_id, audio_path=audio_path)
        else:
            audio_path = source_path  # already audio, no offset, use directly

        # ── Step 2: Transcription ────────────────────────────────────────────
        _emit(file_id, "transcribing")
        db.update_file(file_id, status="transcribing")
        txt_path, json_path = transcribe(audio_path, home_folder, model_name, language)
        db.update_file(
            file_id,
            transcript_txt_path=txt_path,
            transcript_json_path=json_path,
            status="transcribed",
        )

        # ── Step 3 + 4: Diarization + merge (optional) ──────────────────────
        whisper_segments = load_segments_from_json(json_path)

        # Whisper timestamps are relative to the (trimmed) audio. Add the offset
        # back so they align with the ORIGINAL video for the labeling preview.
        if offset_sec and offset_sec > 0:
            for seg in whisper_segments:
                seg["start_sec"] += offset_sec
                seg["end_sec"]   += offset_sec

        if hf_token and is_pyannote_available():
            _emit(file_id, "diarizing")
            db.update_file(file_id, status="diarizing")
            turns = diarize(audio_path, hf_token)
            if offset_sec and offset_sec > 0:
                for t in turns:
                    t["start_sec"] += offset_sec
                    t["end_sec"]   += offset_sec
            merged = merge_diarization_with_segments(whisper_segments, turns)
        else:
            # No diarization: assign all segments to a single placeholder speaker
            merged = [
                {**seg, "speaker_label": "SPEAKER_00"}
                for seg in whisper_segments
            ]

        # Collect representative samples per speaker
        samples = collect_speaker_samples(merged, n_samples=3)

        # Persist speakers
        import json
        speaker_id_map = {}
        for label, segs in samples.items():
            sid = db.upsert_speaker(
                file_id=file_id,
                diarization_label=label,
                display_name=None,
                sample_segments_json=json.dumps(segs),
            )
            speaker_id_map[label] = sid

        # Persist segments
        db.delete_segments(file_id)
        db.insert_segments([
            {
                "file_id":    file_id,
                "start_sec":  seg["start_sec"],
                "end_sec":    seg["end_sec"],
                "speaker_id": speaker_id_map.get(seg.get("speaker_label", "SPEAKER_00")),
                "text":       seg["text"],
            }
            for seg in merged
        ])

        db.update_file(file_id, status="labeled")

        # ── Step 6: Job folder routing ───────────────────────────────────────
        record = db.get_file(file_id)  # re-fetch to get job_id
        job_id = record.get("job_id") or ""
        if job_id.strip() and jobs_root and jobs_root.strip():
            route_to_job_folder(txt_path, stem, job_id, jobs_root)

        db.update_file(file_id, status="done", error_message=None)
        _emit(file_id, "done", done=True)

    except Exception as e:
        err = traceback.format_exc()
        db.update_file(file_id, status="error", error_message=str(e))
        _emit(file_id, "error", done=True, error=str(e))


def run_pipeline(file_ids: list[int], model_name: str, home_folder: str,
                 hf_token: str | None, jobs_root: str | None,
                 language: str | None = None):
    """Launch one thread per file. Per-file offset is read from the DB record."""
    for fid in file_ids:
        db.update_file(fid, status="pending", error_message=None)
        t = threading.Thread(
            target=_process_file,
            args=(fid, model_name, home_folder, hf_token, jobs_root, language),
            daemon=True,
        )
        t.start()
