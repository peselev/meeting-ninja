#!/usr/bin/env python3
"""
cli.py — command-line driver for the meeting transcription pipeline.

Runs the full pipeline (extract → transcribe → diarize → label-prep → job-copy)
on a single file, synchronously, with verbose logging. Designed for testing and
for Claude Code to call programmatically.

Examples:
    python cli.py --file "2026-06-05 11-58-45.mov" --offset 480 \\
        --description "call with Nav about his first full-time role"

    python cli.py --file recording.mov --model medium --language en --no-diarize

The --file argument can be:
    - an absolute path
    - a path relative to the home folder (from Settings)
    - just a filename (searched recursively under the home folder)
"""
from __future__ import annotations
import utils.quiet  # noqa: F401  — MUST be imported before torch/pyannote/whisper
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from db import client as db
from db.client import init_db
from utils.media_info import get_source_type, get_media_info, check_ffmpeg
from processing.extract_audio import extract_audio
from processing.transcribe import transcribe, load_segments_from_json, WHISPER_MODELS
from processing.diarize import (
    is_pyannote_available, diarize,
    merge_diarization_with_segments, collect_speaker_samples,
)
from processing.job_router import route_to_job_folder


log = logging.getLogger("cli")

# Our own module loggers — these get DEBUG when --verbose, without un-muting deps
_OUR_LOGGERS = ("cli", "diarize", "pipeline")


def _setup_logging(verbose: bool):
    # Root stays at INFO so dependency noise we didn't explicitly mute is calm.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    if verbose:
        for name in _OUR_LOGGERS:
            logging.getLogger(name).setLevel(logging.DEBUG)


def _resolve_file(file_arg: str, home_folder: str) -> str | None:
    """Resolve --file into an absolute path."""
    p = Path(file_arg)
    if p.is_absolute() and p.exists():
        return str(p.resolve())

    home = Path(home_folder)
    # relative to home
    candidate = home / file_arg
    if candidate.exists():
        return str(candidate.resolve())

    # search recursively by filename
    matches = list(home.rglob(p.name))
    matches = [m for m in matches if m.is_file()]
    if len(matches) == 1:
        return str(matches[0].resolve())
    if len(matches) > 1:
        log.error("Multiple files match '%s':", p.name)
        for m in matches:
            log.error("  %s", m)
        log.error("Pass a more specific path.")
        return None
    return None


def _find_or_create_file_record(abs_path: str, offset_sec: float,
                                description: str | None, job_id: str | None) -> int:
    """Find existing DB record by path, or create one. Update metadata."""
    existing = next(
        (f for f in db.get_all_files()
         if str(Path(f["source_path"]).resolve()) == abs_path),
        None,
    )
    if existing:
        fid = existing["id"]
        db.update_file(
            fid,
            offset_sec=offset_sec,
            description=description if description is not None else existing.get("description"),
            job_id=job_id if job_id is not None else existing.get("job_id"),
            status="pending",
            error_message=None,
        )
        log.info("Reusing existing DB record id=%s", fid)
        return fid

    info = get_media_info(abs_path)
    fid = db.add_file({
        "source_path":     abs_path,
        "filename":        Path(abs_path).name,
        "source_type":     get_source_type(abs_path),
        "duration_sec":    info["duration_sec"],
        "file_created_at": info["file_created_at"],
        "added_at":        datetime.now().isoformat(),
        "status":          "pending",
        "offset_sec":      offset_sec,
        "description":     description,
        "job_id":          job_id,
    })
    log.info("Created new DB record id=%s", fid)
    return fid


def run(args) -> int:
    init_db()

    # ── Preconditions ────────────────────────────────────────────────────────
    ffmpeg_ok, ffprobe_ok = check_ffmpeg()
    if not ffmpeg_ok or not ffprobe_ok:
        log.error("ffmpeg/ffprobe not found on PATH. Install with `brew install ffmpeg`.")
        return 2

    home_folder = args.home or db.get_setting("home_folder", "")
    if not home_folder:
        log.error("No home folder set. Pass --home or configure it in the app Settings.")
        return 2
    log.info("Home folder: %s", home_folder)

    abs_path = _resolve_file(args.file, home_folder)
    if not abs_path:
        log.error("Could not resolve file: %s", args.file)
        return 2
    log.info("Resolved file: %s", abs_path)

    offset_sec = float(args.offset)
    log.info("Offset: %.1f s   Model: %s   Language: %s   Diarize: %s",
             offset_sec, args.model, args.language, not args.no_diarize)

    fid = _find_or_create_file_record(abs_path, offset_sec, args.description, args.job_id)
    stem = Path(abs_path).stem

    try:
        # ── Step 1: Audio extraction ──────────────────────────────────────────
        source_type = get_source_type(abs_path)
        if source_type == "video" or offset_sec > 0:
            log.info("[1/5] Extracting audio (offset=%.1fs)…", offset_sec)
            db.update_file(fid, status="extracting")
            audio_path = extract_audio(abs_path, home_folder, offset_sec)
            db.update_file(fid, audio_path=audio_path)
            log.info("      → %s", audio_path)
        else:
            audio_path = abs_path
            log.info("[1/5] Audio source, no offset — using original file.")

        # ── Step 2: Transcription ─────────────────────────────────────────────
        log.info("[2/5] Transcribing with Whisper '%s'…", args.model)
        db.update_file(fid, status="transcribing")
        lang = None if args.language == "auto" else args.language
        txt_path, json_path = transcribe(audio_path, home_folder, args.model, lang)
        db.update_file(fid, transcript_txt_path=txt_path,
                       transcript_json_path=json_path, status="transcribed")
        log.info("      → %s", txt_path)
        log.info("      → %s", json_path)

        whisper_segments = load_segments_from_json(json_path)
        log.info("      %d transcript segments", len(whisper_segments))
        if offset_sec > 0:
            for seg in whisper_segments:
                seg["start_sec"] += offset_sec
                seg["end_sec"]   += offset_sec

        # ── Step 3+4: Diarization + merge ─────────────────────────────────────
        hf_token = args.hf_token or db.get_setting("hf_token") or None
        do_diarize = (not args.no_diarize) and hf_token and is_pyannote_available()

        if do_diarize:
            log.info("[3/5] Diarizing with pyannote…")
            db.update_file(fid, status="diarizing")
            turns = diarize(audio_path, hf_token)
            if offset_sec > 0:
                for t in turns:
                    t["start_sec"] += offset_sec
                    t["end_sec"]   += offset_sec
            log.info("      %d speaker turns, %d distinct speakers",
                     len(turns), len({t['label'] for t in turns}))
            merged = merge_diarization_with_segments(whisper_segments, turns)
            log.info("[4/5] Merged diarization with transcript.")
        else:
            reason = ("--no-diarize" if args.no_diarize else
                      "no HF token" if not hf_token else
                      "pyannote not installed")
            log.info("[3/5] Skipping diarization (%s). Single speaker.", reason)
            merged = [{**seg, "speaker_label": "SPEAKER_00"} for seg in whisper_segments]

        # Persist speakers + segments
        import json as _json
        samples = collect_speaker_samples(merged, n_samples=3)
        speaker_id_map = {}
        for label, segs in samples.items():
            sid = db.upsert_speaker(fid, label, None, _json.dumps(segs))
            speaker_id_map[label] = sid
        db.delete_segments(fid)
        db.insert_segments([
            {
                "file_id":    fid,
                "start_sec":  seg["start_sec"],
                "end_sec":    seg["end_sec"],
                "speaker_id": speaker_id_map.get(seg.get("speaker_label", "SPEAKER_00")),
                "text":       seg["text"],
            }
            for seg in merged
        ])
        db.update_file(fid, status="labeled")
        log.info("      Speakers: %s", ", ".join(speaker_id_map.keys()))

        # ── Step 5: Destination folder routing ────────────────────────────────
        rec = db.get_file(fid)
        job_id = rec.get("job_id") or ""
        jobs_root = (getattr(args, "jobs_root", None) or getattr(args, "dest", None)
                     or db.get_setting("destination_folder")
                     or db.get_setting("jobs_root_folder") or "")
        if job_id.strip() and jobs_root.strip():
            ok, msg = route_to_job_folder(txt_path, stem, job_id, jobs_root)
            log.info("[5/5] Destination copy: %s", msg if ok else f"skipped — {msg}")
        else:
            log.info("[5/5] No destination copy (tag='%s', destination set=%s).",
                     job_id, bool(jobs_root.strip()))

        db.update_file(fid, status="done", error_message=None)
        log.info("DONE. Transcript at: %s", txt_path)

        # Print a short preview
        with open(txt_path, encoding="utf-8") as f:
            preview = f.read()[:500]
        print("\n----- transcript preview -----")
        print(preview)
        print("------------------------------\n")
        return 0

    except Exception as e:
        # Print the exception chain WITHOUT touching linecache — some deps
        # (speechbrain lazy imports) raise ImportError when their source is
        # read during traceback formatting, which crashes the formatter itself.
        # traceback.format_exception_only avoids source-line lookups.
        import traceback
        log.error("Pipeline failed: %s", e)
        for line in traceback.format_exception_only(type(e), e):
            print(line, end="")
        db.update_file(fid, status="error", error_message=str(e))
        return 1


def _list_speakers(file_id: int):
    """Print current speakers + a sample excerpt for each."""
    import json
    speakers = db.get_speakers(file_id)
    if not speakers:
        print("No speakers found — has the file been processed?")
        return speakers
    print("\nSpeakers in this file:")
    for sp in speakers:
        name = sp.get("display_name") or "(unnamed)"
        sample = ""
        if sp.get("sample_segments_json"):
            try:
                segs = json.loads(sp["sample_segments_json"])
                if segs:
                    sample = segs[0]["text"][:80]
            except Exception:
                pass
        print(f"  {sp['diarization_label']:<14} → {name}")
        if sample:
            print(f"        e.g. “{sample}…”")
    print()
    return speakers


def run_label(args) -> int:
    """Label speakers for an already-processed file."""
    from processing.labeler import write_labeled_transcript
    from processing.job_router import route_to_job_folder
    init_db()

    home_folder = args.home or db.get_setting("home_folder", "")
    abs_path = _resolve_file(args.file, home_folder)
    if not abs_path:
        log.error("Could not resolve file: %s", args.file)
        return 2

    record = next(
        (f for f in db.get_all_files()
         if str(Path(f["source_path"]).resolve()) == abs_path),
        None,
    )
    if not record:
        log.error("No DB record for this file. Process it first.")
        return 2
    file_id = record["id"]

    speakers = _list_speakers(file_id)
    if not speakers:
        return 2

    label_to_id = {sp["diarization_label"]: sp["id"] for sp in speakers}

    # ── Mode 1: names supplied via --speaker LABEL=Name (repeatable) ──────────
    assignments: dict[str, str] = {}
    for item in (args.speaker or []):
        if "=" not in item:
            log.error("Bad --speaker '%s' (expected LABEL=Name).", item)
            return 2
        lbl, name = item.split("=", 1)
        assignments[lbl.strip()] = name.strip()

    # ── Mode 2: interactive prompts for any not supplied via flags ───────────
    if args.interactive:
        import json
        for sp in speakers:
            lbl = sp["diarization_label"]
            if lbl in assignments:
                continue
            samples = []
            if sp.get("sample_segments_json"):
                try:
                    samples = json.loads(sp["sample_segments_json"])
                except Exception:
                    pass
            print(f"\n── {lbl} ─────────────────────────────")
            for seg in samples[:3]:
                print(f"   [{seg['start_sec']:.0f}s] {seg['text'][:100]}")
            current = sp.get("display_name") or ""
            prompt = f"   Name for {lbl}" + (f" [{current}]" if current else "") + ": "
            try:
                entered = input(prompt).strip()
            except EOFError:
                entered = ""
            if entered:
                assignments[lbl] = entered

    if not assignments:
        log.info("No names provided. Use --speaker LABEL=Name or --interactive.")
        return 0

    # Apply
    for lbl, name in assignments.items():
        sid = label_to_id.get(lbl)
        if sid is None:
            log.warning("Unknown speaker label '%s' — skipping.", lbl)
            continue
        db.update_speaker_name(sid, name or None)
        log.info("Set %s → %s", lbl, name)

    db.update_file(file_id, status="done")
    txt_path = write_labeled_transcript(file_id)
    log.info("Rewrote labeled transcript: %s", txt_path)

    # Copy to destination if tag set
    rec = db.get_file(file_id)
    tag = rec.get("job_id") or ""
    dest = (args.dest or db.get_setting("destination_folder")
            or db.get_setting("jobs_root_folder") or "")
    if tag.strip() and dest.strip() and txt_path:
        stem = Path(rec["source_path"]).stem
        ok, msg = route_to_job_folder(txt_path, stem, tag, dest)
        log.info("Destination copy: %s", msg if ok else f"skipped — {msg}")

    return 0


def main():
    p = argparse.ArgumentParser(description="Meeting transcription pipeline (CLI).")
    sub = p.add_subparsers(dest="command")

    # ── process (default) ────────────────────────────────────────────────────
    pp = sub.add_parser("process", help="Transcribe + diarize a file (default).")
    for parser in (p, pp):  # accept these on both root and 'process' for back-compat
        parser.add_argument("--file", help="Path or filename (searched under home folder).")
        parser.add_argument("--offset", type=float, default=0, help="Seconds to skip at start.")
        parser.add_argument("--model", default="base", choices=WHISPER_MODELS, help="Whisper model.")
        parser.add_argument("--language", default="en", help="ISO code (en, ru, he…) or 'auto'.")
        parser.add_argument("--description", default=None, help="Free-text description.")
        parser.add_argument("--tag", "--job-id", dest="job_id", default=None,
                            help="Tag/ID for transcript routing to destination folder.")
        parser.add_argument("--no-diarize", action="store_true", help="Skip speaker diarization.")
        parser.add_argument("--home", default=None, help="Override home folder.")
        parser.add_argument("--dest", "--jobs-root", dest="dest", default=None,
                            help="Override destination folder.")
        parser.add_argument("--hf-token", default=None, help="Override HuggingFace token.")
        parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")

    # ── label ──────────────────────────────────────────────────────────────────
    pl = sub.add_parser("label", help="Name speakers for an already-processed file.")
    pl.add_argument("--file", required=True, help="Path or filename.")
    pl.add_argument("--speaker", action="append", metavar="LABEL=Name",
                    help="Assign a name, e.g. --speaker SPEAKER_00=Konstantin (repeatable).")
    pl.add_argument("--interactive", action="store_true",
                    help="Prompt for each unnamed speaker, showing excerpts.")
    pl.add_argument("--list", action="store_true", help="Just list speakers and exit.")
    pl.add_argument("--home", default=None, help="Override home folder.")
    pl.add_argument("--dest", default=None, help="Override destination folder.")
    pl.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")

    args = p.parse_args()
    _setup_logging(getattr(args, "verbose", False))

    if args.command == "label":
        if getattr(args, "list", False):
            init_db()
            home_folder = args.home or db.get_setting("home_folder", "")
            abs_path = _resolve_file(args.file, home_folder)
            if not abs_path:
                log.error("Could not resolve file: %s", args.file)
                sys.exit(2)
            rec = next((f for f in db.get_all_files()
                        if str(Path(f["source_path"]).resolve()) == abs_path), None)
            if not rec:
                log.error("No DB record. Process the file first.")
                sys.exit(2)
            _list_speakers(rec["id"])
            sys.exit(0)
        sys.exit(run_label(args))

    # default → process
    if not getattr(args, "file", None):
        p.error("--file is required for processing.")
    # normalize back-compat attr name used in run()
    if not hasattr(args, "jobs_root"):
        args.jobs_root = args.dest
    sys.exit(run(args))


if __name__ == "__main__":
    main()
