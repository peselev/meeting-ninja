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
import meeting_ninja.utils.quiet  # noqa: F401  — MUST be imported before torch/pyannote/whisper
import argparse
import json as _json_mod
import logging
import sys
from datetime import datetime
from pathlib import Path

from meeting_ninja.db import client as db
from meeting_ninja.db.client import init_db
from meeting_ninja.utils.media_info import get_source_type, get_media_info, check_ffmpeg, SOUNDFILE_NATIVE
from meeting_ninja.processing.extract_audio import extract_audio
from meeting_ninja.processing.transcribe import transcribe, load_segments_from_json, WHISPER_MODELS
from meeting_ninja.processing.diarize import (
    is_pyannote_available, diarize,
    merge_diarization_with_segments, collect_speaker_samples,
)
from meeting_ninja.processing.job_router import route_to_job_folder
from meeting_ninja.processing.file_naming import derive_output_stem


log = logging.getLogger("cli")

# Our own module loggers — these get DEBUG when --verbose, without un-muting deps
_OUR_LOGGERS = ("cli", "diarize", "pipeline")


def _setup_logging(verbose: bool):
    # Root stays at INFO so dependency noise we didn't explicitly mute is calm.
    # Logs ALWAYS go to stderr so --json keeps stdout as a single clean object.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    if verbose:
        for name in _OUR_LOGGERS:
            logging.getLogger(name).setLevel(logging.DEBUG)


def _emit_json(payload: dict):
    """Print exactly one JSON object to stdout (the machine-readable contract)."""
    sys.stdout.write(_json_mod.dumps(payload, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _speakers_payload(file_id: int) -> list[dict]:
    """Structured speaker list for JSON output: label, display_name, first sample."""
    out = []
    for sp in db.get_speakers(file_id):
        sample = None
        if sp.get("sample_segments_json"):
            try:
                segs = _json_mod.loads(sp["sample_segments_json"])
                if segs:
                    sample = segs[0]["text"].strip()
            except Exception:
                pass
        out.append({
            "label":        sp["diarization_label"],
            "display_name": sp.get("display_name"),
            "sample":       sample,
        })
    return out


def _resolve_file(file_arg: str, search_root: str) -> str | None:
    """Resolve --file into an absolute path.

    `search_root` is only used to locate the file when `file_arg` is relative
    or a bare filename. It does not determine where outputs are written.
    """
    p = Path(file_arg)
    if p.is_absolute() and p.exists():
        return str(p.resolve())

    root = Path(search_root) if search_root else Path.cwd()
    # relative to the search root
    candidate = root / file_arg
    if candidate.exists():
        return str(candidate.resolve())

    # search recursively by filename
    matches = list(root.rglob(p.name)) if root.exists() else []
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

    # Find the file first. The search root only helps locate a relative/bare
    # filename; it does not decide where outputs go.
    search_root = args.home or db.get_setting("home_folder", "") or str(Path.cwd())
    abs_path = _resolve_file(args.file, search_root)
    if not abs_path:
        log.error("Could not resolve file: %s", args.file)
        if getattr(args, "json", False):
            _emit_json({"ok": False, "command": "process", "file_id": None,
                        "source_path": args.file, "status": "error",
                        "audio_path": None, "transcript_txt_path": None,
                        "transcript_json_path": None, "diarized": False,
                        "speakers": [], "destination_copy": None,
                        "error": f"Could not resolve file: {args.file}"})
        return 2
    log.info("Resolved file: %s", abs_path)

    # Output root: an explicit --home wins; otherwise outputs (audio/,
    # transcripts/) land next to the source recording.
    home_folder = args.home or str(Path(abs_path).parent)
    log.info("Output root: %s", home_folder)

    offset_sec = float(args.offset)
    log.info("Offset: %.1f s   Model: %s   Language: %s   Diarize: %s",
             offset_sec, args.model, args.language, not args.no_diarize)

    fid = _find_or_create_file_record(abs_path, offset_sec, args.description, args.job_id)

    # When a description is set, name the derived files (audio/, transcripts/)
    # after it instead of the source filename. The source itself is never
    # renamed. derive_output_stem keeps the name unique across other files'
    # outputs while staying stable when reprocessing this same file.
    out_stem = (derive_output_stem(args.description, fid, db.get_all_files())
                if args.description else None)
    stem = out_stem or Path(abs_path).stem
    if out_stem:
        log.info("Naming outputs after description: %s", out_stem)

    try:
        # ── Step 1: Audio extraction ──────────────────────────────────────────
        # Transcode to 16kHz mono WAV for video, for an offset trim, OR when the
        # source isn't directly readable by soundfile (diarization's loader).
        # That last case covers .opus/.m4a/.mp3/.aac: Whisper can decode them via
        # ffmpeg, but pyannote's soundfile path can't, so they need a WAV first.
        source_type = get_source_type(abs_path)
        suffix = Path(abs_path).suffix.lower()
        needs_extract = (
            source_type == "video"
            or offset_sec > 0
            or suffix not in SOUNDFILE_NATIVE
        )
        if needs_extract:
            log.info("[1/5] Extracting audio (offset=%.1fs)…", offset_sec)
            db.update_file(fid, status="extracting")
            audio_path = extract_audio(abs_path, home_folder, offset_sec, out_stem=out_stem)
            db.update_file(fid, audio_path=audio_path)
            log.info("      → %s", audio_path)
        else:
            audio_path = abs_path
            log.info("[1/5] Soundfile-native source, no offset — using original file.")

        # ── Step 2: Transcription ─────────────────────────────────────────────
        log.info("[2/5] Transcribing with Whisper '%s'…", args.model)
        db.update_file(fid, status="transcribing")
        lang = None if args.language == "auto" else args.language
        txt_path, json_path = transcribe(audio_path, home_folder, args.model, lang, out_stem=out_stem)
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
        destination_copy = None
        if job_id.strip() and jobs_root.strip():
            ok, msg = route_to_job_folder(txt_path, stem, job_id, jobs_root)
            if ok:
                destination_copy = msg
            log.info("[5/5] Destination copy: %s", msg if ok else f"skipped — {msg}")
        else:
            log.info("[5/5] No destination copy (tag='%s', destination set=%s).",
                     job_id, bool(jobs_root.strip()))

        db.update_file(fid, status="done", error_message=None)
        log.info("DONE. Transcript at: %s", txt_path)

        final = db.get_file(fid)
        if getattr(args, "json", False):
            _emit_json({
                "ok":                   True,
                "command":              "process",
                "file_id":              fid,
                "source_path":          abs_path,
                "filename":             Path(abs_path).name,
                "status":               final.get("status"),
                "audio_path":           final.get("audio_path"),
                "transcript_txt_path":  txt_path,
                "transcript_json_path": final.get("transcript_json_path"),
                "diarized":             bool(do_diarize),
                "speakers":             _speakers_payload(fid),
                "destination_copy":     destination_copy,
                "error":                None,
            })
        else:
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
        db.update_file(fid, status="error", error_message=str(e))
        if getattr(args, "json", False):
            _emit_json({
                "ok":                   False,
                "command":              "process",
                "file_id":              fid,
                "source_path":          abs_path,
                "status":               "error",
                "audio_path":           None,
                "transcript_txt_path":  None,
                "transcript_json_path": None,
                "diarized":             False,
                "speakers":             [],
                "destination_copy":     None,
                "error":                str(e),
            })
        else:
            for line in traceback.format_exception_only(type(e), e):
                print(line, end="", file=sys.stderr)
        return 1


def _list_speakers(file_id: int, quiet: bool = False):
    """Print current speakers + a sample excerpt for each (unless quiet)."""
    import json
    speakers = db.get_speakers(file_id)
    if not speakers:
        if not quiet:
            print("No speakers found — has the file been processed?", file=sys.stderr)
        return speakers
    if quiet:
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
    from meeting_ninja.processing.labeler import write_labeled_transcript
    from meeting_ninja.processing.job_router import route_to_job_folder
    init_db()

    search_root = args.home or db.get_setting("home_folder", "") or str(Path.cwd())
    abs_path = _resolve_file(args.file, search_root)
    _as_json = getattr(args, "json", False)
    if not abs_path:
        log.error("Could not resolve file: %s", args.file)
        if _as_json:
            _emit_json({"ok": False, "command": "label", "file_id": None,
                        "error": f"Could not resolve file: {args.file}",
                        "speakers": [], "transcript_txt_path": None,
                        "destination_copy": None})
        return 2

    record = next(
        (f for f in db.get_all_files()
         if str(Path(f["source_path"]).resolve()) == abs_path),
        None,
    )
    if not record:
        log.error("No DB record for this file. Process it first.")
        if _as_json:
            _emit_json({"ok": False, "command": "label", "file_id": None,
                        "error": "No DB record for this file. Process it first.",
                        "speakers": [], "transcript_txt_path": None,
                        "destination_copy": None})
        return 2
    file_id = record["id"]

    speakers = _list_speakers(file_id, quiet=_as_json)
    if not speakers:
        if _as_json:
            _emit_json({"ok": False, "command": "label", "file_id": file_id,
                        "error": "No speakers found — has the file been processed?",
                        "speakers": [], "transcript_txt_path": None,
                        "destination_copy": None})
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
        if _as_json:
            _emit_json({"ok": True, "command": "label", "file_id": file_id,
                        "error": None, "speakers": _speakers_payload(file_id),
                        "transcript_txt_path": record.get("transcript_txt_path"),
                        "destination_copy": None})
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
    destination_copy = None
    if tag.strip() and dest.strip() and txt_path:
        stem = Path(rec["source_path"]).stem
        ok, msg = route_to_job_folder(txt_path, stem, tag, dest)
        if ok:
            destination_copy = msg
        log.info("Destination copy: %s", msg if ok else f"skipped — {msg}")

    if _as_json:
        _emit_json({"ok": True, "command": "label", "file_id": file_id,
                    "error": None, "speakers": _speakers_payload(file_id),
                    "transcript_txt_path": txt_path,
                    "destination_copy": destination_copy})
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
        parser.add_argument("--description", default=None, help="Free-text description. Names the derived audio/transcript files.")
        parser.add_argument("--tag", "--job-id", dest="job_id", default=None,
                            help="Tag/ID for transcript routing to destination folder.")
        parser.add_argument("--no-diarize", action="store_true", help="Skip speaker diarization.")
        parser.add_argument("--home", default=None,
                            help="Optional. Output root for audio/ and transcripts/. "
                                 "Defaults to the source file's folder.")
        parser.add_argument("--dest", "--jobs-root", dest="dest", default=None,
                            help="Override destination folder.")
        parser.add_argument("--hf-token", default=None, help="Override HuggingFace token.")
        parser.add_argument("--json", action="store_true",
                            help="Emit a single JSON result object to stdout (logs go to stderr).")
        parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")

    # ── label ──────────────────────────────────────────────────────────────────
    pl = sub.add_parser("label", help="Name speakers for an already-processed file.")
    pl.add_argument("--file", required=True, help="Path or filename.")
    pl.add_argument("--speaker", action="append", metavar="LABEL=Name",
                    help="Assign a name, e.g. --speaker SPEAKER_00=Konstantin (repeatable).")
    pl.add_argument("--interactive", action="store_true",
                    help="Prompt for each unnamed speaker, showing excerpts.")
    pl.add_argument("--list", action="store_true", help="Just list speakers and exit.")
    pl.add_argument("--home", default=None,
                    help="Optional. Folder to search when --file is a bare name.")
    pl.add_argument("--dest", default=None, help="Override destination folder.")
    pl.add_argument("--json", action="store_true",
                    help="Emit a single JSON result object to stdout (logs go to stderr).")
    pl.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")

    args = p.parse_args()
    _setup_logging(getattr(args, "verbose", False))

    if args.command == "label":
        if getattr(args, "list", False):
            init_db()
            _as_json = getattr(args, "json", False)
            search_root = args.home or db.get_setting("home_folder", "") or str(Path.cwd())
            abs_path = _resolve_file(args.file, search_root)
            if not abs_path:
                log.error("Could not resolve file: %s", args.file)
                if _as_json:
                    _emit_json({"ok": False, "command": "label-list", "file_id": None,
                                "error": f"Could not resolve file: {args.file}",
                                "speakers": []})
                sys.exit(2)
            rec = next((f for f in db.get_all_files()
                        if str(Path(f["source_path"]).resolve()) == abs_path), None)
            if not rec:
                log.error("No DB record. Process the file first.")
                if _as_json:
                    _emit_json({"ok": False, "command": "label-list", "file_id": None,
                                "error": "No DB record. Process the file first.",
                                "speakers": []})
                sys.exit(2)
            if _as_json:
                _emit_json({"ok": True, "command": "label-list", "file_id": rec["id"],
                            "error": None, "speakers": _speakers_payload(rec["id"])})
            else:
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
