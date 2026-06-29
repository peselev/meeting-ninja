---
name: meeting-ninja
description: >-
  Transcribe a local meeting recording (any audio or video file) and label its
  speakers, fully offline. Use when the user wants a transcript of a recording,
  speaker-attributed notes, or to name the speakers in an already-processed file.
  Drives the installed `meeting-ninja` CLI and reads its JSON output.
---

# meeting-ninja

A local CLI that transcribes a recording with Whisper, optionally diarizes
speakers with pyannote, and writes a labeled plain-text transcript. Audio never
leaves the machine. This skill covers calling it from the command line and
parsing its results.

## When to use

- The user points at a recording (`.mov`, `.mp4`, `.m4a`, `.mp3`, `.opus`, …)
  and wants a transcript or meeting notes.
- The user wants to assign real names to the detected speakers in a file that
  was already processed.

## Prerequisites

- The `meeting-ninja` command is on PATH (`pipx install .` from the repo).
- `ffmpeg` and `ffprobe` are on PATH.
- Speaker diarization is optional. It needs `pip install '.[diarization]'` plus a
  HuggingFace token configured in the app, or it is skipped. Without it, every
  segment is assigned to a single speaker (`SPEAKER_00`) for manual labeling.

## The contract

Always pass `--json`. With it:

- **stdout** carries exactly one JSON object. Parse that.
- **stderr** carries human progress logs. Ignore unless debugging.
- **Exit code** is `0` on success, `1` on a pipeline failure, `2` on a bad
  argument or an unresolvable file. Branch on the code, then read `ok` and
  `error` in the JSON to confirm.

Never parse the stderr logs. The JSON on stdout is the only contract.

## Commands

### 1. Process a file

```bash
meeting-ninja --file "<path-or-name>" --json [options]
```

`--file` accepts an absolute path, a path relative to the search folder, or a
bare filename searched for recursively. An absolute path is the most reliable
and is preferred when calling programmatically.

Options:

| Flag | Meaning |
|---|---|
| `--description "<text>"` | Names the derived files (audio/, transcripts/) after this text. The source file is never renamed. See "Naming outputs" below. |
| `--offset <seconds>` | Skip dead air at the start (e.g. `--offset 30`). |
| `--model <name>` | `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`. Larger is more accurate and slower. Default `base`. |
| `--language <code>` | ISO code like `en`, `ru`, `he`, or `auto` to detect. Default `en`. |
| `--no-diarize` | Skip speaker diarization (faster; one speaker). |
| `--tag "<id>"` | Copy the finished transcript to `<destination>/<id>/transcript-<name>.txt` if a destination folder is configured. |
| `--home "<dir>"` | Override the output root. By default outputs land next to the source file. |

Success output:

```json
{
  "ok": true,
  "command": "process",
  "file_id": 12,
  "source_path": "/abs/path/raw recording.mov",
  "filename": "raw recording.mov",
  "status": "done",
  "audio_path": "/abs/path/audio/Weekly sync.wav",
  "transcript_txt_path": "/abs/path/transcripts/Weekly sync.txt",
  "transcript_json_path": "/abs/path/transcripts/Weekly sync.json",
  "diarized": true,
  "speakers": [
    { "label": "SPEAKER_00", "display_name": null, "sample": "first words of a sample segment" }
  ],
  "destination_copy": null,
  "error": null
}
```

Field notes:

- `transcript_txt_path` is the labeled plain-text transcript. Read this for the
  result. It contains `[HH:MM:SS] Speaker: text` lines.
- `source_path` and `filename` are always the original recording. It is never
  renamed; only the derived files take the description name.
- `speakers` lists each detected speaker with a sample excerpt. `display_name`
  is `null` until labeled.
- `diarized` is `false` when diarization was skipped or unavailable.

Failure output has `"ok": false`, populated `"error"`, and a non-zero exit code:

```json
{ "ok": false, "command": "process", "file_id": null,
  "source_path": "nope.mov", "status": "error", "error": "Could not resolve file: nope.mov",
  "audio_path": null, "transcript_txt_path": null, "transcript_json_path": null,
  "diarized": false, "speakers": [], "destination_copy": null }
```

### 2. List the speakers in a processed file

```bash
meeting-ninja label --file "<path>" --list --json
```

```json
{ "ok": true, "command": "label-list", "file_id": 12, "error": null,
  "speakers": [ { "label": "SPEAKER_00", "display_name": null, "sample": "..." } ] }
```

Use this to see the diarization labels and a sample of each speaker's speech
before assigning names.

### 3. Assign speaker names

```bash
meeting-ninja label --file "<path>" \
  --speaker SPEAKER_00=Konstantin \
  --speaker SPEAKER_01=Interviewer \
  --json
```

Repeat `--speaker LABEL=Name` per speaker. This rewrites the labeled transcript
with the real names and, if a `--tag` and destination were set, refreshes the
destination copy.

```json
{ "ok": true, "command": "label", "file_id": 12, "error": null,
  "speakers": [ { "label": "SPEAKER_00", "display_name": "Konstantin", "sample": "..." } ],
  "transcript_txt_path": "/abs/path/transcripts/Weekly sync.txt",
  "destination_copy": null }
```

## Naming outputs with `--description`

When `--description` is set, the derived files are named after the description
text instead of the source filename: `audio/<description>.wav`,
`transcripts/<description>.txt` / `.json`. Case and spaces are kept; only
path-breaking characters are stripped. The source recording itself is never
renamed or moved.

Naming is unique per file. If a different recording already produced outputs
under the same description, the next one gets a Finder-style ` (2)`, ` (3)`
suffix so nothing is overwritten. Reprocessing the same file reuses its own
name. Without a description, outputs fall back to the source filename's stem.

```bash
meeting-ninja --file "/recordings/2026-06-05 11-58-45.mov" \
  --description "Intro call with Nav" --json
# → transcripts/Intro call with Nav.txt
```

## Formats

Any file `ffprobe` can decode for an audio track works, including ones not in
the built-in extension list (e.g. `.opus`, `.wma`, `.m4b`). The tool transcodes
to a WAV internally when needed. Files with no decodable audio are rejected with
exit code `2`.

## Typical agent workflow

1. Run `process` with an absolute `--file` and `--json`. Add `--description` if
   the user gave the meeting a name, `--no-diarize` for a one-person recording.
2. Check the exit code and `ok`. On failure, report `error`.
3. Read `transcript_txt_path` for the transcript.
4. If the user wants named speakers: run `label --list` to see the labels and
   samples, infer or ask for names, then run `label` with `--speaker`
   assignments, and re-read `transcript_txt_path`.

## Notes

- Diarization is CPU-heavy. A 35-minute recording takes a few minutes.
- `process` re-runs the full pipeline each call. Calling it again on the same
  file reprocesses it.
- Interactive flags (`label --interactive`) are for humans, not agents. Use
  `--speaker` assignments instead.
