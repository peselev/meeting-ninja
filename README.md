# 🎙️ Meeting Ninja

<p align="center">
  <img src="docs/hero.jpg" alt="Meeting Ninja — blending meeting platforms into a unified transcript" width="380">
</p>

A fully local tool for transcribing meeting recordings, with automatic speaker diarization and a manual labeling step. Everything runs on your machine — audio never leaves your computer.

Built with [OpenAI Whisper](https://github.com/openai/whisper) for transcription and [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization, wrapped in a [Streamlit](https://streamlit.io) UI and a scriptable CLI.

---

## What it does

1. **Watches a folder** for recordings (video or audio).
2. **Extracts audio** from video via ffmpeg, with an optional per-file offset to skip dead air at the start.
3. **Transcribes** locally with Whisper (model selectable: `tiny` → `large-v3`).
4. **Diarizes** — figures out *who spoke when* — using pyannote (optional; needs a free HuggingFace token).
5. **Labels speakers** — you review excerpts and assign names (e.g. `SPEAKER_00` → "Alex"), with a video-preview jump to check if unsure.
6. **Routes the transcript** — if a file is tagged, its labeled transcript is copied to `{destination}/{tag}/transcript-{name}.txt` for easy filing.

The labeled transcript is plain text with timestamps and speaker names, ready to paste into any LLM for notes, summaries, or feedback.

---

## Why

I built this to process my own meeting recordings without sending audio to a cloud service. Off-the-shelf tools (Zoom, Otter, Chorus) do transcription and speaker detection, but only for calls *they* host. This works on any local video (`.mov`, `.mp4`, `.mkv`, …) or audio (`.mp3`, `.wav`, `.m4a`, `.amr`, …) file — including screen recordings and phone voice memos — and keeps everything private.

It's also a small study in stitching together a fragile ML dependency stack (Whisper + pyannote + torch) into something reliable, with structural fixes for the version-incompatibility issues those libraries are prone to.

---

## Requirements

- Python 3.9+
- ffmpeg + ffprobe on your PATH
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: `winget install ffmpeg` or [download a build](https://ffmpeg.org/download.html)

## Setup

### Install as a command (recommended)

```bash
git clone https://github.com/peselev/meeting-ninja.git
cd meeting-ninja

pipx install .                 # installs the `meeting-ninja` command
pipx install '.[diarization]'  # same, plus pyannote for speaker diarization
```

This puts a `meeting-ninja` command on your PATH. To reinstall after pulling
changes, run `pipx uninstall meeting-ninja` then `pipx install .`.

### Or run from a virtualenv (for the Streamlit UI / development)

```bash
git clone https://github.com/peselev/meeting-ninja.git
cd meeting-ninja

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e '.[ui,diarization]'
```

### Optional: speaker diarization

Diarization needs pyannote and a free HuggingFace token:

```bash
pip install pyannote.audio
```

1. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (role: read).
2. Accept the model license (one click):
   - https://huggingface.co/pyannote/speaker-diarization-community-1
3. Paste the token into the app's Settings sidebar.

Without a token, the app still transcribes — it just assigns everything to a single speaker for you to label manually.

---

## Running the app (UI)

```bash
streamlit run app.py
```

Opens at http://localhost:8501. On first run, set your **Home folder** in the Settings sidebar. Drop recordings there (or upload via the UI), set per-file options (Whisper model, language, skip-start seconds, an optional tag), and click **Process**. When done, click **🏷️ Label** on a file to name its speakers.

---

## Running headless (CLI)

The CLI runs the same pipeline, synchronously, with verbose logging on stderr.
Handy for scripting, batch use, and calling from agents like Claude Code.

**Process a file:**

```bash
meeting-ninja --file "team-sync.mov" \
  --offset 30 \
  --model medium \
  --language en \
  --tag project-x \
  --description "Weekly planning sync"
```

Skip diarization for casual recordings:

```bash
meeting-ninja --file "casual-call.mov" --no-diarize
```

**Label speakers** for an already-processed file:

```bash
# List detected speakers with sample excerpts
meeting-ninja label --file "team-sync.mov" --list

# Assign names directly
meeting-ninja label --file "team-sync.mov" \
  --speaker SPEAKER_00=Alex \
  --speaker SPEAKER_01=Jordan

# Or label interactively (shows excerpts, prompts for each name)
meeting-ninja label --file "team-sync.mov" --interactive
```

**Machine-readable output (`--json`):** add `--json` to any `process` or
`label` command and the tool prints exactly one JSON object to stdout, with all
human logs on stderr. This is the contract for scripting and agents.

```bash
meeting-ninja --file "team-sync.mov" --json
```

```json
{
  "ok": true,
  "command": "process",
  "file_id": 12,
  "source_path": "/abs/path/team-sync.mov",
  "status": "done",
  "transcript_txt_path": "/.../transcripts/team-sync.txt",
  "transcript_json_path": "/.../transcripts/team-sync.json",
  "diarized": true,
  "speakers": [{ "label": "SPEAKER_00", "display_name": null, "sample": "…" }],
  "destination_copy": null,
  "error": null
}
```

On failure the object has `"ok": false`, an `"error"` string, and a non-zero
exit code, so callers can branch on either.

---

## How files are organized

```
{home_folder}/
├── incoming/       uploaded files
├── audio/          extracted 16 kHz mono WAVs
└── transcripts/    {name}.txt (labeled) + {name}.json (raw, with timestamps)

{destination_folder}/        (optional)
└── {tag}/
    └── transcript-{name}.txt
```

State (file list, statuses, speaker labels) lives in a local SQLite DB at `~/.meeting-transcriber/db.sqlite`.

---

## Architecture

| Layer | What |
|---|---|
| `meeting_ninja/cli.py` | Headless driver with `process` and `label` subcommands; `--json` output. Entry point for the `meeting-ninja` command. |
| `meeting_ninja/db/` | SQLite schema + thin client |
| `meeting_ninja/processing/` | ffmpeg extraction, Whisper transcription, pyannote diarization + merge, transcript labeling, destination routing, threaded pipeline orchestrator |
| `meeting_ninja/utils/` | Media probing, warning suppression |
| `app.py`, `ui/` | Legacy Streamlit UI (being replaced by a FastAPI + React UI) |

Transcription and diarization run in background threads so the UI stays responsive. The diarization step shows a live progress bar (it's CPU-heavy: budget a few minutes for a long recording).

---

## Notes & limitations

- Diarization is slow on CPU. A 35-minute recording takes a few minutes. There's no GPU acceleration wired in.
- Non-WAV/FLAC/OGG sources (e.g. `.mp3`, `.m4a`, `.amr`) are transcoded to 16 kHz mono WAV via ffmpeg before processing, since diarization reads audio through soundfile, which can't decode them directly. AMR-NB is an 8 kHz narrowband speech codec — it transcribes fine, but accuracy trails wideband sources. Decoding `.amr` needs ffmpeg's `amrnb`/`amrwb` decoders, which ship with standard builds (`brew install ffmpeg`); verify with `ffmpeg -decoders | grep amr`.
- The dependency stack (torch 2.6 + pyannote) has known incompatibilities; this repo includes the workarounds (forcing `weights_only=False`, bypassing torchaudio's ffmpeg backend via soundfile, version-proofing the HF auth path).
- Analysis (notes, summaries) is intentionally *not* built in — the labeled transcript is designed to be pasted into the LLM of your choice.

## License

MIT