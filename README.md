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

I built this to process my own meeting recordings without sending audio to a cloud service. Off-the-shelf tools (Zoom, Otter, Chorus) do transcription and speaker detection, but only for calls *they* host. This works on any local `.mov` / `.mp4` / `.mp3` / `.wav` file — including screen recordings — and keeps everything private.

It's also a small study in stitching together a fragile ML dependency stack (Whisper + pyannote + torch) into something reliable, with structural fixes for the version-incompatibility issues those libraries are prone to.

---

## Requirements

- Python 3.9+
- ffmpeg + ffprobe on your PATH
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: `winget install ffmpeg` or [download a build](https://ffmpeg.org/download.html)

## Setup

```bash
git clone https://github.com/<your-username>/meeting-ninja.git
cd meeting-ninja

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
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

The CLI does the same pipeline, synchronously, with verbose logging — handy for scripting or batch use.

**Process a file:**

```bash
python cli.py --file "team-sync.mov" \
  --offset 30 \
  --model medium \
  --language en \
  --tag project-x \
  --description "Weekly planning sync"
```

Skip diarization for casual recordings:

```bash
python cli.py --file "casual-call.mov" --no-diarize
```

**Label speakers** for an already-processed file:

```bash
# List detected speakers with sample excerpts
python cli.py label --file "team-sync.mov" --list

# Assign names directly
python cli.py label --file "team-sync.mov" \
  --speaker SPEAKER_00=Alex \
  --speaker SPEAKER_01=Jordan

# Or label interactively (shows excerpts, prompts for each name)
python cli.py label --file "team-sync.mov" --interactive
```

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
| `app.py` | Streamlit entry point; sidebar settings + main view |
| `cli.py` | Headless driver with `process` and `label` subcommands |
| `db/` | SQLite schema + thin client |
| `processing/` | ffmpeg extraction, Whisper transcription, pyannote diarization + merge, transcript labeling, destination routing, threaded pipeline orchestrator |
| `ui/` | File manager, speaker labeling, settings screens |
| `utils/` | Media probing, warning suppression |

Transcription and diarization run in background threads so the UI stays responsive. The diarization step shows a live progress bar (it's CPU-heavy: budget a few minutes for a long recording).

---

## Notes & limitations

- Diarization is slow on CPU. A 35-minute recording takes a few minutes. There's no GPU acceleration wired in.
- The dependency stack (torch 2.6 + pyannote) has known incompatibilities; this repo includes the workarounds (forcing `weights_only=False`, bypassing torchaudio's ffmpeg backend via soundfile, version-proofing the HF auth path).
- Analysis (notes, summaries) is intentionally *not* built in — the labeled transcript is designed to be pasted into the LLM of your choice.

## License

MIT