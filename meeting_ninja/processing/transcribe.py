from __future__ import annotations
import json
from pathlib import Path

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]


def transcribe(audio_path: str, home_folder: str, model_name: str = "base",
               language: str | None = None, out_stem: str | None = None) -> tuple[str, str]:
    """
    Transcribe audio using Whisper.
    language: ISO code like 'en' to force, or None to auto-detect.
    out_stem: name the .txt/.json after this stem instead of the audio filename.
    Returns (txt_path, json_path).
    Saves both to {home_folder}/transcripts/.
    """
    import whisper  # imported here so the app loads even if whisper isn't installed yet

    transcript_dir = Path(home_folder) / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    stem = out_stem or Path(audio_path).stem
    txt_path = transcript_dir / f"{stem}.txt"
    json_path = transcript_dir / f"{stem}.json"

    model = whisper.load_model(model_name)
    transcribe_kwargs = {
        "verbose": False,
        # Do NOT feed the previous window's text back into the decoder. On quiet
        # or low-SNR audio (phone calls, a soft-spoken speaker) that feedback makes
        # Whisper collapse into a repetition loop ("Yes. Yes. Yes.") and then skip
        # over real speech. Disabling it trades a little cross-sentence fluency for
        # far fewer hallucinated loops and dropped segments. Temperature fallback
        # (Whisper's default 0.0→1.0 ladder) stays on to recover low-confidence spans.
        "condition_on_previous_text": False,
    }
    if language and language != "auto":
        transcribe_kwargs["language"] = language
    result = model.transcribe(audio_path, **transcribe_kwargs)

    # Save plain text
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"].strip())

    # Save full JSON (segments with timestamps — required for diarization merge)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return str(txt_path), str(json_path)


def load_segments_from_json(json_path: str) -> list[dict]:
    """
    Load Whisper segments from saved JSON.
    Returns list of {start, end, text}.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        {"start_sec": seg["start"], "end_sec": seg["end"], "text": seg["text"].strip()}
        for seg in data.get("segments", [])
    ]
