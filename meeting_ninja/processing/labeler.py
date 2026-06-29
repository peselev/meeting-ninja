from __future__ import annotations
from meeting_ninja.db import client as db
from meeting_ninja.utils.media_info import format_duration


def write_labeled_transcript(file_id: int) -> str | None:
    """
    Rewrite the .txt transcript with speaker names prepended to each segment:
        [HH:MM:SS] SpeakerName: text
    Falls back to the raw diarization label when a speaker is unnamed.
    Returns the transcript path, or None if there's nothing to write.
    """
    record = db.get_file(file_id)
    txt_path = record.get("transcript_txt_path")
    if not txt_path:
        return None

    segments = db.get_segments(file_id)
    lines = []
    for seg in segments:
        ts = format_duration(seg["start_sec"])
        name = seg.get("display_name") or seg.get("diarization_label") or "Unknown"
        lines.append(f"[{ts}] {name}: {seg['text']}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return txt_path
