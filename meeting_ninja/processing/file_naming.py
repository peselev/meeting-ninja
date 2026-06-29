"""
file_naming.py — derive output filenames from a file's description.

The source recording is never renamed. When a file has a description, its
derived files (audio/, transcripts/) are named after the sanitized description
instead of the source filename. Naming is unique per file: a *different* file
that happens to share a description gets a Finder-style ` (N)` suffix so nothing
is overwritten, while reprocessing the *same* file reuses its own stem.
"""
from __future__ import annotations
import re
from pathlib import Path

# Characters that break paths on macOS / Unix: path separators, colon (the
# classic HFS separator, which Finder also rejects), and control chars.
_ILLEGAL = re.compile(r"[/\\:\x00-\x1f]")


def sanitize_stem(description: str) -> str | None:
    """Turn free-text into a filesystem-safe filename stem (no extension).

    Keeps the original case and spaces; strips only characters that would break
    a path. Returns None if nothing usable remains.
    """
    base = _ILLEGAL.sub(" ", description)
    base = re.sub(r"\s+", " ", base).strip().strip(".")
    return base or None


def derive_output_stem(description: str, file_id: int, all_files: list[dict]) -> str | None:
    """Pick the output stem for this file from its description.

    Unique across *other* files' outputs (so two recordings sharing a
    description don't overwrite each other), but stable when reprocessing the
    same file (its own outputs are excluded from the collision check). Returns
    None when the description sanitizes to nothing, signalling the caller to
    fall back to the source filename.
    """
    base = sanitize_stem(description)
    if not base:
        return None

    # Stems already claimed by OTHER files' derived outputs.
    claimed: set[str] = set()
    for f in all_files:
        if f.get("id") == file_id:
            continue
        for key in ("transcript_txt_path", "transcript_json_path", "audio_path"):
            path = f.get(key)
            if path:
                claimed.add(Path(path).stem)

    candidate = base
    n = 2
    while candidate in claimed:
        candidate = f"{base} ({n})"
        n += 1
    return candidate
