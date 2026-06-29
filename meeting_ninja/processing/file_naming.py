"""
file_naming.py — rename a source recording to match its description.

When a file is processed with a description, the original is renamed in place to
the (filesystem-safe) description text, keeping its extension. Outputs derive
their names from the source stem, so they follow automatically.
"""
from __future__ import annotations
import re
from pathlib import Path

# Characters that break paths on macOS / Unix: path separators, colon
# (the classic HFS separator, which Finder also rejects), and control chars.
_ILLEGAL = re.compile(r"[/\\:\x00-\x1f]")


def slug_from_description(description: str, ext: str) -> str | None:
    """Turn free-text into a filesystem-safe filename.

    Keeps the original case and spaces; only strips characters that would break
    a path. Returns None if nothing usable remains.
    """
    base = _ILLEGAL.sub(" ", description)
    base = re.sub(r"\s+", " ", base).strip().strip(".")
    if not base:
        return None
    return f"{base}{ext}"


def unique_path(directory: Path, filename: str) -> Path:
    """Finder-style collision handling: name.ext, name (2).ext, name (3).ext …"""
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem
    ext = Path(filename).suffix
    n = 2
    while True:
        candidate = directory / f"{stem} ({n}){ext}"
        if not candidate.exists():
            return candidate
        n += 1


def rename_source_with_description(source_path: str, description: str) -> str:
    """Rename the original recording in place to match its description.

    Returns the new absolute path, or the original path unchanged when the
    description is empty after sanitizing or the file is already named correctly.
    Never overwrites an existing file; collisions get a ` (N)` suffix.
    """
    src = Path(source_path)
    target_name = slug_from_description(description, src.suffix)
    if not target_name or target_name == src.name:
        return str(src.resolve())
    dest = unique_path(src.parent, target_name)
    src.rename(dest)
    return str(dest.resolve())
