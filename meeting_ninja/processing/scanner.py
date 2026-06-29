from __future__ import annotations
from pathlib import Path
from datetime import datetime

from meeting_ninja.db import client as db
from meeting_ninja.utils.media_info import get_source_type, get_media_info, ALL_EXTENSIONS

# Folders the app creates itself — never scan these for "new" recordings
EXCLUDED_DIRNAMES = {"audio", "transcripts", "incoming"}


def scan_home_folder(home_folder: str) -> int:
    """
    Recursively scan home_folder for media files not yet in the DB.
    Adds new ones with status 'pending'. Returns count of files added.
    Dedup is by absolute source_path.
    """
    home = Path(home_folder)
    if not home.exists():
        return 0

    # Existing paths in DB (normalized to absolute)
    existing_paths = {
        str(Path(f["source_path"]).resolve())
        for f in db.get_all_files()
    }

    added = 0
    for path in home.rglob("*"):
        if not path.is_file():
            continue
        # Skip files inside app-managed output folders
        if any(part in EXCLUDED_DIRNAMES for part in path.parts):
            continue
        if path.suffix.lower() not in ALL_EXTENSIONS:
            continue

        abs_path = str(path.resolve())
        if abs_path in existing_paths:
            continue

        try:
            source_type = get_source_type(abs_path)
        except ValueError:
            continue

        info = get_media_info(abs_path)
        db.add_file({
            "source_path":     abs_path,
            "filename":        path.name,
            "source_type":     source_type,
            "duration_sec":    info["duration_sec"],
            "file_created_at": info["file_created_at"],
            "added_at":        datetime.now().isoformat(),
            "status":          "pending",
        })
        existing_paths.add(abs_path)
        added += 1

    return added
