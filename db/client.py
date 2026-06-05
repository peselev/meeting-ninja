from __future__ import annotations
import sqlite3
import os
from pathlib import Path

DB_DIR = Path.home() / ".meeting-transcriber"
DB_PATH = DB_DIR / "db.sqlite"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_conn() -> sqlite3.Connection:
    DB_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db():
    conn = get_conn()
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())

    # ── Migrations for existing DBs (add columns that may be missing) ────────
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(files)")}
    if "offset_sec" not in existing_cols:
        conn.execute("ALTER TABLE files ADD COLUMN offset_sec REAL NOT NULL DEFAULT 0")

    conn.commit()
    conn.close()


# ── Settings ────────────────────────────────────────────────────────────────

def get_setting(key: str, default=None):
    conn = get_conn()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key: str, value: str):
    conn = get_conn()
    conn.execute(
        "INSERT INTO settings(key, value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()
    conn.close()


# ── Files ────────────────────────────────────────────────────────────────────

def add_file(record: dict) -> int:
    conn = get_conn()
    cols = ", ".join(record.keys())
    placeholders = ", ".join("?" for _ in record)
    cur = conn.execute(
        f"INSERT INTO files ({cols}) VALUES ({placeholders})",
        list(record.values()),
    )
    conn.commit()
    file_id = cur.lastrowid
    conn.close()
    return file_id


def get_all_files() -> list:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM files ORDER BY added_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_file(file_id: int) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_file(file_id: int, **kwargs):
    if not kwargs:
        return
    set_clause = ", ".join(f"{k} = ?" for k in kwargs)
    conn = get_conn()
    conn.execute(
        f"UPDATE files SET {set_clause} WHERE id = ?",
        list(kwargs.values()) + [file_id],
    )
    conn.commit()
    conn.close()


def delete_file(file_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
    conn.commit()
    conn.close()


# ── Speakers ─────────────────────────────────────────────────────────────────

def upsert_speaker(file_id: int, diarization_label: str, display_name: str = None,
                   sample_segments_json: str = None) -> int:
    conn = get_conn()
    row = conn.execute(
        "SELECT id FROM speakers WHERE file_id = ? AND diarization_label = ?",
        (file_id, diarization_label),
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE speakers SET display_name = ?, sample_segments_json = ? WHERE id = ?",
            (display_name, sample_segments_json, row["id"]),
        )
        speaker_id = row["id"]
    else:
        cur = conn.execute(
            "INSERT INTO speakers(file_id, diarization_label, display_name, sample_segments_json) VALUES(?,?,?,?)",
            (file_id, diarization_label, display_name, sample_segments_json),
        )
        speaker_id = cur.lastrowid
    conn.commit()
    conn.close()
    return speaker_id


def get_speakers(file_id: int) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM speakers WHERE file_id = ? ORDER BY diarization_label",
        (file_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_speaker_name(speaker_id: int, display_name: str):
    conn = get_conn()
    conn.execute("UPDATE speakers SET display_name = ? WHERE id = ?", (display_name, speaker_id))
    conn.commit()
    conn.close()


# ── Segments ──────────────────────────────────────────────────────────────────

def insert_segments(segments: list[dict]):
    """Bulk insert; each dict must have file_id, start_sec, end_sec, speaker_id (nullable), text."""
    if not segments:
        return
    conn = get_conn()
    conn.executemany(
        "INSERT INTO segments(file_id, start_sec, end_sec, speaker_id, text) VALUES(?,?,?,?,?)",
        [(s["file_id"], s["start_sec"], s["end_sec"], s.get("speaker_id"), s["text"]) for s in segments],
    )
    conn.commit()
    conn.close()


def get_segments(file_id: int) -> list:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT seg.id, seg.start_sec, seg.end_sec, seg.text,
               sp.diarization_label, sp.display_name
        FROM segments seg
        LEFT JOIN speakers sp ON seg.speaker_id = sp.id
        WHERE seg.file_id = ?
        ORDER BY seg.start_sec
        """,
        (file_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_segments(file_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM segments WHERE file_id = ?", (file_id,))
    conn.commit()
    conn.close()


def assign_speaker_to_segments(file_id: int, diarization_label: str, speaker_id: int):
    conn = get_conn()
    conn.execute(
        """
        UPDATE segments SET speaker_id = ?
        WHERE file_id = ? AND id IN (
            SELECT seg.id FROM segments seg
            LEFT JOIN speakers sp ON seg.speaker_id = sp.id
            WHERE seg.file_id = ? AND sp.diarization_label = ?
        )
        """,
        (speaker_id, file_id, file_id, diarization_label),
    )
    conn.commit()
    conn.close()
