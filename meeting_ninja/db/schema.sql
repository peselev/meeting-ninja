CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS files (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path          TEXT NOT NULL,
    filename             TEXT NOT NULL,
    source_type          TEXT NOT NULL CHECK(source_type IN ('video','audio')),
    duration_sec         REAL,
    file_created_at      TEXT,
    added_at             TEXT NOT NULL,
    status               TEXT NOT NULL DEFAULT 'pending',
    audio_path           TEXT,
    transcript_txt_path  TEXT,
    transcript_json_path TEXT,
    job_id               TEXT,
    description          TEXT,
    offset_sec           REAL NOT NULL DEFAULT 0,
    error_message        TEXT
);

CREATE TABLE IF NOT EXISTS speakers (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id              INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    diarization_label    TEXT NOT NULL,
    display_name         TEXT,
    sample_segments_json TEXT
);

CREATE TABLE IF NOT EXISTS segments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    start_sec   REAL NOT NULL,
    end_sec     REAL NOT NULL,
    speaker_id  INTEGER REFERENCES speakers(id),
    text        TEXT NOT NULL
);
