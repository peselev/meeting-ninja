"""One-time utility: wipe all file/speaker/segment rows from the DB.
Does NOT touch your actual recordings or transcript files on disk.
Run:  python cleanup.py
"""
from meeting_ninja.db.client import get_conn

conn = get_conn()
conn.execute("DELETE FROM segments")
conn.execute("DELETE FROM speakers")
conn.execute("DELETE FROM files")
conn.commit()
conn.close()
print("Cleared all file rows. Your recordings and transcripts on disk are untouched.")
