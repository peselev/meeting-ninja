import shutil
from pathlib import Path


def route_to_job_folder(
    transcript_txt_path: str,
    stem: str,
    job_id: str,
    jobs_root: str,
) -> tuple[bool, str]:
    """
    Copy transcript to {jobs_root}/{job_id}/transcript-{stem}.txt.
    Creates the job folder if it doesn't exist.
    Returns (success, message).
    """
    if not job_id or not job_id.strip():
        return False, "No job ID set — skipping job folder copy."
    if not jobs_root or not jobs_root.strip():
        return False, "Destination folder not configured — skipping copy."

    job_dir = Path(jobs_root) / job_id.strip()
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return False, f"Could not create job folder {job_dir}: {e}"

    dest = job_dir / f"transcript-{stem}.txt"
    try:
        shutil.copy2(transcript_txt_path, dest)
    except OSError as e:
        return False, f"Copy failed: {e}"

    return True, str(dest)
