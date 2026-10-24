"""One durable worker. Database reservations survive browser and API restarts."""

import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import threading

from .store import Settings, database, initialize
from .results import finished_reels

log = logging.getLogger("podclipper.web.worker")


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def update(settings, job_id, **values):
    values.update(updated_at=time.time(), heartbeat=time.time())
    with database(settings) as db:
        db.execute("UPDATE jobs SET " + ",".join(f"{key}=?" for key in values) + " WHERE id=?",
                   (*values.values(), job_id))
    (settings.data / "worker-heartbeat").touch()


def collect(settings, job_id, directory):
    progress = directory / "progress.json"
    if not progress.exists():
        return
    try:
        state = json.loads(progress.read_text())
    except (OSError, ValueError):
        return
    stage = state.get("stage")
    if stage in ("transcribing", "selecting", "exporting"):
        update(settings, job_id, stage=stage)
    for value in finished_reels(directory, state):
        path = Path(value).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or path.suffix != ".mp4":
            continue
        sidecar = path.with_suffix(".txt")
        if not sidecar.exists() or not path.with_suffix(".captions.json").exists():
            continue
        relative = str(path.relative_to(directory))
        title = path.stem
        for line in sidecar.read_text().splitlines():
            if line.startswith("title: "):
                title = line[7:][:200]
                break
        key = hashlib.sha256((job_id + relative).encode()).hexdigest()[:24]
        with database(settings) as db:
            if db.execute("SELECT 1 FROM reels WHERE id=?", (key,)).fetchone():
                continue
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                 "-of", "json", str(path)], capture_output=True, text=True, timeout=30, check=True)
        duration = float(json.loads(result.stdout)["format"]["duration"])
        with database(settings) as db:
            db.execute("INSERT OR IGNORE INTO reels VALUES (?,?,?,?,?,?)",
                       (key, job_id, title, relative, duration, time.time()))


def download(settings, job, directory):
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError
    source = directory / "source.mp4"
    ready = directory / "source-ready.json"
    if source.is_file() and ready.is_file():
        return
    source.unlink(missing_ok=True)
    if shutil.disk_usage(settings.data).free < 12 * 1024**3:
        raise ValueError("The server is low on storage. Your allowance has been returned; try again later.")
    options = {"quiet": True, "no_warnings": True, "noplaylist": True,
               "socket_timeout": 30, "retries": 3, "extractor_retries": 2,
               "format": "bv*[height<=1080]+ba/b[height<=1080]",
               "merge_output_format": "mp4", "max_filesize": 3 * 1024**3,
               "outtmpl": str(directory / "source.%(ext)s"),
               "restrictfilenames": True, "overwrites": False}
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(job["url"], download=False)
            if (info.get("is_live") or not info.get("duration")
                    or info["duration"] > settings.max_seconds):
                raise ValueError(f"Choose a recorded video up to {settings.max_seconds // 60} minutes long.")
            if info.get("age_limit", 0) >= 18 or info.get("availability") in ("private", "premium_only", "subscriber_only", "needs_auth"):
                raise ValueError("Choose a public video that does not require signing into YouTube.")
            update(settings, job["id"], title=str(info.get("title") or "YouTube podcast")[:300], stage="downloading")
            ydl.process_ie_result(info, download=True)
    except DownloadError as exc:
        log.warning("YouTube download failed for job %s: %s", job["id"], exc)
        raise ValueError("YouTube could not provide this video. Check it is public and try again; your allowance has been returned.") from None
    if not source.is_file():
        raise ValueError("The video could not be downloaded within the server's file-size limit. Your allowance has been returned.")
    if source.stat().st_size > 3 * 1024**3:
        source.unlink()
        raise ValueError("This video's download exceeds the 3 GB server limit. Your allowance has been returned.")
    save(ready, {"bytes": source.stat().st_size})


def process(settings, job, lock_fd):
    directory = settings.data / "jobs" / job["id"]
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    save(directory / "request.json", {"clips": job["clips"]})
    process = None
    try:
        update(settings, job["id"], stage="downloading", error=None)
        download(settings, job, directory)
        update(settings, job["id"], stage="transcribing")
        with (directory / "processing.log").open("a", buffering=1) as output:
            # Child keeps the global flock alive if the worker crashes. A replacement
            # worker waits for it, then resumes through the pipeline's checkpoints.
            process = subprocess.Popen([sys.executable, "-u", "-m", "podclipper_web.render", str(directory)],
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                       pass_fds=(lock_fd,))
            started = time.monotonic()
            while process.poll() is None:
                collect(settings, job["id"], directory)
                update(settings, job["id"])
                if time.monotonic() - started > 6 * 3600:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise ValueError("Processing reached its time limit. Completed reels are saved; retry the remaining work.")
                time.sleep(3)
        collect(settings, job["id"], directory)
        with database(settings) as db:
            count = db.execute("SELECT COUNT(*) FROM reels WHERE job_id=?", (job["id"],)).fetchone()[0]
        if not count:
            raise ValueError("No reels could be completed. Your allowance has been returned. Retry the job or contact support.")
        progress = json.loads((directory / "progress.json").read_text())
        expected = progress.get("selected_count", job["clips"])
        state = "completed" if process.returncode == 0 and count >= expected else "partial"
        update(settings, job["id"], state=state, stage=state, finished_at=time.time(),
               error=None if state == "completed" else "Some reels were not produced. You can keep these results or retry the remaining work.")
        if state == "completed":
            source = directory / "source.mp4"
            source.unlink(missing_ok=True)
            shutil.rmtree(directory / ".cache", ignore_errors=True)
    except Exception as exc:
        log.exception("Job %s failed", job["id"])
        with database(settings) as db:
            count = db.execute("SELECT COUNT(*) FROM reels WHERE job_id=?", (job["id"],)).fetchone()[0]
        message = str(exc) if isinstance(exc, ValueError) else "Processing was interrupted. Retry this job or contact support."
        update(settings, job["id"], state="partial" if count else "failed", stage="partial" if count else "failed",
               reserved=1 if count else 0, error=message[:500], finished_at=time.time())
    finally:
        if process is not None and process.poll() is None:
            process.terminate()


def expire(settings):
    cutoff = time.time() - settings.retention_days * 86400
    with database(settings) as db:
        rows = db.execute("SELECT id FROM jobs WHERE state IN ('completed','partial','failed') AND finished_at<?", (cutoff,)).fetchall()
        for row in rows:
            shutil.rmtree(settings.data / "jobs" / row["id"], ignore_errors=True)
            db.execute("DELETE FROM reels WHERE job_id=?", (row["id"],))
            db.execute("UPDATE jobs SET state='expired',stage='expired',updated_at=? WHERE id=?", (time.time(), row["id"]))


def main():
    settings = Settings.load()
    initialize(settings)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not os.environ.get("LLM_API_KEY"):
        raise ValueError("Set LLM_API_KEY in the backend .env")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise ValueError("Install ffmpeg and include it in the service PATH")
    with (settings.data / "worker.lock").open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                log.info("Another worker or its pipeline child is still active; waiting")
                time.sleep(10)
        with database(settings) as db:
            db.execute("UPDATE jobs SET state='queued',stage='queued' WHERE state='running'")
        def heartbeat():
            while True:
                (settings.data / "worker-heartbeat").touch()
                time.sleep(10)
        threading.Thread(target=heartbeat, daemon=True).start()
        while True:
            (settings.data / "worker-heartbeat").touch()
            expire(settings)
            with database(settings) as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created_at LIMIT 1").fetchone()
                if row:
                    db.execute("UPDATE jobs SET state='running',stage='downloading',attempts=attempts+1,updated_at=? WHERE id=?",
                               (time.time(), row["id"]))
            if row:
                process(settings, dict(row), lock.fileno())
            else:
                time.sleep(5)


if __name__ == "__main__":
    main()
