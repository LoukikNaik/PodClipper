"""Durable preparation and render tasks supervised with bounded concurrency."""

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
import uuid

from .tasks import claim, complete, enqueue, renew

from .store import Settings, database, initialize
from .results import finished_reels, reel_quality, reel_poster

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


def sync_metadata(settings, job_id, directory=None, posters=False):
    """Persist worker-produced summaries; HTTP status reads need no sidecars."""
    directory = directory or settings.data / 'jobs' / job_id
    try:
        selection = json.loads((directory / 'selection.json').read_text())
        clips = selection['clips']
        if not isinstance(clips,list):
            raise ValueError('Invalid selection')
        with database(settings) as db:
            db.execute('UPDATE jobs SET selected_count=?,discovery_complete=? WHERE id=?',
                       (len(clips),int(selection.get('discovery_complete',True) is True),job_id))
    except (OSError,ValueError,KeyError,TypeError):
        pass
    with database(settings) as db:
        reels = db.execute('SELECT id,path,verdict FROM reels WHERE job_id=?',(job_id,)).fetchall()
    for reel in reels:
        path = (directory / reel['path']).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            continue
        if reel['verdict'] is None:
            verdict = reel_quality(path)
            if verdict:
                with database(settings) as db:
                    db.execute('UPDATE reels SET verdict=? WHERE id=?',(verdict,reel['id']))
        if posters:
            try:
                reel_poster(path)
            except (OSError,ValueError,subprocess.SubprocessError):
                log.warning('Poster preparation failed for %s; a later worker pass can retry',reel['id'])


def collect(settings, job_id, directory):
    sync_metadata(settings,job_id,directory,posters=True)
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
        try:
            reel_poster(path)
        except (OSError,ValueError,subprocess.SubprocessError):
            log.warning('Poster preparation failed for %s; a later worker pass can retry',key)

        with database(settings) as db:
            db.execute("INSERT OR IGNORE INTO reels (id,job_id,title,path,duration,created_at,verdict) VALUES (?,?,?,?,?,?,?)",
                       (key, job_id, title, relative, duration, time.time(),reel_quality(path)))


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
    save(ready, {"bytes": source.stat().st_size, "duration": info["duration"]})


def prepare_request(settings, job, directory):
    if not job['clips']:
        from podclipper.config import load_default_config
        from podclipper.reel_limits import reel_cap
        metadata = json.loads((directory / 'source-ready.json').read_text())
        duration = metadata.get('duration')
        if duration is None:
            from podclipper.ingest import ingest
            duration = ingest(directory / 'source.mp4').duration
        job['clips'] = reel_cap(duration, load_default_config())
        update(settings, job['id'], clips=job['clips'])
    save(directory / 'request.json', {'clips': job['clips']})


def process(settings, job, lock_fds, preparation=False):
    directory = settings.data / "jobs" / job["id"]
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    process = None
    try:
        if preparation:
            update(settings, job['id'], stage='validating' if job.get('source_type') == 'upload' else 'downloading', error=None)
            if job.get('source_type') == 'upload':
                from .uploads import finalize_upload
                if not finalize_upload(settings, job['id']):
                    return False
            download(settings, job, directory)
            prepare_request(settings, job, directory)
        update(settings, job['id'], stage='transcribing')
        with (directory / "processing.log").open("a", buffering=1) as output:
            # Children retain this job's lock and, for exports, the render slot.
            command = [sys.executable, '-u', '-m', 'podclipper_web.render', str(directory)]
            if preparation:
                command.append('transcribe')
            process = subprocess.Popen(command,
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                       pass_fds=lock_fds)
            started = time.monotonic()
            while process.poll() is None:
                if not preparation:
                    collect(settings, job['id'], directory)
                update(settings, job['id'])
                if time.monotonic() - started > 6 * 3600:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise ValueError("Processing reached its time limit. Completed reels are saved; retry the remaining work.")
                time.sleep(3)
        if preparation:
            if process.returncode != 0:
                raise ValueError('Source transcription was interrupted. Retry to resume its saved chunks.')
            return True
        collect(settings, job['id'], directory)
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
    from .uploads import expire_uploads
    expire_uploads(settings)
    with database(settings) as db:
        rows = db.execute("SELECT id,user_id,finished_at,state FROM jobs WHERE "
                          "state IN ('completed','partial','failed') OR (state='expired' AND purged_at IS NULL)").fetchall()
    for row in rows:
        days = settings.access_for(row['user_id']).retention_days
        if days is None:
            continue
        cutoff = time.time() - days * 86400
        if row['state'] != 'expired' and (row['finished_at'] is None or row['finished_at'] >= cutoff):
            continue
        with database(settings) as db:
            db.execute('BEGIN IMMEDIATE')
            changed=db.execute("UPDATE jobs SET state='expired',stage='expired',updated_at=? WHERE id=? AND "
                               "((state IN ('completed','partial','failed') AND finished_at<?) OR (state='expired' AND purged_at IS NULL))",
                               (time.time(),row['id'],cutoff))
            if not changed.rowcount:
                continue
            db.execute('DELETE FROM reels WHERE job_id=?',(row['id'],))
        directory=settings.data/'jobs'/row['id']
        shutil.rmtree(directory,ignore_errors=True)
        if not directory.exists():
            with database(settings) as db:
                db.execute('UPDATE jobs SET purged_at=? WHERE id=?',(time.time(),row['id']))


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
        owner = uuid.uuid4().hex
        stop = threading.Event()
        with database(settings) as db:
            for row in db.execute("SELECT id FROM jobs WHERE state='queued'").fetchall():
                enqueue(db,row['id'],'prepare')

        def run_tasks(kind):
            while not stop.is_set():
                try:
                    with claim(settings,kind,owner) as task:
                        if task is None:
                            stop.wait(2)
                            continue
                        prepared = process(settings,task,task['lock_fds'],preparation=kind=='prepare')
                        complete(settings,task,next_kind='render' if kind=='prepare' and prepared else None)
                except Exception:
                    log.exception('Web task runner failed; its lease and checkpoints allow recovery')
                    stop.wait(2)

        threads = [threading.Thread(target=run_tasks,args=(kind,),name=f'podclipper-{kind}-{i}',daemon=True)
                   for kind,count in (('prepare',2),('render',1)) for i in range(count)]
        for thread in threads:
            thread.start()
        def backfill():
            with database(settings) as db:
                ids=[row['id'] for row in db.execute("SELECT id FROM jobs WHERE state IN ('completed','partial')")]
            for job_id in ids:
                sync_metadata(settings,job_id,posters=True)
        threading.Thread(target=backfill,name='podclipper-metadata',daemon=True).start()
        try:
            while True:
                (settings.data / 'worker-heartbeat').touch()
                renew(settings,owner)
                expire(settings)
                stop.wait(10)
        finally:
            stop.set()


if __name__ == "__main__":
    main()
