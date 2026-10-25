"""Resumable owned video uploads with bounded disk streaming."""

import asyncio
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import uuid

from anyio import CancelScope, to_thread

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.requests import ClientDisconnect

from podclipper.render_cache import write_json
from .store import database, month, usage
from .tasks import enqueue

MAX_UPLOAD_BYTES = 1_000_000_000
CHUNK_BYTES = 32_000_000
PARALLEL_REQUESTS = 4
MIN_FREE_BYTES = 12 * 1024**3
UPLOAD_TTL = 24 * 3600


def is_chunk_request(scope):
    return scope.get('method') == 'POST' and re.fullmatch(r'/uploads/[A-Za-z0-9-]+/chunks/\d+', scope.get('path', '')) is not None


class UploadSubmission(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0, lt=MAX_UPLOAD_BYTES, strict=True)
    request_key: str = Field(min_length=16, max_length=80)


def upload_info(db, job_id):
    row = db.execute('SELECT filename,size FROM uploads WHERE job_id=?', (job_id,)).fetchone()
    if row is None:
        return None
    parts = db.execute('SELECT chunk_index,size FROM upload_chunks WHERE job_id=? ORDER BY chunk_index', (job_id,)).fetchall()
    return {'filename': row['filename'], 'size': row['size'], 'chunk_bytes': CHUNK_BYTES,
            'parallel_requests': PARALLEL_REQUESTS, 'total_chunks': math.ceil(row['size'] / CHUNK_BYTES),
            'completed_chunks': [r['chunk_index'] for r in parts], 'received_bytes': sum(r['size'] for r in parts)}


@contextmanager
def upload_lock(directory, exclusive=False):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / 'upload.lock').open('a') as lock:
        try:
            fcntl.flock(lock, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise HTTPException(409, 'This upload is busy. Please retry shortly.') from None
        yield


def probe_video(path, max_seconds):
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe',
                                 '-show_entries', 'format=duration,format_name:stream=codec_type',
                                 '-of', 'json', str(path)], capture_output=True, text=True, timeout=30, check=True)
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        kinds = {s.get('codec_type') for s in data['streams']}
        formats = set(data['format'].get('format_name', '').split(','))
        if not formats.intersection({'mov', 'mp4', 'matroska', 'webm'}):
            raise ValueError('Use an MP4, MOV, M4V, MKV, or WebM video.')
        if not {'video', 'audio'}.issubset(kinds):
            raise ValueError('Upload a video with an audio track.')
        if not math.isfinite(duration) or not 0 < duration <= max_seconds:
            raise ValueError(f'Upload a video up to {max_seconds // 60} minutes long.')
        return duration
    except (subprocess.CalledProcessError, KeyError, TypeError, json.JSONDecodeError):
        raise ValueError('This file could not be read as a video with audio.') from None


def expire_uploads(settings):
    with database(settings) as db:
        rows = db.execute("SELECT u.job_id FROM uploads u JOIN jobs j ON j.id=u.job_id "
                          "WHERE j.state='uploading' AND u.updated_at<?", (time.time() - UPLOAD_TTL,)).fetchall()
    for row in rows:
        directory = settings.data / 'jobs' / row['job_id']
        try:
            with upload_lock(directory, exclusive=True), database(settings) as db:
                db.execute('BEGIN IMMEDIATE')
                active = db.execute("SELECT u.job_id FROM uploads u JOIN jobs j ON j.id=u.job_id "
                                    "WHERE j.id=? AND j.state='uploading' AND u.updated_at<?",
                                    (row['job_id'], time.time() - UPLOAD_TTL)).fetchone()
                if active:
                    db.execute('DELETE FROM upload_chunks WHERE job_id=?', (row['job_id'],))
                    db.execute("UPDATE jobs SET state='expired',stage='expired',reserved=0,error=?,updated_at=? WHERE id=?",
                               ('Incomplete upload expired after 24 hours. Your allowance was returned.', time.time(), row['job_id']))
            if active:
                shutil.rmtree(directory / 'upload', ignore_errors=True)
        except HTTPException:
            continue


class ChunkReceiver:
    """Synchronous disk work dispatched through a bounded API I/O thread pool."""
    def __init__(self, settings, job_id, index, upload, validate):
        self.settings, self.job_id, self.index, self.validate = settings, job_id, index, validate
        count = math.ceil(upload['size'] / CHUNK_BYTES)
        if not 0 <= index < count:
            raise HTTPException(422, 'Invalid upload chunk number.')
        self.expected = min(CHUNK_BYTES, upload['size'] - index * CHUNK_BYTES)
        self.stack, self.temporary, self.stream = ExitStack(), None, None
        self.digest, self.size = hashlib.sha256(), 0
        directory = settings.data / 'jobs' / job_id
        try:
            self.stack.enter_context(upload_lock(directory))
            self.chunks = directory / 'upload' / 'chunks'
            self.chunks.mkdir(parents=True, exist_ok=True, mode=0o700)
            lock = self.stack.enter_context((self.chunks / f'{index:04d}.lock').open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise HTTPException(409, 'This chunk is already uploading. Please retry.') from None
            if shutil.disk_usage(directory).free < MIN_FREE_BYTES + upload['size'] + self.expected:
                raise HTTPException(507, 'The server is low on storage. Your upload is saved; try again later.')
            self.stream = self.stack.enter_context(tempfile.NamedTemporaryFile(dir=self.chunks, suffix='.tmp', delete=False))
            self.temporary = Path(self.stream.name)
        except BaseException:
            self.close()
            raise

    def write(self, block):
        self.size += len(block)
        if self.size > self.expected:
            raise HTTPException(413, 'Upload chunk exceeds its expected size.')
        self.digest.update(block)
        self.stream.write(block)

    def finish(self):
        if self.size != self.expected:
            raise HTTPException(422, 'Upload chunk is incomplete. Retry this chunk.')
        self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close()
        with database(self.settings) as db:
            db.execute('BEGIN IMMEDIATE')
            self.validate(db, self.job_id, active=True)
            prior = db.execute('SELECT size,sha256 FROM upload_chunks WHERE job_id=? AND chunk_index=?',
                               (self.job_id, self.index)).fetchone()
            if prior:
                if prior['size'] != self.size or prior['sha256'] != self.digest.hexdigest():
                    raise HTTPException(409, 'This chunk was already uploaded with different contents.')
            else:
                self.temporary.replace(self.chunks / f'{self.index:04d}.part')
                db.execute('INSERT INTO upload_chunks VALUES (?,?,?,?)',
                           (self.job_id, self.index, self.size, self.digest.hexdigest()))
            db.execute('UPDATE uploads SET updated_at=? WHERE job_id=?', (time.time(), self.job_id))
        return {'id': self.job_id, 'chunk': self.index, 'received_bytes': self.size}

    def close(self):
        if self.stream is not None:
            self.stream.close()
        if self.temporary is not None:
            self.temporary.unlink(missing_ok=True)
        self.stack.close()


def finalize_upload(settings, job_id):
    """Assemble and validate saved chunks in a worker, never an HTTP request."""
    directory = settings.data / 'jobs' / job_id
    with upload_lock(directory, exclusive=True):
        with database(settings) as db:
            job = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
            upload = db.execute('SELECT * FROM uploads WHERE job_id=?', (job_id,)).fetchone()
            if not job or not upload or job['source_type'] != 'upload':
                raise ValueError('Upload not found')
            if job['state'] not in ('queued', 'running'):
                return False
            parts = db.execute('SELECT * FROM upload_chunks WHERE job_id=? ORDER BY chunk_index', (job_id,)).fetchall()
        ready = directory / 'source-ready.json'
        if ready.is_file() and (directory / 'source.mp4').is_file():
            return True
        count = math.ceil(upload['size'] / CHUNK_BYTES)
        if [r['chunk_index'] for r in parts] != list(range(count)):
            with database(settings) as db:
                db.execute("UPDATE jobs SET state='uploading',stage='uploading',error=?,updated_at=? WHERE id=?",
                           ('Some upload chunks are missing. Choose the same file to resume.', time.time(), job_id))
            return False
        if shutil.disk_usage(directory).free < MIN_FREE_BYTES + upload['size']:
            raise OSError('Not enough free space to assemble this upload')
        assembled = directory / 'upload' / 'assembled.tmp'
        with assembled.open('wb') as output:
            for part in parts:
                source = directory / 'upload' / 'chunks' / f'{part["chunk_index"]:04d}.part'
                digest = hashlib.sha256(); size = 0
                if source.is_file():
                    with source.open('rb') as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            output.write(block); digest.update(block); size += len(block)
                if size != part['size'] or digest.hexdigest() != part['sha256']:
                    with database(settings) as db:
                        db.execute('DELETE FROM upload_chunks WHERE job_id=? AND chunk_index=?', (job_id, part['chunk_index']))
                        db.execute("UPDATE jobs SET state='uploading',stage='uploading',error=?,updated_at=? WHERE id=?",
                                   ('A saved chunk needs uploading again. Choose the same file to resume.', time.time(), job_id))
                    source.unlink(missing_ok=True)
                    assembled.unlink(missing_ok=True)
                    return False
            output.flush(); os.fsync(output.fileno())
        try:
            if assembled.stat().st_size != upload['size']:
                raise ValueError('The completed upload has an unexpected file size.')
            duration = probe_video(assembled, settings.max_seconds)
        except ValueError as exc:
            with database(settings) as db:
                db.execute("UPDATE jobs SET state='failed',stage='failed',reserved=0,error=?,finished_at=?,updated_at=? WHERE id=?",
                           (str(exc)+' Your allowance has been returned. Choose a valid video to start a new upload.', time.time(), time.time(), job_id))
                db.execute('DELETE FROM upload_chunks WHERE job_id=?', (job_id,))
            shutil.rmtree(directory / 'upload', ignore_errors=True)
            return False
        assembled.replace(directory / 'source.mp4')
        write_json(ready, {'bytes': upload['size'], 'duration': duration, 'source_type': 'upload'})
        shutil.rmtree(directory / 'upload', ignore_errors=True)
        return True


def register_upload_routes(app, settings, auth, owned, serialize, rate_limit):
    slots = asyncio.Semaphore(PARALLEL_REQUESTS)

    def session(db, job_id, user, active=False):
        job = owned(db, job_id, user)
        if job['source_type'] != 'upload':
            raise HTTPException(404, 'Upload not found')
        row = db.execute('SELECT * FROM uploads WHERE job_id=?', (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, 'Upload not found')
        if active and job['state'] != 'uploading':
            raise HTTPException(409, 'This upload is closed. Start a new upload if needed.')
        return job, row

    @app.post('/uploads')
    def start_upload(payload: UploadSubmission, user=Depends(auth)):
        rate_limit('submit:' + user['id'], 10)
        filename = Path(payload.filename.replace('\\', '/')).name.strip()
        if any(ord(char) < 32 for char in filename):
            raise HTTPException(422, 'Choose a video with a valid filename.')
        if Path(filename).suffix.lower() not in {'.mp4', '.mov', '.m4v', '.mkv', '.webm'}:
            raise HTTPException(422, 'Choose an MP4, MOV, M4V, MKV, or WebM video.')
        with database(settings) as db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT * FROM jobs WHERE user_id=? AND request_key=?',
                                  (user['id'], payload.request_key)).fetchone()
            if previous:
                info = upload_info(db, previous['id'])
                if not info or info['filename'] != filename or info['size'] != payload.size:
                    raise HTTPException(409, 'This request was already used for another submission.')
                return serialize(db, previous)
            if (limit := settings.access_for(user['id']).monthly_videos) is not None and usage(db, user['id']) >= limit:
                raise HTTPException(429, f'You have used all {limit} videos for this month. Your allowance resets on the 1st (UTC).')
            if db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('uploading','queued','running')").fetchone()[0] >= 100:
                raise HTTPException(503, 'The processing queue is full. Try again later.')
            if shutil.disk_usage(settings.data).free < MIN_FREE_BYTES + 2 * payload.size:
                raise HTTPException(507, 'The server is low on storage. Please try again later.')
            job_id = str(uuid.uuid4()); now = time.time()
            db.execute("INSERT INTO jobs (id,user_id,video_id,url,title,clips,quota_month,request_key,created_at,updated_at,state,stage,source_type) "
                       "VALUES (?,?,?,'',?,0,?,?,?,?,'uploading','uploading','upload')",
                       (job_id, user['id'], 'upload:' + job_id, Path(filename).stem[:300], month(), payload.request_key, now, now))
            db.execute('INSERT INTO uploads VALUES (?,?,?,?)', (job_id, filename, payload.size, now))
            return serialize(db, db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone())

    @app.get('/uploads/{job_id}')
    def upload_status(job_id: str, user=Depends(auth)):
        with database(settings) as db:
            job, _ = session(db, job_id, user)
            return {'id': job_id, 'state': job['state'], **upload_info(db, job_id)}

    @app.post('/uploads/{job_id}/chunks/{index}')
    async def receive_chunk(job_id: str, index: int, request: Request, user=Depends(auth)):
        await to_thread.run_sync(rate_limit, 'chunk:' + user['id'], 240)
        def begin():
            with database(settings) as db:
                _, upload = session(db, job_id, user, active=True)
            return ChunkReceiver(settings, job_id, index, upload,
                                 lambda db, jid, active: session(db, jid, user, active=active))
        async with slots:
            receiver = await to_thread.run_sync(begin)
            try:
                buffer, size = bytearray(), 0
                async for block in request.stream():
                    size += len(block)
                    if size > receiver.expected:
                        raise HTTPException(413, 'Upload chunk exceeds its expected size.')
                    buffer.extend(block)
                    if len(buffer) >= 1024 * 1024:
                        await to_thread.run_sync(receiver.write, bytes(buffer))
                        buffer.clear()
                if buffer:
                    await to_thread.run_sync(receiver.write, bytes(buffer))
                return await to_thread.run_sync(receiver.finish)
            except ClientDisconnect:
                raise HTTPException(400, 'Upload interrupted. Retry the missing chunk.') from None
            finally:
                with CancelScope(shield=True):
                    await to_thread.run_sync(receiver.close)

    @app.post('/uploads/{job_id}/complete', status_code=202)
    def finish_upload(job_id: str, user=Depends(auth)):
        rate_limit('complete:' + user['id'], 20)
        with database(settings) as db:
            db.execute('BEGIN IMMEDIATE')
            job, upload = session(db, job_id, user)
            if job['state'] in {'queued', 'running', 'completed', 'partial'}:
                return serialize(db, job)
            if job['state'] != 'uploading':
                raise HTTPException(409, 'This upload is closed.')
            parts = db.execute('SELECT chunk_index FROM upload_chunks WHERE job_id=? ORDER BY chunk_index', (job_id,)).fetchall()
            if [r['chunk_index'] for r in parts] != list(range(math.ceil(upload['size'] / CHUNK_BYTES))):
                raise HTTPException(409, 'Some upload chunks are missing. Resume the upload.')
            db.execute("UPDATE jobs SET state='queued',stage='validating',error=NULL,updated_at=? WHERE id=?", (time.time(), job_id))
            enqueue(db,job_id,'prepare',reset=True)
            return serialize(db, db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone())

    @app.post('/uploads/{job_id}/cancel')
    def cancel_upload(job_id: str, user=Depends(auth)):
        with database(settings) as db:
            job, _ = session(db, job_id, user)
            if job['state'] == 'cancelled':
                return {'ok': True}
            if job['state'] != 'uploading':
                raise HTTPException(409, 'This video has already left the upload stage.')
        directory = settings.data / 'jobs' / job_id
        with upload_lock(directory, exclusive=True):
            with database(settings) as db:
                session(db, job_id, user, active=True)
                db.execute('DELETE FROM upload_chunks WHERE job_id=?', (job_id,))
                db.execute("UPDATE jobs SET state='cancelled',stage='cancelled',reserved=0,finished_at=?,updated_at=? WHERE id=?",
                           (time.time(), time.time(), job_id))
            shutil.rmtree(directory / 'upload', ignore_errors=True)
        return {'ok': True}
