"""Durable users, sessions, quotas, jobs, and private reel manifests."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3


@dataclass
class Settings:
    data: Path
    client_id: str
    origin: str
    secure: bool
    max_seconds: int
    retention_days: int
    admin_user_id: str = ""

    @classmethod
    def load(cls):
        from podclipper.main import _load_dotenv
        _load_dotenv(Path(os.environ.get("PODCLIPPER_WEB_ENV", ".env")))
        origin = os.environ.get("WEB_ORIGIN", "https://podclipper.loukik.dev").rstrip("/")
        secure = origin.startswith("https://")
        if not secure and origin not in ("http://localhost:5173", "http://127.0.0.1:5173"):
            raise ValueError("WEB_ORIGIN must use HTTPS, except for local development")
        settings = cls(Path(os.environ.get("WEB_DATA_DIR", "web-data")).expanduser().resolve(),
                       os.environ.get("GOOGLE_CLIENT_ID", ""), origin, secure,
                       int(os.environ.get("WEB_MAX_VIDEO_SECONDS", "10800")),
                       int(os.environ.get("WEB_RETENTION_DAYS", "7")),
                       os.environ.get("WEB_ADMIN_GOOGLE_SUB", "").strip())
        if not settings.client_id.endswith(".apps.googleusercontent.com"):
            raise ValueError("Set GOOGLE_CLIENT_ID to the Google OAuth web client ID")
        if not 60 <= settings.max_seconds <= 14400 or not 1 <= settings.retention_days <= 30:
            raise ValueError("Invalid video duration or retention settings")
        settings.data.mkdir(parents=True, exist_ok=True, mode=0o700)
        settings.data.chmod(0o700)
        return settings

    @property
    def cookie(self):
        return "__Host-podclipper_session" if self.secure else "podclipper_session"

    @property
    def challenge_cookie(self):
        return "__Host-podclipper_challenge" if self.secure else "podclipper_challenge"


@contextmanager
def database(settings):
    db = sqlite3.connect(settings.data / "web.sqlite3", timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    try:
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def initialize(settings):
    with database(settings) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
          id TEXT PRIMARY KEY, email TEXT NOT NULL, name TEXT NOT NULL, created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS sessions (
          digest TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          csrf TEXT NOT NULL, expires_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS challenges (digest TEXT PRIMARY KEY, expires_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), video_id TEXT NOT NULL,
          url TEXT NOT NULL, title TEXT NOT NULL DEFAULT '', clips INTEGER NOT NULL,
          state TEXT NOT NULL DEFAULT 'queued', stage TEXT NOT NULL DEFAULT 'queued',
          quota_month TEXT NOT NULL, reserved INTEGER NOT NULL DEFAULT 1,
          request_key TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
          created_at REAL NOT NULL, updated_at REAL NOT NULL, finished_at REAL,
          heartbeat REAL, error TEXT, UNIQUE(user_id, request_key));
        CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(user_id, created_at);
        CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(state,created_at);
        CREATE INDEX IF NOT EXISTS jobs_quota ON jobs(user_id, quota_month, reserved);
        CREATE TABLE IF NOT EXISTS reels (
          id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
          title TEXT NOT NULL, path TEXT NOT NULL, duration REAL NOT NULL,
          created_at REAL NOT NULL, UNIQUE(job_id, path));
        CREATE TABLE IF NOT EXISTS tasks (
          job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE, kind TEXT NOT NULL,
          state TEXT NOT NULL DEFAULT 'pending', owner TEXT, lease_until REAL,
          attempts INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(job_id,kind));
        CREATE INDEX IF NOT EXISTS tasks_ready ON tasks(kind,state,lease_until);
        CREATE TABLE IF NOT EXISTS rate_limits (
          key TEXT PRIMARY KEY, window INTEGER NOT NULL, count INTEGER NOT NULL);
        """)
        db.execute("BEGIN IMMEDIATE")
        if "source_type" not in {r["name"] for r in db.execute("PRAGMA table_info(jobs)")}:
            db.execute("ALTER TABLE jobs ADD COLUMN source_type TEXT NOT NULL DEFAULT 'youtube'")
        for table, additions in (('jobs', [('selected_count','INTEGER'),('discovery_complete','INTEGER'),('purged_at','REAL')]),
                                 ('reels', [('verdict','TEXT')])):
            columns={r['name'] for r in db.execute(f'PRAGMA table_info({table})')}
            for name,kind in additions:
                if name not in columns:
                    db.execute(f'ALTER TABLE {table} ADD COLUMN {name} {kind}')
        db.executescript("""
        CREATE TABLE IF NOT EXISTS uploads (
          job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
          filename TEXT NOT NULL, size INTEGER NOT NULL, updated_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS upload_chunks (
          job_id TEXT NOT NULL REFERENCES uploads(job_id) ON DELETE CASCADE,
          chunk_index INTEGER NOT NULL, size INTEGER NOT NULL, sha256 TEXT NOT NULL,
          PRIMARY KEY(job_id, chunk_index));
        """)
    (settings.data / "web.sqlite3").chmod(0o600)


def month():
    return datetime.now(timezone.utc).strftime("%Y-%m")


def usage(db, user_id):
    return db.execute("SELECT COUNT(*) FROM jobs WHERE user_id=? AND quota_month=? AND reserved=1",
                      (user_id, month())).fetchone()[0]
