"""Small authenticated HTTP API. Heavy work belongs to the separate worker."""

from contextlib import asynccontextmanager
import hashlib
import secrets
import subprocess
import time
from urllib.parse import parse_qs, urlparse
import re
import uuid

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .store import Settings, database, initialize, month, usage
from .results import reel_poster, reel_quality


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def youtube_url(value):
    parsed = urlparse(value.strip())
    if (parsed.scheme != "https" or parsed.username or parsed.password
            or parsed.port not in (None, 443)):
        raise ValueError("Paste an HTTPS YouTube video link")
    if parsed.hostname == "youtu.be":
        video_id = parsed.path.strip("/")
    elif parsed.hostname in ("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif re.fullmatch(r"/(?:live|shorts|embed)/[\w-]{11}/?", parsed.path):
            video_id = parsed.path.strip("/").split("/")[1]
        else:
            raise ValueError("Use a link to one YouTube video")
    else:
        raise ValueError("Only YouTube video links are supported")
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("That YouTube video link is incomplete")
    return video_id, f"https://www.youtube.com/watch?v={video_id}"


class Credential(BaseModel):
    credential: str = Field(min_length=20, max_length=12000)


class Submission(BaseModel):
    url: str = Field(max_length=2048)
    clips: int = Field(default=5, ge=1, le=15, strict=True)
    request_key: str = Field(min_length=16, max_length=80)


class SmallBodies:
    """Bound JSON bodies before parsing, including requests sent without a length."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        messages, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > 16384:
                await send({"type": "http.response.start", "status": 413,
                            "headers": [(b"content-type", b"application/json"), (b"cache-control", b"private, no-store")]})
                return await send({"type": "http.response.body", "body": b'{"detail":"Request is too large"}'})
            messages.append(message)
            if not message.get("more_body", False):
                break
        async def replay():
            return messages.pop(0) if messages else await receive()
        await self.app(scope, replay, send)


def create_app():
    settings = Settings.load()

    @asynccontextmanager
    async def lifespan(app):
        initialize(settings)
        yield

    app = FastAPI(title="PodClipper", lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.add_middleware(SmallBodies)
    app.add_middleware(CORSMiddleware, allow_origins=[settings.origin],
                       allow_credentials=True, allow_methods=["GET", "HEAD", "POST"],
                       allow_headers=["Content-Type", "Range", "X-CSRF-Token", "X-Podclipper-Request"],
                       expose_headers=["Content-Length", "Content-Range", "Accept-Ranges"])

    @app.middleware("http")
    async def private_responses(request, call_next):
        if request.headers.get("content-length", "0").isdigit() and int(request.headers.get("content-length", 0)) > 16384:
            return Response(status_code=413)
        if request.method == "POST" and request.headers.get("origin") != settings.origin:
            return Response(status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def rate_limit(key, count):
        now = int(time.time() // 60)
        with database(settings) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT window,count FROM rate_limits WHERE key=?", (key,)).fetchone()
            used = row["count"] if row and row["window"] == now else 0
            if used >= count:
                raise HTTPException(429, "Too many requests. Please wait a minute.")
            db.execute("INSERT OR REPLACE INTO rate_limits VALUES (?,?,?)", (key, now, used + 1))
            db.execute("DELETE FROM rate_limits WHERE window < ?", (now - 60,))

    def auth(request: Request):
        token = request.cookies.get(settings.cookie, "")
        with database(settings) as db:
            row = db.execute("SELECT users.*, sessions.csrf FROM sessions JOIN users ON users.id=sessions.user_id "
                             "WHERE digest=? AND expires_at>?", (digest(token), time.time())).fetchone()
        if not row:
            raise HTTPException(401, "Sign in with Google to continue")
        if request.method == "POST" and not secrets.compare_digest(
                request.headers.get("X-CSRF-Token", ""), row["csrf"]):
            raise HTTPException(403, "Your session changed. Refresh and try again.")
        return dict(row)

    def owned(db, job_id, user):
        row = db.execute("SELECT * FROM jobs WHERE id=? AND user_id=?", (job_id, user["id"])).fetchone()
        if not row:
            raise HTTPException(404, "Podcast not found")
        return row

    def is_admin(user):
        return bool(settings.admin_user_id) and secrets.compare_digest(user["id"], settings.admin_user_id)

    def admin(user=Depends(auth)):
        if not is_admin(user):
            raise HTTPException(403, "This dashboard is only available to the administrator")
        return user

    def serialize(db, row):
        result = {k: row[k] for k in ("id", "url", "title", "clips", "state", "stage", "created_at",
                                     "updated_at", "finished_at", "error")}
        result["reels"] = []
        directory = settings.data / "jobs" / row["id"]
        for reel in db.execute("SELECT id,title,duration,path FROM reels WHERE job_id=? ORDER BY path", (row["id"],)):
            path = (directory / reel["path"]).resolve()
            result["reels"].append({"id": reel["id"], "title": reel["title"], "duration": reel["duration"],
                                    "verdict": reel_quality(path) if path.is_relative_to(directory) else None})
        return result

    def serialize_admin(db, row):
        result = serialize(db, row)
        owner = db.execute("SELECT name,email FROM users WHERE id=?", (row["user_id"],)).fetchone()
        result["owner"] = dict(owner)
        return result

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/source")
    def source_code():
        archive = settings.data.parent / "podclipper-source.tar.gz"
        if not archive.is_file():
            raise HTTPException(404, "Source archive is unavailable")
        return FileResponse(archive, filename="podclipper-source.tar.gz", media_type="application/gzip")

    @app.get("/config")
    def config():
        with database(settings) as db:
            return {"google_client_id": settings.client_id, "monthly_limit": 5,
                    "max_video_minutes": settings.max_seconds // 60,
                    "retention_days": settings.retention_days,
                    "worker_available": (settings.data / "worker-heartbeat").exists()
                    and time.time() - (settings.data / "worker-heartbeat").stat().st_mtime < 90}

    @app.get("/auth/challenge")
    def challenge(request: Request, response: Response):
        rate_limit("challenge:" + (request.headers.get("cf-connecting-ip") or request.client.host), 20)
        nonce = secrets.token_urlsafe(32)
        with database(settings) as db:
            db.execute("DELETE FROM challenges WHERE expires_at<?", (time.time(),))
            db.execute("INSERT INTO challenges VALUES (?,?)", (digest(nonce), time.time() + 600))
        response.set_cookie(settings.challenge_cookie, nonce, max_age=600, secure=settings.secure,
                            httponly=True, samesite="lax", path="/")
        return {"nonce": nonce}

    @app.post("/auth/google")
    def google_login(payload: Credential, request: Request, response: Response):
        from google.auth.transport.requests import Request as GoogleRequest
        from google.oauth2.id_token import verify_oauth2_token
        rate_limit("login:" + (request.headers.get("cf-connecting-ip") or request.client.host), 10)
        nonce = request.cookies.get(settings.challenge_cookie, "")
        if not nonce or request.headers.get("X-Podclipper-Request") != "1":
            raise HTTPException(403, "Reload the sign-in page and try again")
        try:
            info = verify_oauth2_token(payload.credential, GoogleRequest(), settings.client_id)
        except ValueError:
            raise HTTPException(401, "Google sign-in could not be verified") from None
        except Exception:
            raise HTTPException(503, "Google sign-in is temporarily unavailable. Try again shortly.") from None
        if (info.get("iss") not in ("accounts.google.com", "https://accounts.google.com")
                or not info.get("email_verified") or not info.get("sub") or not info.get("email")
                or not secrets.compare_digest(str(info.get("nonce", "")), nonce)):
            raise HTTPException(401, "Google sign-in could not be verified")
        token, csrf = secrets.token_urlsafe(48), secrets.token_urlsafe(32)
        now = time.time()
        with database(settings) as db:
            db.execute("BEGIN IMMEDIATE")
            used = db.execute("DELETE FROM challenges WHERE digest=? AND expires_at>?", (digest(nonce), now))
            if used.rowcount != 1:
                raise HTTPException(401, "Sign-in expired. Please try again.")
            db.execute("INSERT INTO users VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET email=excluded.email,name=excluded.name",
                       (info["sub"], str(info["email"])[:320], str(info.get("name", ""))[:200], now))
            db.execute("DELETE FROM sessions WHERE expires_at<?", (now,))
            db.execute("INSERT INTO sessions VALUES (?,?,?,?)", (digest(token), info["sub"], csrf, now + 30 * 86400))
        response.set_cookie(settings.cookie, token, max_age=30 * 86400, secure=settings.secure,
                            httponly=True, samesite="lax", path="/")
        response.delete_cookie(settings.challenge_cookie, secure=settings.secure, httponly=True, samesite="lax", path="/")
        return {"ok": True}

    @app.get("/me")
    def me(user=Depends(auth)):
        with database(settings) as db:
            return {"name": user["name"], "email": user["email"], "csrf": user["csrf"],
                    "used": usage(db, user["id"]), "limit": 5, "month": month(), "timezone": "UTC",
                    "is_admin": is_admin(user)}

    @app.post("/auth/logout")
    def logout(request: Request, response: Response, user=Depends(auth)):
        with database(settings) as db:
            db.execute("DELETE FROM sessions WHERE digest=?", (digest(request.cookies.get(settings.cookie, "")),))
        response.delete_cookie(settings.cookie, secure=settings.secure, httponly=True, samesite="lax", path="/")
        return {"ok": True}

    @app.get("/jobs")
    def jobs(user=Depends(auth)):
        with database(settings) as db:
            return [serialize(db, row) for row in db.execute(
                "SELECT * FROM jobs WHERE user_id=? ORDER BY created_at DESC LIMIT 100", (user["id"],)).fetchall()]

    @app.get("/admin/jobs")
    def admin_jobs(offset: int = Query(default=0, ge=0), user=Depends(admin)):
        with database(settings) as db:
            counts = {row["state"]: row["count"] for row in db.execute(
                "SELECT state,COUNT(*) AS count FROM jobs GROUP BY state")}
            rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC,id LIMIT 50 OFFSET ?", (offset,)).fetchall()
            return {"jobs": [serialize_admin(db, row) for row in rows], "total": sum(counts.values()),
                    "counts": counts, "offset": offset, "limit": 50}

    @app.get("/admin/jobs/{job_id}")
    def admin_job(job_id: str, user=Depends(admin)):
        with database(settings) as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                raise HTTPException(404, "Podcast not found")
            return serialize_admin(db, row)

    @app.post("/jobs")
    def submit(payload: Submission, user=Depends(auth)):
        rate_limit("submit:" + user["id"], 10)
        try:
            video_id, url = youtube_url(payload.url)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        now = time.time()
        with database(settings) as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT * FROM jobs WHERE user_id=? AND request_key=?",
                                  (user["id"], payload.request_key)).fetchone()
            if previous:
                if previous["video_id"] != video_id or previous["clips"] != payload.clips:
                    raise HTTPException(409, "This request was already used for another submission")
                return serialize(db, previous)
            existing = db.execute("SELECT * FROM jobs WHERE user_id=? AND video_id=? "
                                  "AND state IN ('queued','running','completed','partial') ORDER BY created_at DESC LIMIT 1",
                                  (user["id"], video_id)).fetchone()
            if existing:
                return serialize(db, existing)
            if usage(db, user["id"]) >= 5:
                raise HTTPException(429, "You have used all 5 videos for this month. Your allowance resets on the 1st (UTC).")
            if db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0] >= 100:
                raise HTTPException(503, "The processing queue is full. Try again later.")
            job_id = str(uuid.uuid4())
            db.execute("INSERT INTO jobs (id,user_id,video_id,url,clips,quota_month,request_key,created_at,updated_at) "
                       "VALUES (?,?,?,?,?,?,?,?,?)", (job_id, user["id"], video_id, url, payload.clips,
                                                      month(), payload.request_key, now, now))
            return serialize(db, db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    @app.get("/jobs/{job_id}")
    def job(job_id: str, user=Depends(auth)):
        with database(settings) as db:
            return serialize(db, owned(db, job_id, user))

    @app.post("/jobs/{job_id}/retry")
    def retry(job_id: str, user=Depends(auth)):
        rate_limit("retry:" + user["id"], 5)
        with database(settings) as db:
            db.execute("BEGIN IMMEDIATE")
            row = owned(db, job_id, user)
            if row["state"] not in ("failed", "partial") or row["attempts"] >= 3:
                raise HTTPException(409, "This job cannot be retried. Contact support if it needs repair.")
            quota_month = row["quota_month"]
            if not row["reserved"]:
                if usage(db, user["id"]) >= 5:
                    raise HTTPException(429, "Your monthly allowance is used up")
                quota_month = month()
            db.execute("UPDATE jobs SET state='queued',stage='queued',error=NULL,reserved=1,quota_month=?,"
                       "finished_at=NULL,updated_at=? WHERE id=?", (quota_month, time.time(), job_id))
            return {"ok": True}

    @app.api_route("/jobs/{job_id}/reels/{reel_id}", methods=["GET", "HEAD"])
    def reel(job_id: str, reel_id: str, download: bool = False, poster: bool = False, user=Depends(auth)):
        with database(settings) as db:
            row = owned(db, job_id, user)
            return reel_response(db, row, reel_id, download, poster)

    @app.api_route("/admin/jobs/{job_id}/reels/{reel_id}", methods=["GET", "HEAD"])
    def admin_reel(job_id: str, reel_id: str, download: bool = False, poster: bool = False, user=Depends(admin)):
        with database(settings) as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                raise HTTPException(404, "Podcast not found")
            return reel_response(db, row, reel_id, download, poster)

    def reel_response(db, row, reel_id, download, poster):
        job_id = row["id"]
        item = db.execute("SELECT * FROM reels WHERE id=? AND job_id=?", (reel_id, job_id)).fetchone()
        if not item or row["state"] == "expired":
            raise HTTPException(404, "Reel not found or no longer available")
        path = (settings.data / "jobs" / job_id / item["path"]).resolve()
        if not path.is_relative_to(settings.data / "jobs" / job_id) or not path.is_file():
            raise HTTPException(404, "Reel is no longer available")
        if poster:
            try:
                preview = reel_poster(path)
            except ValueError:
                raise HTTPException(404, "Preview is unavailable") from None
            except (OSError, subprocess.SubprocessError):
                raise HTTPException(503, "Preview is temporarily unavailable; the reel can still be played") from None
            return FileResponse(preview, media_type="image/jpeg")
        return FileResponse(path, media_type="video/mp4", filename=path.name if download else None,
                            headers={"Cache-Control": "private, no-store"})

    return app
