# PodClipper hosted workspace

The existing React/Vite site talks to a private FastAPI service on the always-on
Mac (`luv`). Google Identity Services is the only login method. A separate worker
process runs the existing PodClipper pipeline; the API never renders a video.

## Install on the Mac

Use Python 3.12, FFmpeg, and yt-dlp's supported JavaScript runtime (Deno recommended).
Install the built wheel with the optional `web` dependencies into `.venv`, copy
`.env.example` to the project root as `.env`, and set the LLM key privately.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install '/path/to/podclipper.whl[web]'
chmod 600 .env
.venv/bin/python backend/install-services.py
```

The service installer registers `dev.podclipper.web-api` and
`dev.podclipper.web-worker` in the current user's GUI LaunchAgent domain. Keep
the Mac awake, online, and logged in. Credentials stay in `.env`, never in plists.
The API binds only `127.0.0.1:8000`; publish it through a named Cloudflare Tunnel.

## Cloudflare

Create a named tunnel and route `podclipper-api.loukik.dev` to it. On the Mac:

```yaml
tunnel: TUNNEL_UUID
credentials-file: /absolute/private/path/TUNNEL_UUID.json
ingress:
  - hostname: podclipper-api.loukik.dev
    service: http://127.0.0.1:8000
  - service: http_status:404
```

Run `cloudflared tunnel --config /absolute/path/config.yml run` as its own
LaunchAgent with KeepAlive and RunAtLoad. Bypass Cloudflare caching for this
hostname; API and media responses also carry `Cache-Control: private, no-store`.
Google login and application permissions are enforced by FastAPI, not Tunnel.

## Frontend and Google

Set `VITE_API_URL` in `landing/.env.local` or the build environment. The API
provides its public Google client ID to the frontend. Register
`https://podclipper.loukik.dev` as an authorized JavaScript origin in Google Auth
Platform. While the OAuth application is Testing, add the intended Google account
under Test users. The privacy page lives at `/privacy/` for production branding.

For local development use `WEB_ORIGIN=http://localhost:5173`,
`VITE_API_URL=http://localhost:8000`, and a local WEB_DATA_DIR. Cookies are Secure
and host-only in production; localhost mode relaxes Secure only for development.

## Persistence and authorization

New website submissions contain the YouTube URL and an idempotency key, with no
reel-count choice. Once the download duration is known, the worker derives the
cap: at least three, one per five source minutes, at most thirty. Selection scans
overlapping transcript windows and reviews a larger candidate pool. The UI shows
actual selected/finished totals. Existing saved selections are preserved on retry;
older API clients may still supply an explicit ceiling using `clips`.

### Admin dashboard

Set `WEB_ADMIN_GOOGLE_SUB` in the server's private `.env` to the administrator's
verified Google account identifier (`users.id` in SQLite), then restart the API.
Leave it blank to disable admin access. This is a single account ID, not an email
allowlist; only server configuration can grant the role.

The owner sees **Admin dashboard** in the workspace header (`/admin`). It lists
all users' jobs, current stages and finished reels, with playback and downloads.
It refreshes every five seconds and paginates in groups of 50. The admin view is
read-only; retry controls remain in each user's own workspace. All `/admin/*`
endpoints, including video GET/HEAD requests, enforce this role on the server.
Ordinary `/jobs/*` routes continue to enforce account ownership for everyone.
The configured administrator’s own results do not expire automatically, including
existing jobs. Other accounts retain the configured `WEB_RETENTION_DAYS` policy.
Abandoned uploads still expire, and successful jobs still clear source/cache files.

- `web-data/web.sqlite3` contains users, hashed session tokens, one-time login
  challenges, monthly quota reservations, job status, and finished reel manifests.
- Only Google ID tokens verified for this client ID can create sessions. Login
  uses a server-issued, single-use nonce; mutations require the exact frontend
  Origin and the session's CSRF token.
- Ordinary job and reel endpoints check account ownership; the separate admin
  endpoints require the configured Google account ID. Sources, caches, raw
  logs, and arbitrary filesystem paths are never served to customers.
- A transaction reserves one of five monthly slots before enqueueing. Months use
  UTC; queued/running/completed/partial jobs count. Failed jobs without outputs
  release their slot. Idempotency keys and duplicate-video lookup prevent network
  retries from charging twice. Retrying a failed job reacquires a slot when needed.
- One supervisor runs two preparation slots and one render slot. Preparation
  validates/uploads or downloads the source and caches cloud transcription, so a
  second podcast can prepare while another exports. Local Whisper remains in the
  single render slot when cloud transcription is disabled.
- SQLite tasks have expiring ownership leases. Per-job process locks and the
  render-slot lock are inherited by child processes, preventing duplicate work
  after a supervisor restart. Recovery resumes expired tasks through checkpoints;
  startup never resets every running job to queued.
- All finished, evaluated reels enter the manifest, including `skip` verdicts.
  Quality verdicts are advisory and shown to users beside playback and download.
  Selection reviews each discovery section as it completes, while one export
  runs alongside the next search. The website counts only finished reels and
  displays “reels ready for review” without a duration-based denominator.
  Partial retries continue unfinished discovery and reuse approved selections,
  skipping clips whose source bounds already have a completed export.
- Caption corrections request strict structured output. OpenRouter routing
  always requires parameter support for these requests, including when no
  provider allowlist is configured. Invalid JSON/mappings receive bounded retries
  and one subdivision into smaller chunks; successful chunks and subdivision
  decisions persist, so another attempt resumes the failed portion.
- Full-source Scribe chunks have a separate bounded repair budget
  (`transcribe.scribe.source_repair_max_windows`, default 64). The short reel
  caption budget remains eight windows. After retries, small unresolved source
  gaps are excluded from podcast/comedy clip selection with surrounding context;
  successful repairs are retained. Unresolved speech above 10% of a source chunk
  still stops processing (`transcribe.scribe.max_unreliable_fraction`). Exported
  reel captions must pass the strict coverage checks. Successful audio requests
  stay cached on retry.
- A job can be retried at most twice after its initial attempt. A pipeline run
  has a six-hour timeout. Public videos are limited to three hours and 3 GiB;
  there must be 12 GiB free before a new download.
- Finished/failed job files expire after seven days. Expiry runs between jobs;
  account and job history remain. Back up SQLite independently from video files.

No automated Instagram posting is connected to this web workspace.

## Operations

```sh
curl http://127.0.0.1:8000/health
launchctl print gui/$(id -u)/dev.podclipper.web-api
tail -50 web-data/logs/worker-error.log
```

Do not print `.env` or log Google credentials. Job details show customer-safe
errors; full diagnostics are private in each job's `processing.log`.

Deploy the exact corresponding source archive as `podclipper-source.tar.gz`
beside `web-data/`. `/source` offers it publicly under AGPL-3.0; it must exclude
credentials, sessions, user data, model weights, and private runtime configuration.

Application data is independent of Instagram posting history in podcast-agent.


## Video uploads

The workspace accepts MP4, MOV, M4V, MKV, and WebM files strictly below
1,000,000,000 bytes, with an audio track and the same three-hour video limit.
Uploads and YouTube links share the five-video monthly allowance.

The API reserves one allowance slot when an owned upload session starts. The
browser transfers 32,000,000-byte chunks with four parallel requests; the API
streams at most four chunks concurrently to private temporary files. Each chunk
has a saved SHA-256 receipt, may arrive out of order, and can be retried without
duplicating bytes. Select the same file after an interruption to resume. The
browser remembers only an upload identifier and a small file fingerprint.

Only the signed-in owner can send, resume, finalize, or cancel an upload. The
existing Origin and CSRF checks still apply, and ordinary JSON bodies remain
limited to 16 KiB. Each raw upload request is limited to 32 MB, so uploads work
through the existing Cloudflare Tunnel without sending a 1 GB request.

`POST /uploads` starts a session; `GET /uploads/{id}` reports received chunks;
`POST /uploads/{id}/chunks/{index}` transfers one raw part;
`POST /uploads/{id}/complete` registers background validation and returns HTTP 202;
a preparation task assembles the file, verifies checksums, and runs ffprobe before
allowing transcription or rendering; `POST /uploads/{id}/cancel` removes incomplete data and returns the
allowance. The worker consumes the assembled source through its normal pipeline.
Invalid media returns the allowance. Abandoned sessions expire after 24 hours of
inactivity; cleanup runs in the worker supervisor. File deletion stays outside
database write transactions.
Successful jobs retain the existing source/cache cleanup and result retention.
No storage provider, API key, or additional environment setting is required.


## API responsiveness

Upload streams remain bounded to four concurrent 32 MB chunks. Disk writes,
hashing, durable flushes, and chunk database updates run in I/O threads rather
than the API event loop. Completion requests do not assemble or probe files.
If a worker finds a corrupt saved part, the job returns to uploading and the
browser can resume the same session; invalid media releases the quota slot.

Workers persist selected counts, discovery state, and reel verdicts in SQLite,
and prepare posters before publishing new reel rows. A background startup pass
fills metadata/posters for legacy results. Status endpoints do not open progress
or scorecard files, and poster requests never launch FFmpeg.

Job lists support `?summary=true`: batched reel/owner reads return a preview and
actual `ready_count`, while job detail endpoints return the full reel list.
The web app polls only the open job, avoids repeatedly requesting account data,
skips polling in hidden tabs, and slows idle library polling. The API continues
using one process so the global four-upload limit remains intact.
