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

### Admin dashboard

Set `WEB_ADMIN_GOOGLE_SUB` in the server's private `.env` to the administrator's
verified Google account identifier (`users.id` in SQLite), then restart the API.
Leave it blank to disable admin access. This is a single account ID, not an email
allowlist; only server configuration can grant the role.

The owner sees **Admin dashboard** in the workspace header (`#/admin`). It lists
all users' jobs, current stages and finished reels, with playback and downloads.
It refreshes every five seconds and paginates in groups of 50. The admin view is
read-only; retry controls remain in each user's own workspace. All `/admin/*`
endpoints, including video GET/HEAD requests, enforce this role on the server.
Ordinary `/jobs/*` routes continue to enforce account ownership for everyone.

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
- One worker holds an OS flock; pipeline children inherit the lock. Following a
  restart the worker waits for any surviving child, then requeues interrupted jobs
  and resumes through existing cache checkpoints.
- All finished, evaluated reels enter the manifest, including `skip` verdicts.
  Quality verdicts are advisory and shown to users beside playback and download.
  Partial retries reuse saved
  selection bounds and render missing clips, preserving completed results.
- Full-source Scribe chunks have a separate bounded repair budget
  (`transcribe.scribe.source_repair_max_windows`, default 64). The short reel
  caption budget remains eight windows. After retries, small unresolved source
  gaps are excluded from podcast/comedy clip selection with surrounding context;
  successful repairs are retained. Unresolved speech above 5% of a source chunk
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
