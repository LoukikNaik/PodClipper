#!/usr/bin/env python3
"""Post a reel via instagrapi's private mobile API.

Replaces the fragile CDP/web-UI path. The caption is just a string param
on clip_upload — Instagram's web contenteditable React state mess is
sidestepped entirely.

Usage:
    python post_via_instagrapi.py outputs/<run>/reel_NN_<slug>.mp4
    python post_via_instagrapi.py <reel.mp4> --dry-run
    python post_via_instagrapi.py <reel.mp4> --caption "custom caption"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.llm import LLMError, build_provider
from src.logging_util import setup_logging


SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_ROOT = SCRIPT_DIR / ".cache"
SESSION_PATH = CACHE_ROOT / "instagrapi_session.json"
MAX_HASHTAGS = 15
INSTAGRAM_CAPTION_LIMIT = 2200


CAPTION_SYSTEM = """You write Instagram Reel captions and hashtags for short vertical clips cut from podcast videos.

You will receive the reel's title, the editor's note about why it was picked, and the transcript of what is actually said in the clip.

Output STRICT JSON with exactly two fields:
  {
    "caption": "<1-3 short lines, hook-style, no hashtags inside>",
    "hashtags": ["#one", "#two", ...]
  }

Rules:
- Caption is 1-3 lines. Newlines separate lines. No hashtags inside the caption — they go in the array.
- Open with a scroll-stopping hook grounded in what was ACTUALLY said. Address the viewer ("you/your") when natural.
- Do not paraphrase the whole clip; tease the payoff, leave the answer in the video.
- 8-15 hashtags. Mix broad (#mindset) with niche (#stoicfounder). Lowercase. No spaces. No duplicates.
- Hashtags must be relevant to the transcript content, not generic filler.
- No emojis unless the transcript itself has a strongly emotional beat that justifies one.
- Output JSON ONLY. No prose, no code fence, no trailing comma.
"""


def _read_sidecar(path: Path) -> dict:
    if not path.exists():
        return {}
    out: dict = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        out[key.strip()] = val.strip()
    return out


def _locate_words_json(reel_slug: str) -> Optional[Path]:
    """Find a reel's cached Whisper transcript anywhere under .cache/.

    Reels keep their slug into the cache verbatim (see pipeline.py), so
    one glob lands the right file without needing the source-video hash.
    If multiple matches exist (same slug across runs) pick the newest.
    """
    if not CACHE_ROOT.exists():
        return None
    matches = sorted(
        CACHE_ROOT.glob(f"*/{reel_slug}/words.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _transcript_from_words(path: Path) -> str:
    words = json.loads(path.read_text())
    return "".join(w.get("text", "") for w in words if isinstance(w, dict)).strip()


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    first, last = text.find("{"), text.rfind("}")
    if first == -1 or last == -1 or last < first:
        raise ValueError(f"no JSON object in LLM output: {text[:200]!r}")
    return json.loads(text[first : last + 1])


def _sanitize_hashtag(tag: str) -> str:
    tag = tag.strip()
    if not tag.startswith("#"):
        tag = "#" + tag
    tag = "#" + re.sub(r"[^A-Za-z0-9_]", "", tag[1:])
    return tag if len(tag) > 1 else ""


def _format_caption(caption: str, hashtags: list[str]) -> str:
    seen: set[str] = set()
    deduped: list[str] = []
    for h in hashtags:
        clean = _sanitize_hashtag(h)
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            deduped.append(clean)
    deduped = deduped[:MAX_HASHTAGS]
    full = caption.strip()
    if deduped:
        full = f"{full}\n\n{' '.join(deduped)}"
    return full[:INSTAGRAM_CAPTION_LIMIT]


def generate_caption(reel_path: Path, cfg, log) -> str:
    """Build caption + hashtags from the reel's sidecar + cached transcript."""
    slug = reel_path.stem
    meta = _read_sidecar(reel_path.with_suffix(".txt"))
    title = meta.get("title") or slug
    reason = meta.get("reason", "")

    words_path = _locate_words_json(slug)
    if words_path is None:
        log.warning(f"no words.json found for slug {slug!r}; caption will lean on title alone")
        transcript = ""
    else:
        transcript = _transcript_from_words(words_path)
        log.info(f"transcript loaded from {words_path.relative_to(SCRIPT_DIR)} ({len(transcript)} chars)")

    user_prompt = (
        f"Title: {title}\n"
        f"Editor's note: {reason}\n\n"
        f"Transcript:\n{transcript or '(transcript unavailable — work from the title)'}\n"
    )
    provider = build_provider(cfg.llm)
    log.info(f"asking {provider.name} for caption + hashtags...")
    try:
        response = provider.complete(
            user_prompt=user_prompt,
            system_prompt=CAPTION_SYSTEM,
            max_tokens=600,
        )
    except LLMError as e:
        log.error(f"LLM call failed: {e}")
        raise

    parsed = _extract_json_object(response)
    caption = str(parsed.get("caption", "")).strip()
    hashtags = parsed.get("hashtags", []) or []
    if not isinstance(hashtags, list):
        hashtags = []
    if not caption:
        raise ValueError(f"LLM returned empty caption; raw: {response[:300]!r}")
    return _format_caption(caption, [str(h) for h in hashtags])


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip(); value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _login(log):
    from instagrapi import Client
    user = os.environ.get("INSTAGRAM_USER")
    pw = os.environ.get("INSTAGRAM_PASS")
    if not user or not pw:
        raise SystemExit("Set INSTAGRAM_USER and INSTAGRAM_PASS in .env")

    cl = Client()
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)

    if SESSION_PATH.exists():
        # Restore prior session — avoids re-triggering email/2FA challenges
        # on every run. instagrapi will silently re-login if the cookies expired.
        try:
            cl.load_settings(SESSION_PATH)
            cl.login(user, pw)
            cl.get_timeline_feed()  # cheap auth check
            log.info(f"reused session: {SESSION_PATH.relative_to(SCRIPT_DIR)}")
            return cl
        except Exception as e:
            log.warning(f"saved session invalid ({e}); fresh login")
            cl = Client()

    cl.login(user, pw)
    cl.dump_settings(SESSION_PATH)
    log.info(f"fresh login complete; session saved to {SESSION_PATH.relative_to(SCRIPT_DIR)}")
    return cl


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Post a reel via instagrapi (Instagram's private mobile API)."
    )
    ap.add_argument("reel", type=Path)
    ap.add_argument("-c", "--config", type=Path, default=Path("config/default.yaml"))
    ap.add_argument("--caption", default=None,
                    help="Skip the LLM and post this caption verbatim")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the caption and exit without uploading")
    ap.add_argument("--trial", action="store_true",
                    help="Post as a Trial Reel — only shown to non-followers initially, "
                         "hidden from your grid until you graduate it manually")
    args = ap.parse_args()

    _load_dotenv()
    log = setup_logging("INFO").getChild("ig")

    if not args.reel.exists():
        log.error(f"reel not found: {args.reel}")
        return 2

    cfg = load_config(args.config)
    caption = args.caption or generate_caption(args.reel, cfg, log)

    print("\n----- caption -----")
    print(caption)
    print("-------------------\n")

    if args.dry_run:
        return 0

    cl = _login(log)

    if args.trial:
        # Server-side gate — Instagram only enables trial reels for some accounts.
        # Failing fast here is cleaner than catching a configure error mid-upload.
        if not cl.clip_trial_eligible():
            log.error(
                "this account isn't currently eligible for Trial Reels per IG's preflight; "
                "drop --trial or wait for the feature to roll out"
            )
            return 1
        log.info("posting as TRIAL reel (graduation strategy: manual)")

    log.info(f"uploading reel: {args.reel.resolve()}")
    media = cl.clip_upload(
        path=args.reel.resolve(),
        caption=caption,
        trial=args.trial,
    )
    log.info(f"posted! code={media.code} pk={media.pk} url=https://www.instagram.com/reel/{media.code}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
