"""Test LLM song tagging via lrclib lyrics → TokenRouter (Claude).

For each test song:
  1. Fetch lyrics from lrclib (the winner from test 01)
  2. Send (artist, title, lyrics) to the LLM
  3. Ask for structured JSON: mood, themes, energy, use_cases, avoid_for,
     viral_context, iconic_lyric

Measure: latency per call, JSON validity, semantic quality (eyeball).
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

# Load .env into os.environ so TOKENROUTER_API_KEY is available
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from podclipper.main import _load_dotenv
_load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from podclipper.config import load_default_config
from podclipper.llm import build_provider


SONGS = [
    ("Lana Del Rey",    "Video Games"),
    ("Sabrina Carpenter", "Espresso"),
    ("Bad Bunny",       "Monáco"),
    ("Phoebe Bridgers", "Funny"),
]

SYSTEM_PROMPT = """You analyze song lyrics to determine what kinds of short-form video
reels (TikTok / Instagram Reels / YouTube Shorts) this song would work well as a
background bed for.

Given the song's artist, title, and lyrics, return a single JSON object on one
line with these fields:

{
  "mood": ["one to three single-word mood tags, e.g. melancholic, upbeat, dramatic"],
  "themes": ["specific lyrical themes, e.g. heartbreak, nostalgia, self-deprecation, revenge"],
  "energy": "low | med | high",
  "use_cases": ["specific reel topics this would work UNDER, e.g. 'post-breakup montage', 'ironic detachment from grief'"],
  "avoid_for": ["reel topics where this song would clash, e.g. 'celebration', 'comedy punchline'"],
  "viral_context": "if you recognize this as a viral sound from TikTok/Reels/Shorts, describe how it's been used. Otherwise null.",
  "iconic_lyric": "the most quotable / hook line from the lyrics"
}

Return ONLY the JSON object, no prose, no code fence."""


def fetch_lrclib(artist: str, title: str) -> str | None:
    q = urllib.parse.urlencode({"artist_name": artist, "track_name": title})
    req = urllib.request.Request(
        f"https://lrclib.net/api/search?{q}",
        headers={"User-Agent": "PodClipper-test"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
    if not data:
        return None
    return data[0].get("plainLyrics") or data[0].get("syncedLyrics")


def main() -> None:
    cfg = load_default_config()
    cfg.llm.provider = "litellm"
    cfg.llm.model = "openai/anthropic/claude-sonnet-4.5"
    cfg.llm.litellm.api_base = "https://api.tokenrouter.com/v1"
    cfg.llm.litellm.api_key_env = "TOKENROUTER_API_KEY"
    cfg.llm.litellm.num_retries = 0
    provider = build_provider(cfg.llm)

    for artist, title in SONGS:
        print(f"\n--- {artist} — {title} ---")
        lyrics = fetch_lrclib(artist, title)
        if not lyrics:
            print("  no lyrics — skip")
            continue
        user_msg = f"Artist: {artist}\nTitle: {title}\n\nLyrics:\n{lyrics[:3000]}"
        t0 = time.time()
        try:
            raw = provider.complete(user_msg, system_prompt=SYSTEM_PROMPT, max_tokens=600)
        except Exception as e:
            print(f"  LLM error: {e}")
            continue
        latency = time.time() - t0

        # JSON validity check
        try:
            tags = json.loads(raw.strip().strip("`").lstrip("json\n"))
            print(f"  ✓ {latency:.1f}s")
            print(f"    mood:        {tags.get('mood')}")
            print(f"    energy:      {tags.get('energy')}")
            print(f"    themes:      {tags.get('themes')}")
            print(f"    use_cases:   {tags.get('use_cases')}")
            print(f"    avoid_for:   {tags.get('avoid_for')}")
            print(f"    viral:       {tags.get('viral_context')}")
            print(f"    iconic line: {(tags.get('iconic_lyric') or '')[:80]}")
        except json.JSONDecodeError as e:
            print(f"  ✗ {latency:.1f}s — JSON parse failed: {e}")
            print(f"    raw: {raw[:200]}")


if __name__ == "__main__":
    main()
