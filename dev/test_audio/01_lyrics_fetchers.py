"""Test lyrics fetching approaches.

Runs each fetcher against 5 reference songs and reports hit/miss + latency.

Fetchers tested:
  - lrclib.net  (free, no auth, plain + synced lyrics)
  - AZ Lyrics scrape  (free, no auth, often blocked)
  - DuckDuckGo HTML search → lyrics page scrape (free fallback)

Skipped (no creds available locally):
  - Genius API (needs OAuth token)
  - Firecrawl (paid API key)
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

SONGS = [
    # (artist, title, expected_keyword_in_lyrics)
    ("Phoebe Bridgers", "Funny",          "not crying"),
    ("Lana Del Rey",    "Video Games",    "playing video games"),
    ("Sabrina Carpenter", "Espresso",     "espresso"),
    ("Hannah Cohen",    "Watching You Fall", "watching"),     # indie / less mainstream
    ("Bad Bunny",       "Monáco",    "monaco"),           # Spanish-language
]

USER_AGENT = "Mozilla/5.0 (PodClipper-test/0.1)"


@dataclass
class Result:
    fetcher: str
    artist: str
    title: str
    hit: bool
    latency_s: float
    lyrics_excerpt: str = ""
    error: str = ""


def _http_get(url: str, headers: dict | None = None, timeout: int = 10) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", errors="replace")


def fetch_lrclib(artist: str, title: str) -> Result:
    t0 = time.time()
    try:
        q = urllib.parse.urlencode({"artist_name": artist, "track_name": title})
        status, body = _http_get(f"https://lrclib.net/api/search?{q}")
        import json
        data = json.loads(body)
        if not data:
            return Result("lrclib", artist, title, False, time.time() - t0, error="no match")
        top = data[0]
        lyrics = top.get("plainLyrics") or top.get("syncedLyrics") or ""
        return Result(
            "lrclib", artist, title,
            hit=bool(lyrics.strip()),
            latency_s=time.time() - t0,
            lyrics_excerpt=lyrics[:200],
        )
    except Exception as e:
        return Result("lrclib", artist, title, False, time.time() - t0, error=str(e)[:100])


def fetch_azlyrics(artist: str, title: str) -> Result:
    """AZ Lyrics URL pattern: /lyrics/<artistslug>/<titleslug>.html (slugify both)."""
    t0 = time.time()
    try:
        def slug(s: str) -> str:
            return "".join(c.lower() for c in s if c.isalnum())
        url = f"https://www.azlyrics.com/lyrics/{slug(artist)}/{slug(title)}.html"
        status, body = _http_get(url)
        # AZL lyrics live in a <div> right after a long HTML comment "<!-- Usage of azlyrics..."
        import re
        m = re.search(
            r'<!-- Usage of azlyrics\.com.*?-->\s*<div>(.*?)</div>',
            body, re.S,
        )
        if not m:
            return Result("azlyrics", artist, title, False, time.time() - t0, error="lyrics div not found")
        raw = m.group(1)
        text = re.sub(r"<.*?>", "", raw).strip()
        return Result(
            "azlyrics", artist, title,
            hit=bool(text),
            latency_s=time.time() - t0,
            lyrics_excerpt=text[:200],
        )
    except urllib.error.HTTPError as e:
        return Result("azlyrics", artist, title, False, time.time() - t0, error=f"HTTP {e.code}")
    except Exception as e:
        return Result("azlyrics", artist, title, False, time.time() - t0, error=str(e)[:100])


def fetch_duckduckgo(artist: str, title: str) -> Result:
    """Use DDG's HTML endpoint to find a lyrics page, then scrape any /lyrics/ result."""
    t0 = time.time()
    try:
        q = urllib.parse.quote_plus(f"{artist} {title} lyrics")
        status, body = _http_get(f"https://html.duckduckgo.com/html/?q={q}")
        import re
        # extract result URLs from the markup
        urls = re.findall(r'href="(https?://[^"]+lyrics[^"]+)"', body)
        if not urls:
            return Result("ddg", artist, title, False, time.time() - t0, error="no lyrics URL in DDG results")
        # Try first URL — fetch and grep for the artist name as a sanity signal
        target = urls[0]
        try:
            _, lyrics_body = _http_get(target)
            # super-naive: take longest plaintext block
            text = re.sub(r"<.*?>", " ", lyrics_body)
            text = re.sub(r"\s+", " ", text)
            return Result(
                "ddg", artist, title,
                hit=artist.split()[0].lower() in text.lower(),
                latency_s=time.time() - t0,
                lyrics_excerpt=text[:200],
            )
        except Exception as e:
            return Result("ddg", artist, title, False, time.time() - t0, error=f"target fetch: {e}"[:100])
    except Exception as e:
        return Result("ddg", artist, title, False, time.time() - t0, error=str(e)[:100])


def main() -> None:
    fetchers = [
        ("lrclib", fetch_lrclib),
        ("azlyrics", fetch_azlyrics),
        ("ddg", fetch_duckduckgo),
    ]
    results: list[Result] = []
    for artist, title, _ in SONGS:
        for name, fn in fetchers:
            r = fn(artist, title)
            results.append(r)
            mark = "✓" if r.hit else "✗"
            extra = r.lyrics_excerpt[:60].replace("\n", " ") if r.hit else r.error
            print(f"  {mark} {name:10} {artist:22} | {title:24} | {r.latency_s:5.2f}s | {extra}")

    print("\n=== HIT RATE BY FETCHER ===")
    by_fetcher: dict[str, list[Result]] = {}
    for r in results:
        by_fetcher.setdefault(r.fetcher, []).append(r)
    for name, rs in by_fetcher.items():
        hits = sum(r.hit for r in rs)
        mean_latency = sum(r.latency_s for r in rs) / len(rs)
        print(f"  {name:10}: {hits}/{len(rs)} hits, avg latency {mean_latency:.2f}s")


if __name__ == "__main__":
    main()
