"""Test scraping trending song lists from public sources.

Tokboard.com is dead (DNS doesn't resolve). Trying public chart alternatives:

  1. Apple Music's Top 100 charts (public web page, no auth)
  2. Billboard Hot 100 (public web page, no auth) — less viral, more mainstream
  3. Spotify Charts (the Top Songs page, no auth needed for web)
  4. iTunes RSS feed (XML, public)

We want artist + title pairs we can feed to the lrclib + LLM tagger.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request

USER_AGENT = "Mozilla/5.0 (PodClipper-test/0.1)"


def http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read().decode("utf-8", errors="replace")


def fetch_itunes_rss_top_songs(country: str = "us", limit: int = 25) -> list[tuple[str, str]]:
    """Apple/iTunes provides public RSS feeds for charts. JSON variant is easier."""
    url = f"https://rss.applemarketingtools.com/api/v2/{country}/music/most-played/{limit}/songs.json"
    body = http_get(url)
    data = json.loads(body)
    out = []
    for item in data.get("feed", {}).get("results", []):
        out.append((item["artistName"], item["name"]))
    return out


def fetch_billboard_hot_100() -> list[tuple[str, str]]:
    """Scrape billboard.com/charts/hot-100 — public HTML."""
    body = http_get("https://www.billboard.com/charts/hot-100/")
    # Billboard renders song rows with a structure like:
    #   <h3 ... class="c-title">SONG NAME</h3>
    #   <span ... class="c-label">ARTIST</span>
    # Highly dynamic; parsing is fragile. Try a sample regex.
    titles = re.findall(
        r'<h3[^>]*class="[^"]*c-title[^"]*"[^>]*>\s*([^<]+?)\s*</h3>',
        body, re.S,
    )
    artists = re.findall(
        r'<span[^>]*class="[^"]*c-label[^"]*"[^>]*>\s*([^<]+?)\s*</span>',
        body, re.S,
    )
    pairs = []
    # The first ~100 of each list correspond to chart entries
    for t, a in zip(titles[:25], artists[:25]):
        t_clean = re.sub(r"\s+", " ", t).strip()
        a_clean = re.sub(r"\s+", " ", a).strip()
        if t_clean and a_clean and not a_clean.isdigit():
            pairs.append((a_clean, t_clean))
    return pairs


def main() -> None:
    print("=== Apple/iTunes Top Songs (US, JSON RSS) ===")
    t0 = time.time()
    try:
        songs = fetch_itunes_rss_top_songs("us", 15)
        print(f"  fetched {len(songs)} songs in {time.time() - t0:.2f}s")
        for i, (a, t) in enumerate(songs[:15], 1):
            print(f"  {i:2}. {a:30} — {t}")
    except Exception as e:
        print(f"  failed: {e}")

    print("\n=== Billboard Hot 100 (HTML scrape) ===")
    t0 = time.time()
    try:
        songs = fetch_billboard_hot_100()
        print(f"  fetched {len(songs)} songs in {time.time() - t0:.2f}s")
        for i, (a, t) in enumerate(songs[:15], 1):
            print(f"  {i:2}. {a[:40]:40} — {t[:50]}")
    except Exception as e:
        print(f"  failed: {e}")


if __name__ == "__main__":
    main()
