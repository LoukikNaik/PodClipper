"""Test scraping Tokboard for TikTok trending sound data.

Tokboard ranks TikTok sounds by usage count. We want artist + title +
TikTok URL (so yt-dlp can fetch later).
"""

from __future__ import annotations

import re
import time
import urllib.request

USER_AGENT = "Mozilla/5.0 (PodClipper-test/0.1)"


def http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read().decode("utf-8", errors="replace")


def main() -> None:
    print("=== Tokboard scrape: top trending TikTok sounds ===\n")
    t0 = time.time()
    try:
        html = http_get("https://tokboard.com/")
    except Exception as e:
        print(f"  fetch failed: {e}")
        return
    print(f"  fetched homepage in {time.time() - t0:.2f}s, {len(html)/1024:.0f} KB")

    # Tokboard's page has track rows with artist/title and links to TikTok sound
    # pages. Try several regex shapes since their HTML changes.
    # Pattern 1: looks for a track title + artist pair.
    candidates = []

    # Pattern A: <a href="/songs/<slug>"> followed by track text
    for m in re.finditer(
        r'<a[^>]*href="(/songs?/[^"]+)"[^>]*>\s*<[^>]*>([^<]+)<',
        html,
    ):
        candidates.append({"href": m.group(1), "snippet": m.group(2).strip()})

    # Pattern B: any `tiktok.com/music` links
    for m in re.finditer(
        r'href="(https?://(?:www\.)?tiktok\.com/music/[^"]+)"',
        html,
    ):
        candidates.append({"tiktok_url": m.group(1)})

    # Pattern C: any explicit "rank" / "title" / "artist" class names
    rows = re.findall(
        r'<tr[^>]*>.*?<td[^>]*>([^<]{2,60})</td>.*?<td[^>]*>([^<]{2,60})</td>',
        html, re.S,
    )

    print(f"  pattern A matches:        {len([c for c in candidates if 'href' in c])}")
    print(f"  pattern B matches (tt):   {len([c for c in candidates if 'tiktok_url' in c])}")
    print(f"  pattern C matches (rows): {len(rows)}")

    if not candidates and not rows:
        # Last resort: dump the first 2000 chars of the page so we can see what
        # the HTML actually looks like
        print("\n  No matches with any pattern. Page sample:")
        print(html[:2000].replace("\n", " ")[:1500])
        return

    print("\n=== Sample extracted candidates ===")
    for c in candidates[:8]:
        print(f"  {c}")
    if rows:
        print("\n=== Sample table rows (first 8) ===")
        for r in rows[:8]:
            print(f"  {r}")


if __name__ == "__main__":
    main()
