"""Test scraping Myinstants for popular meme/SFX audio.

Steps:
  1. Fetch the homepage (which lists trending/popular instants)
  2. Extract: name, page URL, mp3 URL, tags (from each detail page if needed)
  3. Download the top N mp3 files to dev/test_audio/downloads/sfx/
  4. Report: scrape success rate, download success rate, average file size
"""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "Mozilla/5.0 (PodClipper-test/0.1)"
TOP_N = 10
DOWNLOAD_DIR = Path(__file__).parent / "downloads" / "sfx"


def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read()


def scrape_homepage() -> list[dict]:
    """Return list of {name, page_url, mp3_url} from the homepage trending grid.

    Current (May 2026) layout:
      <button class="small-button"
              onclick="play('/media/sounds/xxx.mp3', 'loader-NNN', 'xxx-NNN')"
              title="Play NAME sound" type="button"></button>
    """
    html = http_get("https://www.myinstants.com/").decode("utf-8", errors="replace")

    # Each play button gives us mp3_url + the instant slug-id.
    # Title attribute gives us the display name.
    pattern = re.compile(
        r"onclick=\"play\('([^']+)'(?:,\s*'[^']*')*,\s*'([^']+)'\)\"\s*"
        r'title="Play\s+(.+?)\s+sound"',
        re.S,
    )
    items = []
    seen_slugs = set()
    for m in pattern.finditer(html):
        if len(items) >= TOP_N:
            break
        mp3_path, slug, raw_name = m.groups()
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        mp3_url = f"https://www.myinstants.com{mp3_path}" if mp3_path.startswith("/") else mp3_path
        page_url = f"https://www.myinstants.com/en/instant/{slug}/"
        items.append({"name": raw_name.strip(), "page_url": page_url, "mp3_url": mp3_url})
    return items


def fetch_tags_from_page(page_url: str) -> list[str]:
    """Page detail has <a class="tag-button">tag-name</a> for each tag."""
    try:
        html = http_get(page_url).decode("utf-8", errors="replace")
        tags = re.findall(r'<a[^>]*class="[^"]*\btag-button\b[^"]*"[^>]*>([^<]+)</a>', html)
        return [t.strip() for t in tags]
    except Exception:
        return []


def download(url: str, dest: Path) -> tuple[bool, int]:
    try:
        data = http_get(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True, len(data)
    except Exception as e:
        print(f"    download failed: {e}")
        return False, 0


def safe_filename(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower()
    return s[:60] or "sound"


def main() -> None:
    print(f"=== Myinstants scrape: top {TOP_N} ===")
    t_scrape = time.time()
    items = scrape_homepage()
    print(f"  scraped {len(items)} items in {time.time() - t_scrape:.2f}s")
    if not items:
        print("  no items extracted — page structure may have changed")
        return

    total_bytes = 0
    success = 0
    for it in items:
        print(f"\n  {it['name'][:50]:50}  {it['mp3_url']}")
        tags = fetch_tags_from_page(it["page_url"])
        print(f"    tags: {tags}")
        ext = ".mp3" if it["mp3_url"].endswith(".mp3") else ".wav"
        dest = DOWNLOAD_DIR / (safe_filename(it["name"]) + ext)
        ok, size = download(it["mp3_url"], dest)
        if ok:
            success += 1
            total_bytes += size
            print(f"    ✓ downloaded {size/1024:.0f} KB → {dest.relative_to(Path.cwd()) if dest.is_relative_to(Path.cwd()) else dest}")
        else:
            print(f"    ✗ download failed")

    print(f"\n=== SUMMARY ===")
    print(f"  download success: {success}/{len(items)}")
    print(f"  total downloaded: {total_bytes/1024/1024:.2f} MB")
    print(f"  avg file size:    {total_bytes / max(success, 1) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
