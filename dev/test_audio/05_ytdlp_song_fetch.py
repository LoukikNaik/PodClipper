"""Test yt-dlp for fetching song audio by artist+title.

Uses `ytsearch1:<artist> <title> audio` query, downloads bestaudio,
re-encodes to mp3 ~128 kbps. Measures success rate, time, file size.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

SONGS = [
    ("Lana Del Rey",    "Video Games"),
    ("Sabrina Carpenter", "Espresso"),
    ("Bad Bunny",       "Monáco"),
    ("Phoebe Bridgers", "Funny"),
    ("Some Indie Band That Doesnt Exist", "ImaginarySong"),  # negative test
]

DOWNLOAD_DIR = Path(__file__).parent / "downloads" / "songs"


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")[:60].lower()


def _run_ytdlp(query: str, out_template: str) -> tuple[int, str]:
    """Single yt-dlp invocation. Returns (returncode, stderr)."""
    result = subprocess.run(
        [
            "yt-dlp", "-x",
            "--audio-format", "mp3",
            "--audio-quality", "5",
            "-o", out_template,
            "--no-playlist", "--quiet",
            query,
        ],
        capture_output=True, text=True, timeout=120,
    )
    return result.returncode, (result.stderr or result.stdout)


def fetch_one(artist: str, title: str) -> dict:
    """Try SoundCloud first (no bot-check), fall back to YouTube."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stem = safe_filename(f"{artist}_{title}")
    out_template = str(DOWNLOAD_DIR / f"{stem}.%(ext)s")

    sources = [
        ("soundcloud", f"scsearch1:{artist} {title}"),
        ("youtube",    f"ytsearch1:{artist} {title} audio"),
    ]

    t0 = time.time()
    last_err = ""
    for src_name, query in sources:
        try:
            rc, err = _run_ytdlp(query, out_template)
        except subprocess.TimeoutExpired:
            last_err = f"{src_name}: timeout"
            continue
        if rc == 0:
            files = list(DOWNLOAD_DIR.glob(f"{stem}.*"))
            if files:
                f = files[0]
                return {
                    "artist": artist, "title": title, "ok": True,
                    "source": src_name,
                    "path": str(f), "size_mb": f.stat().st_size / 1024 / 1024,
                    "t_s": time.time() - t0,
                }
        last_err = f"{src_name}: {err[:150]}"

    return {
        "artist": artist, "title": title, "ok": False,
        "err": last_err, "t_s": time.time() - t0,
    }


def main() -> None:
    if not shutil.which("yt-dlp"):
        print("yt-dlp not found on PATH — install with `pip install yt-dlp` or `brew install yt-dlp`")
        return

    print(f"=== yt-dlp song fetch ({len(SONGS)} tracks) ===\n")
    results = []
    for artist, title in SONGS:
        print(f"  fetching: {artist} — {title}")
        r = fetch_one(artist, title)
        results.append(r)
        if r["ok"]:
            print(f"    ✓ {r['t_s']:.1f}s, {r['size_mb']:.2f} MB via {r['source']:10} → {Path(r['path']).name}")
        else:
            print(f"    ✗ {r.get('t_s', 0):.1f}s — {r['err'][:120]}")

    print("\n=== SUMMARY ===")
    ok = [r for r in results if r["ok"]]
    print(f"  success: {len(ok)}/{len(results)}")
    if ok:
        print(f"  avg time: {sum(r['t_s'] for r in ok)/len(ok):.1f}s/track")
        print(f"  avg size: {sum(r['size_mb'] for r in ok)/len(ok):.2f} MB/track")


if __name__ == "__main__":
    main()
