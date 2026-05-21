"""Render one cached reel's segment with --subtitle-style pop for visual review."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from podclipper.config import load_default_config
from podclipper.subtitles import burn_subtitles
from podclipper.types import Word


def main() -> int:
    cache_dir = Path(".cache/yt_9FtradY1AI4_first20-cd9eb04f85/reel_01_confidence-is-an-output-not-input")
    segment = cache_dir / "cropped.mp4"
    words_json = cache_dir / "words.json"
    out_path = Path("outputs/pop_sample.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not segment.exists() or not words_json.exists():
        print(f"missing inputs under {cache_dir}", file=sys.stderr)
        return 2

    words = [Word(**w) for w in json.loads(words_json.read_text())]
    cfg = load_default_config()
    cfg.subtitles.style = "pop"
    cfg.subtitles.fade_out_seconds = 0.0

    burn_subtitles(segment, words, out_path, cfg, title="")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
