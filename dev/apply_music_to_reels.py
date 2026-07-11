#!/usr/bin/env python3
"""Lay LLM-matched background music under already-rendered reels.

Uses the music library (music/library.json): for each reel, matches a track by
transcript+title, then mixes a ducked bed. Reel transcripts are pulled from the
per-clip words.json cache.

Usage: python dev/apply_music_to_reels.py <reels_dir> [--gain 0.32]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from podclipper.config import load_default_config
from podclipper.llm import build_provider
from podclipper.main import _load_dotenv
from podclipper.music import load_library, mix_music, select_track

_load_dotenv()


def _title(sidecar: Path, stem: str) -> str:
    if sidecar.exists():
        for line in sidecar.read_text().splitlines():
            if line.startswith("title:"):
                return line.split(":", 1)[1].strip()
    return stem.replace("_", " ")


def _transcript(stem: str) -> str:
    hits = list(Path(".cache").glob(f"*/{stem}/words.json"))
    if not hits:
        return ""
    words = json.loads(hits[0].read_text())
    return " ".join((w.get("text") or "").strip() for w in words)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reels_dir", type=Path)
    ap.add_argument("--gain", type=float, default=None)
    a = ap.parse_args()

    cfg = load_default_config()
    cfg.music.enabled = True
    if a.gain is not None:
        cfg.music.gain = a.gain
    provider = build_provider(cfg.llm)
    lib = load_library(Path(cfg.music.library_path))
    if lib is None:
        print("no music library"); return 1

    out_dir = a.reels_dir.parent / f"{a.reels_dir.name}_music"
    out_dir.mkdir(parents=True, exist_ok=True)

    reels = sorted(p for p in a.reels_dir.glob("reel_*.mp4"))
    print(f"applying music to {len(reels)} reels -> {out_dir}\n")
    for mp4 in reels:
        stem = mp4.stem
        title = _title(a.reels_dir / f"{stem}.txt", stem)
        transcript = _transcript(stem)
        if not transcript:
            print(f"  {stem}: no transcript cache — skipped"); continue
        track = select_track(title, transcript, lib, provider, cfg)
        out = out_dir / f"{stem}_music.mp4"
        mix_music(mp4, track, out, cfg)
        print(f"  {title[:38]:40s} -> {track['song_id']}/{track['section_id']}")
    print(f"\ndone -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
