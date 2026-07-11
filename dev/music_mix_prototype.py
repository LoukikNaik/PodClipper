#!/usr/bin/env python3
"""Prototype: mix a ducked music bed under a rendered reel's speech.

Usage: python dev/music_mix_prototype.py <reel.mp4> <music.(mp3|wav)> <out.mp4>
       [--music-start SEC] [--music-gain 0..1] [--duck-ratio N] [--fade SEC]

Ducking: the reel's speech sidechains a compressor on the music, so the bed
drops while the speaker talks and swells in the gaps. Speech loudness is
preserved (amix normalize=0).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def mix(reel: Path, music: Path, out: Path, music_start: float,
        music_gain: float, duck_ratio: float, fade: float) -> Path:
    dur = _duration(reel)
    fade_out_start = max(0.0, dur - fade)

    filt = (
        f"[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
        f"afade=t=in:st=0:d={fade:.2f},afade=t=out:st={fade_out_start:.2f}:d={fade:.2f},"
        f"volume={music_gain}[musraw];"
        f"[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
        f"asplit=2[sp][sc];"
        f"[musraw][sc]sidechaincompress=threshold=0.03:ratio={duck_ratio}:"
        f"attack=5:release=250[mus];"
        f"[sp][mus]amix=inputs=2:duration=first:normalize=0[aout]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(reel),
        "-ss", f"{music_start:.3f}", "-t", f"{dur:.3f}", "-i", str(music),
        "-filter_complex", filt,
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("reel", type=Path)
    p.add_argument("music", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--music-start", type=float, default=0.0)
    p.add_argument("--music-gain", type=float, default=0.40)
    p.add_argument("--duck-ratio", type=float, default=8.0)
    p.add_argument("--fade", type=float, default=0.8)
    a = p.parse_args()
    out = mix(a.reel, a.music, a.out, a.music_start, a.music_gain, a.duck_ratio, a.fade)
    print(f"wrote {out} ({_duration(out):.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
