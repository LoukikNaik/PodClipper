"""Scene segmentation: ffmpeg scene-score cut detection + the cut→scene
boundary algorithm. The cut-detection call is IO; the parsing and the
boundary math are pure and tested."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

DEFAULT_THRESHOLD = 0.25
DEFAULT_MIN_SCENE = 1.5
DEFAULT_MAX_SCENE = 8.0


def parse_showinfo(stderr: str) -> list[float]:
    """Extract sorted scene-change timestamps from ffmpeg showinfo stderr."""
    return sorted(float(m) for m in re.findall(r"pts_time:([0-9.]+)", stderr))


def detect_cuts(
    path: Path, threshold: float = DEFAULT_THRESHOLD, stop_at: float | None = None,
) -> list[float]:
    """Run ffmpeg's scene-score filter and return cut timestamps (seconds).
    `stop_at` limits decoding to the first N seconds (huge speedup when only a
    slice is being indexed — ffmpeg otherwise decodes the whole file)."""
    cmd = ["ffmpeg"]
    if stop_at is not None:
        cmd += ["-t", f"{stop_at:.3f}"]
    cmd += ["-i", str(path), "-filter:v",
            f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return parse_showinfo(proc.stderr)


def words_in_range(words: list[dict], start: float, end: float) -> str:
    """Join the text of words whose midpoint falls inside [start, end)."""
    out = [w["text"] for w in words
           if start <= (float(w["start"]) + float(w["end"])) / 2 < end]
    return "".join(out).strip()


def cuts_to_scenes(
    cuts: list[float],
    duration: float,
    min_len: float = DEFAULT_MIN_SCENE,
    max_len: float = DEFAULT_MAX_SCENE,
) -> list[tuple[float, float]]:
    """Turn cut timestamps into [start, end) scene ranges: merge scenes shorter
    than `min_len` into the previous one, then split any scene longer than
    `max_len` into even sub-windows (keeps captions frame-distinct even when cut
    detection finds nothing — slow fades, screen recordings, locked cameras)."""
    bounds = sorted(set([0.0] + [c for c in cuts if 0.0 < c < duration] + [duration]))
    merged: list[list[float]] = []
    for s, e in zip(bounds, bounds[1:]):
        if merged and (e - s) < min_len:
            merged[-1][1] = e          # absorb a too-short scene into the previous
        else:
            merged.append([s, e])
    # a too-short leading scene has no previous to merge into — fold it forward
    if len(merged) > 1 and (merged[0][1] - merged[0][0]) < min_len:
        merged[1][0] = merged[0][0]
        merged.pop(0)

    scenes: list[tuple[float, float]] = []
    for s, e in merged:
        span = e - s
        if span <= max_len:
            scenes.append((s, e))
            continue
        n = int(span // max_len) + 1
        step = span / n
        for k in range(n):
            scenes.append((s + k * step, s + (k + 1) * step if k < n - 1 else e))
    return scenes
