#!/usr/bin/env python3
"""Find a song's hook section from a lyrics transcript: the most-repeated line.

Combines lyrics (the hook is the line sung most often), acoustic energy (pick the
fullest occurrence), and the beat grid (snap boundaries). Fuzzy line clustering
tolerates a rough transcript where the same hook transcribes slightly differently
each time.

Usage: python dev/find_hook.py <transcript.json> <audio> [--min 14] [--max 26]
transcript.json: {"segments":[{"start","end","text"}, ...]}
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import scipy.signal
import scipy.signal.windows
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

import librosa
import numpy as np


def _norm(text: str) -> str:
    """Lowercase, strip combining marks/punctuation — script-agnostic skeleton."""
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join("".join(c for c in w if c.isalnum()) for w in t.split()).strip()


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def find_hook(transcript: dict, audio: Path, length: float = 50.0,
              hook_phrase: str | None = None) -> list[dict]:
    lines = [{"start": s["start"], "end": s["end"], "norm": _norm(s["text"]),
              "text": s["text"]}
             for s in transcript["segments"] if _norm(s["text"]).count(" ") >= 1]

    if hook_phrase:
        # Known hook from web lyrics: match each ASR line against it (low
        # threshold tolerates a garbled sung transcript).
        hp = _norm(hook_phrase)
        hook_idxs = [i for i, ln in enumerate(lines) if _similar(ln["norm"], hp) >= 0.35]
        hook_text = hook_phrase
    else:
        # Cluster lines by fuzzy similarity; the biggest cluster is the hook.
        clusters: list[list[int]] = []
        for i, ln in enumerate(lines):
            for c in clusters:
                if _similar(ln["norm"], lines[c[0]]["norm"]) >= 0.6:
                    c.append(i)
                    break
            else:
                clusters.append([i])
        clusters.sort(key=len, reverse=True)
        hook_idxs = clusters[0] if clusters else []
        hook_text = lines[hook_idxs[0]]["text"] if hook_idxs else ""
    if len(hook_idxs) < 1:
        return []

    y, sr = librosa.load(str(audio), sr=22050, mono=True)
    rms = librosa.feature.rms(y=y)[0]
    rt = librosa.times_like(rms, sr=sr)
    _, beats = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    bt = librosa.frames_to_time(beats, sr=sr)

    def energy(a, b):
        m = (rt >= a) & (rt <= b)
        return float(rms[m].mean()) if m.any() else 0.0

    def snap(t):
        return round(float(bt[np.argmin(np.abs(bt - t))]), 2) if bt.size else round(t, 2)

    # Rank hook occurrences by loudness (the fullest chorus); each section STARTS
    # on the hook (impactful entry) and runs forward `length` s of runway.
    song_dur = len(y) / sr
    occ = sorted(hook_idxs, key=lambda i: energy(lines[i]["start"], lines[i]["end"]),
                 reverse=True)
    sections = []
    used = []
    for i in occ:
        start = snap(lines[i]["start"])
        end = round(min(song_dur, start + length), 2)
        if any(abs(start - u) < length * 0.5 for u in used):
            continue
        used.append(start)
        sections.append({"start": start, "end": end,
                         "energy_db": round(20 * np.log10(energy(start, end) + 1e-9), 1),
                         "hook_text": hook_text})
        if len(sections) >= 2:
            break
    sections.sort(key=lambda s: s["start"])
    return sections


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", type=Path)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--length", type=float, default=50.0,
                    help="section runway in seconds, starting on the hook")
    ap.add_argument("--hook-phrase", default=None,
                    help="known hook line (e.g. from web lyrics) to match against")
    a = ap.parse_args()
    tr = json.loads(a.transcript.read_text())
    secs = find_hook(tr, a.audio, a.length, a.hook_phrase)
    if not secs:
        print("no repeated hook line found")
        return 0
    print(f"hook line: {secs[0]['hook_text']!r}")
    for s in secs:
        print(f"  {s['start']:.1f}-{s['end']:.1f}s  E{s['energy_db']}dB")
    print(json.dumps(secs, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
