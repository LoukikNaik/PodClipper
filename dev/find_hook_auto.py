#!/usr/bin/env python3
"""Automatically find a vocal song's hook section and pin its onset precisely.

Recipe (the one that worked by hand on tera_naam):
  1. Transcribe the whole song with mlx-whisper (word timestamps).
  2. The most-REPEATED lyric line is the hook (not the loudest window).
  3. Pick the occurrence with the most energy behind it.
  4. Cut -> transcribe the clip -> measure where the hook's first word actually
     lands -> shift the start -> repeat. This defeats whole-song timestamp
     drift, which is multiple seconds on sung/produced audio.

Usage: python dev/find_hook_auto.py <audio> --language pa [--length 50]
       [--manifest music/library.json --id tera_naam]  (writes section if given)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import scipy.signal
import scipy.signal.windows
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

import librosa
import numpy as np
import mlx_whisper

MODEL = "mlx-community/whisper-large-v3-mlx"


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join("".join(c for c in w if c.isalnum()) for w in t.split()).strip()


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _transcribe(audio: Path, language: str) -> list[dict]:
    r = mlx_whisper.transcribe(str(audio), path_or_hf_repo=MODEL,
                               word_timestamps=True, language=language)
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip(),
             "words": [{"w": w["word"], "s": w["start"]} for w in s.get("words", [])]}
            for s in r["segments"]]


def _cut(audio: Path, start: float, dur: float, out: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{max(0, start):.3f}", "-t", f"{dur:.3f}",
                    "-i", str(audio), "-ar", "16000", "-ac", "1", str(out)],
                   check=True)
    return out


def find_hook_line(segs: list[dict]) -> tuple[str, str, list[dict]]:
    """Cluster lines by similarity; biggest cluster is the hook. Returns
    (hook_text, hook_norm, [occurrences sorted by time])."""
    lines = [dict(l, norm=_norm(l["text"])) for l in segs if _norm(l["text"]).count(" ") >= 2]
    clusters: list[list[int]] = []
    for i, ln in enumerate(lines):
        for c in clusters:
            if _sim(ln["norm"], lines[c[0]]["norm"]) >= 0.55:
                c.append(i)
                break
        else:
            clusters.append([i])
    clusters.sort(key=len, reverse=True)
    if not clusters or len(clusters[0]) < 2:
        return "", "", []
    hook = [lines[i] for i in clusters[0]]
    rep = hook[0]
    return rep["text"], rep["norm"], sorted(hook, key=lambda l: l["start"])


def _locate_phrase(words: list[dict], hook_norm: str) -> float | None:
    """Slide the hook phrase over the clip's words; return the onset of the
    best-matching window (robust to per-word spelling noise)."""
    k = max(2, len(hook_norm.split()))
    best_r, best_t = 0.0, None
    for i in range(len(words)):
        window = " ".join(_norm(w["w"]) for w in words[i:i + k])
        r = _sim(window, hook_norm)
        if r > best_r:
            best_r, best_t = r, words[i]["s"]
    return best_t if best_r >= 0.5 else None


def pin_onset(audio: Path, approx: float, hook_norm: str,
              language: str, preroll: float = 0.3) -> float:
    """Cut->transcribe->correct until the hook PHRASE sits at the clip start
    (defeats whole-song timestamp drift)."""
    start = approx
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "c.wav"
        for _ in range(4):
            clip_start = max(0.0, start - 5.0)
            _cut(audio, clip_start, 18.0, clip)
            words = [w for s in _transcribe(clip, language) for w in s["words"]]
            onset = _locate_phrase(words, hook_norm)
            if onset is None:
                break
            new_start = round(clip_start + onset - preroll, 2)
            converged = abs(new_start - start) < 0.3
            start = new_start
            if converged:
                break
    return max(0.0, start)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--language", default="hi")
    ap.add_argument("--length", type=float, default=50.0)
    ap.add_argument("--hook-hint", default=None,
                    help="known hook phrase (e.g. song title) to anchor on instead of clustering")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--id", default=None)
    a = ap.parse_args()

    print(f"transcribing {a.audio.name} ({a.language}) ...")
    segs = _transcribe(a.audio, a.language)
    if a.hook_hint:
        hook_text, hook_norm = a.hook_hint, _norm(a.hook_hint)
        occ = [dict(l, norm=_norm(l["text"])) for l in segs
               if _sim(_norm(l["text"]), hook_norm) >= 0.4]
        print(f"hook hint: {hook_text!r}  ({len(occ)} matches)")
    else:
        hook_text, hook_norm, occ = find_hook_line(segs)
        print(f"hook line: {hook_text!r}  ({len(occ)} repeats)")
    if not occ:
        print("no hook found — leave to acoustic analyzer")
        return 0

    y, sr = librosa.load(str(a.audio), sr=22050, mono=True)
    rms = librosa.feature.rms(y=y)[0]
    rt = librosa.times_like(rms, sr=sr)
    dur = len(y) / sr

    def energy(t):
        m = (rt >= t) & (rt <= t + 6)
        return float(rms[m].mean()) if m.any() else 0.0

    best = max(occ, key=lambda l: energy(l["start"]))
    print(f"best occurrence ~{best['start']:.1f}s -> pinning onset ...")
    start = pin_onset(a.audio, best["start"], hook_norm, a.language)
    end = round(min(dur, start + a.length), 2)
    print(f"PINNED start={start}s  end={end}s")

    if a.manifest and a.id:
        lib = json.loads(a.manifest.read_text())
        song = next(s for s in lib["songs"] if s["id"] == a.id)
        song["method"] = "lyric"
        song["sections"] = [{"id": "hook", "start": start, "end": end,
                             "description": f"Opens on hook {hook_text!r} (auto, clip-verified onset)."}]
        a.manifest.write_text(json.dumps(lib, indent=2, ensure_ascii=False) + "\n")
        print(f"updated {a.manifest} [{a.id}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
