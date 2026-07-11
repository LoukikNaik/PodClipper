#!/usr/bin/env python3
"""Detect hit-sections for the music library, combining vocals + beat + energy.

For each song in a library manifest, computes per-frame:
  - music energy   (RMS of the full spectrogram)
  - vocal presence (RMS of the librosa nn_filter foreground / vocal mask)
  - rhythmic drive (onset strength)
combines them into a "hook" curve, finds the strongest sustained region(s),
and snaps their boundaries to the beat grid. Writes start/end + factor
metadata back into the manifest (preserving each song's vibe description).

Usage: python dev/analyze_music.py music/library.json [--sections 2] [--min 16] [--max 40]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import scipy.signal
import scipy.signal.windows
if not hasattr(scipy.signal, "hann"):       # librosa 0.10 vs scipy >=1.13 shim
    scipy.signal.hann = scipy.signal.windows.hann

import librosa
import numpy as np
from scipy.ndimage import median_filter

HOP = 2048          # feature hop (~93ms @ 22050) — keeps nn_filter tractable on full songs
SR = 22050


def _norm(x: np.ndarray) -> np.ndarray:
    """Robust 0..1 normalize by the 95th percentile."""
    hi = np.percentile(x, 95) or 1.0
    return np.clip(x / hi, 0.0, 1.0)


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = np.ones(win) / win
    return np.convolve(x, k, mode="same")


def analyze_song(path: Path, n_sections: int, length: float) -> dict:
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    dur = len(y) / sr

    S_full = np.abs(librosa.stft(y, hop_length=HOP))
    times = librosa.frames_to_time(np.arange(S_full.shape[1]), sr=sr, hop_length=HOP)

    energy = librosa.feature.rms(S=S_full, hop_length=HOP)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    onset = librosa.util.fix_length(onset, size=len(energy))

    # Vocal / foreground isolation: accompaniment is temporally repetitive, so a
    # per-freq running median across time estimates the background; the residual
    # foreground is dominated by vocals (REPET idea, no sklearn dependency).
    w = max(3, int(librosa.time_to_frames(2, sr=sr, hop_length=HOP)))
    S_bg = np.minimum(S_full, median_filter(S_full, size=(1, w)))
    mask_v = librosa.util.softmask(S_full - S_bg, 10 * S_bg, power=2)
    vocal = librosa.feature.rms(S=mask_v * S_full, hop_length=HOP)[0]

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beats, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    e_n, v_n, o_n = _norm(energy), _norm(vocal), _norm(onset)
    vocal_frac = float((vocal > np.percentile(vocal, 60)).mean())
    has_vocals = bool(np.median(v_n) > 0.12 and vocal_frac > 0.25)

    # Weight vocals more for songs that actually have them.
    w_v = 0.30 if has_vocals else 0.05
    hook = _smooth((1.0 - w_v - 0.20) * e_n + w_v * v_n + 0.20 * o_n,
                   win=max(1, int(round(2.0 / (HOP / sr)))))

    def _snap(t: float) -> float:
        if beat_times.size == 0:
            return round(float(t), 2)
        return round(float(beat_times[np.argmin(np.abs(beat_times - t))]), 2)

    # Each section STARTS on the impactful peak and runs forward `length` s.
    sections = []
    hook_work = hook.copy()
    peak0 = float(hook.max())
    guard = int(round(3.0 / (HOP / sr)))
    for _ in range(n_sections):
        pk = int(np.argmax(hook_work))
        if hook_work[pk] < 0.55 * peak0:
            break
        start = _snap(times[pk])
        end = round(min(dur, start + length), 2)
        seg_mask = (times >= start) & (times <= end)
        sections.append({
            "start": start,
            "end": end,
            "energy_db": round(float(20 * np.log10(energy[seg_mask].mean() + 1e-9)), 1),
            "vocal_pct": round(float(v_n[seg_mask].mean()) * 100, 1),
            "hook": round(float(hook[seg_mask].mean()), 3),
        })
        hi = int(np.searchsorted(times, end))
        hook_work[max(0, pk - guard):min(len(hook_work), hi + guard)] = 0

    sections.sort(key=lambda s: s["start"])
    return {
        "duration": round(dur, 1),
        "tempo_bpm": round(tempo, 1),
        "has_vocals": has_vocals,
        "sections": sections,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--sections", type=int, default=2)
    ap.add_argument("--length", type=float, default=50.0,
                    help="section runway in seconds, starting on the peak")
    ap.add_argument("--dry-run", action="store_true", help="print, don't write")
    a = ap.parse_args()

    lib = json.loads(a.manifest.read_text())
    base = a.manifest.parent
    for song in lib["songs"]:
        if song.get("method") == "lyric":
            print(f"  KEEP {song['id']}: lyric-anchored (skipping acoustic pass)")
            continue
        f = base / song["file"]
        if not f.exists():
            print(f"  SKIP {song['id']}: file missing ({f})")
            continue
        res = analyze_song(f, a.sections, a.length)
        song["has_vocals"] = res["has_vocals"]
        song["tempo_bpm"] = res["tempo_bpm"]
        new_secs = []
        for i, s in enumerate(res["sections"]):
            prev = song.get("sections", [])
            desc = prev[i]["description"] if i < len(prev) and "description" in prev[i] else ""
            new_secs.append({
                "id": (prev[i]["id"] if i < len(prev) else f"hit{i+1}"),
                "start": s["start"], "end": s["end"],
                "energy_db": s["energy_db"], "vocal_pct": s["vocal_pct"],
                "description": desc,
            })
        song["sections"] = new_secs
        secs = "  ".join(f"{s['start']:.1f}-{s['end']:.1f}s "
                         f"(E{s['energy_db']}dB V{s['vocal_pct']}%)" for s in new_secs)
        print(f"  {song['id']:16s} {res['tempo_bpm']:5.0f}bpm "
              f"vocals={str(res['has_vocals']):5s}  {secs}")

    if not a.dry_run:
        a.manifest.write_text(json.dumps(lib, indent=2) + "\n")
        print(f"\nupdated {a.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
