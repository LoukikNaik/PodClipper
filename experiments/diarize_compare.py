#!/usr/bin/env python3
"""Run multiple diarization backends on the same clip and print their
speaker timelines side-by-side, so we can eyeball which one separates
voices best on a real sample.

Backends:
  1. pyannote.audio 3.1 — the current baseline (HF gated, slow, accurate)
  2. Resemblyzer + WebRTC VAD + Agglomerative clustering — small, fast,
     no HF gating

Usage:
    python diarize_compare.py PATH/TO/AUDIO.wav [--n-speakers 2]

Audio must be a mono 16 kHz WAV (use `ffmpeg -i in.mp4 -vn -ac 1 -ar 16000
-c:a pcm_s16le out.wav`).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Pyannote backend (uses our existing src.diarize wrapper)
# ──────────────────────────────────────────────────────────────────────

def run_pyannote(audio_path: Path) -> list[tuple[float, float, str]]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.diarize import diarize_clip

    cfg = SimpleNamespace(
        diarize=SimpleNamespace(
            model="pyannote/speaker-diarization-3.1",
            hf_token_env="HF_TOKEN",
            min_speakers=None,
            max_speakers=None,
        ),
        paths=SimpleNamespace(cache_dir=".cache"),
    )
    segs = diarize_clip(audio_path, cfg)
    if segs is None:
        return []
    return [(s.start, s.end, s.speaker_id) for s in segs]


# ──────────────────────────────────────────────────────────────────────
# Resemblyzer + WebRTC VAD + clustering backend
# ──────────────────────────────────────────────────────────────────────

def run_resemblyzer(audio_path: Path, n_speakers: int = 2) -> list[tuple[float, float, str]]:
    """Embed the audio in overlapping windows with Resemblyzer's d-vector
    encoder, cluster the embeddings into `n_speakers` groups, then collapse
    consecutive same-cluster windows into segments.
    """
    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder()
    wav = preprocess_wav(audio_path)   # 16 kHz mono float32, VAD-trimmed silence

    # Continuous-windowed embeddings: 1.4s windows every 0.4s (75% overlap).
    # Resemblyzer's `embed_utterance` does this when continuous=True and
    # returns the per-window embeddings + a partials_slices array.
    _, partial_embeds, wav_splits = encoder.embed_utterance(
        wav, return_partials=True, rate=2.5  # 2.5 windows/s → 400ms hop
    )

    # Cluster embeddings into n_speakers groups
    from sklearn.cluster import AgglomerativeClustering
    labels = AgglomerativeClustering(
        n_clusters=n_speakers,
        metric="cosine",
        linkage="average",
    ).fit_predict(partial_embeds)

    # Walk windows in time order, collapse adjacent same-cluster ones.
    sr = 16000
    segs: list[tuple[float, float, str]] = []
    cur_start, cur_end, cur_label = None, None, None
    for sl, lbl in zip(wav_splits, labels):
        start_s = sl.start / sr
        end_s = sl.stop / sr
        spk = f"SPEAKER_{int(lbl):02d}"
        if cur_label is None:
            cur_start, cur_end, cur_label = start_s, end_s, spk
        elif spk == cur_label and start_s <= cur_end + 0.5:
            cur_end = end_s
        else:
            segs.append((cur_start, cur_end, cur_label))
            cur_start, cur_end, cur_label = start_s, end_s, spk
    if cur_label is not None:
        segs.append((cur_start, cur_end, cur_label))
    return segs


# ──────────────────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────────────────

def fmt_segs(segs: list[tuple[float, float, str]]) -> str:
    if not segs:
        return "  (no segments)"
    lines = []
    speakers = sorted({s[2] for s in segs})
    lines.append(f"  {len(segs)} segments, {len(speakers)} speakers: {speakers}")
    for start, end, spk in segs:
        lines.append(f"    [{start:6.2f} - {end:6.2f}s]  ({end-start:5.2f}s)  {spk}")
    return "\n".join(lines)


def speaker_share(segs: list[tuple[float, float, str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for start, end, spk in segs:
        out[spk] = out.get(spk, 0.0) + (end - start)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path, help="Mono 16kHz WAV")
    parser.add_argument("--n-speakers", type=int, default=2,
                        help="Number of speakers to cluster Resemblyzer to (default 2)")
    parser.add_argument("--skip-pyannote", action="store_true")
    parser.add_argument("--skip-resemblyzer", action="store_true")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"not found: {args.audio}")
        return 2

    # Load .env so HF_TOKEN is available
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            k, _, v = line.partition("=")
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k.strip(), v)

    print(f"\nInput: {args.audio}")

    if not args.skip_pyannote:
        print("\n===== pyannote/speaker-diarization-3.1 =====")
        t0 = time.perf_counter()
        try:
            segs = run_pyannote(args.audio)
            dt = time.perf_counter() - t0
            print(f"  elapsed: {dt:.1f}s")
            print(fmt_segs(segs))
            share = speaker_share(segs)
            print(f"  speaker share (s): {dict((k, round(v, 1)) for k, v in share.items())}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}")

    if not args.skip_resemblyzer:
        print(f"\n===== Resemblyzer (n_clusters={args.n_speakers}) =====")
        t0 = time.perf_counter()
        try:
            segs = run_resemblyzer(args.audio, n_speakers=args.n_speakers)
            dt = time.perf_counter() - t0
            print(f"  elapsed: {dt:.1f}s")
            print(fmt_segs(segs))
            share = speaker_share(segs)
            print(f"  speaker share (s): {dict((k, round(v, 1)) for k, v in share.items())}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
