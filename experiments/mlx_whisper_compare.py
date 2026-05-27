"""Side-by-side test of faster-whisper (current 2nd-pass) vs mlx-whisper on a
clip's audio. Compares wall time, word count, language, and prints a unified
diff of the joined word streams.

Usage:
    python experiments/mlx_whisper_compare.py path/to/segment.mp4
    python experiments/mlx_whisper_compare.py            # uses a default cached clip
"""

from __future__ import annotations

import difflib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

SAMPLE_RATE = 16000

# faster-whisper model size used by the pipeline 2nd pass
FW_MODEL = "large-v3"
FW_COMPUTE = "int8"

# mlx-whisper HF repo (community-converted weights)
MLX_MODEL = "mlx-community/whisper-large-v3-mlx"


def decode_audio(path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", str(path),
        "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-",
    ]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


def run_faster_whisper(audio: np.ndarray) -> dict:
    from faster_whisper import WhisperModel
    t_load = time.perf_counter()
    model = WhisperModel(FW_MODEL, device="cpu", compute_type=FW_COMPUTE)
    load_s = time.perf_counter() - t_load

    t = time.perf_counter()
    segs_iter, info = model.transcribe(
        audio, language=None, beam_size=5, word_timestamps=True, vad_filter=False,
    )
    words = []
    text_parts = []
    for seg in segs_iter:
        text_parts.append(seg.text.strip())
        if seg.words:
            for w in seg.words:
                words.append({
                    "start": float(w.start),
                    "end": float(w.end),
                    "text": w.word,
                    "prob": float(getattr(w, "probability", 1.0)),
                })
    return {
        "engine": "faster-whisper",
        "load_s": load_s,
        "transcribe_s": time.perf_counter() - t,
        "language": info.language,
        "text": " ".join(text_parts).strip(),
        "words": words,
    }


def run_mlx_whisper(audio: np.ndarray) -> dict:
    import mlx_whisper
    t = time.perf_counter()
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MLX_MODEL,
        word_timestamps=True,
        verbose=None,
    )
    elapsed = time.perf_counter() - t

    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []) or []:
            words.append({
                "start": float(w["start"]),
                "end": float(w["end"]),
                "text": w["word"],
                "prob": float(w.get("probability", 1.0)),
            })
    return {
        "engine": "mlx-whisper",
        "load_s": 0.0,  # mlx loads on first call; rolled into transcribe_s
        "transcribe_s": elapsed,
        "language": result.get("language", "?"),
        "text": result.get("text", "").strip(),
        "words": words,
    }


def normalize(s: str) -> str:
    return " ".join(s.lower().split())


def compare(a: dict, b: dict) -> None:
    print()
    print("=" * 80)
    print(f"COMPARE  {a['engine']}   vs   {b['engine']}")
    print("=" * 80)
    for r in (a, b):
        print(
            f"  {r['engine']:>16}  load={r['load_s']:6.2f}s  "
            f"transcribe={r['transcribe_s']:6.2f}s  "
            f"words={len(r['words']):4d}  lang={r['language']}"
        )

    na, nb = normalize(a["text"]), normalize(b["text"])
    if na == nb:
        print("\n  TEXT: identical after normalization")
    else:
        ratio = difflib.SequenceMatcher(None, na, nb).ratio()
        print(f"\n  TEXT: similarity={ratio:.3f}")
        diff = difflib.unified_diff(
            na.split(), nb.split(),
            fromfile=a["engine"], tofile=b["engine"], n=3, lineterm="",
        )
        printed = 0
        for line in diff:
            print(f"    {line}")
            printed += 1
            if printed > 60:
                print("    ... (diff truncated)")
                break

    n = min(len(a["words"]), len(b["words"]))
    if n:
        deltas = [abs(a["words"][i]["start"] - b["words"][i]["start"]) for i in range(n)]
        print(
            f"\n  TIMING: aligned-by-index over {n} words  "
            f"mean Δstart={np.mean(deltas):.3f}s  "
            f"median={np.median(deltas):.3f}s  "
            f"max={np.max(deltas):.3f}s"
        )


def main() -> None:
    default_clip = (
        REPO
        / ".cache/seg_000-b4f4381614/reel_01_elons-dad-married-his-stepdaughter/segment.mp4"
    )
    clip = Path(sys.argv[1]) if len(sys.argv) > 1 else default_clip
    if not clip.exists():
        sys.exit(f"clip not found: {clip}")

    print(f"clip: {clip}")
    audio = decode_audio(clip)
    print(f"audio: {len(audio) / SAMPLE_RATE:.2f}s  shape={audio.shape}")

    fw = run_faster_whisper(audio)
    mlx = run_mlx_whisper(audio)
    compare(fw, mlx)


if __name__ == "__main__":
    main()
