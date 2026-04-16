"""Stage 2: Audio extraction + chunking.

- `extract_audio(video, out_path, sample_rate)` produces a mono WAV for Whisper.
- `plan_chunks(duration, chunk_s, overlap_s)` returns [(start, end), ...] time ranges.
  Actual per-chunk decoding is done lazily inside `transcribe.py` via ffmpeg stdin —
  we don't write N WAV chunk files to disk.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("ave.audio")


class AudioError(Exception):
    pass


@dataclass(frozen=True)
class ChunkRange:
    index: int
    start: float    # seconds
    end: float      # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


def extract_audio(
    video_path: Path,
    out_path: Path,
    sample_rate: int = 16000,
    codec: str = "pcm_s16le",
    overwrite: bool = False,
) -> Path:
    """Extract audio from `video_path` into a single WAV file at `out_path`.

    Mono, specified sample rate and codec. If `out_path` exists and overwrite
    is False, returns the existing path without re-running ffmpeg.
    """
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        log.debug(f"Reusing cached audio at {out_path}")
        return out_path

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                         # drop video
        "-ac", "1",                    # mono
        "-ar", str(sample_rate),
        "-c:a", codec,
        str(out_path),
    ]
    log.info(f"Extracting audio → {out_path.name} ({sample_rate} Hz, mono)")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    except subprocess.CalledProcessError as e:
        raise AudioError(f"ffmpeg audio extract failed: {e.stderr.strip()[-500:]}") from e
    except subprocess.TimeoutExpired as e:
        raise AudioError(f"ffmpeg audio extract timed out on {video_path}") from e

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise AudioError(f"ffmpeg produced no audio output at {out_path}")

    return out_path


def plan_chunks(
    duration: float,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[ChunkRange]:
    """Split a [0, duration] range into overlapping chunks for parallel transcription.

    Each chunk (except possibly the last) has length `chunk_seconds`.
    Consecutive chunks overlap by `overlap_seconds`. The final chunk is clipped
    to the total duration. Returns ≥1 chunk always.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be in [0, chunk_seconds)")

    step = chunk_seconds - overlap_seconds
    chunks: list[ChunkRange] = []
    i = 0
    start = 0.0
    while start < duration:
        end = min(start + chunk_seconds, duration)
        chunks.append(ChunkRange(index=i, start=start, end=end))
        if end >= duration:
            break
        start += step
        i += 1

    log.debug(
        f"Planned {len(chunks)} chunks over {duration:.1f}s "
        f"(chunk={chunk_seconds}s, overlap={overlap_seconds}s)"
    )
    return chunks
