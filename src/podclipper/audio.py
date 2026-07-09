"""Audio extraction + chunk planning."""

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
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def extract_audio(
    video_path: Path,
    out_path: Path,
    sample_rate: int = 16000,
    codec: str = "pcm_s16le",
    overwrite: bool = False,
    duration_limit: float | None = None,
) -> Path:
    """Extract mono audio from `video_path` to `out_path`. `duration_limit`
    caps extraction to the first N seconds. Returns the existing path unchanged
    when it exists and `overwrite=False`."""
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        log.debug(f"Reusing cached audio at {out_path}")
        return out_path

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", codec,
    ]
    if duration_limit is not None:
        cmd += ["-t", f"{duration_limit:.3f}"]
    cmd.append(str(out_path))
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
    """Split [0, duration] into overlapping chunks for parallel transcription."""
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
