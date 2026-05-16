"""Validate the input video and probe basic metadata via ffprobe."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from .types import VideoMeta

log = logging.getLogger("ave.ingest")


class IngestError(Exception):
    pass


def _run_ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError as e:
        raise IngestError("ffprobe not found on PATH — install ffmpeg") from e
    except subprocess.TimeoutExpired as e:
        raise IngestError(f"ffprobe timed out on {path}") from e
    except subprocess.CalledProcessError as e:
        raise IngestError(f"ffprobe failed: {e.stderr.strip()}") from e

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise IngestError(f"ffprobe returned invalid JSON: {e}") from e


def _parse_fps(rate: str) -> float:
    """ffprobe returns fps as 'num/den' (e.g. '30000/1001'); convert to float."""
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_f = float(den)
        if den_f == 0:
            return 0.0
        return float(num) / den_f
    return float(rate)


def ingest(path: Path) -> VideoMeta:
    """Validate the input video and return its metadata."""
    path = Path(path).resolve()
    if not path.exists():
        raise IngestError(f"input video not found: {path}")
    if not path.is_file():
        raise IngestError(f"input path is not a file: {path}")

    probe = _run_ffprobe(path)
    streams = probe.get("streams", [])

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video_stream is None:
        raise IngestError(f"no video stream found in {path}")

    duration_s = video_stream.get("duration") or probe.get("format", {}).get("duration")
    if duration_s is None:
        raise IngestError(f"could not determine duration of {path}")
    duration = float(duration_s)

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise IngestError(f"invalid video dimensions {width}x{height}")

    # avg_frame_rate accounts for VFR; r_frame_rate is the fallback.
    fps_raw = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/0"
    fps = _parse_fps(fps_raw)
    if fps <= 0:
        raise IngestError(f"invalid frame rate {fps_raw!r}")

    codec = video_stream.get("codec_name", "unknown")
    has_audio = audio_stream is not None

    meta = VideoMeta(
        path=path,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        has_audio=has_audio,
        codec=codec,
    )

    log.info(
        f"Ingested {path.name}: {width}x{height} @ {fps:.2f}fps, "
        f"{duration:.1f}s, codec={codec}, audio={'yes' if has_audio else 'NO'}"
    )

    if not has_audio:
        log.warning("No audio stream — transcription and clip selection will fail.")

    return meta
