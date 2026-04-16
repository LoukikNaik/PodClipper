"""End-to-end pipeline orchestration.

Stages (see plan.md for design rationale):
  1. Ingest — validate + probe metadata
  2. Audio   — extract whole-video audio
  3. Transcribe (first pass) — fast, parallel, for LLM analysis
  4. Analyze — LLM picks reel-worthy clips
  5. Per clip:
     a. Extract segment with ffmpeg (pad buffer)
     b. Detect persons per frame (YOLO)
     c. Concurrently: second-pass transcribe (large Whisper)
                    + optional diarization (stub until post-MVP)
     d. Build speaker timeline
     e. Smart 9:16 crop (OpenCV, timeline-driven)
     f. Burn karaoke subtitles (PIL + OpenCV)
     g. Save to output_dir

Intermediate artifacts go under cfg.paths.cache_dir/<video_stem>/. When
use_cache=True (default), existing artifacts are reused — enables resume.
"""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .analyze import analyze_for_reels
from .audio import extract_audio
from .crop import smart_crop_916
from .detect import detect_humans_per_frame
from .ingest import ingest
from .llm import build_provider
from .logging_util import get_console
from .subtitles import burn_subtitles
from .timeline import apply_min_dwell, build_speaker_timeline
from .transcribe import transcribe_first_pass, transcribe_second_pass
from .types import Clip, VideoMeta

log = logging.getLogger("ave.pipeline")


# ---------- helpers ----------

def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:max_len] or "clip"


def _cache_dir_for(cfg: SimpleNamespace, video_path: Path) -> Path:
    # Hash absolute path so the same video always maps to the same cache dir.
    h = hashlib.sha1(str(video_path.resolve()).encode()).hexdigest()[:10]
    base = Path(cfg.paths.cache_dir) / f"{video_path.stem}-{h}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _extract_clip_segment(
    video_path: Path,
    clip: Clip,
    video_duration: float,
    pad_s: float,
    out_path: Path,
    cfg: SimpleNamespace,
) -> Path:
    """Cut [clip.start - pad, clip.end + pad] out of the source, clamped."""
    start = max(0.0, clip.start - pad_s)
    end = min(video_duration, clip.end + pad_s)
    duration = end - start
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c:v", cfg.clip_extract.video_codec,
        "-crf", str(cfg.clip_extract.crf),
        "-preset", cfg.clip_extract.preset,
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg clip extract failed: {e.stderr.strip()[-500:]}") from e
    return out_path


def _maybe_diarize(segment_path: Path, per_frame_bboxes, cfg: SimpleNamespace):
    """Post-MVP hook. Currently returns None; enabling flips in diarize.py."""
    if not getattr(cfg.diarize, "enabled", False):
        return None
    try:
        from .diarize import diarize_clip  # noqa: F401 — future module
    except ImportError:
        log.warning("diarize.enabled=true but diarize module not available; skipping.")
        return None
    from .diarize import diarize_clip
    return diarize_clip(segment_path, per_frame_bboxes, cfg)


# ---------- main entry ----------

def run_pipeline(
    input_path: Path,
    cfg: SimpleNamespace,
    use_cache: bool = True,
) -> list[Path]:
    """Run the full pipeline; return paths of produced reels."""
    input_path = Path(input_path)
    console = get_console()

    # --- Stage 1: Ingest ---
    meta: VideoMeta = ingest(input_path)

    # --- Stage 2: Audio extract ---
    cache = _cache_dir_for(cfg, input_path)
    audio_wav = cache / "audio.wav"
    extract_audio(
        meta.path, audio_wav,
        sample_rate=cfg.audio.sample_rate,
        codec=cfg.audio.codec,
        overwrite=not use_cache,
    )

    # --- Stage 3: First-pass transcription ---
    transcript = transcribe_first_pass(audio_wav, meta.duration, cfg)
    # TODO: could persist transcript JSON here for resume; skipping for MVP

    # --- Stage 4: LLM analysis ---
    provider = build_provider(cfg.llm)
    clips = analyze_for_reels(transcript, meta.duration, provider, cfg)
    if not clips:
        log.warning("LLM returned no clips — nothing to produce")
        return []

    # --- Stage 5: Per-clip loop ---
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    produced: list[Path] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task(f"Processing {len(clips)} clip(s)", total=len(clips))

        for i, clip in enumerate(clips, 1):
            slug = _slug(clip.title)
            stem = f"reel_{i:02d}_{slug}"
            clip_cache = cache / stem
            clip_cache.mkdir(parents=True, exist_ok=True)

            progress.update(task_id, description=f"[{i}/{len(clips)}] {clip.title[:40]}")

            try:
                segment_path = clip_cache / "segment.mp4"
                if not segment_path.exists() or not use_cache:
                    _extract_clip_segment(
                        meta.path, clip, meta.duration,
                        cfg.clip_extract.pad_seconds, segment_path, cfg,
                    )

                # Per-frame detection up front (provides the bbox list to
                # both timeline and diarization stages).
                per_frame, clip_fps, clip_w, clip_h = detect_humans_per_frame(segment_path, cfg)

                # Run transcription + diarization concurrently (I/O + CPU bound; they don't share state).
                with ThreadPoolExecutor(max_workers=2) as ex:
                    f_words = ex.submit(transcribe_second_pass, segment_path, cfg)
                    f_diar = ex.submit(_maybe_diarize, segment_path, per_frame, cfg)
                    words = f_words.result()
                    diar_segments = f_diar.result()

                # Clip duration from frame count (ffmpeg clip may be slightly longer than requested)
                clip_duration = len(per_frame) / clip_fps if clip_fps else (clip.duration + 2 * cfg.clip_extract.pad_seconds)

                timeline = build_speaker_timeline(
                    per_frame_bboxes=per_frame,
                    clip_duration=clip_duration,
                    fps=clip_fps,
                    source_width=clip_w,
                    source_height=clip_h,
                    diar_segments=diar_segments,
                    cfg=cfg,
                )
                timeline = apply_min_dwell(timeline, cfg.crop.min_segment_dwell_seconds)

                # Crop
                cropped_path = clip_cache / "cropped.mp4"
                smart_crop_916(segment_path, timeline, cropped_path, cfg)

                # Subtitles
                final_path = output_dir / f"{stem}.mp4"
                burn_subtitles(cropped_path, words, final_path, cfg)

                # Persist a tiny sidecar describing the clip
                (output_dir / f"{stem}.txt").write_text(
                    f"title: {clip.title}\n"
                    f"reason: {clip.reason}\n"
                    f"source_start: {clip.start:.2f}\n"
                    f"source_end: {clip.end:.2f}\n"
                    f"hook_score: {clip.hook_score:.2f}\n"
                )
                produced.append(final_path)
                log.info(f"[{i}/{len(clips)}] Done: {final_path}")

            except Exception as exc:  # noqa: BLE001 — per-clip boundary, keep going
                log.exception(f"[{i}/{len(clips)}] Failed: {clip.title}: {exc}")

            progress.advance(task_id)

    log.info(f"Pipeline complete — {len(produced)}/{len(clips)} reels written to {output_dir}")
    for p in produced:
        log.info(f"  → {p}")
    return produced
