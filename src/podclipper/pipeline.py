"""End-to-end pipeline orchestration for reels and trailer modes."""

from __future__ import annotations

import hashlib
import json
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
from .crop import smart_crop_916, smart_crop_916_stacked
from .detect import detect_humans_per_frame, detect_humans_all_per_frame
from .ingest import ingest
from .llm import build_provider
from .logging_util import get_console
from .evaluate import evaluate_reel, evaluate_trailer
from .subtitles import burn_subtitles
from .timeline import apply_min_dwell, build_speaker_timeline, classify_wide_shot_frames
from .trailer import (
    concat_with_black_gaps,
    build_trailer_words,
    cut_segment as trailer_cut_segment,
    pick_quotables,
    refine_cut_bounds_with_llm,
)
from .transcribe import (
    engine_suffix,
    first_pass_cache_path,
    transcribe_first_pass,
    transcribe_second_pass_cached,
    words_cache_path,
)
from .transcribe_cleanup import cleanup_words
from .types import Clip, Transcript, TranscriptSegment, VideoMeta, Word

log = logging.getLogger("ave.pipeline")


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:max_len] or "clip"


def _cache_dir_for(cfg: SimpleNamespace, video_path: Path) -> Path:
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


def _maybe_diarize(segment_path: Path, cfg: SimpleNamespace):
    """DEPRECATED — only called from `crop.mode: single`."""
    if not getattr(cfg.diarize, "enabled", False):
        return None
    try:
        from .diarize import diarize_clip
    except ImportError as e:
        log.warning(f"diarize.enabled=true but dependencies missing: {e}")
        return None
    return diarize_clip(segment_path, cfg)


def run_pipeline(
    input_path: Path,
    cfg: SimpleNamespace,
    use_cache: bool = True,
) -> list[Path]:
    """Run the full reels pipeline; return paths of produced reels."""
    input_path = Path(input_path)
    console = get_console()

    meta: VideoMeta = ingest(input_path)

    cache = _cache_dir_for(cfg, input_path)
    audio_wav = cache / "audio.wav"
    extract_audio(
        meta.path, audio_wav,
        sample_rate=cfg.audio.sample_rate,
        codec=cfg.audio.codec,
        overwrite=not use_cache,
    )

    transcript_cache = first_pass_cache_path(cache, cfg.transcribe.engine)
    if use_cache and transcript_cache.exists():
        log.info(f"Reusing cached first-pass transcript: {transcript_cache}")
        transcript = _transcript_from_json(json.loads(transcript_cache.read_text()))
    else:
        transcript = transcribe_first_pass(audio_wav, meta.duration, cfg)
        transcript_cache.write_text(json.dumps(_transcript_to_json(transcript)))
        log.info(f"Cached first-pass transcript → {transcript_cache}")

    # Pin 2nd-pass language to 1st-pass detection so Whisper doesn't flip
    # mid-episode (Hindi/Urdu/etc. swap produces unreadable subtitles).
    if cfg.transcribe.language is None and transcript.language:
        log.info(f"Pinning transcribe language to first-pass detection: {transcript.language}")
        cfg.transcribe.language = transcript.language

    provider = build_provider(cfg.llm)
    clips = analyze_for_reels(transcript, meta.duration, provider, cfg, debug_cache_dir=cache)
    if not clips:
        log.warning("LLM returned no clips — nothing to produce")
        return []

    from datetime import datetime
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(cfg.paths.output_dir) / run_stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {output_dir}")

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

                crop_mode = getattr(cfg.crop, "mode", "auto")
                words_cache = words_cache_path(clip_cache, cfg.transcribe.engine) if use_cache else None

                per_frame_persons: list[list] = []

                if crop_mode == "auto":
                    per_frame_persons, _per_frame_has_face, clip_fps, clip_w, clip_h = (
                        detect_humans_all_per_frame(segment_path, cfg)
                    )
                    words = transcribe_second_pass_cached(segment_path, words_cache, cfg)
                    per_frame = [
                        max(ps, key=lambda b: b.area) if ps else None
                        for ps in per_frame_persons
                    ]
                else:
                    # DEPRECATED `crop.mode: single` branch.
                    per_frame, clip_fps, clip_w, clip_h = detect_humans_per_frame(segment_path, cfg)
                    with ThreadPoolExecutor(max_workers=2) as ex:
                        f_words = ex.submit(transcribe_second_pass_cached, segment_path, words_cache, cfg)
                        f_diar = ex.submit(_maybe_diarize, segment_path, cfg)
                        words = f_words.result()
                        diar_segments = f_diar.result()

                words = cleanup_words(words, provider, cfg)

                clip_duration = len(per_frame) / clip_fps if clip_fps else (clip.duration + 2 * cfg.clip_extract.pad_seconds)

                cropped_path = clip_cache / "cropped.mp4"
                if crop_mode == "auto":
                    is_wide = classify_wide_shot_frames(
                        per_frame_persons,
                        source_width=clip_w,
                        source_height=clip_h,
                        sep_threshold_frac=getattr(cfg.crop, "shot_sep_frac", 0.20),
                        height_cap_frac=getattr(cfg.crop, "shot_height_cap_frac", 0.70),
                        smooth_window_frames=getattr(cfg.crop, "shot_smooth_window_frames", 15),
                    )
                    smart_crop_916_stacked(
                        segment_path, per_frame_persons, is_wide, cropped_path, cfg,
                    )
                else:
                    timeline = build_speaker_timeline(
                        per_frame_bboxes=per_frame,
                        clip_duration=clip_duration,
                        fps=clip_fps,
                        source_width=clip_w,
                        source_height=clip_h,
                        diar_segments=diar_segments,
                        cfg=cfg,
                        video_path=segment_path,
                    )
                    timeline = apply_min_dwell(timeline, cfg.crop.min_segment_dwell_seconds)
                    smart_crop_916(segment_path, timeline, cropped_path, cfg)

                final_path = output_dir / f"{stem}.mp4"
                burn_subtitles(cropped_path, words, final_path, cfg, title=clip.title)

                sidecar_path = output_dir / f"{stem}.txt"
                sidecar_path.write_text(
                    f"title: {clip.title}\n"
                    f"reason: {clip.reason}\n"
                    f"source_start: {clip.start:.2f}\n"
                    f"source_end: {clip.end:.2f}\n"
                    f"hook_score: {clip.hook_score:.2f}\n"
                )

                non_null = sum(1 for b in per_frame if b is not None)
                x_centers = [b.x_center for b in per_frame if b is not None]
                scorecard = evaluate_reel(
                    title=clip.title,
                    words=words,
                    clip_duration=clip_duration,
                    face_hits=non_null,
                    total_frames=len(per_frame),
                    person_frames=non_null,
                    x_centers=x_centers,
                    source_width=clip_w,
                    provider=provider,
                    cfg=cfg,
                )
                scorecard.write_to(sidecar_path)

                eval_cfg = getattr(cfg, "evaluate", None)
                if (eval_cfg and getattr(eval_cfg, "auto_skip", False)
                        and scorecard.verdict == "skip"):
                    skip_dir = output_dir / "skipped"
                    skip_dir.mkdir(exist_ok=True)
                    final_path.rename(skip_dir / final_path.name)
                    sidecar_path.rename(skip_dir / sidecar_path.name)
                    log.warning(f"[{i}/{len(clips)}] Auto-skipped (score {scorecard.final_score:.2f}): {clip.title}")
                else:
                    produced.append(final_path)
                    log.info(f"[{i}/{len(clips)}] Done ({scorecard.verdict}): {final_path}")

            except Exception as exc:  # noqa: BLE001
                log.exception(f"[{i}/{len(clips)}] Failed: {clip.title}: {exc}")

            progress.advance(task_id)

    log.info(f"Pipeline complete — {len(produced)}/{len(clips)} reels written to {output_dir}")
    for p in produced:
        log.info(f"  → {p}")
    return produced


def _transcript_to_json(t: Transcript) -> dict:
    return {
        "language": t.language,
        "segments": [
            {
                "start": s.start, "end": s.end, "text": s.text,
                "words": [
                    {"start": w.start, "end": w.end, "text": w.text,
                     "confidence": w.confidence}
                    for w in s.words
                ],
            }
            for s in t.segments
        ],
    }


def _transcript_from_json(d: dict) -> Transcript:
    segs = []
    for s in d["segments"]:
        words = [Word(**w) for w in s.get("words", [])]
        segs.append(TranscriptSegment(
            start=s["start"], end=s["end"], text=s["text"], words=words,
        ))
    return Transcript(language=d["language"], segments=segs)


def run_trailer_pipeline(
    input_path: Path,
    cfg: SimpleNamespace,
    use_cache: bool = True,
) -> Path | None:
    """Build one trailer for the input video (4-5 quotable picks stitched
    with black-frame transitions)."""
    import json
    from datetime import datetime

    input_path = Path(input_path)
    console = get_console()

    meta: VideoMeta = ingest(input_path)
    log.info(f"Source: {meta.path.name}  {meta.width}x{meta.height} "
             f"{meta.fps:.2f}fps  {meta.duration:.1f}s")

    cache = _cache_dir_for(cfg, input_path)
    trailer_cache = cache / "trailer"
    trailer_cache.mkdir(parents=True, exist_ok=True)

    audio_wav = cache / "audio.wav"
    extract_audio(
        meta.path, audio_wav,
        sample_rate=cfg.audio.sample_rate,
        codec=cfg.audio.codec,
        overwrite=not use_cache,
    )

    transcript_cache = trailer_cache / f"full_transcript_{engine_suffix(cfg.transcribe.engine)}.json"
    if use_cache and transcript_cache.exists():
        log.info(f"Reusing cached transcript: {transcript_cache}")
        transcript = _transcript_from_json(json.loads(transcript_cache.read_text()))
    else:
        transcript = transcribe_first_pass(audio_wav, meta.duration, cfg)
        transcript_cache.write_text(json.dumps(_transcript_to_json(transcript)))
        log.info(f"Cached transcript → {transcript_cache}")

    if cfg.transcribe.language is None and transcript.language:
        log.info(f"Pinning transcribe language to first-pass detection: {transcript.language}")
        cfg.transcribe.language = transcript.language

    provider = build_provider(cfg.llm)
    quotables_cache = trailer_cache / "quotables.json"
    if use_cache and quotables_cache.exists():
        log.info(f"Reusing cached quotables: {quotables_cache}")
        picks = json.loads(quotables_cache.read_text())
    else:
        log.info("Asking LLM to pick 4-5 quotable sentences for the trailer...")
        picks = pick_quotables(transcript, provider, cfg, meta.duration)
        quotables_cache.write_text(json.dumps(picks, indent=2))
        log.info(f"Cached quotables → {quotables_cache}")

    if not picks:
        log.error("LLM returned zero picks — nothing to produce")
        return None

    refined_cache = trailer_cache / "refined_bounds.json"
    refined: list[dict] | None = None
    if use_cache and refined_cache.exists():
        try:
            cached = json.loads(refined_cache.read_text())
            keys_old = [(c.get("start"), c.get("end"), c.get("sentence")) for c in cached]
            keys_new = [(p["start"], p["end"], p["sentence"]) for p in picks]
            if keys_old == keys_new:
                refined = cached
                log.info(f"Reusing cached refined bounds: {refined_cache}")
            else:
                log.info("Cached bounds stale (picks changed); refining fresh")
        except (KeyError, json.JSONDecodeError):
            pass

    if refined is None:
        log.info(f"Refining cut bounds for {len(picks)} picks via LLM...")
        refined = []
        for q in picks:
            cs, ce, refined_sentence = refine_cut_bounds_with_llm(
                q, transcript, provider, cfg,
            )
            refined.append({
                **q,
                "cut_start": cs,
                "cut_end": ce,
                "refined_sentence": refined_sentence,
            })
        refined_cache.write_text(json.dumps(refined, indent=2))
        log.info(f"Cached refined bounds → {refined_cache}")

    picks = refined

    spoken_total = sum(p["end"] - p["start"] for p in picks)
    log.info(f"Got {len(picks)} picks; spoken duration ≈ {spoken_total:.1f}s")
    for i, p in enumerate(picks, 1):
        log.info(
            f"  [{i:02d}] cut {p['cut_start']:7.2f}-{p['cut_end']:7.2f}  "
            f"({p['cut_end']-p['cut_start']:.2f}s)  "
            f"{p['refined_sentence'][:80]}"
        )

    cropped_clips: list[Path] = []
    pick_words_list: list[list[Word]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task(f"Trailer: cutting {len(picks)} picks", total=len(picks))
        for i, q in enumerate(picks):
            progress.update(task_id, description=f"[{i+1}/{len(picks)}] cut + crop")
            seg_path = trailer_cache / f"q_{i:02d}_segment.mp4"
            words_cache = trailer_cache / f"q_{i:02d}_words_{engine_suffix(cfg.transcribe.engine)}.json"
            cropped_path = trailer_cache / f"q_{i:02d}_cropped.mp4"

            if not seg_path.exists() or not use_cache:
                trailer_cut_segment(meta.path, q["cut_start"], q["cut_end"], seg_path, cfg)

            seg_words = transcribe_second_pass_cached(
                seg_path, words_cache if use_cache else None, cfg,
            )
            pick_words_list.append(seg_words)

            if not cropped_path.exists() or not use_cache:
                per_frame_persons, _hf, _fps, clip_w, clip_h = (
                    detect_humans_all_per_frame(seg_path, cfg)
                )
                is_wide = classify_wide_shot_frames(
                    per_frame_persons,
                    source_width=clip_w,
                    source_height=clip_h,
                    sep_threshold_frac=getattr(cfg.crop, "shot_sep_frac", 0.20),
                    height_cap_frac=getattr(cfg.crop, "shot_height_cap_frac", 0.70),
                    smooth_window_frames=getattr(cfg.crop, "shot_smooth_window_frames", 15),
                )
                smart_crop_916_stacked(seg_path, per_frame_persons, is_wide, cropped_path, cfg)
            cropped_clips.append(cropped_path)
            progress.advance(task_id)

    stitched = trailer_cache / "stitched.mp4"
    log.info(f"Stitching {len(cropped_clips)} clips with black gaps...")
    concat_with_black_gaps(cropped_clips, stitched, cfg)

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(cfg.paths.output_dir) / run_stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "trailer.mp4"

    log.info("Building trailer-time word timeline + burning subtitles...")
    trailer_words = build_trailer_words(picks, pick_words_list, cfg)
    burn_subtitles(stitched, trailer_words, final_path, cfg, title="")

    sidecar_path = output_dir / "trailer.txt"
    lines = [
        f"source: {meta.path.name}",
        f"source_duration: {meta.duration:.1f}s",
        f"trailer_duration: ~{spoken_total + (len(picks)-1) * 0.6:.1f}s "
        f"(spoken {spoken_total:.1f}s + gaps)",
        f"picks: {len(picks)}",
        "",
        "=== PICKS ===",
    ]
    for i, q in enumerate(picks, 1):
        lines.append(
            f"[{i:02d}] {q['cut_start']:7.2f}-{q['cut_end']:7.2f}s "
            f"({q['cut_end']-q['cut_start']:.2f}s) {q['refined_sentence']}"
        )
    sidecar_path.write_text("\n".join(lines) + "\n")

    total_trailer_duration = spoken_total + max(0, len(picks) - 1) * float(
        getattr(getattr(cfg, "trailer", SimpleNamespace()), "gap_seconds", 0.6)
    )
    scorecard = evaluate_trailer(picks, total_trailer_duration, provider, cfg)
    scorecard.write_to(sidecar_path)

    log.info(f"Trailer → {final_path}")
    log.info(f"Sidecar → {sidecar_path}")
    return final_path
