"""Two-pass Whisper transcription (parallel-chunked first pass, single-shot
high-quality second pass per clip)."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

import numpy as np

from .audio import ChunkRange, plan_chunks
from .types import Transcript, TranscriptSegment, Word

log = logging.getLogger("ave.transcribe")

_WHISPER_SAMPLE_RATE = 16000

_model_cache: dict[tuple, object] = {}
_model_cache_lock = threading.Lock()


def _resolve_device(requested: str) -> str:
    """faster-whisper uses CTranslate2 (cpu/cuda only — no MPS)."""
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _get_model(model_size: str, compute_type: str, device: str):
    from faster_whisper import WhisperModel
    resolved_device = _resolve_device(device)

    # CPU only reliably supports int8/float32 in CTranslate2.
    if resolved_device == "cpu" and compute_type in {"int8_float16", "float16"}:
        log.warning(
            f"compute_type={compute_type} requires a GPU; falling back to int8 on CPU."
        )
        compute_type = "int8"

    key = (model_size, compute_type, resolved_device)
    with _model_cache_lock:
        if key not in _model_cache:
            log.info(f"Loading faster-whisper model: {model_size} ({compute_type} on {resolved_device})")
            _model_cache[key] = WhisperModel(
                model_size,
                device=resolved_device,
                compute_type=compute_type,
            )
        return _model_cache[key]


def _decode_audio_to_float32(audio_path: Path) -> np.ndarray:
    """Decode any audio file to 16kHz mono float32 PCM via ffmpeg."""
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", str(audio_path),
        "-f", "s16le",
        "-ac", "1",
        "-ar", str(_WHISPER_SAMPLE_RATE),
        "-",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True)
    pcm = np.frombuffer(result.stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def _transcribe_array(
    model,
    audio: np.ndarray,
    *,
    language: str | None,
    beam_size: int,
    word_timestamps: bool,
    time_offset: float = 0.0,
) -> tuple[list[TranscriptSegment], str]:
    """Run faster-whisper on an in-memory audio array; timestamps shifted by
    `time_offset` so they're video-relative for chunked first-pass calls."""
    segments_iter, info = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        vad_filter=False,
    )

    segments: list[TranscriptSegment] = []
    for seg in segments_iter:
        words: list[Word] = []
        if word_timestamps and seg.words:
            for w in seg.words:
                words.append(Word(
                    start=float(w.start) + time_offset,
                    end=float(w.end) + time_offset,
                    text=w.word,
                    confidence=float(getattr(w, "probability", 1.0)),
                ))
        segments.append(TranscriptSegment(
            start=float(seg.start) + time_offset,
            end=float(seg.end) + time_offset,
            text=seg.text.strip(),
            words=words,
        ))
    return segments, info.language


def _merge_segments(
    per_chunk_segments: list[list[TranscriptSegment]],
) -> list[TranscriptSegment]:
    """Merge per-chunk segments, dropping duplicates from overlap regions."""
    merged: list[TranscriptSegment] = []
    last_end = -1.0
    for chunk_segs in per_chunk_segments:
        for seg in chunk_segs:
            if seg.start < last_end - 0.1:
                continue
            if seg.words:
                seg = TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    words=[w for w in seg.words if w.start >= last_end - 0.1],
                )
            merged.append(seg)
            last_end = max(last_end, seg.end)
    return merged


def transcribe_first_pass(
    audio_path: Path,
    duration: float,
    cfg: SimpleNamespace,
) -> Transcript:
    """Parallel-chunk transcription of the full video audio for LLM analysis."""
    fp = cfg.transcribe.first_pass
    model = _get_model(fp.model, fp.compute_type, fp.device)
    language = cfg.transcribe.language

    log.info(f"Decoding audio {audio_path.name} for first-pass transcription...")
    audio = _decode_audio_to_float32(audio_path)

    chunks = plan_chunks(
        duration=duration,
        chunk_seconds=cfg.audio.chunk_seconds,
        overlap_seconds=cfg.audio.chunk_overlap_seconds,
    )
    log.info(f"First pass: transcribing {len(chunks)} chunks with {fp.max_workers} workers")

    def _work(chunk: ChunkRange) -> tuple[int, list[TranscriptSegment], str]:
        start_i = int(chunk.start * _WHISPER_SAMPLE_RATE)
        end_i = int(chunk.end * _WHISPER_SAMPLE_RATE)
        slice_ = audio[start_i:end_i]
        segs, lang = _transcribe_array(
            model, slice_,
            language=language,
            beam_size=fp.beam_size,
            word_timestamps=True,
            time_offset=chunk.start,
        )
        log.debug(f"Chunk {chunk.index}: {len(segs)} segments ({chunk.start:.1f}-{chunk.end:.1f}s)")
        return chunk.index, segs, lang

    per_chunk: list[list[TranscriptSegment]] = [[] for _ in chunks]
    detected_language: str | None = None

    with ThreadPoolExecutor(max_workers=fp.max_workers) as ex:
        futures = [ex.submit(_work, c) for c in chunks]
        for fut in as_completed(futures):
            idx, segs, lang = fut.result()
            per_chunk[idx] = segs
            if detected_language is None:
                detected_language = lang

    merged = _merge_segments(per_chunk)
    log.info(
        f"First pass complete: {len(merged)} segments, "
        f"{sum(len(s.words) for s in merged)} words, language={detected_language}"
    )
    return Transcript(language=detected_language or "unknown", segments=merged)


def transcribe_second_pass(
    clip_audio_or_video: Path,
    cfg: SimpleNamespace,
) -> list[Word]:
    """High-quality single-shot transcription of an extracted clip; returns
    word-level timestamps, clip-relative."""
    sp = cfg.transcribe.second_pass
    model = _get_model(sp.model, sp.compute_type, sp.device)

    audio = _decode_audio_to_float32(clip_audio_or_video)
    segs, _ = _transcribe_array(
        model, audio,
        language=cfg.transcribe.language,
        beam_size=sp.beam_size,
        word_timestamps=sp.word_timestamps,
        time_offset=0.0,
    )

    words: list[Word] = []
    for seg in segs:
        words.extend(seg.words)
    log.info(f"Second pass on {clip_audio_or_video.name}: {len(words)} words")
    return words


def _words_to_json(words: list[Word]) -> list[dict]:
    return [{"start": w.start, "end": w.end, "text": w.text, "confidence": w.confidence}
            for w in words]


def _words_from_json(data: list[dict]) -> list[Word]:
    return [Word(start=float(d["start"]), end=float(d["end"]),
                 text=str(d["text"]), confidence=float(d.get("confidence", 1.0)))
            for d in data]


def transcribe_second_pass_cached(
    clip_audio_or_video: Path,
    cache_path: Optional[Path],
    cfg: SimpleNamespace,
) -> list[Word]:
    """Same as `transcribe_second_pass` but caches to `cache_path` as JSON.
    Pass cache_path=None to bypass caching."""
    if cache_path is not None and cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            words = _words_from_json(data)
            log.info(f"Loaded {len(words)} cached words from {cache_path.name}")
            return words
        except Exception as e:  # noqa: BLE001
            log.warning(f"Failed to load words cache {cache_path} ({e}); re-transcribing")

    words = transcribe_second_pass(clip_audio_or_video, cfg)
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(_words_to_json(words), ensure_ascii=False, indent=2))
            log.info(f"Cached {len(words)} words → {cache_path.name}")
        except Exception as e:  # noqa: BLE001
            log.warning(f"Failed to write words cache {cache_path}: {e}")
    return words


def transcript_to_timestamped_text(
    transcript: Transcript,
    resolution: str = "seconds",
) -> str:
    """Compact [start-end] text format for LLM consumption. Each line is one
    speech segment bounded by natural pauses — the LLM can use segment ends
    as candidate clip boundaries."""
    def _fmt(t: float) -> str:
        if resolution == "seconds":
            return f"{int(t // 60):02d}:{int(t % 60):02d}"
        return f"{t:.2f}"

    lines: list[str] = []
    for seg in transcript.segments:
        if not seg.text.strip():
            continue
        lines.append(f"[{_fmt(seg.start)}-{_fmt(seg.end)}] {seg.text.strip()}")
    return "\n".join(lines)
