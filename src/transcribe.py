"""Stage 3: Transcription (two-pass strategy).

First pass — `transcribe_first_pass(audio_path, cfg)`:
  Split whole-video audio into overlapping chunks, transcribe in parallel
  threads with a shared faster-whisper model (CTranslate2 releases the GIL),
  merge results via simple timestamp dedup. Coarse accuracy is fine — output
  feeds the LLM for clip selection.

Second pass — `transcribe_second_pass(clip_audio, cfg)`:
  Single-shot high-quality transcription of an already-extracted short clip.
  No chunking, clip-relative timestamps. Output feeds the subtitle burner.

faster-whisper accepts numpy arrays directly, so we load audio once and slice
in-memory for parallel chunk workers — no per-chunk WAV files on disk.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np

from .audio import ChunkRange, plan_chunks
from .types import Transcript, TranscriptSegment, Word

log = logging.getLogger("ave.transcribe")

_WHISPER_SAMPLE_RATE = 16000  # faster-whisper's internal sample rate


# ---------- Model loading ----------

# We lazy-load WhisperModel per (model_size, compute_type, device) key so first-pass
# and second-pass models coexist without re-loading on every call.
_model_cache: dict[tuple, object] = {}
_model_cache_lock = threading.Lock()


def _resolve_device(requested: str) -> str:
    """Map 'auto' to the best available backend for faster-whisper.

    faster-whisper uses CTranslate2 which supports 'cpu' and 'cuda' (no MPS).
    On Apple Silicon, 'cpu' with int8 quantization is the pragmatic choice.
    """
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

    # CPU only reliably supports int8/float32 in CTranslate2. If a GPU-flavored
    # type is requested but we're on CPU, fall back to int8 silently.
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


# ---------- Audio decoding ----------

def _decode_audio_to_float32(audio_path: Path) -> np.ndarray:
    """Decode any audio file to 16kHz mono float32 PCM numpy array via ffmpeg.

    This matches faster-whisper's expected input format. We bypass pydub/librosa
    to avoid extra deps; ffmpeg is already a hard dependency.
    """
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


# ---------- Core transcription (per chunk) ----------

def _transcribe_array(
    model,
    audio: np.ndarray,
    *,
    language: str | None,
    beam_size: int,
    word_timestamps: bool,
    time_offset: float = 0.0,
) -> tuple[list[TranscriptSegment], str]:
    """Run faster-whisper on an in-memory audio array; return segments + detected language.

    Timestamps are shifted by `time_offset` seconds so they're video-relative
    (for first-pass chunks) or clip-relative (for second-pass, offset=0).
    """
    segments_iter, info = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        vad_filter=False,  # keep simple; VAD can drop short utterances we want
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


# ---------- First pass (whole video, parallel chunks) ----------

def _merge_segments(
    per_chunk_segments: list[list[TranscriptSegment]],
) -> list[TranscriptSegment]:
    """Merge per-chunk segments into a single timeline, dropping duplicates
    from overlap regions.

    Rule: a segment from chunk N+1 is dropped if its start is before the last
    accepted segment's end. Same rule applies to words inside each segment.
    This is the "simple timestamp dedup" we agreed on for the first pass —
    accuracy is coarse anyway, and the LLM doesn't care about boundary noise.
    """
    merged: list[TranscriptSegment] = []
    last_end = -1.0
    for chunk_segs in per_chunk_segments:
        for seg in chunk_segs:
            if seg.start < last_end - 0.1:  # small tolerance for timestamp jitter
                continue
            # Filter out words that fall before last_end
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


# ---------- Second pass (per clip, single shot, high-quality) ----------

def transcribe_second_pass(
    clip_audio_or_video: Path,
    cfg: SimpleNamespace,
) -> list[Word]:
    """Re-transcribe a single extracted clip with a high-quality model.

    Accepts either a video or audio file — ffmpeg decodes whatever's fed in.
    Returns word-level timestamps, clip-relative (t=0 is clip start).
    """
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


# ---------- Formatting helpers for LLM prompt ----------

def transcript_to_timestamped_text(
    transcript: Transcript,
    resolution: str = "seconds",
) -> str:
    """Compact [MM:SS] text format for LLM consumption."""
    lines: list[str] = []
    for seg in transcript.segments:
        if not seg.text.strip():
            continue
        t = seg.start
        if resolution == "seconds":
            stamp = f"{int(t // 60):02d}:{int(t % 60):02d}"
        else:
            stamp = f"{t:.2f}"
        lines.append(f"[{stamp}] {seg.text.strip()}")
    return "\n".join(lines)
