"""Trailer mode — pick 4-5 quotable sentences from across an episode and
splice them into one short reel with black-frame transitions."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from .llm import LLMProvider, LLMError
from .types import Transcript, Word

log = logging.getLogger("ave.trailer")


# Fallback values used only if a key is missing from cfg.trailer.*
_DEFAULT_GAP_SECONDS = 0.6
_DEFAULT_HEAD_PAD = 0.10
_DEFAULT_TAIL_PAD = 0.10
_DEFAULT_AUDIO_FADE_IN_S = 0.08
_DEFAULT_AUDIO_FADE_OUT_S = 0.30
_DEFAULT_REFINER_WINDOW_S = 10.0
_DEFAULT_SEG_OUT_FPS = 30


def _t_get(cfg: SimpleNamespace, key: str, default):
    tcfg = getattr(cfg, "trailer", None)
    if tcfg is None:
        return default
    return getattr(tcfg, key, default)


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _extract_json_array(raw: str) -> list[dict]:
    """Tolerant JSON-array extractor — handles code fences + trailing prose."""
    try:
        v = json.loads(raw)
        if isinstance(v, list):
            return v
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[\s*\{.*?\}\s*\]", raw, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array found in LLM output:\n{raw[:300]}...")
    return json.loads(m.group(0))


def _extract_json_object(raw: str) -> dict:
    """Tolerant JSON-object extractor."""
    try:
        v = json.loads(raw)
        if isinstance(v, dict):
            return v
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]+\}", raw, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in LLM output:\n{raw[:300]}...")
    return json.loads(m.group(0))


def _fmt_transcript_for_llm(transcript: Transcript) -> str:
    lines = []
    for s in transcript.segments:
        m1, s1 = divmod(int(s.start), 60)
        m2, s2 = divmod(int(s.end), 60)
        lines.append(f"[{m1:02d}:{s1:02d}-{m2:02d}:{s2:02d}] {s.text.strip()}")
    return "\n".join(lines)


def pick_quotables(
    transcript: Transcript,
    provider: LLMProvider,
    cfg: SimpleNamespace,
    video_duration: float,
) -> list[dict]:
    """LLM #1 — pick 4-5 quotable sentences from the full transcript."""
    system_prompt = _load_prompt("trailer_picks.txt")
    transcript_text = _fmt_transcript_for_llm(transcript)

    raw = provider.complete(
        user_prompt=transcript_text,
        system_prompt=system_prompt,
        max_tokens=cfg.llm.max_tokens,
    )
    items = _extract_json_array(raw)

    out: list[dict] = []
    for it in items:
        try:
            s = float(it["start"])
            e = float(it["end"])
            txt = str(it["sentence"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if e <= s or not txt:
            continue
        # LLMs occasionally hallucinate timestamps past source end
        if e > video_duration:
            log.warning(f"Skipping past-source pick ({s:.1f}-{e:.1f}s): {txt!r}")
            continue
        out.append({"start": s, "end": e, "sentence": txt})

    out.sort(key=lambda x: x["start"])
    deduped: list[dict] = []
    for q in out:
        if deduped and q["start"] < deduped[-1]["end"]:
            log.warning(f"Skipping overlapping pick: {q['sentence']!r}")
            continue
        deduped.append(q)
    return deduped


_REFINE_USER_TEMPLATE = """\
Picked sentence:
  "{sentence}"

Rough timestamps: {q_start:.2f}s to {q_end:.2f}s.

Word-by-word context window (each line = "start_time-end_time  word"):

{word_lines}
"""


def derive_cut_bounds(
    quotable: dict, transcript: Transcript, cfg: SimpleNamespace,
) -> tuple[float, float]:
    """Mechanical fallback bounds when the refinement LLM call fails."""
    head_pad = _t_get(cfg, "head_pad", _DEFAULT_HEAD_PAD)
    tail_pad = _t_get(cfg, "tail_pad", _DEFAULT_TAIL_PAD)

    q_start = quotable["start"]
    q_end = quotable["end"]

    candidates: list[Word] = []
    for seg in transcript.segments:
        if seg.end < q_start - 0.5 or seg.start > q_end + 0.5:
            continue
        candidates.extend(seg.words)

    in_window = [
        w for w in candidates
        if q_start - 0.1 <= w.start <= q_end + 0.1
    ]
    if not in_window:
        return max(0.0, q_start - head_pad), q_end + tail_pad
    return max(0.0, in_window[0].start - head_pad), in_window[-1].end + tail_pad


def refine_cut_bounds_with_llm(
    quotable: dict,
    transcript: Transcript,
    provider: LLMProvider,
    cfg: SimpleNamespace,
) -> tuple[float, float, str]:
    """LLM #2 — pick exact first/last word for one sentence's cut. Returns
    (cut_start, cut_end, refined_sentence); falls back to `derive_cut_bounds`
    on any failure."""
    window_s = _t_get(cfg, "refiner_window_s", _DEFAULT_REFINER_WINDOW_S)
    head_pad = _t_get(cfg, "head_pad", _DEFAULT_HEAD_PAD)
    tail_pad = _t_get(cfg, "tail_pad", _DEFAULT_TAIL_PAD)

    q_start = quotable["start"]
    q_end = quotable["end"]
    sentence = quotable["sentence"]

    win_lo = max(0.0, q_start - window_s)
    win_hi = q_end + window_s
    win_words: list[Word] = []
    for seg in transcript.segments:
        if seg.end < win_lo or seg.start > win_hi:
            continue
        for w in seg.words:
            if win_lo <= w.start <= win_hi:
                win_words.append(w)

    if len(win_words) < 3:
        log.warning(f"Refiner window too sparse, falling back: {sentence[:60]!r}")
        cs, ce = derive_cut_bounds(quotable, transcript, cfg)
        return cs, ce, sentence

    word_lines = "\n".join(
        f"  {w.start:7.2f}-{w.end:7.2f}  {w.text}"
        for w in win_words
    )

    user_prompt = _REFINE_USER_TEMPLATE.format(
        sentence=sentence,
        q_start=q_start,
        q_end=q_end,
        word_lines=word_lines,
    )
    system_prompt = _load_prompt("trailer_refiner.txt")

    try:
        raw = provider.complete(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=300,
        )
        data = _extract_json_object(raw)
        new_start = float(data["start"])
        new_end = float(data["end"])
    except (LLMError, ValueError, KeyError, json.JSONDecodeError) as e:
        log.warning(f"Refiner failed ({e}): falling back. {sentence[:60]!r}")
        cs, ce = derive_cut_bounds(quotable, transcript, cfg)
        return cs, ce, sentence

    first_word = min(win_words, key=lambda w: abs(w.start - new_start))
    last_word = min(win_words, key=lambda w: abs(w.end - new_end))

    if last_word.end <= first_word.start:
        log.warning(f"Refiner returned inverted range; falling back. {sentence[:60]!r}")
        cs, ce = derive_cut_bounds(quotable, transcript, cfg)
        return cs, ce, sentence

    span_words = [
        w for w in win_words
        if first_word.start <= w.start and w.end <= last_word.end
    ]
    refined_sentence = " ".join(w.text.strip() for w in span_words)

    return (
        max(0.0, first_word.start - head_pad),
        last_word.end + tail_pad,
        refined_sentence,
    )


def _ffmpeg(args: list[str], desc: str) -> None:
    log.debug(f"ffmpeg: {desc}")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *args],
            check=True, capture_output=True, text=True, timeout=600,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed ({desc}): {e.stderr.strip()[-500:]}"
        ) from e


def cut_segment(
    video: Path, cut_start: float, cut_end: float, out: Path, cfg: SimpleNamespace,
) -> None:
    """ffmpeg cut [cut_start, cut_end] of `video` into `out`."""
    _ffmpeg([
        "-ss", f"{cut_start:.3f}",
        "-i", str(video),
        "-t", f"{cut_end - cut_start:.3f}",
        "-c:v", cfg.clip_extract.video_codec,
        "-crf", str(cfg.clip_extract.crf),
        "-preset", cfg.clip_extract.preset,
        "-c:a", "aac", "-b:a", "192k",
        str(out),
    ], desc=f"cut {cut_start:.2f}-{cut_end:.2f}")


def _probe_duration(p: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(p)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return float(out.stdout.strip())


def concat_with_black_gaps(
    clips: list[Path], out: Path, cfg: SimpleNamespace,
) -> None:
    """Concatenate cropped clips with a black + silent gap between each, using
    ffmpeg's concat filter (one re-encode pass). Each clip's audio gets a
    small fade-in + longer fade-out so the cut into the next gap doesn't
    sound mid-thought."""
    gap_s = _t_get(cfg, "gap_seconds", _DEFAULT_GAP_SECONDS)
    fade_in_s = _t_get(cfg, "audio_fade_in_s", _DEFAULT_AUDIO_FADE_IN_S)
    fade_out_s = _t_get(cfg, "audio_fade_out_s", _DEFAULT_AUDIO_FADE_OUT_S)
    rate = _t_get(cfg, "seg_out_fps", _DEFAULT_SEG_OUT_FPS)

    tgt_w = cfg.crop.target_width
    tgt_h = cfg.crop.target_height
    sr = 48000

    n_clips = len(clips)
    if n_clips == 0:
        raise ValueError("concat_with_black_gaps: no clips provided")

    inputs: list[str] = []
    for c in clips:
        inputs += ["-i", str(c)]

    clip_durations = [_probe_duration(c) for c in clips]

    gap_count = max(0, n_clips - 1)
    for _ in range(gap_count):
        inputs += [
            "-f", "lavfi", "-t", f"{gap_s:.3f}",
            "-i", f"color=black:size={tgt_w}x{tgt_h}:rate={rate}",
            "-f", "lavfi", "-t", f"{gap_s:.3f}",
            "-i", f"anullsrc=channel_layout=stereo:sample_rate={sr}",
        ]

    filter_parts: list[str] = []
    concat_inputs: list[str] = []
    for i in range(n_clips):
        dur = clip_durations[i]
        fade_out_start = max(0.0, dur - fade_out_s)
        filter_parts.append(
            f"[{i}:v]scale={tgt_w}:{tgt_h}:force_original_aspect_ratio=decrease,"
            f"pad={tgt_w}:{tgt_h}:(ow-iw)/2:(oh-ih)/2,fps={rate},"
            f"format=yuv420p,setsar=1[v{i}];"
            f"[{i}:a]aresample={sr}:async=1,"
            f"afade=t=in:st=0:d={fade_in_s},"
            f"afade=t=out:st={fade_out_start:.3f}:d={fade_out_s}[a{i}];"
        )
        concat_inputs.append(f"[v{i}][a{i}]")
        if i < n_clips - 1:
            v_idx = n_clips + 2 * i
            a_idx = n_clips + 2 * i + 1
            concat_inputs.append(f"[{v_idx}:v][{a_idx}:a]")

    n_streams = 2 * n_clips - 1
    filter_complex = (
        "".join(filter_parts)
        + "".join(concat_inputs)
        + f"concat=n={n_streams}:v=1:a=1[outv][outa]"
    )

    _ffmpeg([
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", cfg.crop.ffmpeg_encoder,
        "-crf", str(cfg.crop.ffmpeg_crf),
        "-preset", cfg.crop.ffmpeg_preset,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(out),
    ], desc=f"concat {n_clips} clips + {gap_count} gaps")


def build_trailer_words(
    picks: list[dict],
    pick_second_pass_words: list[list[Word]],
    cfg: SimpleNamespace,
) -> list[Word]:
    """Remap each pick's clip-local 2nd-pass words into trailer time,
    accounting for GAP_SECONDS of black between picks."""
    gap_s = _t_get(cfg, "gap_seconds", _DEFAULT_GAP_SECONDS)

    out: list[Word] = []
    cum = 0.0
    for i, q in enumerate(picks):
        clip_duration = q["cut_end"] - q["cut_start"]
        for w in pick_second_pass_words[i]:
            if w.start < 0 or w.start > clip_duration:
                continue
            out.append(Word(
                start=cum + w.start,
                end=cum + min(w.end, clip_duration),
                text=w.text,
                confidence=w.confidence,
            ))
        cum += clip_duration + gap_s
    return out
