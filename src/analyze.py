"""LLM-driven reel moment detection: pick clips, snap to natural pauses,
refine word-precise bounds, shorten titles."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

from .llm import LLMError, LLMProvider
from .transcribe import transcript_to_timestamped_text
from .types import Clip, Transcript, TranscriptSegment, Word

log = logging.getLogger("ave.analyze")

_PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "prompts" / "reel_detector.txt"


class AnalyzeError(Exception):
    pass


def _load_prompt(min_s: int, max_s: int, target: int) -> str:
    template = _PROMPT_TEMPLATE_PATH.read_text()
    return (
        template
        .replace("{{MIN}}", str(min_s))
        .replace("{{MAX}}", str(max_s))
        .replace("{{TARGET}}", str(target))
    )


def _extract_json_array(text: str) -> list:
    """Pull a JSON array out of LLM output. Tolerates code fences and trailing prose."""
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    first = text.find("[")
    if first == -1:
        raise AnalyzeError(f"no JSON array found in LLM output: {text[:200]!r}")
    # raw_decode stops at end of first valid value, ignoring trailing content
    try:
        parsed, _end = json.JSONDecoder().raw_decode(text[first:])
    except json.JSONDecodeError as e:
        raise AnalyzeError(
            f"could not parse JSON array: {e}; snippet: {text[first:first+300]!r}"
        ) from e
    if not isinstance(parsed, list):
        raise AnalyzeError(f"expected JSON array, got {type(parsed).__name__}")
    return parsed


_TITLE_HARD_MAX = 38
_TITLE_REWRITE_SYSTEM = """You rewrite video reel titles to be scroll-stopping hooks.

You will receive the existing title, the clip's transcript, and an editor note.
Read the transcript and base the new title on what was ACTUALLY said.

Output: ONE rewritten title on a single line. No quotes, no prose.

Rules:
- 38 characters max. 3-7 words.
- Write like a viral TikTok/Reel hook, not an article headline.
- Address the viewer ("you/your") when it fits.
- Lead with a number, authority, or action verb.
- Create a curiosity gap — withhold the answer.

Use one of these templates:
  "The #1 [thing] for [outcome]"       → "The #1 Rule for Inner Peace"
  "Why [surprising/painful thing]"     → "Why Meditation Makes It Worse"
  "Stop [action] if you want [goal]"   → "Stop Chasing Happiness"
  "This is why you [feel/do X]"        → "This Is Why You Feel Empty"
  "[Number] [things] that [outcome]"   → "3 Signs You're Spiritually Lost"
  "[Counterintuitive claim]"           → "Your Best Trait Is Destroying You"
  "[Vivid specific detail]"            → "10,000kg Bells Made of Pure Gold"

REJECTED: vague summaries, "You Won't Believe This", passive voice, fragments.
"""


def _normalize_title(title: str) -> str:
    return " ".join(title.split()).rstrip(" .,!?;:")


def _transcript_excerpt(transcript: Transcript, clip: Clip, max_chars: int = 3000) -> str:
    """Return the plain-text transcript of the clip's time range, trimmed."""
    parts: list[str] = []
    for seg in transcript.segments:
        if seg.end <= clip.start or seg.start >= clip.end:
            continue
        text = seg.text.strip()
        if text:
            parts.append(text)
    joined = " ".join(parts)
    if len(joined) > max_chars:
        joined = joined[:max_chars].rsplit(" ", 1)[0] + " …"
    return joined


def _rewrite_title_with_llm(
    long_title: str,
    transcript_excerpt: str,
    reason: str,
    provider: LLMProvider,
) -> str:
    """Ask the LLM for a short, self-contained title; returns the original on failure."""
    user_prompt = (
        f"Existing title: {long_title}\n\n"
        f"Clip transcript:\n{transcript_excerpt}\n\n"
        f"Editor's note: {reason}\n\n"
        f"Rewrite the title:"
    )
    try:
        response = provider.complete(
            user_prompt=user_prompt,
            system_prompt=_TITLE_REWRITE_SYSTEM,
            max_tokens=64,
        )
    except LLMError as e:
        log.warning(f"title rewrite failed ({e}); keeping original")
        return long_title

    candidate = ""
    for line in response.splitlines():
        line = line.strip().lstrip("-*•").strip().strip('"').strip("'")
        if line:
            candidate = line
            break
    candidate = _normalize_title(candidate)
    if not candidate or len(candidate) > _TITLE_HARD_MAX:
        log.warning(f"title rewrite still too long ({len(candidate)} chars): {candidate!r}")
        return long_title
    return candidate


def _snap_to_segment_boundaries(
    start: float,
    end: float,
    segments: list[TranscriptSegment],
    tolerance_s: float = 3.0,
) -> tuple[float, float]:
    """Snap (start, end) to the nearest transcript-segment boundaries within tolerance."""
    if not segments:
        return start, end

    starts = [s.start for s in segments]
    ends = [s.end for s in segments]

    nearest_start = min(starts, key=lambda s: abs(s - start))
    if abs(nearest_start - start) <= tolerance_s:
        start = nearest_start

    nearest_end = min(ends, key=lambda e: abs(e - end))
    if abs(nearest_end - end) <= tolerance_s:
        end = nearest_end

    return start, end


_REEL_REFINER_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "reel_refiner.txt"
)


def _load_reel_refiner_prompt() -> str:
    return _REEL_REFINER_PROMPT_PATH.read_text(encoding="utf-8")


_REEL_REFINE_USER_TEMPLATE = """\
Clip title:
  "{title}"

Why this clip was picked (the upstream selector's note):
  "{reason}"

Current rough bounds: {current_start:.2f}s to {current_end:.2f}s
(equivalent to indices [{current_first_idx}..{current_last_idx}] in the word list below).

Word list (every word, with index | start_time-end_time | text).
  - Indices before [{current_first_idx}] are CONTEXT BEFORE the clip.
  - Indices [{current_first_idx}..{current_last_idx}] are INSIDE the clip today.
  - Indices after [{current_last_idx}] are CONTEXT AFTER the clip.

{word_lines}
"""


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


def refine_clip_bounds_with_llm(
    clip: Clip,
    transcript: Transcript,
    provider: LLMProvider,
    cfg: SimpleNamespace,
    window_s: float = 10.0,
) -> tuple[float, float, dict]:
    """LLM picks word-precise first/last indices over a clip+context word list.
    Returns (refined_start, refined_end, trace); falls back to original bounds on failure."""
    win_lo = max(0.0, clip.start - window_s)
    win_hi = clip.end + window_s

    words: list[Word] = []
    for seg in transcript.segments:
        if seg.end < win_lo or seg.start > win_hi:
            continue
        for w in seg.words:
            if win_lo <= w.start <= win_hi:
                words.append(w)

    trace: dict = {
        "title": clip.title,
        "input_clip": {"start": clip.start, "end": clip.end, "reason": clip.reason},
        "window": {
            "win_lo": win_lo, "win_hi": win_hi, "num_words": len(words),
            "first_word": words[0].text if words else None,
            "last_word": words[-1].text if words else None,
        },
    }

    if len(words) < 8:
        log.warning(f"Refiner window sparse for {clip.title!r}; keeping bounds")
        trace["outcome"] = "sparse_window_fallback"
        return clip.start, clip.end, trace

    current_first_idx = min(
        range(len(words)), key=lambda i: abs(words[i].start - clip.start)
    )
    current_last_idx = min(
        range(len(words)), key=lambda i: abs(words[i].end - clip.end)
    )
    trace["current_indices"] = {
        "first_idx": current_first_idx,
        "last_idx": current_last_idx,
        "first_word": words[current_first_idx].text,
        "last_word": words[current_last_idx].text,
    }

    word_lines = "\n".join(
        f"  [{i:4d}] {w.start:7.2f}-{w.end:7.2f}  {w.text}"
        for i, w in enumerate(words)
    )

    user_prompt = _REEL_REFINE_USER_TEMPLATE.format(
        title=clip.title,
        reason=clip.reason,
        current_start=clip.start,
        current_end=clip.end,
        current_first_idx=current_first_idx,
        current_last_idx=current_last_idx,
        word_lines=word_lines,
    )
    system_prompt = _load_reel_refiner_prompt()
    trace["llm_user_prompt"] = user_prompt

    try:
        raw = provider.complete(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=200,
        )
        trace["llm_raw_response"] = raw
        log.debug(f"Refiner LLM raw response: {raw[:300]!r}")
        data = _extract_json_object(raw)
        first_idx = int(data["first_word_idx"])
        last_idx = int(data["last_word_idx"])
        log.info(
            f"Refiner picked words [{first_idx}..{last_idx}] for {clip.title!r} "
            f"(was [{current_first_idx}..{current_last_idx}])"
        )
    except (LLMError, ValueError, KeyError, json.JSONDecodeError) as e:
        log.warning(f"Reel refiner failed ({e}); keeping bounds for {clip.title!r}")
        trace["outcome"] = f"llm_failed: {e}"
        return clip.start, clip.end, trace

    if not (0 <= first_idx <= last_idx < len(words)):
        log.warning(
            f"Reel refiner returned out-of-range indices "
            f"({first_idx}..{last_idx} of {len(words)}); keeping bounds for {clip.title!r}"
        )
        trace["outcome"] = f"out_of_range: [{first_idx}..{last_idx}] of {len(words)}"
        return clip.start, clip.end, trace

    trace["llm_picked_indices"] = {
        "first_idx": first_idx, "last_idx": last_idx,
        "first_word": words[first_idx].text, "last_word": words[last_idx].text,
    }
    trace["outcome"] = "llm_picked"
    trace["final_bounds"] = {
        "start": words[first_idx].start, "end": words[last_idx].end,
    }

    return words[first_idx].start, words[last_idx].end, trace


def _coerce_clip(raw: dict, video_duration: float, min_s: float, max_s: float) -> Clip | None:
    """Validate a raw clip dict; return Clip or None if unsalvageable."""
    try:
        start = float(raw["start"])
        end = float(raw["end"])
    except (KeyError, TypeError, ValueError):
        log.warning(f"dropping clip with invalid start/end: {raw!r}")
        return None

    start = max(0.0, min(start, video_duration))
    end = max(0.0, min(end, video_duration))

    if end <= start:
        log.warning(f"dropping clip with end<=start: {raw!r}")
        return None

    duration = end - start
    if duration < min_s * 0.5 or duration > max_s * 1.5:
        log.warning(f"dropping clip with out-of-bounds duration {duration:.1f}s: {raw!r}")
        return None

    title = _normalize_title(str(raw.get("title", ""))) or f"Clip {start:.0f}-{end:.0f}"

    return Clip(
        start=start,
        end=end,
        title=title,
        reason=str(raw.get("reason", "")).strip()[:300],
        hook_score=float(raw.get("hook_score", 0.5)),
    )


def _dump_debug(cache_dir, name: str, payload) -> None:
    """Write a debug JSON to cache_dir/analyze/; no-op if cache_dir is None."""
    if cache_dir is None:
        return
    out_dir = Path(cache_dir) / "analyze"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _clip_to_dict(c: "Clip") -> dict:
    return {
        "start": c.start, "end": c.end, "duration": c.duration,
        "title": c.title, "reason": c.reason, "hook_score": c.hook_score,
    }


def analyze_for_reels(
    transcript: Transcript,
    video_duration: float,
    provider: LLMProvider,
    cfg: SimpleNamespace,
    debug_cache_dir: "Path | None" = None,
) -> list[Clip]:
    """Ask the LLM to pick reel-worthy clips, snap to pauses, refine to word
    boundaries, then shorten over-long titles. Dumps per-stage JSON under
    `<debug_cache_dir>/analyze/` when provided."""
    min_s = cfg.analyze.min_clip_seconds
    max_s = cfg.analyze.max_clip_seconds
    target = cfg.analyze.target_clips

    system_prompt = _load_prompt(min_s, max_s, target)
    user_prompt = transcript_to_timestamped_text(
        transcript,
        resolution=cfg.analyze.transcript_timestamp_resolution,
    )
    if not user_prompt.strip():
        raise AnalyzeError("empty transcript — nothing to analyze")

    log.info(
        f"Sending transcript to LLM ({provider.name}): "
        f"{len(transcript.segments)} segments, "
        f"~{len(user_prompt)} chars"
    )
    try:
        response = provider.complete(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=cfg.llm.max_tokens,
        )
    except LLMError as e:
        raise AnalyzeError(f"LLM call failed: {e}") from e

    log.debug(f"LLM raw response: {response[:500]}")

    raw_clips = _extract_json_array(response)
    _dump_debug(debug_cache_dir, "01_raw_picks.json", {
        "llm_raw_response": response,
        "parsed_clips": raw_clips,
    })

    clips: list[Clip] = []
    snap_log: list[dict] = []
    for raw in raw_clips:
        if not isinstance(raw, dict):
            log.warning(f"dropping non-dict entry: {raw!r}")
            continue
        c = _coerce_clip(raw, video_duration, min_s, max_s)
        if c is None:
            continue

        new_start, new_end = _snap_to_segment_boundaries(
            c.start, c.end, transcript.segments, tolerance_s=3.0,
        )
        snap_log.append({
            "title": c.title,
            "before": {"start": c.start, "end": c.end},
            "after": {"start": new_start, "end": new_end},
            "changed": (new_start, new_end) != (c.start, c.end),
        })
        if (new_start, new_end) != (c.start, c.end):
            log.debug(
                f"Snapped clip boundaries: [{c.start:.2f}, {c.end:.2f}] → "
                f"[{new_start:.2f}, {new_end:.2f}]"
            )
            c = Clip(start=new_start, end=new_end, title=c.title, reason=c.reason, hook_score=c.hook_score)
        clips.append(c)

    clips.sort(key=lambda c: c.hook_score, reverse=True)
    if len(clips) > target:
        log.info(f"LLM returned {len(clips)} clips; keeping top {target}")
        clips = clips[:target]

    _dump_debug(debug_cache_dir, "02_snapped_and_capped.json", {
        "snap_log": snap_log,
        "after_cap": [_clip_to_dict(c) for c in clips],
    })

    refiner_traces: list[dict] = []
    if bool(getattr(cfg.analyze, "refine_bounds", True)):
        window_s = float(getattr(cfg.analyze, "refiner_window_s", 15.0))
        log.info(f"Refining cut bounds for {len(clips)} clips via LLM...")
        for i, c in enumerate(clips):
            rs, re_, trace = refine_clip_bounds_with_llm(
                c, transcript, provider, cfg, window_s=window_s,
            )
            refiner_traces.append(trace)
            if (rs, re_) != (c.start, c.end):
                log.info(
                    f"Refined [{c.start:.2f}-{c.end:.2f}] → "
                    f"[{rs:.2f}-{re_:.2f}]  {c.title!r}"
                )
                clips[i] = Clip(
                    start=rs, end=re_,
                    title=c.title, reason=c.reason, hook_score=c.hook_score,
                )

    _dump_debug(debug_cache_dir, "03_refined.json", {
        "traces": refiner_traces,
        "after_refine": [_clip_to_dict(c) for c in clips],
    })

    title_rewrites: list[dict] = []
    for i, c in enumerate(clips):
        if len(c.title) > _TITLE_HARD_MAX:
            excerpt = _transcript_excerpt(transcript, c)
            short = _rewrite_title_with_llm(c.title, excerpt, c.reason, provider)
            title_rewrites.append({"from": c.title, "to": short, "from_len": len(c.title), "to_len": len(short)})
            if short != c.title:
                log.info(f"Rewrote title ({len(c.title)}→{len(short)} chars): {c.title!r} → {short!r}")
                clips[i] = Clip(
                    start=c.start, end=c.end,
                    title=short, reason=c.reason, hook_score=c.hook_score,
                )

    _dump_debug(debug_cache_dir, "04_final.json", {
        "title_rewrites": title_rewrites,
        "final_clips": [_clip_to_dict(c) for c in clips],
    })

    log.info(f"Analyze complete: {len(clips)} clips selected")
    for i, c in enumerate(clips, 1):
        log.info(f"  #{i} [{c.start:.1f}-{c.end:.1f}] ({c.duration:.1f}s, score={c.hook_score:.2f}) {c.title}")
    return clips
