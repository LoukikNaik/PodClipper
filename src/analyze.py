"""Stage 4: LLM-driven reel moment detection.

Sends the compact timestamped transcript to an LLM provider, parses the
returned JSON, validates + clamps the results, and returns a list of Clip
objects sorted by LLM-reported hook_score.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

from .llm import LLMError, LLMProvider
from .transcribe import transcript_to_timestamped_text
from .types import Clip, Transcript

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
    """Pull a JSON array out of LLM output. Tolerant of code fences or prose."""
    text = text.strip()

    # Strip markdown code fence if present
    fence_match = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Find the first '[' and last ']' and parse that slice — tolerates
    # leading/trailing prose the LLM might sneak in.
    first = text.find("[")
    last = text.rfind("]")
    if first == -1 or last == -1 or last < first:
        raise AnalyzeError(f"no JSON array found in LLM output: {text[:200]!r}")
    candidate = text[first : last + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise AnalyzeError(f"could not parse JSON array: {e}; snippet: {candidate[:200]!r}") from e
    if not isinstance(parsed, list):
        raise AnalyzeError(f"expected JSON array, got {type(parsed).__name__}")
    return parsed


def _coerce_clip(raw: dict, video_duration: float, min_s: float, max_s: float) -> Clip | None:
    """Validate a single raw clip dict; return Clip or None if invalid/unsalvageable."""
    try:
        start = float(raw["start"])
        end = float(raw["end"])
    except (KeyError, TypeError, ValueError):
        log.warning(f"dropping clip with invalid start/end: {raw!r}")
        return None

    # Clamp to video bounds
    start = max(0.0, min(start, video_duration))
    end = max(0.0, min(end, video_duration))

    if end <= start:
        log.warning(f"dropping clip with end<=start: {raw!r}")
        return None

    duration = end - start
    # Enforce length bounds loosely — clamp extremes, drop if still wrong
    if duration < min_s * 0.5 or duration > max_s * 1.5:
        log.warning(f"dropping clip with out-of-bounds duration {duration:.1f}s: {raw!r}")
        return None

    return Clip(
        start=start,
        end=end,
        title=str(raw.get("title", "")).strip()[:80] or f"Clip {start:.0f}-{end:.0f}",
        reason=str(raw.get("reason", "")).strip()[:300],
        hook_score=float(raw.get("hook_score", 0.5)),
    )


def analyze_for_reels(
    transcript: Transcript,
    video_duration: float,
    provider: LLMProvider,
    cfg: SimpleNamespace,
) -> list[Clip]:
    """Ask the LLM to pick reel-worthy clips from the transcript."""
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
    clips: list[Clip] = []
    for raw in raw_clips:
        if not isinstance(raw, dict):
            log.warning(f"dropping non-dict entry: {raw!r}")
            continue
        c = _coerce_clip(raw, video_duration, min_s, max_s)
        if c is not None:
            clips.append(c)

    # Sort by hook_score desc; cap to target if LLM over-delivered
    clips.sort(key=lambda c: c.hook_score, reverse=True)
    if len(clips) > target:
        log.info(f"LLM returned {len(clips)} clips; keeping top {target}")
        clips = clips[:target]

    log.info(f"Analyze complete: {len(clips)} clips selected")
    for i, c in enumerate(clips, 1):
        log.info(f"  #{i} [{c.start:.1f}-{c.end:.1f}] ({c.duration:.1f}s, score={c.hook_score:.2f}) {c.title}")
    return clips
