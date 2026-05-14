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
from .types import Clip, Transcript, TranscriptSegment

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
    """Return the plain-text transcript of the clip's time range.

    Includes segments that overlap [clip.start, clip.end]. Trimmed to
    `max_chars` so pathologically long clips don't blow up the prompt.
    """
    parts: list[str] = []
    for seg in transcript.segments:
        # Keep any segment with meaningful overlap with the clip
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
    """Ask the LLM to produce a short, self-contained title, using the
    actual clip transcript as grounding. Returns the rewritten title, or
    the original if the rewrite fails or can't meet the length budget.
    """
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

    # Take the first non-empty line, strip quotes/bullets the LLM might have added
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
    """Snap (start, end) to the nearest transcript-segment boundaries within tolerance.

    This fixes the common failure mode where the LLM picks a clean start/end
    but the numeric value lands mid-segment (e.g. rounded to the nearest 5s).
    By snapping to segment edges, our cuts land at natural speech pauses.

    If no segment boundary is within `tolerance_s` of the request, we leave
    the original value alone (the LLM may have had a reason).
    """
    if not segments:
        return start, end

    starts = [s.start for s in segments]
    ends = [s.end for s in segments]

    # Snap start to the nearest segment START (we want to begin at the top of a segment)
    nearest_start = min(starts, key=lambda s: abs(s - start))
    if abs(nearest_start - start) <= tolerance_s:
        start = nearest_start

    # Snap end to the nearest segment END (we want to end on a natural pause)
    nearest_end = min(ends, key=lambda e: abs(e - end))
    if abs(nearest_end - end) <= tolerance_s:
        end = nearest_end

    return start, end


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

    title = _normalize_title(str(raw.get("title", ""))) or f"Clip {start:.0f}-{end:.0f}"

    return Clip(
        start=start,
        end=end,
        title=title,
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
        if c is None:
            continue

        # Snap boundaries to the transcript so cuts land on natural pauses.
        new_start, new_end = _snap_to_segment_boundaries(
            c.start, c.end, transcript.segments, tolerance_s=3.0,
        )
        if (new_start, new_end) != (c.start, c.end):
            log.debug(
                f"Snapped clip boundaries: [{c.start:.2f}, {c.end:.2f}] → "
                f"[{new_start:.2f}, {new_end:.2f}]"
            )
            c = Clip(start=new_start, end=new_end, title=c.title, reason=c.reason, hook_score=c.hook_score)
        clips.append(c)

    # Sort by hook_score desc; cap to target if LLM over-delivered
    clips.sort(key=lambda c: c.hook_score, reverse=True)
    if len(clips) > target:
        log.info(f"LLM returned {len(clips)} clips; keeping top {target}")
        clips = clips[:target]

    # If any titles exceeded the overlay's character budget, ask the LLM to
    # rewrite them — truncating mid-sentence produces bad standalone titles.
    # Ground the rewrite in the actual clip transcript, not just the editor note.
    for i, c in enumerate(clips):
        if len(c.title) > _TITLE_HARD_MAX:
            excerpt = _transcript_excerpt(transcript, c)
            short = _rewrite_title_with_llm(c.title, excerpt, c.reason, provider)
            if short != c.title:
                log.info(f"Rewrote title ({len(c.title)}→{len(short)} chars): {c.title!r} → {short!r}")
                clips[i] = Clip(
                    start=c.start, end=c.end,
                    title=short, reason=c.reason, hook_score=c.hook_score,
                )

    log.info(f"Analyze complete: {len(clips)} clips selected")
    for i, c in enumerate(clips, 1):
        log.info(f"  #{i} [{c.start:.1f}-{c.end:.1f}] ({c.duration:.1f}s, score={c.hook_score:.2f}) {c.title}")
    return clips
