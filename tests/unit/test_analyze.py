"""Characterization tests for `src/analyze.py` (pure-function layer).

Module under test: LLM-driven reel moment selection + word-precise refinement.

We characterize the pure helpers here — the orchestrator `analyze_for_reels`
and `refine_clip_bounds_with_llm` involve heavy LLM mocking + prompt file
reads + transcript fixtures; deferred to a later session.

Behaviors locked:
  - `_extract_json_array(text)`     — tolerant JSON-array parser (handles
                                      ```json fences, trailing prose,
                                      raw_decode's tolerance for content
                                      after a valid value).
  - `_extract_json_object(text)`    — tolerant JSON-object parser.
  - `_normalize_title(title)`       — whitespace collapse + trailing-punct strip.
  - `_snap_to_segment_boundaries`   — snap (start, end) to nearest transcript
                                      segment boundary within tolerance.
  - `_coerce_clip(raw, ...)`        — raw dict → Clip, drops invalid entries.
  - `_transcript_excerpt`           — extract clip-range text, trim to max chars.
"""

from __future__ import annotations

import pytest

from podclipper.analyze import (
    AnalyzeError,
    _coerce_clip,
    _extract_json_array,
    _extract_json_object,
    _normalize_title,
    _snap_to_segment_boundaries,
    _transcript_excerpt,
)
from podclipper.types import Clip, Transcript, TranscriptSegment, Word


# --------------------------------------------------------------------------- #
# _extract_json_array
# --------------------------------------------------------------------------- #

def test_extract_json_array_parses_raw_json_array() -> None:
    """Plain JSON array → parsed list."""
    assert _extract_json_array('[1, 2, 3]') == [1, 2, 3]


def test_extract_json_array_strips_code_fence_with_json_marker() -> None:
    """A ```json ... ``` fence is stripped before parsing."""
    text = '```json\n[{"a": 1}]\n```'
    assert _extract_json_array(text) == [{"a": 1}]


def test_extract_json_array_strips_bare_code_fence_without_json_marker() -> None:
    """A bare ``` ... ``` fence (no language tag) is also stripped."""
    text = '```\n[1, 2]\n```'
    assert _extract_json_array(text) == [1, 2]


def test_extract_json_array_tolerates_trailing_prose_after_array() -> None:
    """`raw_decode` stops at end of first valid value — trailing chatter ignored."""
    text = '[1, 2, 3] and here is my explanation of why these are good'
    assert _extract_json_array(text) == [1, 2, 3]


def test_extract_json_array_finds_array_after_leading_prose() -> None:
    """Prose before the `[` is skipped — parsing starts at the first `[`."""
    text = 'Here are the clips I picked:\n[1, 2]'
    assert _extract_json_array(text) == [1, 2]


def test_extract_json_array_raises_analyzeerror_when_no_array_in_text() -> None:
    """No `[` anywhere → AnalyzeError with snippet for diagnosis."""
    with pytest.raises(AnalyzeError, match="no JSON array found"):
        _extract_json_array("just some text, no brackets here")


def test_extract_json_array_raises_analyzeerror_on_invalid_json_inside_brackets() -> None:
    """Bracket present but malformed JSON → AnalyzeError 'could not parse'."""
    with pytest.raises(AnalyzeError, match="could not parse"):
        _extract_json_array("[not actually json{")


def test_extract_json_array_returns_empty_list_for_empty_array_input() -> None:
    """Empty JSON array → empty Python list (parses cleanly, not an error)."""
    assert _extract_json_array("[]") == []


# --------------------------------------------------------------------------- #
# _extract_json_object
# --------------------------------------------------------------------------- #

def test_extract_json_object_parses_raw_object() -> None:
    """Plain JSON object → parsed dict."""
    assert _extract_json_object('{"a": 1, "b": 2}') == {"a": 1, "b": 2}


def test_extract_json_object_finds_object_inside_surrounding_prose() -> None:
    """Object embedded in chatter → regex picks the {...}."""
    text = 'Result: {"first_word_idx": 5, "last_word_idx": 12} done.'
    assert _extract_json_object(text) == {"first_word_idx": 5, "last_word_idx": 12}


def test_extract_json_object_raises_valueerror_when_no_object_found() -> None:
    """No `{...}` substring → ValueError mentioning 'No JSON object'."""
    with pytest.raises(ValueError, match="No JSON object"):
        _extract_json_object("nothing here but prose")


def test_extract_json_object_regex_fallback_silently_returns_inner_object_for_nested() -> None:
    """SURPRISE / KNOWN LIMITATION: prose-wrapped responses containing nested
    objects silently return the INNER object — the regex `\\{[^{}]+\\}` only
    matches flat objects (no braces inside), so it skips the outer `{...}`
    (whose body contains braces) and grabs the inner one.

    Acceptable today: all callers expect flat objects
    ({first_word_idx, last_word_idx}, {start, end}). Locking so anyone who
    later tries to extract nested config from an LLM response knows this
    will silently corrupt the result rather than raise."""
    result = _extract_json_object('Result: {"outer": {"inner": 1}} done.')
    assert result == {"inner": 1}  # NOT the outer object!


# --------------------------------------------------------------------------- #
# _normalize_title
# --------------------------------------------------------------------------- #

def test_normalize_title_collapses_internal_whitespace() -> None:
    """Multiple spaces / tabs / newlines become a single space."""
    assert _normalize_title("  hello   world\tfoo\nbar  ") == "hello world foo bar"


def test_normalize_title_strips_trailing_punctuation() -> None:
    """Trailing `. , ! ? ; :` characters are removed (with surrounding spaces)."""
    assert _normalize_title("hello world!!") == "hello world"
    assert _normalize_title("hello world.") == "hello world"
    assert _normalize_title("hello world ;;;") == "hello world"


def test_normalize_title_leaves_internal_punctuation_alone() -> None:
    """`!` and `?` inside the title are preserved — only trailing ones stripped."""
    assert _normalize_title("Why? Because!") == "Why? Because"


# --------------------------------------------------------------------------- #
# _snap_to_segment_boundaries
# --------------------------------------------------------------------------- #

def _seg(start: float, end: float) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text="x", words=[])


def test_snap_to_segment_boundaries_returns_unchanged_when_no_segments() -> None:
    """Empty segment list → bounds passed through."""
    s, e = _snap_to_segment_boundaries(1.0, 2.0, segments=[])
    assert (s, e) == (1.0, 2.0)


def test_snap_to_segment_boundaries_snaps_when_within_tolerance() -> None:
    """A start within `tolerance_s` of a segment-start is snapped to that boundary."""
    segments = [_seg(0.0, 10.0), _seg(10.0, 20.0)]
    s, e = _snap_to_segment_boundaries(10.5, 19.5, segments, tolerance_s=3.0)
    assert s == 10.0
    assert e == 20.0


def test_snap_to_segment_boundaries_leaves_unchanged_when_beyond_tolerance() -> None:
    """Beyond `tolerance_s` of any boundary → original value kept."""
    segments = [_seg(0.0, 10.0), _seg(50.0, 60.0)]
    s, e = _snap_to_segment_boundaries(25.0, 30.0, segments, tolerance_s=3.0)
    assert s == 25.0
    assert e == 30.0


def test_snap_to_segment_boundaries_can_snap_only_one_end() -> None:
    """Independent decisions for start and end — only the in-tolerance end snaps."""
    segments = [_seg(0.0, 10.0), _seg(100.0, 110.0)]
    s, e = _snap_to_segment_boundaries(1.0, 50.0, segments, tolerance_s=3.0)
    assert s == 0.0
    assert e == 50.0


# --------------------------------------------------------------------------- #
# _coerce_clip
# --------------------------------------------------------------------------- #

def test_coerce_clip_returns_clip_for_valid_raw_dict() -> None:
    """Valid input → Clip with all fields populated."""
    raw = {"start": 10.0, "end": 40.0, "title": "Test", "reason": "good", "hook_score": 0.8}
    c = _coerce_clip(raw, video_duration=300.0, min_s=15, max_s=60)
    assert isinstance(c, Clip)
    assert c.start == 10.0
    assert c.end == 40.0
    assert c.title == "Test"
    assert c.hook_score == 0.8


def test_coerce_clip_returns_none_when_end_less_than_or_equal_start() -> None:
    """end <= start → drop the clip."""
    raw = {"start": 10.0, "end": 10.0, "title": "T"}
    assert _coerce_clip(raw, video_duration=300.0, min_s=15, max_s=60) is None


def test_coerce_clip_returns_none_for_missing_or_unparseable_timestamps() -> None:
    """Missing start/end → drop. Non-numeric → drop."""
    assert _coerce_clip({"end": 40}, video_duration=300, min_s=15, max_s=60) is None
    assert _coerce_clip({"start": "abc", "end": 40}, video_duration=300, min_s=15, max_s=60) is None


def test_coerce_clip_clamps_bounds_to_video_duration() -> None:
    """start/end > video_duration get clamped before duration check."""
    raw = {"start": 290.0, "end": 350.0, "title": "T"}
    c = _coerce_clip(raw, video_duration=300.0, min_s=5, max_s=60)
    assert c is not None
    assert c.end == 300.0


def test_coerce_clip_drops_clip_with_duration_below_half_of_min_s() -> None:
    """Duration < min_s * 0.5 → drop ('clearly not a clip')."""
    raw = {"start": 10.0, "end": 12.0, "title": "T"}
    assert _coerce_clip(raw, video_duration=300, min_s=15, max_s=60) is None


def test_coerce_clip_drops_clip_with_duration_above_1_5x_max_s() -> None:
    """Duration > max_s * 1.5 → drop ('LLM hallucinated, too long')."""
    raw = {"start": 0.0, "end": 100.0, "title": "T"}
    assert _coerce_clip(raw, video_duration=300, min_s=15, max_s=60) is None


def test_coerce_clip_falls_back_to_generated_title_when_title_missing() -> None:
    """No `title` field → `Clip 10-40` derived from bounds."""
    raw = {"start": 10.0, "end": 40.0}
    c = _coerce_clip(raw, video_duration=300, min_s=15, max_s=60)
    assert c is not None
    assert c.title == "Clip 10-40"


def test_coerce_clip_truncates_reason_to_300_characters() -> None:
    """Reason longer than 300 chars → truncated."""
    raw = {"start": 10.0, "end": 40.0, "reason": "x" * 500, "title": "T"}
    c = _coerce_clip(raw, video_duration=300, min_s=15, max_s=60)
    assert c is not None
    assert len(c.reason) == 300


# --------------------------------------------------------------------------- #
# _transcript_excerpt
# --------------------------------------------------------------------------- #

def test_transcript_excerpt_returns_segments_overlapping_clip_range() -> None:
    """Only segments overlapping [clip.start, clip.end] are included."""
    transcript = Transcript(language="en", segments=[
        TranscriptSegment(start=0, end=10, text="before", words=[]),
        TranscriptSegment(start=15, end=25, text="inside", words=[]),
        TranscriptSegment(start=50, end=60, text="after", words=[]),
    ])
    clip = Clip(start=12, end=30, title="t", reason="r")

    excerpt = _transcript_excerpt(transcript, clip)

    assert excerpt == "inside"


def test_transcript_excerpt_joins_multiple_overlapping_segments_with_space() -> None:
    """Multiple overlapping segments joined with single space."""
    transcript = Transcript(language="en", segments=[
        TranscriptSegment(start=10, end=20, text="first", words=[]),
        TranscriptSegment(start=20, end=30, text="second", words=[]),
    ])
    clip = Clip(start=10, end=30, title="t", reason="r")

    excerpt = _transcript_excerpt(transcript, clip)

    assert excerpt == "first second"


def test_transcript_excerpt_truncates_at_max_chars_and_appends_ellipsis() -> None:
    """Excerpt longer than max_chars → rsplit at last space, append ' …'."""
    long_text = "word " * 1000  # 5000 chars
    transcript = Transcript(language="en", segments=[
        TranscriptSegment(start=0, end=100, text=long_text, words=[]),
    ])
    clip = Clip(start=0, end=100, title="t", reason="r")

    excerpt = _transcript_excerpt(transcript, clip, max_chars=50)

    assert excerpt.endswith(" …")
    assert len(excerpt) <= 50 + 3  # 50 chars + " …" but rsplit may shorten further
