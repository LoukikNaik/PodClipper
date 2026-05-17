"""Characterization tests for `src/trailer.py` (pure-function layer).

Module under test: trailer-mode functions — pick quotable sentences, refine
their cut bounds, splice with black gaps.

Deferred (need video fixtures + ffmpeg): `cut_segment`, `_ffmpeg`,
`_probe_duration`, `concat_with_black_gaps`.

Behaviors locked:
  - `_extract_json_array`            — tolerant JSON array parser (different
                                       impl from analyze.py — uses regex
                                       not raw_decode).
  - `_extract_json_object`           — tolerant JSON object parser.
  - `_fmt_transcript_for_llm`        — formats transcript as `[mm:ss-mm:ss] text`.
  - `_t_get(cfg, key, default)`      — cfg.trailer.{key} with fallback.
  - `derive_cut_bounds`              — mechanical fallback when refiner LLM fails.
  - `build_trailer_words`            — remap clip-local words into trailer time.
  - `pick_quotables`                 — LLM #1 + filter/dedup (LLM mocked).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.trailer import (
    _extract_json_array,
    _extract_json_object,
    _fmt_transcript_for_llm,
    _t_get,
    build_trailer_words,
    derive_cut_bounds,
    pick_quotables,
)
from src.types import Transcript, TranscriptSegment, Word


# --------------------------------------------------------------------------- #
# JSON extractors
# --------------------------------------------------------------------------- #

def test_extract_json_array_parses_raw_json_array() -> None:
    """Plain JSON array → parsed list."""
    assert _extract_json_array('[{"a": 1}]') == [{"a": 1}]


def test_extract_json_array_finds_array_inside_surrounding_prose() -> None:
    """Regex fallback grabs `[{...},...]` from prose-wrapped output."""
    text = 'Here are picks:\n[{"x": 1}, {"x": 2}]\nThanks.'
    assert _extract_json_array(text) == [{"x": 1}, {"x": 2}]


def test_extract_json_array_raises_valueerror_when_no_array_present() -> None:
    """No `[...{...}...]` → ValueError 'No JSON array'."""
    with pytest.raises(ValueError, match="No JSON array"):
        _extract_json_array("nothing here")


def test_extract_json_object_parses_raw_object() -> None:
    assert _extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_object_finds_object_in_prose() -> None:
    """Regex `\\{[^{}]+\\}` finds a flat (non-nested) object inside prose."""
    text = 'Result: {"start": 1.0, "end": 2.0} done.'
    assert _extract_json_object(text) == {"start": 1.0, "end": 2.0}


def test_extract_json_object_raises_valueerror_when_no_object_present() -> None:
    with pytest.raises(ValueError, match="No JSON object"):
        _extract_json_object("nothing here")


def test_extract_json_object_regex_fallback_silently_returns_inner_object_for_nested() -> None:
    """SURPRISE / KNOWN LIMITATION: same as analyze.py — the regex
    `\\{[^{}]+\\}` matches only flat objects, so a prose-wrapped response
    with nested objects silently returns the INNER object. Refiner callers
    expect flat {start, end} so this hasn't bitten yet; locking so a future
    caller expecting nested data sees the silent corruption immediately."""
    result = _extract_json_object('Refined: {"meta": {"start": 1.0, "end": 2.0}} done.')
    assert result == {"start": 1.0, "end": 2.0}


# --------------------------------------------------------------------------- #
# _fmt_transcript_for_llm
# --------------------------------------------------------------------------- #

def test_fmt_transcript_for_llm_renders_each_segment_with_mmss_brackets() -> None:
    """Each segment becomes `[mm:ss-mm:ss] text`."""
    transcript = Transcript(language="en", segments=[
        TranscriptSegment(start=5.0, end=12.5, text="hello world", words=[]),
        TranscriptSegment(start=65.0, end=70.0, text="another", words=[]),
    ])

    result = _fmt_transcript_for_llm(transcript)

    assert result == "[00:05-00:12] hello world\n[01:05-01:10] another"


def test_fmt_transcript_for_llm_returns_empty_string_for_empty_transcript() -> None:
    """No segments → empty string."""
    transcript = Transcript(language="en", segments=[])
    assert _fmt_transcript_for_llm(transcript) == ""


# --------------------------------------------------------------------------- #
# _t_get
# --------------------------------------------------------------------------- #

def test_t_get_returns_cfg_trailer_value_when_set() -> None:
    cfg = SimpleNamespace(trailer=SimpleNamespace(gap_seconds=1.5))
    assert _t_get(cfg, "gap_seconds", default=0.6) == 1.5


def test_t_get_returns_default_when_trailer_section_missing() -> None:
    cfg = SimpleNamespace()  # no `trailer` attribute
    assert _t_get(cfg, "gap_seconds", default=0.6) == 0.6


def test_t_get_returns_default_when_specific_key_missing_under_trailer() -> None:
    cfg = SimpleNamespace(trailer=SimpleNamespace())  # trailer exists, key doesn't
    assert _t_get(cfg, "gap_seconds", default=0.6) == 0.6


# --------------------------------------------------------------------------- #
# derive_cut_bounds
# --------------------------------------------------------------------------- #

def _word(start: float, end: float, text: str = "x") -> Word:
    return Word(start=start, end=end, text=text)


def _transcript_with_words(words: list[Word]) -> Transcript:
    return Transcript(language="en", segments=[
        TranscriptSegment(
            start=words[0].start, end=words[-1].end,
            text=" ".join(w.text for w in words), words=words,
        )
    ])


def test_derive_cut_bounds_anchors_to_first_and_last_word_in_window() -> None:
    """In-window first/last words define the cut, plus head/tail pad."""
    words = [_word(10.0, 10.5), _word(11.0, 11.5), _word(12.0, 12.5)]
    transcript = _transcript_with_words(words)
    quotable = {"start": 10.0, "end": 12.5}
    cfg = SimpleNamespace(trailer=SimpleNamespace(head_pad=0.1, tail_pad=0.1))

    cs, ce = derive_cut_bounds(quotable, transcript, cfg)

    assert cs == pytest.approx(10.0 - 0.1, abs=1e-6)
    assert ce == pytest.approx(12.5 + 0.1, abs=1e-6)


def test_derive_cut_bounds_falls_back_to_quotable_bounds_when_no_in_window_words() -> None:
    """No words inside the quotable window → fall back to quotable bounds + pad."""
    words = [_word(100.0, 100.5)]  # far from quotable
    transcript = _transcript_with_words(words)
    quotable = {"start": 10.0, "end": 12.0}
    cfg = SimpleNamespace(trailer=SimpleNamespace(head_pad=0.2, tail_pad=0.3))

    cs, ce = derive_cut_bounds(quotable, transcript, cfg)

    assert cs == pytest.approx(10.0 - 0.2)
    assert ce == pytest.approx(12.0 + 0.3)


def test_derive_cut_bounds_clamps_head_to_zero_when_pad_would_go_negative() -> None:
    """Start very close to 0 + head_pad → max(0, ...) clamp."""
    words = [_word(0.05, 0.5)]
    transcript = _transcript_with_words(words)
    quotable = {"start": 0.05, "end": 0.5}
    cfg = SimpleNamespace(trailer=SimpleNamespace(head_pad=1.0, tail_pad=0.0))

    cs, _ = derive_cut_bounds(quotable, transcript, cfg)

    assert cs == 0.0


# --------------------------------------------------------------------------- #
# build_trailer_words
# --------------------------------------------------------------------------- #

def test_build_trailer_words_remaps_clip_local_words_into_cumulative_trailer_time() -> None:
    """Each pick's words get cumulative offset = sum of prior pick durations + gaps."""
    picks = [
        {"cut_start": 10.0, "cut_end": 12.0},  # 2s duration
        {"cut_start": 50.0, "cut_end": 53.0},  # 3s duration
    ]
    pick_words = [
        [Word(start=0.0, end=0.5, text="hello"), Word(start=0.5, end=1.5, text="world")],
        [Word(start=0.0, end=1.0, text="foo"), Word(start=1.0, end=2.5, text="bar")],
    ]
    cfg = SimpleNamespace(trailer=SimpleNamespace(gap_seconds=0.5))

    out = build_trailer_words(picks, pick_words, cfg)

    assert len(out) == 4
    assert out[0].text == "hello"; assert out[0].start == 0.0; assert out[0].end == 0.5
    assert out[1].text == "world"; assert out[1].start == 0.5; assert out[1].end == 1.5
    # gap of 0.5s after pick 1 (2s duration + 0.5 gap = 2.5s offset)
    assert out[2].text == "foo"; assert out[2].start == 2.5; assert out[2].end == 3.5
    assert out[3].text == "bar"; assert out[3].start == 3.5; assert out[3].end == 5.0


def test_build_trailer_words_drops_words_outside_clip_duration() -> None:
    """A word.start > clip_duration is filtered (LLM/timing artifact)."""
    picks = [{"cut_start": 0.0, "cut_end": 2.0}]
    pick_words = [[
        Word(start=0.5, end=1.0, text="good"),
        Word(start=5.0, end=6.0, text="bad"),  # past 2s clip duration
    ]]
    cfg = SimpleNamespace(trailer=SimpleNamespace(gap_seconds=0.5))

    out = build_trailer_words(picks, pick_words, cfg)

    assert [w.text for w in out] == ["good"]


def test_build_trailer_words_clamps_word_end_to_clip_duration() -> None:
    """A word.end past clip_duration is truncated to the clip boundary."""
    picks = [{"cut_start": 0.0, "cut_end": 2.0}]
    pick_words = [[Word(start=1.0, end=3.0, text="clamped")]]
    cfg = SimpleNamespace(trailer=SimpleNamespace(gap_seconds=0.5))

    out = build_trailer_words(picks, pick_words, cfg)

    assert out[0].end == 2.0


# --------------------------------------------------------------------------- #
# pick_quotables (with mocked provider)
# --------------------------------------------------------------------------- #

def _provider_returning(mocker, raw_response: str):
    p = mocker.MagicMock()
    p.complete.return_value = raw_response
    return p


def _cfg_for_picks(max_tokens: int = 1000) -> SimpleNamespace:
    return SimpleNamespace(llm=SimpleNamespace(max_tokens=max_tokens))


def test_pick_quotables_returns_valid_picks_sorted_by_start_time(mocker) -> None:
    """Happy path: LLM returns 3 picks → sorted ascending by start."""
    transcript = Transcript(language="en", segments=[
        TranscriptSegment(start=0, end=100, text="x", words=[]),
    ])
    picks_json = json.dumps([
        {"start": 50.0, "end": 55.0, "sentence": "B"},
        {"start": 10.0, "end": 15.0, "sentence": "A"},
        {"start": 80.0, "end": 85.0, "sentence": "C"},
    ])
    provider = _provider_returning(mocker, picks_json)

    out = pick_quotables(transcript, provider, _cfg_for_picks(), video_duration=100.0)

    assert [p["sentence"] for p in out] == ["A", "B", "C"]


def test_pick_quotables_drops_picks_past_video_duration(mocker) -> None:
    """An LLM-hallucinated pick beyond video_duration is logged-and-dropped."""
    transcript = Transcript(language="en", segments=[
        TranscriptSegment(start=0, end=100, text="x", words=[]),
    ])
    picks_json = json.dumps([
        {"start": 10.0, "end": 15.0, "sentence": "A"},
        {"start": 150.0, "end": 155.0, "sentence": "PAST_END"},
    ])
    provider = _provider_returning(mocker, picks_json)

    out = pick_quotables(transcript, provider, _cfg_for_picks(), video_duration=100.0)

    assert [p["sentence"] for p in out] == ["A"]


def test_pick_quotables_deduplicates_overlapping_picks(mocker) -> None:
    """If a pick starts before the previous one's end (after sorting), it's dropped."""
    transcript = Transcript(language="en", segments=[
        TranscriptSegment(start=0, end=100, text="x", words=[]),
    ])
    picks_json = json.dumps([
        {"start": 10.0, "end": 20.0, "sentence": "A"},
        {"start": 15.0, "end": 25.0, "sentence": "OVERLAPS_A"},
        {"start": 30.0, "end": 35.0, "sentence": "B"},
    ])
    provider = _provider_returning(mocker, picks_json)

    out = pick_quotables(transcript, provider, _cfg_for_picks(), video_duration=100.0)

    assert [p["sentence"] for p in out] == ["A", "B"]


def test_pick_quotables_drops_picks_with_invalid_fields(mocker) -> None:
    """Missing/non-numeric start, empty sentence, end<=start → dropped silently."""
    transcript = Transcript(language="en", segments=[
        TranscriptSegment(start=0, end=100, text="x", words=[]),
    ])
    picks_json = json.dumps([
        {"start": 10.0, "end": 5.0, "sentence": "INVERTED"},
        {"start": "x", "end": 20.0, "sentence": "BAD_START"},
        {"start": 30.0, "end": 35.0, "sentence": ""},
        {"start": 40.0, "end": 45.0, "sentence": "OK"},
    ])
    provider = _provider_returning(mocker, picks_json)

    out = pick_quotables(transcript, provider, _cfg_for_picks(), video_duration=100.0)

    assert [p["sentence"] for p in out] == ["OK"]
