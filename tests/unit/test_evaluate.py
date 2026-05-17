"""Characterization tests for `src/evaluate.py`.

Module under test: reel quality evaluator — technical metrics + LLM-as-judge.

Behaviors locked:
  - `_extract_json_object`                — tolerant JSON object parser with
                                           fence support.
  - `compute_tech_metrics`                — face/crop/speaker/wps/subtitle math.
  - `TechMetrics` properties              — sweet-spot bands and score weighting.
  - `ContentScores.score`                 — average of 6 dimension scores.
  - `ReelScorecard.__post_init__`         — 30% tech + 70% content weighting,
                                           verdict carried from content.
  - `evaluate_content`                    — fallback paths (disabled, empty
                                           transcript, LLM fails, parse fails);
                                           happy path with mocked LLM.
  - `evaluate_trailer`                    — same shape as evaluate_content for
                                           the 5-axis trailer scorecard.
  - `TrailerScorecard` defaults           — pinned shape.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.evaluate import (
    ContentScores,
    ReelScorecard,
    TechMetrics,
    TrailerScorecard,
    _extract_json_object,
    compute_tech_metrics,
    evaluate_content,
    evaluate_trailer,
)
from src.types import Word


# --------------------------------------------------------------------------- #
# _extract_json_object
# --------------------------------------------------------------------------- #

def test_extract_json_object_parses_raw_object() -> None:
    assert _extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_object_strips_json_code_fence() -> None:
    assert _extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_object_finds_first_and_last_brace_through_prose() -> None:
    """Slices from first `{` to last `}` — tolerates outer prose."""
    text = 'Here is the verdict: {"verdict": "publish", "overall": 4.5} done.'
    assert _extract_json_object(text) == {"verdict": "publish", "overall": 4.5}


def test_extract_json_object_raises_valueerror_when_no_braces() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        _extract_json_object("no braces here")


# --------------------------------------------------------------------------- #
# compute_tech_metrics
# --------------------------------------------------------------------------- #

def _words_for(seconds: float, n: int) -> list[Word]:
    """Generate `n` words evenly spread over `seconds`."""
    if n == 0:
        return []
    step = seconds / n
    return [
        Word(start=i * step, end=(i + 1) * step, text=f"w{i}")
        for i in range(n)
    ]


def test_compute_tech_metrics_returns_correct_face_visibility_ratio() -> None:
    """face_visibility = face_hits / total_frames."""
    m = compute_tech_metrics(
        words=_words_for(30, 60), clip_duration=30.0,
        face_hits=600, total_frames=900,
        person_frames=900, x_centers=[100.0] * 900, source_width=1920,
    )
    assert m.face_visibility == pytest.approx(600 / 900)


def test_compute_tech_metrics_returns_perfect_crop_stability_with_constant_x_centers() -> None:
    """All x_centers equal → std=0 → crop_stability=1.0."""
    m = compute_tech_metrics(
        words=_words_for(30, 60), clip_duration=30.0,
        face_hits=900, total_frames=900,
        person_frames=900, x_centers=[100.0] * 900, source_width=1920,
    )
    assert m.crop_stability == 1.0


def test_compute_tech_metrics_returns_perfect_crop_stability_when_too_few_x_centers() -> None:
    """Fewer than 2 x_centers → crop_stability defaults to 1.0 (no penalty)."""
    m = compute_tech_metrics(
        words=_words_for(30, 60), clip_duration=30.0,
        face_hits=900, total_frames=900,
        person_frames=900, x_centers=[100.0], source_width=1920,
    )
    assert m.crop_stability == 1.0


def test_compute_tech_metrics_words_per_second_uses_word_count_over_duration() -> None:
    """wps = len(words) / max(0.1, clip_duration)."""
    m = compute_tech_metrics(
        words=_words_for(30, 60), clip_duration=30.0,
        face_hits=900, total_frames=900,
        person_frames=900, x_centers=[100.0] * 2, source_width=1920,
    )
    assert m.words_per_second == pytest.approx(60 / 30)


def test_compute_tech_metrics_subtitle_coverage_zero_for_no_words() -> None:
    """No words → sub_cov = 0.0."""
    m = compute_tech_metrics(
        words=[], clip_duration=30.0,
        face_hits=0, total_frames=900,
        person_frames=0, x_centers=[], source_width=1920,
    )
    assert m.subtitle_coverage == 0.0


# --------------------------------------------------------------------------- #
# TechMetrics properties
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("duration,expected", [
    (24.9, False),
    (25.0, True),
    (45.0, True),
    (60.0, True),
    (60.1, False),
])
def test_tech_metrics_duration_in_sweet_spot_25_to_60(duration: float, expected: bool) -> None:
    m = TechMetrics(
        face_visibility=1.0, crop_stability=1.0, speaker_coverage=1.0,
        duration_seconds=duration, words_per_second=2.5, subtitle_coverage=1.0,
    )
    assert m.duration_in_sweet_spot is expected


@pytest.mark.parametrize("wps,expected", [
    (1.4, False),
    (1.5, True),
    (2.5, True),
    (3.5, True),
    (3.6, False),
])
def test_tech_metrics_wps_in_range_1_5_to_3_5(wps: float, expected: bool) -> None:
    m = TechMetrics(
        face_visibility=1.0, crop_stability=1.0, speaker_coverage=1.0,
        duration_seconds=30, words_per_second=wps, subtitle_coverage=1.0,
    )
    assert m.wps_in_range is expected


def test_tech_metrics_score_uses_distinct_weight_per_dimension() -> None:
    """Weights: face 0.25, crop 0.20, speaker 0.15, duration 0.15, wps 0.15, subtitle 0.10.

    Uses DISTINCT input values per dimension so any weight swap shifts
    the result. (All-1.0 inputs would compute to 1.0 regardless of how
    the weights were distributed — useless as a canary.)"""
    m = TechMetrics(
        face_visibility=0.4,       # 0.4 * 0.25 = 0.10
        crop_stability=0.5,        # 0.5 * 0.20 = 0.10
        speaker_coverage=0.6,      # 0.6 * 0.15 = 0.09
        duration_seconds=30,       # in 25-60 sweet spot → 1.0 * 0.15 = 0.15
        words_per_second=2.0,      # in 1.5-3.5 range  → 1.0 * 0.15 = 0.15
        subtitle_coverage=0.8,     # 0.8 * 0.10 = 0.08
    )
    # 0.10 + 0.10 + 0.09 + 0.15 + 0.15 + 0.08 = 0.67
    assert m.score == pytest.approx(0.67)


def test_tech_metrics_score_uses_half_weight_when_duration_and_wps_out_of_band() -> None:
    """Out-of-band duration / wps contribute 0.5 (not 0) of their 0.15 weight."""
    m = TechMetrics(
        face_visibility=0.0, crop_stability=0.0, speaker_coverage=0.0,
        duration_seconds=10,         # below 25 → 0.5 * 0.15 = 0.075
        words_per_second=0.5,        # below 1.5 → 0.5 * 0.15 = 0.075
        subtitle_coverage=0.0,
    )
    # 0 + 0 + 0 + 0.075 + 0.075 + 0 = 0.15
    assert m.score == pytest.approx(0.15)


# --------------------------------------------------------------------------- #
# ContentScores.score
# --------------------------------------------------------------------------- #

def test_content_scores_score_is_mean_of_six_dimensions() -> None:
    """score = sum(6 dim scores) / 6."""
    cs = ContentScores(hook=5, arc=4, ending=3, standalone=4, shareability=5, title_match=3)
    assert cs.score == pytest.approx((5 + 4 + 3 + 4 + 5 + 3) / 6)


# --------------------------------------------------------------------------- #
# ReelScorecard.__post_init__ weighting
# --------------------------------------------------------------------------- #

def test_reel_scorecard_final_score_uses_30pct_tech_plus_70pct_normalized_content() -> None:
    """final = round(tech * 0.3 + (content_score / 5.0) * 0.7, 2).

    Uses DISTINCT tech (0.67) and content (4.0) scores so a weight swap
    (e.g. 0.7 tech / 0.3 content) would change the result and flag the bug.
    Equal scores would not catch a swap."""
    tech = TechMetrics(
        face_visibility=0.4, crop_stability=0.5, speaker_coverage=0.6,
        duration_seconds=30, words_per_second=2.0, subtitle_coverage=0.8,
    )  # tech.score = 0.67 (pinned in test_tech_metrics_score_uses_distinct_weight_per_dimension)
    content = ContentScores(
        hook=4, arc=4, ending=4, standalone=4, shareability=4, title_match=4,
        verdict="publish",
    )  # content.score = 24/6 = 4.0
    sc = ReelScorecard(tech=tech, content=content)
    # 0.67 * 0.3 + (4.0 / 5.0) * 0.7  =  0.201 + 0.56  =  0.761  →  round to 0.76
    assert sc.final_score == pytest.approx(0.76)
    assert sc.verdict == "publish"


# --------------------------------------------------------------------------- #
# evaluate_content (mocked LLM)
# --------------------------------------------------------------------------- #

def _mock_provider(mocker, raw_response: str):
    p = mocker.MagicMock()
    p.complete.return_value = raw_response
    return p


def _eval_cfg(enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(evaluate=SimpleNamespace(enabled=enabled))


def test_evaluate_content_returns_review_default_when_evaluation_disabled(mocker) -> None:
    """`evaluate.enabled = False` → ContentScores(verdict='review', feedback='evaluation disabled')."""
    cfg = _eval_cfg(enabled=False)
    provider = _mock_provider(mocker, "irrelevant")

    cs = evaluate_content("title", [_w("hi")], 30.0, provider, cfg)

    assert cs.verdict == "review"
    assert cs.feedback == "evaluation disabled"
    provider.complete.assert_not_called()


def test_evaluate_content_returns_skip_when_transcript_is_empty(mocker) -> None:
    """No words / whitespace-only → verdict='skip', feedback='empty transcript'."""
    cfg = _eval_cfg()
    provider = _mock_provider(mocker, "irrelevant")

    cs = evaluate_content("title", [], 30.0, provider, cfg)

    assert cs.verdict == "skip"
    assert cs.feedback == "empty transcript"


def test_evaluate_content_returns_review_when_llm_raises(mocker) -> None:
    """LLMError from provider → verdict='review', feedback includes 'LLM eval failed'."""
    from src.llm import LLMError
    cfg = _eval_cfg()
    provider = mocker.MagicMock()
    provider.complete.side_effect = LLMError("network died")

    cs = evaluate_content("title", [_w("hi")], 30.0, provider, cfg)

    assert cs.verdict == "review"
    assert "LLM eval failed" in cs.feedback


def test_evaluate_content_returns_review_when_response_is_unparseable_json(mocker) -> None:
    """Bad JSON → verdict='review', feedback includes 'parse error'."""
    cfg = _eval_cfg()
    provider = _mock_provider(mocker, "not json at all")

    cs = evaluate_content("title", [_w("hi")], 30.0, provider, cfg)

    assert cs.verdict == "review"
    assert "parse error" in cs.feedback


def test_evaluate_content_extracts_scores_and_verdict_from_valid_llm_response(mocker) -> None:
    """Happy path: dim scores, overall, verdict, feedback all populated."""
    cfg = _eval_cfg()
    response = json.dumps({
        "hook":         {"reasoning": "strong opener", "score": 5},
        "arc":          {"reasoning": "tight arc",     "score": 4},
        "ending":       {"reasoning": "mic drop",      "score": 5},
        "standalone":   {"reasoning": "clear",         "score": 4},
        "shareability": {"reasoning": "shareable",     "score": 4},
        "title_match":  {"reasoning": "matches",       "score": 4},
        "overall": 4.3,
        "verdict": "publish",
        "one_line_feedback": "ship it",
    })
    provider = _mock_provider(mocker, response)

    cs = evaluate_content("title", [_w("hi")], 30.0, provider, cfg)

    assert cs.hook == 5
    assert cs.arc == 4
    assert cs.ending == 5
    assert cs.verdict == "publish"
    assert cs.overall == 4.3
    assert cs.feedback == "ship it"


def test_evaluate_content_recomputes_verdict_when_llm_returns_invalid_verdict(mocker) -> None:
    """If verdict ∉ {publish,review,skip}, derive from `overall`: >=4 publish, >=3 review, else skip."""
    cfg = _eval_cfg()
    response = json.dumps({
        "hook":  {"reasoning": "", "score": 2},
        "arc":   {"reasoning": "", "score": 2},
        "ending": {"reasoning": "", "score": 2},
        "standalone": {"reasoning": "", "score": 2},
        "shareability": {"reasoning": "", "score": 2},
        "title_match": {"reasoning": "", "score": 2},
        "overall": 2.0,
        "verdict": "garbage_value",
    })
    provider = _mock_provider(mocker, response)

    cs = evaluate_content("title", [_w("hi")], 30.0, provider, cfg)

    assert cs.verdict == "skip"  # overall < 3.0


# --------------------------------------------------------------------------- #
# evaluate_trailer (mocked LLM)
# --------------------------------------------------------------------------- #

def test_evaluate_trailer_returns_disabled_when_evaluation_disabled(mocker) -> None:
    cfg = _eval_cfg(enabled=False)
    provider = _mock_provider(mocker, "irrelevant")

    sc = evaluate_trailer(picks=[], total_duration=10.0, provider=provider, cfg=cfg)

    assert isinstance(sc, TrailerScorecard)
    assert sc.verdict == "review"
    assert sc.feedback == "evaluation disabled"
    provider.complete.assert_not_called()


def test_evaluate_trailer_falls_back_to_review_when_llm_raises(mocker) -> None:
    from src.llm import LLMError
    cfg = _eval_cfg()
    provider = mocker.MagicMock()
    provider.complete.side_effect = LLMError("died")

    sc = evaluate_trailer(picks=[{"sentence": "x", "cut_start": 0, "cut_end": 1}],
                          total_duration=10.0, provider=provider, cfg=cfg)

    assert sc.verdict == "review"
    assert "LLM eval failed" in sc.feedback


def test_evaluate_trailer_extracts_5_axis_scores_from_valid_response(mocker) -> None:
    """Happy path for the 5-axis trailer rubric."""
    cfg = _eval_cfg()
    response = json.dumps({
        "opener_hook":         {"reasoning": "strong", "score": 5},
        "thematic_coherence":  {"reasoning": "unified", "score": 4},
        "pacing":              {"reasoning": "good", "score": 4},
        "closer_punch":        {"reasoning": "great", "score": 5},
        "standalone_quality":  {"reasoning": "clear", "score": 4},
        "overall": 4.4,
        "verdict": "publish",
        "one_line_feedback": "trailer is ready",
    })
    provider = _mock_provider(mocker, response)

    sc = evaluate_trailer(
        picks=[{"sentence": "a", "cut_start": 0, "cut_end": 2}],
        total_duration=5.0, provider=provider, cfg=cfg,
    )

    assert sc.opener_hook == 5
    assert sc.thematic_coherence == 4
    assert sc.pacing == 4
    assert sc.closer_punch == 5
    assert sc.standalone_quality == 4
    assert sc.verdict == "publish"
    assert sc.overall == 4.4


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _w(text: str) -> Word:
    """Build a 1-second-long Word at t=0 for transcript-string purposes."""
    return Word(start=0.0, end=1.0, text=text)
