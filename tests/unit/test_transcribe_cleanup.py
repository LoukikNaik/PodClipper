"""Characterization tests for `src/transcribe_cleanup.py`.

Module under test: LLM post-pass that fixes Whisper mis-spellings and
transliterates non-Latin scripts so captions are readable.

Behaviors locked:
  - `_extract_json_array`         — tolerant array parser with fence support.
  - `cleanup_words`               — all fallback paths (returns input unchanged
                                    on any failure); happy path with mocked LLM.
                                    Preserves word timings while updating text.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from podclipper.llm import LLMError
from podclipper.transcribe_cleanup import CleanupError, _extract_json_array, cleanup_words
from podclipper.types import Word


# --------------------------------------------------------------------------- #
# _extract_json_array
# --------------------------------------------------------------------------- #

def test_extract_json_array_parses_raw_array() -> None:
    assert _extract_json_array('[{"i": 0, "w": "hi"}]') == [{"i": 0, "w": "hi"}]


def test_extract_json_array_strips_json_code_fence() -> None:
    assert _extract_json_array('```json\n[1, 2]\n```') == [1, 2]


def test_extract_json_array_raises_cleanuperror_when_no_brackets() -> None:
    with pytest.raises(CleanupError, match="no JSON array"):
        _extract_json_array("no brackets here")


# --------------------------------------------------------------------------- #
# cleanup_words — fallback paths
# --------------------------------------------------------------------------- #

def _provider_returning(mocker, raw_response: str):
    p = mocker.MagicMock()
    p.complete.return_value = raw_response
    return p


def _cfg(enabled: bool = True, max_tokens: int = 4096) -> SimpleNamespace:
    return SimpleNamespace(
        transcribe=SimpleNamespace(
            cleanup=SimpleNamespace(enabled=enabled, max_tokens_per_call=max_tokens),
        ),
    )


def _cfg_without_cleanup_section() -> SimpleNamespace:
    return SimpleNamespace(transcribe=SimpleNamespace())


def _words(*texts: str) -> list[Word]:
    return [Word(start=float(i), end=float(i) + 0.5, text=t) for i, t in enumerate(texts)]


def test_cleanup_words_returns_input_unchanged_when_cleanup_disabled(mocker) -> None:
    """`cleanup.enabled = False` → no LLM call, input returned as-is."""
    provider = mocker.MagicMock()
    words = _words("hello", "world")

    out = cleanup_words(words, provider, _cfg(enabled=False))

    assert out is words
    provider.complete.assert_not_called()


def test_cleanup_words_returns_input_unchanged_when_cleanup_section_missing(mocker) -> None:
    """Missing `transcribe.cleanup` section → no LLM call, input passed through."""
    provider = mocker.MagicMock()
    words = _words("hello")

    out = cleanup_words(words, provider, _cfg_without_cleanup_section())

    assert out is words
    provider.complete.assert_not_called()


def test_cleanup_words_returns_input_unchanged_when_word_list_is_empty(mocker) -> None:
    """Empty input → return empty without calling LLM."""
    provider = mocker.MagicMock()

    out = cleanup_words([], provider, _cfg())

    assert out == []
    provider.complete.assert_not_called()


def test_cleanup_words_returns_original_when_llm_raises_llmerror(mocker) -> None:
    """LLMError → keep original (karaoke must keep working)."""
    provider = mocker.MagicMock()
    provider.complete.side_effect = LLMError("network died")
    words = _words("hello", "world")

    out = cleanup_words(words, provider, _cfg())

    assert out is words


def test_cleanup_words_returns_original_when_response_is_unparseable(mocker) -> None:
    """Garbled response → keep original."""
    provider = _provider_returning(mocker, "not json at all")
    words = _words("hello", "world")

    out = cleanup_words(words, provider, _cfg())

    assert out is words


def test_cleanup_words_returns_original_when_response_count_mismatches(mocker) -> None:
    """LLM returned wrong number of tokens → keep original (safer than re-aligning)."""
    provider = _provider_returning(mocker, json.dumps([{"i": 0, "w": "only_one"}]))
    words = _words("hello", "world")  # 2 words

    out = cleanup_words(words, provider, _cfg())

    assert out is words


def test_cleanup_words_returns_original_when_item_missing_w_key(mocker) -> None:
    """Malformed item (no 'w' field) → keep original."""
    provider = _provider_returning(mocker, json.dumps([{"i": 0}, {"i": 1, "w": "world"}]))
    words = _words("hello", "world")

    out = cleanup_words(words, provider, _cfg())

    assert out is words


# --------------------------------------------------------------------------- #
# cleanup_words — happy path
# --------------------------------------------------------------------------- #

def test_cleanup_words_returns_new_words_with_updated_text_when_llm_succeeds(mocker) -> None:
    """Valid response → new Word objects with updated text, original timings preserved."""
    provider = _provider_returning(mocker, json.dumps([
        {"i": 0, "w": "Hello"},          # capitalized
        {"i": 1, "w": "namaste"},        # transliterated from नमस्ते
    ]))
    words = [
        Word(start=0.0, end=0.5, text="hello"),
        Word(start=0.5, end=1.0, text="नमस्ते"),
    ]

    out = cleanup_words(words, provider, _cfg())

    assert [w.text for w in out] == ["Hello", "namaste"]
    # Timings preserved
    assert out[0].start == 0.0 and out[0].end == 0.5
    assert out[1].start == 0.5 and out[1].end == 1.0


def test_cleanup_words_preserves_word_confidence_field(mocker) -> None:
    """Original Word.confidence carries through into the rewritten Word."""
    provider = _provider_returning(mocker, json.dumps([{"i": 0, "w": "fixed"}]))
    words = [Word(start=0.0, end=0.5, text="brokn", confidence=0.42)]

    out = cleanup_words(words, provider, _cfg())

    assert out[0].confidence == 0.42
