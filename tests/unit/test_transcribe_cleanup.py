"""Caption corrections preserve timing and fail explicitly after bounded retries.

Module under test: LLM post-pass that fixes Whisper mis-spellings and
transliterates non-Latin scripts so captions are readable.

Behaviors locked:
  - `_extract_corrections`        — strict correction-object parsing.
  - `cleanup_words`               — bounded retries and validated mapped corrections.
                                    Preserves word timings while updating text.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from podclipper.llm import LLMError
from podclipper.transcribe_cleanup import CleanupError, _extract_corrections, cleanup_words
from podclipper.types import Word


# --------------------------------------------------------------------------- #
# Structured correction responses
# --------------------------------------------------------------------------- #


def test_extract_corrections_parses_schema_object():
    assert _extract_corrections('{"corrections": [{"source_indices": [0], "words": ["hi"]}]}') == [
        {"source_indices": [0], "words": ["hi"]},
    ]


@pytest.mark.parametrize('response', [
    'not json', '[]', '{"corrections": [], "extra": true}',
    '```json\n{"corrections": []}\n```', '{"corrections": null}',
])
def test_extract_corrections_rejects_responses_outside_schema(response):
    with pytest.raises(CleanupError):
        _extract_corrections(response)


# --------------------------------------------------------------------------- #
# cleanup_words — fallback paths
# --------------------------------------------------------------------------- #

def _provider_returning(mocker, raw_response: str):
    p = mocker.MagicMock()
    try:
        data = json.loads(raw_response)
        if isinstance(data, list):
            raw_response = json.dumps({'corrections': data})
    except json.JSONDecodeError:
        pass
    p.complete.return_value = raw_response
    return p


def _cfg(enabled: bool = True, max_tokens: int | None = 4096) -> SimpleNamespace:
    return SimpleNamespace(
        llm=SimpleNamespace(max_tokens=32768, endpoint='https://api.example.com/v1', model='test'),
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


def test_cleanup_words_reports_error_when_llm_retries_are_exhausted(mocker) -> None:
    """An exhausted network failure must not silently publish raw words."""
    provider = mocker.MagicMock()
    provider.complete.side_effect = LLMError("network died")
    words = _words("hello", "world")

    with pytest.raises(CleanupError, match='Caption cleanup failed'):
        cleanup_words(words, provider, _cfg())
    assert provider.complete.call_count == 2
    assert [w.text for w in words] == ['hello', 'world']


def test_cleanup_words_reports_error_when_responses_remain_unparseable(mocker) -> None:
    """Garbled responses exhaust retries and stop caption export."""
    provider = _provider_returning(mocker, "not json at all")
    words = _words("hello", "world")

    with pytest.raises(CleanupError, match='Caption cleanup failed'):
        cleanup_words(words, provider, _cfg())
    assert provider.complete.call_count == 4  # Two parent attempts, then two on the first smaller chunk.


def test_cleanup_words_reports_error_when_source_indices_are_missing(mocker) -> None:
    """Every original word must be covered by the correction mapping."""
    provider = _provider_returning(mocker, json.dumps([
        {"source_indices": [0], "words": ["only_one"]},
    ]))
    words = _words("hello", "world")  # 2 words
    def omit_last_word(**kwargs):
        tokens = json.loads(kwargs['user_prompt'])['tokens']
        return json.dumps({'corrections': [
            {'source_indices': [t['i']], 'words': [t['w']]} for t in tokens[:-1]
        ]})
    provider.complete.side_effect = omit_last_word

    with pytest.raises(CleanupError):
        cleanup_words(words, provider, _cfg())


def test_cleanup_words_reports_error_when_corrected_words_are_missing(mocker) -> None:
    """Each correction must supply its replacement words."""
    provider = _provider_returning(mocker, json.dumps([
        {"source_indices": [0]}, {"source_indices": [1], "words": ["world"]},
    ]))
    words = _words("hello", "world")

    with pytest.raises(CleanupError):
        cleanup_words(words, provider, _cfg())


# --------------------------------------------------------------------------- #
# cleanup_words — happy path
# --------------------------------------------------------------------------- #

def test_cleanup_words_returns_new_words_with_updated_text_when_llm_succeeds(mocker) -> None:
    """Valid response → new Word objects with updated text, original timings preserved."""
    provider = _provider_returning(mocker, json.dumps([
        {"source_indices": [0], "words": ["Hello"]},
        {"source_indices": [1], "words": ["namaste"]},
    ]))
    words = [
        Word(start=0.0, end=0.5, text="hello"),
        Word(start=0.5, end=1.0, text="नमस्ते"),
    ]

    out = cleanup_words(words, provider, _cfg())

    assert [w.text for w in out] == ["Hello", "namaste"]
    assert out is not words
    assert [w.text for w in words] == ["hello", "नमस्ते"]
    assert out[0].start == 0.0 and out[0].end == 0.5
    assert out[1].start == 0.5 and out[1].end == 1.0


def test_cleanup_words_preserves_word_confidence_field(mocker) -> None:
    """Original Word.confidence carries through into the rewritten Word."""
    provider = _provider_returning(mocker, json.dumps([
        {"source_indices": [0], "words": ["fixed"]},
    ]))
    words = [Word(start=0.0, end=0.5, text="brokn", confidence=0.42)]

    out = cleanup_words(words, provider, _cfg())

    assert out[0].confidence == 0.42
    assert out[0].text == "fixed"


def test_cleanup_words_merges_adjacent_tokens_with_their_original_time_span(mocker) -> None:
    """Merged words span both source tokens and retain the lower confidence."""
    provider = _provider_returning(mocker, json.dumps([
        {"source_indices": [0, 1], "words": ["podcast"]},
        {"source_indices": [2], "words": ["today"]},
    ]))
    words = [Word(0, 0.4, "pod", 0.9), Word(0.5, 1, "cast", 0.7), Word(1.2, 2, "today", 0.8)]

    out = cleanup_words(words, provider, _cfg())

    assert out == [Word(0, 1, "podcast", 0.7), Word(1.2, 2, "today", 0.8)]


def test_cleanup_words_splits_a_token_within_its_original_time_span(mocker) -> None:
    """Splits share the original duration in proportion to corrected word lengths."""
    provider = _provider_returning(mocker, json.dumps([
        {"source_indices": [0], "words": ["a", "lot!"]},
    ]))

    out = cleanup_words([Word(1, 2, "alot!", 0.6)], provider, _cfg())

    assert out == [Word(1, 1.2, "a", 0.6), Word(1.2, 2, "lot!", 0.6)]


@pytest.mark.parametrize("corrections", [
    [{"source_indices": [1], "words": ["world"]}, {"source_indices": [0], "words": ["hello"]}],
    [{"source_indices": [0], "words": ["hello"]}, {"source_indices": [0], "words": ["again"]}],
    [{"source_indices": [0, 2], "words": ["combined"]}],
    [{"source_indices": [0, 1, 2], "words": ["combined"]}],
    [{"source_indices": [True, 1], "words": ["combined"]}],
    [{"source_indices": [0, 1], "words": [""]}],
    [{"source_indices": [0, 1], "words": ["two words"]}],
])
def test_cleanup_words_rejects_invalid_mappings(mocker, corrections) -> None:
    """Invalid mappings stop export without mutating the original words."""
    provider = _provider_returning(mocker, json.dumps(corrections))
    words = _words("hello", "world")

    with pytest.raises(CleanupError):
        cleanup_words(words, provider, _cfg())
    assert words == _words('hello', 'world')


def test_cleanup_retries_only_failed_chunks_and_reuses_completed_chunks(mocker, tmp_path):
    cfg = _cfg()
    cfg.transcribe.cleanup.chunk_words = 1
    provider = mocker.MagicMock()
    provider.complete.side_effect = [
        json.dumps({'corrections': [{'source_indices': [0], 'words': ['Hello']}]}),
        LLMError('unavailable'), LLMError('unavailable'),
        json.dumps({'corrections': [{'source_indices': [0], 'words': ['World']}]}),
    ]
    words = _words('hello', 'world')
    with pytest.raises(CleanupError):
        cleanup_words(words, provider, cfg, cache_dir=tmp_path)
    assert provider.complete.call_count == 3
    result = cleanup_words(words, provider, cfg, cache_dir=tmp_path)
    assert [w.text for w in result] == ['Hello', 'World']
    assert provider.complete.call_count == 4
    assert json.loads(provider.complete.call_args.kwargs['user_prompt'])['tokens'] == [{'i': 0, 'w': 'world'}]


def test_cleanup_words_uses_shared_token_budget_when_no_cleanup_override(mocker) -> None:
    """The shared LLM budget applies unless cleanup explicitly overrides it."""
    provider = _provider_returning(mocker, json.dumps([
        {"source_indices": [0], "words": ["hello"]},
    ]))
    cfg = _cfg(max_tokens=None)

    cleanup_words(_words("hello"), provider, cfg)

    assert provider.complete.call_args.kwargs["max_tokens"] == 32768


def test_caption_corrections_request_an_enforced_schema(mocker):
    provider = _provider_returning(mocker, json.dumps({'corrections': [
        {'source_indices': [0], 'words': ['Hello']},
    ]}))
    result = cleanup_words(_words('hello'), provider, _cfg())
    assert result[0].text == 'Hello'
    format = provider.complete.call_args.kwargs['response_format']
    assert format['type'] == 'json_schema'
    assert format['json_schema']['strict'] is True
    schema = format['json_schema']['schema']
    assert schema['required'] == ['corrections']
    assert schema['additionalProperties'] is False
    item = schema['properties']['corrections']['items']
    assert set(item['required']) == {'source_indices', 'words'}
    assert item['additionalProperties'] is False


def test_format_failure_splits_only_that_chunk_and_resumes_successful_halves(mocker, tmp_path):
    provider = mocker.MagicMock()
    provider.complete.side_effect = [
        'broken json', 'broken json',
        json.dumps({'corrections': [
            {'source_indices': [0], 'words': ['One']}, {'source_indices': [1], 'words': ['two']},
        ]}),
        LLMError('temporary outage'), LLMError('temporary outage'),
        json.dumps({'corrections': [
            {'source_indices': [0], 'words': ['three']}, {'source_indices': [1], 'words': ['four']},
        ]}),
    ]
    original = _words('one', 'two', 'three', 'four')
    with pytest.raises(CleanupError):
        cleanup_words(original, provider, _cfg(), cache_dir=tmp_path)
    assert provider.complete.call_count == 5
    result = cleanup_words(original, provider, _cfg(), cache_dir=tmp_path)
    assert provider.complete.call_count == 6
    assert [w.text for w in result] == ['One', 'two', 'three', 'four']
    assert [(w.start, w.end) for w in result] == [(w.start, w.end) for w in original]
    sizes = [len(json.loads(call.kwargs['user_prompt'])['tokens']) for call in provider.complete.call_args_list]
    assert sizes == [4, 4, 2, 2, 2, 2]
