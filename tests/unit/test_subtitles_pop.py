"""TDD-driven tests for the pop-style subtitle path in `src/subtitles.py`."""

from __future__ import annotations

from podclipper.subtitles import active_word_index, generate_pop_popups
from podclipper.types import Word


def _w(text: str, start: float, end: float) -> Word:
    return Word(start=start, end=end, text=text)


def test_active_word_index_returns_index_or_none() -> None:
    popup = generate_pop_popups(
        [_w("hello", 0.0, 0.3), _w("world", 0.4, 0.7)],
        max_words_per_popup=2,
        max_gap_seconds=1.0,
    )[0]
    assert active_word_index(popup, 0.1) == 0
    assert active_word_index(popup, 0.5) == 1
    assert active_word_index(popup, 0.35) is None
    assert active_word_index(popup, 5.0) is None


def test_pop_flushes_on_sentence_end() -> None:
    words = [
        _w("done.", 0.0, 0.3),
        _w("Next", 0.4, 0.7),
    ]
    popups = generate_pop_popups(words, max_words_per_popup=2, max_gap_seconds=1.0)
    assert len(popups) == 2
    assert [[w.text for w in p.words] for p in popups] == [["done."], ["Next"]]


def test_pop_flushes_on_long_gap() -> None:
    words = [
        _w("hello", 0.0, 0.3),
        _w("world", 2.0, 2.3),
    ]
    popups = generate_pop_popups(words, max_words_per_popup=2, max_gap_seconds=1.0)
    assert len(popups) == 2
    assert [[w.text for w in p.words] for p in popups] == [["hello"], ["world"]]


def test_pop_pairs_adjacent_words_at_cap_two() -> None:
    words = [
        _w("hello", 0.0, 0.3),
        _w("world", 0.4, 0.7),
        _w("again", 0.8, 1.1),
        _w("now", 1.2, 1.4),
    ]
    popups = generate_pop_popups(words, max_words_per_popup=2, max_gap_seconds=1.0)
    assert len(popups) == 2
    assert [[w.text for w in p.words] for p in popups] == [
        ["hello", "world"],
        ["again", "now"],
    ]
    assert popups[0].start == 0.0 and popups[0].end == 0.7
    assert popups[1].start == 0.8 and popups[1].end == 1.4


def test_pop_one_word_per_popup_by_default() -> None:
    words = [
        _w("hello", 0.0, 0.3),
        _w("world", 0.4, 0.7),
        _w("again", 0.8, 1.1),
    ]
    popups = generate_pop_popups(words, max_words_per_popup=1, max_gap_seconds=1.0)
    assert len(popups) == 3
    assert [p.start for p in popups] == [0.0, 0.4, 0.8]
    assert [p.end for p in popups] == [0.3, 0.7, 1.1]
    assert [[w.text for w in p.words] for p in popups] == [["hello"], ["world"], ["again"]]
