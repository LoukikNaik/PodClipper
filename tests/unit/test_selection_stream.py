"""Search can advance during an export without losing late failures."""
from threading import Event

import pytest

from podclipper.selection_stream import prefetch_selection


def test_selection_advances_while_consumer_is_rendering():
    rendering, advanced = Event(), Event()

    def selections():
        yield 'first approved clip'
        assert rendering.wait(2)
        advanced.set()
        yield 'second approved clip'

    with prefetch_selection(selections()) as clips:
        assert next(clips) == 'first approved clip'
        rendering.set()
        assert advanced.wait(2), 'search must advance without waiting for the next export'
        assert list(clips) == ['second approved clip']


def test_queued_clips_are_consumed_before_a_late_search_error_is_reported():
    def selections():
        yield 'finished review'
        raise ValueError('section failed')

    with prefetch_selection(selections()) as clips:
        assert next(clips) == 'finished review'
        with pytest.raises(ValueError, match='section failed'):
            next(clips)
