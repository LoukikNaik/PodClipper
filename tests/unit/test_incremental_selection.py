"""Approved moments are available before later discovery requests finish."""
import json
from unittest.mock import Mock

import pytest

from podclipper.analyze import AnalyzeError, iter_analyzed_clips
from podclipper.config import load_default_config
from podclipper.llm import LLMError
from podclipper.types import Transcript, TranscriptSegment, Word


def transcript():
    return Transcript('en', [TranscriptSegment(t, t + 10, 'A complete thought.', [
        Word(t, t + .4, 'A'), Word(t + .5, t + .9, 'complete'), Word(t + 1, t + 1.4, 'thought.'),
    ]) for t in range(0, 1500, 10)])


def pick(start):
    return {'start': start, 'end': start + 20, 'title': 'A Complete Thought',
            'reason': 'Complete beat', 'hook_score': .9}


def review():
    return json.dumps({'title': 'A Complete Thought', 'opening_clear': True, 'payoff_complete': True,
                       'title_supported': True, 'opening_context': 'The subject is introduced.',
                       'title_evidence': 'A complete thought.', 'payoff_evidence': 'A complete thought.'})


def setup():
    cfg = load_default_config()
    cfg.analyze.refine_bounds = False
    provider = Mock(name='external-model')
    provider.name = 'test'
    return cfg, provider


def test_first_approved_reel_is_yielded_before_searching_next_section(tmp_path):
    cfg, provider = setup()
    provider.complete.side_effect = [json.dumps({'clips': [pick(20)]}), review(),
                                     json.dumps({'clips': [pick(1300)]}), review()]
    clips = iter_analyzed_clips(transcript(), 1500, provider, cfg, debug_cache_dir=tmp_path)
    assert next(clips).start == 20
    assert provider.complete.call_count == 2
    saved = json.loads((tmp_path / 'analyze/04_final.json').read_text())
    assert saved['discovery_complete'] is False
    assert len(saved['final_clips']) == 1
    assert [c.start for c in clips] == [1300]
    assert json.loads((tmp_path / 'analyze/04_final.json').read_text())['discovery_complete'] is True


def test_boundary_duplicates_are_not_reviewed_or_exported_twice_and_retry_reuses_reviews(tmp_path):
    cfg, provider = setup()
    provider.complete.side_effect = [json.dumps({'clips': [pick(1120)]}), review(),
                                     json.dumps({'clips': [pick(1120), pick(1300)]}), review()]
    first = list(iter_analyzed_clips(transcript(), 1500, provider, cfg, debug_cache_dir=tmp_path))
    assert [c.start for c in first] == [1120, 1300]
    assert provider.complete.call_count == 4
    provider.complete.side_effect = AssertionError('successful decisions must be cached')
    again = list(iter_analyzed_clips(transcript(), 1500, provider, cfg, debug_cache_dir=tmp_path))
    assert again == first


def test_late_discovery_failure_keeps_approved_reels_and_retry_resumes(tmp_path):
    cfg, provider = setup()
    provider.complete.side_effect = [json.dumps({'clips': [pick(20)]}), review(), LLMError('temporary outage')]
    clips = iter_analyzed_clips(transcript(), 1500, provider, cfg, debug_cache_dir=tmp_path)
    assert next(clips).start == 20
    with pytest.raises(AnalyzeError, match='section 2'):
        next(clips)
    saved = json.loads((tmp_path / 'analyze/04_final.json').read_text())
    assert saved['discovery_complete'] is False
    assert len(saved['final_clips']) == 1
    provider.complete.side_effect = [json.dumps({'clips': [pick(1300)]}), review()]
    assert [c.start for c in iter_analyzed_clips(transcript(), 1500, provider, cfg, debug_cache_dir=tmp_path)] == [20, 1300]


def test_early_section_cannot_take_slots_reserved_for_later_source_sections(tmp_path):
    cfg, provider = setup()
    cfg.analyze.target_clips = 3
    provider.complete.side_effect = [json.dumps({'clips': [pick(t) for t in (20, 100, 200, 300)]}),
                                     review(), review(), json.dumps({'clips': [pick(1300)]}), review()]
    result = list(iter_analyzed_clips(transcript(), 1500, provider, cfg, debug_cache_dir=tmp_path))
    assert [c.start for c in result] == [20, 100, 1300]


def test_retry_keeps_a_committed_clip_when_a_previous_failed_candidate_overlaps_it(tmp_path):
    cfg, provider = setup()
    provider.complete.side_effect = [json.dumps({'clips': [pick(20), pick(25)]}),
                                     LLMError('review interrupted'), review(), LLMError('search interrupted')]
    clips = iter_analyzed_clips(transcript(), 1500, provider, cfg, debug_cache_dir=tmp_path)
    assert next(clips).start == 25
    with pytest.raises(AnalyzeError):
        next(clips)
    provider.complete.side_effect = [json.dumps({'clips': [pick(1300)]}), review()]
    assert [c.start for c in iter_analyzed_clips(transcript(), 1500, provider, cfg, debug_cache_dir=tmp_path)] == [25, 1300]


def test_unused_late_section_slots_can_be_filled_by_deferred_early_candidates(tmp_path):
    cfg, provider = setup()
    cfg.analyze.target_clips = 3
    provider.complete.side_effect = [json.dumps({'clips': [pick(t) for t in (20, 100, 200)]}),
                                     review(), review(), json.dumps({'clips': []}), review()]
    assert [c.start for c in iter_analyzed_clips(transcript(), 1500, provider, cfg, debug_cache_dir=tmp_path)] == [20, 100, 200]
