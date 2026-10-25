import pytest

from podclipper.config import load_default_config
from podclipper.reel_limits import reel_cap
from podclipper.analyze import analyze_for_reels
from podclipper.types import Transcript, TranscriptSegment, Word
from unittest.mock import Mock
import json


@pytest.mark.parametrize('minutes,expected', [(5, 3), (15, 3), (30, 6), (60, 12), (120, 24), (180, 30)])
def test_default_cap_grows_with_duration_and_has_a_hard_limit(minutes, expected):
    assert reel_cap(minutes * 60, load_default_config()) == expected


def test_explicit_cli_or_config_count_remains_an_optional_ceiling():
    cfg = load_default_config()
    cfg.analyze.target_clips = 4
    assert reel_cap(7200, cfg) == 4
    cfg.analyze.target_clips = 100
    assert reel_cap(7200, cfg) == 30


def test_default_selection_has_no_fixed_reel_count():
    cfg = load_default_config()
    assert cfg.analyze.target_clips is None
    assert cfg.content.clips is None


def test_long_video_searches_overlapping_sections_and_can_finish_below_cap(tmp_path):
    cfg = load_default_config()
    cfg.analyze.refine_bounds = False
    transcript = Transcript('en', [TranscriptSegment(t, t + 10, 'A complete thought.', [
        Word(t, t + .4, 'A'), Word(t + .5, t + .9, 'complete'), Word(t + 1, t + 1.4, 'thought.'),
    ]) for t in range(0, 2500, 10)])
    provider = Mock()
    provider.name = 'test'
    provider.complete.side_effect = [
        '[]',
        json.dumps([{'start': 2100, 'end': 2120, 'title': 'A Complete Thought', 'reason': 'Complete beat', 'hook_score': .9}]),
        json.dumps({'title': 'A Complete Thought', 'opening_clear': True, 'payoff_complete': True,
                    'title_supported': True, 'opening_context': 'The subject is introduced.',
                    'title_evidence': 'A complete thought.', 'payoff_evidence': 'A complete thought.'}),
        '[]',
    ]
    result = analyze_for_reels(transcript, 2500, provider, cfg, debug_cache_dir=tmp_path)
    assert len(result) == 1
    assert result[0].start == 2100
    assert provider.complete.call_count == 4
    prompts = [call.kwargs['user_prompt'] for call in provider.complete.call_args_list
               if 'Search source interval' in call.kwargs['user_prompt']]
    assert '1110' in prompts[1]
    assert '2220' in prompts[2]
    provider.reset_mock()
    provider.complete.side_effect = None
    provider.complete.return_value = json.dumps({
        'title': 'A Complete Thought', 'opening_clear': True, 'payoff_complete': True,
        'title_supported': True, 'opening_context': 'The subject is introduced.',
        'title_evidence': 'A complete thought.', 'payoff_evidence': 'A complete thought.',
    })
    again = analyze_for_reels(transcript, 2500, provider, cfg, debug_cache_dir=tmp_path)
    assert len(again) == 1
    assert provider.complete.call_count == 0  # Discovery and approved decisions are cached.
