"""Exercise orchestration and real checkpoints with media/API boundaries replaced."""
import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

from podclipper import pipeline
from podclipper.config import load_default_config
from podclipper.types import Clip, Transcript, TranscriptSegment, VideoMeta, Word


def media_boundaries(monkeypatch, tmp_path, crop):
    source = tmp_path / 'source.mp4'
    source.touch()
    cfg = load_default_config()
    cfg.paths.cache_dir = str(tmp_path / 'cache')
    cfg.paths.output_dir = str(tmp_path / 'exports')
    monkeypatch.setattr(pipeline, 'ingest', lambda _: VideoMeta(source, 1500, 1920, 1080, 30, True, 'h264'))
    monkeypatch.setattr(pipeline, 'extract_audio', lambda _, path, **kw: path.touch())
    transcript = Transcript('en', [TranscriptSegment(0, 1500, 'A complete thought.', [])])
    monkeypatch.setattr(pipeline, 'transcribe_first_pass', lambda *a: transcript)
    monkeypatch.setattr(pipeline, 'build_provider', lambda *a: Mock())
    monkeypatch.setattr(pipeline, '_extract_clip_segment', lambda *a: a[4].touch())
    monkeypatch.setattr(pipeline, 'review_scenes', lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, 'detect_humans_all_per_frame', lambda *a: ([[]] * 20, [], 1, 1920, 1080))
    monkeypatch.setattr(pipeline, 'smart_crop_916_stacked', crop)
    def transcribe(_, cache, *a, **kw):
        if cache is not None:
            cache.write_text(json.dumps([{'start': 0, 'end': 1, 'text': 'Hello', 'confidence': 1.0}]))
        return [Word(0, 1, 'Hello')]
    monkeypatch.setattr(pipeline, 'transcribe_second_pass_cached', transcribe)
    monkeypatch.setattr(pipeline, 'cleanup_words', lambda words, *a, **kw: words)
    monkeypatch.setattr(pipeline, 'burn_subtitles', lambda *a, **kw: a[2].touch())
    monkeypatch.setattr(pipeline, 'evaluate_reel', lambda **kw: SimpleNamespace(
        verdict='publish', final_score=8, write_to=lambda path: None))
    return source, cfg


def test_export_starts_while_later_selection_is_still_running(monkeypatch, tmp_path):
    rendering, searching = Event(), Event()

    def selection(*a, **kw):
        yield Clip(20, 40, 'First Complete Thought', 'complete', .9)
        assert rendering.wait(2)
        searching.set()
        yield Clip(1300, 1320, 'Second Complete Thought', 'complete', .8)

    def crop(*a, **kw):
        rendering.set()
        assert searching.wait(2), 'later search must overlap the first export'
        a[3].touch()

    source, cfg = media_boundaries(monkeypatch, tmp_path, crop)
    monkeypatch.setattr(pipeline, 'iter_analyzed_clips', selection)
    result = pipeline.run_pipeline(source, cfg)
    assert len(result) == 2
    assert all(p.exists() and p.with_suffix('.captions.json').exists() for p in result)


def test_resuming_partial_selection_skips_already_exported_source_bounds(monkeypatch, tmp_path):
    rendered = []

    def crop(*a, **kw):
        rendered.append(a[0])
        a[3].touch()

    source, cfg = media_boundaries(monkeypatch, tmp_path, crop)
    monkeypatch.setattr(pipeline, 'iter_analyzed_clips', lambda *a, **kw: iter([
        Clip(20, 40, 'Already Finished Thought', 'complete', .9),
        Clip(1300, 1320, 'New Complete Thought', 'complete', .8),
    ]))
    result = pipeline.run_pipeline(source, cfg, completed_bounds=[(20, 40)])
    assert len(result) == len(rendered) == 1
    assert 'new-complete-thought' in result[0].name


def test_preparation_stops_after_caching_transcript_without_selecting_or_rendering(monkeypatch,tmp_path):
    def no_render(*a,**kw):
        raise AssertionError('preparation must not render')
    source,cfg=media_boundaries(monkeypatch,tmp_path,no_render)
    provider=Mock(side_effect=AssertionError('preparation must not select clips'))
    monkeypatch.setattr(pipeline,'build_provider',provider)
    result=pipeline.run_pipeline(source,cfg,transcribe_only=True)
    assert len(result)==1 and result[0].name=='first_pass_transcript.json'
    assert result[0].is_file()
    provider.assert_not_called()
