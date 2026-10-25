"""Partial selection manifests must resume discovery, not terminate it early."""
import json
import logging
import sys
from types import SimpleNamespace

from podclipper import pipeline
from podclipper_web import render as renderer


def test_partial_selection_resumes_discovery_and_does_not_regress_export_progress(monkeypatch, tmp_path, caplog):
    directory = tmp_path / 'jobs' / 'job'
    directory.mkdir(parents=True)
    (directory / "request.json").write_text(json.dumps({"clips": 5}))
    prior = directory / 'reels' / 'finished.mp4'
    prior.parent.mkdir()
    prior.touch()
    prior.with_suffix('.txt').write_text('source_start: 20.00\nsource_end: 40.00\n')
    (directory / 'progress.json').write_text(json.dumps({'completed': [str(prior)]}))
    selected = [{'start': 20, 'end': 40, 'title': 'First complete thought', 'reason': 'Complete', 'hook_score': .9}]
    selected.append({**selected[0], 'start': 1300, 'end': 1320, 'title': 'Second complete thought'})
    (directory / 'selection.json').write_text(json.dumps({'source': str(directory / 'source.mp4'),
                                                         'clips': selected, 'discovery_complete': False}))
    final = directory / '.cache' / 'source' / 'analyze' / '04_final.json'
    final.parent.mkdir(parents=True)
    monkeypatch.setattr(renderer.Settings, 'load', lambda: SimpleNamespace(data=tmp_path))
    monkeypatch.setattr(sys, 'argv', ['render', str(directory)])
    monkeypatch.setattr('podclipper.logging_util.setup_logging', lambda *a: None)
    caplog.set_level(logging.INFO, logger='ave')
    for name in ('ave.analyze', 'ave.pipeline'):
        child = logging.getLogger(name)
        caplog.set_level(logging.INFO, logger=name)
        monkeypatch.setattr(child, 'disabled', False)
        monkeypatch.setattr(child, 'propagate', True)

    called = []

    def process(*a, selection_path=None, completed_bounds=None, **kw):
        called.append(True)
        assert selection_path is None, 'partial selection must not stop further discovery'
        assert completed_bounds == [(20, 40)]
        logging.getLogger('ave.pipeline').info('Exporting approved clip: Second complete thought')
        logging.getLogger('ave.analyze').info('Searching transcript section 2/8 (1110–2310s)')
        assert json.loads((directory / 'progress.json').read_text())['stage'] == 'exporting'
        final.write_text(json.dumps({'final_clips': selected, 'discovery_complete': True}))
        logging.getLogger('ave.analyze').info('Selection progress: 1 clips approved; discovery complete')
        assert json.loads((directory / 'selection.json').read_text())['discovery_complete'] is True
        return []

    monkeypatch.setattr(pipeline, 'run_pipeline', process)
    assert renderer.main() == 0
    assert called == [True]
