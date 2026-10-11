"""Config merging, validation and public setting resolution tests."""
import pytest
from podclipper.config import load_config, load_default_config, ns_to_dict


def test_nested_override_preserves_siblings(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("crop:\n  stacked_transition_frames: 24\n")
    cfg = load_config(path)
    assert cfg.crop.stacked_transition_frames == 24
    assert cfg.crop.mode == "auto"
    assert cfg.transcribe.second_pass.model == "large-v3"


def test_public_settings_resolve_and_internal_overrides_win(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("content:\n  type: comedy\n  clips: 3\n  duration: [20, 45]\noutput:\n  captions: pop\n  music: true\nanalyze:\n  target_clips: 5\n")
    cfg = load_config(path)
    assert cfg.crop.comedy_mode
    assert cfg.analyze.target_clips == 5
    assert (cfg.analyze.min_clip_seconds, cfg.analyze.max_clip_seconds) == (20, 45)
    assert cfg.subtitles.style == "pop"
    assert cfg.music.enabled


@pytest.mark.parametrize("text,match", [
    ("crop:\n  typo: 1", "crop.typo"),
    ("llm:\n  provider: claude_cli", "llm.provider"),
    ("content:\n  type: invalid", "content.type"),
    ("content:\n  duration: [50, 10]", "content.duration"),
    ("crop: false", "mapping"),
    ("output:\n  music: 'false'", "type"),
    ("- example", "mapping"),
])
def test_invalid_config_rejected(tmp_path, text, match):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_empty_config_inherits_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("")
    assert ns_to_dict(load_config(path)) == ns_to_dict(load_default_config())


def test_default_llm_budget_is_32k():
    assert load_default_config().llm.max_tokens == 32768


def test_resolved_public_settings_match_internal_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("content:\n  clips: 3\nanalyze:\n  target_clips: 5\n")
    cfg = load_config(path)
    assert cfg.content.clips == cfg.analyze.target_clips == 5


@pytest.mark.parametrize('setting,value', [
    ('panel_dividers.margin_pixels', 'true'),
    ('panel_dividers.margin_pixels', '0'),
    ('panel_dividers.min_coverage', '1.1'),
    ('scene_cuts.crossfade_seconds', '-0.1'),
    ('scene_cuts.crossfade_seconds', '.nan'),
    ('scene_cuts.difference_threshold', '0'),
])
def test_invalid_crop_controls_rejected(tmp_path, setting, value):
    section, field = setting.split('.')
    path = tmp_path / 'config.yaml'
    path.write_text(f'crop:\n  {section}:\n    {field}: {value}\n')
    with pytest.raises(ValueError, match=field):
        load_config(path)


def test_hard_cuts_and_disabled_dividers_are_overridable(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text('crop:\n  panel_dividers:\n    enabled: false\n  scene_cuts:\n    crossfade_seconds: 0\n')
    cfg = load_config(path)
    assert not cfg.crop.panel_dividers.enabled
    assert cfg.crop.scene_cuts.crossfade_seconds == 0
