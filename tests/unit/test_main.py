"""Characterization tests for `main.py`.

Module under test: the CLI entry point.

Functionality being locked down here:
  - `_load_dotenv(path)`           — minimal .env parser that hydrates os.environ
                                    BEFORE the src.llm modules are imported.
                                    Tested behaviors: quote handling, `export`
                                    prefix, comments/blanks, precedence rules.
  - `build_parser()`               — argparse setup. We pin the positional/optional
                                    argument shape, defaults, and `choices=` lists
                                    so future CLI changes are intentional and
                                    visible in a diff (especially the
                                    `--llm-provider` choices — they need to
                                    change in Phase 1 when we add litellm).
  - `apply_cli_overrides(cfg, args)` — applies parsed CLI flags onto the loaded
                                    config namespace. Tested behaviors: which
                                    cfg sub-attributes each CLI flag targets,
                                    and the no-op behavior when flags are unset.
  - `main(argv)` error paths       — return codes 2 (missing input/config),
                                    without spinning up the real pipeline.

These tests document current behavior. DO NOT modify `main.py` to make a
failing test pass — record surprises with a `# SURPRISE:` comment instead.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from podclipper.main import _load_dotenv, apply_cli_overrides, build_parser, main


# --------------------------------------------------------------------------- #
# _load_dotenv: file presence and parsing
# --------------------------------------------------------------------------- #

def test_load_dotenv_silently_returns_when_file_does_not_exist(
    tmp_path: Path,
) -> None:
    """Missing .env file is not an error — the function returns without raising."""
    missing = tmp_path / "does_not_exist.env"

    _load_dotenv(missing)  # must not raise


def test_load_dotenv_reads_simple_key_equals_value_into_os_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`FOO=bar` in the file populates `os.environ['FOO'] = 'bar'`."""
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\n")
    monkeypatch.delenv("FOO", raising=False)

    _load_dotenv(env_file)

    import os
    assert os.environ["FOO"] == "bar"


def test_load_dotenv_skips_blank_lines_and_hash_comments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blank lines and lines starting with `#` are ignored — neighboring keys still set."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n"
        "# this is a comment\n"
        "FOO=bar\n"
        "\n"
        "# another comment\n"
        "BAZ=qux\n"
    )
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAZ", raising=False)

    _load_dotenv(env_file)

    import os
    assert os.environ["FOO"] == "bar"
    assert os.environ["BAZ"] == "qux"


def test_load_dotenv_strips_export_prefix_so_shell_export_syntax_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`export FOO=bar` is treated as `FOO=bar` — supports source-able .env files."""
    env_file = tmp_path / ".env"
    env_file.write_text("export FOO=bar\n")
    monkeypatch.delenv("FOO", raising=False)

    _load_dotenv(env_file)

    import os
    assert os.environ["FOO"] == "bar"


def test_load_dotenv_strips_paired_double_quotes_from_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`FOO="bar baz"` yields the value `bar baz` (quotes are NOT stored)."""
    env_file = tmp_path / ".env"
    env_file.write_text('FOO="bar baz"\n')
    monkeypatch.delenv("FOO", raising=False)

    _load_dotenv(env_file)

    import os
    assert os.environ["FOO"] == "bar baz"


def test_load_dotenv_strips_paired_single_quotes_from_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`FOO='bar baz'` yields the value `bar baz` (single quotes also stripped)."""
    env_file = tmp_path / ".env"
    env_file.write_text("FOO='bar baz'\n")
    monkeypatch.delenv("FOO", raising=False)

    _load_dotenv(env_file)

    import os
    assert os.environ["FOO"] == "bar baz"


def test_load_dotenv_does_not_strip_mismatched_or_unbalanced_quotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`FOO="bar` (only opening quote) keeps the quote — the parser only strips paired matching quotes."""
    env_file = tmp_path / ".env"
    env_file.write_text('FOO="bar\n')
    monkeypatch.delenv("FOO", raising=False)

    _load_dotenv(env_file)

    import os
    assert os.environ["FOO"] == '"bar'


def test_load_dotenv_does_not_override_already_set_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shell env wins: if `FOO` is already in os.environ, the .env value is IGNORED."""
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=from_file\n")
    monkeypatch.setenv("FOO", "from_shell")

    _load_dotenv(env_file)

    import os
    assert os.environ["FOO"] == "from_shell"


def test_load_dotenv_skips_lines_without_equals_sign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed lines (no `=`) are silently ignored, neighboring valid lines still applied."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "this is junk with no equals sign\n"
        "FOO=bar\n"
    )
    monkeypatch.delenv("FOO", raising=False)

    _load_dotenv(env_file)

    import os
    assert os.environ["FOO"] == "bar"


# --------------------------------------------------------------------------- #
# build_parser: argument shape and defaults
# --------------------------------------------------------------------------- #

def test_parser_exits_with_systemexit_when_no_input_path_given() -> None:
    """`input` is positional and required — argparse exits with SystemExit on missing."""
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_defaults_match_documented_pipeline_defaults() -> None:
    """Defaults: mode=reels, no config (→ packaged default.yaml), no overrides.

    Locking these so a behavior change shows up as a one-line test diff.
    """
    parser = build_parser()
    args = parser.parse_args(["video.mp4"])

    assert args.input == Path("video.mp4")
    assert args.mode == "reels"
    assert args.config is None
    assert args.output_dir is None
    assert args.language is None
    assert args.llm_provider is None
    assert args.max_clips is None
    assert args.debug_crop is False
    assert args.debug_detect is False
    assert args.no_cache is False
    assert args.keep_cache is False
    assert args.verbose is False


def test_parser_mode_choices_are_reels_and_trailer_only() -> None:
    """`--mode` accepts exactly `reels` or `trailer` — any other value exits."""
    parser = build_parser()

    parser.parse_args(["video.mp4", "--mode", "reels"])
    parser.parse_args(["video.mp4", "--mode", "trailer"])

    with pytest.raises(SystemExit):
        parser.parse_args(["video.mp4", "--mode", "invalid_mode"])


def test_parser_llm_provider_choices_are_claude_cli_and_litellm() -> None:
    """`--llm-provider` choices are {claude_cli, litellm} after Phase 1."""
    parser = build_parser()

    parser.parse_args(["video.mp4", "--llm-provider", "claude_cli"])
    parser.parse_args(["video.mp4", "--llm-provider", "litellm"])

    with pytest.raises(SystemExit):
        parser.parse_args(["video.mp4", "--llm-provider", "anthropic_api"])


# --------------------------------------------------------------------------- #
# apply_cli_overrides: flag → cfg.<path> mappings
# --------------------------------------------------------------------------- #

def _make_minimal_cfg() -> SimpleNamespace:
    """Build the minimum cfg shape apply_cli_overrides reads/writes."""
    return SimpleNamespace(
        paths=SimpleNamespace(output_dir="outputs"),
        transcribe=SimpleNamespace(language=None),
        llm=SimpleNamespace(provider="claude_cli"),
        analyze=SimpleNamespace(target_clips=10),
        crop=SimpleNamespace(debug_overlay=False),
        detect=SimpleNamespace(debug_overlay=False),
        logging=SimpleNamespace(level="INFO"),
    )


def _make_default_args() -> SimpleNamespace:
    """Build an args object matching `build_parser().parse_args(['video.mp4'])`."""
    return SimpleNamespace(
        output_dir=None,
        language=None,
        whisper_engine=None,
        llm_provider=None,
        max_clips=None,
        limit_minutes=None,
        debug_crop=False,
        debug_detect=False,
        subtitle_style=None,
        intro_zoom=False,
        music=False,
        verbose=False,
    )


def test_apply_cli_overrides_is_noop_when_all_flags_unset() -> None:
    """With default args, cfg passes through unchanged."""
    cfg = _make_minimal_cfg()
    args = _make_default_args()

    apply_cli_overrides(cfg, args)

    assert cfg.paths.output_dir == "outputs"
    assert cfg.transcribe.language is None
    assert cfg.llm.provider == "claude_cli"
    assert cfg.analyze.target_clips == 10
    assert cfg.crop.debug_overlay is False
    assert cfg.detect.debug_overlay is False
    assert cfg.logging.level == "INFO"


def test_apply_cli_overrides_sets_output_dir_as_string_not_path() -> None:
    """`--output-dir foo` → `cfg.paths.output_dir = 'foo'` (str, not Path)."""
    cfg = _make_minimal_cfg()
    args = _make_default_args()
    args.output_dir = Path("custom_out")

    apply_cli_overrides(cfg, args)

    assert cfg.paths.output_dir == "custom_out"
    assert isinstance(cfg.paths.output_dir, str)


def test_apply_cli_overrides_sets_transcribe_language() -> None:
    """`--language hi` → `cfg.transcribe.language = 'hi'`."""
    cfg = _make_minimal_cfg()
    args = _make_default_args()
    args.language = "hi"

    apply_cli_overrides(cfg, args)

    assert cfg.transcribe.language == "hi"


def test_apply_cli_overrides_sets_llm_provider() -> None:
    """`--llm-provider litellm` → `cfg.llm.provider = 'litellm'`."""
    cfg = _make_minimal_cfg()
    args = _make_default_args()
    args.llm_provider = "litellm"

    apply_cli_overrides(cfg, args)

    assert cfg.llm.provider == "litellm"


def test_apply_cli_overrides_sets_max_clips_via_analyze_target_clips() -> None:
    """`--max-clips 5` → `cfg.analyze.target_clips = 5` (the CLI flag name differs from the cfg key)."""
    cfg = _make_minimal_cfg()
    args = _make_default_args()
    args.max_clips = 5

    apply_cli_overrides(cfg, args)

    assert cfg.analyze.target_clips == 5


def test_apply_cli_overrides_sets_crop_debug_overlay_when_debug_crop_flag_true() -> None:
    """`--debug-crop` → `cfg.crop.debug_overlay = True`."""
    cfg = _make_minimal_cfg()
    args = _make_default_args()
    args.debug_crop = True

    apply_cli_overrides(cfg, args)

    assert cfg.crop.debug_overlay is True


def test_apply_cli_overrides_sets_detect_debug_overlay_when_debug_detect_flag_true() -> None:
    """`--debug-detect` → `cfg.detect.debug_overlay = True`."""
    cfg = _make_minimal_cfg()
    args = _make_default_args()
    args.debug_detect = True

    apply_cli_overrides(cfg, args)

    assert cfg.detect.debug_overlay is True


def test_apply_cli_overrides_leaves_subtitles_style_when_flag_omitted() -> None:
    """No `--subtitle-style` → `cfg.subtitles.style` is not touched."""
    cfg = _make_minimal_cfg()
    cfg.subtitles = SimpleNamespace(style="classic")
    args = _make_default_args()  # subtitle_style=None

    apply_cli_overrides(cfg, args)

    assert cfg.subtitles.style == "classic"


def test_build_parser_rejects_unknown_subtitle_style() -> None:
    """argparse `choices=` rejects values outside {classic, pop}."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["video.mp4", "--subtitle-style", "neon"])


def test_apply_cli_overrides_sets_subtitles_style_when_flag_pop() -> None:
    """`--subtitle-style pop` → `cfg.subtitles.style = 'pop'`."""
    cfg = _make_minimal_cfg()
    cfg.subtitles = SimpleNamespace(style="classic")
    args = _make_default_args()
    args.subtitle_style = "pop"

    apply_cli_overrides(cfg, args)

    assert cfg.subtitles.style == "pop"


def test_apply_cli_overrides_sets_logging_level_to_debug_when_verbose_true() -> None:
    """`--verbose` → `cfg.logging.level = 'DEBUG'`."""
    cfg = _make_minimal_cfg()
    args = _make_default_args()
    args.verbose = True

    apply_cli_overrides(cfg, args)

    assert cfg.logging.level == "DEBUG"


# --------------------------------------------------------------------------- #
# main() error-path return codes
# --------------------------------------------------------------------------- #

def test_main_returns_2_when_input_video_file_does_not_exist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing input → return code 2, error message on stderr, pipeline never invoked."""
    missing_input = tmp_path / "no_such_video.mp4"

    rc = main([str(missing_input)])

    assert rc == 2
    captured = capsys.readouterr()
    assert "input video not found" in captured.err


def test_main_returns_2_when_config_file_does_not_exist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing config → return code 2 (input check passes but config check fails)."""
    fake_input = tmp_path / "fake.mp4"
    fake_input.write_text("not a real video, but exists")
    missing_config = tmp_path / "missing.yaml"

    rc = main([str(fake_input), "-c", str(missing_config)])

    assert rc == 2
    captured = capsys.readouterr()
    assert "config file not found" in captured.err


def test_main_with_no_config_flag_dispatches_to_load_default_config(
    tmp_path: Path, mocker,
) -> None:
    """No -c flag → main() calls load_default_config() (NOT load_config) and runs the pipeline.

    Guards the install-time happy path: `podclipper video.mp4` (no -c) must
    use the in-package default.yaml via importlib.resources, not look for a
    file on disk. Without this test, an inverted condition or a missing
    import would only surface at user runtime.
    """
    fake_input = tmp_path / "fake.mp4"
    fake_input.write_text("not a real video, but exists")
    spy_default = mocker.spy(__import__("podclipper.main", fromlist=["load_default_config"]), "load_default_config")
    spy_disk = mocker.spy(__import__("podclipper.main", fromlist=["load_config"]), "load_config")
    mock_run = mocker.patch("podclipper.main.run_pipeline")

    rc = main([str(fake_input)])

    assert rc == 0
    assert spy_default.call_count == 1, "load_default_config() must be called when -c is omitted"
    assert spy_disk.call_count == 0, "load_config() (filesystem path) must NOT be called when -c is omitted"
    assert mock_run.call_count == 1, "run_pipeline must still be invoked with the packaged cfg"
