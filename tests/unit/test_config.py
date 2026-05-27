"""Characterization tests for `src/config.py`.

Module under test: the YAML config loader and namespace converter.

Functionality being locked down here:
  - `load_config(path)`           — reads a YAML file from disk and returns
                                    a SimpleNamespace tree with dotted access.
  - `ns_to_dict(ns)`              — inverse of the loader: converts a
                                    SimpleNamespace tree back to a plain dict
                                    (for logging / serialization).
  - container handling             — how dicts, lists-of-dicts, lists-of-scalars,
                                    and bare scalars survive the round-trip.
  - filesystem error surface       — what happens when the YAML file is missing.
  - real-config smoke              — that `config/default.yaml` actually loads
                                    and exposes the top-level sections the
                                    pipeline depends on.

These tests are characterization tests (Feathers). They document current
behavior. They are expected to pass on first run. A failure here means
either the code does something different than we thought (record it as
SURPRISE in a comment, update the assertion to match reality) or the
test has a plumbing bug (fix it and re-run).

DO NOT modify `src/config.py` to make a failing test pass.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from podclipper.config import load_config, ns_to_dict


# --------------------------------------------------------------------------- #
# load_config: container handling
# --------------------------------------------------------------------------- #

def test_load_config_returns_simplenamespace_with_dotted_access_for_top_level_keys(
    tmp_path: Path,
) -> None:
    """A flat YAML mapping becomes a SimpleNamespace whose attributes are the keys."""
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("foo: bar\nnumber: 42\n")

    cfg = load_config(yaml_file)

    assert isinstance(cfg, SimpleNamespace)
    assert cfg.foo == "bar"
    assert cfg.number == 42


def test_load_config_converts_nested_dicts_to_nested_simplenamespaces(
    tmp_path: Path,
) -> None:
    """Nested YAML mappings become nested SimpleNamespaces, all the way down."""
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text(
        "a:\n"
        "  b:\n"
        "    c: 42\n"
    )

    cfg = load_config(yaml_file)

    assert isinstance(cfg.a, SimpleNamespace)
    assert isinstance(cfg.a.b, SimpleNamespace)
    assert cfg.a.b.c == 42


def test_load_config_preserves_lists_of_dicts_as_lists_of_simplenamespaces(
    tmp_path: Path,
) -> None:
    """A YAML list of mappings becomes a list whose items are SimpleNamespaces."""
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text(
        "items:\n"
        "  - name: alice\n"
        "    age: 30\n"
        "  - name: bob\n"
        "    age: 25\n"
    )

    cfg = load_config(yaml_file)

    assert isinstance(cfg.items, list)
    assert len(cfg.items) == 2
    assert isinstance(cfg.items[0], SimpleNamespace)
    assert cfg.items[0].name == "alice"
    assert cfg.items[1].name == "bob"


def test_load_config_preserves_lists_of_scalars_as_plain_lists(
    tmp_path: Path,
) -> None:
    """A YAML list of scalars stays a plain list (no SimpleNamespace wrapping)."""
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("names: [alice, bob, carol]\n")

    cfg = load_config(yaml_file)

    assert cfg.names == ["alice", "bob", "carol"]


def test_load_config_accepts_string_path_as_well_as_path_object(
    tmp_path: Path,
) -> None:
    """The signature `path: str | Path` actually accepts both."""
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("foo: bar\n")

    cfg_from_str = load_config(str(yaml_file))
    cfg_from_path = load_config(yaml_file)

    assert cfg_from_str.foo == cfg_from_path.foo == "bar"


# --------------------------------------------------------------------------- #
# load_config: filesystem error surface
# --------------------------------------------------------------------------- #

def test_load_config_raises_filenotfounderror_when_path_does_not_exist(
    tmp_path: Path,
) -> None:
    """No file → FileNotFoundError bubbles up from `path.open()`."""
    missing = tmp_path / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError):
        load_config(missing)


# --------------------------------------------------------------------------- #
# ns_to_dict: round-trip and pass-through behavior
# --------------------------------------------------------------------------- #

def test_ns_to_dict_converts_simplenamespace_tree_to_plain_dict(
    tmp_path: Path,
) -> None:
    """Inverse of `load_config` for nested SimpleNamespaces."""
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text(
        "a:\n"
        "  b:\n"
        "    c: 42\n"
    )
    cfg = load_config(yaml_file)

    as_dict = ns_to_dict(cfg)

    assert as_dict == {"a": {"b": {"c": 42}}}


def test_ns_to_dict_converts_lists_of_simplenamespaces_back_to_lists_of_dicts(
    tmp_path: Path,
) -> None:
    """Lists-of-namespaces collapse back into lists-of-dicts."""
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text(
        "items:\n"
        "  - name: alice\n"
        "  - name: bob\n"
    )
    cfg = load_config(yaml_file)

    as_dict = ns_to_dict(cfg)

    assert as_dict == {"items": [{"name": "alice"}, {"name": "bob"}]}


def test_ns_to_dict_passes_scalars_through_unchanged() -> None:
    """Non-namespace, non-list values are returned as-is."""
    assert ns_to_dict("hello") == "hello"
    assert ns_to_dict(42) == 42
    assert ns_to_dict(3.14) == 3.14
    assert ns_to_dict(None) is None
    assert ns_to_dict(True) is True


def test_ns_to_dict_passes_plain_lists_through_recursively() -> None:
    """A bare list of scalars is returned as a list of the same scalars."""
    assert ns_to_dict([1, 2, 3]) == [1, 2, 3]
    assert ns_to_dict(["a", "b"]) == ["a", "b"]


# --------------------------------------------------------------------------- #
# Smoke test: the real config/default.yaml
# --------------------------------------------------------------------------- #

def test_load_default_config_returns_simplenamespace_with_pipeline_sections() -> None:
    """`load_default_config()` reads the YAML bundled inside the package itself.

    No filesystem path required — uses importlib.resources so it works from a
    wheel install where the YAML lives inside site-packages.
    """
    from podclipper.config import load_default_config

    cfg = load_default_config()

    assert isinstance(cfg, SimpleNamespace)
    expected_sections = {
        "logging", "paths", "transcribe", "analyze",
        "crop", "detect", "llm", "subtitles",
    }
    assert expected_sections <= set(vars(cfg).keys())


# Real-config smoke is now covered by
# test_load_default_config_returns_simplenamespace_with_pipeline_sections above,
# which loads via importlib.resources from the in-package default.yaml.


# --------------------------------------------------------------------------- #
# transcribe engine selection
# --------------------------------------------------------------------------- #

def test_default_config_transcribe_engine_is_faster_whisper() -> None:
    """`cfg.transcribe.engine` defaults to `faster_whisper` for back-compat.

    Adding mlx-whisper as an alternative — existing users get the unchanged
    behavior unless they explicitly switch.
    """
    from podclipper.config import load_default_config

    cfg = load_default_config()

    assert cfg.transcribe.engine == "faster_whisper"


def test_default_config_first_pass_mlx_repo_points_to_base_model() -> None:
    """First-pass mlx_repo defaults to the base model — matches the size used
    for faster-whisper first-pass (fast, transcript is LLM-consumed not user-facing)."""
    from podclipper.config import load_default_config

    cfg = load_default_config()

    assert cfg.transcribe.first_pass.mlx_repo == "mlx-community/whisper-base-mlx"


def test_default_config_second_pass_mlx_repo_points_to_large_v3() -> None:
    """Second-pass mlx_repo defaults to large-v3 — accuracy matters for captions."""
    from podclipper.config import load_default_config

    cfg = load_default_config()

    assert cfg.transcribe.second_pass.mlx_repo == "mlx-community/whisper-large-v3-mlx"
