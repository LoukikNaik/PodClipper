"""Tests for podclipper.prompts.load_prompt — packaged-prompt loader."""

from __future__ import annotations

import pytest


def test_load_prompt_returns_file_contents_via_importlib_resources() -> None:
    """`load_prompt('reel_detector.txt')` returns the bundled prompt text."""
    from podclipper.prompts import load_prompt

    text = load_prompt("reel_detector.txt")

    assert isinstance(text, str)
    assert text.strip()  # non-empty


def test_load_prompt_raises_for_unknown_prompt_name() -> None:
    """Missing prompt → FileNotFoundError (or equivalent) — not silently empty."""
    from podclipper.prompts import load_prompt

    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist.txt")
