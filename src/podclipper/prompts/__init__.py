"""Packaged LLM prompts. Use `load_prompt(name)` for wheel-safe access."""

from __future__ import annotations

from importlib import resources


def load_prompt(name: str) -> str:
    """Return the text of a bundled prompt file (e.g. 'reel_detector.txt')."""
    return resources.files("podclipper.prompts").joinpath(name).read_text()
