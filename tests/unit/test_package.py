"""Lock package-level invariants: name, version, importability, CLI."""

from __future__ import annotations

import shutil
import subprocess

import pytest


def test_package_version_is_importable_from_podclipper() -> None:
    """`from podclipper import __version__` succeeds and returns a non-empty str."""
    from podclipper import __version__

    assert isinstance(__version__, str)
    assert __version__  # not the empty string


@pytest.mark.skipif(
    shutil.which("podclipper") is None,
    reason="podclipper not on PATH — run `pip install -e .` first",
)
def test_console_entry_point_podclipper_prints_help_and_exits_zero() -> None:
    """`podclipper --help` runs, exits 0, mentions the CLI name in output."""
    result = subprocess.run(
        ["podclipper", "--help"], capture_output=True, text=True, timeout=15,
    )

    assert result.returncode == 0
    assert "podclipper" in result.stdout.lower()
    assert "--llm-provider" in result.stdout
