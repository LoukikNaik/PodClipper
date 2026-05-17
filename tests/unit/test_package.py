"""Lock package-level invariants: name, version, importability."""

from __future__ import annotations


def test_package_version_is_importable_from_podclipper() -> None:
    """`from podclipper import __version__` succeeds and returns a non-empty str."""
    from podclipper import __version__

    assert isinstance(__version__, str)
    assert __version__  # not the empty string
