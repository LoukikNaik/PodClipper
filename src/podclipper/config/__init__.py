"""YAML config loader → nested SimpleNamespace with dotted access."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    return _to_ns(raw)


def load_default_config() -> SimpleNamespace:
    """Load the YAML bundled with the package (works from a wheel install)."""
    raw = resources.files("podclipper.config").joinpath("default.yaml").read_text()
    return _to_ns(yaml.safe_load(raw))


def ns_to_dict(ns: Any) -> Any:
    """Inverse of `_to_ns` for logging / serialization."""
    if isinstance(ns, SimpleNamespace):
        return {k: ns_to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, list):
        return [ns_to_dict(v) for v in ns]
    return ns
