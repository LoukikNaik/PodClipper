"""Load validated YAML overrides over packaged pipeline defaults."""

from importlib import resources
from pathlib import Path
import math
from types import SimpleNamespace
from typing import Any

import yaml


def _to_ns(obj: Any) -> Any:
    """Convert nested mappings to namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _merge(defaults: dict, overrides: dict, prefix: str = "") -> dict:
    """Merge known keys recursively, replacing lists and scalar values."""
    result = dict(defaults)
    for key, value in overrides.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if key not in defaults:
            raise ValueError(f"Unknown config setting: {name}")
        original = defaults[key]
        if isinstance(original, dict):
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a mapping")
            result[key] = _merge(original, value, name)
        else:
            if original is not None:
                valid = (isinstance(value, (int, float)) and not isinstance(value, bool)
                         if isinstance(original, float)
                         else isinstance(value, type(original)))
                if not valid:
                    raise ValueError(f"Invalid value type for {name}")
            result[key] = value
    return result


def _resolve(raw: dict, overrides: dict) -> SimpleNamespace:
    """Resolve public settings into pipeline fields; explicit internal overrides win."""
    aliases = {
        "content": {"type": ("crop", "comedy_mode"), "clips": ("analyze", "target_clips"),
                    "duration": ("analyze", "min_clip_seconds")},
        "output": {"directory": ("paths", "output_dir"), "captions": ("subtitles", "style"),
                   "music": ("music", "enabled"), "intro_zoom": ("crop", "intro_zoom")},
    }
    merged = _merge(raw, overrides)
    llm = merged["llm"]
    for key in ("max_tokens", "retry_max_tokens"):
        if isinstance(llm[key], bool) or not isinstance(llm[key], int) or llm[key] < 1:
            raise ValueError(f"llm.{key} must be a positive integer")
    if llm["num_retries"] < 0 or isinstance(llm["num_retries"], bool):
        raise ValueError("llm.num_retries must be a nonnegative integer")
    if llm["timeout_seconds"] <= 0 or isinstance(llm["timeout_seconds"], bool):
        raise ValueError("llm.timeout_seconds must be greater than zero")
    repair = merged["transcribe"]["caption_repair"]
    for key in ("window_seconds", "min_speech_seconds", "min_words_per_speech_second",
                "max_word_seconds", "max_retry_seconds", "reference_ratio"):
        if isinstance(repair[key], bool) or not math.isfinite(repair[key]) or repair[key] <= 0:
            raise ValueError(f"transcribe.caption_repair.{key} must be finite and greater than zero")
    for key in ("max_attempts", "max_windows", "reference_min_words"):
        if type(repair[key]) is not int or repair[key] < 1:
            raise ValueError(f"transcribe.caption_repair.{key} must be a positive integer")
    if (isinstance(repair["context_seconds"], bool) or not math.isfinite(repair["context_seconds"])
            or repair["context_seconds"] < 0 or repair["max_retry_seconds"] <= 2 * repair["context_seconds"]):
        raise ValueError("transcribe.caption_repair.max_retry_seconds must exceed twice the nonnegative context_seconds")
    if repair["reference_ratio"] > 1:
        raise ValueError("transcribe.caption_repair.reference_ratio must be at most one")
    dividers = merged["crop"]["panel_dividers"]
    cuts = merged["crop"]["scene_cuts"]
    if not isinstance(cuts["enabled"], bool):
        raise ValueError("crop.scene_cuts.enabled must be a boolean")
    if (isinstance(cuts["crossfade_seconds"], bool)
            or not 0 <= cuts["crossfade_seconds"] <= 1):
        raise ValueError("crop.scene_cuts.crossfade_seconds must be between zero and one")
    if (isinstance(cuts["difference_threshold"], bool)
            or not 0 < cuts["difference_threshold"] <= 255):
        raise ValueError("crop.scene_cuts.difference_threshold must be greater than zero and at most 255")
    for key, maximum in (("min_coverage", 1.0), ("max_width_frac", 0.05),
                         ("min_panel_frac", 0.5), ("min_contrast", 255.0)):
        if isinstance(dividers[key], bool) or not 0 < dividers[key] <= maximum:
            raise ValueError(f"crop.panel_dividers.{key} must be greater than zero and at most {maximum}")
    if isinstance(dividers["margin_pixels"], bool) or dividers["margin_pixels"] < 1:
        raise ValueError("crop.panel_dividers.margin_pixels must be at least 1")
    content, output = merged["content"], merged["output"]
    if content["type"] not in ("podcast", "comedy"):
        raise ValueError("content.type must be podcast or comedy")
    if output["captions"] not in ("classic", "pop", "off"):
        raise ValueError("output.captions must be classic, pop or off")
    duration = content["duration"]
    if (len(duration) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in duration)
            or not 0 < duration[0] <= duration[1]):
        raise ValueError("content.duration must contain positive minimum and maximum seconds")
    if not isinstance(content["clips"], int) or content["clips"] < 1:
        raise ValueError("content.clips must be a positive integer")
    for section, entries in aliases.items():
        for key, (target, field) in entries.items():
            value = merged[section][key]
            if key == "type":
                value = value == "comedy"
            if key == "duration":
                for bound, seconds in zip(("min_clip_seconds", "max_clip_seconds"), value):
                    if bound not in overrides.get(target, {}):
                        merged[target][bound] = seconds
            elif key == "intro_zoom":
                if "enabled" not in overrides.get("crop", {}).get("intro_zoom", {}):
                    merged["crop"]["intro_zoom"]["enabled"] = value
            elif field not in overrides.get(target, {}):
                merged[target][field] = value
    if merged["subtitles"]["style"] not in ("classic", "pop", "off"):
        raise ValueError("subtitles.style must be classic, pop or off")
    content["type"] = "comedy" if merged["crop"]["comedy_mode"] else "podcast"
    content["clips"] = merged["analyze"]["target_clips"]
    content["duration"] = [merged["analyze"]["min_clip_seconds"], merged["analyze"]["max_clip_seconds"]]
    output["directory"] = merged["paths"]["output_dir"]
    output["captions"] = merged["subtitles"]["style"]
    output["music"] = merged["music"]["enabled"]
    output["intro_zoom"] = merged["crop"]["intro_zoom"]["enabled"]
    return _to_ns(merged)


def load_config(path: str | Path) -> SimpleNamespace:
    """Load partial user overrides with all omitted settings inherited."""
    overrides = yaml.safe_load(Path(path).read_text())
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("Config must be a YAML mapping")
    raw = yaml.safe_load(resources.files("podclipper.config").joinpath("default.yaml").read_text())
    return _resolve(raw, overrides)


def load_default_config() -> SimpleNamespace:
    """Load the packaged defaults without user overrides."""
    raw = yaml.safe_load(resources.files("podclipper.config").joinpath("default.yaml").read_text())
    return _resolve(raw, {})


def ns_to_dict(ns: Any) -> Any:
    """Convert namespaces to mappings for serialization."""
    if isinstance(ns, SimpleNamespace):
        return {k: ns_to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, list):
        return [ns_to_dict(v) for v in ns]
    return ns
