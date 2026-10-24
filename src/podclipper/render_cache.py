"""Input fingerprints and atomic checkpoints for clip rendering."""

import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace


def fingerprint(value):
    """Hash nested configuration and artifact inputs without including credentials."""
    encoded = json.dumps(value, sort_keys=True, default=lambda obj: vars(obj) if isinstance(obj, SimpleNamespace) else str(obj))
    return hashlib.sha256(encoded.encode()).hexdigest()


def file_identity(path):
    """Record a file identity for cache invalidation."""
    info = Path(path).stat()
    return [str(Path(path).resolve()), info.st_size, info.st_mtime_ns]


def clip_cache_dir(cache, source, clip, cfg):
    """Keep cache identity independent of selection order and title."""
    key = fingerprint([file_identity(source), clip.start, clip.end, cfg.clip_extract])[:16]
    target = cache / f"clip-{key}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path, value):
    """Atomically replace a JSON artifact."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def checkpoint_matches(artifact, inputs):
    """Require matching inputs and an unchanged output artifact."""
    record = artifact.with_suffix(artifact.suffix + ".checkpoint.json")
    try:
        state = json.loads(record.read_text())
        return state == {"inputs": fingerprint(inputs), "artifact": file_identity(artifact)}
    except (OSError, ValueError, TypeError):
        return False


def checkpoint(artifact, inputs):
    """Record a completed stage and its exact output identity."""
    write_json(artifact.with_suffix(artifact.suffix + ".checkpoint.json"),
               {"inputs": fingerprint(inputs), "artifact": file_identity(artifact)})


def caption_inputs(segment, words, cfg):
    """Identify inputs that affect mapped caption cleanup."""
    return [file_identity(segment), [asdict(w) for w in words], cfg.transcribe.cleanup,
            cfg.llm.endpoint, cfg.llm.model]


def crop_inputs(segment, cfg):
    """Identify inputs that affect speaker detection and framing."""
    vision = ([cfg.llm.endpoint, cfg.llm.model]
              if getattr(getattr(cfg.crop, "scene_review", None), "enabled", False) else None)
    return ["face-and-scene-v3", file_identity(segment), cfg.detect, cfg.crop, cfg.diarize, vision]


def asr_inputs(segment, cfg):
    """Identify audio and settings that affect raw transcription."""
    options = {k: v for k, v in vars(cfg.transcribe).items() if k not in ("cleanup", "caption_repair", "first_pass")}
    if not getattr(options.get("scribe"), "enabled", False):
        options.pop("scribe", None)
    second_pass = dict(vars(options["second_pass"]))
    if second_pass.get("task", "transcribe") == "transcribe":
        second_pass.pop("task", None)
    if not second_pass.get("initial_prompt"):
        second_pass.pop("initial_prompt", None)
    options["second_pass"] = SimpleNamespace(**second_pass)
    return [file_identity(segment), options]


def word_provenance(source, start, end, artifact, cfg):
    """Record source bounds for reusing timed words in a trim."""
    options = asr_inputs(artifact, cfg)[1]
    return {"source": file_identity(source), "start": start, "end": end,
            "settings": fingerprint(options), "artifact": file_identity(artifact)}


def reuse_containing_words(cache, source, start, end, cfg):
    """Shift validated parent-clip timings when a repair only trims the same audio."""
    settings = fingerprint(asr_inputs(source, cfg)[1])
    for manifest in cache.glob("clip-*/words.provenance.json"):
        try:
            state = json.loads(manifest.read_text())
            artifact = manifest.with_name("words.json")
            if (state["source"] != file_identity(source) or state["settings"] != settings
                    or state["artifact"] != file_identity(artifact)
                    or not state["start"] <= start < end <= state["end"]):
                continue
            result = []
            for word in json.loads(artifact.read_text()):
                a, b = word["start"] + state["start"], word["end"] + state["start"]
                if a < end and b > start:
                    result.append(word | {"start": max(0, a - start), "end": min(end, b) - start})
            if result:
                return result
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return None
