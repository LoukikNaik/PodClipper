"""Tests for the video-index schema — the contract between the indexer
(Phase 1, MLX env) and the editing agent (Phase 2, LiteLLM env).

Pure dataclasses + (de)serialization; no models, no IO.
"""

from __future__ import annotations

from podclipper.index.schema import Scene, VideoIndex


def _scene(**kw) -> Scene:
    base = dict(
        id=0, start=0.0, end=5.0, duration=5.0,
        visual="a person speaks to camera", on_screen_text="HELLO",
        mood="calm", transcript="hi there", words=[],
        has_speech=True, shot_type="single",
    )
    base.update(kw)
    return Scene(**base)


def test_scene_round_trips_through_dict() -> None:
    """Scene.from_dict(s.to_dict()) reconstructs an equal Scene."""
    s = _scene()

    assert Scene.from_dict(s.to_dict()) == s


def test_scene_carries_clean_transcript_field() -> None:
    """transcript_clean (LLM-fixed + transliterated) round-trips and defaults
    to empty when absent from the dict (back-compat with older indexes)."""
    s = _scene(transcript_clean="aaj best performance dunga")
    assert Scene.from_dict(s.to_dict()).transcript_clean == "aaj best performance dunga"

    legacy = {k: v for k, v in s.to_dict().items() if k != "transcript_clean"}
    assert Scene.from_dict(legacy).transcript_clean == ""


def test_video_index_round_trips_through_json() -> None:
    """VideoIndex.from_json(idx.to_json()) reconstructs an equal index, and
    the JSON carries the contract's top-level keys."""
    import json

    idx = VideoIndex(
        source="x.mp4", duration_seconds=10.0,
        vision_model="gemma", transcript_model="whisper",
        scenes=[_scene(id=0), _scene(id=1, start=5.0, end=10.0)],
    )

    payload = idx.to_json()
    parsed = json.loads(payload)
    assert set(parsed) >= {"source", "duration_seconds", "scene_count", "scenes"}
    assert parsed["scene_count"] == 2

    assert VideoIndex.from_json(payload) == idx


def test_markdown_has_scene_headers_and_typed_lines() -> None:
    """to_markdown emits one [MM:SS–MM:SS] header per scene with typed lines,
    and omits empty fields."""
    idx = VideoIndex(
        source="x.mp4", duration_seconds=10.0,
        vision_model="gemma", transcript_model="whisper",
        scenes=[
            _scene(id=0, start=0.0, end=5.0, visual="two people talk",
                   on_screen_text="EP 1", transcript="hello world"),
            _scene(id=1, start=5.0, end=8.0, visual="a wide landscape",
                   on_screen_text="", transcript=""),  # no text / no speech
        ],
    )

    md = idx.to_markdown()

    # one header per scene, with timestamps
    assert md.count("## ") == 2
    assert "00:00" in md and "00:05" in md
    # typed lines present for scene 0
    assert "two people talk" in md
    assert "EP 1" in md
    assert "hello world" in md
    # empty on-screen text line omitted for scene 1 (no stray "EP 1" leak)
    assert "a wide landscape" in md
    assert md.count("on-screen") == 1  # only scene 0 had on-screen text
