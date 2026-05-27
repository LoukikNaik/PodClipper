"""Tests for the index transcript-cleanup LLM layer. The LLM call is injected
(a fake `complete` fn) so these run with no network."""

from __future__ import annotations

import json

from podclipper.index.cleanup import clean_index
from podclipper.index.schema import Scene, VideoIndex


def _idx() -> VideoIndex:
    def sc(i, t):
        return Scene(id=i, start=float(i), end=float(i) + 1, duration=1.0,
                     visual="", on_screen_text="", mood="", transcript=t,
                     words=[], has_speech=bool(t))
    return VideoIndex(
        source="x", duration_seconds=10.0, vision_model="g", transcript_model="w",
        scenes=[sc(0, ""), sc(1, "आज परफॉरमेंस दूँगा"), sc(2, "मंजीट डागर")],
    )


def test_clean_index_only_sends_speech_scenes_and_applies_fixes() -> None:
    """Only scenes with a transcript are sent to the LLM; returned fixes land in
    transcript_clean, keyed by id; the raw transcript is left untouched."""
    sent = {}

    def fake_complete(system_prompt: str, user_prompt: str) -> str:
        sent["user"] = user_prompt
        return json.dumps([
            {"id": 1, "text": "aaj performance dunga"},
            {"id": 2, "text": "Manjeet Dagar"},
        ])

    out = clean_index(_idx(), fake_complete)

    # only speech-scene ids (1,2) were sent, not the empty scene 0
    assert "\"id\": 1" in sent["user"] and "\"id\": 2" in sent["user"]
    assert "\"id\": 0" not in sent["user"]
    # fixes applied to transcript_clean, raw transcript preserved
    by_id = {s.id: s for s in out.scenes}
    assert by_id[1].transcript_clean == "aaj performance dunga"
    assert by_id[1].transcript == "आज परफॉरमेंस दूँगा"
    assert by_id[2].transcript_clean == "Manjeet Dagar"
    assert by_id[0].transcript_clean == ""  # never had speech


def test_clean_index_preserves_scenes_missing_from_response() -> None:
    """If the LLM omits a scene id, that scene's transcript_clean stays empty
    (we never lose the raw transcript)."""
    def fake_complete(system_prompt: str, user_prompt: str) -> str:
        return json.dumps([{"id": 1, "text": "aaj performance dunga"}])  # drops id 2

    out = clean_index(_idx(), fake_complete)
    by_id = {s.id: s for s in out.scenes}
    assert by_id[1].transcript_clean == "aaj performance dunga"
    assert by_id[2].transcript_clean == ""
    assert by_id[2].transcript == "मंजीट डागर"  # raw intact


def test_clean_index_batches_speech_scenes() -> None:
    """With batch_size=1, each speech scene is sent in its own LLM call (so a
    big video doesn't blow the request timeout)."""
    calls = []

    def fake_complete(system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt)
        calls.append([p["id"] for p in payload])
        return json.dumps([{"id": p["id"], "text": f"fixed-{p['id']}"} for p in payload])

    out = clean_index(_idx(), fake_complete, batch_size=1)

    assert calls == [[1], [2]]  # two single-scene calls, empty scene 0 skipped
    by_id = {s.id: s for s in out.scenes}
    assert by_id[1].transcript_clean == "fixed-1"
    assert by_id[2].transcript_clean == "fixed-2"
