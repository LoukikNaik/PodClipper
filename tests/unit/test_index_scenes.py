"""Tests for scene segmentation — pure logic (ffmpeg-output parsing + the
cut→scene boundary algorithm). No subprocess/IO here."""

from __future__ import annotations

from podclipper.index.scenes import cuts_to_scenes, parse_showinfo, words_in_range


def test_parse_showinfo_extracts_sorted_pts_times() -> None:
    """Pull pts_time floats out of ffmpeg showinfo stderr, sorted ascending."""
    stderr = (
        "frame info ... pts_time:12.500 ... \n"
        "noise\n"
        "[Parsed_showinfo] n:1 pts_time:3.20 type:I\n"
        "[Parsed_showinfo] n:2 pts_time:25.0 type:P\n"
    )

    assert parse_showinfo(stderr) == [3.2, 12.5, 25.0]


def test_no_cuts_splits_into_even_windows() -> None:
    """No detected cuts → even sub-windows so captions stay frame-distinct."""
    scenes = cuts_to_scenes([], duration=30.0, min_len=1.5, max_len=8.0)
    assert len(scenes) == 4
    assert scenes[0] == (0.0, 7.5)
    assert scenes[-1][1] == 30.0


def test_real_cuts_become_scene_boundaries() -> None:
    """Cuts inside [0, duration] become scene edges."""
    scenes = cuts_to_scenes([5.0, 7.0], duration=12.0, min_len=1.5, max_len=8.0)
    assert (0.0, 5.0) in scenes
    assert (5.0, 7.0) in scenes


def test_short_scene_after_long_one_is_absorbed() -> None:
    """A sub-min_len scene is merged into the previous one regardless of how
    long the previous scene is (fixes 0.1s sliver scenes from clustered cuts)."""
    # cut at 5.08 creates a 0.08s sliver right after a 5s scene
    scenes = cuts_to_scenes([5.0, 5.08], duration=12.0, min_len=1.5, max_len=8.0)
    # no scene shorter than min_len
    assert all((e - s) >= 1.5 for s, e in scenes), scenes
    assert (0.0, 5.08) in scenes


def test_long_scene_is_split_under_max_len() -> None:
    """A scene longer than max_len is sub-divided; none exceed max_len."""
    scenes = cuts_to_scenes([], duration=40.0, min_len=1.5, max_len=8.0)
    assert all((e - s) <= 8.0 + 1e-9 for s, e in scenes)


def test_words_in_range_selects_by_midpoint_and_joins() -> None:
    """Words whose midpoint falls in [start, end) are joined and stripped."""
    words = [
        {"start": 0.0, "end": 0.4, "text": " hello"},
        {"start": 0.5, "end": 0.9, "text": " world"},
        {"start": 5.0, "end": 5.4, "text": " later"},   # outside
    ]
    assert words_in_range(words, 0.0, 1.0) == "hello world"
    assert words_in_range(words, 4.0, 6.0) == "later"
