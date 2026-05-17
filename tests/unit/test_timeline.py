"""Characterization tests for `src/timeline.py`.

Module under test: timeline builders for the crop stage.

Behaviors locked:
  - `classify_wide_shot_frames`     — pure bbox-geometry shot classifier
                                      (active path for `crop.mode: auto`).
  - `_cluster_x_centers`            — 1D NN clustering of bbox x-centers.
  - `_longest_contiguous_run`       — gap-tolerant run-length over frame idx.
  - `_cluster_persistence`          — ratio of cluster size to total frames.
  - `apply_min_dwell`               — DEPRECATED (legacy single-mode) but pure.
  - `build_speaker_timeline`        — DEPRECATED, only sanity-check the two
                                      simple-fallback branches (no persons →
                                      CENTER, one person → PRIMARY).
"""

from __future__ import annotations

import numpy as np

from src.timeline import (
    _cluster_persistence,
    _cluster_x_centers,
    _longest_contiguous_run,
    apply_min_dwell,
    build_speaker_timeline,
    classify_wide_shot_frames,
)
from src.types import BBox, TimelineSegment


# --------------------------------------------------------------------------- #
# classify_wide_shot_frames
# --------------------------------------------------------------------------- #

def test_classify_wide_shot_returns_empty_array_for_empty_input() -> None:
    out = classify_wide_shot_frames([], source_width=1920, source_height=1080)
    assert isinstance(out, np.ndarray)
    assert out.shape == (0,)


def test_classify_wide_shot_returns_all_false_when_only_one_person_per_frame() -> None:
    """Single bbox per frame → never wide (needs ≥2 persons)."""
    persons = [[_bbox(100, 100, 200, 600)] for _ in range(30)]
    out = classify_wide_shot_frames(persons, source_width=1920, source_height=1080)
    assert not out.any()


def test_classify_wide_shot_marks_wide_when_two_short_separated_persons() -> None:
    """Two bboxes, both shorter than 70% of source height, separated by ≥ 20% of width → wide."""
    # source 1920x1080, height cap = 756, sep threshold = 384px
    persons = [
        [_bbox(100, 200, 300, 500), _bbox(1400, 200, 300, 500)]  # centers ~250 and ~1550 → sep 1300
        for _ in range(30)
    ]
    out = classify_wide_shot_frames(persons, source_width=1920, source_height=1080)
    assert out.all()


def test_classify_wide_shot_marks_false_when_persons_too_tall() -> None:
    """Persons taller than `height_cap_frac` (default 0.70) → not qualifying as wide."""
    # source 1920x1080, both persons 900 tall (> 756 = 0.70*1080)
    persons = [
        [_bbox(100, 50, 300, 900), _bbox(1400, 50, 300, 900)]
        for _ in range(30)
    ]
    out = classify_wide_shot_frames(persons, source_width=1920, source_height=1080)
    assert not out.any()


def test_classify_wide_shot_marks_false_when_persons_too_close_horizontally() -> None:
    """Separation < `sep_threshold_frac` → single shot, not wide."""
    persons = [
        [_bbox(100, 200, 300, 500), _bbox(300, 200, 300, 500)]  # centers ~250 and ~450 → sep 200 (< 384)
        for _ in range(30)
    ]
    out = classify_wide_shot_frames(persons, source_width=1920, source_height=1080)
    assert not out.any()


def test_classify_wide_shot_smooths_single_frame_flicker_to_majority() -> None:
    """One single-shot frame surrounded by wide-shot frames is smoothed to wide via the window."""
    wide_frame = [_bbox(100, 200, 300, 500), _bbox(1400, 200, 300, 500)]
    single_frame = [_bbox(800, 200, 300, 500)]
    persons = [wide_frame] * 20 + [single_frame] + [wide_frame] * 20

    out = classify_wide_shot_frames(
        persons, source_width=1920, source_height=1080,
        smooth_window_frames=15,
    )

    # The single-frame outlier should be smoothed to wide because the surrounding
    # majority within the 15-frame window is wide.
    assert out[20]  # the flicker frame is now True


# --------------------------------------------------------------------------- #
# _cluster_x_centers
# --------------------------------------------------------------------------- #

def test_cluster_x_centers_groups_nearby_centers_into_single_cluster() -> None:
    """Two bboxes at similar x → one cluster."""
    bboxes = [_bbox(100, 0, 50, 50), _bbox(120, 0, 50, 50), _bbox(140, 0, 50, 50)]
    clusters = _cluster_x_centers(bboxes, merge_tolerance_px=120.0)
    assert len(clusters) == 1
    assert clusters[0] == [0, 1, 2]


def test_cluster_x_centers_separates_distant_centers_into_separate_clusters() -> None:
    """Bboxes far apart in x → separate clusters."""
    bboxes = [_bbox(100, 0, 50, 50), _bbox(1000, 0, 50, 50)]
    clusters = _cluster_x_centers(bboxes, merge_tolerance_px=120.0)
    assert len(clusters) == 2


def test_cluster_x_centers_skips_none_bboxes() -> None:
    """None entries are skipped, indices of remaining bboxes preserved."""
    bboxes = [_bbox(100, 0, 50, 50), None, _bbox(110, 0, 50, 50)]
    clusters = _cluster_x_centers(bboxes, merge_tolerance_px=120.0)
    assert clusters == [[0, 2]]


# --------------------------------------------------------------------------- #
# _longest_contiguous_run
# --------------------------------------------------------------------------- #

def test_longest_contiguous_run_returns_span_within_gap_tolerance() -> None:
    """Frames [0,1,2,3] with gap_tolerance=1 → span 4 (3-0+1)."""
    assert _longest_contiguous_run([0, 1, 2, 3], gap_tolerance=1) == 4


def test_longest_contiguous_run_breaks_when_gap_exceeds_tolerance() -> None:
    """Gap > tolerance splits the run; returns span of longest sub-run."""
    # frames 0,1 (run 2), then big gap, then 100,101,102 (run 3)
    assert _longest_contiguous_run([0, 1, 100, 101, 102], gap_tolerance=2) == 3


def test_longest_contiguous_run_returns_zero_for_empty_input() -> None:
    assert _longest_contiguous_run([], gap_tolerance=1) == 0


# --------------------------------------------------------------------------- #
# _cluster_persistence
# --------------------------------------------------------------------------- #

def test_cluster_persistence_is_size_over_total() -> None:
    assert _cluster_persistence([1, 2, 3], total_frames=10) == 0.3


def test_cluster_persistence_returns_zero_for_zero_total_frames() -> None:
    assert _cluster_persistence([1, 2, 3], total_frames=0) == 0.0


# --------------------------------------------------------------------------- #
# apply_min_dwell (DEPRECATED)
# --------------------------------------------------------------------------- #

def _seg(start: float, end: float, label: str = "X") -> TimelineSegment:
    return TimelineSegment(start=start, end=end, label=label, bbox_at=lambda i: None)


def test_apply_min_dwell_returns_input_unchanged_when_single_segment() -> None:
    """1 segment → can't merge with anything → return as-is."""
    timeline = [_seg(0, 5)]
    assert apply_min_dwell(timeline, min_dwell_seconds=10.0) is timeline


def test_apply_min_dwell_absorbs_too_short_segment_into_previous() -> None:
    """A segment shorter than min_dwell is absorbed: end-time extends previous, label stays."""
    timeline = [_seg(0, 10, label="A"), _seg(10, 11, label="B")]
    out = apply_min_dwell(timeline, min_dwell_seconds=2.0)
    assert len(out) == 1
    assert out[0].start == 0
    assert out[0].end == 11
    assert out[0].label == "A"


def test_apply_min_dwell_keeps_segment_when_long_enough() -> None:
    """Segments >= min_dwell stay as separate segments."""
    timeline = [_seg(0, 10, label="A"), _seg(10, 20, label="B")]
    out = apply_min_dwell(timeline, min_dwell_seconds=2.0)
    assert len(out) == 2


# --------------------------------------------------------------------------- #
# build_speaker_timeline (DEPRECATED — only the two simple fallbacks)
# --------------------------------------------------------------------------- #

def test_build_speaker_timeline_returns_center_fallback_when_no_persons_detected() -> None:
    """No bbox in any frame → single CENTER segment spanning the whole clip."""
    per_frame = [None, None, None]
    out = build_speaker_timeline(
        per_frame_bboxes=per_frame, clip_duration=3.0, fps=30.0,
        source_width=1920, source_height=1080,
    )
    assert len(out) == 1
    assert out[0].label == "CENTER"
    assert out[0].start == 0.0 and out[0].end == 3.0


def test_build_speaker_timeline_returns_primary_when_one_persistent_cluster() -> None:
    """All bboxes in one cluster, persistent → single PRIMARY segment."""
    # 60 frames at fps=30 = 2s, all same person
    per_frame = [_bbox(500, 100, 200, 600) for _ in range(60)]
    out = build_speaker_timeline(
        per_frame_bboxes=per_frame, clip_duration=2.0, fps=30.0,
        source_width=1920, source_height=1080,
    )
    assert len(out) == 1
    assert out[0].label == "PRIMARY"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _bbox(x: float, y: float, w: float, h: float, conf: float = 1.0) -> BBox:
    return BBox(x=x, y=y, w=w, h=h, confidence=conf)
