"""Characterization tests for `src/types.py` — shared dataclasses.

Module under test: BBox, Word, Clip, Transcript, TimelineSegment, etc.

Most are plain dataclasses — only the computed properties / methods need
locking down:
  - `BBox.x_center`, `y_center`, `area`
  - `BBox.iou` (intersection-over-union)
  - `Clip.duration`
  - `Transcript.full_text`
"""

from __future__ import annotations

import pytest

from podclipper.types import BBox, Clip, Transcript, TranscriptSegment


# --------------------------------------------------------------------------- #
# BBox properties
# --------------------------------------------------------------------------- #

def test_bbox_x_center_is_x_plus_half_width() -> None:
    b = BBox(x=100, y=200, w=400, h=600)
    assert b.x_center == 300


def test_bbox_y_center_is_y_plus_half_height() -> None:
    b = BBox(x=100, y=200, w=400, h=600)
    assert b.y_center == 500


def test_bbox_area_is_w_times_h() -> None:
    b = BBox(x=0, y=0, w=10, h=20)
    assert b.area == 200


# --------------------------------------------------------------------------- #
# BBox.iou
# --------------------------------------------------------------------------- #

def test_bbox_iou_returns_1_for_identical_boxes() -> None:
    a = BBox(x=0, y=0, w=10, h=10)
    b = BBox(x=0, y=0, w=10, h=10)
    assert a.iou(b) == 1.0


def test_bbox_iou_returns_0_for_non_overlapping_boxes() -> None:
    a = BBox(x=0, y=0, w=10, h=10)
    b = BBox(x=100, y=100, w=10, h=10)
    assert a.iou(b) == 0.0


def test_bbox_iou_computes_correct_partial_overlap() -> None:
    """Two 10x10 boxes overlapping by 5x5 → intersection 25, union 175, IoU = 1/7."""
    a = BBox(x=0, y=0, w=10, h=10)
    b = BBox(x=5, y=5, w=10, h=10)
    assert a.iou(b) == pytest.approx(25 / 175, abs=1e-6)


def test_bbox_iou_returns_0_when_union_is_zero_degenerate_case() -> None:
    """Both boxes have zero area → no union → IoU = 0 (avoids divide-by-zero)."""
    a = BBox(x=0, y=0, w=0, h=0)
    b = BBox(x=0, y=0, w=0, h=0)
    assert a.iou(b) == 0.0


# --------------------------------------------------------------------------- #
# Clip.duration
# --------------------------------------------------------------------------- #

def test_clip_duration_is_end_minus_start() -> None:
    c = Clip(start=10.5, end=42.0, title="t", reason="r")
    assert c.duration == pytest.approx(31.5)


# --------------------------------------------------------------------------- #
# Transcript.full_text
# --------------------------------------------------------------------------- #

def test_transcript_full_text_joins_stripped_segment_texts_with_single_space() -> None:
    t = Transcript(language="en", segments=[
        TranscriptSegment(start=0, end=5, text="  hello  ", words=[]),
        TranscriptSegment(start=5, end=10, text="world", words=[]),
    ])
    assert t.full_text == "hello world"


def test_transcript_full_text_returns_empty_string_for_no_segments() -> None:
    t = Transcript(language="en", segments=[])
    assert t.full_text == ""
