"""Characterization tests for crop helpers.

Locked behavior:
  - `_pick_performer` — comedy-mode single-performer selection: among persons
    above the size floor, the lit + full-body/high person wins over audience
    (in shadow, short silhouettes low in the foreground).
"""

from __future__ import annotations

import numpy as np

from podclipper.crop import _pick_performer
from podclipper.types import BBox


def _bbox(x: float, y: float, w: float, h: float) -> BBox:
    return BBox(x=x, y=y, w=w, h=h, confidence=1.0)


SRC_W, SRC_H = 1920, 1080


def _frame() -> np.ndarray:
    return np.zeros((SRC_H, SRC_W, 3), dtype=np.uint8)


def test_pick_performer_returns_none_when_none_clears_floor() -> None:
    """All persons below the size floor → no performer."""
    frame = _frame()
    persons = [_bbox(500, 900, 80, 100)]  # 100px < 0.30*1080=324
    assert _pick_performer(frame, persons, floor=324, src_w=SRC_W, src_h=SRC_H,
                           bright_w=1.0, top_w=0.5) is None


def test_pick_performer_prefers_the_lit_person_over_shadow_audience() -> None:
    """Two floor-passing persons of similar geometry: the brighter (lit) one wins."""
    frame = _frame()
    performer = _bbox(900, 200, 300, 700)     # lit region
    audience = _bbox(300, 200, 300, 700)      # left as dark (shadow)
    frame[200:900, 900:1200] = 180            # performer bbox is bright
    frame[200:900, 300:600] = 15              # audience bbox stays near-black
    chosen = _pick_performer(frame, [audience, performer], floor=324,
                             src_w=SRC_W, src_h=SRC_H, bright_w=1.0, top_w=0.5)
    assert chosen is performer


def test_pick_performer_rejects_short_foreground_audience_head() -> None:
    """A big-but-short foreground audience head low in frame loses to the taller,
    higher standing performer even if both are equally lit."""
    frame = _frame()
    frame[:] = 120  # uniform brightness so geometry decides
    performer = _bbox(900, 150, 300, 800)     # top high (y=150), tall
    fg_head = _bbox(850, 820, 320, 240)       # low (y=820), short — foreground head
    chosen = _pick_performer(frame, [fg_head, performer], floor=200,
                             src_w=SRC_W, src_h=SRC_H, bright_w=1.0, top_w=0.5)
    assert chosen is performer


def test_pick_performer_clamps_out_of_bounds_bbox() -> None:
    """A bbox partly outside the frame is clamped, not crashed, and still scored."""
    frame = _frame()
    frame[:] = 100
    p = _bbox(-50, -30, 400, 900)  # extends past top-left
    assert _pick_performer(frame, [p], floor=200, src_w=SRC_W, src_h=SRC_H,
                           bright_w=1.0, top_w=0.5) is p
