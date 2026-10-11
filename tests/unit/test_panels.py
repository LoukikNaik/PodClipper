"""Divider geometry and crop containment regressions."""
from types import SimpleNamespace
import numpy as np
import pytest
from podclipper.panels import SourcePanel, constrain_crop, detect_source_panels


def split_frame(line=255):
    frame = np.empty((180, 600, 3), dtype=np.uint8)
    frame[:, :300] = (70, 100, 140)
    frame[:, 300:] = (140, 80, 100)
    frame[:, 298:302] = line
    return frame


@pytest.mark.parametrize('line', [0, 255])
def test_dividers_excluded_from_bounds(line):
    panels = detect_source_panels(split_frame(line))
    assert panels == (SourcePanel(0, 296), SourcePanel(304, 600))


def test_disabled_or_missing_dividers_keep_whole_frame():
    assert detect_source_panels(split_frame(), SimpleNamespace(enabled=False)) == (SourcePanel(0, 600),)
    assert detect_source_panels(np.full((180, 600, 3), 120, dtype=np.uint8)) == (SourcePanel(0, 600),)


def test_excessive_margin_never_creates_empty_views():
    panels = detect_source_panels(split_frame(), SimpleNamespace(margin_pixels=1000))
    assert all(0 <= p.left < p.right <= 600 for p in panels)


@pytest.mark.parametrize('bbox', [(-100, -50, 600, 400), (500, 300, 300, 200), (0, 0, 900, 600)])
def test_crop_keeps_aspect_and_stays_inside_view(bbox):
    panel = SourcePanel(304, 600)
    x, y, w, h = constrain_crop(bbox, panel, 180, focus_y=80)
    assert panel.left <= x < x + w <= panel.right
    assert 0 <= y < y + h <= 180
    assert w / h == pytest.approx(bbox[2] / bbox[3], abs=.02)
