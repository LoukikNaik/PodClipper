"""Find graphic dividers and keep crops inside individual source panels."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


@dataclass(frozen=True)
class SourcePanel:
    """Horizontal source bounds, excluding the divider itself (right exclusive)."""

    left: int
    right: int

    @property
    def width(self) -> int:
        return self.right - self.left


def detect_source_panels(frame: np.ndarray, cfg: SimpleNamespace | None = None) -> tuple[SourcePanel, ...]:
    """Recognize thin, full-height light/dark lines with contrast on both sides.

    Sample rows at original horizontal resolution so even a one-pixel divider
    survives. Tall scene edges alone are insufficient: the stripe must have a
    nearly solid neutral color, be narrow, and contrast with both adjacent views.
    No positions are carried across frames, so a camera cut removes old bounds.
    """
    height, width = frame.shape[:2]
    whole = (SourcePanel(0, width),)
    if not getattr(cfg, "enabled", True) or height < 16 or width < 32:
        return whole
    coverage = float(getattr(cfg, "min_coverage", 0.90))
    max_width = max(1, int(width * float(getattr(cfg, "max_width_frac", 0.008))))
    minimum_panel = max(2, int(width * float(getattr(cfg, "min_panel_frac", 0.15))))
    contrast = float(getattr(cfg, "min_contrast", 25.0))
    margin = max(1, int(getattr(cfg, "margin_pixels", 2)))
    rows = np.linspace(int(height * 0.04), int(height * 0.96) - 1, min(height, 96)).astype(int)
    pixels = frame[rows].astype(np.int16)
    neutral = pixels.max(axis=2) - pixels.min(axis=2) < 38
    light = (pixels.min(axis=2) >= 190) & neutral
    dark = (pixels.max(axis=2) <= 40) & neutral
    # A dark background is common; require very little variation for dark lines.
    dark_columns = (dark.mean(axis=0) >= coverage) & (pixels.std(axis=0).max(axis=1) < 10)
    mask = (light.mean(axis=0) >= coverage) | dark_columns
    columns = np.flatnonzero(mask)
    if not len(columns):
        return whole
    groups = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
    candidates = []
    for group in groups:
        left, right = int(group[0]), int(group[-1]) + 1
        if right - left > max_width or left < minimum_panel or width - right < minimum_panel:
            continue
        offset = max(3, width // 300)
        stripe = pixels[:, left:right].mean(axis=1)
        before = pixels[:, max(0, left - offset)]
        after = pixels[:, min(width - 1, right + offset)]
        differences = np.minimum(np.abs(stripe - before).mean(axis=1),
                                 np.abs(stripe - after).mean(axis=1))
        if (differences >= contrast).mean() < coverage * 0.8:
            continue
        candidates.append((float(np.median(differences)), left, right))
    accepted = []
    for score, left, right in sorted(candidates, reverse=True):
        center = (left + right) / 2
        if all(abs(center - (a + b) / 2) >= minimum_panel for a, b in accepted):
            accepted.append((left, right))
    if not accepted:
        return whole
    panels, start = [], 0
    for left, right in sorted(accepted):
        # User margins must not turn narrow views into empty/negative crops.
        # Ignore a divider whose exclusion zone cannot fit in the source.
        if left - margin <= start or right + margin >= width:
            continue
        panels.append(SourcePanel(start, left - margin))
        start = right + margin
    panels.append(SourcePanel(start, width))
    return tuple(panels)


def panel_for_x(panels: tuple[SourcePanel, ...], x: float) -> SourcePanel:
    """Choose the view containing a subject, including subjects near a divider."""
    return min(panels, key=lambda panel: max(panel.left - x, 0, x - (panel.right - 1)))


def constrain_crop(bbox: tuple[int, int, int, int], panel: SourcePanel,
                   source_height: int, focus_y: float | None = None) -> tuple[int, int, int, int]:
    """Fit a crop inside its view without stretching or crossing a divider.

    Reduce width and height together and keep a supplied face anchor in view.
    """
    x, y, width, height = bbox
    scale = min(1.0, panel.width / max(1, width), source_height / max(1, height))
    crop_w = max(1, min(panel.width, int(round(width * scale))))
    crop_h = max(1, min(source_height, int(round(height * scale))))
    x = max(panel.left, min(panel.right - crop_w, int(round(x + (width - crop_w) / 2))))
    if focus_y is not None and crop_h < height:
        y = int(round(focus_y - crop_h * 0.5))
    y = max(0, min(source_height - crop_h, y))
    return x, y, crop_w, crop_h

