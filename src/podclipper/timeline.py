"""Timeline builders for the crop stage.

`classify_wide_shot_frames` is the active API used by `crop.mode: auto`.
Everything else (`build_speaker_timeline`, `apply_min_dwell`, the cluster
helpers) is DEPRECATED — only reached by `crop.mode: single`."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Optional

import numpy as np

from .types import BBox, Timeline, TimelineSegment

log = logging.getLogger("ave.timeline")


def _cluster_x_centers(
    bboxes: list[Optional[BBox]],
    merge_tolerance_px: float = 120.0,
) -> list[list[int]]:
    """Group frame indices by persistent bbox x-center (1D nearest-neighbor)."""
    clusters: list[list[int]] = []
    centers: list[float] = []

    for idx, b in enumerate(bboxes):
        if b is None:
            continue
        cx = b.x_center
        if not centers:
            centers.append(cx)
            clusters.append([idx])
            continue
        distances = [abs(cx - c) for c in centers]
        nearest = min(range(len(centers)), key=lambda i: distances[i])
        if distances[nearest] <= merge_tolerance_px:
            n = len(clusters[nearest])
            centers[nearest] = (centers[nearest] * n + cx) / (n + 1)
            clusters[nearest].append(idx)
        else:
            centers.append(cx)
            clusters.append([idx])

    return clusters


def _cluster_persistence(cluster: list[int], total_frames: int) -> float:
    if total_frames == 0:
        return 0.0
    return len(cluster) / total_frames


def _longest_contiguous_run(frames: list[int], gap_tolerance: int) -> int:
    """Frame-span of the longest run of detections in `frames`, treating
    gaps <= `gap_tolerance` as continuous. Spans (not counts) so a deliberate
    3.5s shot with 70% detection density still scores 3.5s."""
    if not frames:
        return 0
    sorted_frames = sorted(frames)
    best = 1
    run_start = sorted_frames[0]
    prev = sorted_frames[0]
    for f in sorted_frames[1:]:
        if f - prev > gap_tolerance:
            best = max(best, prev - run_start + 1)
            run_start = f
        prev = f
    best = max(best, prev - run_start + 1)
    return best


def _make_bbox_source(bboxes: list[Optional[BBox]], member_frames: set[int]):
    """Return-bbox-if-in-member-set closure."""
    def bbox_at(frame_idx: int) -> Optional[BBox]:
        if frame_idx in member_frames and 0 <= frame_idx < len(bboxes):
            return bboxes[frame_idx]
        return None
    return bbox_at


def _make_any_bbox_source(bboxes: list[Optional[BBox]]):
    """Pass-through closure — any detected bbox, no cluster filtering."""
    def bbox_at(frame_idx: int) -> Optional[BBox]:
        if 0 <= frame_idx < len(bboxes):
            return bboxes[frame_idx]
        return None
    return bbox_at


def _multi_shot_timeline(
    significant_clusters: list[list[int]],
    per_frame_bboxes: list[Optional[BBox]],
    clip_duration: float,
    fps: float,
) -> "Timeline":
    """Emit one timeline segment per active-cluster window — follows the editor's cuts."""
    total_frames = len(per_frame_bboxes)
    if total_frames == 0:
        return []

    frame_to_sig = [-1] * total_frames
    for sig_idx, frames in enumerate(significant_clusters):
        for f in frames:
            if 0 <= f < total_frames:
                frame_to_sig[f] = sig_idx

    # Fill gap frames by assigning to whichever significant cluster is nearer
    # in time. Pure carry-forward would leave entire gaps with the old shot
    # and delay the crop switch past the visual cut.
    sig_positions = [i for i in range(total_frames) if frame_to_sig[i] >= 0]
    if sig_positions:
        prev_sig = [-1] * total_frames
        last = -1
        for i in range(total_frames):
            if frame_to_sig[i] >= 0:
                last = i
            prev_sig[i] = last
        next_sig = [-1] * total_frames
        nxt = -1
        for i in range(total_frames - 1, -1, -1):
            if frame_to_sig[i] >= 0:
                nxt = i
            next_sig[i] = nxt
        for i in range(total_frames):
            if frame_to_sig[i] >= 0:
                continue
            p, n = prev_sig[i], next_sig[i]
            if p < 0:
                frame_to_sig[i] = frame_to_sig[n]
            elif n < 0:
                frame_to_sig[i] = frame_to_sig[p]
            else:
                frame_to_sig[i] = frame_to_sig[n] if (n - i) <= (i - p) else frame_to_sig[p]
    else:
        for i in range(total_frames):
            frame_to_sig[i] = 0

    cluster_frame_sets = [set(c) for c in significant_clusters]

    segments: list[TimelineSegment] = []
    start_f = 0
    cur = frame_to_sig[0]
    for i in range(1, total_frames + 1):
        if i == total_frames or frame_to_sig[i] != cur:
            start_t = start_f / fps
            end_t = clip_duration if i == total_frames else i / fps
            segments.append(TimelineSegment(
                start=start_t,
                end=end_t,
                label=f"SHOT_{cur}",
                bbox_at=_make_bbox_source(per_frame_bboxes, cluster_frame_sets[cur]),
            ))
            if i < total_frames:
                start_f = i
                cur = frame_to_sig[i]

    return segments


def _make_center_source(center_x: float, center_y: float, w: float, h: float):
    """Synthetic bbox anchored at frame center — used when no persons are detected."""
    static = BBox(
        x=center_x - w / 2,
        y=center_y - h / 2,
        w=w, h=h,
        confidence=0.0,
    )
    def bbox_at(frame_idx: int) -> Optional[BBox]:
        return static
    return bbox_at


def build_speaker_timeline(
    per_frame_bboxes: list[Optional[BBox]],
    clip_duration: float,
    fps: float,
    source_width: int,
    source_height: int,
    diar_segments: Optional[list] = None,
    cfg: SimpleNamespace | None = None,
    video_path: "Optional[object]" = None,
) -> Timeline:
    """DEPRECATED — legacy `crop.mode: single` only. Turns per-frame YOLO
    output (+ optional diarization) into a Timeline."""
    total_frames = len(per_frame_bboxes)
    non_null = sum(1 for b in per_frame_bboxes if b is not None)

    if non_null == 0:
        log.info("No persons detected in clip — using center-frame fallback")
        return [TimelineSegment(
            start=0.0,
            end=clip_duration,
            label="CENTER",
            bbox_at=_make_center_source(
                source_width / 2, source_height / 2,
                source_height * 9 / 16,
                source_height * 0.9,
            ),
        )]

    clusters = _cluster_x_centers(per_frame_bboxes)
    log.info(f"Found {len(clusters)} persistent bbox cluster(s)")

    # "Real" cluster = longest contiguous run >= 1s. The naive 10%-of-total
    # rule wrongly dropped legitimate ~3.5s cutaway shots.
    min_run_seconds = 1.0
    min_run_frames = max(1, int(min_run_seconds * fps)) if fps > 0 else 1
    gap_tolerance = max(2, int(0.25 * fps)) if fps > 0 else 2
    significant = [
        c for c in clusters
        if _longest_contiguous_run(c, gap_tolerance) >= min_run_frames
    ]
    if not significant:
        log.debug("No persistent cluster; using all detections as single group")
        return [TimelineSegment(
            start=0.0,
            end=clip_duration,
            label="PRIMARY",
            bbox_at=_make_any_bbox_source(per_frame_bboxes),
        )]

    if diar_segments is not None and len(significant) >= 2 and video_path is not None:
        from .diarize import link_timeline
        try:
            return link_timeline(
                diar_segments=diar_segments,
                bbox_clusters=significant,
                per_frame_bboxes=per_frame_bboxes,
                fps=fps,
                clip_duration=clip_duration,
                video_path=video_path,
                cfg=cfg,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(f"link_timeline failed ({e}); falling back to multi-shot timeline")

    if len(significant) >= 2:
        log.info(
            f"Multi-shot clip: {len(significant)} camera angles; "
            "emitting per-shot timeline segments"
        )
        return _multi_shot_timeline(significant, per_frame_bboxes, clip_duration, fps)

    primary_cluster = significant[0]
    persistence = _cluster_persistence(primary_cluster, total_frames)
    log.info(
        f"Using primary cluster: {len(primary_cluster)}/{total_frames} frames "
        f"({persistence:.0%})"
    )
    member_set = set(primary_cluster)

    return [TimelineSegment(
        start=0.0,
        end=clip_duration,
        label="PRIMARY",
        bbox_at=_make_bbox_source(per_frame_bboxes, member_set),
    )]


def apply_min_dwell(timeline: Timeline, min_dwell_seconds: float) -> Timeline:
    """DEPRECATED — legacy `crop.mode: single` only. Absorbs short segments
    into the previous one."""
    if len(timeline) <= 1:
        return timeline
    out: list[TimelineSegment] = []
    for seg in timeline:
        if (seg.end - seg.start) < min_dwell_seconds and out:
            prev = out[-1]
            out[-1] = TimelineSegment(
                start=prev.start,
                end=seg.end,
                label=prev.label,
                bbox_at=prev.bbox_at,
            )
        else:
            out.append(seg)
    return out


def _min_dwell_debounce(flags: "np.ndarray", min_frames: int) -> "np.ndarray":
    """Only switch layout state after the new state has held for `min_frames`
    continuously — so brief two-shot cuts don't flip to stacked."""
    if min_frames <= 1 or len(flags) == 0:
        return flags
    out = flags.copy()
    stable = bool(flags[0])
    pending_count = 0
    for i in range(len(flags)):
        if bool(flags[i]) == stable:
            pending_count = 0
        else:
            pending_count += 1
            if pending_count >= min_frames:
                stable = bool(flags[i])
                pending_count = 0
        out[i] = stable
    return out


def classify_wide_shot_frames(
    per_frame_persons: list[list[BBox]],
    source_width: int,
    source_height: int,
    sep_threshold_frac: float = 0.20,
    height_cap_frac: float = 1.0,
    smooth_window_frames: int = 15,
    min_dwell_frames: int = 0,
    min_person_frac: float = 0.20,
) -> "np.ndarray":
    """Per-frame wide-shot flag (two real people on screen) from YOLO bbox
    geometry. Raw wide = ≥2 people that are foreground (taller than
    `min_person_frac` of the frame — ignores tiny background people/posters)
    AND separated by ≥`sep_threshold_frac` of source width. `height_cap_frac`
    is an optional upper bound (default 1.0 = off). Smoothed by majority over
    `smooth_window_frames`, then a min-dwell debounce."""
    n = len(per_frame_persons)
    if n == 0:
        return np.zeros(0, dtype=bool)

    sep_thresh = sep_threshold_frac * source_width
    height_cap = height_cap_frac * source_height
    height_floor = min_person_frac * source_height

    raw = np.zeros(n, dtype=bool)
    for i, persons in enumerate(per_frame_persons):
        qual = [p.x_center for p in persons if height_floor <= p.h <= height_cap]
        if len(qual) >= 2 and (max(qual) - min(qual)) >= sep_thresh:
            raw[i] = True

    win = max(1, int(smooth_window_frames))
    half = win // 2
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = raw[lo:hi].mean() >= 0.5
    return _min_dwell_debounce(out, int(min_dwell_frames))
