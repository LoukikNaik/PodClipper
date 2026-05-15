"""Stage 5b: Speaker timeline builder.

A speaker timeline is `list[TimelineSegment]` where each segment answers:
  "For this time range, which bbox should the 9:16 crop window target?"

MVP path: cluster per-frame bboxes to find one or more persistent positions,
then return a single-entry timeline pointing at the most-persistent one.

Post-MVP path (already wired through): when `diar_segments` is supplied by
`diarize.py`, emit one timeline entry per diarization segment with its bbox
chosen via mouth-motion face linking.

The crop stage is identical for both paths — it consumes the timeline with no
knowledge of how it was built.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Optional

import numpy as np

from .types import BBox, Timeline, TimelineSegment

log = logging.getLogger("ave.timeline")


# ---------- Bbox clustering ----------

def _cluster_x_centers(
    bboxes: list[Optional[BBox]],
    merge_tolerance_px: float = 120.0,
) -> list[list[int]]:
    """Group frame indices by persistent bbox x-center.

    Simple 1D clustering: iterate frames, attach each detection to the nearest
    existing cluster within `merge_tolerance_px`; otherwise start a new cluster.
    Returns a list of clusters; each cluster is a list of frame indices where
    that position was observed. Frames with None are ignored.
    """
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
        # Nearest existing cluster
        distances = [abs(cx - c) for c in centers]
        nearest = min(range(len(centers)), key=lambda i: distances[i])
        if distances[nearest] <= merge_tolerance_px:
            # Update running mean
            n = len(clusters[nearest])
            centers[nearest] = (centers[nearest] * n + cx) / (n + 1)
            clusters[nearest].append(idx)
        else:
            centers.append(cx)
            clusters.append([idx])

    return clusters


def _cluster_persistence(cluster: list[int], total_frames: int) -> float:
    """Fraction of frames this cluster is present in — 0.0 to 1.0."""
    if total_frames == 0:
        return 0.0
    return len(cluster) / total_frames


def _longest_contiguous_run(frames: list[int], gap_tolerance: int) -> int:
    """Return the time-span (in frames) of the longest contiguous run of
    detections in `frames`, where consecutive entries are considered part of
    the same run if they're within `gap_tolerance` frames of each other.

    A run's "span" is (last - first + 1), not the count of members. This way
    a deliberately framed shot of 3.5 s that only has ~70% detection density
    still gets credit for being on screen 3.5 s. Used to distinguish real
    shots (long contiguous screen time) from scattered false positives.
    """
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


# ---------- bbox_at callables ----------

def _make_bbox_source(bboxes: list[Optional[BBox]], member_frames: set[int]):
    """Build a callable that returns the bbox for a given frame, but only if
    the frame is in `member_frames`. Used to filter per-frame bboxes down to
    the ones belonging to a specific persistent position.
    """
    def bbox_at(frame_idx: int) -> Optional[BBox]:
        if frame_idx in member_frames and 0 <= frame_idx < len(bboxes):
            return bboxes[frame_idx]
        return None
    return bbox_at


def _make_any_bbox_source(bboxes: list[Optional[BBox]]):
    """Simple pass-through — any detected bbox, no cluster filtering.
    Used when only one persistent position exists and cluster filtering adds no value.
    """
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
    """Build a timeline that follows the active cluster per time window.

    When a source has multiple camera angles (common in podcast edits),
    clusters represent SHOTS. Each cluster has one or more contiguous time
    windows where it was active. We emit one timeline segment per window,
    targeting whichever cluster's bbox is active at that time. The editor
    already cut cameras at these frame boundaries — we just follow them.

    Frames in insignificant clusters or with no detection carry forward the
    last-known significant cluster so we never emit gap segments.
    """
    total_frames = len(per_frame_bboxes)
    if total_frames == 0:
        return []

    # frame_idx → which significant cluster (as index into `significant_clusters`)
    frame_to_sig = [-1] * total_frames
    for sig_idx, frames in enumerate(significant_clusters):
        for f in frames:
            if 0 <= f < total_frames:
                frame_to_sig[f] = sig_idx

    # Fill gaps by assigning each gap frame to whichever significant cluster
    # is nearer in time. This handles shot cuts correctly: a gap caused by
    # motion-blurred transition frames gets split — frames near the old shot
    # stay with the old cluster, frames near the new shot move to the new
    # one. Pure carry-forward would leave the entire gap with the old shot,
    # delaying the crop's segment switch past the visual cut.
    sig_positions = [i for i in range(total_frames) if frame_to_sig[i] >= 0]
    if sig_positions:
        # For each frame, find the nearest sig_position via two-pointer sweep.
        # Build prev_sig (last sig at or before i) and next_sig (first sig at or after i).
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

    # Pre-compute the frame-set lookup each cluster's bbox_at will need.
    cluster_frame_sets = [set(c) for c in significant_clusters]

    # Collapse contiguous runs into timeline segments.
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
    """Fallback: a synthetic 'bbox' anchored at frame center (used when no
    persons are ever detected)."""
    static = BBox(
        x=center_x - w / 2,
        y=center_y - h / 2,
        w=w, h=h,
        confidence=0.0,
    )
    def bbox_at(frame_idx: int) -> Optional[BBox]:
        return static
    return bbox_at


# ---------- Main API ----------

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
    """Turn per-frame YOLO output (+ optional diarization) into a Timeline.

    MVP behavior (diar_segments is None, which is the only path enabled today):
      1. If no person was ever detected → single-entry timeline pointing at frame center
      2. If one persistent position → single-entry timeline filtered to that position
      3. If multiple persistent positions → single-entry timeline on the most persistent
         (largest cluster by frame count). Full multi-speaker switching requires diar_segments.

    Post-MVP: when diar_segments is provided, build an N-entry timeline by
    linking diarized speakers to persistent bboxes via mouth-motion correlation.
    """
    total_frames = len(per_frame_bboxes)
    non_null = sum(1 for b in per_frame_bboxes if b is not None)

    # --- Case: no people detected at all ---
    if non_null == 0:
        log.info("No persons detected in clip — using center-frame fallback")
        return [TimelineSegment(
            start=0.0,
            end=clip_duration,
            label="CENTER",
            bbox_at=_make_center_source(
                source_width / 2, source_height / 2,
                source_height * 9 / 16,  # approximate speaker-bbox width for 9:16 crop
                source_height * 0.9,
            ),
        )]

    # --- Cluster persistent positions ---
    clusters = _cluster_x_centers(per_frame_bboxes)
    log.info(f"Found {len(clusters)} persistent bbox cluster(s)")

    # Drop transient clusters. A cluster qualifies as "real" if its longest
    # contiguous on-screen run is at least ~1s — that's a deliberate shot,
    # not a stray false positive. Counting raw frames (the old 10%-of-total
    # rule) wrongly dropped legitimate ~3.5s cutaway shots whenever the main
    # cluster dominated the rest of the clip.
    min_run_seconds = 1.0
    min_run_frames = max(1, int(min_run_seconds * fps)) if fps > 0 else 1
    gap_tolerance = max(2, int(0.25 * fps)) if fps > 0 else 2
    significant = [
        c for c in clusters
        if _longest_contiguous_run(c, gap_tolerance) >= min_run_frames
    ]
    if not significant:
        # Nothing persistent — treat all detections as "one speaker"
        log.debug("No persistent cluster; using all detections as single group")
        return [TimelineSegment(
            start=0.0,
            end=clip_duration,
            label="PRIMARY",
            bbox_at=_make_any_bbox_source(per_frame_bboxes),
        )]

    # --- Multi-speaker path: diarization-driven timeline with mouth-motion linking ---
    if diar_segments is not None and len(significant) >= 2 and video_path is not None:
        from .diarize import link_timeline  # lazy import — optional dep
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
        except Exception as e:  # noqa: BLE001 — fall back to multi-shot on any failure
            log.warning(f"link_timeline failed ({e}); falling back to multi-shot timeline")

    # --- Multi-shot path (MVP): multiple camera angles detected. Produce one
    # timeline segment per shot rather than picking one cluster globally. ---
    if len(significant) >= 2:
        log.info(
            f"Multi-shot clip: {len(significant)} camera angles; "
            "emitting per-shot timeline segments"
        )
        return _multi_shot_timeline(significant, per_frame_bboxes, clip_duration, fps)

    # --- Single-shot path: one persistent cluster, one timeline segment ---
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
    """Merge or drop segments shorter than `min_dwell_seconds`.

    Short segments get absorbed into whichever neighbor they're more similar to
    (here: just the previous segment). Single-segment timelines pass through.
    """
    if len(timeline) <= 1:
        return timeline
    out: list[TimelineSegment] = []
    for seg in timeline:
        if (seg.end - seg.start) < min_dwell_seconds and out:
            # Extend previous segment to absorb this one
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


# ---------- Wide-shot classifier (for the shot-aware stacked crop) -------

def classify_wide_shot_frames(
    per_frame_persons: list[list[BBox]],
    source_width: int,
    source_height: int,
    sep_threshold_frac: float = 0.20,
    height_cap_frac: float = 0.70,
    smooth_window_frames: int = 15,
) -> "np.ndarray":
    """Decide, for each frame, whether it's a wide shot (two visible people)
    or a single-person/close-up shot, based on YOLO bbox geometry alone.

    A frame qualifies as "raw wide" if:
      * ≥ 2 person bboxes each shorter than `height_cap_frac * source_height`,
        which excludes the typical single-person close-up where one bbox
        fills most of the frame, and
      * the leftmost and rightmost qualifying bboxes are separated by at
        least `sep_threshold_frac * source_width`, ruling out two
        near-overlapping detections of the same person.

    A centered window of `smooth_window_frames` frames is then applied:
    `is_wide[t]` is True iff at least half of the raw decisions inside that
    window are True. This eliminates flicker when YOLO momentarily misses
    one person during a wide shot, or briefly double-detects in a close-up.

    Returns a bool ndarray of length `len(per_frame_persons)`.
    """
    n = len(per_frame_persons)
    if n == 0:
        return np.zeros(0, dtype=bool)

    sep_thresh = sep_threshold_frac * source_width
    height_cap = height_cap_frac * source_height

    raw = np.zeros(n, dtype=bool)
    for i, persons in enumerate(per_frame_persons):
        qual = [p.x_center for p in persons if p.h < height_cap]
        if len(qual) >= 2 and (max(qual) - min(qual)) >= sep_thresh:
            raw[i] = True

    win = max(1, int(smooth_window_frames))
    half = win // 2
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = raw[lo:hi].mean() >= 0.5
    return out
