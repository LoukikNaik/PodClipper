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

    # Drop clusters that are too transient (< 10% of frames) — usually false positives
    min_cluster_frames = max(1, int(0.1 * total_frames))
    significant = [c for c in clusters if len(c) >= min_cluster_frames]
    if not significant:
        # Nothing persistent — treat all detections as "one speaker"
        log.debug("No persistent cluster; using all detections as single group")
        return [TimelineSegment(
            start=0.0,
            end=clip_duration,
            label="PRIMARY",
            bbox_at=_make_any_bbox_source(per_frame_bboxes),
        )]

    # --- Multi-speaker path (post-MVP) ---
    if diar_segments is not None and len(significant) >= 2:
        # Delegate to diarize module for full multi-segment timeline construction
        from .diarize import link_timeline  # lazy import, optional dep
        return link_timeline(
            diar_segments=diar_segments,
            bbox_clusters=significant,
            per_frame_bboxes=per_frame_bboxes,
            fps=fps,
            clip_duration=clip_duration,
        )

    # --- MVP: pick the most persistent cluster, single-entry timeline ---
    primary_cluster = max(significant, key=len)
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
