#!/usr/bin/env python3
"""Diagnose why a specific frame's crop is wrong: print YOLO bbox, cluster
assignment, and timeline segment for a window around a target frame."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from podclipper.config import load_config
from podclipper.detect import detect_humans_per_frame
from podclipper.logging_util import setup_logging
from podclipper.timeline import _cluster_x_centers, build_speaker_timeline, apply_min_dwell


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python diag_frame.py SEGMENT.mp4 TARGET_FRAME [WINDOW=30]")
        return 2
    segment = Path(sys.argv[1])
    target = int(sys.argv[2])
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    cfg = load_config(Path("config/default.yaml"))
    setup_logging("INFO")

    bboxes, fps, w, h = detect_humans_per_frame(segment, cfg)
    print(f"\nTotal frames: {len(bboxes)} @ {fps:.2f}fps, frame size {w}x{h}")
    print(f"Target frame {target} ≈ t={target/fps:.2f}s\n")

    # Clusters
    clusters = _cluster_x_centers(bboxes)
    centers = []
    for c in clusters:
        xs = [bboxes[i].x_center for i in c if bboxes[i] is not None]
        centers.append((sum(xs)/len(xs) if xs else 0.0, len(c)))
    print(f"Clusters: {len(clusters)}")
    for i, (cx, n) in enumerate(centers):
        print(f"  cluster {i}: center_x={cx:.0f}  frames={n}  "
              f"first_frame={clusters[i][0] if clusters[i] else '-'}  "
              f"last_frame={clusters[i][-1] if clusters[i] else '-'}")

    duration = len(bboxes) / fps if fps else 0.0
    timeline = build_speaker_timeline(
        per_frame_bboxes=bboxes, clip_duration=duration, fps=fps,
        source_width=w, source_height=h, diar_segments=None,
        cfg=cfg, video_path=segment,
    )
    timeline = apply_min_dwell(timeline, cfg.crop.min_segment_dwell_seconds)
    print(f"\nTimeline ({len(timeline)} segments):")
    for s in timeline:
        print(f"  [{s.start:6.2f}-{s.end:6.2f}s] {s.label}")

    print(f"\nFrames {target-window}..{target+window}:")
    print(f"  frame   yolo_x_center   in_seg(label)   seg.bbox_at_returns")
    for f in range(max(0, target - window), min(len(bboxes), target + window + 1)):
        b = bboxes[f]
        bx = f"{b.x_center:6.0f}" if b is not None else "  None"
        seg_for_f = next((s for s in timeline if s.start <= f/fps < s.end), timeline[-1] if timeline else None)
        sb = seg_for_f.bbox_at(f) if seg_for_f else None
        sbx = f"{sb.x_center:6.0f}" if sb is not None else "  None"
        print(f"  {f:5d}   {bx}          {seg_for_f.label if seg_for_f else '-':10s}    {sbx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
