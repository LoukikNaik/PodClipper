"""Stage 5a: Person detection.

Runs YOLO v8 nano on (every Nth) frame of an extracted clip and returns a
per-frame list of `BBox | None`. A simple greedy IoU matcher tracks the
"primary" subject across frames — when multiple persons are visible, the
largest bbox on frame 0 is anchored and subsequent frames pick the detection
with highest IoU against the previous anchor (falling back to largest).

For frames we skipped (sample_every_n_frames > 1), we carry the last-known
bbox forward so the output list is dense with one entry per source frame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2

from .types import BBox

log = logging.getLogger("ave.detect")

_model_cache: dict[tuple, object] = {}


class DetectError(Exception):
    pass


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _get_yolo(model_path: str, device: str):
    from ultralytics import YOLO
    resolved = _resolve_device(device)
    key = (model_path, resolved)
    if key not in _model_cache:
        log.info(f"Loading YOLO model {model_path} on {resolved}")
        model = YOLO(model_path)
        _model_cache[key] = (model, resolved)
    return _model_cache[key]


def _detections_to_bboxes(result, person_class_id: int, conf_threshold: float) -> list[BBox]:
    """Pull person bboxes out of a single ultralytics Result."""
    bboxes: list[BBox] = []
    if result.boxes is None:
        return bboxes
    for box in result.boxes:
        cls_id = int(box.cls.item())
        if cls_id != person_class_id:
            continue
        conf = float(box.conf.item())
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bboxes.append(BBox(
            x=float(x1),
            y=float(y1),
            w=float(x2 - x1),
            h=float(y2 - y1),
            confidence=conf,
        ))
    return bboxes


def _pick_primary(
    candidates: list[BBox],
    anchor: Optional[BBox],
) -> Optional[BBox]:
    """Choose the bbox most likely to be 'the same person' as `anchor`."""
    if not candidates:
        return None
    if anchor is None:
        # No prior — pick the largest bbox (usually the main subject / closest to camera)
        return max(candidates, key=lambda b: b.area)
    # Pick highest IoU with anchor; if no overlap at all, fall back to largest
    best = max(candidates, key=lambda b: anchor.iou(b))
    if anchor.iou(best) < 1e-4:
        return max(candidates, key=lambda b: b.area)
    return best


def detect_humans_per_frame(
    video_path: Path,
    cfg: SimpleNamespace,
) -> tuple[list[Optional[BBox]], float, int, int]:
    """Run per-frame person detection across the video.

    Returns:
        (bboxes, fps, width, height) — `bboxes[i]` is the primary-subject BBox
        for source frame i (or None if no person was found). Length equals
        the source frame count.
    """
    video_path = Path(video_path)
    det_cfg = cfg.detect

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise DetectError(f"OpenCV could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model, device = _get_yolo(det_cfg.model, det_cfg.device)
    every_n = max(1, int(det_cfg.sample_every_n_frames))

    log.info(
        f"Detecting persons in {video_path.name}: "
        f"{total_frames} frames @ {fps:.2f}fps, sampling every {every_n}"
    )

    out: list[Optional[BBox]] = []
    anchor: Optional[BBox] = None
    frame_idx = 0
    last_bbox: Optional[BBox] = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % every_n == 0:
            # ultralytics wants BGR numpy arrays, which OpenCV already gives us
            results = model.predict(
                frame,
                conf=det_cfg.confidence,
                iou=det_cfg.iou,
                device=device,
                verbose=False,
            )
            if results:
                bboxes = _detections_to_bboxes(
                    results[0],
                    person_class_id=det_cfg.person_class_id,
                    conf_threshold=det_cfg.confidence,
                )
                primary = _pick_primary(bboxes, anchor)
                if primary is not None:
                    anchor = primary
                    last_bbox = primary
                out.append(primary)
            else:
                out.append(None)
        else:
            # Skipped frame — carry forward the last known detection.
            # Crop stage still has a bbox per frame, and EMA smoothing handles
            # any staleness.
            out.append(last_bbox)
        frame_idx += 1

    cap.release()

    non_null = sum(1 for b in out if b is not None)
    log.info(
        f"Detection complete: {non_null}/{len(out)} frames had a person "
        f"({100 * non_null / max(1, len(out)):.1f}%)"
    )
    return out, fps, width, height
