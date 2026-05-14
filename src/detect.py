"""Stage 5a: Person detection.

Runs YOLO v8 nano on (every Nth) frame of an extracted clip and returns a
per-frame list of `BBox | None`. A simple greedy IoU matcher tracks the
"primary" subject across frames — when multiple persons are visible, the
largest bbox on frame 0 is anchored and subsequent frames pick the detection
with highest IoU against the previous anchor (falling back to largest).

Front-face preference: when face_aware=True, face detection is run on ALL
candidate persons per frame (not just the YOLO winner). Candidates with a
visible face (front-facing) are always preferred over back-of-head detections.

For frames we skipped (sample_every_n_frames > 1), we carry the last-known
bbox forward so the output list is dense with one entry per source frame.

Debug overlay (cfg.detect.debug_overlay=True or --debug-detect):
  Writes <clip_cache>/debug_detect.mp4 with per-frame annotations:
    - Green border  = front-facing person (face detected)
    - Orange border = back-of-head (no face found)
    - Thick border  = selected primary subject
    - Cyan rect     = face bbox within primary
"""

from __future__ import annotations

import logging
import threading
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

from .types import BBox

log = logging.getLogger("ave.detect")

_model_cache: dict[tuple, object] = {}
_face_detector = None
_face_detector_lock = threading.Lock()

# MediaPipe short-range face detector — designed for selfie-distance faces
# (~2m), which fits podcast framing. ~250KB tflite.
_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


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


def _ensure_face_model(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "blaze_face_short_range.tflite"
    if not target.exists():
        log.info(f"Downloading MediaPipe face detector → {target}")
        urllib.request.urlretrieve(_FACE_MODEL_URL, target)
    return target


def _get_face_detector(cfg: SimpleNamespace):
    """Load MediaPipe FaceDetector once per process. Returns None if unavailable."""
    global _face_detector
    with _face_detector_lock:
        if _face_detector is not None:
            return _face_detector
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            model_path = _ensure_face_model(Path(cfg.paths.cache_dir))
            options = mp_vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                min_detection_confidence=0.4,
                running_mode=mp_vision.RunningMode.IMAGE,
            )
            _face_detector = mp_vision.FaceDetector.create_from_options(options)
        except Exception as e:  # noqa: BLE001
            log.warning(f"face detector unavailable ({e}); falling back to body bbox")
            _face_detector = False  # sentinel — don't retry
    return _face_detector if _face_detector else None


def _face_bbox_inside(frame_bgr, person: BBox, detector) -> Optional[BBox]:
    """Run face detection on the region around `person` and return the face
    bbox whose center falls inside `person`, or None if no suitable face found.

    Coordinates are converted back to the source frame (not the crop window).
    """
    import mediapipe as mp

    H, W = frame_bgr.shape[:2]
    # Pad the crop so a face near the bbox edge isn't missed
    pad_x = int(0.15 * person.w)
    pad_y = int(0.15 * person.h)
    x0 = max(0, int(person.x) - pad_x)
    y0 = max(0, int(person.y) - pad_y)
    x1 = min(W, int(person.x + person.w) + pad_x)
    y1 = min(H, int(person.y + person.h) + pad_y)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = frame_bgr[y0:y1, x0:x1]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        result = detector.detect(mp_image)
    except Exception:  # noqa: BLE001
        return None
    if not result.detections:
        return None

    # Pick the face whose center lies inside the person bbox (there may be
    # multiple face detections if two people are near each other).
    best: Optional[BBox] = None
    best_area = 0.0
    for det in result.detections:
        bb = det.bounding_box
        fx = x0 + bb.origin_x
        fy = y0 + bb.origin_y
        fw = bb.width
        fh = bb.height
        face_cx = fx + fw / 2
        face_cy = fy + fh / 2
        # Require the face center to lie within the person bbox
        if not (person.x <= face_cx <= person.x + person.w and
                person.y <= face_cy <= person.y + person.h):
            continue
        if fw * fh > best_area:
            best_area = fw * fh
            best = BBox(x=fx, y=fy, w=fw, h=fh, confidence=person.confidence)
    return best


def _pick_primary(
    candidates: list[BBox],
    anchor: Optional[BBox],
) -> Optional[BBox]:
    """Choose the bbox most likely to be 'the same person' as `anchor`."""
    if not candidates:
        return None
    if anchor is None:
        return max(candidates, key=lambda b: b.area)
    best = max(candidates, key=lambda b: anchor.iou(b))
    if anchor.iou(best) < 1e-4:
        return max(candidates, key=lambda b: b.area)
    return best


def _run_face_on_all(
    frame_bgr: np.ndarray,
    candidates: list[BBox],
    detector,
) -> list[tuple[BBox, Optional[BBox]]]:
    """Run face detection on every candidate person bbox.

    Returns a list of (person_bbox, face_bbox_or_None).
    A non-None face_bbox means the person is front-facing.
    """
    return [(person, _face_bbox_inside(frame_bgr, person, detector))
            for person in candidates]


def _pick_primary_face_aware(
    tagged: list[tuple[BBox, Optional[BBox]]],
    anchor: Optional[BBox],
) -> tuple[Optional[BBox], Optional[BBox]]:
    """Like _pick_primary but excludes back-of-head candidates when a
    front-facing person is available.

    Returns (person_bbox, face_bbox_or_None).
    """
    if not tagged:
        return None, None

    front = [(p, f) for p, f in tagged if f is not None]
    pool = front if front else tagged  # fall back to back-facing if all are back

    persons = [p for p, _ in pool]
    primary = _pick_primary(persons, anchor)
    if primary is None:
        return None, None

    for p, f in pool:
        if p is primary:
            return primary, f
    return primary, None  # fallback, shouldn't reach


def _draw_detection_debug(
    frame: np.ndarray,
    all_tagged: list[tuple[BBox, Optional[BBox]]],
    primary: Optional[BBox],
    primary_face: Optional[BBox],
    frame_idx: int,
    fps: float,
) -> np.ndarray:
    """Render colored person/face bboxes onto a copy of `frame` for debugging.

    Color convention:
      - Green border  = front-facing (face detector found a face)
      - Orange border = back-of-head (no face detected)
      - Thick (4px)   = selected primary subject
      - Cyan rect     = face bbox within the primary
    """
    annotated = frame.copy()

    def _is_primary(p: BBox) -> bool:
        if primary is None:
            return False
        return abs(p.x - primary.x) < 2 and abs(p.y - primary.y) < 2

    for person, face in all_tagged:
        front = face is not None
        is_prim = _is_primary(person)
        color = (0, 200, 0) if front else (0, 140, 255)  # green vs orange (BGR)
        thickness = 4 if is_prim else 2
        x, y = int(person.x), int(person.y)
        w, h = int(person.w), int(person.h)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        label = ("PRIMARY:" if is_prim else "") + ("FRONT" if front else "BACK")
        cv2.putText(annotated, label, (x, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Face bbox in cyan for the primary
    if primary_face is not None:
        fx, fy = int(primary_face.x), int(primary_face.y)
        fw, fh = int(primary_face.w), int(primary_face.h)
        cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (255, 220, 0), 2)
        cv2.putText(annotated, "face", (fx, max(12, fy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1, cv2.LINE_AA)

    # Timestamp HUD
    t = frame_idx / fps if fps > 0 else 0.0
    cv2.putText(annotated, f"f{frame_idx}  t={t:.2f}s", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


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
    face_detector = _get_face_detector(cfg) if getattr(det_cfg, "face_aware", True) else None
    every_n = max(1, int(det_cfg.sample_every_n_frames))

    debug_overlay = getattr(det_cfg, "debug_overlay", False)
    log.info(
        f"Detecting persons in {video_path.name}: "
        f"{total_frames} frames @ {fps:.2f}fps, sampling every {every_n}"
        + (", face-aware crop enabled" if face_detector else "")
        + (", debug overlay ON" if debug_overlay else "")
    )

    # Initialise debug video writer if requested.
    debug_writer = None
    if debug_overlay:
        debug_path = video_path.parent / "debug_detect.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        debug_writer = cv2.VideoWriter(str(debug_path), fourcc, fps, (width, height))
        if not debug_writer.isOpened():
            log.warning("Could not open debug VideoWriter; disabling detect overlay")
            debug_writer = None
        else:
            log.info(f"Debug detect overlay → {debug_path}")

    out: list[Optional[BBox]] = []
    anchor: Optional[BBox] = None
    frame_idx = 0
    last_bbox: Optional[BBox] = None
    last_tagged: list[tuple[BBox, Optional[BBox]]] = []
    last_face: Optional[BBox] = None
    face_hits = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % every_n == 0:
            results = model.predict(
                frame,
                conf=det_cfg.confidence,
                iou=det_cfg.iou,
                device=device,
                verbose=False,
            )

            primary: Optional[BBox] = None
            face_bbox: Optional[BBox] = None
            all_tagged: list[tuple[BBox, Optional[BBox]]] = []

            if results:
                bboxes = _detections_to_bboxes(
                    results[0],
                    person_class_id=det_cfg.person_class_id,
                    conf_threshold=det_cfg.confidence,
                )
                if face_detector is not None and bboxes:
                    # Run face detection on ALL candidates so we can prefer
                    # front-facing persons over back-of-head ones.
                    all_tagged = _run_face_on_all(frame, bboxes, face_detector)
                    primary, face_bbox = _pick_primary_face_aware(all_tagged, anchor)
                else:
                    primary = _pick_primary(bboxes, anchor)
                    all_tagged = [(p, None) for p in bboxes]

            # Refine: keep the person bbox dimensions but re-anchor its
            # x-center on the detected face x-center. The body bbox drifts
            # to hands/gestures; the face doesn't.
            if primary is not None and face_bbox is not None:
                face_cx = face_bbox.x + face_bbox.w / 2
                new_x = face_cx - primary.w / 2
                primary = BBox(
                    x=new_x, y=primary.y,
                    w=primary.w, h=primary.h,
                    confidence=primary.confidence,
                )
                face_hits += 1

            last_tagged = all_tagged
            last_face = face_bbox

            if debug_writer is not None:
                annotated = _draw_detection_debug(
                    frame, all_tagged, primary, face_bbox, frame_idx, fps)
                debug_writer.write(annotated)

            if primary is not None:
                anchor = primary
                last_bbox = primary
            out.append(primary)
        else:
            # Skipped frame — carry forward last known detection.
            if debug_writer is not None:
                annotated = _draw_detection_debug(
                    frame, last_tagged, last_bbox, last_face, frame_idx, fps)
                debug_writer.write(annotated)
            out.append(last_bbox)

        frame_idx += 1

    cap.release()
    if debug_writer is not None:
        debug_writer.release()
        log.info(f"Debug detect video written: {video_path.parent}/debug_detect.mp4")

    if face_detector is not None and frame_idx > 0:
        log.info(f"Face-aware crop: {face_hits}/{frame_idx} frames ({100*face_hits/frame_idx:.0f}%) used face center")

    non_null = sum(1 for b in out if b is not None)
    log.info(
        f"Detection complete: {non_null}/{len(out)} frames had a person "
        f"({100 * non_null / max(1, len(out)):.1f}%)"
    )
    return out, fps, width, height
