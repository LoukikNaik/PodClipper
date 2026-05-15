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


def _detect_faces_full_frame(frame_bgr, detector) -> list[BBox]:
    """Run face detection once on the full frame. Returns face bboxes in source
    pixel coordinates. Used for single-pass attribution to person bboxes —
    avoids the cross-leakage bug where a neighbor's face falls inside another
    person's padded crop and is misattributed."""
    import mediapipe as mp

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        result = detector.detect(mp_image)
    except Exception:  # noqa: BLE001
        return []
    out: list[BBox] = []
    for det in result.detections:
        bb = det.bounding_box
        score = det.categories[0].score if det.categories else 0.0
        out.append(BBox(
            x=float(bb.origin_x),
            y=float(bb.origin_y),
            w=float(bb.width),
            h=float(bb.height),
            confidence=float(score),
        ))
    return out


def _face_person_score(face: BBox, person: BBox) -> float:
    """Score how strongly `face` belongs to `person`.

    Returns 0.0 if any of these strict tests fail:
      - Face center vertically in the upper 60% of the person bbox
        (a real face sits on top of a body; a face leaking in from a
        neighbor often appears at mic/chest height).
      - Majority of the face's area lies inside the person bbox (≥ 0.5).
      - Face is wider than ~5% of person width (filters tiny FP detections).

    Otherwise returns the fraction of face area inside the person bbox,
    so greedy assignment naturally prefers the most-enclosed face.
    """
    if person.w <= 0 or person.h <= 0 or face.w <= 0 or face.h <= 0:
        return 0.0

    face_cy = face.y + face.h / 2
    if face_cy < person.y or face_cy > person.y + 0.6 * person.h:
        return 0.0

    ix1 = max(face.x, person.x)
    iy1 = max(face.y, person.y)
    ix2 = min(face.x + face.w, person.x + person.w)
    iy2 = min(face.y + face.h, person.y + person.h)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    face_area = face.w * face.h
    enclosed = inter / face_area
    if enclosed < 0.5:
        return 0.0

    if face.w < 0.05 * person.w:
        return 0.0
    return enclosed


def _attribute_faces_to_persons(
    persons: list[BBox],
    faces: list[BBox],
) -> list[Optional[BBox]]:
    """Greedily assign each face to at most one person.

    Returns `face_for_person[i]` aligned with `persons`. A face is assigned to
    its highest-scoring person (per `_face_person_score`), and each face/person
    can only be matched once — so a face that overlaps two person bboxes does
    NOT get attributed to both.
    """
    out: list[Optional[BBox]] = [None] * len(persons)
    if not persons or not faces:
        return out

    # Build all (face_idx, person_idx, score) triples, sort high → low.
    pairs: list[tuple[int, int, float]] = []
    for fi, face in enumerate(faces):
        for pi, person in enumerate(persons):
            s = _face_person_score(face, person)
            if s > 0.0:
                pairs.append((fi, pi, s))
    pairs.sort(key=lambda t: t[2], reverse=True)

    used_face: set[int] = set()
    used_person: set[int] = set()
    for fi, pi, _ in pairs:
        if fi in used_face or pi in used_person:
            continue
        out[pi] = faces[fi]
        used_face.add(fi)
        used_person.add(pi)
    return out


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
    """Detect faces once on the full frame, then attribute each face to at most
    one person bbox via `_attribute_faces_to_persons`.

    Returns a list of (person_bbox, face_bbox_or_None).
    A non-None face_bbox means the person is front-facing AND owns that face
    exclusively — preventing the back-of-head / neighbor-face leakage bug.
    """
    faces = _detect_faces_full_frame(frame_bgr, detector)
    face_per_person = _attribute_faces_to_persons(candidates, faces)
    return list(zip(candidates, face_per_person))


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


def detect_humans_all_per_frame(
    video_path: Path,
    cfg: SimpleNamespace,
) -> tuple[list[list[BBox]], list[list[bool]], float, int, int]:
    """Like `detect_humans_per_frame` but returns ALL detected persons per
    frame (not just the primary), plus a parallel face-attribution flag list.

    Used by the shot-aware stacked crop renderer, which needs to know about
    multiple people in wide shots and can't make do with the single primary
    bbox stored by `detect_humans_per_frame`.

    Returns:
        (per_frame_persons, per_frame_has_face, fps, width, height)
          - per_frame_persons[i] = list of BBox detected in frame i
          - per_frame_has_face[i][j] = True iff the jth person in frame i
            had a face that passed _attribute_faces_to_persons' strict gate
          - Length always equals the source frame count; entries may be
            empty lists when no person was detected.

    Always runs detection per frame (no sample_every_n_frames carry-forward)
    because the stacked path's body-IoU hysteresis needs current detections
    every frame to decide if the locked crop should hold or update.
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

    log.info(
        f"Multi-person detection on {video_path.name}: "
        f"{total_frames} frames @ {fps:.2f}fps"
        + (", face-aware tagging enabled" if face_detector else "")
    )

    per_frame_persons: list[list[BBox]] = []
    per_frame_has_face: list[list[bool]] = []
    frame_idx = 0
    multi_person_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            frame,
            conf=det_cfg.confidence,
            iou=det_cfg.iou,
            device=device,
            verbose=False,
            classes=[det_cfg.person_class_id],
        )

        bboxes: list[BBox] = []
        has_face: list[bool] = []
        if results:
            bboxes = _detections_to_bboxes(
                results[0],
                person_class_id=det_cfg.person_class_id,
                conf_threshold=det_cfg.confidence,
            )
            if face_detector is not None and bboxes:
                tagged = _run_face_on_all(frame, bboxes, face_detector)
                has_face = [f is not None for _, f in tagged]
            else:
                has_face = [False] * len(bboxes)

        per_frame_persons.append(bboxes)
        per_frame_has_face.append(has_face)
        if len(bboxes) >= 2:
            multi_person_frames += 1
        frame_idx += 1

    cap.release()

    log.info(
        f"Multi-person detection complete: {frame_idx} frames, "
        f"{multi_person_frames} ({100*multi_person_frames/max(1,frame_idx):.1f}%) "
        f"with ≥2 persons"
    )
    return per_frame_persons, per_frame_has_face, fps, width, height
