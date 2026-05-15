#!/usr/bin/env python3
"""Test: when a frame contains two people (wide shot), crop each person
separately and stack their panels vertically into a 9:16 reel. When only
one person is visible (close-up), fall back to the standard single 9:16
crop. Side-steps the "which speaker is talking" problem for wide shots.

Two passes:

Pass 1 — per frame, run YOLO + MediaPipe face attribution. Record:
         - list of person bboxes
         - list of attributed face bboxes (None if back-of-head)

Pass 2 — temporal smoothing of "is this a wide shot or close-up frame":
         a frame is "wide" if ≥2 people with faces are visible in at least
         half of a small surrounding window. Then, walk the video again
         and produce the composite output per frame.

Audio is muxed back in at the end via ffmpeg.

Usage:
    python stacked_crop_test.py PATH/TO/CLIP.mp4 [--out PATH] \\
        [--window-frames 15] [--top-is-left | --top-is-right]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_CACHE_DIR = Path.home() / ".cache" / "agentic-video-editor"

# MediaPipe Pose Landmarker indices we care about (out of 33).
POSE_NOSE = 0
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24

TARGET_W = 1080
TARGET_H = 1920
PANEL_H = TARGET_H // 2          # 960
PANEL_W = TARGET_W                # 1080
PANEL_ASPECT = PANEL_W / PANEL_H  # 1.125 (9:8)


def _ensure_face_model() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = _CACHE_DIR / "blaze_face_short_range.tflite"
    if not target.exists():
        print(f"Downloading face detector → {target}")
        urllib.request.urlretrieve(_FACE_MODEL_URL, target)
    return target


def _ensure_pose_model() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = _CACHE_DIR / "pose_landmarker_lite.task"
    if not target.exists():
        print(f"Downloading pose landmarker → {target}")
        urllib.request.urlretrieve(_POSE_MODEL_URL, target)
    return target


def _pose_anchors_for_person(
    frame_bgr: np.ndarray,
    person_bbox: tuple[int, int, int, int],
    pose_landmarker,
) -> dict | None:
    """Run Pose Landmarker on a padded person crop. Return source-pixel
    coordinates for the anchor points we need (nose + shoulder midpoint +
    hip midpoint), or None if the detection failed."""
    H, W = frame_bgr.shape[:2]
    px, py, pw, ph = person_bbox

    # Pad the crop so the pose model has some context above the head.
    pad_x = int(0.15 * pw)
    pad_y = int(0.20 * ph)
    x0 = max(0, px - pad_x)
    y0 = max(0, py - pad_y)
    x1 = min(W, px + pw + pad_x)
    y1 = min(H, py + ph + pad_y)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = frame_bgr[y0:y1, x0:x1]
    crop_h, crop_w = crop.shape[:2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        res = pose_landmarker.detect(mp_image)
    except Exception:  # noqa: BLE001
        return None
    if not res.pose_landmarks:
        return None
    lm = res.pose_landmarks[0]

    def _to_src(idx: int) -> tuple[float, float, float]:
        p = lm[idx]
        return (x0 + p.x * crop_w, y0 + p.y * crop_h, getattr(p, "visibility", 1.0))

    nose = _to_src(POSE_NOSE)
    ls = _to_src(POSE_LEFT_SHOULDER)
    rs = _to_src(POSE_RIGHT_SHOULDER)
    lh = _to_src(POSE_LEFT_HIP)
    rh = _to_src(POSE_RIGHT_HIP)

    # All required points must have at least token visibility.
    if min(nose[2], ls[2], rs[2]) < 0.3:
        return None

    shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    return {
        "nose": (nose[0], nose[1]),
        "shoulder_mid": shoulder_mid,
        "hip_mid": hip_mid,
        "hip_visible": min(lh[2], rh[2]) >= 0.3,
    }


def _bbox_from_pose(anchors: dict, src_w: int, src_h: int, snap: int = 8) -> tuple[int, int, int, int]:
    """Compute the source-pixel crop bbox (x0, y0, w, h) implied by pose anchors.

    Same geometry as _crop_panel_from_pose but returns the bbox tuple instead
    of cropping the frame — so we can do IoU hysteresis against a locked bbox.
    """
    nose_x, nose_y = anchors["nose"]
    sx, sy = anchors["shoulder_mid"]
    head_h = max(8.0, abs(sy - nose_y))
    panel_top = nose_y - 1.5 * head_h
    panel_bot = nose_y + 3.5 * head_h
    panel_h = panel_bot - panel_top
    panel_w = panel_h * PANEL_ASPECT
    cx = nose_x

    def _snap(v: float) -> int:
        return int(round(v / snap) * snap)

    crop_h = _snap(min(panel_h, src_h))
    crop_w = _snap(min(panel_w, src_w))
    y0 = _snap(panel_top)
    x0 = _snap(cx - crop_w / 2)
    y0 = max(0, min(src_h - crop_h, y0))
    x0 = max(0, min(src_w - crop_w, x0))
    return (x0, y0, crop_w, crop_h)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix0 = max(ax, bx); iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw); iy1 = min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _render_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Slice + resize a frame to PANEL_W x PANEL_H using the given source bbox."""
    x0, y0, w, h = bbox
    region = frame[y0:y0 + h, x0:x0 + w]
    return cv2.resize(region, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)


def _crop_panel_from_pose(
    frame: np.ndarray,
    anchors: dict,
    src_w: int,
    src_h: int,
    snap: int = 8,
) -> np.ndarray:
    """Build a tight head+shoulders panel using pose anchors. Output is
    PANEL_W x PANEL_H (9:8).

    Vertical layout:
      panel_top    = nose - 1.5 * head_height    (face sits ~30% from top)
      panel_bottom = nose + 3.5 * head_height    (chest-level framing)

    The final crop window (x0, y0, crop_w, crop_h) is rounded to multiples of
    `snap` pixels so sub-pixel wobble in the anchors can't propagate to the
    output. This kills "size pumping" entirely.
    """
    nose_x, nose_y = anchors["nose"]
    sx, sy = anchors["shoulder_mid"]

    head_h = max(8.0, abs(sy - nose_y))
    panel_top = nose_y - 1.5 * head_h
    panel_bot = nose_y + 3.5 * head_h
    panel_h = panel_bot - panel_top
    panel_w = panel_h * PANEL_ASPECT
    cx = nose_x

    def _snap(v: float) -> int:
        return int(round(v / snap) * snap)

    # Clamp + snap each dimension to multiples of `snap` so frame-to-frame
    # sub-pixel drift can't move the crop window.
    crop_h = _snap(min(panel_h, src_h))
    crop_w = _snap(min(panel_w, src_w))
    y0 = _snap(panel_top)
    x0 = _snap(cx - crop_w / 2)
    y0 = max(0, min(src_h - crop_h, y0))
    x0 = max(0, min(src_w - crop_w, x0))

    region = frame[y0:y0 + crop_h, x0:x0 + crop_w]
    return cv2.resize(region, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)


def _detect_faces(detector, frame_bgr) -> list[tuple[int, int, int, int]]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        res = detector.detect(mp_image)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for d in res.detections:
        bb = d.bounding_box
        out.append((int(bb.origin_x), int(bb.origin_y), int(bb.width), int(bb.height)))
    return out


def _attribute_faces(persons: list[tuple[int, int, int, int]],
                     faces: list[tuple[int, int, int, int]]) -> list[bool]:
    """For each person bbox, return True iff some face has center in the upper
    60% of that person AND ≥50% of face area inside, AND face is wider than
    ~5% of person width. Mirrors src/detect.py's strict gate."""
    out = [False] * len(persons)
    if not persons or not faces:
        return out
    pairs = []  # (face_idx, person_idx, score)
    for fi, (fx, fy, fw, fh) in enumerate(faces):
        f_cy = fy + fh / 2
        f_area = fw * fh
        if f_area <= 0:
            continue
        for pi, (px, py, pw, ph) in enumerate(persons):
            if pw <= 0 or ph <= 0:
                continue
            if f_cy < py or f_cy > py + 0.6 * ph:
                continue
            ix1 = max(fx, px); iy1 = max(fy, py)
            ix2 = min(fx + fw, px + pw); iy2 = min(fy + fh, py + ph)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            enclosed = inter / f_area
            if enclosed < 0.5:
                continue
            if fw < 0.05 * pw:
                continue
            pairs.append((fi, pi, enclosed))
    pairs.sort(key=lambda t: t[2], reverse=True)
    used_face: set[int] = set()
    used_pers: set[int] = set()
    for fi, pi, _ in pairs:
        if fi in used_face or pi in used_pers:
            continue
        out[pi] = True
        used_face.add(fi); used_pers.add(pi)
    return out


def _crop_around(frame: np.ndarray, person_x_center: float, src_h: int, src_w: int,
                 panel_aspect: float) -> np.ndarray:
    """Crop a window of aspect=panel_aspect (w/h) centered horizontally on
    person_x_center, using full source height. Pads/clamps at frame edges.
    Then resizes to (PANEL_W, PANEL_H).
    """
    crop_h = src_h
    crop_w = int(round(crop_h * panel_aspect))
    if crop_w > src_w:
        crop_w = src_w
    x0 = int(round(person_x_center - crop_w / 2))
    x0 = max(0, min(src_w - crop_w, x0))
    x1 = x0 + crop_w
    region = frame[:, x0:x1]
    return cv2.resize(region, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--window-frames", type=int, default=15,
                        help="Temporal window for shot-type smoothing")
    parser.add_argument("--yolo-conf", type=float, default=0.4)
    parser.add_argument("--top-is-left", action="store_true",
                        help="Put left-side person in top panel (default)")
    parser.add_argument("--top-is-right", action="store_true",
                        help="Put right-side person in top panel")
    parser.add_argument("--smoothing-alpha", type=float, default=0.05,
                        help="EMA factor for crop anchors (lower = smoother, "
                             "0=frozen, 1=no smoothing). Default 0.05.")
    parser.add_argument("--snap-px", type=int, default=8,
                        help="Snap final crop window dimensions to multiples "
                             "of this many pixels. Higher = stronger stability "
                             "against sub-pixel jitter. Default 8.")
    parser.add_argument("--iou-threshold", type=float, default=0.70,
                        help="Keep the locked crop bbox while new candidate "
                             "IoU >= this. Below = lock onto candidate. "
                             "Lower (e.g. 0.6) = stiffer (more lock retention). "
                             "Higher (e.g. 0.9) = more responsive. Default 0.70.")
    parser.add_argument("--miss-tolerance", type=int, default=15,
                        help="Hold the locked bbox during up to this many "
                             "consecutive frames of YOLO/Pose detection misses "
                             "before falling back to a safe crop. ~0.5s at 30fps. "
                             "Default 15.")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"not found: {args.video}")
        return 2

    out_path = args.out or args.video.with_name(f"stacked_{args.video.stem}.mp4")

    # Models
    from ultralytics import YOLO
    print("Loading YOLO yolov8n.pt")
    yolo = YOLO("yolov8n.pt")
    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = "cpu"
    print(f"Device: {device}")

    model_path = _ensure_face_model()
    options = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        min_detection_confidence=0.4,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    face_detector = mp_vision.FaceDetector.create_from_options(options)

    pose_model_path = _ensure_pose_model()
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(pose_model_path)),
            num_poses=1,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"could not open: {args.video}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Source: {src_w}x{src_h} @ {fps:.2f}fps, {total} frames")
    print(f"Output: {TARGET_W}x{TARGET_H} → {out_path}")

    # Pass 1: detections per frame
    print("\nPass 1: detect persons + faces")
    per_frame_persons: list[list[tuple[int, int, int, int]]] = []
    per_frame_has_face: list[list[bool]] = []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = yolo.predict(frame, conf=args.yolo_conf, iou=0.5, device=device,
                               verbose=False, classes=[0])
        persons: list[tuple[int, int, int, int]] = []
        if results:
            for box in results[0].boxes or []:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                persons.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        faces = _detect_faces(face_detector, frame)
        has_face = _attribute_faces(persons, faces)
        per_frame_persons.append(persons)
        per_frame_has_face.append(has_face)
        if total and fi % 60 == 0:
            print(f"  {fi}/{total} ({100*fi/total:.0f}%)", end="\r", flush=True)
        fi += 1
    cap.release()
    total_frames = fi
    print(f"\n  {total_frames} frames analysed")

    # Pass 2: smoothed wide-vs-single decision per frame.
    #
    # Geometry-only rule (face attribution can fail on small wide-shot faces
    # at 360p sources): a frame is "wide" if
    #   - ≥ 2 YOLO person bboxes,
    #   - the leftmost and rightmost are separated by ≥ 20% of source width
    #     (so two near-overlapping detections of one person don't qualify),
    #   - and each qualifying bbox is < 70% of source height (a true wide
    #     shot has smaller persons than a single close-up).
    print("Pass 2: smoothing shot-type decisions")
    SEP_THRESHOLD = 0.20 * src_w
    HEIGHT_CAP = 0.70 * src_h

    def _is_wide_raw(persons: list[tuple[int, int, int, int]]) -> bool:
        qual = [(p[0] + p[2] / 2, p[3]) for p in persons if p[3] < HEIGHT_CAP]
        if len(qual) < 2:
            return False
        xs = [cx for cx, _ in qual]
        return (max(xs) - min(xs)) >= SEP_THRESHOLD

    is_wide_raw = np.array([_is_wide_raw(p) for p in per_frame_persons], dtype=bool)
    win = max(1, args.window_frames)
    half = win // 2
    is_wide = np.zeros(total_frames, dtype=bool)
    for i in range(total_frames):
        lo = max(0, i - half); hi = min(total_frames, i + half + 1)
        is_wide[i] = is_wide_raw[lo:hi].mean() >= 0.5

    n_wide = int(is_wide.sum())
    print(f"  wide-shot frames: {n_wide}/{total_frames} ({100*n_wide/total_frames:.1f}%)")

    # Pass 3: render
    print("\nPass 3: rendering")
    tmp_video = Path(tempfile.mkstemp(prefix="stacked_", suffix=".mp4")[1])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (TARGET_W, TARGET_H))
    if not writer.isOpened():
        print(f"failed to open writer: {tmp_video}")
        return 2

    # ── Body-IoU-hysteresis bbox lock per slot, with miss tolerance ─
    # Each slot holds:
    #   "body": the YOLO full-body bbox at the time the crop was last set
    #   "bbox": the actual crop bbox (head+upper-torso panel)
    #   "miss": consecutive_miss_count
    #
    # The lock trigger is IoU of the *body* bbox between the current frame
    # and when the crop was last computed. Body bboxes are large and stable
    # (~100x300px) so IoU stays high unless the person actually moves;
    # tiny pose jitter in the small crop bbox no longer triggers updates.
    IOU_THRESHOLD = float(args.iou_threshold)
    MISS_TOLERANCE = int(args.miss_tolerance)

    state: dict[str, dict] = {
        "top": {"body": None, "bbox": None, "miss": 0},
        "bot": {"body": None, "bbox": None, "miss": 0},
        "single": {"body": None, "bbox": None, "miss": 0},
    }

    cap = cv2.VideoCapture(str(args.video))
    fi = 0
    prev_is_wide = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        persons = per_frame_persons[fi]
        has_face = per_frame_has_face[fi]
        wide = bool(is_wide[fi])

        # Layout transition → drop stale state for slots about to be used.
        if prev_is_wide is not None and wide != prev_is_wide:
            if wide:
                state["top"] = {"body": None, "bbox": None, "miss": 0}
                state["bot"] = {"body": None, "bbox": None, "miss": 0}
            else:
                state["single"] = {"body": None, "bbox": None, "miss": 0}
        prev_is_wide = wide

        if wide:
            # Pick the leftmost and rightmost qualifying bboxes. May yield 0
            # or 1 entries if YOLO momentarily missed someone — miss handling
            # in _panel_for keeps the panel anchored to its last lock.
            qual = [(persons[i][0] + persons[i][2] / 2, persons[i])
                    for i in range(len(persons)) if persons[i][3] < HEIGHT_CAP]
            qual.sort(key=lambda t: t[0])
            left_bbox = qual[0][1] if len(qual) >= 1 else None
            right_bbox = qual[-1][1] if len(qual) >= 2 else None
            if args.top_is_right:
                top_bbox, bot_bbox = right_bbox, left_bbox
            else:
                top_bbox, bot_bbox = left_bbox, right_bbox

            def _panel_for(slot: str, person_bbox):
                slot_state = state[slot]

                # Case 1: no YOLO person bbox this frame for this slot.
                if person_bbox is None:
                    slot_state["miss"] += 1
                    if slot_state["bbox"] is not None and slot_state["miss"] <= MISS_TOLERANCE:
                        return _render_bbox(frame, slot_state["bbox"])
                    return _crop_around(frame, src_w / 2, src_h, src_w, PANEL_ASPECT)

                # We have the person. Compare BODY bbox to the locked body
                # bbox; if the person hasn't really moved (high body IoU),
                # keep the existing crop and don't even run pose.
                if (slot_state["body"] is not None
                        and slot_state["bbox"] is not None
                        and _bbox_iou(person_bbox, slot_state["body"]) >= IOU_THRESHOLD):
                    slot_state["miss"] = 0
                    return _render_bbox(frame, slot_state["bbox"])

                # Body bbox moved enough (or first frame) → recompute crop.
                anchors = _pose_anchors_for_person(frame, person_bbox, pose_landmarker)
                if anchors is None:
                    slot_state["miss"] += 1
                    if slot_state["bbox"] is not None and slot_state["miss"] <= MISS_TOLERANCE:
                        return _render_bbox(frame, slot_state["bbox"])
                    cx = person_bbox[0] + person_bbox[2] / 2
                    return _crop_around(frame, cx, src_h, src_w, PANEL_ASPECT)

                slot_state["miss"] = 0
                new_crop = _bbox_from_pose(anchors, src_w, src_h, snap=args.snap_px)
                slot_state["body"] = person_bbox
                slot_state["bbox"] = new_crop
                return _render_bbox(frame, new_crop)

            top_panel = _panel_for("top", top_bbox)
            bot_panel = _panel_for("bot", bot_bbox)
            composite = np.vstack([top_panel, bot_panel])
            # Thin divider line for visual separation
            composite[PANEL_H - 1:PANEL_H + 1, :] = (10, 10, 10)
        else:
            # Single-shot crop. Same body-IoU hysteresis: keep the crop if
            # the YOLO body bbox barely moved since we last set it.
            slot_state = state["single"]
            picks = [(persons[i], has_face[i]) for i in range(len(persons))]
            front = [p for p, h in picks if h]
            chosen = front[0] if front else (persons[0] if persons else None)

            crop_h = src_h
            crop_w = max(1, int(round(crop_h * TARGET_W / TARGET_H)))
            if crop_w > src_w:
                crop_w = src_w

            if chosen is None:
                slot_state["miss"] += 1
                if slot_state["bbox"] is not None and slot_state["miss"] <= MISS_TOLERANCE:
                    locked = slot_state["bbox"]
                else:
                    locked = (max(0, (src_w - crop_w) // 2), 0, crop_w, crop_h)
            elif (slot_state["body"] is not None
                  and slot_state["bbox"] is not None
                  and _bbox_iou(chosen, slot_state["body"]) >= IOU_THRESHOLD):
                # Body barely moved — keep the existing crop.
                slot_state["miss"] = 0
                locked = slot_state["bbox"]
            else:
                slot_state["miss"] = 0
                raw_cx = chosen[0] + chosen[2] / 2
                x0 = int(round(raw_cx - crop_w / 2))
                x0 = max(0, min(src_w - crop_w, x0))
                x0 = int(round(x0 / args.snap_px) * args.snap_px)
                x0 = max(0, min(src_w - crop_w, x0))
                new_crop = (x0, 0, crop_w, crop_h)
                slot_state["body"] = chosen
                slot_state["bbox"] = new_crop
                locked = new_crop

            composite = cv2.resize(
                frame[locked[1]:locked[1] + locked[3], locked[0]:locked[0] + locked[2]],
                (TARGET_W, TARGET_H),
                interpolation=cv2.INTER_AREA,
            )

        writer.write(composite)
        fi += 1

    cap.release()
    writer.release()
    print(f"  wrote {fi} frames → {tmp_video}")

    # Mux audio back in
    print("\nMuxing audio from source")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(tmp_video),
        "-i", str(args.video),
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest", str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as e:
        print(f"audio mux failed: {e.stderr.strip()[-300:]}")
        return 1
    finally:
        tmp_video.unlink(missing_ok=True)

    print(f"\nDone → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
