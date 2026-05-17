"""Smart 9:16 crop renderers (legacy single-panel + shot-aware stacked)."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

from .types import BBox, Timeline, TimelineSegment

log = logging.getLogger("ave.crop")


class CropError(Exception):
    pass


_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

_POSE_NOSE = 0
_POSE_LEFT_SHOULDER = 11
_POSE_RIGHT_SHOULDER = 12
_POSE_LEFT_HIP = 23
_POSE_RIGHT_HIP = 24

_pose_landmarker = None
_pose_landmarker_lock = threading.Lock()


@dataclass
class _FrameContext:
    segment_idx: int
    bbox: Optional[BBox]
    target_x_center: float


def _segment_for_time(timeline: Timeline, t: float) -> tuple[int, TimelineSegment]:
    for i, seg in enumerate(timeline):
        if seg.start <= t < seg.end:
            return i, seg
    return len(timeline) - 1, timeline[-1]


def _compute_crop_window(
    target_x_center: float,
    source_width: int,
    crop_width: int,
) -> tuple[int, int]:
    x_start = target_x_center - crop_width / 2
    x_start = max(0, min(int(round(x_start)), source_width - crop_width))
    return x_start, x_start + crop_width


def _open_ffmpeg_pipe(
    out_path: Path,
    width: int,
    height: int,
    fps: float,
    cfg: SimpleNamespace,
) -> subprocess.Popen:
    """Open an ffmpeg subprocess that reads raw BGR frames on stdin."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps}",
        "-i", "-",
        "-c:v", cfg.crop.ffmpeg_encoder,
        "-pix_fmt", "yuv420p",
        "-crf", str(cfg.crop.ffmpeg_crf),
        "-preset", cfg.crop.ffmpeg_preset,
        "-an",
        str(out_path),
    ]
    log.debug(f"ffmpeg encoder cmd: {cmd}")
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _mux_audio(source_video: Path, video_only: Path, final_out: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_only),
        "-i", str(source_video),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        str(final_out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as e:
        raise CropError(f"audio mux failed: {e.stderr.strip()[-500:]}") from e


def smart_crop_916(
    source_video: Path,
    timeline: Timeline,
    out_path: Path,
    cfg: SimpleNamespace,
    debug_out: Optional[Path] = None,
) -> Path:
    """DEPRECATED — legacy `crop.mode: single` renderer. Replaced by
    `smart_crop_916_stacked`. Kept for backward compatibility."""
    source_video = Path(source_video)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_w = int(cfg.crop.target_width)
    target_h = int(cfg.crop.target_height)
    if target_w <= 0 or target_h <= 0:
        raise CropError(f"invalid target dims: {target_w}x{target_h}")

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise CropError(f"OpenCV could not open {source_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop_h = src_h
    crop_w = max(1, int(round(crop_h * target_w / target_h)))
    if crop_w > src_w:
        crop_w = src_w
    log.info(
        f"Cropping {source_video.name}: {src_w}x{src_h} → {target_w}x{target_h} "
        f"(crop window: {crop_w}x{crop_h}, fps={fps:.2f})"
    )

    tmp_video = Path(tempfile.mkstemp(prefix="ave_crop_", suffix=".mp4")[1])
    encoder = _open_ffmpeg_pipe(tmp_video, target_w, target_h, fps, cfg)

    debug_encoder = None
    debug_tmp = None
    want_debug = (debug_out is not None) or getattr(cfg.crop, "debug_overlay", False)
    if want_debug:
        debug_tmp = Path(tempfile.mkstemp(prefix="ave_debug_", suffix=".mp4")[1])
        debug_encoder = _open_ffmpeg_pipe(debug_tmp, src_w, src_h, fps, cfg)

    alpha = float(cfg.crop.smoothing_alpha)
    smoothed_x: Optional[float] = None
    last_segment_idx = -1
    last_known_x: Optional[float] = None

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t = frame_idx / fps
            seg_idx, seg = _segment_for_time(timeline, t)

            # On hard cut: reset EMA + seed last_known_x from the new segment's
            # first available bbox so the crop doesn't strand at the old shot's
            # x-center during the bbox-gap (mic / wall artifact).
            if seg_idx != last_segment_idx:
                smoothed_x = None
                last_segment_idx = seg_idx
                last_known_x = None
                seg_end_frame = int(seg.end * fps) if fps else frame_idx
                look_limit = min(frame_idx + int(0.5 * fps), seg_end_frame)
                for look in range(frame_idx, look_limit + 1):
                    b = seg.bbox_at(look)
                    if b is not None:
                        last_known_x = b.x_center
                        break

            bbox = seg.bbox_at(frame_idx)
            if bbox is not None:
                target_x = bbox.x_center
                last_known_x = target_x
            elif last_known_x is not None:
                target_x = last_known_x
            else:
                target_x = src_w / 2

            smoothed_x = target_x if smoothed_x is None else (alpha * target_x + (1 - alpha) * smoothed_x)

            x_start, x_end = _compute_crop_window(smoothed_x, src_w, crop_w)
            cropped = frame[:, x_start:x_end]
            if cropped.shape[1] != crop_w:
                pad = crop_w - cropped.shape[1]
                cropped = cv2.copyMakeBorder(cropped, 0, 0, 0, pad, cv2.BORDER_CONSTANT)

            resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)

            try:
                encoder.stdin.write(resized.tobytes())
            except BrokenPipeError as e:
                stderr = encoder.stderr.read().decode("utf-8", errors="replace")[-500:]
                raise CropError(f"ffmpeg encoder died: {stderr}") from e

            if debug_encoder is not None:
                dbg = frame.copy()
                if bbox is not None:
                    cv2.rectangle(
                        dbg,
                        (int(bbox.x), int(bbox.y)),
                        (int(bbox.x + bbox.w), int(bbox.y + bbox.h)),
                        (0, 255, 255), 2,
                    )
                cv2.rectangle(dbg, (x_start, 0), (x_end, src_h), (0, 255, 0), 3)
                cv2.putText(
                    dbg, f"seg {seg_idx}: {seg.label} t={t:.2f}s",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                )
                debug_encoder.stdin.write(dbg.tobytes())

            frame_idx += 1
    finally:
        cap.release()

    encoder.stdin.close()
    ret = encoder.wait(timeout=120)
    if ret != 0:
        stderr = encoder.stderr.read().decode("utf-8", errors="replace")[-500:]
        raise CropError(f"ffmpeg encoder exited with {ret}: {stderr}")

    if debug_encoder is not None:
        debug_encoder.stdin.close()
        debug_encoder.wait(timeout=120)

    log.info(f"Cropped {frame_idx} frames → muxing audio...")

    try:
        _mux_audio(source_video, tmp_video, out_path)
    finally:
        tmp_video.unlink(missing_ok=True)

    if debug_encoder is not None and debug_tmp is not None:
        if debug_out is None:
            debug_out = out_path.with_name(out_path.stem + "_debug.mp4")
        _mux_audio(source_video, debug_tmp, debug_out)
        debug_tmp.unlink(missing_ok=True)
        log.info(f"Debug overlay → {debug_out}")

    log.info(f"Crop complete → {out_path}")
    return out_path


def _ensure_pose_model(cache_dir: Path) -> Path:
    """Download MediaPipe Pose Landmarker on first use."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "pose_landmarker_lite.task"
    if not target.exists():
        log.info(f"Downloading MediaPipe pose landmarker → {target}")
        urllib.request.urlretrieve(_POSE_MODEL_URL, target)
    return target


def _get_pose_landmarker(cfg: SimpleNamespace):
    """Load the Pose Landmarker once per process (thread-safe)."""
    global _pose_landmarker
    with _pose_landmarker_lock:
        if _pose_landmarker is None:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = _ensure_pose_model(Path(cfg.paths.cache_dir))
            options = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                num_poses=1,
                running_mode=mp_vision.RunningMode.IMAGE,
            )
            _pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    return _pose_landmarker


def _pose_anchors_for_person(
    frame_bgr: np.ndarray,
    person_bbox: BBox,
    pose_landmarker,
) -> Optional[dict]:
    """Pose Landmarker on a padded person crop; returns source-pixel
    {nose, shoulder_mid} or None on miss / low visibility."""
    import mediapipe as mp

    H, W = frame_bgr.shape[:2]
    px, py, pw, ph = int(person_bbox.x), int(person_bbox.y), int(person_bbox.w), int(person_bbox.h)
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

    def _to_src(idx: int):
        p = lm[idx]
        return (x0 + p.x * crop_w, y0 + p.y * crop_h, getattr(p, "visibility", 1.0))

    nose = _to_src(_POSE_NOSE)
    ls = _to_src(_POSE_LEFT_SHOULDER)
    rs = _to_src(_POSE_RIGHT_SHOULDER)
    if min(nose[2], ls[2], rs[2]) < 0.3:
        return None
    shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    return {
        "nose": (nose[0], nose[1]),
        "shoulder_mid": shoulder_mid,
    }


def _bbox_from_pose_anchors(
    anchors: dict,
    src_w: int,
    src_h: int,
    panel_aspect: float,
    snap_px: int = 8,
) -> tuple[int, int, int, int]:
    """Build a (x, y, w, h) crop bbox of `panel_aspect` framed around the
    speaker's head + upper torso. Vertical: top = nose - 1.5·head_h, bottom =
    nose + 3.5·head_h. Dims snapped to multiples of `snap_px` to suppress
    sub-pixel drift before IoU comparisons."""
    nose_x, nose_y = anchors["nose"]
    sx, sy = anchors["shoulder_mid"]
    head_h = max(8.0, abs(sy - nose_y))
    panel_top = nose_y - 1.5 * head_h
    panel_bot = nose_y + 3.5 * head_h
    panel_h = panel_bot - panel_top
    panel_w = panel_h * panel_aspect
    cx = nose_x

    def _snap(v: float) -> int:
        return int(round(v / snap_px) * snap_px)

    crop_h = _snap(min(panel_h, src_h))
    crop_w = _snap(min(panel_w, src_w))
    y0 = _snap(panel_top)
    x0 = _snap(cx - crop_w / 2)
    y0 = max(0, min(src_h - crop_h, y0))
    x0 = max(0, min(src_w - crop_w, x0))
    return (x0, y0, crop_w, crop_h)


def _bbox_iou_tuple(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0 = max(ax, bx); iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw); iy1 = min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _bbox_to_tuple(b: BBox) -> tuple[int, int, int, int]:
    return (int(b.x), int(b.y), int(b.w), int(b.h))


def _lerp_bbox(a: tuple[int, int, int, int],
               b: tuple[int, int, int, int],
               t: float) -> tuple[int, int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(round(av + (bv - av) * t)) for av, bv in zip(a, b))


def _bbox_within(a: tuple[int, int, int, int],
                 b: tuple[int, int, int, int],
                 tol: int) -> bool:
    return all(abs(av - bv) <= tol for av, bv in zip(a, b))


def _render_from_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int],
                     panel_w: int, panel_h: int) -> np.ndarray:
    x, y, w, h = bbox
    return cv2.resize(frame[y:y + h, x:x + w], (panel_w, panel_h),
                      interpolation=cv2.INTER_AREA)


def _fallback_single_panel(
    frame: np.ndarray, cx: float, src_w: int, src_h: int,
    panel_w: int, panel_h: int, panel_aspect: float,
) -> np.ndarray:
    """Centered, full-height crop at panel_aspect — used when pose fails."""
    crop_h = src_h
    crop_w = max(1, int(round(crop_h * panel_aspect)))
    if crop_w > src_w:
        crop_w = src_w
    x0 = int(round(cx - crop_w / 2))
    x0 = max(0, min(src_w - crop_w, x0))
    return _render_from_bbox(frame, (x0, 0, crop_w, crop_h), panel_w, panel_h)


def smart_crop_916_stacked(
    source_video: Path,
    per_frame_persons: list[list[BBox]],
    is_wide: "np.ndarray",
    out_path: Path,
    cfg: SimpleNamespace,
) -> Path:
    """Shot-aware 9:16 renderer: single panel for close-ups, stacked dual
    panels (9:8 each) for wide shots. Pose-anchored framing with body-bbox
    IoU hysteresis and lerp transitions so the crop only updates when the
    speaker actually moves."""
    source_video = Path(source_video)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_w = int(cfg.crop.target_width)
    target_h = int(cfg.crop.target_height)
    panel_w = target_w
    panel_h = target_h // 2
    panel_aspect = panel_w / panel_h
    single_aspect = target_w / target_h

    iou_threshold = float(getattr(cfg.crop, "stacked_iou_threshold", 0.70))
    miss_tolerance = int(getattr(cfg.crop, "stacked_miss_tolerance", 15))
    snap_px = int(getattr(cfg.crop, "stacked_snap_px", 8))
    height_cap_frac = float(getattr(cfg.crop, "shot_height_cap_frac", 0.70))
    top_is_right = bool(getattr(cfg.crop, "stacked_top_is_right", False))
    transition_frames = int(getattr(cfg.crop, "stacked_transition_frames", 12))
    # per-frame alpha so a sustained target change settles in ~transition_frames
    transition_alpha = 1.0 / max(1, transition_frames)

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise CropError(f"OpenCV could not open {source_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height_cap = height_cap_frac * src_h

    n_wide = int(is_wide.sum()) if hasattr(is_wide, "sum") else sum(1 for v in is_wide if v)
    log.info(
        f"Stacked crop {source_video.name}: {src_w}x{src_h} → {target_w}x{target_h} "
        f"(wide-shot frames: {n_wide}/{len(per_frame_persons)} "
        f"({100*n_wide/max(1,len(per_frame_persons)):.1f}%), fps={fps:.2f})"
    )

    pose_landmarker = _get_pose_landmarker(cfg)

    tmp_video = Path(tempfile.mkstemp(prefix="ave_stack_", suffix=".mp4")[1])
    encoder = _open_ffmpeg_pipe(tmp_video, target_w, target_h, fps, cfg)

    # Per-slot lock state:
    #   body:     YOLO body bbox when target was set (the hysteresis signal —
    #             the small crop bbox would be too sensitive to pose jitter)
    #   target:   destination crop bbox set when body-IoU lock breaks
    #   rendered: lerps toward `target` by `transition_alpha` per frame
    #   miss:     consecutive YOLO/Pose failure count
    def _new_slot() -> dict:
        return {"body": None, "target": None, "rendered": None, "miss": 0}

    state: dict[str, dict] = {
        "top": _new_slot(),
        "bot": _new_slot(),
        "single": _new_slot(),
    }

    def _reset(slot: str) -> None:
        state[slot] = _new_slot()

    def _advance_render(s: dict) -> tuple[int, int, int, int]:
        if s["rendered"] is None:
            s["rendered"] = s["target"]
        elif s["target"] is not None and s["rendered"] != s["target"]:
            new_rendered = _lerp_bbox(s["rendered"], s["target"], transition_alpha)
            # snap to target once within 1px to avoid forever-wobble
            if _bbox_within(new_rendered, s["target"], 1):
                new_rendered = s["target"]
            s["rendered"] = new_rendered
        return s["rendered"]

    def _panel_with_pose(slot: str, frame: np.ndarray,
                         person_bbox: Optional[BBox]) -> np.ndarray:
        s = state[slot]

        if person_bbox is None:
            s["miss"] += 1
            if s["rendered"] is not None and s["miss"] <= miss_tolerance:
                return _render_from_bbox(frame, _advance_render(s),
                                         panel_w, panel_h)
            return _fallback_single_panel(frame, src_w / 2, src_w, src_h,
                                          panel_w, panel_h, panel_aspect)

        cur_body = _bbox_to_tuple(person_bbox)
        if (s["body"] is not None and s["target"] is not None
                and _bbox_iou_tuple(cur_body, s["body"]) >= iou_threshold):
            s["miss"] = 0
            return _render_from_bbox(frame, _advance_render(s),
                                     panel_w, panel_h)

        anchors = _pose_anchors_for_person(frame, person_bbox, pose_landmarker)
        if anchors is None:
            s["miss"] += 1
            if s["rendered"] is not None and s["miss"] <= miss_tolerance:
                return _render_from_bbox(frame, _advance_render(s),
                                         panel_w, panel_h)
            return _fallback_single_panel(frame, person_bbox.x_center, src_w, src_h,
                                          panel_w, panel_h, panel_aspect)

        s["miss"] = 0
        new_target = _bbox_from_pose_anchors(anchors, src_w, src_h,
                                             panel_aspect, snap_px=snap_px)
        s["body"] = cur_body
        s["target"] = new_target
        return _render_from_bbox(frame, _advance_render(s),
                                 panel_w, panel_h)

    prev_is_wide: Optional[bool] = None
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx >= len(per_frame_persons):
                persons: list[BBox] = []
                wide = False
            else:
                persons = per_frame_persons[frame_idx]
                wide = bool(is_wide[frame_idx])

            # Reset locks across layout transitions
            if prev_is_wide is not None and wide != prev_is_wide:
                if wide:
                    _reset("top"); _reset("bot")
                else:
                    _reset("single")
            prev_is_wide = wide

            if wide:
                qual = [(p.x_center, p) for p in persons if p.h < height_cap]
                qual.sort(key=lambda t: t[0])
                left_bbox = qual[0][1] if len(qual) >= 1 else None
                right_bbox = qual[-1][1] if len(qual) >= 2 else None
                if top_is_right:
                    top_b, bot_b = right_bbox, left_bbox
                else:
                    top_b, bot_b = left_bbox, right_bbox

                top_panel = _panel_with_pose("top", frame, top_b)
                bot_panel = _panel_with_pose("bot", frame, bot_b)
                composite = np.vstack([top_panel, bot_panel])
                composite[panel_h - 1:panel_h + 1, :] = (10, 10, 10)
            else:
                s = state["single"]
                chosen: Optional[BBox] = None
                for p in persons:
                    if p.h < height_cap or len(persons) == 1:
                        chosen = p
                        break
                if chosen is None and persons:
                    chosen = max(persons, key=lambda p: p.area)

                crop_h = src_h
                crop_w = max(1, int(round(crop_h * single_aspect)))
                if crop_w > src_w:
                    crop_w = src_w

                if chosen is None:
                    s["miss"] += 1
                    if s["rendered"] is not None and s["miss"] <= miss_tolerance:
                        locked = _advance_render(s)
                    else:
                        locked = (max(0, (src_w - crop_w) // 2), 0, crop_w, crop_h)
                else:
                    cur_body = _bbox_to_tuple(chosen)
                    if (s["body"] is not None and s["target"] is not None
                            and _bbox_iou_tuple(cur_body, s["body"]) >= iou_threshold):
                        s["miss"] = 0
                    else:
                        s["miss"] = 0
                        raw_cx = chosen.x_center
                        x0 = int(round(raw_cx - crop_w / 2))
                        x0 = max(0, min(src_w - crop_w, x0))
                        x0 = int(round(x0 / snap_px) * snap_px)
                        x0 = max(0, min(src_w - crop_w, x0))
                        s["target"] = (x0, 0, crop_w, crop_h)
                        s["body"] = cur_body
                    locked = _advance_render(s)

                composite = _render_from_bbox(frame, locked, target_w, target_h)

            try:
                encoder.stdin.write(composite.tobytes())
            except BrokenPipeError as e:
                stderr = encoder.stderr.read().decode("utf-8", errors="replace")[-500:]
                raise CropError(f"ffmpeg encoder died: {stderr}") from e
            frame_idx += 1
    finally:
        cap.release()

    encoder.stdin.close()
    ret = encoder.wait(timeout=180)
    if ret != 0:
        stderr = encoder.stderr.read().decode("utf-8", errors="replace")[-500:]
        raise CropError(f"ffmpeg encoder exited with {ret}: {stderr}")

    log.info(f"Stacked-cropped {frame_idx} frames → muxing audio...")
    try:
        _mux_audio(source_video, tmp_video, out_path)
    finally:
        tmp_video.unlink(missing_ok=True)

    log.info(f"Stacked crop complete → {out_path}")
    return out_path
