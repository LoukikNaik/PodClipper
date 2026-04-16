"""Stage 5c: Smart 9:16 crop driven by a speaker timeline.

Pipeline:
  1. Decode source clip frame-by-frame with OpenCV.
  2. For each frame, look up the active timeline segment and its target bbox.
  3. Compute target crop-center x, apply EMA smoothing within a segment,
     reset smoothing on segment boundaries (hard cut).
  4. Clamp x to stay inside frame, slice, resize to target WxH.
  5. Pipe raw BGR frames into an ffmpeg encoder subprocess.

Audio is preserved by running a second ffmpeg pass after the video pipe finishes
(copying audio from the source clip into the cropped video).

Optional debug overlay mode draws bboxes, crop rects, and segment labels on
an auxiliary video so smoothing parameters can be tuned visually.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
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


@dataclass
class _FrameContext:
    """What segment/bbox/x-center applies to one source frame."""
    segment_idx: int
    bbox: Optional[BBox]
    target_x_center: float


def _segment_for_time(timeline: Timeline, t: float) -> tuple[int, TimelineSegment]:
    """Find the timeline segment covering time `t`. Clamps to last on overflow."""
    for i, seg in enumerate(timeline):
        if seg.start <= t < seg.end:
            return i, seg
    return len(timeline) - 1, timeline[-1]


def _compute_crop_window(
    target_x_center: float,
    source_width: int,
    crop_width: int,
) -> tuple[int, int]:
    """Return (x_start, x_end) for a centered crop window, clamped to frame."""
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
        "-an",  # no audio in this pass; muxed in later
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
    """Copy the audio track from source into the cropped (video-only) output."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_only),
        "-i", str(source_video),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",   # optional — don't fail if source has no audio
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
    """Produce a 9:16 crop of `source_video` following `timeline`.

    Returns the final output path (with audio re-muxed from source).
    If `debug_out` is set OR `cfg.crop.debug_overlay` is true, also emits a
    debug video with bbox + crop-rect overlays.
    """
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

    # Crop window in source pixels — 9:16 aspect based on source height.
    crop_h = src_h
    crop_w = max(1, int(round(crop_h * target_w / target_h)))
    if crop_w > src_w:
        # Source is already taller than 9:16 — crop_w clamped, use whole width
        crop_w = src_w
    log.info(
        f"Cropping {source_video.name}: {src_w}x{src_h} → {target_w}x{target_h} "
        f"(crop window: {crop_w}x{crop_h}, fps={fps:.2f})"
    )

    # --- Open encoder(s) ---
    tmp_video = Path(tempfile.mkstemp(prefix="ave_crop_", suffix=".mp4")[1])
    encoder = _open_ffmpeg_pipe(tmp_video, target_w, target_h, fps, cfg)

    debug_encoder = None
    debug_tmp = None
    want_debug = (debug_out is not None) or getattr(cfg.crop, "debug_overlay", False)
    if want_debug:
        debug_tmp = Path(tempfile.mkstemp(prefix="ave_debug_", suffix=".mp4")[1])
        debug_encoder = _open_ffmpeg_pipe(debug_tmp, src_w, src_h, fps, cfg)

    # --- Main frame loop ---
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

            # Hard cut on segment change: reset EMA to the new target
            if seg_idx != last_segment_idx:
                smoothed_x = None
                last_segment_idx = seg_idx

            bbox = seg.bbox_at(frame_idx)
            if bbox is not None:
                target_x = bbox.x_center
                last_known_x = target_x
            elif last_known_x is not None:
                target_x = last_known_x
            else:
                target_x = src_w / 2  # fallback: frame center

            smoothed_x = target_x if smoothed_x is None else (alpha * target_x + (1 - alpha) * smoothed_x)

            x_start, x_end = _compute_crop_window(smoothed_x, src_w, crop_w)
            cropped = frame[:, x_start:x_end]
            if cropped.shape[1] != crop_w:
                # Edge case: source narrower than crop_w, pad with black
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
                # Draw bbox (yellow)
                if bbox is not None:
                    cv2.rectangle(
                        dbg,
                        (int(bbox.x), int(bbox.y)),
                        (int(bbox.x + bbox.w), int(bbox.y + bbox.h)),
                        (0, 255, 255), 2,
                    )
                # Draw crop window (green)
                cv2.rectangle(dbg, (x_start, 0), (x_end, src_h), (0, 255, 0), 3)
                # Segment label
                cv2.putText(
                    dbg, f"seg {seg_idx}: {seg.label} t={t:.2f}s",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                )
                debug_encoder.stdin.write(dbg.tobytes())

            frame_idx += 1
    finally:
        cap.release()

    # --- Close encoders ---
    encoder.stdin.close()
    ret = encoder.wait(timeout=120)
    if ret != 0:
        stderr = encoder.stderr.read().decode("utf-8", errors="replace")[-500:]
        raise CropError(f"ffmpeg encoder exited with {ret}: {stderr}")

    if debug_encoder is not None:
        debug_encoder.stdin.close()
        debug_encoder.wait(timeout=120)

    log.info(f"Cropped {frame_idx} frames → muxing audio...")

    # --- Mux audio ---
    try:
        _mux_audio(source_video, tmp_video, out_path)
    finally:
        tmp_video.unlink(missing_ok=True)

    # --- Finalize debug output ---
    if debug_encoder is not None and debug_tmp is not None:
        if debug_out is None:
            debug_out = out_path.with_name(out_path.stem + "_debug.mp4")
        _mux_audio(source_video, debug_tmp, debug_out)
        debug_tmp.unlink(missing_ok=True)
        log.info(f"Debug overlay → {debug_out}")

    log.info(f"Crop complete → {out_path}")
    return out_path
