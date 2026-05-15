#!/usr/bin/env python3
"""Test: can mouth motion alone identify the active speaker?

Two-pass design so the speaker label at frame T can use frames T+1, T+2, ...
in addition to past frames — useful because mouth-motion at the start of a
word only becomes detectable a few frames later.

Pass 1: For every frame, run YOLO to find all persons, cluster each by
        x-center into stable "tracks" (one per chair), and measure each
        person's mouth-openness via MediaPipe FaceLandmarker. Result is a
        per-track timeseries of (frame_idx, openness).

Pass 2: For every frame, compute the variance of openness within a window
        [t - left, t + right] for each track. The track with the highest
        variance is the speaker at frame t. By default the window is
        centered (looks 0.5s back and 0.5s forward) so a speaker is tagged
        as soon as they start talking, not 1s later.

Pass 3: Re-walk the video, render the debug overlay using the precomputed
        decisions, and write debug_mouth_speaker_<stem>.mp4.

Usage:
    python mouth_speaker_test.py PATH/TO/CLIP.mp4 \\
        [--window-seconds 1.0] [--look-ahead-fraction 0.5] \\
        [--yolo-conf 0.4] [--variance-floor 1e-5]

  --look-ahead-fraction = 0.0  → trailing window (past only)
                          0.5  → centered window (default)
                          1.0  → forward-only window (future only)
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# MediaPipe FaceLandmarker indices (mirrors src/diarize.py).
_UPPER_LIP_IDX = 13
_LOWER_LIP_IDX = 14
_FACE_TOP_IDX = 10
_FACE_BOTTOM_IDX = 152

_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_CACHE_DIR = Path.home() / ".cache" / "agentic-video-editor"

_CLUSTER_TOLERANCE_PX = 120.0
_TRACK_COLORS = [
    (255, 200, 80),
    (80, 180, 255),
    (200, 100, 255),
    (160, 255, 100),
]


def _ensure_landmarker_model() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = _CACHE_DIR / "face_landmarker.task"
    if not target.exists():
        print(f"Downloading FaceLandmarker → {target}")
        urllib.request.urlretrieve(_FACE_LANDMARKER_URL, target)
    return target


def _mouth_openness(frame_bgr: np.ndarray, x: int, y: int, w: int, h: int, landmarker) -> float | None:
    H, W = frame_bgr.shape[:2]
    pad_x = int(0.12 * w)
    pad_y = int(0.18 * h)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(W, x + w + pad_x)
    y1 = min(H, y + h + pad_y)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        result = landmarker.detect(mp_image)
    except Exception:  # noqa: BLE001
        return None
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    mouth_gap = abs(lm[_LOWER_LIP_IDX].y - lm[_UPPER_LIP_IDX].y)
    face_height = abs(lm[_FACE_BOTTOM_IDX].y - lm[_FACE_TOP_IDX].y)
    if face_height <= 0:
        return None
    return float(mouth_gap / face_height)


def _assign_track(x_center: float, track_centers: list[float]) -> int:
    if not track_centers:
        track_centers.append(x_center)
        return 0
    distances = [abs(x_center - c) for c in track_centers]
    nearest = min(range(len(track_centers)), key=lambda i: distances[i])
    if distances[nearest] <= _CLUSTER_TOLERANCE_PX:
        track_centers[nearest] = 0.9 * track_centers[nearest] + 0.1 * x_center
        return nearest
    track_centers.append(x_center)
    return len(track_centers) - 1


def _centered_variance(
    series: np.ndarray,
    mask: np.ndarray,
    center: int,
    left: int,
    right: int,
) -> float | None:
    """Variance of `series` over [center-left, center+right], using only
    entries where `mask` is True. Returns None if fewer than 3 valid samples."""
    n = len(series)
    lo = max(0, center - left)
    hi = min(n, center + right + 1)
    window = series[lo:hi]
    mwin = mask[lo:hi]
    valid = window[mwin]
    if valid.size < 3:
        return None
    return float(valid.var())


def main() -> int:
    parser = argparse.ArgumentParser(description="Mouth-motion speaker identification test.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--window-seconds", type=float, default=1.0,
                        help="Total variance window length in seconds (default 1.0)")
    parser.add_argument("--look-ahead-fraction", type=float, default=0.5,
                        help="0=trailing past, 0.5=centered, 1=forward-only (default 0.5)")
    parser.add_argument("--yolo-conf", type=float, default=0.4)
    parser.add_argument("--variance-floor", type=float, default=1e-5,
                        help="Below this variance, no one is 'speaking'")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"not found: {args.video}", file=sys.stderr)
        return 2

    from ultralytics import YOLO
    print(f"Loading YOLO yolov8n.pt")
    yolo = YOLO("yolov8n.pt")
    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = "cpu"
    print(f"Device: {device}")

    model_path = _ensure_landmarker_model()
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        num_faces=1,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"could not open: {args.video}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {W}x{H} @ {fps:.2f}fps, {total} frames")

    window_n = max(2, int(round(args.window_seconds * fps)))
    ahead = max(0.0, min(1.0, args.look_ahead_fraction))
    right = int(round(window_n * ahead))
    left = window_n - right
    print(f"Variance window: {window_n} frames ({args.window_seconds:.2f}s)  "
          f"→ left={left}f back / right={right}f forward")

    # ----- PASS 1: detect + measure openness per frame, store per track -----
    # Per-frame state we'll need in pass 2/3:
    #   per_frame_persons[i] = list of (track_idx, (x,y,w,h), openness_or_None)
    per_frame_persons: list[list[tuple[int, tuple[int, int, int, int], float | None]]] = []
    track_centers: list[float] = []

    print("\nPass 1: YOLO + FaceLandmarker per frame")
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = yolo.predict(frame, conf=args.yolo_conf, iou=0.5, device=device,
                               verbose=False, classes=[0])
        persons_this_frame: list[tuple[int, tuple[int, int, int, int], float | None]] = []
        seen_tracks: set[int] = set()
        if results:
            for box in results[0].boxes or []:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                xi, yi = int(x1), int(y1)
                wi, hi = int(x2 - x1), int(y2 - y1)
                xc = xi + wi / 2
                track = _assign_track(xc, track_centers)
                if track in seen_tracks:
                    continue
                seen_tracks.add(track)
                openness = _mouth_openness(frame, xi, yi, wi, hi, landmarker)
                persons_this_frame.append((track, (xi, yi, wi, hi), openness))
        per_frame_persons.append(persons_this_frame)

        if total and frame_idx % 60 == 0:
            pct = 100.0 * frame_idx / total
            print(f"  {frame_idx}/{total} ({pct:.0f}%)", end="\r", flush=True)
        frame_idx += 1
    cap.release()
    total_frames = frame_idx
    print(f"\n  {total_frames} frames analysed, {len(track_centers)} tracks: "
          f"{[f'{c:.0f}' for c in track_centers]}")

    # ----- PASS 2: per-track openness timeseries → centered-window variance -----
    n_tracks = len(track_centers)
    openness_series = np.zeros((n_tracks, total_frames), dtype=np.float64)
    openness_mask = np.zeros((n_tracks, total_frames), dtype=bool)
    for fi, persons in enumerate(per_frame_persons):
        for track, _, o in persons:
            if o is not None:
                openness_series[track, fi] = o
                openness_mask[track, fi] = True

    print("\nPass 2: rolling variance per track")
    winner_per_frame: list[int | None] = [None] * total_frames
    variance_per_frame: list[dict[int, float]] = [dict() for _ in range(total_frames)]
    for fi in range(total_frames):
        vmap: dict[int, float] = {}
        for tk in range(n_tracks):
            v = _centered_variance(openness_series[tk], openness_mask[tk], fi, left, right)
            if v is not None:
                vmap[tk] = v
        variance_per_frame[fi] = vmap
        if vmap:
            best = max(vmap, key=lambda t: vmap[t])
            if vmap[best] >= args.variance_floor:
                winner_per_frame[fi] = best

    # ----- PASS 3: render debug overlay using precomputed decisions -----
    out_path = args.video.with_name(f"debug_mouth_speaker_{args.video.stem}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        print(f"could not open writer: {out_path}", file=sys.stderr)
        return 2

    print(f"\nPass 3: rendering → {out_path}")
    cap = cv2.VideoCapture(str(args.video))
    fi = 0
    last_sec = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        winner = winner_per_frame[fi]
        vmap = variance_per_frame[fi]
        persons = per_frame_persons[fi]

        for track, (x, y, w, h), openness in persons:
            color = _TRACK_COLORS[track % len(_TRACK_COLORS)]
            is_speaker = (track == winner)
            outline = (0, 255, 0) if is_speaker else color
            thickness = 4 if is_speaker else 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), outline, thickness)
            v = vmap.get(track)
            v_str = f"σ²={v:.5f}" if v is not None else "σ²=—"
            o_str = f"o={openness:.3f}" if openness is not None else "o=—"
            tag = "SPEAKING" if is_speaker else f"T{track}"
            label = f"{tag}  {o_str}  {v_str}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y), outline, -1)
            cv2.putText(frame, label, (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        t = fi / fps
        cv2.putText(frame, f"f{fi}  t={t:.2f}s",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)

        sec = int(t)
        if sec != last_sec:
            last_sec = sec
            vs = " ".join(f"T{tk}=σ²{v:.5f}" for tk, v in sorted(vmap.items()))
            w_str = f"SPEAKER=T{winner}" if winner is not None else "SPEAKER=—"
            print(f"  t={sec:3d}s  {w_str:14s}  {vs}")
        fi += 1
    cap.release()
    writer.release()
    print(f"\nDone. {fi} frames → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
