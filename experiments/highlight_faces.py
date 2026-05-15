#!/usr/bin/env python3
"""Highlight faces in a video by drawing bounding boxes on each frame.

Uses MediaPipe Tasks API (BlazeFace short-range, ~2m framing).

Usage:
    python highlight_faces.py INPUT.mp4 OUTPUT.mp4 [--confidence 0.5]
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_CACHE_DIR = Path.home() / ".cache" / "agentic-video-editor"


def _ensure_face_model() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = _CACHE_DIR / "blaze_face_short_range.tflite"
    if not target.exists():
        print(f"Downloading MediaPipe face detector → {target}")
        urllib.request.urlretrieve(_FACE_MODEL_URL, target)
    return target


def highlight_faces(input_path: str, output_path: str, confidence: float) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"Could not open input video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        sys.exit(f"Could not open output video for writing: {output_path}")

    model_path = _ensure_face_model()
    options = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        min_detection_confidence=confidence,
        running_mode=mp_vision.RunningMode.VIDEO,
    )

    frame_idx = 0
    with mp_vision.FaceDetector.create_from_options(options) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000.0 / fps)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            for det in result.detections:
                bb = det.bounding_box
                x, y, w, h = bb.origin_x, bb.origin_y, bb.width, bb.height
                score = det.categories[0].score if det.categories else 0.0

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{score:.2f}"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, y - th - baseline - 4), (x + tw + 4, y), (0, 255, 0), -1)
                cv2.putText(frame, label, (x + 2, y - baseline - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1
            if total_frames and frame_idx % 30 == 0:
                pct = 100.0 * frame_idx / total_frames
                print(f"\rProcessed {frame_idx}/{total_frames} frames ({pct:.1f}%)", end="", flush=True)

    cap.release()
    writer.release()
    print(f"\nWrote {frame_idx} frames to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Highlight faces in a video with MediaPipe.")
    parser.add_argument("input", help="Path to the input video.")
    parser.add_argument("output", help="Path to the output video (mp4).")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Minimum detection confidence (0.0–1.0). Default: 0.5")
    args = parser.parse_args()

    highlight_faces(args.input, args.output, args.confidence)


if __name__ == "__main__":
    main()
