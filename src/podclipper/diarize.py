"""DEPRECATED — legacy `crop.mode: single` only.

Speaker diarization (pyannote.audio) + mouth-motion face linking. Not reached
in the default `auto` mode. Kept for single-locked-camera podcasts and as
historical reference; replaced by `src/crop.py::smart_crop_916_stacked`."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

from .types import BBox, DiarSegment, Timeline, TimelineSegment

log = logging.getLogger("ave.diarize")


class DiarizeError(Exception):
    pass


_pipeline = None
_pipeline_key: tuple | None = None
_pipeline_lock = threading.Lock()


def _get_pipeline(cfg: SimpleNamespace):
    """Load the pyannote diarization pipeline once per process."""
    global _pipeline, _pipeline_key
    from pyannote.audio import Pipeline
    import torch

    model_name = cfg.diarize.model
    token_env = getattr(cfg.diarize, "hf_token_env", "HF_TOKEN")
    token = os.environ.get(token_env)
    key = (model_name, bool(token))

    with _pipeline_lock:
        if _pipeline is None or _pipeline_key != key:
            if not token:
                raise DiarizeError(
                    f"{token_env} env var not set — pyannote models are gated. "
                    f"Accept terms at https://huggingface.co/{model_name} "
                    f"then export {token_env}=hf_xxx"
                )
            log.info(f"Loading pyannote pipeline: {model_name}")
            pipe = Pipeline.from_pretrained(model_name, token=token)
            try:
                if torch.cuda.is_available():
                    pipe.to(torch.device("cuda"))
                elif torch.backends.mps.is_available():
                    pipe.to(torch.device("mps"))
            except Exception as e:  # noqa: BLE001
                log.debug(f"pyannote device move failed ({e}); staying on CPU")
            _pipeline = pipe
            _pipeline_key = key
    return _pipeline


def diarize_clip(
    segment_path: Path,
    cfg: SimpleNamespace,
) -> Optional[list[DiarSegment]]:
    """Run pyannote on the clip audio; returns DiarSegments or None on failure."""
    try:
        pipe = _get_pipeline(cfg)
    except DiarizeError as e:
        log.warning(f"Diarization disabled: {e}")
        return None
    except Exception as e:  # noqa: BLE001
        log.warning(f"Failed to load pyannote pipeline: {e}")
        return None

    kwargs: dict = {}
    min_spk = getattr(cfg.diarize, "min_speakers", None)
    max_spk = getattr(cfg.diarize, "max_speakers", None)
    if min_spk is not None:
        kwargs["min_speakers"] = int(min_spk)
    if max_spk is not None:
        kwargs["max_speakers"] = int(max_spk)

    try:
        result = pipe(str(segment_path), **kwargs)
    except Exception as e:  # noqa: BLE001
        log.warning(f"pyannote run failed on {segment_path.name}: {e}")
        return None

    # pyannote 4.x returns a wrapper; 3.x returned Annotation directly
    annotation = getattr(result, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(result, "speaker_diarization", result)

    segments: list[DiarSegment] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(DiarSegment(
            start=float(turn.start),
            end=float(turn.end),
            speaker_id=str(speaker),
        ))
    segments.sort(key=lambda s: s.start)
    unique = sorted({s.speaker_id for s in segments})
    log.info(
        f"Diarized {segment_path.name}: "
        f"{len(segments)} turns, {len(unique)} speakers {unique}"
    )
    return segments


# MediaPipe FaceLandmarker indices: 13/14 = upper/lower lip, 10/152 = forehead/chin.
# Vertical distance between lips, normalized by face height, gives mouth-openness.
_UPPER_LIP_IDX = 13
_LOWER_LIP_IDX = 14
_FACE_TOP_IDX = 10
_FACE_BOTTOM_IDX = 152

_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

_landmarker = None
_landmarker_lock = threading.Lock()


def _ensure_face_model(cache_dir: Path) -> Path:
    import urllib.request

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "face_landmarker.task"
    if not target.exists():
        log.info(f"Downloading MediaPipe face_landmarker model → {target}")
        urllib.request.urlretrieve(_FACE_LANDMARKER_URL, target)
    return target


def _get_landmarker(cfg: SimpleNamespace):
    """Load the FaceLandmarker once per process."""
    global _landmarker
    with _landmarker_lock:
        if _landmarker is None:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = _ensure_face_model(Path(cfg.paths.cache_dir))
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                num_faces=1,
                running_mode=mp_vision.RunningMode.IMAGE,
            )
            _landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def _mouth_openness(frame_bgr: np.ndarray, bbox: BBox, landmarker) -> Optional[float]:
    """Mouth-openness ratio (mouth gap / face height) at one face bbox."""
    import mediapipe as mp

    H, W = frame_bgr.shape[:2]
    pad_x = int(0.12 * bbox.w)
    pad_y = int(0.18 * bbox.h)
    x0 = max(0, int(bbox.x) - pad_x)
    y0 = max(0, int(bbox.y) - pad_y)
    x1 = min(W, int(bbox.x + bbox.w) + pad_x)
    y1 = min(H, int(bbox.y + bbox.h) + pad_y)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]

    upper = lm[_UPPER_LIP_IDX]
    lower = lm[_LOWER_LIP_IDX]
    face_top = lm[_FACE_TOP_IDX]
    face_bottom = lm[_FACE_BOTTOM_IDX]
    mouth_gap = abs(lower.y - upper.y)
    face_height = abs(face_bottom.y - face_top.y)
    if face_height <= 0:
        return None
    return float(mouth_gap / face_height)


def _cluster_frames_in_window(
    cluster_frames: set[int],
    start_frame: int,
    end_frame: int,
    max_samples: int = 40,
) -> list[int]:
    """Up to `max_samples` evenly-spaced frame indices in [start, end) that
    belong to `cluster_frames`."""
    in_window = [f for f in sorted(cluster_frames) if start_frame <= f < end_frame]
    if not in_window:
        return []
    if len(in_window) <= max_samples:
        return in_window
    step = len(in_window) / max_samples
    return [in_window[int(i * step)] for i in range(max_samples)]


def _sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.var())


def _mouth_variance_for_cluster(
    video_path: Path,
    cluster_frames_sample: list[int],
    per_frame_bboxes: list[Optional[BBox]],
    landmarker,
    debug_collect: Optional[list] = None,
) -> float:
    """Mouth-openness variance across sampled frames in a cluster — the
    higher the variance, the more likely this face is the active speaker."""
    if not cluster_frames_sample:
        return 0.0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    readings: list[float] = []
    try:
        for frame_idx in cluster_frames_sample:
            bbox = per_frame_bboxes[frame_idx] if 0 <= frame_idx < len(per_frame_bboxes) else None
            if bbox is None:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            m = _mouth_openness(frame, bbox, landmarker)
            if m is not None:
                readings.append(m)
            if debug_collect is not None:
                debug_collect.append((frame_idx, frame, bbox, m))
    finally:
        cap.release()
    return _sample_variance(readings)


def _draw_mouth_debug_frame(
    frame: np.ndarray,
    face_bbox: BBox,
    openness: Optional[float],
    frame_idx: int,
    fps: float,
    cluster_idx: int,
    speaker_id: Optional[str] = None,
) -> np.ndarray:
    """Mouth-region box colored by openness: green=speaking, blue=silent,
    orange=no landmarks."""
    annotated = frame.copy()
    H, W = annotated.shape[:2]

    px, py = int(face_bbox.x), int(face_bbox.y)
    pw, ph = int(face_bbox.w), int(face_bbox.h)
    cv2.rectangle(annotated, (px, py), (px + pw, py + ph), (180, 180, 180), 1)

    mouth_y = py + int(ph * 0.68)
    mouth_h = int(ph * 0.22)
    mouth_w = int(pw * 0.45)
    mouth_x = px + int((pw - mouth_w) / 2)
    mouth_x = max(0, min(W - mouth_w, mouth_x))
    mouth_y = max(0, min(H - mouth_h, mouth_y))

    if openness is None:
        color = (0, 165, 255)
        label = "no landmarks"
    elif openness > 0.03:
        color = (0, 220, 0)
        label = f"SPEAK {openness:.3f}"
    else:
        color = (220, 60, 60)
        label = f"silent {openness:.3f}"

    cv2.rectangle(annotated, (mouth_x, mouth_y),
                  (mouth_x + mouth_w, mouth_y + mouth_h), color, 2)
    cv2.putText(annotated, label, (mouth_x, max(12, mouth_y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    t = frame_idx / fps if fps > 0 else 0.0
    spk = f"  spk={speaker_id}" if speaker_id else ""
    cv2.putText(annotated, f"f{frame_idx}  t={t:.2f}s  cluster={cluster_idx}{spk}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def _speaker_primary_window(
    diar_segments: list[DiarSegment],
    speaker_id: str,
) -> Optional[tuple[float, float]]:
    """(start, end) of the longest contiguous speaking window for this speaker."""
    spans = [(s.start, s.end) for s in diar_segments if s.speaker_id == speaker_id]
    if not spans:
        return None
    return max(spans, key=lambda se: se[1] - se[0])


def _link_speakers_to_clusters(
    diar_segments: list[DiarSegment],
    bbox_clusters: list[list[int]],
    per_frame_bboxes: list[Optional[BBox]],
    fps: float,
    video_path: Path,
    cfg: SimpleNamespace,
) -> dict[str, int]:
    """Map each diarized speaker_id → bbox-cluster index with highest mouth-motion
    variance during that speaker's primary window."""
    unique_speakers = sorted({s.speaker_id for s in diar_segments})
    if not unique_speakers or not bbox_clusters:
        return {}

    landmarker = _get_landmarker(cfg)
    cluster_frame_sets = [set(c) for c in bbox_clusters]
    mapping: dict[str, int] = {}

    debug_overlay = getattr(getattr(cfg, "detect", object()), "debug_overlay", False)
    debug_frames: list[tuple[int, np.ndarray, BBox, Optional[float], int, str]] = []

    for speaker in unique_speakers:
        window = _speaker_primary_window(diar_segments, speaker)
        if window is None:
            continue
        start_frame = int(window[0] * fps)
        end_frame = int(window[1] * fps)
        if end_frame <= start_frame:
            continue

        scores: dict[int, float] = {}
        for cluster_idx, frames in enumerate(cluster_frame_sets):
            sample = _cluster_frames_in_window(frames, start_frame, end_frame)
            collect: list = [] if debug_overlay else None
            scores[cluster_idx] = _mouth_variance_for_cluster(
                video_path, sample, per_frame_bboxes, landmarker,
                debug_collect=collect,
            )
            if collect:
                for fi, fr, bb, m in collect:
                    debug_frames.append((fi, fr, bb, m, cluster_idx, speaker))

        if not scores or max(scores.values()) == 0.0:
            log.debug(f"Speaker {speaker}: mouth-motion signal unavailable; falling back to presence")
            densities = {
                i: len([f for f in frames if start_frame <= f < end_frame])
                for i, frames in enumerate(cluster_frame_sets)
            }
            best = max(densities, key=lambda k: densities[k])
        else:
            best = max(scores, key=lambda k: scores[k])
        mapping[speaker] = best
        log.info(
            f"Speaker {speaker} → cluster {best} "
            f"(mouth-variance scores: {dict((i, round(v, 5)) for i, v in scores.items())})"
        )

    if debug_overlay and debug_frames:
        debug_frames.sort(key=lambda t: t[0])
        debug_path = video_path.parent / "debug_mouth.mp4"
        H, W = debug_frames[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(debug_path), fourcc, min(fps, 10.0), (W, H))
        if writer.isOpened():
            for fi, fr, bb, m, cidx, spk in debug_frames:
                annotated = _draw_mouth_debug_frame(fr, bb, m, fi, fps, cidx, spk)
                writer.write(annotated)
            writer.release()
            log.info(f"Mouth debug video written: {debug_path}")

    # If two speakers landed on the same cluster, re-assign the second by density.
    seen: dict[int, str] = {}
    for speaker, cluster_idx in list(mapping.items()):
        if cluster_idx in seen:
            window = _speaker_primary_window(diar_segments, speaker)
            start_frame = int(window[0] * fps)
            end_frame = int(window[1] * fps)
            alt_scores = {
                i: len([f for f in bbox_clusters[i] if start_frame <= f < end_frame])
                for i in range(len(bbox_clusters))
                if i != cluster_idx
            }
            if alt_scores:
                alt = max(alt_scores, key=lambda k: alt_scores[k])
                log.warning(
                    f"Speakers {seen[cluster_idx]} and {speaker} both linked "
                    f"to cluster {cluster_idx}; re-assigning {speaker} → {alt}"
                )
                mapping[speaker] = alt
                cluster_idx = alt
        seen[cluster_idx] = speaker

    return mapping


def _make_bbox_source(per_frame_bboxes: list[Optional[BBox]], member_frames: set[int]):
    def bbox_at(frame_idx: int) -> Optional[BBox]:
        if frame_idx in member_frames and 0 <= frame_idx < len(per_frame_bboxes):
            return per_frame_bboxes[frame_idx]
        return None
    return bbox_at


def _make_center_source(per_frame_bboxes: list[Optional[BBox]]):
    def bbox_at(frame_idx: int) -> Optional[BBox]:
        if 0 <= frame_idx < len(per_frame_bboxes):
            return per_frame_bboxes[frame_idx]
        return None
    return bbox_at


def link_timeline(
    diar_segments: list[DiarSegment],
    bbox_clusters: list[list[int]],
    per_frame_bboxes: list[Optional[BBox]],
    fps: float,
    clip_duration: float,
    video_path: Path,
    cfg: Optional[SimpleNamespace] = None,
) -> Timeline:
    """Turn diarization + bbox clusters into a speaker-following Timeline."""
    if not diar_segments:
        return []
    if cfg is None:
        cfg = SimpleNamespace(paths=SimpleNamespace(cache_dir=".cache"))

    mapping = _link_speakers_to_clusters(
        diar_segments, bbox_clusters, per_frame_bboxes, fps, video_path, cfg,
    )
    if not mapping:
        log.warning("No speaker→cluster mapping; falling back to all-bbox source")
        return [TimelineSegment(
            start=0.0, end=clip_duration, label="FALLBACK",
            bbox_at=_make_center_source(per_frame_bboxes),
        )]

    cluster_frame_sets = [set(c) for c in bbox_clusters]

    timeline: Timeline = []
    for seg in diar_segments:
        cluster_idx = mapping.get(seg.speaker_id)
        if cluster_idx is None:
            continue
        timeline.append(TimelineSegment(
            start=max(0.0, seg.start),
            end=min(clip_duration, seg.end),
            label=seg.speaker_id,
            bbox_at=_make_bbox_source(per_frame_bboxes, cluster_frame_sets[cluster_idx]),
        ))

    # Collapse adjacent same-cluster segments to avoid useless cuts
    merged: Timeline = []
    for seg in timeline:
        if merged and merged[-1].label == seg.label and seg.start - merged[-1].end < 0.15:
            prev = merged[-1]
            merged[-1] = TimelineSegment(
                start=prev.start, end=seg.end,
                label=prev.label, bbox_at=prev.bbox_at,
            )
        else:
            merged.append(seg)

    log.info(f"Built diarization-driven timeline with {len(merged)} segments")
    return merged
