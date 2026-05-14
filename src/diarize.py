"""Stage 5 (post-MVP): Speaker diarization + mouth-motion face linking.

Purpose: when the source video is a single-camera interview (or any edit
where the director didn't cut between speakers visually), our multi-shot
timeline falls back to one bbox cluster and we lose "follow the speaker"
cropping. Diarization + face linking fixes that:

  1. `diarize_clip(segment_path, cfg)` runs pyannote.audio on the clip's
     audio track. Returns a list of `DiarSegment{start, end, speaker_id}`.
     pyannote knows *when* each unique voice was speaking but has no idea
     where those speakers are in the frame.

  2. `link_timeline(diar_segments, bbox_clusters, per_frame_bboxes, fps,
     clip_duration, video_path, source_dims)` bridges the gap:
       a. For each unique speaker, find the longest contiguous window they
          were actively speaking.
       b. Within that window, for each persistent bbox cluster, crop the
          face region per frame and run MediaPipe FaceMesh to measure
          mouth-opening amplitude. Variance over the window is a clean
          "is this face talking" signal.
       c. Pick the cluster with highest mouth-opening variance as that
          speaker's visual position.
       d. Translate the full diarization timeline into TimelineSegments
          that target the correct bbox for each speaking interval.
       e. Apply min-dwell smoothing to avoid flicker.

Auth: pyannote's speaker-diarization-3.1 model is gated on HuggingFace.
Users must accept terms on https://huggingface.co/pyannote/speaker-diarization-3.1
and export their token to `HF_TOKEN` (or whatever is set in
`cfg.diarize.hf_token_env`).

Graceful degradation: any failure in this module causes `diarize_clip` to
return None and the pipeline falls back to the multi-shot/single-cluster
timeline.
"""

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


# ---------- Pyannote pipeline (cached) ----------

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
            # Prefer MPS / CUDA if available — diarization is fairly light but
            # GPU still helps.
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


# ---------- Entry point: diarize ----------

def diarize_clip(
    segment_path: Path,
    cfg: SimpleNamespace,
) -> Optional[list[DiarSegment]]:
    """Run pyannote on the clip audio; return DiarSegments or None on failure."""
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

    # pyannote 4.x returns a DiarizeOutput wrapper; 3.x returned the
    # Annotation directly. Handle both.
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


# ---------- MediaPipe mouth-opening analysis ----------

# Lip landmarks in MediaPipe FaceLandmarker — index 13 is the top of the upper
# lip, 14 is the bottom of the lower lip. The vertical distance between them,
# normalized by the face's vertical extent (10=forehead, 152=chin), is a
# compact "mouth openness" signal.
_UPPER_LIP_IDX = 13
_LOWER_LIP_IDX = 14
_FACE_TOP_IDX = 10
_FACE_BOTTOM_IDX = 152

# MediaPipe model-asset URL (provided by Google for the FaceLandmarker task).
_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

_landmarker = None
_landmarker_lock = threading.Lock()


def _ensure_face_model(cache_dir: Path) -> Path:
    """Download the FaceLandmarker .task model on first use; cache it."""
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
    """Return a mouth-openness ratio (mouth gap / face height) for the given
    face bbox, or None if landmarks weren't detected."""
    import mediapipe as mp

    H, W = frame_bgr.shape[:2]
    # Pad the bbox so the whole face is in frame.
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
    lm = result.face_landmarks[0]  # list of NormalizedLandmark

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
    """Pick up to `max_samples` frame indices that both belong to `cluster_frames`
    and fall inside [start_frame, end_frame). Evenly spaced."""
    in_window = [f for f in sorted(cluster_frames) if start_frame <= f < end_frame]
    if not in_window:
        return []
    if len(in_window) <= max_samples:
        return in_window
    step = len(in_window) / max_samples
    return [in_window[int(i * step)] for i in range(max_samples)]


def _sample_variance(values: list[float]) -> float:
    """Population variance of values (0.0 if fewer than 2 values)."""
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
    """Open the video and compute mouth-openness at each sampled frame, return variance.

    If `debug_collect` is a list, appends (frame_idx, frame_bgr, bbox, openness_or_None)
    tuples so the caller can render a debug overlay video.
    """
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
    """Draw a mouth-region box on the frame coloured by openness.

    Green  = mouth open / likely speaking  (openness > 0.03)
    Blue   = mouth closed / likely silent  (openness <= 0.03)
    Orange = landmarks not detected
    """
    annotated = frame.copy()
    H, W = annotated.shape[:2]

    # Person bbox outline
    px, py = int(face_bbox.x), int(face_bbox.y)
    pw, ph = int(face_bbox.w), int(face_bbox.h)
    cv2.rectangle(annotated, (px, py), (px + pw, py + ph), (180, 180, 180), 1)

    # Estimated mouth region: roughly bottom 30% of face, centred horizontally
    mouth_y = py + int(ph * 0.68)
    mouth_h = int(ph * 0.22)
    mouth_w = int(pw * 0.45)
    mouth_x = px + int((pw - mouth_w) / 2)
    mouth_x = max(0, min(W - mouth_w, mouth_x))
    mouth_y = max(0, min(H - mouth_h, mouth_y))

    if openness is None:
        color = (0, 165, 255)   # orange — no landmarks
        label = "no landmarks"
    elif openness > 0.03:
        color = (0, 220, 0)     # green — speaking
        label = f"SPEAK {openness:.3f}"
    else:
        color = (220, 60, 60)   # blue-ish — silent
        label = f"silent {openness:.3f}"

    cv2.rectangle(annotated, (mouth_x, mouth_y),
                  (mouth_x + mouth_w, mouth_y + mouth_h), color, 2)
    cv2.putText(annotated, label, (mouth_x, max(12, mouth_y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # HUD
    t = frame_idx / fps if fps > 0 else 0.0
    spk = f"  spk={speaker_id}" if speaker_id else ""
    cv2.putText(annotated, f"f{frame_idx}  t={t:.2f}s  cluster={cluster_idx}{spk}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


# ---------- Speaker → cluster linking ----------

def _speaker_primary_window(
    diar_segments: list[DiarSegment],
    speaker_id: str,
) -> Optional[tuple[float, float]]:
    """Return (start, end) of the longest contiguous speaking window for this speaker."""
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
    """Map each diarized speaker_id → index into bbox_clusters with the
    highest mouth-opening variance during that speaker's primary window."""
    unique_speakers = sorted({s.speaker_id for s in diar_segments})
    if not unique_speakers or not bbox_clusters:
        return {}

    landmarker = _get_landmarker(cfg)
    cluster_frame_sets = [set(c) for c in bbox_clusters]
    mapping: dict[str, int] = {}

    debug_overlay = getattr(getattr(cfg, "detect", object()), "debug_overlay", False)
    # (frame_idx, frame_bgr, bbox, openness, cluster_idx, speaker_id)
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

    # Write mouth debug video if requested (sorted by frame index)
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

    # Sanity: if two speakers landed on the same cluster, swap the second to
    # whichever other cluster has the next-best signal. This should be rare
    # for properly diarized multi-speaker clips.
    seen: dict[int, str] = {}
    for speaker, cluster_idx in list(mapping.items()):
        if cluster_idx in seen:
            # Re-pick: try the remaining clusters for this speaker
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


# ---------- Timeline builder ----------

def _make_bbox_source(per_frame_bboxes: list[Optional[BBox]], member_frames: set[int]):
    """Same helper as in timeline.py, duplicated here to avoid circular import."""
    def bbox_at(frame_idx: int) -> Optional[BBox]:
        if frame_idx in member_frames and 0 <= frame_idx < len(per_frame_bboxes):
            return per_frame_bboxes[frame_idx]
        return None
    return bbox_at


def _make_center_source(per_frame_bboxes: list[Optional[BBox]]):
    """Fallback: return any available bbox for the frame, else None."""
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
    """Turn diarization + bbox clusters into a speaker-following Timeline.

    Requires `video_path` because mouth-motion linking reads frames.
    `cfg` is needed by `_get_landmarker` to locate the model cache dir.
    """
    if not diar_segments:
        return []
    if cfg is None:
        # Lazy default so callers without access to cfg still work
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

    # Collapse adjacent segments with the same cluster to avoid useless cuts
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
