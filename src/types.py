"""Shared dataclasses passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------- Stage 1: Ingest ----------

@dataclass
class VideoMeta:
    path: Path
    duration: float         # seconds
    width: int
    height: int
    fps: float
    has_audio: bool
    codec: str


# ---------- Stage 3: Transcribe ----------

@dataclass
class Word:
    start: float            # seconds (clip-relative for second pass, video-relative for first)
    end: float
    text: str
    confidence: float = 1.0


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    segments: list[TranscriptSegment]

    @property
    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)


# ---------- Stage 4: Analyze ----------

@dataclass
class Clip:
    start: float            # video-relative seconds
    end: float
    title: str
    reason: str             # LLM's justification (why this is reel-worthy)
    hook_score: float = 0.0 # LLM confidence / ranking

    @property
    def duration(self) -> float:
        return self.end - self.start


# ---------- Stage 5a: Person detection ----------

@dataclass
class BBox:
    x: float                # left
    y: float                # top
    w: float
    h: float
    confidence: float = 1.0

    @property
    def x_center(self) -> float:
        return self.x + self.w / 2

    @property
    def y_center(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> float:
        return self.w * self.h

    def iou(self, other: "BBox") -> float:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


# ---------- Stage 5b: Timeline ----------

@dataclass
class TimelineSegment:
    """One slice of a clip pointing at a bbox source for smart cropping.

    `bbox_at(frame_idx)` returns the bbox to center on for that frame,
    or None if no detection was available (crop falls back to last-known x).
    """
    start: float
    end: float
    label: str              # "SPEAKER_00", "PRIMARY", etc. — for debug only

    # frame_idx is clip-relative. Implementations typically close over a
    # per-frame bbox list filtered to a single persistent position.
    bbox_at: "callable"     # Callable[[int], Optional[BBox]]


Timeline = list[TimelineSegment]


# ---------- Stage 5c: Extracted clip artifacts ----------

@dataclass
class ClipArtifacts:
    """Everything produced for a single clip, passed through the per-clip pipeline."""
    clip: Clip
    segment_path: Path                       # raw extracted video
    per_frame_bboxes: list[Optional[BBox]]   # one entry per frame (None if no person)
    fps: float
    source_width: int
    source_height: int
    words: list[Word] = field(default_factory=list)   # from second-pass transcription
    diar_segments: Optional[list] = None              # post-MVP: pyannote output
    timeline: Optional[Timeline] = None
    cropped_path: Optional[Path] = None
    final_path: Optional[Path] = None
