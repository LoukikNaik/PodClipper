"""Shared dataclasses passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class VideoMeta:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    codec: str


@dataclass
class Word:
    start: float
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


@dataclass
class DiarSegment:
    """One speaker-active window from pyannote.audio."""
    start: float
    end: float
    speaker_id: str


@dataclass
class Clip:
    start: float
    end: float
    title: str
    reason: str
    hook_score: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class BBox:
    x: float
    y: float
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


@dataclass
class TimelineSegment:
    """One slice of a clip pointing at a per-frame bbox source for cropping."""
    start: float
    end: float
    label: str
    bbox_at: "callable"  # Callable[[int], Optional[BBox]]


Timeline = list[TimelineSegment]


@dataclass
class ClipArtifacts:
    """Everything produced for a single clip in the per-clip pipeline."""
    clip: Clip
    segment_path: Path
    per_frame_bboxes: list[Optional[BBox]]
    fps: float
    source_width: int
    source_height: int
    words: list[Word] = field(default_factory=list)
    diar_segments: Optional[list] = None
    timeline: Optional[Timeline] = None
    cropped_path: Optional[Path] = None
    final_path: Optional[Path] = None
