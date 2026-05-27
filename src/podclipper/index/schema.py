"""Video-index schema: the contract between the indexer and the editing agent.

Pure dataclasses + JSON/markdown (de)serialization. Importable from either
Python env (no model/IO deps) since both halves agree on this shape.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Scene:
    """One cut-to-cut unit of the video, enriched with visual + speech."""
    id: int
    start: float
    end: float
    duration: float
    visual: str
    on_screen_text: str
    mood: str
    transcript: str
    words: list[dict]
    has_speech: bool
    shot_type: str = ""
    transcript_clean: str = ""   # LLM-fixed + transliterated; what the agent reads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        return cls(
            id=int(d["id"]),
            start=float(d["start"]),
            end=float(d["end"]),
            duration=float(d["duration"]),
            visual=str(d.get("visual", "")),
            on_screen_text=str(d.get("on_screen_text", "")),
            mood=str(d.get("mood", "")),
            transcript=str(d.get("transcript", "")),
            words=list(d.get("words", [])),
            has_speech=bool(d.get("has_speech", bool(d.get("transcript", "")))),
            shot_type=str(d.get("shot_type", "")),
            transcript_clean=str(d.get("transcript_clean", "")),
        )


@dataclass
class VideoIndex:
    """The full timestamped index for one video — the agent reasons over this."""
    source: str
    duration_seconds: float
    vision_model: str
    transcript_model: str
    scenes: list[Scene] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "duration_seconds": round(self.duration_seconds, 2),
            "scene_count": len(self.scenes),
            "vision_model": self.vision_model,
            "transcript_model": self.transcript_model,
            "scenes": [s.to_dict() for s in self.scenes],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "VideoIndex":
        return cls(
            source=str(d["source"]),
            duration_seconds=float(d["duration_seconds"]),
            vision_model=str(d.get("vision_model", "")),
            transcript_model=str(d.get("transcript_model", "")),
            scenes=[Scene.from_dict(s) for s in d.get("scenes", [])],
        )

    @classmethod
    def from_json(cls, payload: str) -> "VideoIndex":
        return cls.from_dict(json.loads(payload))

    def to_markdown(self) -> str:
        lines = [
            f"# Video index — {self.source}",
            f"duration: {self.duration_seconds:.0f}s · {len(self.scenes)} scenes",
            f"vision: {self.vision_model} · transcript: {self.transcript_model}",
            "",
        ]
        for sc in self.scenes:
            lines.append(
                f"## Scene {sc.id:03d}  [{_clock(sc.start)}–{_clock(sc.end)}]  ({sc.duration:.1f}s)"
            )
            if sc.mood:
                lines.append(f"*mood: {sc.mood}*")
            if sc.shot_type:
                lines.append(f"*shot: {sc.shot_type}*")
            if sc.visual:
                lines.append(f"**visual:** {sc.visual}")
            if sc.on_screen_text:
                lines.append(f"**on-screen:** {sc.on_screen_text}")
            lines.append(f"**said:** {sc.transcript_clean or sc.transcript or '—'}")
            lines.append("")
        return "\n".join(lines)


def _clock(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"
