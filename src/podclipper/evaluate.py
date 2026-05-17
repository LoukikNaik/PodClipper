"""Reel quality evaluation — technical metrics + LLM-as-judge."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

from .llm import LLMError, LLMProvider
from .types import Word

log = logging.getLogger("ave.evaluate")


# The evaluator prompt must NOT share context with the selector — the judge
# scores as if encountering the reel cold while scrolling.
_EVAL_SYSTEM = """You are a social media content evaluator. You will receive the title and transcript of a short-form video clip (Instagram Reel / TikTok / YouTube Short). Evaluate it as if you're seeing it for the first time while scrolling.

For EACH of the 6 criteria below:
1. Write 1-2 sentences of reasoning citing specific words/sentences from the transcript.
2. Then give a score from 1 to 5 using the rubric.

## Criteria

### 1. HOOK (first 2 seconds)
Does the opening line make a stranger stop scrolling?
  1 = generic filler ("So today...", "Let me tell you...")
  2 = mildly interesting but not urgent
  3 = clear topic but no tension or curiosity gap
  4 = strong hook: bold claim, specific number, provocative question, or named authority
  5 = irresistible — you MUST know what happens next

### 2. ARC (setup → payoff)
Does the clip tell a complete story or argument?
  1 = random fragment, no discernible point
  2 = has a topic but meanders with tangents
  3 = clear topic, weak or missing payoff
  4 = clean setup → development → payoff, minor filler
  5 = tight zero-filler arc, every sentence advances the argument

### 3. ENDING (last sentence)
Does the final line feel like a natural, satisfying close?
  1 = cuts off mid-sentence or mid-thought
  2 = complete sentence but topic unresolved
  3 = acceptable — finished a thought, but not the climax
  4 = lands on the payoff, viewer got the point
  5 = mic-drop — reframes everything or delivers a revelation

### 4. STANDALONE (context-free)
Could a stranger with zero context understand and enjoy this?
  1 = requires significant prior knowledge
  2 = mostly needs context
  3 = partially self-contained, some unclear references
  4 = 90%+ understandable to a stranger
  5 = fully self-contained, nothing missed

### 5. SHAREABILITY (engagement potential)
Would someone share this, comment, or save it?
  1 = forgettable
  2 = mildly interesting, not share-worthy
  3 = good but not remarkable
  4 = share-worthy — "my friend needs to hear this"
  5 = viral potential — provokes surprise, recognition, or debate

### 6. TITLE MATCH
Does the title represent the content and create appropriate curiosity?
  1 = misleading or generic
  2 = loosely related, oversells
  3 = accurate but boring
  4 = hooks the right audience, content delivers
  5 = perfect curiosity gap that the content resolves satisfyingly

## Output format

Return ONLY a JSON object, no prose before or after:
{
  "hook": {"reasoning": "...", "score": N},
  "arc": {"reasoning": "...", "score": N},
  "ending": {"reasoning": "...", "score": N},
  "standalone": {"reasoning": "...", "score": N},
  "shareability": {"reasoning": "...", "score": N},
  "title_match": {"reasoning": "...", "score": N},
  "overall": <average of all 6 scores, 1 decimal place>,
  "verdict": "publish" | "review" | "skip",
  "one_line_feedback": "<single actionable sentence for improvement>"
}

Verdict rules:
  overall >= 4.0 → "publish"
  3.0 <= overall < 4.0 → "review"
  overall < 3.0 → "skip"
"""


@dataclass
class TechMetrics:
    face_visibility: float
    crop_stability: float
    speaker_coverage: float
    duration_seconds: float
    words_per_second: float
    subtitle_coverage: float

    @property
    def duration_in_sweet_spot(self) -> bool:
        return 25 <= self.duration_seconds <= 60

    @property
    def wps_in_range(self) -> bool:
        return 1.5 <= self.words_per_second <= 3.5

    @property
    def score(self) -> float:
        return (
            self.face_visibility * 0.25
            + self.crop_stability * 0.20
            + self.speaker_coverage * 0.15
            + (1.0 if self.duration_in_sweet_spot else 0.5) * 0.15
            + (1.0 if self.wps_in_range else 0.5) * 0.15
            + self.subtitle_coverage * 0.10
        )


def compute_tech_metrics(
    words: list[Word],
    clip_duration: float,
    face_hits: int,
    total_frames: int,
    person_frames: int,
    x_centers: list[float],
    source_width: int,
) -> TechMetrics:
    """Objective technical quality metrics from pipeline data."""
    import numpy as np

    face_vis = face_hits / max(1, total_frames)
    speaker_cov = person_frames / max(1, total_frames)

    if len(x_centers) >= 2:
        std = float(np.std(x_centers))
        crop_stab = max(0.0, 1.0 - std / max(1.0, source_width * 0.15))
    else:
        crop_stab = 1.0

    wps = len(words) / max(0.1, clip_duration)

    if words:
        word_time = sum(max(0.0, w.end - w.start) for w in words)
        sub_cov = min(1.0, word_time / max(0.1, clip_duration))
    else:
        sub_cov = 0.0

    return TechMetrics(
        face_visibility=face_vis,
        crop_stability=crop_stab,
        speaker_coverage=speaker_cov,
        duration_seconds=clip_duration,
        words_per_second=wps,
        subtitle_coverage=sub_cov,
    )


@dataclass
class ContentScores:
    hook: int = 0
    hook_reasoning: str = ""
    arc: int = 0
    arc_reasoning: str = ""
    ending: int = 0
    ending_reasoning: str = ""
    standalone: int = 0
    standalone_reasoning: str = ""
    shareability: int = 0
    shareability_reasoning: str = ""
    title_match: int = 0
    title_match_reasoning: str = ""
    overall: float = 0.0
    verdict: str = "review"
    feedback: str = ""

    @property
    def score(self) -> float:
        dims = [self.hook, self.arc, self.ending, self.standalone, self.shareability, self.title_match]
        return sum(dims) / max(1, len(dims))


def _extract_json_object(text: str) -> dict:
    """Pull a JSON object from LLM output, tolerant of fences and prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1:
        raise ValueError(f"no JSON object in LLM output: {text[:200]!r}")
    return json.loads(text[first : last + 1])


def evaluate_content(
    title: str,
    words: list[Word],
    clip_duration: float,
    provider: LLMProvider,
    cfg: SimpleNamespace,
) -> ContentScores:
    """Send title + transcript to LLM; returns ContentScores. On any failure
    returns default "review" so the reel is never silently dropped."""
    eval_cfg = getattr(cfg, "evaluate", None)
    if eval_cfg is not None and not bool(getattr(eval_cfg, "enabled", True)):
        return ContentScores(verdict="review", feedback="evaluation disabled")

    transcript = " ".join(w.text.strip() for w in words if w.text.strip())
    if not transcript:
        return ContentScores(verdict="skip", feedback="empty transcript")

    first_sentence = transcript.split(".")[0].strip() + "." if "." in transcript else transcript[:80]
    last_sentence = transcript.rsplit(".", 2)[-2].strip() + "." if transcript.count(".") >= 2 else transcript[-80:]

    user_prompt = json.dumps({
        "title": title,
        "transcript": transcript,
        "duration_seconds": round(clip_duration, 1),
        "word_count": len(words),
        "first_sentence": first_sentence,
        "last_sentence": last_sentence,
    }, ensure_ascii=False, indent=2)

    try:
        response = provider.complete(
            user_prompt=user_prompt,
            system_prompt=_EVAL_SYSTEM,
            max_tokens=1500,
        )
    except LLMError as e:
        log.warning(f"evaluation LLM call failed ({e}); defaulting to 'review'")
        return ContentScores(verdict="review", feedback=f"LLM eval failed: {e}")

    try:
        data = _extract_json_object(response)
    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"evaluation returned unparseable JSON ({e}); defaulting to 'review'")
        return ContentScores(verdict="review", feedback=f"parse error: {e}")

    def _dim(key: str) -> tuple[int, str]:
        entry = data.get(key, {})
        if isinstance(entry, dict):
            return int(entry.get("score", 3)), str(entry.get("reasoning", ""))
        return int(entry) if isinstance(entry, (int, float)) else 3, ""

    h_s, h_r = _dim("hook")
    a_s, a_r = _dim("arc")
    e_s, e_r = _dim("ending")
    st_s, st_r = _dim("standalone")
    sh_s, sh_r = _dim("shareability")
    tm_s, tm_r = _dim("title_match")

    overall = float(data.get("overall", 0)) or round((h_s + a_s + e_s + st_s + sh_s + tm_s) / 6, 1)
    verdict = str(data.get("verdict", "review"))
    if verdict not in {"publish", "review", "skip"}:
        verdict = "publish" if overall >= 4.0 else ("review" if overall >= 3.0 else "skip")
    feedback = str(data.get("one_line_feedback", ""))

    return ContentScores(
        hook=h_s, hook_reasoning=h_r,
        arc=a_s, arc_reasoning=a_r,
        ending=e_s, ending_reasoning=e_r,
        standalone=st_s, standalone_reasoning=st_r,
        shareability=sh_s, shareability_reasoning=sh_r,
        title_match=tm_s, title_match_reasoning=tm_r,
        overall=overall, verdict=verdict, feedback=feedback,
    )


@dataclass
class ReelScorecard:
    tech: TechMetrics
    content: ContentScores
    final_score: float = 0.0
    verdict: str = "review"

    def __post_init__(self):
        # 30% tech, 70% content
        self.final_score = round(
            self.tech.score * 0.3 + (self.content.score / 5.0) * 0.7,
            2,
        )
        self.verdict = self.content.verdict

    def format(self) -> str:
        t = self.tech
        c = self.content
        lines = [
            "",
            "=== QUALITY SCORECARD ===",
            "",
            "Technical:",
            f"  Face visibility:     {t.face_visibility:.0%}",
            f"  Crop stability:      {t.crop_stability:.2f}",
            f"  Speaker coverage:    {t.speaker_coverage:.0%}",
            f"  Duration:            {t.duration_seconds:.0f}s {'✓' if t.duration_in_sweet_spot else '⚠ outside 25-60s'}",
            f"  Words/sec:           {t.words_per_second:.1f} {'✓' if t.wps_in_range else '⚠ outside 1.5-3.5'}",
            f"  Subtitle coverage:   {t.subtitle_coverage:.0%}",
            f"  Tech score:          {t.score:.2f}",
            "",
            "Content (LLM evaluation):",
            f"  Hook:                {c.hook}/5  \"{c.hook_reasoning[:80]}\"",
            f"  Arc:                 {c.arc}/5  \"{c.arc_reasoning[:80]}\"",
            f"  Ending:              {c.ending}/5  \"{c.ending_reasoning[:80]}\"",
            f"  Standalone:          {c.standalone}/5  \"{c.standalone_reasoning[:80]}\"",
            f"  Shareability:        {c.shareability}/5  \"{c.shareability_reasoning[:80]}\"",
            f"  Title match:         {c.title_match}/5  \"{c.title_match_reasoning[:80]}\"",
            f"  Content score:       {c.overall}/5",
            "",
            f"Overall:               {self.final_score:.2f}  →  {self.verdict.upper()}",
            f"Feedback:              \"{c.feedback}\"",
        ]
        return "\n".join(lines)

    def write_to(self, path) -> None:
        from pathlib import Path
        p = Path(path)
        with p.open("a") as f:
            f.write(self.format() + "\n")


def evaluate_reel(
    title: str,
    words: list[Word],
    clip_duration: float,
    face_hits: int,
    total_frames: int,
    person_frames: int,
    x_centers: list[float],
    source_width: int,
    provider: LLMProvider,
    cfg: SimpleNamespace,
) -> ReelScorecard:
    """Run full two-layer evaluation and return a scorecard."""
    tech = compute_tech_metrics(
        words, clip_duration, face_hits, total_frames,
        person_frames, x_centers, source_width,
    )
    content = evaluate_content(title, words, clip_duration, provider, cfg)

    scorecard = ReelScorecard(tech=tech, content=content)
    log.info(
        f"Eval: {scorecard.verdict.upper()} "
        f"(tech={tech.score:.2f}, content={content.overall}/5, "
        f"final={scorecard.final_score:.2f}) — {content.feedback}"
    )
    return scorecard


@dataclass
class TrailerScorecard:
    """Five-axis scorecard for spliced trailers (no hook/arc/ending — those
    don't apply to a sequence of picks). See prompts/trailer_evaluator.txt."""
    opener_hook: int = 3
    opener_hook_reasoning: str = ""
    thematic_coherence: int = 3
    thematic_coherence_reasoning: str = ""
    pacing: int = 3
    pacing_reasoning: str = ""
    closer_punch: int = 3
    closer_punch_reasoning: str = ""
    standalone_quality: int = 3
    standalone_quality_reasoning: str = ""
    overall: float = 3.0
    verdict: str = "review"
    feedback: str = ""

    def format(self) -> str:
        lines = [
            "",
            "=== TRAILER SCORECARD ===",
            "",
            f"  Opener hook:         {self.opener_hook}/5  \"{self.opener_hook_reasoning[:80]}\"",
            f"  Thematic coherence:  {self.thematic_coherence}/5  \"{self.thematic_coherence_reasoning[:80]}\"",
            f"  Pacing:              {self.pacing}/5  \"{self.pacing_reasoning[:80]}\"",
            f"  Closer punch:        {self.closer_punch}/5  \"{self.closer_punch_reasoning[:80]}\"",
            f"  Standalone quality:  {self.standalone_quality}/5  \"{self.standalone_quality_reasoning[:80]}\"",
            "",
            f"Overall:               {self.overall:.1f}/5  →  {self.verdict.upper()}",
            f"Feedback:              \"{self.feedback}\"",
        ]
        return "\n".join(lines)

    def write_to(self, path) -> None:
        from pathlib import Path
        with Path(path).open("a") as f:
            f.write(self.format() + "\n")


def evaluate_trailer(
    picks: list[dict],
    total_duration: float,
    provider: LLMProvider,
    cfg: SimpleNamespace,
) -> TrailerScorecard:
    """LLM-as-judge for trailer mode. Each pick dict must have at least
    "refined_sentence" or "sentence", "cut_start", "cut_end"."""
    from pathlib import Path
    eval_cfg = getattr(cfg, "evaluate", None)
    if eval_cfg is not None and not bool(getattr(eval_cfg, "enabled", True)):
        return TrailerScorecard(verdict="review", feedback="evaluation disabled")

    system_prompt = (
        Path(__file__).resolve().parent.parent.parent
        / "prompts" / "trailer_evaluator.txt"
    ).read_text(encoding="utf-8")

    picks_payload = []
    for i, p in enumerate(picks, 1):
        sentence = p.get("refined_sentence") or p.get("sentence") or ""
        dur = float(p.get("cut_end", 0)) - float(p.get("cut_start", 0))
        picks_payload.append({
            "index": i, "sentence": sentence.strip(), "duration_seconds": round(dur, 2),
        })

    user_prompt = json.dumps({
        "picks": picks_payload,
        "total_duration_seconds": round(total_duration, 2),
    }, ensure_ascii=False, indent=2)

    try:
        response = provider.complete(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=1500,
        )
    except LLMError as e:
        log.warning(f"trailer evaluation LLM call failed ({e}); defaulting to 'review'")
        return TrailerScorecard(verdict="review", feedback=f"LLM eval failed: {e}")

    try:
        data = _extract_json_object(response)
    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"trailer eval returned unparseable JSON ({e}); defaulting to 'review'")
        return TrailerScorecard(verdict="review", feedback=f"parse error: {e}")

    def _dim(key: str) -> tuple[int, str]:
        entry = data.get(key, {})
        if isinstance(entry, dict):
            return int(entry.get("score", 3)), str(entry.get("reasoning", ""))
        return int(entry) if isinstance(entry, (int, float)) else 3, ""

    o_s, o_r = _dim("opener_hook")
    tc_s, tc_r = _dim("thematic_coherence")
    p_s, p_r = _dim("pacing")
    cp_s, cp_r = _dim("closer_punch")
    sq_s, sq_r = _dim("standalone_quality")

    overall = float(data.get("overall", 0)) or round(
        (o_s + tc_s + p_s + cp_s + sq_s) / 5, 1
    )
    verdict = str(data.get("verdict", "review"))
    if verdict not in {"publish", "review", "skip"}:
        verdict = "publish" if overall >= 4.0 else ("review" if overall >= 3.0 else "skip")
    feedback = str(data.get("one_line_feedback", ""))

    sc = TrailerScorecard(
        opener_hook=o_s, opener_hook_reasoning=o_r,
        thematic_coherence=tc_s, thematic_coherence_reasoning=tc_r,
        pacing=p_s, pacing_reasoning=p_r,
        closer_punch=cp_s, closer_punch_reasoning=cp_r,
        standalone_quality=sq_s, standalone_quality_reasoning=sq_r,
        overall=overall, verdict=verdict, feedback=feedback,
    )
    log.info(
        f"Trailer eval: {sc.verdict.upper()} (overall={overall}/5) — {feedback}"
    )
    return sc
