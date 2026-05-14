"""Stage 3b: LLM-based cleanup of second-pass whisper output.

Problems this solves:
  - Whisper sometimes mis-transcribes English words ("Agentec" → "Agentic",
    "Karmanyeva" → "Karmanye va"), which burns awkwardly into subtitles.
  - Non-English speech (Hindi Devanagari, Urdu, Arabic, ...) gets written in
    its native script — but our target reel audience typically wants Latin
    characters. We transliterate phonetically so viewers can read along.

Approach: send the full word list of a clip as an indexed JSON array to the
LLM, ask for the same structure back with corrected/romanized `w` values,
then align 1:1 with the original `Word` objects to preserve timestamps.

Failure modes:
  - LLM returns a different word count → we reject and keep originals.
  - LLM returns malformed JSON → reject, keep originals.
  - LLM times out → reject, keep originals.
Either way, karaoke keeps working; worst case subs look the same as before.
"""

from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace

from .llm import LLMError, LLMProvider
from .types import Word

log = logging.getLogger("ave.cleanup")


_SYSTEM_PROMPT = """You clean up automatically-transcribed subtitle text for short-form videos.

You will receive a JSON array of indexed word tokens from a transcription:
  [{"i": 0, "w": "Hello"}, {"i": 1, "w": "वर्ल्ड"}, ...]

Return the SAME array with corrected `w` values. Two transformations:

1. **Transliterate any non-Latin script** (Devanagari, Urdu, Arabic, Chinese, etc.)
   to the phonetic English / Latin alphabet. Keep it natural for a reader who
   doesn't know the source language. Prefer common romanizations (e.g.
   "नमस्ते" → "Namaste", "धन्यवाद" → "Dhanyavaad", "کتاب" → "kitaab"). For
   common Hindi loanwords that already have an English spelling, use it
   ("dharma", "karma", "guru", "yoga", "namaste").

2. **Fix obvious English mis-transcriptions**. Examples:
   - "Agentec" → "Agentic"
   - "Karmanyeva" → "Karmanye va" (when it's a Sanskrit compound that was joined)
   - "Shiva jee" → "Shivaji"
   - "Machina learning" → "Machine learning"
   Only correct clear errors; do NOT rewrite the speaker's words, normalize
   grammar, or add punctuation that wasn't already on the token.

Critical rules:
- Return EXACTLY the same number of tokens, in the same order, with the same `i`.
- Never merge or split tokens (no "is not" → "isn't", no "isn't" → "is not").
  If whisper gave you one token, return one token. If two, return two.
- Preserve case where sensible (proper nouns capitalized; otherwise match input).
- Preserve any trailing punctuation that's already part of the token
  (e.g. "Hello," stays "Hello,"; "world." stays "world.").
- Output ONLY the JSON array. No prose, no code fence, nothing else.
"""


class CleanupError(Exception):
    pass


def _extract_json_array(text: str) -> list:
    """Same tolerant extractor as analyze.py."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    first = text.find("[")
    last = text.rfind("]")
    if first == -1 or last == -1:
        raise CleanupError(f"no JSON array in LLM output: {text[:200]!r}")
    return json.loads(text[first : last + 1])


def cleanup_words(
    words: list[Word],
    provider: LLMProvider,
    cfg: SimpleNamespace,
) -> list[Word]:
    """Post-process words: transliterate + fix spelling via LLM.

    Returns a NEW list of Word objects with corrected text but original
    timestamps. If cleanup fails for any reason, returns `words` unchanged.
    """
    cleanup_cfg = getattr(cfg.transcribe, "cleanup", None)
    if cleanup_cfg is None or not bool(cleanup_cfg.enabled):
        return words
    if not words:
        return words

    # Build indexed payload; keep tokens in their raw form (with attached
    # punctuation / leading space as whisper emitted them).
    payload = [{"i": i, "w": w.text} for i, w in enumerate(words)]
    user_prompt = json.dumps(payload, ensure_ascii=False)

    try:
        response = provider.complete(
            user_prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=int(cleanup_cfg.max_tokens_per_call),
        )
    except LLMError as e:
        log.warning(f"cleanup LLM call failed ({e}); keeping raw words")
        return words

    try:
        fixed = _extract_json_array(response)
    except (json.JSONDecodeError, CleanupError) as e:
        log.warning(f"cleanup returned unparseable JSON ({e}); keeping raw words")
        return words

    if len(fixed) != len(words):
        log.warning(
            f"cleanup count mismatch: {len(fixed)} vs {len(words)}; keeping raw words"
        )
        return words

    out: list[Word] = []
    changes = 0
    for original, item in zip(words, fixed):
        if not isinstance(item, dict) or "w" not in item:
            log.warning(f"cleanup item malformed: {item!r}; keeping raw")
            return words
        new_text = str(item["w"])
        if new_text != original.text:
            changes += 1
        out.append(Word(
            start=original.start,
            end=original.end,
            text=new_text,
            confidence=original.confidence,
        ))
    log.info(f"Cleanup: {changes}/{len(words)} words rewritten")
    return out
