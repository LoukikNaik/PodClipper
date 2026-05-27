"""LLM transcript-cleanup layer for the index: fixes ASR mis-transcriptions
and transliterates non-Latin scripts to readable Latin, so the editing agent
reasons over clean text. Only the transcripts are sent to the LLM — never the
visuals. The LLM call is injected (`complete_fn`) for testability.

Raw `transcript` + word timings are left untouched (needed for subtitle
export); the cleaned text lands in `transcript_clean`.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from .schema import VideoIndex

CompleteFn = Callable[[str, str], str]  # (system_prompt, user_prompt) -> raw text

SYSTEM_PROMPT = """You clean up automatically-transcribed video subtitles.

You receive a JSON array of scene transcripts:
  [{"id": 1, "text": "आज परफॉरमेंस दूँगा"}, ...]

Return the SAME array with corrected `text`. Two jobs:

1. TRANSLITERATE any non-Latin script (Devanagari, Urdu, Arabic, ...) to natural
   phonetic Latin. Use common romanizations a reader who doesn't know the source
   language can follow (e.g. "आज" -> "aaj", "परफॉरमेंस" -> "performance").
   English loanwords already in the speech keep their English spelling.
2. FIX obvious ASR errors and drop pure repetition/noise artifacts. Do NOT
   rewrite the speaker's meaning, add content, or translate to English — only
   romanize and correct.

Rules:
- Keep the SAME ids. You may omit a scene only if its text is pure noise.
- Output ONLY the JSON array. No prose, no markdown fences."""


def speech_payload(index: VideoIndex) -> list[dict]:
    """The minimal payload sent to the LLM: id + raw transcript for scenes that
    actually have speech."""
    return [{"id": s.id, "text": s.transcript}
            for s in index.scenes if s.transcript.strip()]


def _extract_json_array(text: str) -> list:
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    first, last = s.find("["), s.rfind("]")
    if first == -1 or last == -1:
        raise ValueError(f"no JSON array in cleanup response: {text[:200]!r}")
    return json.loads(s[first:last + 1])


def apply_cleanup(index: VideoIndex, fixed_by_id: dict[int, str]) -> VideoIndex:
    """Write fixed text into each scene's transcript_clean by id. Scenes absent
    from the mapping keep transcript_clean == '' (raw transcript untouched)."""
    for s in index.scenes:
        if s.id in fixed_by_id:
            s.transcript_clean = str(fixed_by_id[s.id]).strip()
    return index


def clean_index(index: VideoIndex, complete_fn: CompleteFn, batch_size: int = 20) -> VideoIndex:
    """Send only the transcripts to the LLM (in batches of `batch_size` to stay
    under request timeouts), apply fixes to transcript_clean. Best-effort: a
    failed batch leaves those scenes' transcript_clean empty, the rest proceed."""
    payload = speech_payload(index)
    if not payload:
        return index

    fixed_by_id: dict[int, str] = {}
    for i in range(0, len(payload), max(1, batch_size)):
        batch = payload[i:i + batch_size]
        user_prompt = json.dumps(batch, ensure_ascii=False)
        try:
            items = _extract_json_array(complete_fn(SYSTEM_PROMPT, user_prompt))
        except Exception:  # noqa: BLE001 — best-effort: skip a failed/slow batch
            continue
        for item in items:
            if isinstance(item, dict) and "id" in item and "text" in item:
                try:
                    fixed_by_id[int(item["id"])] = item["text"]
                except (TypeError, ValueError):
                    continue
    return apply_cleanup(index, fixed_by_id)


# --------------------------------------------------------------------------- #
# Real LLM backend (Kimi via TokenRouter) + CLI. Runs in the litellm env.
# --------------------------------------------------------------------------- #

# Cleanup is mechanical (fix + transliterate), not reasoning — a fast cheap
# model is the right tool. Kimi stays the *agent* brain; Haiku does cleanup.
DEFAULT_MODEL = "openai/anthropic/claude-haiku-4.5"
DEFAULT_API_BASE = "https://api.tokenrouter.com/v1"
DEFAULT_KEY_ENV = "TOKENROUTER_API_KEY"


def tokenrouter_complete(
    model: str = DEFAULT_MODEL, api_base: str = DEFAULT_API_BASE,
    key_env: str = DEFAULT_KEY_ENV,
) -> CompleteFn:
    """Build a CompleteFn that routes to an OpenAI-compatible gateway via litellm."""
    import os
    import litellm

    def _complete(system_prompt: str, user_prompt: str) -> str:
        resp = litellm.completion(
            model=model, api_base=api_base, api_key=os.environ[key_env],
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            timeout=180,
        )
        return resp.choices[0].message.content or ""

    return _complete


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    from ..main import _load_dotenv
    from .schema import VideoIndex

    p = argparse.ArgumentParser(prog="podclipper.index.cleanup")
    p.add_argument("scenes_json", type=Path, help="path to an index scenes.json")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-base", default=DEFAULT_API_BASE)
    p.add_argument("--key-env", default=DEFAULT_KEY_ENV)
    p.add_argument("--batch-size", type=int, default=15)
    args = p.parse_args(argv)

    _load_dotenv()
    index = VideoIndex.from_json(args.scenes_json.read_text())
    n = len(speech_payload(index))
    print(f"[cleanup] {n} speech scenes → {args.model} (batch={args.batch_size})", flush=True)

    index = clean_index(
        index, tokenrouter_complete(args.model, args.api_base, args.key_env),
        batch_size=args.batch_size,
    )

    args.scenes_json.write_text(index.to_json())
    md = args.scenes_json.with_name(args.scenes_json.parent.name + ".md")
    md.write_text(index.to_markdown())
    cleaned = sum(1 for s in index.scenes if s.transcript_clean)
    print(f"[cleanup] wrote {cleaned} cleaned transcripts → {args.scenes_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
