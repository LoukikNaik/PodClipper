"""Test LLM tagging of SFX/meme audio.

Input: SFX names from Myinstants (test 03 output).
Output: structured JSON per SFX with: use_cases, mood, when_to_use_in_reel,
        intensity, related_memes.

Since Myinstants didn't return tags on detail pages, the LLM gets ONLY the
name. We're testing whether the name alone gives enough context for usable
tags.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from podclipper.main import _load_dotenv
_load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from podclipper.config import load_default_config
from podclipper.llm import build_provider


SFX_NAMES = [
    "Vine Boom",
    "Bruh",
    "Sad Violin",
    "Record Scratch",
    "Among Us role reveal sound",
    "Inception Horn",
    "Anime Wow",
    "FAHHHHHHHHHHHHHH",     # nonsense name, test fallback
    "Drum Roll",
    "Rizz Sound Effect",
]


SYSTEM_PROMPT = """You tag short viral/meme sound effects (SFX) so they can be auto-picked
to punctuate moments in short-form video reels.

Given a sound effect's filename or display name (often crude — these come from
Myinstants), return a single JSON object on one line with:

{
  "canonical_name": "cleaned-up name, e.g. 'vine_boom' for 'Vine Boom'",
  "category": "reaction | reveal | transition | punchline | suspense | celebration | sad_irony | meme_quote | other",
  "mood": ["one to three tags, e.g. dramatic, comedic, ominous"],
  "intensity": "low | med | high",
  "duration_estimate_s": "rough guess: short (<1s), medium (1-3s), long (>3s)",
  "when_to_use": ["specific moments in a reel where this lands, e.g. 'after a surprising reveal', 'over a slow zoom on a face', 'punctuating the punchline of a joke'"],
  "viral_origin": "where this meme came from if you know — TV show / Vine / TikTok / Inception / etc., or null",
  "recognizable": true | false   // would a typical Gen-Z viewer recognize this?
}

Return ONLY the JSON object, no prose, no code fence."""


def main() -> None:
    cfg = load_default_config()
    cfg.llm.provider = "litellm"
    cfg.llm.model = "openai/anthropic/claude-sonnet-4.5"
    cfg.llm.litellm.api_base = "https://api.tokenrouter.com/v1"
    cfg.llm.litellm.api_key_env = "TOKENROUTER_API_KEY"
    cfg.llm.litellm.num_retries = 0
    provider = build_provider(cfg.llm)

    for name in SFX_NAMES:
        print(f"\n--- {name} ---")
        t0 = time.time()
        try:
            raw = provider.complete(
                f"SFX name: {name}",
                system_prompt=SYSTEM_PROMPT,
                max_tokens=400,
            )
        except Exception as e:
            print(f"  LLM error: {e}")
            continue
        latency = time.time() - t0
        try:
            tags = json.loads(raw.strip().strip("`").lstrip("json\n"))
        except json.JSONDecodeError as e:
            print(f"  ✗ JSON parse: {e}; raw: {raw[:120]}")
            continue
        print(f"  ✓ {latency:.1f}s")
        print(f"    category:      {tags.get('category')}")
        print(f"    mood:          {tags.get('mood')}")
        print(f"    intensity:     {tags.get('intensity')}")
        print(f"    when_to_use:   {tags.get('when_to_use')}")
        print(f"    viral_origin:  {tags.get('viral_origin')}")
        print(f"    recognizable:  {tags.get('recognizable')}")


if __name__ == "__main__":
    main()
