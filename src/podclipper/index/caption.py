"""MLX-Gemma per-scene visual captioning. Frames are sampled within a scene
and passed as images (mlx_vlm's video= path silently drops frames for gemma4).
Runs in the MLX env."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

MODEL_ID = "mlx-community/gemma-4-e2b-it-4bit"
FRAMES_PER_SCENE = 4
MAX_TOKENS = 512

SCENE_PROMPT = """These images are frames sampled in order from ONE continuous shot of a video. Return ONE JSON object, nothing else:
{
  "visual": "1-2 sentences: who/what is on screen, the setting, the action, the format (live-action, animation, screen-recording, etc.)",
  "on_screen_text": "any visible text/captions/UI labels verbatim, or \\"\\" if none",
  "mood": "one or two words for the visual mood (e.g. tense, calm, energetic, somber)"
}
Output JSON only, no prose, no markdown fences."""


def sample_frames(src: Path, start: float, dur: float, n: int, dst_dir: Path) -> list[str]:
    """Extract up to `n` evenly-spaced JPEG frames from [start, start+dur).
    Per-frame failures (e.g. seeking near EOF) are skipped, not fatal."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    usable = max(0.0, dur - 0.2)
    paths: list[str] = []
    for i in range(n):
        t = start + (usable * (i + 0.5) / max(n, 1))
        out = dst_dir / f"f{i:02d}.jpg"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}",
                 "-i", str(src), "-vframes", "1", "-vf", "scale=768:-2", str(out)],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError:
            continue
        if out.exists() and out.stat().st_size > 0:
            paths.append(str(out))
    return paths


def parse_caption_json(text: str) -> dict:
    """Tolerant extraction of the scene-caption JSON object."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                return decoder.raw_decode(s[i:])[0]
            except json.JSONDecodeError:
                continue
    return {"visual": "", "on_screen_text": "", "mood": "", "_raw": text}


def load_captioner(model_id: str = MODEL_ID):
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    model, processor = load(model_id)
    config = load_config(model_id)
    return model, processor, config


def caption_scene(captioner, frames: list[str]) -> dict:
    """Return {visual, on_screen_text, mood} for a scene's frames."""
    if not frames:
        return {"visual": "", "on_screen_text": "", "mood": ""}
    model, processor, config = captioner
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    formatted = apply_chat_template(processor, config, SCENE_PROMPT, num_images=len(frames))
    out = generate(model, processor, formatted, image=frames,
                   max_tokens=MAX_TOKENS, verbose=False)
    text = out.text if hasattr(out, "text") else str(out)
    return parse_caption_json(text)
