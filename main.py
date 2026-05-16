#!/usr/bin/env python3
"""Agentic Video Editor — CLI entry point.

Usage:
    python main.py INPUT_VIDEO [options]

Example:
    python main.py talk.mp4 --output-dir outputs --language en
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader — populates os.environ from `KEY=value` lines.

    Shell env vars take precedence, so pre-set values are never overwritten.
    Accepts optional surrounding quotes and `export KEY=value` prefix.
    Comments (`#`) and blank lines are skipped.
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip paired quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


# Load `.env` BEFORE importing anything that reads env vars (e.g. src.llm).
_load_dotenv()

from src.config import load_config  # noqa: E402
from src.logging_util import setup_logging  # noqa: E402
from src.pipeline import run_pipeline, run_trailer_pipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agentic-video-editor",
        description="Turn a long video into vertical 9:16 reels with smart cropping and subtitles.",
    )
    p.add_argument("input", type=Path, help="Path to input video file")
    p.add_argument(
        "--mode", choices=["reels", "trailer"], default="reels",
        help=(
            "reels (default): produce N independent 9:16 reels covering "
            "different topics. "
            "trailer: produce ONE short trailer that splices 4-5 quotable "
            "sentences from across the episode together with black-frame "
            "transitions, like a movie trailer."
        ),
    )
    p.add_argument(
        "-c", "--config", type=Path, default=Path("config/default.yaml"),
        help="Path to YAML config (default: config/default.yaml)",
    )
    p.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Where to write final reels (overrides config.paths.output_dir)",
    )
    p.add_argument(
        "--language", default=None,
        help="Force Whisper language (e.g. 'en', 'hi'). Default: auto-detect",
    )
    p.add_argument(
        "--llm-provider", choices=["claude_cli", "anthropic_api"], default=None,
        help="Override config.llm.provider",
    )
    p.add_argument(
        "--max-clips", type=int, default=None,
        help="Cap the number of reels produced (after LLM analysis)",
    )
    p.add_argument(
        "--debug-crop", action="store_true",
        help="Render auxiliary debug video with bbox + crop overlay",
    )
    p.add_argument(
        "--debug-detect", action="store_true",
        help=(
            "Write debug_detect.mp4 and debug_mouth.mp4 per clip — "
            "shows front/back face labels (green=front, orange=back), "
            "selected primary (thick border), face bbox (cyan), "
            "and mouth-openness boxes (green=talking, blue=silent)"
        ),
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Ignore cached intermediate artifacts and rebuild from scratch",
    )
    p.add_argument(
        "--keep-cache", action="store_true",
        help="Keep cache dir after completion (default: preserved for resume)",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level logging",
    )
    return p


def apply_cli_overrides(cfg, args) -> None:
    """Mutate cfg in place with CLI flag overrides."""
    if args.output_dir is not None:
        cfg.paths.output_dir = str(args.output_dir)
    if args.language is not None:
        cfg.transcribe.language = args.language
    if args.llm_provider is not None:
        cfg.llm.provider = args.llm_provider
    if args.max_clips is not None:
        cfg.analyze.target_clips = args.max_clips
    if args.debug_crop:
        cfg.crop.debug_overlay = True
    if args.debug_detect:
        cfg.detect.debug_overlay = True
    if args.verbose:
        cfg.logging.level = "DEBUG"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.exists():
        print(f"error: input video not found: {args.input}", file=sys.stderr)
        return 2

    if not args.config.exists():
        print(f"error: config file not found: {args.config}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)

    log = setup_logging(cfg.logging.level)
    log.info(f"Input: {args.input}")
    log.info(f"Output dir: {cfg.paths.output_dir}")
    log.info(f"LLM provider: {cfg.llm.provider}")

    try:
        if args.mode == "trailer":
            log.info("Mode: trailer (splice quotable sentences into one reel)")
            run_trailer_pipeline(
                input_path=args.input,
                cfg=cfg,
                use_cache=not args.no_cache,
            )
        else:
            log.info("Mode: reels (independent 9:16 clips)")
            run_pipeline(
                input_path=args.input,
                cfg=cfg,
                use_cache=not args.no_cache,
            )
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001  — top-level boundary
        log.exception(f"Pipeline failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
