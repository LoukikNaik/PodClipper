#!/usr/bin/env python3
"""Agentic Video Editor CLI."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from podclipper.config import load_config, load_default_config
from podclipper.logging_util import setup_logging


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader. Shell env vars take precedence; supports `export`
    prefix and paired quotes."""
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
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def build_parser(show_advanced: bool = False) -> argparse.ArgumentParser:
    def advanced(text: str) -> str:
        return text if show_advanced else argparse.SUPPRESS

    p = argparse.ArgumentParser(
        prog="podclipper",
        description="Turn a podcast into vertical reels with captions. First time? Run: podclipper init",
        epilog="Local podclipper.yaml is loaded automatically. More options: podclipper --help-all",
    )
    p.add_argument("input", type=Path, help="Path to input video file")
    p.add_argument(
        "--mode", choices=["reels", "trailer"], default="reels",
        help=advanced(
            "reels (default): produce N independent 9:16 reels covering "
            "different topics. "
            "trailer: produce ONE short trailer that splices 4-5 quotable "
            "sentences from across the episode together with black-frame "
            "transitions, like a movie trailer."
        ),
    )
    p.add_argument(
        "-c", "--config", type=Path, default=None,
        help="Config file (default: ./podclipper.yaml, then packaged defaults)",
    )
    p.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Where to save reels (default: outputs)",
    )
    p.add_argument(
        "--language", default=None,
        help=advanced("Set source speech language (e.g. 'en', 'hi'). Default: auto-detect"),
    )
    p.add_argument(
        "--whisper-engine", choices=["mlx", "faster"], default=None,
        help=advanced("Transcription engine: faster (default, CPU/CUDA) or mlx (Apple-Silicon GPU)"),
    )
    p.add_argument(
        "--clips", dest="max_clips", type=int, default=None,
        help="Cap the number of reels produced (after LLM analysis)",
    )
    p.add_argument(
        "--limit-minutes", type=float, default=None,
        help=advanced(
            "Only consider the first N minutes of the source for clip "
            "selection (skips transcribing the rest). Overrides "
            "cfg.analyze.source_limit_seconds."
        ),
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help=advanced("Ignore cached intermediate artifacts and rebuild from scratch"),
    )
    p.add_argument(
        "--captions", dest="subtitle_style", choices=["classic", "pop", "off"], default=None,
        help="Caption style: classic, pop, or off (default: classic)",
    )
    p.add_argument(
        "--intro-zoom", action="store_true",
        help=advanced("Start each reel zoomed in, then pull out over the first ~0.7s"),
    )
    p.add_argument(
        "--music", action="store_true",
        help=advanced("Add background music matched from the curated library"),
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level logging",
    )
    p.add_argument("--type", choices=["podcast", "comedy"], default=None, help=advanced("Content type (default: podcast)"))
    p.add_argument("--debug", action="store_true", help=advanced("Write detection and crop debug overlays"))
    p.add_argument("--select-only", action="store_true", help="Save clip selection for review and stop before clip processing (reels mode)")
    p.add_argument("--scribe", action="store_true", help="Experimental: transcribe once with Scribe via OpenRouter and reuse its words for captions")
    p.add_argument("--selection", type=Path, help="Render clips from a saved selection.json without selecting again (reels mode)")
    return p


def _config_path(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    local = Path("podclipper.yaml")
    return local if local.exists() else None


def _load_runtime_config(config_path: Path | None):
    """Apply shell/.env connection settings over the selected YAML config."""
    _load_dotenv()
    if config_path is not None:
        _load_dotenv(config_path.resolve().parent / ".env")
    cfg = load_config(config_path) if config_path else load_default_config()
    for name, field in (("LLM_ENDPOINT", "endpoint"), ("LLM_MODEL", "model")):
        value = os.environ.get(name, "").strip()
        if value:
            setattr(cfg.llm, field, value)
    return cfg


def _init_project() -> int:
    """Create a small starter config and key file without replacing user files."""
    import yaml

    config_path = Path("podclipper.yaml")
    try:
        if config_path.exists():
            cfg = load_config(config_path)
            print("Keeping existing podclipper.yaml")
        else:
            cfg = load_default_config()
            starter = {
                "llm": {key: getattr(cfg.llm, key) for key in ("endpoint", "model")},
                "content": {"clips": cfg.content.clips},
                "output": {"captions": cfg.output.captions},
            }
            with config_path.open("x") as handle:
                handle.write(yaml.safe_dump(starter, sort_keys=False))
            print("Created podclipper.yaml")
        key = "LLM_API_KEY"
        env_path = Path(".env")
        if env_path.exists():
            print("Keeping existing .env")
        else:
            # Restrict access before writing, including on machines with a permissive umask.
            fd = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as handle:
                handle.write(
                    f"# One API key for your chosen LLM host. Keep this file private.\n{key}=\n"
                    "\n# Optional: override the endpoint and model from podclipper.yaml.\n"
                    "# LLM_ENDPOINT=https://your-host/v1\n# LLM_MODEL=your-model\n"
                )
            print("Created .env")
        print(f"\nSet {key} in .env (or your shell), then run:")
        print('  podclipper "path/to/podcast.mp4"')
        return 0
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: could not initialize project: {exc}", file=sys.stderr)
        return 2


def _check_setup(cfg) -> None:
    """Fail before transcription when required tools or credentials are missing."""
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise ValueError(
            f"Missing {', '.join(missing)}. Install FFmpeg and put it on PATH "
            "(macOS: brew install ffmpeg; Ubuntu/Debian: sudo apt install ffmpeg)."
        )
    endpoint = urlparse(cfg.llm.endpoint)
    if endpoint.scheme not in ("http", "https") or not endpoint.netloc:
        raise ValueError("Set an http:// or https:// LLM_ENDPOINT in .env, or llm.endpoint in podclipper.yaml")
    if not cfg.llm.model.strip():
        raise ValueError("Set LLM_MODEL in .env, or llm.model in podclipper.yaml")
    key = "LLM_API_KEY"
    if not os.environ.get(key, "").strip():
        raise ValueError(f"Missing {key}. Add {key}=your-api-key to .env or export it in your shell.")
    if getattr(getattr(cfg.transcribe, "scribe", None), "enabled", False):
        from .scribe import validate_connection
        validate_connection(cfg)


def apply_cli_overrides(cfg, args) -> None:
    if getattr(args, "scribe", False):
        cfg.transcribe.scribe.enabled = True
    if args.output_dir is not None:
        cfg.paths.output_dir = str(args.output_dir)
    if args.language is not None:
        cfg.transcribe.language = args.language
    if args.whisper_engine is not None:
        cfg.transcribe.engine = args.whisper_engine
    if args.max_clips is not None:
        cfg.analyze.target_clips = args.max_clips
    if args.limit_minutes is not None:
        cfg.analyze.source_limit_seconds = args.limit_minutes * 60.0
    if args.debug:
        cfg.crop.debug_overlay = True
        cfg.detect.debug_overlay = True
    if args.subtitle_style is not None:
        cfg.subtitles.style = args.subtitle_style
    if args.intro_zoom:
        if not hasattr(cfg.crop, "intro_zoom"):
            cfg.crop.intro_zoom = SimpleNamespace()
        cfg.crop.intro_zoom.enabled = True
    if args.music:
        if not hasattr(cfg, "music"):
            cfg.music = SimpleNamespace()
        cfg.music.enabled = True
    if args.type is not None:
        cfg.crop.comedy_mode = args.type == "comedy"
    if args.verbose:
        cfg.logging.level = "DEBUG"


def main(argv: list[str] | None = None) -> int:
    import yaml

    raw = sys.argv[1:] if argv is None else argv
    if raw and raw[0] == "init":
        argparse.ArgumentParser(prog="podclipper init", description="Create starter config and .env without overwriting existing files.").parse_args(raw[1:])
        return _init_project()
    if raw and raw[0] == "config":
        from podclipper.config import ns_to_dict
        parser = argparse.ArgumentParser(prog="podclipper config")
        parser.add_argument("--resolved", action="store_true", required=True)
        parser.add_argument("-c", "--config", type=Path)
        args = parser.parse_args(raw[1:])
        try:
            config_path = _config_path(args.config)
            cfg = _load_runtime_config(config_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            print(f"error: could not load config: {exc}", file=sys.stderr)
            return 2
        print(yaml.safe_dump(ns_to_dict(cfg), sort_keys=False), end="")
        return 0
    show_advanced = "--help-all" in raw
    args = build_parser(show_advanced=show_advanced).parse_args(
        ["--help" if arg == "--help-all" else arg for arg in raw]
    )

    if args.select_only and args.mode != "reels":
        print("error: --select-only requires --mode reels", file=sys.stderr)
        return 2
    if args.selection is not None and (args.mode != "reels" or args.select_only):
        print("error: --selection requires reels mode without --select-only", file=sys.stderr)
        return 2
    if args.selection is not None and not args.selection.is_file():
        print(f"error: saved selection not found: {args.selection}", file=sys.stderr)
        return 2

    if not args.input.is_file():
        print(f"error: input video not found: {args.input}", file=sys.stderr)
        return 2

    config_path = _config_path(args.config)
    if config_path is not None and not config_path.is_file():
        print(f"error: config file not found: {config_path}", file=sys.stderr)
        return 2
    try:
        cfg = _load_runtime_config(config_path)
        apply_cli_overrides(cfg, args)
        if cfg.transcribe.scribe.enabled and args.mode != "reels":
            raise ValueError("--scribe is supported only in reels mode")
        if cfg.analyze.target_clips < 1:
            raise ValueError("--clips must be at least 1")
        if args.limit_minutes is not None and args.limit_minutes <= 0:
            raise ValueError("--limit-minutes must be greater than zero")
        _check_setup(cfg)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    log = setup_logging(cfg.logging.level)
    log.info(f"Input: {args.input}")
    log.info(f"Output dir: {cfg.paths.output_dir}")
    log.info(f"LLM model: {cfg.llm.model}")
    log.info(f"Config: {config_path if config_path else 'packaged defaults'}")

    try:
        if args.mode == "trailer":
            from podclipper.pipeline import run_trailer_pipeline
            log.info("Mode: trailer (splice quotable sentences into one reel)")
            run_trailer_pipeline(
                input_path=args.input,
                cfg=cfg,
                use_cache=not args.no_cache,
            )
        else:
            from podclipper.pipeline import run_pipeline
            log.info("Mode: reels (independent 9:16 clips)")
            produced = run_pipeline(
                input_path=args.input,
                cfg=cfg,
                use_cache=not args.no_cache,
                select_only=args.select_only,
                selection_path=args.selection,
            )
            if not produced:
                print("No reels were exported. See the log for rejected clips or processing failures.", file=sys.stderr)
                return 1
            print("\nSelection saved:" if args.select_only else f"\nSaved {len(produced)} reel(s):")
            for path in produced:
                print(f"  {path}")
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001
        if args.verbose:
            log.exception(f"Pipeline failed: {exc}")
        else:
            log.error(f"Pipeline failed: {exc}. Use --verbose for details.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
