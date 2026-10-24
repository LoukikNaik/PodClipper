"""Isolated pipeline process; publish only finished reels to the worker."""

import json
import logging
from pathlib import Path
import sys

from .store import Settings
from .results import finished_reel_path


def main():
    from podclipper.config import load_default_config
    from podclipper.logging_util import setup_logging
    from podclipper.pipeline import run_pipeline
    from podclipper.render_cache import write_json

    settings = Settings.load()
    directory = Path(sys.argv[1]).resolve()
    if directory.parent != settings.data / "jobs":
        raise ValueError("Invalid processing directory")
    job = json.loads((directory / "request.json").read_text())
    cfg = load_default_config()
    import os
    cfg.llm.endpoint = os.environ.get("LLM_ENDPOINT", cfg.llm.endpoint)
    cfg.llm.model = os.environ.get("LLM_MODEL", cfg.llm.model)
    cfg.paths.cache_dir = str(directory / ".cache")
    cfg.paths.output_dir = str(directory / "reels")
    cfg.analyze.target_clips = job["clips"]
    cfg.transcribe.scribe.enabled = os.environ.get("WEB_SCRIBE", "true").lower() == "true"
    cfg.transcribe.scribe.tag_audio_events = True
    cfg.transcribe.scribe.timeout_seconds = 300
    cfg.subtitles.ffmpeg_preset = "veryfast"
    cfg.evaluate.auto_skip = False
    setup_logging("INFO")
    status_path = directory / "progress.json"
    try:
        prior = json.loads(status_path.read_text())
    except (OSError, ValueError):
        prior = {}
    completed = [p for p in prior.get("completed", []) if Path(p).is_file()]
    status = {"stage": "transcribing", "completed": completed, "done": False}
    selection = directory / "selection.json"
    if not selection.exists():
        candidates = list((directory / ".cache").glob("*/analyze/04_final.json"))
        if len(candidates) == 1:
            selected = json.loads(candidates[0].read_text()).get("final_clips", [])
            if selected:
                write_json(selection, {"source": str(directory / "source.mp4"), "clips": selected})
    pending = None
    if selection.exists():
        selected = json.loads(selection.read_text())
        status["selected_count"] = len(selected["clips"])
        bounds = []
        for value in completed:
            fields = {}
            for line in Path(value).with_suffix(".txt").read_text().splitlines():
                if line.startswith(("source_start: ", "source_end: ")):
                    key, number = line.split(": ", 1)
                    fields[key] = float(number)
            if len(fields) == 2:
                bounds.append((fields["source_start"], fields["source_end"]))
        remaining = [c for c in selected["clips"] if not any(
            abs(c["start"] - a) < .05 and abs(c["end"] - b) < .05 for a, b in bounds)]
        if not remaining:
            status.update(done=True, stage="completed", produced=completed)
            write_json(status_path, status)
            return 0
        pending = directory / "selection-pending.json"
        write_json(pending, {**selected, "clips": remaining})
    write_json(status_path, status)

    class Progress(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if message.startswith(("Full Scribe transcript:", "Cached first-pass transcript", "Reusing cached first-pass transcript")):
                status["stage"] = "selecting"
            elif record.name == "ave.pipeline" and message.startswith("Output directory:"):
                status["stage"] = "exporting"
                if not selection.exists():
                    candidates = list((directory / ".cache").glob("*/analyze/04_final.json"))
                    if len(candidates) == 1:
                        selected = json.loads(candidates[0].read_text()).get("final_clips", [])
                        status["selected_count"] = len(selected)
                        write_json(selection, {"source": str(directory / "source.mp4"), "clips": selected})
            elif record.name == "ave.pipeline":
                path = finished_reel_path(message, directory)
                if path is not None and str(path) not in status["completed"]:
                    status["completed"].append(str(path))
            write_json(status_path, status)

    handler = Progress()
    logging.getLogger("ave").addHandler(handler)
    try:
        produced = run_pipeline(directory / "source.mp4", cfg, use_cache=True, selection_path=pending)
        status.update(done=True, stage="completed", produced=[str(p.resolve()) for p in produced])
        write_json(status_path, status)
        return 0 if produced else 1
    except Exception:
        logging.getLogger("ave.web").exception("Processing failed")
        status.update(done=True, stage="failed")
        write_json(status_path, status)
        return 1
    finally:
        logging.getLogger("ave").removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
