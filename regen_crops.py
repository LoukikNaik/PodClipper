#!/usr/bin/env python3
"""Re-render existing cached reels with the latest detect.py crop logic.

Skips LLM clip selection and diarization. Runs Whisper second-pass (cached
to `words.json` under each reel cache dir, so repeat runs are instant) and
re-burns karaoke subtitles + title overlay using the previous reel's title.

Usage:
    python regen_crops.py CACHE_DIR OUTPUT_DIR [SIDECAR_DIR]

  CACHE_DIR     e.g. .cache/RiHi4solYHE_part6-adb85d8f31
  OUTPUT_DIR    a fresh directory to write reels into
  SIDECAR_DIR   optional — directory containing the original reel_*.txt
                files; titles are parsed from them and used for the title
                overlay. Defaults to the matching outputs/<timestamp>/ if
                one can be inferred; otherwise titles are empty.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.crop import smart_crop_916
from src.detect import detect_humans_per_frame
from src.logging_util import setup_logging
from src.subtitles import burn_subtitles
from src.timeline import apply_min_dwell, build_speaker_timeline
from src.transcribe import transcribe_second_pass_cached


def _title_from_sidecar(path: Path) -> str:
    """Parse `title: ...` out of a reel_*.txt sidecar."""
    if not path.exists():
        return ""
    for line in path.read_text().splitlines():
        m = re.match(r"^title:\s*(.+)$", line)
        if m:
            return m.group(1).strip()
    return ""


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python regen_crops.py CACHE_DIR OUTPUT_DIR [SIDECAR_DIR]")
        return 2
    cache_dir = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    sidecar_dir = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else None
    if not cache_dir.is_dir():
        print(f"not found: {cache_dir}")
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(Path("config/default.yaml"))
    log = setup_logging("INFO")

    reel_dirs = sorted(p for p in cache_dir.iterdir() if p.is_dir() and p.name.startswith("reel_"))
    if not reel_dirs:
        log.warning(f"No reel_* dirs found in {cache_dir}")
        return 1
    log.info(f"Found {len(reel_dirs)} cached reels; regenerating with fixed detect.py")
    if sidecar_dir:
        log.info(f"Reading titles from sidecars in {sidecar_dir}")

    failures: list[str] = []
    for i, reel_dir in enumerate(reel_dirs, 1):
        segment = reel_dir / "segment.mp4"
        if not segment.exists():
            log.warning(f"[{i}/{len(reel_dirs)}] {reel_dir.name}: no segment.mp4 — skip")
            failures.append(reel_dir.name)
            continue

        log.info(f"[{i}/{len(reel_dirs)}] {reel_dir.name}")
        try:
            per_frame, fps, w, h = detect_humans_per_frame(segment, cfg)
            duration = len(per_frame) / fps if fps else 0.0
            timeline = build_speaker_timeline(
                per_frame_bboxes=per_frame,
                clip_duration=duration,
                fps=fps,
                source_width=w,
                source_height=h,
                diar_segments=None,
                cfg=cfg,
                video_path=segment,
            )
            timeline = apply_min_dwell(timeline, cfg.crop.min_segment_dwell_seconds)

            cropped_path = reel_dir / "cropped_v2.mp4"
            smart_crop_916(segment, timeline, cropped_path, cfg)

            words = transcribe_second_pass_cached(segment, reel_dir / "words.json", cfg)

            title = _title_from_sidecar(sidecar_dir / f"{reel_dir.name}.txt") if sidecar_dir else ""

            final_path = output_dir / f"{reel_dir.name}.mp4"
            burn_subtitles(cropped_path, words, final_path, cfg, title=title)
            log.info(f"  → {final_path}")
        except Exception as e:  # noqa: BLE001
            log.exception(f"  failed: {e}")
            failures.append(reel_dir.name)

    log.info(f"Done. {len(reel_dirs) - len(failures)}/{len(reel_dirs)} reels → {output_dir}")
    if failures:
        log.warning(f"Failures: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
