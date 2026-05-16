#!/usr/bin/env python3
"""Re-render existing cached reels with the latest crop logic.

Usage:
    python regen_crops.py CACHE_DIR OUTPUT_DIR [SIDECAR_DIR]
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.crop import smart_crop_916, smart_crop_916_stacked
from src.detect import detect_humans_per_frame, detect_humans_all_per_frame
from src.logging_util import setup_logging
from src.subtitles import burn_subtitles
from src.timeline import (
    apply_min_dwell,
    build_speaker_timeline,
    classify_wide_shot_frames,
)
from src.transcribe import transcribe_second_pass_cached


def _title_from_sidecar(path: Path) -> str:
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

    crop_mode = getattr(cfg.crop, "mode", "auto")

    reel_dirs = sorted(p for p in cache_dir.iterdir()
                       if p.is_dir() and p.name.startswith("reel_"))
    if not reel_dirs:
        log.warning(f"No reel_* dirs found in {cache_dir}")
        return 1

    # When a sidecar dir is given, restrict to reels whose sidecar exists
    # there (lets us target a single previous run when the cache holds many).
    if sidecar_dir:
        wanted = {p.stem for p in sidecar_dir.glob("reel_*.txt")}
        if wanted:
            filtered = [d for d in reel_dirs if d.name in wanted]
            log.info(f"Filtering to {len(filtered)}/{len(reel_dirs)} reels "
                     f"matching sidecars in {sidecar_dir}")
            reel_dirs = filtered
        log.info(f"Reading titles from sidecars in {sidecar_dir}")

    log.info(f"Found {len(reel_dirs)} cached reels; crop mode = {crop_mode!r}")

    failures: list[str] = []
    for i, reel_dir in enumerate(reel_dirs, 1):
        segment = reel_dir / "segment.mp4"
        if not segment.exists():
            log.warning(f"[{i}/{len(reel_dirs)}] {reel_dir.name}: no segment.mp4 — skip")
            failures.append(reel_dir.name)
            continue

        log.info(f"[{i}/{len(reel_dirs)}] {reel_dir.name}")
        try:
            cropped_path = reel_dir / "cropped_regen.mp4"
            if crop_mode == "auto":
                persons, _hf, fps, w, h = detect_humans_all_per_frame(segment, cfg)
                is_wide = classify_wide_shot_frames(
                    persons,
                    source_width=w,
                    source_height=h,
                    sep_threshold_frac=getattr(cfg.crop, "shot_sep_frac", 0.20),
                    height_cap_frac=getattr(cfg.crop, "shot_height_cap_frac", 0.70),
                    smooth_window_frames=getattr(cfg.crop, "shot_smooth_window_frames", 15),
                )
                smart_crop_916_stacked(segment, persons, is_wide, cropped_path, cfg)
            else:
                # DEPRECATED `crop.mode: single` branch.
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
                smart_crop_916(segment, timeline, cropped_path, cfg)

            words = transcribe_second_pass_cached(
                segment, reel_dir / "words.json", cfg,
            )

            title = (_title_from_sidecar(sidecar_dir / f"{reel_dir.name}.txt")
                     if sidecar_dir else "")

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
