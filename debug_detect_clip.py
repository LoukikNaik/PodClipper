#!/usr/bin/env python3
"""Run only the detect stage on a clip with debug overlay enabled. Writes
`debug_detect.mp4` next to the input clip."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.detect import detect_humans_per_frame
from src.logging_util import setup_logging


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python debug_detect_clip.py PATH/TO/CLIP.mp4")
        return 2
    clip = Path(sys.argv[1]).resolve()
    if not clip.exists():
        print(f"not found: {clip}")
        return 2

    cfg = load_config(Path("config/default.yaml"))
    cfg.detect.debug_overlay = True
    cfg.detect.face_aware = True
    setup_logging("INFO")

    bboxes, fps, w, h = detect_humans_per_frame(clip, cfg)
    print(f"\nReturned {len(bboxes)} frames; "
          f"{sum(1 for b in bboxes if b is not None)} with a primary bbox; "
          f"fps={fps:.2f}, size={w}x{h}")
    print(f"Debug overlay → {clip.parent / 'debug_detect.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
