"""Stage 5d: Subtitle rendering and burn-in.

Renders word-level karaoke subtitles directly onto video frames using OpenCV
+ Pillow, avoiding any dependency on libass (which is commonly missing from
homebrew ffmpeg builds on macOS). Two public entry points:

  generate_subtitle_lines(words, cfg) → list[SubLine]
      Group raw words into reel-sized lines (can be cached/debugged).

  burn_subtitles(video_path, words, out_path, cfg) → Path
      Decode video frame-by-frame, composite karaoke subtitles, pipe to ffmpeg
      encoder, mux back the source audio.

Styling knobs come from config.subtitles; fonts are resolved against the
system (macOS path first, then Pillow default).
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import Word

log = logging.getLogger("ave.subtitles")


class SubtitleError(Exception):
    pass


# ---------- Line grouping ----------

@dataclass
class SubLine:
    start: float
    end: float
    words: list[Word]


def generate_subtitle_lines(
    words: list[Word],
    cfg: SimpleNamespace,
    max_gap_seconds: float = 0.8,
) -> list[SubLine]:
    """Group word tokens into screen-sized lines.

    Breaks on: (a) max chars reached, (b) sentence-ending punctuation, (c) long pause.
    """
    max_chars = int(cfg.subtitles.max_chars_per_line)
    SENT_END = {".", "!", "?"}

    lines: list[SubLine] = []
    buf: list[Word] = []
    buf_chars = 0

    def flush() -> None:
        nonlocal buf, buf_chars
        if not buf:
            return
        lines.append(SubLine(start=buf[0].start, end=buf[-1].end, words=list(buf)))
        buf = []
        buf_chars = 0

    for w in words:
        token = w.text.strip()
        if not token:
            continue
        if buf and (w.start - buf[-1].end) > max_gap_seconds:
            flush()
        if buf and (buf_chars + 1 + len(token)) > max_chars:
            flush()
        buf.append(w)
        buf_chars += len(token) + (1 if buf_chars > 0 else 0)
        if token[-1] in SENT_END:
            flush()
    flush()
    return lines


# ---------- ASS color helper ----------

def _parse_ass_color(c: str) -> tuple[int, int, int, int]:
    """Parse an ASS &HAABBGGRR color (or plain hex) into RGBA for Pillow."""
    s = c.strip().lstrip("&").lstrip("H").lstrip("h")
    s = s.zfill(8) if len(s) <= 8 else s
    try:
        val = int(s, 16)
    except ValueError:
        return (255, 255, 255, 255)
    a = (val >> 24) & 0xFF
    b = (val >> 16) & 0xFF
    g = (val >> 8) & 0xFF
    r = val & 0xFF
    # ASS alpha semantic: 0=opaque, 255=transparent — invert for Pillow
    return (r, g, b, 255 - a)


# ---------- Font resolution ----------

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int, preferred_name: str = "Arial") -> ImageFont.ImageFont:
    # Try to respect preferred_name by substring match first
    name_lc = preferred_name.lower()
    ordered = sorted(_FONT_CANDIDATES, key=lambda p: 0 if name_lc in Path(p).stem.lower() else 1)
    for p in ordered:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
    log.warning(f"No system TTF found — falling back to PIL default bitmap font (size won't apply)")
    return ImageFont.load_default()


# ---------- Frame rendering ----------

def _render_line_onto_frame(
    frame: np.ndarray,
    line: SubLine,
    t: float,
    font: ImageFont.ImageFont,
    primary_rgba: tuple[int, int, int, int],
    highlight_rgba: tuple[int, int, int, int],
    outline_rgba: tuple[int, int, int, int],
    outline_width: int,
    margin_v: int,
) -> np.ndarray:
    """Composite one subtitle line onto `frame` with karaoke highlight."""
    h, w = frame.shape[:2]
    # BGR (OpenCV) → RGB (PIL)
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Measure line width to center it
    text_parts: list[tuple[str, bool]] = []  # (word_with_leading_space, is_active)
    for i, wobj in enumerate(line.words):
        token = wobj.text.strip()
        if not token:
            continue
        is_active = wobj.start <= t <= wobj.end
        prefix = "" if i == 0 else " "
        text_parts.append((prefix + token, is_active))

    # Total width
    total_width = 0
    for txt, _ in text_parts:
        bbox = draw.textbbox((0, 0), txt, font=font)
        total_width += bbox[2] - bbox[0]

    # Baseline y: lower third of frame, controlled by margin_v
    line_bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_height = line_bbox[3] - line_bbox[1]
    y = h - margin_v - line_height

    # Center horizontally
    x = (w - total_width) // 2

    for txt, is_active in text_parts:
        color = highlight_rgba if is_active else primary_rgba
        # Outline: draw the text multiple times with offset
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), txt, font=font, fill=outline_rgba)
        draw.text((x, y), txt, font=font, fill=color)
        advance = draw.textbbox((0, 0), txt, font=font)
        x += advance[2] - advance[0]

    merged = Image.alpha_composite(pil, overlay).convert("RGB")
    return cv2.cvtColor(np.array(merged), cv2.COLOR_RGB2BGR)


def _active_line(lines: list[SubLine], t: float) -> Optional[SubLine]:
    for line in lines:
        if line.start <= t <= line.end:
            return line
    return None


# ---------- ffmpeg encoder pipe (reused from crop) ----------

def _open_ffmpeg_pipe(
    out_path: Path,
    width: int,
    height: int,
    fps: float,
    cfg: SimpleNamespace,
) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps}",
        "-i", "-",
        "-c:v", cfg.crop.ffmpeg_encoder,
        "-pix_fmt", "yuv420p",
        "-crf", str(cfg.crop.ffmpeg_crf),
        "-preset", cfg.crop.ffmpeg_preset,
        "-an",
        str(out_path),
    ]
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def _mux_audio(source_video: Path, video_only: Path, final_out: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_only),
        "-i", str(source_video),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        str(final_out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as e:
        raise SubtitleError(f"audio mux failed: {e.stderr.strip()[-500:]}") from e


# ---------- Main API ----------

def burn_subtitles(
    video_path: Path,
    words: list[Word],
    out_path: Path,
    cfg: SimpleNamespace,
) -> Path:
    """Render karaoke subtitles onto `video_path` and write to `out_path`.

    Word timestamps are clip-relative (seconds from video start).
    """
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not words:
        raise SubtitleError("empty word list — nothing to render")

    lines = generate_subtitle_lines(words, cfg)
    log.info(f"Subtitles: {len(words)} words → {len(lines)} lines")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SubtitleError(f"OpenCV could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    font = _load_font(int(cfg.subtitles.font_size), cfg.subtitles.font_name)
    primary = _parse_ass_color(cfg.subtitles.primary_color)
    highlight = _parse_ass_color(cfg.subtitles.highlight_color)
    outline = _parse_ass_color(cfg.subtitles.outline_color)
    outline_w = int(cfg.subtitles.outline_width)
    margin_v = int(cfg.subtitles.margin_v)

    tmp_video = Path(tempfile.mkstemp(prefix="ave_sub_", suffix=".mp4")[1])
    encoder = _open_ffmpeg_pipe(tmp_video, width, height, fps, cfg)

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps
            line = _active_line(lines, t)
            if line is not None:
                frame = _render_line_onto_frame(
                    frame, line, t, font,
                    primary, highlight, outline, outline_w, margin_v,
                )
            try:
                encoder.stdin.write(frame.tobytes())
            except BrokenPipeError as e:
                stderr = encoder.stderr.read().decode("utf-8", errors="replace")[-500:]
                raise SubtitleError(f"ffmpeg encoder died: {stderr}") from e
            frame_idx += 1
    finally:
        cap.release()

    encoder.stdin.close()
    ret = encoder.wait(timeout=120)
    if ret != 0:
        stderr = encoder.stderr.read().decode("utf-8", errors="replace")[-500:]
        raise SubtitleError(f"ffmpeg encoder exited with {ret}: {stderr}")

    log.info(f"Subtitled {frame_idx} frames → muxing audio...")
    try:
        _mux_audio(video_path, tmp_video, out_path)
    finally:
        tmp_video.unlink(missing_ok=True)

    log.info(f"Subtitle burn complete → {out_path}")
    return out_path
