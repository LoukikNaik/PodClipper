"""Karaoke subtitle + fading title overlay burner (OpenCV + Pillow)."""

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


@dataclass
class SubLine:
    start: float
    end: float
    words: list[Word]


@dataclass
class PopPopup:
    start: float
    end: float
    words: list[Word]


def generate_pop_popups(
    words: list[Word],
    max_words_per_popup: int = 1,
    max_gap_seconds: float = 1.0,
) -> list[PopPopup]:
    """Group word tokens into 1–N word pop-style popups."""
    popups: list[PopPopup] = []
    buf: list[Word] = []

    def flush() -> None:
        nonlocal buf
        if buf:
            popups.append(PopPopup(start=buf[0].start, end=buf[-1].end, words=list(buf)))
            buf = []

    for w in words:
        if not w.text.strip():
            continue
        if buf and (w.start - buf[-1].end) > max_gap_seconds:
            flush()
        buf.append(w)
        if len(buf) >= max_words_per_popup:
            flush()
    flush()
    return popups


def generate_subtitle_lines(
    words: list[Word],
    cfg: SimpleNamespace,
    max_gap_seconds: float = 0.8,
) -> list[SubLine]:
    """Group word tokens into screen-sized lines, breaking on max chars,
    sentence-ending punctuation, or long pauses."""
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
    # ASS alpha: 0=opaque, 255=transparent — invert for Pillow.
    return (r, g, b, 255 - a)


_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int, preferred_name: str = "Arial") -> ImageFont.ImageFont:
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
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    text_parts: list[tuple[str, bool]] = []
    for i, wobj in enumerate(line.words):
        token = wobj.text.strip()
        if not token:
            continue
        is_active = wobj.start <= t <= wobj.end
        prefix = "" if i == 0 else " "
        text_parts.append((prefix + token, is_active))

    total_width = 0
    for txt, _ in text_parts:
        bbox = draw.textbbox((0, 0), txt, font=font)
        total_width += bbox[2] - bbox[0]

    line_bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_height = line_bbox[3] - line_bbox[1]
    y = h - margin_v - line_height

    x = (w - total_width) // 2

    for txt, is_active in text_parts:
        color = highlight_rgba if is_active else primary_rgba
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


def _wrap_title(title: str, max_chars: int) -> list[str]:
    """Greedy word-wrap into lines no longer than `max_chars`."""
    words = title.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        candidate = f"{cur} {w}".strip()
        if len(candidate) <= max_chars or not cur:
            cur = candidate
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _title_alpha(t: float, duration: float, fade: float) -> float:
    """0.0–1.0 opacity for the title overlay at time t."""
    if t < 0 or t >= duration:
        return 0.0
    fade_start = max(0.0, duration - fade)
    if t <= fade_start:
        return 1.0
    return max(0.0, 1.0 - (t - fade_start) / fade)


def _apply_alpha(rgba: tuple[int, int, int, int], factor: float) -> tuple[int, int, int, int]:
    r, g, b, a = rgba
    return (r, g, b, int(round(a * factor)))


def _render_title_onto_frame(
    frame: np.ndarray,
    title_lines: list[str],
    alpha: float,
    font: ImageFont.ImageFont,
    cfg: SimpleNamespace,
    frame_size: tuple[int, int],
) -> np.ndarray:
    """Composite the title at the top of the frame. Supports gradient
    backdrop (default) or pill backdrop (legacy)."""
    tcfg = cfg.subtitles.title_overlay
    frame_w, frame_h = frame_size

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    line_heights: list[int] = []
    line_widths: list[int] = []
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    if not title_lines:
        return frame

    total_text_h = sum(line_heights) + int(tcfg.line_spacing) * max(0, len(title_lines) - 1)
    max_line_w = max(line_widths)

    y0 = int(tcfg.y_offset)
    x_center = frame_w // 2

    if bool(getattr(tcfg, "gradient_enabled", False)):
        grad_opacity = float(getattr(tcfg, "gradient_opacity", 0.55)) * alpha
        grad_h = int(frame_h * 0.35)
        alpha_col = np.linspace(grad_opacity * 255, 0, grad_h, dtype=np.uint8)
        grad_arr = np.zeros((grad_h, frame_w, 4), dtype=np.uint8)
        grad_arr[:, :, 3] = alpha_col[:, np.newaxis]
        grad = Image.fromarray(grad_arr, "RGBA")
        overlay.paste(grad, (0, 0))

    elif bool(getattr(tcfg, "pill_enabled", False)):
        pad_x = int(tcfg.pill_padding_x)
        pad_y = int(tcfg.pill_padding_y)
        radius = int(tcfg.pill_corner_radius)
        pill_w = max_line_w + 2 * pad_x
        pill_h = total_text_h + 2 * pad_y
        pill_x = x_center - pill_w // 2
        pill_y = y0 - pad_y
        pill_color = _apply_alpha(_parse_ass_color(tcfg.pill_color), alpha)
        draw.rounded_rectangle(
            [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
            radius=radius, fill=pill_color,
        )

    text_rgba = _apply_alpha(_parse_ass_color(tcfg.color), alpha)
    outline_rgba = _apply_alpha(_parse_ass_color(tcfg.outline_color), alpha)
    outline_w = int(tcfg.outline_width)

    y = y0
    for line, lw, lh in zip(title_lines, line_widths, line_heights):
        x = x_center - lw // 2
        try:
            draw.text(
                (x, y), line, font=font, fill=text_rgba,
                stroke_width=outline_w, stroke_fill=outline_rgba,
            )
        except TypeError:
            # Older Pillow without stroke_width — manual offset
            for dx in range(-outline_w, outline_w + 1):
                for dy in range(-outline_w, outline_w + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((x + dx, y + dy), line, font=font, fill=outline_rgba)
            draw.text((x, y), line, font=font, fill=text_rgba)
        y += lh + int(tcfg.line_spacing)

    merged = Image.alpha_composite(pil, overlay).convert("RGB")
    return cv2.cvtColor(np.array(merged), cv2.COLOR_RGB2BGR)


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


def _mux_audio(
    source_video: Path,
    video_only: Path,
    final_out: Path,
    audio_fade_out: float = 0.0,
    video_duration: float = 0.0,
) -> None:
    """Mux source-video audio onto the rendered video, optionally fading
    audio out over the last `audio_fade_out` seconds (uses rendered duration,
    not source, since they may differ slightly)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_only),
        "-i", str(source_video),
        "-c:v", "copy",
    ]
    if audio_fade_out > 0 and video_duration > audio_fade_out:
        fade_start = max(0.0, video_duration - audio_fade_out)
        cmd += [
            "-c:a", "aac",
            "-af", f"afade=t=out:st={fade_start:.3f}:d={audio_fade_out:.3f}",
        ]
    else:
        cmd += ["-c:a", "aac"]
    cmd += [
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        str(final_out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as e:
        raise SubtitleError(f"audio mux failed: {e.stderr.strip()[-500:]}") from e


def burn_subtitles(
    video_path: Path,
    words: list[Word],
    out_path: Path,
    cfg: SimpleNamespace,
    title: str = "",
) -> Path:
    """Render karaoke subtitles + optional fading title overlay onto `video_path`."""
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not words:
        raise SubtitleError("empty word list — nothing to render")

    lines = generate_subtitle_lines(words, cfg)
    log.info(f"Subtitles: {len(words)} words → {len(lines)} lines" + (f" + title={title!r}" if title else ""))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SubtitleError(f"OpenCV could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fade_out_seconds = float(getattr(cfg.subtitles, "fade_out_seconds", 0.0))
    fade_out_start_frame = (
        max(0, total_frames - int(round(fade_out_seconds * fps)))
        if (fade_out_seconds > 0 and total_frames > 0)
        else None
    )

    font = _load_font(int(cfg.subtitles.font_size), cfg.subtitles.font_name)
    primary = _parse_ass_color(cfg.subtitles.primary_color)
    highlight = _parse_ass_color(cfg.subtitles.highlight_color)
    outline = _parse_ass_color(cfg.subtitles.outline_color)
    outline_w = int(cfg.subtitles.outline_width)
    margin_v = int(cfg.subtitles.margin_v)

    tcfg = cfg.subtitles.title_overlay
    show_title = bool(title) and bool(getattr(tcfg, "enabled", False))
    title_lines: list[str] = []
    title_font: Optional[ImageFont.ImageFont] = None
    title_duration = float(getattr(tcfg, "duration_seconds", 0.0))
    title_fade = float(getattr(tcfg, "fade_seconds", 0.0))
    if show_title:
        title_lines = _wrap_title(title, int(tcfg.max_chars_per_line))
        title_font = _load_font(int(tcfg.font_size), cfg.subtitles.font_name)

    tmp_video = Path(tempfile.mkstemp(prefix="ave_sub_", suffix=".mp4")[1])
    encoder = _open_ffmpeg_pipe(tmp_video, width, height, fps, cfg)

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps

            if show_title:
                alpha = _title_alpha(t, title_duration, title_fade)
                if alpha > 0.0 and title_font is not None:
                    frame = _render_title_onto_frame(
                        frame, title_lines, alpha, title_font, cfg, (width, height),
                    )

            line = _active_line(lines, t)
            if line is not None:
                frame = _render_line_onto_frame(
                    frame, line, t, font,
                    primary, highlight, outline, outline_w, margin_v,
                )

            if fade_out_start_frame is not None and frame_idx >= fade_out_start_frame:
                progress = (frame_idx - fade_out_start_frame) / max(
                    1, total_frames - fade_out_start_frame - 1
                )
                alpha = max(0.0, 1.0 - progress)
                frame = (frame.astype(np.float32) * alpha).astype(np.uint8)

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

    rendered_duration = frame_idx / fps if fps > 0 else 0.0
    log.info(
        f"Subtitled {frame_idx} frames → muxing audio"
        + (f" (fade-out {fade_out_seconds:.2f}s)" if fade_out_seconds > 0 else "")
        + "..."
    )
    try:
        _mux_audio(
            video_path, tmp_video, out_path,
            audio_fade_out=fade_out_seconds,
            video_duration=rendered_duration,
        )
    finally:
        tmp_video.unlink(missing_ok=True)

    log.info(f"Subtitle burn complete → {out_path}")
    return out_path
