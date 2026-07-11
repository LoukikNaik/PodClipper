"""Background-music library: load, LLM-match a track to a reel, mix a ducked bed."""

from __future__ import annotations

import json
import logging
import random
import re
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from .prompts import load_prompt

log = logging.getLogger("ave.music")


class MusicError(Exception):
    pass


def load_library(path: Path) -> Optional[dict]:
    """Load the music library manifest, dropping `disabled` songs/sections;
    return None if missing/empty."""
    path = Path(path)
    if not path.exists():
        log.warning(f"music library not found: {path}")
        return None
    lib = json.loads(path.read_text())
    songs = []
    for s in lib.get("songs") or []:
        if s.get("disabled"):
            continue
        s["sections"] = [sec for sec in s.get("sections", []) if not sec.get("disabled")]
        if s["sections"]:
            songs.append(s)
    if not songs:
        log.warning(f"music library has no enabled songs: {path}")
        return None
    lib["songs"] = songs
    lib["_dir"] = path.parent
    return lib


def _resolve(lib: dict, song: dict) -> Path:
    return Path(lib["_dir"]) / song["file"]


def _candidates(songs: list[dict]) -> list[tuple[dict, dict]]:
    """Flatten to (song, section) pairs — the unit the selector scores."""
    return [(song, sec) for song in songs for sec in (song.get("sections") or [])]


def _track_menu(cands: list[tuple[dict, dict]]) -> str:
    """One numbered block per section, with the section's own vocals + context
    so the selector scores THIS passage, not the whole song."""
    blocks = []
    for i, (song, sec) in enumerate(cands):
        vocals = sec.get("vocals") or sec.get("lyrics", "")
        blocks.append(
            f"{i}: {song['title']}\n"
            f"   song: {song.get('description', '')}\n"
            f"   this section — {sec.get('context', '')}\n"
            f"   lyrics: {vocals}\n"
            f"   use when: {sec.get('description', '')}"
        )
    return "\n".join(blocks)


def _parse_scores(raw: str) -> list[float]:
    """Pull the `scores` array out of the selector's JSON reply; [] on failure."""
    obj = re.search(r"\{.*\}", raw, re.DOTALL)
    if not obj:
        return []
    try:
        scores = json.loads(obj.group(0)).get("scores", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    if not isinstance(scores, list):
        return []
    return [float(s) for s in scores if isinstance(s, (int, float))]


def select_track(title: str, transcript: str, lib: dict,
                 provider, cfg: SimpleNamespace) -> Optional[dict]:
    """Score every library SECTION for the reel in one LLM call, pick the
    highest; fall back to a random section if none clears `music.min_score`.
    Returns {file, start, end, song_id, section_id} or None."""
    cands = _candidates(lib["songs"])
    if not cands:
        return None
    min_score = float(getattr(getattr(cfg, "music", SimpleNamespace()), "min_score", 5.0))
    scores: list[float] = []
    try:
        raw = provider.complete(
            user_prompt=f"REEL TITLE: {title}\n\nTRANSCRIPT:\n{transcript}\n\n"
                        f"AVAILABLE TRACKS:\n{_track_menu(cands)}",
            system_prompt=load_prompt("music_selector.txt"),
            max_tokens=400,
        )
        scores = _parse_scores(raw)
    except Exception as e:  # noqa: BLE001
        log.warning(f"music selection LLM call failed ({e}); picking at random")

    if len(scores) == len(cands):
        best = max(range(len(cands)), key=lambda i: scores[i])
        if scores[best] >= min_score:
            song, section = cands[best]
            log.info(f"music: matched '{song['title']}' / {section.get('id', '')} "
                     f"(score {scores[best]:.0f}/10) to reel '{title[:40]}'")
        else:
            song, section = random.choice(cands)
            log.info(f"music: top score {scores[best]:.0f} < {min_score:.0f} — "
                     f"random pick '{song['title']}' / {section.get('id', '')}")
    else:
        song, section = random.choice(cands)
        log.info(f"music: no usable scores — random pick '{song['title']}' / "
                 f"{section.get('id', '')}")

    return {
        "file": _resolve(lib, song),
        "start": float(section.get("start", 0.0)),
        "end": section.get("end"),
        "song_id": song["id"],
        "section_id": section.get("id", ""),
    }


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def mix_music(reel_video: Path, track: dict, out_path: Path,
              cfg: SimpleNamespace) -> Path:
    """Mix a gentle, ducked music bed under the reel's existing speech audio."""
    mcfg = getattr(cfg, "music", SimpleNamespace())
    gain = float(getattr(mcfg, "gain", 0.32))
    ratio = float(getattr(mcfg, "duck_ratio", 3.0))
    threshold = float(getattr(mcfg, "duck_threshold", 0.04))
    release = float(getattr(mcfg, "duck_release", 450))
    fade = float(getattr(mcfg, "fade_seconds", 0.8))

    dur = _duration(reel_video)
    fade_out = max(0.0, dur - fade)
    src = Path(track["file"])
    if not src.exists():
        raise MusicError(f"music file missing: {src}")

    start = float(track["start"])
    total = _duration(src)
    avail = total - start
    if avail <= 0:
        start, avail = 0.0, total
    seg_len = min(avail, dur)

    filt = (
        f"[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
        f"asplit=2[sp][sc];"
        f"[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
        f"afade=t=in:st=0:d={fade:.2f},afade=t=out:st={fade_out:.2f}:d={fade:.2f},"
        f"volume={gain}[musraw];"
        f"[musraw][sc]sidechaincompress=threshold={threshold}:ratio={ratio}:"
        f"attack=15:release={release:.0f}[mus];"
        f"[sp][mus]amix=inputs=2:duration=first:normalize=0[aout]"
    )
    with tempfile.TemporaryDirectory() as td:
        # Play from the section start through the natural end of the song; only
        # loop back to the start once the mp3 itself runs out (reels are short and
        # the section start is the intended entry, so a natural continuation past
        # the section end beats an early repeat). `end` is now advisory metadata.
        seg = Path(td) / "seg.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", f"{start:.3f}", "-t", f"{seg_len:.3f}", "-i", str(src),
             "-ar", "44100", "-ac", "2", str(seg)],
            check=True, capture_output=True,
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(reel_video),
            "-stream_loop", "-1", "-t", f"{dur:.3f}", "-i", str(seg),
            "-filter_complex", filt,
            "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)
        except subprocess.CalledProcessError as e:
            raise MusicError(f"music mix failed: {e.stderr.strip()[-500:]}") from e
    return out_path
