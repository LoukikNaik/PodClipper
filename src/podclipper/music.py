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


def _pick_section(song: dict) -> dict:
    secs = song.get("sections") or []
    return random.choice(secs) if secs else {"start": 0.0, "end": None}


def _track_menu(songs: list[dict]) -> str:
    return "\n".join(f"{i}: {s['title']} — {s.get('description', '')}"
                     for i, s in enumerate(songs))


def _parse_index(raw: str) -> int:
    m = re.search(r'"index"\s*:\s*(-?\d+)', raw)
    if m:
        return int(m.group(1))
    obj = re.search(r"\{.*\}", raw, re.DOTALL)
    if obj:
        return int(json.loads(obj.group(0)).get("index", -1))
    return -1


def select_track(title: str, transcript: str, lib: dict,
                 provider, cfg: SimpleNamespace) -> Optional[dict]:
    """LLM-match a reel to a library track; fall back to random. Returns
    {file, start, end, song_id, section_id} or None."""
    songs = lib["songs"]
    idx = -1
    try:
        raw = provider.complete(
            user_prompt=f"REEL TITLE: {title}\n\nTRANSCRIPT:\n{transcript}\n\n"
                        f"AVAILABLE TRACKS:\n{_track_menu(songs)}",
            system_prompt=load_prompt("music_selector.txt"),
            max_tokens=300,
        )
        idx = _parse_index(raw)
    except Exception as e:  # noqa: BLE001
        log.warning(f"music selection LLM call failed ({e}); picking at random")

    if 0 <= idx < len(songs):
        song = songs[idx]
        log.info(f"music: matched '{song['title']}' to reel '{title[:40]}'")
    else:
        song = random.choice(songs)
        log.info(f"music: no confident match — random pick '{song['title']}'")

    section = _pick_section(song)
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
    end = track.get("end")
    seg_len = (float(end) - start) if end else dur

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
        # Cut the curated section, then loop IT to fill the reel — never bleed
        # past the section boundary into unwanted (e.g. vocal) parts of the song.
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
