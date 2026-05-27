#!/usr/bin/env python3
"""Run mlx-community/Qwen2.5-VL-3B-Instruct-4bit on a video and emit
Marlin-shaped {scene, events: [{start, end, description}]} JSON + readout."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MODEL_ID = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
CHUNK_SECONDS = 30
FPS = 2.0
MAX_TOKENS = 1024


CAPTION_PROMPT = """You are a video captioner. Watch this clip carefully and return ONE JSON object, nothing else.

Schema:
{
  "scene": "1-3 sentences describing the setting, people, and overall format of the clip",
  "events": [
    {"start": <float seconds from clip start>, "end": <float seconds>, "description": "<short phrase>"},
    ...
  ]
}

Rules:
- Events should partition the clip with no gaps and no overlaps.
- Each event is 1-5 seconds, describing what is visually happening (who speaks, gestures, cuts, overlays).
- Times are relative to the clip start (0.0 = first frame).
- Output JSON only, no prose, no markdown fences."""


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)]
    )
    return float(out.strip())


def cut_chunk(src: Path, start: float, dur: float, dst: Path) -> None:
    subprocess.check_call(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
         "-c", "copy", "-avoid_negative_ts", "make_zero", str(dst)]
    )


def _parse_json_object(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(s[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return {"scene": "", "events": [], "_raw": text}


def caption_chunk(model, processor, config, chunk_path: Path) -> tuple[dict, str]:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    formatted = apply_chat_template(
        processor, config, CAPTION_PROMPT, num_images=0, num_videos=1,
    )
    out = generate(
        model, processor, formatted,
        video=[str(chunk_path)], max_tokens=MAX_TOKENS, verbose=False,
    )
    text = out.text if hasattr(out, "text") else str(out)
    return _parse_json_object(text), text


def offset_events(result: dict, offset: float) -> dict:
    events = []
    for ev in result.get("events", []) or []:
        try:
            s, e = float(ev.get("start", 0.0)), float(ev.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        events.append(
            {"start": s + offset, "end": e + offset,
             "description": str(ev.get("description", ""))}
        )
    return {
        "chunk_offset_seconds": offset,
        "scene": result.get("scene", ""),
        "events": events,
    }


def run(video: Path, out_dir: Path, chunk_seconds: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem
    json_out = out_dir / f"{stem}.qwen_mlx.json"
    txt_out = out_dir / f"{stem}.qwen_mlx.txt"

    duration = probe_duration(video)
    print(f"[input] {video.name}  duration={duration:.1f}s  chunk={chunk_seconds}s", flush=True)

    print(f"[load] {MODEL_ID}", flush=True)
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    t0 = time.time()
    model, processor = load(MODEL_ID)
    config = load_config(MODEL_ID)
    print(f"[load] done in {time.time() - t0:.1f}s", flush=True)

    starts = []
    t = 0.0
    while t < duration:
        starts.append((t, min(chunk_seconds, duration - t)))
        t += chunk_seconds

    aggregated = {
        "video": str(video),
        "duration_seconds": duration,
        "chunk_seconds": chunk_seconds,
        "model": MODEL_ID,
        "chunks": [],
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for i, (start, dur) in enumerate(starts):
            chunk_path = tmp_path / f"chunk_{i:03d}.mp4"
            cut_chunk(video, start, dur, chunk_path)
            t0 = time.time()
            try:
                parsed, raw = caption_chunk(model, processor, config, chunk_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[chunk {i}] FAILED: {exc}", flush=True)
                aggregated["chunks"].append(
                    {"chunk_offset_seconds": start, "error": str(exc)}
                )
                continue
            dt = time.time() - t0
            out = offset_events(parsed, start)
            out["elapsed_seconds"] = round(dt, 2)
            out["raw_response"] = raw
            aggregated["chunks"].append(out)
            print(
                f"[chunk {i}] {start:7.1f}-{start+dur:7.1f}s  "
                f"events={len(out['events'])}  took={dt:.1f}s",
                flush=True,
            )
            json_out.write_text(json.dumps(aggregated, indent=2))

    json_out.write_text(json.dumps(aggregated, indent=2))

    lines = [
        f"# Qwen2.5-VL-3B-MLX caption: {video.name}",
        f"duration: {duration:.1f}s   model: {MODEL_ID}",
        "",
    ]
    for c in aggregated["chunks"]:
        off = c.get("chunk_offset_seconds", 0.0)
        lines.append(f"=== chunk @ {off:.1f}s  (took {c.get('elapsed_seconds', '?')}s) ===")
        if "error" in c:
            lines.append(f"  ERROR: {c['error']}")
            lines.append("")
            continue
        if c.get("scene"):
            lines.append(f"scene: {c['scene']}")
        for ev in c.get("events", []):
            lines.append(
                f"  [{ev['start']:7.2f} - {ev['end']:7.2f}]  {ev['description']}"
            )
        lines.append("")
    txt_out.write_text("\n".join(lines))

    print(f"[done] wrote {json_out} and {txt_out}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("-o", "--out-dir", type=Path, default=Path("outputs/marlin"))
    p.add_argument("--chunk-seconds", type=float, default=CHUNK_SECONDS)
    args = p.parse_args(argv)
    for v in args.videos:
        if not v.exists():
            print(f"missing: {v}", file=sys.stderr)
            return 2
        run(v, args.out_dir, args.chunk_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
