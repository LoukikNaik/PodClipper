#!/usr/bin/env python3
"""Run NemoStation/Marlin-2B caption mode over long videos in 2-min chunks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

def _load_dotenv(p: Path = Path(".env")) -> None:
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in {'"', "'"}:
            v = v[1:-1]
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()
if "HUGGINGFACE_HUB_TOKEN" not in os.environ and "HF_TOKEN" in os.environ:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
os.environ.setdefault("VIDEO_MAX_PIXELS", "200704")
os.environ.setdefault("FPS", "2.0")
os.environ.setdefault("FPS_MAX_FRAMES", "240")
os.environ.setdefault("FPS_MIN_FRAMES", "4")

MODEL_ID = "NemoStation/Marlin-2B"
CHUNK_SECONDS = 110


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path),
        ]
    )
    return float(out.strip())


def cut_chunk(src: Path, start: float, dur: float, dst: Path) -> None:
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
            "-c", "copy", "-avoid_negative_ts", "make_zero", str(dst),
        ]
    )


def load_marlin():
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    print(f"[load] device={device} dtype={dtype}", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=dtype,
    ).to(device).eval()
    return model, processor, device


def caption_chunk(model, processor, chunk_path: Path) -> dict:
    if hasattr(model, "caption"):
        return model.caption(str(chunk_path))
    raise RuntimeError("model has no caption() helper — check trust_remote_code load")


def offset_events(result: dict, offset: float) -> dict:
    events = []
    for ev in result.get("events", []) or []:
        events.append(
            {
                "start": float(ev.get("start", 0.0)) + offset,
                "end": float(ev.get("end", 0.0)) + offset,
                "description": ev.get("description", ""),
            }
        )
    return {
        "chunk_offset_seconds": offset,
        "scene": result.get("scene", ""),
        "caption": result.get("caption", ""),
        "events": events,
    }


def run(video: Path, out_dir: Path, chunk_seconds: float = CHUNK_SECONDS) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem
    json_out = out_dir / f"{stem}.marlin.json"
    txt_out = out_dir / f"{stem}.marlin.txt"

    duration = probe_duration(video)
    print(f"[input] {video.name}  duration={duration:.1f}s  chunk={chunk_seconds}s", flush=True)

    model, processor, device = load_marlin()

    chunks = []
    starts = []
    t = 0.0
    while t < duration:
        dur = min(chunk_seconds, duration - t)
        starts.append((t, dur))
        t += chunk_seconds

    aggregated = {
        "video": str(video),
        "duration_seconds": duration,
        "chunk_seconds": chunk_seconds,
        "model": MODEL_ID,
        "device": device,
        "chunks": [],
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for i, (start, dur) in enumerate(starts):
            chunk_path = tmp_path / f"chunk_{i:03d}.mp4"
            cut_chunk(video, start, dur, chunk_path)
            t0 = time.time()
            try:
                raw = caption_chunk(model, processor, chunk_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[chunk {i}] FAILED: {exc}", flush=True)
                aggregated["chunks"].append(
                    {"chunk_offset_seconds": start, "error": str(exc)}
                )
                continue
            dt = time.time() - t0
            out = offset_events(raw, start)
            out["elapsed_seconds"] = round(dt, 2)
            aggregated["chunks"].append(out)
            print(
                f"[chunk {i}] {start:7.1f}-{start+dur:7.1f}s  "
                f"events={len(out['events'])}  took={dt:.1f}s",
                flush=True,
            )
            json_out.write_text(json.dumps(aggregated, indent=2))

    json_out.write_text(json.dumps(aggregated, indent=2))

    lines = [
        f"# Marlin-2B caption: {video.name}",
        f"duration: {duration:.1f}s   model: {MODEL_ID}   device: {device}",
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
        if c.get("caption"):
            lines.append(f"caption: {c['caption']}")
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
