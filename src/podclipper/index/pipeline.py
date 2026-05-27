"""Phase 1 indexer: video (local path OR YouTube URL) → one timestamped
VideoIndex (scenes.json + .md) the editing agent reasons over.

Runs in the MLX env:
    .venv/bin/python -m podclipper.index <path-or-url> -o index/
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .caption import FRAMES_PER_SCENE, MODEL_ID, caption_scene, load_captioner, sample_frames
from .scenes import cuts_to_scenes, detect_cuts, words_in_range
from .schema import Scene, VideoIndex

WHISPER_REPO = "mlx-community/whisper-large-v3-mlx"


def resolve_input(arg: str, work_dir: Path) -> Path:
    if re.match(r"^https?://", arg):
        out = work_dir / "source.mp4"
        print(f"[index] downloading {arg}", flush=True)
        subprocess.check_call([
            "yt-dlp", "-f", "bv*[height<=720]+ba/b[height<=720]/b",
            "--merge-output-format", "mp4", "-o", str(out), arg,
        ])
        return out
    p = Path(arg)
    if not p.exists():
        sys.exit(f"input not found: {arg}")
    return p


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)]
    )
    return float(out.strip())


def transcribe_words(
    path: Path, repo: str = WHISPER_REPO,
    start_at: float = 0.0, stop_at: float | None = None,
    language: str | None = None,
) -> list[dict]:
    """Transcribe (a window of) the video with word-level timestamps. When a
    window is given, only that audio is decoded and word times are offset back
    to absolute video time."""
    import mlx_whisper
    src = str(path)
    offset = start_at
    tmp_audio: Path | None = None
    if start_at > 0.0 or stop_at is not None:
        tmp_audio = path.with_suffix(".window.wav")
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        if start_at > 0.0:
            cmd += ["-ss", f"{start_at:.3f}"]
        if stop_at is not None:
            cmd += ["-t", f"{max(0.0, stop_at - start_at):.3f}"]
        cmd += ["-i", str(path), "-ac", "1", "-ar", "16000", str(tmp_audio)]
        subprocess.check_call(cmd)
        src = str(tmp_audio)
    print(f"[index] transcribing with {repo} (language={language or 'auto'})", flush=True)
    result = mlx_whisper.transcribe(
        src, path_or_hf_repo=repo, word_timestamps=True, verbose=None,
        language=language, condition_on_previous_text=False,
    )
    words: list[dict] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []) or []:
            words.append({"start": float(w["start"]) + offset,
                          "end": float(w["end"]) + offset, "text": w["word"]})
    if tmp_audio is not None and tmp_audio.exists():
        tmp_audio.unlink()
    return words


def build_index(
    video: Path, source_label: str, out_dir: Path, start_at: float = 0.0,
    stop_at: float | None = None, language: str | None = None,
) -> VideoIndex:
    duration = probe_duration(video)
    if stop_at is not None:
        duration = min(duration, stop_at)

    cuts = detect_cuts(video, stop_at=stop_at)
    scene_ranges = [(max(s, start_at), e) for s, e in cuts_to_scenes(cuts, duration)
                    if e > start_at]
    print(f"[index] {video.name}  {duration:.1f}s  {len(cuts)} cuts → {len(scene_ranges)} scenes", flush=True)

    words = transcribe_words(video, start_at=start_at, stop_at=stop_at, language=language)

    print(f"[index] loading {MODEL_ID}", flush=True)
    captioner = load_captioner()

    index = VideoIndex(
        source=source_label, duration_seconds=duration,
        vision_model=MODEL_ID, transcript_model=WHISPER_REPO, scenes=[],
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for i, (s, e) in enumerate(scene_ranges):
            frames = sample_frames(video, s, e - s, FRAMES_PER_SCENE, tmp_path / f"sc{i:04d}")
            t0 = time.time()
            cap = caption_scene(captioner, frames)
            transcript = words_in_range(words, s, e)
            scene_words = [w for w in words if s <= (w["start"] + w["end"]) / 2 < e]
            index.scenes.append(Scene(
                id=i, start=round(s, 2), end=round(e, 2), duration=round(e - s, 2),
                visual=cap.get("visual", ""), on_screen_text=cap.get("on_screen_text", ""),
                mood=cap.get("mood", ""), transcript=transcript, words=scene_words,
                has_speech=bool(transcript),
            ))
            print(f"[scene {i:04d}] {s:7.1f}-{e:7.1f}s  ({e-s:4.1f}s)  "
                  f"speech={'Y' if transcript else 'n'}  cap={time.time()-t0:.1f}s  "
                  f"{cap.get('visual','')[:55]}", flush=True)
            (out_dir / "scenes.json").write_text(index.to_json())

    (out_dir / "scenes.json").write_text(index.to_json())
    return index


def run(arg: str, out_root: Path, start_at: float = 0.0, stop_at: float | None = None,
        language: str | None = None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        video = resolve_input(arg, Path(tmp))
        stem = video.stem if not re.match(r"^https?://", arg) else "video"
        out_dir = out_root / stem
        index = build_index(video, arg, out_dir, start_at, stop_at, language)
        (out_dir / f"{stem}.md").write_text(index.to_markdown())
        print(f"[done] {out_dir}/scenes.json  +  {out_dir}/{stem}.md  "
              f"({len(index.scenes)} scenes)", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="podclipper.index")
    p.add_argument("input", help="local video path OR YouTube URL")
    p.add_argument("-o", "--out-root", type=Path, default=Path("index"))
    p.add_argument("--start-at", type=float, default=0.0, help="index from this second")
    p.add_argument("--stop-at", type=float, default=None, help="stop indexing at this second")
    p.add_argument("--language", default=None, help="force Whisper language (e.g. 'hi'); default auto-detect")
    args = p.parse_args(argv)
    run(args.input, args.out_root, args.start_at, args.stop_at, args.language)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
