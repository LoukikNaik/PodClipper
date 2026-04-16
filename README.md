# Agentic Video Editor

Turn a long-form video (podcast, talk, interview) into vertical 9:16 reels with smart speaker-tracking crops and karaoke subtitles. Pipeline is local-first: Whisper transcribes, an LLM (Claude CLI or Anthropic API) picks reel-worthy moments, YOLO tracks the speaker, OpenCV crops, and PIL burns in subtitles.

## Pipeline

```
Video
  ↓ ingest        (ffprobe — validate, metadata)
  ↓ audio         (ffmpeg  — extract mono 16kHz WAV)
  ↓ transcribe    (faster-whisper — fast 1st pass, parallel chunks)
  ↓ analyze       (LLM — pick reel-worthy clips as JSON)
  ↓ per clip:
       ├─ extract       (ffmpeg cut ±2s pad)
       ├─ detect        (YOLO v8 nano — per-frame person bboxes)
       ├─ transcribe 2  (faster-whisper large — clean word timestamps)
       ├─ timeline      (stub: largest persistent bbox → single segment;
       │                 post-MVP: pyannote diarization → multi-segment)
       ├─ crop          (OpenCV — 9:16, EMA smoothing, hard cut between segments)
       └─ subtitles     (PIL + OpenCV — karaoke word highlight, burn in)
  ↓ outputs/reel_NN_<title>.mp4
```

Full design rationale in `.cursor/plans/plan.md`.

## Setup

Requires Python 3.10+ and `ffmpeg` + `ffprobe` on PATH.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### LLM provider

Either works; pick one via `config/default.yaml` or `--llm-provider`:

- **Claude CLI** (default) — install Claude Code so the `claude` binary is on PATH.
- **Anthropic API** — set `ANTHROPIC_API_KEY` in your environment.

### YOLO + Whisper models

- `yolov8n.pt` (6 MB) downloads automatically on first run, cached by ultralytics.
- Whisper models (`base`, `large-v3`) download from HuggingFace on first use.
- Default compute type is `int8` on CPU — works on Apple Silicon without extra config. For CUDA, switch to `float16` / `int8_float16` in config.

## Usage

```bash
python main.py path/to/video.mp4
```

Common flags:

```bash
python main.py video.mp4 \
  --output-dir outputs/my-reels \
  --language en \
  --max-clips 5 \
  --debug-crop \
  -v
```

Run `python main.py --help` for the full list.

Outputs land in `outputs/reel_NN_<slugified-title>.mp4` alongside a `.txt` sidecar with the LLM's reasoning and source timestamps.

## Configuration

Everything is knob-tunable in `config/default.yaml`:

- `transcribe.first_pass.model` / `second_pass.model` — Whisper model sizes
- `analyze.min/max_clip_seconds`, `target_clips` — clip length and count hints to the LLM
- `detect.sample_every_n_frames` — bump for speed, drop to 1 for smoothest tracking
- `crop.smoothing_alpha` — lower = smoother panning
- `crop.min_segment_dwell_seconds` — min time between hard cuts (post-MVP multi-speaker)
- `subtitles.font_size`, `margin_v`, `primary_color`, `highlight_color` — styling

## Caching + resume

Intermediate artifacts are persisted under `.cache/<video_stem>-<hash>/` per source video:

```
.cache/mytalk-a2340ae50a/
  audio.wav                           # whole-video extracted audio
  reel_01_i-quit-my-job/
    segment.mp4                       # extracted clip + pad
    cropped.mp4                       # after 9:16 smart crop
```

Re-running reuses these — useful while iterating on subtitles or the LLM prompt. Pass `--no-cache` to force a rebuild.

## Project layout

```
config/default.yaml
prompts/reel_detector.txt     # LLM system prompt (editable)
main.py                       # CLI entry point
src/
  config.py   logging_util.py  types.py
  ingest.py   audio.py         transcribe.py
  llm/base.py llm/claude_cli.py llm/anthropic_api.py
  analyze.py
  detect.py   timeline.py      crop.py
  subtitles.py
  pipeline.py
```

## Known limitations (MVP)

- **Single-speaker assumption** — for multi-person content, we follow the most persistent bbox. Multi-speaker podcast-style follow-the-speaker cropping is designed in (`timeline.py` accepts diarization input) but the `diarize.py` implementation with pyannote + MediaPipe is post-MVP.
- **No vision in LLM analysis** — reel selection is transcript-only today. Visual-only moments (gestures, reactions) may be missed.
- **No resumability within a stage** — we cache between stages but don't checkpoint inside long transcription runs. Fine for videos under ~1 hour.
- **Homebrew ffmpeg on macOS often lacks libass** — we render subtitles in Python instead of via the `ass` filter, so this isn't a blocker. Tradeoff: slower than filter-based rendering.

Full roadmap in `.cursor/plans/plan.md`.
