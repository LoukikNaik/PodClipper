# PodClipper

Turn a long-form podcast, talk or interview into vertical 9:16 reels — locally, on your machine. Whisper transcribes, Claude picks the reel-worthy moments, YOLO + MediaPipe lock on the active speaker, OpenCV crops with smooth pans across shot cuts, and karaoke captions burn in.

🔗 **Live demo:** [podclipper.loukik.dev](https://podclipper.loukik.dev)

## Demo reels

Three reels generated end-to-end from a single ~60-minute podcast episode. Previews loop silently below — **click any tile to open the full MP4 with audio.**

<table>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/LoukikNaik/PodClipper/releases/download/demo-assets-v1/reel_01_confidence-is-an-output-not-input.mp4">
        <img src="docs/demos/reel_01.gif" width="240" alt="Reel 1 preview"/>
      </a>
      <br/><em>"Confidence Is an Output, Not Input"</em>
      <br/><sub><a href="https://github.com/LoukikNaik/PodClipper/releases/download/demo-assets-v1/reel_01_confidence-is-an-output-not-input.mp4">▶ play with audio</a></sub>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/LoukikNaik/PodClipper/releases/download/demo-assets-v1/reel_02_make-anxiety-sit-in-the-back-seat.mp4">
        <img src="docs/demos/reel_02.gif" width="240" alt="Reel 2 preview"/>
      </a>
      <br/><em>"Make Anxiety Sit in the Back Seat"</em>
      <br/><sub><a href="https://github.com/LoukikNaik/PodClipper/releases/download/demo-assets-v1/reel_02_make-anxiety-sit-in-the-back-seat.mp4">▶ play with audio</a></sub>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/LoukikNaik/PodClipper/releases/download/demo-assets-v1/reel_03_how-hanuman-knew-it-was-sita.mp4">
        <img src="docs/demos/reel_03.gif" width="240" alt="Reel 3 preview"/>
      </a>
      <br/><em>"How Hanuman Knew It Was Sita"</em>
      <br/><sub><a href="https://github.com/LoukikNaik/PodClipper/releases/download/demo-assets-v1/reel_03_how-hanuman-knew-it-was-sita.mp4">▶ play with audio</a></sub>
    </td>
  </tr>
</table>

For full-screen playback with audio inline, visit the [live demo site](https://podclipper.loukik.dev).

## Pipeline

```
Video
  ↓ ingest        ffprobe — validate, metadata
  ↓ audio         ffmpeg — extract mono 16kHz WAV
  ↓ transcribe    faster-whisper — fast 1st pass, parallel chunks
  ↓ analyze       LLM — pick reel-worthy clips as JSON
  ↓ per clip:
       ├─ extract       ffmpeg cut ±2s pad
       ├─ detect        YOLOv8 + MediaPipe face attribution
       │                (prefers front-facing person; rejects back-of-head)
       ├─ transcribe 2  faster-whisper large — clean word timestamps
       │                (cached to words.json for fast re-runs)
       ├─ diarize       pyannote.audio + mouth-motion linking
       │                (optional; follows the active speaker in single-camera clips)
       ├─ timeline      x-center clustering with contiguous-run threshold
       │                + per-shot segments for multi-camera edits
       ├─ crop          OpenCV — 9:16, EMA smoothing, look-ahead seed
       │                across segment boundaries (so the crop follows
       │                the speaker through a hard cut)
       ├─ subtitles     PIL + OpenCV — karaoke word highlight + fading title
       └─ evaluate      LLM-as-judge — publish / skip verdict + scorecard
  ↓ outputs/<timestamp>/reel_NN_<title>.mp4
```

## Setup

Requires Python 3.10+ and `ffmpeg` + `ffprobe` on PATH.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### LLM provider

Pick one via `config/default.yaml` or `--llm-provider`:

- **Claude CLI** (default) — install Claude Code so the `claude` binary is on PATH.
- **Anthropic API** — set `ANTHROPIC_API_KEY` in your environment.

### Optional: speaker diarization

For single-camera interview footage where the editor didn't cut between speakers, enable pyannote-driven follow-the-speaker:

1. Accept terms at [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).
2. Add `HF_TOKEN=hf_xxx` to a `.env` file in the repo root (see `.env.example`).
3. `diarize.enabled: true` in `config/default.yaml`.

The pipeline runs without it — it just won't follow speaker turns visually in single-cam edits.

### Models

- `yolov8n.pt` (6 MB) — auto-downloaded by ultralytics on first run.
- MediaPipe face detector + landmarker — auto-downloaded to `.cache/`.
- Whisper models (`base`, `large-v3`) — auto-downloaded from HuggingFace.
- Default compute type is `int8` on CPU (works on Apple Silicon out of the box). For CUDA, set `float16` / `int8_float16` in config.

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
  --debug-detect \
  --debug-crop \
  -v
```

Run `python main.py --help` for the full list.

Outputs land in `outputs/<timestamp>/reel_NN_<slugified-title>.mp4` alongside a `.txt` sidecar with the LLM's reasoning, source timestamps, and the LLM-judge scorecard.

### Iterating on crop changes only

If you've already run the full pipeline and just want to re-render reels with updated detect / timeline / crop logic (without re-running Whisper or the LLM):

```bash
python regen_crops.py .cache/<video_stem>-<hash> outputs/regen_run sidecar_dir/
```

This reuses cached `segment.mp4` and `words.json` and only re-runs detect + timeline + crop + subtitle burn.

## Configuration

Everything is knob-tunable in `config/default.yaml`:

- `transcribe.first_pass.model` / `second_pass.model` — Whisper model sizes
- `analyze.min/max_clip_seconds`, `target_clips` — clip length and count hints to the LLM
- `detect.sample_every_n_frames` — bump for speed, drop to 1 for smoothest tracking
- `detect.face_aware` — face-attribution gate that prefers front-facing speakers
- `crop.smoothing_alpha` — lower = smoother panning
- `crop.min_segment_dwell_seconds` — min time between hard cuts
- `diarize.enabled` — multi-speaker follow-the-speaker (needs `HF_TOKEN`)
- `evaluate.publish_threshold` / `skip_threshold` — LLM-judge gating
- `subtitles.font_size`, `margin_v`, `primary_color`, `highlight_color` — styling

## Caching + resume

Intermediate artifacts persist under `.cache/<video_stem>-<hash>/`:

```
.cache/mytalk-a2340ae50a/
  audio.wav                    # whole-video extracted audio
  reel_01_i-quit-my-job/
    segment.mp4                # extracted clip + pad
    words.json                 # cached 2nd-pass Whisper words
    cropped.mp4                # after 9:16 smart crop
```

Re-running reuses these — fast iteration on subtitles, prompt or crop. Pass `--no-cache` to force a rebuild.

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
  diarize.py  subtitles.py     transcribe_cleanup.py
  evaluate.py pipeline.py
landing/                      # Vite + React landing page
.github/workflows/            # GH Pages deploy for the landing
docs/demos/                   # README GIFs
```

## Deployment

The landing page deploys to **podclipper.loukik.dev** via GitHub Pages on every push to `main` that touches `landing/**`. See [`.github/workflows/deploy-landing.yml`](.github/workflows/deploy-landing.yml).

## Known limitations

- **No vision in LLM analysis** — reel selection is transcript-only today. Purely visual moments (gestures, reactions) may be missed.
- **No resumability within a stage** — caching is between stages, not inside a long Whisper run. Fine for sources under ~1 hour.
- **Homebrew ffmpeg on macOS often lacks libass** — subtitles render in Python (PIL + OpenCV) instead of via the `ass` filter, so this isn't a blocker. Tradeoff: slower than filter-based rendering.

## License

MIT — see [LICENSE](LICENSE) if present, otherwise treat as MIT.
