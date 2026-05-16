# PodClipper — Project Context for Claude

> Local-first pipeline that turns long-form podcast videos into short
> vertical (9:16) reels for TikTok / Reels / Shorts. Whisper transcribes,
> Claude picks reel-worthy moments, YOLO + MediaPipe lock on the active
> speaker, OpenCV/FFmpeg renders the crop, and PIL burns karaoke captions.

Repo: `github.com/LoukikNaik/PodClipper` · Live demo: `podclipper.loukik.dev`

## Read this before naming or renaming anything

`docs/ubiquitous-language.md` is the single source of truth for every named
thing in this repo (entity, state, worker, action, concept). The contract:

1. **Look it up before naming.** About to call something "the scheduled
   item card"? Check the table first — there may already be a term for it.
2. **Add before use.** Introducing a new concept? Write the row in the
   glossary first, then use the name in code.
3. **Update on rename.** Rename in code → rename in the table in the
   *same commit*.

If a name isn't in the table and you're tempted to use it, that's a signal
to add it (or pick an existing term). The glossary is alphabetized, flat,
one row per name, no sections. Keep it that way.

## Pipeline (high level)

```
Source video
  ↓  ingest             ffprobe metadata
  ↓  audio              ffmpeg → 16 kHz mono WAV
  ↓  transcribe (1st)   faster-whisper base, parallel chunks
  ↓  analyze            Claude picks reel-worthy clips (JSON list)
  │
  ↓  per clip:
  │     extract         ffmpeg cut with ±2 s pad
  │     detect          YOLOv8 person bboxes + MediaPipe face attribution
  │     transcribe (2)  faster-whisper large-v3 (cached to words.json)
  │     shot-classify   per-frame single vs wide (≥2 people)
  │     crop            shot-aware:
  │                       single  → 9:16 follow-the-speaker
  │                       stacked → two 9:8 panels, one person each
  │                       (pose-anchored, body-IoU lock, lerp transitions)
  │     subtitles       PIL karaoke word highlight + fading title
  │     evaluate        LLM-as-judge scorecard + publish/review/skip verdict
  │
  ↓  outputs/<timestamp>/reel_NN_<slug>.mp4
```

## Module map

| File | Role |
|---|---|
| `main.py` | CLI entry point. Loads `.env`, parses flags, calls `run_pipeline`. |
| `src/pipeline.py` | Orchestrator. Holds the per-clip loop and the `crop.mode` branch. |
| `src/ingest.py` | ffprobe wrapper, returns `VideoMeta`. |
| `src/audio.py` | Whole-video audio extraction. |
| `src/transcribe.py` | Whisper 1st/2nd-pass + `transcribe_second_pass_cached` (JSON cache). |
| `src/transcribe_cleanup.py` | LLM post-pass to fix Whisper mis-spellings / transliterate non-Latin. |
| `src/analyze.py` | LLM clip selection — reads transcript, returns `Clip[]`. |
| `src/llm/` | Provider abstraction. `claude_cli.py` (default, configurable timeout), `anthropic_api.py`. |
| `src/detect.py` | YOLOv8 + MediaPipe BlazeFace. **Two entry points:** `detect_humans_per_frame` (single primary, legacy) and `detect_humans_all_per_frame` (all persons + face flags, used by shot-aware path). |
| `src/timeline.py` | Two responsibilities: (1) legacy `build_speaker_timeline` for the single-crop path, (2) `classify_wide_shot_frames` for the new shot-aware path. |
| `src/crop.py` | Two renderers: legacy `smart_crop_916` (single-panel timeline-driven) and new `smart_crop_916_stacked` (shot-aware, single ↔ stacked dual-panel). |
| `src/subtitles.py` | Karaoke word-highlight burner + fading title overlay + audio fade-out. |
| `src/evaluate.py` | LLM-as-judge scorer; writes `verdict` + numeric scores into reel sidecar. |
| `src/diarize.py` | **No longer in the hot path.** pyannote.audio + mouth-motion linking; kept for reference / single-locked-camera future use. |
| `src/types.py` | Shared dataclasses: `BBox`, `Word`, `Clip`, `Timeline`, etc. |
| `src/config.py` | YAML loader → `SimpleNamespace` tree. |
| `config/default.yaml` | All knobs. Key: `crop.mode` = `auto` (shot-aware) or `single` (legacy). |

## The shot-aware crop path (current default)

This is the load-bearing recent work. It replaces the old timeline +
diarization-driven crop that had jitter and mis-framing in wide shots.

**Per frame:**

1. YOLO detects all persons in the frame.
2. **Shot classifier** (`classify_wide_shot_frames`) — frame is `wide` iff
   ≥ 2 person bboxes, each shorter than 70 % of source height, separated
   by ≥ 20 % of source width. Temporally smoothed over 15 frames.
3. **Single mode** → one 9:16 crop centered on the largest person.
   **Wide mode** → two 9:8 stacked panels: leftmost person on top,
   rightmost on bottom (each tightly framed on their face + shoulders
   via MediaPipe Pose Landmarker).

**Stability mechanism (key to it not looking like garbage):**

- **Body-IoU lock**: the crop bbox stays locked while the YOLO body bbox
  IoU stays ≥ 0.7 vs the body bbox at the moment we last set the crop.
  Below 0.7 → recompute crop from new pose anchors.
- **Lerp transitions**: when the lock breaks, the rendered crop **lerps**
  toward the new target over 12 frames (~0.4 s @ 30 fps) — each
  lock-break becomes a tiny pan-zoom instead of a 1-frame snap.
- **Miss tolerance**: if YOLO or Pose Landmarker fails for ≤ 15 frames
  in a row, hold the last lock. Bridges brief dropouts.
- **Snap-to-target**: when rendered is within 1 px of target, snap
  exactly (no infinite micro-wobble).

The body-IoU lock matters specifically because the small crop bbox is too
sensitive to pose jitter; using the much larger YOLO body bbox as the
hysteresis signal means only real human movement triggers crop updates,
not landmark noise.

Tunable knobs (`config/default.yaml` → `crop:`):
- `stacked_iou_threshold` (0.70) — lower = more lock retention
- `stacked_transition_frames` (12) — higher = smoother but slower settle
- `stacked_miss_tolerance` (15) — bridge this many bad frames
- `stacked_snap_px` (8) — snap crop dims to multiples of this
- `shot_sep_frac` (0.20), `shot_height_cap_frac` (0.70),
  `shot_smooth_window_frames` (15) — wide-shot detector

## Caching layout

Per source video, intermediates persist under `.cache/<stem>-<hash>/`:

```
.cache/myvideo-a2340ae50a/
  audio.wav                                  # full-video audio
  reel_01_some-title/
    segment.mp4                              # extracted clip
    words.json                               # cached Whisper 2nd pass
    cropped.mp4 / cropped_regen.mp4          # working crop output
```

`regen_crops.py` exploits this cache to re-render reels from cached
segments + words.json without re-running LLM clip selection, audio
extraction, or Whisper. Crop mode follows `cfg.crop.mode`.

## Output

Final reels land at `outputs/<YYYY-MM-DD_HH-MM-SS>/reel_NN_<slug>.mp4`,
each with a `.txt` sidecar containing:
- LLM's `title`, `reason`, `hook_score`
- Source-clip timestamps (`source_start`, `source_end`)
- Evaluation scorecard (`verdict`, `final_score`, `tech_score`,
  `content_score`, per-axis breakdown, freeform feedback)

## Important CLI commands

```bash
# Full pipeline, default config (shot-aware crop mode)
python main.py path/to/video.mp4

# Common flags
python main.py video.mp4 --output-dir outputs --max-clips 5 -v

# Force legacy single-crop path
# (edit cfg.crop.mode to "single" in config, or copy + override)

# Re-render reels from cache, optionally filtered to a previous run's sidecars
python regen_crops.py .cache/<video>-<hash> outputs/regen_run [outputs/<orig>]

# Standalone debug (kept at root — active tools)
python debug_detect_clip.py outputs/segment.mp4         # YOLO + face overlay
python diag_frame.py path/to/segment.mp4 1380 40        # per-frame bbox/cluster dump

# One-off experiments (moved under experiments/; results encoded in
# this doc — re-run only if a hypothesis comes back)
python experiments/stacked_crop_test.py path/to/video.mp4   # shot-aware prototype (now in src/crop.py)
python experiments/diarize_compare.py audio.wav             # pyannote vs Resemblyzer — both lost
python experiments/mouth_speaker_test.py video.mp4          # mouth-motion speaker ID — negative result
python experiments/highlight_faces.py in.mp4 out.mp4        # MediaPipe face-bbox overlay utility
```

## LLM provider

Configured at `cfg.llm.provider`:
- `claude_cli` (default) — shells out to `claude -p`. Needs Claude Code
  CLI on PATH. Timeout is `cfg.llm.claude_cli.timeout_seconds` (default 900).
- `anthropic_api` — uses `ANTHROPIC_API_KEY` from `.env`.

## Optional: speaker diarization (legacy single-mode only)

For single-locked-camera podcasts where the editor never cuts between
speakers, set `crop.mode: single` AND `diarize.enabled: true`. Requires:
- `HF_TOKEN` in `.env`
- Accept terms at `huggingface.co/pyannote/speaker-diarization-3.1`

Diarization is **not** invoked in the default `auto` mode — the
shot-aware stacked layout sidesteps the "who's speaking right now?"
problem by showing both people whenever the source goes wide.

## Landing page

`landing/` — Vite + React. Deploys to `podclipper.loukik.dev` via
`.github/workflows/deploy-landing.yml` (push to `main`, path-filtered to
`landing/**`). Demo reels are bundled at `landing/public/videos/`
(60 MB, exception added in `.gitignore` to track them).

## Known limitations / non-obvious gotchas

- **Source resolution affects shot classification.** At 360p (yt-dlp
  format 18 fallback when JS challenge solver isn't available),
  MediaPipe FaceLandmarker fails on the small faces in wide shots. The
  shot detector deliberately uses pure bbox geometry — no face-attribution
  requirement — so it still works at low res.
- **Whisper's `int8` on CPU** is the default and works on Apple Silicon
  without GPU config. For CUDA, switch to `int8_float16` / `float16` in
  config.
- **Audio doesn't leave the machine — transcript does.** The `100% Local`
  framing on the landing page refers to audio. LLM clip selection sends
  the transcript to Claude (CLI or API).
- **The diarization path is brittle.** Pyannote-3.1 frequently merges or
  splits speakers on conversational podcast audio, and even when correct
  it provides no signal for which side of the screen the speaker is on.
  The shot-aware path was built specifically to retire it for the typical
  multi-camera podcast use case.

## Where to look first when something breaks

| Symptom | Likely file(s) |
|---|---|
| Wrong clip selected | `prompts/reel_detector.txt`, `src/analyze.py` |
| Wrong person being cropped | `src/detect.py` (face attribution), `src/timeline.py` (shot classifier) |
| Vibrating / jumpy stacked panel | `src/crop.py` (`smart_crop_916_stacked`) — tweak `stacked_iou_threshold` / `stacked_transition_frames` |
| Crop misses head | `src/crop.py` (`_bbox_from_pose_anchors`) — tweak the 1.5/3.5 head-height multipliers |
| Subtitles look off | `src/subtitles.py`, `config.subtitles.*` |
| Reel marked skip | `src/evaluate.py`, sidecar `.txt` has the LLM's full feedback |
| Pipeline times out at LLM | `cfg.llm.claude_cli.timeout_seconds` |
| HF / pyannote errors | only relevant in `crop.mode = single` + diarize.enabled |

## Recent architectural shift (May 2026)

Moved from **"follow the speaker"** (timeline + diarization + cluster
locking) to **"follow the editor"** (shot-aware single + stacked). The
new design mirrors the editor's cuts rather than trying to override them.
This eliminated three classes of bugs:
- Back-of-head wrongly attributed faces in cluster-derived timelines
- Single-cluster crop drifting into mics/empty space during wide shots
- Diarization quirks (1-speaker output on monologue clips, similar-voice
  merging) leaking into crop framing

The legacy single-crop + timeline path is still present and reachable via
`cfg.crop.mode = single` for backward compatibility / experiments.

## Comment style (load-bearing — please respect)

Comments in this codebase are kept deliberately minimal. The rule:

- **Each function gets ONE docstring of 1–2 lines max.** State the contract,
  not the implementation. No multi-paragraph rationale.
- **No inline comments explaining what the code does.** Well-named identifiers
  already say what; comments rot, identifiers don't.
- **No change-log comments.** "Added for X bug", "was 2.0, lowered to 0.15
  after Huberman test", "May 2026 refactor" — that history belongs in `git
  log` and PR descriptions, not in source.
- **Keep comments only when the WHY is non-obvious and load-bearing.**
  Examples: "ASS alpha: 0=opaque, 255=transparent — invert for Pillow",
  "CPU only supports int8 in CTranslate2", "raw_decode stops at end of
  first valid value so trailing prose is tolerated". A future reader
  would be confused without it.
- **DEPRECATED markers stay** as 1-line tags on the function/module that
  also says what replaced it. Don't expand them into rationale.
- **Module docstrings: one line** that says what the module is for.
  Pipeline diagrams, stage lists, "Approach:" / "Failure modes:" sections
  — all belong in CLAUDE.md, not at the top of every file.

When editing existing code, **don't reintroduce explanatory comments.** If
you're tempted to add one, ask: would removing this confuse a reader who
already understands the surrounding code? If no → don't add it.
