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
  ↓  transcribe (1st)   mlx-whisper base, per-clip language auto-detect
  ↓  analyze            Claude picks reel-worthy clips (JSON list)
  │
  ↓  per clip:
  │     extract         ffmpeg cut with ±2 s pad
  │     detect          YOLOv8 person bboxes + MediaPipe face attribution
  │     transcribe (2)  mlx-whisper large-v3 (VAD + garble-retry, cached)
  │     shot-classify   per-frame single vs two-shot (≥2 real people)
  │     crop            shot-aware:
  │                       single  → 9:16 follow-the-speaker
  │                       stacked → two 9:8 panels, one person each
  │                       (pose-anchored, body-IoU lock, lerp + snap-on-cut)
  │                     optional --intro-zoom punch-in→pull-out opener
  │     subtitles       karaoke (classic) or 1-2-word pop overlay
  │                     cfg.subtitles.style picks the renderer
  │     music           optional --music: LLM-matched ducked bed from library
  │     evaluate        LLM-as-judge scorecard + publish/review/skip verdict
  │
  ↓  outputs/<timestamp>/reel_NN_<slug>.mp4
```

## Module map

| File | Role |
|---|---|
| `src/podclipper/main.py` | CLI entry point (`podclipper = "podclipper.main:main"` in pyproject.toml). Loads `.env`, parses flags, calls `run_pipeline`. |
| `src/podclipper/pipeline.py` | Orchestrator. Holds the per-clip loop and the `crop.mode` branch. |
| `src/podclipper/ingest.py` | ffprobe wrapper, returns `VideoMeta`. |
| `src/podclipper/audio.py` | Whole-video audio extraction. |
| `src/podclipper/transcribe.py` | Whisper 1st/2nd-pass + `transcribe_second_pass_cached` (JSON cache). Engine behind `cfg.transcribe.engine` = `mlx` (default, Apple-Silicon GPU) \| `faster`. Auto-detects language per clip; VAD + `condition_on_previous_text=false` guard against hallucination; garble-retry re-runs with `fallback_languages` when auto-detect returns a suspiciously sparse transcript. |
| `src/podclipper/transcribe_cleanup.py` | LLM post-pass to fix Whisper mis-spellings / transliterate non-Latin. |
| `src/podclipper/analyze.py` | LLM clip selection — reads transcript, returns `Clip[]`. |
| `src/podclipper/llm/` | Provider abstraction. `claude_cli.py` (default, configurable timeout), `litellm_provider.py` (unified gateway: Anthropic, OpenAI, Gemini, Groq, Ollama, ...). |
| `src/podclipper/detect.py` | YOLOv8 + MediaPipe BlazeFace. **Two entry points:** `detect_humans_per_frame` (single primary, legacy) and `detect_humans_all_per_frame` (all persons + face flags, used by shot-aware path). |
| `src/podclipper/timeline.py` | Two responsibilities: (1) legacy `build_speaker_timeline` for the single-crop path, (2) `classify_wide_shot_frames` for the shot-aware path — two-shot = ≥2 people above a min-size **floor** (`shot_min_person_frac`, ignores tiny background people) separated by ≥`shot_sep_frac`, with an optional min-dwell debounce. |
| `src/podclipper/crop.py` | Two renderers: legacy `smart_crop_916` (single-panel timeline-driven) and `smart_crop_916_stacked` (shot-aware, single ↔ stacked dual-panel; body-IoU lock + lerp + **snap-on-cut**, optional intro-zoom, `debug_out` horizontal overlay). |
| `src/podclipper/music.py` | Optional background music: `load_library` (manifest), `select_track` (LLM vibe-match → random fallback), `mix_music` (ducked bed, loops the section to fill the reel). Behind `--music`. |
| `src/podclipper/subtitles.py` | Two renderers: `_burn_classic` (karaoke word highlight + fading title + audio fade-out) and `_burn_pop` (1–2 huge sheared words, alternating highlight color). `burn_subtitles` dispatches by `cfg.subtitles.style` (`classic` \| `pop`). |
| `src/podclipper/evaluate.py` | LLM-as-judge scorer; writes `verdict` + numeric scores into reel sidecar. |
| `src/podclipper/diarize.py` | **No longer in the hot path.** pyannote.audio + mouth-motion linking; kept for reference / single-locked-camera future use. |
| `src/podclipper/types.py` | Shared dataclasses: `BBox`, `Word`, `Clip`, `Timeline`, etc. |
| `src/podclipper/config.py` | YAML loader → `SimpleNamespace` tree. |
| `src/podclipper/config/default.yaml` | All knobs (bundled inside the wheel via `importlib.resources`). Key: `crop.mode` = `auto` (shot-aware) or `single` (legacy). |
| `src/podclipper/config/__init__.py` | `load_config(path)` for explicit `-c` flag; `load_default_config()` for the packaged file (used when no `-c` given). |
| `src/podclipper/prompts/` | 5 system prompts (`reel_detector`, `reel_refiner`, `trailer_picks/refiner/evaluator`) loaded via `load_prompt(name)` — also bundled in the wheel. |

## The shot-aware crop path (current default)

This is the load-bearing recent work. It replaces the old timeline +
diarization-driven crop that had jitter and mis-framing in wide shots.

**Per frame:**

1. YOLO detects all persons in the frame.
2. **Shot classifier** (`classify_wide_shot_frames`) — frame is a two-shot
   iff ≥ 2 people **taller than `shot_min_person_frac` (0.20) of source
   height** (a size *floor* that ignores tiny background people/posters) and
   separated by ≥ 20 % of width. Temporally smoothed (15 frames), optional
   min-dwell debounce. NB: this used to be a size *cap* ("both people small =
   wide establishing shot"), which wrongly read seated podcast guests filling
   70–90 % of frame as close-ups; a floor matches "2 real people → stack".
3. **Single mode** → one 9:16 crop centered on the largest person.
   **Wide mode** → two 9:8 stacked panels: leftmost person on top,
   rightmost on bottom (each tightly framed on their face + shoulders
   via MediaPipe Pose Landmarker).

**Stability mechanism (key to it not looking like garbage):**

- **Body-IoU lock**: the crop bbox stays locked while the YOLO body bbox
  IoU stays ≥ 0.7 vs the body bbox at the moment we last set the crop.
  Below 0.7 → recompute crop from new pose anchors.
- **Lerp transitions**: when the lock breaks, the rendered crop **lerps**
  toward the new target over ~16 frames — each lock-break becomes a tiny
  pan-zoom instead of a 1-frame snap.
- **Snap-on-cut**: but when the body-IoU collapses below `snap_cut_iou`
  (0.15) — a source *camera cut*, person jumped — snap instantly instead
  of gliding (a glide after a hard cut reads as shake).
- **Miss tolerance**: if YOLO or Pose Landmarker fails for ≤ 15 frames
  in a row, hold the last lock. Bridges brief dropouts.
- **Snap-to-target**: when rendered is within 1 px of target, snap
  exactly (no infinite micro-wobble).

The body-IoU lock matters specifically because the small crop bbox is too
sensitive to pose jitter; using the much larger YOLO body bbox as the
hysteresis signal means only real human movement triggers crop updates,
not landmark noise.

Tunable knobs (`src/podclipper/config/default.yaml` → `crop:`):
- `stacked_iou_threshold` (0.50) — lower = more lock retention
- `snap_cut_iou` (0.15) — below this overlap = a cut → snap, don't glide
- `stacked_transition_frames` (16) — higher = smoother but slower settle
- `stacked_miss_tolerance` (15) — bridge this many bad frames
- `stacked_snap_px` (8) — snap crop dims to multiples of this
- `shot_sep_frac` (0.20), `shot_min_person_frac` (0.20),
  `shot_smooth_window_frames` (15), `shot_min_dwell_frames` (0),
  `shot_height_cap_frac` (1.0 = off) — two-shot detector
- `intro_zoom.{enabled,duration_seconds,max_zoom}` — the --intro-zoom opener

## The comedy / single-performer path (optional)

`--comedy` (or `cfg.crop.comedy_mode`) is for stand-up / single-performer
footage where splitting into stacked panels is wrong and audience members
must never be cropped to. It changes the shot-aware path in two ways:

1. **Never stacks.** The pipeline forces `is_wide` to all-False, so
   `smart_crop_916_stacked` only ever renders the single 9:16 panel — no
   dual-panel layout regardless of how many people are on screen.
2. **Locks on the performer, ignores audience.** `_pick_performer` scores each
   person (above the `comedy_min_person_frac` 0.30 size floor) per frame and
   takes the max. The score exploits how stand-up is shot: the performer is
   **spotlit** while the audience sit in **shadow**, and the performer stands
   **full-body high** in frame while the audience are **short silhouettes low
   in the foreground**. So `score = comedy_brightness_weight·(mean bbox
   luma) + comedy_top_weight·(how high the bbox top sits) + 0.3·(bbox
   height)`. Brightness is the dominant signal (audience in shadow score near
   zero). If nobody clears the floor — e.g. an audience reaction cutaway — it
   **holds the last performer lock** (via the existing miss-tolerance) instead
   of jumping to an audience face.

Non-comedy runs are bit-identical (the whole thing is gated on `comedy_mode`).
Knobs (`crop:`): `comedy_mode`, `comedy_min_person_frac`,
`comedy_brightness_weight`, `comedy_top_weight`.

**Known limit:** a *long* audience cutaway (longer than
`stacked_miss_tolerance` frames) will eventually fall back to a centered crop
of whatever's on screen — true fix would need source scene-cut detection.
The heuristic targets the common single-camera-on-stage case.

## The pop subtitle path (optional)

Alternative to the classic karaoke renderer for TikTok/Reels-style
captions — 1–2 huge italic-sheared words at a time, active word in a
cycling highlight color (red → green by default) with a slight scale-up,
heavy black outline. Enabled per-run with `--subtitle-style pop` or
per-config with `cfg.subtitles.style: "pop"`. Default is `classic`.

**Per popup:**

1. `generate_pop_popups` groups words into 1–N word popups, flushing on
   the `max_words_per_popup` cap, sentence-ending punctuation, or any
   inter-word gap > `max_gap_seconds`.
2. `active_word_index(popup, t)` returns which word in the popup is
   currently being spoken.
3. `_render_pop_onto_frame` draws the popup onto a small canvas, applies
   a horizontal shear via affine transform, and uniformly scales it
   down if the sheared width exceeds 92% of the frame width (overflow
   guard — keeps long words like "CONFIDENCE" from clipping past the
   edges).
4. `pick_highlight_color(popup_idx, colors)` cycles through
   `cfg.subtitles.pop.highlight_colors` by popup index.

Tunable knobs (`src/podclipper/config/default.yaml` → `subtitles.pop:`):
- `font_size` (140), `outline_width` (6) — both larger than classic.
- `highlight_colors` (list of ASS colors, cycled per popup) — defaults
  to `[red, green]`.
- `scale` (1.08) — active-word scale-up factor.
- `shear_deg` (8.0) — italic skew applied to all words in the popup.
- `max_words_per_popup` (2), `max_gap_seconds` (0.6) — popup grouping.
- `y_position_frac` (0.72) — vertical anchor as fraction of frame
  height (`0` = top, `1` = bottom; `0.72` = TikTok lower-middle zone).

The classic path stays bit-identical when `style == "classic"` —
`burn_subtitles` is a thin dispatcher to `_burn_classic` (body unchanged
from before the pop feature) or `_burn_pop`.

## The intro-zoom opener (optional)

`--intro-zoom` (or `cfg.crop.intro_zoom.enabled`) makes each reel open
**punched-in on the subject and ease out** to the normal framing over
`duration_seconds` (0.7) via an eased center digital zoom — a scroll-stopper
"dopamine" hook. Applied in both the single and stacked crop paths
(`_apply_intro_zoom` on the composed 1080×1920 frame). Off by default →
existing runs bit-identical.

## The background-music path (optional)

`--music` (or `cfg.music.enabled`) lays a ducked music bed under each reel.
The **library is curated offline** (by Claude Code, not the runtime): songs
downloaded, their genuine hit-sections found, and each tagged with a vibe
description — persisted in `music/library.json` (both audio **and** the
manifest are gitignored, since it catalogs real copyrighted tracks;
`music/library.json.example` is the tracked schema template — copy it to
`library.json` and populate). At runtime the pipeline:

1. `load_library` reads the manifest (skips `disabled` songs/sections).
2. `select_track` — a **separate LLM call** that **scores every enabled
   *section* 0–10** for the reel (transcript+title) in one shot and picks the
   **argmax** (`music.min_score`, default 5.0; below it → random section).
   Scoring is per-section (`_candidates` flattens song→sections), so the
   selector judges the actual passage — its own vocals + context — not the
   whole song. Below `min_score` or on parse failure → random section.
3. `mix_music` — trims the chosen section and **loops it** to fill the reel
   (never bleeds past the section into unwanted parts), then sidechain-ducks
   it under the speech (`gain`, `duck_ratio`, `duck_release`, `fade_seconds`).

Section anchoring for vocal songs uses **web lyrics + mlx-whisper**: identify
the hook line, then a cut→transcribe-the-clip→correct loop pins its exact
onset (defeats whole-song timestamp drift). Instrumentals use energy/beat
section detection. Tooling lives under `dev/` (`analyze_music.py`,
`find_hook_auto.py`, `apply_music_to_reels.py`).

Manifest shape — one entry per song, `sections[]` each with
`{id, start, end, lyrics, vocals, context, description}`; `method` = `lyric`
(onset pinned to a verified lyric) or `acoustic` (energy/beat). The signals the
selector LLM reads are per-section: `vocals` (the lines actually sung in that
window, from mlx-whisper of the cut section), `context` (what that passage is
about, grounded in its own vocals + provenance), and `description` ("vibe +
when to use"). Grounding on the section's real vocals — not the whole song —
is deliberate: a 30–50 s window may be one verse or an instrumental bridge with
a different tone than the song overall (e.g. `chaudhary`'s section is vocal, not
the instrumental its old label claimed).

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

Install first: `pip install -e .` (or `pip install podclipper` after PyPI release).

```bash
# Full pipeline, packaged default config (shot-aware crop mode)
podclipper path/to/video.mp4
# equivalent: python -m podclipper path/to/video.mp4

# Common flags
podclipper video.mp4 --llm-provider litellm --output-dir outputs --max-clips 5 -v

# Override config (pass a FULL yaml — no partial merging)
podclipper video.mp4 -c my-overrides.yaml

# Pop subtitle style (TikTok-style 1-2 huge sheared words at a time)
podclipper video.mp4 --subtitle-style pop

# Intro zoom (punch-in → pull-out opener) + background music (LLM-matched bed)
podclipper video.mp4 --intro-zoom --music

# Comedy / single-performer: never stack, lock on the performer, ignore audience
podclipper video.mp4 --comedy

# Only consider the first N minutes for clip selection (no re-encode of the rest)
podclipper video.mp4 --limit-minutes 20

# Transcription engine (mlx is default; faster-whisper on non-Apple / CUDA)
podclipper video.mp4 --whisper-engine faster

# Force legacy single-crop path
# (set cfg.crop.mode to "single" in your override config)

# Re-render reels from cache, optionally filtered to a previous run's sidecars
python dev/regen_crops.py .cache/<video>-<hash> outputs/regen_run [outputs/<orig>]

# Standalone debug (under dev/ — not installed with the package)
python dev/debug_detect_clip.py outputs/segment.mp4         # YOLO + face overlay
python dev/diag_frame.py path/to/segment.mp4 1380 40        # per-frame bbox/cluster dump

# One-off experiments (moved under experiments/; results encoded in
# this doc — re-run only if a hypothesis comes back)
python experiments/stacked_crop_test.py path/to/video.mp4   # shot-aware prototype (now in src/podclipper/crop.py)
python experiments/diarize_compare.py audio.wav             # pyannote vs Resemblyzer — both lost
python experiments/mouth_speaker_test.py video.mp4          # mouth-motion speaker ID — negative result
python experiments/highlight_faces.py in.mp4 out.mp4        # MediaPipe face-bbox overlay utility
```

## LLM provider

Configured at `cfg.llm.provider`:
- `claude_cli` (default) — shells out to `claude -p`. Needs Claude Code
  CLI on PATH. Timeout is `cfg.llm.claude_cli.timeout_seconds` (default 900).
- `litellm` — unified SDK. Vendor is picked from `cfg.llm.model`
  (`anthropic/claude-sonnet-4-5`, `openai/gpt-5-mini`, `ollama/llama3`,
  `groq/llama-3-70b`, ...). API key env var follows LiteLLM's convention
  (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...). For local/self-hosted
  models, set `cfg.llm.litellm.api_base`.

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
- **mlx-whisper is the default engine** (Apple-Silicon GPU) and is markedly
  better than faster-whisper on accented/sung/code-switched audio. Its first
  pass runs **sequentially** — the numba backend isn't thread-safe and aborts
  under the ThreadPool. Use `--whisper-engine faster` on non-Apple / CUDA
  (that path keeps the parallel chunks and the `int8` compute types).
- **Language: auto-detect per clip, don't force globally.** Forcing `--language
  en` makes Whisper *translate* Hindi to English; forcing `hi` garbles English.
  Auto-detect keeps each clip in its own language; the **garble-retry** safety
  net re-runs with `fallback_languages` only when auto returns a suspiciously
  sparse transcript (the accented-English-heard-as-Hindi failure). The old
  episode-level language pinning was removed — it broke odd-language-out clips.
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
| Wrong clip selected | `prompts/reel_detector.txt`, `src/podclipper/analyze.py` |
| Wrong person being cropped | `src/podclipper/detect.py` (face attribution), `src/podclipper/timeline.py` (shot classifier) |
| Vibrating / jumpy stacked panel | `src/podclipper/crop.py` (`smart_crop_916_stacked`) — tweak `stacked_iou_threshold` / `stacked_transition_frames` |
| Crop misses head | `src/podclipper/crop.py` (`_bbox_from_pose_anchors`) — tweak the 1.5/3.5 head-height multipliers |
| Subtitles look off | `src/podclipper/subtitles.py` — check `cfg.subtitles.style` first to know which renderer is in play (`_burn_classic` vs `_burn_pop`); knobs under `config.subtitles.*` and `config.subtitles.pop.*` |
| Reel marked skip | `src/podclipper/evaluate.py`, sidecar `.txt` has the LLM's full feedback |
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
