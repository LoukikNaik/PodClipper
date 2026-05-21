# Backlog

Things we know we want to revisit but aren't blocking right now.

## LLM token-limit + batching audit

**What:** Audit every place the pipeline sends data to an LLM and make sure
we don't bust the model's input or output token limits on long sources,
and add batching/chunking where the risk is real.

**Where to look:**

| LLM call site | File | Risk surface |
|---|---|---|
| Reel selection | `src/podclipper/analyze.py::analyze_for_reels` | Sends full first-pass transcript. ~10-12K tokens per hour of audio. Multi-hour episodes could approach limits. |
| Transcript cleanup | `src/podclipper/transcribe_cleanup.py::cleanup_words` | Already enforces `max_tokens_per_call: 2000` — verify it actually batches and doesn't silently truncate. |
| Reel title rewrite | `src/podclipper/analyze.py::_rewrite_title_with_llm` | Tiny per-call; safe. |
| **Reel content evaluator** | `src/podclipper/evaluate.py::evaluate_content` | Sends one reel's title + transcript per call. ~250-500 tokens. Safe today, but if we ever batch multiple reels into one call to save round-trips, mind the limit. |
| Trailer pick selection | `src/podclipper/trailer.py::pick_quotables` | Sends full first-pass transcript. Same risk as reel selection. |
| Trailer bounds refiner | `src/podclipper/trailer.py::refine_cut_bounds_with_llm` | Per-pick ±10s word window. Small. Safe. |
| **Trailer evaluator** | `src/podclipper/evaluate.py::evaluate_trailer` | 4-5 picks worth of sentences + durations. Tiny. Safe today. |

**What to do:**

1. Add a sanity-check helper `estimate_tokens(text) -> int` and log a warning
   before any call where the estimate is within 20% of the provider's input cap.
2. For the long-transcript callers (reel selection + trailer pick selection):
   chunk the transcript into ~30-minute windows, pick within each window,
   then run a second meta-LLM call to select across the per-window picks for
   thematic coherence and to enforce the 4-5 hard cap.
3. For the evaluators: if we ever batch multiple reels per call (saves
   wall-clock and money on long runs), add a per-batch token check that
   splits the batch if it would exceed the input cap.
4. Provider limits to encode somewhere config-readable:
   - Anthropic Claude (current default models): 200K input / 8K output
   - Claude CLI: same underlying model, no separate limit
   Output limit hasn't been a real concern — every JSON response is <1K tokens.

Not blocking — leaving here until a long-source run actually surfaces a
truncation or context-overflow error.

## Trailer mode: dramatic music between picks

**What:** Right now the gap between picks is `color=black` video + an
`anullsrc` silent audio source. That gives the trailer a deliberate beat
but feels clinical. Real trailers use a music bed (often a tense build
that hits on each cut) to glue the picks together emotionally.

**Where to look:**

`src/podclipper/trailer.py::concat_with_black_gaps` — the filter_complex builder
that currently splices each cropped clip + a `color=black` + `anullsrc`
gap. The audio gap is the substitution point.

**Two implementation paths to consider when we come back to this:**

1. **Stinger per gap** (simplest): one short 0.6-1.0 s music hit per
   transition. Bring-your-own asset under `assets/trailer_stinger.mp3`
   referenced from `cfg.trailer.gap_audio`. During gaps, replace
   `anullsrc` with this asset instead of silence. No mixing with
   speech audio — keeps the pipeline simple.

2. **Continuous music bed** (richer, more work): one music track plays
   under the whole trailer at low volume (e.g. -18 dB), ducks to -28 dB
   when the speaker is talking, returns to -18 dB during the black gaps.
   Needs `amix` + `sidechaincompress` in the filter graph, or a separate
   ffmpeg pass post-concat. Way nicer effect; way more knobs.

**Other things to nail down before building this:**

- Licensing: ship one CC0/CC-BY track in `assets/` for default, let users
  swap via `cfg.trailer.gap_audio` (path to mp3/wav).
- Bring-your-own per-source: maybe `--trailer-music path.mp3` CLI flag
  for one-off overrides.
- Audio fade-in/out on the music itself at the trailer's very start/end
  so it doesn't pop in.
- Per-gap volume curve: for path (1), a tiny fade-in / fade-out on the
  stinger so it doesn't clip the speech on either side.

Defer until we have at least one CC-licensed stinger asset to test with.

## Reels mode: background music bed

**What:** Standard 9:16 reels (`--mode reels`, the default) currently play
the source audio raw. A subtle music bed underneath the speaker would lift
the production value to "looks edited" without much extra work — same
direction as the trailer music idea above, but applied to each reel
independently.

**Where to look:**

`src/podclipper/subtitles.py::burn_subtitles` — the final stage that muxes audio back
into each cropped reel. The audio mux is the splice point; we'd add a
second audio input (the music bed) and `amix` them with sidechain ducking.
Alternative: do this as a separate pass right before the audio fade-out
mux that already runs at the tail of each reel.

**Sketch of the audio chain:**

  speaker audio   ──┐
                    ├── amix ──► afade=out ──► final reel audio
  music bed     ──┤  (with sidechaincompress to duck music
  (loops if         under speaker by ~10 dB)
   shorter than
   clip)

**Things to figure out:**

- **Selection.** Three options, increasing in complexity:
  (a) One default track in `assets/reel_bed.mp3`, used for all reels.
  (b) Per-config picker: `cfg.subtitles.music_bed` with a small library
      keyed by mood, user-set per source video.
  (c) LLM-driven mood pick: feed the clip transcript to a tiny LLM call
      that returns a mood label, then pick from a library. Most polish,
      most knobs.
- **Volume.** Default music at -22 dB with ducking to -32 dB during speech
  is the safe starting point. Make this config-tunable.
- **Length.** Loop the bed (`-stream_loop -1`) and clip to the reel
  duration via `-shortest` on the output. Need to also fade the music in
  the first 0.3s and out matching the existing 0.6s video fade-out.
- **Genre/mood library.** Same licensing problem as the trailer stinger
  — ship a small CC0/CC-BY default library (4-6 tracks across moods)
  under `assets/music/`, let users swap.

Likely natural to build this alongside the trailer music work — they
share the audio-mixing chain. Tackle trailer first (simpler: gap-only,
no ducking) to validate the asset/licensing flow, then extend to the
full-reel bed.

## Fancy title overlay (typography + color + motion)

**What:** The top-of-frame title card right now is plain — Arial 96pt
white with a black outline over a dark gradient backdrop. Functional
but generic. Real social-media reels lean on bold display typography,
brand-consistent colors, and a touch of motion to land the hook in the
first second.

**Where to look:**

`src/podclipper/subtitles.py::_draw_title_overlay` (or wherever the title is
rendered onto each frame in the first 3.5s) — that's the splice point
for typography/color/animation changes. Knobs already live in
`cfg.subtitles.title_overlay.*` (`font_size`, `color`, `outline_color`,
`outline_width`, `y_offset`, `gradient_enabled`, etc.) — extending the
config schema is straightforward.

**Things to nail down:**

- **Font.** Stop using Arial. Ship a CC-licensed display font under
  `assets/fonts/` (e.g. Inter Display Bold, Anton, Bebas Neue, or an
  SIL-licensed brand-friendly face). Add `cfg.subtitles.title_overlay.font_path`
  pointing at the bundled file so PIL doesn't fall back to system fonts.
- **Color schemes.** Move beyond white-on-dark. Two options:
  (a) Fixed palette per source video, settable in config (e.g. accent
      color = brand color)
  (b) Mood-driven via a tiny LLM call on the reel transcript: returns a
      label (energetic / contemplative / dramatic / informative) and we
      map to a curated palette (warm yellow, cool blue, hot red, etc.)
- **Gradient/glow.** Replace the flat gradient backdrop with a subtle
  radial glow behind the title, or a tasteful drop shadow. Pillow's
  `ImageFilter.GaussianBlur` on an alpha layer is the cheapest path.
- **Motion.** Two easy wins:
  (a) Per-word stagger — letters/words fade in left-to-right over
      ~0.5s instead of the whole title appearing at once.
  (b) A scale-from-0.95 entry so the title "lands" with weight.
  Both are per-frame alpha + transform tweaks in the existing burn loop.
- **Templates.** Eventually let users pick from named templates
  (`minimal`, `mrbeast`, `documentary`, `tiktok`) — each is a preset
  bundle of font + palette + motion params in the config.

## Fancy synced captions (typography + color + animation)

**What:** Karaoke word highlighting is on (white text, yellow current
word, black outline, Arial 72pt). Functional but visually flat compared
to what TikTok/Reels editors ship today.

**Where to look:**

`src/podclipper/subtitles.py::burn_subtitles` — the per-frame caption renderer.
The active-word lookup is `[w for w in words if w.start <= t <= w.end]`;
that's where per-word styling kicks in.

**Things to nail down:**

- **Font.** Same as title overlay — ship a display font under
  `assets/fonts/` (Inter Bold or similar tight sans-serif). Stop relying
  on system Arial.
- **Per-word color/animation.** Right now the active word turns yellow.
  Add options:
  - **Scale pop**: the active word renders at 1.1× scale and ramps back
    to 1.0 over its duration. Adds energy without distracting.
  - **Two-tone**: alternate active-word color across words to keep the
    eye moving (e.g. yellow → cyan → yellow → cyan).
  - **Bounce**: tiny y-translate on word start, easing back to baseline.
- **Sentence-level chunking colors.** Currently every word in a line
  shares one base color. Could color each sentence differently for
  longer reels — gives visual rhythm beyond just the active-word
  highlight.
- **Backdrop options.** A semi-opaque rounded-rect behind the caption
  line (TikTok-style) is more legible on bright source video. Off by
  default; toggle in config.
- **Templates.** Same idea as the title overlay — named template
  presets (`minimal`, `tiktok-pop`, `mrbeast-bounce`, `documentary`)
  that bundle font + colors + motion choices in one knob.

Likely build these two together — they share the font-loading + per-frame
PIL composition path. Doing them in the same pass means we only set up
the `assets/fonts/` directory and the template-config schema once.

## Character-highlight reels (fan-edit mode)

**What:** Different beast from podcast clip extraction. Take N episodes
of a reality show / multi-episode YouTube series / TikTok creator's back
catalog, identify each recurring person on screen, and let the user pick
one ("give me a Hannah appreciation reel") + a style ("baddie", "emo",
"comedic", "dramatic"). Output is a music-driven highlight reel with
genre-appropriate cuts, slow-mo on the right beats, viral-song bed,
and styled captions/effects matching the chosen aesthetic.

This is a wholly new pipeline, not a flag on the existing one. New
`--mode highlights` if/when we build it.

**Sketch of the pipeline:**

  episode files (N)
    ↓ ingest      ffprobe per file
    ↓ face embed  per frame: YOLO person + face crop + embedding
                  (insightface/arcface or face_recognition lib)
    ↓ cluster     agglomerative over embeddings → "character" groups
    ↓ label       user names each cluster, OR vision-LLM labels by
                  showing it a 3-frame thumbnail per cluster
    ↓ per-character screen-time index + per-frame "is this character
       primary subject?" track
    ↓ moment      vision-LLM scores frame windows for the chosen style
       scorer    (comedic, emotional, dramatic, attractive). Sort by
                  score, pick top N moments for the character.
    ↓ style       apply the style template:
       template   - music bed (viral song from licensed library)
                  - slow-mo on beat hits (BPM-detected, ffmpeg setpts)
                  - color grade preset (warm/cool/saturated/desaturated)
                  - caption font + animation matching style
                  - transition style (hard cut / whip pan / fade)
    ↓ render      ffmpeg concat with per-beat alignment, music ducking
                  during dialogue, music fade in/out, final encode

**Major technical pieces we'd need:**

| Piece | Likely tool | Notes |
|---|---|---|
| Face embedding | `insightface` (arcface) or `face_recognition` | Per-face 512-dim vector; cluster across all episodes |
| Cross-episode clustering | scikit-learn `AgglomerativeClustering` on cosine distance | Need a sensible distance threshold; same person across lighting/angle changes is the hard case |
| Character labeling | Vision-capable Claude call per cluster (3-5 representative thumbnails) | Returns a short label like "blonde host" or "guy in red hoodie"; user can rename |
| Screen-time tracking | YOLO + face-match per frame | Builds `character_id → list[time-window]` index |
| Moment scoring | Vision-LLM per frame-window | Style-specific prompt: "score this 10s window 1-5 for COMEDY/DRAMA/ATTRACTIVENESS" |
| Beat detection | `librosa` `onset_detect` | For aligning cuts/slow-mo to music beats |
| Slow-mo | ffmpeg `setpts=2.0*PTS` (video) + `atempo=0.5` (audio, or mute) | Beat-aware: trigger on detected hits |
| Music library | CC0/CC-BY licensed tracks under `assets/music/highlights/` keyed by mood | DMCA-safe defaults; users can supply their own |
| Color grade | ffmpeg `colorchannelmixer` + `eq` + `curves` filters | Preset per style (warm/cool/contrast/saturation) |
| Transitions | ffmpeg `xfade` filter | Per-style transition vocabulary |

**Open questions to settle before building:**

- **Licensing.** "Viral songs" are almost universally copyrighted; we can
  ship a small CC library of mood-matched alternatives and let users
  supply their own audio for personal use. Don't ship anything that
  invites a DMCA strike on the demo site.
- **Privacy.** Face recognition on third-party reality TV / creator
  videos has real consent questions. Need a clear "this is for personal
  fan edits, not commercial use" disclaimer and probably no public
  hosting of outputs by default.
- **Scope.** Reality TV episodes are 30-60 min × N — way more compute
  than current podcast workflow. Need batching, GPU acceleration for
  the face-embed step (insightface has CUDA + CoreML backends).
- **Style templates.** How many do we ship vs let users define? Start
  with 3-4 named templates ("appreciation", "baddie", "emo", "comedic")
  + a YAML schema users can extend.
- **Moment scoring needs vision.** The current pipeline is transcript-
  only; this feature would need vision-LLM calls. We'd want to send
  3-5 keyframes per candidate window, not full video. Token-budget the
  scorer carefully.

**Why this is interesting:** completely different audience (reality-TV
super-fans, creator-stans) than the podcaster audience the rest of the
tool targets. Built on top of the same shot-aware crop and subtitle
plumbing, but driven by entirely different selection + styling logic.
Likely a separate `src/podclipper/highlights.py` module + `run_highlights_pipeline`
in pipeline.py, mirroring the structure of trailer mode.

Multi-month effort. Not blocking. Listed here so the moving pieces are
captured the moment someone wants to start it.

## LiteLLM as the unified API provider (drop `anthropic_api`)

**What:** Replace the in-tree `anthropic_api` provider with a `litellm`
provider so any model on any vendor (OpenAI, Gemini, Groq, Bedrock, Ollama,
…) is reachable via one config knob. LiteLLM speaks Anthropic natively
via `anthropic/<model>` strings, so a separate native provider is redundant.

**Scope:**

- Add `src/podclipper/llm/litellm_provider.py` implementing the `LLMProvider` Protocol.
- Register `"litellm"` in `build_provider()`.
- **Delete** `src/podclipper/llm/anthropic_api.py` and its branch in `build_provider()`.
- Update `--llm-provider` CLI choices: `{claude_cli, litellm}` (was
  `{claude_cli, anthropic_api}`).
- Three characterization tests in `tests/unit/` are time-bombed for this
  change — they'll fail loudly when `anthropic_api` is removed, signaling
  exactly what to update.

**Config shape:**

```yaml
llm:
  provider: litellm                          # or "claude_cli"
  model: "anthropic/claude-sonnet-4-5"       # vendor/model — LiteLLM routes by prefix
  max_tokens: 4096
  temperature: 0.7
  litellm:
    api_key_env: ANTHROPIC_API_KEY
    api_base: null                           # optional — set for Ollama/vLLM/proxies
    timeout_seconds: 300
    num_retries: 2
  claude_cli:                                # unchanged
    timeout_seconds: 900
```

**One model for everything for now.** Per-task overrides
(`llm.analyze.model`, `llm.evaluate.model`) are explicitly deferred to a
follow-up.

**Survivors:** `claude_cli` provider stays as the no-API-key default for
users with a Claude Code subscription.

**Mocking strategy:** unit tests mock `litellm.completion` at the SDK
boundary. No real API calls in the test suite. One optional integration
test gated behind `RUN_INTEGRATION=1` can be added later if needed.

**Method:** strict TDD — Phase 0 characterization tests are the safety net
that proves `claude_cli` parity through the swap.

## PyPI package conversion (pip + uv)

**What:** Ship the pipeline as `pip install podclipper` so users on macOS
and Linux can install + run without cloning the repo. Both pip and uv
read the same `pyproject.toml`, so supporting both is one configuration.

**Scope:**

- Move source under `src/podclipper/` (rename current `src/` →
  `src/podclipper/`); update all `from src.X import …` → `from podclipper.X`.
- Add `pyproject.toml` (PEP 621). Python `>=3.10,<3.13` (mediapipe wheel
  constraint).
- Bundle `config/default.yaml` and `prompts/` as package data via
  `importlib.resources`.
- One console entry point only: `podclipper = "podclipper.main:main"`.
- Move `regen_crops.py` and `debug_detect_clip.py` to `dev/` — debugging
  tools, not user-facing features, so they don't go on user PATH.
- GitHub Action publishes to PyPI on git tag using
  `pypa/gh-action-pypi-publish`.

**Extras (heavy deps opt-in, core install stays lean):**

```toml
[project.optional-dependencies]
diarize = ["pyannote.audio>=3.1", "torch>=2.0"]   # legacy single-mode only
llm     = ["litellm>=1.50"]                       # if not core
```

`mediapipe` and `ultralytics` stay in core — required for the default
shot-aware crop path.

**Out of scope:** Docker image, conda package, Windows native build.

**Method:** TDD where the surface is pure (entry-point dispatch, package-
data discovery via `importlib.resources`); smoke tests + manual venv
install verification for the rest.

**Execution order overall:**

1. Phase 0 — characterization test suite (DONE).
2. Manual smoke: run pipeline on a known video, confirm output unchanged.
3. Phase 1 — LiteLLM swap (TDD; safety net is Phase 0).
4. Manual smoke: rerun with `provider: litellm`, confirm parity.
5. Phase 2 — package conversion (TDD where pure, smoke for the rest).

## Explore Marlin 2B for clip selection / evaluation

**What:** Evaluate [NemoStation/Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B)
as a candidate model for one of the LLM call sites — most likely reel
selection (`analyze.py`) or LLM-as-judge (`evaluate.py`).

**Why it might be worth a look:** 2B is small enough to run locally
(CPU on Apple Silicon, or any consumer GPU) — would close the loop on
PodClipper's "100% local" framing, which today applies only to audio.
Today the transcript still leaves the machine for Claude/OpenAI/etc.

**What to find out:**
- Context window — long enough for ~10-12K-token hour-long transcripts?
- JSON-mode / function-calling support — our prompts expect strict JSON
  back (`reel_detector`, `reel_evaluator`). If it can't reliably emit
  JSON, it's a non-starter without a heavier post-parser.
- Quality on the reel-selection task vs Claude Sonnet — eyeball on 2-3
  episodes via `cfg.llm.provider: litellm` + `cfg.llm.model: ollama/marlin-2b`
  (assuming it lands on Ollama or LiteLLM supports it directly).
- Latency on the reference rig (M-series Mac, no GPU) — fast enough that
  the local-only mode isn't a 30-minute wait?

**Where it plugs in:** No new code path needed if LiteLLM exposes it —
just a config swap. If not, add a Marlin-specific provider under
`src/podclipper/llm/`.
