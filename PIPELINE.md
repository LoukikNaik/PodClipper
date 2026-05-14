# Pipeline

End-to-end walkthrough of how `agentic-video-editor` turns a long video into vertical 9:16 reels.

## Input → Output

A long video file (`.mp4`, `.mov`, etc.) → N vertical 9:16 reels (1080×1920) with karaoke subs, a fading title card, and a clean fade-out. Orchestrated by `src/pipeline.py::run_pipeline`, invoked from `main.py`.

```
video.mp4
   │
   ▼
┌──────────┐
│  ingest  │  ffprobe → VideoMeta(dims, fps, duration, audio, codec)
└──────────┘
   │
   ▼
┌──────────┐
│  audio   │  ffmpeg → 16 kHz mono WAV (cached)
└──────────┘
   │
   ▼
┌───────────────────────────┐
│  first-pass transcribe    │  ThreadPool × faster-whisper(base)
│  parallel 5-min chunks    │  per-chunk → segment-merge via timestamp dedup
│  overlap 10 s             │  → Transcript{language, segments, words}
└───────────────────────────┘
   │
   ▼
┌───────────────────────────┐
│  analyze (Claude)         │  prompts/reel_detector.txt
│  pick reel-worthy clips   │  JSON → coerce → snap start/end to segment
│  rewrite long titles      │  boundaries → (if title > 38 chars) second
│                           │    LLM call grounded in clip transcript
└───────────────────────────┘
   │
   ▼ list[Clip]
   │
   │  for each clip:
   ▼
┌────────────────────────────────────────────────────────────────────┐
│  PER-CLIP LOOP                                                     │
│                                                                    │
│  extract clip ±2 s buffer (ffmpeg)                                 │
│       │                                                            │
│       ▼                                                            │
│  YOLOv8n per-frame detection → list[BBox | None] + fps, w, h       │
│       │                                                            │
│       ├────────────────┐                                           │
│       ▼                ▼                                           │
│   second-pass      maybe_diarize                                   │
│   whisper(base)    (stub → None)    ◄─── ThreadPool(2)             │
│       │                │                                           │
│       ▼                ▼                                           │
│  cleanup_words (LLM) ──────┐    transliterate non-Latin, fix       │
│                            │    English typos, 1:1 word count      │
│                            ▼                                       │
│                  build_speaker_timeline                            │
│                     - 1 bbox cluster → 1-segment timeline          │
│                     - N bbox clusters → N camera-shot segments     │
│                     - diar_segments present → multi-speaker        │
│                    apply_min_dwell                                 │
│                            │                                       │
│                            ▼                                       │
│                   smart_crop_916 (OpenCV)                          │
│                     decode → slice → resize →                      │
│                     pipe BGR to ffmpeg encoder →                   │
│                     re-mux source audio                            │
│                            │                                       │
│                            ▼                                       │
│                   burn_subtitles (PIL + OpenCV)                    │
│                     karaoke word highlight at bottom               │
│                     fading title pill at top (3 s, 0.6 s fade)     │
│                     video fade-to-black last 0.6 s                 │
│                     audio afade out                                │
│                            │                                       │
│                            ▼                                       │
│                   outputs/reel_NN_<slug>.mp4 + .txt sidecar        │
└────────────────────────────────────────────────────────────────────┘
```

---

## Stage details

### 1. Ingest — `src/ingest.py`

Shells out to `ffprobe`, parses JSON, returns a `VideoMeta` dataclass with duration, width×height, fps, codec, and whether the file has an audio stream. Raises `IngestError` if probing fails or there's no video stream.

### 2. Audio extraction — `src/audio.py`

`extract_audio(video, out_path, sample_rate=16000)` invokes ffmpeg to produce a mono WAV at the Whisper-friendly sample rate. Result is cached at `.cache/<video_stem>-<hash>/audio.wav` and reused on re-runs unless `--no-cache` is passed.

The helper `plan_chunks(duration, chunk_s, overlap_s)` returns `[(start, end), …]` ranges used by the next stage — it doesn't write N chunk files, just computes time windows.

### 3. First-pass transcription — `src/transcribe.py::transcribe_first_pass`

- Decodes the WAV into a single `float32` numpy array (`_decode_audio_to_float32` via ffmpeg pipe).
- Splits into 5-min overlapping chunks (from stage 2's planner).
- Parallelizes across `ThreadPoolExecutor` workers. A **single** faster-whisper model is shared — CTranslate2 releases the GIL so threads run truly concurrently without per-worker model memory overhead.
- Each worker calls `model.transcribe(array_slice, beam_size=1, word_timestamps=True)` and shifts timestamps by the chunk's start offset.
- Merge step: iterate per-chunk segments and drop any whose start falls inside the previous chunk's range (simple timestamp dedup — no fuzzy text alignment, good enough since accuracy is coarse here).
- Output: `Transcript{language, segments: [TranscriptSegment{start, end, text, words}]}`.

Uses the **smaller model** (`base`) because we only need enough quality for the LLM to identify topics.

### 4. Analyze — `src/analyze.py`

Builds the LLM prompt from two pieces:

- **System prompt**: `prompts/reel_detector.txt`, which specifies:
  - The four-ingredient "sweet spot": hook / setup / payoff / close.
  - Topic-coherence rules: the clip must contain ONE complete topic unit.
  - Boundary rules: end must land on a sentence-final `.`, `!`, or `?`.
  - Two mandatory self-checks: **"Stop early"** (cut 5s before end — is the full point there?) and **"Continue"** (does the next segment continue the same thought?).
  - Strict title guidance with curiosity-gap patterns, 38-char hard limit.
- **User prompt**: compact `[MM:SS-MM:SS] text` lines so the LLM sees natural pause boundaries between segments.

Sends to the provider via `src/llm/__init__.py::build_provider(cfg.llm)` — either `ClaudeCLIProvider` (subprocess) or `AnthropicAPIProvider` (SDK).

Response pipeline:

- `_extract_json_array` tolerates code fences and prose prefix.
- Per clip: `_coerce_clip` validates start/end/title/hook_score, clamps to video bounds, rejects out-of-range durations.
- `_snap_to_segment_boundaries` snaps start to the nearest segment start and end to the nearest segment end (within 3 s tolerance). Lands cuts on real speech pauses.
- `_rewrite_title_with_llm` fires a **second LLM call** if any title is over 38 chars — the prompt includes the actual clip transcript so the rewrite is grounded in real content, not a mechanical truncation.
- Sort by `hook_score` descending, cap to `target_clips`.

Output: `list[Clip{start, end, title, reason, hook_score}]`.

### 5. Per-clip loop — `src/pipeline.py`

#### 5a. Extract clip

`ffmpeg -ss <start-2s> -t <duration+4s> ... segment.mp4` — 2-second buffer on each side so crops and transcription have context. Written to `.cache/<video>/reel_NN_<slug>/segment.mp4` for cache reuse.

#### 5b. Detect — `src/detect.py`

YOLOv8n (`ultralytics`). Device auto-detects CUDA > MPS > CPU. Per-frame inference, filtered to the `person` class above the confidence threshold. `_pick_primary` chooses the "main subject" per frame using greedy IoU against an anchor bbox (largest on the first frame, updated as you go). Frames with no detection get `None`; frames sampled less often than once per frame carry forward the last-known bbox. Returns `(list[BBox|None], fps, width, height)`.

#### 5c. Concurrent 2nd-pass Whisper + diarization stub

`ThreadPoolExecutor(max_workers=2)` runs two independent tasks on the same clip:

- `transcribe_second_pass(segment, cfg)` — single-shot high-quality Whisper transcribe. No chunking — clip is short. Clip-relative timestamps.
- `_maybe_diarize(segment, per_frame_bboxes, cfg)` — returns `None` today (MVP); post-MVP hook for `pyannote.audio` + MediaPipe mouth-motion linking to enable multi-speaker follow-the-speaker cropping.

#### 5d. LLM transcript cleanup — `src/transcribe_cleanup.py`

Serial step after second-pass. `cleanup_words(words, provider, cfg)`:

- Sends indexed JSON array `[{"i": 0, "w": "…"}, …]` to Claude.
- System prompt: transliterate non-Latin scripts (Devanagari / Urdu / Arabic → Latin), fix obvious English typos, preserve 1:1 word count.
- Validates response: if the count mismatches or JSON is invalid → return raw words unchanged. Timestamps are preserved from the original `Word` objects; only `.text` is replaced.
- Effect: `गुरुकुल` renders as `gurukul`, "Shiva jee" becomes "Shivaji", etc. Karaoke timing intact.

#### 5e. Build speaker timeline — `src/timeline.py`

Central data structure: `list[TimelineSegment{start, end, label, bbox_at: Callable[frame_idx → BBox|None]}]`. Four code paths:

1. **No persons detected** → single center-frame segment.
2. **1 persistent bbox cluster** → one timeline segment, `bbox_at` returns the actual per-frame bbox filtered to that cluster.
3. **Multiple persistent clusters (camera angles)** → `_multi_shot_timeline` maps each cluster's contiguous frame runs to its own timeline segment. The editor already cut cameras; we follow their cuts. Key fix for podcast-style multi-camera edits.
4. **Diarization provided** (post-MVP) → multi-speaker timeline per pyannote segments.

`apply_min_dwell(timeline, 0.8s)` merges flickers so we don't hard-cut every 0.3 s.

#### 5f. Smart 9:16 crop — `src/crop.py`

OpenCV decode, per-frame:

1. `t = frame_idx / fps` → find the active timeline segment.
2. `bbox = seg.bbox_at(frame_idx)`; if `None`, carry the last-known x-center; else use `bbox.x_center`.
3. EMA smooth within a segment (`α=0.1`); hard reset on segment change (so camera cuts land cleanly).
4. Compute crop window `[x_start, x_start+crop_w]` clamped to source width; `crop_w = source_h × 9/16`.
5. Slice, resize to 1080×1920, pipe BGR to an `ffmpeg` encoder subprocess.
6. Second ffmpeg pass: re-mux source audio onto the rendered video (`-c:v copy -c:a aac`).

Optional `--debug-crop` renders an auxiliary video with bboxes + crop rectangles drawn on the source for tuning.

#### 5g. Subtitle burn + title overlay + fade-out — `src/subtitles.py`

- `generate_subtitle_lines` groups words into karaoke lines (char budget + sentence punctuation + long-pause breaks).
- `burn_subtitles` walks the cropped video frame by frame. PIL overlay composited per frame:
  - **Bottom karaoke**: white text, yellow highlight on the currently-spoken word, black outline, centered in the lower third.
  - **Top title pill** (first 3 s, 0.6 s fade-out): rounded rectangle + bold multi-line wrapped title, alpha-blended.
  - **Last 0.6 s**: linear fade-to-black on pixel values.
- Pipe BGR frames to ffmpeg encoder.
- Audio muxed in with `afade=t=out:st=<end-0.6>:d=0.6` — audio fades in lockstep with the visual fade.

Output: `outputs/reel_NN_<slug>.mp4` + sidecar `.txt` with title, LLM reasoning, source timestamps, hook score.

---

## Cross-cutting concerns

- **Caching**: `.cache/<video>-<hash>/{audio.wav, reel_NN/{segment.mp4, cropped.mp4}}`. On re-run, existing artifacts skip the expensive stages. `--no-cache` forces a rebuild.
- **Config-driven**: `config/default.yaml` holds everything (Whisper model sizes, YOLO confidence, EMA alpha, fade duration, subtitle style, LLM provider). CLI flags in `main.py` override individual keys.
- **LLM abstraction**: `src/llm/` exposes a single `Protocol.complete(user, system, max_tokens)`. Swapping between Claude CLI and Anthropic API is one config change.
- **Per-clip failure isolation**: clip-level `try/except` — one bad clip doesn't kill the whole run.
- **Logging**: Rich-based logger, progress bar over clips in the per-clip loop.

---

## Post-MVP roadmap

- **Diarization** (`src/diarize.py`): pyannote.audio + MediaPipe mouth-motion linking to switch crops between speakers in podcast-style multi-person clips. Design already wired through the timeline API.
- **Vision-enabled LLM analysis**: send sampled keyframes alongside the transcript so the LLM can catch visual-only moments (gestures, reactions, props).
- **Resumability within stages**: checkpoint per-chunk in first-pass transcription so 1-hour videos can resume mid-flight.
