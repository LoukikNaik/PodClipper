# Pipeline Walkthrough — Step by Step

What happens the moment you run `python main.py video.mp4`.

---

## 0 · Entry point — `main.py`

Before any import runs, the script reads `.env` from the working directory and
populates `os.environ`. This ensures `HF_TOKEN`, `ANTHROPIC_API_KEY`, and any
other secrets are available by the time the pipeline modules load.

```python
# main.py
def _load_dotenv(path: Path = Path(".env")) -> None:
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        # Shell env vars take precedence — never overwrite pre-set vars
        if key and key not in os.environ:
            os.environ[key] = value

_load_dotenv()            # ← runs before any src.* import
from src.pipeline import run_pipeline
```

`argparse` then parses flags (`--language`, `--llm-provider`, `--max-clips`,
`--debug-crop`, `--debug-detect`, `--no-cache`) and merges them into the YAML
config as simple attribute overrides.

---

## Stage 1 · Ingest — `src/ingest.py`

`ffprobe` is called once to read metadata. The result is stored in a `VideoMeta`
dataclass that every downstream stage shares.

```python
@dataclass
class VideoMeta:
    path: Path
    duration: float    # seconds
    width: int
    height: int
    fps: float
    has_audio: bool
    codec: str
```

Nothing is decoded here — it's a pure probe. If the file is unreadable or has
no video stream, the pipeline aborts with a clear error message before any GPU
or model resources are allocated.

---

## Stage 2 · Audio Extraction — `src/audio.py`

The whole-video audio track is ripped to a 16 kHz mono WAV and saved to the
clip cache.

```python
# ffmpeg command issued under the hood
[
    "ffmpeg", "-y",
    "-i", str(video_path),
    "-vn",                          # no video
    "-ac", "1",                     # mono
    "-ar", str(sample_rate),        # 16 000 Hz — Whisper-native
    "-acodec", "pcm_s16le",         # uncompressed, fastest decode
    str(audio_wav),
]
```

**Why 16 kHz mono WAV?** faster-whisper's CTranslate2 backend ingests raw
`float32` arrays decoded from this format. By producing a simple uncompressed
WAV we avoid any lossy-decode step later and save the `sf.read()` roundtrip
that pydub or librosa would introduce.

The output path is deterministic: `.cache/<video_stem>-<sha1[:10]>/audio.wav`.
If it already exists and `use_cache=True` (the default), this stage is skipped
entirely — re-runs of the same video go straight to transcription.

---

## Stage 3a · First-pass Transcription — `src/transcribe.py`

The 16 kHz WAV is decoded to a flat `float32` numpy array in memory, then
split into 5-minute overlapping chunks (10 s overlap) that are transcribed in
parallel.

```python
def transcribe_first_pass(audio_path, duration, cfg) -> Transcript:
    audio = _decode_audio_to_float32(audio_path)   # ffmpeg → stdout → np.frombuffer
    chunks = plan_chunks(duration, chunk_seconds=300, overlap_seconds=10)

    def _work(chunk):
        slice_ = audio[int(chunk.start * 16000) : int(chunk.end * 16000)]
        segs, lang = _transcribe_array(model, slice_, time_offset=chunk.start, ...)
        return chunk.index, segs, lang

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_work, c) for c in chunks]
        ...   # collect results in chunk-index order
```

**Model used:** `base` (or `tiny`) with `int8` quantization — fast but coarse.
Accuracy doesn't need to be perfect here: this transcript is only sent to the
LLM for clip selection, not used for subtitles.

Overlapping chunks mean no speech near a boundary is cut off. Duplicates from
the overlap are eliminated by a simple "drop any segment whose `start` < last
accepted `end`" rule:

```python
def _merge_segments(per_chunk_segments):
    last_end = -1.0
    for chunk_segs in per_chunk_segments:
        for seg in chunk_segs:
            if seg.start < last_end - 0.1:   # inside overlap window → skip
                continue
            merged.append(seg)
            last_end = max(last_end, seg.end)
```

The detected language (e.g. `"hi"`, `"en"`) is stored on the `Transcript` and
**pinned** onto `cfg.transcribe.language` so every subsequent Whisper call uses
the same language. Without pinning, the high-quality second-pass model might
flip between Hindi and Urdu on different clips, producing unreadable mixed-script
subtitles.

```python
# pipeline.py — language pinning
if cfg.transcribe.language is None and transcript.language:
    cfg.transcribe.language = transcript.language
```

---

## Stage 3b · LLM Analysis — `src/analyze.py`

The merged transcript is formatted as compact timestamped lines:

```
[00:00-00:12] So the biggest enemy of a teenager today is comparison.
[00:12-00:28] You compare your behind the scenes to everyone else's highlight reel.
[00:28-00:45] The second enemy is overthinking. You sit and replay conversations...
```

This block of text, preceded by the `reel_detector.txt` system prompt, is sent
to Claude. The LLM returns a JSON array of clip candidates:

```json
[
  {
    "start": 0.0,
    "end": 47.0,
    "title": "3 Things Ruining Your Peace",
    "reason": "Complete 3-point arc with a strong hook and a clean directive ending.",
    "hook_score": 4.8
  },
  ...
]
```

The `analyze_for_reels` function then:

1. **Validates** each entry (clamps timestamps to video bounds, drops clips with
   `end <= start` or duration outside the 15–90 s window).

2. **Snaps** clip boundaries to the nearest transcript-segment edge (±3 s
   tolerance). This makes cuts land on natural speech pauses instead of
   mid-sentence.

   ```python
   def _snap_to_segment_boundaries(start, end, segments, tolerance_s=3.0):
       nearest_start = min(seg_starts, key=lambda s: abs(s - start))
       if abs(nearest_start - start) <= tolerance_s:
           start = nearest_start
       # same for end → nearest seg_end
   ```

3. **Rewrites long titles** via a second LLM call if the title exceeds 38 chars
   (the two-line budget on a 1080×1920 frame). The rewrite is grounded in the
   actual clip transcript — not a mechanical truncation.

   ```python
   if len(clip.title) > 38:
       excerpt = _transcript_excerpt(transcript, clip)   # pull clip text
       short = _rewrite_title_with_llm(clip.title, excerpt, clip.reason, provider)
   ```

4. **Sorts** by `hook_score` descending and caps at `target_clips` (default 8).

---

## Stage 4 · Per-clip Loop — `src/pipeline.py`

All remaining stages run once per selected clip. A Rich progress bar tracks progress.

The output directory is timestamped (`outputs/2026-04-22_14-30-00/`) so every
run is non-destructive.

### 4a · Clip Extraction

`ffmpeg` cuts `[clip.start − pad, clip.end + pad]` out of the source video.
The pad (default 2 s) gives the subtitle and crop stages a soft boundary to
work with; it's trimmed away later by the smart-crop smoother.

```python
cmd = [
    "ffmpeg", "-y",
    "-ss", f"{start:.3f}",          # input seek — fast keyframe seek
    "-i", str(video_path),
    "-t", f"{duration:.3f}",
    "-c:v", "libx264", "-crf", "18",
    "-preset", "medium",
    "-c:a", "aac",
    str(out_path),
]
```

### 4b · Person Detection — `src/detect.py`

YOLOv8-nano runs on every frame of the extracted clip segment. Each frame
produces a list of person bounding boxes.

```python
results = model.predict(frame, conf=0.35, iou=0.5, verbose=False)
bboxes = _detections_to_bboxes(results[0], person_class_id=0, conf_threshold=0.35)
```

**Front-face preference**: MediaPipe BlazeFace is then run on *all* person
candidates simultaneously — not just the YOLO winner. Any candidate where a
face is detectable (front-facing) automatically beats back-of-head candidates
before the IoU tracker even runs.

```python
# Run face detection on every candidate
all_tagged = _run_face_on_all(frame, bboxes, face_detector)
# → [(person_bbox, face_bbox_or_None), ...]

# Prefer front-facing: filter pool to front-facing candidates first
front = [(p, f) for p, f in all_tagged if f is not None]
pool = front if front else all_tagged    # fall back only if everyone is back-facing

primary = _pick_primary([p for p, _ in pool], anchor)   # IoU / largest
```

Once the primary person is selected, the crop's x-center is shifted from the
body centroid to the face centroid (body bbox dimensions are kept unchanged so
downstream clustering is stable):

```python
if face_bbox is not None:
    face_cx = face_bbox.x + face_bbox.w / 2
    new_x = face_cx - primary.w / 2          # shift x, keep w/h
    primary = BBox(x=new_x, y=primary.y, w=primary.w, h=primary.h, ...)
```

Between sampled frames, the last-known bbox is carried forward so the output
list is always dense (one entry per source frame). The anchor updates every
sampled frame so tracking survives cuts and pans.

**Debug overlay** (`--debug-detect`): writes `debug_detect.mp4` in the clip
cache with per-frame annotations:
- Green border = front-facing (face found)
- Orange border = back-of-head (no face)
- Thick border = selected primary
- Cyan rect = face bbox

### 4c · Concurrent Transcription + Diarization

Second-pass transcription and speaker diarization are completely independent
(one is I/O bound, the other is CPU/GPU bound). They run concurrently inside a
`ThreadPoolExecutor(2)`:

```python
with ThreadPoolExecutor(max_workers=2) as ex:
    f_words = ex.submit(transcribe_second_pass, segment_path, cfg)
    f_diar  = ex.submit(_maybe_diarize, segment_path, cfg)
    words        = f_words.result()
    diar_segments = f_diar.result()
```

**Second-pass transcription** (`large-v3`, `beam_size=5`, `word_timestamps=True`)
re-transcribes only the extracted clip. The output is a flat list of `Word`
objects with clip-relative timestamps (t=0 = clip start):

```python
@dataclass
class Word:
    start: float       # seconds from clip start
    end: float
    text: str
    confidence: float
```

**Diarization** (`pyannote/speaker-diarization-3.1`, requires `HF_TOKEN`) runs
on the clip's audio track and returns a sequence of speaker-active intervals:

```python
@dataclass
class DiarSegment:
    start: float       # clip-relative
    end: float
    speaker_id: str    # e.g. "SPEAKER_00", "SPEAKER_01"
```

If diarization is disabled or fails for any reason, it returns `None` and the
pipeline gracefully falls through to single-speaker mode.

### 4d · LLM Cleanup — `src/transcribe_cleanup.py`

The second-pass word list is sent to Claude as an indexed JSON array. Claude
corrects Whisper mis-transcriptions and transliterates non-Latin scripts to
phonetic Latin — without altering timestamps or word count.

```python
# Input to LLM
[{"i": 0, "w": "Karmanyeva"}, {"i": 1, "w": "वाधिकारस्ते"}, ...]

# LLM returns same structure with corrected text
[{"i": 0, "w": "Karmanye va"}, {"i": 1, "w": "Vadhikaraste"}, ...]
```

The alignment is 1-to-1 on `i`. If the LLM returns a different word count or
malformed JSON, the raw words are used as-is — subtitles keep working, they
just reflect the raw Whisper output.

### 4e · Speaker Timeline — `src/timeline.py`

The per-frame bbox list is clustered by x-center position (1D, tolerance 120 px)
into "persistent positions" — spatial groupings that represent distinct speaker
locations or camera angles.

```python
def _cluster_x_centers(bboxes, merge_tolerance_px=120.0):
    # For each frame's bbox, find the nearest existing cluster center.
    # If within tolerance: join it (update running mean).
    # If outside: start a new cluster.
```

The timeline is then built via one of four paths, in priority order:

| Condition | Path | Result |
|---|---|---|
| No persons detected | CENTER fallback | Single segment targeting frame center |
| No persistent cluster (< 10% frames) | PRIMARY fallback | Single segment, all detections |
| 2+ clusters + diarization available | **Diarization-driven** | N segments, speaker-to-cluster linked via mouth motion |
| 2+ clusters, no diarization | **Multi-shot** | N segments, one per contiguous camera-angle run |
| 1 cluster | **Single-shot** | Single segment targeting that cluster |

**Diarization-driven linking** (`src/diarize.py → link_timeline`): for each
speaker, the pipeline finds their longest speaking window, then for each bbox
cluster, it samples up to 40 frames from that window and runs MediaPipe
FaceLandmarker to measure mouth openness (gap between upper/lower lip as a
fraction of face height). The cluster with the highest variance in mouth
openness is the one that was talking — and gets linked to that speaker.

```python
# Mouth openness signal
upper = landmarks[13]   # upper lip center
lower = landmarks[14]   # lower lip center
face_top = landmarks[10]
face_bottom = landmarks[152]
openness = abs(lower.y - upper.y) / abs(face_bottom.y - face_top.y)

# Variance over a window = "how much did this mouth move?"
# High variance → this face was speaking
```

**Debug overlay** (`--debug-detect`): writes `debug_mouth.mp4` with sampled
frames annotated with mouth boxes:
- Green = `SPEAK` (openness > 0.03)
- Blue = `silent`
- Orange = `no landmarks`

After building the timeline, short segments (< 0.8 s by default) are merged
into their predecessor to prevent flicker during speaker switches:

```python
def apply_min_dwell(timeline, min_dwell_seconds=0.8):
    for seg in timeline:
        if (seg.end - seg.start) < min_dwell_seconds and out:
            out[-1].end = seg.end     # absorb into prev
        else:
            out.append(seg)
```

### 4f · Smart 9:16 Crop — `src/crop.py`

OpenCV reads the clip frame-by-frame. For each frame, the target x-center is
looked up via `timeline_seg.bbox_at(frame_idx)`. An EMA smoother (α = 0.1)
dampens jitter within a timeline segment; at segment boundaries the EMA resets
hard to the new speaker's position (no slow pan across speakers).

```python
# Within a segment: smooth
x_smooth = alpha * x_target + (1 - alpha) * x_smooth

# At a segment boundary: hard cut to new target
if crossed_boundary:
    x_smooth = x_target
```

The 9:16 crop window (1080 × 1920 px) is centered on `x_smooth`, clamped to
source frame edges, and written frame-by-frame to a new video file.
Audio is passed through untouched.

### 4g · Subtitle Burn — `src/subtitles.py`

The cropped video is read frame-by-frame again. For each frame at timestamp `t`,
the word that spans `t` is highlighted in yellow; all other visible words are
white. Subtitles are rendered with PIL (TrueType font, outline via
`stroke_width`), composited onto the frame with OpenCV, and written to the
final output file.

```python
# For each frame at timestamp t:
active_words = [w for w in words if w.start <= t <= w.end]
context_words = [w for w in words if w.start <= t + lookahead]

for word in context_words:
    color = YELLOW if word in active_words else WHITE
    draw.text((x, y), word.text, font=font, fill=color, stroke_fill=BLACK, stroke_width=3)
```

**Title overlay**: for the first `duration_seconds` (default 3.5 s) of the
clip, the title is rendered in large bold text at the top of the frame. A
dark-to-transparent gradient sweep covers the top 30% of the frame so white
text stays readable on any background. The title fades out over 0.7 s.

```python
# Gradient backdrop — numpy broadcast, no per-pixel loop
alpha_col = np.linspace(grad_opacity * 255, 0, grad_h, dtype=np.uint8)
grad_arr[:, :, 3] = alpha_col[:, np.newaxis]   # broadcast col → full-width
```

**Video fade**: the last 0.6 s of the clip fades to black. Each frame's pixels
are multiplied by a linear alpha ramp:

```python
t_from_end = clip_duration - t
if t_from_end < fade_out_seconds:
    alpha = t_from_end / fade_out_seconds
    frame = (frame.astype(np.float32) * alpha).astype(np.uint8)
```

Audio gets a matching `afade=t=out` applied when muxing back in via ffmpeg.

### 4h · Quality Evaluation — `src/evaluate.py`

After the final reel is rendered, a two-layer quality check runs automatically.

**Layer 1 — Technical metrics** (computed from pipeline data, zero cost):

| Metric | Formula | Good range |
|---|---|---|
| Face visibility | `face_hits / total_frames` | ≥ 80% |
| Crop stability | `1 − std(x_centers) / source_width` | ≥ 0.85 |
| Speaker coverage | `person_frames / total_frames` | ≥ 70% |
| Duration | Range check | 25–60 s |
| Words per second | `len(words) / clip_duration` | 1.5–3.5 |
| Subtitle coverage | `sum(word_durations) / clip_duration` | ≥ 60% |

Composite: `tech_score = weighted_sum(all_metrics)` → 0.0–1.0.

**Layer 2 — LLM content evaluation** (one Claude call per reel):

The evaluator receives only the title + transcript — no selection prompt, no
hook score, no `reason`. It scores six dimensions 1–5 with chain-of-thought
reasoning:

```json
{
  "hook":         {"reasoning": "...", "score": 4},
  "arc":          {"reasoning": "...", "score": 5},
  "ending":       {"reasoning": "...", "score": 4},
  "standalone":   {"reasoning": "...", "score": 5},
  "shareability": {"reasoning": "...", "score": 4},
  "title_match":  {"reasoning": "...", "score": 5},
  "overall": 4.5,
  "verdict": "publish",
  "one_line_feedback": "Strong listicle — cut the 'So' opener for a harder hook."
}
```

**Final score**:
```
final = tech_score × 0.3 + (content_score / 5.0) × 0.7
```

Content is weighted 70% because a technically perfect reel with weak content
is still a bad reel.

**Verdict thresholds**:
- `overall ≥ 4.0` → **publish**
- `3.0 ≤ overall < 4.0` → **review**
- `overall < 3.0` → **skip**

The full scorecard is appended to a sidecar `.txt` next to the reel:

```
=== QUALITY SCORECARD ===

Technical:
  Face visibility:     92%   ✓
  Crop stability:      0.91
  Speaker coverage:    100%  ✓
  Duration:            48s   ✓
  Words/sec:           2.3   ✓
  Subtitle coverage:   84%   ✓
  Tech score:          0.94

Content (LLM evaluation):
  Hook:                4/5  "Direct address, slight filler opener"
  Arc:                 5/5  "Clean 3-part listicle, zero tangents"
  ...
  Content score:       4.5/5

Overall:               0.88  →  PUBLISH
Feedback:              "Tighten the opener — cut the 'So'."
```

### 4i · Auto-skip or Save

If `evaluate.auto_skip: true` and `verdict == "skip"`, the rendered `.mp4` and
sidecar `.txt` are moved to `outputs/<timestamp>/skipped/` instead of staying
in the main directory.

```python
if eval_cfg.auto_skip and scorecard.verdict == "skip":
    skip_dir = output_dir / "skipped"
    final_path.rename(skip_dir / final_path.name)
    sidecar_path.rename(skip_dir / sidecar_path.name)
else:
    produced.append(final_path)
```

---

## Output Structure

```
outputs/
└── 2026-04-22_14-30-00/
    ├── reel_01_3-things-ruining-your-peace.mp4   ← final 9:16 reel
    ├── reel_01_3-things-ruining-your-peace.txt   ← sidecar + scorecard
    ├── reel_02_why-meditation-makes-it-worse.mp4
    ├── reel_02_why-meditation-makes-it-worse.txt
    └── skipped/
        └── reel_03_...mp4                        ← auto-skipped if score < 3.0

.cache/
└── <video_stem>-<sha1>/
    ├── audio.wav                                 ← extracted audio (reused)
    ├── reel_01_.../
    │   ├── segment.mp4                           ← raw padded clip
    │   ├── cropped.mp4                           ← 9:16 smart-cropped
    │   ├── debug_detect.mp4                      ← (if --debug-detect)
    │   └── debug_mouth.mp4                       ← (if --debug-detect + diarize)
    └── reel_02_.../
        └── ...
```

---

## Data Flow Summary

```
main.py
  └─ load_config + CLI overrides
       └─ run_pipeline()
            │
            ├─ [1] ingest        → VideoMeta (duration, fps, dimensions)
            ├─ [2] extract_audio → audio.wav (16 kHz mono WAV)
            ├─ [3a] transcribe_first_pass  → Transcript (segments + language)
            │       └─ language pinning → cfg.transcribe.language
            ├─ [3b] analyze_for_reels     → list[Clip] (start/end/title/score)
            │
            └─ for each Clip:
                 ├─ extract_clip_segment   → segment.mp4 (padded)
                 ├─ detect_humans_per_frame → per_frame_bboxes, fps, w, h
                 │   └─ face detection on ALL candidates → front-face preference
                 │
                 ├─ [concurrent]
                 │   ├─ transcribe_second_pass → list[Word] (word timestamps)
                 │   └─ diarize_clip           → list[DiarSegment] | None
                 │
                 ├─ cleanup_words          → list[Word] (corrected text, same timestamps)
                 ├─ build_speaker_timeline → Timeline (per-segment bbox_at callables)
                 │   └─ mouth-motion linking if diarization ran
                 ├─ apply_min_dwell        → Timeline (short segments merged)
                 ├─ smart_crop_916         → cropped.mp4 (1080×1920, EMA-smoothed)
                 ├─ burn_subtitles         → reel_XX_slug.mp4 (karaoke + title overlay)
                 └─ evaluate_reel          → ReelScorecard → appended to .txt sidecar
                      └─ auto-skip to skipped/ if verdict=="skip" and auto_skip=true
```
