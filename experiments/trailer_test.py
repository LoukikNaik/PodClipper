#!/usr/bin/env python3
"""Trailer mode test: stitch standalone quotable sentences from across a
podcast into one trailer-style reel, with brief black-frame transitions
between cuts.

What it does:
  1. Transcribe the full episode (reuses .cache/ if present).
  2. Ask the LLM to pick every quotable standalone sentence — each must
     make sense on its own, no time/count limit.
  3. Extract each sentence's source clip (±0.2 s pad).
  4. Run the shot-aware crop on each clip independently.
  5. Concatenate the cropped clips with 0.3 s of black + silence between.
  6. Burn karaoke subtitles over the concatenated trailer with timestamps
     remapped into trailer coordinates.

Usage:
    python experiments/trailer_test.py path/to/video.mp4 [--out trailer.mp4]
"""

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.analyze import _snap_to_segment_boundaries
from src.audio import extract_audio
from src.config import load_config
from src.crop import smart_crop_916_stacked
from src.detect import detect_humans_all_per_frame
from src.ingest import ingest
from src.llm import build_provider
from src.logging_util import setup_logging
from src.subtitles import burn_subtitles
from src.timeline import classify_wide_shot_frames
from src.transcribe import transcribe_first_pass
from src.types import Transcript, Word

log = logging.getLogger("trailer")

GAP_SECONDS = 0.6        # black-frame duration between sentences — long
                         # enough to feel like a deliberate beat, not a join
HEAD_PAD = 0.10          # buffer before the first spoken word in each cut
TAIL_PAD = 0.10          # buffer after the last spoken word in each cut
SEG_OUT_FPS = 30         # framerate for the trailer output

# Per-clip audio crossfades into/out of each segment so the cut doesn't
# sound like the speaker is mid-thought. The fade-out consumes the TAIL_PAD
# region so no spoken content is faded.
AUDIO_FADE_IN_S = 0.08
AUDIO_FADE_OUT_S = 0.30


# ---------- .env loader (so HF_TOKEN / ANTHROPIC_API_KEY are visible) ----------
def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        if key and key not in os.environ:
            os.environ[key] = value


# ---------- LLM prompt + parser ----------
_TRAILER_SYSTEM_PROMPT = """\
You are building a short trailer for a podcast episode by picking 4 to 5
sentences from the transcript that, played back-to-back, work as one
coherent mini-narrative.

Hard requirements for what you pick:

  1. EXACTLY 4 OR 5 PICKS. Not 6, not 10, not 20. Be ruthless. If a candidate
     is only "good," cut it.

  2. EACH PICK IS A SELF-SUFFICIENT HOOK LINE. A stranger encountering it
     cold on social media would understand it AND want to keep watching.
     No mid-thought picks. No "...and that's why..." picks that depend on
     prior context.

  3. THE 4-5 PICKS, IN ORDER, FORM A COHERENT ARC. They should feel like
     they belong in the same trailer:
        opener (the hook of the episode)
        →  development (one or two reframes / sharp claims)
        →  payoff (the line you want viewers leaving with)
     Picks from totally different parts of the conversation are fine ONLY
     IF they thematically connect.

  4. EACH PICK IS ONE COMPLETE SENTENCE (or at most two tightly-coupled
     sentences forming a single thought). Begin at the start of the
     sentence, end at the end of the sentence — never mid-clause.

Output a JSON array of EXACTLY 4 or 5 items, no prose around it. Each item:
  - "sentence": the verbatim sentence text from the transcript (so we can
                locate it precisely)
  - "start":    seconds (float) when the sentence begins
  - "end":      seconds (float) when the sentence ends — be GENEROUS,
                better to overshoot than to clip the last word

The transcript below has [MM:SS-MM:SS] timestamps per segment. Use those
to derive your "start" and "end" values.

Example output format:
[
  {"start": 109.82, "end": 113.40, "sentence": "Confidence is an output, not an input."},
  {"start": 681.50, "end": 685.20, "sentence": "Make anxiety sit in the back seat."}
]
"""


def _fmt_transcript_for_llm(transcript: Transcript) -> str:
    lines = []
    for s in transcript.segments:
        m1, s1 = divmod(int(s.start), 60)
        m2, s2 = divmod(int(s.end), 60)
        lines.append(f"[{m1:02d}:{s1:02d}-{m2:02d}:{s2:02d}] {s.text.strip()}")
    return "\n".join(lines)


def _extract_json_array(raw: str) -> list[dict]:
    """Tolerant JSON-array extractor — copes with ```json fences and prose."""
    # Try direct parse first
    try:
        v = json.loads(raw)
        if isinstance(v, list):
            return v
    except json.JSONDecodeError:
        pass
    # Otherwise find first '[' ... matching ']'
    m = re.search(r"\[\s*\{.*?\}\s*\]", raw, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array found in LLM output:\n{raw[:300]}...")
    return json.loads(m.group(0))


def pick_quotables(transcript: Transcript, provider, cfg, video_duration: float) -> list[dict]:
    transcript_text = _fmt_transcript_for_llm(transcript)
    raw = provider.complete(
        user_prompt=transcript_text,
        system_prompt=_TRAILER_SYSTEM_PROMPT,
        max_tokens=cfg.llm.max_tokens,
    )
    items = _extract_json_array(raw)
    # Light validation; tolerate small schema slips
    out = []
    for it in items:
        try:
            s = float(it["start"])
            e = float(it["end"])
            txt = str(it["sentence"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if e <= s or not txt:
            continue
        # Drop picks that extend past the source (LLMs sometimes hallucinate
        # continuation timestamps beyond the transcript's actual end).
        if e > video_duration:
            log.warning(f"Skipping past-source pick ({s:.1f}-{e:.1f}s): {txt!r}")
            continue
        # Snap to the nearest Whisper segment boundaries so cuts land on
        # natural speech pauses, not mid-word. Tolerance is large (5s)
        # because the LLM frequently rounds to whole seconds.
        s_snapped, e_snapped = _snap_to_segment_boundaries(
            s, e, transcript.segments, tolerance_s=5.0,
        )
        if (s_snapped, e_snapped) != (s, e):
            log.info(
                f"Snapped {s:.2f}-{e:.2f} → {s_snapped:.2f}-{e_snapped:.2f}"
                f" : {txt[:60]!r}"
            )
        out.append({"start": s_snapped, "end": e_snapped, "sentence": txt})
    # Sort and dedupe overlaps (keep first)
    out.sort(key=lambda x: x["start"])
    deduped = []
    for q in out:
        if deduped and q["start"] < deduped[-1]["end"]:
            log.warning(f"Skipping overlap: {q['sentence']!r}")
            continue
        deduped.append(q)
    return deduped


# ---------- LLM-refined cut bounds (word-precise) ----------
_REFINE_SYSTEM_PROMPT = """\
You refine the start/end timestamps of trailer picks using word-precise
transcript data. Output ONLY a JSON object — no prose, no markdown fences.
"""

_REFINE_USER_TEMPLATE = """\
You previously picked this sentence as a trailer hook:

  "{sentence}"

You estimated it spans roughly from {q_start:.2f}s to {q_end:.2f}s in the audio.

You now have the WORD-BY-WORD timestamps around that pick. Your job is to
find the EXACT word timestamps where the complete standalone thought TRULY
begins and ends, so the clip plays as ONE self-contained idea when shown
on social media.

The most common failure mode you must fix:

  THE SPEAKER FINISHES THE SETUP THEN CONTINUES INTO THE PAYOFF, AND
  WHISPER ADDS A SEGMENT BREAK BETWEEN THEM. If you cut at the segment
  break, the clip ends right as the speaker is about to land the point —
  the listener feels jolted, like "wait, they were about to say
  something." YOU MUST EXTEND PAST THAT BREAK to include the payoff or
  wrap-up if it belongs to the same idea.

Concrete decision rules:

  1. EXTEND THE END forward through phrases that COMPLETE the thought:
     "...and that's why X.", "...so that's the reason.", "...because of Y.",
     "...which is what I do now.", "...every single time."
     Even if Whisper put a segment boundary right before, these belong to
     the same idea — INCLUDE them.

  2. DO NOT EXTEND past a phrase that opens a NEW idea or example. Signals
     that a new idea is starting: "Another thing is...", "But then there's...",
     "Let me give you an example of...", a shift to a different subject.

  3. The END should land where the listener feels "yes, that landed, the
     thought is done." Not where punctuation says so — where MEANING says so.

  4. The START should be the first word that makes sense WITHOUT prior
     context. If the picked sentence begins with "And then..." / "So..."
     / "But..." and those words add nothing, skip them. If a short setup
     right before improves the hook ("Here's the thing:"), include it.

  5. Pick from the words below. Do not invent timestamps. Use the exact
     start/end of an actual word in the list.

  6. ERR ON THE SIDE OF LETTING THE THOUGHT BREATHE. A clip that feels
     complete by being a touch long is far better than a clip that
     feels cut off by being a touch short.

Word-by-word window (each line = "start_time-end_time  word"):

{word_lines}

Output a single JSON object on a single line, no prose, no markdown:
  {{"start": <float, start time of the first word of the thought>,
    "end":   <float, end time of the last word of the thought>}}
"""


def refine_cut_bounds_with_llm(
    quotable: dict,
    transcript: Transcript,
    provider,
    window_s: float = 10.0,
) -> tuple[float, float, str]:
    """Ask the LLM to find word-precise start/end of the standalone thought.

    Returns (start, end, refined_sentence) where start/end are exact word
    timestamps from the transcript, and refined_sentence is the actual span
    text. Falls back to derive_cut_bounds() on any parse failure.
    """
    q_start = quotable["start"]
    q_end = quotable["end"]
    sentence = quotable["sentence"]

    # Gather words from segments overlapping the window
    win_lo = max(0.0, q_start - window_s)
    win_hi = q_end + window_s
    win_words: list[Word] = []
    for seg in transcript.segments:
        if seg.end < win_lo or seg.start > win_hi:
            continue
        for w in seg.words:
            if win_lo <= w.start <= win_hi:
                win_words.append(w)

    if len(win_words) < 3:
        log.warning(f"Window too sparse, falling back: {sentence[:50]!r}")
        cs, ce = derive_cut_bounds(quotable, transcript)
        return cs, ce, sentence

    word_lines = "\n".join(
        f"  {w.start:7.2f}-{w.end:7.2f}  {w.text}"
        for w in win_words
    )

    user_prompt = _REFINE_USER_TEMPLATE.format(
        sentence=sentence,
        q_start=q_start,
        q_end=q_end,
        word_lines=word_lines,
    )

    try:
        raw = provider.complete(
            user_prompt=user_prompt,
            system_prompt=_REFINE_SYSTEM_PROMPT,
            max_tokens=200,
        )
        m = re.search(r"\{[^{}]+\}", raw, flags=re.DOTALL)
        if not m:
            raise ValueError("no JSON object found in response")
        data = json.loads(m.group(0))
        new_start = float(data["start"])
        new_end = float(data["end"])
    except (Exception,) as e:  # noqa: BLE001
        log.warning(f"LLM refine failed ({e}): falling back. {sentence[:50]!r}")
        cs, ce = derive_cut_bounds(quotable, transcript)
        return cs, ce, sentence

    # Snap the LLM's reply to the nearest actual word boundaries — defends
    # against rounding (e.g. LLM says "1010.5" but no word ends there).
    first_word = min(win_words, key=lambda w: abs(w.start - new_start))
    last_word = min(win_words, key=lambda w: abs(w.end - new_end))

    if last_word.end <= first_word.start:
        log.warning(f"LLM refine gave inverted range, falling back. {sentence[:50]!r}")
        cs, ce = derive_cut_bounds(quotable, transcript)
        return cs, ce, sentence

    # Build the refined sentence text from the actual word span
    span_words = [
        w for w in win_words
        if first_word.start <= w.start and w.end <= last_word.end
    ]
    refined_sentence = " ".join(w.text.strip() for w in span_words)

    return (
        max(0.0, first_word.start - HEAD_PAD),
        last_word.end + TAIL_PAD,
        refined_sentence,
    )


# ---------- Mechanical cut-bounds (fallback) ----------
def derive_cut_bounds(
    quotable: dict, transcript: Transcript,
) -> tuple[float, float]:
    """Find the tightest [cut_start, cut_end] for a picked sentence by
    looking at Whisper's PER-WORD timestamps in the overlapping segments.

    Segments report end = end-of-segment (often includes trailing silence
    until the next segment starts), so cutting at segment.end + uniform pad
    routinely catches the first word or two of the next sentence. Using
    `last_word.end + small TAIL_PAD` keeps the cut tight on actual speech.
    """
    q_start = quotable["start"]
    q_end = quotable["end"]

    # All words from segments that touch our pick window
    candidates: list[Word] = []
    for seg in transcript.segments:
        if seg.end < q_start - 0.5 or seg.start > q_end + 0.5:
            continue
        candidates.extend(seg.words)

    # Keep only words whose START falls within the pick window (with a
    # small slop). This drops trailing words from the NEXT sentence that
    # happen to share a segment-ish boundary.
    in_window = [
        w for w in candidates
        if q_start - 0.1 <= w.start <= q_end + 0.1
    ]
    if not in_window:
        # Fall back to the snapped LLM bounds
        return max(0.0, q_start - HEAD_PAD), q_end + TAIL_PAD

    cut_start = max(0.0, in_window[0].start - HEAD_PAD)
    cut_end = in_window[-1].end + TAIL_PAD
    return cut_start, cut_end


# ---------- ffmpeg helpers ----------
def _ffmpeg(args: list[str], desc: str) -> None:
    log.debug(f"ffmpeg: {desc}")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *args],
            check=True, capture_output=True, text=True, timeout=600,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed ({desc}): {e.stderr.strip()[-500:]}"
        ) from e


def cut_segment(video: Path, cut_start: float, cut_end: float, out: Path, cfg) -> None:
    _ffmpeg([
        "-ss", f"{cut_start:.3f}",
        "-i", str(video),
        "-t", f"{cut_end - cut_start:.3f}",
        "-c:v", cfg.clip_extract.video_codec,
        "-crf", str(cfg.clip_extract.crf),
        "-preset", cfg.clip_extract.preset,
        "-c:a", "aac", "-b:a", "192k",
        str(out),
    ], desc=f"cut {cut_start:.2f}-{cut_end:.2f}")


def shot_aware_crop(segment: Path, out: Path, cfg) -> None:
    persons, _hf, _fps, w, h = detect_humans_all_per_frame(segment, cfg)
    is_wide = classify_wide_shot_frames(
        persons,
        source_width=w,
        source_height=h,
        sep_threshold_frac=getattr(cfg.crop, "shot_sep_frac", 0.20),
        height_cap_frac=getattr(cfg.crop, "shot_height_cap_frac", 0.70),
        smooth_window_frames=getattr(cfg.crop, "shot_smooth_window_frames", 15),
    )
    smart_crop_916_stacked(segment, persons, is_wide, out, cfg)


def _probe_duration(p: Path) -> float:
    """Return the duration in seconds of `p` via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(p)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return float(out.stdout.strip())


def concat_with_black(clips: list[Path], out: Path, gap_s: float, cfg) -> None:
    """Re-encode all clips into one trailer via the concat filter.

    Between clips, insert a `color=black` video of duration `gap_s` plus a
    matching silent audio source. The concat filter forces uniform codec
    params, so we don't have to fuss over re-encoding params per clip.

    Each clip's audio gets a small afade in at the start and a longer afade
    out at the end, so the cut into the next black gap doesn't sound like
    the speaker is mid-thought.
    """
    tgt_w = cfg.crop.target_width
    tgt_h = cfg.crop.target_height
    rate = SEG_OUT_FPS
    sr = 48000
    inputs: list[str] = []
    # Interleave: clip, [gap], clip, [gap], ..., clip
    n_clips = len(clips)
    parts: list[str] = []
    # Probe each clip's duration so we can place afade-out at the right spot
    clip_durations = [_probe_duration(c) for c in clips]
    # Build inputs and filter labels
    for i, c in enumerate(clips):
        inputs += ["-i", str(c)]
    # Gap source is a lavfi color + anullsrc concatenated into one input each
    # — but lavfi can only produce a single video or audio stream per -i. So
    # we use two lavfi inputs per gap (video + audio), and reference them by
    # index in the filter graph.
    gap_count = max(0, n_clips - 1)
    for _ in range(gap_count):
        inputs += [
            "-f", "lavfi", "-t", f"{gap_s:.3f}",
            "-i", f"color=black:size={tgt_w}x{tgt_h}:rate={rate}",
            "-f", "lavfi", "-t", f"{gap_s:.3f}",
            "-i", f"anullsrc=channel_layout=stereo:sample_rate={sr}",
        ]
    # Build the concat-filter input list. Indexing:
    #   clip i is input i in [0..n_clips)
    #   gap j is inputs at (n_clips + 2*j) for video and (n_clips + 2*j + 1) for audio
    filter_parts = []
    concat_inputs = []
    for i in range(n_clips):
        # Normalize each clip's resolution and fps to match (defensive).
        dur = clip_durations[i]
        fade_out_start = max(0.0, dur - AUDIO_FADE_OUT_S)
        filter_parts.append(
            f"[{i}:v]scale={tgt_w}:{tgt_h}:force_original_aspect_ratio=decrease,"
            f"pad={tgt_w}:{tgt_h}:(ow-iw)/2:(oh-ih)/2,fps={rate},"
            f"format=yuv420p,setsar=1[v{i}];"
            f"[{i}:a]aresample={sr}:async=1,"
            f"afade=t=in:st=0:d={AUDIO_FADE_IN_S},"
            f"afade=t=out:st={fade_out_start:.3f}:d={AUDIO_FADE_OUT_S}[a{i}];"
        )
        concat_inputs.append(f"[v{i}][a{i}]")
        if i < n_clips - 1:
            v_idx = n_clips + 2 * i
            a_idx = n_clips + 2 * i + 1
            concat_inputs.append(f"[{v_idx}:v][{a_idx}:a]")
    n_streams = 2 * n_clips - 1  # n clips + (n-1) gaps
    filter_complex = "".join(filter_parts) + \
        "".join(concat_inputs) + f"concat=n={n_streams}:v=1:a=1[outv][outa]"
    _ffmpeg([
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", cfg.crop.ffmpeg_encoder,
        "-crf", str(cfg.crop.ffmpeg_crf),
        "-preset", cfg.crop.ffmpeg_preset,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(out),
    ], desc=f"concat {n_clips} clips + {gap_count} gaps")


# ---------- Word remapping ----------
def build_trailer_words(
    quotables: list[dict],
    transcript: Transcript,
    gap_s: float,
) -> list[Word]:
    """Pull words from the first-pass transcript that fall inside each
    quotable's spoken window, and remap their times into trailer
    coordinates.

    Each quotable carries pre-computed `cut_start` / `cut_end` (the actual
    ffmpeg cut bounds). A word at video-time t inside cut [cut_start, cut_end]
    maps to trailer-time:
        cum_offset + (t - cut_start)
    where cum_offset = sum of (clip durations + gap_s) for all prior clips.
    """
    out: list[Word] = []
    cum = 0.0
    # Flatten all first-pass words once
    all_words: list[Word] = []
    for seg in transcript.segments:
        all_words.extend(seg.words)
    all_words.sort(key=lambda w: w.start)

    for q in quotables:
        cut_start = q["cut_start"]
        cut_end = q["cut_end"]
        clip_duration = cut_end - cut_start

        for w in all_words:
            # Only words within the spoken window (NOT the head/tail pad),
            # so the head/tail audio fade region has no captions on it.
            if w.start < q["start"] - 0.05 or w.end > q["end"] + 0.05:
                continue
            t_in_clip = w.start - cut_start
            t_end_in_clip = w.end - cut_start
            out.append(Word(
                start=cum + t_in_clip,
                end=cum + t_end_in_clip,
                text=w.text,
                confidence=w.confidence,
            ))
        cum += clip_duration + gap_s

    return out


# ---------- Cache helpers ----------
def _cache_dir(cfg, video: Path) -> Path:
    h = hashlib.sha1(str(video.resolve()).encode()).hexdigest()[:10]
    base = Path(cfg.paths.cache_dir) / f"{video.stem}-{h}"
    (base / "trailer").mkdir(parents=True, exist_ok=True)
    return base


def _transcript_to_json(t: Transcript) -> dict:
    return {
        "language": t.language,
        "segments": [
            {
                "start": s.start, "end": s.end, "text": s.text,
                "words": [
                    {"start": w.start, "end": w.end, "text": w.text,
                     "confidence": w.confidence}
                    for w in s.words
                ],
            }
            for s in t.segments
        ],
    }


def _transcript_from_json(d: dict) -> Transcript:
    from src.types import TranscriptSegment
    segs = []
    for s in d["segments"]:
        words = [Word(**w) for w in s.get("words", [])]
        segs.append(TranscriptSegment(
            start=s["start"], end=s["end"], text=s["text"], words=words,
        ))
    return Transcript(language=d["language"], segments=segs)


# ---------- Main ----------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("video", help="Path to source podcast video")
    ap.add_argument("--out", default=None,
                    help="Output trailer path (default: /tmp/trailer_<stem>.mp4)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    _load_dotenv(REPO / ".env")
    setup_logging("DEBUG" if args.verbose else "INFO")

    video = Path(args.video).resolve()
    if not video.exists():
        log.error(f"Not found: {video}")
        return 2

    out = Path(args.out) if args.out \
        else Path(f"/tmp/trailer_{video.stem}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(REPO / "config" / "default.yaml")
    provider = build_provider(cfg.llm)

    cache = _cache_dir(cfg, video)
    trailer_cache = cache / "trailer"

    # 1. Probe + audio
    meta = ingest(video)
    log.info(f"Source: {video.name}  {meta.width}x{meta.height} "
             f"{meta.fps:.2f}fps  {meta.duration:.1f}s")
    audio_wav = extract_audio(
        video, cache / "audio.wav",
        sample_rate=cfg.audio.sample_rate, codec=cfg.audio.codec,
    )

    # 2. Transcribe (cached as full_transcript.json under trailer/)
    transcript_cache = trailer_cache / "full_transcript.json"
    if transcript_cache.exists():
        log.info(f"Reusing cached transcript: {transcript_cache}")
        transcript = _transcript_from_json(json.loads(transcript_cache.read_text()))
    else:
        transcript = transcribe_first_pass(audio_wav, meta.duration, cfg)
        transcript_cache.write_text(json.dumps(_transcript_to_json(transcript)))
        log.info(f"Cached transcript → {transcript_cache}")

    # 3. LLM picks quotables (cached)
    quotables_cache = trailer_cache / "quotables.json"
    if quotables_cache.exists():
        log.info(f"Reusing cached quotables: {quotables_cache}")
        quotables = json.loads(quotables_cache.read_text())
    else:
        log.info("Asking LLM to pick quotable standalone sentences...")
        quotables = pick_quotables(transcript, provider, cfg, meta.duration)
        quotables_cache.write_text(json.dumps(quotables, indent=2))
        log.info(f"Cached quotables → {quotables_cache}")

    if not quotables:
        log.error("LLM returned zero quotables — bailing.")
        return 1

    # Compute LLM-refined, word-precise cut bounds for each pick (cached).
    refined_cache = trailer_cache / "refined_bounds.json"
    refined_data: dict | None = None
    if refined_cache.exists():
        try:
            refined_data = json.loads(refined_cache.read_text())
            # Invalidate if the picks changed
            cached_keys = [(d["start"], d["end"], d["sentence"]) for d in refined_data]
            current_keys = [(q["start"], q["end"], q["sentence"]) for q in quotables]
            if cached_keys != current_keys:
                log.info("Refined-bounds cache stale (picks changed); recomputing.")
                refined_data = None
        except (KeyError, json.JSONDecodeError):
            refined_data = None

    if refined_data is None:
        log.info("Asking LLM to find word-precise cut bounds for each pick...")
        refined_data = []
        for q in quotables:
            cs, ce, refined_sentence = refine_cut_bounds_with_llm(
                q, transcript, provider,
            )
            refined_data.append({
                **q,
                "cut_start": cs,
                "cut_end": ce,
                "refined_sentence": refined_sentence,
            })
        refined_cache.write_text(json.dumps(refined_data, indent=2))
        log.info(f"Cached refined bounds → {refined_cache}")
    else:
        log.info(f"Reusing cached refined bounds: {refined_cache}")

    quotables = refined_data

    log.info(f"Got {len(quotables)} quotables; "
             f"total spoken duration ≈ "
             f"{sum(q['cut_end']-q['cut_start'] for q in quotables):.1f}s")
    for i, q in enumerate(quotables, 1):
        log.info(
            f"  [{i:02d}] cut {q['cut_start']:7.2f}-{q['cut_end']:7.2f}  "
            f"({q['cut_end']-q['cut_start']:.2f}s)  "
            f"refined: {q['refined_sentence']}"
        )

    # 4. Cut + shot-aware crop each quotable
    cropped_clips: list[Path] = []
    for i, q in enumerate(quotables):
        seg = trailer_cache / f"q_{i:02d}_segment.mp4"
        crop = trailer_cache / f"q_{i:02d}_cropped.mp4"
        if not seg.exists():
            cut_segment(video, q["cut_start"], q["cut_end"], seg, cfg)
        if not crop.exists():
            log.info(f"  [{i+1}/{len(quotables)}] crop ...")
            shot_aware_crop(seg, crop, cfg)
        cropped_clips.append(crop)

    # 5. Concat cropped clips with black-frame gaps
    stitched = trailer_cache / "stitched.mp4"
    log.info(f"Stitching {len(cropped_clips)} clips with "
             f"{GAP_SECONDS:.2f}s black gaps...")
    concat_with_black(cropped_clips, stitched, gap_s=GAP_SECONDS, cfg=cfg)

    # 6. Burn subtitles with remapped trailer-time words
    log.info("Building trailer-time word timeline + burning subtitles...")
    trailer_words = build_trailer_words(quotables, transcript, gap_s=GAP_SECONDS)
    burn_subtitles(stitched, trailer_words, out, cfg, title="")

    log.info(f"Trailer → {out}  ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
