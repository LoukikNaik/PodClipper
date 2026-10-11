# Improvements

## Implemented: local retries for missing caption speech

Found in the v0.1.2 agent reel **Work All-Out Till 35, Then Enjoy**, from
source `JqwfZdW4WvI`, bounds 1318.64–1395.04 seconds. Around reel time
20–40 seconds, second-pass Whisper drops speech and assigns `करेंगे` to
28.06–37.64 seconds, followed by `करने` at 37.64–40.32 seconds.

The cached transcript contains 145 words over 76.67 seconds, above the current
1.5 words/second retry threshold. That whole-clip average misses local failures.
Cache loading skips transcription checks, and caption cleanup cannot recover
speech absent from its input. The agent's sampled review approved this draft
despite noting transcript uncertainty.

An isolated large-v3 retry of reel time 20–44 seconds recovered 57 words,
including the missing discussion of ambitions, money and building a home.
A corrected caption export is saved locally under the agent's ignored
`data/reels/v0.1.2/2026-10-11/caption-fix/` directory.

The automatic caption pass now checks Silero speech regions, unusually long
word timings, local word density, and available first-pass word counts. It
validates cached captions too and retries only suspicious audio sections with
context. Replacement words use clip-relative timestamps and matching context
words to preserve boundaries. Repaired caches are written atomically.

Defaults allow two attempts per affected section, at most eight retry windows,
and at most 24 seconds of audio per window including four seconds of context on
each side. Unresolved coverage stops that clip before cleanup or export.
`words.quality.json` records outcomes. Settings remain internal defaults under
`transcribe.caption_repair`, with optional YAML overrides.

These checks detect likely omissions, not every wrong word. Speech detection
and the first pass can make mistakes. Slower speech may need threshold tuning;
first-pass text is used only to flag omissions, never as replacement captions.

Running the new recovery on the original broken cache detected reel time
24–42 seconds, retried 20–44 seconds once, and increased the caption transcript
from 145 to 179 words. The repaired transcript passed all coverage checks.
Local evidence is under ignored `outputs/caption_coverage_repair/`.

## Implemented: divider-aware podcast crops and cut transitions

The Rashmika/Vijay/Shubhankar episode contains edited two- and three-view
layouts. Crops now respect thin vertical divider lines, including pose input
and fallback paths. Auto mode shows at most two source views and keeps the
chosen views stable within a layout. Selection uses person size, not active
speaker identification.

Detected camera/layout cuts reset tracking even when the same participant
remains visible. Output rows stay filled. A 0.12-second crossfade softens cuts
without changing audio timing; `crop.scene_cuts.crossfade_seconds: 0` preserves
hard cuts. No zoom animation or side padding is added.

An eight-second preview from source 690–698 seconds covers 200 frames and two
layout changes. All 406 crop-bound checks passed, and the output decoded fully.
The accepted preview and source overlay are under ignored
`outputs/podcast_dividers_review/cut-crossfade-*.mp4`.

Release review added regression coverage for divider detection, crop bounds,
three-view selection, scene-cut resets and crossfade timing. It also fixed
oversized divider margins creating invalid views and stale wide-shot flags
stacking a single foreground subject with a tiny background detection.

Limitations: faint/broken/decorative dividers can be missed. Cut detection uses
image differences; subtle cuts can be missed and lighting changes can trigger
resets. General face containment and caption/title quality still need work.

## Deferred: face cropping and incorrect stacked layouts

**Status:** Open; deferred by request. Continue with the podcast-to-reels flow
for now. No crop behavior was changed during this investigation.

### Reproduction and evidence

Found in the **v0.1.1** fresh macOS Homebrew end-to-end run, using the first
five minutes of Dan Pink's TED talk, *The puzzle of motivation*. The pipeline
exported a captioned 1080 × 1920 reel successfully, but visual quality was poor.

- [Runner and artifacts](https://github.com/LoukikNaik/homebrew-tap/actions/runs/38145569445)
- Output: `reel_01_cash-rewards-made-people-slower.mp4`, 90.12 seconds.
- Selected source bounds: 171.82–261.78 seconds; extraction includes padding.
- [Detailed local report](outputs/release_0.1.1_e2e/REPORT.md)
- [Local frame samples and scores](outputs/release_0.1.1_e2e/visual_review/sampling.json)

The local evidence is in ignored `outputs/` and is not included in a clone.
Runner artifacts expire; the observations below preserve the findings.

| Reel time | Observed defect |
| --- | --- |
| 0.00–0.04 s | One frame of the preceding candle slide flashes before the speaker. The opening title overlaps his hair. |
| 8.56 s, 13.24 s | Framing changes leave part of the face outside the crop. |
| 36.80–38.20 s | Most of his face is outside the right edge in every sample taken 0.2 seconds apart. |
| 49.08–51.28 s | Repeated switches between one view and stacked panels; panels show an audience head, empty background, or the stage screen. |
| 54.48 s, 59.68 s, 60.00 s | Unsuitable stacked views persist in these samples, including the speaker and his projected image. One view returns at 60.24 s. |
| 65.52 s | Much of the face is outside the left edge. |
| 72.04–72.36 s | Brief stacked-layout flash duplicates the speaker, then follows his raised hand in the lower panel. One view returns by 72.44 s. |
| 74.00 s, 76.76 s, 77.24–77.48 s | Further face clipping at the right edge. |

Review covered 15 regularly spaced frames, 12 seeded random frames, before/at/
after samples around 12 large visual changes, and 30 focused follow-up samples.
All 2,253 output frames were numerically scored for visual changes. These are
sampled observations, not exact start/end times for every defect.

### Code findings and proposed fixes

- **Keep the face inside the crop.** In `src/podclipper/crop.py`, the active
  single-panel branch centers on the body box and can retain a stale crop while
  body overlap remains high. Use face/head tracking with a safety margin;
  override locking and smoothing when the face approaches the crop edge.
- **Make stacking conservative.** `classify_wide_shot_frames` in
  `src/podclipper/timeline.py` uses person-box geometry, not participant identity.
  The default `shot_min_dwell_frames` is zero. Require persistent qualified
  participant tracks; handle audience and projected images; prefer one view
  when uncertain.
- **Use consistent candidates.** Stacked rendering filters by maximum person
  height but omits the classifier's minimum size filter. Apply the same
  qualification and track selection in both places.
- **Handle actual camera cuts.** The renderer infers cuts from body overlap.
  Add image-based shot boundaries, reset framing on cuts, and smooth movement
  within each shot.
- **Use a safe fallback.** Fit a wider view when portrait cropping cannot keep
  the face visible. Keep the title clear of the head and inspect extraction
  padding to remove the preceding-slide flash without losing opening speech.
- **Improve verification.** The scorecard reported 100% face visibility despite
  visible face loss; crop stability was 0.29. Measure face containment and layout
  switches, preserve source/detection traces, and include random and transition
  samples in future reports. The current CI verifier saves only two fixed frames.

### When resuming

Re-render this same selection, compare the timestamps above, then verify a
genuine multi-speaker podcast. Preserve appropriate stacked layouts for actual
participants. This talk exposed defects that may also affect podcasts; the
successful export alone does not establish visual quality for every podcast.
