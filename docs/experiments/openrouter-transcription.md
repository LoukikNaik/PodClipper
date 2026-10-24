# OpenRouter transcription experiments — October 16, 2026

## Recommendation

Qwen3 ASR 0.6B is worth a larger trial for English episode transcripts. On two
random English excerpts its text was broadly consistent with Scribe and it
passed the existing timing and speech-coverage checks. It costs about $0.012 per
audio hour through the currently available OpenRouter endpoint.

For Hindi-English reel captions, Scribe v2 is the stronger candidate from this
experiment. It consistently retained more intelligible mixed-language speech
and returned usable word timestamps. Use it as an optional caption fallback
before considering a replacement for all local transcription. These comparisons
left the production defaults unchanged. A subsequent experimental `--scribe`
flag now offers full-source transcription and caption reuse; see the README.

These are preliminary observations, not word-error rates: no human reference
transcript or word-by-word audio alignment review was produced. Agreement with
Scribe is not proof of accuracy. A local coverage check can pass inaccurate text.

## Pricing checked against OpenRouter

Prices are USD per hour of input audio, calculated from the API's per-second
rates. Excludes cleanup/translation LLM calls, retries, and credit purchase fees.

| Model | Current advertised price/hour | Observation |
|---|---:|---|
| Qwen3 ASR 0.6B | $0.011988 | Promising on English; mixed-language errors, repetition, and timing failures on harder excerpts. |
| Whisper Large V3 Turbo | $0.011988 | Fast, but dropped speech or garbled code switching on several excerpts. |
| NVIDIA Nemotron 3.5 ASR 0.6B | $0.011988 | Tested on the two difficult Javed excerpts; text and timing defects. |
| Qwen3 ASR 1.7B | $0.027 | Did not resolve Hindi-English issues consistently; slower than 0.6B in this run. |
| Whisper Large V3 | From $0.027 | Routing varied: actual effective rates in the two requests were about $0.09 and $0.111/hour; one returned no words. |
| ElevenLabs Scribe v2 | $0.11 promotional / $0.22 regular | Strongest observed mixed-language output and most consistent caption timing checks. |

The Scribe discount ends October 19 at 8 AM Pacific, according to OpenRouter's
launch announcement. Budget using the regular price. Five one-minute caption
transcriptions would cost about $0.0183/day or $0.55 per 30 days at that rate;
whole episode transcription would be additional.

Sources:

- [OpenRouter STT guide](https://openrouter.ai/docs/guides/overview/multimodal/stt)
- [Scribe v2 model and pricing](https://openrouter.ai/elevenlabs/scribe-v2)
- [ElevenLabs launch offer](https://openrouter.ai/blog/announcements/elevenlabs-on-openrouter/)
- [Qwen3 ASR 0.6B](https://openrouter.ai/qwen/qwen3-asr-0.6b)
- [Qwen3 ASR 1.7B](https://openrouter.ai/qwen/qwen3-asr-1.7b)
- [Whisper Turbo](https://openrouter.ai/openai/whisper-large-v3-turbo)
- [Live transcription model catalog](https://openrouter.ai/api/v1/models?output_modalities=transcription)

## Initial difficult excerpts

Two previously problematic excerpts from Javed Akhtar's interview with Saurabh
Dwivedi (`q8SE1OV5w0o`):

- Film villains: source 1124.59–1180.29 seconds, 55.7 seconds.
- Faith and rationality: source 2699.98–2787.98 seconds, 88 seconds.

Six models, 16 total requests including auto-detection and explicit-Hindi
variants. Scribe's two language variants were nearly identical; manually forcing
Hindi did not produce a substantial improvement over its automatic detection.

Scribe recovered the film title Shree 420 and the phrase about the rich man and
industrialist where cheaper outputs contained unrelated phonetic text. In the
faith excerpt it retained the concluding English argument that the earlier
local export had left incomplete. This is a qualitative comparison of returned
transcripts, not a verified transcript of the source audio.

Qwen 0.6B returned some timestamps beyond the 88-second recording. Qwen 1.7B
returned plausible-looking words but failed coverage checks in multiple ranges.
Nemotron returned overlapping timing that failed the pipeline's word validator.
Whisper Turbo also left coverage gaps. All four Scribe requests passed both the
pipeline's timing validator and read-only speech-coverage audit.

Initial requests cost **$0.01600404** in total, from `usage.cost` receipts.

## Additional random excerpts

Convenience sample of four other already downloaded podcast recordings.
Random seed `20261016`; one 60-second excerpt per recording, selected uniformly
after the first three minutes and before the final minute of the available
file. Selection happened before requesting any model output. Tim Ferriss was
sampled from an existing 5–25 minute file; Leila Hormozi from a first-20-minutes
file. This is not uniform sampling across each full episode or across YouTube.

| Podcast | Original source time | Group |
|---|---|---|
| [Tim Ferriss: The #1 Reason You Feel Stuck](https://www.youtube.com/watch?v=s0DohADRRlY&t=616s) | 10:16–11:16 | English |
| [Leila Hormozi: The Psychology of Making Money](https://www.youtube.com/watch?v=9FtradY1AI4&t=925s) | 15:25–16:25 | English |
| [Rashmika & Vijay / Shubhankar Mishra](https://www.youtube.com/watch?v=JqwfZdW4WvI&t=4773s) | 79:33–80:33 | Hindi-English |
| [Ekta Kapoor / TRS](https://www.youtube.com/watch?v=lqOH-zxtCDE&t=2288s) | 38:08–39:08 | Hindi-English |

Same extracted 16 kHz mono audio per model, automatic language detection,
temperature zero, verbose JSON with word timestamps, no vocabulary hints or LLM
cleanup. Four models × four excerpts = 16 additional billed requests.

### Existing caption checks

Pass means finite ordered words within the recording and no suspicious speech
coverage ranges using the current `transcribe.caption_repair` defaults. This
does not establish semantic correctness or precise karaoke alignment. A flagged
range is evidence for review or repair, not proof that every word there is absent.

| Model | Tim Ferriss | Leila Hormozi | Rashmika/Vijay | Ekta Kapoor |
|---|---|---|---|---|
| Scribe v2 | Pass | Pass | Pass | Pass |
| Qwen3 ASR 0.6B | Pass | Pass | Coverage flagged | Invalid timing + coverage flagged |
| Qwen3 ASR 1.7B | Pass | Pass | Coverage flagged | Invalid timing + coverage flagged |
| Whisper Turbo | Coverage flagged | Pass | Coverage flagged | Pass |

### Text observations

- English: Qwen 0.6B and 1.7B retained broadly similar arguments to Scribe. Qwen
  0.6B confused a person's name in the Tim Ferriss excerpt. Whisper Turbo omitted
  part of the explanation in that excerpt, matching a flagged coverage range.
- Rashmika/Vijay: Scribe preserved the mixed-language conversation, names, and
  life-lessons discussion. Qwen retained much of it with mistakes; Whisper
  Turbo's output contained major omissions and repeated filler-like text.
- Ekta Kapoor: Qwen 0.6B emitted a long repeated Hindi phrase and invalid word
  timestamps. Qwen 1.7B was better but still failed timing and coverage checks.
  Whisper Turbo covered the speech but translated/paraphrased parts of Hindi
  and garbled some wording. Scribe retained the code switching more consistently.
- All systems cut off at arbitrary excerpt boundaries; these samples were
  chosen to compare ASR, not to serve as selected reels.
- Scribe writes Hindi primarily in Devanagari and preserves much of the spoken
  English in Latin script. Roman Hinglish still needs cleanup/transliteration.
  English subtitle translation is a separate operation; do not assume an STT
  endpoint also supports Whisper's translation task.

### Cost and latency

| Model | Cost for four one-minute samples | Observed request latency |
|---|---:|---:|
| Qwen3 ASR 0.6B | $0.0007992 | 3.04–8.71 seconds |
| Whisper Turbo | $0.0007992 | 2.71–7.63 seconds |
| Qwen3 ASR 1.7B | $0.0018 | 16.77–20.87 seconds |
| Scribe v2 | $0.0073333 | 2.27–3.33 seconds |

Random-sample requests cost **$0.01073173**. Both experiment sets together cost
**$0.02673578** (about 2.7 US cents). Latencies include HTTP request time but
exclude extraction. Requests ran with concurrency, so this is not a controlled
service latency benchmark.

## Full-podcast flag trial

The experimental `--scribe` flow was then run on the complete 4,532-second Javed
Akhtar podcast. It produced **13,185 words** from 16 overlapping five-minute
requests. A small boundary retry from the first attempt brought measured total
transcription charges to **$0.14048314** at promotional pricing. Selection,
cleanup, and evaluation LLM charges are additional and not included here.

A coverage check initially flagged the start of an overlapping chunk even
though that speech belonged to the preceding chunk. Coverage now audits each
chunk's owned interval, retaining original audio context for repairs. Successful
request caches were reused on resume. A separate read-only audit of the entire
merged transcript, including the joins, found valid word timings and no
suspicious speech-coverage ranges.

The same film-villains moment was exported from a saved selection with `--scribe`:

- Cached full transcript reused; 116 caption words sliced from its timestamps.
- No normal second-pass transcription and no extra Scribe requests for export.
- Output: 56 seconds, 1080×1920 H.264 video, AAC audio, burned Romanized captions.
- Content score: 3.2/5, verdict **review**, not automatic publication approval.
- Longer-context Scribe still misheard the film title as “Shri chaar B”, whereas
  the earlier short standalone request recovered Shree 420. Passing coverage is
  insufficient to establish word accuracy. Targeted semantic review/repair
  remains necessary even if the mandatory second pass is removed.

Fresh selection also produced three candidates. When caption transliteration is
enabled, the Scribe selection/title-review path requests English or Romanized
Hinglish titles to avoid carrying Hindi script into the existing title renderer.
This instruction is limited to the opt-in flow; the Whisper default is unchanged.

Local artifacts are under `outputs/scribe-full-trial/`, including
`full-coverage-audit.json`, `render.log`, the saved selection reports, the exported
MP4 with exact caption words, and the source request receipts in the isolated
cache. The export was inspected with ffprobe and a sampled frame; every frame
and caption word has not been manually reviewed.

## Reproduce and inspect

`dev/try_scribe.py` is a standalone experiment using the existing `LLM_API_KEY`.
It always calls the official OpenRouter transcription endpoint, keeps credentials
out of saved metadata, caps excerpt length at 180 seconds, estimates cost using
the highest listed endpoint rate, and makes no automatic billed retries.

```sh
python dev/try_scribe.py /path/to/podcast.mp4 \
  --start 600 --end 660 \
  --model qwen/qwen3-asr-0.6b \
  --env-file .env \
  --output outputs/asr-example
```

Requires FFmpeg and `python-dotenv`. Supported comparison models are an explicit
duration-priced allowlist. `--language hi` enables an explicit-language trial;
`--keyterm` is an optional Scribe-specific probe whose forwarding was not tested
here. No separate provider API key is needed. Model selection is experimental;
the shared production LLM model and local Whisper configuration are unchanged.

Local ignored artifacts:

- `outputs/scribe-experiment-2026-10-16/*/`: initial audio, raw response, text,
  normalized words, usage/latency summaries, and read-only coverage audits.
- `outputs/scribe-experiment-2026-10-16/random/manifest.json`: random sample plan.
- `outputs/scribe-experiment-2026-10-16/random/audited-results.json`: measured results.
- `outputs/scribe-experiment-2026-10-16/random/comparison.html`: side-by-side raw
  transcripts with audio playback, source links, prices, and diagnostic links.

The CLI saves structural timing diagnostics. Coverage audits were additionally
run against `podclipper.transcribe_repair` without invoking any repair or touching
the agent's active run caches. No new reels were rendered, queued, or posted by
these experiments.
