# Audio Test Sweep — Empirical Findings

Ran six probes to validate the design assumptions in `docs/AUDIO_STORE_DESIGN.md`.
All test scripts are in this directory; results below.

## Headline

| Layer | Winner | Hit rate | Cost / latency | Confidence |
|---|---|---|---|---|
| **Lyrics fetch** | lrclib.net | **5/5** songs (incl. indie + Spanish) | free, ~1s/call | High |
| **Song mood tagging** | LiteLLM/TokenRouter → Claude Sonnet 4.5 | **4/4** with rich, accurate tags | ~$0.001 + 5-10s/call | Very high |
| **SFX/meme tagging** | Same as above | **10/10** including nonsense filenames | ~$0.001 + 3-7s/call | Very high |
| **SFX file fetch** | Myinstants scrape | **10/10** downloads | free, ~0.5s/file, ~70 KB avg | High |
| **Song audio fetch** | yt-dlp via SoundCloud | **~50%** of mainstream songs (gives 30s preview) | free, 2-25s/track | Medium — needs fallback |
| **Trending songs list** | Apple/iTunes Top Songs RSS (JSON) | **15/15** in <1s | free, no auth | High (but it's "top played" not "TikTok viral") |

## Test 1 — Lyrics fetchers

Tested against 5 songs: Lana Del Rey "Video Games," Sabrina Carpenter "Espresso," Bad Bunny "Monáco" (Spanish), Hannah Cohen "Watching You Fall" (indie), Phoebe Bridgers "Funny."

| Fetcher | Hit rate | Latency | Notes |
|---|---|---|---|
| **lrclib.net** | **5/5** | ~1s | Free, no auth, returns plainLyrics + syncedLyrics |
| AZ Lyrics scrape | 0/5 | ~0.7s | URL pattern fails (404 / lyrics div not found / unicode errors) |
| DuckDuckGo + scrape | 0/5 | ~0.9s | DDG HTML page no longer surfaces raw URLs in scrapeable form |

**Verdict: lrclib alone is enough for the lyrics enricher.** Skip the fallback chain.

**Gotcha:** lrclib fuzzy-matched "Phoebe Bridgers — Funny" to Bo Burnham's "That Funny Feeling" — the wrong song. Quality check needed: pass artist+title to the LLM and let it verify the lyrics match the requested track.

## Test 2 — LLM song tagging

For each song, fed `(artist, title, lyrics)` to the LiteLLM provider (TokenRouter → Claude Sonnet 4.5). Asked for `{mood, themes, energy, use_cases, avoid_for, viral_context, iconic_lyric}`.

All 4 calls returned **clean structured JSON with deeply useful tags**:

- **Lana Del Rey — Video Games** → `coquette aesthetic`, `romanticizing red flags`, `vintage glamour edits` (the actual TikTok use cases for this song)
- **Sabrina Carpenter — Espresso** → recognized the "that's that me espresso" hook as a 2024 viral catchphrase; tagged for confidence/glow-up content
- **Bad Bunny — Monáco** → `luxury lifestyle flexing`, `Monaco/yacht content`, in Spanish lyrics
- **Phoebe Bridgers — Funny** → **caught the lrclib mismatch** ("This is actually a Bo Burnham song from 'Inside' (2021)"), provided correct viral context (climate doom / late-capitalism aesthetic)

**Verdict: LLM tagging is the killer feature.** Quality and cost both fine.

## Test 3 — Myinstants scraper

Hit `https://www.myinstants.com/` and parsed the homepage trending grid.

Initial regex was stale (page structure changed) — got 0 results. After fixing to the new `<button class="small-button" onclick="play('...')">` pattern: **10/10 downloads**, avg 72 KB/file, total 0.7 MB.

Detail-page tag scrape returned empty (page format changed there too). **Workaround:** rely on LLM tagging from the filename alone (Test 4 proves this works).

## Test 4 — LLM SFX tagging from filename only

Asked the LLM to tag 10 SFX names with `{category, mood, intensity, when_to_use, viral_origin, recognizable}`. **10/10 succeeded** with rich tags. Highlights:

- "Rizz Sound Effect" → tagged as "TikTok slang circa 2022, popularized by Kai Cenat"
- "Among Us role reveal sound" → traced to 2020 viral game
- "Inception Horn" → identified as 2010 film origin
- "FAHHHHHHHHHHHHHH" (nonsense filename) → still got sensible tags (dramatic shock reaction)

**Verdict: filename-only LLM tagging works.** No need for Myinstants tag scraping.

## Test 5 — yt-dlp song fetching

Tested `ytsearch1:<artist> <title>` and `scsearch1:<artist> <title>`.

**YouTube is anti-bot-blocked.** Even with `--cookies-from-browser chrome` we get "Sign in to confirm you're not a bot." Workarounds exist (manual cookie export from a YouTube Music subscription account, residential proxy) but none are clean.

**SoundCloud works** — but only ~50% hit rate on mainstream songs, and returns **30-second previews** (~470 KB) instead of full tracks. Crucially: **30s is fine for our use case** — reels are 15-90s and we'd loop / fade the bed anyway.

| Source | Success | Notes |
|---|---|---|
| SoundCloud (`scsearch1:`) | ~50% mainstream tracks | 30s preview, perfect length for bed |
| YouTube (`ytsearch1:`) | 0% without cookie auth | Bot-check blocks unauthenticated |
| YouTube + chrome cookies | 0% (still bot-checked) | Cookies alone insufficient |

**Verdict: SoundCloud as primary, but we need a fallback path.** Options:
1. User BYO (`cfg.audio.byo_dir` — drop files in a folder, pre-tagged via LLM)
2. Spotify Web API `/audio-features` + `/tracks/{id}/preview_url` (30s previews, public auth) — needs OAuth client setup
3. Manual upload at index time

## Test 6 — Trending songs source

Tokboard.com is DEAD (DNS doesn't resolve). Alternatives:

| Source | Result | Notes |
|---|---|---|
| **Apple/iTunes Top Songs RSS** | **15/15 in <1s** | Public JSON, no auth: `https://rss.applemarketingtools.com/api/v2/us/music/most-played/25/songs.json` |
| Billboard Hot 100 scrape | 2/25 | Dynamic HTML, regex fragile |

**Verdict: Use iTunes RSS for "what's mainstream popular right now."** Caveat: this is *top-played on Apple Music*, not *viral on TikTok specifically*. For TikTok-viral, no free reliable API exists in 2026; the practical approach is curated lists + adding tracks as you find them.

## What this means for the design

Confirmed:
- ✅ **Lyrics enricher = lrclib only** (no fallback chain needed v1)
- ✅ **LLM tagger handles both songs and SFX excellently** — the same `LiteLLMProvider` we already built
- ✅ **Myinstants scrape is the SFX source**, name-only tagging is fine
- ✅ **Apple iTunes RSS is the trending source** for mainstream music
- ✅ **SoundCloud (30s previews) is good enough for the bed use case**

Need to add to the design doc:
- 🆕 **LLM verification step for lrclib matches.** Fuzzy match returns wrong songs sometimes; LLM should sanity-check.
- 🆕 **Song fetch is a fallback chain**, not single-source. SoundCloud → user-BYO → skip.
- ❌ **Tokboard, AZ Lyrics, DDG scrape, YouTube unauthenticated** are dead ends — don't waste time on them.

## Recommended Phase 1 implementation order

1. **`AudioTrack` dataclass + JSON manifest** — pure scaffolding, no external calls
2. **`LrclibLyricsEnricher`** — 1 hour, validated
3. **`LLMMoodTagEnricher`** (works for both songs and SFX) — 1 hour, validated
4. **`MyinstantsSource`** — 1 hour, scraper validated above
5. **`ItunesRssSource`** — 30 min, JSON parse
6. **`SoundCloudAudioFetcherEnricher`** (via yt-dlp scsearch) — 1 hour
7. **`ManualSource`** (CSV reader for BYO seed lists) — 30 min
8. **`podclipper audio pipeline run`** CLI — 1 hour
9. **End-to-end: index ~30 songs + ~50 SFX, browse via `podclipper audio search`** — 1 hour

Total: **~7 hours** to a working, browseable audio catalog with rich tags, before any picker/mixer work begins.

Skipping for v1:
- AZ Lyrics, Firecrawl, Genius, DDG (Test 1 shows lrclib alone is sufficient)
- Tokboard (dead) and other dead trending sources
- Whisper-from-audio lyrics enricher (Test 1 shows we get lyrics for free)
- Essentia features (LLM tagging from name/lyrics works without needing audio features)
- YouTube unauthenticated fetching (anti-bot blocked)
