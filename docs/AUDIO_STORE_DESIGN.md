# Audio Store — Design Doc

> A pluggable catalog of audio (songs, viral SFX, instrumental beds) that
> PodClipper's picker can query at reel-build time. **The store knows what
> exists and what it means. It does NOT know how to pick or how to mix —
> those are downstream concerns.**

## Why a store

Three independent forces make this worth designing carefully:

1. **Sources will multiply.** Today: Tokboard, Myinstants, manual lists. Tomorrow: YouTube Shorts trending scraper, Spotify Viral 50, user's iTunes library, Apple Music API, a specific subreddit's weekly thread. Hard-coding any of them locks us out of the others.
2. **Enrichment is a pipeline, not a function.** A song row goes from `{artist, title}` → `{... lyrics}` → `{... mood, themes}` → `{... local mp3}`. Each step has its own latency, cost, and failure mode. They run independently, idempotently, and out of order.
3. **The taxonomy keeps changing.** Today we tag `mood: melancholic`. Tomorrow we want `viral_phase: rising | peaking | declining`, `cultural_context: post-breakup TikTok 2024`, `clip_match_quality: snippet-aligned | full-song-vibe`. Schema needs to grow without migration pain.

The right abstraction is a **content-addressable catalog with composable enrichers**, not a one-shot script.

## Components

```
┌─────────────────────────┐    ┌─────────────────────────┐
│   SOURCES (pluggable)   │    │  ENRICHERS (pluggable)  │
│                         │    │                         │
│  • TokboardSource       │    │  • LrclibLyricsEnricher │
│  • MyinstantsSource     │    │  • FirecrawlLyricsEnr.. │
│  • YouTubeShortsSource  │    │  • GeniusEnricher       │
│  • SpotifyPlaylistSource│    │  • WhisperLyricsEnr..   │
│  • LocalFolderSource    │    │  • LLMMoodTagEnricher   │
│  • ManualSource         │    │  • EssentiaFeatureEnr.. │
│                         │    │  • DurationEnricher     │
│  yields →               │    │  • YtdlpAudioFetcherEnr │
│  AudioTrack stubs       │    │                         │
└────────────┬────────────┘    └────────────┬────────────┘
             │                              │
             ↓                              ↓
       ┌─────────────────────────────────────────┐
       │     AUDIO MANIFEST  (SQLite or JSON)    │
       │                                         │
       │  • AudioTrack records                   │
       │  • Provenance (which source, when)      │
       │  • Enrichment state (which enrichers    │
       │    have run, last run timestamp)        │
       │  • Local-cache pointers (path on disk)  │
       │  • User overrides (always win)          │
       └────────────┬────────────────────────────┘
                    │
                    ↓
         downstream consumers (NOT part of the store):
         • Picker (LLM-driven reel ↔ track matching)
         • Mixer (ffmpeg audio chain)
         • CLI search / browse
         • Future: web UI, tag editor, etc.
```

## Core data model

```python
from typing import Literal
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

Category = Literal["sfx_drop", "song_clip", "instrumental_bed"]

@dataclass
class AudioTrack:
    # Identity
    track_id: str                       # uuid or stable content hash
    category: Category

    # Provenance
    source_name: str                    # "tokboard", "myinstants", "manual"
    source_url: str | None              # original URL where we found it
    source_metadata: dict = field(default_factory=dict)  # source-specific blob

    # Basic identity
    title: str | None = None
    artist: str | None = None
    duration_s: float | None = None

    # Enriched fields (any can be None until enricher fills it)
    lyrics: str | None = None
    lyrics_provider: str | None = None      # "lrclib", "firecrawl", "whisper"
    mood: list[str] = field(default_factory=list)         # ["melancholic", "ironic"]
    themes: list[str] = field(default_factory=list)       # ["heartbreak", "denial"]
    energy: Literal["low", "med", "high"] | None = None
    tempo_bpm: float | None = None
    key: str | None = None
    use_cases: list[str] = field(default_factory=list)
    avoid_for: list[str] = field(default_factory=list)
    viral_origin: str | None = None         # "TikTok 2024 — 1.2M uses"
    iconic_snippet: dict | None = None      # {"start_s": 45, "end_s": 75, "lyrics": "..."}

    # Local cache
    local_audio_path: Path | None = None
    audio_content_hash: str | None = None   # sha256 of audio file for change detection

    # Indexing metadata
    indexed_at: datetime = field(default_factory=datetime.utcnow)
    enrichers_run: list[str] = field(default_factory=list)
    enrichment_errors: dict[str, str] = field(default_factory=dict)

    # User-set fields (these ALWAYS win over enricher-set values)
    user_tags: list[str] = field(default_factory=list)
    user_notes: str | None = None
    user_disabled: bool = False             # exclude from picker without deleting
```

## Storage layer

**v1: JSON manifest** (one file, `~/.podclipper/audio/manifest.json`).
- Simple, diffable, version-controllable, easy to back up.
- Fine up to ~5000 tracks.
- Bad: re-reads the whole file on every update, no concurrent writes.

**v2 (when v1 hurts): SQLite** (`~/.podclipper/audio/manifest.db`).
- Single file, no server, supports indexes for tag queries.
- Concurrent reads, serialized writes.
- One-time migration from v1 JSON.

Don't ship SQLite on day 1 — over-engineering. JSON works until it doesn't.

## Source plugin Protocol

```python
class AudioSource(Protocol):
    name: str                                       # "tokboard", "myinstants"
    category: Category                              # what kind of audio this source provides
    needs_credentials: bool                         # True if API key required

    def fetch_candidates(
        self,
        limit: int = 30,
        cursor: str | None = None,
    ) -> Iterator[AudioTrack]:
        """Yield bare-minimum AudioTrack stubs.

        Each yielded track has at minimum: track_id, category, source_name,
        source_url. Most other fields will be None — enrichers fill them.
        Source-specific extras (e.g. Tokboard's usage_count, Myinstants's
        play_count) go in source_metadata.
        """
        ...
```

**Initial sources** (build in order of value):

| Source | Category | Notes |
|---|---|---|
| `ManualSource` | any | Read a CSV / YAML of `(artist, title, url)` — the bootstrap source |
| `MyinstantsSource` | sfx_drop | Scrape the "most played" top N from myinstants.com |
| `TokboardSource` | song_clip | Scrape top trending TikTok sounds from tokboard.com |
| `LocalFolderSource` | any | User points at a folder; we treat each file as a track |
| `YouTubePlaylistSource` | any | Given a YT playlist URL, ingest each video as a track |
| `SpotifyPlaylistSource` | song_clip | Read a public Spotify playlist (needs only a token) |
| `YouTubeShortsTrendingSource` | song_clip | Scrape Shorts trending — fragile but valuable when it works |

## Enricher plugin Protocol

```python
class AudioEnricher(Protocol):
    name: str
    requires: list[str]                # AudioTrack field names this needs to read
    produces: list[str]                # AudioTrack field names this writes
    needs_local_audio: bool            # True → ensure track.local_audio_path is set first
    cost_estimate: Literal["free", "cheap", "expensive"]  # for prioritization

    def should_run(self, track: AudioTrack) -> bool:
        """Skip if produces[*] already populated AND no force-rerun flag."""
        ...

    def enrich(self, track: AudioTrack) -> AudioTrack:
        """Mutate in place, return. Errors → track.enrichment_errors[self.name] = msg."""
        ...
```

**Initial enrichers** (build in order of value):

| Enricher | Requires | Produces | needs_local_audio | Cost |
|---|---|---|---|---|
| `LrclibLyricsEnricher` | artist, title | lyrics | no | free |
| `FirecrawlLyricsEnricher` | artist, title | lyrics | no | cheap |
| `GeniusEnricher` | artist, title | source_metadata.genres, themes | no | free |
| `WhisperLyricsEnricher` | local audio | lyrics | yes | free but slow |
| `LLMMoodTagEnricher` | lyrics OR title | mood, themes, use_cases, avoid_for | no | cheap |
| `EssentiaFeatureEnricher` | local audio | tempo_bpm, energy, key | yes | free |
| `DurationEnricher` | local audio | duration_s | yes | free |
| `YtdlpAudioFetcherEnricher` | source_url OR (artist+title) | local_audio_path, audio_content_hash | n/a (it IS the fetcher) | free |

**Enricher dependency graph** is enforced via `requires` / `produces`:
- Running `LLMMoodTagEnricher` before `LrclibLyricsEnricher` is a no-op (no lyrics to tag from).
- An orchestrator can topologically sort enrichers by `produces → requires` chains and run them in dependency order.

## How sources + enrichers compose: pipelines

Pipelines are **declarative configs**, not code. Each pipeline = a source + an ordered enricher list.

```yaml
# ~/.podclipper/audio/pipelines.yaml

pipelines:
  tiktok_viral_songs:
    source:
      kind: tokboard
      limit: 50
    enrichers:
      - lrclib_lyrics
      - firecrawl_lyrics            # fallback for lrclib misses
      - llm_mood_tag
    # No YtdlpAudioFetcher here — fetch lazily at picker time

  myinstants_sfx:
    source:
      kind: myinstants
      limit: 100
    enrichers:
      - ytdlp_audio_fetcher         # SFX files are tiny, always fetch
      - duration                    # ffprobe to lock the real duration
      - llm_mood_tag                # turn Myinstants tags into our taxonomy

  user_folder_byo:
    source:
      kind: local_folder
      path: ~/Music/podclipper-bring-your-own
    enrichers:
      - duration
      - essentia_features           # no lyrics for user files; use audio analysis
      - whisper_lyrics              # for vocal tracks
      - llm_mood_tag

  curated_seed_list:
    source:
      kind: manual
      path: ~/.podclipper/audio/seed_songs.csv
    enrichers:
      - lrclib_lyrics
      - llm_mood_tag
```

## CLI surface

```bash
# Discovery
podclipper audio source list                          # available source plugins
podclipper audio enricher list                        # available enricher plugins
podclipper audio pipeline list                        # configured pipelines

# Ingestion
podclipper audio pipeline run tiktok_viral_songs      # source → enrichers → manifest
podclipper audio pipeline run --all                   # run all pipelines that are due

# Per-track operations
podclipper audio show <track_id>                      # full record dump
podclipper audio enrich <track_id> --enricher llm_mood_tag   # re-run one enricher
podclipper audio enrich <track_id> --all              # run any should_run() enrichers
podclipper audio fetch <track_id>                     # download local audio NOW
podclipper audio prune <track_id>                     # delete from manifest + cache

# Browsing
podclipper audio search "ironic heartbreak"           # tag/lyrics fuzzy search
podclipper audio list --category song_clip --mood melancholic
podclipper audio tag <track_id> --add "post-breakup"  # manual tag (always wins)
podclipper audio disable <track_id>                   # exclude from picker

# Maintenance
podclipper audio doctor                               # check for orphaned files, broken URLs
podclipper audio export manifest.json                 # for backup/sharing
podclipper audio import shared_manifest.json          # merge a friend's library
```

## On-disk layout

```
~/.podclipper/audio/
├── manifest.json              # the catalog (v1)
├── pipelines.yaml             # pipeline configs
├── cache/                     # downloaded audio files
│   ├── <sha256>.mp3
│   ├── <sha256>.wav
│   └── ...
└── seed_songs.csv             # manual source examples (optional)
```

Path is configurable via `cfg.audio.store_path` if a user wants the catalog elsewhere.

## Idempotency + change detection

- **Track identity** is content-hash for downloaded audio, source-URL-hash for not-yet-fetched tracks. Re-running a source ingest never creates duplicates.
- **Enricher idempotency**: each enricher's `should_run()` returns False if its `produces` fields are already populated. Force re-run via `--enricher X --force`.
- **Stale data**: tracks have an `indexed_at` timestamp. A `--max-age 30d` flag re-runs enrichers older than that.

## How a picker uses the store (preview — not in scope for the store itself)

```python
# Picker code (downstream of the store)
def pick_audio_for_reel(reel: Clip, transcript: Transcript, store: AudioStore) -> AudioPick:
    # Filter candidates by category and category-specific criteria
    songs = store.query(
        category="song_clip",
        min_duration_s=reel.duration,
        not_disabled=True,
        enrichers_must_include=["llm_mood_tag"],   # only fully-tagged tracks
    )
    sfx = store.query(category="sfx_drop", not_disabled=True)

    # Build LLM prompt: reel context + candidate manifest (just relevant fields)
    candidates_for_llm = [
        {"id": t.track_id, "artist": t.artist, "title": t.title,
         "lyrics": t.lyrics, "mood": t.mood, "themes": t.themes,
         "use_cases": t.use_cases}
        for t in songs
    ]
    plan = llm.complete(audio_picker_prompt, json=candidates_for_llm)
    # ... return AudioPick with chosen bed + sfx drops
```

The store exposes a clean `query()` API. The picker does the LLM logic. Mixer reads the chosen track's `local_audio_path` (auto-fetching if not present).

## Phasing

| Phase | Deliverable | Risk |
|---|---|---|
| **0 — design** (this doc) | Agreement on Protocol shapes, manifest schema, CLI surface | Bikeshedding |
| **1 — minimal store** | `AudioTrack` dataclass + JSON manifest + `ManualSource` (CSV) + one enricher (`LrclibLyricsEnricher`) + `podclipper audio pipeline run` working end-to-end | None — pure mechanical |
| **2 — broader enrichers** | `LLMMoodTagEnricher`, `YtdlpAudioFetcherEnricher`, `DurationEnricher` | None |
| **3 — viral sources** | `MyinstantsSource`, `TokboardSource` (scrapers will need maintenance) | Scrapers break |
| **4 — picker + mixer** | The actual "use the store" layer that produces audio-enhanced reels | Real integration risk |
| **5 — polish** | `EssentiaFeatureEnricher`, `WhisperLyricsEnricher`, SQLite migration, user web UI, etc. | None pressing |

Phases 1–3 are the store. Phase 4 is the consumer. They're independently shippable.

## Open questions

1. **Manifest format v1 — JSON or JSONL?** JSON is human-readable; JSONL streams better at scale and works with `grep`/`jq`. Default to JSON.
2. **Store path default — `~/.podclipper/audio/` or under the repo?** Home dir is more correct (cross-project), but repo-local is easier for development. Default to home, configurable.
3. **How does the user discover what's IN their store?** A `podclipper audio list` CLI is basic. A `podclipper audio web` that spins up a browse-by-tag UI is much nicer but real work.
4. **Pipeline scheduling.** Do we want `podclipper audio pipeline run --schedule daily` to auto-refresh viral sources? Probably yes eventually, not now.
5. **Sharing manifests.** If User A spends a week curating + tagging a great library, can User B `podclipper audio import` it and get the same enrichment without rerunning the LLM/scrapers? Manifests should be portable — point this in design.
6. **Per-enricher cost control.** LLM-tagging 500 tracks at $0.001 each is $0.50 — fine. But what about 50k? Need a `--dry-run` mode that estimates cost before running.

## Non-goals (deliberately)

- **No mood inference from audio in v1.** Essentia is great but adds 150 MB of native dep weight. Use LLM-from-lyrics where possible; fall back to user tagging.
- **No web UI in the store layer.** Future polish, not foundational.
- **No automatic copyright clearance.** User decides what's in their library; we don't gate.
- **No streaming / no on-the-fly fetch during reel render.** Audio must be locally cached before the mixer runs (with auto-fetch via `YtdlpAudioFetcherEnricher` on first use being acceptable).
