# Audio Store — Architecture & Interface Contracts

Lean review doc. For the full design rationale, see [`AUDIO_STORE_DESIGN.md`](AUDIO_STORE_DESIGN.md).
For the empirical research that grounds the choices below, see [`../dev/test_audio/FINDINGS.md`](../dev/test_audio/FINDINGS.md).

---

## What this component is

A pluggable catalog of audio (songs, viral SFX, instrumental beds) with rich
LLM-derived tags. The picker (separate component, not yet built) queries this
catalog to choose a track per reel.

**The store knows what audio exists and what it means. It does not know how to
pick or how to mix — those are downstream concerns.**

---

## Architecture

```
                                  ┌───────────────────────────────────┐
                                  │   `podclipper audio pipeline run` │
                                  │            (CLI entry)            │
                                  └────────────────┬──────────────────┘
                                                   │
                          ┌────────────────────────▼──────────────────────┐
                          │              Orchestrator                     │
                          │   for each (source, [enricher, ...]):         │
                          │     stubs = source.fetch_candidates(limit)    │
                          │     for stub in stubs:                        │
                          │       for enricher in pipeline.enrichers:     │
                          │         if enricher.should_run(stub):         │
                          │           enricher.enrich(stub)               │
                          │       store.upsert(stub)                      │
                          └──────┬─────────────────────────────────┬──────┘
                                 │                                 │
                ┌────────────────▼────────────┐    ┌───────────────▼──────────────┐
                │       SOURCES (plug)        │    │      ENRICHERS (plug)        │
                │                             │    │                              │
                │  • ManualSource (CSV)       │    │  • LrclibLyrics              │
                │  • MyinstantsSource         │    │  • LLMLyricsVerifier         │
                │  • ItunesRssSource          │    │  • LLMSongMood               │
                │                             │    │  • LLMSfxMood                │
                │  yield AudioTrack stubs     │    │  • SoundCloudAudioFetcher    │
                │  with: source_*, title,     │    │  • MyinstantsAudioFetcher    │
                │  artist, category, url      │    │  • FFprobeDuration           │
                │                             │    │                              │
                │  (future: SpotifyPlaylist,  │    │  fields read/written         │
                │   YouTubePlaylist, TikTok)  │    │  declared via                │
                │                             │    │  requires/produces           │
                └─────────────────────────────┘    └──────────────────────────────┘
                                 │                                 │
                                 └──────────┬──────────────────────┘
                                            │
                          ┌─────────────────▼──────────────────┐
                          │           AudioStore               │
                          │   ~/.podclipper/audio/             │
                          │     ├─ manifest.json               │
                          │     └─ cache/<content_hash>.mp3    │
                          │                                    │
                          │   API: upsert, get, delete,        │
                          │        query(filters), all, stats  │
                          └─────────────────┬──────────────────┘
                                            │
                                            ▼
                  ─────────── downstream consumers (out of scope here) ───────────
                              Picker · Mixer · CLI browse · Web UI


Inside an enricher run:

   AudioTrack (partial)                  AudioTrack (more populated)
   ┌─────────────────────────┐           ┌─────────────────────────┐
   │ title: "Vine Boom"      │           │ title: "Vine Boom"      │
   │ category: sfx_drop      │           │ category: sfx_drop      │
   │ source_url: <myi-mp3>   │  enrich   │ source_url: <myi-mp3>   │
   │ local_audio_path: None  │ ─────────▶│ local_audio_path: <hash>│  ← MyinstantsAudioFetcher
   │ duration_s: None        │           │ duration_s: 1.18        │  ← FFprobeDuration
   │ mood: []                │           │ mood: ["comedic","drama"]│  ← LLMSfxMood
   │ sfx_category: None      │           │ sfx_category: "reaction"│  ← LLMSfxMood
   │ enrichers_run: []       │           │ enrichers_run: [3 names]│
   └─────────────────────────┘           └─────────────────────────┘
```

---

## Interface contracts

The whole store rides on **three Protocols + one data class + one storage facade**.

### `AudioTrack` — the unit of data

```python
@dataclass
class AudioTrack:
    track_id: str                          # stable id (e.g. "myinstants:vine-boom-70972")
    category: "sfx_drop" | "song_clip" | "instrumental_bed"

    # Provenance (set by Source, never mutated)
    source_name: str
    source_url: str | None
    source_metadata: dict

    # Identity (set by Source, possibly refined by enrichers)
    title: str | None
    artist: str | None
    duration_s: float | None

    # Enrichment fields (each set by exactly one enricher)
    lyrics: str | None
    lyrics_provider: str | None
    lyrics_match_verified: bool
    mood: list[str]
    themes: list[str]
    energy: "low" | "med" | "high" | None
    use_cases: list[str]
    avoid_for: list[str]
    viral_context: str | None
    iconic_lyric: str | None               # songs
    iconic_moment: str | None              # SFX
    sfx_category: str | None               # "reaction" | "reveal" | …
    sfx_intensity: "low" | "med" | "high" | None
    sfx_recognizable: bool | None

    # Local cache
    local_audio_path: Path | None
    audio_content_hash: str | None

    # Bookkeeping
    indexed_at: datetime
    enrichers_run: list[str]
    enrichment_errors: dict[str, str]

    # User overrides (always win over enricher output)
    user_tags: list[str]
    user_disabled: bool
```

### `AudioSource` — pulls in candidates

```python
class AudioSource(Protocol):
    name: str
    category: Category

    def fetch_candidates(self, limit: int = 30) -> Iterator[AudioTrack]: ...
```

Yields `AudioTrack` stubs with provenance + minimum identity (`title`,
optionally `artist` and `source_url`). All enrichment fields start empty.

### `AudioEnricher` — adds one or more fields

```python
class AudioEnricher(Protocol):
    name: str
    requires: list[str]                    # fields it reads
    produces: list[str]                    # fields it writes
    needs_local_audio: bool                # orchestrator force-fetches first if True
    cost: "free" | "cheap" | "expensive"

    def should_run(self, track: AudioTrack) -> bool: ...
    def enrich(self, track: AudioTrack) -> AudioTrack: ...
```

Idempotent. `should_run()` returns False when `produces` fields are already populated.
Errors caught and stored in `track.enrichment_errors[self.name]` — never raised.

### `AudioStore` — persistence + query

```python
class AudioStore:
    def __init__(self, manifest_path: Path): ...

    # Mutation
    def upsert(self, track: AudioTrack) -> None: ...
    def get(self, track_id: str) -> AudioTrack | None: ...
    def delete(self, track_id: str) -> None: ...
    def all(self) -> Iterator[AudioTrack]: ...

    # Query (for the future picker)
    def query(
        self,
        category: Category | None = None,
        mood_any: list[str] | None = None,
        themes_all: list[str] | None = None,
        min_duration_s: float | None = None,
        max_duration_s: float | None = None,
        require_local_audio: bool = False,
        require_enrichers: list[str] | None = None,
        not_disabled: bool = True,
    ) -> list[AudioTrack]: ...

    # Maintenance
    def stats(self) -> dict: ...
```

### `Pipeline` — declarative wiring

Defined in `default_pipelines.yaml`, not code:

```yaml
pipelines:
  <name>:
    source:
      kind: <source_name>           # registered AudioSource class
      <source-specific kwargs>
    enrichers:
      - <enricher_name>             # registered AudioEnricher class, run in order
      - ...
```

The orchestrator reads this, instantiates the named source and enrichers,
and runs them.

---

## Phase-1 concrete implementations

### Sources

| Source | Category | What it does | Validated |
|---|---|---|---|
| `ManualSource(csv_path)` | any | Reads `artist,title,category,source_url?` rows from CSV | trivial |
| `MyinstantsSource(limit)` | `sfx_drop` | Scrapes myinstants.com homepage | ✅ Test 3: 10/10 |
| `ItunesRssSource(country, limit)` | `song_clip` | Fetches Apple/iTunes Top Songs JSON RSS | ✅ Test 6: 15/15 |

### Enrichers

| Enricher | Reads | Writes | Validated |
|---|---|---|---|
| `LrclibLyricsEnricher` | artist, title | lyrics, lyrics_provider | ✅ Test 1: 5/5 |
| `LLMLyricsVerifierEnricher` | lyrics, artist, title | lyrics_match_verified | required per Test 1 caveat |
| `LLMSongMoodEnricher` | lyrics, artist, title | mood, themes, energy, use_cases, avoid_for, viral_context, iconic_lyric | ✅ Test 2: 4/4 |
| `LLMSfxMoodEnricher` | title | sfx_category, mood, sfx_intensity, use_cases, viral_context, iconic_moment, sfx_recognizable | ✅ Test 4: 10/10 |
| `MyinstantsAudioFetcher` | source_url | local_audio_path, audio_content_hash | direct download |
| `SoundCloudAudioFetcher` | artist, title | local_audio_path, audio_content_hash, duration_s | ⚠ Test 5: ~50% — falls back gracefully on miss |
| `FFprobeDurationEnricher` | local_audio_path | duration_s | trivial |

### Pipelines that ship

```yaml
pipelines:
  viral_sfx:
    source:    { kind: myinstants, limit: 50 }
    enrichers: [myinstants_audio_fetcher, ffprobe_duration, llm_sfx_mood]

  trending_songs:
    source:    { kind: itunes_rss, country: us, limit: 25 }
    enrichers: [lrclib_lyrics, llm_lyrics_verifier, llm_song_mood]
    # audio fetch deferred until picker actually uses the track

  user_seed_songs:
    source:    { kind: manual, csv_path: ~/.podclipper/audio/seed.csv }
    enrichers: [lrclib_lyrics, llm_lyrics_verifier, llm_song_mood]
```

---

## On-disk layout

```
~/.podclipper/audio/
├─ manifest.json                       # the catalog (JSON v1; SQLite if/when scale demands)
├─ pipelines.yaml                      # pipeline configs (or use defaults)
├─ seed.csv                            # user's hand-curated source list (optional)
└─ cache/
   ├─ <content_hash>.mp3
   └─ ...
```

Location overridable via `cfg.audio.store_path`.

---

## CLI surface

```bash
# Ingestion
podclipper audio pipeline list
podclipper audio pipeline run viral_sfx
podclipper audio pipeline run --all

# Browse
podclipper audio list --category sfx_drop --mood comedic
podclipper audio search "ironic heartbreak"
podclipper audio show <track_id>
podclipper audio stats

# Maintenance
podclipper audio re-enrich <track_id> --enricher llm_song_mood
podclipper audio tag <track_id> --add wholesome --add thanksgiving
podclipper audio disable <track_id>
podclipper audio fetch <track_id>
```

---

## Module layout

```
src/podclipper/audio/
├─ __init__.py
├─ types.py            # AudioTrack, Category, Protocols
├─ source.py           # AudioSource Protocol
├─ enricher.py         # AudioEnricher Protocol
├─ store.py            # AudioStore (JSON manifest impl)
├─ orchestrator.py     # pipeline runner
├─ registry.py         # name → class lookup for plugins
├─ default_pipelines.yaml
├─ cli.py              # `podclipper audio …` subcommands
├─ sources/
│  ├─ manual.py
│  ├─ myinstants.py
│  └─ itunes_rss.py
└─ enrichers/
   ├─ lrclib.py
   ├─ llm_song_mood.py
   ├─ llm_sfx_mood.py
   ├─ llm_lyrics_verifier.py
   ├─ soundcloud_fetcher.py
   ├─ myinstants_fetcher.py
   └─ ffprobe_duration.py
```

---

## What ships at end of Phase 1

A browsable, queryable catalog with ~75 fully-tagged tracks (50 SFX + 25 trending songs).
The contracts above are stable — the picker, mixer, and reel-pipeline integration
(Phase 2+) consume `AudioStore.query()` and never touch sources or enrichers.

## What is deliberately NOT in Phase 1

- Picker (LLM-driven reel ↔ track matching) → Phase 2
- Mixer (ffmpeg audio chain) → Phase 2
- Reels-pipeline integration → Phase 3
- Trailer-pipeline integration → Phase 3
- SQLite migration → only if JSON gets painful
- Whisper-from-audio lyrics, Essentia features, AZ Lyrics, DDG, Tokboard,
  YouTube unauthenticated fetch — all dead ends per `FINDINGS.md`
