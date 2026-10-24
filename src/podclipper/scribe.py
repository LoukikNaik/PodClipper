"""Experimental full-source Scribe transcription and reuse of timed words for reel captions."""

import base64
from collections import Counter
from copy import copy
from dataclasses import asdict, replace
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
import wave

import numpy as np

from .audio import plan_chunks
from .render_cache import checkpoint, checkpoint_matches, file_identity, fingerprint, write_json
from .transcribe_repair import CaptionCoverageError, _valid_words, repair_caption_words
from .types import Transcript, TranscriptSegment, Word

log = logging.getLogger("ave.scribe")
MODEL = "elevenlabs/scribe-v2"
SAMPLE_RATE = 16000


def validate_connection(cfg):
    """Require the existing shared connection to point at OpenRouter before uploading audio."""
    endpoint = urlparse(cfg.llm.endpoint)
    if (endpoint.scheme != "https" or endpoint.hostname != "openrouter.ai"
            or endpoint.port not in (None, 443) or endpoint.path.rstrip("/") != "/api/v1"
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError("--scribe requires LLM_ENDPOINT=https://openrouter.ai/api/v1 and an OpenRouter LLM_API_KEY")
    if not os.environ.get("LLM_API_KEY", "").strip():
        raise ValueError("--scribe requires the existing LLM_API_KEY")


def _segments(words):
    """Group timed words into short speech segments for clip selection."""
    result, current = [], []
    for word in words:
        if current and (word.start - current[-1].end > .7 or word.end - current[0].start > 15):
            result.append(TranscriptSegment(current[0].start, current[-1].end,
                                             " ".join(w.text for w in current), current))
            current = []
        current.append(word)
    if current:
        result.append(TranscriptSegment(current[0].start, current[-1].end,
                                         " ".join(w.text for w in current), current))
    return result


class ScribeTranscriber:
    """Cache successful requests and repair only suspicious audio windows with the same backend."""

    def __init__(self, cfg, cache, use_cache=True):
        validate_connection(cfg)
        self.cfg = cfg
        self.options = cfg.transcribe.scribe
        self.cache = Path(cache) / "scribe"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self.transcript_path = None
        self.cost = 0.0
        event_setting = getattr(self.options, "tag_audio_events", None)
        self.tag_audio_events = (bool(getattr(cfg.crop, "comedy_mode", False))
                                 if event_setting is None else event_setting)
        self.audio_events = []
        self.last_audio_events = []
        self.unreliable_ranges = []

    def _request(self, audio):
        """Return validated clip-relative words and detected language, recording each billed receipt."""
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        duration = len(pcm) / (2 * SAMPLE_RATE)
        inputs = ["scribe-request-v1", MODEL, hashlib.sha256(pcm).hexdigest(), self.cfg.transcribe.language]
        if self.tag_audio_events:
            inputs.append("audio-events-v1")
        path = self.cache / "requests" / f"{fingerprint(inputs)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.use_cache and checkpoint_matches(path, inputs):
            response = json.loads(path.read_text())
            return self._parse(response, duration)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(SAMPLE_RATE)
            audio_file.writeframes(pcm)
        body = {
            "model": MODEL, "input_audio": {"data": base64.b64encode(buffer.getvalue()).decode(), "format": "wav"},
            "temperature": 0, "response_format": "verbose_json", "timestamp_granularities": ["word", "segment"],
            "provider": {"options": {"elevenlabs": {"tag_audio_events": self.tag_audio_events}}},
        }
        if self.cfg.transcribe.language:
            body["language"] = self.cfg.transcribe.language
        key = os.environ["LLM_API_KEY"]
        encoded = json.dumps(body).encode()
        for attempt in range(self.options.max_attempts):
            started = time.perf_counter()
            try:
                request = urllib.request.Request(self.cfg.llm.endpoint.rstrip("/") + "/audio/transcriptions",
                    data=encoded, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=self.options.timeout_seconds) as result:
                    response = json.load(result)
                    generation = result.headers.get("X-Generation-Id")
                usage = response.get("usage") or {}
                cost = float(usage.get("cost") or 0)
                self.cost += cost
                receipt = {"model": MODEL, "seconds": duration, "cost_usd": cost, "usage": usage,
                           "generation_id": generation, "elapsed_seconds": round(time.perf_counter() - started, 2),
                           "response": str(path), "received_at": time.time()}
                with (self.cache / "usage.jsonl").open("a") as journal:
                    journal.write(json.dumps(receipt) + "\n")
                write_json(path, response)
                words, language = self._parse(response, duration)
                checkpoint(path, inputs)
                log.info("Scribe: %.1fs audio, %s words, %.2fs request, $%.6f", duration, len(words),
                         receipt["elapsed_seconds"], cost)
                return words, language
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace").replace(key, "[redacted]")[:500]
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt + 1 == self.options.max_attempts:
                    raise RuntimeError(f"Scribe HTTP {exc.code}: {detail}") from None
                log.warning("Scribe HTTP %s; retrying this audio chunk", exc.code)
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError) as exc:
                if attempt + 1 == self.options.max_attempts:
                    raise RuntimeError(f"Scribe request failed ({type(exc).__name__}); successful chunks remain cached") from None
                log.warning("Scribe %s; retrying only this request (may incur another charge)", type(exc).__name__)
            time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("Scribe request exhausted retries")

    def _parse(self, response, duration):
        """Reject missing, nonfinite, overlapping, or out-of-range word timestamps."""
        if not isinstance(response.get("words"), list):
            raise ValueError("Scribe did not return word timestamps")
        words = []
        self.last_audio_events = []
        for item in response["words"]:
            if item.get("type") == "spacing":
                continue
            text = item.get("word", item.get("text"))
            start, end = item["start"], item["end"]
            confidence = item.get("confidence", 1.0)
            if (not isinstance(text, str) or not text.strip() or type(start) not in (int, float)
                    or type(end) not in (int, float) or not math.isfinite(start) or not math.isfinite(end)
                    or not 0 <= start <= end <= duration + .1):
                raise ValueError("Scribe returned invalid word timestamps")
            if type(confidence) not in (int, float) or not math.isfinite(confidence):
                confidence = 1.0
            if item.get("type") == "audio_event":
                if (confidence >= .5 and 0 < min(end, duration) - min(start, duration) <= 10
                        and re.search(r"laugh|applau|clap|cheer|हँस|हंस|हँसी|हंसी|ताल", text, re.I)):
                    self.last_audio_events.append({"start": min(start, duration), "end": min(end, duration), "text": text})
                continue
            words.append(Word(min(start, duration), min(end, duration), text.strip(), confidence))
        event_only = bool(response["words"]) and all(
            w.get("type") in ("audio_event", "spacing") for w in response["words"])
        if not _valid_words(words, duration) or (not words and not event_only and str(response.get("text", "")).strip()):
            raise ValueError("Scribe returned unusable word timestamps")
        return words, str(response.get("language") or "auto")

    def _repair(self, audio, words, report, *, owned_start=0.0, owned_end=None, audio_events=None,
                max_windows=None):
        """Audit speech coverage and retranscribe only failed sections with surrounding context."""
        if not self.cfg.transcribe.caption_repair.enabled:
            return words
        repair_cfg = copy(self.cfg.transcribe.caption_repair)
        if max_windows is not None:
            repair_cfg.max_windows = max_windows

        audit_audio = audio
        if owned_start or owned_end is not None:
            audit_audio = audio.copy()
            audit_audio[:round(owned_start * SAMPLE_RATE)] = 0
            if owned_end is not None:
                audit_audio[round(owned_end * SAMPLE_RATE):] = 0
        events = audio_events if audio_events is not None else []
        protected_words = list(words)
        attempts = Counter()

        def event_gaps():
            """Exclude tagged reactions without masking recognized spoken words."""
            excluded = []
            for event in events:
                gaps = [(event["start"], event["end"])]
                for word in protected_words:
                    gaps = [(a, b) for start, end in gaps
                            for a, b in ((start, min(end, word.start - .15)),
                                         (max(start, word.end + .15), end)) if b > a]
                excluded.extend(gaps)
            return excluded

        excluded = event_gaps()

        def retry(left, right):
            """Retry one suspect section with more context on repeated attempts."""
            attempt = attempts[(left, right)]
            attempts[(left, right)] += 1
            context = min((attempt + 1) * repair_cfg.context_seconds,
                          max(0, (repair_cfg.max_retry_seconds - (right - left)) / 2))
            request_left = max(0, left - context)
            request_right = min(len(audio) / SAMPLE_RATE, right + context)
            replacement, _ = self._request(audio[round(request_left * SAMPLE_RATE):round(request_right * SAMPLE_RATE)])
            replacement = [replace(w, start=max(left, w.start + request_left),
                                   end=min(right, w.end + request_left)) for w in replacement
                           if left <= (w.start + w.end) / 2 + request_left < right]
            for event in self.last_audio_events:
                start, end = max(left, event["start"] + request_left), min(right, event["end"] + request_left)
                if start < end:
                    events.append({"start": start, "end": end, "text": event["text"]})
            protected_words.extend(replacement)
            excluded[:] = event_gaps()
            return replacement
        result = repair_caption_words(audit_audio, words, repair_cfg, retry,
                                      report_path=report, non_speech_regions=excluded)
        if audit_audio is not audio:
            metadata = json.loads(report.read_text())
            metadata["owned_audio_range"] = [owned_start, owned_end]
            write_json(report, metadata)
        return result

    def full_transcript(self, audio_path, duration):
        """Transcribe overlapping chunks once, merge their owned words, and cache the full transcript."""
        inputs = ["scribe-transcript-v2", file_identity(audio_path), duration, MODEL,
                  self.cfg.transcribe.language, self.options.chunk_seconds, self.options.overlap_seconds,
                  self.cfg.transcribe.caption_repair,
                  ["source-repair-budget-v1", self.options.source_repair_max_windows]]
        inputs.append(["source-coverage-v2", self.options.max_unreliable_fraction])
        if self.tag_audio_events:
            inputs.append("audio-events-v1")
        workspace = self.cache / fingerprint(inputs)[:16]
        workspace.mkdir(parents=True, exist_ok=True)
        self.transcript_path = workspace / "transcript.json"
        if self.use_cache and checkpoint_matches(self.transcript_path, inputs):
            data = json.loads(self.transcript_path.read_text())
            self.audio_events = data.get("audio_events", [])
            self.unreliable_ranges = data.get("unreliable_ranges", [])
            log.info("Reusing full Scribe transcript: %s", self.transcript_path)
            return Transcript(data["language"], [TranscriptSegment(s["start"], s["end"], s["text"],
                              [Word(**w) for w in s["words"]]) for s in data["segments"]])
        chunks = plan_chunks(duration, self.options.chunk_seconds, self.options.overlap_seconds)
        log.info("Scribe opt-in: uploading %.1f minutes in %s resumable chunks; no regular Whisper passes",
                 duration / 60, len(chunks))
        merged, languages = [], Counter()
        self.audio_events = []
        self.unreliable_ranges = []
        with wave.open(str(audio_path)) as audio_file:
            if audio_file.getframerate() != SAMPLE_RATE or audio_file.getnchannels() != 1 or audio_file.getsampwidth() != 2:
                raise ValueError("Scribe source audio must be mono 16 kHz PCM16 WAV")
            for index, chunk in enumerate(chunks):
                left = (chunks[index - 1].end + chunk.start) / 2 if index else 0
                right = (chunk.end + chunks[index + 1].start) / 2 if index + 1 < len(chunks) else duration
                chunk_path = workspace / f"chunk-{index:04d}.json"
                chunk_key = [inputs, chunk.start, chunk.end]
                if self.use_cache and checkpoint_matches(chunk_path, chunk_key):
                    data = json.loads(chunk_path.read_text())
                    words, language = [Word(**w) for w in data["words"]], data["language"]
                    events = data.get("audio_events", [])
                    unreliable = data.get("unreliable_ranges", [])
                    log.info("Reusing Scribe chunk %s/%s", index + 1, len(chunks))
                else:
                    log.info("Scribe chunk %s/%s: %.1f–%.1fs", index + 1, len(chunks), chunk.start, chunk.end)
                    audio_file.setpos(round(chunk.start * SAMPLE_RATE))
                    pcm = audio_file.readframes(round(chunk.duration * SAMPLE_RATE))
                    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
                    words, language = self._request(audio)
                    events = list(self.last_audio_events)
                    quality = workspace / f"chunk-{index:04d}.quality.json"
                    unreliable = []
                    try:
                        words = self._repair(audio, words, quality,
                                             owned_start=left - chunk.start, owned_end=right - chunk.start,
                                             audio_events=events, max_windows=self.options.source_repair_max_windows)
                    except CaptionCoverageError as exc:
                        remaining = exc.remaining_issues
                        fraction = getattr(self.options, "max_unreliable_fraction", .05)
                        if (exc.words is None or not remaining
                                or sum(b - a for a, b in remaining) > chunk.duration * fraction):
                            raise
                        words = exc.words
                        context = self.cfg.transcribe.caption_repair.context_seconds
                        unreliable = [[max(0, a - context), min(chunk.duration, b + context)] for a, b in remaining]
                        log.warning("Keeping unresolved source spans out of clip selection: %s", unreliable)
                    write_json(chunk_path, {"language": language, "words": [asdict(w) for w in words],
                                            "audio_events": events, "unreliable_ranges": unreliable})
                    checkpoint(chunk_path, chunk_key)
                self.unreliable_ranges.extend([[max(left, a + chunk.start), min(right, b + chunk.start)]
                                               for a, b in unreliable if a + chunk.start < right and b + chunk.start > left])
                for event in events:
                    start, end = max(left, event["start"] + chunk.start), min(right, event["end"] + chunk.start)
                    if start < end:
                        self.audio_events.append({"start": start, "end": end, "text": event["text"]})
                languages[language] += len(words)
                for word in words:
                    start, end = word.start + chunk.start, word.end + chunk.start
                    if not left <= (start + end) / 2 < right:
                        continue
                    word = replace(word, start=start, end=end)
                    if merged and word.start < merged[-1].end:
                        previous = merged[-1]
                        normalize = lambda text: text.casefold().strip(".,!?;:।…\"'“”‘’()[]{}")
                        if word.start < left < previous.end and normalize(previous.text) == normalize(word.text):
                            continue
                        boundary = (previous.end + word.start) / 2
                        merged[-1] = replace(previous, end=boundary)
                        word = replace(word, start=boundary)
                    merged.append(word)
        if not merged or not _valid_words(merged, duration):
            raise ValueError("Scribe produced no usable full-source transcript")
        transcript = Transcript(languages.most_common(1)[0][0], _segments(merged))
        write_json(self.transcript_path, {**asdict(transcript), "audio_events": self.audio_events,
                                         "unreliable_ranges": self.unreliable_ranges})
        checkpoint(self.transcript_path, inputs)
        log.info("Full Scribe transcript: %s words; new request cost $%.6f", len(merged), self.cost)
        return transcript

    def clip_words(self, segment, transcript, start, end, clip_cache, *, caption_start=None, caption_end=None):
        """Shift full-source word timestamps into a clip and repair gaps without a second full pass."""
        from .transcribe import _decode_audio_to_float32

        path = clip_cache / "words.json"
        caption_start = start if caption_start is None else caption_start
        caption_end = end if caption_end is None else caption_end
        inputs = ["scribe-clip-v2", file_identity(self.transcript_path), file_identity(segment), start, end,
                  caption_start, caption_end,
                  self.cfg.transcribe.caption_repair]
        if self.use_cache and checkpoint_matches(path, inputs):
            words = [Word(**w) for w in json.loads(path.read_text())]
            log.info("Reusing checked Scribe clip captions: %s", path)
            return words
        words = [replace(w, start=max(caption_start, w.start) - start, end=min(caption_end, w.end) - start)
                 for s in transcript.segments for w in s.words
                 if w.start < caption_end and w.end > caption_start]
        audio = _decode_audio_to_float32(segment)
        events = [{**e, "start": max(start, e["start"]) - start, "end": min(end, e["end"]) - start}
                  for e in self.audio_events if e["start"] < end and e["end"] > start]
        words = self._repair(audio, words, clip_cache / "words.quality.json",
                             owned_start=caption_start - start, owned_end=caption_end - start,
                             audio_events=events)
        left, right = caption_start - start, caption_end - start
        words = [replace(w, start=max(left, w.start), end=min(right, w.end))
                 for w in words if w.start < right and w.end > left]
        write_json(path, [asdict(w) for w in words])
        checkpoint(path, inputs)
        write_json(clip_cache / "words.provenance.json", {
            "backend": "scribe", "transcript": str(self.transcript_path), "start": start, "end": end,
            "artifact": file_identity(path),
        })
        log.info("Reused %s Scribe words for %.1f–%.1fs; skipped normal second pass", len(words), start, end)
        return words
