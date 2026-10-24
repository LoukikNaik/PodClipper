"""Detect missing caption speech and retry bounded audio sections."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import re

import numpy as np

from .types import Word

log = logging.getLogger("ave.transcribe.repair")
_SAMPLE_RATE = 16000


class CaptionCoverageError(RuntimeError):
    """Caption speech coverage remains unreliable after local retries."""

    def __init__(self, message, *, words=None, remaining_issues=None):
        super().__init__(message)
        self.words = list(words) if words is not None else None
        self.remaining_issues = list(remaining_issues or [])


def _speech_regions(audio: np.ndarray) -> list[tuple[float, float]]:
    """Return unpadded speech regions using the bundled Silero detector."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(min_silence_duration_ms=250, speech_pad_ms=0)
    return [(span["start"] / _SAMPLE_RATE, span["end"] / _SAMPLE_RATE)
            for span in get_speech_timestamps(audio, options, sampling_rate=_SAMPLE_RATE)]


def _overlap(regions, start, end):
    """Measure speech seconds inside a time range."""
    return sum(max(0.0, min(end, right) - max(start, left)) for left, right in regions)


def _count(words, start, end, max_word_seconds=math.inf):
    """Count plausible words by their midpoint inside a time range."""
    return sum(start <= (w.start + w.end) / 2 < end
               and w.end - w.start <= max_word_seconds for w in words)


def _merge_ranges(ranges):
    """Combine overlapping ranges in chronological order."""
    result = []
    for start, end in sorted(ranges):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def _issues(words, speech, reference, duration, cfg):
    """Find suspicious durations, uncovered speech, and locally sparse words."""
    ranges = []
    score = 0.0
    for word in words:
        if word.end - word.start > cfg.max_word_seconds:
            spoken = _overlap(speech, word.start, word.end)
            if spoken >= cfg.min_speech_seconds or _count(reference, word.start, word.end) >= cfg.reference_min_words:
                ranges.append((word.start, word.end))
                score += word.end - word.start
    covered = _merge_ranges([(max(0, w.start - .15), min(duration, w.end + .15))
                             for w in words if w.end - w.start <= cfg.max_word_seconds])
    cursor = 0.0
    for start, end in covered + [(duration, duration)]:
        if _overlap(speech, cursor, start) >= cfg.min_speech_seconds:
            missing = [(max(cursor, left), min(start, right)) for left, right in speech
                       if right > cursor and left < start]
            ranges.append((missing[0][0], missing[-1][1]))
            score += _overlap(speech, cursor, start)
        cursor = max(cursor, end)
    step = cfg.window_seconds / 2
    for start in np.arange(0, duration, step):
        end = min(duration, float(start) + cfg.window_seconds)
        spoken = _overlap(speech, start, end)
        count = _count(words, start, end, cfg.max_word_seconds)
        expected = _count(reference, start, end)
        minimum = spoken * cfg.min_words_per_speech_second
        reference_minimum = expected * cfg.reference_ratio if expected >= cfg.reference_min_words else 0
        deficit = max(minimum, reference_minimum) - count
        if deficit > 0 and (spoken >= cfg.min_speech_seconds or reference_minimum > 0):
            ranges.append((float(start), end))
            score += deficit
    return _merge_ranges(ranges), score


def _valid_words(words, duration):
    """Require finite ordered nonempty words within the decoded audio."""
    previous = 0.0
    for word in words:
        if (not word.text.strip() or not math.isfinite(word.start) or not math.isfinite(word.end)
                or word.start < previous - .02 or word.start < 0
                or word.end < word.start or word.end > duration + .1):
            return False
        previous = word.end
    return True


def _splice(words, replacement, start, end):
    """Replace an affected range while retaining whole boundary words."""
    def key(word):
        return re.sub(r"[^\w]", "", word.text.casefold())

    before = [(i, j) for i, old in enumerate(words) for j, new in enumerate(replacement)
              if old.end <= start and key(old) == key(new) and key(old)
              and abs((old.start + old.end - new.start - new.end) / 2) < 1]
    after = [(i, j) for i, old in enumerate(words) for j, new in enumerate(replacement)
             if old.start >= end and key(old) == key(new) and key(old)
             and abs((old.start + old.end - new.start - new.end) / 2) < 1]
    if before and after:
        left_i, left_j = max(before)
        right_i, right_j = min(after)
        if left_j < right_j:
            middle = [Word(max(words[left_i].end, w.start), min(words[right_i].start, w.end),
                           w.text, w.confidence) for w in replacement[left_j + 1:right_j]]
            return words[:left_i + 1] + middle + words[right_i:]
    crossing = [w for w in words if w.start < end and w.end > start]
    if crossing:
        start = min(start, crossing[0].start)
        end = max(end, crossing[-1].end)
    selected = [Word(max(start, w.start), min(end, w.end), w.text, w.confidence)
                for w in replacement if start <= (w.start + w.end) / 2 < end]
    return [w for w in words if w.end <= start] + selected + [w for w in words if w.start >= end]


def repair_caption_words(audio, words, cfg, transcribe_window, *, reference_words=None, report_path=None,
                         non_speech_regions=None):
    """Retry suspicious speech windows; fail before export if coverage stays unreliable."""
    duration = len(audio) / _SAMPLE_RATE
    reference = reference_words or []
    excluded = _merge_ranges(non_speech_regions or [])
    if any(not math.isfinite(a) or not math.isfinite(b) or not 0 <= a < b <= duration for a, b in excluded):
        raise CaptionCoverageError("Invalid non-speech event timestamps")
    report = {"status": "checking", "duration": duration, "settings": vars(cfg),
              "non_speech_regions": excluded,
              "first_pass_reference": bool(reference), "initial_issues": [], "retries": []}
    if report_path is not None and Path(report_path).exists():
        try:
            previous = json.loads(Path(report_path).read_text())
            if previous.get("status") == "repaired":
                report["last_repair"] = {key: previous[key] for key in ("initial_issues", "retries")}
            elif previous.get("last_repair"):
                report["last_repair"] = previous["last_repair"]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass

    def save(status, remaining):
        report.update(status=status, remaining_issues=remaining)
        if report_path is not None:
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2))
            temporary.replace(path)

    def audited_speech(regions):
        """Subtract confirmed non-speech events, including new section-retry results."""
        excluded = _merge_ranges(non_speech_regions or [])
        if any(not math.isfinite(a) or not math.isfinite(b) or not 0 <= a < b <= duration for a, b in excluded):
            raise CaptionCoverageError("Invalid non-speech event timestamps")
        report["non_speech_regions"] = excluded
        for left, right in excluded:
            regions = [(a, b) for start, end in regions
                       for a, b in ((start, min(end, left)), (max(start, right), end)) if b > a]
        return regions

    try:
        if not _valid_words(words, duration):
            raise CaptionCoverageError("Invalid caption word timings; regenerate the transcript cache.")
        detected_speech = _speech_regions(audio)
        speech = audited_speech(detected_speech)
        issues, score = _issues(words, speech, reference, duration, cfg)
        report["initial_issues"] = issues
        if not issues:
            save("passed", [])
            return words
        log.warning("Caption coverage: suspicious speech at %s; retrying only affected sections", issues)
        for attempt in range(cfg.max_attempts):
            pending = []
            core_seconds = cfg.max_retry_seconds - 2 * cfg.context_seconds
            for left, right in issues:
                for start in np.arange(left, right, core_seconds):
                    pending.append((float(start), min(right, float(start) + core_seconds)))
            for start, end in pending:
                if len(report["retries"]) >= cfg.max_windows:
                    break
                left = max(0, start - cfg.context_seconds)
                right = min(duration, end + cfg.context_seconds)
                entry = {"attempt": attempt + 1, "start": left, "end": right, "accepted": False}
                report["retries"].append(entry)
                save("retrying", issues)
                log.info("Retrying caption audio %.2f–%.2fs (attempt %s)", left, right, attempt + 1)
                try:
                    replacement = transcribe_window(left, right)
                    if not _valid_words(replacement, duration):
                        raise CaptionCoverageError("Retry returned invalid word timings")
                    if any(w.start < left - .1 or w.end > right + .1 for w in replacement):
                        raise CaptionCoverageError("Retry word timings escaped the audio window")
                    speech = audited_speech(detected_speech)
                    issues, score = _issues(words, speech, reference, duration, cfg)
                    spoken = _overlap(speech, left, right)
                    if len(replacement) > max(10, spoken * 8):
                        raise CaptionCoverageError("Retry returned implausibly dense speech")
                    candidate = _splice(words, replacement, start, end)
                    new_issues, new_score = _issues(candidate, speech, reference, duration, cfg)
                    if _valid_words(candidate, duration) and new_score < score - .01:
                        words, issues, score = candidate, new_issues, new_score
                        entry["accepted"] = True
                    entry["words"] = len(replacement)
                except Exception as exc:
                    entry["error"] = type(exc).__name__
                    log.warning("Caption section retry failed: %s", type(exc).__name__)
                save("retrying", issues)
                if not issues:
                    save("repaired", [])
                    return words
            if len(report["retries"]) >= cfg.max_windows:
                break
        save("failed", issues)
        raise CaptionCoverageError(
            f"Missing or unreliable caption speech remains at {issues}; clip export stopped. "
            "Inspect words.quality.json and retry with a different transcription model.",
            words=words, remaining_issues=issues,
        )
    except Exception as exc:
        if report["status"] != "failed":
            report["error"] = type(exc).__name__
            save("failed", report.get("remaining_issues", []))
        if isinstance(exc, CaptionCoverageError):
            raise
        raise CaptionCoverageError(f"Caption coverage check failed ({type(exc).__name__}); clip export stopped.") from exc
