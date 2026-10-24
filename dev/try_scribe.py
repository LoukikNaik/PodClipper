"""Measure OpenRouter Scribe transcription on a short source excerpt without changing pipeline caches."""

import argparse
import base64
import json
import math
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
import wave

from dotenv import dotenv_values


API = "https://openrouter.ai/api/v1"
MODEL = "elevenlabs/scribe-v2"
DURATION_MODELS = [MODEL, "qwen/qwen3-asr-0.6b", "qwen/qwen3-asr-1.7b",
                   "nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b",
                   "openai/whisper-large-v3-turbo", "openai/whisper-large-v3"]


def main():
    """Save raw output, normalized timed words, latency, and billed cost for one bounded request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", choices=DURATION_MODELS, default=MODEL)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--keyterm", action="append", default=[])
    parser.add_argument("--text-only", action="store_true", help="Probe models that reject verbose timestamps")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    duration = args.end - args.start
    if not (math.isfinite(args.start) and math.isfinite(args.end) and args.start >= 0 and 0 < duration <= 180):
        parser.error("Choose a source excerpt no longer than 180 seconds.")
    values = dotenv_values(args.env_file)
    key = os.environ.get("LLM_API_KEY") or values.get("LLM_API_KEY")
    if not key:
        parser.error("LLM_API_KEY is missing.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "response.json").exists():
        parser.error("Response already exists; choose a new output folder to avoid another billed request.")
    with urllib.request.urlopen(f"{API}/models/{args.model}/endpoints", timeout=30) as response:
        endpoints = json.load(response)
    (output / "endpoints.json").write_text(json.dumps(endpoints, indent=2))
    rates = [float(e["pricing"]["prompt"]) for e in endpoints["data"]["endpoints"]]
    rate = max(rates)
    estimate = duration * rate * (1.2 if args.keyterm else 1)
    if estimate > 0.05:
        parser.error(f"Estimated request cost ${estimate:.4f} exceeds the $0.05 experiment limit.")
    audio = output / "audio.wav"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(args.start), "-i", str(args.source),
        "-t", str(duration), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio),
    ], check=True, timeout=60)
    with wave.open(str(audio)) as waveform:
        audio_seconds = waveform.getnframes() / waveform.getframerate()
    body = {
        "model": args.model,
        "input_audio": {"data": base64.b64encode(audio.read_bytes()).decode(), "format": "wav"},
        "response_format": "json" if args.text_only else "verbose_json",
        "temperature": 0,
    }
    if not args.text_only:
        body["timestamp_granularities"] = ["word", "segment"]
    if args.model == MODEL:
        body["provider"] = {"options": {"elevenlabs": {"tag_audio_events": False}}}
    if args.language != "auto":
        body["language"] = args.language
    if args.keyterm:
        if args.model != MODEL:
            parser.error("Keyterms in this experiment are supported only for Scribe.")
        body["provider"]["options"]["elevenlabs"]["keyterms"] = args.keyterm
    metadata = {
        "source": str(args.source.resolve()), "source_start": args.start, "source_end": args.end,
        "audio_seconds": audio_seconds, "model": args.model, "language_requested": args.language,
        "response_format": body["response_format"],
        "options": body.get("provider", {}).get("options", {}), "estimated_usd": estimate,
        "rate_usd_per_second": rate, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "endpoint_rates_usd_per_second": rates,
    }
    (output / "request-metadata.json").write_text(json.dumps(metadata, indent=2))
    request = urllib.request.Request(f"{API}/audio/transcriptions", data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
    })
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=100) as response:
            result = json.load(response)
            generation = response.headers.get("X-Generation-Id")
    except urllib.error.HTTPError as exc:
        error = exc.read().decode(errors="replace").replace(key, "[redacted]")[:2000]
        (output / "error.json").write_text(json.dumps({"status": exc.code, "body": error}, indent=2))
        raise SystemExit(f"HTTP {exc.code}: {error}") from None
    (output / "response.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (output / "transcript.txt").write_text(result.get("text", "") + "\n")
    words = [{"text": w.get("word", w.get("text", "")), "start": w.get("start"), "end": w.get("end")}
             for w in result.get("words", [])]
    problems = []
    if not words:
        problems.append("No word timestamps returned")
    previous_start = -1
    for index, word in enumerate(words):
        start, end = word["start"], word["end"]
        if (not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not math.isfinite(start) or not math.isfinite(end)
                or not 0 <= start <= end <= audio_seconds + 0.25):
            problems.append(f"Invalid word timing at index {index}")
            continue
        if start < previous_start:
            problems.append(f"Unordered word at index {index}")
        previous_start = start
    summary = metadata | {
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "usage": result.get("usage"), "generation_id": generation,
        "language_returned": result.get("language"), "word_count": len(words),
        "timing_problems": problems,
        "limitations": "Structural timing validation only; no reference WER or audio alignment accuracy measurement.",
    }
    (output / "words.json").write_text(json.dumps(words, ensure_ascii=False, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
