"""Run a standalone local Whisper transcription and timing trial."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def main() -> None:
    """Transcribe a file to Roman Hinglish and save text plus timestamped JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Oriserve/Whisper-Hindi2Hinglish-Apex")
    parser.add_argument("--language", default="en", help="en for Apex; hi for standard Whisper Hindi/Hinglish; auto to detect")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--timestamps", choices=["segment", "word"], default="segment")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    started = time.perf_counter()
    decoded = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(args.input),
         "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        check=True, capture_output=True,
    )
    audio = np.frombuffer(decoded.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    print(f"Loading {args.model} on {device}; audio={len(audio) / 16000:.2f}s", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model, dtype=dtype, use_safetensors=True,
    )
    if args.timestamps == "word" and not model.generation_config.alignment_heads:
        parser.error("This checkpoint has no alignment heads; use segment timestamps or a separate audio aligner.")
    recognizer = pipeline(
        "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor, dtype=dtype,
        device=device, chunk_length_s=30, stride_length_s=5,
    )
    loaded = time.perf_counter()
    print("Model ready; transcribing without LLM cleanup", flush=True)
    result = recognizer(
        {"raw": audio, "sampling_rate": 16000},
        return_timestamps="word" if args.timestamps == "word" else True,
        generate_kwargs={"task": "transcribe", "language": None if args.language == "auto" else args.language, "num_beams": 5},
    )
    elapsed = time.perf_counter() - loaded
    result.update({
        "model": args.model, "input": str(args.input.resolve()), "device": device,
        "audio_seconds": len(audio) / 16000, "inference_seconds": round(elapsed, 2),
        "total_seconds": round(time.perf_counter() - started, 2),
        "word_count": len(result["text"].split()), "timestamp_granularity": args.timestamps,
        "language": args.language,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    args.output.with_suffix(".txt").write_text(result["text"].strip() + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
