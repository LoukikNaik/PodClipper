"""Tests for `transcribe.py` engine dispatcher and mlx-whisper adapter.

The existing module body (faster-whisper path) is exercised end-to-end via
the larger pipeline tests; this file focuses on the new dispatcher logic
that selects between faster-whisper and mlx-whisper based on
`cfg.transcribe.engine`.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from podclipper.transcribe import _resolve_engine


def _install_fake_mlx_whisper(monkeypatch, fake_result: dict, recorder: list) -> None:
    """Inject a fake `mlx_whisper` module whose `transcribe` records its
    kwargs into `recorder` and returns `fake_result`. Tests then assert on
    both the call and the return shape."""
    fake = types.ModuleType("mlx_whisper")

    def fake_transcribe(audio, **kwargs):
        recorder.append({"audio": audio, **kwargs})
        return fake_result

    fake.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)


def _cfg_with_engine(engine: str) -> SimpleNamespace:
    return SimpleNamespace(transcribe=SimpleNamespace(engine=engine))


# --------------------------------------------------------------------------- #
# _resolve_engine
# --------------------------------------------------------------------------- #

def test_resolve_engine_returns_faster_whisper_when_set() -> None:
    """`cfg.transcribe.engine = 'faster_whisper'` resolves to the same string."""
    cfg = _cfg_with_engine("faster_whisper")

    assert _resolve_engine(cfg) == "faster_whisper"


def test_resolve_engine_returns_mlx_whisper_when_set() -> None:
    """`cfg.transcribe.engine = 'mlx_whisper'` resolves to the same string."""
    cfg = _cfg_with_engine("mlx_whisper")

    assert _resolve_engine(cfg) == "mlx_whisper"


def test_resolve_engine_raises_value_error_on_unknown_engine() -> None:
    """Unknown engine names raise early so misconfiguration surfaces before any model load."""
    cfg = _cfg_with_engine("openai_whisper")

    with pytest.raises(ValueError, match="openai_whisper"):
        _resolve_engine(cfg)


# --------------------------------------------------------------------------- #
# _transcribe_array_mlx adapter
# --------------------------------------------------------------------------- #

_EMPTY_MLX_RESULT = {"text": "", "language": "en", "segments": []}


def test_mlx_adapter_calls_mlx_whisper_transcribe_with_repo_and_language(monkeypatch) -> None:
    """The adapter forwards the audio buffer plus repo/language/word_timestamps
    to `mlx_whisper.transcribe` in the documented kwarg shape."""
    from podclipper.transcribe import _transcribe_array_mlx

    recorder: list = []
    _install_fake_mlx_whisper(monkeypatch, _EMPTY_MLX_RESULT, recorder)

    audio = np.zeros(16000, dtype=np.float32)

    _transcribe_array_mlx(
        audio,
        repo="mlx-community/whisper-base-mlx",
        language="hi",
        word_timestamps=True,
    )

    assert len(recorder) == 1
    call = recorder[0]
    assert call["audio"] is audio
    assert call["path_or_hf_repo"] == "mlx-community/whisper-base-mlx"
    assert call["language"] == "hi"
    assert call["word_timestamps"] is True


def test_mlx_adapter_returns_words_shifted_by_time_offset_with_probability(monkeypatch) -> None:
    """Each Word's start/end is shifted by `time_offset` (so chunked first-pass
    calls produce video-relative times); confidence reads from `probability`."""
    from podclipper.transcribe import _transcribe_array_mlx
    from podclipper.types import TranscriptSegment, Word

    mlx_result = {
        "text": " hello world",
        "language": "en",
        "segments": [
            {
                "start": 0.0, "end": 1.2, "text": " hello world",
                "words": [
                    {"word": " hello", "start": 0.0, "end": 0.5, "probability": 0.95},
                    {"word": " world", "start": 0.6, "end": 1.2, "probability": 0.88},
                ],
            }
        ],
    }
    recorder: list = []
    _install_fake_mlx_whisper(monkeypatch, mlx_result, recorder)

    audio = np.zeros(16000, dtype=np.float32)
    segs, lang = _transcribe_array_mlx(
        audio,
        repo="mlx-community/whisper-base-mlx",
        language=None,
        word_timestamps=True,
        time_offset=10.0,
    )

    assert lang == "en"
    assert len(segs) == 1
    seg = segs[0]
    assert isinstance(seg, TranscriptSegment)
    assert seg.start == pytest.approx(10.0)
    assert seg.end == pytest.approx(11.2)
    assert seg.text == "hello world"  # leading space stripped
    assert len(seg.words) == 2
    w0, w1 = seg.words
    assert isinstance(w0, Word)
    assert w0.start == pytest.approx(10.0)
    assert w0.end == pytest.approx(10.5)
    assert w0.text == " hello"  # leading space preserved (matches faster-whisper)
    assert w0.confidence == pytest.approx(0.95)
    assert w1.start == pytest.approx(10.6)
    assert w1.confidence == pytest.approx(0.88)


def test_mlx_adapter_handles_segment_without_words_key(monkeypatch) -> None:
    """A segment that omits the `words` key (some mlx-whisper builds when
    word_timestamps=False) yields an empty Word list, not a KeyError."""
    from podclipper.transcribe import _transcribe_array_mlx

    mlx_result = {
        "text": " hi",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 0.5, "text": " hi"},  # no "words" key
        ],
    }
    recorder: list = []
    _install_fake_mlx_whisper(monkeypatch, mlx_result, recorder)

    segs, _ = _transcribe_array_mlx(
        np.zeros(16000, dtype=np.float32),
        repo="x",
        language=None,
        word_timestamps=True,
    )

    assert len(segs) == 1
    assert segs[0].words == []


# --------------------------------------------------------------------------- #
# _run_pass dispatcher (per-pass entry point used by first_pass + second_pass)
# --------------------------------------------------------------------------- #

def _make_pass_cfg(*, model="base", compute_type="int8", device="cpu",
                   beam_size=1, word_timestamps=True,
                   mlx_repo="mlx-community/whisper-base-mlx") -> SimpleNamespace:
    return SimpleNamespace(
        model=model, compute_type=compute_type, device=device,
        beam_size=beam_size, word_timestamps=word_timestamps,
        mlx_repo=mlx_repo,
    )


def test_run_pass_dispatches_to_mlx_when_engine_is_mlx_whisper(monkeypatch) -> None:
    """`_run_pass` with engine=mlx_whisper invokes mlx_whisper.transcribe and
    must NOT load a faster-whisper model."""
    from podclipper import transcribe as tx

    recorder: list = []
    _install_fake_mlx_whisper(monkeypatch, _EMPTY_MLX_RESULT, recorder)

    # Sentinel so we'd see it if _get_model accidentally got called.
    def boom(*a, **kw):
        raise AssertionError("_get_model must not be called for engine=mlx_whisper")
    monkeypatch.setattr(tx, "_get_model", boom)

    pass_cfg = _make_pass_cfg()
    audio = np.zeros(16000, dtype=np.float32)

    tx._run_pass(
        audio,
        engine="mlx_whisper",
        pass_cfg=pass_cfg,
        language="en",
        word_timestamps=True,
        time_offset=0.0,
    )

    assert len(recorder) == 1
    assert recorder[0]["path_or_hf_repo"] == "mlx-community/whisper-base-mlx"


def test_run_pass_dispatches_to_faster_whisper_when_engine_is_faster_whisper(monkeypatch) -> None:
    """`_run_pass` with engine=faster_whisper loads a faster-whisper model via
    `_get_model` and must NOT touch mlx_whisper."""
    from podclipper import transcribe as tx

    # Trip wire: if mlx is imported we fail loud.
    def boom_import(name, *a, **kw):
        if name == "mlx_whisper":
            raise AssertionError("mlx_whisper must not be imported for engine=faster_whisper")
        return original_import(name, *a, **kw)
    original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    monkeypatch.setattr("builtins.__import__", boom_import)

    get_model_calls: list = []
    fake_model = object()

    def fake_get_model(model, compute_type, device):
        get_model_calls.append((model, compute_type, device))
        return fake_model

    transcribe_array_calls: list = []
    def fake_transcribe_array(model, audio, **kwargs):
        transcribe_array_calls.append({"model": model, "audio": audio, **kwargs})
        return [], "en"

    monkeypatch.setattr(tx, "_get_model", fake_get_model)
    monkeypatch.setattr(tx, "_transcribe_array", fake_transcribe_array)

    pass_cfg = _make_pass_cfg(model="base", compute_type="int8", device="cpu", beam_size=5)
    audio = np.zeros(16000, dtype=np.float32)

    tx._run_pass(
        audio,
        engine="faster_whisper",
        pass_cfg=pass_cfg,
        language="en",
        word_timestamps=True,
        time_offset=2.5,
    )

    assert get_model_calls == [("base", "int8", "cpu")]
    assert len(transcribe_array_calls) == 1
    call = transcribe_array_calls[0]
    assert call["model"] is fake_model
    assert call["beam_size"] == 5
    assert call["time_offset"] == 2.5


def test_mlx_adapter_uses_default_confidence_when_probability_missing(monkeypatch) -> None:
    """A word dict without `probability` defaults to confidence=1.0."""
    from podclipper.transcribe import _transcribe_array_mlx

    mlx_result = {
        "text": " hi", "language": "en",
        "segments": [{
            "start": 0.0, "end": 0.3, "text": " hi",
            "words": [{"word": " hi", "start": 0.0, "end": 0.3}],
        }],
    }
    recorder: list = []
    _install_fake_mlx_whisper(monkeypatch, mlx_result, recorder)

    segs, _ = _transcribe_array_mlx(
        np.zeros(16000, dtype=np.float32),
        repo="x",
        language=None,
        word_timestamps=True,
    )

    assert segs[0].words[0].confidence == 1.0


def test_mlx_adapter_skips_word_loop_when_word_timestamps_false(monkeypatch) -> None:
    """word_timestamps=False yields empty Word lists even if mlx returns words."""
    from podclipper.transcribe import _transcribe_array_mlx

    mlx_result = {
        "text": " hi", "language": "en",
        "segments": [{
            "start": 0.0, "end": 0.3, "text": " hi",
            "words": [{"word": " hi", "start": 0.0, "end": 0.3, "probability": 0.9}],
        }],
    }
    recorder: list = []
    _install_fake_mlx_whisper(monkeypatch, mlx_result, recorder)

    segs, _ = _transcribe_array_mlx(
        np.zeros(16000, dtype=np.float32),
        repo="x",
        language=None,
        word_timestamps=False,
    )

    assert segs[0].words == []


# --------------------------------------------------------------------------- #
# Public callers honor cfg.transcribe.engine
# --------------------------------------------------------------------------- #

def _full_cfg(engine: str) -> SimpleNamespace:
    """Minimal cfg shape transcribe_second_pass reads."""
    return SimpleNamespace(
        transcribe=SimpleNamespace(
            engine=engine,
            language=None,
            second_pass=_make_pass_cfg(),
        ),
    )


def test_transcribe_second_pass_uses_mlx_when_engine_is_mlx_whisper(
    monkeypatch, tmp_path
) -> None:
    """End-to-end at the public API: setting cfg.transcribe.engine='mlx_whisper'
    causes transcribe_second_pass to call mlx_whisper.transcribe instead of
    loading a faster-whisper model."""
    from podclipper import transcribe as tx

    mlx_result = {
        "text": " hi", "language": "en",
        "segments": [{
            "start": 0.0, "end": 0.3, "text": " hi",
            "words": [{"word": " hi", "start": 0.0, "end": 0.3, "probability": 0.9}],
        }],
    }
    recorder: list = []
    _install_fake_mlx_whisper(monkeypatch, mlx_result, recorder)

    def boom(*a, **kw):
        raise AssertionError("_get_model must not be called for engine=mlx_whisper")
    monkeypatch.setattr(tx, "_get_model", boom)

    fake_audio = np.zeros(16000, dtype=np.float32)
    monkeypatch.setattr(tx, "_decode_audio_to_float32", lambda p: fake_audio)

    cfg = _full_cfg("mlx_whisper")
    fake_clip_path = tmp_path / "clip.mp4"
    fake_clip_path.write_bytes(b"")

    words = tx.transcribe_second_pass(fake_clip_path, cfg)

    assert len(recorder) == 1
    assert recorder[0]["path_or_hf_repo"] == "mlx-community/whisper-base-mlx"
    assert len(words) == 1
    assert words[0].text == " hi"


def test_transcribe_first_pass_uses_mlx_when_engine_is_mlx_whisper(
    monkeypatch, tmp_path
) -> None:
    """transcribe_first_pass also dispatches via cfg.transcribe.engine — both
    passes are switched by a single engine knob."""
    from podclipper import transcribe as tx

    mlx_result = {
        "text": " hi", "language": "en",
        "segments": [{
            "start": 0.0, "end": 0.3, "text": " hi",
            "words": [{"word": " hi", "start": 0.0, "end": 0.3, "probability": 0.9}],
        }],
    }
    recorder: list = []
    _install_fake_mlx_whisper(monkeypatch, mlx_result, recorder)

    def boom(*a, **kw):
        raise AssertionError("_get_model must not be called for engine=mlx_whisper")
    monkeypatch.setattr(tx, "_get_model", boom)

    fake_audio = np.zeros(16000, dtype=np.float32)
    monkeypatch.setattr(tx, "_decode_audio_to_float32", lambda p: fake_audio)

    cfg = SimpleNamespace(
        transcribe=SimpleNamespace(
            engine="mlx_whisper",
            language=None,
            first_pass=_make_pass_cfg(mlx_repo="mlx-community/whisper-base-mlx"),
        ),
        audio=SimpleNamespace(chunk_seconds=300, chunk_overlap_seconds=10),
    )
    cfg.transcribe.first_pass.max_workers = 1  # single-threaded for determinism
    fake_audio_path = tmp_path / "audio.wav"
    fake_audio_path.write_bytes(b"")

    transcript = tx.transcribe_first_pass(fake_audio_path, duration=1.0, cfg=cfg)

    assert any(call["path_or_hf_repo"] == "mlx-community/whisper-base-mlx"
               for call in recorder)
    assert transcript.language == "en"


# --------------------------------------------------------------------------- #
# Cache path helpers (engine-suffixed so fw and mlx outputs don't collide)
# --------------------------------------------------------------------------- #

def test_engine_suffix_returns_fw_for_faster_whisper() -> None:
    """Suffix is the short engine tag — used in cache filenames so a single
    source video can be transcribed by both engines without overwriting."""
    from podclipper.transcribe import engine_suffix

    assert engine_suffix("faster_whisper") == "fw"


def test_engine_suffix_returns_mlx_for_mlx_whisper() -> None:
    from podclipper.transcribe import engine_suffix

    assert engine_suffix("mlx_whisper") == "mlx"


def test_first_pass_cache_path_includes_engine_suffix(tmp_path) -> None:
    """The cache filename includes the engine tag so fw and mlx outputs never
    overwrite each other — required for side-by-side comparison runs."""
    from podclipper.transcribe import first_pass_cache_path

    fw_path = first_pass_cache_path(tmp_path, "faster_whisper")
    mlx_path = first_pass_cache_path(tmp_path, "mlx_whisper")

    assert fw_path == tmp_path / "first_pass_transcript_fw.json"
    assert mlx_path == tmp_path / "first_pass_transcript_mlx.json"
    assert fw_path != mlx_path


def test_words_cache_path_includes_engine_suffix(tmp_path) -> None:
    """Per-clip 2nd-pass cache also engine-tagged."""
    from podclipper.transcribe import words_cache_path

    fw_path = words_cache_path(tmp_path, "faster_whisper")
    mlx_path = words_cache_path(tmp_path, "mlx_whisper")

    assert fw_path == tmp_path / "words_fw.json"
    assert mlx_path == tmp_path / "words_mlx.json"


def test_first_pass_with_mlx_engine_calls_transcribe_serially_not_in_parallel(
    monkeypatch, tmp_path
) -> None:
    """mlx-whisper uses numba internally and is not thread-safe. With multiple
    chunks the ThreadPoolExecutor must serialize calls (max_workers=1) or the
    numba workqueue layer crashes mid-run."""
    from podclipper import transcribe as tx

    mlx_result = {
        "text": " hi", "language": "en",
        "segments": [{"start": 0.0, "end": 0.5, "text": " hi", "words": []}],
    }

    # Track concurrent mlx invocations — if any two overlap, fail.
    import threading
    in_flight = [0]
    max_concurrent = [0]
    lock = threading.Lock()

    fake = types.ModuleType("mlx_whisper")
    def fake_transcribe(audio, **kwargs):
        with lock:
            in_flight[0] += 1
            max_concurrent[0] = max(max_concurrent[0], in_flight[0])
        try:
            import time
            time.sleep(0.05)  # widen the race window
            return mlx_result
        finally:
            with lock:
                in_flight[0] -= 1
    fake.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)

    fake_audio = np.zeros(16000 * 900, dtype=np.float32)  # 15 min → 3 chunks
    monkeypatch.setattr(tx, "_decode_audio_to_float32", lambda p: fake_audio)

    cfg = SimpleNamespace(
        transcribe=SimpleNamespace(
            engine="mlx_whisper",
            language=None,
            first_pass=_make_pass_cfg(mlx_repo="mlx-community/whisper-base-mlx"),
        ),
        audio=SimpleNamespace(chunk_seconds=300, chunk_overlap_seconds=10),
    )
    cfg.transcribe.first_pass.max_workers = 4  # config requests parallelism — must be overridden

    fake_audio_path = tmp_path / "audio.wav"
    fake_audio_path.write_bytes(b"")

    tx.transcribe_first_pass(fake_audio_path, duration=900.0, cfg=cfg)

    assert max_concurrent[0] == 1, (
        f"mlx_whisper must be called serially (max_workers=1); "
        f"observed {max_concurrent[0]} concurrent calls"
    )
