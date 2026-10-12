from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from quotable import transcribe as tx
from quotable.config import Settings
from quotable.models import TranscriptionError

SR = tx.SAMPLE_RATE


@dataclass
class FWWord:
    start: float
    end: float
    word: str


@dataclass
class FWSegment:
    start: float
    end: float
    text: str
    words: list = field(default_factory=list)


@dataclass
class FWInfo:
    language: str = "en"
    duration: float = 10.0


SEGMENTS = [
    FWSegment(0.0, 2.0, " ask not", [FWWord(0.0, 1.0, " ask"), FWWord(1.0, 2.0, " not")]),
    FWSegment(2.0, 4.0, " what", [FWWord(2.0, 4.0, " what"), FWWord(4.0, 4.0, " ")]),
]


class FakeModel:
    """Records constructions and transcribe calls; can fail like a broken CUDA install."""

    built: list[tuple] = []
    calls: list[dict] = []
    fail_load_on: set[str] = set()
    fail_first_segment_on: set[str] = set()
    load_error = RuntimeError("CUDA failed with error no CUDA-capable device is detected")

    def __init__(self, name, device, compute_type):
        if device in self.fail_load_on:
            raise self.load_error
        self.device = device
        FakeModel.built.append((name, device, compute_type))

    def transcribe(self, audio, **kwargs):
        FakeModel.calls.append({"device": self.device, **kwargs})

        def gen():
            if self.device in self.fail_first_segment_on:
                raise RuntimeError("Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so}")
            yield from SEGMENTS

        return gen(), FWInfo()


@pytest.fixture(autouse=True)
def fake_whisper(monkeypatch):
    FakeModel.built, FakeModel.calls = [], []
    FakeModel.fail_load_on, FakeModel.fail_first_segment_on = set(), set()
    monkeypatch.setattr(tx, "_whisper_model_cls", lambda: FakeModel)
    monkeypatch.setattr(tx, "_decode", lambda path: np.zeros(SR * 10, dtype=np.float32))
    monkeypatch.setattr(tx, "_model", None)
    monkeypatch.setattr(tx, "_cuda_broken", False)


def _settings(device="cpu", model="base", compute_type=None) -> Settings:
    compute_type = compute_type or ("float16" if device == "cuda" else "int8")
    return Settings(
        output_dir=Path("/tmp/q"), model=model, device=device, compute_type=compute_type
    )


def _run(settings, start_s=0.0, language=None):
    with tx.transcribe_iter(Path("a.flac"), language, start_s, settings) as run:
        return run, list(run)


# --- model singleton ------------------------------------------------------------------


def test_model_is_loaded_once_and_reused():
    s = _settings()
    assert tx.get_model(s) is tx.get_model(s)
    _run(s)
    _run(s)
    assert FakeModel.built == [("base", "cpu", "int8")]


def test_a_different_model_replaces_the_singleton():
    tx.get_model(_settings(model="base"))
    tx.get_model(_settings(model="tiny"))
    assert [b[0] for b in FakeModel.built] == ["base", "tiny"]
    assert tx._model[0] == ("tiny", "cpu", "int8")


def test_cuda_load_failure_falls_back_to_cpu_and_stays_there(capsys):
    FakeModel.fail_load_on = {"cuda"}
    s = _settings(device="cuda", model="turbo")
    run, segs = _run(s)
    assert FakeModel.built == [("turbo", "cpu", "int8")]
    assert "falling back to CPU" in capsys.readouterr().err
    assert segs
    _run(s)  # the next job does not retry CUDA
    assert FakeModel.built == [("turbo", "cpu", "int8")]


def test_lazy_cudnn_failure_on_first_segment_retries_on_cpu(capsys):
    FakeModel.fail_first_segment_on = {"cuda"}
    run, segs = _run(_settings(device="cuda", model="turbo"))
    assert [c["device"] for c in FakeModel.calls] == ["cuda", "cpu"]
    assert [s.text for s in segs] == ["ask not", "what"]
    assert "retrying on CPU" in capsys.readouterr().err


def test_non_cuda_load_failure_is_a_transcription_error(monkeypatch):
    monkeypatch.setattr(FakeModel, "load_error", OSError("model 'nope' not found on the Hub"))
    FakeModel.fail_load_on = {"cpu"}
    with pytest.raises(TranscriptionError, match="nope"):
        _run(_settings(model="nope"))


# --- decoding options and resume ------------------------------------------------------


def test_fresh_run_uses_vad_and_word_timestamps_without_clips():
    run, _ = _run(_settings(), language="en")
    (call,) = FakeModel.calls
    assert call["vad_filter"] is True and call["word_timestamps"] is True
    assert call["language"] == "en"
    assert "clip_timestamps" not in call
    assert (run.language, run.duration) == ("en", 10.0)


def test_resume_passes_vad_speech_chunks_from_the_resume_point(monkeypatch):
    import faster_whisper.vad

    chunks = [(0.5, 2.0), (3.0, 6.0), (7.0, 9.5)]
    monkeypatch.setattr(
        faster_whisper.vad,
        "get_speech_timestamps",
        lambda audio, *a, **k: [{"start": int(s * SR), "end": int(e * SR)} for s, e in chunks],
    )
    _run(_settings(), start_s=4.0)
    clips = FakeModel.calls[0]["clip_timestamps"]
    assert clips == [4.0, 6.0, 7.0, 9.5]  # chunk before 4 s dropped, straddling one clipped
    assert clips != [4.0]
    assert all(c >= 4.0 for c in clips)


def test_resume_into_trailing_silence_decodes_nothing(monkeypatch):
    import faster_whisper.vad

    monkeypatch.setattr(
        faster_whisper.vad,
        "get_speech_timestamps",
        lambda audio, *a, **k: [{"start": 0, "end": 2 * SR}],
    )
    run, segs = _run(_settings(), start_s=5.0)
    assert segs == [] and FakeModel.calls == []
    assert run.duration == 10.0


# --- segments and locking -------------------------------------------------------------


def test_segments_are_converted_with_clean_words():
    _, segs = _run(_settings())
    assert [s.text for s in segs] == ["ask not", "what"]
    assert [(w.text, w.start, w.end) for w in segs[0].words] == [
        ("ask", 0.0, 1.0),
        ("not", 1.0, 2.0),
    ]
    assert [w.text for w in segs[1].words] == ["what"]  # blank word dropped


def test_inference_lock_is_held_for_the_whole_run_and_released_on_early_exit():
    with tx.transcribe_iter(Path("a.flac"), settings=_settings()) as run:
        assert tx.INFERENCE_LOCK.locked()
        for _ in run:
            break  # a cancel between segments
    assert not tx.INFERENCE_LOCK.locked()


def test_lock_released_when_start_fails(monkeypatch):
    monkeypatch.setattr(FakeModel, "load_error", OSError("disk full"))
    FakeModel.fail_load_on = {"cpu"}
    with pytest.raises(TranscriptionError):
        _run(_settings())
    assert not tx.INFERENCE_LOCK.locked()
