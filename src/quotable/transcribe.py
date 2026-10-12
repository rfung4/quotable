"""faster-whisper as a segment iterator, with one warm model per process.

``transcribe_iter`` returns a ``WhisperRun``: enter it to start (it takes the process-wide
inference lock, so two runs never share an 8 GB card), iterate it for ``Segment``s as
they are decoded, and leave it to release the lock — leaving early is how cancel works.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from quotable.config import Settings
from quotable.models import Segment, TranscriptionError, Word

SAMPLE_RATE = 16000
_CUDA_HINTS = ("cuda", "cudnn", "cublas", "libcu", "gpu")

INFERENCE_LOCK = threading.Lock()
_MODEL_LOCK = threading.Lock()
_model: tuple[tuple[str, str, str], Any] | None = None
_cuda_broken = False


def _whisper_model_cls():
    from faster_whisper import WhisperModel

    return WhisperModel


def _is_cuda_failure(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(hint in text for hint in _CUDA_HINTS)


def _warn(msg: str) -> None:
    print(f"quotable: {msg}", file=sys.stderr)


def _cpu_compute_type(settings: Settings) -> str:
    return settings.compute_type if settings.device == "cpu" else "int8"


def get_model(settings: Settings, *, force_cpu: bool = False):
    """The process's one ``WhisperModel``; replaced (freeing VRAM) when the key changes.

    A CUDA load failure warns on stderr and loads on CPU instead; later calls in this
    process then go straight to CPU rather than failing again.
    """
    global _model, _cuda_broken
    device = "cpu" if force_cpu or _cuda_broken else settings.device
    compute_type = _cpu_compute_type(settings) if device == "cpu" else settings.compute_type
    key = (settings.model, device, compute_type)
    with _MODEL_LOCK:
        if _model is not None and _model[0] == key:
            return _model[1]
        _model = None  # drop the old one before loading another
        cls = _whisper_model_cls()
        try:
            model = cls(settings.model, device=device, compute_type=compute_type)
        except Exception as exc:
            if device != "cuda" or not _is_cuda_failure(exc):
                raise TranscriptionError(
                    f"Could not load Whisper model {settings.model!r}: {exc}"
                ) from exc
            _cuda_broken = True
            _warn(f"CUDA unavailable ({exc}); falling back to CPU. See the README's GPU notes.")
            device, compute_type = "cpu", _cpu_compute_type(settings)
            key = (settings.model, device, compute_type)
            model = cls(settings.model, device=device, compute_type=compute_type)
        _model = (key, model)
        return model


def _mark_cuda_broken() -> None:
    global _cuda_broken, _model
    _cuda_broken = True
    with _MODEL_LOCK:
        _model = None


def resume_clips(audio: Any, start_s: float) -> list[float]:
    """VAD speech chunks at or after ``start_s`` as a flat ``[s1, e1, s2, e2, …]``.

    faster-whisper ignores ``vad_filter`` whenever ``clip_timestamps`` is set, so a bare
    ``[start_s]`` would decode silence unfiltered (and hallucinate). Running VAD ourselves
    and passing only speech keeps a resumed run equivalent to a fresh one.
    """
    from faster_whisper.vad import get_speech_timestamps

    clips: list[float] = []
    for chunk in get_speech_timestamps(audio):
        start, end = chunk["start"] / SAMPLE_RATE, chunk["end"] / SAMPLE_RATE
        if end <= start_s:
            continue
        clips += [max(start, start_s), end]
    return clips


def _decode(path: Path):
    from faster_whisper.audio import decode_audio

    return decode_audio(str(path), sampling_rate=SAMPLE_RATE)


def _to_segment(seg: Any) -> Segment:
    words = [
        Word(start=w.start, end=max(w.end, w.start), text=w.word.strip())
        for w in (seg.words or [])
        if w.word.strip()
    ]
    return Segment(start=seg.start, end=seg.end, text=seg.text.strip(), words=words)


class WhisperRun:
    def __init__(self, audio: Path, language: str | None, start_s: float, settings: Settings):
        self.audio_path = audio
        self.requested_language = language
        self.start_s = start_s
        self.settings = settings
        self.language: str | None = None
        self.duration: float | None = None
        self._segments: Iterator[Any] | None = None
        self._audio: Any = None
        self._yielded = False

    def _start(self, *, force_cpu: bool = False) -> None:
        model = get_model(self.settings, force_cpu=force_cpu)
        kwargs: dict[str, Any] = {
            "language": self.requested_language,
            "word_timestamps": True,
            "vad_filter": True,
        }
        if self._audio is None:
            self._audio = _decode(self.audio_path)
        if self.start_s > 0:
            clips = resume_clips(self._audio, self.start_s)
            if not clips:  # nothing but silence after the resume point
                self.duration = len(self._audio) / SAMPLE_RATE
                self._segments = iter(())
                return
            kwargs["clip_timestamps"] = clips
        segments, info = model.transcribe(self._audio, **kwargs)
        self._segments = iter(segments)
        self.language = info.language
        self.duration = info.duration

    def __enter__(self) -> WhisperRun:
        INFERENCE_LOCK.acquire()
        try:
            try:
                self._start()
            except TranscriptionError:
                raise
            except Exception as exc:
                if not (self.settings.device == "cuda" and _is_cuda_failure(exc)):
                    raise TranscriptionError(f"Whisper failed to start: {exc}") from exc
                self._fall_back(exc)
        except BaseException:
            INFERENCE_LOCK.release()
            raise
        return self

    def _fall_back(self, exc: BaseException) -> None:
        _warn(f"CUDA failed during inference ({exc}); retrying on CPU.")
        _mark_cuda_broken()
        self._start(force_cpu=True)

    def __iter__(self) -> Iterator[Segment]:
        assert self._segments is not None, "enter the run before iterating it"
        while True:
            try:
                seg = next(self._segments)
            except StopIteration:
                return
            except Exception as exc:
                # cuDNN/cuBLAS load lazily, so a broken GPU often fails on the first segment.
                if self._yielded or not (
                    self.settings.device == "cuda" and _is_cuda_failure(exc) and not _cuda_broken
                ):
                    raise TranscriptionError(f"Whisper failed mid-run: {exc}") from exc
                self._fall_back(exc)
                continue
            self._yielded = True
            yield _to_segment(seg)

    def __exit__(self, *exc_info: object) -> None:
        close = getattr(self._segments, "close", None)
        if close is not None:
            close()
        self._segments = None
        INFERENCE_LOCK.release()


def transcribe_iter(
    audio: Path,
    language: str | None = None,
    start_s: float = 0.0,
    settings: Settings | None = None,
) -> WhisperRun:
    """``with transcribe_iter(path) as run: for seg in run: …`` — see the module docstring."""
    return WhisperRun(audio, language, start_s, settings or Settings.from_env())
