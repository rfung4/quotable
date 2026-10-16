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

# The VAD guard (QUOTABLE_VAD=auto). Silero keeps 1.0–1.4× the audio above -50 dBFS on every
# speech recording in the eval corpus (it pads its chunks), and 0.29× on a space-to-ground radio
# call whose speech it mostly drops. Below MIN_VAD_SHARE the run goes on without VAD.
LOUD_DBFS = -50.0
LOUD_FRAME = 480  # 30 ms
MIN_LOUD_S = 30.0  # too little sound to judge the VAD by
MIN_VAD_SHARE = 0.5

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


def speech_chunks(audio: Any) -> list[tuple[float, float]]:
    """Silero's speech chunks in seconds, with faster-whisper's default VAD options."""
    from faster_whisper.vad import get_speech_timestamps

    return [
        (c["start"] / SAMPLE_RATE, c["end"] / SAMPLE_RATE) for c in get_speech_timestamps(audio)
    ]


def resume_clips(
    audio: Any, start_s: float, chunks: list[tuple[float, float]] | None = None
) -> list[float]:
    """VAD speech chunks at or after ``start_s`` as a flat ``[s1, e1, s2, e2, …]``.

    faster-whisper ignores ``vad_filter`` whenever ``clip_timestamps`` is set, so a bare
    ``[start_s]`` would decode silence unfiltered (and hallucinate). Running VAD ourselves
    and passing only speech keeps a resumed run equivalent to a fresh one.
    """
    clips: list[float] = []
    for start, end in speech_chunks(audio) if chunks is None else chunks:
        if end <= start_s:
            continue
        clips += [max(start, start_s), end]
    return clips


def loud_seconds(audio: Any) -> float:
    """Seconds of ``audio`` in 30 ms frames whose RMS is above ``LOUD_DBFS``."""
    import numpy as np

    n = len(audio) // LOUD_FRAME
    if n == 0:
        return 0.0
    frames = np.asarray(audio[: n * LOUD_FRAME], dtype=np.float32).reshape(n, LOUD_FRAME)
    power = (frames * frames).mean(axis=1)
    loud = int((power > 10 ** (LOUD_DBFS / 10)).sum())
    return loud * LOUD_FRAME / SAMPLE_RATE


def vad_drops_sound(vad_s: float, loud_s: float) -> bool:
    """True when the VAD kept far less than the audio's sound: it is missing speech."""
    return loud_s >= MIN_LOUD_S and vad_s < MIN_VAD_SHARE * loud_s


def _vad_off_warning(vad_s: float, loud_s: float) -> str:
    return (
        f"the voice filter kept {vad_s:.0f} s of {loud_s:.0f} s of sound, so it is likely "
        "missing speech; transcribing without it (QUOTABLE_VAD=on keeps it)."
    )


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
        # None until decided; then whether this run filters with the VAD. A CPU fallback
        # restarts the run and keeps the decision.
        self.vad: bool | None = None if settings.vad == "auto" else settings.vad == "on"

    def _guard(self, vad_s: float) -> bool:
        """Decide the VAD for ``auto`` from the seconds it kept; warn when it is dropped."""
        loud_s = loud_seconds(self._audio)
        if vad_drops_sound(vad_s, loud_s):
            _warn(_vad_off_warning(vad_s, loud_s))
            return False
        return True

    def _start(self, *, force_cpu: bool = False) -> None:
        model = get_model(self.settings, force_cpu=force_cpu)
        kwargs: dict[str, Any] = {"language": self.requested_language, "word_timestamps": True}
        if self._audio is None:
            self._audio = _decode(self.audio_path)
        if self.start_s > 0:
            if self.vad is False:
                clips = [self.start_s]
            else:
                chunks = speech_chunks(self._audio)
                if self.vad is None:
                    self.vad = self._guard(sum(end - start for start, end in chunks))
                clips = (
                    resume_clips(self._audio, self.start_s, chunks) if self.vad else [self.start_s]
                )
            if not clips:  # nothing but silence after the resume point
                self.duration = len(self._audio) / SAMPLE_RATE
                self._segments = iter(())
                return
            kwargs["clip_timestamps"] = clips
            segments, info = model.transcribe(self._audio, vad_filter=False, **kwargs)
        else:
            segments, info = model.transcribe(
                self._audio, vad_filter=self.vad is not False, **kwargs
            )
            if self.vad is None:
                # faster-whisper runs the VAD before it decodes anything, so the guard reads
                # what it kept for free and only a tripped guard pays for a second call.
                self.vad = self._guard(info.duration_after_vad)
                if not self.vad:
                    segments, info = model.transcribe(self._audio, vad_filter=False, **kwargs)
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
