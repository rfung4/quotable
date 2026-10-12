"""Optional speaker diarization (``quotable[diarize]``): pyannote turns → word speakers.

pyannote is imported lazily so the base install never needs torch. ``assign_speakers`` is
a pure function and carries all the alignment logic; it needs no pyannote to test.
"""

from __future__ import annotations

import bisect
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quotable.config import Settings
from quotable.models import DiarizationUnavailableError, Segment, Word
from quotable.transcribe import INFERENCE_LOCK, SAMPLE_RATE

MODEL_ID = "pyannote/speaker-diarization-community-1"
TERMS_URL = f"https://huggingface.co/{MODEL_ID}"
NEAREST_TURN_S = 1.0
_TOKEN_FIX = (
    f"Accept the model's terms at {TERMS_URL} with your Hugging Face account, "
    "then set QUOTABLE_HF_TOKEN (or HF_TOKEN) to a read token."
)
_AUTH_HINTS = ("401", "403", "gated", "unauthorized", "forbidden", "access", "token")

_LOAD_LOCK = threading.Lock()
_pipeline: tuple[str, Any] | None = None


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


def load_pipeline(token: str | None) -> Any:
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationUnavailableError(
            "Speaker labels need the optional diarize extra, which is not installed.",
            fix="pip install 'quotable[diarize]' (or uvx 'quotable[diarize]'), then retry.",
        ) from exc
    if not token:
        raise DiarizationUnavailableError(
            "Speaker labels need a Hugging Face token for the gated pyannote model.",
            fix=_TOKEN_FIX,
        )
    try:
        pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
    except Exception as exc:
        if any(hint in str(exc).lower() for hint in _AUTH_HINTS):
            raise DiarizationUnavailableError(
                f"Hugging Face refused the pyannote model: {exc}", fix=_TOKEN_FIX
            ) from exc
        raise DiarizationUnavailableError(f"Could not load the pyannote pipeline: {exc}") from exc
    if pipeline is None:  # pyannote returns None (and prints) when the repo is gated for you
        raise DiarizationUnavailableError(
            "Hugging Face did not grant access to the pyannote model.", fix=_TOKEN_FIX
        )
    try:
        import torch

        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception:  # CPU diarization is slow but correct
        pass
    return pipeline


def get_pipeline(token: str | None) -> Any:
    global _pipeline
    with _LOAD_LOCK:
        if _pipeline is None or _pipeline[0] != (token or ""):
            _pipeline = (token or "", load_pipeline(token))
        return _pipeline[1]


def _waveform(audio: Path) -> dict:
    import torch
    from faster_whisper.audio import decode_audio

    samples = decode_audio(str(audio), sampling_rate=SAMPLE_RATE)
    # In-memory input avoids pyannote's torchcodec decoder (and its FFmpeg libraries).
    return {"waveform": torch.from_numpy(samples).unsqueeze(0), "sample_rate": SAMPLE_RATE}


def _annotation(output: Any) -> Any:
    # pyannote 4 returns DiarizeOutput; the exclusive (non-overlapping) view is the one
    # meant for aligning with STT words. pyannote 3 returned the Annotation itself.
    for name in ("exclusive_speaker_diarization", "speaker_diarization"):
        if getattr(output, name, None) is not None:
            return getattr(output, name)
    return output


def diarize(audio: Path, settings: Settings) -> list[Turn]:
    pipeline = get_pipeline(settings.hf_token)
    with INFERENCE_LOCK:
        output = pipeline(_waveform(audio))
    turns = [
        Turn(start=float(seg.start), end=float(seg.end), speaker=str(label))
        for seg, _, label in _annotation(output).itertracks(yield_label=True)
    ]
    return sorted(turns, key=lambda t: (t.start, t.end))


# --- alignment ------------------------------------------------------------------------


class _TurnIndex:
    def __init__(self, turns: Sequence[Turn]) -> None:
        self.turns = sorted(turns, key=lambda t: (t.start, t.end))
        self.starts = [t.start for t in self.turns]
        self.max_len = max((t.end - t.start for t in self.turns), default=0.0)

    def near(self, start: float, end: float, slack: float) -> list[Turn]:
        """Turns that come within ``slack`` of [start, end], in time order."""
        hi = bisect.bisect_right(self.starts, end + slack)
        lo = bisect.bisect_left(self.starts, start - slack - self.max_len)
        return [t for t in self.turns[lo:hi] if t.end >= start - slack]

    def speaker_for(self, start: float, end: float) -> str | None:
        """Max overlap (ties → earlier turn); no overlap → nearest turn within 1 s; else None."""
        candidates = self.near(start, end, NEAREST_TURN_S)
        best, best_overlap = None, 0.0
        for t in candidates:
            overlap = min(end, t.end) - max(start, t.start)
            if overlap > best_overlap:
                best, best_overlap = t, overlap
        if best is not None:
            return best.speaker
        nearest, nearest_gap = None, 0.0
        for t in candidates:
            gap = max(t.start - end, start - t.end, 0.0)
            if gap <= NEAREST_TURN_S and (nearest is None or gap < nearest_gap):
                nearest, nearest_gap = t, gap
        return nearest.speaker if nearest is not None else None


def _majority(labels: Sequence[str | None]) -> str | None:
    counts = Counter(label for label in labels if label is not None)
    if not counts:
        return None
    top = max(counts.values())
    return next(label for label in labels if label is not None and counts[label] == top)


def assign_speakers(segments: Sequence[Segment], turns: Sequence[Turn]) -> list[Segment]:
    """Label every word by its turn, and every segment by its words' majority."""
    index = _TurnIndex(turns)
    out: list[Segment] = []
    for seg in segments:
        words = [
            Word(start=w.start, end=w.end, text=w.text, speaker=index.speaker_for(w.start, w.end))
            for w in seg.words
        ]
        speaker = (
            _majority([w.speaker for w in words])
            if words
            else index.speaker_for(seg.start, seg.end)
        )
        out.append(seg.model_copy(update={"words": words, "speaker": speaker}))
    return out
