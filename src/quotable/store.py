"""On-disk transcript cache: one directory per (source, variant), written atomically.

Layout under ``OUTPUT_DIR``::

    {source_key}/{variant}/transcript.{json,txt,srt}   finished, read-only
    {source_key}/.{variant}.work/partial.jsonl         in progress; one segment per line
    .locks/{source_key}__{variant}.lock                cross-process variant lock

Ids reach this module from the agent, so ``parse_id`` vets them before any path is built.
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path

from filelock import FileLock

from quotable.models import (
    Method,
    Segment,
    SourceError,
    Transcript,
    TranscriptionError,
    TranscriptPage,
)

FILES = ("transcript.json", "transcript.txt", "transcript.srt")
PARTIAL = "partial.jsonl"
DEFAULT_LIMIT = 20_000
MAX_LIMIT = 100_000

_ID = re.compile(r"^([A-Za-z0-9_-]+):([A-Za-z0-9_.-]+)$")
_UNSAFE_PART = re.compile(r"[^A-Za-z0-9_.-]+")
_BAD_ID = "Unknown transcript id."
_BAD_ID_FIX = "Use the `id` returned by `transcribe`, e.g. `Youtube-abc123:whisper-en-turbo`."


def variant_name(
    method: Method | str, language: str | None, model: str | None, diarized: bool = False
) -> str:
    """``{method}-{language}-{model|captions}[-diarized]`` path-safe."""
    method = Method(method)
    model_part = "captions" if method is Method.CAPTIONS else (model or "whisper")
    parts = [method.value, language or "und", Path(model_part).name or model_part]
    name = "-".join(_UNSAFE_PART.sub("_", p).strip("._") or "x" for p in parts)
    return f"{name}-diarized" if diarized else name


def _check_part(part: str) -> None:
    if part in (".", "..") or part.startswith("."):
        raise SourceError(_BAD_ID, fix=_BAD_ID_FIX)


def parse_id(transcript_id: str) -> tuple[str, str]:
    """``source_key:variant`` → its two parts; pure string checks, no filesystem access."""
    m = _ID.fullmatch(transcript_id or "")
    if not m:
        raise SourceError(_BAD_ID, fix=_BAD_ID_FIX)
    key, variant = m.groups()
    _check_part(key)
    _check_part(variant)
    return key, variant


def _ts(seconds: float, sep: str) -> str:
    ms = max(0, round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _line(seg: Segment) -> str:
    return f"{seg.speaker}: {seg.text}" if seg.speaker else seg.text


def render_txt(segments: list[Segment]) -> str:
    return "".join(_line(s) + "\n" for s in segments)


def render_timestamped(segments: list[Segment]) -> str:
    return "".join(f"[{_hms(s.start)}] {_line(s)}\n" for s in segments)


def render_srt(segments: list[Segment]) -> str:
    blocks = [
        f"{i}\n{_ts(s.start, ',')} --> {_ts(s.end, ',')}\n{_line(s)}\n"
        for i, s in enumerate(segments, start=1)
    ]
    return "\n".join(blocks)


class Store:
    def __init__(self, output_dir: Path) -> None:
        self.root = Path(output_dir)

    # --- paths ------------------------------------------------------------------------

    def _under_root(self, path: Path) -> Path:
        root = self.root.resolve()
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise SourceError(_BAD_ID, fix=_BAD_ID_FIX)
        return path

    def variant_dir(self, key: str, variant: str) -> Path:
        _check_part(key)
        _check_part(variant)
        return self._under_root(self.root / key / variant)

    def work_dir(self, key: str, variant: str) -> Path:
        _check_part(key)
        _check_part(variant)
        return self._under_root(self.root / key / f".{variant}.work")

    def variant_lock(self, key: str, variant: str) -> FileLock:
        _check_part(key)
        _check_part(variant)
        locks = self.root / ".locks"
        locks.mkdir(parents=True, exist_ok=True)
        return FileLock(str(locks / f"{key}__{variant}.lock"))

    # --- partial work -----------------------------------------------------------------

    def append_partial(self, key: str, variant: str, segment: Segment) -> None:
        work = self.work_dir(key, variant)
        work.mkdir(parents=True, exist_ok=True)
        with (work / PARTIAL).open("a", encoding="utf-8") as fh:
            fh.write(segment.model_dump_json() + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def partial_segments(self, key: str, variant: str) -> list[Segment]:
        path = self.work_dir(key, variant) / PARTIAL
        if not path.is_file():
            return []
        out: list[Segment] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(Segment.model_validate_json(line))
            except ValueError:
                break  # a torn final line from a crash mid-append; resume before it
        return out

    def resume_point(self, key: str, variant: str) -> float:
        segments = self.partial_segments(key, variant)
        return segments[-1].end if segments else 0.0

    def discard_partial(self, key: str, variant: str) -> None:
        shutil.rmtree(self.work_dir(key, variant), ignore_errors=True)

    # --- finished transcripts ---------------------------------------------------------

    def finalize(self, transcript: Transcript) -> Path:
        """Write all three files to a staging dir, then rename it into place.

        A failure at any point leaves no cache entry. An empty transcript is refused so
        that nothing that could be a failed run is ever stored as a result.
        """
        if not transcript.segments:
            raise TranscriptionError("No speech was found, so nothing was cached.")
        key, variant = transcript.source_key, transcript.variant
        final = self.variant_dir(key, variant)
        staging = self._under_root(self.root / key / f".{variant}.staging-{uuid.uuid4().hex}")
        staging.mkdir(parents=True)
        try:
            (staging / "transcript.json").write_text(
                transcript.model_dump_json(indent=1), encoding="utf-8"
            )
            (staging / "transcript.txt").write_text(
                render_txt(transcript.segments), encoding="utf-8"
            )
            (staging / "transcript.srt").write_text(
                render_srt(transcript.segments), encoding="utf-8"
            )
            old = None
            if final.exists():  # a forced re-run replaces the old entry
                old = final.with_name(f".{variant}.old-{uuid.uuid4().hex}")
                final.rename(old)
            staging.rename(final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
        self.discard_partial(key, variant)
        return final

    def load_variant(self, key: str, variant: str) -> Transcript | None:
        path = self.variant_dir(key, variant) / "transcript.json"
        try:
            return Transcript.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except ValueError:  # unreadable entry: treat as a miss so a re-run replaces it
            return None

    def load(self, transcript_id: str) -> Transcript | None:
        key, variant = parse_id(transcript_id)
        return self.load_variant(key, variant)

    def page(
        self,
        transcript_id: str,
        offset: int = 0,
        limit: int = DEFAULT_LIMIT,
        start_s: float | None = None,
        end_s: float | None = None,
        timestamps: bool = False,
    ) -> TranscriptPage:
        """A character page of the rendered text, optionally within a time window.

        Pages end on a line boundary when they can, and ``next_offset`` is where the
        next one starts (``None`` at the end). With a window, offsets count within it.
        """
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if not 1 <= limit <= MAX_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
        if start_s is not None and end_s is not None and end_s < start_s:
            raise ValueError("end_s must be >= start_s")

        transcript = self.load(transcript_id)
        if transcript is None:
            raise SourceError(_BAD_ID, fix=_BAD_ID_FIX)
        segments = [
            s
            for s in transcript.segments
            if (start_s is None or s.end > start_s) and (end_s is None or s.start < end_s)
        ]
        text = (render_timestamped if timestamps else render_txt)(segments)
        chunk = text[offset : offset + limit]
        next_offset: int | None = None
        if offset + limit < len(text):
            cut = chunk.rfind("\n")
            if cut >= 0:
                chunk = chunk[: cut + 1]
            next_offset = offset + len(chunk)
        return TranscriptPage(
            id=transcript_id,
            text=chunk,
            offset=offset,
            next_offset=next_offset,
            total_chars=len(text),
            start_s=start_s,
            end_s=end_s,
        )
