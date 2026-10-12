"""Domain models and typed errors.

The Pydantic models double as MCP tool output schemas, so field names are part of the
public surface. Errors carry a ``fix`` hint the server forwards to the agent.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Method(str, Enum):
    CAPTIONS = "captions"
    WHISPER = "whisper"


class SourceKind(str, Enum):
    FILE = "file"
    FEED = "feed"
    URL = "url"


class Word(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[Word] = Field(default_factory=list)


class Chapter(BaseModel):
    title: str
    start: float
    end: float


class Transcript(BaseModel):
    source_key: str
    variant: str
    method: Method
    language: str | None = None
    segments: list[Segment] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)
    title: str | None = None
    duration: float | None = None
    cite_url: str | None = None

    @property
    def id(self) -> str:
        return f"{self.source_key}:{self.variant}"

    @property
    def speakers(self) -> list[str]:
        seen: dict[str, None] = {}
        for seg in self.segments:
            if seg.speaker:
                seen.setdefault(seg.speaker)
        return list(seen)


class EpisodeInfo(BaseModel):
    guid: str | None = None
    index: int
    title: str
    published: str | None = None
    duration: float | None = None
    has_transcript: bool = False


class MediaInfo(BaseModel):
    source: str
    kind: SourceKind
    title: str | None = None
    duration: float | None = None
    has_captions: bool = False
    languages: list[str] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)
    cite_url: str | None = None
    episodes: list[EpisodeInfo] = Field(default_factory=list)


class QuotableError(Exception):
    """Base class. ``fix`` says what the caller can do about it."""

    default_fix = ""

    def __init__(self, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.fix = fix if fix is not None else self.default_fix


class SourceError(QuotableError):
    default_fix = "Check the path or URL, and that the source is reachable."


class NoCaptionsError(QuotableError):
    default_fix = "Retry with mode='auto' or mode='whisper' to transcribe the audio."


class DownloadError(QuotableError):
    default_fix = "Update yt-dlp, or check the network and that the media is available."


class TranscriptionError(QuotableError):
    default_fix = "Retry with a smaller QUOTABLE_MODEL or QUOTABLE_DEVICE=cpu."


class DiarizationUnavailableError(QuotableError):
    default_fix = "pip install 'quotable[diarize]' and set QUOTABLE_HF_TOKEN."


class CancelledError(QuotableError):
    default_fix = "Call again to resume from the saved partial transcript."


class TranscriptPage(BaseModel):
    id: str
    text: str
    offset: int
    next_offset: int | None = Field(
        default=None, description="Pass back as `offset` for the next page; null when done."
    )
    total_chars: int
    start_s: float | None = None
    end_s: float | None = None
