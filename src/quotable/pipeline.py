"""Orchestration: one source in, one cached transcript out.

``transcribe`` resolves the source (local file, podcast feed or yt-dlp URL) to its cache
key, then works per variant under that variant's file lock: the cache, then whatever
transcript the source already ships, then Whisper on the audio. Whisper segments go to
``on_segment`` and to ``partial.jsonl`` as they arrive, so a cancelled or crashed run
resumes where it stopped.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from yt_dlp import YoutubeDL

import quotable.diarize as dz
import quotable.transcribe as tx
from quotable import audio, captions, feeds
from quotable.audio import AudioSource
from quotable.config import Settings
from quotable.models import (
    CancelledError,
    Chapter,
    MediaInfo,
    Method,
    NoCaptionsError,
    Segment,
    SourceError,
    SourceKind,
    Transcript,
)
from quotable.store import Store, variant_name

Mode = Literal["auto", "captions", "whisper"]
MODES = ("auto", "captions", "whisper")
RECENT_EPISODES = 20
OnSegment = Callable[[Segment, float, "float | None"], None]
OnWait = Callable[[], None]
_SOURCE_FIX = "Pass an existing file path, a podcast feed URL or a media URL."


@dataclass(frozen=True)
class Outcome:
    transcript: Transcript
    path: Path  # the variant's directory: transcript.json, .txt, .srt
    cached: bool


@dataclass
class _Source:
    """A resolved source: its cache key and metadata, plus what fetching needs per kind."""

    kind: SourceKind
    key: str
    title: str | None = None
    duration: float | None = None
    cite_url: str | None = None
    chapters: list[Chapter] = field(default_factory=list)
    local: AudioSource | None = None  # file
    url: str = ""  # yt-dlp URL
    info: dict = field(default_factory=dict)  # yt-dlp URL
    ydl: Any = None  # yt-dlp URL
    feed: feeds.Feed | None = None  # feed
    episode: feeds.Episode | None = None  # feed


def _log(msg: str) -> None:
    print(f"quotable: {msg}", file=sys.stderr)


def _youtube_dl(outdir: Path) -> YoutubeDL:
    return YoutubeDL(audio.ydl_options(outdir))


@contextmanager
def _scratch(keep_audio: bool) -> Iterator[Path]:
    """A temp dir for downloaded audio, removed afterwards unless QUOTABLE_KEEP_AUDIO."""
    path = Path(tempfile.mkdtemp(prefix="quotable-"))
    try:
        yield path
    finally:
        if keep_audio and any(path.iterdir()):
            _log(f"kept the downloaded audio in {path}")
        else:
            shutil.rmtree(path, ignore_errors=True)


def _reject_episode(episode: str | int | None) -> None:
    if episode is not None and episode != "":
        raise SourceError(
            "`episode` applies only to podcast RSS feeds, and this source is not one.",
            fix="Drop `episode`, or pass the feed's URL.",
        )


def _resolve(source: str, episode: str | int | None, work: Path, stack: ExitStack) -> _Source:
    """Source → cache key and metadata. Reads metadata only; downloads no audio."""
    path = audio.local_path(source)
    if path is not None:
        _reject_episode(episode)
        local = audio.local_source(path)
        return _Source(
            kind=SourceKind.FILE,
            key=local.source_key,
            title=local.title,
            cite_url=local.cite_url,
            local=local,
        )
    if not audio.is_url(source):
        raise SourceError(f"No file at {source!r}, and it is not an http(s) URL.", fix=_SOURCE_FIX)
    if feeds.is_feed(source):
        feed = feeds.load_feed(source)
        ep = feeds.pick_episode(feed, episode)
        return _Source(
            kind=SourceKind.FEED,
            key=feeds.episode_key(ep),
            title=ep.title,
            duration=ep.duration,
            cite_url=feeds.episode_cite(ep),
            feed=feed,
            episode=ep,
        )
    _reject_episode(episode)
    ydl = stack.enter_context(_youtube_dl(work))
    info = audio.probe(ydl, source)
    return _Source(
        kind=SourceKind.URL,
        key=audio.url_key(info),
        title=info.get("title"),
        duration=info.get("duration"),
        cite_url=audio.url_cite(info, source),
        chapters=captions.chapters_from_info(info),
        url=source,
        info=info,
        ydl=ydl,
    )


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise CancelledError("Cancelled; the segments transcribed so far are kept.")


def _transcript(
    src: _Source,
    variant: str,
    method: Method,
    language: str | None,
    segments: list[Segment],
    duration: float | None = None,
) -> Transcript:
    return Transcript(
        source_key=src.key,
        variant=variant,
        method=method,
        language=language,
        segments=segments,
        chapters=src.chapters,
        title=src.title,
        duration=duration or src.duration,
        cite_url=src.cite_url,
    )


def _cached(store: Store, key: str, variant: str) -> Outcome | None:
    hit = store.load_variant(key, variant)
    return Outcome(hit, store.variant_dir(key, variant), cached=True) if hit else None


# --- captions the source already ships ------------------------------------------------


def _caption_plan(
    src: _Source, language: str | None
) -> tuple[str | None, Callable[[], list[Segment]]]:
    """The track's language (it names the variant) and a fetcher; nothing fetched yet."""
    if src.kind is SourceKind.FILE:
        raise NoCaptionsError("Local files carry no captions.")
    if src.kind is SourceKind.FEED:
        ep = src.episode
        ref = feeds.choose_transcript(ep, language)
        if ref is None:
            if ep.transcripts:
                raise NoCaptionsError(f"This episode ships no transcript in {language!r}.")
            raise NoCaptionsError("This episode ships no transcript.")
        return ref.language or language, lambda: feeds.fetch_transcript(ep, language) or []
    track = captions.pick_track(src.info, language)
    if track is None:
        raise captions.no_track_error(src.info, language)
    return track.language, lambda: captions.fetch(src.ydl, src.info, language)[0]


def _from_captions(src: _Source, store: Store, language: str | None, force: bool) -> Outcome:
    track_language, fetch = _caption_plan(src, language)
    variant = variant_name(Method.CAPTIONS, track_language, None)
    with store.variant_lock(src.key, variant):
        if not force and (hit := _cached(store, src.key, variant)) is not None:
            return hit
        transcript = _transcript(src, variant, Method.CAPTIONS, track_language, fetch())
        return Outcome(transcript, store.finalize(transcript), cached=False)


# --- Whisper on the audio ---------------------------------------------------------------


def _fetch_audio(src: _Source, work: Path) -> AudioSource:
    if src.kind is SourceKind.FILE:
        return src.local
    if src.kind is SourceKind.FEED:
        return feeds.download_enclosure(src.episode, work)
    return audio.download(src.ydl, src.info, src.url)


def _from_audio(
    src: _Source,
    store: Store,
    work: Path,
    *,
    language: str | None,
    diarize: bool,
    force: bool,
    on_segment: OnSegment | None,
    on_wait: OnWait | None,
    cancel: threading.Event | None,
    settings: Settings,
) -> Outcome:
    key = src.key
    language = tx.whisper_language(language)
    variant = variant_name(Method.WHISPER, language, settings.model, diarize)
    with store.variant_lock(key, variant):
        if force:
            store.discard_partial(key, variant)
        elif (hit := _cached(store, key, variant)) is not None:
            return hit
        if diarize:
            dz.get_pipeline(settings)  # a missing extra or token fails before Whisper
        _check_cancel(cancel)

        media = _fetch_audio(src, work)
        segments = store.partial_segments(key, variant)
        start_s = segments[-1].end if segments else 0.0
        with tx.transcribe_iter(
            media.path, language, start_s, settings, cancel=cancel, on_wait=on_wait
        ) as run:
            for seg in run:
                store.append_partial(key, variant, seg)
                segments.append(seg)
                if on_segment is not None:
                    on_segment(seg, seg.end, run.duration)
                _check_cancel(cancel)
            detected, duration = run.language, run.duration

        if diarize:
            _check_cancel(cancel)
            turns = dz.diarize(media.path, settings, cancel=cancel, on_wait=on_wait)
            segments = dz.assign_speakers(segments, turns)
        transcript = _transcript(
            src,
            variant,
            Method.WHISPER,
            detected or language,
            segments,
            duration=duration or media.duration,
        )
        return Outcome(transcript, store.finalize(transcript), cached=False)


# --- public API -------------------------------------------------------------------------


def transcribe(
    source: str,
    *,
    mode: Mode = "auto",
    language: str | None = None,
    diarize: bool = False,
    episode: str | int | None = None,
    force: bool = False,
    on_segment: OnSegment | None = None,
    on_wait: OnWait | None = None,
    cancel: threading.Event | None = None,
    settings: Settings | None = None,
) -> Outcome:
    """Transcribe ``source``, from the cache when this variant has been done before.

    ``mode``: ``auto`` prefers the source's own captions or feed transcript and falls back
    to Whisper; ``captions`` never runs Whisper (``NoCaptionsError`` instead); ``whisper``
    never looks for captions. ``diarize`` always takes the audio path, since captions
    carry no speakers. ``on_segment(segment, done_s, total_s)`` is called per Whisper
    segment; ``on_wait()`` once when the run queues behind another on the GPU. Setting
    ``cancel`` stops between segments (or in the queue) with ``CancelledError``, keeping
    the work so far for the next call. ``force`` ignores the cache and any partial run.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
    if diarize and mode == "captions":
        raise ValueError(
            "diarize=True needs the audio (captions carry no speakers); "
            "use mode='auto' or mode='whisper'."
        )
    language = language or None
    settings = settings or Settings.from_env()
    store = Store(settings.output_dir)
    with ExitStack() as stack:
        work = stack.enter_context(_scratch(settings.keep_audio))
        src = _resolve(source, episode, work, stack)
        if mode != "whisper" and not diarize:
            try:
                return _from_captions(src, store, language, force)
            except NoCaptionsError as exc:
                if mode == "captions":
                    raise
                if src.kind is not SourceKind.FILE:
                    _log(f"{exc.message} Transcribing the audio instead.")
        return _from_audio(
            src,
            store,
            work,
            language=language,
            diarize=diarize,
            force=force,
            on_segment=on_segment,
            on_wait=on_wait,
            cancel=cancel,
            settings=settings,
        )


def _file_duration(path: Path) -> float | None:
    import av

    try:
        with av.open(str(path)) as container:
            return container.duration / av.time_base if container.duration else None
    except (av.error.FFmpegError, OSError):
        return None


def media_info(source: str, episode: str | int | None = None) -> MediaInfo:
    """What a source is and what ``transcribe`` would find there. Downloads no audio.

    For a feed it describes the episode ``transcribe`` would pick (default: the newest)
    and lists the most recent episodes.
    """
    path = audio.local_path(source)
    if path is not None:
        _reject_episode(episode)
        return MediaInfo(
            source=source,
            kind=SourceKind.FILE,
            title=path.name,
            duration=_file_duration(path),
            cite_url=audio.file_cite(path),
        )
    with ExitStack() as stack:
        src = _resolve(source, episode, stack.enter_context(_scratch(False)), stack)
    if src.kind is SourceKind.FEED:
        refs = src.episode.transcripts
        has_captions = bool(refs)
        languages = [ref.language for ref in refs if ref.language]
        episodes = [ep.info() for ep in src.feed.episodes[:RECENT_EPISODES]]
    else:
        track = captions.pick_track(src.info)
        has_captions = track is not None
        languages = [lang for lang in src.info.get("subtitles") or {} if lang != "live_chat"]
        if track is not None and track.auto:
            languages.append(track.language)
        episodes = []
    title = src.title
    if src.kind is SourceKind.FEED and episode in (None, "") and src.feed.title:
        title = src.feed.title  # no episode asked for: this is the show
    return MediaInfo(
        source=source,
        kind=src.kind,
        title=title,
        duration=src.duration,
        has_captions=has_captions,
        languages=list(dict.fromkeys(languages)),
        chapters=src.chapters,
        cite_url=src.cite_url,
        episodes=episodes,
    )
