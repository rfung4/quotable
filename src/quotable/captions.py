"""Captions the source already ships: track choice, json3 / WebVTT parsing, chapters.

Caption text is third-party data. Parsing never trusts it to be well-formed: empty or
unparseable tracks are a miss (``NoCaptionsError``), never an empty transcript.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

from quotable.models import Chapter, NoCaptionsError, Segment, Word

FORMATS = ("json3", "vtt")
_SKIP_TRACKS = {"live_chat"}


class Opener(Protocol):
    def urlopen(self, req: str) -> Any: ...


@dataclass(frozen=True)
class CaptionTrack:
    language: str
    ext: str
    url: str
    auto: bool


def _base(lang: str) -> str:
    return lang.removesuffix("-orig").split("-")[0].lower()


def _best_format(formats: list[dict], prefer_json3: bool) -> dict | None:
    by_ext = {f.get("ext"): f for f in formats if f.get("url")}
    order = FORMATS if prefer_json3 else ("vtt",)
    return next((by_ext[ext] for ext in order if ext in by_ext), None)


def _track(lang: str, formats: list[dict], auto: bool, prefer_json3: bool) -> CaptionTrack | None:
    fmt = _best_format(formats, prefer_json3)
    if fmt is None:
        return None
    return CaptionTrack(
        language=lang.removesuffix("-orig"), ext=fmt["ext"], url=fmt["url"], auto=auto
    )


def _candidates(tracks: dict[str, list[dict]], wanted: str | None) -> list[str]:
    keys = [k for k in tracks if k not in _SKIP_TRACKS]
    if wanted is None:
        return keys
    exact = [k for k in keys if k.lower() in (wanted.lower(), f"{wanted.lower()}-orig")]
    base = [k for k in keys if _base(k) == _base(wanted) and k not in exact]
    return exact + base


def _usable_manual(info: dict) -> list[str]:
    manual: dict[str, list[dict]] = info.get("subtitles") or {}
    prefer_json3 = info.get("extractor_key") == "Youtube"
    return [k for k in _candidates(manual, None) if _best_format(manual[k], prefer_json3)]


def unnamed_language(info: dict, language: str | None) -> bool:
    """Several manual tracks and nothing names the video's language: any pick is a guess.

    TED talks look like this: 43 manual tracks, ``language`` None, no ``-orig`` track. The
    first track in dict order was Albanian, a translation returned as the talk.
    """
    return language is None and not info.get("language") and len(_usable_manual(info)) > 1


def no_track_error(info: dict, language: str | None) -> NoCaptionsError:
    if unnamed_language(info, language):
        return NoCaptionsError(
            f"The video has captions in {len(_usable_manual(info))} languages and names none "
            "as its own, so none was picked (most would be translations).",
            fix="Pass `language` (for example 'en') to choose a track, or use mode='auto' or "
            "mode='whisper' to transcribe the audio.",
        )
    wanted = f" in {language!r}" if language else ""
    return NoCaptionsError(f"No caption track{wanted} for this source.")


def pick_track(info: dict, language: str | None = None) -> CaptionTrack | None:
    """Choose a caption track: requested language first, manual over automatic.

    With no language requested, manual subtitles in the video's language win, then any
    manual track, then the *original* automatic track. Automatic tracks in other languages
    are machine translations and are only used when asked for by name. When the video names
    no language and has several manual tracks, none of them is chosen: one is the original
    and the rest are translations, and nothing says which (``unnamed_language``).
    """
    prefer_json3 = info.get("extractor_key") == "Youtube"
    manual: dict[str, list[dict]] = info.get("subtitles") or {}
    auto: dict[str, list[dict]] = info.get("automatic_captions") or {}
    video_lang = info.get("language")

    if language is not None:
        ordered = [(k, manual[k], False) for k in _candidates(manual, language)]
        ordered += [(k, auto[k], True) for k in _candidates(auto, language)]
    else:
        manual_keys = [] if unnamed_language(info, None) else _candidates(manual, None)
        if video_lang:
            manual_keys.sort(key=lambda k: _base(k) != _base(video_lang))
        ordered = [(k, manual[k], False) for k in manual_keys]
        auto_keys = [k for k in auto if k.endswith("-orig")]
        if video_lang:
            auto_keys += [k for k in _candidates(auto, video_lang) if k not in auto_keys]
        ordered += [(k, auto[k], True) for k in auto_keys]

    for lang, formats, is_auto in ordered:
        track = _track(lang, formats, is_auto, prefer_json3)
        if track is not None:
            return track
    return None


def fetch(
    ydl: Opener, info: dict, language: str | None = None
) -> tuple[list[Segment], CaptionTrack]:
    """Download and parse the chosen track through the ``YoutubeDL`` instance's opener."""
    track = pick_track(info, language)
    if track is None:
        raise no_track_error(info, language)
    if urlsplit(track.url).scheme not in ("http", "https"):
        raise NoCaptionsError(f"Caption track URL is not http(s): {urlsplit(track.url).scheme!r}.")
    try:
        body = ydl.urlopen(track.url).read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise NoCaptionsError(f"Could not fetch the caption track: {exc}") from exc

    try:
        segments = parse_json3(body, words=track.auto) if track.ext == "json3" else parse_vtt(body)
    except (ValueError, KeyError, TypeError, OverflowError) as exc:  # int(Infinity) overflows
        raise NoCaptionsError(f"Caption track could not be parsed ({track.ext}).") from exc
    if not segments:
        raise NoCaptionsError(f"Caption track ({track.language}, {track.ext}) is empty.")
    return segments, track


def _clean(text: str) -> str:
    return " ".join(html.unescape(text).split())


def parse_json3(body: str, *, words: bool) -> list[Segment]:
    """YouTube json3: one segment per event; automatic tracks carry per-word ``tOffsetMs``.

    Display durations overlap the next event, so each segment ends where the next begins.
    Manual tracks put whole lines in a seg, so ``words`` is only set for automatic ones.
    """
    events = [e for e in json.loads(body).get("events", []) if e.get("segs")]
    out: list[Segment] = []
    for i, event in enumerate(events):
        start_ms = int(event.get("tStartMs", 0))
        end_ms = start_ms + int(event.get("dDurationMs", 0))
        if i + 1 < len(events):
            end_ms = min(end_ms, int(events[i + 1].get("tStartMs", end_ms)))
        end_ms = max(end_ms, start_ms)

        segs = [s for s in event["segs"] if s.get("utf8", "").strip()]
        text = _clean("".join(s.get("utf8", "") for s in event["segs"]))
        if not text:
            continue

        seg_words: list[Word] = []
        if words:
            offsets = [start_ms + int(s.get("tOffsetMs", 0)) for s in segs]
            for j, s in enumerate(segs):
                w_end = offsets[j + 1] if j + 1 < len(offsets) else end_ms
                w_start = min(offsets[j], end_ms)
                seg_words.append(
                    Word(
                        start=w_start / 1000, end=max(w_end, w_start) / 1000, text=_clean(s["utf8"])
                    )
                )
        out.append(Segment(start=start_ms / 1000, end=end_ms / 1000, text=text, words=seg_words))
    return out


_TIMING = re.compile(
    r"^(?P<start>(?:\d+:)?\d{2}:\d{2}\.\d{3})\s+-->\s+(?P<end>(?:\d+:)?\d{2}:\d{2}\.\d{3})"
)
_TAG = re.compile(r"<[^>]*>")


def _vtt_seconds(stamp: str) -> float:
    parts = stamp.split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2])
    hours = int(parts[-3]) if len(parts) == 3 else 0
    return hours * 3600 + minutes * 60 + seconds


def parse_vtt(body: str) -> list[Segment]:
    """WebVTT cues → segments, with inline tags stripped and rolling repeats collapsed.

    YouTube's automatic VTT re-shows the previous line above each new one, plus short
    cues repeating it alone. A line equal to the last emitted line is that repeat.
    """
    out: list[Segment] = []
    # Split on truly empty lines only: YouTube cues contain lines that are a single space.
    for block in re.split(r"(?:\r?\n){2,}", body.replace("\ufeff", "")):
        lines = block.strip().splitlines()
        timing_at = next((i for i, line in enumerate(lines) if _TIMING.match(line)), None)
        if timing_at is None:
            continue
        m = _TIMING.match(lines[timing_at])
        start, end = _vtt_seconds(m["start"]), _vtt_seconds(m["end"])
        for raw in lines[timing_at + 1 :]:
            text = _clean(_TAG.sub("", raw))
            if not text or (out and out[-1].text == text):
                continue
            out.append(Segment(start=start, end=max(end, start), text=text))
    return out


def chapters_from_info(info: dict) -> list[Chapter]:
    """``info["chapters"]`` → ``Chapter``s; a missing end runs to the next start or the end."""
    raw = [c for c in info.get("chapters") or [] if c.get("start_time") is not None]
    raw.sort(key=lambda c: float(c["start_time"]))
    duration = info.get("duration")
    out: list[Chapter] = []
    for i, c in enumerate(raw):
        start = float(c["start_time"])
        end = c.get("end_time")
        if end is None:
            end = raw[i + 1]["start_time"] if i + 1 < len(raw) else (duration or start)
        title = _clean(str(c.get("title") or "")) or f"Chapter {i + 1}"
        out.append(Chapter(title=title, start=start, end=max(float(end), start)))
    return out
