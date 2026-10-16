"""Podcast RSS: sniff, pick an episode, prefer its ``<podcast:transcript>``, else the enclosure.

Everything in a feed is attacker-controlled, including every URL it names. All feed I/O
goes through ``_fetch``/``_stream_to``: http(s) only (before, during and after redirects),
a timeout on every call, and a byte cap on everything (a generous one on audio).
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from quotable import __version__
from quotable.audio import AudioSource, fragment_cite
from quotable.captions import parse_vtt
from quotable.models import (
    DownloadError,
    EpisodeInfo,
    NoCaptionsError,
    Segment,
    SourceError,
    SourceKind,
    Word,
)

TIMEOUT_S = 30
# Some hosts (podcastindex.org) answer 403 to a bare or generic agent and 200 to one that
# says who is asking and where to find them.
USER_AGENT = f"quotable/{__version__} (+https://github.com/rfung4/quotable)"
SNIFF_BYTES = 64 * 1024
FEED_MAX_BYTES = 50 * 1024 * 1024
TRANSCRIPT_MAX_BYTES = 10 * 1024 * 1024
# Far past any audio episode (10 h at 320 kb/s is 1.4 GB); a bound on what a hostile or
# broken enclosure can write to disk.
ENCLOSURE_MAX_BYTES = 8 * 1024**3
_ALLOWED_SCHEMES = ("http", "https")
_NETWORK_ERRORS = (OSError, http.client.HTTPException)

# <podcast:transcript type=…> → our parser, in preference order (JSON > SRT > VTT).
_TRANSCRIPT_TYPES = {
    "application/json": "json",
    "application/x-subrip": "srt",
    "application/srt": "srt",
    "text/srt": "srt",
    "text/vtt": "vtt",
}
_PREFERENCE = ("json", "srt", "vtt")
_AUDIO_SUFFIX = re.compile(r"\.[a-z0-9]{1,5}")


# --- safe fetching --------------------------------------------------------------------


def _check_scheme(url: str) -> None:
    scheme = urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise SourceError(
            f"Refusing a non-http(s) URL from a feed (scheme {scheme or 'none'!r}).",
            fix="Only http and https URLs are fetched; this feed may be malformed or hostile.",
        )


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Refuses redirects off http(s) with our ``SourceError``.

    The stdlib refuses ``file:`` itself but lets ``ftp:`` through, and its refusal is a
    plain ``HTTPError`` that would read as a network miss, so check the target first.
    """

    def http_error_302(self, req, fp, code, msg, headers):
        location = headers.get("location") or headers.get("uri")
        if location:
            _check_scheme(urljoin(req.full_url, location))
        return super().http_error_302(req, fp, code, msg, headers)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_scheme(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _build_opener() -> urllib.request.OpenerDirector:
    """An opener with http(s) handlers only — no file:, ftp: or data: handler exists."""
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler(),
        urllib.request.HTTPHandler(),
        urllib.request.HTTPSHandler(),
        _SafeRedirect(),
        urllib.request.HTTPDefaultErrorHandler(),
        urllib.request.HTTPErrorProcessor(),
    ):
        opener.add_handler(handler)
    return opener


_OPENER = _build_opener()


def _open(url: str):
    _check_scheme(url)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    resp = _OPENER.open(req, timeout=TIMEOUT_S)
    try:
        _check_scheme(resp.geturl())
    except SourceError:
        resp.close()
        raise
    return resp


def _fetch(url: str, max_bytes: int, *, truncate: bool = False) -> bytes:
    """Read at most ``max_bytes``; over the cap is an error unless ``truncate`` (sniffing)."""
    resp = _open(url)
    try:
        body = resp.read(max_bytes + 1)
    finally:
        resp.close()
    if len(body) > max_bytes:
        if truncate:
            return body[:max_bytes]
        raise SourceError(f"Response from {urlsplit(url).netloc} exceeds {max_bytes} bytes.")
    return body


def _stream_to(url: str, path: Path) -> None:
    resp = _open(url)
    written, max_bytes = 0, ENCLOSURE_MAX_BYTES
    try:
        with path.open("wb") as fh:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                written += len(chunk)
                if written > max_bytes:
                    raise SourceError(
                        f"Download from {urlsplit(url).netloc} exceeds {max_bytes} bytes.",
                        fix="That is far larger than any audio episode. Download it yourself and "
                        "pass the file's path.",
                    )
                fh.write(chunk)
    finally:
        resp.close()


# --- feed model -----------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptRef:
    url: str
    kind: str  # json | srt | vtt
    language: str | None = None


@dataclass(frozen=True)
class Episode:
    index: int  # 1 = newest
    title: str
    guid: str | None
    published: str | None
    duration: float | None
    enclosure_url: str | None
    transcripts: tuple[TranscriptRef, ...] = ()
    transcript_types: tuple[str, ...] = ()
    feed_url: str = ""

    def info(self) -> EpisodeInfo:
        return EpisodeInfo(
            guid=self.guid,
            index=self.index,
            title=self.title,
            published=self.published,
            duration=self.duration,
            has_transcript=bool(self.transcripts),
            transcript_types=list(self.transcript_types),
        )


@dataclass(frozen=True)
class Feed:
    url: str
    title: str | None
    episodes: list[Episode] = field(default_factory=list)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _ns(tag: str) -> str:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") else ""


def _child_text(item: ET.Element, name: str, ns_hint: str | None = None) -> str | None:
    for child in item:
        if _local(child.tag) == name and (ns_hint is None or ns_hint in _ns(child.tag)):
            text = (child.text or "").strip()
            if text:
                return text
    return None


def parse_duration(raw: str | None) -> float | None:
    """``itunes:duration``: seconds, ``MM:SS`` or ``HH:MM:SS``."""
    if not raw:
        return None
    try:
        parts = [float(p) for p in raw.strip().split(":")]
    except ValueError:
        return None
    if not 1 <= len(parts) <= 3:
        return None
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def _published(raw: str | None) -> tuple[str | None, float | None]:
    if not raw:
        return None, None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return raw, None
    return dt.isoformat(), dt.timestamp()


def _transcripts(
    item: ET.Element, feed_language: str | None = None
) -> tuple[tuple[TranscriptRef, ...], tuple[str, ...]]:
    """The usable transcript refs, and every declared type (usable or not, deduplicated).

    A transcript without a ``language`` is in the channel's ``<language>`` (podcast namespace).
    """
    refs: list[TranscriptRef] = []
    types: list[str] = []
    for child in item:
        # Both namespace URLs are in the wild: podcastindex.org/namespace/1.0 and the
        # original github.com/Podcastindex-org/... one, so the check is case-insensitive.
        if _local(child.tag) != "transcript" or "podcastindex" not in _ns(child.tag).lower():
            continue
        mime = (child.get("type") or "").split(";")[0].strip().lower()
        if mime and mime not in types:
            types.append(mime)
        kind = _TRANSCRIPT_TYPES.get(mime)
        url = (child.get("url") or "").strip()
        if kind and url:
            language = child.get("language") or feed_language
            refs.append(TranscriptRef(url=url, kind=kind, language=language))
    return tuple(refs), tuple(types)


def _enclosure_url(enclosure: ET.Element | None) -> str | None:
    if enclosure is None:
        return None
    return (enclosure.get("url") or "").strip() or None


def parse_feed(body: bytes, url: str) -> Feed:
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise SourceError(f"Feed is not valid XML: {exc}") from exc
    if _local(root.tag) != "rss" or (channel := root.find("channel")) is None:
        raise SourceError("Not an RSS feed (no <rss><channel>).")

    feed_language = _child_text(channel, "language")
    raw = []
    for order, item in enumerate(channel.findall("item")):
        enclosure = item.find("enclosure")
        published, ts = _published(_child_text(item, "pubDate"))
        transcripts, transcript_types = _transcripts(item, feed_language)
        raw.append(
            (
                ts,
                order,
                dict(
                    title=_child_text(item, "title") or f"Episode {order + 1}",
                    guid=_child_text(item, "guid"),
                    published=published,
                    duration=parse_duration(_child_text(item, "duration", "itunes")),
                    enclosure_url=_enclosure_url(enclosure),
                    transcripts=transcripts,
                    transcript_types=transcript_types,
                ),
            )
        )
    # Newest first by pubDate; undated items keep feed order after the dated ones.
    raw.sort(key=lambda r: (r[0] is None, -(r[0] or 0), r[1]))
    episodes = [
        Episode(index=i + 1, feed_url=url, **fields) for i, (_, _, fields) in enumerate(raw)
    ]
    return Feed(url=url, title=_child_text(channel, "title"), episodes=episodes)


# --- public API -----------------------------------------------------------------------


def is_feed(url: str) -> bool:
    """True when the first 64 KB parse as an XML document whose root is ``<rss>``."""
    if urlsplit(url).scheme.lower() not in _ALLOWED_SCHEMES:
        return False
    try:
        head = _fetch(url, SNIFF_BYTES, truncate=True)
    except (*_NETWORK_ERRORS, SourceError):
        return False
    parser = ET.XMLPullParser(events=("start",))
    try:
        parser.feed(head)
        for _, elem in parser.read_events():
            return _local(elem.tag) == "rss"
    except ET.ParseError:
        return False
    return False


def load_feed(url: str) -> Feed:
    try:
        body = _fetch(url, FEED_MAX_BYTES)
    except _NETWORK_ERRORS as exc:
        raise SourceError(f"Could not fetch the feed: {exc}") from exc
    return parse_feed(body, url)


def list_episodes(url: str) -> list[EpisodeInfo]:
    return [ep.info() for ep in load_feed(url).episodes]


def pick_episode(feed: Feed, episode: str | int | None = None) -> Episode:
    """Default newest. An ``int`` is a 1-based index (1 = newest); a string is tried as an
    exact guid, then as an index, then as a title: an exact title, else the one title that
    contains it (guids may be digits). Words in several titles are an error, not a guess."""
    if not feed.episodes:
        raise SourceError("The feed has no episodes.")
    if episode is None or episode == "":
        return feed.episodes[0]
    if isinstance(episode, int):
        if 1 <= episode <= len(feed.episodes):
            return feed.episodes[episode - 1]
        raise SourceError(
            f"Episode index {episode} is out of range.",
            fix=f"The feed has {len(feed.episodes)} episodes; 1 is the newest.",
        )
    wanted = episode.strip()
    for ep in feed.episodes:
        if ep.guid is not None and ep.guid == wanted:
            return ep
    if wanted.isdigit() and 1 <= int(wanted) <= len(feed.episodes):
        return feed.episodes[int(wanted) - 1]
    lowered = wanted.lower()
    matches = [ep for ep in feed.episodes if lowered in ep.title.lower()]
    exact = [ep for ep in matches if ep.title.strip().lower() == lowered]
    if len(exact) == 1:
        return exact[0]
    if len(matches) == 1:
        return matches[0]
    if matches:
        listed = "; ".join(f"{ep.index}: {ep.title}" for ep in matches[:5])
        more = f" (and {len(matches) - 5} more)" if len(matches) > 5 else ""
        raise SourceError(
            f"{len(matches)} episodes match {wanted!r}.",
            fix=f"Pass a guid, a 1-based index or more of the title. Matches: {listed}{more}",
        )
    recent = "; ".join(f"{ep.index}: {ep.title}" for ep in feed.episodes[:5])
    raise SourceError(
        f"No episode matches {wanted!r}.",
        fix=f"Pass a guid, a 1-based index (1 = newest) or part of a title. Newest: {recent}",
    )


def _base(code: str) -> str:
    return code.split("-")[0].lower()


def choose_transcript(ep: Episode, language: str | None = None) -> TranscriptRef | None:
    """The preferred format, in ``language`` when one is asked for. A transcript declared in
    another language is never one in this language; one that names none may be."""
    refs = ep.transcripts
    if language:
        refs = tuple(r for r in refs if r.language is None or _base(r.language) == _base(language))

    def rank(ref: TranscriptRef) -> tuple[bool, int]:
        return bool(language) and ref.language is None, _PREFERENCE.index(ref.kind)

    return min(refs, key=rank) if refs else None


def fetch_transcript(ep: Episode, language: str | None = None) -> list[Segment] | None:
    """Segments from the episode's own transcript, or ``None`` when it ships none.

    A transcript that can't be fetched or parsed, or is empty, is a miss (``NoCaptionsError``)
    so it can never be cached as a result and auto mode moves on to the enclosure. That
    includes a URL ``_fetch`` refuses (not http(s), oversize), which it refuses before opening.
    """
    ref = choose_transcript(ep, language)
    if ref is None:
        return None
    try:
        body = _fetch(ref.url, TRANSCRIPT_MAX_BYTES).decode("utf-8", errors="replace")
    except (SourceError, *_NETWORK_ERRORS) as exc:
        raise NoCaptionsError(f"Could not fetch the episode transcript: {exc}") from exc
    try:
        segments = {"json": parse_json_transcript, "srt": parse_srt, "vtt": parse_vtt}[ref.kind](
            body
        )
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise NoCaptionsError(f"Episode transcript could not be parsed ({ref.kind}).") from exc
    if not segments:
        raise NoCaptionsError(f"Episode transcript ({ref.kind}) is empty.")
    return segments


def episode_key(ep: Episode) -> str:
    """Guids are only unique within a feed (plenty are "1" or "ep-12"): the feed is in the key."""
    if ep.guid:
        basis = f"{ep.feed_url}\n{ep.guid}"
    elif ep.enclosure_url:
        basis = ep.enclosure_url
    else:
        raise SourceError(f"Episode {ep.title!r} has neither a guid nor an enclosure.")
    return f"rss-{hashlib.sha256(basis.encode()).hexdigest()[:16]}"


def episode_cite(ep: Episode) -> str | None:
    """The enclosure as a citation link, or None: a javascript: or file: URL is no citation."""
    if not ep.enclosure_url or urlsplit(ep.enclosure_url).scheme.lower() not in _ALLOWED_SCHEMES:
        return None
    return fragment_cite(ep.enclosure_url)


def download_enclosure(ep: Episode, outdir: Path) -> AudioSource:
    if not ep.enclosure_url:
        raise SourceError(
            f"Episode {ep.title!r} has no audio enclosure.",
            fix="Pick another episode; this one has nothing to transcribe.",
        )
    suffix = Path(urlsplit(ep.enclosure_url).path).suffix.lower()
    if not _AUDIO_SUFFIX.fullmatch(suffix):
        suffix = ".bin"
    path = outdir / f"audio{suffix}"
    try:
        _stream_to(ep.enclosure_url, path)
    except _NETWORK_ERRORS as exc:
        raise DownloadError(f"Could not download the episode audio: {exc}") from exc
    return AudioSource(
        kind=SourceKind.FEED,
        source_key=episode_key(ep),
        path=path,
        cite_url=episode_cite(ep),
        title=ep.title,
        duration=ep.duration,
    )


# --- transcript formats ---------------------------------------------------------------

_SRT_TIMING = re.compile(
    r"(?P<sh>\d+):(?P<sm>\d{2}):(?P<ss>\d{2})[,.](?P<sms>\d{1,3})\s*-->\s*"
    r"(?P<eh>\d+):(?P<em>\d{2}):(?P<es>\d{2})[,.](?P<ems>\d{1,3})"
)
_TAG = re.compile(r"<[^>]*>")


def _srt_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(body: str) -> list[Segment]:
    out: list[Segment] = []
    for block in re.split(r"(?:\r?\n){2,}", body.replace("﻿", "").strip()):
        lines = block.splitlines()
        timing_at = next((i for i, line in enumerate(lines) if _SRT_TIMING.search(line)), None)
        if timing_at is None:
            continue
        m = _SRT_TIMING.search(lines[timing_at])
        start = _srt_seconds(m["sh"], m["sm"], m["ss"], m["sms"])
        end = _srt_seconds(m["eh"], m["em"], m["es"], m["ems"])
        text = " ".join(_TAG.sub("", " ".join(lines[timing_at + 1 :])).split())
        if text:
            out.append(Segment(start=start, end=max(end, start), text=text))
    return out


_SENTENCE_END = (".", "?", "!", "…")
_WORD_GAP_S = 1.0


def parse_json_transcript(body: str) -> list[Segment]:
    """Podcast namespace JSON (``segments[]`` of ``startTime``/``endTime``/``body``/``speaker``).

    Hosts emit either phrase-level segments or one word per segment. Word-level files are
    regrouped into sentences (split on speaker change, end punctuation or a >1 s gap) with
    the words kept, so they read as text and still cite to the word.
    """
    raw = json.loads(body)["segments"]
    items = []
    for seg in raw:
        text = " ".join(str(seg.get("body") or "").split())
        if not text:
            continue
        start = float(seg["startTime"])
        end = float(seg.get("endTime", start))
        if not (math.isfinite(start) and math.isfinite(end)):  # json.loads takes NaN, Infinity
            raise ValueError(f"non-finite time in segment {text[:40]!r}")
        end = max(end, start)
        items.append((start, end, text, seg.get("speaker") or None))
    if not items:
        return []

    word_level = sorted(len(t.split()) for _, _, t, _ in items)[len(items) // 2] <= 1
    if not word_level:
        return [Segment(start=s, end=e, text=t, speaker=sp) for s, e, t, sp in items]

    out: list[Segment] = []
    words: list[Word] = []

    def flush() -> None:
        if words:
            out.append(
                Segment(
                    start=words[0].start,
                    end=words[-1].end,
                    text=" ".join(w.text for w in words),
                    speaker=words[0].speaker,
                    words=list(words),
                )
            )
            words.clear()

    for start, end, text, speaker in items:
        if words and (speaker != words[-1].speaker or start - words[-1].end > _WORD_GAP_S):
            flush()
        words.append(Word(start=start, end=end, text=text, speaker=speaker))
        if text.endswith(_SENTENCE_END):
            flush()
    flush()
    return out
