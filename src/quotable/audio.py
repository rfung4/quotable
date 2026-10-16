"""Audio sources: local files pass through, URLs go through yt-dlp as a library.

Audio is downloaded as-is (``bestaudio``) with no post-processing, so no ffmpeg binary is
needed; faster-whisper decodes it with PyAV.
"""

from __future__ import annotations

import hashlib
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from quotable.models import DownloadError, SourceError, SourceKind

_CLEAN_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_JS_RUNTIME_HINTS = ("javascript runtime", "challenge solver", "js runtime")
JS_RUNTIME_FIX = (
    "Install Deno (https://deno.com) or Node.js 20+ and make sure it is on PATH, then retry."
)


@dataclass(frozen=True)
class AudioSource:
    kind: SourceKind
    source_key: str
    path: Path
    cite_url: str
    title: str | None = None
    duration: float | None = None


def is_url(source: str) -> bool:
    return urlsplit(source).scheme in ("http", "https")


def local_path(source: str) -> Path | None:
    """The source as an existing local file, or ``None`` when it isn't one."""
    if is_url(source):
        return None
    path = Path(source).expanduser()
    return path.resolve() if path.is_file() else None


def file_key(path: Path) -> str:
    """Content hash, so a moved or renamed recording still hits its cache."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return f"file-{digest.hexdigest()[:16]}"


def url_key(info: dict) -> str:
    """``{extractor_key}-{id}``; ids outside ``[A-Za-z0-9_-]`` are hashed to stay path-safe."""
    extractor = re.sub(r"[^A-Za-z0-9]", "", str(info.get("extractor_key") or "url")) or "url"
    video_id = str(info.get("id") or info.get("webpage_url") or "")
    if not video_id:
        raise SourceError("yt-dlp returned no id for this URL.")
    if not _CLEAN_ID.match(video_id):
        video_id = hashlib.sha256(video_id.encode()).hexdigest()[:16]
    return f"{extractor}-{video_id}"


def fragment_cite(url: str) -> str:
    """``url#t={t}`` with any existing fragment replaced (media fragment syntax)."""
    parts = urlsplit(url)
    return urlunsplit(parts._replace(fragment="")) + "#t={t}"


def file_cite(path: Path) -> str:
    return path.resolve().as_uri() + "#t={t}"


def url_cite(info: dict, url: str) -> str:
    """Citation template for a yt-dlp source; ``{t}`` is whole seconds."""
    if info.get("extractor_key") == "Youtube" and info.get("id"):
        return f"https://youtu.be/{info['id']}?t={{t}}"
    return fragment_cite(str(info.get("webpage_url") or url))


def local_source(path: Path) -> AudioSource:
    return AudioSource(
        kind=SourceKind.FILE,
        source_key=file_key(path),
        path=path,
        cite_url=file_cite(path),
        title=path.name,
    )


@dataclass
class YdlLogger:
    """Sends yt-dlp's warnings and errors to stderr and keeps the warnings.

    The JS-runtime failure surfaces only as a warning; the error raised afterwards is a
    generic "Requested format is not available", so error mapping needs both.
    """

    warnings: list[str] = field(default_factory=list)

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)
        print(msg, file=sys.stderr)

    def error(self, msg: str) -> None:
        print(msg, file=sys.stderr)


def ydl_options(outdir: Path, logger: YdlLogger | None = None) -> dict[str, Any]:
    return {
        "logger": logger or YdlLogger(),
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
        "outtmpl": str(outdir / "audio.%(ext)s"),
        # yt-dlp enables only Deno by default; Node 20+ works for YouTube's JS challenges too.
        "js_runtimes": {"deno": {}, "node": {}},
    }


def map_ytdlp_error(exc: Exception, warnings: Sequence[str] = ()) -> SourceError | DownloadError:
    message = str(exc).removeprefix("ERROR: ").strip() or type(exc).__name__
    lowered = message.lower()
    if "unsupported url" in lowered:
        return SourceError(message, fix="Pass a local file path, a podcast feed or a media URL.")
    context = " ".join([lowered, *warnings]).lower()
    if any(hint in context for hint in _JS_RUNTIME_HINTS):
        return DownloadError(message, fix=JS_RUNTIME_FIX)
    return DownloadError(message)


def _warnings(ydl: Any) -> list[str]:
    logger = getattr(ydl, "params", {}).get("logger")
    return list(getattr(logger, "warnings", []))


def probe(ydl: Any, url: str) -> dict:
    """Metadata only (title, duration, captions, chapters); downloads nothing."""
    try:
        info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise map_ytdlp_error(exc, _warnings(ydl)) from exc
    if not info:
        raise SourceError("yt-dlp returned no metadata for this URL.")
    if info.get("_type") in ("playlist", "multi_video"):
        raise SourceError(
            "Playlists and channels are not supported.",
            fix="Pass the URL of a single item; loop over items yourself if you need several.",
        )
    return info


def download(ydl: Any, info: dict, url: str) -> AudioSource:
    """Download the best audio-only format for an already-probed ``info`` dict."""
    try:
        result = ydl.process_ie_result(info, download=True)
    except Exception as exc:
        raise map_ytdlp_error(exc, _warnings(ydl)) from exc
    downloads = result.get("requested_downloads") or []
    filepath = downloads[0].get("filepath") if downloads else result.get("filepath")
    if not filepath or not Path(filepath).is_file():
        raise DownloadError("yt-dlp finished without producing an audio file.")
    return AudioSource(
        kind=SourceKind.URL,
        source_key=url_key(info),
        path=Path(filepath),
        cite_url=url_cite(info, url),
        title=info.get("title"),
        duration=info.get("duration"),
    )
