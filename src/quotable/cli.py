"""Command line: ``quotable get``, ``info`` and ``serve`` over the same pipeline as the server.

``get`` prints only the transcript on stdout; segment lines stream to stderr as they arrive.
Failures are one stderr line with the fix and exit code 1.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from quotable import __version__, pipeline
from quotable.models import QuotableError, Segment
from quotable.store import render_srt, render_timestamped, render_txt

FORMATS = ("txt", "srt", "json")
MODES = ("auto", "captions", "whisper")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quotable",
        description="Media to cite-ready transcripts: captions first, Whisper next.",
    )
    parser.add_argument("--version", action="version", version=f"quotable {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    get = sub.add_parser("get", help="print a transcript (cached per variant)")
    get.add_argument("source", help="a file path, a podcast feed URL or a media URL")
    get.add_argument("--mode", choices=MODES, default="auto")
    get.add_argument("--lang", help="language code, e.g. en")
    get.add_argument("--diarize", action="store_true", help="label speakers (needs the extra)")
    get.add_argument("--episode", help="feed episode: guid, index or title (default: latest)")
    get.add_argument("--format", choices=FORMATS, default="txt")
    get.add_argument("--force", action="store_true", help="ignore the cache and any partial run")

    info = sub.add_parser("info", help="show a source's title, duration, chapters and episodes")
    info.add_argument("source")
    info.add_argument("--episode")

    sub.add_parser("serve", help="run the MCP server on stdio")
    return parser


def _stream(seg: Segment, done_s: float, total_s: float | None) -> None:
    sys.stderr.write(render_timestamped([seg]))


def _get(args: argparse.Namespace) -> None:
    outcome = pipeline.transcribe(
        args.source,
        mode=args.mode,
        language=args.lang,
        diarize=args.diarize,
        episode=args.episode,
        force=args.force,
        on_segment=_stream,
    )
    transcript = outcome.transcript
    if args.format == "json":
        text = transcript.model_dump_json(indent=2) + "\n"
    elif args.format == "srt":
        text = render_srt(transcript.segments)
    else:
        text = render_txt(transcript.segments)
    sys.stdout.write(text)


def _info(args: argparse.Namespace) -> None:
    info = pipeline.media_info(args.source, episode=args.episode)
    sys.stdout.write(info.model_dump_json(indent=2) + "\n")


def _serve(args: argparse.Namespace) -> None:
    from quotable import server

    server.run()


def _daemon(args: argparse.Namespace) -> None:
    from quotable import daemon

    daemon.run()


_COMMANDS = {"get": _get, "info": _info, "serve": _serve, "daemon": _daemon}


def main(argv: Sequence[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv == ["daemon"]:  # spawned by the shim; off argparse, whose SUPPRESS still lists it
        args = argparse.Namespace(command="daemon")
    else:
        args = _parser().parse_args(argv)
    try:
        _COMMANDS[args.command](args)
    except QuotableError as exc:
        fix = f" Fix: {exc.fix}" if exc.fix else ""
        print(f"quotable: {exc.message}{fix}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"quotable: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("quotable: interrupted; rerun to resume from the last segment.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":  # the shim spawns ``python -m quotable.cli daemon``
    sys.exit(main())
