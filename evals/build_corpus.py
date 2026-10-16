"""The evaluation corpus: generate the local media, check everything the manifest promises.

``python -m evals.build_corpus`` generates the missing local files with ffmpeg, then validates;
``--validate`` only validates. Validation touches the network (YouTube metadata, feeds) by
design: this is a script, not a test, and the manifest's expectations are about the live world.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import av
from yt_dlp import YoutubeDL

from quotable import audio, captions, feeds

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).resolve().parent / "corpus.json"
SRC_PLACEHOLDER = "{src}"
FIXTURE = "tests/fixtures/jfk.flac"

KINDS = ("file", "youtube", "feed", "tts")
TIERS = ("short", "long")
CAPTION_KINDS = ("manual", "asr", "none")
ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
EXPECT_KEYS = ("captions", "transcript_tag", "language", "speakers", "duration_s", "chapters")
ENTRY_KEYS = ("id", "kind", "source", "license", "expect", "tier", "diarize", "readme_row", "note")


def load_manifest(path: Path = MANIFEST) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["entries"]


def schema_problems(entries: list[dict[str, Any]]) -> list[str]:
    """Everything wrong with the manifest that needs no file and no network."""
    problems: list[str] = []
    seen: set[str] = set()
    for i, e in enumerate(entries):
        eid = e.get("id", f"#{i}")
        missing = [k for k in ENTRY_KEYS if k not in e]
        if missing:
            problems.append(f"{eid}: missing keys {missing}")
            continue
        if not ID_RE.match(eid):
            problems.append(f"{eid}: id must match {ID_RE.pattern}")
        if eid in seen:
            problems.append(f"{eid}: duplicate id")
        seen.add(eid)
        if e["kind"] not in KINDS:
            problems.append(f"{eid}: kind {e['kind']!r} not in {KINDS}")
        if e["tier"] not in TIERS:
            problems.append(f"{eid}: tier {e['tier']!r} not in {TIERS}")
        lic = e["license"]
        if not (isinstance(lic, dict) and lic.get("name") and lic.get("source_url")):
            problems.append(f"{eid}: license needs a name and a source_url")
        expect = e["expect"]
        gaps = [k for k in EXPECT_KEYS if k not in expect]
        if gaps:
            problems.append(f"{eid}: expect is missing {gaps}")
            continue
        if expect["captions"] not in CAPTION_KINDS:
            problems.append(f"{eid}: expect.captions {expect['captions']!r} not in {CAPTION_KINDS}")
        if not expect["duration_s"] or expect["duration_s"] <= 0:
            problems.append(f"{eid}: expect.duration_s must be positive")
        if e["kind"] == "feed" and not e.get("episode"):
            problems.append(f"{eid}: a feed entry pins its episode by guid")
        if e["kind"] == "youtube" and not str(e["source"]).startswith("https://"):
            problems.append(f"{eid}: a youtube source is an https URL")
        if e["kind"] == "tts" and not e.get("script"):
            problems.append(f"{eid}: a tts entry names its script")
        if e.get("build") and e["kind"] != "file":
            problems.append(f"{eid}: only file entries are built")
    return problems


def _within(measured: float, expected: float, tolerance: float = 0.10) -> bool:
    return abs(measured - expected) <= max(tolerance * expected, 0.3)


def _ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError("ffmpeg is not on PATH; it is an eval-only dependency (evals/README.md)")
    return found


def build_file(entry: dict[str, Any]) -> str:
    """Generate one local file; returns ``built``, ``present`` or raises."""
    path = ROOT / entry["source"]
    if path.is_file() and path.stat().st_size > 0:
        return "present"
    src = ROOT / entry["build"].get("src", FIXTURE)
    args = [str(src) if a == SRC_PLACEHOLDER else a for a in entry["build"]["args"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the target and rename: a half-written file must never look like a result.
    partial = path.with_name("partial-" + path.name)
    cmd = [_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args, str(partial)]
    done = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if done.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed: {done.stderr.strip()[-300:]}")
    os.replace(partial, path)
    return "built"


def inspect_file(path: Path) -> dict[str, Any]:
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("no audio stream")
        duration = container.duration / av.time_base if container.duration else None
        return {
            "sample_rate": stream.rate,
            "channels": stream.channels,
            "video": any(s.type == "video" for s in container.streams),
            "duration_s": duration,
        }


def check_file(entry: dict[str, Any]) -> list[str]:
    path = ROOT / entry["source"]
    if not path.is_file():
        return [f"file missing: {entry['source']} (run `python -m evals.build_corpus`)"]
    try:
        got = inspect_file(path)
    except Exception as exc:
        return [f"unreadable: {type(exc).__name__}: {exc}"]
    expect = entry["expect"]
    problems = []
    for key in ("sample_rate", "channels", "video"):
        if key in expect and got[key] != expect[key]:
            problems.append(f"{key} is {got[key]}, expected {expect[key]}")
    if got["duration_s"] is None or not _within(got["duration_s"], expect["duration_s"]):
        problems.append(f"duration {got['duration_s']} s, expected ~{expect['duration_s']} s")
    return problems


def check_youtube(entry: dict[str, Any]) -> list[str]:
    expect = entry["expect"]
    with tempfile.TemporaryDirectory() as tmp:
        ydl = YoutubeDL(audio.ydl_options(Path(tmp)))
        try:
            info = audio.probe(ydl, entry["source"])
        except Exception as exc:
            return [f"probe failed: {type(exc).__name__}: {exc}"]
    problems = []
    # The pipeline's own choice, not "subtitles is non-empty": a livestream carries only
    # live_chat there, which pick_track skips.
    track = captions.pick_track(info)
    if expect.get("default_captions") == "refused":
        # Nothing names the video's language, so the default choice must be no track; the
        # kind shipped is what asking for the language gets.
        if track is not None:
            problems.append(f"default captions track {track.language!r} chosen, expected none")
        track = captions.pick_track(info, expect["language"])
    kind = "none" if track is None else ("asr" if track.auto else "manual")
    if kind != expect["captions"]:
        problems.append(f"captions are {kind}, expected {expect['captions']}")
    duration = info.get("duration")
    if not duration or not _within(float(duration), expect["duration_s"]):
        problems.append(f"duration {duration} s, expected ~{expect['duration_s']} s")
    has_chapters = bool(info.get("chapters"))
    if has_chapters != expect["chapters"]:
        problems.append(f"chapters present is {has_chapters}, expected {expect['chapters']}")
    language = info.get("language")
    if language and language.split("-")[0].lower() != expect["language"]:
        problems.append(f"language is {language}, expected {expect['language']}")
    return problems


def check_feed(entry: dict[str, Any]) -> list[str]:
    expect = entry["expect"]
    try:
        feed = feeds.load_feed(entry["source"])
        ep = feeds.pick_episode(feed, entry["episode"])
    except Exception as exc:
        return [f"feed failed: {type(exc).__name__}: {exc}"]
    problems = []
    # pick_episode falls back to an index and then a title substring; a guid that dropped
    # off the feed must not silently become some other episode.
    if ep.guid != entry["episode"]:
        problems.append(f"guid {entry['episode']!r} is not in the feed (picked {ep.guid!r})")
    has_tag = bool(ep.transcripts)
    if has_tag != expect["transcript_tag"]:
        problems.append(f"usable transcript tag is {has_tag}, expected {expect['transcript_tag']}")
    if ep.duration is None:
        problems.append("the feed gives this episode no duration")
    else:
        # The feed's declared length can differ from the audio's (NPR inserts ads), so the
        # manifest may carry both: duration_s is the audio, feed_duration_s the feed's claim.
        declared = expect.get("feed_duration_s", expect["duration_s"])
        if not _within(ep.duration, declared):
            problems.append(f"declared duration {ep.duration} s, expected ~{declared} s")
    if not ep.enclosure_url:
        problems.append("no enclosure URL")
    return problems


def check_tts(entry: dict[str, Any]) -> tuple[str, list[str]]:
    """Absent until Task 2 has run; present means it must be a readable file of the right length."""
    if not (ROOT / entry["source"]).is_file():
        return "absent", []
    return "ok", check_file(entry)


def validate(entries: list[dict[str, Any]], *, only: set[str] | None = None) -> int:
    failures = 0
    for problem in schema_problems(entries):
        print(f"FAIL  manifest   {problem}")
        failures += 1
    if failures:
        return failures
    for entry in entries:
        if only and entry["id"] not in only:
            continue
        kind = entry["kind"]
        status = "ok"
        if kind == "file":
            problems = check_file(entry)
        elif kind == "youtube":
            problems = check_youtube(entry)
        elif kind == "feed":
            problems = check_feed(entry)
        else:
            status, problems = check_tts(entry)
        if problems:
            failures += 1
            print(f"FAIL  {entry['id']:<22} {'; '.join(problems)}")
        else:
            print(f"{status.upper():<5} {entry['id']:<22} {kind}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.build_corpus", description=__doc__)
    parser.add_argument("--validate", action="store_true", help="validate only, build nothing")
    parser.add_argument("--only", action="append", default=[], help="entry id (repeatable)")
    args = parser.parse_args(argv)

    entries = load_manifest()
    only = set(args.only) or None
    if not args.validate:
        problems = schema_problems(entries)
        for problem in problems:
            print(f"FAIL  manifest   {problem}")
        if problems:
            return 1
        for entry in entries:
            if entry["kind"] != "file" or "build" not in entry:
                continue
            if only and entry["id"] not in only:
                continue
            try:
                print(f"{build_file(entry).upper():<7} {entry['id']}")
            except RuntimeError as exc:
                print(f"FAIL    {entry['id']}: {exc}", file=sys.stderr)
                return 1
    failures = validate(entries, only=only)
    print(f"{len(entries)} entries, {failures} failing")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
