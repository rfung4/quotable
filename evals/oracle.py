"""The audio cache and the oracle references.

``python -m evals.oracle`` downloads every URL and feed entry's audio once into
``evals/media/cache/{id}/`` (the matrix runs Whisper on that file, so no cell re-downloads),
then transcribes each real-audio entry with Whisper ``large-v3`` on CUDA and stores:

- ``evals/refs/{id}.json``       the oracle transcript (``Transcript`` JSON)
- ``evals/refs/{id}.meta.json``  a sidecar naming the model that made it (``--check`` rejects
  a reference made by any other model: ``turbo`` is distilled from large-v3, so a ``turbo``
  reference would score ``turbo`` at ~0 WER)
- ``evals/refs/{id}.captions.json``  the captions or feed transcript, where the entry has one

The silence clip's reference is empty and the 3 h loop's is the JFK reference repeated, by
construction. TTS entries need no oracle: their reference is exact (``evals.make_tts``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL

from evals.build_corpus import ROOT, _within, inspect_file, load_manifest
from quotable import audio, feeds, pipeline, transcribe
from quotable.config import Settings
from quotable.models import Method, Transcript

CACHE = ROOT / "evals" / "media" / "cache"
INDEX = CACHE / "index.json"
REFS = ROOT / "evals" / "refs"
ORACLE_MODEL = "large-v3"
ORACLE_DEVICE = "cuda"
ORACLE_COMPUTE = "float16"
VARIANT = f"oracle-{ORACLE_MODEL}"
HEARTBEAT_S = 600.0

EMPTY_BY_CONSTRUCTION = {"silence-30s"}
LOOP_OF = {"jfk-loop-3h": "jfk.flac"}
# Shorter than one VAD window: the oracle may legitimately say nothing.
MAY_BE_EMPTY = {"jfk-clip-0.6s"}


def applicable(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in entries if e["kind"] != "tts"]


def wants_captions_ref(entry: dict[str, Any]) -> bool:
    expect = entry["expect"]
    if entry["kind"] == "youtube":
        return expect["captions"] != "none"
    return entry["kind"] == "feed" and bool(expect["transcript_tag"])


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name("partial-" + path.name)
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


# --- audio cache ------------------------------------------------------------------------


def load_index() -> dict[str, dict[str, Any]]:
    if not INDEX.is_file():
        return {}
    return json.loads(INDEX.read_text(encoding="utf-8"))


def cached_path(index: dict[str, dict[str, Any]], entry_id: str) -> Path | None:
    hit = index.get(entry_id)
    if not hit:
        return None
    path = ROOT / hit["path"]
    return path if path.is_file() and path.stat().st_size > 0 else None


def download_entry(entry: dict[str, Any]) -> Path:
    """Fetch one URL or feed entry's audio with the product's own helpers."""
    outdir = CACHE / entry["id"]
    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True)
    if entry["kind"] == "youtube":
        ydl = YoutubeDL(audio.ydl_options(outdir))
        info = audio.probe(ydl, entry["source"])
        path = audio.download(ydl, info, entry["source"]).path
    else:
        episode = feeds.pick_episode(feeds.load_feed(entry["source"]), entry["episode"])
        if episode.guid != entry["episode"]:  # pick_episode falls back to index and title
            raise RuntimeError(f"guid {entry['episode']!r} is no longer in the feed")
        path = feeds.download_enclosure(episode, outdir).path
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("the download produced no file")
    return path


def ensure_audio(entry: dict[str, Any], index: dict[str, dict[str, Any]]) -> Path:
    if entry["kind"] == "file":
        path = ROOT / entry["source"]
        if not path.is_file():
            raise RuntimeError(f"{entry['source']} is missing; run `python -m evals.build_corpus`")
        return path
    existing = cached_path(index, entry["id"])
    if existing is not None:
        return existing
    path = download_entry(entry)
    # Index only after the file is real: a failed download must not look like a cache hit.
    index[entry["id"]] = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size}
    _atomic_write(INDEX, json.dumps(index, indent=1, sort_keys=True))
    return path


# --- references -------------------------------------------------------------------------


def _ref_path(entry_id: str, suffix: str = ".json") -> Path:
    return REFS / f"{entry_id}{suffix}"


def _meta_ok(entry: dict[str, Any]) -> bool:
    path = _ref_path(entry["id"], ".meta.json")
    if not (path.is_file() and _ref_path(entry["id"]).is_file()):
        return False
    meta = json.loads(path.read_text(encoding="utf-8"))
    return meta.get("model") == ORACLE_MODEL or "derived" in meta


def _refuse_cpu() -> None:
    if transcribe._cuda_broken:
        raise RuntimeError("CUDA fell back to CPU; an oracle pass on CPU would take hours")


def run_oracle(entry: dict[str, Any], path: Path) -> None:
    language = entry["expect"]["language"]
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            output_dir=Path(tmp),
            model=ORACLE_MODEL,
            device=ORACLE_DEVICE,
            compute_type=ORACLE_COMPUTE,
        )
        started, last_beat = time.monotonic(), 0.0
        segments = []
        with transcribe.transcribe_iter(path, language=language, settings=settings) as run:
            _refuse_cpu()  # a load-time fallback is known before the first segment
            for seg in run:
                _refuse_cpu()  # a first-segment fallback restarts the run on CPU
                segments.append(seg)
                if seg.end - last_beat >= HEARTBEAT_S:
                    last_beat = seg.end
                    print(f"  {entry['id']}: {seg.end / 60:.0f} min", flush=True)
            detected, duration = run.language, run.duration
        wall = time.monotonic() - started
    _refuse_cpu()
    transcript = Transcript(
        source_key=entry["id"],
        variant=VARIANT,
        method=Method.WHISPER,
        language=detected,
        segments=segments,
        duration=duration,
        title=entry["id"],
    )
    _atomic_write(_ref_path(entry["id"]), transcript.model_dump_json(indent=1))
    meta = {
        "model": ORACLE_MODEL,
        "device": ORACLE_DEVICE,
        "compute_type": ORACLE_COMPUTE,
        "language": language,
        "detected_language": detected,
        "duration_s": duration,
        "segments": len(segments),
        "wall_s": round(wall, 1),
    }
    _atomic_write(_ref_path(entry["id"], ".meta.json"), json.dumps(meta, indent=1))
    print(f"  {entry['id']}: {len(segments)} segments, {wall:.0f} s", flush=True)


def write_empty(entry: dict[str, Any]) -> None:
    transcript = Transcript(
        source_key=entry["id"], variant=VARIANT, method=Method.WHISPER, language="en", segments=[]
    )
    _atomic_write(_ref_path(entry["id"]), transcript.model_dump_json(indent=1))
    meta = {"derived": "empty by construction (generated silence)", "language": "en"}
    _atomic_write(_ref_path(entry["id"], ".meta.json"), json.dumps(meta, indent=1))


def write_loop(entry: dict[str, Any], base_id: str) -> None:
    base = Transcript.model_validate_json(_ref_path(base_id).read_text(encoding="utf-8"))
    period = inspect_file(ROOT / next(e for e in load_manifest() if e["id"] == base_id)["source"])[
        "duration_s"
    ]
    repeats = round(entry["expect"]["duration_s"] / period)
    segments = []
    for k in range(repeats):
        shift = k * period
        for seg in base.segments:
            moved = seg.model_copy(
                update={
                    "start": seg.start + shift,
                    "end": seg.end + shift,
                    "words": [
                        w.model_copy(update={"start": w.start + shift, "end": w.end + shift})
                        for w in seg.words
                    ],
                }
            )
            segments.append(moved)
    transcript = Transcript(
        source_key=entry["id"],
        variant=VARIANT,
        method=Method.WHISPER,
        language=base.language,
        segments=segments,
        duration=repeats * period,
        title=entry["id"],
    )
    _atomic_write(_ref_path(entry["id"]), transcript.model_dump_json())
    meta = {
        "derived": f"{base_id} repeated {repeats} times, offset by {period:.4f} s",
        "derived_from": base_id,
        "language": entry["expect"]["language"],
        "duration_s": repeats * period,
    }
    _atomic_write(_ref_path(entry["id"], ".meta.json"), json.dumps(meta, indent=1))
    print(f"  {entry['id']}: {repeats} repeats, {len(segments)} segments", flush=True)


def write_captions(entry: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            output_dir=Path(tmp),
            model=ORACLE_MODEL,
            device=ORACLE_DEVICE,
            compute_type=ORACLE_COMPUTE,
        )
        outcome = pipeline.transcribe(
            entry["source"],
            mode="captions",
            language=entry["expect"]["language"],
            episode=entry.get("episode"),
            settings=settings,
        )
    transcript = outcome.transcript
    if not transcript.segments:
        raise RuntimeError("the captions transcript came back empty")
    _atomic_write(_ref_path(entry["id"], ".captions.json"), transcript.model_dump_json(indent=1))
    words = sum(len(s.words) for s in transcript.segments)
    print(f"  {entry['id']}: captions, {len(transcript.segments)} segments, {words} words")


def build(entries: list[dict[str, Any]], *, redo: bool) -> int:
    index = load_index()
    failures = 0
    ordered = sorted(entries, key=lambda e: (e["id"] in LOOP_OF, e["kind"] != "file"))
    for entry in ordered:
        eid = entry["id"]
        try:
            if eid in EMPTY_BY_CONSTRUCTION:
                if redo or not _meta_ok(entry):
                    write_empty(entry)
                    print(f"{eid}: empty reference")
                continue
            if eid in LOOP_OF:
                if redo or not _meta_ok(entry):
                    write_loop(entry, LOOP_OF[eid])
                continue
            path = ensure_audio(entry, index)
            if redo or not _meta_ok(entry):
                print(f"{eid}: oracle {ORACLE_MODEL} on {path.name}", flush=True)
                run_oracle(entry, path)
            if wants_captions_ref(entry) and (
                redo or not _ref_path(eid, ".captions.json").is_file()
            ):
                write_captions(entry)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            failures += 1
            print(f"FAIL  {eid}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    return failures


# --- check ------------------------------------------------------------------------------


def _load_ref(path: Path) -> Transcript:
    return Transcript.model_validate_json(path.read_text(encoding="utf-8"))


def check_entry(entry: dict[str, Any], index: dict[str, dict[str, Any]]) -> list[str]:
    eid, problems = entry["id"], []
    if entry["kind"] == "file":
        if not (ROOT / entry["source"]).is_file():
            problems.append(f"audio missing: {entry['source']}")
    elif cached_path(index, eid) is None:
        problems.append("no cached audio (run `python -m evals.oracle`)")

    ref_path, meta_path = _ref_path(eid), _ref_path(eid, ".meta.json")
    if not ref_path.is_file() or not meta_path.is_file():
        return [*problems, "reference or sidecar missing"]
    try:
        ref = _load_ref(ref_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [*problems, f"reference unreadable: {type(exc).__name__}: {exc}"]

    if "derived" in meta:
        if eid in EMPTY_BY_CONSTRUCTION and ref.segments:
            problems.append("the silence reference must be empty")
        if eid in LOOP_OF:
            base_path = _ref_path(LOOP_OF[eid])
            if not base_path.is_file():
                problems.append(f"base reference {LOOP_OF[eid]} missing")
            elif len(ref.segments) % max(len(_load_ref(base_path).segments), 1) != 0:
                problems.append("loop reference is not a whole number of repeats")
            if not ref.segments:
                problems.append("loop reference is empty")
    else:
        if meta.get("model") != ORACLE_MODEL:
            problems.append(f"reference made by {meta.get('model')!r}, not {ORACLE_MODEL}")
        if meta.get("language") != entry["expect"]["language"]:
            problems.append(f"reference language {meta.get('language')!r}, expected the manifest's")
        measured = meta.get("duration_s")
        if measured is None or not _within(float(measured), entry["expect"]["duration_s"]):
            wanted = entry["expect"]["duration_s"]
            problems.append(f"audio duration {measured} s, expected ~{wanted}")
        if not ref.segments and eid not in MAY_BE_EMPTY:
            problems.append("the oracle reference is empty")
    if wants_captions_ref(entry):
        cap = _ref_path(eid, ".captions.json")
        if not cap.is_file():
            problems.append("captions reference missing")
        elif not _load_ref(cap).segments:
            problems.append("captions reference is empty")
    return problems


def check(entries: list[dict[str, Any]]) -> int:
    index = load_index()
    failures = 0
    for entry in entries:
        problems = check_entry(entry, index)
        failures += bool(problems)
        print(f"{'FAIL' if problems else 'OK  '}  {entry['id']:<22} {'; '.join(problems)}")
    print(f"{len(entries)} entries, {failures} failing")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.oracle", description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify cache and references")
    parser.add_argument("--redo", action="store_true", help="regenerate references")
    parser.add_argument("--only", action="append", default=[], help="entry id (repeatable)")
    args = parser.parse_args(argv)

    entries = applicable(load_manifest())
    if args.only:
        entries = [e for e in entries if e["id"] in set(args.only)]
    if args.check:
        return 1 if check(entries) else 0
    failures = build(entries, redo=args.redo)
    print(f"done, {failures} failing")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
