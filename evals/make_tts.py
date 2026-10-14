"""TTS dialogues with exact references: scripts in, WAV and turn-timed reference out.

``python -m evals.make_tts`` calls OpenAI's speech endpoint once per turn (cached under
``evals/media/tts/turns/``, so a re-run never re-bills), concatenates the turns with known
gaps and writes ``evals/media/tts/{id}.wav`` and ``evals/refs/{id}.json``. The reference
times come from the measured turn lengths, so they are exact.

``OPENAI_API_KEY`` is read from the environment, sent only in the Authorization header to
api.openai.com, and never written to a file, a log line or an error message.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from evals.build_corpus import ROOT, inspect_file, load_manifest

API_URL = "https://api.openai.com/v1/audio/speech"
MODEL = "gpt-4o-mini-tts"
TURN_CACHE = ROOT / "evals" / "media" / "tts" / "turns"
REFS = ROOT / "evals" / "refs"
MAX_RESPONSE = 50_000_000
RETRY_CODES = {429, 500, 502, 503, 504}
ATTEMPTS = 6
WORKERS = 4

# Redaction is deliberately loose (a 401 body echoes a masked fragment such as "sk-proj-****abcd");
# the leak scan is strict (twenty key-alphabet characters, which no audio sample run matches).
REDACT_RE = re.compile(r"sk-[A-Za-z0-9_*.-]+")
LEAK_RE = re.compile(rb"sk-[A-Za-z0-9_-]{20,}")
DURATION_TOLERANCE_S = 0.05


class TtsError(RuntimeError):
    pass


def redact(text: str, key: str | None = None) -> str:
    if key:
        text = text.replace(key, "[redacted]")
    return REDACT_RE.sub("[redacted]", text)


def synthesise(text: str, voice: str, key: str) -> bytes:
    body = json.dumps(
        {"model": MODEL, "input": text, "voice": voice, "response_format": "wav"}
    ).encode()
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    # Long paragraphs take longer to generate; the endpoint also stalls now and then.
    timeout = 25 + len(text) / 10
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return response.read(MAX_RESPONSE)
        except urllib.error.HTTPError as exc:
            detail = redact(exc.read(2000).decode("utf-8", "replace"), key).strip()
            if exc.code in RETRY_CODES and attempt < ATTEMPTS:
                time.sleep(2 * attempt)
                continue
            raise TtsError(f"OpenAI TTS returned HTTP {exc.code}: {detail[:300]}") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < ATTEMPTS:
                time.sleep(2 * attempt)
                continue
            raise TtsError(f"Could not reach api.openai.com: {redact(str(exc), key)}") from None
    raise TtsError("OpenAI TTS: no response")  # unreachable; keeps the type checker honest


def parse_wav(data: bytes) -> tuple[int, int, int, bytes]:
    """(rate, channels, sample_width_bytes, pcm). Tolerates a streamed header whose size
    fields are 0 or 0xFFFFFFFF, which the speech endpoint may send."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise TtsError("The speech endpoint did not return a WAV file.")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        tag, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = pos + 8
        if tag == b"fmt ":
            _, channels, rate, _, _, bits = struct.unpack("<HHIIHH", data[body : body + 16])
            fmt = (rate, channels, bits // 8)
        elif tag == b"data":
            end = len(data) if size in (0, 0xFFFFFFFF) or body + size > len(data) else body + size
            pcm = data[body:end]
            break
        pos = body + size + (size & 1)
    if fmt is None or pcm is None:
        raise TtsError("The WAV from the speech endpoint has no fmt or data chunk.")
    return (*fmt, pcm)


def turn_audio(script_id: str, index: int, voice: str, text: str, key: str) -> tuple[bytes, bool]:
    digest = hashlib.sha1(f"{MODEL}\n{voice}\n{text}".encode()).hexdigest()[:12]
    path = TURN_CACHE / f"{script_id}-{index:03d}-{digest}.wav"
    if path.is_file() and path.stat().st_size > 0:
        return path.read_bytes(), True
    data = synthesise(text, voice, key)
    parse_wav(data)  # never cache something that is not audio
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name("partial-" + path.name)
    partial.write_bytes(data)
    os.replace(partial, path)
    return data, False


def _write_atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name("partial-" + path.name)
    write(partial)
    os.replace(partial, path)


def build(entry: dict[str, Any], key: str) -> None:
    script = json.loads((ROOT / entry["script"]).read_text(encoding="utf-8"))
    speakers = script["speakers"]
    gap = float(script.get("gap_s", 0.4))
    params: tuple[int, int, int] | None = None
    pcm = bytearray()
    turns: list[dict[str, Any]] = []
    total = len(script["turns"])

    def fetch(item: tuple[int, dict[str, Any]]) -> tuple[bytes, bool]:
        i, turn = item
        got = turn_audio(script["id"], i, speakers[turn["speaker"]]["voice"], turn["text"], key)
        print(f"{script['id']} {i + 1}/{total} {'cached' if got[1] else 'synthesised'}", flush=True)
        return got

    # Requests overlap (the endpoint stalls now and then, and one at a time waits on each
    # stall); assembly below stays in order, so the output does not depend on completion order.
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        fetched = list(pool.map(fetch, enumerate(script["turns"])))
    for i, turn in enumerate(script["turns"]):
        data, _ = fetched[i]
        rate, channels, width, samples = parse_wav(data)
        if params is None:
            params = (rate, channels, width)
        elif params != (rate, channels, width):
            raise TtsError(f"{script['id']} turn {i}: audio format changed mid-script")
        frame = channels * width
        start = len(pcm) / frame / rate
        pcm += samples
        end = len(pcm) / frame / rate
        turns.append(
            {
                "speaker": turn["speaker"],
                "text": turn["text"],
                "start": round(start, 3),
                "end": round(end, 3),
            }
        )
        if i < total - 1:
            silence = float(turn.get("pause_after_s", gap))
            pcm += b"\x00" * (round(silence * rate) * frame)
    assert params is not None
    rate, channels, width = params
    duration = len(pcm) / (channels * width) / rate

    def write_wav(path: Path) -> None:
        with wave.open(str(path), "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(width)
            out.setframerate(rate)
            out.writeframes(bytes(pcm))

    _write_atomic(ROOT / entry["source"], write_wav)
    reference = {"id": script["id"], "duration_s": round(duration, 3), "turns": turns}
    _write_atomic(
        REFS / f"{entry['id']}.json",
        lambda p: p.write_text(json.dumps(reference, indent=1, ensure_ascii=False), "utf-8"),
    )
    print(f"{entry['id']}: {len(turns)} turns, {duration:.1f} s", flush=True)


def leaks(key: str | None) -> list[str]:
    found = []
    for base in (ROOT / "evals" / "media" / "tts", REFS):
        for path in sorted(p for p in base.rglob("*") if p.is_file()) if base.is_dir() else []:
            blob = path.read_bytes()
            if LEAK_RE.search(blob) or (key and key.encode() in blob):
                found.append(str(path.relative_to(ROOT)))
    return found


def check_entry(entry: dict[str, Any]) -> list[str]:
    problems = []
    script_path, media, ref_path = ROOT / entry["script"], ROOT / entry["source"], None
    ref_path = REFS / f"{entry['id']}.json"
    for label, path in (("script", script_path), ("media", media), ("reference", ref_path)):
        if not path.is_file():
            problems.append(f"{label} missing: {path.relative_to(ROOT)}")
    if problems:
        return problems
    script = json.loads(script_path.read_text(encoding="utf-8"))
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    turns = ref["turns"]
    if [t["text"] for t in turns] != [t["text"] for t in script["turns"]]:
        problems.append("reference turns differ from the script (re-run make_tts)")
    if [t["speaker"] for t in turns] != [t["speaker"] for t in script["turns"]]:
        problems.append("reference speakers differ from the script")
    labels = {t["speaker"] for t in turns}
    wanted = entry["expect"]["speakers"]
    if not (len(labels) == len(script["speakers"]) == wanted):
        problems.append(
            f"speakers: {len(labels)} in the reference, {len(script['speakers'])} in the script,"
            f" {wanted} in the manifest"
        )
    previous_end = 0.0
    for i, t in enumerate(turns):
        if t["end"] <= t["start"] or t["start"] < previous_end:
            problems.append(f"turn {i} is empty or overlaps the previous one")
            break
        previous_end = t["end"]
    try:
        measured = inspect_file(media)["duration_s"]
    except Exception as exc:
        problems.append(f"media unreadable: {type(exc).__name__}: {exc}")
        return problems
    if measured is None or abs(measured - turns[-1]["end"]) > DURATION_TOLERANCE_S:
        problems.append(
            f"media is {measured} s, the reference ends at {turns[-1]['end']} s"
            f" (tolerance {DURATION_TOLERANCE_S} s)"
        )
    return problems


def tts_entries(only: set[str] | None = None) -> list[dict[str, Any]]:
    return [e for e in load_manifest() if e["kind"] == "tts" and (not only or e["id"] in only)]


def check(only: set[str] | None = None) -> int:
    failures = 0
    for entry in tts_entries(only):
        problems = check_entry(entry)
        failures += bool(problems)
        status = "FAIL" if problems else "OK  "
        print(f"{status}  {entry['id']:<14} {'; '.join(problems)}")
    for path in leaks(os.environ.get("OPENAI_API_KEY")):
        failures += 1
        print(f"FAIL  key-shaped string found in {path}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.make_tts", description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify what exists; no network")
    parser.add_argument("--only", action="append", default=[], help="script id (repeatable)")
    args = parser.parse_args(argv)
    only = set(args.only) or None

    if args.check:
        failures = check(only)
        print(f"{failures} failing")
        return 1 if failures else 0

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print(
            "make_tts: OPENAI_API_KEY is not set. Export it in the environment;"
            " it is never written to a file.",
            file=sys.stderr,
        )
        return 1
    try:
        for entry in tts_entries(only):
            build(entry, key)
    except TtsError as exc:
        print(f"make_tts: {redact(str(exc), key)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
