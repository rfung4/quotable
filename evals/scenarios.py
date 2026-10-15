"""Long-run scenarios: straight run, cancel and resume, paging, memory, the feed transcript path.

``python -m evals.scenarios`` runs them on the GPU (about half an hour) and writes
``evals/results/scenarios.json`` with a ``pass`` and the measured numbers per scenario;
``--check`` replays that file. Run it on an idle machine: it shares the card with nothing.

On each long entry (Lex, cached audio; the 3 h loop), ``turbo`` · ``cuda``, each run in its
own output directory so no scenario is a cache hit of another:

- ``straight`` — one uninterrupted run, the baseline for the others
- ``memory`` — RSS and VRAM sampled every 30 s during the straight run; no growth above 10 %
  from the first sample after the first segment (audio decoded, VAD done, model warm) to the
  end. The run takes ~5 min, so the plan's "minute 5" would compare neighbouring samples
- ``cancel_resume`` — cancel once 600 s of audio are done; the partial file keeps segments,
  the variant lock is free, the next call resumes at or after 600 s, and the text past 600 s
  matches the straight run within a provisional WER of 0.05 (a resume decodes VAD clips,
  so some drift is by construction); words lost or doubled near the boundary are counted
- ``paging`` — offset pages concatenate to the full text exactly; 600 s window pages
  concatenate to the overlapping segments exactly, every segment is in some window, and
  only segments straddling a boundary are in two

And ``feed_transcript``: the Darknet 95 min episode, live feed and guid, ``mode=auto`` with
the enclosure download made to raise: the transcript-tag path must be taken.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from filelock import Timeout

from evals import oracle, score
from evals.build_corpus import ROOT, load_manifest
from evals.run import NVIDIA_SMI, RESULTS
from quotable import feeds, pipeline, transcribe
from quotable.config import Settings
from quotable.models import CancelledError, Method, Transcript
from quotable.store import Store, render_txt

LONG_ENTRIES = ("lex-497", "jfk-loop-3h")
FEED_ENTRY = "darknet-179"
CANCEL_AT_S = 600.0
WINDOW_S = 600.0
WINDOW_PAGE_CHARS = 2000  # small, so a window spans several pages and offsets are exercised
SAMPLE_S = 30.0
GROWTH_MAX = 0.10
RESUME_WER_MAX = 0.05  # provisional (Decision 10)
BOUNDARY_S = 30.0


def settings(output_dir: Path) -> Settings:
    return Settings(output_dir=output_dir, model="turbo", device="cuda", compute_type="float16")


def _source(entry: dict[str, Any]) -> str:
    if entry["kind"] == "file":
        return str(ROOT / entry["source"])
    path = oracle.cached_path(oracle.load_index(), entry["id"])
    if path is None:
        raise RuntimeError(f"no cached audio for {entry['id']}; run `python -m evals.oracle`")
    return str(path)


class MemorySampler:
    """RSS (from /proc) and this pid's VRAM (from nvidia-smi) every ``period`` seconds."""

    def __init__(self, period: float = SAMPLE_S) -> None:
        self.period = period
        self.samples: list[dict[str, float | None]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._t0 = 0.0

    def __enter__(self) -> MemorySampler:
        self._t0 = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join()
        self.sample()

    def _loop(self) -> None:
        while not self._stop.wait(self.period):
            self.sample()

    def sample(self) -> None:
        rss = None
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1024
        vram = None
        try:
            out = subprocess.run(
                NVIDIA_SMI, capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL
            ).stdout
            for line in out.splitlines():
                pid, _, used = (part.strip() for part in line.partition(","))
                if pid == str(os.getpid()) and used.isdigit():
                    vram = float(used)
        except (OSError, subprocess.SubprocessError):
            pass
        t = round(time.monotonic() - self._t0, 1)
        self.samples.append({"t_s": t, "rss_mb": rss, "vram_mb": vram})


def growth(samples: list[dict[str, Any]], key: str, baseline_s: float) -> float | None:
    """Largest rise over the first sample at or after ``baseline_s``, as a fraction."""
    after = [s[key] for s in samples if s["t_s"] >= baseline_s and s[key] is not None]
    if len(after) < 2:
        return None
    return round(max(after[1:]) / after[0] - 1, 4)


# --- scenarios --------------------------------------------------------------------------


def straight(source: str, out: Path) -> tuple[dict[str, Any], dict[str, Any], Transcript]:
    transcribe.get_model(settings(out))  # loaded before sampling, so load is not "growth"
    first: list[float] = []
    with MemorySampler() as sampler:
        started = time.monotonic()

        def mark_first(seg: Any, done_s: float, total_s: float | None) -> None:
            if not first:
                first.append(time.monotonic() - started)

        outcome = pipeline.transcribe(
            source, mode="whisper", force=True, on_segment=mark_first, settings=settings(out)
        )
        wall = time.monotonic() - started
    full = outcome.transcript
    words = sum(len(s.text.split()) for s in full.segments)
    run = {
        "pass": bool(full.segments) and not transcribe._cuda_broken,
        "wall_s": round(wall, 1),
        "audio_s": full.duration,
        "realtime": round((full.duration or 0) / wall, 1),
        "segments": len(full.segments),
        "words": words,
        "cuda_fallback": transcribe._cuda_broken,
    }
    baseline = first[0] if first else wall
    rss = growth(sampler.samples, "rss_mb", baseline)
    vram = growth(sampler.samples, "vram_mb", baseline)
    memory = {
        "pass": rss is not None and rss <= GROWTH_MAX and (vram is None or vram <= GROWTH_MAX),
        "rss_growth": rss,
        "vram_growth": vram,
        "baseline_after_s": round(baseline, 1),
        "samples": sampler.samples,
    }
    return run, memory, full


def _boundary_edits(full: Transcript, resumed: Transcript, point: float) -> dict[str, Any]:
    """WER past the cancel point, and edits whose word sits within 30 s of the resume point."""
    ref = score.side([s.model_dump() for s in full.segments if s.start >= CANCEL_AT_S])
    hyp = score.side([s.model_dump() for s in resumed.segments if s.start >= CANCEL_AT_S])
    a, b = score._encode(ref.norm, hyp.norm)
    pairs = score.align(a, b)
    edits = sum(1 for i, j in pairs if i is None or j is None or a[i] != b[j])

    def near(t: float | None) -> bool:
        return t is not None and abs(t - point) <= BOUNDARY_S

    lost = sum(1 for i, j in pairs if j is None and near(ref.start[i]))
    doubled = sum(1 for i, j in pairs if i is None and near(hyp.start[j]))
    return {
        "wer_past_cancel": round(edits / len(ref.norm), 4) if ref.norm else None,
        "words_lost_near_boundary": lost,
        "words_doubled_near_boundary": doubled,
    }


def cancel_resume(source: str, full: Transcript, out: Path) -> dict[str, Any]:
    store = Store(out)
    cancel = threading.Event()

    def stop_after(seg: Any, done_s: float, total_s: float | None) -> None:
        if done_s >= CANCEL_AT_S:
            cancel.set()

    cancelled = False
    try:
        pipeline.transcribe(
            source,
            mode="whisper",
            force=True,
            on_segment=stop_after,
            cancel=cancel,
            settings=settings(out),
        )
    except CancelledError:
        cancelled = True
    key, variant = full.source_key, full.variant
    partial = store.partial_segments(key, variant)
    point = store.resume_point(key, variant)
    lock = store.variant_lock(key, variant)  # a second instance: refused while held
    try:
        lock.acquire(timeout=0)
        lock.release()
        lock_free = True
    except Timeout:
        lock_free = False

    firsts: list[float] = []

    def first_start(seg: Any, done_s: float, total_s: float | None) -> None:
        if not firsts:
            firsts.append(seg.start)

    resumed = pipeline.transcribe(
        source, mode="whisper", on_segment=first_start, settings=settings(out)
    ).transcript
    found = {
        "cancelled": cancelled,
        "partial_segments": len(partial),
        "resume_point_s": round(point, 3),
        "lock_free": lock_free,
        "first_resumed_start_s": round(firsts[0], 3) if firsts else None,
        **_boundary_edits(full, resumed, point),
    }
    found["pass"] = (
        cancelled
        and len(partial) > 0
        and lock_free
        and found["first_resumed_start_s"] is not None
        and found["first_resumed_start_s"] >= CANCEL_AT_S
        and found["wer_past_cancel"] is not None
        and found["wer_past_cancel"] <= RESUME_WER_MAX
        and not transcribe._cuda_broken
    )
    return found


def paging(store: Store, full: Transcript) -> dict[str, Any]:
    tid = full.id
    pages, text, offset = 0, "", 0
    while True:
        page = store.page(tid, offset)
        pages, text = pages + 1, text + page.text
        if page.next_offset is None:
            break
        offset = page.next_offset
    offset_ok = text == render_txt(full.segments)

    end = max(full.duration or 0.0, full.segments[-1].end if full.segments else 0.0)
    windows = [(k * WINDOW_S, (k + 1) * WINDOW_S) for k in range(math.ceil(end / WINDOW_S))]
    window_mismatches, window_pages = [], 0
    for lo, hi in windows:
        joined, offset = "", 0
        while True:
            page = store.page(tid, offset, limit=WINDOW_PAGE_CHARS, start_s=lo, end_s=hi)
            window_pages, joined = window_pages + 1, joined + page.text
            if page.next_offset is None:
                break
            offset = page.next_offset
        expected = render_txt([s for s in full.segments if s.end > lo and s.start < hi])
        if joined != expected:
            window_mismatches.append(lo)

    inside = [sum(1 for lo, hi in windows if s.end > lo and s.start < hi) for s in full.segments]
    orphans = [k for k, n in enumerate(inside) if n == 0]
    doubled = {k for k, n in enumerate(inside) if n >= 2}
    bounds = [hi for _, hi in windows[:-1]]
    straddling = {
        k for k, s in enumerate(full.segments) if any(s.start < b < s.end for b in bounds)
    }
    return {
        "pass": offset_ok and not window_mismatches and not orphans and doubled == straddling,
        "offset_pages": pages,
        "offset_text_exact": offset_ok,
        "windows": len(windows),
        "window_pages": window_pages,
        "window_mismatches_at_s": window_mismatches,
        "segments_in_no_window": len(orphans),
        "segments_in_two_windows": len(doubled),
        "segments_straddling_a_boundary": len(straddling),
    }


def long_entry(entry: dict[str, Any], work: Path) -> dict[str, Any]:
    source = _source(entry)
    print(f"{entry['id']}: straight run", flush=True)
    run, memory, full = straight(source, work / "straight")
    print(f"{entry['id']}: paging", flush=True)
    pages = paging(Store(work / "straight"), full)
    print(f"{entry['id']}: cancel at {CANCEL_AT_S:.0f} s and resume", flush=True)
    resume = cancel_resume(source, full, work / "resume")
    return {"straight": run, "memory": memory, "paging": pages, "cancel_resume": resume}


def feed_transcript(entry: dict[str, Any], work: Path) -> dict[str, Any]:
    calls: list[str] = []
    real = feeds.download_enclosure

    def refuse(*args: Any, **kwargs: Any) -> Any:
        calls.append("download_enclosure")
        raise RuntimeError("the transcript-tag path must not download the enclosure")

    feeds.download_enclosure = refuse
    try:
        outcome = pipeline.transcribe(
            entry["source"], mode="auto", episode=entry["episode"], settings=settings(work)
        )
    finally:
        feeds.download_enclosure = real
    t = outcome.transcript
    return {
        "pass": t.method == Method.CAPTIONS and not calls and bool(t.segments),
        "method": t.method.value,
        "enclosure_downloads": len(calls),
        "segments": len(t.segments),
        "words": sum(len(s.text.split()) for s in t.segments),
    }


# --- driver -----------------------------------------------------------------------------

SCENARIOS = ("straight", "memory", "paging", "cancel_resume")


def run_all(only: set[str], out: Path) -> int:
    entries = {e["id"]: e for e in load_manifest()}
    path = out / "scenarios.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    with tempfile.TemporaryDirectory(prefix="scenarios-", dir=out) as tmp:
        for eid in LONG_ENTRIES:
            if only and eid not in only:
                continue
            try:
                results[eid] = long_entry(entries[eid], Path(tmp) / eid)
            except Exception as exc:
                results[eid] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"{eid}: {results[eid]['error']}", file=sys.stderr, flush=True)
        if not only or FEED_ENTRY in only:
            try:
                found = feed_transcript(entries[FEED_ENTRY], Path(tmp) / FEED_ENTRY)
                results[FEED_ENTRY] = {"feed_transcript": found}
            except Exception as exc:
                results[FEED_ENTRY] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"{FEED_ENTRY}: {results[FEED_ENTRY]['error']}", file=sys.stderr, flush=True)
    partial = path.with_name("partial-" + path.name)
    partial.write_text(json.dumps(results, indent=1), encoding="utf-8")
    os.replace(partial, path)
    return check(out)


def check(out: Path) -> int:
    path = out / "scenarios.json"
    if not path.is_file():
        print(f"FAIL  {path} is missing (run `python -m evals.scenarios`)")
        return 1
    results = json.loads(path.read_text(encoding="utf-8"))
    expected = [(eid, name) for eid in LONG_ENTRIES for name in SCENARIOS]
    expected.append((FEED_ENTRY, "feed_transcript"))
    failures = 0
    for eid, name in expected:
        found = results.get(eid, {})
        record = found.get(name)
        if record is None:
            reason = found.get("error") or "not recorded"
            print(f"FAIL  {eid:<12} {name:<15} {reason}")
            failures += 1
            continue
        numbers = {k: v for k, v in record.items() if k not in ("pass", "samples")}
        print(f"{'OK  ' if record['pass'] else 'FAIL'}  {eid:<12} {name:<15} {numbers}")
        failures += not record["pass"]
    print(f"{len(expected)} scenarios, {failures} failing")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.scenarios", description=__doc__)
    parser.add_argument("--check", action="store_true", help="replay scenarios.json")
    parser.add_argument("--only", action="append", default=[], help="entry id (repeatable)")
    parser.add_argument("--out", default=str(RESULTS), help="results directory")
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    if args.check:
        return check(out)
    out.mkdir(parents=True, exist_ok=True)
    return run_all(set(args.only), out)


if __name__ == "__main__":
    sys.exit(main())
