"""The matrix runner: every cell in its own process, one JSON row per cell.

``python -m evals.run`` expands the manifest into cells (``evals/README.md`` → The matrix),
runs each one as ``python -m evals.run --cell <id> --out <dir> --row <file>`` and collects
the rows into ``<out>/cells.jsonl`` (default ``evals/results/``). A cell that already has a
row is not run again unless ``--redo``; a ``skipped`` row is retried on every run, because it
records something that did not happen yet (no token, no media).

A child process per cell keeps model load, memory and a CUDA fallback from leaking between
cells, and ``os.wait4`` gives each cell's peak RSS. The child writes its row to the ``--row``
file rather than stdout, which the libraries share.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals import oracle
from evals.build_corpus import ROOT, load_manifest
from quotable import diarize, feeds, pipeline, transcribe
from quotable.config import Settings
from quotable.models import QuotableError

RESULTS = ROOT / "evals" / "results"
MODELS = ("tiny", "base", "small", "turbo")
COMPUTE = {"cpu": "int8", "cuda": "float16"}
GPU_CELL = ("turbo", "cuda")  # long items, real-path cells, diarize cells
README_CPU = ("base", "cpu")  # the README's no-GPU row
SMOKE = "jfk.flac|whisper|tiny|cpu|false|false"
NVIDIA_SMI = [
    "nvidia-smi",
    "--query-compute-apps=pid,used_memory",
    "--format=csv,noheader,nounits",
]
LOG_TAIL = 4000

ROW_KEYS = (
    "cell_id",
    "entry",
    "mode",
    "model",
    "device",
    "compute_type",
    "diarize",
    "real_path",
    "status",
    "reason",
    "error",
    "wall_s",
    "load_s",
    "audio_s",
    "realtime",
    "cached_s",
    "segments",
    "words",
    "progress_calls",
    "first_progress_s",
    "max_rss_mb",
    "vram_peak_mb",
    "transcript_path",
)


@dataclass(frozen=True)
class Cell:
    entry: str
    mode: str  # "whisper" | "captions"
    model: str | None  # None on captions cells, which load no model
    device: str | None
    diarize: bool
    real: bool  # Whisper on the live URL or feed, the only cells that exercise the download

    @property
    def id(self) -> str:
        flags = [str(self.diarize).lower(), str(self.real).lower()]
        return "|".join([self.entry, self.mode, self.model or "-", self.device or "-", *flags])

    @property
    def slug(self) -> str:
        return self.id.replace("|", "__")

    @classmethod
    def parse(cls, cell_id: str) -> Cell:
        parts = cell_id.split("|")
        if len(parts) != 6 or parts[1] not in ("whisper", "captions"):
            raise ValueError(f"not a cell id: {cell_id!r}")
        entry, mode, model, device, diarized, real = parts
        if mode == "whisper":
            shape_ok = model != "-" and device in COMPUTE
        else:
            shape_ok = model == "-" and device == "-"
        if not shape_ok or not {diarized, real} <= {"true", "false"}:
            raise ValueError(f"not a cell id: {cell_id!r}")
        return cls(
            entry,
            mode,
            None if model == "-" else model,
            None if device == "-" else device,
            diarized == "true",
            real == "true",
        )


# --- the matrix -------------------------------------------------------------------------


def entry_cells(entry: dict[str, Any]) -> list[Cell]:
    eid = entry["id"]
    cells = []
    if oracle.wants_captions_ref(entry):  # captions on YouTube, a transcript tag on a feed
        cells.append(Cell(eid, "captions", None, None, False, False))
    grid = [(m, d) for m in MODELS for d in COMPUTE] if entry["tier"] == "short" else [GPU_CELL]
    if entry["readme_row"] and README_CPU not in grid:
        grid.append(README_CPU)
    cells += [Cell(eid, "whisper", m, d, False, False) for m, d in grid]
    if entry["diarize"]:
        cells.append(Cell(eid, "whisper", *GPU_CELL, True, False))
    if entry["kind"] in ("youtube", "feed"):
        cells.append(Cell(eid, "whisper", *GPU_CELL, False, True))
    return cells


def matrix(entries: list[dict[str, Any]], name: str) -> list[Cell]:
    if name == "smoke":
        return [Cell.parse(SMOKE)]
    return [cell for entry in entries for cell in entry_cells(entry)]


def hf_token() -> str | None:
    # The same two variables, in the same order, as Settings.from_env (config.py:64).
    return os.environ.get("QUOTABLE_HF_TOKEN") or os.environ.get("HF_TOKEN") or None


def cell_source(cell: Cell, entry: dict[str, Any], index: dict[str, Any]) -> tuple[str, Any]:
    """(source, episode) for ``pipeline.transcribe``; ``FileNotFoundError`` if media is absent.

    Whisper cells on URL and feed entries run on the audio Task 3 cached; captions cells
    and real-path cells go to the live source.
    """
    if entry["kind"] in ("file", "tts"):
        path = ROOT / entry["source"]
        if not path.is_file():
            raise FileNotFoundError(entry["source"])
        return str(path), None
    if cell.mode == "captions" or cell.real:
        return entry["source"], entry.get("episode")
    path = oracle.cached_path(index, entry["id"])
    if path is None:
        raise FileNotFoundError(f"cached audio for {entry['id']} (run `python -m evals.oracle`)")
    return str(path), None


def new_row(cell: Cell) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(ROW_KEYS)
    row.update(
        cell_id=cell.id,
        entry=cell.entry,
        mode=cell.mode,
        model=cell.model,
        device=cell.device,
        compute_type=COMPUTE[cell.device] if cell.device else None,
        diarize=cell.diarize,
        real_path=cell.real,
    )
    return row


# --- the child: one cell ----------------------------------------------------------------


class VramSampler:
    """Peak GPU memory of one pid; the device total would include the desktop's."""

    def __init__(self, pid: int, period: float = 1.0) -> None:
        self.pid, self.period = str(pid), period
        self.peak: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> VramSampler:
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
        try:
            out = subprocess.run(
                NVIDIA_SMI, capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return
        for line in out.splitlines():
            pid, _, used = (part.strip() for part in line.partition(","))
            if pid == self.pid and used.isdigit():
                self.peak = max(self.peak or 0, int(used))


class Progress:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.calls = 0
        self.first_s: float | None = None

    def __call__(self, segment: Any, done_s: float, total_s: float | None) -> None:
        self.calls += 1
        if self.first_s is None:
            self.first_s = round(time.monotonic() - self.started, 3)


def _expected_key(entry: dict[str, Any]) -> str:
    # pick_episode falls back to index and title matching (feeds.py), so a guid that left
    # the feed would quietly transcribe another episode; the cache key says which one ran.
    episode = feeds.Episode(
        index=0,
        title="",
        guid=entry["episode"],
        published=None,
        duration=None,
        enclosure_url=None,
    )
    return feeds.episode_key(episode)


def _measure(cell: Cell, entry: dict[str, Any], out: Path, row: dict[str, Any]) -> None:
    source, episode = cell_source(cell, entry, oracle.load_index())
    device = cell.device or "cpu"
    settings = Settings(
        output_dir=out / "cache" / cell.slug,
        model=cell.model or "base",  # unused on captions cells
        device=device,
        compute_type=COMPUTE[device],
        hf_token=hf_token(),
    )
    # Product defaults: no language, so Whisper detects it and captions take the default track.
    call = dict(mode=cell.mode, diarize=cell.diarize, episode=episode, settings=settings)
    sampler = VramSampler(os.getpid()) if cell.device == "cuda" else None
    try:
        with sampler or nullcontext():
            if cell.mode == "whisper":
                started = time.monotonic()
                transcribe.get_model(settings)
                if cell.diarize:
                    diarize.get_pipeline(settings)
                row["load_s"] = round(time.monotonic() - started, 3)

            progress = Progress()
            first = pipeline.transcribe(source, force=True, on_segment=progress, **call)
            wall = time.monotonic() - progress.started
            transcript = first.transcript
            audio_s = transcript.duration or entry["expect"]["duration_s"]
            row.update(
                status="ok",
                wall_s=round(wall, 3),
                audio_s=round(audio_s, 3),
                realtime=round(audio_s / wall, 2) if wall > 0 else None,
                segments=len(transcript.segments),
                words=sum(len(s.text.split()) for s in transcript.segments),
                progress_calls=progress.calls,
                first_progress_s=progress.first_s,
                transcript_path=str((first.path / "transcript.json").relative_to(out)),
            )
            if (
                entry["kind"] == "feed"
                and episode
                and transcript.source_key != _expected_key(entry)
            ):
                row.update(status="fail", reason="guid_moved")
                return

            started = time.monotonic()
            second = pipeline.transcribe(source, **call)
            row["cached_s"] = round(time.monotonic() - started, 3)
            if not second.cached:
                row.update(status="fail", reason="not_cached")
    finally:
        if sampler is not None:
            row["vram_peak_mb"] = sampler.peak


def run_cell(cell: Cell, out: Path) -> dict[str, Any]:
    entry = {e["id"]: e for e in load_manifest()}[cell.entry]
    row = new_row(cell)
    try:
        _measure(cell, entry, out, row)
    except QuotableError as exc:
        name = type(exc).__name__
        reason = name if row["status"] is None else f"{name} on the cached call"
        error = {"type": name, "message": exc.message, "fix": exc.fix}
        row.update(status="error", reason=reason, error=error)
    except Exception as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        row.update(status="crash", reason="exception", error=error)
    # Both fallback paths set this (load time and mid-inference); the pipeline carries on on
    # CPU without saying so, which would poison every number in a cuda row.
    if cell.device == "cuda" and transcribe._cuda_broken and row["status"] in ("ok", "error"):
        row.update(status="fail", reason="cuda_fallback")
    # load_pipeline leaves pyannote on CPU without a word when torch sees no CUDA; pyannote
    # sets ``device`` only when the pipeline is moved.
    if cell.device == "cuda" and cell.diarize and row["status"] == "ok" and diarize._pipeline:
        if getattr(getattr(diarize._pipeline[1], "device", None), "type", None) != "cuda":
            row.update(status="fail", reason="cuda_fallback")
    return row


def child_main(cell_id: str, out: Path, row_path: Path) -> int:
    row = run_cell(Cell.parse(cell_id), out)
    row_path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return 0


# --- the parent: the matrix -------------------------------------------------------------


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["cell_id"]] = row
    return rows


def save_rows(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    partial = path.with_name("partial-" + path.name)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.values())
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def prefetch_models(models: set[str]) -> None:
    """Download missing Whisper models up front, so no cell's ``load_s`` includes one."""
    from faster_whisper.utils import download_model

    for name in sorted(models):
        try:
            download_model(name, local_files_only=True)
        except Exception:
            print(f"downloading Whisper {name} before timing anything", flush=True)
            download_model(name)


def timeout_s(cell: Cell, entry: dict[str, Any]) -> float:
    # A hang guard, not a budget: CPU cells get four times the audio length on top.
    return 900 + entry["expect"]["duration_s"] * (4 if cell.device == "cpu" else 1)


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # the cell and whatever it spawned (ffmpeg, deno)
    except ProcessLookupError:
        pass


def spawn(cell: Cell, out: Path, timeout: float) -> dict[str, Any]:
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    row_path, log_path = logs / f"{cell.slug}.row.json", logs / f"{cell.slug}.log"
    row_path.unlink(missing_ok=True)
    shutil.rmtree(out / "cache" / cell.slug, ignore_errors=True)
    cmd = [sys.executable, "-m", "evals.run", "--cell", cell.id, "--out", str(out)]
    cmd += ["--row", str(row_path)]
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    timed_out = threading.Event()

    def expire() -> None:
        timed_out.set()
        _kill(proc)

    timer = threading.Timer(timeout, expire)
    timer.daemon = True
    timer.start()
    try:
        _, status, usage = os.wait4(proc.pid, 0)
    except BaseException:
        _kill(proc)
        os.wait4(proc.pid, 0)
        proc.returncode = -signal.SIGKILL
        raise
    finally:
        timer.cancel()
    proc.returncode = os.waitstatus_to_exitcode(status)

    if row_path.is_file() and not timed_out.is_set():
        row = json.loads(row_path.read_text(encoding="utf-8"))
        row_path.unlink()
    else:
        row = new_row(cell)
        code = proc.returncode
        if timed_out.is_set():
            reason = f"timeout after {timeout:.0f} s"
        else:
            reason = f"signal {-code}" if code < 0 else f"exit {code}"
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-LOG_TAIL:]
        error = {"type": "ChildDied", "message": f"the cell process ended: {reason}"}
        row.update(status="crash", reason=reason, error={**error, "traceback": tail})
    row["max_rss_mb"] = round(usage.ru_maxrss / 1024, 1)  # Linux reports kilobytes
    return row


def _summary(row: dict[str, Any]) -> str:
    if row["status"] == "ok":
        return f"ok  {row['wall_s']:.1f} s  {row['realtime']}x"
    return f"{row['status']}  {row['reason']}"


def parent_main(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    entries = {e["id"]: e for e in load_manifest()}
    cells = matrix(list(entries.values()), args.matrix)
    if args.only:
        cells = [c for c in cells if any(fnmatch.fnmatchcase(c.id, g) for g in args.only)]
    if not cells:
        print("evals.run: no cell matches", file=sys.stderr)
        return 1

    results = out / "cells.jsonl"
    rows = load_rows(results)
    todo = [c for c in cells if args.redo or rows.get(c.id, {}).get("status") in (None, "skipped")]
    index, token = oracle.load_index(), hf_token()
    runnable, skipped, missing = [], [], []
    for cell in todo:
        entry = entries[cell.entry]
        if cell.diarize and not token:
            skipped.append((cell, "no token"))
            continue
        try:
            cell_source(cell, entry, index)
        except FileNotFoundError as exc:
            if entry["kind"] == "tts":  # Task 2 may not have run; recorded, never absent
                skipped.append((cell, "no media"))
            else:
                missing.append(f"{cell.id}: {exc}")
            continue
        runnable.append(cell)
    if missing:
        print("evals.run: media missing, nothing run:", *missing, sep="\n  ", file=sys.stderr)
        return 1

    for cell, reason in skipped:
        rows[cell.id] = {**new_row(cell), "status": "skipped", "reason": reason}
    save_rows(results, rows)
    prefetch_models({c.model for c in runnable if c.model})

    print(f"{len(runnable)} to run, {len(cells) - len(todo)} done, {len(skipped)} skipped")
    print(f"rows: {results}  logs: {out / 'logs'}", flush=True)
    crashes = 0
    for n, cell in enumerate(runnable, 1):
        print(f"[{n}/{len(runnable)}] {cell.id}", end="  ", flush=True)
        row = spawn(cell, out, timeout_s(cell, entries[cell.entry]))
        crashes += row["status"] == "crash"
        rows[cell.id] = row
        save_rows(results, rows)
        print(_summary(row), flush=True)

    counts: dict[str, int] = {}
    for cell in cells:
        status = rows[cell.id]["status"]
        counts[status] = counts.get(status, 0) + 1
    print("  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 1 if crashes else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.run", description=__doc__)
    parser.add_argument("--matrix", choices=("full", "smoke"), default="full")
    parser.add_argument("--only", action="append", default=[], help="glob on cell_id")
    parser.add_argument("--redo", action="store_true", help="re-run cells that have a row")
    parser.add_argument("--out", default=str(RESULTS), help="results directory")
    parser.add_argument("--cell", help=argparse.SUPPRESS)  # child mode
    parser.add_argument("--row", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.cell:
        if not args.row:
            parser.error("--cell needs --row")
        return child_main(args.cell, Path(args.out).resolve(), Path(args.row))
    try:
        return parent_main(args)
    except KeyboardInterrupt:
        print("\nevals.run: interrupted; the running cell was not recorded", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
