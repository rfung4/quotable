"""Benchmark the pipeline per (source, model, device) and write a markdown table.

    uv run python scripts/bench.py <source>... --models tiny,base --devices cpu,cuda --out bench.md

Each Whisper run uses ``force=True`` in a throwaway cache; the cached column is a second,
identical call. The first run of a model includes loading it. A source with captions also
gets a captions-only row (no model, no device).
"""

import argparse
import dataclasses
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

from quotable import pipeline
from quotable.config import Settings
from quotable.models import QuotableError


def _hardware() -> str:
    gpu = "no GPU"
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            import subprocess

            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=10,
            ).stdout.strip()
            gpu = out.splitlines()[0] if out else "CUDA GPU"
    except Exception:
        pass
    return f"{gpu} · {os.cpu_count()} CPUs · {platform.platform()}"


def _timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def _row(source, model, device, wall, duration, cached_s) -> str:
    rtf = f"{duration / wall:.1f}x" if duration and wall else "n/a"
    dur = f"{duration:.0f}s" if duration else "n/a"
    return f"| {source} | {model} | {device} | {dur} | {wall:.1f}s | {rtf} | {cached_s:.2f}s |"


def _bench_one(source: str, base: Settings, model: str | None, device: str | None) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        settings = dataclasses.replace(base, output_dir=Path(tmp))
        if model:
            compute = "float16" if device == "cuda" else "int8"
            settings = dataclasses.replace(
                settings, model=model, device=device, compute_type=compute
            )
        mode = "whisper" if model else "captions"
        run = lambda force: pipeline.transcribe(  # noqa: E731
            source, mode=mode, force=force, settings=settings
        )
        outcome, wall = _timed(lambda: run(True))
        _, cached_s = _timed(lambda: run(False))
        return _row(
            Path(source).name if Path(source).exists() else source,
            model or "captions",
            device or "-",
            wall,
            outcome.transcript.duration,
            cached_s,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--models", default="tiny,base")
    parser.add_argument("--devices", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("bench.md"))
    parser.add_argument("--print", action="store_true", help="also print the table")
    args = parser.parse_args()

    base = Settings.from_env()
    rows: list[str] = []
    for source in args.sources:
        try:
            if pipeline.media_info(source).has_captions:
                rows.append(_bench_one(source, base, None, None))
        except QuotableError as exc:
            print(f"skipping captions row for {source}: {exc.message}", file=sys.stderr)
        for device in args.devices.split(","):
            for model in args.models.split(","):
                print(f"bench: {source} {model} {device}", file=sys.stderr)
                rows.append(_bench_one(source, base, model, device))

    table = "\n".join(
        [
            f"Hardware: {_hardware()}",
            "",
            "| source | model | device | audio | wall | realtime | cached |",
            "|---|---|---|---|---|---|---|",
            *rows,
            "",
        ]
    )
    args.out.write_text(table)
    if args.print:
        print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
