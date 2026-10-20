"""GPU end-to-end check for the shared process: N sessions, one process on the GPU, then none.

``QUOTABLE_MODEL=turbo QUOTABLE_MODEL_TTL=5 uv run python evals/shared_process_check.py
tests/fixtures/jfk.flac`` starts N ``quotable serve`` sessions (stdio shims) that each
transcribe the file at once. Each session gets its own copy with a few ms of silence in
front: one source would make the later sessions wait for the first one's cached result
instead of queueing for the GPU. A sampler polls
``nvidia-smi`` and keeps only processes started by this check (their environment carries its
private ``XDG_RUNTIME_DIR``). After the sessions close it waits out the TTL and checks that
nothing is left. Exit 0 when every check passes; Linux only (reads ``/proc``).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import anyio
import numpy as np
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

from quotable.server import QUEUE_MESSAGE

QUOTABLE = Path(sys.executable).with_name("quotable")
SERVING = re.compile(r"^quotable daemon\[(\d+)\]: serving ", re.M)
POLL_S = 0.25


def _ours(pid: int, runtime: str) -> bool:
    try:
        env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return False
    return f"XDG_RUNTIME_DIR={runtime}".encode() in env


def _gpu_apps() -> list[tuple[int, int]]:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [(int(p), int(m)) for p, m in (line.split(",") for line in out.splitlines() if line)]


def _copies(source: Path, n: int, into: Path) -> list[Path]:
    """``n`` versions of ``source`` with different content hashes, so different cache keys."""
    import wave

    from faster_whisper import decode_audio

    samples = decode_audio(str(source), sampling_rate=16000)
    paths = []
    for i in range(n):
        padded = np.concatenate([np.zeros(160 * (i + 1), dtype=samples.dtype), samples])
        path = into / f"session{i + 1}.wav"
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes((np.clip(padded, -1, 1) * 32767).astype("<i2").tobytes())
        paths.append(path)
    return paths


def _alive(pid: int) -> bool:
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return state.stdout.strip()[:1] not in ("", "Z")


class Sampler(threading.Thread):
    def __init__(self, runtime: str) -> None:
        super().__init__(daemon=True)
        self.runtime = runtime
        self.stop = threading.Event()
        self.max_at_once = 0
        self.pids: set[int] = set()
        self.peak_mib = 0

    def run(self) -> None:
        while not self.stop.is_set():
            ours = [(p, m) for p, m in _gpu_apps() if _ours(p, self.runtime)]
            self.max_at_once = max(self.max_at_once, len(ours))
            self.pids |= {p for p, _ in ours}
            self.peak_mib = max([self.peak_mib, *(m for _, m in ours)])
            time.sleep(POLL_S)


async def _session(env: dict, source: Path, out: dict, name: str) -> None:
    params = StdioServerParameters(command=str(QUOTABLE), args=["serve"], env=env)
    messages: list[str | None] = []

    async def progress(done, total, message):
        messages.append(message)

    async with Client(stdio_client(params)) as client:
        start = time.monotonic()
        result = await client.call_tool(
            "transcribe", {"source": str(source), "mode": "whisper"}, progress_callback=progress
        )
    out[name] = {
        "ok": not result.is_error,
        "cached": (result.structured_content or {}).get("cached"),
        "queued": QUEUE_MESSAGE in messages,
        "seconds": time.monotonic() - start,
        "error": result.content[0].text if result.is_error else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="an audio file every session transcribes")
    parser.add_argument("--sessions", type=int, default=2)
    args = parser.parse_args()
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found: this check needs a CUDA GPU", file=sys.stderr)
        return 2

    ttl = float(os.environ.get("QUOTABLE_MODEL_TTL") or 300)
    runtime = tempfile.mkdtemp(prefix="qchk")  # short: AF_UNIX paths max out near 104 bytes
    work = Path(tempfile.mkdtemp(prefix="quotable-check-"))
    env = {
        **os.environ,
        "QUOTABLE_DAEMON": "on",
        "XDG_RUNTIME_DIR": runtime,
        "XDG_STATE_HOME": str(work / "state"),
        "QUOTABLE_OUTPUT_DIR": str(work / "out"),
    }
    log = work / "state" / "quotable" / "daemon.log"
    sources = _copies(args.source.resolve(), args.sessions, work)

    sampler = Sampler(runtime)
    sampler.start()
    results: dict[str, dict] = {}

    async def run_all() -> None:
        async with anyio.create_task_group() as tg:
            for n in range(args.sessions):
                tg.start_soon(_session, env, sources[n], results, f"session {n + 1}")

    started = time.monotonic()
    anyio.run(run_all)
    elapsed = time.monotonic() - started
    text = log.read_text() if log.exists() else ""
    daemons = [int(p) for p in SERVING.findall(text)]

    deadline = time.monotonic() + ttl + 5  # TTL, plus the once-a-second idle check
    while time.monotonic() < deadline and any(_alive(p) for p in daemons):
        time.sleep(POLL_S)
    time.sleep(2 * POLL_S)  # one more GPU sample after the exit
    sampler.stop.set()
    sampler.join()
    left_on_gpu = [p for p, _ in _gpu_apps() if _ours(p, runtime)]

    checks = [
        (
            "every session got its transcript",
            all(r["ok"] and r["cached"] is False for r in results.values()),
        ),
        ("one shared process served them", len(daemons) == 1),
        ("one model load", text.count("quotable: loaded Whisper") == 1),
        ("at most one process on the GPU at once", sampler.max_at_once == 1),
        ("and it was the shared process", sampler.pids == set(daemons)),
        (
            "the second session queued, and was told",
            args.sessions < 2 or any(r["queued"] for r in results.values()),
        ),
        (f"no process left {ttl:g} s after the last request", not any(map(_alive, daemons))),
        ("nothing of ours left on the GPU", not left_on_gpu),
    ]
    for name, r in results.items():
        print(f"{name}: {r}")
    print(f"wall time {elapsed:.1f} s · peak VRAM of the shared process {sampler.peak_mib} MiB")
    for label, ok in checks:
        print(f"{'✓' if ok else '✗'} {label}")
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(runtime, ignore_errors=True)
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
