"""``quotable serve`` through the shim and one shared process: real processes, ``tiny`` on CPU.

Every test runs with ``QUOTABLE_DAEMON=on`` in its own runtime and state dirs (see
``tests/conftest.py``, which also kills the processes a test spawned). Processes are found by
the pid in their ``serving`` log line. A process that exits while its shim is idle stays a
zombie until the shim next wakes, so "gone" means no such process or a zombie.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import anyio
import numpy as np
import pytest
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.exceptions import MCPError

from quotable import shim
from quotable.server import QUEUE_MESSAGE

pytestmark = pytest.mark.integration

JFK = Path(__file__).parents[1] / "fixtures" / "jfk.flac"
QUOTABLE = Path(sys.executable).with_name("quotable")
MODES = ["legacy", "auto"]
SERVING = re.compile(r"^quotable daemon\[(\d+)\]: serving (\S+)$", re.M)


@pytest.fixture(autouse=True)
def shared(monkeypatch):
    monkeypatch.setenv("QUOTABLE_DAEMON", "on")


@pytest.fixture(scope="module")
def long_clip(tmp_path_factory) -> Path:
    """JFK 16 times over (~3 min; ~9 s for ``tiny`` on CPU): time to act while a run is going."""
    from faster_whisper import decode_audio

    samples = np.tile(decode_audio(str(JFK), sampling_rate=16000), 16)
    path = tmp_path_factory.mktemp("clips") / "jfk_x16.wav"
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def _log() -> str:
    path = shim.log_path()
    return path.read_text() if path.exists() else ""


def _served() -> list[tuple[int, str]]:
    return [(int(pid), sock) for pid, sock in SERVING.findall(_log())]


def _gone(pid: int) -> bool:
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return state.stdout.strip()[:1] in ("", "Z")


def _run(main, timeout: float):
    async def bounded():
        with anyio.fail_after(timeout):
            return await main()

    return anyio.run(bounded)


async def _wait_for(predicate, timeout=20.0):
    with anyio.fail_after(timeout):
        while not predicate():
            await anyio.sleep(0.05)


def _client(mode: str = "auto", errlog=None) -> Client:
    params = StdioServerParameters(command=str(QUOTABLE), args=["serve"], env=dict(os.environ))
    return Client(stdio_client(params, errlog=errlog or sys.stderr), mode=mode)


async def _transcribe(client: Client, source: Path, progress=None, **args) -> dict:
    result = await client.call_tool(
        "transcribe", {"source": str(source), **args}, progress_callback=progress
    )
    assert not result.is_error, result.content
    return result.structured_content


# --- one process, one model ---------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_two_sessions_share_one_process_and_one_model_load(mode, long_clip):
    async def main():
        async with _client(mode) as a, _client(mode) as b:
            first = await _transcribe(a, JFK)
            second = await _transcribe(b, long_clip, mode="whisper")
            again = await _transcribe(b, JFK)
        return first, second, again

    first, second, again = _run(main, 120)
    assert "fellow Americans" in first["text"] and not first["cached"]
    assert "fellow Americans" in second["text"]
    assert again["cached"] and again["id"] == first["id"]  # b sees a's cache: one store
    assert len(_served()) == 1
    assert _log().count("quotable: loaded Whisper tiny on cpu") == 1


def test_ten_shims_started_at_once_make_one_process():
    async def main():
        results = []

        async def one():
            async with _client() as client:
                results.append([t.name for t in (await client.list_tools()).tools])

        async with anyio.create_task_group() as tg:
            for _ in range(10):
                tg.start_soon(one)
        return results

    results = _run(main, 120)
    assert len(results) == 10 and all(len(r) == 3 for r in results)
    assert len(_served()) == 1


def test_concurrent_requests_for_one_source_make_one_run(long_clip):
    async def main():
        out = {}
        async with _client() as a, _client() as b:
            async with anyio.create_task_group() as tg:

                async def call(name, client):
                    out[name] = await _transcribe(client, long_clip, mode="whisper")

                tg.start_soon(call, "a", a)
                tg.start_soon(call, "b", b)
        return out

    out = _run(main, 180)
    assert out["a"]["id"] == out["b"]["id"]
    assert sorted([out["a"]["cached"], out["b"]["cached"]]) == [False, True]


# --- planned exit: bye and resend -------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_requests_around_the_idle_exit_all_succeed(mode, monkeypatch):
    monkeypatch.setenv("QUOTABLE_MODEL_TTL", "1")

    async def main():
        async with _client(mode) as client:
            # The process checks once a second, so it exits 1-2 s after the last request;
            # these gaps straddle that window from both sides.
            for gap in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4):
                await anyio.sleep(gap)
                result = await client.call_tool("read_transcript", {"id": "nope"})
                # Answered by the server, not failed by the shim.
                assert "Unknown transcript id" in result.content[0].text
            info = await client.call_tool("media_info", {"source": str(JFK)})
            assert not info.is_error

    _run(main, 120)
    exits = _log().count("exiting (idle)")
    assert exits >= 3  # the gaps above 2 s each outlived a process
    assert len(_served()) == exits + 1 or len(_served()) == exits  # last one may still be up


def test_no_process_remains_after_the_ttl(monkeypatch):
    monkeypatch.setenv("QUOTABLE_MODEL_TTL", "1")

    async def main():
        async with _client() as client:
            await _transcribe(client, JFK)
            ((pid, _),) = _served()
            await _wait_for(lambda: _gone(pid), timeout=10)
            return pid

    pid = _run(main, 60)
    assert _gone(pid)
    assert not any(p.suffix == ".sock" for p in shim.runtime_dir().iterdir())


def test_a_touched_source_file_moves_to_a_new_process(monkeypatch, tmp_path):
    # A private copy, imported first through PYTHONPATH: touching the real source would also
    # retire the developer's own shared process when the tool is an editable install.
    shutil.copytree(
        shim.PACKAGE_DIR, tmp_path / "quotable", ignore=shutil.ignore_patterns("__pycache__")
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    source = tmp_path / "quotable" / "feeds.py"
    real_mtime = (shim.PACKAGE_DIR / "feeds.py").stat().st_mtime_ns

    async def main():
        async with _client() as client:
            await client.list_tools()
            ((old_pid, old_sock),) = _served()
            os.utime(source)  # newer code: a new socket key
            await _wait_for(lambda: _gone(old_pid))  # idle, so it leaves at once
            assert "exiting (outdated)" in _log()
            assert [t.name for t in (await client.list_tools()).tools]
            ((_, new_sock),) = [s for s in _served() if s[0] != old_pid]
            return old_sock, new_sock

    old_sock, new_sock = _run(main, 60)
    assert old_sock != new_sock
    assert (shim.PACKAGE_DIR / "feeds.py").stat().st_mtime_ns == real_mtime


# --- crash and cancel ----------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_a_killed_process_fails_the_call_and_the_next_call_works(mode, long_clip):
    async def main():
        async with _client(mode) as client:
            started = anyio.Event()

            async def progress(done, total, message):
                started.set()

            async with anyio.create_task_group() as tg:
                box = {}

                async def call():
                    try:
                        await client.call_tool(
                            "transcribe",
                            {"source": str(long_clip), "mode": "whisper"},
                            progress_callback=progress,
                        )
                    except MCPError as exc:  # the shim's JSON-RPC error
                        box["error"] = exc

                tg.start_soon(call)
                await started.wait()
                ((pid, _),) = _served()
                os.kill(pid, 9)
            failed = box["error"]
            after = await _transcribe(client, JFK)
        return failed, after, pid

    failed, after, pid = _run(main, 120)
    assert "may or may not have run" in str(failed) and "daemon.log" in str(failed)
    assert "fellow Americans" in after["text"]
    assert len({p for p, _ in _served()}) == 2


def test_a_closed_session_frees_the_queue(long_clip):
    async def main():
        seen_b = []
        a_started = anyio.Event()

        async def a_progress(done, total, message):
            a_started.set()

        async def b_progress(done, total, message):
            seen_b.append(message)

        async with _client() as b:
            async with anyio.create_task_group() as tg:
                a_scope = anyio.CancelScope()

                async def run_a():
                    with a_scope:
                        async with _client() as a:
                            await a.call_tool(
                                "transcribe",
                                {"source": str(long_clip), "mode": "whisper"},
                                progress_callback=a_progress,
                            )

                tg.start_soon(run_a)
                await a_started.wait()
                box = {}

                async def run_b():
                    start = time.monotonic()
                    box["result"] = await _transcribe(b, JFK, progress=b_progress)
                    box["seconds"] = time.monotonic() - start

                tg.start_soon(run_b)
                await _wait_for(lambda: QUEUE_MESSAGE in seen_b)
                a_scope.cancel()  # a's session closes: its shim exits, its run is cancelled
            # Had a's run finished instead, this would be a cache hit.
            box["a_after"] = await _transcribe(b, long_clip, mode="whisper")
        return box, seen_b

    box, seen_b = _run(main, 120)
    assert "fellow Americans" in box["result"]["text"]
    assert seen_b[0] == QUEUE_MESSAGE
    assert box["a_after"]["cached"] is False


# --- process hygiene -------------------------------------------------------------------------


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_the_shared_process_holds_no_pipe_and_the_shim_stays_small():
    proc = subprocess.Popen(
        [str(QUOTABLE), "serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=dict(os.environ),
    )
    try:

        def call(message):
            proc.stdin.write(json.dumps({"jsonrpc": "2.0", **message}).encode() + b"\n")
            proc.stdin.flush()
            if "id" in message:
                return json.loads(proc.stdout.readline())

        info = {"name": "t", "version": "0"}
        init = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": info}
        assert "result" in call({"id": 1, "method": "initialize", "params": init})
        call({"method": "notifications/initialized"})
        assert len(call({"id": 2, "method": "tools/list"})["result"]["tools"]) == 3

        ((pid, _),) = _served()
        assert os.readlink(f"/proc/{pid}/fd/0") == "/dev/null"
        assert os.readlink(f"/proc/{pid}/fd/1") == str(shim.log_path())
        assert os.readlink(f"/proc/{pid}/cwd") == "/"

        rollup = Path(f"/proc/{proc.pid}/smaps_rollup").read_text()
        private_kb = sum(
            int(line.split()[1])
            for line in rollup.splitlines()
            if line.startswith(("Private_Clean:", "Private_Dirty:"))
        )
        assert private_kb < 15 * 1024, f"shim private RSS {private_kb} kB"
    finally:
        proc.stdin.close()
        assert proc.wait(10) == 0
