import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import anyio
import pytest
from anyio.streams.buffered import BufferedByteReceiveStream
from mcp.server import MCPServer
from mcp.types import LATEST_PROTOCOL_VERSION

from quotable import daemon
from quotable import server as server_module
from quotable.config import Settings
from quotable.models import QuotableError
from quotable.server import create_server

TICK = 0.02


@pytest.fixture
def sock():
    # pytest's tmp_path can exceed the ~104-108 byte AF_UNIX path limit.
    short = Path(tempfile.mkdtemp(prefix="qd"))
    yield short / "k.sock"
    shutil.rmtree(short, ignore_errors=True)


class Client:
    def __init__(self, stream):
        self.stream = stream
        self.buffered = BufferedByteReceiveStream(stream)
        self.sent = 0

    @classmethod
    async def connect(cls, sock, cwd="/work"):
        client = cls(await anyio.connect_unix(sock))
        await client.stream.send(json.dumps({"cwd": cwd}).encode() + b"\n")
        return client

    async def send(self, message):
        await self.stream.send(json.dumps({"jsonrpc": "2.0", **message}).encode() + b"\n")
        self.sent += 1

    async def recv(self):
        return json.loads(await self.buffered.receive_until(b"\n", daemon.MAX_LINE))

    async def response(self, id):
        while True:
            message = await self.recv()
            if message.get("id") == id and ("result" in message or "error" in message):
                return message

    async def initialize(self):
        await self.send(
            {
                "id": "init",
                "method": "initialize",
                "params": {
                    "protocolVersion": LATEST_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            }
        )
        result = (await self.response("init"))["result"]
        await self.send({"method": "notifications/initialized"})
        return result

    async def call(self, id, method, params=None):
        await self.send({"id": id, "method": method, "params": params or {}})
        return await self.response(id)


def _daemon(app, sock, *, ttl_s=0.0, key="k", current_key=None, busy=lambda: False):
    return daemon.Daemon(
        daemon.lowlevel_server(app),
        sock,
        key=key,
        current_key=current_key or (lambda: key),
        ttl_s=ttl_s,
        busy=busy,
        tick_s=TICK,
    )


async def _wait_for(sock):
    with anyio.fail_after(5):
        while not sock.exists():
            await anyio.sleep(0.01)


def _slow_app(release: anyio.Event, started: anyio.Event):
    app = MCPServer("slow")

    @app.tool()
    async def wait() -> str:
        started.set()
        await release.wait()
        return "done"

    return app


# --- lifetime lock -----------------------------------------------------------


def test_lock_is_exclusive(sock):
    first = daemon.take_lock(sock)
    assert first is not None
    assert daemon.take_lock(sock) is None
    first.close()
    again = daemon.take_lock(sock)
    assert again is not None
    again.close()


def test_daemon_command_exits_when_the_lock_is_held(tmp_path):
    runtime = Path(tempfile.mkdtemp(prefix="qd"))
    try:
        env = {
            **os.environ,
            "XDG_RUNTIME_DIR": str(runtime),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        }
        probe = "import sys; from quotable import shim; print(shim.socket_path())"
        sock = Path(
            subprocess.run(
                [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        held = daemon.take_lock(sock)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "quotable.cli", "daemon"], env=env, timeout=30
            )
        finally:
            held.close()
        assert proc.returncode == 0
        assert not sock.exists()
        assert not (tmp_path / "state").exists()  # returned before detaching
    finally:
        shutil.rmtree(runtime, ignore_errors=True)


# --- serving -------------------------------------------------------------------


def test_each_connection_is_its_own_session_on_one_server(sock, tmp_path):
    app = create_server(
        Settings(output_dir=tmp_path, model="tiny", device="cpu", compute_type="int8")
    )
    d = _daemon(app, sock)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(d.serve)
            await _wait_for(sock)
            a, b = await Client.connect(sock), await Client.connect(sock)
            assert (await a.initialize())["serverInfo"]["name"] == "quotable"
            await b.initialize()
            tools = await b.call(2, "tools/list")
            assert {t["name"] for t in tools["result"]["tools"]} == {
                "media_info",
                "transcribe",
                "read_transcript",
            }
            assert len(d.connections) == 2
            await a.stream.aclose()
            await anyio.sleep(0.1)
            assert len(d.connections) == 1
            assert (await b.call(3, "ping"))["result"] == {}
            tg.cancel_scope.cancel()

    anyio.run(main)


def test_bad_json_line_gets_an_error_and_the_session_survives(sock):
    d = _daemon(MCPServer("t"), sock)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(d.serve)
            await _wait_for(sock)
            c = await Client.connect(sock)
            await c.initialize()
            await c.stream.send(b"{not json\n")
            assert (await c.call(5, "ping"))["result"] == {}
            tg.cancel_scope.cancel()

    anyio.run(main)


def test_connection_without_a_hello_is_dropped(sock):
    d = _daemon(MCPServer("t"), sock)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(d.serve)
            await _wait_for(sock)
            stream = await anyio.connect_unix(sock)
            await stream.send(b"not json\n")
            with anyio.fail_after(2), pytest.raises(anyio.EndOfStream):
                await stream.receive()
            tg.cancel_scope.cancel()

    anyio.run(main)


def test_hello_cwd_is_kept_per_connection(sock):
    d = _daemon(MCPServer("t"), sock)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(d.serve)
            await _wait_for(sock)
            c = await Client.connect(sock, cwd="/home/someone/project")
            await c.initialize()
            assert [conn.cwd for conn in d.connections] == ["/home/someone/project"]
            tg.cancel_scope.cancel()

    anyio.run(main)


def test_stale_socket_file_is_replaced(sock):
    sock.write_text("left by a crashed process")
    d = _daemon(MCPServer("t"), sock)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(d.serve)
            await anyio.sleep(0.1)
            c = await Client.connect(sock)
            await c.initialize()
            tg.cancel_scope.cancel()

    anyio.run(main)


# --- bye and exit ----------------------------------------------------------------


def test_idle_sends_bye_with_lines_read_then_exits_and_unlinks(sock):
    d = _daemon(MCPServer("t"), sock, ttl_s=0.2)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(d.serve)
            await _wait_for(sock)
            a, b = await Client.connect(sock), await Client.connect(sock)
            await a.initialize()
            await a.call(2, "ping")
            with anyio.fail_after(3):
                bye_a, bye_b = await a.recv(), await b.recv()
            assert bye_a == {daemon.BYE: a.sent} and a.sent == 3
            assert bye_b == {daemon.BYE: 0}  # b never got past its hello
            with anyio.fail_after(3), pytest.raises(anyio.EndOfStream):
                await a.buffered.receive(1)

    anyio.run(main)
    assert not sock.exists()


def test_serve_returns_after_bye(sock):
    d = _daemon(MCPServer("t"), sock, ttl_s=0.1)

    async def main():
        with anyio.fail_after(3):
            await d.serve()

    anyio.run(main)
    assert not sock.exists()


def test_activity_postpones_idle_exit(sock):
    d = _daemon(MCPServer("t"), sock, ttl_s=0.4)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(d.serve)
            await _wait_for(sock)
            c = await Client.connect(sock)
            await c.initialize()
            for i in range(6):  # 0.6 s of traffic, each gap under the TTL
                await anyio.sleep(0.1)
                assert (await c.call(10 + i, "ping"))["result"] == {}
            tg.cancel_scope.cancel()

    anyio.run(main)


def test_no_bye_while_a_request_is_in_flight(sock):
    async def main():
        release, started = anyio.Event(), anyio.Event()
        d = _daemon(_slow_app(release, started), sock, ttl_s=0.1)
        async with anyio.create_task_group() as tg:
            tg.start_soon(d.serve)
            await _wait_for(sock)
            c = await Client.connect(sock)
            await c.initialize()
            await c.send({"id": 7, "method": "tools/call", "params": {"name": "wait"}})
            await started.wait()
            await anyio.sleep(0.4)  # four TTLs
            assert sock.exists()
            release.set()
            assert (await c.response(7))["result"]["content"][0]["text"] == "done"
            with anyio.fail_after(3):
                assert await c.recv() == {daemon.BYE: c.sent}

    anyio.run(main)


def test_a_cancelled_request_is_no_longer_in_flight(sock):
    async def main():
        release, started = anyio.Event(), anyio.Event()
        d = _daemon(_slow_app(release, started), sock, ttl_s=0.1)
        async with anyio.create_task_group() as tg:
            tg.start_soon(d.serve)
            await _wait_for(sock)
            c = await Client.connect(sock)
            await c.initialize()
            await c.send({"id": 8, "method": "tools/call", "params": {"name": "wait"}})
            await started.wait()
            await c.send({"method": "notifications/cancelled", "params": {"requestId": 8}})
            with anyio.fail_after(3):
                message = await c.recv()
                while daemon.BYE not in message:
                    message = await c.recv()
            assert message == {daemon.BYE: c.sent}

    anyio.run(main)


def test_busy_inference_postpones_exit(sock):
    busy = [True]
    d = _daemon(MCPServer("t"), sock, ttl_s=0.05, busy=lambda: busy[0])

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(d.serve)
            await _wait_for(sock)
            await anyio.sleep(0.3)
            assert sock.exists()
            busy[0] = False
            with anyio.fail_after(3):
                while sock.exists():
                    await anyio.sleep(0.02)

    anyio.run(main)


def test_ttl_zero_never_idles_but_an_outdated_key_exits(sock):
    key = ["k"]
    d = _daemon(MCPServer("t"), sock, ttl_s=0, current_key=lambda: key[0])

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(d.serve)
            await _wait_for(sock)
            c = await Client.connect(sock)
            await c.initialize()
            await anyio.sleep(0.3)
            assert sock.exists()
            key[0] = "changed"
            with anyio.fail_after(3):
                assert await c.recv() == {daemon.BYE: c.sent}

    anyio.run(main)


def test_a_line_sent_after_the_bye_is_not_counted(sock):
    d = _daemon(MCPServer("t"), sock, ttl_s=0.1)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(d.serve)
            await _wait_for(sock)
            c = await Client.connect(sock)
            await c.initialize()
            with anyio.fail_after(3):
                bye = await c.recv()
            await c.send({"id": 9, "method": "ping"})  # the shim would resend this one
            assert bye == {daemon.BYE: c.sent - 1}
            # EOF, or a reset because the process closed with that line unread; never a pong.
            with (
                anyio.fail_after(3),
                pytest.raises((anyio.IncompleteRead, anyio.BrokenResourceError)),
            ):
                await c.recv()

    anyio.run(main)


# --- SDK guard and process setup --------------------------------------------------


def test_sdk_lowlevel_run_takes_read_write_and_options():
    lowlevel = daemon.lowlevel_server(MCPServer("t"))
    params = list(inspect.signature(lowlevel.run).parameters)
    assert params[:3] == ["read_stream", "write_stream", "initialization_options"]


def test_missing_sdk_api_raises_a_typed_error_with_a_fix():
    with pytest.raises(QuotableError) as err:
        daemon.lowlevel_server(object())
    assert "mcp>=2.3,<2.4" in err.value.fix and "QUOTABLE_DAEMON=off" in err.value.fix


def test_detach_points_stdin_at_null_and_stdout_at_the_log(tmp_path):
    log = tmp_path / "state" / "daemon.log"
    probe = (
        "import os, subprocess, sys; from pathlib import Path; from quotable import daemon; "
        f"daemon.detach(Path({str(log)!r})); "
        "print('from-python', flush=True); "
        "subprocess.run(['sh', '-c', 'echo from-child; cat; pwd']); "
        "print('stderr-line', file=sys.stderr, flush=True)"
    )
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=30, input="wire\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert time.monotonic() - start < 20  # `cat` got EOF instead of blocking
    assert proc.stdout == "" and proc.stderr == ""
    assert log.read_text().splitlines() == ["from-python", "from-child", "/", "stderr-line"]
    assert oct(log.stat().st_mode & 0o777) == oct(0o600)


def test_cli_module_runs_as_main():
    proc = subprocess.run(
        [sys.executable, "-m", "quotable.cli", "--version"], capture_output=True, text=True
    )
    assert proc.returncode == 0 and proc.stdout.startswith("quotable ")


def test_daemon_command_is_hidden_from_help():
    proc = subprocess.run(
        [sys.executable, "-m", "quotable.cli", "--help"], capture_output=True, text=True
    )
    assert "daemon" not in proc.stdout


# --- per-session working directory ------------------------------------------------


@pytest.mark.parametrize("hello", [{"cwd": "relative/dir"}, {"cwd": ""}, {}, {"cwd": 3}, []])
def test_hello_without_an_absolute_cwd_is_refused(sock, hello):
    d = _daemon(MCPServer("t"), sock)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(d.serve)
            await _wait_for(sock)
            stream = await anyio.connect_unix(sock)
            await stream.send(json.dumps(hello).encode() + b"\n")
            with anyio.fail_after(2), pytest.raises(anyio.EndOfStream):
                await stream.receive()
            tg.cancel_scope.cancel()

    anyio.run(main)


@pytest.mark.parametrize(
    "source, expected",
    [
        ("clip.wav", "/work/a/clip.wav"),
        ("../b/clip.wav", "/work/a/../b/clip.wav"),
        ("/abs/clip.wav", "/abs/clip.wav"),
        ("~/clip.wav", str(Path("~/clip.wav").expanduser())),
        ("https://example.com/clip.wav", "https://example.com/clip.wav"),
        ("http://example.com/feed.xml", "http://example.com/feed.xml"),
    ],
)
def test_session_cwd_resolves_relative_sources(source, expected):
    token = daemon.SESSION_CWD.set("/work/a")
    try:
        assert server_module.session_source(source) == expected
    finally:
        daemon.SESSION_CWD.reset(token)


def test_no_session_cwd_leaves_sources_alone():
    # In-process serving (QUOTABLE_DAEMON=off): the process cwd is the session's.
    assert server_module.session_source("clip.wav") == "clip.wav"


def test_each_connection_resolves_against_its_own_cwd(sock, tmp_path, monkeypatch):
    seen = []

    def fake_media_info(source, episode=None):
        seen.append(source)
        raise QuotableError(f"no file at {source}", fix="check the path")

    def fake_transcribe(source, **kwargs):
        seen.append(source)
        raise QuotableError(f"no file at {source}", fix="check the path")

    monkeypatch.setattr(server_module.pipeline, "media_info", fake_media_info)
    monkeypatch.setattr(server_module.pipeline, "transcribe", fake_transcribe)
    app = create_server(
        Settings(output_dir=tmp_path, model="tiny", device="cpu", compute_type="int8")
    )
    d = _daemon(app, sock)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(d.serve)
            await _wait_for(sock)
            a = await Client.connect(sock, cwd="/proj/a")
            b = await Client.connect(sock, cwd="/proj/b")
            await a.initialize()
            await b.initialize()
            ra = await a.call(
                2, "tools/call", {"name": "media_info", "arguments": {"source": "x.mp3"}}
            )
            rb = await b.call(
                2, "tools/call", {"name": "transcribe", "arguments": {"source": "y.mp3"}}
            )
            tg.cancel_scope.cancel()
        return ra, rb

    ra, rb = anyio.run(main)
    assert seen == ["/proj/a/x.mp3", "/proj/b/y.mp3"]
    assert "/proj/a/x.mp3" in ra["result"]["content"][0]["text"]
    assert ra["result"]["isError"] and rb["result"]["isError"]


def test_a_session_closing_mid_call_does_not_take_down_the_others(sock):
    async def main():
        release, started = anyio.Event(), anyio.Event()
        d = _daemon(_slow_app(release, started), sock)
        async with anyio.create_task_group() as tg:
            tg.start_soon(d.serve)
            await _wait_for(sock)
            a, b = await Client.connect(sock), await Client.connect(sock)
            await a.initialize()
            await b.initialize()
            await a.send({"id": 7, "method": "tools/call", "params": {"name": "wait"}})
            await started.wait()
            await a.stream.aclose()  # a's session ends while its call is running
            release.set()  # its response now goes to a closed socket
            await anyio.sleep(0.2)
            with anyio.fail_after(3):
                assert (await b.call(2, "ping"))["result"] == {}
            c = await Client.connect(sock)
            await c.initialize()
            tg.cancel_scope.cancel()

    anyio.run(main)
