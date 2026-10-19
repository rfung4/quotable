import contextlib
import io
import json
import os
import select
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from quotable import shim


@pytest.fixture
def code(tmp_path, monkeypatch):
    """A fake package dir and site-packages dir, so mtimes can be moved."""
    package = tmp_path / "pkg"
    site = tmp_path / "site"
    package.mkdir()
    site.mkdir()
    (package / "server.py").write_text("")
    (package / "notes.txt").write_text("")
    monkeypatch.setattr(shim, "PACKAGE_DIR", package)
    monkeypatch.setattr(shim, "_site_packages", lambda: str(site))
    return package, site


def _bump(path, seconds=10):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 1_000_000_000))


def test_key_is_16_hex_and_stable(code):
    key = shim.socket_key({})
    assert len(key) == 16
    int(key, 16)
    assert shim.socket_key({}) == key


@pytest.mark.parametrize(
    "name", ["QUOTABLE_MODEL", "QUOTABLE_DEVICE", "QUOTABLE_MODEL_TTL", "HF_TOKEN", "XDG_DATA_HOME"]
)
def test_key_changes_with_settings(code, name):
    assert shim.socket_key({name: "a"}) != shim.socket_key({name: "b"})


@pytest.mark.parametrize("name", ["QUOTABLE_DAEMON", "PATH", "XDG_RUNTIME_DIR"])
def test_key_ignores_unrelated_env(code, name):
    assert shim.socket_key({name: "a"}) == shim.socket_key({})


def test_key_ignores_env_order(code):
    a = {"QUOTABLE_MODEL": "tiny", "QUOTABLE_DEVICE": "cpu"}
    b = {"QUOTABLE_DEVICE": "cpu", "QUOTABLE_MODEL": "tiny"}
    assert shim.socket_key(a) == shim.socket_key(b)


def test_key_changes_when_source_is_touched(code):
    package, _ = code
    before = shim.socket_key({})
    _bump(package / "server.py")
    assert shim.socket_key({}) != before


def test_key_ignores_non_python_files(code):
    package, _ = code
    before = shim.socket_key({})
    _bump(package / "notes.txt")
    assert shim.socket_key({}) == before


def test_key_changes_when_dependencies_change(code):
    _, site = code
    before = shim.socket_key({})
    _bump(site)
    assert shim.socket_key({}) != before


def test_key_changes_with_version_and_prefix(code, monkeypatch):
    before = shim.socket_key({})
    monkeypatch.setattr(shim, "_version", lambda: "9.9.9")
    after_version = shim.socket_key({})
    monkeypatch.setattr(shim.sys, "prefix", "/elsewhere")
    assert len({before, after_version, shim.socket_key({})}) == 3


def test_key_version_is_the_installed_one_without_importlib_metadata():
    from importlib.metadata import version

    assert shim._version() == version("quotable-mcp")
    probe = (
        "import sys; from quotable import shim; shim.socket_key(); "
        "print('importlib.metadata' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_key_on_the_real_install():
    key = shim.socket_key({})
    assert len(key) == 16 and key == shim.socket_key({})


def test_runtime_dir_under_xdg(tmp_path):
    path = shim.runtime_dir({"XDG_RUNTIME_DIR": str(tmp_path)})
    assert path == tmp_path / "quotable"
    assert stat.S_IMODE(path.lstat().st_mode) == 0o700


def test_runtime_dir_falls_back_to_tempdir(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    path = shim.runtime_dir({})
    assert path == tmp_path / f"quotable-{os.getuid()}"
    assert stat.S_IMODE(path.lstat().st_mode) == 0o700


def test_runtime_dir_is_reused(tmp_path):
    env = {"XDG_RUNTIME_DIR": str(tmp_path)}
    assert shim.runtime_dir(env) == shim.runtime_dir(env)


def test_runtime_dir_refuses_a_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    (tmp_path / "quotable").symlink_to(real)
    with pytest.raises(shim.ShimError, match="not a directory"):
        shim.runtime_dir({"XDG_RUNTIME_DIR": str(tmp_path)})


def test_runtime_dir_refuses_a_file(tmp_path):
    (tmp_path / "quotable").write_text("")
    with pytest.raises(shim.ShimError, match="not a directory"):
        shim.runtime_dir({"XDG_RUNTIME_DIR": str(tmp_path)})


@pytest.mark.parametrize("mode", [0o750, 0o705, 0o770])
def test_runtime_dir_refuses_shared_access(tmp_path, mode):
    path = tmp_path / "quotable"
    path.mkdir()
    path.chmod(mode)
    with pytest.raises(shim.ShimError, match="accessible to other users") as err:
        shim.runtime_dir({"XDG_RUNTIME_DIR": str(tmp_path)})
    assert "chmod 700" in err.value.fix


def test_runtime_dir_refuses_another_owner(tmp_path, monkeypatch):
    uid = os.getuid()
    monkeypatch.setattr(shim.os, "getuid", lambda: uid + 1)
    (tmp_path / "quotable").mkdir(mode=0o700)
    with pytest.raises(shim.ShimError, match="owned by another user"):
        shim.runtime_dir({"XDG_RUNTIME_DIR": str(tmp_path)})


def test_key_names_the_socket(tmp_path, code):
    env = {"XDG_RUNTIME_DIR": str(tmp_path)}
    path = shim.socket_path(env)
    assert path == tmp_path / "quotable" / f"{shim.socket_key(env)}.sock"


def test_runtime_dir_longest_socket_path_fits_macos(tmp_path, code, monkeypatch):
    # macOS: TMPDIR is /var/folders/<2>/<28>/T/; sun_path holds 104 bytes including the NUL.
    macos_tmp = "/var/folders/zz/zyxvpxvq6csfxvn_n0000000000000/T"
    monkeypatch.setattr(shim.os, "getuid", lambda: 4294967294)
    monkeypatch.setattr(tempfile, "tempdir", macos_tmp)
    monkeypatch.setattr(shim, "runtime_dir", lambda env: shim._runtime_dir_path(env))
    path = shim.socket_path({})
    assert str(path).startswith(macos_tmp + "/quotable-4294967294/")
    assert len(os.fsencode(path)) < 104


def test_shim_imports_only_the_stdlib():
    probe = (
        "import sys; before = set(sys.modules); import quotable.shim; "
        "new = {m.split('.')[0] for m in set(sys.modules) - before}; "
        "print(sorted(new - set(sys.stdlib_module_names) - {'quotable'}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[]"


# --- the pump: a fake shared process on a real socket ------------------------------


class FakeConnection:
    def __init__(self, sock, number):
        self.sock = sock
        self.number = number
        self.lines = []  # parsed, hello first
        self.bye_after = None  # (count, gate): after `count` lines past the hello, bye
        self.closed = threading.Event()


class FakeProcess:
    """Stands in for the shared process: records lines, answers requests, says bye on cue.

    A request with method ``hold`` is answered only once ``release`` is set.
    """

    def __init__(self, path):
        self.path = path
        self.connections = []
        self.release = threading.Event()
        self.listening = threading.Event()
        self.reject_initialize = False
        self.next_bye = None
        self._server = None

    def start(self):
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self.path))
        self._server.listen()
        self.listening.set()
        threading.Thread(target=self._accept, daemon=True).start()

    def stop(self):
        if self._server is not None:
            self._server.close()
        for conn in self.connections:
            with contextlib.suppress(OSError):
                conn.sock.close()

    def _accept(self):
        while True:
            try:
                sock, _ = self._server.accept()
            except OSError:
                return
            conn = FakeConnection(sock, len(self.connections))
            conn.bye_after, self.next_bye = self.next_bye, None
            self.connections.append(conn)
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        reader = conn.sock.makefile("rb")
        try:
            for raw in reader:
                try:
                    message = json.loads(raw)
                except ValueError:
                    message = {"unparsed": raw.decode().strip()}
                conn.lines.append(message)
                if isinstance(message, dict) and "method" in message and "id" in message:
                    if message["method"] == "hold":
                        self.release.wait(10)
                    reply = {"method": message["method"], "conn": conn.number}
                    if message["method"] == "initialize" and self.reject_initialize:
                        error = {"code": -32602, "message": "unsupported protocol version"}
                        self._send(conn, {"jsonrpc": "2.0", "id": message["id"], "error": error})
                    else:
                        reply = {"jsonrpc": "2.0", "id": message["id"], "result": reply}
                        self._send(conn, reply)
                if conn.bye_after and len(conn.lines) - 1 == conn.bye_after[0]:
                    conn.bye_after[1].wait(10)
                    self._send(conn, {shim.BYE: conn.bye_after[0]})
                    conn.sock.shutdown(socket.SHUT_RDWR)
                    return
        except OSError:
            pass
        finally:
            conn.closed.set()

    def _send(self, conn, message):
        conn.sock.sendall(json.dumps(message).encode() + b"\n")

    def crash(self, number):
        conn = self.connections[number]
        conn.sock.shutdown(socket.SHUT_RDWR)


class Child:
    def __init__(self, code=None):
        self.code = code

    def poll(self):
        return self.code


class Session:
    """A shim on pipes, run in a thread, as Claude Code would drive it."""

    def __init__(self, env, spawn, spawn_timeout_s=5.0):
        in_r, self.in_w = os.pipe()
        self.out_r, out_w = os.pipe()
        self.shim = shim.Shim(
            stdin_fd=in_r,
            stdout=os.fdopen(out_w, "wb"),
            env=env,
            cwd="/work/project",
            spawn=spawn,
            spawn_timeout_s=spawn_timeout_s,
        )
        self.result = None
        self._buf = b""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        self.result = self.shim.run()

    def send(self, message):
        raw = message if isinstance(message, bytes) else json.dumps(message).encode()
        os.write(self.in_w, raw + b"\n")

    def recv(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buf:
            left = deadline - time.monotonic()
            assert left > 0, "no line from the shim"
            if select.select([self.out_r], [], [], left)[0]:
                self._buf += os.read(self.out_r, 65536)
        raw, self._buf = self._buf.split(b"\n", 1)
        return json.loads(raw)

    def silent(self, seconds=0.3):
        return not select.select([self.out_r], [], [], seconds)[0] and b"\n" not in self._buf

    def close(self):
        os.close(self.in_w)
        self.thread.join(5)
        assert not self.thread.is_alive()


def _request(id, method="ping", **params):
    return {"jsonrpc": "2.0", "id": id, "method": method, "params": params}


INITIALIZE = _request(
    0,
    "initialize",
    protocolVersion="2025-06-18",
    capabilities={},
    clientInfo={"name": "claude-code", "version": "2"},
)
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition never held"
        time.sleep(0.01)


@pytest.fixture
def pump(monkeypatch, tmp_path):
    short = Path(tempfile.mkdtemp(prefix="qs"))
    path = short / "k.sock"
    monkeypatch.setattr(shim, "socket_path", lambda env=None: path)
    fake = FakeProcess(path)
    env = {"XDG_STATE_HOME": str(tmp_path / "state")}
    sessions = []
    spawns = []

    def spawn(env):
        spawns.append(env)
        if not fake.listening.is_set():
            fake.start()
        return Child()

    def open_session(**kwargs):
        session = Session(env, kwargs.pop("spawn", spawn), **kwargs)
        sessions.append(session)
        return session

    yield SimpleNamespace(fake=fake, path=path, env=env, spawns=spawns, open=open_session)
    for session in sessions:
        with contextlib.suppress(OSError):
            os.close(session.in_w)
    fake.stop()
    shutil.rmtree(short, ignore_errors=True)


def _handshake(session):
    session.send(INITIALIZE)
    assert session.recv()["id"] == 0
    session.send(INITIALIZED)


def test_relays_requests_and_sends_the_cwd_hello(pump):
    s = pump.open()
    _handshake(s)
    s.send(_request(1, "tools/list"))
    assert s.recv() == {"jsonrpc": "2.0", "id": 1, "result": {"method": "tools/list", "conn": 0}}
    assert pump.fake.connections[0].lines[0] == {"cwd": "/work/project"}
    assert [m.get("method") for m in pump.fake.connections[0].lines[1:]] == [
        "initialize",
        "notifications/initialized",
        "tools/list",
    ]
    assert len(pump.spawns) == 1
    s.close()
    assert s.result == 0


def test_connects_without_spawning_when_the_process_is_up(pump):
    pump.fake.start()
    s = pump.open()
    _handshake(s)
    assert pump.spawns == []
    s.close()


def test_stdin_eof_exits_without_spawning(pump):
    s = pump.open()
    s.close()
    assert s.result == 0 and pump.spawns == []


def test_non_json_line_is_forwarded_as_is(pump):
    s = pump.open()
    _handshake(s)
    s.send(b"{oops")
    s.send(_request(1))
    assert s.recv()["id"] == 1
    assert pump.fake.connections[0].lines[3] == {"unparsed": "{oops"}


def test_bye_with_everything_read_reconnects_lazily_and_replays_initialize(pump):
    s = pump.open()
    gate = threading.Event()
    gate.set()
    pump.fake.next_bye = (3, gate)
    _handshake(s)
    s.send(_request(1))
    assert s.recv()["id"] == 1
    pump.fake.connections[0].closed.wait(5)
    assert s.silent()
    assert len(pump.fake.connections) == 1  # nothing to resend: no reconnect yet

    s.send(_request(2, "tools/list"))
    assert s.recv() == {"jsonrpc": "2.0", "id": 2, "result": {"method": "tools/list", "conn": 1}}
    second = pump.fake.connections[1].lines
    assert second[0] == {"cwd": "/work/project"}
    replay = second[1]
    assert replay["method"] == "initialize" and replay["params"] == INITIALIZE["params"]
    assert isinstance(replay["id"], str) and replay["id"].startswith("quotable-shim-")
    assert second[2:] == [INITIALIZED, _request(2, "tools/list")]
    assert s.silent()  # the replayed initialize's response was dropped


def test_bye_resends_exactly_the_unread_lines(pump):
    s = pump.open()
    gate = threading.Event()
    pump.fake.next_bye = (3, gate)
    _handshake(s)
    s.send(_request(1))  # line 3: read, answered, then the fake waits on the gate
    assert s.recv()["id"] == 1
    s.send(_request(2, "tools/list"))  # never read by the first connection
    s.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 9}})
    time.sleep(0.1)
    gate.set()
    assert s.recv() == {"jsonrpc": "2.0", "id": 2, "result": {"method": "tools/list", "conn": 1}}
    _wait(lambda: len(pump.fake.connections[1].lines) == 5)
    second = pump.fake.connections[1].lines
    assert second[1]["method"] == "initialize" and second[1]["id"] != 0
    assert second[2:] == [
        INITIALIZED,
        _request(2, "tools/list"),
        {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 9}},
    ]
    assert [m.get("id") for m in pump.fake.connections[0].lines if "id" in m] == [0, 1]


def test_an_unread_initialize_is_resent_not_replayed(pump):
    s = pump.open()
    gate = threading.Event()
    pump.fake.start()
    pump.fake.next_bye = (1, gate)  # read one line, then wait for the gate and say bye
    s.send(_request(5, "tools/list"))  # modern era: no initialize at all
    assert s.recv()["id"] == 5
    s.send(INITIALIZE)
    gate.set()
    assert s.recv()["id"] == 0
    second = pump.fake.connections[1].lines
    assert second[1:] == [INITIALIZE]  # the client's own, once, with its own id


def test_no_initialize_means_no_replay(pump):
    s = pump.open()
    gate = threading.Event()
    gate.set()
    pump.fake.next_bye = (1, gate)
    s.send(_request(1, "tools/list"))
    assert s.recv()["id"] == 1
    pump.fake.connections[0].closed.wait(5)
    s.send(_request(2, "tools/list"))
    assert s.recv()["id"] == 2
    assert pump.fake.connections[1].lines[1:] == [_request(2, "tools/list")]


def test_a_failed_initialize_is_not_replayed(pump):
    pump.fake.reject_initialize = True
    s = pump.open()
    s.send(INITIALIZE)
    assert "error" in s.recv()
    pump.fake.reject_initialize = False
    pump.fake.crash(0)
    s.send(_request(3))
    assert s.recv()["id"] == 3
    assert pump.fake.connections[1].lines[1:] == [_request(3)]


def test_crash_fails_in_flight_calls_and_the_next_call_reconnects(pump):
    s = pump.open()
    _handshake(s)
    s.send(_request(7, "hold"))
    _wait(lambda: len(pump.fake.connections[0].lines) == 4)
    s.send({"jsonrpc": "2.0", "id": "server-req-1", "result": {}})  # a reply, not a request
    s.send({"jsonrpc": "2.0", "method": "notifications/progress", "params": {}})
    _wait(lambda: s.shim._sent == 5)  # both reached the first connection before it dies
    pump.fake.crash(0)
    error = s.recv()
    assert error["id"] == 7 and error["error"]["code"] == shim.ERROR_CODE
    assert "may or may not have run" in error["error"]["message"]
    assert str(shim.log_path(pump.env)) in error["error"]["message"]
    assert s.silent()  # nothing for the reply or the notification
    pump.fake.release.set()

    s.send(_request(8))
    assert s.recv()["result"]["conn"] == 1
    second = pump.fake.connections[1].lines
    assert second[1]["method"] == "initialize" and second[2:] == [INITIALIZED, _request(8)]


def test_spawn_retries_after_a_lock_loser_exits_zero(pump):
    calls = []

    def spawn(env):
        calls.append(time.monotonic())
        if len(calls) == 1:
            return Child(code=0)  # lost the lock to a process on its way out
        pump.fake.start()
        return Child()

    s = pump.open(spawn=spawn)
    _handshake(s)
    assert len(calls) == 2 and calls[1] - calls[0] >= shim.RESPAWN_GAP_S


def test_spawn_failure_answers_with_an_error_naming_the_log(pump):
    s = pump.open(spawn=lambda env: Child(code=3))
    s.send(INITIALIZE)
    error = s.recv()
    assert error["id"] == 0 and "exited with code 3" in error["error"]["message"]
    assert str(shim.log_path(pump.env)) in error["error"]["message"]


def test_spawn_timeout_answers_with_an_error_and_never_unlinks(pump):
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(pump.path))
    stale.close()  # a crashed process's socket file: connect is refused
    s = pump.open(spawn=lambda env: Child(), spawn_timeout_s=0.3)
    s.send(_request(4))
    error = s.recv()
    assert error["id"] == 4 and "did not start within 0.3 s" in error["error"]["message"]
    assert pump.path.exists()
    s.send({"jsonrpc": "2.0", "method": "notifications/initialized"})  # no id: no answer
    assert s.silent()


def test_socket_path_is_recomputed_on_every_connect(pump, monkeypatch):
    calls = []
    real = shim.socket_path
    monkeypatch.setattr(shim, "socket_path", lambda env=None: calls.append(1) or real(env))
    s = pump.open()
    gate = threading.Event()
    gate.set()
    pump.fake.next_bye = (1, gate)
    s.send(_request(1))
    assert s.recv()["id"] == 1
    pump.fake.connections[0].closed.wait(5)
    before = len(calls)
    s.send(_request(2))
    assert s.recv()["id"] == 2
    assert len(calls) > before


def test_a_failed_write_still_finds_the_bye(pump, tmp_path):
    pump.fake.start()
    ours, theirs = socket.socketpair()
    out = io.BytesIO()
    sh = shim.Shim(stdin_fd=-1, stdout=out, env=pump.env, cwd="/w", spawn=lambda env: Child())
    sh.conn = ours
    theirs.sendall(json.dumps({shim.BYE: 0}).encode() + b"\n")
    theirs.close()
    sh._from_client(json.dumps(_request(1)).encode())  # EPIPE, then the bye is drained
    _wait(lambda: pump.fake.connections and len(pump.fake.connections[0].lines) == 2)
    assert pump.fake.connections[0].lines[1:] == [_request(1)]
    _wait(lambda: sh.conn is not None and sh.conn.recv(1, socket.MSG_PEEK))
    sh._read_conn()
    assert json.loads(out.getvalue()) == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"method": "ping", "conn": 0},
    }


def test_spawn_daemon_detaches_and_logs(monkeypatch, tmp_path):
    seen = {}

    class FakePopen:
        def __init__(self, args, **kwargs):
            seen.update(args=args, **kwargs)
            seen["stderr_name"] = kwargs["stderr"].name

    monkeypatch.setattr(shim.subprocess, "Popen", FakePopen)
    env = {"XDG_STATE_HOME": str(tmp_path), "QUOTABLE_MODEL": "tiny"}
    shim.spawn_daemon(env)
    assert seen["args"] == [sys.executable, "-m", "quotable.cli", "daemon"]
    assert seen["stdin"] is subprocess.DEVNULL and seen["stdout"] is subprocess.DEVNULL
    assert seen["stderr_name"] == str(tmp_path / "quotable" / "daemon.log")
    assert seen["start_new_session"] is True
    assert seen["env"] == env
