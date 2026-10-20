"""Stdio shim: every ``quotable serve`` talks to one shared process per user over a socket.

Stdlib imports only: the shim runs once per Claude Code session, so it stays small.
The socket name is a hash of everything that changes what the shared process would do,
so different code, dependencies or settings never share a process.

The pump relays newline-delimited JSON-RPC between stdin/stdout and the socket, parsing lines
only to track in-flight request ids and the client's ``initialize``. When the shared process
exits on purpose it first sends ``{"quotable_bye": N}``: lines after the first N were never
read, so the shim resends them to a fresh process, after replaying ``initialize`` under a
private id. A process that dies without a bye fails each in-flight call instead; nothing is
retried, since those calls may have run.
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import socket
import stat
import subprocess
import sys
import sysconfig
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from quotable import DIST

PACKAGE_DIR = Path(__file__).resolve().parent
# Settings outside the QUOTABLE_ prefix that change a transcript or where it is stored.
_KEY_ENV = ("HF_TOKEN", "XDG_DATA_HOME")
# Chooses shim vs in-process; the shared process never reads it.
_NOT_IN_KEY = ("QUOTABLE_DAEMON",)


class ShimError(Exception):
    """Same shape as ``QuotableError``, which the shim cannot import (it pulls in pydantic)."""

    def __init__(self, message: str, fix: str) -> None:
        super().__init__(message)
        self.message = message
        self.fix = fix


def _version() -> str:
    """From the dist-info directory's name: importlib.metadata would cost the shim ~6 MB."""
    prefix, suffix = DIST.replace("-", "_") + "-", ".dist-info"
    for entry in sys.path:
        try:
            names = os.listdir(entry or ".")
        except OSError:
            continue
        for name in names:
            if name.startswith(prefix) and name.endswith(suffix):
                return name[len(prefix) : -len(suffix)]
    return "unknown"


def _site_packages() -> str:
    return sysconfig.get_paths()["purelib"]


def _mtime_ns(path: str | Path) -> int | None:
    """``None`` for a file listed but gone by the stat (mid-checkout, a dangling symlink)."""
    try:
        return os.stat(path).st_mtime_ns
    except FileNotFoundError:
        return None


def _newest_source_mtime() -> int:
    mtimes = (_mtime_ns(p) for p in PACKAGE_DIR.glob("*.py"))
    return max((m for m in mtimes if m is not None), default=0)


def _installed_dists(site: str) -> list[str]:
    """Each installed distribution with its mtime: installs, upgrades and removals change this.

    Not site-packages' own mtime: the first import after an install writes
    site-packages/__pycache__, which moved it and so the key.
    """
    try:
        entries = [e for e in os.scandir(site) if e.name.endswith(".dist-info")]
    except OSError:
        return []
    stamped = ((e.name, _mtime_ns(e.path)) for e in entries)
    return sorted(f"{name}@{mtime}" for name, mtime in stamped if mtime is not None)


def socket_key(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    settings = sorted(
        (name, value)
        for name, value in env.items()
        if (name.startswith("QUOTABLE_") and name not in _NOT_IN_KEY) or name in _KEY_ENV
    )
    site = _site_packages()
    parts = [
        _version(),
        str(PACKAGE_DIR),
        str(_newest_source_mtime()),
        sys.prefix,
        site,
        *_installed_dists(site),
        *(f"{name}={value}" for name, value in settings),
    ]
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def _runtime_dir_path(env: Mapping[str, str]) -> Path:
    xdg = env.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "quotable"
    return Path(tempfile.gettempdir()) / f"quotable-{os.getuid()}"


def runtime_dir(env: Mapping[str, str] | None = None) -> Path:
    """Create the socket dir 0700, refusing one another user could have planted or can read."""
    path = _runtime_dir_path(os.environ if env is None else env)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    st = path.lstat()
    if not stat.S_ISDIR(st.st_mode):
        raise ShimError(
            f"{path} is not a directory (a symlink or a file)",
            fix=f"remove {path}, or set XDG_RUNTIME_DIR to a private directory",
        )
    if st.st_uid != os.getuid():
        raise ShimError(
            f"{path} is owned by another user",
            fix="set XDG_RUNTIME_DIR to a private directory you own",
        )
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise ShimError(
            f"{path} is accessible to other users",
            fix=f"chmod 700 {path}",
        )
    return path


def log_path(env: Mapping[str, str] | None = None) -> Path:
    """Where the shared process's stderr goes; it has no terminal and no MCP log."""
    env = os.environ if env is None else env
    state = env.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "quotable" / "daemon.log"


def socket_path(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return runtime_dir(env) / f"{socket_key(env)}.sock"


# --- the pump ----------------------------------------------------------------------

BYE = "quotable_bye"  # must match daemon.BYE; not imported, the daemon pulls in the SDK
SPAWN_TIMEOUT_S = 30.0
POLL_S = 0.05
RESPAWN_GAP_S = 0.5
DRAIN_TIMEOUT_S = 2.0
SEND_TIMEOUT_S = 10.0
ERROR_CODE = -32000  # JSON-RPC implementation-defined server error


@dataclass(eq=False)
class _Line:
    raw: bytes
    replay: bool = False  # sent by the shim itself; redone, never resent
    request_id: Any = None
    has_id: bool = False
    index: int = -1  # position on the current connection, after the hello


def _key(request_id: Any) -> str:
    return json.dumps(request_id)  # 1 and "1" are different ids


def spawn_daemon(env: Mapping[str, str]) -> subprocess.Popen:
    """Start the shared process detached from this session: no stdio, its own session."""
    log = log_path(env)
    log.parent.mkdir(parents=True, exist_ok=True)
    # The log from the start, so an import error before the process redirects is kept.
    with open(log, "ab") as stderr:
        return subprocess.Popen(
            [sys.executable, "-m", "quotable.cli", "daemon"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            env=dict(env),
            start_new_session=True,
            cwd="/",  # `-m` puts the cwd on sys.path: a quotable/ there would shadow the install
        )


@dataclass
class Shim:
    stdin_fd: int
    stdout: BinaryIO
    env: Mapping[str, str]
    cwd: str
    spawn: Callable[[Mapping[str, str]], Any] = spawn_daemon
    spawn_timeout_s: float = SPAWN_TIMEOUT_S

    conn: socket.socket | None = None
    _rbuf: bytes = b""
    _sent: int = 0
    _pending: list[_Line] = field(default_factory=list)  # sent here, not yet known read
    _held: list[_Line] = field(default_factory=list)  # waiting for the replayed initialize
    _in_flight: dict[str, _Line] = field(default_factory=dict)
    _init: dict | None = None  # the client's initialize, once it succeeded
    _init_line: _Line | None = None
    _init_candidate: tuple[str, dict, _Line] | None = None
    _initialized_line: _Line | None = None
    _replay_id: str | None = None
    _replays: int = 0
    _children: list[Any] = field(default_factory=list)

    # --- entry -------------------------------------------------------------------

    def run(self) -> int:
        ibuf = b""
        while True:
            self._children = [c for c in self._children if c.poll() is None]  # reap
            fds: list[Any] = [self.stdin_fd] + ([self.conn] if self.conn else [])
            readable, _, _ = select.select(fds, [], [])
            if self.conn is not None and self.conn in readable:
                self._read_conn()
            if self.stdin_fd in readable:
                data = os.read(self.stdin_fd, 65536)
                if not data:
                    self._close()
                    return 0
                ibuf += data
                *lines, ibuf = ibuf.split(b"\n")
                for raw in lines:
                    if raw.strip():
                        self._from_client(raw)

    # --- client → process --------------------------------------------------------

    def _from_client(self, raw: bytes) -> None:
        line = _Line(raw)
        try:
            message = json.loads(raw)
        except ValueError:
            message = None  # forwarded as is; the server answers with a parse error
        if isinstance(message, dict):
            if "method" in message and "id" in message:
                line.request_id, line.has_id = message["id"], True
                self._in_flight[_key(message["id"])] = line
            if message.get("method") == "initialize" and line.has_id:
                self._init_candidate = (_key(message["id"]), message, line)
            elif message.get("method") == "notifications/initialized":
                self._initialized_line = line
        if self.conn is None:
            self._connect([line])  # sends it, or answers it with an error
        else:
            self._send(line)

    def _send(self, line: _Line) -> None:
        if self.conn is None:
            return
        if self._replay_id is not None and not line.replay:
            self._held.append(line)
            return
        line.index = self._sent
        self._sent += 1
        self._pending.append(line)
        try:
            self.conn.sendall(line.raw + b"\n")
        except OSError:
            self._drain_after_failed_write()

    # --- process → client --------------------------------------------------------

    def _read_conn(self) -> None:
        assert self.conn is not None
        try:
            data = self.conn.recv(65536)
        except OSError:
            data = b""
        if not data:
            self._lost()
            return
        self._feed(data)

    def _feed(self, data: bytes) -> None:
        conn = self.conn
        self._rbuf += data
        while conn is self.conn and b"\n" in self._rbuf:
            raw, self._rbuf = self._rbuf.split(b"\n", 1)
            if raw.strip():
                self._from_process(raw)

    def _from_process(self, raw: bytes) -> None:
        try:
            message = json.loads(raw)
        except ValueError:
            message = None
        if isinstance(message, dict):
            if BYE in message and len(message) == 1:
                self._bye(int(message[BYE]))
                return
            if "method" not in message and "id" in message:
                if self._replay_id is not None and message["id"] == self._replay_id:
                    self._replay_done(message)
                    return
                self._answered(_key(message["id"]), "result" in message)
        self._write_out(raw)

    def _answered(self, key: str, ok: bool) -> None:
        line = self._in_flight.pop(key, None)
        if line is None:
            return
        if line.index >= 0:  # the process reads in order: everything up to it was read
            self._pending = [p for p in self._pending if p.index > line.index]
        if self._init_candidate is not None and self._init_candidate[0] == key:
            if ok:
                self._init, self._init_line = self._init_candidate[1], self._init_candidate[2]
            self._init_candidate = None

    def _write_out(self, raw: bytes) -> None:
        self.stdout.write(raw + b"\n")
        self.stdout.flush()

    def _error_out(self, line: _Line, message: str) -> None:
        if not line.has_id or self._in_flight.pop(_key(line.request_id), None) is None:
            return
        error = {"code": ERROR_CODE, "message": message}
        self._write_out(
            json.dumps({"jsonrpc": "2.0", "id": line.request_id, "error": error}).encode()
        )

    # --- connection lifecycle ----------------------------------------------------

    def _connect(self, lines: list[_Line]) -> bool:
        """Connect (spawning if needed), replay the handshake, then send ``lines`` in order."""
        try:
            conn = self._open()
            try:
                conn.sendall(json.dumps({"cwd": self.cwd}).encode() + b"\n")
            except OSError:
                conn.close()
                raise
        except (OSError, ShimError) as exc:
            reason = (
                f"{exc.message} Fix: {exc.fix}"
                if isinstance(exc, ShimError)
                else f"cannot reach the shared process: {exc}"
            )
            for line in lines:
                self._error_out(line, f"quotable: {reason}")
            return False
        self.conn, self._rbuf, self._sent, self._pending = conn, b"", 0, []
        resend = set(lines)
        replay_init = self._init is not None and self._init_line not in resend
        if replay_init:
            assert self._init is not None
            self._replays += 1
            self._replay_id = f"quotable-shim-replay-{self._replays}"
            replay = {**self._init, "id": self._replay_id}
            self._send(_Line(json.dumps(replay).encode(), replay=True))
            if self._initialized_line is not None and self._initialized_line not in resend:
                self._held.append(_Line(self._initialized_line.raw, replay=True))
        for line in lines:
            self._send(line)
        return True

    def _open(self) -> socket.socket:
        conn = self._try_connect()
        if conn is not None:
            return conn
        deadline = time.monotonic() + self.spawn_timeout_s
        child, spawned_at = self.spawn(self.env), time.monotonic()
        self._children.append(child)
        while time.monotonic() < deadline:
            time.sleep(POLL_S)
            conn = self._try_connect()
            if conn is not None:
                return conn
            code = child.poll()
            if code not in (None, 0):
                raise ShimError(
                    f"the shared process exited with code {code} before listening.",
                    fix=f"see {log_path(self.env)}, or set QUOTABLE_DAEMON=off",
                )
            # Exit 0: the lifetime lock was still held by a process on its way out.
            if code == 0 and time.monotonic() - spawned_at >= RESPAWN_GAP_S:
                child, spawned_at = self.spawn(self.env), time.monotonic()
                self._children.append(child)
        raise ShimError(
            f"the shared process did not start within {self.spawn_timeout_s:g} s.",
            fix=f"see {log_path(self.env)}, or set QUOTABLE_DAEMON=off",
        )

    def _try_connect(self) -> socket.socket | None:
        path = socket_path(self.env)  # recomputed: after an outdated bye the key is new
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            conn.connect(str(path))
        except (FileNotFoundError, ConnectionRefusedError):
            conn.close()
            return None
        except OSError:
            conn.close()
            raise
        conn.settimeout(SEND_TIMEOUT_S)
        return conn

    def _replay_done(self, message: dict) -> None:
        self._replay_id = None
        if "result" not in message:
            print(f"quotable: replaying initialize failed: {message}", file=sys.stderr)
        held, self._held = self._held, []
        for line in held:
            self._send(line)

    def _bye(self, count: int) -> None:
        """A planned exit: resend what the process never read; reconnect only if needed."""
        resend = [p for p in self._pending if p.index >= count and not p.replay]
        resend += [h for h in self._held if not h.replay]
        self._close()
        for line in resend:
            line.index = -1
        if resend:
            self._connect(resend)

    def _lost(self) -> None:
        """EOF or a reset with no bye: the process died; fail what was in flight."""
        log = log_path(self.env)
        self._close()
        for line in list(self._in_flight.values()):
            self._error_out(
                line,
                "quotable: the shared process exited during this call; it may or may not "
                f"have run. Fix: retry the call; see {log} if it keeps happening.",
            )
        self._init_candidate = None  # an initialize cut off here never succeeded

    def _drain_after_failed_write(self) -> None:
        """The write failed; a bye may still be waiting in the socket. Read it before deciding."""
        conn = self.conn
        assert conn is not None
        conn.settimeout(DRAIN_TIMEOUT_S)
        while conn is self.conn:
            try:
                data = conn.recv(65536)
            except OSError:
                data = b""
            if not data:
                if conn is self.conn:
                    self._lost()
                return
            self._feed(data)

    def _close(self) -> None:
        if self.conn is not None:
            self.conn.close()
        self.conn, self._rbuf, self._pending, self._held = None, b"", [], []
        self._replay_id = None


def main(env: Mapping[str, str] | None = None) -> int:
    """``quotable serve`` on POSIX: relay this session's stdio to the shared process."""
    env = dict(os.environ if env is None else env)
    return Shim(stdin_fd=0, stdout=sys.stdout.buffer, env=env, cwd=os.getcwd()).run()
