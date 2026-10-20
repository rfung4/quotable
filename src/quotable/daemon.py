"""The shared process: one per user and socket key, serving every session's shim.

The shim spawns it as ``quotable daemon``. It holds ``<socket>.lock`` for life, and only the
lock holder binds or unlinks the socket, so two processes never serve one key.

Wire, per connection: a hello line ``{"cwd": "/abs/path"}``, then newline-delimited JSON-RPC
in both directions, served as its own MCP session on one shared server (one model, one GPU
queue). When nothing is in flight and either ``QUOTABLE_MODEL_TTL`` has passed since the last
message or the code's socket key has changed, it stops reading every connection, sends
``{"quotable_bye": <lines read>}`` on each, unlinks the socket and exits. Lines past that
count were never read, so the shim resends them to the next process without running anything
twice.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import anyio
from anyio.abc import SocketStream
from anyio.streams.buffered import BufferedByteReceiveStream
from anyio.streams.memory import MemoryObjectSendStream
from mcp.shared.message import SessionMessage
from mcp.types import (
    JSONRPCError,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    jsonrpc_message_adapter,
)

from quotable import shim
from quotable.models import QuotableError

MAX_LINE = 16 * 1024 * 1024
TICK_S = 1.0
DRAIN_S = 0.2
BYE = "quotable_bye"
# Requests held open for the whole session (Claude Code opens one at start): not calls, so
# they never keep the process from going idle.
STREAMS = frozenset({"subscriptions/listen"})
# The connection's hello cwd; request handlers inherit it from the connection's task.
# ``None`` when serving in-process, where the process cwd is the session's.
SESSION_CWD: ContextVar[str | None] = ContextVar("quotable_session_cwd", default=None)


def take_lock(sock: Path) -> IO[str] | None:
    """The lifetime lock; ``None`` when another process holds it. Keep the file open."""
    import fcntl  # POSIX only, and Windows imports this module (``server`` uses SESSION_CWD)

    lock = open(f"{sock}.lock", "a")  # held until the process exits
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    return lock


def detach(log: Path) -> None:
    """No terminal and no wire: fd 0 reads EOF, fd 1 and 2 append to the log, cwd is ``/``.

    Children (ffmpeg, deno) inherit these fds. ``/`` as cwd makes a relative path that missed
    the session's cwd fail loudly instead of resolving somewhere plausible.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(fd, 2)
    os.close(fd)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    os.dup2(2, 1)
    os.chdir("/")


def lowlevel_server(app: Any) -> Any:
    """``MCPServer._lowlevel_server`` is private SDK API; pinned by ``mcp<2.4``, checked here."""
    lowlevel = getattr(app, "_lowlevel_server", None)
    if not (
        callable(getattr(lowlevel, "run", None))
        and callable(getattr(lowlevel, "create_initialization_options", None))
    ):
        raise QuotableError(
            "This mcp version has no MCPServer._lowlevel_server.run, which the shared "
            "process needs.",
            fix="install mcp>=2.3,<2.4, or set QUOTABLE_DAEMON=off to serve in-process",
        )
    return lowlevel


def _log(text: str) -> None:
    print(f"quotable daemon[{os.getpid()}]: {text}", file=sys.stderr, flush=True)


@dataclass(eq=False)
class _Connection:
    stream: SocketStream
    cwd: str | None = None
    lines_read: int = 0
    in_flight: set[str | int] = field(default_factory=set)
    reading: anyio.CancelScope = field(default_factory=anyio.CancelScope)
    send_lock: anyio.Lock = field(default_factory=anyio.Lock)

    async def send_line(self, text: str) -> None:
        async with self.send_lock:
            await self.stream.send(text.encode() + b"\n")


class Daemon:
    """Serves ``lowlevel`` (an SDK low-level server) on a Unix socket until idle or outdated."""

    def __init__(
        self,
        lowlevel: Any,
        sock: Path,
        *,
        key: str,
        current_key: Callable[[], str],
        ttl_s: float,
        busy: Callable[[], bool] = lambda: False,
        tick_s: float = TICK_S,
    ) -> None:
        self.lowlevel = lowlevel
        self.sock = sock
        self.key = key
        self.current_key = current_key
        self.ttl_s = ttl_s
        self.busy = busy
        self.tick_s = tick_s
        self.connections: set[_Connection] = set()
        self.exiting = False
        self.last_activity = 0.0

    def _touch(self) -> None:
        self.last_activity = anyio.current_time()

    async def serve(self) -> None:
        with suppress(FileNotFoundError):
            self.sock.unlink()  # a crashed holder's leftover; we hold the lock
        listener = await anyio.create_unix_listener(self.sock)
        self._touch()
        _log(f"serving {self.sock}")
        try:
            async with anyio.create_task_group() as tg:

                tg.start_soon(listener.serve, self._handle, tg)
                reason = await self._watch()
                await self._bye(reason)
                await anyio.sleep(DRAIN_S)  # accept, and bye, whatever was still in the backlog
                tg.cancel_scope.cancel()
        finally:
            with suppress(FileNotFoundError):
                self.sock.unlink()
            await listener.aclose()

    async def _watch(self) -> str:
        while True:
            await anyio.sleep(self.tick_s)
            if self.busy() or any(c.in_flight for c in self.connections):
                continue
            try:
                current = self.current_key()
            except OSError as exc:  # a source file replaced mid-checkout: look again next tick
                _log(f"could not check the key: {exc!r}")
                continue
            if current != self.key:
                reason = "outdated"
            elif self.ttl_s > 0 and anyio.current_time() - self.last_activity >= self.ttl_s:
                reason = "idle"
            else:
                continue
            # No await between the in-flight check and here: no line can be read in between.
            self.exiting = True
            with suppress(FileNotFoundError):
                self.sock.unlink()  # a shim connecting now spawns a fresh process instead
            for conn in self.connections:
                conn.reading.cancel()
            return reason

    async def _bye(self, reason: str) -> None:
        _log(f"exiting ({reason}); {len(self.connections)} connection(s) told to resend")
        for conn in list(self.connections):
            with suppress(anyio.BrokenResourceError, anyio.ClosedResourceError, OSError):
                with anyio.move_on_after(5):
                    await conn.send_line(json.dumps({BYE: conn.lines_read}))

    async def _handle(self, stream: SocketStream) -> None:
        conn = _Connection(stream)
        if self.exiting:  # accepted after the exit was decided: nothing read, nothing to resend
            async with stream:
                with suppress(anyio.BrokenResourceError, anyio.ClosedResourceError, OSError):
                    await conn.send_line(json.dumps({BYE: 0}))
                await anyio.sleep_forever()
        self.connections.add(conn)
        try:
            async with stream:
                buffered = BufferedByteReceiveStream(stream)
                with conn.reading:
                    try:
                        hello = json.loads(await buffered.receive_until(b"\n", MAX_LINE))
                    except (anyio.EndOfStream, anyio.IncompleteRead, ValueError) as exc:
                        _log(f"dropping a connection without a hello: {exc!r}")
                        return
                if conn.reading.cancel_called:
                    await anyio.sleep_forever()  # the bye reaches it with a count of 0
                cwd = hello.get("cwd") if isinstance(hello, dict) else None
                if not (isinstance(cwd, str) and os.path.isabs(cwd)):
                    _log(f"refusing a connection whose hello has no absolute cwd: {hello!r}")
                    return
                conn.cwd = cwd
                SESSION_CWD.set(cwd)
                await self._session(conn, buffered)
        except Exception as exc:  # one connection's failure, hello included, never ends the others'
            _log(f"dropped a connection after an error: {exc!r}")
        finally:
            self.connections.discard(conn)

    async def _session(self, conn: _Connection, buffered: BufferedByteReceiveStream) -> None:
        read_writer, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](0)
        write_stream, write_reader = anyio.create_memory_object_stream[SessionMessage](0)

        async with anyio.create_task_group() as tg:

            async def writer() -> None:
                async with write_reader:
                    async for session_message in write_reader:
                        message = session_message.message
                        line = message.model_dump_json(by_alias=True, exclude_unset=True)
                        try:
                            await conn.send_line(line)
                        except (anyio.BrokenResourceError, anyio.ClosedResourceError, OSError):
                            # The shim is gone: end this session, cancelling its calls, so a
                            # transcription it started stops and frees the GPU queue.
                            tg.cancel_scope.cancel()
                            return
                        if isinstance(message, JSONRPCResponse | JSONRPCError):
                            conn.in_flight.discard(message.id)
                            self._touch()

            tg.start_soon(self._reader, conn, buffered, read_writer)
            tg.start_soon(writer)
            await self.lowlevel.run(
                read_stream, write_stream, self.lowlevel.create_initialization_options()
            )
            tg.cancel_scope.cancel()

    async def _reader(
        self,
        conn: _Connection,
        buffered: BufferedByteReceiveStream,
        out: MemoryObjectSendStream[SessionMessage | Exception],
    ) -> None:
        async with out:
            with conn.reading:
                while True:
                    try:
                        line = await buffered.receive_until(b"\n", MAX_LINE)
                    except (anyio.EndOfStream, anyio.IncompleteRead):
                        return
                    try:
                        item: SessionMessage | Exception = SessionMessage(
                            jsonrpc_message_adapter.validate_json(line, by_name=False)
                        )
                    except Exception as exc:
                        item = exc
                    await out.send(item)
                    # Counted once delivered: a line cut off by the bye is resent, not lost.
                    conn.lines_read += 1
                    self._touch()
                    message = item.message if isinstance(item, SessionMessage) else None
                    if isinstance(message, JSONRPCRequest) and message.method not in STREAMS:
                        conn.in_flight.add(message.id)
                    elif (
                        isinstance(message, JSONRPCNotification)
                        and message.method == "notifications/cancelled"
                    ):
                        # A cancelled request gets no response; its thread still counts
                        # through ``busy`` until it stops.
                        conn.in_flight.discard((message.params or {}).get("requestId"))
            # Stopped by the bye: hold the session open until the process exits, so the
            # server does not see EOF and the shim does not see the socket close first.
            await anyio.sleep_forever()


def run() -> None:
    """Entry point for ``quotable daemon``; returns at once if another process holds the key."""
    sock = shim.socket_path()
    lock = take_lock(sock)
    if lock is None:
        return
    detach(shim.log_path())

    import quotable.transcribe as tx
    from quotable.config import Settings
    from quotable.server import create_server

    settings = Settings.from_env()
    lowlevel = lowlevel_server(create_server(settings))
    daemon = Daemon(
        lowlevel,
        sock,
        key=sock.name.removesuffix(".sock"),
        current_key=shim.socket_key,
        ttl_s=settings.model_ttl,
        busy=tx.INFERENCE_LOCK.locked,
    )
    try:
        anyio.run(daemon.serve)
    finally:
        lock.close()
