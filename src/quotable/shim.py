"""Stdio shim: every ``quotable serve`` talks to one shared process per user over a socket.

Stdlib imports only: the shim runs once per Claude Code session, so it stays small.
The socket name is a hash of everything that changes what the shared process would do,
so different code, dependencies or settings never share a process.
"""

from __future__ import annotations

import hashlib
import os
import stat
import sys
import sysconfig
import tempfile
from collections.abc import Mapping
from pathlib import Path

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
    from importlib.metadata import version

    return version(DIST)


def _site_packages() -> str:
    return sysconfig.get_paths()["purelib"]


def _newest_source_mtime() -> int:
    return max((p.stat().st_mtime_ns for p in PACKAGE_DIR.glob("*.py")), default=0)


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
        str(os.stat(site).st_mtime_ns),
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
