"""Every test gets a private runtime and state dir and an explicit ``QUOTABLE_DAEMON``.

Without these a test running ``quotable serve`` would reach the developer's own shared
process (same uid, same socket key) or leave one running (review W6). Tests that exercise the
shared process opt in with ``QUOTABLE_DAEMON=on``; any process they spawned is killed at
teardown, found by the pid in its log line.
"""

import os
import re
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

import pytest

_SERVING = re.compile(r"^quotable daemon\[(\d+)\]: serving ", re.M)


def _kill_spawned(log: Path) -> None:
    if not log.exists():
        return
    for pid in {int(p) for p in _SERVING.findall(log.read_text(errors="replace"))}:
        args = subprocess.run(["ps", "-o", "args=", "-p", str(pid)], capture_output=True, text=True)
        if "quotable.cli daemon" in args.stdout:  # still ours, not a reused pid
            os.kill(pid, signal.SIGKILL)


@pytest.fixture(autouse=True)
def isolated_shared_process(monkeypatch, tmp_path):
    runtime = Path(tempfile.mkdtemp(prefix="qrt"))  # short: AF_UNIX paths max out near 104 bytes
    state = tmp_path / "xdg-state"
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    monkeypatch.setenv("QUOTABLE_DAEMON", "off")
    yield
    _kill_spawned(state / "quotable" / "daemon.log")
    shutil.rmtree(runtime, ignore_errors=True)
