"""Minimal stdio MCP server used by test_stdio.py.

Its one tool spawns children the way yt-dlp does: no stdin/stdout overrides.
"""

import json
import subprocess
import time

from mcp.server import MCPServer

server = MCPServer("probe")


@server.tool()
def spawn() -> str:
    started = time.monotonic()
    cat = subprocess.run(["cat"], capture_output=True, timeout=5)
    elapsed = time.monotonic() - started
    subprocess.run(["sh", "-c", "echo leak"], timeout=5)
    return json.dumps({"cat_stdout": cat.stdout.decode(), "cat_seconds": elapsed})


if __name__ == "__main__":
    server.run("stdio")
