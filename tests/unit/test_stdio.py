"""Pins the SDK's fd-level stdio isolation, which quotable relies on instead of its own.

If an `mcp` upgrade stops repointing fd 0/1 while serving, children such as ffmpeg inherit
the JSON-RPC pipe: `cat` blocks on stdin and `echo` corrupts the wire.
"""

import json
import sys
from pathlib import Path

import anyio
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

PROBE = Path(__file__).with_name("stdio_probe.py")


async def _run_probe(errlog_path: Path) -> tuple[list[dict], list[str]]:
    params = StdioServerParameters(command=sys.executable, args=[str(PROBE)])
    results: list[dict] = []
    tools: list[str] = []
    with errlog_path.open("w") as errlog:
        with anyio.fail_after(30):
            async with Client(stdio_client(params, errlog=errlog)) as client:
                tools = [t.name for t in (await client.list_tools()).tools]
                for _ in range(2):
                    result = await client.call_tool("spawn", {})
                    assert not result.is_error
                    results.append(json.loads(result.content[0].text))
    return results, tools


def test_children_get_eof_on_stdin_and_stdout_goes_to_stderr(tmp_path):
    errlog_path = tmp_path / "server.stderr"
    results, tools = anyio.run(_run_probe, errlog_path)

    assert tools == ["spawn"]
    for r in results:
        assert r["cat_stdout"] == ""
        assert r["cat_seconds"] < 1.0
    # A second call succeeding on the same session proves the pipe was not corrupted.
    assert len(results) == 2
    assert errlog_path.read_text().count("leak") == 2
