"""``quotable serve`` over real stdio: the real ``tiny`` model, progress, paging, resources.

Wrapped in ``fail_after`` so a server blocked on its stdin fails the test instead of
hanging CI.
"""

import os
import sys
from pathlib import Path

import anyio
import pytest
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

pytestmark = pytest.mark.integration

JFK = Path(__file__).parents[1] / "fixtures" / "jfk.flac"
QUOTABLE = Path(sys.executable).with_name("quotable")


async def _round_trip(errlog_path: Path) -> dict:
    # The SDK gives the child only HOME/PATH/SHELL/TERM/USER/LOGNAME; forward the
    # QUOTABLE_* settings from conftest, or it uses the default model and cache dir.
    params = StdioServerParameters(command=str(QUOTABLE), args=["serve"], env=dict(os.environ))
    progress: list[tuple[float, float | None, str | None]] = []

    async def on_progress(done, total, message):
        progress.append((done, total, message))

    with errlog_path.open("w") as errlog, anyio.fail_after(120):
        async with Client(stdio_client(params, errlog=errlog)) as client:
            result = await client.call_tool(
                "transcribe", {"source": str(JFK)}, progress_callback=on_progress
            )
            assert not result.is_error, result.content
            tid = result.structured_content["id"]
            page = await client.call_tool("read_transcript", {"id": tid})
            assert not page.is_error, page.content
            resource = await client.read_resource(f"transcript://{tid}")
    return {
        "result": result,
        "page": page,
        "resource": resource.contents[0],
        "progress": progress,
    }


def test_serve_transcribes_pages_and_serves_the_resource_over_stdio(tmp_path):
    out = anyio.run(_round_trip, tmp_path / "server.stderr")

    result = out["result"].structured_content
    assert result["method"] == "whisper"
    assert result["cached"] is False
    assert "ask not what your country" in result["text"].lower()

    assert any("ask not" in (message or "").lower() for _, _, message in out["progress"])

    page = out["page"].structured_content
    assert "ask not what your country" in page["text"].lower()
    assert page["next_offset"] is None

    assert out["resource"].mime_type == "text/plain"
    assert out["resource"].text.startswith("[00:00:00]")
    assert "ask not what your country" in out["resource"].text.lower()

    # The variant was written to the output dir the env pointed the server at.
    assert (Path(os.environ["QUOTABLE_OUTPUT_DIR"]) / result["id"].split(":")[0]).is_dir()
