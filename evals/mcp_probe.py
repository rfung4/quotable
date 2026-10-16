"""The MCP round trip against a serve command, and a dump of the surface a client sees.

``python -m evals.mcp_probe`` builds the wheel and serves it with ``uvx`` (so it proves the
wheel, not the checkout), transcribes ``tests/fixtures/jfk.flac`` over stdio, pages it, reads
its resource, and writes ``evals/results/mcp_surface.json``: the tools with their schemas and
annotations, the resource templates, the server instructions and the round trip's outcome.
``--command`` serves something else instead. ``--check`` replays that file and the in-client
checklist, ``evals/results/mcp_checklist.md``.

The child gets ``os.environ`` plus ``QUOTABLE_MODEL=tiny``, ``QUOTABLE_DEVICE=cpu`` and a
temporary ``QUOTABLE_OUTPUT_DIR``: the SDK otherwise hands it only ``HOME``, ``PATH`` and the
like, and the wheel would default to ``turbo`` on ``cuda``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import anyio
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

from quotable.server import UNTRUSTED

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"
SURFACE = RESULTS / "mcp_surface.json"
CHECKLIST = RESULTS / "mcp_checklist.md"
JFK = ROOT / "tests" / "fixtures" / "jfk.flac"
TOOLS = 3
TEMPLATES = 2
TIMEOUT_S = 300


def _wheel_command() -> list[str]:
    subprocess.run(["uv", "build"], cwd=ROOT, check=True, stdin=subprocess.DEVNULL)
    version = subprocess.run(
        ["uv", "version", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    wheel = ROOT / "dist" / f"quotable-{version}-py3-none-any.whl"
    # --refresh-package: a rebuilt wheel keeps its version, and uvx would serve the cached one.
    return ["uvx", "--refresh-package", "quotable", "--from", str(wheel), "quotable", "serve"]


def _dump(model) -> dict:
    return model.model_dump(mode="json", by_alias=True, exclude_none=True)


async def _probe(command: list[str], out_dir: Path, errlog_path: Path) -> dict:
    env = dict(os.environ)
    env.update(QUOTABLE_MODEL="tiny", QUOTABLE_DEVICE="cpu", QUOTABLE_OUTPUT_DIR=str(out_dir))
    params = StdioServerParameters(command=command[0], args=command[1:], env=env)
    progress: list[str | None] = []

    async def on_progress(done, total, message):
        progress.append(message)

    with errlog_path.open("w") as errlog, anyio.fail_after(TIMEOUT_S):
        async with Client(stdio_client(params, errlog=errlog)) as client:
            tools = await client.list_tools()
            templates = await client.list_resource_templates()
            result = await client.call_tool(
                "transcribe", {"source": str(JFK)}, progress_callback=on_progress
            )
            tid = (result.structured_content or {}).get("id")
            page = await client.call_tool("read_transcript", {"id": tid}) if tid else None
            resource = await client.read_resource(f"transcript://{tid}") if tid else None
            instructions = client.instructions
            server_info = _dump(client.server_info)

    text = result.content[0].text if result.content else ""
    page_text = (page.structured_content or {}).get("text", "") if page else ""
    body = resource.contents[0].text if resource else ""
    checks = {
        "transcribe_ok": not result.is_error and bool(tid),
        "untrusted_first_line": text.startswith(UNTRUSTED),
        # What a client that forwards only structured content shows: the notice, first.
        "untrusted_structured": all(
            list(r.structured_content or {})[:1] == ["notice"]
            and r.structured_content["notice"] == UNTRUSTED
            for r in (result, page)
            if r is not None
        ),
        "progress_messages": sum(1 for m in progress if m),
        "page_ok": page is not None and not page.is_error and "ask not" in page_text.lower(),
        "resource_ok": body.startswith("[00:00:00]") and "ask not" in body.lower(),
        "output_dir_used": bool(tid) and (out_dir / tid.split(":")[0]).is_dir(),
    }
    ok = all(checks.values())
    return {
        "server_info": server_info,
        "instructions": instructions,
        "tools": [_dump(t) for t in tools.tools],
        "resource_templates": [_dump(t) for t in templates.resource_templates],
        "round_trip": {"ok": ok, "command": command, "transcript_id": tid, "checks": checks},
    }


def check() -> list[str]:
    problems = []
    if not SURFACE.exists():
        return [f"{SURFACE} missing; run python -m evals.mcp_probe"]
    surface = json.loads(SURFACE.read_text())
    if len(surface.get("tools", [])) != TOOLS:
        problems.append(f"{len(surface.get('tools', []))} tools, want {TOOLS}")
    if len(surface.get("resource_templates", [])) != TEMPLATES:
        problems.append(f"{len(surface.get('resource_templates', []))} templates, want {TEMPLATES}")
    if not surface.get("round_trip", {}).get("ok"):
        problems.append(f"round trip failed: {surface.get('round_trip')}")
    if not CHECKLIST.exists():
        problems.append(f"{CHECKLIST} missing; the in-client checklist has not been sat")
    else:
        open_items = [
            line for line in CHECKLIST.read_text().splitlines() if line.lstrip().startswith("- [ ]")
        ]
        problems += [f"checklist open: {line.strip()}" for line in open_items]
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--command", help="serve command to probe (default: the built wheel)")
    parser.add_argument("--check", action="store_true", help="replay the recorded files only")
    args = parser.parse_args(argv)

    if args.check:
        problems = check()
        for p in problems:
            print(p)
        print(f"mcp probe: {len(problems)} problems")
        return 1 if problems else 0

    command = shlex.split(args.command) if args.command else _wheel_command()
    RESULTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="quotable-mcp-probe-") as tmp:
        errlog = RESULTS / "mcp_probe.stderr"
        surface = anyio.run(_probe, command, Path(tmp), errlog)
    SURFACE.write_text(json.dumps(surface, indent=2) + "\n")
    rt = surface["round_trip"]
    print(json.dumps(rt["checks"]))
    print(
        f"{len(surface['tools'])} tools, {len(surface['resource_templates'])} templates, "
        f"round trip {'ok' if rt['ok'] else 'FAILED'} -> {SURFACE.relative_to(ROOT)}"
    )
    return 0 if rt["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
