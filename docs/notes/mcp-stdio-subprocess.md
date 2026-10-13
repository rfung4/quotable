# MCP over stdio and child processes

> Why `quotable serve` never touches the server's stdin or stdout, and what
> breaks the moment someone does.

## The failure

A stdio MCP server speaks JSON-RPC on fd 0 and fd 1. Anything else that reads
fd 0 or writes fd 1 shares the wire with the client:

- a child that reads stdin blocks on the client's next request, or eats it;
- a child that prints to stdout injects bytes into the JSON-RPC stream, and the
  client drops the connection on the first unparseable line.

quotable spawns none of these itself, but yt-dlp does: ffmpeg for some
downloads, and Deno or Node to solve YouTube's JavaScript challenge. ffmpeg
reads stdin by default. Passing `stdin=subprocess.DEVNULL` to calls we make
does not reach the children of a library we call.

## What the SDK does

`mcp` ≥ 2.3 handles this at the file-descriptor level, inside
`stdio_server()` (`mcp/server/stdio.py`, `_claim_fd`). While serving, it:

1. dups the real stdin and stdout to close-on-exec descriptors at or above fd 3
   and serves the wire from those;
2. points fd 0 at `/dev/null`, so any child sees EOF at once;
3. points fd 1 at a dup of fd 2, so any child print lands on stderr.

The claim is skipped when a caller passes explicit streams to
`stdio_server(stdin, stdout)` ("explicit streams skip the claim", same file).
So the one rule for this repo is:

- serve with `MCPServer.run("stdio")`;
- never pass `stdin=` or `stdout=` to `stdio_server()`;
- never write to stdout in server mode; logs go to stderr.

The original design had quotable own an `stdio.py` that passed its own streams.
That would have switched the SDK's claim off and left the children on the pipe.

## The regression test

`tests/unit/test_stdio.py` pins the behaviour against an SDK upgrade. It
launches `tests/unit/stdio_probe.py`, a one-tool server, over a real
`stdio_client`. The tool spawns `cat` with inherited stdin and prints `leak`
to its own stdout. The test asserts:

- `cat` returns empty output in under a second, so it saw EOF, not the pipe;
- `leak` shows up twice in the server's stderr log, once per call;
- the second tool call on the same session succeeds, so the wire was not
  corrupted by the first.

It was red-checked by disabling `_claim_fd` while writing it.
`tests/integration/test_stdio_server.py` runs the real `quotable serve` over
stdio with the real `tiny` model as the end-to-end version.

## In-process versus a child process per job

The other way to keep children off the pipe is to run each transcription in
its own subprocess, as yueying does. That buys a hard kill and VRAM release per
job. It costs a model load per job: several seconds for `turbo`, every time.

quotable keeps the work in-process with one warm model and relies on the fd
claim for safety. Cancellation is cooperative, checked between Whisper
segments, and a cancelled run keeps its partial result so the next call
resumes. The trade is a few hundred milliseconds of cancel latency for a
multi-second saving per job.

## One more thing clients do

`mcp.client.stdio` starts a server with a minimal environment: `HOME`, `PATH`,
`SHELL`, `TERM`, `USER`, `LOGNAME`. Other clients behave the same way. Any
`QUOTABLE_*` setting has to be passed in the client's `env` block, or the server
silently uses its defaults. The integration test forwards `os.environ` for this
reason.
