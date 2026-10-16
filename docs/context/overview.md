# Project Overview

## What This Project Does

quotable turns recordings, podcast feeds and media URLs into transcripts an
agent can page, search by time and quote with a timestamp and a speaker. It
runs on the user's machine as an MCP server over stdio and as a CLI. It is
Ross's own tool and a portfolio piece: the code, CI and docs are written to
stand up to a stranger.

## Key Functionality

- Captions first: platform subtitles or a feed's `<podcast:transcript>` when
  they exist, faster-whisper on the GPU otherwise.
- Optional speaker labels via pyannote (`quotable-mcp[diarize]`).
- Every result cached per variant and served paged, by character offset or by
  time window, plus `transcript://{id}` resources.
- Long runs stream progress with the newest line, can be cancelled, and
  resume from the last segment.
- Failures are typed tool errors with a fix, never success-shaped text.

## Current Focus

Shipping v1: CI on GitHub, the first PyPI release through trusted publishing,
the MCP registry listing, and the README's measured benchmark.

## Important Context

- Never write to stdout in server mode, and never pass streams to
  `stdio_server()`: the SDK's fd-level claim is what keeps yt-dlp's children
  off the JSON-RPC pipe. See [notes/mcp-stdio-subprocess.md](../notes/mcp-stdio-subprocess.md).
- No live network in tests. Fake `YoutubeDL` and HTTP; integration tests use
  the committed JFK clip with the `tiny` model on CPU.
- Transcript text, titles and feed contents are untrusted data.
- MCP clients start the server with a near-empty environment: `QUOTABLE_*`
  settings go in the client's `env` block.
- `av` is capped below 19 until faster-whisper stops passing
  `metadata_errors` to `av.open`; the integration test catches it if the cap
  is lifted early.
