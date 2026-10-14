<!-- mcp-name: io.github.rfung4/quotable -->

# quotable

Turn recordings, podcast feeds and media URLs into transcripts you can cite,
on your own machine. Every line carries a timestamp, and a speaker label when
you ask for one. Captions first, [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
when there are none. Exposed as an MCP server over stdio and as a CLI.

<!-- TODO(demo): asciinema/GIF of Claude Code citing a speaker at a timestamp,
     recorded on a CC-licensed or own recording. Owner: Ross. -->

## Install

```bash
uvx quotable --help
claude mcp add quotable -- uvx quotable serve
```

With speaker labels (pulls in PyTorch and pyannote, about 2 GB):

```bash
claude mcp add quotable -e QUOTABLE_HF_TOKEN=hf_... -- uvx --from 'quotable[diarize]' quotable serve
```

Diarization uses pyannote's
[`speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1),
a gated model: accept its terms on Hugging Face once, then pass a token.

MCP clients start the server with a near-empty environment, so put `QUOTABLE_*`
settings in the client's `env` block (`-e` above), not in your shell profile.

## Use

From an agent:

> Transcribe `~/meetings/2026-10-13-planning.m4a` with speaker labels and quote
> what SPEAKER_01 said about the deadline, with the timestamp.

From the shell:

```bash
quotable get ~/meetings/planning.m4a --diarize --format srt
quotable get https://feeds.example.com/podcast.xml --episode "pricing" --format txt
quotable info https://www.youtube.com/watch?v=...
quotable serve
```

`get` prints the transcript on stdout and streams segments to stderr as they
are produced. Failures are one stderr line with the fix, exit code 1.

## Tools and resources

| Name | Kind | What it does |
|------|------|--------------|
| `media_info(source, episode?)` | tool, read-only | Title, duration, caption languages, chapters; for a feed, its recent episodes |
| `transcribe(source, mode?, language?, diarize?, episode?, force?)` | tool | Transcribes and caches; returns an `id`, metadata, `cite_url` and the first page |
| `read_transcript(id, offset?, limit?, start_s?, end_s?, timestamps?)` | tool, read-only | Pages a cached transcript by character offset or by time window |
| `transcript://{id}` | resource, `text/plain` | The whole transcript, one `[hh:mm:ss] SPEAKER: text` line per segment |
| `transcript://{id}/meta` | resource, `application/json` | Method, language, chapters, speakers, segment count |

`mode` is `auto` (the source's own captions or feed transcript, else Whisper),
`captions` (never run Whisper) or `whisper` (always transcribe the audio).
`episode` picks a feed episode by guid, 1-based index (1 = newest) or title words.

Tool results are typed structured output plus a short text block. Failures
arrive as tool errors (`isError: true`) that name the cause and the fix, never
as success-shaped text. Long Whisper runs report progress per segment with the
newest line; a cancel stops between segments and the next call resumes from
there.

`cite_url` is a template with `{t}` for whole seconds, so an agent can link the
exact moment: `https://youtu.be/{id}?t={t}` for YouTube, `…#t={t}` for other
URLs and enclosures, `file://{path}#t={t}` for local files.

Transcript text and titles are third-party content. The rendered text says so
in its first line, and an agent should treat it as data, never as instructions.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `QUOTABLE_OUTPUT_DIR` | `$XDG_DATA_HOME/quotable` (`~/.local/share/quotable`) | Cache of transcripts: `transcript.json`, `.txt`, `.srt` per variant |
| `QUOTABLE_DEVICE` | `auto` | `auto`, `cpu` or `cuda`; `auto` picks CUDA when ctranslate2 sees a GPU |
| `QUOTABLE_MODEL` | `turbo` on CUDA, `base` on CPU | Any faster-whisper model name |
| `QUOTABLE_COMPUTE_TYPE` | `float16` on CUDA, `int8` on CPU | ctranslate2 compute type |
| `QUOTABLE_KEEP_AUDIO` | off | Keep downloaded audio in its scratch directory (path logged to stderr) instead of deleting it; nothing reads it back |
| `QUOTABLE_HF_TOKEN` | `HF_TOKEN` | Hugging Face token for the diarization model |

A transcript is cached per *variant*: source × method × language × model ×
diarized. Two concurrent requests for the same variant share one run; the
second waits and then reads the cache.

### GPU

On CUDA the model loads in `float16`. If CUDA or cuDNN is missing or broken,
the load falls back to CPU with a warning on stderr; the server keeps working.
faster-whisper's README covers installing cuBLAS and cuDNN 9 for your platform.
An 8 GB card fits `turbo`.

### YouTube and other sites

Downloads go through [yt-dlp](https://github.com/yt-dlp/yt-dlp). YouTube
requires a JavaScript runtime to solve its challenge: install
[Deno](https://deno.com) or Node.js 20+ and keep it on `PATH`. If it is
missing, the error says so.

## How it works

```mermaid
flowchart LR
    S[file · feed URL · media URL] --> R{resolve}
    R -->|file| A[audio]
    R -->|feed| F[episode: podcast:transcript?]
    R -->|URL| C[captions: json3 / VTT]
    F -->|transcript| T[Transcript]
    C -->|captions| T
    F -->|no transcript| A
    C -->|no captions| A
    A --> W[faster-whisper, streaming segments]
    W -->|optional| D[pyannote speakers]
    W --> T
    D --> T
    T --> K[(cache per variant)]
    K --> M[MCP tools + transcript:// resources · CLI]
```

The pipeline is captions-first: platform subtitles or a feed's
`<podcast:transcript>` cost seconds and keep names right, so they win when
they exist. Whisper runs otherwise, on the GPU when there is one. Segments are
written to disk as they arrive, so a crashed or cancelled run resumes from its
last segment. One model stays warm per process, and one lock serialises
inference so two runs cannot fight for the card.

The server runs the work in-process. yt-dlp spawns ffmpeg and a JS runtime on
its own, and a child that inherits the server's stdin or stdout would block or
corrupt the JSON-RPC pipe. The `mcp` SDK repoints fd 0 and fd 1 while serving,
quotable never overrides that, and a regression test pins the behaviour.
See [docs/notes/mcp-stdio-subprocess.md](docs/notes/mcp-stdio-subprocess.md).

## Benchmark

Measured with `scripts/bench.py`. Realtime is audio duration divided by wall
time; cached is a second call for the same variant.

<!-- TODO(bench): run `uv run python scripts/bench.py <1h recording> --models base,turbo --devices cpu,cuda --out bench.md`
     on the RTX 3070 and replace the pending rows. Owner: Ross. -->

| Source | Model | Device | Audio | Wall | Realtime | Cached |
|--------|-------|--------|-------|------|----------|--------|
| jfk.flac (test clip) | tiny | cpu | 11 s | 0.9 s | 11.9x | 0.00 s |
| ~1 h recording | base | cpu | pending | pending | pending | pending |
| ~1 h recording | turbo | cuda | pending | pending | pending | pending |
| captioned video | captions | - | pending | pending | pending | pending |

The `tiny` row was measured on a 32-core machine; the first run includes
loading the model.

## Why not one of the others

<!-- Every cell below goes through /verify-claims before the repo is public.
     Sources read 2026-10-12; the competitors move. -->

| | quotable | yueying | transcriptor-mcp | hearsay | mcp-video-analyzer |
|---|---|---|---|---|---|
| Captions first, Whisper fallback | yes | yes | not checked | not checked | not checked |
| Typed tool errors, never success-shaped text | yes | no (`"ERROR: …"` strings) | not checked | not checked | not checked |
| Structured output and `transcript://` resources | yes | no | not checked | not checked | not checked |
| Speaker labels over MCP | yes (extra) | no | no | CLI only | no |
| Podcast RSS as an input | yes | no | no | no | no |
| Partial text, resume, cancel on long runs | yes | progress only | not checked | not checked | not checked |
| Keyframes, OCR, visual analysis | no, by design | yes | yes | no | yes |

If you need frames or OCR, use one of those; quotable aims at the transcript
half. The main trade-off against yueying's child-process-per-job design: one
warm model in-process saves a multi-second load per job and lets a cancel keep
the partial result, at the cost of hard-kill and VRAM release.

## Rights

quotable processes what you point it at. You need the rights to the
recordings, feeds and pages you transcribe, and the terms of the site you fetch
from apply to you: YouTube's terms, for one, restrict downloading and automated
access. quotable ships no cookie or anti-bot handling.

## Development

```bash
uv sync
uv run pytest -q                 # unit tests, collaborators faked
uv run pytest -m integration     # real tiny model on CPU, real stdio, no network
uv run ruff check . && uv run black --check . && uv run isort --check-only .
```

The test clip is the public-domain JFK inaugural excerpt from whisper.cpp,
re-encoded to 16 kHz mono FLAC; see `tests/fixtures/README.md`. Python 3.10 to
3.14, MIT licence.
