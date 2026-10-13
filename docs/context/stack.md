# Tech Stack

## Core

| Layer | Technology | Why |
|-------|------------|-----|
| MCP server | `mcp>=2.3,<3` (`mcp.server.MCPServer`) | Official SDK; 2.x owns stdio fd isolation, structured output, annotations, resources |
| Speech to text | `faster-whisper` on ctranslate2 | CUDA `float16` or CPU `int8`, word timestamps, `clip_timestamps` for resume |
| Audio decode | PyAV (`av>=11,<19`) | Bundled by faster-whisper; no ffmpeg binary needed for local decode |
| Media download | `yt-dlp[default]` | Any supported URL, `bestaudio` as-is; `[default]` brings the YouTube JS challenge solver |
| Speaker labels | `pyannote.audio>=4` (optional extra) | Gated `speaker-diarization-community-1`; keeps torch out of the base install |
| Models and validation | Pydantic 2 | Structured tool output validated by the SDK |
| Locking | `filelock` | Per-variant cross-process lock; already in the tree via `huggingface_hub` |

## Runtime

| Component | Choice |
|-----------|--------|
| Python | 3.10 to 3.14 |
| Packaging | `uv`, hatchling, src layout |
| Install | `uvx quotable`; `uvx --from 'quotable[diarize]' quotable` |
| Transport | stdio only |
| Cache | `$XDG_DATA_HOME/quotable`, one directory per source × variant |

## Tooling

| Purpose | Tool |
|---------|------|
| Lint and format | ruff (`E,F,I,B,UP`), black, isort (profile black), line 100 |
| Tests | pytest; `integration` and `diarize` markers deselected by default |
| CI | GitHub Actions: lint, unit matrix 3.10 to 3.14, integration on 3.12 with a Hugging Face cache |
| Release | Tag `v*` → version check → `uv build` → PyPI trusted publishing (OIDC, no token) |
| Registry | `server.json` validated by `scripts/check_server_json.py`; published with `mcp-publisher` |
| Benchmark | `scripts/bench.py` |

## External Services

| Service | Purpose | When |
|---------|---------|------|
| Hugging Face Hub | Whisper and pyannote weights | First use of a model; cached under `~/.cache/huggingface` |
| Media hosts, podcast feeds | The content itself | Only when the user asks for a URL or feed; never in tests |
