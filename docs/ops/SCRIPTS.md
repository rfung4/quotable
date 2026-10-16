# Utility Scripts

## Release Scripts

| Script | Description | Use Case |
|--------|-------------|----------|
| `scripts/check_server_json.py` | Validates `server.json` against its `$schema` (fetched over the network) and requires every version in it to equal the installed package version; `--tag vX.Y.Z` also checks a release tag | Run locally before tagging; the release workflow runs it with the tag |

## Development Scripts

| Script | Description | Use Case |
|--------|-------------|----------|
| `scripts/demo.tape` | [vhs](https://github.com/charmbracelet/vhs) script for `docs/demo.gif`: Claude Code with the project `.mcp.json` transcribes the synthesised `interview-3` with speaker labels and quotes a speaker at a timestamp | `vhs scripts/demo.tape` from a plain terminal (not inside Claude Code), after `uv sync --extra diarize` and `python -m evals.make_tts`, with `QUOTABLE_HF_TOKEN` set |
