# Utility Scripts

## Release Scripts

| Script | Description | Use Case |
|--------|-------------|----------|
| `scripts/check_server_json.py` | Validates `server.json` against its `$schema` (fetched over the network) and requires every version in it to equal the installed package version; `--tag vX.Y.Z` also checks a release tag | Run locally before tagging; the release workflow runs it with the tag |

## Development Scripts

| Script | Description | Use Case |
|--------|-------------|----------|
| `scripts/bench.py` | Runs the pipeline per (source, model, device) with `force=True` in a throwaway cache, times a second cached call, writes a markdown table with the hardware line | `uv run python scripts/bench.py <source> --models base,turbo --devices cpu,cuda --out bench.md --print` for the README numbers |
