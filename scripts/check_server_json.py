"""Validate server.json against its $schema and pin its versions to the package version.

``--tag vX.Y.Z`` also requires the release tag to equal the project version.
"""

import argparse
import json
import sys
import urllib.request
from importlib.metadata import version
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag")
    args = parser.parse_args()

    manifest = json.loads((ROOT / "server.json").read_text())

    with urllib.request.urlopen(manifest["$schema"], timeout=30) as resp:  # noqa: S310
        schema = json.load(resp)
    jsonschema.validate(manifest, schema)

    dist = "quotable-mcp"
    expected = version(dist)
    found = {"server.json version": manifest["version"]}
    for i, pkg in enumerate(manifest.get("packages", [])):
        found[f"server.json packages[{i}].version"] = pkg["version"]
        if pkg["identifier"] != dist:
            print(f"server.json packages[{i}].identifier is not {dist!r}", file=sys.stderr)
            return 1
    if args.tag is not None:
        found["tag"] = args.tag.removeprefix("v")
    bad = {k: v for k, v in found.items() if v != expected}
    if bad:
        print(f"version mismatch, the package has {expected}: {bad}", file=sys.stderr)
        return 1
    print(f"server.json valid, versions all {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
