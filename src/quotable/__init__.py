"""Local-first MCP server and CLI that turns media into cite-ready transcripts."""

# PyPI's `quotable` is an unrelated project; this one is published as quotable-mcp.
DIST = "quotable-mcp"


def __getattr__(name: str) -> str:
    # Lazy: importlib.metadata costs ~6 MB, and the stdio shim never needs it.
    if name == "__version__":
        from importlib.metadata import version

        return version(DIST)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
