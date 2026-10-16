"""Local-first MCP server and CLI that turns media into cite-ready transcripts."""

from importlib.metadata import version

# PyPI's `quotable` is an unrelated project; this one is published as quotable-mcp.
DIST = "quotable-mcp"
__version__ = version(DIST)
