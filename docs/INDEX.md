# quotable Documentation

> Local-first MCP server + CLI: media → cite-ready transcripts. Captions first,
> faster-whisper fallback, optional speaker labels, cached per variant, paged.

**Client:** none; personal portfolio project
**Status:** 🟡 Active

---

## Quick Context (Start Here)

| Document | Purpose |
|----------|---------|
| [context/overview.md](context/overview.md) | What this project does and the gotchas |
| [context/domain.md](context/domain.md) | Source, variant, segment, page, partial |
| [context/stack.md](context/stack.md) | Libraries, runtime, tooling |

---

## Documentation Map

| I need to... | Check |
|--------------|-------|
| Understand the project | [context/](context/) |
| See how a source becomes a transcript | [architecture/quotable.md](architecture/quotable.md) |
| Know why the server never touches stdio | [notes/mcp-stdio-subprocess.md](notes/mcp-stdio-subprocess.md) |
| Run the scripts | [ops/SCRIPTS.md](ops/SCRIPTS.md) |
| See current work | [development/](development/) (local) |
| Explore options | [research/](research/) (local) |

The MCP tool and resource surface is documented in the repo
[README](../README.md); there is no HTTP API and no database.

---

## Active Development

| Feature | Status | Location |
|---------|--------|----------|
| quotable v1 | 🟡 In Progress | [development/quotable/](development/quotable/) (local) |

---

## Recent Updates

| Date | Change |
|------|--------|
| 2026-10-13 | Architecture, stdio note, context docs and README written (Task 18) |
| 2026-10-12 | Initial documentation structure |
