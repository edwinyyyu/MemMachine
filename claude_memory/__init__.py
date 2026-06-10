"""Claude Code <-> EventMemory integration.

A persistent associative memory for Claude Code, backed by EventMemory (the
SQLite-backed VectorStore + SegmentStore). See DESIGN.md for the architecture and
the rationale behind every decision; README.md for installation.

Five modules:

- ``engine``     — config, stores, embedder, and the search/expand/ingest core
- ``transcript`` — Claude transcript JSONL -> timeline events
- ``daemon``     — the single memory process + its Unix-socket IPC (server+client)
- ``cli``        — one entry point: ``mcp`` / ``warm`` / ``ambient`` / ``ingest`` /
                   ``install`` (the hooks and MCP server are thin daemon clients)
- ``smoke``      — dependency-free end-to-end tests
"""
