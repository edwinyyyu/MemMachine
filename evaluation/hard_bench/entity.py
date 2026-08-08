"""Adapter for the R23+R24 prose-fact + DSU entity-resolution memory.

The reference implementation lives in the sibling `semantic_memory` repo:
  - aen6_prose_v2.py (round23): Mention/Fact/EntityRegistry/MemoryStore,
    writer prompt, retrieve, etc.
  - aen7_recursive.py (round24): overrides retrieve + ingest_turns to add
    recursive cognition pass and hybrid LLM dedup.
  - _common.py (round7): Cache, Budget, llm(), embed_batch() — sync.

We import them via sys.path injection and wrap the sync entry points in
asyncio.to_thread so they don't block hard_bench's async event loop.

Public:
    await build_entity_store(turns, cache_path)        -> MemoryStore
    await retrieve_entity(query, store, *, k=14)       -> list[Fact]
    fact_source_turn_id(fact)                          -> int

A Fact's `.ts` carries the writer's chosen target turn_id (one of the
fire's target turns). We use that as the source turn for Hit emission.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

# ---- sys.path injection so the reference modules import cleanly ---------
_REF_ROOT = Path("/Users/eyu/edwinyyyu/mmcc/semantic_memory/evaluation/attribute_memory/research")
_PATHS = [
    _REF_ROOT / "round23_prose_facts" / "architectures",
    _REF_ROOT / "round24_recursive_cognition" / "architectures",
    _REF_ROOT / "round7" / "experiments",
]
for p in _PATHS:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import _common  # noqa: E402
import aen6_prose_v2 as v2  # noqa: E402
import aen7_recursive as v3  # noqa: E402

# Re-export so callers don't have to do their own sys.path dance.
__all__ = [
    "make_cache",
    "make_budget",
    "build_entity_store",
    "retrieve_entity",
    "fact_source_turn_id",
    "v2",
    "v3",
    "_common",
]


def make_cache(path: Path) -> _common.Cache:
    """Make a Cache instance pointing at `path`, patched with a
    multi-process-safe save (fcntl-merge over a sidecar lockfile).

    The upstream `_common.Cache.save` does a naive `write_text(json.dumps)`
    that races on parallel writers — last writer wins, others' entries
    are lost. We don't want to modify the reference module, so we
    monkey-patch the instance's save method with a safe variant.
    """
    cache = _common.Cache(path)

    def _safe_save():
        if not cache._dirty:
            return
        import fcntl
        cache.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = cache.path.with_suffix(cache.path.suffix + ".lock")
        with open(lock_path, "w") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                disk: dict = {}
                if cache.path.exists():
                    try:
                        import json as _json
                        disk = _json.loads(cache.path.read_text())
                    except Exception:
                        disk = {}
                disk.update(cache._d)
                cache._d = disk
                import json as _json
                tmp = cache.path.with_suffix(cache.path.suffix + ".tmp")
                tmp.write_text(_json.dumps(cache._d))
                tmp.replace(cache.path)
                cache._dirty = False
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    cache.save = _safe_save  # type: ignore[method-assign]
    return cache


def make_budget(max_llm: int = 10_000, max_embed: int = 5_000) -> _common.Budget:
    """Generous budget — hard_bench scenarios can be longer than the
    research suite that the defaults were tuned for."""
    return _common.Budget(
        max_llm=max_llm,
        max_embed=max_embed,
        stop_at_llm=max_llm - 1,
        stop_at_embed=max_embed - 1,
    )


async def build_entity_store(
    turns: list[tuple[int, str]],
    cache: _common.Cache,
    budget: _common.Budget,
    *,
    enable_reflection: bool = True,
) -> v2.MemoryStore:
    """Ingest turns into a MemoryStore (R24 recursive cognition).

    `turns` is a list of (turn_id, "Speaker: text") strings, in chronological
    order. Returns the populated MemoryStore.
    """
    def _ingest():
        # aen7.ingest_turns returns (obs_facts, obs_mentions, cog_facts,
        # cog_mentions, store, telemetry).
        _obs_facts, _obs_mentions, _cog_facts, _cog_mentions, store, _telemetry = (
            v3.ingest_turns(turns, cache, budget, enable_reflection=enable_reflection)
        )
        cache.save()
        return store

    return await asyncio.to_thread(_ingest)


async def retrieve_entity(
    query: str,
    store: v2.MemoryStore,
    cache: _common.Cache,
    budget: _common.Budget,
    *,
    k: int = 14,
    expand_hops: int = 1,
    llm_dedup: bool = True,
) -> list[v2.Fact]:
    """Run R24's hybrid retrieve. Returns the list of Facts.

    Uses surface-match (uncapped) ∪ kNN top-k, with optional multi-hop
    entity expansion and LLM dedup at read time.
    """
    def _retrieve():
        facts, _resolution_map = v3.retrieve(
            query,
            store,
            cache,
            budget,
            top_k=k,
            expand_hops=expand_hops,
            llm_dedup=llm_dedup,
        )
        return facts

    return await asyncio.to_thread(_retrieve)


def fact_source_turn_id(fact: v2.Fact) -> int:
    """Best-effort mapping from a Fact back to a source turn_id.

    Writer sets `fact.ts` to one of the fire's target turn_ids (the writer
    picks which target turn the fact "belongs to" via the JSON output's
    `turn` field, with a fallback to the last target turn).
    """
    return int(fact.ts)
