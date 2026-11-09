"""What the reranker reads for a query: one definition shared by scoring, costs and replay.

A unit is a head chunk's text (expand = 0) or that chunk's EventMemory window (expand > 0,
expand-then-rerank). A run scores units in first-stage order until `pool_tokens` tokens of
units, at least `min_depth` of them. Keys are content-addressed (rerankers.ScoreCache.key),
and units with identical text are scored once.
"""

from __future__ import annotations

import numpy as np

from rbench import chunking, retrieval
from rbench.expansion import Windows
from rbench.rerankers import ScoreCache, depth, instruction_text, prompt_key, render
from rbench.samples import sample
from rbench.schema import queries


def texts(table: chunking.ChunkTable, chunk_texts: list[str], head: np.ndarray, windows: Windows | None) -> list[str]:
    return [chunk_texts[c] for c in head] if windows is None else [windows.text(int(c)) for c in head]


def tokens(table: chunking.ChunkTable, head: np.ndarray, windows: Windows | None) -> np.ndarray:
    return table.tokens[head] if windows is None else np.asarray([windows.tokens(int(c)) for c in head])


def keys(prompt: str, mode: str, query: str, unit_texts: list[str]) -> list[str]:
    return [ScoreCache.key(prompt, mode, query, render(u, mode)) for u in unit_texts]


def pool_items(run, chunking_name: str, embedder: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """(query, [(key, rendered unit)]) for every sampled query, to the run's depth, unique by key."""
    table = chunking.build(run.dataset, chunking_name)
    chunk_texts = table.texts(run.dataset)
    pools = retrieval.build(embedder, run.dataset, run.scope, chunking_name, run.variant)
    windows = Windows(run.dataset, table, run.expand) if run.expand else None
    prompt = prompt_key(instruction_text(run.instruction, run.dataset))
    out, seen = [], set()
    for qi in sample(run.dataset, run.sample):
        head, _ = pools.head(qi)
        head = head[: depth(tokens(table, head, windows), run.pool_tokens, run.min_depth)]
        query = queries(run.dataset)[qi]["variants"][run.variant]
        unit_texts = texts(table, chunk_texts, head, windows)
        items = []
        for k, u in zip(keys(prompt, run.render, query, unit_texts), unit_texts, strict=True):
            if k not in seen:
                seen.add(k)
                items.append((k, render(u, run.render)))
        out.append((query, items))
    return out
