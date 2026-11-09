"""First-stage retrieval: rank each query's candidates by embedding similarity.

Scopes (which chunks a query may retrieve):
  corpus      every chunk of the dataset: for queries that stand alone
  doc         the chunks of the query's own document: for queries that assume it
              ("what datasets do they use?" assumes the paper)
  scope_docs  the chunks of the documents in the query's scope_docs (one repository snapshot)

The ranking is kept from the top until HEAD_TOKENS tokens of chunk text. Every reader
budget and every rerank depth used later is within that head, so the rest of the
ranking is never needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rbench import chunking, embedding
from rbench.paths import POOLS
from rbench.schema import corpus, queries

HEAD_TOKENS = 16384


@dataclass(frozen=True)
class Pools:
    """Each query's ranking head, as CSR rows: ids[offsets[q]:offsets[q+1]] best first."""

    qids: np.ndarray
    offsets: np.ndarray
    ids: np.ndarray
    scores: np.ndarray

    def head(self, q: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.offsets[q], self.offsets[q + 1]
        return self.ids[a:b], self.scores[a:b]


def path(embedder: str, dataset: str, scope: str, chunking_name: str, variant: str) -> Path:
    return POOLS / embedder / dataset / scope / f"{chunking_name}__{variant}.npz"


def load(embedder: str, dataset: str, scope: str, chunking_name: str, variant: str) -> Pools:
    """Cached pools, refused if they were built for other queries than the current ones."""
    data = np.load(path(embedder, dataset, scope, chunking_name, variant))
    if data["qids"].tolist() != [q["qid"] for q in queries(dataset)]:
        raise RuntimeError(f"{path(embedder, dataset, scope, chunking_name, variant)} was built for other queries; "
                           "delete it to rebuild")
    return Pools(data["qids"], data["offsets"], data["ids"], data["scores"])


def candidates(dataset: str, scope: str, table: chunking.ChunkTable) -> list[np.ndarray] | None:
    """Candidate chunk ids per query; None means every chunk."""
    if scope == "corpus":
        return None
    docs = corpus(dataset)
    by_doc = table.by_doc(len(docs.doc_ids))
    out = []
    for q in queries(dataset):
        names = [q["doc_id"]] if scope == "doc" else q["scope_docs"]
        out.append(np.sort(np.concatenate([by_doc[docs.index[n]] for n in names])))
    return out


def head(sim: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Indices into sim, best first, until HEAD_TOKENS tokens (or all of them)."""
    keep = min(len(sim), max(64, int(4 * HEAD_TOKENS / max(1, np.median(tokens)))))
    while True:
        top = np.argpartition(-sim, keep - 1)[:keep] if keep < len(sim) else np.arange(len(sim))
        top = top[np.argsort(-sim[top], kind="stable")]
        cum = np.cumsum(tokens[top])
        if cum[-1] >= HEAD_TOKENS or keep == len(sim):
            return top[: int(np.searchsorted(cum, HEAD_TOKENS, "left")) + 1]
        keep = min(len(sim), keep * 2)


def build(embedder: str, dataset: str, scope: str, chunking_name: str, variant: str) -> Pools:
    target = path(embedder, dataset, scope, chunking_name, variant)
    if target.exists():
        return load(embedder, dataset, scope, chunking_name, variant)
    table = chunking.load(dataset, chunking_name)
    vectors = embedding.chunk_vectors(embedder, dataset, chunking_name).astype(np.float32)
    qvectors = embedding.query_vectors(embedder, dataset, variant).astype(np.float32)
    if len(vectors) != len(table) or len(qvectors) != len(queries(dataset)):
        raise RuntimeError(f"{embedder}/{dataset}: cached vectors ({len(vectors)} chunks, {len(qvectors)} queries) do not "
                           f"match the data ({len(table)} chunks, {len(queries(dataset))} queries); delete and re-embed")
    per_query = candidates(dataset, scope, table)
    ids, scores, offsets = [], [], [0]
    for start in range(0, len(qvectors), 256):
        block = qvectors[start : start + 256]
        sims = block @ vectors.T if per_query is None else None
        for j, qv in enumerate(block):
            cand = None if per_query is None else per_query[start + j]
            sim = sims[j] if cand is None else vectors[cand] @ qv
            local = head(sim, table.tokens if cand is None else table.tokens[cand])
            ids.append((local if cand is None else cand[local]).astype(np.int32))
            scores.append(sim[local].astype(np.float32))
            offsets.append(offsets[-1] + len(local))
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, qids=np.asarray([q["qid"] for q in queries(dataset)]), offsets=np.asarray(offsets, np.int64),
             ids=np.concatenate(ids), scores=np.concatenate(scores))
    return load(embedder, dataset, scope, chunking_name, variant)
