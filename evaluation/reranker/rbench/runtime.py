"""MemMachine's runtime retrieval, replayed exactly, for recall at top-k (the asker's metric).

At runtime MemMachine's LongTermMemory asks EventMemory for 4 x k segments
(long_term_memory._EVENT_BACKEND_DEDUP_OVERFETCH, k the episodes requested), EventMemory has its
reranker score them and sorts by score (a stable sort: ties keep vector-search order), and
LongTermMemory keeps the first k distinct episodes, returned whole (one LongMemEval turn is one
episode). This module replays that path for every question:

  1. the vector search: EventMemory's own query at the deepest pool (200 seeds); a vector search
     limit of 4 x k returns the first 4 x k of them (exact search, so a prefix);
  2. the reranker on those 4 x k candidates, the strings EventMemory renders for its reranker. The
     rerankers are MemMachine's own classes, built through its RerankerManager: IdentityReranker,
     BM25Reranker (its default tokenizer: NLTK word_tokenize and English stop words, set up by
     memmachine_server.setup_nltk) and RRFHybridReranker (k = 60, its default); the Qwen3
     cross-encoders are memory.CachedReranker over the score cache (no API calls). BM25 and RRF
     depend on the whole candidate set (BM25's IDF; RRF's ranks), so they are computed afresh for
     each 4 x k; a cross-encoder's score is per pair, so the cached pool scores are its scores;
  3. EventMemory's stable sort by score, then LongTermMemory's dedup: the first k distinct
     episodes.

Recall at k: the share of a question's evidence turns among its k episodes (an episode is
returned whole, so a turn found is a turn read); all evidence: 1 if every evidence turn is among
them. Each is averaged over questions, every question counting equally.

  python -m rbench runtime-topk [--sample all] [--models q4-mem,q8-mem]
"""

from __future__ import annotations

import asyncio
import functools

import pandas as pd

from rbench import partitions as pt
from rbench.paths import SUMMARY
from rbench.report.stats import delta
from rbench.samples import sample
from rbench.schema import queries

from memmachine_server.episodic_memory.long_term_memory.long_term_memory import (  # noqa: E402, I001
    _EVENT_BACKEND_DEDUP_OVERFETCH as OVERFETCH,
)

KS = (1, 2, 3, 5, 10, 20, 30, 50)  # 4 x 50 = 200, the deepest pool scored
POOL = 200
NAMES = {"first_stage": "no reranker (vector search order)", "bm25": "BM25 alone",
         "rrf:identity+bm25": "RRF [identity, BM25]"}


def label(ranker: str) -> str:
    if ranker in NAMES:
        return NAMES[ranker]
    if ranker.startswith("rrf:identity+bm25+"):
        return f"RRF [identity, BM25, {ranker.removeprefix('rrf:identity+bm25+')}]"
    return f"{ranker} alone"


def rankers_for(models: list[str]) -> list[str]:
    """The configurations compared: no reranker, BM25, RRF of identity and BM25, and for each
    cross-encoder model: it alone and RRF of identity, BM25 and it (the two rrf-hybrid configs)."""
    return ["first_stage", "bm25", "rrf:identity+bm25", *(x for m in models for x in (m, f"rrf:identity+bm25+{m}"))]


def first_k_episodes(order: list[int], episode_of: list[int], k: int) -> list[int]:
    """LongTermMemory's dedup: walk the contexts in score order, keep each episode the first time
    it appears, stop at k episodes."""
    out, seen = [], set()
    for i in order:
        e = episode_of[i]
        if e in seen:
            continue
        seen.add(e)
        out.append(e)
        if len(out) >= k:
            break
    return out


def by_score(scores: list[float]) -> list[int]:
    """EventMemory's order: sorted by score, highest first; a stable sort keeps ties in
    vector-search order."""
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


class _NoEmbedders:
    async def get_embedder(self, name: str):  # no embedder-backed reranker is configured here
        raise KeyError(name)


async def _memmachine_rerankers(models: list[str]) -> dict:
    """MemMachine's rerankers, built as its RerankerManager builds them from configuration."""
    from memmachine_server import setup_nltk
    from memmachine_server.common.configuration.reranker_conf import (
        BM25RerankerConf,
        IdentityRerankerConf,
        RerankersConf,
        RRFHybridRerankerConf,
    )
    from memmachine_server.common.reranker.rrf_hybrid_reranker import RRFHybridReranker, RRFHybridRerankerParams
    from memmachine_server.common.resource_manager.reranker_manager import RerankerManager

    from rbench.memory import CachedReranker

    setup_nltk()  # what a MemMachine server runs: NLTK's tokenizer and stop words for BM25
    conf = RerankersConf(bm25={"bm": BM25RerankerConf()}, identity={"id": IdentityRerankerConf()},
                         rrf_hybrid={"rrf": RRFHybridRerankerConf(reranker_ids=["id", "bm"])})
    manager = RerankerManager(conf, _NoEmbedders())
    ident, bm25, rrf2 = [await manager.get_reranker(n) for n in ("id", "bm", "rrf")]
    # the same tokenizer, memoized: a pure function of the text, re-run for every pool size and configuration
    bm25._tokenize = functools.lru_cache(maxsize=None)(bm25._tokenize)
    out = {"bm25": bm25, "rrf:identity+bm25": rrf2}
    k = RRFHybridRerankerConf.model_fields["k"].default
    for m in models:  # rrf-hybrid [id, bm, ce], built as RerankerManager._build_rrf_hybrid_reranker does
        ce = CachedReranker(m, mode="replay")
        out[m] = ce
        out[f"rrf:identity+bm25+{m}"] = RRFHybridReranker(RRFHybridRerankerParams(rerankers=[ident, bm25, ce], k=k))
    return out


async def _replay(embedder: str, dataset: str, chunking: str, sample_name: str, models: list[str]) -> pd.DataFrame:
    from rbench.memory import Memory
    from rbench.progress import Progress
    from rbench.simulation import first_stage

    rows, fmt = queries(dataset), pt.formats(dataset)
    scorers = await _memmachine_rerankers(models)
    rankers = rankers_for(models)
    records = []
    keyed = pt.groups(dataset, sample(dataset, sample_name), "history")
    progress = Progress(f"{dataset} {chunking} runtime top-k", sum(len(v) for v in keyed.values()), "queries")
    for key, members in keyed.items():
        p = pt.part(dataset, rows[members[0]], "history")
        async with Memory(embedder, dataset, chunking) as m:
            seg = await pt.table(m, dataset, p)
            partition = await (await m.event_memory()).open_or_create_partition(key)
            for qi in members:
                q = rows[qi]
                query = q["variants"]["question"]
                seeds, _, _, texts = await first_stage(partition, seg, query, (), fmt.rerank, POOL)
                episode_of = [int(seg.event[s]) for s in seeds]
                turns = pt.gold(pt.gold_spans(dataset, q, p), p, seg).alternatives[0]
                gold = {int(seg.event[span[0][0]]) for span in turns}
                for k in KS:
                    limit = OVERFETCH * k
                    candidates = texts[0][:limit]
                    if len(candidates) < limit:
                        continue  # a history too small for this pool (not in LongMemEval-M)
                    for r in rankers:
                        order = (list(range(limit)) if r == "first_stage"
                                 else by_score(await scorers[r].score(query, candidates)))
                        got = set(first_k_episodes(order, episode_of, k))
                        records.append((q["qid"], q["features"].get("question_type"), r, k, limit,
                                        len(gold & got) / len(gold), float(gold <= got)))
                progress.step()
    return pd.DataFrame(records, columns=["qid", "question_type", "ranker", "k", "limit", "recall", "every"])


def write(embedder: str, dataset: str = "longmemeval_m", chunking: str = "seg500", sample_name: str = "all",
          models: tuple[str, ...] = ("q4-mem", "q8-mem")) -> str:
    """Replay every question and write the report (results/summary/<embedder>/...__runtime_topk.md)."""
    d = asyncio.run(_replay(embedder, dataset, chunking, sample_name, list(models)))
    out = SUMMARY / embedder
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}__history__question__{sample_name}__runtime_topk"
    d.to_parquet(out / f"{stem}.parquet", index=False)
    rankers = rankers_for(list(models))
    n = d.qid.nunique()
    lines = [f"# Recall at top-k as MemMachine retrieves at runtime — {dataset} · {chunking} · {n} questions\n",
             "- **Pool:** for k results, EventMemory retrieves 4 x k segments (LongTermMemory's "
             f"`_EVENT_BACKEND_DEDUP_OVERFETCH` = {OVERFETCH}), and the reranker scores those 4 x k.",
             "- **Results:** after EventMemory's sort by score, LongTermMemory keeps the first k distinct "
             "episodes (turns), returned whole.",
             "- **Recall:** the share of a question's evidence turns among its k episodes. **All evidence:** 1 if "
             "every evidence turn is among them. Each is averaged over questions, every question counting equally.",
             "- **Rerankers:** MemMachine's own IdentityReranker, BM25Reranker (default tokenizer) and "
             "RRFHybridReranker (k = 60), built through its RerankerManager; the Qwen3 cross-encoders' scores come "
             "from the score cache. BM25 and RRF are recomputed for each 4 x k candidate set.",
             "- **Δ:** minus no reranker, paired by question, with its 95% bootstrap interval. k stops at 50 because "
             f"4 x 50 = {POOL}, the deepest pool scored.\n"]
    base = d[d.ranker == "first_stage"].set_index(["qid", "k"]).sort_index()
    for measure, title in (("recall", "Recall at k (evidence turns among the k episodes)"),
                           ("every", "All evidence among the k episodes")):
        lines += [f"## {title}\n", "| k (pool 4k) | " + " | ".join(label(r) for r in rankers) + " |",
                  "|---|" + "---|" * len(rankers)]
        for k in KS:
            cells = []
            b = base.xs(k, level="k")[measure]
            for r in rankers:
                x = d[(d.ranker == r) & (d.k == k)].set_index("qid").sort_index()[measure]
                if x.empty:
                    cells.append("–")
                    continue
                cells.append(f"{x.mean():.3f}" if r == "first_stage"
                             else f"{x.mean():.3f} ({delta((x - b.loc[x.index]).to_numpy(), x.index.to_numpy())})")
            lines.append(f"| {k} ({OVERFETCH * k}) | " + " | ".join(cells) + " |")
        lines.append("")
    path = out / f"{stem}.md"
    path.write_text("\n".join(lines) + "\n")
    return str(path)
