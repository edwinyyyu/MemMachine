"""Replay reading for partitioned (chat-history) datasets: one tidy row per (query, ranker).

Per query, inside its own history (histories.py):
  1. The first stage's head of seeds (exact search), similarity order.
  2. A ranker orders the seeds (rankers.py). Depths are counts of seeds, EventMemory's
     vector_search_limit, for every ranker, so reranking seeds and reranking their windows are
     compared at the same overfetch. A reranked configuration reads only its reranked seeds
     (EventMemory reranks every seed it retrieves); the first stage reads as many as the budget
     takes, or, with a depth L ("first_stage@nL"), only its top L.
  3. Each seed becomes a context: the seed alone, or its EventMemory window (expand e).
     Rerank-then-expand scores seeds; expand-then-rerank (xrerank) scores windows.
  4. Segments are admitted as EventMemory admits them under a growing segment limit
     (eventmemory.admission_order_ids), and the reader cost after each one is the token count
     of the context string EventMemory would render (reading.py): headers with timestamps,
     JSON text, components; overlaps counted once.
  5. metrics.evaluate gives the coverage events (complete and touched), recall and precision
     against the gold turns; the session-level view adds LongMemEval-style recall@k (an
     evidence session counts once any of its segments is admitted).

Expansions: rerank-then-expand and the expanded first stage need no new scores, so they are
replayed at every value in EXPANSIONS; expand-then-rerank needs the windows scored and appears
only where a run scored them (grid.py scores expand 3).

Output: results/sim/<embedder>/<dataset>/history/question/<sample>/seg<size>__<instruction>__em.parquet,
the same columns as simulate.py plus:
  sess_any_k, sess_all_k   admitted segments until some / every evidence session is touched
  admitted                 segments admitted before reading stopped (past MAX_COST, or at the
                           end of the reranked pool)
  <event>_ctx              contexts (seeds) read when the event happens: coverage after
                           reranking at k kept seeds is <event>_ctx <= k
  segs_t<B>, ctx_t<B>      segments and contexts read within reader budget B: EventMemory's
                           max_num_segments, and the first stage's vector_search_limit, at B
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rbench import histories as hs
from rbench import metrics, rankers
from rbench.eventmemory import admission
from rbench.paths import SIM
from rbench.reading import reading
from rbench.rerankers import ScoreCache, instruction_text, pair_overhead, prompt_key
from rbench.samples import sample
from rbench.schema import queries
from rbench.simulate import COLUMNS, scored_prefix

EXPANSIONS = (1, 2, 3, 6)  # expand_context values replayed (EventMemory's default is 0)
MAX_COST = 16384  # reading stops past the largest reader budget measured
RENDER = "em"
EVENTS = ("first", "every", "first_touch", "every_touch")
EXTRA = (["sess_any_k", "sess_all_k", "admitted"] + [f"{e}_ctx" for e in EVENTS]
         + [f"segs_t{b}" for b in metrics.BUDGETS] + [f"ctx_t{b}" for b in metrics.BUDGETS])


def path(embedder: str, dataset: str, sample_name: str, size: int, instruction: str) -> object:
    return SIM / embedder / dataset / "history" / "question" / sample_name / f"seg{size}__{instruction}__{RENDER}.parquet"


def gold_for(dataset: str, qid: str, seg: hs.Segments) -> tuple[metrics.Gold, list[list[int]]]:
    turns, sessions = hs.gold_segments(dataset, qid, seg)
    chars = {i: len(seg.texts[i]) for t in turns for i in t}
    spans = tuple(tuple((i, chars[i]) for i in t) for t in turns)
    return metrics.Gold((spans,), chars, tuple(sorted(chars))), sessions


def keys_for(query: str, prompt: str, texts: list[str]) -> list[str]:
    """Cache keys of rendered units (they are already rendered: no further render step)."""
    return [ScoreCache.key(prompt, RENDER, query, t) for t in texts]


def read(contexts: list[list[int]], seeds: list[int], seg: hs.Segments, lazy: bool = False
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(admitted segment ids, cumulative rendered cost, the context each came from), stopping
    past MAX_COST. lazy: contexts enter the query result only as admission reaches them (the
    first stage without a vector search limit, whose limit is the seeds it reads)."""
    pairs = admission(contexts, seeds)
    order, ctx = [s for s, _ in pairs], [k for _, k in pairs]
    cum = reading(contexts, order, seg.event, seg.inner, seg.run, MAX_COST, ctx if lazy else None)
    return np.asarray(order[: len(cum)], dtype=np.int64), cum, np.asarray(ctx[: len(cum)], dtype=np.int64)


def counts(values: list[float], cum: np.ndarray, ctx: np.ndarray) -> list[float]:
    """The count normalizations: contexts (seeds) read when each event happens, and the
    segments and contexts read within each reader budget (EventMemory's max_num_segments and,
    for the first stage, its vector_search_limit at that budget)."""
    f = dict(zip(metrics.FIELDS, values, strict=True))
    seen = np.maximum.accumulate(ctx) + 1 if len(ctx) else ctx  # contexts reached after each segment
    at = [float(seen[int(f[f"{e}_k"]) - 1]) if np.isfinite(f[f"{e}_k"]) else np.inf for e in EVENTS]
    reach = np.maximum.accumulate(cum) if len(cum) else cum
    segs = [int(np.searchsorted(reach, b, "right")) for b in metrics.BUDGETS]
    return at + segs + [float(seen[n - 1]) if n else 0.0 for n in segs]


def session_positions(admitted: np.ndarray, sessions: list[list[int]]) -> tuple[float, float]:
    position = {int(s): i for i, s in enumerate(admitted)}
    firsts = [min((position[s] for s in sess if s in position), default=np.inf) for sess in sessions]
    return (min(firsts) + 1, max(firsts) + 1) if firsts else (np.inf, np.inf)


def run(embedder: str, dataset: str, size: int, sample_name: str, models: list[str], instruction: str = "default",
        expansions: tuple[int, ...] = EXPANSIONS) -> pd.DataFrame:
    rows = queries(dataset)
    pools = hs.load_pools(embedder, dataset, size, sample_name)
    itext = instruction_text(instruction, dataset)
    prompt = prompt_key(itext)
    caches = {m: ScoreCache(m) for m in models}
    all_rankers = rankers.for_models(models, expansions, rankers.HISTORY_DEPTHS, limited_first_stage=True)
    records: dict[str, list] = {r.id: [] for r in all_rankers}
    dropped: dict[str, str] = {}
    picked = sample(dataset, sample_name)
    for qi in picked:
        q = rows[qi]
        seg, producers = hs.build_segments(dataset, q["qid"], size)
        head, sims = pools.head(qi, seg)
        head = [int(x) for x in head]
        g, sessions = gold_for(dataset, q["qid"], seg)
        query = q["variants"]["question"]
        overhead = pair_overhead(query, itext) if models else 0
        chars = np.asarray([len(t) for t in seg.texts])
        n_seg = len(seg)
        units = {0: [[s] for s in head], **{e: [hs.window(s, n_seg, e) for s in head] for e in expansions}}
        keys = {e: keys_for(query, prompt, [hs.unit_text(seg, producers, u) for u in units[e]]) for e in units}
        scores = {(m, e): scored_prefix(caches[m], m, keys[e]) for m in models for e in units}
        seed_tokens = seg.tokens[head]
        priority = {e: np.asarray([2 if any(i in g.union for i in u) else 0 for u in units[e]]) for e in units}
        for r in all_rankers:
            if r.id not in records:
                continue
            n = r.depth.chunks(seed_tokens) if r.depth else 0
            logits = None
            cost = 0.0
            if r.model is not None:
                logits, pair_tokens = scores[(r.model, r.expand if r.kind == "xrerank" else 0)]
                if n > len(logits):  # some query was not scored this deep: drop the configuration
                    del records[r.id]
                    dropped[r.id] = f"{q['qid']} has {len(logits)} scored, {n} needed"
                    continue
                cost = float(pair_tokens[:n].sum())
            perm = r.order(n, sims, logits, priority[r.expand])
            seeds = [head[i] for i in perm]
            contexts = [units[r.expand][i] for i in perm]
            admitted, cum, ctx = read(contexts, seeds, seg, lazy=r.kind == "first_stage" and r.depth is None)
            values = metrics.evaluate(admitted, cum, chars, g)
            sess_any, sess_all = session_positions(admitted, sessions)
            prefix = cost - max(0, n - 1) * overhead if r.model is not None else 0.0
            records[r.id].append([n if r.model else 0, cost, prefix, n if r.kind == "oracle" else 0, *values, sess_any, sess_all,
                                  len(admitted), *counts(values, cum, ctx)])
    frames = []
    for r in all_rankers:
        if r.id not in records:
            continue
        meta = {"qid": [rows[i]["qid"] for i in picked], "split": [rows[i]["split"] for i in picked],
                "cluster": [rows[i]["qid"] for i in picked], "chunking": f"seg{size}", "ranker": r.id, "kind": r.kind,
                "model": r.model, "depth": r.depth.label if r.depth else None, "weight": r.weight, "expand": r.expand,
                "question_type": [rows[i]["features"]["question_type"] for i in picked]}
        frames.append(pd.concat([pd.DataFrame(meta), pd.DataFrame(records[r.id], columns=[*COLUMNS, *EXTRA])], axis=1))
    report_dropped(dataset, size, dropped, set(records))
    out = pd.concat(frames, ignore_index=True)
    target = path(embedder, dataset, sample_name, size, instruction)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(target, index=False)
    return out


def report_dropped(dataset: str, size: int, dropped: dict[str, str], kept: set[str]) -> None:
    """Say which configurations of the scored models were left out because some question was
    not scored deep enough (models with no scores at all, and windows never scored at an
    expansion, drop out silently: that is by design)."""
    for m in sorted({k.split(":")[1].split("@")[0] for k in kept if ":" in k}):
        for kind, what in (("rerank", "seeds"), ("xrerank", "windows")):
            def expand(k: str) -> str:
                return k.split("+x")[1] if "+x" in k else "0"

            scored = {expand(k) for k in kept if k.startswith(f"{kind}:{m}@")}
            relevant = {k: v for k, v in dropped.items() if k.startswith(f"{kind}:{m}@") and expand(k) in scored}
            lost = sorted({k.split("@")[1].split("+")[0] for k in relevant}, key=lambda d: int(d[1:]))
            if lost:
                example = next(v for k, v in relevant.items() if k.split("@")[1].split("+")[0] == lost[0])
                print(f"{dataset} seg{size}: {m} reranking {what} is left out at vector search limits {', '.join(lost)}: "
                      f"some question was not scored that deep (e.g. {example})", flush=True)


def scoring_work(embedder: str, dataset: str, size: int, sample_name: str, model: str, expand: int,
                 pool_seeds: int, instruction: str = "default"):
    """Pairs to score for a run, as rerankers.Work: units of each query's first pool_seeds seeds
    (the deepest vector search limit the run replays), unique by key, not yet in the model's
    cache."""
    rows = queries(dataset)
    pools = hs.load_pools(embedder, dataset, size, sample_name)
    prompt = prompt_key(instruction_text(instruction, dataset))
    cache = ScoreCache(model)
    work, seen = [], set()
    for qi in sample(dataset, sample_name):
        q = rows[qi]
        seg, producers = hs.build_segments(dataset, q["qid"], size)
        head, _ = pools.head(qi, seg)
        head = [int(x) for x in head][:pool_seeds]
        units = [[s] for s in head] if expand == 0 else [hs.window(s, len(seg), expand) for s in head]
        texts = [hs.unit_text(seg, producers, u) for u in units]
        query = q["variants"]["question"]
        keys = keys_for(query, prompt, texts)
        have = cache.get(keys)
        items = []
        for k, t in zip(keys, texts, strict=True):
            if k not in have and k not in seen:  # scored already, or queued for another query
                seen.add(k)
                items.append((k, t))
        if items:
            work.append((query, items))
    return work
