"""MemMachine's runtime retrieval, replayed exactly, for recall at top-k (the asker's metric).

At runtime MemMachine's LongTermMemory asks EventMemory for 4 x k segments
(long_term_memory._EVENT_BACKEND_DEDUP_OVERFETCH, k the episodes requested), EventMemory has its
reranker score them and sorts by score (a stable sort: ties keep vector-search order), and
LongTermMemory keeps the first k distinct episodes, returned whole (one LongMemEval turn is one
episode). This module replays that path for every question (expand_context 0, the default):

  1. the vector search: EventMemory's own query at the deepest pool (200 seeds); a vector search
     limit of L returns the first L of them (exact search, so a prefix). Each question's seeds,
     their rendered texts, their episodes and its evidence segments are cached beside the store
     on first use (runtime_inputs/), so a re-run needs no store: only BM25 and cached scores;
  2. the reranker on those L candidates, the strings EventMemory renders for its reranker. The
     rerankers are MemMachine's own classes, built through its RerankerManager: IdentityReranker,
     BM25Reranker (its default tokenizer: NLTK word_tokenize and English stop words, set up by
     memmachine_server.setup_nltk) and RRFHybridReranker (k = 60, its default); the Qwen3
     cross-encoders are memory.CachedReranker over the score cache (no API calls). BM25 and RRF
     depend on the whole candidate set (BM25's IDF; RRF's ranks), so they are computed afresh for
     each pool; a cross-encoder's score is per pair, so the cached pool scores are its scores;
  3. EventMemory's stable sort by score, then the k results.

Pools: L = 4 x k (the runtime's), and for comparison a fixed L = 20 (EventMemory's default) and
200 (the deepest pool scored, the earlier reports' "pool"), each for k <= L.

Recall at k is counted two ways. k segments: the earlier reports' measure (report/topk.py), the
first k segments in ranked order, an evidence turn counting when all its segments are among them;
at a fixed pool it reproduces the earlier reports' cells. k episodes: what LongTermMemory returns,
an evidence turn counting when its episode is among the k (an episode is returned whole, so a turn
found is a turn read). All evidence: 1 if every evidence turn counts. Each is averaged over
questions, every question counting equally.

  python -m rbench runtime-topk [--sample all] [--models q4-mem,q8-mem] [--redraw]
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json

import numpy as np
import pandas as pd

from rbench import partitions as pt
from rbench.paths import FIGURES, SUMMARY
from rbench.report.stats import delta, interval
from rbench.samples import sample
from rbench.schema import queries

from memmachine_server.episodic_memory.long_term_memory.long_term_memory import (  # noqa: E402, I001
    _EVENT_BACKEND_DEDUP_OVERFETCH as OVERFETCH,
)

KS = (1, 2, 3, 5, 10, 20, 30, 50)  # 4 x 50 = 200, the deepest pool scored
POOL = 200
POOLS = ("4k", "20", "200")  # the runtime's 4 x k; a fixed 20 (EventMemory's default) and 200
UNITS = {"segments": "k segments (the earlier measure: a turn counts when all its segments are among the first k)",
         "episodes": "k episodes (what MemMachine returns: the first k distinct turns, each whole)"}
NAMES = {"first_stage": "no reranker (vector search order)", "bm25": "BM25 alone",
         "rrf:identity+bm25": "RRF [identity, BM25]"}
LEXICAL = "#8f8e89"  # BM25 and RRF without a cross-encoder: gray, no model
INPUTS_VERSION = 1


# What a cross-encoder is fused with: every subset of MemMachine's other two components. RRF of
# one component orders as that component alone (ties aside), so the single ones are no reranker
# (identity), BM25 alone and the cross-encoder alone, each measured as itself.
RRF_WITH = (("identity", "bm25"), ("identity",), ("bm25",))
PART_NAMES = {"identity": "identity", "bm25": "BM25"}


def rrf_name(parts: tuple[str, ...]) -> str:
    return "rrf:" + "+".join(parts)


def label(ranker: str) -> str:
    if ranker in NAMES:
        return NAMES[ranker]
    if ranker.startswith("rrf:"):
        return "RRF [" + ", ".join(PART_NAMES.get(x, x) for x in ranker.removeprefix("rrf:").split("+")) + "]"
    return f"{ranker} alone"


def rankers_for(models: list[str]) -> list[str]:
    """The configurations compared: no reranker, BM25, RRF of identity and BM25, and for each
    cross-encoder model: it alone and RRF of it with each subset of identity and BM25 (the user's
    two rrf-hybrid configs, [identity, bm25] and [identity, bm25, ce], and the other two)."""
    return ["first_stage", "bm25", "rrf:identity+bm25",
            *(x for m in models for x in (m, *(rrf_name((*c, m)) for c in RRF_WITH)))]


def limit_of(pool: str, k: int) -> int:
    return OVERFETCH * k if pool == "4k" else int(pool)


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


def turns_in_full(order: list[int], seeds: list[int], spans: list[tuple[int, ...]], k: int) -> list[bool]:
    """The earlier reports' measure (report/topk.py, share_full_k): keep the first k seeds in
    ranked order, and an evidence turn counts when every segment of it is among them."""
    kept = {seeds[i] for i in order[:k]}
    return [set(span) <= kept for span in spans]


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
    # the same tokenizer, memoized within a question (cleared after each): a pure function of the
    # text, otherwise re-run for every pool and configuration
    bm25._tokenize = functools.lru_cache(maxsize=None)(bm25._tokenize)
    out = {"bm25": bm25, "rrf:identity+bm25": rrf2}
    k = RRFHybridRerankerConf.model_fields["k"].default
    parts = {"identity": ident, "bm25": bm25}
    for m in models:  # rrf-hybrid [..., ce], built as RerankerManager._build_rrf_hybrid_reranker does
        ce = CachedReranker(m, mode="replay")
        out[m] = ce
        for c in RRF_WITH:
            out[rrf_name((*c, m))] = RRFHybridReranker(
                RRFHybridRerankerParams(rerankers=[*(parts[x] for x in c), ce], k=k))
    return out


def _inputs_path(m, key: str, qid: str, query: str, rerank_format):
    """Where a question's replay inputs are cached: keyed by the partition's encoding (its done
    stamp), the embedder and search width, the question, the pool and the rendering."""
    stamp = [INPUTS_VERSION, m.done(key).read_text(), m.embedder, m.width, qid, query, POOL, repr(rerank_format)]
    return m.root / "runtime_inputs" / key / f"{hashlib.sha1(json.dumps(stamp).encode()).hexdigest()[:24]}.json"


async def _question_inputs(embedder: str, dataset: str, chunking: str, key: str, members: list[int], rows,
                           fmt) -> list[dict]:
    """Per question: {seeds, texts (rendered for the reranker), episode_of (each seed's episode),
    spans (each evidence turn's segments), gold (each evidence turn's episode)}, from the cache,
    or from EventMemory's own query (then cached)."""
    from rbench.dbs import replace_atomically
    from rbench.memory import Memory
    from rbench.simulation import first_stage

    m = Memory(embedder, dataset, chunking)
    paths = [_inputs_path(m, key, rows[qi]["qid"], rows[qi]["variants"]["question"], fmt.rerank) for qi in members]
    out: list[dict | None] = []
    for path in paths:
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, ValueError):  # absent or unreadable: computed below
            out.append(None)
    missing = [i for i, x in enumerate(out) if x is None]
    if missing:
        p = pt.part(dataset, rows[members[0]], "history")
        async with m:
            seg = await pt.table(m, dataset, p)
            partition = await (await m.event_memory()).open_or_create_partition(key)
            for i in missing:
                q = rows[members[i]]
                seeds, _, _, texts = await first_stage(partition, seg, q["variants"]["question"], (), fmt.rerank, POOL)
                spans = [[int(c) for c, _ in span] for span in pt.gold(pt.gold_spans(dataset, q, p), p, seg).alternatives[0]]
                x = {"seeds": [int(s) for s in seeds], "texts": texts[0], "episode_of": [int(seg.event[s]) for s in seeds],
                     "spans": spans, "gold": [int(seg.event[span[0]]) for span in spans]}
                paths[i].parent.mkdir(parents=True, exist_ok=True)
                replace_atomically(paths[i], lambda t, x=x: t.write_text(json.dumps(x)))
                out[i] = x
    return out


async def _replay(embedder: str, dataset: str, chunking: str, sample_name: str, models: list[str]) -> pd.DataFrame:
    from rbench.progress import Progress

    rows, fmt = queries(dataset), pt.formats(dataset)
    scorers = await _memmachine_rerankers(models)
    rankers = rankers_for(models)
    records = []
    keyed = pt.groups(dataset, sample(dataset, sample_name), "history")
    progress = Progress(f"{dataset} {chunking} runtime top-k", sum(len(v) for v in keyed.values()), "queries")
    for key, members in keyed.items():
        for qi, x in zip(members, await _question_inputs(embedder, dataset, chunking, key, members, rows, fmt),
                         strict=True):
            q = rows[qi]
            query, qtype = q["variants"]["question"], q["features"].get("question_type")
            seeds, texts, episode_of = x["seeds"], x["texts"], x["episode_of"]
            spans, gold = [tuple(s) for s in x["spans"]], set(x["gold"])
            orders: dict[tuple[str, int], list[int]] = {}  # (ranker, L): the ranking of the first L
            for pool in POOLS:
                for k in KS:
                    limit = limit_of(pool, k)
                    if k > limit or len(texts) < limit:
                        continue  # k above a fixed pool; or a history too small for it (none in LongMemEval-M)
                    for r in rankers:
                        if (r, limit) not in orders:
                            orders[(r, limit)] = (list(range(limit)) if r == "first_stage"
                                                  else by_score(await scorers[r].score(query, texts[:limit])))
                        order = orders[(r, limit)]
                        got = set(first_k_episodes(order, episode_of, k))
                        full = turns_in_full(order, seeds, spans, k)
                        records.append((q["qid"], qtype, r, pool, k, limit, "episodes",
                                        len(gold & got) / len(gold), float(gold <= got)))
                        records.append((q["qid"], qtype, r, pool, k, limit, "segments",
                                        sum(full) / len(full), float(all(full))))
            scorers["bm25"]._tokenize.cache_clear()
            progress.step()
    return pd.DataFrame(records, columns=["qid", "question_type", "ranker", "pool", "k", "limit", "unit", "recall",
                                          "every"])


def _by_question(d: pd.DataFrame, unit: str, pool: str, ranker: str, measure: str) -> pd.DataFrame:
    """qid x k: one configuration's per-question values."""
    x = d[(d.unit == unit) & (d.pool == pool) & (d.ranker == ranker)]
    return x.pivot(index="qid", columns="k", values=measure).sort_index()


def _curve(a: pd.DataFrame, b: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per k in KS: the mean of a (or of a - b, paired by question) and its 95% interval (nan where
    not measured; no interval for a alone)."""
    mean, low, high = (np.full(len(KS), np.nan) for _ in range(3))
    for j, k in enumerate(KS):
        if k not in a.columns:
            continue
        x = a[k].dropna()
        if b is None:
            mean[j] = x.mean()
            continue
        diff = x - b[k].loc[x.index]
        mean[j] = diff.mean()
        low[j], high[j] = interval(diff.to_numpy(), diff.index.to_numpy()) if diff.abs().sum() else (0.0, 0.0)
    return mean, low, high


def _report(d: pd.DataFrame, models: list[str], dataset: str, chunking: str) -> list[str]:
    rankers = rankers_for(models)
    n = d.qid.nunique()
    lines = [f"# Recall at top-k as MemMachine retrieves at runtime — {dataset} · {chunking} · {n} questions\n",
             "- **Pool:** for k results, EventMemory retrieves 4 x k segments (LongTermMemory's "
             f"`_EVENT_BACKEND_DEDUP_OVERFETCH` = {OVERFETCH}), and the reranker scores those 4 x k. Sections 5 and 6 "
             "compare a fixed pool of 20 (EventMemory's default limit) and 200 (the earlier reports' pool).",
             "- **Results, counted two ways:**",
             "  - **k segments** (sections 1, 2 and 5): the first k segments after EventMemory's sort by score, the "
             "same measure as the earlier reports' recall at top-k (an evidence turn counts when all its segments "
             "are among them). Only the pool differs: 4 x k instead of a fixed 20 or 200. Section 5's fixed pools "
             "are the earlier reports' own cells.",
             "  - **k episodes** (sections 3, 4 and 6), what LongTermMemory returns: walking the sorted segments, it "
             "keeps the first k distinct episodes (turns), each returned whole. The 4 x k overfetch exists for "
             "this step. An evidence turn counts when its episode is among the k.",
             "- **Recall:** the share of a question's evidence turns counted. **All evidence:** 1 if every one is. "
             "Each is averaged over questions, every question counting equally.",
             "- **Rerankers:** MemMachine's own IdentityReranker, BM25Reranker (default tokenizer) and "
             "RRFHybridReranker (k = 60), built through its RerankerManager; the Qwen3 cross-encoders' scores come "
             "from the score cache. BM25 and RRF are recomputed for each candidate set.",
             "- **Δ:** minus no reranker (same pool, same k), paired by question, with its 95% bootstrap interval. k "
             f"stops at 50 because 4 x 50 = {POOL}, the deepest pool scored.\n"]
    sections = (("segments", "recall", "Recall at k segments (evidence turns read in full)"),
                ("segments", "every", "All evidence read in full within k segments"),
                ("episodes", "recall", "Recall at k episodes (evidence turns among the k episodes)"),
                ("episodes", "every", "All evidence among the k episodes"))
    for number, (unit, measure, title) in enumerate(sections, 1):
        base = _by_question(d, unit, "4k", "first_stage", measure)
        lines += [f"## {number}. {title}, pool 4 x k\n",
                  "| k (pool 4k) | " + " | ".join(label(r) for r in rankers) + " |", "|---|" + "---|" * len(rankers)]
        for k in KS:
            cells = []
            for r in rankers:
                x = _by_question(d, unit, "4k", r, measure)
                if k not in x.columns:
                    cells.append("–")
                    continue
                x = x[k].dropna()
                cells.append(f"{x.mean():.3f}" if r == "first_stage" else
                             f"{x.mean():.3f} ({delta((x - base[k].loc[x.index]).to_numpy(), x.index.to_numpy())})")
            lines.append(f"| {k} ({OVERFETCH * k}) | " + " | ".join(cells) + " |")
        lines.append("")
    for number, unit in ((5, "segments"), (6, "episodes")):
        lines += [f"## {number}. The pool: 4 x k against a fixed 20 or 200, recall at {UNITS[unit]}\n",
                  "| k | no reranker | " + " | ".join(f"{m}, pool {p}" for m in models for p in ("4k", "20", "200"))
                  + " |", "|---|---|" + "---|" * (3 * len(models))]
        for k in KS:
            cells = []
            for r, p in [("first_stage", "4k"), *((m, p) for m in models for p in ("4k", "20", "200"))]:
                x = _by_question(d, unit, p, r, "recall")
                cells.append(f"{x[k].mean():.3f}" if k in x.columns else "–")
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def _named(model: str) -> str:
    """The model's chart label, naming the model card's instruction where it is used (the
    memory instruction's label already names it)."""
    from rbench.report.figures import LABEL
    from rbench.rerankers import INSTRUCTION, RERANKERS

    card = RERANKERS[model].template.instruction == INSTRUCTION and "instruction" not in LABEL[model]
    return LABEL[model] + (", the model card's instruction" if card else "")


def figures(d: pd.DataFrame, models: list[str], out, head: str) -> list[str]:
    """runtime_pool_<model>.png (the pool: 4 x k vs a fixed 20 or 200) and runtime_rrf_<unit>.png
    (the RRF hybrids against no reranker and against the cross-encoder alone)."""
    from matplotlib.ticker import MaxNLocator

    from rbench.report import figures as fg
    from rbench.report import palette

    out.mkdir(parents=True, exist_ok=True)
    ks, first = list(KS), fg.SERIES["first_stage"]
    written = []

    def line(ax, ys, color, ls="-", band=None):
        fg._plot(ax, ks, ys, color=color, linestyle=ls, marker="o", markersize=3.5)
        if band is not None:
            ax.fill_between(ks, band[0], band[1], color=color, alpha=0.12, linewidth=0)

    def key(color, ls="-"):
        return fg._key(color, ls, marker="o")

    def k_axis(ax, recall: bool):
        fg._log_x(ax, 10, ks)
        ax.set_xlabel("k results (log)")
        if recall:
            fg._share_y(ax)
        else:
            ax.axhline(0, color=fg.INK2, linewidth=0.6)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))  # round steps: 0.02, 0.05
            ax.yaxis.set_major_formatter(fg.FuncFormatter(lambda v, _: f"{v:+.2f}" if abs(v) > 1e-9 else "0"))

    unit_line = {"segments": "k segments: the earlier reports' measure; an evidence turn counts when all its segments are "
                             "among the first k.",
                 "episodes": "k episodes: what MemMachine returns; walking the reranked segments it keeps the first k "
                             "distinct turns, each whole, and an evidence turn counts when it is among them."}
    for m in models:
        # the darker three of four steps: the lightest drifts toward the first stage's blue (green) or
        # fades on white (amber); the three clear palette.FLOOR for normal and deficient color vision
        shades = palette.family(fg.SERIES[m], 4)
        styles = {"4k": (shades[1], "-"), "200": (shades[0], "--"), "20": (shades[2], ":")}
        fig, axes = fg._figure(2, 2, panel=(4.6, 3.3))
        for j, unit in enumerate(UNITS):
            base = {p: _by_question(d, unit, p, "first_stage", "recall") for p in POOLS}
            ax = axes[0][j]
            line(ax, _curve(base["4k"])[0], first)
            for p, (color, ls) in styles.items():
                line(ax, _curve(_by_question(d, unit, p, m, "recall"))[0], color, ls)
            k_axis(ax, True)
            ax.set_title(f"k {unit}: recall", loc="left")
            ax.set_ylabel("recall")
            ax = axes[1][j]
            for p, (color, ls) in styles.items():
                mean, low, high = _curve(_by_question(d, unit, p, m, "recall"), base[p])
                line(ax, mean, color, ls, band=(low, high))
            k_axis(ax, False)
            ax.set_title(f"k {unit}: gain over no reranker", loc="left")
            ax.set_ylabel("recall gain")
        path = out / f"runtime_pool_{m}.png"
        fg._save(fig, path, f"Candidate pool: 4 × k (MemMachine's runtime) against a fixed 20 or 200, {_named(m)} — "
                 f"{head}", [key(first), *(key(c, ls) for c, ls in styles.values())],
                 ["no reranker", "reranking a pool of 4 × k (MemMachine's runtime)", "reranking a fixed pool of 200",
                  "reranking a fixed pool of 20 (EventMemory's default limit; k ≤ 20 only)"], ncol=2,
                 note=["For k results, the reranker scores a pool of candidates from the vector search: 4 × k, what "
                       "MemMachine's LongTermMemory asks for, or a fixed 20 or 200, as in the earlier reports.",
                       "Left: " + unit_line["segments"] + " Right: " + unit_line["episodes"],
                       "Top: recall, the share of a question's evidence turns counted, averaged over questions. Bottom: "
                       "the gain over no reranker at the same k and pool; shaded: its 95% interval (paired bootstrap "
                       "over questions).",
                       "At k = 5 the 4 × k pool is 20 and at k = 50 it is 200, so the lines meet there."])
        written.append(str(path))
    lexical = palette.family(LEXICAL, 2)
    for unit in UNITS:
        fig, axes = fg._figure(len(models), 3, panel=(4.4, 3.2))
        for i, m in enumerate(models):
            model = palette.family(fg.SERIES[m], 4)[1:3]  # the middle two: visible on white, apart from blue
            series = [("bm25", lexical[1], "-"), ("rrf:identity+bm25", lexical[0], "--"),
                      (m, model[0], "-"), (f"rrf:identity+bm25+{m}", model[1], "--")]
            base = _by_question(d, unit, "4k", "first_stage", "recall")
            ce = _by_question(d, unit, "4k", m, "recall")
            ax = axes[i][0]
            line(ax, _curve(base)[0], first)
            for r, color, ls in series:
                line(ax, _curve(_by_question(d, unit, "4k", r, "recall"))[0], color, ls)
            k_axis(ax, True)
            ax.set_title(f"{_named(m)}: recall at k", loc="left")
            ax.set_ylabel("recall")
            ax = axes[i][1]
            for r, color, ls in series:
                mean, low, high = _curve(_by_question(d, unit, "4k", r, "recall"), base)
                line(ax, mean, color, ls, band=(low, high))
            k_axis(ax, False)
            ax.set_title("against no reranker", loc="left")
            ax.set_ylabel("recall minus no reranker")
            ax = axes[i][2]
            for r, color, ls in (series[1], series[3]):
                mean, low, high = _curve(_by_question(d, unit, "4k", r, "recall"), ce)
                line(ax, mean, color, ls, band=(low, high))
            k_axis(ax, False)
            ax.set_title("against the cross-encoder alone", loc="left")
            ax.set_ylabel("recall minus the cross-encoder alone")
        path = out / f"runtime_rrf_{unit}.png"  # the user's two configurations (subsets: runtime_rrf_subsets_*)
        fg._save(fig, path, f"RRF hybrid with and without the cross-encoder, MemMachine's runtime pool (4 × k), "
                 f"{unit} — {head}",
                 [key(first), key(lexical[1]), key(lexical[0], "--"), key(fg.INK), key(fg.INK, "--")],
                 ["no reranker", "BM25 alone", "RRF [identity, BM25]", "the cross-encoder alone (its color)",
                  "RRF [identity, BM25, cross-encoder] (its color)"], ncol=3,
                 note=["MemMachine's rrf-hybrid reranker (RRF's k = 60; BM25 with MemMachine's default tokenizer), "
                       "with and without the cross-encoder, on the pool MemMachine's runtime reranks: 4 × k candidates.",
                       unit_line[unit],
                       "Left: recall, the share of a question's evidence turns counted, averaged over questions. "
                       "Middle: the difference from no reranker. Right: the difference from the cross-encoder alone "
                       "(below 0: worse than it). Shaded: 95% intervals, paired bootstrap over questions.",
                       "Blue: no reranker. Gray: BM25 alone (solid) and RRF [identity, BM25] (dashed). The model's "
                       "color, one row per model: the cross-encoder alone (solid) and RRF [identity, BM25, "
                       "cross-encoder] (dashed)."])
        written.append(str(path))
    for unit in UNITS:  # every RRF including the cross-encoder, against it alone and no reranker
        fig, axes = fg._figure(len(models), 3, panel=(4.4, 3.2))
        for i, m in enumerate(models):
            # the darker four of five steps (the lightest fades on white); the cross-encoder alone
            # on the step nearest the model's own color
            shades = palette.family(fg.SERIES[m], 5)
            series = [(m, shades[2], "-"), (rrf_name(("identity", m)), shades[1], ":"),
                      (rrf_name(("bm25", m)), shades[3], "-."), (rrf_name(("identity", "bm25", m)), shades[0], "--")]
            base = _by_question(d, unit, "4k", "first_stage", "recall")
            ce = _by_question(d, unit, "4k", m, "recall")
            ax = axes[i][0]
            line(ax, _curve(base)[0], first)
            for r, color, ls in series:
                line(ax, _curve(_by_question(d, unit, "4k", r, "recall"))[0], color, ls)
            k_axis(ax, True)
            ax.set_title(f"{_named(m)}: recall at k", loc="left")
            ax.set_ylabel("recall")
            ax = axes[i][1]
            for r, color, ls in series:
                mean, low, high = _curve(_by_question(d, unit, "4k", r, "recall"), base)
                line(ax, mean, color, ls, band=(low, high))
            k_axis(ax, False)
            ax.set_title("against no reranker", loc="left")
            ax.set_ylabel("recall minus no reranker")
            ax = axes[i][2]
            for r, color, ls in series[1:]:
                mean, low, high = _curve(_by_question(d, unit, "4k", r, "recall"), ce)
                line(ax, mean, color, ls, band=(low, high))
            k_axis(ax, False)
            ax.set_title("against the cross-encoder alone", loc="left")
            ax.set_ylabel("recall minus the cross-encoder alone")
        path = out / f"runtime_rrf_subsets_{unit}.png"
        fg._save(fig, path, f"The cross-encoder in each RRF hybrid, MemMachine's runtime pool (4 × k), {unit} — {head}",
                 [key(first), key(fg.INK), key(fg.INK, ":"), key(fg.INK, "-."), key(fg.INK, "--")],
                 ["no reranker", "the cross-encoder alone", "RRF [identity, cross-encoder]", "RRF [BM25, cross-encoder]",
                  "RRF [identity, BM25, cross-encoder]"], ncol=3,
                 note=["MemMachine's rrf-hybrid reranker with the cross-encoder and each subset of its other components "
                       "(identity: the vector search order; BM25), on the pool MemMachine's runtime reranks: 4 × k "
                       "candidates. RRF's k = 60; BM25 with MemMachine's default tokenizer.",
                       "An RRF of one component orders as that component alone: RRF [identity] is no reranker, RRF "
                       "[cross-encoder] the cross-encoder alone. RRF [identity, BM25] and BM25 alone: "
                       f"runtime_rrf_{unit}.png.",
                       unit_line[unit],
                       "Left: recall, the share of a question's evidence turns counted, averaged over questions. "
                       "Middle: the difference from no reranker. Right: the difference from the cross-encoder alone "
                       "(below 0: worse than it). Shaded: 95% intervals, paired bootstrap over questions.",
                       "Blue: no reranker. The model's color, one row per model, darker to lighter: RRF [identity, "
                       "BM25, cross-encoder] (dashed), RRF [identity, cross-encoder] (dotted), the cross-encoder alone "
                       "(solid), RRF [BM25, cross-encoder] (dash-dot)."])
        written.append(str(path))
    return written


def summary_dir(embedder: str):
    """The runtime replay's reports and rows: a folder of their own, so the core pipeline's
    clean-up of a run's summaries (tables.clear: <run>__*.md beside it) never removes them."""
    return SUMMARY / embedder / "runtime"


def figures_dir(embedder: str, dataset: str, sample_name: str):
    """The runtime replay's charts: a folder of their own, so the core pipeline's redraw (which
    clears its run's chart folder first: figures.write) never removes them."""
    return FIGURES / embedder / "runtime" / f"{dataset}__history__question__{sample_name}"


def write(embedder: str, dataset: str = "longmemeval_m", chunking: str = "seg500", sample_name: str = "all",
          models: tuple[str, ...] = ("q4-mem", "q8-mem"), redraw: bool = False) -> list[str]:
    """Replay every question (or, with redraw, reuse the last replay's rows) and write the report
    (summary_dir: ...__runtime_topk.md, and its rows, .parquet) and its charts (figures_dir)."""
    out = summary_dir(embedder)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}__history__question__{sample_name}__runtime_topk"
    if redraw:
        d = pd.read_parquet(out / f"{stem}.parquet")
    else:
        d = asyncio.run(_replay(embedder, dataset, chunking, sample_name, list(models)))
        d.to_parquet(out / f"{stem}.parquet", index=False)
    path = out / f"{stem}.md"
    path.write_text("\n".join(_report(d, list(models), dataset, chunking)) + "\n")
    head = f"{dataset} · {chunking} · {d.qid.nunique()} questions"
    charts = figures(d, list(models), figures_dir(embedder, dataset, sample_name), head)
    return [str(path), *charts]
