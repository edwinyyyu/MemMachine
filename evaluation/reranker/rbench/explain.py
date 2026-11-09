"""A dry run: one query traced through every step and every measurement, with real numbers.

python -m rbench explain <dataset> <chunking> <qid> [--model q06-local] [--pool-seeds N] [--expand E]
                                                  [--embedder gemma300m-raw]

The query runs in its own EventMemory partition (partitions.py): the first stage, the windows
at expand_context E, both scoring positions, and the rendered reader cost. Scores missing
from the cache are computed with the local 0.6B (free) when the model is local; for a paid
model the query's pool must already be scored.
"""

from __future__ import annotations

import asyncio

import numpy as np

from rbench import config, metrics
from rbench import partitions as pt
from rbench.embedding import DEFAULT as EMBEDDER
from rbench.memory import Memory
from rbench.rankers import Depth, Ranker
from rbench.rerankers import RERANKERS, ScoreCache, pair_overhead, prompt_key, score, to_logit
from rbench.schema import queries
from rbench.simulation import first_stage, keys_for, read

from memmachine_server.episodic_memory.event_memory.data_types import ProducerContext  # noqa: I001

SHOW = 10


def _preview(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "…"


def _source(p: pt.Part, seg: pt.Segments, c: int) -> str:
    """What a segment comes from: its turn and producer, or its document."""
    e = int(seg.event[c])
    context = p.events[e].context
    return f"{p.labels[e]} ({context.producer})" if isinstance(context, ProducerContext) else p.labels[e]


async def _first(embedder: str, dataset: str, chunking: str, p: pt.Part, query: str, expand: int):
    async with Memory(embedder, dataset, chunking) as m:
        seg = await pt.table(m, dataset, p)
        partition = await (await m.event_memory()).open_or_create_partition(p.key)
        return (seg, *await first_stage(partition, seg, query, (expand,), pt.formats(dataset).rerank))


def run(dataset: str, chunking_name: str, qid: str, model: str = "q06-local", pool_seeds: int = 50, expand: int = 3,
        embedder: str = EMBEDDER) -> str:
    from rbench.datasets import info

    rows = queries(dataset)
    qi = next(i for i, q in enumerate(rows) if q["qid"] == qid)
    q = rows[qi]
    query = q["variants"][info(dataset).default_variant]
    p = pt.part(dataset, q)
    pt.ingest(embedder, dataset, chunking_name, [qi])  # encoded once (embedded when not yet)
    seg, head, sims, units_all, texts_all = asyncio.run(_first(embedder, dataset, chunking_name, p, query, expand))
    head = np.asarray(head)
    spans = pt.gold_spans(dataset, q, p)
    g = pt.gold(spans, p, seg)
    union = g.union
    budgets = config.EVAL.budgets
    kind = q.get("features", {}).get("question_type")
    w = [f"# Dry run: {dataset} · {qid} · {chunking_name} · {RERANKERS[model].hf_model}\n",
         "## 1. The query" + (f" ({kind})" if kind else "") + "\n", "> " + query.replace("\n", "\n> "), "",
         f"Its partition ({pt.scope_of(dataset)}): {len(p.events):,} events, {len(seg):,} EventMemory segments "
         f"({chunking_name}; mean {np.mean([len(t) for t in seg.texts]):,.0f} characters), in EventMemory's order.\n"]

    w.append(f"## 2. The gold: {len(spans)} acceptable answer(s); every span of one is needed\n")
    for a, (alternative, mapped) in enumerate(zip(spans, g.alternatives, strict=True)):
        for (e, s, t), span in zip(alternative, mapped, strict=True):
            ids = [c for c, _ in span]
            w.append(f"- Answer {a + 1}: {p.labels[e]} [{s}:{t}] “{_preview(p.texts[e][s:t], 160)}” — segment(s) {ids} "
                     f"({int(seg.tokens[ids].sum())} tokens). Complete once all are read; touched once any is.")
    w.append("")

    w += [f"## 3. First stage: EventMemory.query over this partition ({embedder}, exact search), {len(head)} seeds\n",
          "| rank | segment | from | tokens | cosine | gold | text |", "|---|---|---|---|---|---|---|"]
    for i, (c, sim) in enumerate(zip(head[:SHOW], sims[:SHOW], strict=True)):
        w.append(f"| {i + 1} | {c} | {_source(p, seg, int(c))} | {seg.tokens[c]} | {sim:.3f} | "
                 f"{'yes' if int(c) in union else ''} | {_preview(seg.texts[c])} |")
    w.append("")

    tmpl = RERANKERS[model].template
    prompt = prompt_key(tmpl)
    n = min(pool_seeds, len(head))
    units = {e: units_all[e][:n] for e in (0, expand)}
    cache, scored = ScoreCache(model), {}
    for e in units:
        texts = texts_all[e][:n]
        keys = keys_for(query, prompt, texts)
        have = cache.get(keys)
        missing = [(k, t) for k, t in zip(keys, texts, strict=True) if k not in have]
        if missing:
            if RERANKERS[model].backend != "local":
                raise SystemExit(f"{len(missing)} pairs are not scored for {model}; score the pool first")
            score(model, [(query, missing)], f"explain/{dataset}/{qid}/x{e}")
            have = cache.get(keys)
        scored[e] = (to_logit(model, np.asarray([have[k][0] for k in keys])), np.asarray([have[k][1] for k in keys]), texts)
    overhead = pair_overhead(query, tmpl)
    w += [f"## 4. Rerank pool: the top {n} seeds (vector search limit {n})\n",
          f"Each pair is the model's fixed text + query ({overhead} tokens) + one unit rendered as EventMemory's "
          "reranker sees it. Reranker tokens for this query (API billing / with a prefix cache): "
          f"seeds {int(scored[0][1].sum()):,} / {int(scored[0][1].sum() - (n - 1) * overhead):,}; "
          f"windows (expand_context {expand}) {int(scored[expand][1].sum()):,} / "
          f"{int(scored[expand][1].sum() - (n - 1) * overhead):,}.\n",
          "| new rank (seeds) | old rank | segment | seed logit | window logit | gold in seed / window | seed text |",
          "|---|---|---|---|---|---|---|"]
    for r_, i in enumerate(np.argsort(-scored[0][0], kind="stable")[:SHOW]):
        c = int(head[i])
        in_window = any(x in union for x in units[expand][i])
        w.append(f"| {r_ + 1} | {i + 1} | {c} | {scored[0][0][i]:+.2f} | {scored[expand][0][i]:+.2f} | "
                 f"{'yes' if c in union else 'no'} / {'yes' if in_window else 'no'} | {_preview(seg.texts[c])} |")
    w.append("")

    d = Depth(n)
    configs = {"first stage": Ranker("first_stage"), f"first stage, expand {expand}": Ranker("first_stage", expand=expand),
               f"rerank seeds (depth {n})": Ranker("rerank", model, d),
               f"rerank seeds, then expand {expand}": Ranker("rerank", model, d, expand=expand),
               f"expand {expand}, then rerank windows": Ranker("xrerank", model, d, expand=expand),
               f"perfect reranker (depth {n}, evidence first)": Ranker("oracle", depth=d)}
    chars = np.asarray([len(t) for t in seg.texts])
    w += ["## 5. Reading each order as EventMemory renders it, and every measurement\n",
          "Segments are admitted as EventMemory's segment limit grows; the reader cost is the token count of the "
          "rendered context (headers, JSON text, windows merged). A reranked order reads only its pool. *first* / "
          "*every*: reader tokens when some span of some answer / every span of one answer is complete; *touch*: "
          "when touched.\n",
          "| order | first | every | first touch | every touch | admitted | " + " | ".join(f"every@{b}" for b in budgets)
          + " | recall@1000 |", "|---|---|---|---|---|---|" + "---|" * len(budgets) + "---|"]
    for label, r_ in configs.items():
        e = r_.expand
        logits = None if r_.model is None else scored[expand if r_.kind == "xrerank" else 0][0]
        priority = np.asarray([2 if any(x in union for x in u) else 0 for u in (units[e] if r_.depth else [[0]])])
        perm = r_.order(n if r_.depth else 0, sims, logits, priority)
        seeds = [int(head[i]) for i in perm]
        contexts = [units_all[e][i] for i in perm]
        admitted, cum, _ = read(contexts, seeds, seg, lazy=r_.kind == "first_stage" and r_.depth is None)
        f = dict(zip(metrics.FIELDS, metrics.evaluate(admitted, cum, chars, g), strict=True))
        w.append(f"| {label} | {f['first']:,.0f} | {f['every']:,.0f} | {f['first_touch']:,.0f} | {f['every_touch']:,.0f} | "
                 f"{len(admitted)} | " + " | ".join(str(int(f['every'] <= b)) for b in budgets)
                 + f" | {f['recall_t1000']:.2f} |")
    w += ["", "## 6. From one query to the reported numbers\n",
          "- Every per-query value above is averaged over the queries with equal weight.",
          "- Reader budget B is continuous: each query's costs give its coverage at every B.",
          "- Overfetch depth is free: a shallower depth reorders a prefix of the same scored pool, and that depth is "
          "the vector search limit of the reranked configuration.",
          "- Intervals bootstrap the dataset's clusters (questions for LongMemEval)."]
    return "\n".join(w) + "\n"
