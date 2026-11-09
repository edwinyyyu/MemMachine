"""A dry run: one query traced through every step and every measurement, with real numbers.

python -m rbench explain <dataset> <chunking> <qid> [--model q06-local] [--pool-seeds N | --pool-tokens N --min-depth N]
                                                  [--embedder gemma300m]

For a history dataset (LongMemEval) the chunking is EventMemory's segment length (seg500), and
the trace adds the windows (expand_context 3), both scoring positions and the rendered reader
cost.

Scores missing from the cache are computed with the local 0.6B (free) when the model is
q06-local; for a paid model the query's pool must already be scored.
"""

from __future__ import annotations

import numpy as np

from rbench import chunking, metrics, retrieval
from rbench.datasets import info
from rbench.embedding import DEFAULT as EMBEDDER
from rbench.rankers import Depth, Ranker
from rbench.rerankers import RERANKERS, ScoreCache, depth, pair_overhead, prompt_key, score, to_logit
from rbench.schema import corpus, queries
from rbench.simulate import gold_priority

SHOW = 10
BUDGETS = (500, 1000, 2000, 4000)


def _producers(dataset: str, qid: str, seg) -> list[str]:
    """Each segment's producer (User or Assistant), from the history file."""
    from rbench.datasets.longmemeval import history
    from rbench.histories import PRODUCERS

    sessions = history(dataset, qid)["sessions"]
    return [PRODUCERS[sessions[int(s)]["turns"][int(t)]["role"]] for s, t in zip(seg.session, seg.turn, strict=True)]


def _preview(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "…"


def run(dataset: str, chunking_name: str, qid: str, model: str = "q06-local", pool_tokens: int | None = None,
        min_depth: int | None = None, embedder: str = EMBEDDER, pool_seeds: int | None = None) -> str:
    """History datasets score the top pool_seeds seeds (default: the core stage's, grid.Run);
    the others score pool_tokens tokens, at least min_depth (default 8,192 / 16, as adhoc)."""
    if info(dataset).partitioned:
        from rbench.grid import Run

        return run_history(dataset, chunking_name, qid, model, pool_seeds or Run.pool_seeds, embedder=embedder)
    pool_tokens, min_depth = pool_tokens or 8192, min_depth or 16
    ds, docs = info(dataset), corpus(dataset)
    rows = queries(dataset)
    qi = next(i for i, q in enumerate(rows) if q["qid"] == qid)
    q = rows[qi]
    query = q["variants"][ds.default_variant]
    table = chunking.build(dataset, chunking_name)
    texts = table.texts(dataset)
    pools = retrieval.build(embedder, dataset, ds.scope, chunking_name, ds.default_variant)
    head, sims = pools.head(qi)
    g = metrics.gold(dataset, table)[qi]
    union = g.union
    w = []
    w += [f"# Dry run: {dataset} · {qid} · {chunking_name} · {RERANKERS[model].hf_model}\n",
          f"## 1. The query ({ds.default_variant}, {ds.scope} scope)\n", "> " + query.replace("\n", "\n> "), ""]

    w.append(f"## 2. The gold: {len(q['gold'])} acceptable answer(s); the first has {len(q['gold'][0])} span(s)\n")
    for k, (doc_id, s, e) in enumerate(q["gold"][0][:6]):
        d = docs.index[doc_id]
        s, e = metrics.trim(docs.texts[d], s, e)
        chunks = [int(c) for c, _ in g.alternatives[0][k]]
        w.append(f"- Span {k + 1} in `{doc_id}`: “{_preview(docs.texts[d][s:e], 160)}” — inside chunk(s) {chunks} "
                 f"({int(table.tokens[chunks].sum())} tokens). It counts as read only once all of them are read.")
    w.append("")

    w.append(f"## 3. First stage: {embedder} ranks {len(head)} head chunks of {len(table):,} candidates by cosine\n")
    w.append("| rank | chunk | doc | tokens | cumulative | cosine | gold chars | text |")
    w.append("|---|---|---|---|---|---|---|---|")
    cum = np.cumsum(table.tokens[head])
    for i, (c, sim) in enumerate(zip(head[:SHOW], sims[:SHOW], strict=True)):
        w.append(f"| {i + 1} | {c} | {docs.doc_ids[table.doc_index[c]][:28]} | {table.tokens[c]} | {cum[i]} | {sim:.3f} | "
                 f"{union.get(int(c), 0)} | {_preview(texts[c])} |")
    w.append("")

    tmpl = RERANKERS[model].template
    n = depth(table.tokens[head], pool_tokens, min_depth)
    overhead = pair_overhead(query, tmpl)
    keys = [ScoreCache.key(prompt_key(tmpl), "plain", query, texts[c]) for c in head[:n]]
    cache = ScoreCache(model)
    have = cache.get(keys)
    missing = [(k, texts[c]) for k, c in zip(keys, head[:n], strict=True) if k not in have]
    if missing:
        if RERANKERS[model].backend != "local":
            raise SystemExit(f"{len(missing)} pairs are not scored for {model}; score the pool first")
        score(model, [(query, missing)], f"explain/{dataset}/{qid}")
        have = cache.get(keys)
    logits = to_logit(model, np.asarray([have[k][0] for k in keys]))
    pair_tokens = np.asarray([have[k][1] for k in keys])
    w += [f"## 4. Rerank pool: the head chunks within {pool_tokens:,} tokens, at least {min_depth}: {n} pairs\n",
          f"Each pair is the model card's prompt + instruction + query ({overhead} tokens, repeated in every pair) + one "
          f"chunk. Reranker cost for this query: {int(pair_tokens.sum()):,} tokens as an API bills them; "
          f"{int(pair_tokens.sum() - (n - 1) * overhead):,} with a prefix cache (prompt and query computed once). "
          f"At ${RERANKERS['q8'].usd_per_mtok}/M (8B) that is ${pair_tokens.sum() / 1e6 * RERANKERS['q8'].usd_per_mtok:.5f}.\n"]
    order = np.argsort(-logits, kind="stable")
    w += ["| new rank | old rank | chunk | score (yes−no logit) | gold chars | text |", "|---|---|---|---|---|---|"]
    for r, i in enumerate(order[:SHOW]):
        c = head[i]
        w.append(f"| {r + 1} | {i + 1} | {c} | {logits[i]:+.2f} | {union.get(int(c), 0)} | {_preview(texts[c])} |")
    w.append("")

    chars = table.end - table.start
    d = Depth("n", n)
    configs = {
        "first stage only": Ranker("first_stage"),
        f"reranked (depth {n})": Ranker("rerank", model, d),
        f"perfect reranker (depth {n}, evidence first)": Ranker("oracle", depth=d),
    }
    priority = gold_priority([{int(c)} for c in head], union, set(g.cheapest))
    w += ["## 5. Reading in each order, and every measurement\n",
          "Whole chunks are read in order. *first* = reader tokens when some gold span is completely read; *every* = when "
          "all spans of one answer are. Coverage@B is 1 if that cost is ≤ B. Recall@B = share of the answer's gold "
          "characters inside the chunks read within B; precision@B = share of characters read that are gold.\n",
          "| order | first | every | first_k | " + " | ".join(f"cov@{b} first/every" for b in BUDGETS)
          + " | recall@1000 | precision@1000 | F1@1000 |", "|---|---|---|---|" + "---|" * len(BUDGETS) + "---|---|---|"]
    for label, r in configs.items():
        perm = r.order(n if r.depth else 0, sims, logits if r.model else None, priority)
        f = dict(zip(metrics.FIELDS, metrics.evaluate(head[perm], np.cumsum(table.tokens[head[perm]]), chars, g), strict=True))
        rec, pre = f["recall_t1000"], f["precision_t1000"]
        f1 = 2 * rec * pre / (rec + pre) if rec + pre else 0.0
        w.append(f"| {label} | {f['first']:,.0f} | {f['every']:,.0f} | {f['first_k']:.0f} | "
                 + " | ".join(f"{int(f['first'] <= b)}/{int(f['every'] <= b)}" for b in BUDGETS)
                 + f" | {rec:.2f} | {pre:.3f} | {f1:.3f} |")
    w += ["", "## 6. From one query to the reported numbers\n",
          "- Every per-query value above is averaged over the test queries with equal weight: coverage@B becomes the "
          "share of queries covered, recall@B the mean per-query recall, F1 the mean per-query F1.",
          "- Reader budget B is continuous: each query's *first* and *every* costs give its coverage at every B.",
          "- Overfetch depth is free: a shallower depth reorders a prefix of the same scored pool.",
          "- Reranker cost is the mean of the per-query token counts in §4; total cost = B × LLM price + reranker tokens × "
          "reranker price, compared at a continuous price ratio (break_even.png).",
          "- Differences between configurations are paired on the same queries; intervals bootstrap the dataset's "
          "clusters (documents, dialogues or issues)."]
    return "\n".join(w) + "\n"


def run_history(dataset: str, chunking_name: str, qid: str, model: str = "q06-local", pool_seeds: int = 50,
                expand: int = 3, embedder: str = EMBEDDER) -> str:
    """The same trace for one question inside its own history, as EventMemory would serve it."""
    from rbench import histories as hs
    from rbench.simulate_histories import gold_for, keys_for, read

    rows = queries(dataset)
    qi = next(i for i, q in enumerate(rows) if q["qid"] == qid)
    q = rows[qi]
    query = q["variants"]["question"]
    import asyncio

    from rbench.simulate_histories import first_stage

    hs.ingest(embedder, dataset, chunking_name, [qid])  # encoded once (embedded when not yet)
    seg, head, sims, units_all, texts_all = asyncio.run(first_stage(embedder, dataset, chunking_name, qid, query, (expand,)))
    head = np.asarray(head)
    producer = {u: p for u, p in zip(seg.uuid, _producers(dataset, qid, seg), strict=True)}
    g = gold_for(dataset, qid, seg)
    union = g.union
    w = [f"# Dry run: {dataset} · {qid} · {chunking_name} · {RERANKERS[model].hf_model}\n",
         f"## 1. The question ({q['features']['question_type']}, asked {q['features']['question_date']})\n",
         "> " + query.replace("\n", "\n> "), "",
         f"The history: {len(seg):,} EventMemory segments ({chunking_name}; mean {np.mean([len(t) for t in seg.texts]):,.0f} "
         "characters), "
         f"{int(seg.event.max()) + 1:,} turns, {len(set(seg.session.tolist()))} sessions, in timestamp order.\n"]

    turns = hs.gold_segments(dataset, qid, seg)
    w.append(f"## 2. The gold: {len(turns)} turn(s) marked has_answer, all needed\n")
    for k, ids in enumerate(turns):
        w.append(f"- Turn {k + 1} ({producer[seg.uuid[ids[0]]]}, session {int(seg.session[ids[0]])}): "
                 f"“{_preview(''.join(seg.texts[i] for i in ids), 160)}” "
                 f"— segment(s) {ids} ({int(seg.tokens[ids].sum())} tokens). Complete once all are read; touched once any is.")
    w.append("")

    w += [f"## 3. First stage: EventMemory.query over this history ({embedder}, exact search), {len(head)} seeds\n",
          "| rank | segment | session | producer | tokens | cosine | gold | text |", "|---|---|---|---|---|---|---|---|"]
    for i, (c, sim) in enumerate(zip(head[:SHOW], sims[:SHOW], strict=True)):
        w.append(f"| {i + 1} | {c} | {int(seg.session[c])} | {producer[seg.uuid[c]]} | {seg.tokens[c]} | {sim:.3f} | "
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
          f"Each pair is the model card's prompt + instruction + query ({overhead} tokens) + one unit rendered as "
          "EventMemory's reranker sees it. Reranker tokens for this question (API billing / with a prefix cache): "
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

    d = Depth("n", n)
    configs = {"first stage": Ranker("first_stage"), f"first stage, expand {expand}": Ranker("first_stage", expand=expand),
               f"rerank seeds (depth {n})": Ranker("rerank", model, d),
               f"rerank seeds, then expand {expand}": Ranker("rerank", model, d, expand=expand),
               f"expand {expand}, then rerank windows": Ranker("xrerank", model, d, expand=expand),
               f"perfect reranker (depth {n}, evidence first)": Ranker("oracle", depth=d)}
    chars = np.asarray([len(t) for t in seg.texts])
    w += ["## 5. Reading each order as EventMemory renders it, and every measurement\n",
          "Segments are admitted as EventMemory's segment limit grows; the reader cost is the token count of the "
          "rendered context (headers, JSON text, windows merged). A reranked order reads only its pool. *first* / "
          "*every*: reader tokens when some / every gold turn is complete; *touch*: when touched.\n",
          "| order | first | every | first touch | every touch | admitted | " + " | ".join(f"every@{b}" for b in BUDGETS)
          + " | recall@1000 |", "|---|---|---|---|---|---|" + "---|" * len(BUDGETS) + "---|"]
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
                 f"{len(admitted)} | " + " | ".join(str(int(f['every'] <= b)) for b in BUDGETS)
                 + f" | {f['recall_t1000']:.2f} |")
    w += ["", "## 6. From one question to the reported numbers\n",
          "- Every per-question value above is averaged over the questions with equal weight.",
          "- Reader budget B is continuous: each question's costs give its coverage at every B.",
          "- Overfetch depth is free: a shallower depth reorders a prefix of the same scored pool, and that depth is "
          "the vector search limit of the reranked configuration.",
          "- Intervals bootstrap questions (each history is its own partition)."]
    return "\n".join(w) + "\n"
