"""F5 — Allen-relation evaluation.

Pipeline:
1. Load allen_docs + allen_queries + allen_gold.
2. Run the AllenExtractor over all docs AND queries.
3. Build an EventResolver over all docs; also extract per-query event
   pairs so the query's own (event, time) tuples are available.
4. For each query:
   - Base hybrid (T_and_S): standard interval-overlap retrieval + semantic
     rerank (mirrors eval.py baseline).
   - Allen retrieval: resolve the query's anchor event via EventResolver,
     then apply allen_retrieval.allen_retrieve with the query's relation.
5. Report R@5, R@10, MRR, NDCG@10 overall AND per relation.
6. Extraction quality on query side: did the extractor find the
   (relation, anchor) we expected?

Writes:
- results/allen_relations.json
- results/allen_relations.md
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from advanced_common import (
    Embedder,
    mean,
    ndcg_at_k,
    recall_at_k,
)
from advanced_common import (
    mrr as mrr_fn,
)
from allen_extractor import AllenExtractor
from allen_retrieval import allen_retrieve, te_interval
from allen_schema import AllenExpression
from event_resolver import EventResolver
from schema import TimeExpression, parse_iso, to_us
from scorer import Interval, score_pair
from store import IntervalStore

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DB_PATH = ROOT / "cache" / "allen_intervals.sqlite"
TOP_K = 10


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Build base hybrid (T + S) retrieval — mirrors eval.py rank_hybrid
# ---------------------------------------------------------------------------
def _flatten_query_intervals(te: TimeExpression) -> list[Interval]:
    out: list[Interval] = []
    if te.kind == "instant" and te.instant:
        out.append(
            Interval(
                earliest_us=to_us(te.instant.earliest),
                latest_us=to_us(te.instant.latest),
                best_us=to_us(te.instant.best) if te.instant.best else None,
                granularity=te.instant.granularity,
            )
        )
    elif te.kind == "interval" and te.interval:
        out.append(
            Interval(
                earliest_us=to_us(te.interval.start.earliest),
                latest_us=to_us(te.interval.end.latest),
                best_us=to_us(te.interval.start.best)
                if te.interval.start.best
                else None,
                granularity=te.interval.start.granularity,
            )
        )
    return out


def _temporal_retrieve(
    store: IntervalStore, query_exprs: list[TimeExpression]
) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for te in query_exprs:
        for qi in _flatten_query_intervals(te):
            rows = store.query_overlap(qi.earliest_us, qi.latest_us)
            best_per_doc: dict[str, float] = {}
            for _, doc_id, e_us, l_us, b_us, gran in rows:
                s = Interval(
                    earliest_us=e_us,
                    latest_us=l_us,
                    best_us=b_us,
                    granularity=gran,
                )
                sc = score_pair(qi, s)
                if sc > best_per_doc.get(doc_id, 0.0):
                    best_per_doc[doc_id] = sc
            for d, sc in best_per_doc.items():
                out[d] += sc
    return dict(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    # 1. Load
    if not (DATA_DIR / "allen_docs.jsonl").exists():
        import allen_synth

        allen_synth.main()

    docs = load_jsonl(DATA_DIR / "allen_docs.jsonl")
    queries = load_jsonl(DATA_DIR / "allen_queries.jsonl")
    gold = {
        r["query_id"]: {
            "relevant": set(r["relevant_doc_ids"]),
            "relation": r["relation"],
            "anchor_id": r["anchor_id"],
        }
        for r in load_jsonl(DATA_DIR / "allen_gold.jsonl")
    }
    print(f"Loaded {len(docs)} docs, {len(queries)} queries.")

    # 2. Allen extraction — docs + queries
    ex = AllenExtractor()

    async def extract_items(items):
        async def one(iid, text, ref):
            try:
                return iid, await ex.extract(text, ref)
            except Exception as e:
                print(f"  allen extract failed for {iid}: {e}")
                return iid, []

        return await asyncio.gather(*(one(*x) for x in items))

    print("Allen-extracting docs...")
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    doc_results = await extract_items(doc_items)
    print("Allen-extracting queries...")
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    q_results = await extract_items(q_items)
    ex.save()

    exprs_by_doc: dict[str, list[AllenExpression]] = dict(doc_results)
    exprs_by_query: dict[str, list[AllenExpression]] = dict(q_results)
    print(f"Extractor usage: {ex.usage}, cost=${ex.cost_usd():.4f}")

    # 3. Event resolver — index all docs
    print("Building event resolver index...")
    resolver = EventResolver()
    await resolver.index_docs(docs)
    print(
        f"Indexed {len(resolver.entries)} (event, time) pairs across {len(docs)} docs."
    )

    # 4. Build base hybrid (temporal-overlap + semantic rerank)
    #    Use just the .time field from AllenExpression -> TimeExpression.
    if DB_PATH.exists():
        DB_PATH.unlink()
    store = IntervalStore(DB_PATH)
    for doc_id, aes in exprs_by_doc.items():
        for ae in aes:
            try:
                store.insert_expression(doc_id, ae.time)
            except Exception as e:
                print(f"  insert failed for {doc_id}: {e}")

    # Semantic embeddings
    embedder = Embedder()
    doc_texts = {d["doc_id"]: d["text"] for d in docs}
    query_texts = {q["query_id"]: q["text"] for q in queries}

    all_texts = list(doc_texts.values()) + list(query_texts.values())
    await embedder.embed_batch(all_texts)
    embedder.save()

    async def _emb(text: str) -> np.ndarray:
        return await embedder.embed(text)

    doc_embs: dict[str, np.ndarray] = {}
    for did, t in doc_texts.items():
        doc_embs[did] = await _emb(t)
    query_embs: dict[str, np.ndarray] = {}
    for qid, t in query_texts.items():
        query_embs[qid] = await _emb(t)

    def _cos(a, b) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        return float(np.dot(a, b) / denom)

    def rank_hybrid(qid: str) -> list[tuple[str, float]]:
        q_aes = exprs_by_query.get(qid, [])
        q_tes = [ae.time for ae in q_aes]
        t_scores = _temporal_retrieve(store, q_tes)
        t_ranked = sorted(t_scores.items(), key=lambda x: x[1], reverse=True)
        if not t_ranked:
            # semantic fallback
            return sorted(
                ((did, _cos(query_embs[qid], v)) for did, v in doc_embs.items()),
                key=lambda x: x[1],
                reverse=True,
            )
        cand_ids = [d for d, _ in t_ranked[:20]]
        sem = {d: _cos(query_embs[qid], doc_embs[d]) for d in cand_ids}
        return sorted(sem.items(), key=lambda x: x[1], reverse=True)

    # Pre-resolve all doc-side event-anchor spans so the Allen retriever
    # can look them up synchronously.
    anchor_cache: dict[str, Any] = {}

    async def _prewarm_anchor(span: str) -> None:
        if span in anchor_cache:
            return
        e = await resolver.resolve(span)
        if e is not None:
            anchor_cache[span] = te_interval(e.time)
        else:
            anchor_cache[span] = None

    doc_anchor_spans: set[str] = set()
    for aes in exprs_by_doc.values():
        for ae in aes:
            if ae.anchor and ae.anchor.kind == "event":
                doc_anchor_spans.add(ae.anchor.span)
    for q in queries:
        doc_anchor_spans.add(q["anchor_span"])

    print(f"Prewarming resolver for {len(doc_anchor_spans)} anchor spans...")
    await asyncio.gather(*[_prewarm_anchor(s) for s in doc_anchor_spans])

    def sync_resolve(span: str):
        return anchor_cache.get(span)

    # 5. Allen retrieval
    async def rank_allen(qid: str, q: dict) -> tuple[list[tuple[str, float]], dict]:
        """Returns (ranked, debug_info)."""
        relation = q["relation"]
        anchor_span = q["anchor_span"]
        match = await resolver.resolve(anchor_span)
        debug = {
            "relation": relation,
            "anchor_span": anchor_span,
            "resolved": None,
            "match_span": None,
            "match_doc": None,
        }
        if match is None:
            return rank_hybrid(qid), debug
        debug["match_span"] = match.span
        debug["match_doc"] = match.doc_id
        debug["resolved"] = {
            "earliest": match.time.instant.earliest.isoformat()
            if match.time.instant
            else None,
            "latest": match.time.instant.latest.isoformat()
            if match.time.instant
            else None,
        }
        scores = allen_retrieve(
            relation,
            match.time,
            exprs_by_doc,
            resolve_anchor=sync_resolve,
            anchor_doc_id=match.doc_id,
        )
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if not ranked:
            return rank_hybrid(qid), debug
        return ranked, debug

    # 6. Evaluate
    per_relation: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "base_r5": [],
            "base_r10": [],
            "base_mrr": [],
            "base_ndcg10": [],
            "allen_r5": [],
            "allen_r10": [],
            "allen_mrr": [],
            "allen_ndcg10": [],
        }
    )
    overall = {
        "base_r5": [],
        "base_r10": [],
        "base_mrr": [],
        "base_ndcg10": [],
        "allen_r5": [],
        "allen_r10": [],
        "allen_mrr": [],
        "allen_ndcg10": [],
    }
    debug_per_query: dict[str, Any] = {}

    # Extraction-quality check on queries: did the extractor produce
    # the expected (relation, anchor_span~)?
    ext_hits = 0
    ext_total = 0
    relation_hits = 0
    anchor_hits = 0
    ext_details: list[dict[str, Any]] = []

    for q in queries:
        qid = q["query_id"]
        exp_rel = q["relation"]
        exp_anchor = q["anchor_span"]
        aes = exprs_by_query.get(qid, [])
        # Look for an AllenExpression that has (relation, anchor) labels.
        got_rel = None
        got_anchor = None
        for ae in aes:
            if ae.relation is not None and ae.anchor is not None:
                got_rel = ae.relation
                got_anchor = ae.anchor.span
                break
        ext_total += 1
        rel_ok = got_rel == exp_rel
        anchor_ok = got_anchor is not None and _anchor_text_similar(
            got_anchor, exp_anchor
        )
        if rel_ok:
            relation_hits += 1
        if anchor_ok:
            anchor_hits += 1
        if rel_ok and anchor_ok:
            ext_hits += 1
        ext_details.append(
            {
                "query_id": qid,
                "expected_relation": exp_rel,
                "got_relation": got_rel,
                "expected_anchor": exp_anchor,
                "got_anchor": got_anchor,
                "rel_ok": rel_ok,
                "anchor_ok": anchor_ok,
            }
        )

    # Retrieval eval
    for q in queries:
        qid = q["query_id"]
        g = gold.get(qid)
        if g is None:
            continue
        relevant = g["relevant"]
        relation = g["relation"]

        base_ranked = [d for d, _ in rank_hybrid(qid)]
        allen_ranked_pairs, dbg = await rank_allen(qid, q)
        allen_ranked = [d for d, _ in allen_ranked_pairs]
        debug_per_query[qid] = {
            "relation": relation,
            "anchor": q["anchor_span"],
            "gold": sorted(relevant),
            "base_top10": base_ranked[:10],
            "allen_top10": allen_ranked[:10],
            "resolver": dbg,
        }

        if relevant:
            br5 = recall_at_k(base_ranked, relevant, 5)
            br10 = recall_at_k(base_ranked, relevant, 10)
            bmrr = mrr_fn(base_ranked, relevant)
            bndcg = ndcg_at_k(base_ranked, relevant, 10)

            ar5 = recall_at_k(allen_ranked, relevant, 5)
            ar10 = recall_at_k(allen_ranked, relevant, 10)
            amrr = mrr_fn(allen_ranked, relevant)
            andcg = ndcg_at_k(allen_ranked, relevant, 10)

            overall["base_r5"].append(br5)
            overall["base_r10"].append(br10)
            overall["base_mrr"].append(bmrr)
            overall["base_ndcg10"].append(bndcg)
            overall["allen_r5"].append(ar5)
            overall["allen_r10"].append(ar10)
            overall["allen_mrr"].append(amrr)
            overall["allen_ndcg10"].append(andcg)

            per_relation[relation]["base_r5"].append(br5)
            per_relation[relation]["base_r10"].append(br10)
            per_relation[relation]["base_mrr"].append(bmrr)
            per_relation[relation]["base_ndcg10"].append(bndcg)
            per_relation[relation]["allen_r5"].append(ar5)
            per_relation[relation]["allen_r10"].append(ar10)
            per_relation[relation]["allen_mrr"].append(amrr)
            per_relation[relation]["allen_ndcg10"].append(andcg)

    # 7. Assemble metrics
    def _avg(xs):
        return mean(xs) if xs else 0.0

    overall_metrics = {k: _avg(v) for k, v in overall.items()}
    per_rel_metrics = {
        rel: {k: _avg(v) for k, v in d.items()} for rel, d in per_relation.items()
    }
    ext_metrics = {
        "total_queries": ext_total,
        "both_rel_and_anchor_correct": ext_hits,
        "relation_correct": relation_hits,
        "anchor_similar": anchor_hits,
        "details": ext_details,
    }

    # 8. Event resolver quality: log each resolver resolution
    # (captured in debug_per_query)
    resolver_hit_rate = sum(
        1 for d in debug_per_query.values() if d["resolver"]["match_span"]
    ) / max(1, len(debug_per_query))

    cost = ex.cost_usd() + resolver.cost_usd()

    out = {
        "overall": overall_metrics,
        "per_relation": per_rel_metrics,
        "extraction_quality": {
            "total_queries": ext_metrics["total_queries"],
            "both_rel_and_anchor_correct": ext_metrics["both_rel_and_anchor_correct"],
            "relation_correct": ext_metrics["relation_correct"],
            "anchor_similar": ext_metrics["anchor_similar"],
        },
        "resolver_hit_rate": resolver_hit_rate,
        "cost_usd": cost,
        "extractor_usage": ex.usage,
        "debug": debug_per_query,
        "extraction_details": ext_details,
    }

    out_path_json = RESULTS_DIR / "allen_relations.json"
    with out_path_json.open("w") as f:
        json.dump(out, f, indent=2, default=str)

    # 9. Markdown report
    md_lines = [
        "# Allen-Relation Retrieval — Results\n",
        "\n## Extraction quality (query side)\n",
        f"- Queries: {ext_metrics['total_queries']}\n",
        f"- Relation correct: {ext_metrics['relation_correct']} "
        f"({ext_metrics['relation_correct'] / ext_metrics['total_queries']:.1%})\n",
        f"- Anchor similar (substring match): {ext_metrics['anchor_similar']} "
        f"({ext_metrics['anchor_similar'] / ext_metrics['total_queries']:.1%})\n",
        f"- Both correct: {ext_metrics['both_rel_and_anchor_correct']} "
        f"({ext_metrics['both_rel_and_anchor_correct'] / ext_metrics['total_queries']:.1%})\n",
        "\n## Event resolver hit rate\n",
        f"- Queries whose anchor resolved to a corpus event: {resolver_hit_rate:.1%}\n",
        "\n## Retrieval — Overall\n",
        "| Metric | Base hybrid | Allen | Δ |\n",
        "|---|---:|---:|---:|\n",
    ]
    for k_short, k_base, k_allen in [
        ("R@5", "base_r5", "allen_r5"),
        ("R@10", "base_r10", "allen_r10"),
        ("MRR", "base_mrr", "allen_mrr"),
        ("NDCG@10", "base_ndcg10", "allen_ndcg10"),
    ]:
        b = overall_metrics[k_base]
        a = overall_metrics[k_allen]
        md_lines.append(f"| {k_short} | {b:.3f} | {a:.3f} | {a - b:+.3f} |\n")

    md_lines.append("\n## Retrieval — per relation\n")
    md_lines.append(
        "| Relation | N | Base R@5 | Allen R@5 | Base MRR | Allen MRR | Base NDCG@10 | Allen NDCG@10 |\n"
    )
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for rel in ["before", "after", "during", "overlaps", "contains"]:
        d = per_rel_metrics.get(rel, {})
        if not d:
            continue
        n = len(
            [
                q
                for q in queries
                if gold.get(q["query_id"], {}).get("relation") == rel
                and gold[q["query_id"]]["relevant"]
            ]
        )
        md_lines.append(
            f"| {rel} | {n} | {d.get('base_r5', 0):.3f} | {d.get('allen_r5', 0):.3f} | "
            f"{d.get('base_mrr', 0):.3f} | {d.get('allen_mrr', 0):.3f} | "
            f"{d.get('base_ndcg10', 0):.3f} | {d.get('allen_ndcg10', 0):.3f} |\n"
        )

    md_lines.append("\n## Queries now answerable that weren't before\n")
    new_wins: list[str] = []
    for qid, d in debug_per_query.items():
        g = gold[qid]
        if not g["relevant"]:
            continue
        br5 = recall_at_k(d["base_top10"], g["relevant"], 5)
        ar5 = recall_at_k(d["allen_top10"], g["relevant"], 5)
        if ar5 > br5 + 1e-6:
            new_wins.append(
                f"- **{qid}** ({g['relation']}): base R@5 {br5:.2f} → Allen R@5 {ar5:.2f}"
            )
            new_wins.append(
                f"  - gold: {g['relevant']}; base top-5: {d['base_top10'][:5]}; allen top-5: {d['allen_top10'][:5]}"
            )
    md_lines.append("\n".join(new_wins) + "\n" if new_wins else "(none)\n")

    md_lines.append("\n## Failure analysis — Allen losses vs base\n")
    allen_losses: list[str] = []
    for qid, d in debug_per_query.items():
        g = gold[qid]
        if not g["relevant"]:
            continue
        br5 = recall_at_k(d["base_top10"], g["relevant"], 5)
        ar5 = recall_at_k(d["allen_top10"], g["relevant"], 5)
        if ar5 < br5 - 1e-6:
            allen_losses.append(
                f"- **{qid}** ({g['relation']}): base R@5 {br5:.2f} → Allen R@5 {ar5:.2f}; "
                f"resolver match: {d['resolver'].get('match_span')}"
            )
    md_lines.append("\n".join(allen_losses) + "\n" if allen_losses else "(none)\n")

    md_lines.append("\n## Cost\n")
    md_lines.append(
        f"- Total LLM cost: ${cost:.4f} "
        f"(extractor ${ex.cost_usd():.4f} + resolver ${resolver.cost_usd():.4f})\n"
    )

    out_path_md = RESULTS_DIR / "allen_relations.md"
    with out_path_md.open("w") as f:
        f.writelines(md_lines)

    print(f"Wrote {out_path_md} and {out_path_json}")
    print(json.dumps(overall_metrics, indent=2))


def _anchor_text_similar(got: str, expected: str) -> bool:
    g = got.lower().strip().strip(".")
    e = expected.lower().strip().strip(".")
    if not g or not e:
        return False
    if g == e:
        return True
    # Substring both directions
    if g in e or e in g:
        return True
    # Token overlap
    gs = set(g.replace(",", " ").split())
    es = set(e.replace(",", " ").split())
    if not gs or not es:
        return False
    inter = gs & es
    # Must share at least one non-stopword token
    stop = {"my", "the", "a", "an", "to", "of"}
    if inter - stop:
        return True
    return False


if __name__ == "__main__":
    asyncio.run(main())
