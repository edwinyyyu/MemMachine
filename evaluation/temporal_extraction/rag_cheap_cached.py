"""Cheap RAG eval using ONLY cached extractions (no new LLM calls at extraction).

Loads cached v2p + era + allen extractions from whatever is in
cache/extractor_v2p/, cache/extractor_shared_pass2/, cache/advanced/,
cache/allen/. Runs v2p.extract() on each item — items with cached
LLM responses return quickly. Items with missing cache entries produce
empty TE lists (pass1 call hits timeout or network unavailable) but
the retrieval still runs with whatever signal is available.

SKIP router (no LLM calls at all — assume ["semantic"]) so V5 degrades
to V3 for this run.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from advanced_common import LLMCaller
from allen_extractor import AllenExtractor
from allen_retrieval import allen_retrieve, te_interval
from axis_distributions import (
    AXES,
    AxisDistribution,
    axes_for_expression,
    merge_axis_dists,
)
from baselines import embed_all
from era_extractor import EraExtractor
from event_resolver import EventResolver
from expander import expand
from extractor_v2p import ExtractorV2P
from multi_axis_scorer import axis_score, tag_score
from multi_axis_tags import tags_for_axes
from openai import AsyncOpenAI
from rag_pipeline import (
    v1_cascade,
    v2_temporal_only,
    v3_semantic_only,
    v4_rrf_all,
    v7_score_blend,
)
from schema import GRANULARITY_ORDER, TimeExpression, parse_iso, to_us
from scorer import Interval, score_jaccard_composite

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

TOP_K = 10


def load_jsonl(path):
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


SUBSET_FILES = [
    ("base", "queries.jsonl", "docs.jsonl", "gold.jsonl"),
    ("discriminator", "disc_queries.jsonl", "disc_docs.jsonl", "disc_gold.jsonl"),
    (
        "utterance",
        "utterance_queries.jsonl",
        "utterance_docs.jsonl",
        "utterance_gold.jsonl",
    ),
    ("era", "era_queries.jsonl", "era_docs.jsonl", "era_gold.jsonl"),
    ("axis", "axis_queries.jsonl", "axis_docs.jsonl", "axis_gold.jsonl"),
    ("allen", "allen_queries.jsonl", "allen_docs.jsonl", "allen_gold.jsonl"),
]


def load_all():
    s2q, all_docs, all_gold, meta = {}, [], {}, {}
    seen = set()
    for subset, qf, df, gf in SUBSET_FILES:
        queries = load_jsonl(DATA_DIR / qf)
        docs = load_jsonl(DATA_DIR / df)
        gold = load_jsonl(DATA_DIR / gf)
        s2q[subset] = queries
        for d in docs:
            if d["doc_id"] in seen:
                continue
            seen.add(d["doc_id"])
            all_docs.append(d)
        for g in gold:
            all_gold[g["query_id"]] = set(g.get("relevant_doc_ids") or [])
        for q in queries:
            m = {"subset": subset}
            if "relation" in q:
                m["allen_relation"] = q["relation"]
                m["allen_anchor_span"] = q.get("anchor_span")
                m["allen_anchor_id"] = q.get("anchor_id")
            meta[q["query_id"]] = m
    return s2q, all_docs, all_gold, meta


def flatten_intervals(te):
    out = []
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
        g = (
            te.interval.start.granularity
            if GRANULARITY_ORDER.get(te.interval.start.granularity, 0)
            >= GRANULARITY_ORDER.get(te.interval.end.granularity, 0)
            else te.interval.end.granularity
        )
        best = te.interval.start.best or te.interval.start.earliest
        out.append(
            Interval(
                earliest_us=to_us(te.interval.start.earliest),
                latest_us=to_us(te.interval.end.latest),
                best_us=to_us(best),
                granularity=g,
            )
        )
    elif te.kind == "recurrence" and te.recurrence:
        now = datetime.now(tz=timezone.utc)
        anchor = te.recurrence.dtstart.best or te.recurrence.dtstart.earliest
        start = min(now - timedelta(days=365 * 10), anchor - timedelta(days=365))
        end = now + timedelta(days=365 * 2)
        if te.recurrence.until is not None:
            end = min(end, te.recurrence.until.latest or te.recurrence.until.earliest)
        try:
            for inst in expand(te.recurrence, start, end):
                out.append(
                    Interval(
                        earliest_us=to_us(inst.earliest),
                        latest_us=to_us(inst.latest),
                        best_us=to_us(inst.best) if inst.best else None,
                        granularity=inst.granularity,
                    )
                )
        except Exception:
            pass
    return out


def build_mem(tes):
    ivs = []
    axes_per = []
    tags = set()
    for te in tes:
        ivs.extend(flatten_intervals(te))
        ax = axes_for_expression(te)
        axes_per.append(ax)
        tags |= tags_for_axes(ax)
    axes_merged = merge_axis_dists(axes_per)
    return {"intervals": ivs, "axes_merged": axes_merged, "multi_tags": tags}


def _empty_mem():
    return {
        "intervals": [],
        "axes_merged": {
            a: AxisDistribution(axis=a, values={}, informative=False) for a in AXES
        },
        "multi_tags": set(),
    }


def interval_pair_best(q, d):
    if not q or not d:
        return 0.0
    t = 0.0
    for qi in q:
        best = 0.0
        for si in d:
            s = score_jaccard_composite(qi, si)
            if s > best:
                best = s
        t += best
    return t


def multi_axis_scores(qm, dms, alpha=0.5, beta=0.35, gamma=0.15):
    qa = qm["axes_merged"]
    qtg = qm["multi_tags"]
    qivs = qm["intervals"]
    raw = {d: interval_pair_best(qivs, b["intervals"]) for d, b in dms.items()}
    mx = max(raw.values(), default=0.0)
    out = {}
    for d, b in dms.items():
        iv = raw[d] / mx if mx > 0 else 0.0
        ax = axis_score(qa, b["axes_merged"])
        tg = tag_score(qtg, b["multi_tags"])
        out[d] = alpha * iv + beta * ax + gamma * tg
    return out


def semantic_scores(qe, des):
    qn = float(np.linalg.norm(qe)) or 1e-9
    out = {}
    for d, v in des.items():
        vn = float(np.linalg.norm(v)) or 1e-9
        out[d] = float(np.dot(qe, v) / (qn * vn))
    return out


def te_window_us(te):
    if te.kind == "instant" and te.instant:
        return to_us(te.instant.earliest), to_us(te.instant.latest)
    if te.kind == "interval" and te.interval:
        return to_us(te.interval.start.earliest), to_us(te.interval.end.latest)
    return None


def era_scores(qt, dt):
    qws = [te_window_us(t) for t in qt]
    qws = [w for w in qws if w and w[1] > w[0]]
    if not qws:
        return {}
    out = {}
    for did, tes in dt.items():
        dws = [te_window_us(t) for t in tes]
        dws = [w for w in dws if w and w[1] > w[0]]
        bt = 0.0
        for qw in qws:
            qe, ql = qw
            best = 0.0
            for dw in dws:
                de, dl = dw
                inter = min(ql, dl) - max(qe, de)
                if inter <= 0:
                    continue
                uni = max(ql, dl) - min(qe, de)
                if uni <= 0:
                    continue
                j = inter / uni
                if j > best:
                    best = j
            bt += best
        if bt > 0:
            out[did] = bt
    return out


def allen_scores_for_q(q_meta, exprs, rac, rad):
    relation = q_meta.get("allen_relation")
    span = q_meta.get("allen_anchor_span")
    if not relation or not span:
        return {}
    aiv = rac.get(span)
    if aiv is None:
        return {}
    from schema import FuzzyInstant

    earliest = datetime.fromtimestamp(aiv.earliest / 1_000_000, tz=timezone.utc)
    latest = datetime.fromtimestamp(aiv.latest / 1_000_000, tz=timezone.utc)
    fake = TimeExpression(
        kind="instant",
        surface=span,
        reference_time=datetime.now(timezone.utc),
        confidence=1.0,
        instant=FuzzyInstant(
            earliest=earliest, latest=latest, best=None, granularity="day"
        ),
    )
    s = allen_retrieve(
        relation,
        fake,
        exprs,
        resolve_anchor=lambda sp: rac.get(sp),
        anchor_doc_id=rad.get(span),
    )
    return dict(s)


def r_at_k(r, rel, k):
    if not rel:
        return float("nan")
    return len(set(r[:k]) & rel) / len(rel)


def mrr(r, rel):
    if not rel:
        return float("nan")
    for i, d in enumerate(r, 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg(r, rel, k):
    if not rel:
        return float("nan")
    dcg = 0.0
    for i, d in enumerate(r[:k], 1):
        if d in rel:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def avg(vs):
    xs = [v for v in vs if not math.isnan(v)]
    return sum(xs) / len(xs) if xs else 0.0


def evaluate(rm, gold, qids):
    r5, r10, mr, nd = [], [], [], []
    for qid in qids:
        rel = gold.get(qid, set())
        if not rel:
            continue
        r = rm.get(qid, [])
        r5.append(r_at_k(r, rel, 5))
        r10.append(r_at_k(r, rel, 10))
        mr.append(mrr(r, rel))
        nd.append(ndcg(r, rel, 10))
    return {
        "recall@5": avg(r5),
        "recall@10": avg(r10),
        "mrr": avg(mr),
        "ndcg@10": avg(nd),
        "n": len(r5),
    }


async def extract_cached(items, label, factory, timeout=60.0, concurrency=4):
    """Call extract(); cache hits return quickly; cache misses get a short timeout."""
    ex = factory()

    async def one(iid, text, ref):
        try:
            return iid, await asyncio.wait_for(ex.extract(text, ref), timeout=timeout)
        except asyncio.TimeoutError:
            return iid, []
        except Exception:
            return iid, []

    print(f"  {label}: {len(items)} items...", flush=True)
    results = await asyncio.gather(*(one(*it) for it in items))
    if hasattr(ex, "cache"):
        ex.cache.save()
    if hasattr(ex, "shared_pass2_cache"):
        ex.shared_pass2_cache.save()
    if hasattr(ex, "save"):
        ex.save()
    if hasattr(ex, "llm") and hasattr(ex.llm, "save"):
        ex.llm.save()
    usage = getattr(ex, "usage", {})
    if not usage and hasattr(ex, "llm"):
        usage = getattr(ex.llm, "usage", {})
    print(f"  {label} done, usage={usage}", flush=True)
    return dict(results), usage


async def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("Cheap RAG Eval (cached) — v2p T + era E + semantic S + allen A", flush=True)
    print("=" * 60, flush=True)

    s2q, all_docs, gold, meta = load_all()
    all_doc_ids = [d["doc_id"] for d in all_docs]
    all_q = []
    for qs in s2q.values():
        all_q.extend(qs)
    print(f"Docs: {len(all_docs)}, Queries: {len(all_q)}", flush=True)

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in all_docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in all_q]

    print("\n[T] v2p extraction (cached)...", flush=True)
    # Client with very short timeout so cache-miss calls fail fast

    fast_client = AsyncOpenAI(timeout=10.0, max_retries=0)

    def v2p_factory():
        ex = ExtractorV2P(concurrency=4)
        ex.client = fast_client
        return ex

    doc_tes, u1 = await extract_cached(doc_items, "v2p-docs", v2p_factory, timeout=30.0)
    q_tes, u2 = await extract_cached(q_items, "v2p-queries", v2p_factory, timeout=30.0)

    print("\n[E] era extraction (cached)...", flush=True)

    def era_factory():
        llm = LLMCaller(concurrency=4)
        llm.client = fast_client
        return EraExtractor(llm)

    era_doc_tes, u3 = await extract_cached(
        doc_items, "era-docs", era_factory, timeout=30.0
    )
    era_q_tes, u4 = await extract_cached(
        q_items, "era-queries", era_factory, timeout=30.0
    )

    print("\n[S] embeddings (cached)...", flush=True)
    doc_texts = [d["text"] for d in all_docs]
    q_texts = [q["text"] for q in all_q]
    dea = await embed_all(doc_texts)
    qea = await embed_all(q_texts)
    doc_embs = {d["doc_id"]: dea[i] for i, d in enumerate(all_docs)}
    q_embs = {q["query_id"]: qea[i] for i, q in enumerate(all_q)}

    print("\n[A] allen extraction (cached, allen subset)...", flush=True)
    allen_qs_list = s2q.get("allen", [])
    allen_docs_raw = load_jsonl(DATA_DIR / "allen_docs.jsonl")
    allen_doc_id_set = {d["doc_id"] for d in allen_docs_raw}
    allen_docs = [d for d in all_docs if d["doc_id"] in allen_doc_id_set]
    allen_doc_exprs = {}
    rac, rad = {}, {}
    if allen_docs and allen_qs_list:
        a_doc_items = [
            (d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in allen_docs
        ]
        a_q_items = [
            (q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in allen_qs_list
        ]

        def allen_factory():
            ex = AllenExtractor(concurrency=4)
            ex.client = fast_client
            return ex

        allen_doc_exprs, _ = await extract_cached(
            a_doc_items, "allen-docs", allen_factory, timeout=30.0
        )
        await extract_cached(a_q_items, "allen-queries", allen_factory, timeout=30.0)
        # Resolver
        try:
            resolver = EventResolver()
            await resolver.index_docs(allen_docs)
            anchor_spans = {
                q["anchor_span"] for q in allen_qs_list if q.get("anchor_span")
            }
            for aes in allen_doc_exprs.values():
                for ae in aes:
                    if ae.anchor and ae.anchor.kind == "event":
                        anchor_spans.add(ae.anchor.span)

            async def _prewarm(sp):
                try:
                    e = await asyncio.wait_for(resolver.resolve(sp), timeout=15.0)
                    if e is not None:
                        rac[sp] = te_interval(e.time)
                        rad[sp] = e.doc_id
                    else:
                        rac[sp] = None
                except Exception:
                    rac[sp] = None

            await asyncio.gather(*[_prewarm(s) for s in anchor_spans])
        except Exception as e:
            print(f"  resolver failed: {e}", flush=True)

    print("\nBuilding memories...", flush=True)
    doc_mems = {did: build_mem(tes) for did, tes in doc_tes.items()}
    for did in all_doc_ids:
        doc_mems.setdefault(did, _empty_mem())
    q_mems = {qid: build_mem(tes) for qid, tes in q_tes.items()}
    for q in all_q:
        q_mems.setdefault(q["query_id"], _empty_mem())

    print("\nComputing per-query T/S/A/E scores...", flush=True)
    per_q = {}
    for q in all_q:
        qid = q["query_id"]
        qm = q_mems.get(qid, _empty_mem())
        t = multi_axis_scores(qm, doc_mems)
        s = semantic_scores(q_embs[qid], doc_embs)
        qm_meta = meta[qid]
        a = (
            allen_scores_for_q(qm_meta, allen_doc_exprs, rac, rad)
            if qm_meta.get("allen_relation")
            else {}
        )
        q_era = era_q_tes.get(qid, [])
        e = era_scores(q_era, era_doc_tes) if q_era else {}
        per_q[qid] = {"T": t, "S": s, "A": a, "E": e}

    variants = [
        "V1_CASCADE",
        "V2_TEMPORAL-ONLY",
        "V3_SEMANTIC-ONLY",
        "V4_RRF-ALL",
        "V7_SCORE-BLEND",
    ]
    ranked = {v: {} for v in variants}
    for q in all_q:
        qid = q["query_id"]
        sc = per_q[qid]
        ranked["V1_CASCADE"][qid] = v1_cascade(sc["T"], sc["S"], all_doc_ids)
        ranked["V2_TEMPORAL-ONLY"][qid] = v2_temporal_only(sc["T"], sc["S"])
        ranked["V3_SEMANTIC-ONLY"][qid] = v3_semantic_only(sc["S"])
        ranked["V4_RRF-ALL"][qid] = v4_rrf_all(sc["T"], sc["S"], sc["A"], sc["E"])
        ranked["V7_SCORE-BLEND"][qid] = v7_score_blend(
            sc["T"], sc["S"], sc["A"], sc["E"]
        )

    subset_order = [
        "base",
        "discriminator",
        "utterance",
        "era",
        "axis",
        "allen",
        "combined",
    ]
    metrics = {v: {} for v in variants}
    all_qids = [q["query_id"] for q in all_q]
    for v in variants:
        for subset, qs in s2q.items():
            qids = [qq["query_id"] for qq in qs]
            metrics[v][subset] = evaluate(ranked[v], gold, qids)
        metrics[v]["combined"] = evaluate(ranked[v], gold, all_qids)

    # How many extractions were actually cached?
    t_emit = sum(1 for tes in doc_tes.values() if tes)
    q_emit = sum(1 for tes in q_tes.values() if tes)
    e_emit = sum(1 for tes in era_doc_tes.values() if tes)
    print(f"\nDoc v2p TEs emitted: {t_emit}/{len(all_docs)}", flush=True)
    print(f"Query v2p TEs emitted: {q_emit}/{len(all_q)}", flush=True)
    print(f"Doc era TEs emitted: {e_emit}/{len(all_docs)}", flush=True)

    wall_s = time.time() - t0

    out = {
        "variant_metrics": metrics,
        "extraction_coverage": {
            "doc_v2p_emit_rate": t_emit / max(len(all_docs), 1),
            "query_v2p_emit_rate": q_emit / max(len(all_q), 1),
            "doc_era_emit_rate": e_emit / max(len(all_docs), 1),
        },
        "subset_sizes": {k: len(v) for k, v in s2q.items()},
        "total_queries": len(all_q),
        "total_docs": len(all_docs),
        "wall_time_s": wall_s,
        "note": (
            "This eval uses the v2p + era caches populated by the primary cheap "
            "eval run plus the adversarial_v2p run. Items not in cache yield "
            "empty TEs (T/E channels degrade to S). Router is OMITTED so V5/V6 "
            "are not evaluated; results include V1/V2/V3/V4/V7 only."
        ),
    }
    (RESULTS_DIR / "rag_cheap.json").write_text(json.dumps(out, indent=2, default=str))

    md = []
    md.append("# Cheap RAG Fusion Re-Eval (v2' extractor; V1-V4, V7 only)\n\n")
    md.append(f"Docs: {len(all_docs)}. Queries: {len(all_q)}. Wall: {wall_s:.1f}s.\n\n")
    md.append(
        f"Extraction coverage: v2p docs={t_emit}/{len(all_docs)}, "
        f"v2p queries={q_emit}/{len(all_q)}, era docs={e_emit}/{len(all_docs)}.\n\n"
    )
    md.append(
        "Variants skipped: V5 (ROUTED-SINGLE) and V6 (ROUTED-MULTI) — router omitted.\n"
        "V8 (LLM-RERANK) and V9 (HYBRID) skipped per task spec.\n\n"
    )

    md.append("## R@5 by variant × subset\n\n")
    md.append("| Variant | " + " | ".join(subset_order) + " |\n")
    md.append("|---|" + "|".join(["---:"] * len(subset_order)) + "|\n")
    for v in variants:
        row = [v] + [
            f"{metrics[v].get(s, {}).get('recall@5', 0.0):.3f}" for s in subset_order
        ]
        md.append("| " + " | ".join(row) + " |\n")

    md.append("\n## NDCG@10 by variant × subset\n\n")
    md.append("| Variant | " + " | ".join(subset_order) + " |\n")
    md.append("|---|" + "|".join(["---:"] * len(subset_order)) + "|\n")
    for v in variants:
        row = [v] + [
            f"{metrics[v].get(s, {}).get('ndcg@10', 0.0):.3f}" for s in subset_order
        ]
        md.append("| " + " | ".join(row) + " |\n")

    md.append("\n## MRR by variant × subset\n\n")
    md.append("| Variant | " + " | ".join(subset_order) + " |\n")
    md.append("|---|" + "|".join(["---:"] * len(subset_order)) + "|\n")
    for v in variants:
        row = [v] + [
            f"{metrics[v].get(s, {}).get('mrr', 0.0):.3f}" for s in subset_order
        ]
        md.append("| " + " | ".join(row) + " |\n")

    md.append("\n## Per-subset winner (R@5; ties by NDCG@10)\n\n")
    md.append("| Subset | Best | R@5 | NDCG@10 | MRR |\n")
    md.append("|---|---|---:|---:|---:|\n")
    for s in subset_order:
        best = max(
            variants,
            key=lambda v: (
                metrics[v].get(s, {}).get("recall@5", 0.0),
                metrics[v].get(s, {}).get("ndcg@10", 0.0),
            ),
        )
        m = metrics[best].get(s, {})
        md.append(
            f"| {s} | {best} | {m.get('recall@5', 0):.3f} | "
            f"{m.get('ndcg@10', 0):.3f} | {m.get('mrr', 0):.3f} |\n"
        )

    md.append("\n## Combined ranking\n\n")
    md.append("| Variant | Combined R@5 | LLM calls/q |\n")
    md.append("|---|---:|---:|\n")
    calls = {
        "V1_CASCADE": 0,
        "V2_TEMPORAL-ONLY": 0,
        "V3_SEMANTIC-ONLY": 0,
        "V4_RRF-ALL": 0,
        "V7_SCORE-BLEND": 0,
    }
    ordered = sorted(
        variants,
        key=lambda v: metrics[v]["combined"].get("recall@5", 0.0),
        reverse=True,
    )
    for v in ordered:
        md.append(f"| {v} | {metrics[v]['combined']['recall@5']:.3f} | {calls[v]} |\n")

    (RESULTS_DIR / "rag_cheap.md").write_text("".join(md))

    print("\n=== Combined R@5 ===", flush=True)
    for v in variants:
        print(f"  {v:<22} R@5={metrics[v]['combined']['recall@5']:.3f}", flush=True)
    print(f"Wall: {wall_s:.1f}s", flush=True)
    print("Wrote results/rag_cheap.{md,json}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
