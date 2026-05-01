"""Run extractor_v2' on the axis corpus, compare with v1 extractions under
the multi-axis scorer, and write results/extractor_v2p.{md,json}.

Steps:
 1. Sanity sample on 3 axis queries + 3 axis docs (fail fast).
 2. Full extraction on axis corpus (15 docs + 20 queries) + 10 sampled
    base queries for regression check.
 3. Evaluate:
      (a) v1 extractions + multi-axis scorer (α=0.5/β=0.35/γ=0.15),
      (b) v2' extractions + multi-axis scorer (same weights).
    on axis subset (R@5, MRR, NDCG@10). Same for base regression.
 4. Write results/extractor_v2p.md and .json.

Hard concurrency: Semaphore(5). Per-call timeout: 30s (via asyncio.wait_for).
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from axis_distributions import (
    AXES,
    AxisDistribution,
)
from extractor import Extractor, LLMCache
from extractor_v2p import ExtractorV2P
from multi_axis_eval import (
    build_doc_memory,
    build_query_memory,
    eval_rankings,
    rank_interval,
    rank_multi_axis,
)
from schema import TimeExpression, parse_iso

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR = ROOT / "cache"
V2P_CACHE_DIR = CACHE_DIR / "extractor_v2p"
V2P_CACHE_DIR.mkdir(exist_ok=True, parents=True)


BEST_ALPHA = 0.5
BEST_BETA = 0.35
BEST_GAMMA = 0.15

PER_CALL_TIMEOUT = 30.0
CONCURRENCY = 5

# gpt-5-mini pricing
PRICE_IN = 0.25 / 1_000_000
PRICE_OUT = 2.00 / 1_000_000


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


async def extract_with_v2p(
    items: list[tuple[str, str, datetime]],
    label: str,
) -> tuple[dict[str, list[TimeExpression]], dict[str, int]]:
    ex = ExtractorV2P(concurrency=CONCURRENCY)
    sem = ex.sem  # reuse the semaphore the extractor uses internally

    async def one(iid: str, text: str, ref: datetime):
        try:
            tes = await asyncio.wait_for(
                ex.extract(text, ref), timeout=PER_CALL_TIMEOUT * 4
            )
        except asyncio.TimeoutError:
            print(f"  [v2p] timeout for {iid}")
            tes = []
        except Exception as e:
            print(f"  [v2p] extract failed for {iid}: {e}")
            tes = []
        return iid, tes

    print(f"[v2p] extracting {label} ({len(items)} items)...")
    results = await asyncio.gather(*(one(*it) for it in items))
    ex.cache.save()
    ex.shared_pass2_cache.save()
    print(
        f"[v2p] {label} usage: input={ex.usage['input']}, output={ex.usage['output']}"
    )
    return {i: t for i, t in results}, ex.usage


async def extract_with_v1(
    items: list[tuple[str, str, datetime]],
    cache_file: Path,
    label: str,
) -> tuple[dict[str, list[TimeExpression]], dict[str, int]]:
    ex = Extractor(concurrency=CONCURRENCY)
    ex.cache = LLMCache(path=cache_file)

    async def one(iid: str, text: str, ref: datetime):
        try:
            tes = await asyncio.wait_for(
                ex.extract(text, ref), timeout=PER_CALL_TIMEOUT * 4
            )
        except asyncio.TimeoutError:
            print(f"  [v1] timeout for {iid}")
            tes = []
        except Exception as e:
            print(f"  [v1] extract failed for {iid}: {e}")
            tes = []
        return iid, tes

    print(f"[v1] extracting {label} ({len(items)} items, cache={cache_file.name})...")
    results = await asyncio.gather(*(one(*it) for it in items))
    ex.cache.save()
    print(f"[v1] {label} usage: input={ex.usage['input']}, output={ex.usage['output']}")
    return {i: t for i, t in results}, ex.usage


AXIS_SURFACE_PATTERNS = {
    "bare_month": {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    },
    "quarter": {"q1", "q2", "q3", "q4"},
    "season": {
        "spring",
        "summer",
        "autumn",
        "fall",
        "winter",
    },
    "part_of_day": {
        "morning",
        "afternoon",
        "evening",
        "night",
        "dawn",
        "dusk",
        "mornings",
        "afternoons",
        "evenings",
        "nights",
    },
    "weekend_weekday": {
        "weekend",
        "weekends",
        "weekday",
        "weekdays",
    },
}


def axis_surface_type(surface: str) -> str | None:
    toks = surface.lower().split()
    # Check for compound with a bare axis word.
    for word in toks:
        word_clean = word.strip(",.!?;:'\"")
        for cat, vocab in AXIS_SURFACE_PATTERNS.items():
            if word_clean in vocab:
                return cat
    return None


def is_axis_only_resolution(te: TimeExpression) -> bool:
    """A resolution counts as 'axis-only' if it is a recurrence whose
    anchor has no meaningful earliest<latest spread beyond a second."""
    if te.kind != "recurrence" or te.recurrence is None:
        return False
    r = te.recurrence
    rrule = r.rrule.upper()
    # Any RRULE with a YEARLY/DAILY/WEEKLY freq and BY-clause that
    # constrains some axis is axis-expressive.
    if any(k in rrule for k in ("BYMONTH=", "BYHOUR=", "BYDAY=")):
        return True
    return False


def summarize_axis_extractions(
    texts: dict[str, str],
    extractions: dict[str, list[TimeExpression]],
) -> dict[str, Any]:
    """Rate of axis-surface extraction per category.

    For each text containing a bare axis word, check if at least one
    extraction covered it (by surface containment)."""
    per_text: list[dict[str, Any]] = []
    cat_hit: dict[str, int] = defaultdict(int)
    cat_total: dict[str, int] = defaultdict(int)
    for iid, text in texts.items():
        tlow = text.lower()
        # Find axis words in the text.
        matched_cats: set[str] = set()
        for cat, vocab in AXIS_SURFACE_PATTERNS.items():
            for w in vocab:
                # word-boundary-ish match
                tokens = [t.strip(",.!?;:'\"") for t in tlow.split()]
                if w in tokens:
                    matched_cats.add(cat)
                    break
        if not matched_cats:
            continue
        tes = extractions.get(iid, [])
        surfaces = [te.surface for te in tes]
        surfaces_low = [s.lower() for s in surfaces]
        per_cat_hit: dict[str, bool] = {}
        for cat in matched_cats:
            cat_total[cat] += 1
            # A hit is any extraction whose surface contains an axis word in this cat.
            hit = False
            for s in surfaces_low:
                s_tokens = [t.strip(",.!?;:'\"") for t in s.split()]
                if any(w in s_tokens for w in AXIS_SURFACE_PATTERNS[cat]):
                    hit = True
                    break
            per_cat_hit[cat] = hit
            if hit:
                cat_hit[cat] += 1
        per_text.append(
            {
                "iid": iid,
                "text": text,
                "axis_categories_in_text": sorted(matched_cats),
                "extracted_surfaces": surfaces,
                "hit_by_category": per_cat_hit,
            }
        )
    rates = {
        cat: {
            "hit": cat_hit[cat],
            "total": cat_total[cat],
            "rate": (cat_hit[cat] / cat_total[cat]) if cat_total[cat] else float("nan"),
        }
        for cat in AXIS_SURFACE_PATTERNS
    }
    overall_hit = sum(cat_hit.values())
    overall_total = sum(cat_total.values())
    rates["OVERALL"] = {
        "hit": overall_hit,
        "total": overall_total,
        "rate": (overall_hit / overall_total) if overall_total else float("nan"),
    }
    return {"per_category": rates, "per_text": per_text}


def rank_all(fn, all_qids, query_mem):
    out: dict[str, list[str]] = {}
    for qid in all_qids:
        qm = query_mem.get(qid)
        if qm is None:
            qm = {
                "intervals": [],
                "axes_merged": {
                    a: AxisDistribution(axis=a, values={}, informative=False)
                    for a in AXES
                },
                "axes_per_expr": [],
                "multi_tags": set(),
                "hier_tags": set(),
            }
        ranked = fn(qm)
        out[qid] = [d for d, _ in ranked]
    return out


async def sanity_sample(v2p_ex: ExtractorV2P) -> bool:
    """Run a small sanity probe on 3 axis queries and 3 axis docs.

    Returns True if all extract non-empty axis-expressive refs."""
    ref = parse_iso("2026-04-23T12:00:00Z")
    samples_q = [
        ("sanity_q_mar", "what happens in March?"),
        ("sanity_q_q2", "anything in Q2?"),
        ("sanity_q_afternoon", "afternoon events?"),
    ]
    samples_d = [
        ("sanity_d_june", "I vacation every June"),
        ("sanity_d_tue_ev", "Tuesday evenings are book club"),
        ("sanity_d_q2", "Q2 is my busiest quarter"),
    ]
    print("\n--- Sanity sample ---")
    ok = True
    for iid, text in samples_q + samples_d:
        try:
            tes = await asyncio.wait_for(
                v2p_ex.extract(text, ref), timeout=PER_CALL_TIMEOUT * 4
            )
        except Exception as e:
            print(f"  {iid!r:<25} ({text!r}) -> FAILED: {e}")
            ok = False
            continue
        surfs = [(te.surface, te.kind) for te in tes]
        print(f"  {iid!r:<25} ({text!r}) -> {surfs}")
        if not surfs:
            ok = False
    v2p_ex.cache.save()
    v2p_ex.shared_pass2_cache.save()
    print(f"--- Sanity sample OK={ok} ---\n")
    return ok


async def run() -> None:
    t0 = time.time()

    # ---------- Load corpora ----------
    axis_docs = load_jsonl(DATA_DIR / "axis_docs.jsonl")
    axis_queries = load_jsonl(DATA_DIR / "axis_queries.jsonl")
    axis_gold = {
        r["query_id"]: set(r["relevant_doc_ids"])
        for r in load_jsonl(DATA_DIR / "axis_gold.jsonl")
    }

    base_docs = load_jsonl(DATA_DIR / "docs.jsonl")
    base_queries_all = load_jsonl(DATA_DIR / "queries.jsonl")
    base_gold = {
        r["query_id"]: set(r["relevant_doc_ids"])
        for r in load_jsonl(DATA_DIR / "gold.jsonl")
    }

    # Sample 10 base queries for regression check (seeded).
    rnd = random.Random(42)
    sampled_base_queries = rnd.sample(base_queries_all, 10)
    sampled_base_qids = {q["query_id"] for q in sampled_base_queries}

    print(
        f"Corpus: {len(axis_docs)} axis docs, {len(axis_queries)} axis queries, "
        f"{len(base_docs)} base docs, {len(sampled_base_queries)} sampled base queries"
    )

    # ---------- Sanity sample ----------
    v2p_sanity = ExtractorV2P(concurrency=CONCURRENCY)
    try:
        sanity_ok = await asyncio.wait_for(sanity_sample(v2p_sanity), timeout=180.0)
    except asyncio.TimeoutError:
        print("Sanity sample timed out — aborting.")
        return
    if not sanity_ok:
        print("WARNING: sanity sample has empty extractions. Continuing anyway.")

    # ---------- Build item lists ----------
    # For v2': extract axis docs + axis queries + sampled base queries.
    # We also need v2' extractions on BASE docs so the base regression check
    # includes v2' as the doc side. But that would blow budget. Instead the
    # base regression is run by replacing axis-doc indexes while keeping
    # v1-extracted base docs. Same for base queries on the base-only side.
    # Simpler approach: for multi-axis ranking, we rebuild a doc set that is
    # (base_docs extracted with v1) + (axis_docs extracted with v2') vs
    # (base_docs extracted with v1) + (axis_docs extracted with v1), and
    # queries extracted with v2' and v1 respectively.

    axis_doc_items = [
        (d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in axis_docs
    ]
    axis_query_items = [
        (q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in axis_queries
    ]
    sampled_base_query_items = [
        (q["query_id"], q["text"], parse_iso(q["ref_time"]))
        for q in sampled_base_queries
    ]
    base_doc_items = [
        (d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in base_docs
    ]
    # For axis regression we need ALL base queries ranked (gold is fixed),
    # but cost-wise we only care about sampled_base_qids rows. We can reuse
    # v1 base-queries from the shared cache (already extracted).
    base_query_items_all = [
        (q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in base_queries_all
    ]

    # ---------- v1 extractions (from existing shared cache) ----------
    v1_cache = CACHE_DIR / "llm_cache.json"
    v1_axis_docs, u1a = await extract_with_v1(
        axis_doc_items, CACHE_DIR / "multi_axis" / "llm_cache.json", "v1/axis-docs"
    )
    v1_axis_queries, u1b = await extract_with_v1(
        axis_query_items,
        CACHE_DIR / "multi_axis" / "llm_cache.json",
        "v1/axis-queries",
    )
    v1_base_docs, u1c = await extract_with_v1(base_doc_items, v1_cache, "v1/base-docs")
    v1_base_queries_all, u1d = await extract_with_v1(
        base_query_items_all, v1_cache, "v1/base-queries-all"
    )

    # ---------- v2' extractions ----------
    # Axis docs + axis queries + sampled base queries only.
    v2p_axis_docs, u2a = await extract_with_v2p(axis_doc_items, "v2p/axis-docs")
    v2p_axis_queries, u2b = await extract_with_v2p(axis_query_items, "v2p/axis-queries")
    v2p_base_queries_sampled, u2c = await extract_with_v2p(
        sampled_base_query_items, "v2p/base-queries-sampled"
    )

    v2p_new_input = u2a["input"] + u2b["input"] + u2c["input"]
    v2p_new_output = u2a["output"] + u2b["output"] + u2c["output"]
    v2p_cost = v2p_new_input * PRICE_IN + v2p_new_output * PRICE_OUT
    print(
        f"\nv2' total new LLM cost: ${v2p_cost:.4f} "
        f"(in={v2p_new_input}, out={v2p_new_output})"
    )

    # ---------- Load v2 (existing) extractions for comparison ----------
    # We extract v2 on axis corpus too, since v2's cache may not have it.
    # Skip if cache-present. Keep cost small by only running on axis items.
    from extractor_v2 import ExtractorV2

    v2_ex = ExtractorV2(concurrency=CONCURRENCY)

    async def v2_extract(items, label):
        print(f"[v2] extracting {label} ({len(items)} items)...")

        async def one(iid, text, ref):
            try:
                tes = await asyncio.wait_for(
                    v2_ex.extract(text, ref), timeout=PER_CALL_TIMEOUT * 4
                )
            except Exception as e:
                print(f"  [v2] {iid}: {e}")
                tes = []
            return iid, tes

        res = await asyncio.gather(*(one(*it) for it in items))
        v2_ex.cache.save()
        v2_ex.shared_pass2_cache.save()
        print(
            f"[v2] {label} usage: input={v2_ex.usage['input']}, "
            f"output={v2_ex.usage['output']}"
        )
        return {i: t for i, t in res}

    v2_axis_docs = await v2_extract(axis_doc_items, "axis-docs")
    v2_axis_queries = await v2_extract(axis_query_items, "axis-queries")

    # ---------- Axis-surface extraction rate ----------
    print("\n--- Axis surface extraction rate ---")
    q_texts = {q["query_id"]: q["text"] for q in axis_queries}
    d_texts = {d["doc_id"]: d["text"] for d in axis_docs}

    rate_v1_q = summarize_axis_extractions(q_texts, v1_axis_queries)
    rate_v2_q = summarize_axis_extractions(q_texts, v2_axis_queries)
    rate_v2p_q = summarize_axis_extractions(q_texts, v2p_axis_queries)

    rate_v1_d = summarize_axis_extractions(d_texts, v1_axis_docs)
    rate_v2_d = summarize_axis_extractions(d_texts, v2_axis_docs)
    rate_v2p_d = summarize_axis_extractions(d_texts, v2p_axis_docs)

    def _fmt_rate(x: dict) -> str:
        return " ".join(
            f"{cat}={v['hit']}/{v['total']}"
            for cat, v in x["per_category"].items()
            if v["total"] > 0
        )

    print(f"v1  queries: {_fmt_rate(rate_v1_q)}")
    print(f"v2  queries: {_fmt_rate(rate_v2_q)}")
    print(f"v2p queries: {_fmt_rate(rate_v2p_q)}")
    print(f"v1  docs:    {_fmt_rate(rate_v1_d)}")
    print(f"v2  docs:    {_fmt_rate(rate_v2_d)}")
    print(f"v2p docs:    {_fmt_rate(rate_v2p_d)}")

    # ---------- Multi-axis eval ----------
    # Build doc and query memories for each variant.
    # V1: base_docs(v1) + axis_docs(v1) with queries v1
    # V2': base_docs(v1) + axis_docs(v2p) with queries v2p on axis qids + v1 on base
    v1_doc_ext = {**v1_base_docs, **v1_axis_docs}
    v1_query_ext = {**v1_base_queries_all, **v1_axis_queries}
    v2p_doc_ext = {**v1_base_docs, **v2p_axis_docs}
    # For queries in the v2p variant: use v2p on axis and sampled base; for
    # the REMAINING non-sampled base queries, v1 is used to fill the rank
    # set. Both systems share the same base-doc extractions.
    v2p_query_ext = {**v1_base_queries_all}
    for qid, tes in v2p_axis_queries.items():
        v2p_query_ext[qid] = tes
    for qid, tes in v2p_base_queries_sampled.items():
        v2p_query_ext[qid] = tes

    # Also v2 doc ext + query ext, for completeness.
    v2_doc_ext = {**v1_base_docs, **v2_axis_docs}
    v2_query_ext = {**v1_base_queries_all, **v2_axis_queries}

    print("\n--- Building memory (v1) ---")
    v1_doc_mem = build_doc_memory(v1_doc_ext)
    v1_query_mem = build_query_memory(v1_query_ext)

    print("--- Building memory (v2) ---")
    v2_doc_mem = build_doc_memory(v2_doc_ext)
    v2_query_mem = build_query_memory(v2_query_ext)

    print("--- Building memory (v2p) ---")
    v2p_doc_mem = build_doc_memory(v2p_doc_ext)
    v2p_query_mem = build_query_memory(v2p_query_ext)

    # All doc ids union
    all_doc_ids = [d["doc_id"] for d in base_docs] + [d["doc_id"] for d in axis_docs]
    for mem in (v1_doc_mem, v2_doc_mem, v2p_doc_mem):
        for did in all_doc_ids:
            if did not in mem:
                mem[did] = {
                    "intervals": [],
                    "axes_merged": {
                        a: AxisDistribution(axis=a, values={}, informative=False)
                        for a in AXES
                    },
                    "axes_per_expr": [],
                    "multi_tags": set(),
                    "hier_tags": set(),
                }

    axis_qids = {q["query_id"] for q in axis_queries}
    all_gold = {
        **base_gold,
        **{q["query_id"]: set(q.get("relevant_doc_ids", [])) for q in []},
    }
    # axis_gold is computed above; merge it explicitly:
    all_gold.update(axis_gold)

    all_qids = axis_qids | sampled_base_qids

    def run_variant(
        name: str,
        doc_mem: dict,
        query_mem: dict,
    ) -> dict[str, Any]:
        ma = rank_all(
            lambda qm: rank_multi_axis(qm, doc_mem, BEST_ALPHA, BEST_BETA, BEST_GAMMA),
            all_qids,
            query_mem,
        )
        iv = rank_all(lambda qm: rank_interval(qm, doc_mem), all_qids, query_mem)
        ax = eval_rankings(ma, all_gold, axis_qids)
        base = eval_rankings(ma, all_gold, sampled_base_qids)
        ax_iv = eval_rankings(iv, all_gold, axis_qids)
        base_iv = eval_rankings(iv, all_gold, sampled_base_qids)
        return {
            "multi_axis_best_weights": {
                "alpha": BEST_ALPHA,
                "beta": BEST_BETA,
                "gamma": BEST_GAMMA,
            },
            "multi_axis": {"axis": ax, "base_sampled": base},
            "interval_only": {"axis": ax_iv, "base_sampled": base_iv},
            "rankings_axis_sample": {
                qid: ma.get(qid, [])[:5] for qid in sorted(axis_qids)
            },
        }

    print("\n--- Running v1 + axis scorer ---")
    v1_res = run_variant("v1", v1_doc_mem, v1_query_mem)
    print("--- Running v2 + axis scorer ---")
    v2_res = run_variant("v2", v2_doc_mem, v2_query_mem)
    print("--- Running v2' + axis scorer ---")
    v2p_res = run_variant("v2p", v2p_doc_mem, v2p_query_mem)

    # ---------- Report ----------
    out_json: dict[str, Any] = {
        "weights": {"alpha": BEST_ALPHA, "beta": BEST_BETA, "gamma": BEST_GAMMA},
        "axis_surface_extraction_rate": {
            "queries": {
                "v1": rate_v1_q,
                "v2": rate_v2_q,
                "v2p": rate_v2p_q,
            },
            "docs": {
                "v1": rate_v1_d,
                "v2": rate_v2_d,
                "v2p": rate_v2p_d,
            },
        },
        "results": {
            "v1": v1_res,
            "v2": v2_res,
            "v2p": v2p_res,
        },
        "cost": {
            "v2p_new_usage": {"input": v2p_new_input, "output": v2p_new_output},
            "v2p_new_cost_usd": v2p_cost,
        },
        "meta": {
            "n_axis_docs": len(axis_docs),
            "n_axis_queries": len(axis_queries),
            "n_base_queries_sampled": len(sampled_base_queries),
            "sampled_base_qids": sorted(sampled_base_qids),
            "elapsed_s": time.time() - t0,
        },
    }

    # clean NaN
    def _clean(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, set):
            return sorted(o)
        if isinstance(o, float) and math.isnan(o):
            return None
        return o

    (RESULTS_DIR / "extractor_v2p.json").write_text(
        json.dumps(_clean(out_json), indent=2)
    )

    # Markdown
    md: list[str] = []
    md.append("# v2' extractor — axis-aware single-pass (v2-prime)\n\n")
    md.append(
        f"Weights held fixed at α={BEST_ALPHA}, β={BEST_BETA}, γ={BEST_GAMMA} "
        "(previous best MULTI-AXIS blend).\n\n"
    )
    md.append("## Axis-surface extraction rate (per category)\n\n")
    md.append("### Queries\n\n")
    md.append("| Category | v1 | v2 | v2' |\n|---|---:|---:|---:|\n")
    for cat in list(AXIS_SURFACE_PATTERNS.keys()) + ["OVERALL"]:

        def fmt(x):
            r = x["per_category"][cat]
            if r["total"] == 0:
                return "-"
            return f"{r['hit']}/{r['total']} ({r['rate']:.2f})"

        md.append(
            f"| {cat} | {fmt(rate_v1_q)} | {fmt(rate_v2_q)} | {fmt(rate_v2p_q)} |\n"
        )
    md.append("\n### Docs\n\n")
    md.append("| Category | v1 | v2 | v2' |\n|---|---:|---:|---:|\n")
    for cat in list(AXIS_SURFACE_PATTERNS.keys()) + ["OVERALL"]:

        def fmtd(x):
            r = x["per_category"][cat]
            if r["total"] == 0:
                return "-"
            return f"{r['hit']}/{r['total']} ({r['rate']:.2f})"

        md.append(
            f"| {cat} | {fmtd(rate_v1_d)} | {fmtd(rate_v2_d)} | {fmtd(rate_v2p_d)} |\n"
        )

    md.append("\n## Retrieval metrics (axis subset, 20 queries)\n\n")
    md.append("| Variant | R@5 | R@10 | MRR | NDCG@10 |\n|---|---:|---:|---:|---:|\n")
    for name, r in [
        ("v1 + multi-axis", v1_res),
        ("v2 + multi-axis", v2_res),
        ("v2' + multi-axis", v2p_res),
    ]:
        m = r["multi_axis"]["axis"]
        md.append(
            f"| {name} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | "
            f"{m['mrr']:.3f} | {m['ndcg@10']:.3f} |\n"
        )
    md.append("\n### Interval-only (no axis scorer) for reference:\n\n")
    md.append("| Variant | R@5 | MRR | NDCG@10 |\n|---|---:|---:|---:|\n")
    for name, r in [
        ("v1 interval-only", v1_res),
        ("v2 interval-only", v2_res),
        ("v2' interval-only", v2p_res),
    ]:
        m = r["interval_only"]["axis"]
        md.append(
            f"| {name} | {m['recall@5']:.3f} | {m['mrr']:.3f} | {m['ndcg@10']:.3f} |\n"
        )

    md.append(
        "\n## Base regression check (10 sampled base queries, multi-axis blend)\n\n"
    )
    md.append("| Variant | R@5 | R@10 | MRR | NDCG@10 |\n|---|---:|---:|---:|---:|\n")
    for name, r in [("v1 base", v1_res), ("v2 base", v2_res), ("v2' base", v2p_res)]:
        m = r["multi_axis"]["base_sampled"]
        md.append(
            f"| {name} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | "
            f"{m['mrr']:.3f} | {m['ndcg@10']:.3f} |\n"
        )

    md.append("\n## Cost\n\n")
    md.append(f"- v2' new LLM usage: input={v2p_new_input}, output={v2p_new_output}\n")
    md.append(f"- v2' new LLM cost: ${v2p_cost:.4f}\n")
    md.append(f"- Wall time: {time.time() - t0:.1f} s\n")

    md.append("\n## Sample v2' axis-query extractions\n\n")
    for qid in sorted(axis_qids):
        tes = v2p_axis_queries.get(qid, [])
        md.append(f"- **{qid}**: {q_texts[qid]!r} -> ")
        if not tes:
            md.append("(no extractions)\n")
            continue
        parts = []
        for te in tes:
            if te.kind == "recurrence" and te.recurrence is not None:
                parts.append(f"rec[{te.surface!r} rrule={te.recurrence.rrule}]")
            elif te.kind == "instant" and te.instant is not None:
                parts.append(f"inst[{te.surface!r} gran={te.instant.granularity}]")
            else:
                parts.append(f"{te.kind}[{te.surface!r}]")
        md.append("; ".join(parts) + "\n")

    (RESULTS_DIR / "extractor_v2p.md").write_text("".join(md))

    # Console summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Axis subset R@5 (multi-axis scorer):")
    for name, r in [("v1", v1_res), ("v2", v2_res), ("v2p", v2p_res)]:
        m = r["multi_axis"]["axis"]
        print(
            f"  {name:<5}  R@5={m['recall@5']:.3f}  NDCG@10={m['ndcg@10']:.3f}  "
            f"MRR={m['mrr']:.3f}"
        )
    print("\nBase-sampled R@5 (multi-axis scorer):")
    for name, r in [("v1", v1_res), ("v2", v2_res), ("v2p", v2p_res)]:
        m = r["multi_axis"]["base_sampled"]
        print(f"  {name:<5}  R@5={m['recall@5']:.3f}  NDCG@10={m['ndcg@10']:.3f}")
    print(f"\nv2' cost: ${v2p_cost:.4f}  wall={time.time() - t0:.1f}s")
    print("Wrote results/extractor_v2p.{md,json}")


if __name__ == "__main__":
    asyncio.run(run())
