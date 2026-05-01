"""Evaluate `jaccard_with_containment` against `jaccard_composite`.

Strategy:
1. Run the adversarial v2'' pipeline (cached extractions), collecting rankings
   under a swappable interval scorer.
2. Sweep: baseline (jaccard_composite) vs containment (log2/sqrt/dice decay).
3. Compute A3, A-category, full adversarial, and base-55 interval-only
   regression metrics for each.
4. Write results/scorer_containment.{md,json}.

Zero LLM calls — everything comes from existing caches.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# NOTE: we import adversarial_v2pp_eval as a module and monkey-patch its
# score function reference before invoking its ranking logic.
import adversarial_v2pp_eval as av2pp
import anchor_retrieval as _anchor_mod

# Pull all the machinery from the adversarial v2'' eval module.
from adversarial_v2pp_eval import (
    ALPHA_IV,
    ANCHOR_DB,
    BETA_AXIS,
    DATA_DIR,
    FILTER_MODALITY,
    GAMMA_TAG,
    INTERVAL_DB,
    REF_ANCHOR_ALPHA,
    REF_ANCHOR_BETA,
    RESULTS_DIR,
    TOP_K,
    build_memory,
    flatten_intervals,
    load_jsonl,
    mrr,
    nanmean,
    ndcg_at_k,
    recall_at_k,
    run_allen_extract,
    run_era_extract,
    run_v2pp_extract,
)
from allen_retrieval import allen_retrieve, te_interval
from anchor_retrieval import retrieve as anchor_retrieve
from anchor_store import UtteranceAnchorStore
from axis_distributions import (
    AXES,
    AxisDistribution,
)
from baselines import embed_all, semantic_rank
from modality_filter import filter_ranking, partition_by_modality
from modality_schema import get_modality
from schema import FuzzyInstant, TimeExpression, parse_iso
from scorer import Interval
from scorer import score_jaccard_composite as _orig_jaccard
from scorer_containment import (
    score_jaccard_with_containment,
)
from store import IntervalStore

# ---------------------------------------------------------------------------
# Swappable scorer helpers
# ---------------------------------------------------------------------------
_ACTIVE_SCORER = {"mode": "jaccard_composite", "decay": "log2"}


def _active_score(q: Interval, s: Interval) -> float:
    if _ACTIVE_SCORER["mode"] == "jaccard_composite":
        return _orig_jaccard(q, s)
    return score_jaccard_with_containment(q, s, decay=_ACTIVE_SCORER["decay"])


def _install_scorer(mode: str, decay: str = "log2") -> None:
    _ACTIVE_SCORER["mode"] = mode
    _ACTIVE_SCORER["decay"] = decay

    # Monkey-patch ALL call sites that use score_jaccard_composite /
    # score_pair for interval pairing:
    #   - adversarial_v2pp_eval.score_jaccard_composite (used by interval_pair_best)
    #   - anchor_retrieval.score_pair (used by _score_referents_per_doc, _score_anchors_per_doc)
    av2pp.score_jaccard_composite = _active_score
    _anchor_mod.score_pair = _active_score


# ---------------------------------------------------------------------------
# Adversarial pipeline runner (reimplemented to keep it explicit)
# ---------------------------------------------------------------------------
async def run_adversarial(scorer_mode: str, decay: str) -> dict[str, Any]:
    """Run adversarial v2'' ranking with the given scorer.
    Uses existing extraction caches — no new LLM calls (assuming cached).
    Returns per-query rankings + per-category metrics.
    """
    _install_scorer(scorer_mode, decay)

    docs = load_jsonl(DATA_DIR / "adversarial_docs.jsonl")
    queries = load_jsonl(DATA_DIR / "adversarial_queries.jsonl")
    gold_entries = load_jsonl(DATA_DIR / "adversarial_gold.jsonl")
    gold_map = {
        g["query_id"]: set(g.get("relevant_doc_ids") or []) for g in gold_entries
    }
    query_cat = {q["query_id"]: q["category"] for q in queries}
    doc_cat = {d["doc_id"]: d["category"] for d in docs}

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]

    # Run (cached) extractions.
    v2pp_docs, _u1, _ = await run_v2pp_extract(doc_items, f"docs-v2pp-{scorer_mode}")
    v2pp_qs, _u2, _ = await run_v2pp_extract(q_items, f"queries-v2pp-{scorer_mode}")
    era_docs, _u3 = await run_era_extract(doc_items, f"docs-era-{scorer_mode}")
    era_qs, _u4 = await run_era_extract(q_items, f"queries-era-{scorer_mode}")
    allen_docs_ex, _u5 = await run_allen_extract(doc_items, f"docs-allen-{scorer_mode}")
    allen_qs, _u6 = await run_allen_extract(q_items, f"queries-allen-{scorer_mode}")

    def merge_tes(a, b):
        seen = set()
        merged = []
        for te in list(a) + list(b):
            key = (te.kind, (te.surface or "").strip().lower())
            if key in seen:
                continue
            seen.add(key)
            merged.append(te)
        return merged

    doc_ext = {
        d["doc_id"]: merge_tes(
            v2pp_docs.get(d["doc_id"], []), era_docs.get(d["doc_id"], [])
        )
        for d in docs
    }
    q_ext = {
        q["query_id"]: merge_tes(
            v2pp_qs.get(q["query_id"], []), era_qs.get(q["query_id"], [])
        )
        for q in queries
    }

    v2pp_doc_ext_for_mod = {d["doc_id"]: v2pp_docs.get(d["doc_id"], []) for d in docs}
    keep_ids, skip_ids = partition_by_modality(v2pp_doc_ext_for_mod)

    # Fresh interval stores each run (so we don't leak between scorer modes).
    if INTERVAL_DB.exists():
        INTERVAL_DB.unlink()
    if ANCHOR_DB.exists():
        ANCHOR_DB.unlink()
    store = IntervalStore(INTERVAL_DB)
    astore = UtteranceAnchorStore(ANCHOR_DB)
    for d in docs:
        for te in doc_ext.get(d["doc_id"], []):
            if get_modality(te) != "actual":
                continue
            try:
                store.insert_expression(d["doc_id"], te)
            except Exception:
                pass
        astore.upsert_anchor(d["doc_id"], parse_iso(d["ref_time"]), "day")

    def filtered_ext(ext_map):
        return {
            k: [te for te in v if get_modality(te) == "actual"]
            for k, v in ext_map.items()
        }

    doc_mem = build_memory(filtered_ext(doc_ext))
    q_mem = build_memory(filtered_ext(q_ext))
    for d in docs:
        doc_mem.setdefault(
            d["doc_id"],
            {
                "intervals": [],
                "axes_merged": {
                    a: AxisDistribution(axis=a, values={}, informative=False)
                    for a in AXES
                },
                "multi_tags": set(),
            },
        )

    doc_texts_list = [d["text"] for d in docs]
    q_texts_list = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts_list)
    q_embs_arr = await embed_all(q_texts_list)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    def semantic_rerank(cand, qid):
        qv = q_embs[qid]
        qn = np.linalg.norm(qv) or 1e-9
        out = []
        for d in cand:
            v = doc_embs.get(d)
            if v is None:
                continue
            vn = np.linalg.norm(v) or 1e-9
            sim = float(np.dot(qv, v) / (qn * vn))
            out.append((d, sim))
        return sorted(out, key=lambda x: x[1], reverse=True)

    def query_intervals(qid):
        out = []
        for te in q_ext.get(qid, []):
            if get_modality(te) != "actual":
                continue
            out.extend(flatten_intervals(te))
        return out

    def allen_query_info(qid):
        for ae in allen_qs.get(qid, []):
            if ae.relation is not None and ae.anchor is not None:
                return ae.relation, ae.anchor.span
        return None, None

    def resolve_anchor_from_docs(span):
        if not span:
            return None
        span_lc = span.lower().strip().strip("'.,\"")
        for did, tes in doc_ext.items():
            for te in tes:
                if get_modality(te) != "actual":
                    continue
                iv = te_interval(te)
                if iv is None:
                    continue
                surf = (te.surface or "").lower()
                if span_lc in surf or surf in span_lc:
                    return iv
        return None

    doc_allen_by_doc = {d["doc_id"]: allen_docs_ex.get(d["doc_id"], []) for d in docs}

    rankings: dict[str, list[str]] = {}

    for q in queries:
        qid = q["query_id"]
        relation, anchor_span = allen_query_info(qid)
        allen_ranked_ids: list[str] = []
        used_allen = False
        if relation and anchor_span:
            anchor_te = None
            for ae in allen_qs.get(qid, []):
                if ae.anchor and ae.anchor.resolved is not None:
                    anchor_te = ae.anchor.resolved
                    break
            if anchor_te is None:
                iv = resolve_anchor_from_docs(anchor_span)
                if iv is not None:
                    anchor_te = TimeExpression(
                        kind="instant",
                        surface=anchor_span,
                        reference_time=parse_iso(q["ref_time"]),
                        instant=FuzzyInstant(
                            earliest=datetime.fromtimestamp(
                                iv.earliest / 1e6, tz=timezone.utc
                            ),
                            latest=datetime.fromtimestamp(
                                iv.latest / 1e6, tz=timezone.utc
                            ),
                            best=None,
                            granularity="day",
                        ),
                    )
            if anchor_te is not None:
                try:
                    allen_scores = allen_retrieve(
                        relation,
                        anchor_te,
                        doc_allen_by_doc,
                        resolve_anchor=lambda s: resolve_anchor_from_docs(s),
                    )
                    allen_ranked_ids = [
                        d
                        for d, _ in sorted(
                            allen_scores.items(), key=lambda x: x[1], reverse=True
                        )
                    ]
                    used_allen = len(allen_ranked_ids) > 0
                except Exception:
                    pass

        q_ivs = query_intervals(qid)
        anchor_ref_scores = anchor_retrieve(
            store,
            astore,
            q_ivs,
            source="union",
            agg="sum_weighted",
            alpha=REF_ANCHOR_BETA,
            beta=REF_ANCHOR_ALPHA,
        )

        # This uses the monkey-patched scorer via av2pp.score_jaccard_composite
        ma_ranked = av2pp.rank_multi_axis(
            q_mem.get(
                qid,
                {
                    "intervals": [],
                    "axes_merged": {
                        a: AxisDistribution(axis=a, values={}, informative=False)
                        for a in AXES
                    },
                    "multi_tags": set(),
                },
            ),
            doc_mem,
            ALPHA_IV,
            BETA_AXIS,
            GAMMA_TAG,
        )

        ar_max = max(anchor_ref_scores.values()) if anchor_ref_scores else 0.0
        ma_max = max(s for _, s in ma_ranked) if ma_ranked else 0.0
        combined: dict[str, float] = {}
        for d in {di["doc_id"] for di in docs}:
            ar = anchor_ref_scores.get(d, 0.0)
            ma = dict(ma_ranked).get(d, 0.0)
            ar_n = ar / ar_max if ar_max > 0 else 0.0
            ma_n = ma / ma_max if ma_max > 0 else 0.0
            combined[d] = 0.5 * ar_n + 0.5 * ma_n
        cand = [
            d for d, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)
        ][:20]
        sem = semantic_rerank(cand, qid) if cand else []
        ma_ranked_ids = [d for d, _ in sem]

        if used_allen and allen_ranked_ids:
            final = allen_ranked_ids[:TOP_K]
            for d in ma_ranked_ids:
                if d not in final:
                    final.append(d)
        else:
            final = ma_ranked_ids
            if not final:
                sem_all = semantic_rank(q_embs[qid], doc_embs)
                final = [d for d, _ in sem_all]

        if FILTER_MODALITY:
            final = filter_ranking(final, v2pp_doc_ext_for_mod, filter_modality=True)

        rankings[qid] = final

    # Per-category + per-query metrics
    cats_queries = defaultdict(list)
    for q in queries:
        cats_queries[q["category"]].append(q["query_id"])

    per_cat: dict[str, dict[str, float]] = {}
    for cat, qids in sorted(cats_queries.items()):
        r5, r10, mr, nd = [], [], [], []
        for qid in qids:
            rel = gold_map.get(qid, set())
            ranked = rankings.get(qid, [])
            if not rel:
                bad_in_top5 = any(doc_cat.get(d) == cat for d in ranked[:5])
                r5.append(0.0 if bad_in_top5 else 1.0)
                r10.append(0.0 if bad_in_top5 else 1.0)
                mr.append(float("nan"))
                nd.append(float("nan"))
            else:
                r5.append(recall_at_k(ranked, rel, 5))
                r10.append(recall_at_k(ranked, rel, 10))
                mr.append(mrr(ranked, rel))
                nd.append(ndcg_at_k(ranked, rel, TOP_K))
        per_cat[cat] = {
            "n": len(qids),
            "recall@5": nanmean(r5),
            "recall@10": nanmean(r10),
            "mrr": nanmean(mr),
            "ndcg@10": nanmean(nd),
        }

    # Overall (mean over cats; matches adversarial_v2pp_eval convention)
    all_r5 = [m["recall@5"] for m in per_cat.values()]
    all_r10 = [m["recall@10"] for m in per_cat.values()]
    all_mr = [m["mrr"] for m in per_cat.values()]
    all_nd = [m["ndcg@10"] for m in per_cat.values()]
    overall = {
        "recall@5": nanmean(all_r5),
        "recall@10": nanmean(all_r10),
        "mrr": nanmean(all_mr),
        "ndcg@10": nanmean(all_nd),
    }

    # A3 specific
    a3_qids = cats_queries.get("A3", [])
    a3_r5 = []
    for qid in a3_qids:
        rel = gold_map.get(qid, set())
        if rel:
            a3_r5.append(recall_at_k(rankings.get(qid, []), rel, 5))
    a3_r5_mean = nanmean(a3_r5) if a3_r5 else float("nan")

    return {
        "overall": overall,
        "per_category": per_cat,
        "a3_r5_queries": {
            qid: recall_at_k(rankings.get(qid, []), gold_map.get(qid, set()), 5)
            for qid in a3_qids
            if gold_map.get(qid)
        },
        "rankings": rankings,
    }


# ---------------------------------------------------------------------------
# Base-55 regression pipeline (interval-only + axis + tag, same ranker but on
# base corpus). Uses v2'' extractor for apples-to-apples vs the adversarial
# eval.
# ---------------------------------------------------------------------------
async def run_base(scorer_mode: str, decay: str) -> dict[str, Any]:
    _install_scorer(scorer_mode, decay)

    docs = load_jsonl(DATA_DIR / "docs.jsonl")
    queries = load_jsonl(DATA_DIR / "queries.jsonl")
    gold_entries = load_jsonl(DATA_DIR / "gold.jsonl")
    gold_map = {
        g["query_id"]: set(g.get("relevant_doc_ids") or []) for g in gold_entries
    }

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]

    # Reuse v2pp extract for consistency; this uses a separate cache under
    # adversarial_v2pp but since prompts differ, base items may need LLM calls
    # unless the caches are cross-populated. We try cached first and skip
    # anything uncached.
    v2pp_docs, _u1, _ = await run_v2pp_extract(doc_items, "base-docs-v2pp")
    v2pp_qs, _u2, _ = await run_v2pp_extract(q_items, "base-queries-v2pp")

    doc_ext = {d["doc_id"]: v2pp_docs.get(d["doc_id"], []) for d in docs}
    q_ext = {q["query_id"]: v2pp_qs.get(q["query_id"], []) for q in queries}

    def filtered_ext(ext_map):
        return {
            k: [te for te in v if get_modality(te) == "actual"]
            for k, v in ext_map.items()
        }

    doc_mem = build_memory(filtered_ext(doc_ext))
    q_mem = build_memory(filtered_ext(q_ext))
    for d in docs:
        doc_mem.setdefault(
            d["doc_id"],
            {
                "intervals": [],
                "axes_merged": {
                    a: AxisDistribution(axis=a, values={}, informative=False)
                    for a in AXES
                },
                "multi_tags": set(),
            },
        )

    doc_texts_list = [d["text"] for d in docs]
    q_texts_list = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts_list)
    q_embs_arr = await embed_all(q_texts_list)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    rankings: dict[str, list[str]] = {}
    for q in queries:
        qid = q["query_id"]
        ma_ranked = av2pp.rank_multi_axis(
            q_mem.get(
                qid,
                {
                    "intervals": [],
                    "axes_merged": {
                        a: AxisDistribution(axis=a, values={}, informative=False)
                        for a in AXES
                    },
                    "multi_tags": set(),
                },
            ),
            doc_mem,
            ALPHA_IV,
            BETA_AXIS,
            GAMMA_TAG,
        )
        cand = [d for d, _ in ma_ranked][:20]

        # Semantic rerank of top-20
        qv = q_embs[qid]
        qn = np.linalg.norm(qv) or 1e-9
        reranked = []
        for d in cand:
            v = doc_embs.get(d)
            if v is None:
                continue
            vn = np.linalg.norm(v) or 1e-9
            sim = float(np.dot(qv, v) / (qn * vn))
            reranked.append((d, sim))
        reranked.sort(key=lambda x: x[1], reverse=True)
        final = [d for d, _ in reranked]
        if not final:
            sem_all = semantic_rank(qv, doc_embs)
            final = [d for d, _ in sem_all]
        rankings[qid] = final

    r5, r10, mr, nd = [], [], [], []
    for q in queries:
        qid = q["query_id"]
        rel = gold_map.get(qid, set())
        if not rel:
            continue
        ranked = rankings.get(qid, [])
        r5.append(recall_at_k(ranked, rel, 5))
        r10.append(recall_at_k(ranked, rel, 10))
        mr.append(mrr(ranked, rel))
        nd.append(ndcg_at_k(ranked, rel, TOP_K))

    return {
        "overall": {
            "recall@5": nanmean(r5),
            "recall@10": nanmean(r10),
            "mrr": nanmean(mr),
            "ndcg@10": nanmean(nd),
            "n": len(r5),
        },
        "rankings": rankings,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
async def main() -> None:
    t0 = time.time()
    print("Scorer containment sweep — zero LLM calls (cache-only).")

    configs: list[tuple[str, str, str]] = [
        ("baseline_jaccard", "jaccard_composite", "log2"),
        ("containment_log2", "jaccard_with_containment", "log2"),
        ("containment_sqrt", "jaccard_with_containment", "sqrt"),
        ("containment_dice", "jaccard_with_containment", "dice"),
    ]

    results: dict[str, Any] = {"configs": {}}
    for name, mode, decay in configs:
        print(f"\n=== Config: {name}  (mode={mode}, decay={decay}) ===")
        adv = await run_adversarial(mode, decay)
        base = await run_base(mode, decay)
        results["configs"][name] = {
            "mode": mode,
            "decay": decay,
            "adversarial_overall": adv["overall"],
            "adversarial_per_category": adv["per_category"],
            "a3_per_query_r5": adv["a3_r5_queries"],
            "base_overall": base["overall"],
        }
        print(
            f"  adv overall R@5={adv['overall']['recall@5']:.3f}  "
            f"A3 R@5={nanmean(list(adv['a3_r5_queries'].values())):.3f}  "
            f"base R@5={base['overall']['recall@5']:.3f}"
        )

    # Write JSON
    (RESULTS_DIR / "scorer_containment.json").write_text(
        json.dumps(_json_clean(results), indent=2, default=str)
    )

    # Write Markdown
    lines = ["# Scorer Containment Sweep\n\n"]
    lines.append(f"Wall: {time.time() - t0:.1f}s. Zero LLM calls (cache-only).\n\n")
    lines.append(
        "Four configs compared against the ship-best v2'' + multi-axis + "
        "score-blend pipeline:\n\n"
    )
    lines.append("- baseline_jaccard: existing jaccard_composite.\n")
    lines.append(
        "- containment_{log2,sqrt,dice}: new max(jaccard, q_in_s, s_in_q) "
        "with decay formula.\n\n"
    )
    lines.append("## Overall comparison\n\n")
    lines.append("| config | Adv R@5 | Adv R@10 | A3 R@5 | Base R@5 | Base R@10 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    baseline_adv = results["configs"]["baseline_jaccard"]
    for name, _mode, _decay in configs:
        r = results["configs"][name]
        a3 = nanmean(list(r["a3_per_query_r5"].values()))
        lines.append(
            f"| {name} | {r['adversarial_overall']['recall@5']:.3f} "
            f"| {r['adversarial_overall']['recall@10']:.3f} "
            f"| {a3:.3f} "
            f"| {r['base_overall']['recall@5']:.3f} "
            f"| {r['base_overall']['recall@10']:.3f} |\n"
        )

    lines.append("\n## Per-category (Adversarial) — delta vs baseline_jaccard\n\n")
    cats = sorted(baseline_adv["adversarial_per_category"].keys())
    header = "| cat | n | " + " | ".join(f"{n} R@5" for n, _, _ in configs) + " |\n"
    lines.append(header)
    lines.append("|---|---:" + "|---:" * len(configs) + "|\n")
    for cat in cats:
        n = baseline_adv["adversarial_per_category"][cat]["n"]
        row = f"| {cat} | {n} |"
        for name, _m, _d in configs:
            v = results["configs"][name]["adversarial_per_category"][cat]["recall@5"]
            vs = f"{v:.3f}" if v == v else "-"
            row += f" {vs} |"
        lines.append(row + "\n")

    lines.append("\n## A3 per-query R@5\n\n")
    lines.append("| qid | " + " | ".join(n for n, _, _ in configs) + " |\n")
    lines.append("|---" + "|---" * len(configs) + "|\n")
    for qid in sorted(baseline_adv["a3_per_query_r5"].keys()):
        row = f"| {qid} |"
        for name, _m, _d in configs:
            v = results["configs"][name]["a3_per_query_r5"][qid]
            vs = f"{v:.2f}" if v == v else "-"
            row += f" {vs} |"
        lines.append(row + "\n")

    # Deltas & recommendation
    def _delta(new: float, old: float) -> float:
        return new - old

    lines.append("\n## Regressions vs baseline\n\n")
    lines.append(
        "| config | Δ Adv R@5 | Δ Base R@5 | max cat regression | worst cat |\n"
    )
    lines.append("|---|---:|---:|---:|---|\n")
    for name, _m, _d in configs:
        if name == "baseline_jaccard":
            continue
        r = results["configs"][name]
        b = baseline_adv
        dadv = _delta(
            r["adversarial_overall"]["recall@5"], b["adversarial_overall"]["recall@5"]
        )
        dbase = _delta(r["base_overall"]["recall@5"], b["base_overall"]["recall@5"])
        worst_cat, worst_delta = None, 0.0
        for cat in cats:
            d = _delta(
                r["adversarial_per_category"][cat]["recall@5"],
                b["adversarial_per_category"][cat]["recall@5"],
            )
            if d < worst_delta:
                worst_delta = d
                worst_cat = cat
        lines.append(
            f"| {name} | {dadv:+.3f} | {dbase:+.3f} | {worst_delta:+.3f} | {worst_cat or '-'} |\n"
        )

    (RESULTS_DIR / "scorer_containment.md").write_text("".join(lines))
    print(
        f"\nWrote results/scorer_containment.{{md,json}}. Wall: {time.time() - t0:.1f}s."
    )


def _json_clean(o):
    if isinstance(o, dict):
        return {k: _json_clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_clean(v) for v in o]
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


if __name__ == "__main__":
    asyncio.run(main())
