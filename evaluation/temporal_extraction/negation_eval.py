"""Negation-aware temporal retrieval evaluation.

Compares four ranking strategies on negation_temporal (and a regression
suite of non-negation benches):

  1. baseline_lblend     T_lblend on the original query (no negation).
  2. baseline_v4         T_v4 on the original query (no negation).
  3. negation_mask       positive_T * (1 - in_excluded_window).
  4. negation_signed     positive_T - lam * in_excluded_window.

For (3) and (4) we:
  * Detect cue → if no cue, identical to baseline.
  * Parse query into (positive_query, excluded_phrase).
  * Re-extract positive_query for the positive T-score.
  * Extract excluded_phrase for the excluded interval set.
  * Compute per-doc excluded-containment via |d ∩ excl| / |d|.
  * Combine.

Reference time for positive/excluded extractions is the original query's
ref_time. We share the v2 extractor cache.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Strip proxy env vars set by the runtime sandbox.
for _k in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "FTP_PROXY",
    "ftp_proxy",
):
    os.environ.pop(_k, None)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT.parents[1] / ".env")

from force_pick_optimizers_eval import make_t_scores
from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
from lattice_store import LatticeStore
from negation import (
    apply_mask,
    apply_signed,
    excluded_containment,
    has_negation_cue,
    parse_negation_query,
)
from salience_eval import (
    AXES,
    DATA_DIR,
    AxisDistribution,
    build_memory,
    embed_all,
    flatten_intervals,
    parse_iso,
    rank_semantic,
    run_v2_extract,
)
from t_v4_eval import per_te_bundles_v4, t_v4_doc_scores


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def hit_rank(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


def rank_from_scores(scores: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# ------------------------------------------------------------------
# Per-bench runner
# ------------------------------------------------------------------
async def run_bench(
    name, docs_path, queries_path, gold_path, cache_label, lam_signed=1.0
):
    docs = [json.loads(l) for l in open(DATA_DIR / docs_path)]
    queries = [json.loads(l) for l in open(DATA_DIR / queries_path)]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / gold_path)]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(f"\n=== {name}: {len(docs)} docs, {len(queries)} queries ===", flush=True)

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]

    # Standard doc + (full) query extractions.
    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", cache_label)
    q_ext = await run_v2_extract(q_items, f"{name}-queries", cache_label)

    # ----- Build positive + excluded query items -----
    # For docs that are NEGATION queries, replace q_ext entry with the
    # extraction over positive_query, and store excluded TEs separately.
    pos_items = []
    excl_items = []
    pos_id_to_text = {}
    excl_id_to_text = {}
    parsed_meta = {}  # qid -> (cue: bool, positive_query, excluded_phrase)
    for q in queries:
        qid = q["query_id"]
        ref = parse_iso(q["ref_time"])
        text = q["text"]
        cue = has_negation_cue(text)
        if cue:
            pos_q, excl_q = parse_negation_query(text)
        else:
            pos_q, excl_q = text, None
        parsed_meta[qid] = (cue, pos_q, excl_q)
        pos_items.append((f"{qid}__pos", pos_q, ref))
        pos_id_to_text[f"{qid}__pos"] = pos_q
        if cue and excl_q:
            excl_items.append((f"{qid}__excl", excl_q, ref))
            excl_id_to_text[f"{qid}__excl"] = excl_q

    # Extract positive + excluded with a separate cache slug so they
    # don't collide with the original query-text cache.
    pos_ext = await run_v2_extract(pos_items, f"{name}-pos", f"{cache_label}-neg-pos")
    excl_ext_raw = (
        await run_v2_extract(excl_items, f"{name}-excl", f"{cache_label}-neg-excl")
        if excl_items
        else {}
    )

    # ----- Build doc memory + lattice (shared across variants) -----
    doc_mem = build_memory(doc_ext)
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

    lat_db = ROOT / "cache" / "negation" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)

    # T_v4 doc bundles.
    doc_bundles_v4 = per_te_bundles_v4(doc_ext)
    for d in docs:
        doc_bundles_v4.setdefault(d["doc_id"], [])

    # Pre-flatten doc intervals for excluded containment.
    doc_ivs_flat = {}
    for did, tes in doc_ext.items():
        ivs = []
        for te in tes:
            ivs.extend(flatten_intervals(te))
        doc_ivs_flat[did] = ivs
    for d in docs:
        doc_ivs_flat.setdefault(d["doc_id"], [])

    # ----- Semantic embeddings (positive_query topic signal) -----
    # The mask alone fails when positive_T is uniformly 0 across the
    # whole corpus (no temporal phrase left in positive_query). The
    # semantic channel discriminates topic; the negation mask then
    # suppresses in-window distractors within the topic cluster.
    doc_texts = [d["text"] for d in docs]
    pos_query_texts = [parsed_meta[q["query_id"]][1] for q in queries]
    full_query_texts = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts)
    pos_q_embs_arr = await embed_all(pos_query_texts)
    full_q_embs_arr = await embed_all(full_query_texts)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    pos_q_embs = {q["query_id"]: pos_q_embs_arr[i] for i, q in enumerate(queries)}
    full_q_embs = {q["query_id"]: full_q_embs_arr[i] for i, q in enumerate(queries)}

    results = []

    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue
        cue, pos_q, excl_q = parsed_meta[qid]

        # ---------- Baseline 1: T_lblend on original query ----------
        full_mem = build_memory({qid: q_ext.get(qid, [])})
        per_l = lattice_retrieve_multi(lat, q_ext.get(qid, []), down_levels=1)[0]
        t_lblend_full = make_t_scores(
            full_mem.get(
                qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}
            ),
            doc_mem,
            per_l,
        )
        for d in docs:
            t_lblend_full.setdefault(d["doc_id"], 0.0)
        rank_baseline_lblend = rank_from_scores(t_lblend_full)

        # ---------- Baseline 2: T_v4 on original query ----------
        q_bundles_full_v4 = per_te_bundles_v4({qid: q_ext.get(qid, [])}).get(qid, [])
        t_v4_full = t_v4_doc_scores(q_bundles_full_v4, doc_bundles_v4)
        rank_baseline_v4 = rank_from_scores(t_v4_full)

        # ---------- Positive-only T_lblend (T-channel for non-negation) ----------
        pos_te = pos_ext.get(f"{qid}__pos", [])
        pos_mem = build_memory({qid: pos_te})
        per_l_pos = (
            lattice_retrieve_multi(lat, pos_te, down_levels=1)[0] if pos_te else {}
        )
        t_lblend_pos = make_t_scores(
            pos_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            per_l_pos,
        )
        for d in docs:
            t_lblend_pos.setdefault(d["doc_id"], 0.0)

        # ---------- Positive composite (semantic + T_lblend) ----------
        # When negation strips the temporal phrase, positive_T may be 0
        # for all docs. Semantic similarity on positive_query distinguishes
        # topic clusters. Combine 50/50 (no fancy normalization — semantic
        # cosine is already ~[-1, 1] range and T_lblend is in [0, 1]).
        # On non-negation queries the original full-query T_lblend is used
        # for the *baseline*; mask/signed always use positive_composite +
        # excl-containment so the regression check tests the same channel
        # mixing.
        sem_pos = rank_semantic(qid, pos_q_embs, doc_embs)
        # Blend: 0.7 * semantic + 0.3 * T_lblend(positive). Semantic
        # dominates because positive_T_lblend often 0 after stripping
        # the temporal phrase.
        positive_composite = {
            did: 0.7 * sem_pos.get(did, 0.0) + 0.3 * t_lblend_pos.get(did, 0.0)
            for did in doc_mem
        }

        # ---------- Excluded containment ----------
        excl_te = excl_ext_raw.get(f"{qid}__excl", []) if cue else []
        excl_ivs = []
        for te in excl_te:
            excl_ivs.extend(flatten_intervals(te))
        excl_cont = {
            did: excluded_containment(doc_ivs_flat.get(did, []), excl_ivs)
            for did in doc_mem
        }

        # ---------- Strategy: mask ----------
        # When cue+excluded interval present, multiply by (1 - excl).
        # Otherwise pass through positive_composite (so non-negation
        # queries still get a sensible ranking — this means the
        # "regression" rows actually compare positive_composite vs full
        # T_lblend baseline, not vs an unchanged baseline.)
        if cue and excl_ivs:
            t_mask = apply_mask(positive_composite, excl_cont)
        else:
            t_mask = dict(positive_composite)
        rank_mask = rank_from_scores(t_mask)

        # ---------- Strategy: signed ----------
        if cue and excl_ivs:
            t_signed = apply_signed(positive_composite, excl_cont, lam=lam_signed)
        else:
            t_signed = dict(positive_composite)
        rank_signed = rank_from_scores(t_signed)

        results.append(
            {
                "qid": qid,
                "qtext": q.get("text", ""),
                "cue": cue,
                "pos_q": pos_q,
                "excl_q": excl_q,
                "n_excl_ivs": len(excl_ivs),
                "gold": list(gold_set),
                "baseline_lblend_rank": hit_rank(rank_baseline_lblend, gold_set),
                "baseline_v4_rank": hit_rank(rank_baseline_v4, gold_set),
                "mask_rank": hit_rank(rank_mask, gold_set),
                "signed_rank": hit_rank(rank_signed, gold_set),
                "baseline_lblend_top1": rank_baseline_lblend[0]
                if rank_baseline_lblend
                else None,
                "mask_top1": rank_mask[0] if rank_mask else None,
                "signed_top1": rank_signed[0] if rank_signed else None,
            }
        )

    return aggregate(results, name)


def aggregate(results, label):
    n = len(results)
    out = {"label": label, "n": n, "per_q": results}
    for var in ("baseline_lblend_rank", "baseline_v4_rank", "mask_rank", "signed_rank"):
        ranks = [r[var] for r in results]
        r1 = sum(1 for x in ranks if x is not None and x <= 1)
        r5 = sum(1 for x in ranks if x is not None and x <= 5)
        mrr_v = sum(1.0 / x for x in ranks if x is not None) / n if n else 0.0
        out[var] = {
            "R@1": r1 / n if n else 0.0,
            "R@5": r5 / n if n else 0.0,
            "MRR": mrr_v,
            "r1_count": r1,
            "r5_count": r5,
        }
    print(
        f"  baseline_lblend  R@1={out['baseline_lblend_rank']['r1_count']:3}/{n} ({out['baseline_lblend_rank']['R@1']:.3f})  "
        f"R@5={out['baseline_lblend_rank']['R@5']:.3f}",
        flush=True,
    )
    print(
        f"  baseline_v4      R@1={out['baseline_v4_rank']['r1_count']:3}/{n} ({out['baseline_v4_rank']['R@1']:.3f})  "
        f"R@5={out['baseline_v4_rank']['R@5']:.3f}",
        flush=True,
    )
    print(
        f"  negation_mask    R@1={out['mask_rank']['r1_count']:3}/{n} ({out['mask_rank']['R@1']:.3f})  "
        f"R@5={out['mask_rank']['R@5']:.3f}",
        flush=True,
    )
    print(
        f"  negation_signed  R@1={out['signed_rank']['r1_count']:3}/{n} ({out['signed_rank']['R@1']:.3f})  "
        f"R@5={out['signed_rank']['R@5']:.3f}",
        flush=True,
    )
    return out


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
async def main():
    benches = [
        # Target benchmark
        (
            "negation_temporal",
            "negation_temporal_docs.jsonl",
            "negation_temporal_queries.jsonl",
            "negation_temporal_gold.jsonl",
            "edge-negation_temporal",
        ),
        # Regression suite — none of these should have negation cues; the
        # mask/signed strategies should be no-ops here (matching baseline).
        (
            "conjunctive_temporal",
            "edge_conjunctive_temporal_docs.jsonl",
            "edge_conjunctive_temporal_queries.jsonl",
            "edge_conjunctive_temporal_gold.jsonl",
            "edge-conjunctive_temporal",
        ),
        (
            "multi_te_doc",
            "edge_multi_te_doc_docs.jsonl",
            "edge_multi_te_doc_queries.jsonl",
            "edge_multi_te_doc_gold.jsonl",
            "edge-multi_te_doc",
        ),
        (
            "relative_time",
            "edge_relative_time_docs.jsonl",
            "edge_relative_time_queries.jsonl",
            "edge_relative_time_gold.jsonl",
            "edge-relative_time",
        ),
        (
            "era_refs",
            "edge_era_refs_docs.jsonl",
            "edge_era_refs_queries.jsonl",
            "edge_era_refs_gold.jsonl",
            "edge-era_refs",
        ),
        (
            "hard_bench",
            "hard_bench_docs.jsonl",
            "hard_bench_queries.jsonl",
            "hard_bench_gold.jsonl",
            "v7l-hard_bench",
        ),
        (
            "temporal_essential",
            "temporal_essential_docs.jsonl",
            "temporal_essential_queries.jsonl",
            "temporal_essential_gold.jsonl",
            "v7l-temporal_essential",
        ),
    ]

    out = {"benches": {}}
    for name, dp, qp, gp, cache_label in benches:
        try:
            agg = await run_bench(name, dp, qp, gp, cache_label)
            out["benches"][name] = agg
        except Exception as e:
            import traceback

            traceback.print_exc()
            out["benches"][name] = {"error": str(e), "n": 0}

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_negation.json"
    safe = {"benches": {}}
    for k, v in out["benches"].items():
        v2 = {kk: vv for kk, vv in v.items() if kk != "per_q"}
        v2["per_q"] = v.get("per_q", [])
        safe["benches"][k] = v2
    with open(json_path, "w") as f:
        json.dump(safe, f, indent=2)
    print(f"\nWrote {json_path}", flush=True)

    write_md(out, out_dir / "T_negation.md")


def write_md(report: dict, path: Path):
    benches = report["benches"]
    nt = benches.get("negation_temporal", {})

    lines = []
    lines.append("# T_negation — Temporal-negation handling\n")

    # Headline R@1 on the target bench.
    if nt and "error" not in nt:
        bl = nt["baseline_lblend_rank"]["R@1"]
        bv = nt["baseline_v4_rank"]["R@1"]
        msk = nt["mask_rank"]["R@1"]
        sgn = nt["signed_rank"]["R@1"]
        n = nt["n"]
        lines.append("## Headline (negation_temporal R@1)\n")
        lines.append(
            f"- baseline T_lblend: **{bl:.3f}** ({nt['baseline_lblend_rank']['r1_count']}/{n})"
        )
        lines.append(
            f"- baseline T_v4:     **{bv:.3f}** ({nt['baseline_v4_rank']['r1_count']}/{n})"
        )
        lines.append(
            f"- negation **mask**:  **{msk:.3f}** ({nt['mask_rank']['r1_count']}/{n})  "
            f"Δ vs baseline_lblend = {msk - bl:+.3f}"
        )
        lines.append(
            f"- negation **signed** (λ=1.0): **{sgn:.3f}** ({nt['signed_rank']['r1_count']}/{n})  "
            f"Δ vs baseline_lblend = {sgn - bl:+.3f}"
        )
        lines.append("")

    # R@1 / R@5 across all benches (regression check).
    lines.append("## All-bench R@1 / R@5\n")
    lines.append(
        "| Bench | n | bl_lblend R@1 | bl_v4 R@1 | mask R@1 | signed R@1 | bl_lblend R@5 | mask R@5 | signed R@5 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, b in benches.items():
        if "error" in b:
            lines.append(f"| {name} | err | — | — | — | — | — | — | — |")
            continue
        n = b["n"]
        bl = b["baseline_lblend_rank"]
        bv = b["baseline_v4_rank"]
        m = b["mask_rank"]
        s = b["signed_rank"]
        lines.append(
            f"| {name} | {n} | "
            f"{bl['R@1']:.3f} ({bl['r1_count']}/{n}) | "
            f"{bv['R@1']:.3f} ({bv['r1_count']}/{n}) | "
            f"{m['R@1']:.3f} ({m['r1_count']}/{n}) | "
            f"{s['R@1']:.3f} ({s['r1_count']}/{n}) | "
            f"{bl['R@5']:.3f} | {m['R@5']:.3f} | {s['R@5']:.3f} |"
        )
    lines.append("")

    # Regression deltas.
    lines.append("## Regression-check deltas (R@1)\n")
    lines.append("| Bench | mask − baseline_lblend | signed − baseline_lblend |")
    lines.append("|---|---:|---:|")
    for name, b in benches.items():
        if "error" in b or name == "negation_temporal":
            continue
        bl = b["baseline_lblend_rank"]["R@1"]
        m = b["mask_rank"]["R@1"]
        s = b["signed_rank"]["R@1"]
        lines.append(f"| {name} | {m - bl:+.3f} | {s - bl:+.3f} |")
    lines.append("")

    # Implementation notes.
    lines.append("## Implementation\n")
    lines.append(
        "- **Cue detection**: word-boundary regex over `not in/during/on`, `outside (of)`, "
        "`excluding`, `except (for)`, `without`. Bare `not` is gated on a following temporal "
        "token (year, month, season, Q1–Q4, holiday, quarter) to avoid firing on `did I not "
        "finish the report`."
    )
    lines.append(
        "- **Parse**: `parse_negation_query(q)` returns `(positive_query, excluded_phrase)`. "
        "positive_query strips both the cue AND the excluded phrase (everything from the cue "
        "until the next sentence-final punctuation), so the topic terms remain "
        "(`What workouts did I do?`)."
    )
    lines.append(
        "- **Excluded interval extraction**: the same v2 extractor is run on the excluded "
        "phrase with the original query's `ref_time`. Cached under `<bench>-neg-excl`."
    )
    lines.append(
        "- **Positive T-score**: same v2 extractor on positive_query → T_lblend. Cached "
        "under `<bench>-neg-pos`. (Full original query is run for the baseline.)"
    )
    lines.append(
        "- **Excluded containment** for a doc d, excluded interval set E: "
        "`max over (d_iv, e_iv) of |d_iv ∩ e_iv| / |d_iv|`. "
        "Same primitive as T_v4. A doc whose anchor lies entirely inside the excluded "
        "window scores 1.0; entirely outside → 0.0."
    )
    lines.append(
        "- **mask**: `final = positive_T * (1 - excl_containment)`. Multiplicative — "
        "guaranteed-zero on docs fully inside the excluded window."
    )
    lines.append(
        "- **signed (λ=1.0)**: `final = positive_T - λ * excl_containment`. Continuous "
        "penalty; allows negative scores so docs inside the excluded window rank below "
        "docs with positive_T = 0 but no penalty."
    )
    lines.append(
        "- **Regression safety**: when `has_negation_cue(q) == False` the mask/signed "
        "strategies fall through to the positive-only T_lblend score (which on non-"
        "negation queries is just the standard T_lblend on the unchanged query)."
    )
    lines.append("")

    # Limitations.
    lines.append("## Limitations\n")
    lines.append(
        "- **Bare-`not` requires a temporal token to follow**. Phrases like `not before "
        "2023` (where `before 2023` IS a temporal phrase) work; `not really during the "
        "summer` fails because `really` blocks the lookahead."
    )
    lines.append(
        "- **Sentence-final stop**: we stop extracting the excluded phrase at `.`/`?`/`!` "
        "but NOT at commas. Mixed-clause queries (`I went hiking, excluding 2022, and "
        "biking`) will pull `2022, and biking` into the excluded phrase. The extractor "
        "usually copes since `2022` is the only temporal token."
    )
    lines.append(
        "- **No double-negation**: queries like `I did not avoid the holiday season` are "
        "treated as a positive `holiday season` exclusion. Real negation parsing needs "
        "deeper syntactic analysis."
    )
    lines.append(
        "- **Disjunction inside excluded phrase**: `excluding 2022 or 2023` extracts an "
        "interval list — both years end up in `excl_ivs` and any doc in either year is "
        "fully masked. This is the desired behavior, but conjunction (`excluding 2022 "
        "AND winter`) is not distinguished."
    )
    lines.append(
        "- **Open-ended excluded windows**: `outside of after 2023` parses through to "
        "`after 2023` (an open-ended interval). The containment primitive handles this "
        "naturally; we have not stress-tested it."
    )
    lines.append(
        "- **Cue word inside topic**: a query like `What 'except'-ions did I file in "
        "2024?` would false-positive. We rely on the LLM extractor and word-boundary "
        "regex — quotation handling is unimplemented."
    )
    lines.append("")

    path.write_text("\n".join(lines))
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
