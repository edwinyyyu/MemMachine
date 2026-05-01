"""T_recency evaluation: recency-decay channel for "latest/most recent X" queries.

Compares:
  - rerank_only:                   cross-encoder over union(top-50 sem, top-50 T_v4)
  - T_lblend:                      pre-existing T baseline
  - T_v4:                          single-primitive containment T
  - T_lblend + recency (replace):  recency replaces T_lblend when cue & T_lblend dead
  - T_lblend + recency (additive): (1-α)*T_lblend + α*cue*recency
  - T_v4 + recency (replace):      same logic on T_v4
  - T_v4 + recency (additive):     (1-α)*T_v4 + α*cue*recency
  - rerank + recency (replace):    cross-encoder picks topic candidates → recency within set
  - rerank + recency (mult):       cue * exp(-λΔ) * rerank_score (gated multiplicative)
  - recency_only:                  pure recency-decay ranking (sanity / topic-blind)

Half-life sweep: 7d, 21d, 90d.

The recency cue gate (regex on query text) ensures recency is
*not* applied to queries lacking cue words — so non-recency benches
should remain unchanged.

Key design point: pure recency-decay over the full corpus is topic-
blind. The cross-encoder rerank score carries the topic signal, so
we combine recency with the rerank score (`rerank * recency` for cued
queries) so the channel only picks among topic-similar docs.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Strip proxy env vars set by sandbox.
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

from force_pick_optimizers_eval import (
    RERANK_TOP_K,
    make_t_scores,
    merge_with_tail,
    rerank_topk,
    topk_from_scores,
)
from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
from lattice_store import LatticeStore
from recency import (
    combine_additive,
    combine_replacement,
    has_recency_cue,
    lambda_for_half_life,
    recency_scores_for_docs,
)
from salience_eval import (
    AXES,
    DATA_DIR,
    AxisDistribution,
    build_memory,
    embed_all,
    parse_iso,
    rank_semantic,
    run_v2_extract,
)
from schema import to_us
from t_v4_eval import per_te_bundles_v4, t_v4_doc_scores

HALF_LIVES = [7.0, 21.0, 90.0]


# --------------------------------------------------------------------------
def hit_rank(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


def rank_from_scores(scores: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# --------------------------------------------------------------------------
async def run_bench(name, docs_path, queries_path, gold_path, cache_label, reranker):
    docs = [json.loads(l) for l in open(DATA_DIR / docs_path)]
    queries = [json.loads(l) for l in open(DATA_DIR / queries_path)]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / gold_path)]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(f"\n=== {name}: {len(docs)} docs, {len(queries)} queries ===", flush=True)
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", cache_label)
    q_ext = await run_v2_extract(q_items, f"{name}-queries", cache_label)

    doc_text = {d["doc_id"]: d["text"] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    doc_ref_us = {d["doc_id"]: to_us(parse_iso(d["ref_time"])) for d in docs}
    q_ref_us = {q["query_id"]: to_us(parse_iso(q["ref_time"])) for q in queries}

    # T_lblend memory
    doc_mem = build_memory(doc_ext)
    q_mem = build_memory(q_ext)
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

    # Lattice for T_lblend
    lat_db = ROOT / "cache" / "t_recency" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)
    qids = [q["query_id"] for q in queries]
    per_q_l = {
        qid: lattice_retrieve_multi(lat, q_ext.get(qid, []), down_levels=1)[0]
        for qid in qids
    }

    # T_v4 bundles
    doc_bundles_v4 = per_te_bundles_v4(doc_ext)
    for d in docs:
        doc_bundles_v4.setdefault(d["doc_id"], [])
    q_bundles_v4 = per_te_bundles_v4(q_ext)

    # Semantic + cross-encoder rerank for the rerank_only baseline
    print("  embedding + reranking...", flush=True)
    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    # Rerank: union(top-50 sem, top-50 T_v4) -> cross-encoder
    per_q_t_v4 = {}
    for qid in qids:
        qb4 = q_bundles_v4.get(qid, [])
        per_q_t_v4[qid] = t_v4_doc_scores(qb4, doc_bundles_v4)

    per_q_r: dict[str, dict[str, float]] = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        t_top = topk_from_scores(per_q_t_v4[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + t_top))[:RERANK_TOP_K]
        per_q_r[qid] = await rerank_topk(
            reranker, q_text[qid], union, doc_text, len(union)
        )

    # Per-half-life recency lambdas
    lambdas = {h: lambda_for_half_life(h) for h in HALF_LIVES}

    results = []
    n_cued = 0
    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue

        cue = has_recency_cue(q["text"])
        if cue:
            n_cued += 1

        # rerank_only ranking (top-50 reranked, then tail by semantic)
        rs = per_q_r[qid]
        rerank_only_rank = merge_with_tail(
            [d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)],
            per_q_s[qid],
        )

        # T_lblend
        t_lblend = make_t_scores(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            per_q_l.get(qid, {}),
        )
        for d in docs:
            t_lblend.setdefault(d["doc_id"], 0.0)
        rank_lb = rank_from_scores(t_lblend)

        # T_v4
        t_v4 = dict(per_q_t_v4[qid])
        for d in docs:
            t_v4.setdefault(d["doc_id"], 0.0)
        rank_v4 = rank_from_scores(t_v4)

        # Recency scores at each half-life
        rec_at_h: dict[float, dict[str, float]] = {}
        for h, lam in lambdas.items():
            rec_at_h[h] = recency_scores_for_docs(
                doc_bundles_v4,
                doc_ref_us,
                q_ref_us[qid],
                lam,
            )

        # Rerank scores extended to all docs (with semantic tail) so we can
        # combine with recency cleanly. Docs not in the rerank top-K get
        # a fallback equal to a small fraction of the semantic score so
        # that within-set ranking is dominated by the rerank when present.
        rerank_full = {}
        # Normalize rerank scores to [0,1] so multiplication with recency
        # produces interpretable products. Cross-encoder scores can be
        # arbitrary reals (logits); clip negatives to 0 then min-max scale.
        rs_vals = list(rs.values())
        if rs_vals:
            r_min = min(rs_vals)
            r_max = max(rs_vals)
            r_span = (r_max - r_min) or 1.0
        else:
            r_min, r_span = 0.0, 1.0
        for did in t_v4:  # all doc ids
            if did in rs:
                rerank_full[did] = (rs[did] - r_min) / r_span
            else:
                # Tail: small constant below rerank floor; preserves
                # candidate-set boundary so non-topic docs cannot win.
                rerank_full[did] = -1.0

        rec = {}
        for h in HALF_LIVES:
            rec_scores = rec_at_h[h]

            # recency_only — uses cue gate (no cue → fall back to T_v4)
            if cue:
                rec_only_scores = rec_scores
            else:
                rec_only_scores = t_v4
            rank_recOnly = rank_from_scores(rec_only_scores)

            # T_lblend + recency replacement
            lb_rec = combine_replacement(t_lblend, rec_scores, cue)
            rank_lb_rec = rank_from_scores(lb_rec)

            # T_lblend + recency additive (α=0.5)
            lb_rec_add = combine_additive(t_lblend, rec_scores, cue, alpha=0.5)
            rank_lb_rec_add = rank_from_scores(lb_rec_add)

            # T_v4 + recency replacement
            v4_rec = combine_replacement(t_v4, rec_scores, cue)
            rank_v4_rec = rank_from_scores(v4_rec)

            # T_v4 + recency additive (α=0.5)
            v4_rec_add = combine_additive(t_v4, rec_scores, cue, alpha=0.5)
            rank_v4_rec_add = rank_from_scores(v4_rec_add)

            # Rerank + recency multiplicative (gated)
            # When cue: score = (rerank>=0) ? rerank*recency : rerank (tail).
            # When no cue: pure rerank.
            rer_rec_mult = {}
            for did, rfs in rerank_full.items():
                if cue and rfs >= 0:
                    rer_rec_mult[did] = rfs * rec_scores.get(did, 0.0)
                else:
                    rer_rec_mult[did] = rfs
            rank_rer_rec_mult = rank_from_scores(rer_rec_mult)

            # Rerank + recency additive (α=0.5)
            rer_rec_add = {}
            for did, rfs in rerank_full.items():
                rfs01 = max(rfs, 0.0)
                if cue:
                    rer_rec_add[did] = 0.5 * rfs01 + 0.5 * rec_scores.get(did, 0.0)
                else:
                    rer_rec_add[did] = rfs
            rank_rer_rec_add = rank_from_scores(rer_rec_add)

            # Rerank + recency replacement: when cue, rank by recency
            # AMONG THE RERANK TOP-K (non-rerank docs get tail score 0).
            # Otherwise pure rerank ranking. This tests "candidate set =
            # rerank top, ranking = recency".
            rer_rec_replace = {}
            for did, rfs in rerank_full.items():
                if cue and rfs >= 0:
                    rer_rec_replace[did] = rec_scores.get(did, 0.0)
                else:
                    rer_rec_replace[did] = rfs
            rank_rer_rec_replace = rank_from_scores(rer_rec_replace)

            rec[h] = {
                "recency_only": hit_rank(rank_recOnly, gold_set),
                "lb_plus_rec_replace": hit_rank(rank_lb_rec, gold_set),
                "lb_plus_rec_add": hit_rank(rank_lb_rec_add, gold_set),
                "v4_plus_rec_replace": hit_rank(rank_v4_rec, gold_set),
                "v4_plus_rec_add": hit_rank(rank_v4_rec_add, gold_set),
                "rer_plus_rec_mult": hit_rank(rank_rer_rec_mult, gold_set),
                "rer_plus_rec_add": hit_rank(rank_rer_rec_add, gold_set),
                "rer_plus_rec_replace": hit_rank(rank_rer_rec_replace, gold_set),
            }

        results.append(
            {
                "qid": qid,
                "qtext": q.get("text", ""),
                "gold": list(gold_set),
                "cue": cue,
                "rerank_only_rank": hit_rank(rerank_only_rank, gold_set),
                "t_lblend_rank": hit_rank(rank_lb, gold_set),
                "t_v4_rank": hit_rank(rank_v4, gold_set),
                "by_h": rec,
            }
        )

    return aggregate(results, name, n_cued)


def aggregate(results, label, n_cued):
    n = len(results)
    out = {"label": label, "n": n, "n_cued": n_cued, "per_q": results}

    base_vars = ["rerank_only_rank", "t_lblend_rank", "t_v4_rank"]
    h_vars = [
        "recency_only",
        "lb_plus_rec_replace",
        "lb_plus_rec_add",
        "v4_plus_rec_replace",
        "v4_plus_rec_add",
        "rer_plus_rec_mult",
        "rer_plus_rec_add",
        "rer_plus_rec_replace",
    ]

    for var in base_vars:
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

    for h in HALF_LIVES:
        for var in h_vars:
            ranks = [r["by_h"][h][var] for r in results]
            r1 = sum(1 for x in ranks if x is not None and x <= 1)
            r5 = sum(1 for x in ranks if x is not None and x <= 5)
            mrr_v = sum(1.0 / x for x in ranks if x is not None) / n if n else 0.0
            out[f"{var}_h{int(h)}"] = {
                "R@1": r1 / n if n else 0.0,
                "R@5": r5 / n if n else 0.0,
                "MRR": mrr_v,
                "r1_count": r1,
                "r5_count": r5,
            }

    # Print headline
    def fmt(d):
        return f"{d['R@1']:.3f} ({d['r1_count']}/{n})"

    print(f"  cued: {n_cued}/{n}", flush=True)
    print(f"  rerank_only        R@1={fmt(out['rerank_only_rank'])}", flush=True)
    print(f"  T_lblend           R@1={fmt(out['t_lblend_rank'])}", flush=True)
    print(f"  T_v4               R@1={fmt(out['t_v4_rank'])}", flush=True)
    for h in HALF_LIVES:
        print(f"  --- half-life {int(h)}d ---", flush=True)
        print(
            f"  recency_only       R@1={fmt(out[f'recency_only_h{int(h)}'])}",
            flush=True,
        )
        print(
            f"  lb+rec replace     R@1={fmt(out[f'lb_plus_rec_replace_h{int(h)}'])}",
            flush=True,
        )
        print(
            f"  lb+rec additive    R@1={fmt(out[f'lb_plus_rec_add_h{int(h)}'])}",
            flush=True,
        )
        print(
            f"  v4+rec replace     R@1={fmt(out[f'v4_plus_rec_replace_h{int(h)}'])}",
            flush=True,
        )
        print(
            f"  v4+rec additive    R@1={fmt(out[f'v4_plus_rec_add_h{int(h)}'])}",
            flush=True,
        )
        print(
            f"  rer*rec mult       R@1={fmt(out[f'rer_plus_rec_mult_h{int(h)}'])}",
            flush=True,
        )
        print(
            f"  rer+rec additive   R@1={fmt(out[f'rer_plus_rec_add_h{int(h)}'])}",
            flush=True,
        )
        print(
            f"  rer+rec replace    R@1={fmt(out[f'rer_plus_rec_replace_h{int(h)}'])}",
            flush=True,
        )

    return out


def write_md(report: dict, path: Path):
    lines = []
    lines.append("# T_recency — Recency-decay scoring channel\n")
    lines.append(
        "Recency cue regex (`latest`, `most recent`, `last`, `recently`, `newly`, `just`, "
        "`current(ly)`, `present`, `now`) gates an exponential-decay channel:\n"
    )
    lines.append(
        "```\nrecency_score = exp(-λ * |query_ref - doc_anchor| / 1 day)\n```\n"
    )
    lines.append(
        "Doc anchor = MAX over all TE `best_us`; falls back to doc `ref_time` if no TE. "
        "Cue gate uses simple regex on query text; suppressed when explicit date phrase "
        'appears (e.g. "last week", "last Monday", "current 2024").\n'
    )

    benches = report["benches"]

    # ---- LATEST_RECENT HEADLINE ----
    lines.append("## Headline: latest_recent R@1\n")
    lr = benches.get("latest_recent")
    if lr and "error" not in lr:
        lines.append(f"- n={lr['n']}, cued={lr['n_cued']}/{lr['n']}")
        lines.append(
            f"- rerank_only:        {lr['rerank_only_rank']['R@1']:.3f} ({lr['rerank_only_rank']['r1_count']}/{lr['n']})"
        )
        lines.append(
            f"- T_lblend:           {lr['t_lblend_rank']['R@1']:.3f} ({lr['t_lblend_rank']['r1_count']}/{lr['n']})"
        )
        lines.append(
            f"- T_v4:               {lr['t_v4_rank']['R@1']:.3f} ({lr['t_v4_rank']['r1_count']}/{lr['n']})"
        )
        for h in HALF_LIVES:
            hi = int(h)
            ro = lr[f"recency_only_h{hi}"]
            lr2 = lr[f"lb_plus_rec_replace_h{hi}"]
            la = lr[f"lb_plus_rec_add_h{hi}"]
            v4r = lr[f"v4_plus_rec_replace_h{hi}"]
            v4a = lr[f"v4_plus_rec_add_h{hi}"]
            rrm = lr[f"rer_plus_rec_mult_h{hi}"]
            rra = lr[f"rer_plus_rec_add_h{hi}"]
            rrr = lr[f"rer_plus_rec_replace_h{hi}"]
            lines.append("")
            lines.append(f"  half-life={hi}d:")
            lines.append(
                f"  - recency_only:        {ro['R@1']:.3f} ({ro['r1_count']}/{lr['n']})"
            )
            lines.append(
                f"  - lb + recency replace:{lr2['R@1']:.3f} ({lr2['r1_count']}/{lr['n']})"
            )
            lines.append(
                f"  - lb + recency add:    {la['R@1']:.3f} ({la['r1_count']}/{lr['n']})"
            )
            lines.append(
                f"  - v4 + recency replace:{v4r['R@1']:.3f} ({v4r['r1_count']}/{lr['n']})"
            )
            lines.append(
                f"  - v4 + recency add:    {v4a['R@1']:.3f} ({v4a['r1_count']}/{lr['n']})"
            )
            lines.append(
                f"  - rerank * recency:    {rrm['R@1']:.3f} ({rrm['r1_count']}/{lr['n']})"
            )
            lines.append(
                f"  - rerank + recency add:{rra['R@1']:.3f} ({rra['r1_count']}/{lr['n']})"
            )
            lines.append(
                f"  - rerank > recency:    {rrr['R@1']:.3f} ({rrr['r1_count']}/{lr['n']})  [recipe: candidate-set=rerank, rank=recency]"
            )
    lines.append("")

    # ---- REGRESSION TABLE (R@1) at chosen half-life: pick the best for latest_recent ----
    # Find best half-life for latest_recent across all recipes that include
    # rerank — these are the only ones expected to clear the topic-blind bar.
    best_h = HALF_LIVES[1]  # default 21d
    best_recipe = "rer_plus_rec_replace"
    if lr and "error" not in lr:
        best_score = -1.0
        for h in HALF_LIVES:
            hi = int(h)
            for rk in ("rer_plus_rec_replace", "rer_plus_rec_mult", "rer_plus_rec_add"):
                s = lr[f"{rk}_h{hi}"]["R@1"]
                if s > best_score:
                    best_score = s
                    best_h = h
                    best_recipe = rk
    bh = int(best_h)

    lines.append(
        f"## R@1 regression check (half-life={bh}d, headline recipe = `{best_recipe}`)\n"
    )
    lines.append(
        "| Benchmark | n | cued | rerank_only | T_lblend | T_v4 | rec_only | lb+rec rep | v4+rec rep | rer*rec | rer+rec add | rer>rec |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, b in benches.items():
        if "error" in b:
            lines.append(f"| {name} | err | - | - | - | - | - | - | - | - | - | - |")
            continue
        n = b["n"]
        nc = b["n_cued"]
        ro = b["rerank_only_rank"]
        lb = b["t_lblend_rank"]
        v4 = b["t_v4_rank"]
        ro_r = b[f"recency_only_h{bh}"]
        lb_r = b[f"lb_plus_rec_replace_h{bh}"]
        v4_r = b[f"v4_plus_rec_replace_h{bh}"]
        rrm = b[f"rer_plus_rec_mult_h{bh}"]
        rra = b[f"rer_plus_rec_add_h{bh}"]
        rrr = b[f"rer_plus_rec_replace_h{bh}"]
        lines.append(
            f"| {name} | {n} | {nc} | "
            f"{ro['R@1']:.3f} | {lb['R@1']:.3f} | {v4['R@1']:.3f} | "
            f"{ro_r['R@1']:.3f} | {lb_r['R@1']:.3f} | {v4_r['R@1']:.3f} | "
            f"{rrm['R@1']:.3f} | {rra['R@1']:.3f} | {rrr['R@1']:.3f} |"
        )
    lines.append("")

    # ---- Half-life sweep (latest_recent only) ----
    lines.append("## Half-life sweep — latest_recent\n")
    lines.append(
        "| half-life (d) | recency_only | lb+rec rep | v4+rec rep | rer*rec | rer+rec add | rer>rec |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    if lr and "error" not in lr:
        for h in HALF_LIVES:
            hi = int(h)
            lines.append(
                f"| {hi} | "
                f"{lr[f'recency_only_h{hi}']['R@1']:.3f} | "
                f"{lr[f'lb_plus_rec_replace_h{hi}']['R@1']:.3f} | "
                f"{lr[f'v4_plus_rec_replace_h{hi}']['R@1']:.3f} | "
                f"{lr[f'rer_plus_rec_mult_h{hi}']['R@1']:.3f} | "
                f"{lr[f'rer_plus_rec_add_h{hi}']['R@1']:.3f} | "
                f"{lr[f'rer_plus_rec_replace_h{hi}']['R@1']:.3f} |"
            )
    lines.append("")

    # ---- Summary verdict ----
    lines.append("## Verdict\n")
    if lr and "error" not in lr:
        baseline = max(
            lr["rerank_only_rank"]["R@1"],
            lr["t_lblend_rank"]["R@1"],
            lr["t_v4_rank"]["R@1"],
        )
        best_recency = 0.0
        best_label = ""
        for h in HALF_LIVES:
            hi = int(h)
            for label_key, label_str in [
                (f"recency_only_h{hi}", f"recency_only h={hi}"),
                (f"lb_plus_rec_replace_h{hi}", f"lb+rec replace h={hi}"),
                (f"lb_plus_rec_add_h{hi}", f"lb+rec add h={hi}"),
                (f"v4_plus_rec_replace_h{hi}", f"v4+rec replace h={hi}"),
                (f"v4_plus_rec_add_h{hi}", f"v4+rec add h={hi}"),
                (f"rer_plus_rec_mult_h{hi}", f"rerank*recency h={hi}"),
                (f"rer_plus_rec_add_h{hi}", f"rerank+recency add h={hi}"),
                (f"rer_plus_rec_replace_h{hi}", f"rerank>recency h={hi}"),
            ]:
                s = lr[label_key]["R@1"]
                if s > best_recency:
                    best_recency = s
                    best_label = label_str
        lines.append(
            f"- **latest_recent**: best baseline R@1 = {baseline:.3f}; best recency = {best_recency:.3f} ({best_label}); "
            f"Δ = {best_recency - baseline:+.3f}."
        )

    # Regression check: did any non-latest_recent benchmark drop vs the
    # rerank_only baseline (the canonical reference for the headline recipes)?
    regressions = []
    for name, b in benches.items():
        if name == "latest_recent" or "error" in b:
            continue
        ro_base = b["rerank_only_rank"]["R@1"]
        rrr = b[f"rer_plus_rec_replace_h{bh}"]["R@1"]
        rrm = b[f"rer_plus_rec_mult_h{bh}"]["R@1"]
        rra = b[f"rer_plus_rec_add_h{bh}"]["R@1"]
        deltas = {
            "rerank>recency": rrr - ro_base,
            "rerank*recency": rrm - ro_base,
            "rerank+rec add": rra - ro_base,
        }
        if any(d < -0.005 for d in deltas.values()):
            regressions.append((name, deltas))
    if regressions:
        lines.append(
            f"- **Regressions vs rerank_only baseline** (at h={bh}d, only when at least one recipe drops):"
        )
        for n, deltas in regressions:
            parts = ", ".join(f"{k} Δ={v:+.3f}" for k, v in deltas.items())
            lines.append(f"  - {n}: {parts}")
    else:
        lines.append(
            "- **No regressions detected**: cue gate prevents recency activation on non-recency benches."
        )

    lines.append("")
    lines.append("## Limitations\n")
    lines.append(
        "- **Anti-decay queries** (`earliest`, `first`, `originally`, `started`) are NOT handled — "
        "they need an inverse-decay (small `(ref - doc)` is bad, large is good). Add an `earliest` cue "
        "list and flip the sign of the exponent."
    )
    lines.append(
        "- **Future-leaning queries** (`upcoming`, `next`) need anchor-anchored decay around a future "
        "reference, not `ref_time`."
    )
    lines.append(
        '- **Ambiguous `last`**: the cue list includes `last` ("last appointment") but a suppress-rule '
        'filters out "last week/month/year/Monday/..." since those carry their own date anchors.'
    )
    lines.append(
        '- **Multi-cue conjunctions** ("latest project finished after Q1") are scored on recency only; '
        "the overlap channel is dropped. A weighted sum (additive) preserves both signals at the cost of "
        "muddier ranking when one signal is strong."
    )
    lines.append(
        '- **Cue gate false-positives** on non-recency uses of `now`/`current`/`just`: e.g. "Just then I '
        'thought ...". Suppress-rule covers a few common cases but not all.'
    )

    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------
async def main():
    print("Loading cross-encoder...", flush=True)
    from memmachine_server.common.reranker.cross_encoder_reranker import (
        CrossEncoderReranker,
        CrossEncoderRerankerParams,
    )
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    reranker = CrossEncoderReranker(
        CrossEncoderRerankerParams(
            cross_encoder=ce,
            max_input_length=512,
        )
    )

    benches_main = [
        # Primary
        (
            "latest_recent",
            "latest_recent_docs.jsonl",
            "latest_recent_queries.jsonl",
            "latest_recent_gold.jsonl",
            "edge-latest_recent",
        ),
        # Regression checks
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
        (
            "tempreason_small",
            "real_benchmark_small_docs.jsonl",
            "real_benchmark_small_queries.jsonl",
            "real_benchmark_small_gold.jsonl",
            "v7l-tempreason_small",
        ),
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
            "era_refs",
            "edge_era_refs_docs.jsonl",
            "edge_era_refs_queries.jsonl",
            "edge_era_refs_gold.jsonl",
            "edge-era_refs",
        ),
        (
            "open_ended_date",
            "open_ended_date_docs.jsonl",
            "open_ended_date_queries.jsonl",
            "open_ended_date_gold.jsonl",
            "edge-open_ended_date",
        ),
        (
            "causal_relative",
            "causal_relative_docs.jsonl",
            "causal_relative_queries.jsonl",
            "causal_relative_gold.jsonl",
            "edge-causal_relative",
        ),
        (
            "negation_temporal",
            "negation_temporal_docs.jsonl",
            "negation_temporal_queries.jsonl",
            "negation_temporal_gold.jsonl",
            "edge-negation_temporal",
        ),
    ]

    out = {"benches": {}}
    for name, dp, qp, gp, cache_label in benches_main:
        # Tolerate alternate edge_*-prefixed file names.
        if not (DATA_DIR / dp).exists():
            alt = f"edge_{dp}"
            if (DATA_DIR / alt).exists():
                dp = alt
        if not (DATA_DIR / qp).exists():
            alt = f"edge_{qp}"
            if (DATA_DIR / alt).exists():
                qp = alt
        if not (DATA_DIR / gp).exists():
            alt = f"edge_{gp}"
            if (DATA_DIR / alt).exists():
                gp = alt
        if not (DATA_DIR / dp).exists():
            print(f"  [{name}] missing {dp} - skipping", flush=True)
            continue
        try:
            agg = await run_bench(name, dp, qp, gp, cache_label, reranker)
            out["benches"][name] = agg
        except Exception as e:
            import traceback

            traceback.print_exc()
            out["benches"][name] = {"error": str(e), "n": 0}

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_recency.json"
    json_safe = {"benches": {}}
    for k, v in out["benches"].items():
        if "error" in v:
            json_safe["benches"][k] = v
            continue
        v2 = {kk: vv for kk, vv in v.items() if kk != "per_q"}
        v2["per_q"] = v.get("per_q", [])
        json_safe["benches"][k] = v2
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2, default=str)
    print(f"\nWrote {json_path}", flush=True)

    md_path = out_dir / "T_recency.md"
    write_md(out, md_path)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
