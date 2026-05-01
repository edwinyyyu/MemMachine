"""Pair-comparison switches eval.

For each candidate dimension D ∈ {T, recency}, the gate sees:
  R_0 = baseline ranking (rerank_only top-5)
  R_D = (rerank + D additive) top-5

If gate picks R_D, dim D is active. Active dims combine:
  - T-only         → fuse_T_R (score_blend over {T, R} at w_T=0.4)
  - rec-only       → rerank_only with α=0.5 additive recency
  - both           → fuse_T_R then α=0.5 additive recency on the fused score
  - neither        → rerank_only

Compares against:
  1. rerank_only
  2. fuse_T_R (always-on T)
  3. fuse_T_R + recency_additive (regex-switched recency, current best)
  4. pair_switches  (this design)
  5. query_only_llm_switches (one prompt → JSON {T_active, recency_active})

Uses gpt-5-nano for all gates (cheaper, proven sufficient on pair-pick).
Prompt for pair gate: chrono+set+1-token format from set_pickers_eval.py.
Prompt for query-only: JSON output, no candidate sets shown.

Cache: cache/pair_switches/  (separate from production caches).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
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

# ---------------------------------------------------------------------------
# Override BlindJudge cache + model BEFORE importing it.
# ---------------------------------------------------------------------------
import v7l_ts_blind_eval as base

PAIR_CACHE_DIR = ROOT / "cache" / "pair_switches"
PAIR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
base.CACHE_DIR = PAIR_CACHE_DIR
base.CACHE_FILE = PAIR_CACHE_DIR / "llm_cache.json"
base.MODEL = "gpt-5-nano"

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
from multi_channel_eval import (
    ADDITIVE_ALPHA,
    CV_REF,
    HALF_LIFE_DAYS,
    W_R_FUSE_TR,
    W_T_FUSE_TR,
    additive_with_recency,
    fuse_T_R_blend,
    has_temporal_anchor,
    normalize_rerank_full,
)
from rag_fusion import score_blend
from recency import (
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
from v7l_ts_blind_eval import BlindJudge, _key

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# Mirrors set_pickers_eval.PICK_PROMPT (chrono+set+1-token format), the
# production gate format.
PICK_PROMPT = """Pick the candidate SET whose collection of documents better answers the user's query.

Each set is an UNORDERED pool — documents are listed chronologically (by date), NOT by relevance. Order is not signal. Judge each set as a whole: does its pool contain the information needed to answer the question?

If the query has a temporal anchor (year, quarter, month, date range): prefer the set with more docs inside that period. Docs outside the period are distractors regardless of topical match.
If the query has NO temporal anchor: ignore dates and judge by topical/entity match (right person, right event-type).

QUERY: {query}

{sets}

Output exactly one letter from {{{choices}}} — the better set. No commentary.
"""


QUERY_ONLY_PROMPT = """A retrieval system has two optional ranking signals:

  T = temporal-anchor matching (matches dates/intervals/era cues in the query against doc timestamps and extracted intervals).
  recency = freshness boost (favours docs whose dates are close to the query's reference time, e.g. for "latest" or "most recent" queries).

Decide whether each signal should be ON for this query, then output a one-line JSON object.

Rules:
- T_active = true if the query mentions any explicit or implicit temporal scope (year, quarter, month, season, decade, era like "1990s", relative-time phrase like "after the merger", date range, or any other temporal grounding cue).
- recency_active = true if the query asks for the latest / most recent / current / etc. — i.e. wants the freshest item, not a specific historical period.
- Both can be true; both can be false.

QUERY: {query}

Output exactly: {{"T_active": <true|false>, "recency_active": <true|false>}} — no commentary, no markdown.
"""


# ---------------------------------------------------------------------------
# Set formatting
# ---------------------------------------------------------------------------
def _format_set(label: str, docs: list[tuple[str, str, str]]) -> str:
    """docs: list of (doc_id, date_str, snippet). Sort by date."""
    sorted_docs = sorted(docs, key=lambda x: x[1] or "")
    lines = [f"Set {label}:"]
    for did, date_str, text in sorted_docs:
        snippet = text[:200].replace("\n", " ")
        date_label = f"[{date_str}]" if date_str else "[date unknown]"
        lines.append(f"  - {date_label} {snippet}")
    return "\n".join(lines)


async def pick_n(
    judge: BlindJudge, query: str, sets: list[list[tuple[str, str, str]]], rng_seed: int
) -> int:
    """Pick best of N sets (chrono+set+1-token format)."""
    n = len(sets)
    labels = ["A", "B", "C", "D", "E", "F"][:n]
    rng = random.Random(rng_seed)
    order = list(range(n))
    rng.shuffle(order)
    shuffled = [sets[i] for i in order]
    formatted = "\n\n".join(_format_set(labels[i], s) for i, s in enumerate(shuffled))
    choices = ", ".join(labels)
    prompt = PICK_PROMPT.format(query=query, sets=formatted, choices=choices)
    k = _key(f"pairswitch_pick_{n}", prompt)
    if k in judge.cache:
        raw = judge.cache[k]
    else:
        judge.calls += 1
        raw = await judge._llm(prompt, max_tokens=8)
        if raw:
            judge.cache[k] = raw
            judge._dirty = True
    m = re.search(r"\b([A-F])\b", (raw or "").upper())
    if not m:
        return order[0]
    letter = m.group(1)
    if letter not in labels:
        return order[0]
    pos = labels.index(letter)
    return order[pos]


async def query_only_switch(judge: BlindJudge, query: str) -> tuple[bool, bool]:
    """Single LLM call: JSON {T_active, recency_active}."""
    prompt = QUERY_ONLY_PROMPT.format(query=query)
    k = _key("pairswitch_qonly", prompt)
    if k in judge.cache:
        raw = judge.cache[k]
    else:
        judge.calls += 1
        raw = await judge._llm(prompt, max_tokens=64)
        if raw:
            judge.cache[k] = raw
            judge._dirty = True
    raw = (raw or "").strip()
    # Extract first JSON object.
    m = re.search(r"\{[^{}]*\}", raw)
    if not m:
        return (False, False)
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return (False, False)
    return (bool(obj.get("T_active", False)), bool(obj.get("recency_active", False)))


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------
def hit_rank(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


def rank_from_scores(scores: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def build_rerank_only_scores(
    rerank_partial: dict[str, float], s_scores: dict[str, float]
) -> dict[str, float]:
    """rerank_only as a score dict: rerank scores, with semantic as tail-tiebreak."""
    rmin = min(rerank_partial.values()) if rerank_partial else 0.0
    out = dict(rerank_partial)
    # Tail: docs not in rerank set get a tiny score scaled by semantic.
    smax = max(s_scores.values()) if s_scores else 1.0
    for d, sc in s_scores.items():
        if d not in out:
            out[d] = rmin - 1.0 + (sc / smax if smax > 0 else 0.0) * 1e-3
    return out


def rerank_only_top5(
    rerank_partial: dict[str, float], s_scores: dict[str, float]
) -> list[str]:
    primary = [
        d for d, _ in sorted(rerank_partial.items(), key=lambda x: x[1], reverse=True)
    ]
    return merge_with_tail(primary, s_scores)[:5]


def fuse_T_R_top5(t_scores, r_full, s_scores) -> list[str]:
    primary = fuse_T_R_blend(t_scores, r_full, w_T=W_T_FUSE_TR)
    rest = [d for d in rank_from_scores(s_scores) if d not in set(primary)]
    return (primary + rest)[:5]


def recency_additive_top5(
    rerank_partial, rec_scores, s_scores, alpha=ADDITIVE_ALPHA, all_doc_ids=None
):
    """rerank_only base + α-additive recency (no T)."""
    base_full = build_rerank_only_scores(rerank_partial, s_scores)
    blended = additive_with_recency(base_full, rec_scores, cue=True, alpha=alpha)
    return rank_from_scores(blended)[:5]


def combine_active_dims(
    t_scores, r_full, rec_scores, s_scores, T_active: bool, recency_active: bool
) -> list[str]:
    """Final ranking by active dimension combination."""
    if not T_active and not recency_active:
        primary = [
            d for d, _ in sorted(r_full.items(), key=lambda x: x[1], reverse=True)
        ]
        return merge_with_tail(primary, s_scores)
    if T_active and not recency_active:
        primary = fuse_T_R_blend(t_scores, r_full, w_T=W_T_FUSE_TR)
        rest = [d for d in rank_from_scores(s_scores) if d not in set(primary)]
        return primary + rest
    if not T_active and recency_active:
        # rerank_only base, recency-additive on top.
        base_full = dict(r_full)
        blended = additive_with_recency(
            base_full, rec_scores, cue=True, alpha=ADDITIVE_ALPHA
        )
        primary = rank_from_scores(blended)
        # Tail with semantic for any missing.
        rest = [d for d in rank_from_scores(s_scores) if d not in set(primary)]
        return primary + rest
    # Both active: fuse_T_R + recency_additive (matches variant 3 semantics).
    fused_TR_scores = dict(
        score_blend(
            {"T": t_scores, "R": r_full},
            {"T": W_T_FUSE_TR, "R": W_R_FUSE_TR},
            top_k_per=40,
            dispersion_cv_ref=CV_REF,
        )
    )
    blended = additive_with_recency(
        fused_TR_scores, rec_scores, cue=True, alpha=ADDITIVE_ALPHA
    )
    primary = rank_from_scores(blended)
    rest = [d for d in rank_from_scores(s_scores) if d not in set(primary)]
    return primary + rest


# ---------------------------------------------------------------------------
# Per-query gate (parallel)
# ---------------------------------------------------------------------------
async def gate_T(
    qid,
    q_text,
    t_scores,
    r_full,
    rec_scores,
    s_scores,
    rerank_partial,
    doc_text,
    doc_dates,
    judge,
):
    """Pair-gate on T: rerank_only vs fuse_T_R."""
    R0 = rerank_only_top5(rerank_partial, s_scores)
    RD = fuse_T_R_top5(t_scores, r_full, s_scores)
    if list(R0) == list(RD):
        return False  # identical → T adds nothing
    set_R0 = [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in R0]
    set_RD = [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in RD]
    seed = hash((qid, "pair_T")) & 0xFFFFFFFF
    idx = await pick_n(judge, q_text, [set_R0, set_RD], seed)
    return idx == 1


async def gate_recency(
    qid, q_text, rec_scores, s_scores, rerank_partial, doc_text, doc_dates, judge
):
    """Pair-gate on recency: rerank_only vs rerank_only+recency_additive."""
    R0 = rerank_only_top5(rerank_partial, s_scores)
    RD = recency_additive_top5(rerank_partial, rec_scores, s_scores)
    if list(R0) == list(RD):
        return False
    set_R0 = [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in R0]
    set_RD = [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in RD]
    seed = hash((qid, "pair_rec")) & 0xFFFFFFFF
    idx = await pick_n(judge, q_text, [set_R0, set_RD], seed)
    return idx == 1


# ---------------------------------------------------------------------------
# Bench loop
# ---------------------------------------------------------------------------
async def run_bench(
    name, docs_path, queries_path, gold_path, cache_label, reranker, judge
):
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
    doc_dates = {d["doc_id"]: d["ref_time"][:10] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    doc_ref_us = {d["doc_id"]: to_us(parse_iso(d["ref_time"])) for d in docs}
    q_ref_us = {q["query_id"]: to_us(parse_iso(q["ref_time"])) for q in queries}

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

    # Lattice.
    lat_db = ROOT / "cache" / "pair_switches" / f"lat_{name}.sqlite"
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

    # Recency anchor bundles.
    doc_bundles_for_rec: dict[str, list[dict]] = {}
    for did, mem in doc_mem.items():
        ivs = mem.get("intervals") or []
        doc_bundles_for_rec[did] = [{"intervals": ivs}] if ivs else []

    # Embed + semantic.
    print("  embedding + reranking...", flush=True)
    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    per_q_t = {
        qid: make_t_scores(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            per_q_l.get(qid, {}),
        )
        for qid in qids
    }
    for qid in qids:
        for d in docs:
            per_q_t[qid].setdefault(d["doc_id"], 0.0)

    per_q_r_full: dict[str, dict[str, float]] = {}
    per_q_r_partial: dict[str, dict[str, float]] = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        t_top = topk_from_scores(per_q_t[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + t_top))[:RERANK_TOP_K]
        rs = await rerank_topk(reranker, q_text[qid], union, doc_text, len(union))
        per_q_r_partial[qid] = rs
        per_q_r_full[qid] = normalize_rerank_full(
            rs, [d["doc_id"] for d in docs], tail_score=0.0
        )

    lam = lambda_for_half_life(HALF_LIFE_DAYS)

    # Pre-compute per-query rec_scores.
    per_q_rec: dict[str, dict[str, float]] = {}
    for qid in qids:
        per_q_rec[qid] = recency_scores_for_docs(
            doc_bundles_for_rec,
            doc_ref_us,
            q_ref_us[qid],
            lam,
        )

    # ---- Run gates concurrently per query --------------------------------
    print(
        f"  running pair gates + query-only LLM switches over {len(qids)} queries...",
        flush=True,
    )

    async def per_query_gates(q):
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            return None

        # Regex switches.
        T_regex = has_temporal_anchor(q["text"])
        Rec_regex = has_recency_cue(q["text"])

        # Run pair gates + query-only LLM in parallel.
        T_pair_task = gate_T(
            qid,
            q["text"],
            per_q_t[qid],
            per_q_r_full[qid],
            per_q_rec[qid],
            per_q_s[qid],
            per_q_r_partial[qid],
            doc_text,
            doc_dates,
            judge,
        )
        Rec_pair_task = gate_recency(
            qid,
            q["text"],
            per_q_rec[qid],
            per_q_s[qid],
            per_q_r_partial[qid],
            doc_text,
            doc_dates,
            judge,
        )
        Q_only_task = query_only_switch(judge, q["text"])
        T_pair, Rec_pair, (T_qonly, Rec_qonly) = await asyncio.gather(
            T_pair_task,
            Rec_pair_task,
            Q_only_task,
        )

        t_scores = per_q_t[qid]
        r_full = per_q_r_full[qid]
        rerank_partial = per_q_r_partial[qid]
        s_scores = per_q_s[qid]
        rec_scores = per_q_rec[qid]

        # Variant 1: rerank_only.
        rerank_only_rank = merge_with_tail(
            [
                d
                for d, _ in sorted(
                    rerank_partial.items(), key=lambda x: x[1], reverse=True
                )
            ],
            s_scores,
        )

        # Variant 2: fuse_T_R.
        primary_2 = fuse_T_R_blend(t_scores, r_full, w_T=W_T_FUSE_TR)
        rank_fuse_TR = primary_2 + [
            d for d in rank_from_scores(s_scores) if d not in set(primary_2)
        ]

        # Variant 3: regex-switched recency on top of fuse_T_R.
        fused_TR_scores = dict(
            score_blend(
                {"T": t_scores, "R": r_full},
                {"T": W_T_FUSE_TR, "R": W_R_FUSE_TR},
                top_k_per=40,
                dispersion_cv_ref=CV_REF,
            )
        )
        fused_TR_with_rec = additive_with_recency(
            fused_TR_scores,
            rec_scores,
            cue=Rec_regex,
            alpha=ADDITIVE_ALPHA,
        )
        primary_3 = rank_from_scores(fused_TR_with_rec)
        rank_regex_switched = primary_3 + [
            d for d in rank_from_scores(s_scores) if d not in set(primary_3)
        ]

        # Variant 4: pair_switches.
        rank_pair = combine_active_dims(
            t_scores,
            r_full,
            rec_scores,
            s_scores,
            T_active=T_pair,
            recency_active=Rec_pair,
        )

        # Variant 5: query_only_llm.
        rank_qonly = combine_active_dims(
            t_scores,
            r_full,
            rec_scores,
            s_scores,
            T_active=T_qonly,
            recency_active=Rec_qonly,
        )

        return {
            "qid": qid,
            "qtext": q.get("text", "")[:200],
            "gold": list(gold_set),
            "T_regex": T_regex,
            "Rec_regex": Rec_regex,
            "T_pair": T_pair,
            "Rec_pair": Rec_pair,
            "T_qonly": T_qonly,
            "Rec_qonly": Rec_qonly,
            "rerank_only": hit_rank(rerank_only_rank, gold_set),
            "fuse_T_R": hit_rank(rank_fuse_TR, gold_set),
            "regex_switched": hit_rank(rank_regex_switched, gold_set),
            "pair_switches": hit_rank(rank_pair, gold_set),
            "query_only_llm": hit_rank(rank_qonly, gold_set),
        }

    # Run per-query gates with bounded concurrency (judge has its own sem too).
    sem = asyncio.Semaphore(8)

    async def bounded(q):
        async with sem:
            return await per_query_gates(q)

    raw_results = await asyncio.gather(*(bounded(q) for q in queries))
    results = [r for r in raw_results if r is not None]

    return aggregate(results, name)


def aggregate(results, label):
    n = len(results)
    out = {"label": label, "n": n, "per_q": results}
    variants = [
        "rerank_only",
        "fuse_T_R",
        "regex_switched",
        "pair_switches",
        "query_only_llm",
    ]
    for var in variants:
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

    # Switch firing counts.
    out["fire"] = {
        "T_regex": sum(1 for r in results if r["T_regex"]),
        "Rec_regex": sum(1 for r in results if r["Rec_regex"]),
        "T_pair": sum(1 for r in results if r["T_pair"]),
        "Rec_pair": sum(1 for r in results if r["Rec_pair"]),
        "T_qonly": sum(1 for r in results if r["T_qonly"]),
        "Rec_qonly": sum(1 for r in results if r["Rec_qonly"]),
    }

    print(f"  n={n}  fire: {out['fire']}", flush=True)
    for var in variants:
        d = out[var]
        print(
            f"  {var:18s}  R@1={d['R@1']:.3f} ({d['r1_count']}/{n})  "
            f"R@5={d['R@5']:.3f} ({d['r5_count']}/{n})  MRR={d['MRR']:.3f}",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# MD report
# ---------------------------------------------------------------------------
def write_md(out: dict, path: Path, total_calls: int, n_queries_total: int):
    benches = out["benches"]
    lines = []
    lines.append("# T_pair_switches — pair-comparison switches eval\n")
    lines.append("Gate model: gpt-5-nano. Pair-gate format: chrono+set+1-token.\n")
    lines.append(
        "Per query: 2 pair gates (T, recency) + 1 query-only LLM call (for variant 5). All 3 in parallel.\n"
    )
    lines.append(f"Recency: half-life={HALF_LIFE_DAYS}d, α={ADDITIVE_ALPHA}.\n")

    # ---- R@1 deltas vs regex-switched (lead) -----------------------------
    lines.append("\n## R@1 deltas vs regex_switched (LEAD)\n")
    lines.append(
        "| Benchmark | n | rerank_only | fuse_T_R | regex_switched | pair_switches | Δ pair vs regex | query_only_llm | Δ qonly vs regex |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, b in benches.items():
        if "error" in b:
            lines.append(f"| {name} | err | - | - | - | - | - | - | - |")
            continue
        n = b["n"]
        ro = b["rerank_only"]["R@1"]
        ft = b["fuse_T_R"]["R@1"]
        rs = b["regex_switched"]["R@1"]
        ps = b["pair_switches"]["R@1"]
        qo = b["query_only_llm"]["R@1"]
        d_ps = ps - rs
        d_qo = qo - rs
        lines.append(
            f"| {name} | {n} | {ro:.3f} | {ft:.3f} | {rs:.3f} | {ps:.3f} | "
            f"{d_ps:+.3f} | {qo:.3f} | {d_qo:+.3f} |"
        )
    # Macro avg.
    valid = [b for b in benches.values() if "error" not in b]
    if valid:
        macro = lambda var: sum(b[var]["R@1"] for b in valid) / len(valid)
        m_ro, m_ft, m_rs = (
            macro("rerank_only"),
            macro("fuse_T_R"),
            macro("regex_switched"),
        )
        m_ps, m_qo = macro("pair_switches"), macro("query_only_llm")
        lines.append(
            f"| **macro avg** | - | {m_ro:.3f} | {m_ft:.3f} | {m_rs:.3f} | {m_ps:.3f} | "
            f"{m_ps - m_rs:+.3f} | {m_qo:.3f} | {m_qo - m_rs:+.3f} |"
        )
    lines.append("")

    # ---- R@5 ------------------------------------------------------------
    lines.append("## R@5 by benchmark\n")
    lines.append(
        "| Benchmark | n | rerank_only | fuse_T_R | regex_switched | pair_switches | query_only_llm |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, b in benches.items():
        if "error" in b:
            continue
        n = b["n"]
        lines.append(
            f"| {name} | {n} | "
            f"{b['rerank_only']['R@5']:.3f} | {b['fuse_T_R']['R@5']:.3f} | "
            f"{b['regex_switched']['R@5']:.3f} | {b['pair_switches']['R@5']:.3f} | "
            f"{b['query_only_llm']['R@5']:.3f} |"
        )
    lines.append("")

    # ---- Switch firing pattern ------------------------------------------
    lines.append("## Switch firing pattern\n")
    lines.append(
        "Number of queries where each switch fires (T-active and recency-active counts).\n"
    )
    lines.append(
        "| Benchmark | n | T_regex | T_pair | T_qonly | Rec_regex | Rec_pair | Rec_qonly |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, b in benches.items():
        if "error" in b:
            continue
        f = b["fire"]
        lines.append(
            f"| {name} | {b['n']} | {f['T_regex']} | {f['T_pair']} | {f['T_qonly']} | "
            f"{f['Rec_regex']} | {f['Rec_pair']} | {f['Rec_qonly']} |"
        )
    lines.append("")

    # ---- Cost ------------------------------------------------------------
    lines.append("## Cost\n")
    lines.append(
        f"- Total LLM calls (across cache misses, both pair gates and qonly): **{total_calls}**.\n"
    )
    lines.append(f"- Total queries judged: **{n_queries_total}**.\n")
    lines.append(
        "- pair_switches budget per query: **2** LLM calls (T-gate + rec-gate).\n"
    )
    lines.append("- query_only_llm budget per query: **1** LLM call.\n")
    lines.append("- regex_switched: 0 LLM calls.\n")
    lines.append("")

    # ---- Per-benchmark deltas commentary --------------------------------
    if valid:
        wins = []
        losses = []
        for name, b in benches.items():
            if "error" in b:
                continue
            d = b["pair_switches"]["R@1"] - b["regex_switched"]["R@1"]
            if d >= 0.02:
                wins.append((name, d))
            elif d <= -0.02:
                losses.append((name, d))
        lines.append("## pair_switches vs regex_switched: per-bench breakdown\n")
        if wins:
            lines.append("**pair_switches wins (Δ ≥ +0.02 R@1):**\n")
            for n, d in sorted(wins, key=lambda x: -x[1]):
                lines.append(f"- {n}: {d:+.3f}")
            lines.append("")
        if losses:
            lines.append("**pair_switches losses (Δ ≤ −0.02 R@1):**\n")
            for n, d in sorted(losses, key=lambda x: x[1]):
                lines.append(f"- {n}: {d:+.3f}")
            lines.append("")
        if not wins and not losses:
            lines.append("- All deltas within ±0.02 R@1 (essentially tied).\n")

    # ---- Recommendation -------------------------------------------------
    lines.append("## Recommendation\n")
    if valid:
        d_macro = m_ps - m_rs
        d_qo_macro = m_qo - m_rs
        if d_macro >= 0.01:
            lines.append(
                f"- **Ship pair_switches**: macro R@1 +{d_macro:.3f} over regex_switched, "
                f"at cost of 2 LLM calls per query. "
            )
        elif d_macro <= -0.01:
            lines.append(
                f"- **Stick with regex_switched**: pair_switches macro R@1 is {d_macro:+.3f}; "
                f"the LLM-in-loop pair-gate does not pay off across this benchmark suite. "
            )
        else:
            lines.append(
                f"- **Stick with regex_switched**: pair_switches macro R@1 is {d_macro:+.3f} "
                f"(within noise). The 2-LLM-call budget per query is not justified by this delta. "
            )
        if abs(d_qo_macro - d_macro) >= 0.01:
            if d_qo_macro > d_macro:
                lines.append(
                    f"  query_only_llm beats pair_switches on macro by {d_qo_macro - d_macro:+.3f} "
                    f"R@1 — single-call gating is the better LLM-in-loop design.\n"
                )
            else:
                lines.append(
                    f"  pair_switches beats query_only_llm on macro by {d_macro - d_qo_macro:+.3f} "
                    f"R@1 — looking at candidate sets adds value over query-only.\n"
                )
        else:
            lines.append("\n")

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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

    judge = BlindJudge()
    print(f"Judge model: {base.MODEL}, cache: {base.CACHE_FILE}", flush=True)

    benches_main = [
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
            "latest_recent",
            "latest_recent_docs.jsonl",
            "latest_recent_queries.jsonl",
            "latest_recent_gold.jsonl",
            "edge-latest_recent",
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
    n_queries_total = 0
    for name, dp, qp, gp, cache_label in benches_main:
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
            agg = await run_bench(name, dp, qp, gp, cache_label, reranker, judge)
            out["benches"][name] = agg
            n_queries_total += agg["n"]
        except Exception as e:
            import traceback

            traceback.print_exc()
            out["benches"][name] = {"error": str(e), "n": 0}

    judge.save()

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_pair_switches.json"
    json_safe = {
        "benches": {},
        "judge_calls": judge.calls,
        "judge_failed": judge.failed,
        "judge_usage": judge.usage,
        "n_queries_total": n_queries_total,
    }
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

    md_path = out_dir / "T_pair_switches.md"
    write_md(out, md_path, total_calls=judge.calls, n_queries_total=n_queries_total)
    print(f"Wrote {md_path}", flush=True)
    print(
        f"Judge: calls={judge.calls} failed={judge.failed} usage={judge.usage}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
