"""Hybrid set-picking optimizers: combine the gate's distribution-aware
rerank-only fallback with bisect_thirds' per-query weight tuning.

Variants:
  hybrid_old_gate  Step 1: gate using OLD format (rank-numbered "best match")
                   to decide rerank_only vs fuse_T_R(w=0.4). If picks
                   rerank_only → return. Else → bisect_thirds (new format).
                   Cost: 1 call non-temporal, 5 calls temporal.

  hybrid_new_gate  Same but step 1 uses NEW format gate (chrono-sorted set,
                   "set" framing). Tests if format consistency helps.

  multi_recipe_4   Single call, n=4 recipes (new format):
                   pure_S, rerank_only, fuse_T_R(w=0.4), fuse_T_R(w=0.6).

  fiveway          Single call, n=5: rerank_only + fuse at {0.0, 0.233, 0.467, 0.7}.
                   Includes rerank_only as explicit non-temporal option.

Compared against:
  - rerank_only (safe default)
  - fuse_T_R w=0.6 (oracle)
  - bisect_thirds (current per-corpus champion)

Tested on hard_bench + LME (non-temp) + mixed_cue + dense_cluster + tempreason_small.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT.parents[1] / ".env")

from force_pick_optimizers_eval import (
    RERANK_TOP_K,
    fuse_at_w,
    make_t_scores,
    merge_with_tail,
    rerank_topk,
    topk_from_scores,
)
from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
from lattice_store import LatticeStore
from salience_eval import (  # type: ignore
    AXES,
    DATA_DIR,
    AxisDistribution,
    build_memory,
    embed_all,
    parse_iso,
    rank_semantic,
    run_v2_extract,
)
from set_pickers_eval import (
    hit_rank,
    pick_n,
    run_bisect_thirds,
)
from v7l_ts_blind_eval import BlindJudge, _key

# ---------- OLD-format gate ----------

OLD_PICK_PROMPT = """You are evaluating retrieval candidates for a query. Each CANDIDATE below is a top-5 list of documents produced by a different blend of topic-match and time-match signals. Your job is to pick the candidate whose top-5 best matches the query.

================ QUERY ================
{query}

============ CANDIDATES ============
{sets}

Output rules:
  - Output exactly one integer from {{ {choices} }} — the candidate number.
  - No commentary, just the number.
"""


def _format_old(label: str, docs: list[tuple[str, str, str]]) -> str:
    """OLD format: top-5 numbered list, no date labels, doc-order = relevance order."""
    lines = [f"Candidate {label}:"]
    for i, (did, _date, text) in enumerate(docs):
        snippet = text[:200].replace("\n", " ")
        lines.append(f"  {i + 1}. {snippet}")
    return "\n".join(lines)


async def pick_old_format(
    judge: BlindJudge, query: str, sets: list[list[tuple[str, str, str]]], rng_seed: int
) -> int:
    n = len(sets)
    rng = random.Random(rng_seed)
    order = list(range(n))
    rng.shuffle(order)
    shuffled = [sets[i] for i in order]
    formatted = "\n\n".join(_format_old(str(i + 1), s) for i, s in enumerate(shuffled))
    choices = ", ".join(str(i + 1) for i in range(n))
    prompt = OLD_PICK_PROMPT.format(query=query, sets=formatted, choices=choices)
    k = _key(f"oldfmt_{n}", prompt)
    if k in judge.cache:
        raw = judge.cache[k]
    else:
        judge.calls += 1
        raw = await judge._llm(prompt, max_tokens=8)
        if raw:
            judge.cache[k] = raw
            judge._dirty = True
    m = re.search(r"\d+", raw or "")
    if not m:
        return order[0]
    try:
        v = int(m.group(0))
    except ValueError:
        return order[0]
    if v < 1 or v > n:
        return order[0]
    return order[v - 1]


# ---------- Hybrids ----------


async def run_hybrid_old_gate(
    qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
):
    """OLD-format gate first; if picks rerank_only return it, else bisect_thirds."""
    s_top_50 = topk_from_scores(s_scores, RERANK_TOP_K)
    rs = {did: r_scores.get(did, 0.0) for did in s_top_50}
    rerank_only = merge_with_tail(
        [d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)], s_scores
    )
    fuse_04 = fuse_at_w(t_scores, r_scores, s_scores, 0.4)
    s_r = [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in rerank_only[:5]]
    s_f = [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in fuse_04[:5]]
    seed = hash((qid, "hybrid_oldgate")) & 0xFFFFFFFF
    idx = await pick_old_format(judge, q_text, [s_r, s_f], seed)
    if idx == 0:
        return rerank_only, {"final_w_T": None, "decision": "rerank_only"}
    bisect_rank, bd = await run_bisect_thirds(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )
    return bisect_rank, {"final_w_T": bd["final_w_T"], "decision": "bisect"}


async def run_hybrid_new_gate(
    qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
):
    """NEW-format gate first; if picks rerank_only return it, else bisect_thirds."""
    s_top_50 = topk_from_scores(s_scores, RERANK_TOP_K)
    rs = {did: r_scores.get(did, 0.0) for did in s_top_50}
    rerank_only = merge_with_tail(
        [d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)], s_scores
    )
    fuse_04 = fuse_at_w(t_scores, r_scores, s_scores, 0.4)
    s_r = [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in rerank_only[:5]]
    s_f = [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in fuse_04[:5]]
    seed = hash((qid, "hybrid_newgate")) & 0xFFFFFFFF
    idx = await pick_n(judge, q_text, [s_r, s_f], seed)
    if idx == 0:
        return rerank_only, {"final_w_T": None, "decision": "rerank_only"}
    bisect_rank, bd = await run_bisect_thirds(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )
    return bisect_rank, {"final_w_T": bd["final_w_T"], "decision": "bisect"}


# ---------- Multi-recipe gate (n=4) ----------


async def run_multi_recipe_4(
    qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
):
    """4-way pick over recipes: pure_S, rerank_only, fuse_T_R(0.4), fuse_T_R(0.6)."""
    pure_s_rank = sorted(s_scores.items(), key=lambda x: x[1], reverse=True)
    pure_s = [d for d, _ in pure_s_rank]

    s_top_50 = topk_from_scores(s_scores, RERANK_TOP_K)
    rs = {did: r_scores.get(did, 0.0) for did in s_top_50}
    rerank_only = merge_with_tail(
        [d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)], s_scores
    )
    fuse_04 = fuse_at_w(t_scores, r_scores, s_scores, 0.4)
    fuse_06 = fuse_at_w(t_scores, r_scores, s_scores, 0.6)
    recipes = [pure_s, rerank_only, fuse_04, fuse_06]
    sets = [
        [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in r[:5]] for r in recipes
    ]
    seed = hash((qid, "multi4")) & 0xFFFFFFFF
    idx = await pick_n(judge, q_text, sets, seed)
    return recipes[idx], {"recipe_idx": idx}


# ---------- 5-way: rerank_only + 4 weights ----------


async def run_fiveway(
    qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
):
    s_top_50 = topk_from_scores(s_scores, RERANK_TOP_K)
    rs = {did: r_scores.get(did, 0.0) for did in s_top_50}
    rerank_only = merge_with_tail(
        [d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)], s_scores
    )
    weights = [0.0, 0.233, 0.467, 0.7]
    fuses = [fuse_at_w(t_scores, r_scores, s_scores, w) for w in weights]
    candidates = [rerank_only] + fuses
    sets = [
        [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in r[:5]]
        for r in candidates
    ]
    seed = hash((qid, "fiveway")) & 0xFFFFFFFF
    idx = await pick_n(judge, q_text, sets, seed)
    return candidates[idx], {"choice_idx": idx}


# ---------- per-query evaluation ----------


async def evaluate_query(
    qid, q_text, doc_text, doc_dates, gold_set, t_scores, s_scores, r_scores, judge
):
    s_top_50 = topk_from_scores(s_scores, RERANK_TOP_K)
    rs = {did: r_scores.get(did, 0.0) for did in s_top_50}
    rerank_only = merge_with_tail(
        [d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)], s_scores
    )
    fuse_06 = fuse_at_w(t_scores, r_scores, s_scores, 0.6)

    bisect_rank, bd = await run_bisect_thirds(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )
    h_old, hod = await run_hybrid_old_gate(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )
    h_new, hnd = await run_hybrid_new_gate(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )
    multi_rank, md = await run_multi_recipe_4(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )
    five_rank, fd = await run_fiveway(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )

    return {
        "rerank_only": hit_rank(rerank_only, gold_set),
        "fuse_T_R_w06": hit_rank(fuse_06, gold_set),
        "bisect_thirds": hit_rank(bisect_rank, gold_set),
        "hybrid_old_gate": hit_rank(h_old, gold_set),
        "hybrid_new_gate": hit_rank(h_new, gold_set),
        "multi_recipe_4": hit_rank(multi_rank, gold_set),
        "fiveway": hit_rank(five_rank, gold_set),
        "h_old_decision": hod["decision"],
        "h_new_decision": hnd["decision"],
        "multi_idx": md["recipe_idx"],
        "five_idx": fd["choice_idx"],
    }


# ---------- benchmark loaders ----------


async def run_temporal_bench(name, docs_path, queries_path, gold_path, reranker, judge):
    docs = [json.loads(l) for l in open(DATA_DIR / docs_path)]
    queries = [json.loads(l) for l in open(DATA_DIR / queries_path)]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / gold_path)]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(f"\n=== {name}: {len(docs)} docs, {len(queries)} queries ===")
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", f"v7l-{name}")
    q_ext = await run_v2_extract(q_items, f"{name}-queries", f"v7l-{name}")
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
    doc_text = {d["doc_id"]: d["text"] for d in docs}
    doc_dates = {d["doc_id"]: d["ref_time"][:10] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}
    qids = [q["query_id"] for q in queries]
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    lat_db = ROOT / "cache" / "force" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)
    per_q_l = {
        qid: lattice_retrieve_multi(lat, q_ext.get(qid, []), down_levels=1)[0]
        for qid in qids
    }

    per_q_t = {
        qid: make_t_scores(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            per_q_l.get(qid, {}),
        )
        for qid in qids
    }

    print("  reranking + hybrid optimizers...")
    per_q_r = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        t_top = topk_from_scores(per_q_t[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + t_top))[:RERANK_TOP_K]
        per_q_r[qid] = await rerank_topk(
            reranker, q_text[qid], union, doc_text, len(union)
        )

    results = []
    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue
        r = await evaluate_query(
            qid,
            q_text[qid],
            doc_text,
            doc_dates,
            gold_set,
            per_q_t[qid],
            per_q_s[qid],
            per_q_r[qid],
            judge,
        )
        results.append(r)
    return aggregate(results, name)


async def run_lme_bench(judge, reranker):
    NON_TEMPORAL_TYPES = {
        "single-session-preference",
        "single-session-user",
        "single-session-assistant",
        "knowledge-update",
        "multi-session",
    }
    data = json.load(
        open(ROOT.parent / "associative_recall" / "data" / "longmemeval_s_50q.json")
    )
    non_temp = [q for q in data if q["question_type"] in NON_TEMPORAL_TYPES][:10]
    print(f"\n=== longmemeval (non-temp): {len(non_temp)} queries ===")
    results = []
    for q in non_temp:
        qid = q["question_id"]
        q_text = q["question"]
        q_date = q["question_date"].split(" ")[0].replace("/", "-")
        gold_ids = q["answer_session_ids"]
        gold_set = set(gold_ids)
        sessions_dict = {
            sid: " ".join(t.get("content", "") for t in sess)
            for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"])
        }
        session_dates = {
            sid: d.split(" ")[0].replace("/", "-")
            for sid, d in zip(q["haystack_session_ids"], q["haystack_dates"])
        }
        doc_ids = list(sessions_dict.keys())
        doc_text = sessions_dict
        doc_dates = dict(session_dates)

        doc_items = [
            (did, doc_text[did], parse_iso(session_dates.get(did, q_date)))
            for did in doc_ids
        ]
        q_items = [(qid, q_text, parse_iso(q_date))]
        cache_label = f"lme-q-{qid}"
        doc_ext = await run_v2_extract(doc_items, cache_label + "-docs", cache_label)
        q_ext = await run_v2_extract(q_items, cache_label + "-queries", cache_label)
        doc_mem = build_memory(doc_ext)
        q_mem = build_memory(q_ext)
        for did in doc_ids:
            doc_mem.setdefault(
                did,
                {
                    "intervals": [],
                    "axes_merged": {
                        a: AxisDistribution(axis=a, values={}, informative=False)
                        for a in AXES
                    },
                    "multi_tags": set(),
                },
            )

        doc_embs_arr = await embed_all([doc_text[did] for did in doc_ids])
        q_embs_arr = await embed_all([q_text])
        doc_embs = {did: doc_embs_arr[i] for i, did in enumerate(doc_ids)}
        q_embs = {qid: q_embs_arr[0]}
        s_scores = rank_semantic(qid, q_embs, doc_embs)

        lat_db = ROOT / "cache" / "force_lme" / f"lat_{qid}.sqlite"
        lat_db.parent.mkdir(parents=True, exist_ok=True)
        if lat_db.exists():
            lat_db.unlink()
        lat = LatticeStore(str(lat_db))
        for did, tes in doc_ext.items():
            for te in tes:
                ts = lattice_tags_for_expression(te)
                lat.insert(did, ts.absolute, ts.cyclical)
        l_scores, _ = lattice_retrieve_multi(lat, q_ext.get(qid, []), down_levels=1)
        t_scores = make_t_scores(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            l_scores,
        )

        s_top = topk_from_scores(s_scores, RERANK_TOP_K)
        t_top = topk_from_scores(t_scores, RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + t_top))[:RERANK_TOP_K]
        r_scores = await rerank_topk(reranker, q_text, union, doc_text, len(union))

        r = await evaluate_query(
            qid,
            q_text,
            doc_text,
            doc_dates,
            gold_set,
            t_scores,
            s_scores,
            r_scores,
            judge,
        )
        results.append(r)
    return aggregate(results, "longmemeval (non-temp)")


def aggregate(results, label):
    print(f"\n=== {label} ===")
    variants = [
        "rerank_only",
        "fuse_T_R_w06",
        "bisect_thirds",
        "hybrid_old_gate",
        "hybrid_new_gate",
        "multi_recipe_4",
        "fiveway",
    ]
    n = len(results)
    for var in variants:
        ranks = [r[var] for r in results]
        r1 = sum(1 for x in ranks if x is not None and x <= 1)
        r5 = sum(1 for x in ranks if x is not None and x <= 5)
        mrr = sum(1.0 / x for x in ranks if x is not None) / n if n else 0
        print(
            f"  {var:18} R@1={r1}/{n} ({r1 / n:.3f})  R@5={r5}/{n} ({r5 / n:.3f})  MRR={mrr:.3f}"
        )
    from collections import Counter

    h_old_dec = Counter(r["h_old_decision"] for r in results)
    h_new_dec = Counter(r["h_new_decision"] for r in results)
    multi_idx = Counter(r["multi_idx"] for r in results)
    five_idx = Counter(r["five_idx"] for r in results)
    print(f"  h_old_gate decisions: {dict(h_old_dec)}")
    print(f"  h_new_gate decisions: {dict(h_new_dec)}")
    print(
        f"  multi_recipe_4 picks (0=pure_S, 1=rerank, 2=fuse04, 3=fuse06): {dict(multi_idx)}"
    )
    print(
        f"  fiveway picks (0=rerank, 1-4=fuse@{{0.0, 0.233, 0.467, 0.7}}): {dict(five_idx)}"
    )
    return {"results": results}


async def main():
    print("Loading cross-encoder...")
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

    benches = [
        (
            "hard_bench",
            "hard_bench_docs.jsonl",
            "hard_bench_queries.jsonl",
            "hard_bench_gold.jsonl",
        ),
        (
            "mixed_cue",
            "mixed_cue_docs.jsonl",
            "mixed_cue_queries.jsonl",
            "mixed_cue_gold.jsonl",
        ),
        (
            "dense_cluster",
            "dense_cluster_docs.jsonl",
            "dense_cluster_queries.jsonl",
            "dense_cluster_gold.jsonl",
        ),
        (
            "tempreason_small",
            "real_benchmark_small_docs.jsonl",
            "real_benchmark_small_queries.jsonl",
            "real_benchmark_small_gold.jsonl",
        ),
    ]
    out = {}
    for name, dp, qp, gp in benches:
        try:
            out[name] = await run_temporal_bench(name, dp, qp, gp, reranker, judge)
        except Exception as e:
            print(f"  [{name}] failed: {e}")
            out[name] = {"error": str(e)}
    out["longmemeval_nontemp"] = await run_lme_bench(judge, reranker)

    judge.save()
    out_path = ROOT / "results" / "hybrid_pickers.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"LLM calls: {judge.calls}")


if __name__ == "__main__":
    asyncio.run(main())
