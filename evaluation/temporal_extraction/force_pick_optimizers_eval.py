"""Force-pick optimizers (bisection + two-pointer) on hard_bench + LongMemEval.

Two designs, both with force-pick (no tie option, model must commit):

  bisect_pick: Bisection over weight interval.
    Round r: range [lo, hi]. Compare result sets at c_left=lo+0.25*(hi-lo)
    and c_right=lo+0.75*(hi-lo). Pick winner half. Final = last picked center.

  twoptr_force: Two-pointer with force-pick + winning-side convergence.
    Range [lo=0, hi=1]. Compare result sets AT lo and AT hi. Loser pointer
    moves toward winner by step=0.25. Final = winning side's last position
    (NOT midpoint).

Force-pick eliminates the tie-default bias. Bisection compares well-separated
weights at every round (large contrast). Two-pointer also has large contrast
on round 1 (extremes), then shrinks.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT.parents[1] / ".env")


import v7l_ts_blind_eval as base
from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
from lattice_store import LatticeStore
from rag_fusion import score_blend
from salience_eval import (  # type: ignore
    AXES,
    DATA_DIR,
    AxisDistribution,
    build_memory,
    embed_all,
    interval_pair_best,
    parse_iso,
    rank_semantic,
    run_v2_extract,
    tag_score,
)
from v7l_ts_blind_eval import (
    BlindJudge,
    _format_sets,
    _key,
)

T_ALPHA, T_GAMMA, T_DELTA = 0.20, 0.20, 0.60
RERANK_TOP_K = 50
MAX_ROUNDS_BISECT = 4
MAX_ROUNDS_TWOPTR = 4
TWOPTR_STEP = 0.25


PICK_FORCED = """Pick which candidate result set is a better match for the query. You MUST pick one — no ties, no abstain.

================ QUERY ================
{query}

============ CANDIDATES ============
{sets}

Output exactly one integer from {{1, 2}} — the better candidate.
No 0, no commentary, no explanation.
"""


def make_t_scores(q_mem, doc_mem, l_per_doc):
    qa = q_mem.get("axes_merged") or {}
    q_tags = q_mem.get("multi_tags") or set()
    q_ivs = q_mem.get("intervals") or []
    raw_iv = {
        did: interval_pair_best(q_ivs, b["intervals"]) for did, b in doc_mem.items()
    }
    max_iv = max(raw_iv.values()) if raw_iv else 0.0
    out = {}
    for did, b in doc_mem.items():
        iv_norm = raw_iv[did] / max_iv if max_iv > 0 else 0.0
        l_sc = l_per_doc.get(did, 0.0)
        out[did] = (
            T_ALPHA * iv_norm
            + T_GAMMA * tag_score(q_tags, b["multi_tags"])
            + T_DELTA * l_sc
        )
    return out


def fuse_at_w(t_scores, r_scores, s_scores_for_tail, w_T):
    fused = score_blend(
        {"T": t_scores, "R": r_scores},
        {"T": w_T, "R": 1.0 - w_T},
        top_k_per=40,
        dispersion_cv_ref=0.20,
    )
    primary = [d for d, _ in fused]
    seen = set(primary)
    tail = sorted(s_scores_for_tail.items(), key=lambda x: x[1], reverse=True)
    return primary + [d for d, _ in tail if d not in seen]


def topk_from_scores(scores, k):
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


async def rerank_topk(reranker, query_text, doc_ids, doc_text, k):
    cand_ids = doc_ids[:k]
    cand_texts = [doc_text.get(did, "")[:1000] for did in cand_ids]
    scores = await reranker.score(query_text, cand_texts)
    return {did: float(s) for did, s in zip(cand_ids, scores)}


def hit_rank(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


async def force_pick(
    judge: BlindJudge, query: str, cand_sets: list[list[str]], rng_seed: int
) -> int:
    """Force-pick: no tie option. Returns 0 or 1 (index into cand_sets)."""
    n = len(cand_sets)
    rng = random.Random(rng_seed)
    order = list(range(n))
    rng.shuffle(order)
    shuffled = [cand_sets[i] for i in order]
    prompt = PICK_FORCED.format(query=query, sets=_format_sets(shuffled))
    k = _key("force_pick", prompt)
    if k in judge.cache:
        raw = judge.cache[k]
    else:
        judge.calls += 1
        raw = await judge._llm(prompt, max_tokens=16)
        if raw:
            judge.cache[k] = raw
            judge._dirty = True
    import re

    m = re.search(r"\d+", raw or "")
    if not m:
        return 0  # default to first if model fails
    try:
        v = int(m.group(0))
    except ValueError:
        return 0
    if v not in (1, 2):
        return 0
    return order[v - 1]


def get_top5_at_w(t_scores, r_scores, s_scores, w_T, doc_text):
    ranked = fuse_at_w(t_scores, r_scores, s_scores, w_T)
    return ranked[:5], [doc_text.get(d, "")[: base.MAX_TEXT_LEN] for d in ranked[:5]]


async def run_bisect_pick(
    qid,
    query_text,
    t_scores,
    r_scores,
    s_scores,
    doc_text,
    judge,
    max_rounds=MAX_ROUNDS_BISECT,
):
    lo, hi = 0.0, 1.0
    last_picked_w = 0.5
    history = []
    for r in range(max_rounds):
        c_left = lo + 0.25 * (hi - lo)
        c_right = lo + 0.75 * (hi - lo)
        _, top5_l = get_top5_at_w(t_scores, r_scores, s_scores, c_left, doc_text)
        _, top5_r = get_top5_at_w(t_scores, r_scores, s_scores, c_right, doc_text)
        seed = hash((qid, "bisect", r, lo, hi)) & 0xFFFFFFFF
        idx = await force_pick(judge, query_text, [top5_l, top5_r], seed)
        if idx == 0:
            last_picked_w = c_left
            mid = (lo + hi) / 2
            hi = mid
        else:
            last_picked_w = c_right
            mid = (lo + hi) / 2
            lo = mid
        history.append({"round": r + 1, "lo": lo, "hi": hi, "picked": last_picked_w})
    final_w = last_picked_w
    return fuse_at_w(t_scores, r_scores, s_scores, final_w), {
        "final_w_T": final_w,
        "history": history,
    }


async def run_bisect_thirds_capped(
    qid,
    query_text,
    t_scores,
    r_scores,
    s_scores,
    doc_text,
    judge,
    max_rounds=4,
    lo_init=0.0,
    hi_init=0.7,
):
    """Thirds bisection over capped range [0, 0.7]. Each round: compare at 1/3 and 2/3
    of current range. Drop 1/3 farthest from winner. Range shrinks by 1/3 per round."""
    lo, hi = lo_init, hi_init
    last_picked_w = (lo + hi) / 2
    history = []
    for r in range(max_rounds):
        L = hi - lo
        c_left = lo + L / 3.0
        c_right = lo + 2.0 * L / 3.0
        _, top5_l = get_top5_at_w(t_scores, r_scores, s_scores, c_left, doc_text)
        _, top5_r = get_top5_at_w(t_scores, r_scores, s_scores, c_right, doc_text)
        seed = hash((qid, "bisect_thirds_capped", r, lo, hi)) & 0xFFFFFFFF
        idx = await force_pick(judge, query_text, [top5_l, top5_r], seed)
        if idx == 0:
            last_picked_w = c_left
            hi = c_right  # drop right third
        else:
            last_picked_w = c_right
            lo = c_left  # drop left third
        history.append({"round": r + 1, "lo": lo, "hi": hi, "picked": last_picked_w})
    final_w = last_picked_w
    return fuse_at_w(t_scores, r_scores, s_scores, final_w), {
        "final_w_T": final_w,
        "history": history,
    }


async def run_twoptr_force(
    qid,
    query_text,
    t_scores,
    r_scores,
    s_scores,
    doc_text,
    judge,
    max_rounds=MAX_ROUNDS_TWOPTR,
    step=TWOPTR_STEP,
):
    lo, hi = 0.0, 1.0
    last_winning_side = None
    last_winning_w = 0.5
    history = []
    for r in range(max_rounds):
        if hi - lo < step:
            break
        _, top5_lo = get_top5_at_w(t_scores, r_scores, s_scores, lo, doc_text)
        _, top5_hi = get_top5_at_w(t_scores, r_scores, s_scores, hi, doc_text)
        seed = hash((qid, "twoptr", r, lo, hi)) & 0xFFFFFFFF
        idx = await force_pick(judge, query_text, [top5_lo, top5_hi], seed)
        if idx == 0:
            # lo wins → hi moves toward lo
            last_winning_side = "lo"
            last_winning_w = lo
            hi = round(max(lo, hi - step), 4)
        else:
            last_winning_side = "hi"
            last_winning_w = hi
            lo = round(min(hi, lo + step), 4)
        history.append(
            {"round": r + 1, "lo": lo, "hi": hi, "winner": last_winning_side}
        )
    # Final: winning side's last position (NOT midpoint)
    final_w = last_winning_w if last_winning_w is not None else 0.5
    return fuse_at_w(t_scores, r_scores, s_scores, final_w), {
        "final_w_T": final_w,
        "history": history,
    }


# Reuse from prior eval


def merge_with_tail(primary, tail_scores):
    seen = set(primary)
    tail = sorted(
        ((d, s) for d, s in tail_scores.items() if d not in seen),
        key=lambda x: x[1],
        reverse=True,
    )
    return primary + [d for d, _ in tail]


async def evaluate_query(
    qid, q_text, doc_text, gold_set, t_scores, s_scores, r_scores, judge
):
    """Run all variants for one query."""
    s_top_50 = topk_from_scores(s_scores, RERANK_TOP_K)
    rs = {did: r_scores.get(did, 0.0) for did in s_top_50}
    rerank_only = merge_with_tail(
        [d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)], s_scores
    )

    fuse_04 = fuse_at_w(t_scores, r_scores, s_scores, 0.4)
    fuse_06 = fuse_at_w(t_scores, r_scores, s_scores, 0.6)

    bisect_rank, bd = await run_bisect_pick(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, judge
    )
    twoptr_rank, td = await run_twoptr_force(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, judge
    )
    bisect_capped_rank, bcd = await run_bisect_thirds_capped(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, judge
    )

    return {
        "rerank_only": hit_rank(rerank_only, gold_set),
        "fuse_T_R_w04": hit_rank(fuse_04, gold_set),
        "fuse_T_R_w06": hit_rank(fuse_06, gold_set),
        "bisect_pick": hit_rank(bisect_rank, gold_set),
        "twoptr_force": hit_rank(twoptr_rank, gold_set),
        "bisect_thirds_capped": hit_rank(bisect_capped_rank, gold_set),
        "bisect_w": bd["final_w_T"],
        "twoptr_w": td["final_w_T"],
        "bisect_capped_w": bcd["final_w_T"],
    }


async def run_temporal_bench(
    name, docs_path, queries_path, gold_path, cache_doc, cache_q, reranker, judge
):
    docs = [json.loads(l) for l in open(DATA_DIR / docs_path)]
    queries = [json.loads(l) for l in open(DATA_DIR / queries_path)]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / gold_path)]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(f"\n=== {name}: {len(docs)} docs, {len(queries)} queries ===")
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", cache_doc)
    q_ext = await run_v2_extract(q_items, f"{name}-queries", cache_q)
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
    q_text = {q["query_id"]: q["text"] for q in queries}
    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts)
    q_embs_arr = await embed_all(q_texts)
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

    print("  reranking + force-pick optimizers...")
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
            qid, q_text, doc_text, gold_set, t_scores, s_scores, r_scores, judge
        )
        results.append(r)
    return aggregate(results, "longmemeval (non-temp)")


def aggregate(results, label):
    print(f"\n=== {label} ===")
    variants = [
        "rerank_only",
        "fuse_T_R_w04",
        "fuse_T_R_w06",
        "bisect_pick",
        "twoptr_force",
        "bisect_thirds_capped",
    ]
    n = len(results)
    for var in variants:
        ranks = [r[var] for r in results]
        r1 = sum(1 for x in ranks if x is not None and x <= 1)
        r5 = sum(1 for x in ranks if x is not None and x <= 5)
        mrr = sum(1.0 / x for x in ranks if x is not None) / n if n else 0
        print(
            f"  {var:22} R@1={r1}/{n} ({r1 / n:.3f})  R@5={r5}/{n} ({r5 / n:.3f})  MRR={mrr:.3f}"
        )

    import statistics

    bw = [r["bisect_w"] for r in results]
    tw = [r["twoptr_w"] for r in results]
    bcw = [r["bisect_capped_w"] for r in results]
    print(
        f"  bisect_w (uncapped) : mean={statistics.mean(bw):.3f}  min={min(bw):.3f}  max={max(bw):.3f}"
    )
    print(
        f"  twoptr_w  : mean={statistics.mean(tw):.3f}  min={min(tw):.3f}  max={max(tw):.3f}"
    )
    print(
        f"  bisect_capped_w (0-0.7): mean={statistics.mean(bcw):.3f}  min={min(bcw):.3f}  max={max(bcw):.3f}"
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

    out = {}
    out["hard_bench"] = await run_temporal_bench(
        "hard_bench",
        "hard_bench_docs.jsonl",
        "hard_bench_queries.jsonl",
        "hard_bench_gold.jsonl",
        "v7l-hard_bench",
        "v7l-hard_bench",
        reranker,
        judge,
    )
    out["longmemeval_nontemp"] = await run_lme_bench(judge, reranker)

    judge.save()
    out_path = ROOT / "results" / "force_pick_optimizers.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"LLM calls: {judge.calls}")


if __name__ == "__main__":
    asyncio.run(main())
