"""CoT date-aware judge with extreme-start two-pointer.

Tests two ideas combined:
  1. Two-pointer over [0, 0.7] starting at the endpoints (max contrast).
     w_T=0   → rerank-only (topical winners).
     w_T=0.7 → T-dominated within useful range (preserves enough rerank signal
               to keep results topically grounded, unlike w_T=1.0).
  2. CoT judge: query + top-5 (with snippets) for each side.
     Prompt asks: identify any temporal anchor in query, evaluate
     alignment per side, pick. Reasoning allowed (200 tokens).

Hypothesis: when sets actually differ (extreme weights guarantee this) AND
the judge explicitly considers date alignment, the optimizer can pick
correctly per-distribution: hard_bench → high w_T, LME-nontemp → low w_T.

Compared against:
  - rerank_only (the safe default)
  - fuse_T_R w=0.6 (oracle for hard_bench)
  - gate (the current winner: rerank_only vs fuse_T_R, single LLM call)
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
from v7l_ts_blind_eval import BlindJudge, _key

COT_PROMPT = """You are picking which candidate set of documents better answers the user's query.

Each set is an UNORDERED collection of documents (presented in chronological order, NOT by relevance — order is not signal). Treat each set as a pool and ask: "does this pool contain the information that answers the question?"

QUERY: {query}

Candidate Set A:
{set1}

Candidate Set B:
{set2}

Step 1: Identify any temporal anchor in the query — a year, quarter (Q1-Q4), month, season, or specific date range. Note it.
Step 2: If a temporal anchor exists, check each set: how many of its docs fall inside the anchored period? Docs outside the period are distractors regardless of topical match. The set with more in-period docs wins.
Step 3: If NO temporal anchor exists, ignore dates and judge by topical/entity match: which set's docs more directly address the specific person, event-type, or object in the query?
Step 4: Do NOT favor a set just because its first-listed doc looks good — judge the COLLECTION.

Reason briefly in 1-3 sentences, then on a NEW line output exactly:
FINAL: A
or
FINAL: B
"""


def _format_set_chrono(docs_with_dates: list[tuple[str, str, str]]) -> str:
    """Format as bulleted unordered set, sorted chronologically by date.
    Each item: (doc_id, date_str, snippet). Dates may be empty string."""
    sorted_docs = sorted(docs_with_dates, key=lambda x: x[1] or "")
    lines = []
    for did, date_str, text in sorted_docs:
        snippet = text[:200].replace("\n", " ")
        date_label = f"[{date_str}]" if date_str else "[date unknown]"
        lines.append(f"  - {date_label} {snippet}")
    return "\n".join(lines)


async def cot_pick(
    judge: BlindJudge,
    query: str,
    set1: list[tuple[str, str, str]],
    set2: list[tuple[str, str, str]],
    rng_seed: int,
) -> tuple[int, str]:
    """CoT pick. set1/set2 are list of (doc_id, date_str, text).
    Returns (chosen_index_into_input, reasoning_text)."""
    rng = random.Random(rng_seed)
    order = [0, 1]
    rng.shuffle(order)
    shown = [set1, set2]
    s1 = shown[order[0]]
    s2 = shown[order[1]]
    prompt = COT_PROMPT.format(
        query=query, set1=_format_set_chrono(s1), set2=_format_set_chrono(s2)
    )
    k = _key("cot_pick", prompt)
    if k in judge.cache:
        raw = judge.cache[k]
    else:
        judge.calls += 1
        raw = await judge._llm(prompt, max_tokens=300)
        if raw:
            judge.cache[k] = raw
            judge._dirty = True
    # Parse FINAL: A or B
    m = re.search(r"FINAL\s*:?\s*([AB])", raw or "", re.IGNORECASE)
    if not m:
        return order[0], raw or ""
    letter = m.group(1).upper()
    v = 1 if letter == "A" else 2
    return order[v - 1], raw or ""


def get_top5_with_text(t_scores, r_scores, s_scores, w_T, doc_text, doc_dates):
    ranked = fuse_at_w(t_scores, r_scores, s_scores, w_T)
    return [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in ranked[:5]]


def hit_rank(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


async def run_twoptr_extreme_cot(
    qid,
    query_text,
    t_scores,
    r_scores,
    s_scores,
    doc_text,
    doc_dates,
    judge,
    lo_init=0.0,
    hi_init=0.7,
    max_rounds=4,
    step=0.175,
):
    """Two-pointer over [0, 0.7] starting at endpoints. CoT judge.
    step=0.175 means after 4 rounds the loser pointer can move 4*0.175=0.7 toward winner."""
    lo, hi = lo_init, hi_init
    last_winning_w = (lo + hi) / 2
    history = []
    for r in range(max_rounds):
        if hi - lo < 0.01:
            break
        s_lo = get_top5_with_text(t_scores, r_scores, s_scores, lo, doc_text, doc_dates)
        s_hi = get_top5_with_text(t_scores, r_scores, s_scores, hi, doc_text, doc_dates)
        # Skip if sets identical — pure noise pick
        ids_lo = tuple(d for d, _, _ in s_lo)
        ids_hi = tuple(d for d, _, _ in s_hi)
        if ids_lo == ids_hi:
            history.append({"round": r + 1, "lo": lo, "hi": hi, "skipped": "identical"})
            break  # no signal — stop; final = current midpoint
        seed = hash((qid, "twoptr_extreme_cot", r, lo, hi)) & 0xFFFFFFFF
        idx, _ = await cot_pick(judge, query_text, s_lo, s_hi, seed)
        if idx == 0:
            last_winning_w = lo
            hi = round(max(lo, hi - step), 4)
            history.append({"round": r + 1, "winner": "lo", "lo": lo, "hi": hi})
        else:
            last_winning_w = hi
            lo = round(min(hi, lo + step), 4)
            history.append({"round": r + 1, "winner": "hi", "lo": lo, "hi": hi})
    final_w = last_winning_w
    return fuse_at_w(t_scores, r_scores, s_scores, final_w), {
        "final_w_T": final_w,
        "history": history,
    }


async def run_gate_baseline(
    qid, query_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
):
    """Gate: pick rerank_only vs fuse_T_R(w=0.4). Comparison-based, single call."""
    s_top_50 = topk_from_scores(s_scores, RERANK_TOP_K)
    rs = {did: r_scores.get(did, 0.0) for did in s_top_50}
    rerank_only = merge_with_tail(
        [d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)], s_scores
    )
    fuse = fuse_at_w(t_scores, r_scores, s_scores, 0.4)
    s1 = [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in rerank_only[:5]]
    s2 = [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in fuse[:5]]
    seed = hash((qid, "gate_cot")) & 0xFFFFFFFF
    idx, _ = await cot_pick(judge, query_text, s1, s2, seed)
    return rerank_only if idx == 0 else fuse


async def evaluate_query(
    qid, q_text, doc_text, doc_dates, gold_set, t_scores, s_scores, r_scores, judge
):
    s_top_50 = topk_from_scores(s_scores, RERANK_TOP_K)
    rs = {did: r_scores.get(did, 0.0) for did in s_top_50}
    rerank_only = merge_with_tail(
        [d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)], s_scores
    )
    fuse_06 = fuse_at_w(t_scores, r_scores, s_scores, 0.6)

    twoptr_rank, td = await run_twoptr_extreme_cot(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )
    gate_rank = await run_gate_baseline(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )

    return {
        "rerank_only": hit_rank(rerank_only, gold_set),
        "fuse_T_R_w06": hit_rank(fuse_06, gold_set),
        "twoptr_extreme_cot": hit_rank(twoptr_rank, gold_set),
        "gate_cot": hit_rank(gate_rank, gold_set),
        "twoptr_w": td["final_w_T"],
        "twoptr_history": td["history"],
    }


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

    print("  reranking + CoT optimizers...")
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

        doc_dates = dict(session_dates)
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
    variants = ["rerank_only", "fuse_T_R_w06", "twoptr_extreme_cot", "gate_cot"]
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

    tw = [r["twoptr_w"] for r in results]
    print(
        f"  twoptr_w final: mean={statistics.mean(tw):.3f}  min={min(tw):.3f}  max={max(tw):.3f}  "
        f"unique={sorted(set(round(x, 3) for x in tw))}"
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
        reranker,
        judge,
    )
    out["longmemeval_nontemp"] = await run_lme_bench(judge, reranker)

    judge.save()
    out_path = ROOT / "results" / "cot_judge_extreme.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"LLM calls: {judge.calls}")


if __name__ == "__main__":
    asyncio.run(main())
