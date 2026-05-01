"""Set-picking optimizers under the new framework: chrono-sorted unordered
sets with date labels, 1-token output (no CoT), explicit "set" framing.

Tests four set-picking designs:

  gate            n=2: rerank_only vs fuse_T_R(w=0.4). Single call. Prior winner.
  twoptr [0,0.7]  n=2 at endpoints; loser pointer moves toward winner by step.
                  4 rounds → final = winning side's last position.
  bisect_thirds   n=2 at lo+L/3, lo+2L/3; drop opposite third. 4 rounds.
  fourway         n=4 at fixed weights {0.0, 0.233, 0.467, 0.7}. Single call.
                  Final = chosen weight.

Format applied to ALL designs:
  - Each candidate set is shown as a bulleted UNORDERED collection
  - Sorted chronologically by doc date (date label in brackets)
  - Prompt frames as "set" not "ranked list"
  - 1-token output (A/B/C/D), max_tokens=8

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
from v7l_ts_blind_eval import BlindJudge, _key

PICK_PROMPT = """Pick the candidate SET whose collection of documents better answers the user's query.

Each set is an UNORDERED pool — documents are listed chronologically (by date), NOT by relevance. Order is not signal. Judge each set as a whole: does its pool contain the information needed to answer the question?

If the query has a temporal anchor (year, quarter, month, date range): prefer the set with more docs inside that period. Docs outside the period are distractors regardless of topical match.
If the query has NO temporal anchor: ignore dates and judge by topical/entity match (right person, right event-type).

QUERY: {query}

{sets}

Output exactly one letter from {{{choices}}} — the better set. No commentary.
"""


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
    """Pick best of N sets. Returns index into input sets (after un-shuffling)."""
    n = len(sets)
    labels = ["A", "B", "C", "D", "E", "F"][:n]
    rng = random.Random(rng_seed)
    order = list(range(n))
    rng.shuffle(order)
    shuffled = [sets[i] for i in order]
    formatted = "\n\n".join(_format_set(labels[i], s) for i, s in enumerate(shuffled))
    choices = ", ".join(labels)
    prompt = PICK_PROMPT.format(query=query, sets=formatted, choices=choices)
    k = _key(f"setpick_{n}", prompt)
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


def get_set_at_w(t_scores, r_scores, s_scores, w_T, doc_text, doc_dates):
    ranked = fuse_at_w(t_scores, r_scores, s_scores, w_T)
    return [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in ranked[:5]]


def hit_rank(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


# ---------- gate: rerank_only vs fuse_T_R(w=0.4) ----------


async def run_gate(
    qid, query_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
):
    s_top_50 = topk_from_scores(s_scores, RERANK_TOP_K)
    rs = {did: r_scores.get(did, 0.0) for did in s_top_50}
    rerank_only = merge_with_tail(
        [d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)], s_scores
    )
    fuse = fuse_at_w(t_scores, r_scores, s_scores, 0.4)
    s_rerank = [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in rerank_only[:5]]
    s_fuse = [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in fuse[:5]]
    seed = hash((qid, "gate")) & 0xFFFFFFFF
    idx = await pick_n(judge, query_text, [s_rerank, s_fuse], seed)
    return rerank_only if idx == 0 else fuse


# ---------- twoptr [0, 0.7] ----------


async def run_twoptr(
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
    lo, hi = lo_init, hi_init
    last_winning_w = (lo + hi) / 2
    history = []
    for r in range(max_rounds):
        if hi - lo < 0.01:
            break
        s_lo = get_set_at_w(t_scores, r_scores, s_scores, lo, doc_text, doc_dates)
        s_hi = get_set_at_w(t_scores, r_scores, s_scores, hi, doc_text, doc_dates)
        ids_lo = tuple(d for d, _, _ in s_lo)
        ids_hi = tuple(d for d, _, _ in s_hi)
        if ids_lo == ids_hi:
            history.append({"round": r + 1, "skipped": "identical"})
            break
        seed = hash((qid, "twoptr_new", r, lo, hi)) & 0xFFFFFFFF
        idx = await pick_n(judge, query_text, [s_lo, s_hi], seed)
        if idx == 0:
            last_winning_w = lo
            hi = round(max(lo, hi - step), 4)
            history.append({"round": r + 1, "winner": "lo", "lo": lo, "hi": hi})
        else:
            last_winning_w = hi
            lo = round(min(hi, lo + step), 4)
            history.append({"round": r + 1, "winner": "hi", "lo": lo, "hi": hi})
    return fuse_at_w(t_scores, r_scores, s_scores, last_winning_w), {
        "final_w_T": last_winning_w,
        "history": history,
    }


# ---------- bisect_thirds_capped [0, 0.7] ----------


async def run_bisect_thirds(
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
):
    lo, hi = lo_init, hi_init
    last_picked_w = (lo + hi) / 2
    history = []
    for r in range(max_rounds):
        L = hi - lo
        c_left = lo + L / 3.0
        c_right = lo + 2.0 * L / 3.0
        s_left = get_set_at_w(t_scores, r_scores, s_scores, c_left, doc_text, doc_dates)
        s_right = get_set_at_w(
            t_scores, r_scores, s_scores, c_right, doc_text, doc_dates
        )
        ids_left = tuple(d for d, _, _ in s_left)
        ids_right = tuple(d for d, _, _ in s_right)
        if ids_left == ids_right:
            history.append({"round": r + 1, "skipped": "identical"})
            break
        seed = hash((qid, "bisect_thirds_new", r, lo, hi)) & 0xFFFFFFFF
        idx = await pick_n(judge, query_text, [s_left, s_right], seed)
        if idx == 0:
            last_picked_w = c_left
            hi = c_right
            history.append(
                {
                    "round": r + 1,
                    "winner": "left",
                    "lo": lo,
                    "hi": hi,
                    "picked": last_picked_w,
                }
            )
        else:
            last_picked_w = c_right
            lo = c_left
            history.append(
                {
                    "round": r + 1,
                    "winner": "right",
                    "lo": lo,
                    "hi": hi,
                    "picked": last_picked_w,
                }
            )
    return fuse_at_w(t_scores, r_scores, s_scores, last_picked_w), {
        "final_w_T": last_picked_w,
        "history": history,
    }


# ---------- 4-way pick at fixed weights ----------

FOURWAY_WEIGHTS = [0.0, 0.233, 0.467, 0.7]


async def run_fourway(
    qid, query_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
):
    sets = [
        get_set_at_w(t_scores, r_scores, s_scores, w, doc_text, doc_dates)
        for w in FOURWAY_WEIGHTS
    ]
    seed = hash((qid, "fourway")) & 0xFFFFFFFF
    idx = await pick_n(judge, query_text, sets, seed)
    chosen_w = FOURWAY_WEIGHTS[idx]
    return fuse_at_w(t_scores, r_scores, s_scores, chosen_w), {"final_w_T": chosen_w}


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

    gate_rank = await run_gate(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )
    twoptr_rank, td = await run_twoptr(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )
    bisect_rank, bd = await run_bisect_thirds(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )
    fourway_rank, fd = await run_fourway(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )

    return {
        "rerank_only": hit_rank(rerank_only, gold_set),
        "fuse_T_R_w06": hit_rank(fuse_06, gold_set),
        "gate": hit_rank(gate_rank, gold_set),
        "twoptr": hit_rank(twoptr_rank, gold_set),
        "bisect_thirds": hit_rank(bisect_rank, gold_set),
        "fourway": hit_rank(fourway_rank, gold_set),
        "twoptr_w": td["final_w_T"],
        "bisect_w": bd["final_w_T"],
        "fourway_w": fd["final_w_T"],
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

    print("  reranking + set-picker optimizers...")
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
        "gate",
        "twoptr",
        "bisect_thirds",
        "fourway",
    ]
    n = len(results)
    for var in variants:
        ranks = [r[var] for r in results]
        r1 = sum(1 for x in ranks if x is not None and x <= 1)
        r5 = sum(1 for x in ranks if x is not None and x <= 5)
        mrr = sum(1.0 / x for x in ranks if x is not None) / n if n else 0
        print(
            f"  {var:14} R@1={r1}/{n} ({r1 / n:.3f})  R@5={r5}/{n} ({r5 / n:.3f})  MRR={mrr:.3f}"
        )
    import statistics

    for key in ["twoptr_w", "bisect_w", "fourway_w"]:
        ws = [r[key] for r in results]
        print(
            f"  {key:14}: mean={statistics.mean(ws):.3f}  unique={sorted(set(round(x, 3) for x in ws))}"
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
    out_path = ROOT / "results" / "set_pickers_new_format.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"LLM calls: {judge.calls}")


if __name__ == "__main__":
    asyncio.run(main())
