"""Compare judge models on goldilocks: gate (1-call) and bisect_score (4-call).

Tests how cheaper LLMs handle the set-picking task. If a smaller model can
match gpt-5-mini's performance, the production cost drops dramatically.

Models tested (in order of cheapness):
  gpt-5-nano   (cheapest)
  gpt-5-mini   (current)

For each model: gate + bisect_score + multi_recipe_4 (4-way recipe pick).

Goldilocks is the right benchmark for model differentiation because it's
adversarial — middle weights matter, so the judge has to make subtle
discriminations the smaller model might fail at.
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
from openai import AsyncOpenAI
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
from set_pickers_eval import PICK_PROMPT, _format_set
from v7l_ts_blind_eval import _key

PER_CALL_TIMEOUT_S = 30.0
CONCURRENCY = 8


class ParamJudge:
    """Judge with configurable model name."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.client = AsyncOpenAI(timeout=PER_CALL_TIMEOUT_S, max_retries=1)
        self.sem = asyncio.Semaphore(CONCURRENCY)
        self.cache: dict = {}  # per-model cache (memory only — won't persist)
        self.calls = 0
        self.failed = 0

    async def _llm(self, prompt: str, max_tokens: int = 512) -> str:
        async with self.sem:
            try:
                resp = await asyncio.wait_for(
                    self.client.responses.create(
                        model=self.model,
                        input=prompt,
                        max_output_tokens=max_tokens,
                    ),
                    timeout=PER_CALL_TIMEOUT_S,
                )
            except Exception:
                self.failed += 1
                return ""
        return resp.output_text or ""


async def pick_n_with_judge(judge: ParamJudge, query: str, sets, rng_seed: int) -> int:
    n = len(sets)
    labels = ["A", "B", "C", "D", "E", "F"][:n]
    rng = random.Random(rng_seed)
    order = list(range(n))
    rng.shuffle(order)
    shuffled = [sets[i] for i in order]
    formatted = "\n\n".join(_format_set(labels[i], s) for i, s in enumerate(shuffled))
    choices = ", ".join(labels)
    prompt = PICK_PROMPT.format(query=query, sets=formatted, choices=choices)
    k = _key(f"setpick_{n}_{judge.model}", prompt)
    if k in judge.cache:
        raw = judge.cache[k]
    else:
        judge.calls += 1
        raw = await judge._llm(prompt, max_tokens=512)
        if raw:
            judge.cache[k] = raw
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


async def run_gate(
    qid, query_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
):
    s_top_50 = topk_from_scores(s_scores, RERANK_TOP_K)
    rs = {did: r_scores.get(did, 0.0) for did in s_top_50}
    rerank_only = merge_with_tail(
        [d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)], s_scores
    )
    fuse = fuse_at_w(t_scores, r_scores, s_scores, 0.4)
    s_r = [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in rerank_only[:5]]
    s_f = [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in fuse[:5]]
    seed = hash((qid, "gate")) & 0xFFFFFFFF
    idx = await pick_n_with_judge(judge, query_text, [s_r, s_f], seed)
    return rerank_only if idx == 0 else fuse


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
            break
        seed = hash((qid, "bisect", r, lo, hi)) & 0xFFFFFFFF
        idx = await pick_n_with_judge(judge, query_text, [s_left, s_right], seed)
        if idx == 0:
            last_picked_w = c_left
            hi = c_right
        else:
            last_picked_w = c_right
            lo = c_left
    return fuse_at_w(t_scores, r_scores, s_scores, last_picked_w), last_picked_w


async def run_multi_recipe_4(
    qid, query_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
):
    pure_s_rank = sorted(s_scores.items(), key=lambda x: x[1], reverse=True)
    pure_s = [d for d, _ in pure_s_rank]
    s_top_50 = topk_from_scores(s_scores, RERANK_TOP_K)
    rs = {did: r_scores.get(did, 0.0) for did in s_top_50}
    rerank_only = merge_with_tail(
        [d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)], s_scores
    )
    fuse_02 = fuse_at_w(t_scores, r_scores, s_scores, 0.2)
    fuse_04 = fuse_at_w(t_scores, r_scores, s_scores, 0.4)
    fuse_06 = fuse_at_w(t_scores, r_scores, s_scores, 0.6)
    recipes = [rerank_only, fuse_02, fuse_04, fuse_06]
    sets = [
        [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in r[:5]] for r in recipes
    ]
    seed = hash((qid, "multi4_v2")) & 0xFFFFFFFF
    idx = await pick_n_with_judge(judge, query_text, sets, seed)
    return recipes[idx]


async def evaluate_query(
    qid, q_text, doc_text, doc_dates, gold_set, t_scores, s_scores, r_scores, judge
):
    s_top_50 = topk_from_scores(s_scores, RERANK_TOP_K)
    rs = {did: r_scores.get(did, 0.0) for did in s_top_50}
    rerank_only = merge_with_tail(
        [d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)], s_scores
    )
    fuse_06 = fuse_at_w(t_scores, r_scores, s_scores, 0.6)
    fuse_02 = fuse_at_w(t_scores, r_scores, s_scores, 0.2)

    gate_rank = await run_gate(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )
    bisect_rank, bw = await run_bisect_thirds(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )
    multi_rank = await run_multi_recipe_4(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )

    return {
        "rerank_only": hit_rank(rerank_only, gold_set),
        "fuse_T_R_w02": hit_rank(fuse_02, gold_set),
        "fuse_T_R_w06": hit_rank(fuse_06, gold_set),
        "gate": hit_rank(gate_rank, gold_set),
        "bisect_score": hit_rank(bisect_rank, gold_set),
        "multi_recipe_4": hit_rank(multi_rank, gold_set),
        "bisect_w": bw,
    }


async def run_goldilocks(name, judge, reranker, suffix=""):
    docs = [json.loads(l) for l in open(DATA_DIR / f"goldilocks{suffix}_docs.jsonl")]
    queries = [
        json.loads(l) for l in open(DATA_DIR / f"goldilocks{suffix}_queries.jsonl")
    ]
    gold_rows = [
        json.loads(l) for l in open(DATA_DIR / f"goldilocks{suffix}_gold.jsonl")
    ]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(
        f"\n=== goldilocks{suffix} @ {judge.model}: {len(docs)} docs, {len(queries)} queries ==="
    )
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(
        doc_items, f"goldilocks{suffix}-docs", f"v7l-goldilocks{suffix}"
    )
    q_ext = await run_v2_extract(
        q_items, f"goldilocks{suffix}-queries", f"v7l-goldilocks{suffix}"
    )
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

    lat_db = (
        ROOT / "cache" / "modelcmp" / f"lat_goldilocks{suffix}_{judge.model}.sqlite"
    )
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

    print("  reranking...")
    per_q_r = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        t_top = topk_from_scores(per_q_t[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + t_top))[:RERANK_TOP_K]
        per_q_r[qid] = await rerank_topk(
            reranker, q_text[qid], union, doc_text, len(union)
        )

    print(f"  judging with {judge.model}...")
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

    print(f"\n=== goldilocks{suffix} @ {judge.model} ===")
    variants = [
        "rerank_only",
        "fuse_T_R_w02",
        "fuse_T_R_w06",
        "gate",
        "bisect_score",
        "multi_recipe_4",
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
    import statistics

    bws = [r["bisect_w"] for r in results]
    print(
        f"  bisect_w mean={statistics.mean(bws):.3f} unique={sorted(set(round(x, 3) for x in bws))}"
    )
    print(f"  Calls: {judge.calls}  Failed: {judge.failed}")
    return {"results": results, "calls": judge.calls, "failed": judge.failed}


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

    models = ["gpt-5-nano", "gpt-5-mini", "gpt-5.4-nano", "gpt-5.4-mini"]
    out = {}
    for suffix in ["", "_v2"]:
        for m in models:
            key = f"{m}{suffix}"
            try:
                judge = ParamJudge(m)
                out[key] = await run_goldilocks(
                    "goldilocks", judge, reranker, suffix=suffix
                )
            except Exception as e:
                print(f"  [{key}] failed: {type(e).__name__}: {str(e)[:200]}")
                out[key] = {"error": str(e)[:200]}

    out_path = ROOT / "results" / "model_comparison_goldilocks.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
