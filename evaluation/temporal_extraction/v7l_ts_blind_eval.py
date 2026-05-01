"""V7L (T+S only, single dial) + blind LLM-in-loop weight tuning.

Single dial: w_T ∈ [0, 1], w_S = 1 - w_T. No L channel (shown dispensable in
channel_contribution_eval.py).

Six designs, all with weight numbers HIDDEN from the model:

  blind_3way        Pick best of {w-step, w, w+step} blinded.
  blind_pair        Pick better of {prev_set, curr_set} blinded.
  blind_3way_anch   blind_3way + pure_S (w=0) and pure_T (w=1) anchor sets.
  blind_pair_anch   blind_pair + same two anchors.
  two_pointer       lo, hi pointers; loser moves toward winner by fixed step.
  blind_dir_pair    Direct UP/DOWN/STOP vote after seeing prev+curr blinded
                    (model knows it's tuning a temporal-vs-topic emphasis dial,
                    but never sees a weight number).

Compares against:
  V7L_TS (w_T=0.2)  best fixed across these benchmarks (per channel sweep).
  V7L_TS oracle     per-query best w_T over a 11-point grid.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT.parents[1] / ".env")

import extractor_common
import numpy as np
from openai import AsyncOpenAI

_orig_call = extractor_common.BaseImprovedExtractor._call


async def _patched_call(self, *args, **kwargs):
    original_create = self.client.chat.completions.create

    async def patched_create(**call_kwargs):
        model = call_kwargs.get("model", "")
        if isinstance(model, str) and model.startswith("gpt-5"):
            call_kwargs["reasoning_effort"] = "minimal"
        return await original_create(**call_kwargs)

    self.client.chat.completions.create = patched_create
    try:
        return await _orig_call(self, *args, **kwargs)
    finally:
        self.client.chat.completions.create = original_create


extractor_common.BaseImprovedExtractor._call = _patched_call

from rag_fusion import score_blend
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
from salience_eval import (
    rank_t as rank_multi_axis_t,
)

MODEL = "gpt-5-mini"
CACHE_DIR = ROOT / "cache" / "ts_blind"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_cache.json"

PER_CALL_TIMEOUT_S = 30.0
CONCURRENCY = 8
MAX_TEXT_LEN = 180
MAX_ROUNDS = 4

STEP_3WAY = 0.2
STEP_PAIR = 0.15
STEP_DIR = 0.15
STEP_TWOPTR = 0.1
VISITED_TOL = 0.025

ANCHOR_S_W = 0.0  # pure semantic
ANCHOR_T_W = 1.0  # pure temporal


def _load_cache() -> dict[str, str]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    tmp = CACHE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache))
    tmp.replace(CACHE_FILE)


def _key(stage: str, body: str) -> str:
    return hashlib.sha256(f"{stage}|{MODEL}|{body}".encode()).hexdigest()


PICK_PROMPT = """You are evaluating retrieval candidates for a query. Each CANDIDATE below is a top-5 list of documents produced by a different blend of topic-match and time-match signals. Your job is to pick the candidate whose top-5 best matches the query.

================ QUERY ================
{query}

============ CANDIDATES ============
{sets}

Output rules:
  - Output exactly one integer from {{ {choices} }} — the candidate number.
  - Output 0 if all candidates look equally good.
  - No commentary, just the number.
"""


PICK_WITH_REF_PROMPT = """You are evaluating retrieval candidates for a query. The retrieval system blends two signals: topic-match (semantic content) and time-match (date/interval). The blend weight varies — the goal is to find the weight that gives the best result for THIS query.

Below you will see two REFERENCE retrievals showing what each pure-extreme weighting produces. The references are CONTEXT ONLY — they help you tell whether a candidate looks too topic-leaning or too time-leaning. They are NOT eligible to be picked.

Then you will see CANDIDATES from blended weightings. These ARE the only options.

================ QUERY ================
{query}

============ REFERENCE A ============
This is what PURE-TOPICAL retrieval produces (date/time signal is OFF, only topic matters):
{ref_S}

============ REFERENCE B ============
This is what PURE-TEMPORAL retrieval produces (topic signal is OFF, only date/time matters):
{ref_T}

============ CANDIDATES ============
The candidates below are the actual options (selectable). Each was produced by a different blend of topic and time signals.

{sets}

Pick the CANDIDATE whose top-5 best matches the query (considering both topic relevance AND temporal context, since the query may want either or both). Use References A and B as anchors so you can tell whether a candidate is leaning too far one way.

Output rules:
  - Output exactly one integer from {{ {choices} }} — the candidate number.
  - Output 0 if all candidates look equally good.
  - Do NOT output A, B, or any reference label. References are not options.
  - No commentary, just the number.
"""


DIR_PROMPT = """A retrieval system tunes how it ranks documents for a query. It just made a tentative adjustment to its ranking and produced a CURRENT result set; the result set BEFORE that adjustment is shown as PREVIOUS.

Your job is to compare the two sets and tell whether the adjustment helped, hurt, or made no difference.

================ QUERY ================
{query}

============ PREVIOUS RESULT SET ============
{prev_set}

============ CURRENT RESULT SET ============
{cur_set}

Compare PREVIOUS to CURRENT against the query:
  CURRENT_BETTER  — CURRENT is a better top-5 match for the query than PREVIOUS.
  PREVIOUS_BETTER — PREVIOUS was a better match; CURRENT is worse.
  TIE             — both look about equally good (or equally bad) for this query.

Output exactly one of: CURRENT_BETTER, PREVIOUS_BETTER, TIE. No commentary."""


DIR_WITH_REF_PROMPT = """A retrieval system tunes how it ranks documents for a query. It just made a tentative adjustment to its ranking and produced a CURRENT result set; the result set BEFORE that adjustment is shown as PREVIOUS.

Two REFERENCE retrievals are also shown to help you calibrate: they are produced by the two extreme weightings of the system. References are CONTEXT ONLY — they are not options and are not being compared. They are there to help you read where PREVIOUS and CURRENT sit on the topical-vs-temporal spectrum.

================ QUERY ================
{query}

============ REFERENCE A (pure topical — date/time signal OFF) ============
{ref_S}

============ REFERENCE B (pure temporal — topic signal OFF) ============
{ref_T}

============ PREVIOUS RESULT SET ============
{prev_set}

============ CURRENT RESULT SET ============
{cur_set}

Compare PREVIOUS to CURRENT against the query (Use References A and B only as anchors for understanding what direction each result leans):
  CURRENT_BETTER  — CURRENT is a better top-5 match for the query than PREVIOUS.
  PREVIOUS_BETTER — PREVIOUS was a better match; CURRENT is worse.
  TIE             — both look about equally good (or equally bad) for this query.

Output exactly one of: CURRENT_BETTER, PREVIOUS_BETTER, TIE. No commentary."""


def _format_sets(sets: list[list[str]]) -> str:
    out_lines = []
    for i, s in enumerate(sets):
        out_lines.append(f"Set {i + 1}:")
        for j, t in enumerate(s):
            out_lines.append(f"  {j + 1}. {t[:MAX_TEXT_LEN]}")
        out_lines.append("")
    return "\n".join(out_lines)


def _format_one_set(s: list[str]) -> str:
    return "\n".join(f"  {j + 1}. {t[:MAX_TEXT_LEN]}" for j, t in enumerate(s))


class BlindJudge:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(timeout=PER_CALL_TIMEOUT_S, max_retries=1)
        self.sem = asyncio.Semaphore(CONCURRENCY)
        self.cache = _load_cache()
        self._dirty = False
        self.usage = {"input": 0, "output": 0}
        self.calls = 0
        self.failed = 0

    async def _llm(self, prompt: str, max_tokens: int = 64) -> str:
        async with self.sem:
            try:
                resp = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=max_tokens,
                        reasoning_effort="minimal",
                    ),
                    timeout=PER_CALL_TIMEOUT_S,
                )
            except Exception:
                self.failed += 1
                return ""
        u = resp.usage
        if u:
            self.usage["input"] += getattr(u, "prompt_tokens", 0) or 0
            self.usage["output"] += getattr(u, "completion_tokens", 0) or 0
        return resp.choices[0].message.content or ""

    async def pick_best(
        self,
        query: str,
        candidate_sets: list[list[str]],
        rng_seed: int,
        ref_S: list[str] | None = None,
        ref_T: list[str] | None = None,
    ) -> int:
        n = len(candidate_sets)
        rng = random.Random(rng_seed)
        order = list(range(n))
        rng.shuffle(order)
        shuffled_sets = [candidate_sets[i] for i in order]
        choices_str = ", ".join(str(i + 1) for i in range(n))
        if ref_S is not None and ref_T is not None:
            prompt = PICK_WITH_REF_PROMPT.format(
                query=query,
                ref_S=_format_one_set(ref_S),
                ref_T=_format_one_set(ref_T),
                sets=_format_sets(shuffled_sets),
                choices=choices_str,
            )
        else:
            prompt = PICK_PROMPT.format(
                query=query,
                sets=_format_sets(shuffled_sets),
                choices=choices_str,
            )
        k = _key("pick", prompt)
        if k in self.cache:
            raw = self.cache[k]
        else:
            self.calls += 1
            raw = await self._llm(prompt, max_tokens=64)
            if raw:
                self.cache[k] = raw
                self._dirty = True
        m = re.search(r"\d+", raw or "")
        if not m:
            return -1
        try:
            v = int(m.group(0))
        except ValueError:
            return -1
        if v == 0 or v < 1 or v > n:
            return -1
        return order[v - 1]

    async def vote_direction(
        self,
        query: str,
        prev_set: list[str],
        cur_set: list[str],
        rng_seed: int,
        ref_S: list[str] | None = None,
        ref_T: list[str] | None = None,
    ) -> str:
        # Note: prev/cur labels carry temporal meaning ("which way did the dial just move"),
        # so we do NOT shuffle them — order matters for direction inference.
        if ref_S is not None and ref_T is not None:
            prompt = DIR_WITH_REF_PROMPT.format(
                query=query,
                ref_S=_format_one_set(ref_S),
                ref_T=_format_one_set(ref_T),
                prev_set=_format_one_set(prev_set),
                cur_set=_format_one_set(cur_set),
            )
        else:
            prompt = DIR_PROMPT.format(
                query=query,
                prev_set=_format_one_set(prev_set),
                cur_set=_format_one_set(cur_set),
            )
        k = _key("direction", prompt)
        if k in self.cache:
            raw = self.cache[k]
        else:
            self.calls += 1
            raw = await self._llm(prompt, max_tokens=32)
            if raw:
                self.cache[k] = raw
                self._dirty = True
        u = (raw or "").strip().upper()
        # New comparison vocabulary: model judges quality, not direction.
        if "CURRENT_BETTER" in u:
            return "CURRENT_BETTER"
        if "PREVIOUS_BETTER" in u:
            return "PREVIOUS_BETTER"
        return "TIE"

    def save(self) -> None:
        if self._dirty:
            _save_cache(self.cache)
            self._dirty = False


def rank_blend_ts(t, s, w_T: float) -> list[str]:
    """T+S only blend. w_S = 1 - w_T. No L."""
    w_S = max(0.0, 1.0 - w_T)
    chans = {"T": t, "S": s}
    weights = {"T": w_T, "S": w_S}
    fused = score_blend(chans, weights, top_k_per=40, dispersion_cv_ref=0.20)
    ranked = [d for d, _ in fused]
    seen = set(ranked)
    tail = [
        d
        for d, _ in sorted(s.items(), key=lambda x: x[1], reverse=True)
        if d not in seen
    ]
    return ranked + tail


class RetrievalCache:
    def __init__(self, t, s, doc_text):
        self.t, self.s, self.doc_text = t, s, doc_text
        self._cache: dict[float, tuple[list[str], list[str]]] = {}

    def get(self, w_T: float) -> tuple[list[str], list[str]]:
        key = round(max(0.0, min(1.0, w_T)), 3)
        if key not in self._cache:
            ranked = rank_blend_ts(self.t, self.s, key)
            top5_text = [self.doc_text.get(d, "")[:MAX_TEXT_LEN] for d in ranked[:5]]
            self._cache[key] = (ranked, top5_text)
        return self._cache[key]


def _abab(visited: list[float], proposed: float) -> bool:
    if len(visited) < 3:
        return False
    return (
        abs(visited[-3] - visited[-1]) < VISITED_TOL
        and abs(proposed - visited[-2]) < VISITED_TOL
    )


async def run_3way(
    qid,
    query_text,
    cache,
    judge,
    with_references=False,
    max_rounds=MAX_ROUNDS,
    init_w=0.4,
):
    w = init_w
    visited = [w]
    trace = [{"round": 0, "w_T": w}]
    stopped = "max_rounds"
    ref_S = cache.get(ANCHOR_S_W)[1] if with_references else None
    ref_T = cache.get(ANCHOR_T_W)[1] if with_references else None
    for r in range(max_rounds):
        cands_w = [
            max(0.0, round(w - STEP_3WAY, 3)),
            w,
            min(1.0, round(w + STEP_3WAY, 3)),
        ]
        dedup_w = []
        for cw in cands_w:
            if not dedup_w or all(abs(cw - x) > VISITED_TOL for x in dedup_w):
                dedup_w.append(cw)
        cand_sets = [cache.get(cw)[1] for cw in dedup_w]
        seed = hash((qid, "3way", with_references, r, w)) & 0xFFFFFFFF
        idx = await judge.pick_best(
            query_text, cand_sets, rng_seed=seed, ref_S=ref_S, ref_T=ref_T
        )
        trace.append({"round": r + 1, "candidates": dedup_w, "picked_idx": idx})
        if idx < 0:
            stopped = "tie"
            break
        chosen_w = dedup_w[idx]
        if abs(chosen_w - w) < VISITED_TOL:
            stopped = "stay"
            break
        if _abab(visited, chosen_w):
            stopped = "abab_cycle"
            break
        w = chosen_w
        visited.append(w)
    return cache.get(w)[0], {
        "final_w_T": w,
        "rounds": len(trace) - 1,
        "stopped": stopped,
    }


async def run_pair(
    qid,
    query_text,
    cache,
    judge,
    with_references=False,
    max_rounds=MAX_ROUNDS,
    init_w=0.4,
    step=STEP_PAIR,
):
    w = init_w
    direction = -1.0  # bias downward initially (empirical optimum is low)
    visited = [w]
    trace = [{"round": 0, "w_T": w}]
    stopped = "max_rounds"
    ref_S = cache.get(ANCHOR_S_W)[1] if with_references else None
    ref_T = cache.get(ANCHOR_T_W)[1] if with_references else None
    for r in range(max_rounds):
        proposed = max(0.0, min(1.0, round(w + direction * step, 3)))
        if abs(proposed - w) < VISITED_TOL:
            direction = -direction
            proposed = max(0.0, min(1.0, round(w + direction * step, 3)))
            if abs(proposed - w) < VISITED_TOL:
                stopped = "boundary"
                break
        if _abab(visited, proposed):
            stopped = "abab_cycle"
            break
        cands_w = [w, proposed]
        cand_sets = [cache.get(cw)[1] for cw in cands_w]
        seed = hash((qid, "pair", with_references, r, w, proposed)) & 0xFFFFFFFF
        idx = await judge.pick_best(
            query_text, cand_sets, rng_seed=seed, ref_S=ref_S, ref_T=ref_T
        )
        trace.append(
            {
                "round": r + 1,
                "current": w,
                "proposed": proposed,
                "candidates": cands_w,
                "picked_idx": idx,
            }
        )
        if idx < 0:
            stopped = "tie"
            break
        chosen_w = cands_w[idx]
        if abs(chosen_w - w) < VISITED_TOL:
            direction = -direction
            continue
        w = chosen_w
        visited.append(w)
    return cache.get(w)[0], {
        "final_w_T": w,
        "rounds": len(trace) - 1,
        "stopped": stopped,
    }


async def run_two_pointer(
    qid,
    query_text,
    cache,
    judge,
    max_rounds=MAX_ROUNDS,
    lo_init=0.0,
    hi_init=1.0,
    step=STEP_TWOPTR,
):
    lo, hi = lo_init, hi_init
    trace = [{"round": 0, "lo": lo, "hi": hi}]
    stopped = "max_rounds"
    for r in range(max_rounds):
        if hi - lo < step:
            stopped = "converged"
            break
        cand_sets = [cache.get(lo)[1], cache.get(hi)[1]]
        seed = hash((qid, "twoptr", r, lo, hi)) & 0xFFFFFFFF
        idx = await judge.pick_best(query_text, cand_sets, rng_seed=seed)
        trace.append({"round": r + 1, "lo": lo, "hi": hi, "picked_idx": idx})
        if idx < 0:
            stopped = "tie"
            break
        if idx == 0:
            hi = round(max(lo, hi - step), 3)
        else:
            lo = round(min(hi, lo + step), 3)
    final_w = round((lo + hi) / 2.0, 3)
    return cache.get(final_w)[0], {
        "final_w_T": final_w,
        "lo": lo,
        "hi": hi,
        "rounds": len(trace) - 1,
        "stopped": stopped,
    }


async def run_dir_pair(
    qid,
    query_text,
    cache,
    judge,
    max_rounds=MAX_ROUNDS,
    init_w=0.4,
    step=STEP_DIR,
    with_references=False,
):
    """Comparison-vocabulary direction tuner.

    Each round, optimizer takes a tentative step in `direction`. Model judges:
    CURRENT_BETTER → accept move and continue same direction; PREVIOUS_BETTER →
    reject move and reverse direction; TIE → stop. The model never sees weight
    numbers, only the result sets and (optionally) reference extremes.
    """
    w = init_w
    direction = -1.0  # bias downward initially (empirical optimum is low)
    visited = [w]
    trace = [{"round": 0, "w_T": w}]
    stopped = "max_rounds"
    ref_S = cache.get(ANCHOR_S_W)[1] if with_references else None
    ref_T = cache.get(ANCHOR_T_W)[1] if with_references else None
    for r in range(max_rounds):
        proposed = max(0.0, min(1.0, round(w + direction * step, 3)))
        if abs(proposed - w) < VISITED_TOL:
            direction = -direction
            proposed = max(0.0, min(1.0, round(w + direction * step, 3)))
            if abs(proposed - w) < VISITED_TOL:
                stopped = "boundary"
                break
        if _abab(visited, proposed):
            stopped = "abab_cycle"
            break
        prev_set = cache.get(w)[1]
        cur_set = cache.get(proposed)[1]
        seed = hash((qid, "dir", with_references, r, w, proposed)) & 0xFFFFFFFF
        verdict = await judge.vote_direction(
            query_text, prev_set, cur_set, rng_seed=seed, ref_S=ref_S, ref_T=ref_T
        )
        trace.append(
            {
                "round": r + 1,
                "current": w,
                "proposed": proposed,
                "direction": direction,
                "verdict": verdict,
            }
        )
        if verdict == "TIE":
            stopped = "tie"
            break
        if verdict == "PREVIOUS_BETTER":
            # Move was bad — reverse direction, don't accept move
            direction = -direction
            continue
        # CURRENT_BETTER — accept the move
        w = proposed
        visited.append(w)
    return cache.get(w)[0], {
        "final_w_T": w,
        "rounds": len(trace) - 1,
        "stopped": stopped,
    }


def metrics(rankings, gold, qids):
    r1 = r3 = r5 = r10 = 0
    mrr_sum = 0.0
    n = 0
    for qid in qids:
        rel = set(gold.get(qid, []))
        if not rel:
            continue
        r = rankings.get(qid, [])
        hit = None
        for i, d in enumerate(r[:10]):
            if d in rel:
                hit = i + 1
                break
        if hit:
            if hit <= 1:
                r1 += 1
            if hit <= 3:
                r3 += 1
            if hit <= 5:
                r5 += 1
            if hit <= 10:
                r10 += 1
            mrr_sum += 1.0 / hit
        n += 1
    return {
        "r@1": r1 / n if n else 0,
        "r@3": r3 / n if n else 0,
        "r@5": r5 / n if n else 0,
        "r@10": r10 / n if n else 0,
        "mrr": mrr_sum / n if n else 0,
        "n": n,
    }


WEIGHT_GRID_ORACLE = [round(i * 0.1, 1) for i in range(11)]


def best_w_T_for_query(t, s, gold: set[str]) -> float:
    """Pick w_T from grid that ranks gold highest. Tie-break: smallest |w-0.4|."""
    best = (None, 11, 1.0)
    for w_T in WEIGHT_GRID_ORACLE:
        ranked = rank_blend_ts(t, s, w_T)
        h = None
        for i, d in enumerate(ranked[:10]):
            if d in gold:
                h = i + 1
                break
        if h is None:
            h = 11
        dist = abs(w_T - 0.4)
        if (h, dist) < (best[1], best[2]):
            best = (w_T, h, dist)
    return best[0]


async def run_bench(
    name, docs_path, queries_path, gold_path, cache_doc, cache_q, judge
):
    docs = [json.loads(l) for l in open(DATA_DIR / docs_path)]
    queries = [json.loads(l) for l in open(DATA_DIR / queries_path)]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / gold_path)]
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}

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

    per_q_t, per_q_s = {}, {}
    for q in queries:
        qid = q["query_id"]
        per_q_t[qid] = rank_multi_axis_t(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
        )
        per_q_s[qid] = rank_semantic(qid, q_embs, doc_embs)

    qids = [q["query_id"] for q in queries]

    # Baselines
    fixed_02 = {qid: rank_blend_ts(per_q_t[qid], per_q_s[qid], w_T=0.2) for qid in qids}
    pure_S = {qid: rank_blend_ts(per_q_t[qid], per_q_s[qid], w_T=0.0) for qid in qids}

    # Oracle
    oracle = {}
    for qid in qids:
        rel = gold.get(qid, set())
        if not rel:
            oracle[qid] = pure_S[qid]
            continue
        w_best = best_w_T_for_query(per_q_t[qid], per_q_s[qid], rel)
        oracle[qid] = rank_blend_ts(per_q_t[qid], per_q_s[qid], w_best)

    designs = (
        "blind_3way",
        "blind_3way_ref",
        "blind_pair",
        "blind_pair_ref",
        "blind_dir_pair",
        "blind_dir_pair_ref",
    )
    opt_results: dict[str, dict[str, list[str]]] = {d: {} for d in designs}
    opt_diag: dict[str, dict[str, dict]] = {d: {} for d in designs}

    async def run_one(qid: str, design: str):
        cache = RetrievalCache(per_q_t[qid], per_q_s[qid], doc_text)
        if design == "blind_3way":
            r, diag = await run_3way(
                qid, q_text[qid], cache, judge, with_references=False
            )
        elif design == "blind_3way_ref":
            r, diag = await run_3way(
                qid, q_text[qid], cache, judge, with_references=True
            )
        elif design == "blind_pair":
            r, diag = await run_pair(
                qid, q_text[qid], cache, judge, with_references=False
            )
        elif design == "blind_pair_ref":
            r, diag = await run_pair(
                qid, q_text[qid], cache, judge, with_references=True
            )
        elif design == "blind_dir_pair":
            r, diag = await run_dir_pair(
                qid, q_text[qid], cache, judge, with_references=False
            )
        elif design == "blind_dir_pair_ref":
            r, diag = await run_dir_pair(
                qid, q_text[qid], cache, judge, with_references=True
            )
        else:
            raise ValueError(design)
        opt_results[design][qid] = r
        opt_diag[design][qid] = diag

    tasks = []
    for d in designs:
        tasks.extend(run_one(qid, d) for qid in qids)
    await asyncio.gather(*tasks)
    judge.save()

    variants = {
        "pure_S (w=0)": pure_S,
        "V7L_TS (w=0.2)": fixed_02,
        "ORACLE": oracle,
    }
    for d in designs:
        variants[d] = opt_results[d]
    results = {
        var: metrics(ranks, {k: list(v) for k, v in gold.items()}, qids)
        for var, ranks in variants.items()
    }
    print(f"{'Variant':24} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6}")
    for var, m in results.items():
        print(
            f"{var:24} {m['r@1']:>6.3f} {m['r@3']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f}"
        )

    for d in designs:
        finals = [diag["final_w_T"] for diag in opt_diag[d].values()]
        rounds = [diag["rounds"] for diag in opt_diag[d].values()]
        reasons = Counter(diag["stopped"] for diag in opt_diag[d].values())
        print(
            f"  {d:18}: avg_final={np.mean(finals):.2f}  std={np.std(finals):.2f}  "
            f"min={np.min(finals):.2f}  max={np.max(finals):.2f}  "
            f"avg_rounds={np.mean(rounds):.2f}  stopped={dict(reasons)}"
        )

    return results, {
        d: {
            "final_w_Ts": [diag["final_w_T"] for diag in opt_diag[d].values()],
            "rounds": [diag["rounds"] for diag in opt_diag[d].values()],
            "stopped": dict(Counter(diag["stopped"] for diag in opt_diag[d].values())),
        }
        for d in designs
    }


async def main():
    judge = BlindJudge()
    benches = [
        (
            "mixed_cue",
            "mixed_cue_docs.jsonl",
            "mixed_cue_queries.jsonl",
            "mixed_cue_gold.jsonl",
            "v7l-mixed_cue",
            "v7l-mixed_cue",
        ),
        (
            "dense_cluster",
            "dense_cluster_docs.jsonl",
            "dense_cluster_queries.jsonl",
            "dense_cluster_gold.jsonl",
            "v7l-dense_cluster",
            "v7l-dense_cluster",
        ),
        (
            "tempreason_small",
            "real_benchmark_small_docs.jsonl",
            "real_benchmark_small_queries.jsonl",
            "real_benchmark_small_gold.jsonl",
            "v7l-tempreason",
            "v7l-tempreason",
        ),
        (
            "hard_bench",
            "hard_bench_docs.jsonl",
            "hard_bench_queries.jsonl",
            "hard_bench_gold.jsonl",
            "v7l-hard_bench",
            "v7l-hard_bench",
        ),
    ]
    all_results = {}
    all_diag = {}
    for name, *paths in benches:
        try:
            r, d = await run_bench(name, *paths, judge=judge)
            all_results[name] = r
            all_diag[name] = d
        except Exception as e:
            print(f"  [{name}] failed: {e}")
            import traceback

            traceback.print_exc()

    out_path = ROOT / "results" / "v7l_ts_blind.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"metrics": all_results, "diagnostics": all_diag}, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"\nLLM calls: {judge.calls}, failed: {judge.failed}, usage: {judge.usage}")

    print("\n=== SUMMARY (R@1) ===")
    print(f"{'Benchmark':22} {'Variant':24} {'R@1':>6} {'R@5':>6} {'MRR':>6}")
    for bname, vmap in all_results.items():
        for var, m in vmap.items():
            print(
                f"{bname:22} {var:24} {m['r@1']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f}"
            )


if __name__ == "__main__":
    asyncio.run(main())
