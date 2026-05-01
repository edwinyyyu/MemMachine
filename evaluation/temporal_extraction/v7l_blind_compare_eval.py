"""V7L + blinded result-set comparison.

The model never sees weight values. It only compares result sets and picks
which one best matches the query. The optimizer translates the pick into a
weight change.

Five designs:

  blind_3way        — Each round: retrieve at {w-step, w, w+step}; show all 3
                      shuffled+blinded; model picks best (or "tie"); jump to
                      picked w. Stop on middle/tie/cycle.

  blind_pair        — Each round: take a tentative step; show {prev_set,
                      cur_set} shuffled+blinded; model picks. Continue same
                      direction if cur wins; reverse if prev wins; stop on tie.

  blind_3way_anch   — blind_3way + always include anchor sets at w=0 (pure
                      semantic) and w=W_CAP (pure temporal). Picks among 5
                      candidates; anchor pick jumps w_T to that extreme.

  blind_pair_anch   — blind_pair + same two anchors. Picks among 4 candidates.

  two_pointer       — lo, hi pointers (initial 0.0 and W_CAP). Each round:
                      compare retrievals at lo and hi (blinded); loser moves
                      toward winner by FIXED step (0.1). Stop when lo > hi
                      crosses, or model says tie. Final w_T = midpoint.

All designs ship the model only blinded "Set 1, Set 2, ..." labels. Random
order each call. Optimizer privately maps blind index → weight.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
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
    parse_iso,
    rank_semantic,
    run_v2_extract,
)
from salience_eval import (
    rank_t as rank_multi_axis_t,
)

MODEL = "gpt-5-mini"
CACHE_DIR = ROOT / "cache" / "blind_compare"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_cache.json"

PER_CALL_TIMEOUT_S = 30.0
CONCURRENCY = 8
MAX_TEXT_LEN = 180
MAX_ROUNDS = 4
W_L = 0.2
W_CAP = 1.0 - W_L  # 0.8

STEP_3WAY = 0.2
STEP_PAIR = 0.15
STEP_TWOPTR = 0.1
VISITED_TOL = 0.025

# Anchor sentinel weights — these intentionally use w_L=0 so the model sees
# truly pure-channel retrievals as reference points. Local candidates still
# use w_L=0.2 (the system's actual operating mode).
ANCHOR_S_W_T = 0.0  # pure semantic: w_S=1, w_T=0, w_L=0
ANCHOR_T_W_T = 1.0  # pure temporal: w_T=1, w_S=0, w_L=0
ANCHOR_W_L = 0.0
ANCHOR_S_KEY = "anchor_S"
ANCHOR_T_KEY = "anchor_T"


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


PICK_PROMPT = """Look at the query and the candidate result sets below. Each set is a top-5 list of documents that might match the query. Pick the set that best matches the query.

Query: {query}

{sets}

Output exactly one number identifying the best set ({choices}).
Output 0 if all sets look equally good or no set is clearly best.
No commentary."""


def _format_sets(sets: list[list[str]]) -> str:
    out_lines = []
    for i, s in enumerate(sets):
        out_lines.append(f"Set {i + 1}:")
        for j, t in enumerate(s):
            out_lines.append(f"  {j + 1}. {t[:MAX_TEXT_LEN]}")
        out_lines.append("")
    return "\n".join(out_lines)


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
    ) -> int:
        """Returns index into `candidate_sets` of the model's pick, or -1 for tie/none.

        Shuffles candidates (deterministic per rng_seed) before showing the model;
        unshuffles the answer back to the original index space.
        """
        n = len(candidate_sets)
        rng = random.Random(rng_seed)
        order = list(range(n))
        rng.shuffle(order)
        shuffled_sets = [candidate_sets[i] for i in order]
        choices_str = ", ".join(str(i + 1) for i in range(n))
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
        if v == 0:
            return -1  # tie / none
        if v < 1 or v > n:
            return -1
        # unshuffle: shuffled position v-1 corresponds to original position order[v-1]
        return order[v - 1]

    def save(self) -> None:
        if self._dirty:
            _save_cache(self.cache)
            self._dirty = False


def ingest_lattice(store: LatticeStore, extracted) -> None:
    for doc_id, tes in extracted.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            store.insert(doc_id, ts.absolute, ts.cyclical)


def lattice_scores_for_query(store, q_extracted, query_ids):
    out = {}
    for qid in query_ids:
        tes = q_extracted.get(qid, [])
        scores, _ = lattice_retrieve_multi(store, tes, down_levels=1)
        out[qid] = scores
    return out


def rank_blend(t, s, l, w_T: float, w_L: float | None = None) -> list[str]:
    if w_L is None:
        w_L = W_L
    w_S = max(0.0, 1.0 - w_T - w_L)
    chans = {"T": t, "S": s, "L": l}
    weights = {"T": w_T, "S": w_S, "L": w_L}
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
    """Per-query cache. Keys are either floats (local w_T candidates with w_L=0.2)
    or sentinel strings ANCHOR_S_KEY / ANCHOR_T_KEY for true pure-channel anchors
    (w_L = 0)."""

    def __init__(self, t, s, l, doc_text):
        self.t, self.s, self.l = t, s, l
        self.doc_text = doc_text
        self._cache: dict = {}

    def get(self, key) -> tuple[list[str], list[str]]:
        if key == ANCHOR_S_KEY:
            cache_key = ANCHOR_S_KEY
            if cache_key not in self._cache:
                ranked = rank_blend(
                    self.t, self.s, self.l, ANCHOR_S_W_T, w_L=ANCHOR_W_L
                )
                top5_text = [
                    self.doc_text.get(d, "")[:MAX_TEXT_LEN] for d in ranked[:5]
                ]
                self._cache[cache_key] = (ranked, top5_text)
            return self._cache[cache_key]
        if key == ANCHOR_T_KEY:
            cache_key = ANCHOR_T_KEY
            if cache_key not in self._cache:
                ranked = rank_blend(
                    self.t, self.s, self.l, ANCHOR_T_W_T, w_L=ANCHOR_W_L
                )
                top5_text = [
                    self.doc_text.get(d, "")[:MAX_TEXT_LEN] for d in ranked[:5]
                ]
                self._cache[cache_key] = (ranked, top5_text)
            return self._cache[cache_key]
        # Float w_T (local candidate) — clip to W_CAP, w_L stays at default 0.2
        cache_key = round(max(0.0, min(W_CAP, float(key))), 3)
        if cache_key not in self._cache:
            ranked = rank_blend(self.t, self.s, self.l, cache_key)
            top5_text = [self.doc_text.get(d, "")[:MAX_TEXT_LEN] for d in ranked[:5]]
            self._cache[cache_key] = (ranked, top5_text)
        return self._cache[cache_key]


def _abab(visited: list[float], proposed: float) -> bool:
    if len(visited) < 3:
        return False
    return (
        abs(visited[-3] - visited[-1]) < VISITED_TOL
        and abs(proposed - visited[-2]) < VISITED_TOL
    )


async def run_3way(
    qid: str,
    query_text: str,
    cache: RetrievalCache,
    judge: BlindJudge,
    anchored: bool = False,
    max_rounds: int = MAX_ROUNDS,
    init_w: float = 0.4,
) -> tuple[list[str], dict]:
    w = init_w
    visited = [w]
    trace = [{"round": 0, "w_T": w}]
    stopped = "max_rounds"
    for r in range(max_rounds):
        cands_w = [
            max(0.0, round(w - STEP_3WAY, 3)),
            w,
            min(W_CAP, round(w + STEP_3WAY, 3)),
        ]
        if anchored:
            cands_w = cands_w + [ANCHOR_S, ANCHOR_T]
        # Dedupe (e.g., near boundary lower==current)
        dedup_idx_to_w: list[float] = []
        for cw in cands_w:
            if not dedup_idx_to_w or all(
                abs(cw - x) > VISITED_TOL for x in dedup_idx_to_w
            ):
                dedup_idx_to_w.append(cw)
        cand_sets = [cache.get(cw)[1] for cw in dedup_idx_to_w]
        seed = hash((qid, "3way", anchored, r, w)) & 0xFFFFFFFF
        idx = await judge.pick_best(query_text, cand_sets, rng_seed=seed)
        trace.append({"round": r + 1, "candidates": dedup_idx_to_w, "picked_idx": idx})
        if idx < 0:
            stopped = "tie"
            break
        chosen_w = dedup_idx_to_w[idx]
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
        "trace": trace,
    }


async def run_pair(
    qid: str,
    query_text: str,
    cache: RetrievalCache,
    judge: BlindJudge,
    anchored: bool = False,
    max_rounds: int = MAX_ROUNDS,
    init_w: float = 0.4,
    step: float = STEP_PAIR,
) -> tuple[list[str], dict]:
    """Pairwise: take a tentative step in a probing direction; show prev/curr blinded.
    If model picks current → continue same direction; else → reverse.
    """
    w = init_w
    direction = -1.0  # bias toward exploring downward first (empirical optimum is low)
    prev_w = None
    visited = [w]
    trace = [{"round": 0, "w_T": w}]
    stopped = "max_rounds"
    for r in range(max_rounds):
        proposed = max(0.0, min(W_CAP, round(w + direction * step, 3)))
        if abs(proposed - w) < VISITED_TOL:
            # at boundary already, try opposite direction
            direction = -direction
            proposed = max(0.0, min(W_CAP, round(w + direction * step, 3)))
            if abs(proposed - w) < VISITED_TOL:
                stopped = "boundary"
                break
        if _abab(visited, proposed):
            stopped = "abab_cycle"
            break

        cands_w: list[float] = [w, proposed]
        if anchored:
            cands_w = cands_w + [ANCHOR_S, ANCHOR_T]
        dedup_idx_to_w: list[float] = []
        for cw in cands_w:
            if not dedup_idx_to_w or all(
                abs(cw - x) > VISITED_TOL for x in dedup_idx_to_w
            ):
                dedup_idx_to_w.append(cw)
        cand_sets = [cache.get(cw)[1] for cw in dedup_idx_to_w]
        seed = hash((qid, "pair", anchored, r, w, proposed)) & 0xFFFFFFFF
        idx = await judge.pick_best(query_text, cand_sets, rng_seed=seed)
        trace.append(
            {
                "round": r + 1,
                "current": w,
                "proposed": proposed,
                "candidates": dedup_idx_to_w,
                "picked_idx": idx,
            }
        )
        if idx < 0:
            stopped = "tie"
            break
        chosen_w = dedup_idx_to_w[idx]
        if abs(chosen_w - w) < VISITED_TOL:
            # current wins → reverse direction next round
            direction = -direction
            # we still record but don't move
            continue
        # accept the move (could be the proposed one or an anchor)
        prev_w = w
        w = chosen_w
        visited.append(w)
        # If we accepted toward proposed, keep direction; if anchor jump, set direction toward S or T
        if abs(chosen_w - proposed) < VISITED_TOL:
            pass  # keep direction
        elif abs(chosen_w - ANCHOR_S) < VISITED_TOL:
            direction = -1.0
        elif abs(chosen_w - ANCHOR_T) < VISITED_TOL:
            direction = 1.0
    return cache.get(w)[0], {
        "final_w_T": w,
        "rounds": len(trace) - 1,
        "stopped": stopped,
        "trace": trace,
    }


async def run_two_pointer(
    qid: str,
    query_text: str,
    cache: RetrievalCache,
    judge: BlindJudge,
    max_rounds: int = MAX_ROUNDS,
    lo_init: float = 0.0,
    hi_init: float = W_CAP,
    step: float = STEP_TWOPTR,
) -> tuple[list[str], dict]:
    """lo and hi pointers; each round model compares results at lo vs hi; loser moves toward winner by fixed step."""
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
            # lo wins → hi moves down by step (toward lo)
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
        "trace": trace,
    }


def metrics(rankings, gold, qids):
    r1 = r3 = r5 = r10 = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
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
            dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(r[:10]) if d in rel)
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), 10)))
            ndcg_sum += dcg / ideal if ideal else 0.0
        n += 1
    return {
        "n": n,
        "r@1": r1 / n if n else 0,
        "r@3": r3 / n if n else 0,
        "r@5": r5 / n if n else 0,
        "r@10": r10 / n if n else 0,
        "mrr": mrr_sum / n if n else 0,
        "ndcg@10": ndcg_sum / n if n else 0,
    }


async def run_bench(
    name, docs_path, queries_path, gold_path, cache_doc, cache_q, judge
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

    per_q_t, per_q_s = {}, {}
    for q in queries:
        qid = q["query_id"]
        per_q_t[qid] = rank_multi_axis_t(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
        )
        per_q_s[qid] = rank_semantic(qid, q_embs, doc_embs)

    lat_db = ROOT / "cache" / "blind_compare" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    ingest_lattice(lat, doc_ext)
    per_q_l = lattice_scores_for_query(lat, q_ext, [q["query_id"] for q in queries])

    qids = [q["query_id"] for q in queries]

    fixed_04 = {
        qid: rank_blend(per_q_t[qid], per_q_s[qid], per_q_l[qid], w_T=0.4)
        for qid in qids
    }
    fixed_02 = {
        qid: rank_blend(per_q_t[qid], per_q_s[qid], per_q_l[qid], w_T=0.2)
        for qid in qids
    }

    designs = (
        "blind_3way",
        "blind_pair",
        "blind_3way_anch",
        "blind_pair_anch",
        "two_pointer",
    )
    opt_results: dict[str, dict[str, list[str]]] = {d: {} for d in designs}
    opt_diag: dict[str, dict[str, dict]] = {d: {} for d in designs}

    async def run_one(qid: str, design: str):
        cache = RetrievalCache(per_q_t[qid], per_q_s[qid], per_q_l[qid], doc_text)
        if design == "blind_3way":
            r, diag = await run_3way(qid, q_text[qid], cache, judge, anchored=False)
        elif design == "blind_3way_anch":
            r, diag = await run_3way(qid, q_text[qid], cache, judge, anchored=True)
        elif design == "blind_pair":
            r, diag = await run_pair(qid, q_text[qid], cache, judge, anchored=False)
        elif design == "blind_pair_anch":
            r, diag = await run_pair(qid, q_text[qid], cache, judge, anchored=True)
        elif design == "two_pointer":
            r, diag = await run_two_pointer(qid, q_text[qid], cache, judge)
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
        "V7L (w=0.4)": fixed_04,
        "V7L (w=0.2)": fixed_02,
    }
    for d in designs:
        variants[f"V7L+{d}"] = opt_results[d]
    results = {var: metrics(ranks, gold, qids) for var, ranks in variants.items()}
    print(f"{'Variant':28} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6} {'NDCG':>6}")
    for var, m in results.items():
        print(
            f"{var:28} {m['r@1']:>6.3f} {m['r@3']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f} {m['ndcg@10']:>6.3f}"
        )

    for d in designs:
        finals = [diag["final_w_T"] for diag in opt_diag[d].values()]
        rounds = [diag["rounds"] for diag in opt_diag[d].values()]
        reasons = Counter(diag["stopped"] for diag in opt_diag[d].values())
        print(
            f"  {d:18}: avg_final_w_T={np.mean(finals):.2f}  std={np.std(finals):.2f}  "
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

    out_path = ROOT / "results" / "v7l_blind_compare.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"metrics": all_results, "diagnostics": all_diag}, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"\nLLM calls: {judge.calls}, failed: {judge.failed}, usage: {judge.usage}")

    print("\n=== SUMMARY ===")
    print(f"{'Benchmark':22} {'Variant':28} {'R@1':>6} {'R@5':>6} {'MRR':>6}")
    for bname, vmap in all_results.items():
        for var, m in vmap.items():
            print(
                f"{bname:22} {var:28} {m['r@1']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f}"
            )


if __name__ == "__main__":
    asyncio.run(main())
