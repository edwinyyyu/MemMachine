"""V7L + stepwise weight-adjustment optimizers.

Per query, the model is shown CURRENT and PREVIOUS top-5 result sets and the
weights that produced each. It has real comparison evidence ("did the last
step help?") rather than judging a single result set in isolation.

Three optimizer variants tested:

  - sgd:        fixed-step UP/DOWN/STOP (delta = 0.1).
  - momentum:   accumulate consecutive-direction votes; reverse zeros momentum.
  - magnitude:  model picks step from {-0.2, -0.1, 0, +0.1, +0.2}.

All start at w_T=0.4 and run up to 4 rounds (4 judge calls per query).

Compares against:
  - V7L (fixed w_T=0.4, original baseline)
  - V7L-tuned (fixed w_T=0.2, empirically best-fixed from oracle sweep)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sys
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
CACHE_DIR = ROOT / "cache" / "step_opt"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_cache.json"

PER_CALL_TIMEOUT_S = 30.0
CONCURRENCY = 8
MAX_TEXT_LEN = 220
MAX_ROUNDS = 4
W_L = 0.2
W_CAP = 1.0 - W_L  # T can occupy at most this much weight; 0.8

# Linear-space optimizers
SGD_STEP = 0.1
MOMENTUM_BASE = 0.1
MOMENTUM_MAX = 0.3
MAGNITUDE_OPTIONS = [-0.2, -0.1, 0.0, 0.1, 0.2]

# Logit-space optimizers (z = logit(w_T / W_CAP); w_T = W_CAP * sigmoid(z))
# Logit step 0.5 ≈ Δw 0.10 near middle, ≈ Δw 0.04 near edges
LOGIT_SGD_STEP = 0.5
LOGIT_MAGNITUDE_SCALE = 5.0  # model picks linear delta, we scale into logit
LOGIT_INIT_Z = 0.0  # logit(0.5) = 0 → w_T = 0.4 (matches linear init)

# Cycle avoidance: if the proposed next w_T is within VISITED_TOL of any
# already-visited w_T, declare convergence and stop. Prevents UP/DOWN
# oscillation around the optimum without needing binary-search-style decay.
VISITED_TOL = 0.025


def w_to_z(w: float) -> float:
    eps = 1e-6
    p = max(eps, min(1.0 - eps, w / W_CAP))
    return math.log(p / (1.0 - p))


def z_to_w(z: float) -> float:
    return W_CAP / (1.0 + math.exp(-z))


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


FIRST_ROUND_DIRECTION = """You're tuning a temporal-aware retrieval system. The temporal weight w_T controls how much the system weights date/interval match vs topic content.

Query: {query}

Current attempt (w_T = {cur_w}):
{cur_top5}

There is no previous attempt yet. Look at the current results.
- Are they over-anchored to dates but missing the query's topic? → DOWN (lower w_T)
- Are they topically scattered when the query has a clear time anchor? → UP (raise w_T)
- Do they look right? → STOP

Output exactly one word: UP, DOWN, or STOP. No commentary."""


WITH_PREV_DIRECTION = """You're tuning a temporal-aware retrieval system. The temporal weight w_T controls how much the system weights date/interval match vs topic content.

Query: {query}

Previous attempt (w_T = {prev_w}):
{prev_top5}

Current attempt (w_T = {cur_w}):
{cur_top5}

Compare. Did the change from previous to current improve the match to the query?

- UP: continue raising w_T (current is better and is higher; or current is worse and is lower → reverse upward)
- DOWN: continue lowering w_T (current is better and is lower; or current is worse and is higher → reverse downward)
- STOP: current results are good

Output exactly one word: UP, DOWN, or STOP. No commentary."""


FIRST_ROUND_MAGNITUDE = """You're tuning a temporal-aware retrieval system. The temporal weight w_T controls how much the system weights date/interval match vs topic content.

Query: {query}

Current attempt (w_T = {cur_w}):
{cur_top5}

There is no previous attempt yet. Pick a step size to adjust w_T:
  -0.2  large decrease (results badly over-anchored to dates)
  -0.1  small decrease
   0    leave alone (results look right)
  +0.1  small increase
  +0.2  large increase (results miss the temporal anchor)

Output exactly one number from {{-0.2, -0.1, 0, 0.1, 0.2}}. No commentary."""


WITH_PREV_MAGNITUDE = """You're tuning a temporal-aware retrieval system. The temporal weight w_T controls how much the system weights date/interval match vs topic content.

Query: {query}

Previous attempt (w_T = {prev_w}):
{prev_top5}

Current attempt (w_T = {cur_w}):
{cur_top5}

Compare. Pick the next step for w_T:
  -0.2  large decrease (lower w_T more)
  -0.1  small decrease
   0    stop (current is good)
  +0.1  small increase
  +0.2  large increase (raise w_T more)

Reverse direction if the previous-to-current change made things worse.

Output exactly one number from {{-0.2, -0.1, 0, 0.1, 0.2}}. No commentary."""


def _format_top5(top5_texts: list[str]) -> str:
    return "\n".join(f"  {i + 1}. {t[:MAX_TEXT_LEN]}" for i, t in enumerate(top5_texts))


class StepJudge:
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

    async def _cached(self, stage: str, prompt: str) -> str:
        k = _key(stage, prompt)
        if k in self.cache:
            return self.cache[k]
        self.calls += 1
        raw = await self._llm(prompt)
        if raw:
            self.cache[k] = raw
            self._dirty = True
        return raw

    async def vote_direction(
        self,
        query: str,
        cur_w: float,
        cur_top5: list[str],
        prev_w: float | None,
        prev_top5: list[str] | None,
    ) -> str:
        if prev_w is None:
            prompt = FIRST_ROUND_DIRECTION.format(
                query=query, cur_w=f"{cur_w:.2f}", cur_top5=_format_top5(cur_top5)
            )
        else:
            prompt = WITH_PREV_DIRECTION.format(
                query=query,
                prev_w=f"{prev_w:.2f}",
                prev_top5=_format_top5(prev_top5 or []),
                cur_w=f"{cur_w:.2f}",
                cur_top5=_format_top5(cur_top5),
            )
        raw = await self._cached("direction", prompt)
        u = (raw or "").strip().upper()
        if u.startswith("UP"):
            return "UP"
        if u.startswith("DOWN"):
            return "DOWN"
        return "STOP"

    async def vote_magnitude(
        self,
        query: str,
        cur_w: float,
        cur_top5: list[str],
        prev_w: float | None,
        prev_top5: list[str] | None,
    ) -> float:
        if prev_w is None:
            prompt = FIRST_ROUND_MAGNITUDE.format(
                query=query, cur_w=f"{cur_w:.2f}", cur_top5=_format_top5(cur_top5)
            )
        else:
            prompt = WITH_PREV_MAGNITUDE.format(
                query=query,
                prev_w=f"{prev_w:.2f}",
                prev_top5=_format_top5(prev_top5 or []),
                cur_w=f"{cur_w:.2f}",
                cur_top5=_format_top5(cur_top5),
            )
        raw = await self._cached("magnitude", prompt)
        m = re.search(r"[-+]?0?\.\d+|[-+]?[01]", raw or "")
        if not m:
            return 0.0
        try:
            v = float(m.group(0))
        except ValueError:
            return 0.0
        return min(MAGNITUDE_OPTIONS, key=lambda x: abs(x - v))

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


def rank_blend(t, s, l, w_T: float) -> list[str]:
    w_S = max(0.0, 1.0 - w_T - W_L)
    chans = {"T": t, "S": s, "L": l}
    weights = {"T": w_T, "S": w_S, "L": W_L}
    fused = score_blend(chans, weights, top_k_per=40, dispersion_cv_ref=0.20)
    ranked = [d for d, _ in fused]
    seen = set(ranked)
    tail = [
        d
        for d, _ in sorted(s.items(), key=lambda x: x[1], reverse=True)
        if d not in seen
    ]
    return ranked + tail


def _linear_step(
    base: str, sign: float, mag: float, momentum: float
) -> tuple[float, float]:
    """Returns (delta_w, new_momentum) for a linear-space step."""
    if base == "sgd":
        return sign * SGD_STEP, momentum
    if base == "momentum":
        if (sign > 0 and momentum > 0) or (sign < 0 and momentum < 0):
            momentum = max(
                -MOMENTUM_MAX, min(MOMENTUM_MAX, momentum + sign * MOMENTUM_BASE)
            )
        else:
            momentum = sign * MOMENTUM_BASE
        return momentum, momentum
    if base == "magnitude":
        return sign * mag, momentum
    raise ValueError(f"unknown base: {base}")


def _logit_step(
    base: str, sign: float, mag: float, momentum: float
) -> tuple[float, float]:
    """Returns (delta_z, new_momentum) for a logit-space step."""
    if base == "sgd":
        return sign * LOGIT_SGD_STEP, momentum
    if base == "momentum":
        # Logit-space momentum scales with the logit step size
        base_z = LOGIT_SGD_STEP
        max_z = LOGIT_SGD_STEP * 3.0
        if (sign > 0 and momentum > 0) or (sign < 0 and momentum < 0):
            momentum = max(-max_z, min(max_z, momentum + sign * base_z))
        else:
            momentum = sign * base_z
        return momentum, momentum
    if base == "magnitude":
        return sign * mag * LOGIT_MAGNITUDE_SCALE, momentum
    raise ValueError(f"unknown base: {base}")


def _is_abab(
    weights_seen: list[float], proposed: float, tol: float = VISITED_TOL
) -> bool:
    """True if applying `proposed` would form an A-B-A-B pattern at the trajectory tail.

    Allows A-B-A (correction) but stops A-B-A-B (oscillation). Requires at least
    3 prior visits — so the very first revisit is always allowed.
    """
    if len(weights_seen) < 3:
        return False
    a_prev = weights_seen[-3]
    b_prev = weights_seen[-2]
    a_cur = weights_seen[-1]
    return (
        abs(a_prev - a_cur) < tol  # A repeated at positions n-3 and n-1
        and abs(proposed - b_prev) < tol  # proposed equals B from position n-2
    )


async def run_optimizer(
    optimizer: str,
    query_text: str,
    t: dict,
    s: dict,
    l: dict,
    doc_text: dict[str, str],
    judge: StepJudge,
    initial_w_T: float = 0.4,
    max_rounds: int = MAX_ROUNDS,
) -> tuple[list[str], dict]:
    """Run a stepwise optimizer in linear or logit space.

    optimizer: one of 'sgd', 'momentum', 'magnitude', 'logit_sgd', 'logit_magnitude'.
    """
    is_logit = optimizer.startswith("logit_")
    base_opt = optimizer.removeprefix("logit_")

    if is_logit:
        z = LOGIT_INIT_Z
        w_T = z_to_w(z)
    else:
        z = None
        w_T = initial_w_T

    momentum = 0.0
    history: list[tuple[float, list[str]]] = []  # (w_T, top5_doc_ids)
    weights_seen: list[float] = []  # for cycle detection

    ranked = rank_blend(t, s, l, w_T)
    history.append((w_T, ranked[:5]))
    weights_seen.append(w_T)
    trace = [{"round": 0, "w_T": w_T, "vote": None}]
    stopped_reason = "max_rounds"

    for r in range(max_rounds):
        cur_w, cur_doc_ids = history[-1]
        cur_top5_text = [doc_text.get(d, "")[:MAX_TEXT_LEN] for d in cur_doc_ids]
        if len(history) >= 2:
            prev_w, prev_doc_ids = history[-2]
            prev_top5_text = [doc_text.get(d, "")[:MAX_TEXT_LEN] for d in prev_doc_ids]
        else:
            prev_w, prev_top5_text = None, None

        if base_opt in ("sgd", "momentum"):
            vote = await judge.vote_direction(
                query_text, cur_w, cur_top5_text, prev_w, prev_top5_text
            )
            trace.append({"round": r + 1, "vote": vote})
            if vote == "STOP":
                stopped_reason = "stop_vote"
                break
            sign = 1.0 if vote == "UP" else -1.0
            mag = 0.0  # unused for sgd/momentum
        elif base_opt == "magnitude":
            delta_lin = await judge.vote_magnitude(
                query_text, cur_w, cur_top5_text, prev_w, prev_top5_text
            )
            trace.append({"round": r + 1, "vote": delta_lin})
            if abs(delta_lin) < 1e-6:
                stopped_reason = "stop_vote"
                break
            sign = 1.0 if delta_lin > 0 else -1.0
            mag = abs(delta_lin)
        else:
            raise ValueError(f"unknown optimizer: {optimizer}")

        if is_logit:
            dz, momentum = _logit_step(base_opt, sign, mag, momentum)
            new_z = z + dz
            proposed_w = z_to_w(new_z)
        else:
            dw, momentum = _linear_step(base_opt, sign, mag, momentum)
            proposed_w = max(0.0, min(W_CAP, w_T + dw))

        if _is_abab(weights_seen, proposed_w):
            stopped_reason = "abab_cycle"
            break

        if is_logit:
            z = new_z
        w_T = proposed_w
        ranked = rank_blend(t, s, l, w_T)
        history.append((w_T, ranked[:5]))
        weights_seen.append(w_T)
        trace[-1]["w_T_after"] = w_T

    return ranked, {
        "trace": trace,
        "final_w_T": w_T,
        "rounds_used": len(trace) - 1,
        "stopped_reason": stopped_reason,
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

    lat_db = ROOT / "cache" / "step_opt" / f"lat_{name}.sqlite"
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

    optimizers = ("sgd", "momentum", "magnitude", "logit_sgd", "logit_magnitude")
    opt_results: dict[str, dict[str, list[str]]] = {o: {} for o in optimizers}
    opt_diag: dict[str, dict[str, dict]] = {o: {} for o in optimizers}

    async def run_one(qid: str, optimizer: str):
        r, diag = await run_optimizer(
            optimizer,
            q_text[qid],
            per_q_t[qid],
            per_q_s[qid],
            per_q_l[qid],
            doc_text,
            judge,
        )
        opt_results[optimizer][qid] = r
        opt_diag[optimizer][qid] = diag

    tasks = []
    for opt in optimizers:
        tasks.extend(run_one(qid, opt) for qid in qids)
    await asyncio.gather(*tasks)
    judge.save()

    variants = {
        "V7L (w=0.4)": fixed_04,
        "V7L (w=0.2)": fixed_02,
    }
    for o in optimizers:
        variants[f"V7L+{o}"] = opt_results[o]
    results = {var: metrics(ranks, gold, qids) for var, ranks in variants.items()}
    print(f"{'Variant':28} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6} {'NDCG':>6}")
    for var, m in results.items():
        print(
            f"{var:28} {m['r@1']:>6.3f} {m['r@3']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f} {m['ndcg@10']:>6.3f}"
        )

    from collections import Counter

    for opt in optimizers:
        finals = [d["final_w_T"] for d in opt_diag[opt].values()]
        rounds = [d["rounds_used"] for d in opt_diag[opt].values()]
        reasons = Counter(d["stopped_reason"] for d in opt_diag[opt].values())
        print(
            f"  {opt:18}: avg_final_w_T={np.mean(finals):.2f}  std={np.std(finals):.2f}  "
            f"min={np.min(finals):.2f}  max={np.max(finals):.2f}  "
            f"avg_rounds={np.mean(rounds):.2f}  stopped={dict(reasons)}"
        )

    return results, {
        opt: {
            "final_w_Ts": [d["final_w_T"] for d in opt_diag[opt].values()],
            "rounds_used": [d["rounds_used"] for d in opt_diag[opt].values()],
            "stopped_reasons": dict(
                Counter(d["stopped_reason"] for d in opt_diag[opt].values())
            ),
        }
        for opt in optimizers
    }


async def main():
    judge = StepJudge()

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

    out_path = ROOT / "results" / "v7l_step_opt.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"metrics": all_results, "diagnostics": all_diag}, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"\nLLM calls: {judge.calls}, failed: {judge.failed}, usage: {judge.usage}")

    print("\n=== SUMMARY ===")
    print(f"{'Benchmark':22} {'Variant':24} {'R@1':>6} {'R@5':>6} {'MRR':>6}")
    for bname, vmap in all_results.items():
        for var, m in vmap.items():
            print(
                f"{bname:22} {var:24} {m['r@1']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f}"
            )


if __name__ == "__main__":
    asyncio.run(main())
