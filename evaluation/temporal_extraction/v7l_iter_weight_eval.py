"""V7L + iterative model-in-the-loop weight tuning.

Per query, the model adjusts the temporal-channel weight w_T using metacognitive
monitoring of the retrieval result set:

  Stage 0 (initial): model sees query alone, picks w_T from {0.2, 0.4, 0.6, 0.8}.
  Stages 1-3 (refine): model sees query + current top-5, says UP / DOWN / STOP;
                       step starts at 0.2 and halves each round.

Three variants per benchmark:
  - V7L: fixed w_T=0.4 (baseline)
  - V7L+initial: model picks initial w_T, no iteration
  - V7L+iter: initial + binary-search refinement

Costs ~4 LLM calls per query at gpt-5-mini minimal-reasoning (pennies-per-bench).
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

# Patch any extraction calls that slip through to use minimal reasoning.
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
CACHE_DIR = ROOT / "cache" / "iter_weight"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_cache.json"

PER_CALL_TIMEOUT_S = 30.0
CONCURRENCY = 8
MAX_TEXT_LEN = 240


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


INITIAL_PROMPT = """You're tuning a temporal-aware retrieval system. The system fuses three signals:
  - Temporal (T): match on dates/intervals extracted from text
  - Semantic (S): topic/content match via embeddings
  - Lattice (L): recurrence/axis patterns (weekday, season, era), fixed at 0.2

Given ONLY a user query (no results yet), pick how much weight to put on the temporal channel w_T. (Semantic gets 1 - w_T - 0.2.)

Query: {query}

Output exactly one number from {{0.2, 0.4, 0.6, 0.8}}:
- 0.2 — query is mostly topical with no time anchor (e.g. "what hobbies does Alex enjoy?")
- 0.4 — query has soft temporal context (e.g. "what was Alex doing recently?")
- 0.6 — query is anchored to a specific time (e.g. "what happened in March 2024?")
- 0.8 — query is purely a time lookup (e.g. "what occurred on 2024-03-15?")

Output one number only, no commentary."""


JUDGE_PROMPT = """You're tuning a temporal-aware retrieval system. The system fuses temporal score (T = date/interval match) with semantic score (S = topic match), with w_S = 1 - w_T - 0.2 (lattice fixed at 0.2).

Query: {query}
Current temporal weight: w_T = {w_T}

Top-5 results currently retrieved:
{results}

Decide whether to adjust w_T:
- UP — top results miss the query's temporal anchor / are temporally scattered when the query is time-specific. Raise w_T.
- DOWN — top results are over-anchored to a date but miss the query's topic intent. Lower w_T.
- STOP — top results look correct given the query.

Output exactly one word: UP, DOWN, or STOP. No commentary."""


class IterativeJudge:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(timeout=PER_CALL_TIMEOUT_S, max_retries=1)
        self.sem = asyncio.Semaphore(CONCURRENCY)
        self.cache = _load_cache()
        self._dirty = False
        self.usage = {"input": 0, "output": 0}
        self.calls = {"initial": 0, "judge": 0}
        self.failed = 0

    async def _call(self, prompt: str, max_tokens: int = 64) -> str:
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

    async def initial_weight(self, query: str) -> float:
        k = _key("initial", query)
        if k in self.cache:
            raw = self.cache[k]
        else:
            self.calls["initial"] += 1
            raw = await self._call(INITIAL_PROMPT.format(query=query), max_tokens=64)
            if raw:
                self.cache[k] = raw
                self._dirty = True
        return self._parse_weight(raw)

    @staticmethod
    def _parse_weight(raw: str) -> float:
        m = re.search(r"0?\.\d+", raw or "")
        if m:
            try:
                v = float(m.group(0))
                # Snap to grid; clamp
                v = max(0.2, min(0.8, v))
                return v
            except ValueError:
                pass
        return 0.4

    async def judge(self, query: str, results: list[str], w_T: float) -> str:
        body = f"{query}|{w_T:.3f}|{'||'.join(results)}"
        k = _key("judge", body)
        if k in self.cache:
            raw = self.cache[k]
        else:
            self.calls["judge"] += 1
            res_str = "\n".join(
                f"{i + 1}. {r[:MAX_TEXT_LEN]}" for i, r in enumerate(results)
            )
            raw = await self._call(
                JUDGE_PROMPT.format(query=query, w_T=f"{w_T:.2f}", results=res_str),
                max_tokens=64,
            )
            if raw:
                self.cache[k] = raw
                self._dirty = True
        return self._parse_judge(raw)

    @staticmethod
    def _parse_judge(raw: str) -> str:
        if not raw:
            return "STOP"
        u = raw.strip().upper()
        if u.startswith("UP"):
            return "UP"
        if u.startswith("DOWN"):
            return "DOWN"
        return "STOP"

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


def rank_blend(t, s, l, w_T: float, w_L: float = 0.2) -> list[str]:
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


async def iterative_rank(
    qid: str,
    query_text: str,
    t: dict,
    s: dict,
    l: dict,
    doc_text: dict[str, str],
    judge: IterativeJudge,
    max_rounds: int = 3,
    initial_step: float = 0.2,
) -> tuple[float, list[str], list[str], dict]:
    """Returns (initial_w_T, ranking_after_initial, ranking_after_iter, diagnostics)."""
    w_T_init = await judge.initial_weight(query_text)
    ranked_initial = rank_blend(t, s, l, w_T_init)

    w_T = w_T_init
    step = initial_step
    ranked = ranked_initial
    history = [("init", w_T)]

    for r in range(max_rounds):
        top5 = [doc_text.get(d, "")[:MAX_TEXT_LEN] for d in ranked[:5]]
        decision = await judge.judge(query_text, top5, w_T)
        history.append((decision, w_T))
        if decision == "STOP":
            break
        if decision == "UP":
            w_T = min(0.8, w_T + step)
        else:
            w_T = max(0.0, w_T - step)
        step /= 2.0
        ranked = rank_blend(t, s, l, w_T)

    diag = {
        "history": history,
        "initial_w_T": w_T_init,
        "final_w_T": w_T,
        "rounds_used": len([h for h in history if h[0] != "init"]),
    }
    return w_T_init, ranked_initial, ranked, diag


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

    lat_db = ROOT / "cache" / "iter_weight" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    ingest_lattice(lat, doc_ext)
    per_q_l = lattice_scores_for_query(lat, q_ext, [q["query_id"] for q in queries])

    qids = [q["query_id"] for q in queries]

    baseline = {
        qid: rank_blend(per_q_t[qid], per_q_s[qid], per_q_l[qid], w_T=0.4)
        for qid in qids
    }

    initial_only: dict[str, list[str]] = {}
    iter_results: dict[str, list[str]] = {}
    iter_diag: dict[str, dict] = {}

    async def one(qid: str):
        w_T_init, r_init, r_iter, diag = await iterative_rank(
            qid,
            q_text[qid],
            per_q_t[qid],
            per_q_s[qid],
            per_q_l[qid],
            doc_text,
            judge,
        )
        initial_only[qid] = r_init
        iter_results[qid] = r_iter
        iter_diag[qid] = diag

    await asyncio.gather(*(one(qid) for qid in qids))
    judge.save()

    variants = {
        "V7L": baseline,
        "V7L+initial": initial_only,
        "V7L+iter": iter_results,
    }
    results = {var: metrics(ranks, gold, qids) for var, ranks in variants.items()}
    print(f"{'Variant':24} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6} {'NDCG':>6}")
    for var, m in results.items():
        print(
            f"{var:24} {m['r@1']:>6.3f} {m['r@3']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f} {m['ndcg@10']:>6.3f}"
        )

    init_ws = [d["initial_w_T"] for d in iter_diag.values()]
    final_ws = [d["final_w_T"] for d in iter_diag.values()]
    rounds = [d["rounds_used"] for d in iter_diag.values()]
    print(
        f"  iter: avg_initial_w_T={np.mean(init_ws):.2f}  avg_final_w_T={np.mean(final_ws):.2f}  "
        f"std_final={np.std(final_ws):.2f}  avg_rounds={np.mean(rounds):.2f}"
    )

    return results, {
        "initial_w_Ts": init_ws,
        "final_w_Ts": final_ws,
        "rounds_used": rounds,
        "per_query": {
            qid: {
                "init": d["initial_w_T"],
                "final": d["final_w_T"],
                "history": d["history"],
            }
            for qid, d in iter_diag.items()
        },
    }


async def main():
    judge = IterativeJudge()

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

    out_path = ROOT / "results" / "v7l_iter_weight.json"
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
