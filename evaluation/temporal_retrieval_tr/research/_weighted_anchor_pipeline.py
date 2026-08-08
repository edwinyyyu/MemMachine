"""Weighted-anchor pipeline — planner + retriever + bench in one module.

The proposed architecture (probe):
- Plan = list[(interval, weight)] anchors + proximity_anchor
- Negation expressed via signed weight, NOT via complement intervals
- Graded importance / soft preferences expressed via fractional weights
- Single interval per anchor — no IntervalSet, no set algebra in the prompt
- Scoring: weighted_score = Σ wᵢ · pair_overlap(intervalᵢ, doc_anchor)
- Pool admission: any positive weighted contribution OR no positive anchor

A/B'd against the v7.1 proximity-anchor ship to determine: better, worse,
or net-neutral.
"""
from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from openai import AsyncOpenAI
from openai.types.responses import ResponseTextConfigParam
from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)

from temporal_retrieval_min.core import build_pool, Interval as V1Interval
from temporal_retrieval_min.schema import parse_iso, to_us
from temporal_retrieval_tr import Doc, TemporalExtractor
from temporal_retrieval_tr.planner import (
    NEG_INF, POS_INF,
    _iso_to_us, _proximity_from_str,
)
from temporal_retrieval_tr.retriever import Result, TemporalRetriever
from temporal_retrieval_tr.scoring import pair_overlap
from temporal_retrieval_tr.time_range import (
    Endpoint, Interval, IntervalSet, is_inf, is_infinite_measure,
)
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)

setup_env()

if not os.environ.get("OPENAI_API_KEY"):
    from dotenv import load_dotenv
    load_dotenv(Path("/Users/eyu/edwinyyyu/mmcc/extra_memory/.env"))


# ============================================================================
# WEIGHTED-ANCHOR PROMPT
# ============================================================================

MODEL = "gpt-5-mini"
PROMPT_VERSION = "wa-v1"

WEIGHTED_PROMPT = """You translate a natural-language query into a TIME-RANGE PLAN.

A query implies WHICH MOMENTS IN TIME a matching document's date anchor
should fall on. You describe that intent as a list of WEIGHTED ANCHORS
plus an optional proximity anchor.

OUTPUT SHAPE
============
{{
  "anchors": [
    {{
      "interval": {{"lo": "YYYY-MM-DD"|null, "hi": "YYYY-MM-DD"|null}},
      "weight": <float>
    }}
  ],
  "proximity_anchor": "YYYY-MM-DD" | "latest" | "earliest" | null
}}

Each anchor is ONE half-open interval [lo, hi) carrying a real-valued
weight. The doc's match score is `Σ weight_i · overlap_i(doc, interval_i)`
where `overlap` is 1 if the doc anchor is inside the interval (frac-min
on intersection / min-measure for partial overlap). Null lo/hi mean
unbounded (lo=null is -infinity; hi=null is +infinity).

WEIGHT SEMANTICS
================
- Positive (default +1.0): the user WANTS docs anchored in this interval.
- Negative (typically -1.0): the user wants to EXCLUDE docs anchored here.
- Fractional positive (e.g. +0.3): supporting/context interval, weaker preference.
- Omit anchors with no preference.

COMPOSITION RULES (do these AT THE LLM LEVEL — emit the weighted result)
========================================================================
- "in X" → one anchor [X.lo, X.hi) weight +1.0
- "after X" → one anchor [X.hi, +inf) weight +1.0
- "before X" → one anchor (-inf, X.lo) weight +1.0
- "between X and Y" → one anchor [X.lo, Y.hi) weight +1.0
- "since X" → one anchor [X.lo, +inf) weight +1.0
- "until X" → one anchor (-inf, X.hi) weight +1.0
- "in A and B" (disjoint) → TWO anchors, both +1.0 (graded coverage)
- "in A or B" (disjoint) → TWO anchors, both +1.0
- "in Q1 or Q4 of YYYY" → TWO anchors, both +1.0 (each quarter)
- "not in X" / "outside X" / "excluding X" → one anchor X weight -1.0
- "in A not in B" (B inside A) → +1.0 A, -1.0 B
- "mostly in A but consider B" → +1.0 A, +0.3 B

VERB-POLARITY RULE — CRITICAL (unchanged from current planner)
==============================================================
"not" / "didn't" / "did not" / "wasn't" attached to a VERB is EVENT
POLARITY, not temporal scoping. IGNORE it. Emit the same plan as if the
verb were affirmative.

  "what did not happen in 2024" — "not" attaches to verb "happen".
    → one anchor [2024], weight +1.0   (NOT -1.0)
  "what wasn't completed by March" — "wasn't" is verb polarity.
    → one anchor (-inf, March), weight +1.0

  Contrast with TEMPORAL-scoping negation:
  "what happened EXCLUDING 2024" → one anchor [2024], weight -1.0
  "what happened OUTSIDE 2024" → one anchor [2024], weight -1.0

EMPTY OUTPUT
============
Emit empty anchors when:
1. No temporal flavor: "how do I plan my morning?", "lessons from the launch"
2. Anaphoric refs with unknown date: "since the v3 launch"
3. "Most recently X" / "most recent X" with a specific subject:
   emit empty anchors with proximity_anchor="latest".
4. COMPARATIVE queries naming TWO alternatives ("before or after Y",
   "which came first, A or B"): empty anchors, proximity_anchor=null.
   Both events stay surfaced for the answer system to compare.

RECURRING / HABITUAL PATTERNS — past 3 + future 1
=================================================
For recurring queries (plural day-name, bare weekday/month/quarter, "every X"),
emit FOUR anchors, each weight +1.0, covering the 3 most-recent past
occurrences + 1 next upcoming occurrence relative to REF_TIME. Time-of-day
bands ("morning" 03-13, "afternoon" 11-19, "evening" 16-00+1, "night"
19-07+1, "noon" 10-14) apply per occurrence.

PROXIMITY ANCHOR
================
`proximity_anchor` is a SEPARATE channel for closeness ranking. Values:
- "latest" — prefer later in time (most recent)
- "earliest" — prefer earlier in time
- ISO date — prefer docs whose anchor is closest to that date
- null — no closeness scoring

Set proximity_anchor when the user is pointing at a SPECIFIC TIME-POINT
they want answers to cluster around. Do NOT fire proximity for:
- Pure set-membership ("in March 2024")
- Comparative queries naming two alternatives
- "Deictic-now context for past question" ("X happening today — any past cases?")

POINT-DAY rule: when an anchor's interval is ONE DAY (a specific
calendar day, e.g. [2024-03-15, 2024-03-16)), ALSO emit
proximity_anchor at the same date — catches metadata-anchored corpora.

EXAMPLES (neutral, illustrative — not from any dataset)
=======================================================

Query: "What was the inventory check on 2019-11-30?"
{{"anchors":[{{"interval":{{"lo":"2019-11-30","hi":"2019-12-01"}},"weight":1.0}}],"proximity_anchor":"2019-11-30"}}

Query: "Pull up notes from any time in 2023"
{{"anchors":[{{"interval":{{"lo":"2023-01-01","hi":"2024-01-01"}},"weight":1.0}}],"proximity_anchor":null}}

Query: "Anything from 2020 or 2024"
{{"anchors":[{{"interval":{{"lo":"2020-01-01","hi":"2021-01-01"}},"weight":1.0}},{{"interval":{{"lo":"2024-01-01","hi":"2025-01-01"}},"weight":1.0}}],"proximity_anchor":null}}

Query: "What didn't happen in March 2024?"   (verb-polarity)
{{"anchors":[{{"interval":{{"lo":"2024-03-01","hi":"2024-04-01"}},"weight":1.0}}],"proximity_anchor":null}}

Query: "What happened EXCLUDING March 2024?"   (true exclusion)
{{"anchors":[{{"interval":{{"lo":"2024-03-01","hi":"2024-04-01"}},"weight":-1.0}}],"proximity_anchor":null}}

Query: "Events in 2024 but not March"
{{"anchors":[{{"interval":{{"lo":"2024-01-01","hi":"2025-01-01"}},"weight":1.0}},{{"interval":{{"lo":"2024-03-01","hi":"2024-04-01"}},"weight":-1.0}}],"proximity_anchor":null}}

Query: "Anything around June 2022"
{{"anchors":[],"proximity_anchor":"2022-06-15"}}

Query: "Most recent inventory log entry"
{{"anchors":[],"proximity_anchor":"latest"}}

Query: "Did the design review come before or after the demo?"   (comparative)
{{"anchors":[],"proximity_anchor":null}}

Query: "Bakery stock count is off this morning — any past discrepancies?"   (deictic-now)
{{"anchors":[],"proximity_anchor":null}}

NOW PRODUCE THE PLAN FOR:

Query: {query}
Reference time: {ref_time}
"""


SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["anchors", "proximity_anchor"],
    "properties": {
        "anchors": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["interval", "weight"],
                "properties": {
                    "interval": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["lo", "hi"],
                        "properties": {
                            "lo": {"type": ["string", "null"]},
                            "hi": {"type": ["string", "null"]},
                        },
                    },
                    "weight": {"type": "number"},
                },
            },
        },
        "proximity_anchor": {"type": ["string", "null"]},
    },
}


# ============================================================================
# WEIGHTED PLAN + PLANNER
# ============================================================================


@dataclass
class WeightedAnchor:
    interval: Interval
    weight: float


@dataclass
class WeightedPlan:
    anchors: list[WeightedAnchor] = field(default_factory=list)
    proximity_anchor_us: Endpoint | None = None
    raw: str | None = field(default=None, repr=False)
    parse_error: str | None = field(default=None, repr=False)


def _interval_from_json(j: dict) -> Interval | None:
    lo_s = j.get("lo")
    hi_s = j.get("hi")
    try:
        lo = NEG_INF if lo_s is None else _iso_to_us(lo_s)
        hi = POS_INF if hi_s is None else _iso_to_us(hi_s)
    except ValueError:
        return None
    if lo >= hi:
        return None
    return Interval(lo, hi)


def _json_to_weighted_plan(obj: dict) -> tuple[list[WeightedAnchor], Endpoint | None]:
    anchors: list[WeightedAnchor] = []
    for a in obj.get("anchors", []):
        iv = _interval_from_json(a.get("interval", {}))
        if iv is None:
            continue
        weight = float(a.get("weight", 1.0))
        anchors.append(WeightedAnchor(iv, weight))
    pa = _proximity_from_str(obj.get("proximity_anchor"))
    return anchors, pa


_CACHE_DIR = (
    Path("/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation") /
    "temporal_retrieval_tr" / "cache" / "weighted_planner"
)
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_FILE = _CACHE_DIR / "llm_plan_cache.json"


def _cache_key(query: str, ref_time: str) -> str:
    h = hashlib.sha256()
    h.update(MODEL.encode())
    h.update(b"|")
    h.update(PROMPT_VERSION.encode())
    h.update(b"|")
    h.update(query.encode())
    h.update(b"|")
    h.update(ref_time.encode())
    return h.hexdigest()


class WeightedQueryPlanner:
    PER_CALL_TIMEOUT_S = 45.0
    CONCURRENCY = 8

    def __init__(self) -> None:
        self._client = AsyncOpenAI(timeout=self.PER_CALL_TIMEOUT_S)
        self._sem = asyncio.Semaphore(self.CONCURRENCY)
        if _CACHE_FILE.exists():
            try:
                self._cache = json.loads(_CACHE_FILE.read_text())
            except Exception:
                self._cache = {}
        else:
            self._cache = {}

    def _save_cache(self) -> None:
        import fcntl
        try:
            lock_path = _CACHE_FILE.with_suffix(_CACHE_FILE.suffix + ".lock")
            with open(lock_path, "w") as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    disk: dict = {}
                    if _CACHE_FILE.exists():
                        try:
                            disk = json.loads(_CACHE_FILE.read_text())
                        except Exception:
                            disk = {}
                    disk.update(self._cache)
                    self._cache = disk
                    tmp = _CACHE_FILE.with_suffix(_CACHE_FILE.suffix + ".tmp")
                    tmp.write_text(json.dumps(self._cache))
                    tmp.replace(_CACHE_FILE)
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

    async def plan(self, query: str, ref_time: str) -> WeightedPlan:
        key = _cache_key(query, ref_time)
        if key in self._cache:
            obj = self._cache[key]
            try:
                anchors, pa = _json_to_weighted_plan(obj)
                return WeightedPlan(
                    anchors=anchors, proximity_anchor_us=pa,
                    raw=json.dumps(obj),
                )
            except Exception:
                pass

        prompt = WEIGHTED_PROMPT.format(query=query, ref_time=ref_time)
        format_config: ResponseFormatTextJSONSchemaConfigParam = {
            "type": "json_schema", "name": "weighted_plan", "strict": True,
            "schema": SCHEMA,
        }
        text_config: ResponseTextConfigParam = {"format": format_config}
        async with self._sem:
            try:
                resp = await self._client.responses.create(
                    model=MODEL, input=prompt, text=text_config,
                )
                raw = resp.output_text
                obj = json.loads(raw)
                anchors, pa = _json_to_weighted_plan(obj)
                self._cache[key] = obj
                self._save_cache()
                return WeightedPlan(
                    anchors=anchors, proximity_anchor_us=pa, raw=raw,
                )
            except Exception as e:
                return WeightedPlan(parse_error=str(e), raw="")

    def stats(self) -> dict:
        return {"model": MODEL, "version": PROMPT_VERSION}


# ============================================================================
# WEIGHTED RETRIEVER
# ============================================================================


def weighted_match_score(
    anchors: list[WeightedAnchor],
    doc_anchors: list[IntervalSet],
) -> float:
    """Weighted sum: Σ weight_i · best_overlap_i.

    For each weighted anchor, take the best overlap with any doc anchor.
    Sum up weight · best_overlap. Doc can have positive (net match),
    negative (excluded by negative weights), or zero score.
    """
    if not anchors or not doc_anchors:
        return 0.0
    total = 0.0
    for wa in anchors:
        ivset = IntervalSet.from_intervals([wa.interval])
        best = 0.0
        for d in doc_anchors:
            f = pair_overlap(ivset, d)
            if f > best:
                best = f
        total += wa.weight * best
    return total


class WeightedTemporalRetriever(TemporalRetriever):
    """Reuses TemporalRetriever for index/embed/rerank; overrides query()
    to use weighted-anchor match scoring + WeightedPlan.

    Heterogeneous-match handling is preserved as a separate concern, same
    as the proximity-anchor pipeline. Pool admission rule:
    - If the plan has any POSITIVE-weight anchor (bounded scope present),
      a doc is eligible iff its weighted_match_score > 0.
    - If no positive-weight bounded anchor → no membership filter;
      semantic+proximity decide.
    """

    def __init__(self, weighted_planner: WeightedQueryPlanner, **kwargs):
        # Pass a dummy planner to satisfy parent; we override query() entirely.
        super().__init__(**kwargs)
        self._weighted_planner = weighted_planner

    async def query(self, query: str, ref_time: str, k: int = 10) -> list[Result]:
        plan: WeightedPlan = await self._weighted_planner.plan(query, ref_time)
        anchors = plan.anchors

        # Has bounded positive-weight scope? Determines timeless admission.
        has_pos_bounded = any(
            a.weight > 0
            and not (
                is_inf(a.interval.earliest_us) and is_inf(a.interval.latest_us)
            )
            and not (
                isinstance(a.interval.earliest_us, type(NEG_INF))
                and a.interval.earliest_us == NEG_INF
                and isinstance(a.interval.latest_us, type(POS_INF))
                and a.interval.latest_us == POS_INF
            )
            for a in anchors
        )

        # Semantic
        q_emb = (await self.embed_fn([query]))[0]
        q_emb = np.asarray(q_emb, dtype=np.float32)
        sem_scores = self._cosine_all(q_emb)

        all_dids = list(self._doc_ref_us.keys())
        match_all: dict[str, float] = {}
        eligible: list[str] = []
        timeless_in_scope: set[str] = set()

        for did in all_dids:
            d_anchors = self._doc_anchors.get(did, [])
            if not d_anchors:
                if not anchors or not has_pos_bounded:
                    match_all[did] = 1.0
                    eligible.append(did)
                else:
                    if isinstance(self.timeless_match_in_scope, str):
                        timeless_in_scope.add(did)
                    else:
                        match_all[did] = self.timeless_match_in_scope
            else:
                s = weighted_match_score(anchors, d_anchors)
                match_all[did] = s
                if s > 0.0:
                    eligible.append(did)

        pool = build_pool(sem_scores, all_dids, eligible, self.pool_size)
        if not pool:
            return []

        pool_texts = [self._docs[did].text for did in pool]
        rerank_scores = await self.rerank_fn(query, pool_texts)
        base = dict(zip(pool, rerank_scores, strict=False))

        if isinstance(self.timeless_match_in_scope, str) and \
                self.timeless_match_in_scope == "base":
            for did in pool:
                if did in timeless_in_scope:
                    match_all[did] = base.get(did, 0.0)

        if base:
            base_vals = list(base.values())
            pool_spread = max(base_vals) - min(base_vals)
        else:
            pool_spread = 1.0
        match_eff = {did: match_all.get(did, 0.0) * pool_spread for did in pool}

        if plan.proximity_anchor_us is not None and self.proximity_copeland:
            return self._copeland_proximity_rerank(
                pool, base, match_eff, plan.proximity_anchor_us, k,
            )

        anchored = {did for did in pool if self._doc_anchors.get(did)}
        return self._copeland_pairwise_rerank(
            pool, base, match_eff, anchored, k,
        )


# ============================================================================
# BENCH
# ============================================================================

# v7.1 proximity-anchor baseline (the shipped pipeline).
BASELINE_R1: dict[str, float] = {
    "adversarial": 0.800, "allen": 0.350, "ambiguous_year": 0.917,
    "ambiguous_year_adv": 0.833, "axis": 0.950, "causal_relative": 0.200,
    "composition": 0.400, "cotemporal": 0.950, "dense_cluster": 0.967,
    "disc": 0.667, "edge_conjunctive_temporal": 0.833, "edge_era_refs": 0.167,
    "edge_multi_te_doc": 1.000, "edge_relative_time": 0.917,
    "engagement_disjoint": 0.800, "era": 0.900, "goldilocks": 0.933,
    "goldilocks_v2": 0.667, "hard_bench": 0.960, "hard_dense_cluster": 1.000,
    "latest_recent": 1.000, "lattice": 1.000, "mixed_cue": 0.975,
    "negation_temporal": 0.800, "notin_multi_interval": 0.250,
    "open_ended_date": 0.800, "polarity": 0.933, "precedents": 1.000,
    "realq": 0.769, "realq_deictic": 1.000, "realq_v2": 0.912,
    "sensitivity_curated": 0.727, "speculative_anchors": 0.500,
    "temporal_essential": 1.000, "timeless_policies": 0.467,
    "utterance": 0.800, "v7_compound_hard": 0.944, "v7_doc_directional": 0.750,
    "same_topic_recency": 0.967, "same_topic_recency_hard": 0.967,
    "recency_stress_deep": 1.000, "recency_vs_rerank": 0.450,
    "state_vs_event": 1.000, "state_vs_event_v2": 0.960,
    "comparative_recency": 0.917, "metadata_only": 0.500,
    "deictic_in_content": 0.857,
}
BASELINE_R5: dict[str, float] = {
    "adversarial": 0.914, "allen": 1.000, "ambiguous_year": 1.000,
    "ambiguous_year_adv": 0.917, "axis": 1.000, "causal_relative": 1.000,
    "composition": 0.720, "cotemporal": 1.000, "dense_cluster": 1.000,
    "disc": 0.767, "edge_conjunctive_temporal": 1.000, "edge_era_refs": 1.000,
    "edge_multi_te_doc": 1.000, "edge_relative_time": 1.000,
    "engagement_disjoint": 0.900, "era": 1.000, "goldilocks": 1.000,
    "goldilocks_v2": 1.000, "hard_bench": 1.000, "hard_dense_cluster": 1.000,
    "latest_recent": 1.000, "lattice": 1.000, "mixed_cue": 1.000,
    "negation_temporal": 0.933, "notin_multi_interval": 1.000,
    "open_ended_date": 0.867, "polarity": 1.000, "precedents": 1.000,
    "realq": 1.000, "realq_deictic": 1.000, "realq_v2": 1.000,
    "sensitivity_curated": 0.818, "speculative_anchors": 1.000,
    "temporal_essential": 1.000, "timeless_policies": 1.000,
    "utterance": 0.900, "v7_compound_hard": 1.000, "v7_doc_directional": 1.000,
    "same_topic_recency": 1.000, "same_topic_recency_hard": 1.000,
    "recency_stress_deep": 1.000, "recency_vs_rerank": 0.600,
    "state_vs_event": 1.000, "state_vs_event_v2": 1.000,
    "comparative_recency": 1.000, "metadata_only": 0.786,
    "deictic_in_content": 1.000,
}


async def run_bench(bench, extractor, weighted_planner, embed_fn, rerank_fn):
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = WeightedTemporalRetriever(
        weighted_planner=weighted_planner,
        embed_fn=embed_fn, rerank_fn=rerank_fn,
        extractor=extractor,
        ranking_method="copeland_pairwise",
    )
    await vd.index(docs)
    rk = {
        q["query_id"]: [
            x.doc_id for x in await vd.query(q["text"], q["ref_time"], k=10)
        ]
        for q in queries
    }
    m = metrics(rk, gold)
    del vd, docs
    gc.collect()
    return m


async def main():
    raw = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    extractor = TemporalExtractor()
    weighted_planner = WeightedQueryPlanner()

    print("=== Weighted-anchor pipeline (WA-v1) vs v7.1 proximity-anchor baseline ===",
          flush=True)
    print(f"{'bench':30s}  {'R@1':>6s} {'Δ':>8s}  {'R@5':>6s} {'Δ':>8s}",
          flush=True)
    print("-" * 70, flush=True)

    rows: list[tuple[str, dict]] = []
    d1_total = 0.0
    d5_total = 0.0
    n_used = 0
    for bench in BENCH_NAMES:
        try:
            m = await run_bench(bench, extractor, weighted_planner,
                                embed_fn, rerank_fn)
        except Exception as e:
            print(f"  ERROR {bench}: {e}", flush=True)
            continue
        if m is None:
            continue
        rows.append((bench, m))
        b1 = BASELINE_R1.get(bench)
        b5 = BASELINE_R5.get(bench)
        if b1 is None or b5 is None:
            print(f"  {bench:30s}  {m['R@1']:.3f}        {m['R@5']:.3f}",
                  flush=True)
            continue
        d1 = m['R@1'] - b1
        d5 = m['R@5'] - b5
        print(
            f"  {bench:30s}  {m['R@1']:.3f} ({d1:+.3f})  "
            f"{m['R@5']:.3f} ({d5:+.3f})",
            flush=True,
        )
        d1_total += d1
        d5_total += d5
        n_used += 1

    if n_used > 0:
        print()
        print(f"MACRO Δ over {n_used} benches:  "
              f"ΔR@1={d1_total/n_used:+.4f}  ΔR@5={d5_total/n_used:+.4f}",
              flush=True)


if __name__ == "__main__":
    asyncio.run(main())
