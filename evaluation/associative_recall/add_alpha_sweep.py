"""Additive-only embedding steering alpha sweep on LME-hard-POC (30 Q).

Prior v2 add-only at alpha=0.1 showed mild negatives (-0.45pp vs v1,
and on this POC R@50 = 0.8040 vs baseline 0.8169 -> -1.3pp). This sweep
asks: does a smaller alpha land in positive territory?

Variants (all ADD-only, no subtract):
    add_a0.01_1r : alpha=0.01, 1 round
    add_a0.03_1r : alpha=0.03, 1 round
    add_a0.05_1r : alpha=0.05, 1 round
    add_a0.1_1r  : alpha=0.1,  1 round (sanity check vs prior)
    add_a0.2_1r  : alpha=0.2,  1 round
    add_a0.05_3r : alpha=0.05, 3 rounds (small nudge iterated)

Plus a formulation variant:
    add_score_merge : no embedding arithmetic; run 3 separate retrievals
        using embed(add_phrase_i) as probes and merge by sum-of-cosine-score
        with the baseline retrieval.

Baseline:
    add_baseline : v2f speaker-format direct retrieve (round 0 only).

Reuses existing steerv2 LLM + embedding caches so the round-1 LLM prompt
(alpha-independent) is a cache-hit for every variant.

Writes:
    results/add_alpha_sweep.json
    results/add_alpha_sweep.md
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openai
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)

from active_steering import EmbeddingCache, _query_em_by_vector, cached_embed
from active_steering_v2 import (
    STEER_V2_PROMPT,
    _format_retrieved_snippet_indexed,
    _parse_v2_json,
)
from em_architectures import (
    EMHit,
    V2F_MODEL,
    _MergedLLMCache,
    _dedupe_by_turn_id,
    _query_em,
    format_primer_context,
)
from em_lme_tuned_cues import (
    LMETUNE_V2F_MIXED7030_CACHE,
    V2F_LME_MIXED_7030_PROMPT,
    parse_speaker_cues,
)


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

ASSOC_DIR = Path(__file__).resolve().parent
DATA_DIR = ASSOC_DIR / "data"
RESULTS_DIR = ASSOC_DIR / "results"
CACHE_DIR = ASSOC_DIR / "cache"

LME_QUESTIONS_FILE = DATA_DIR / "questions_longmemeval_hard.json"
LME_COLLECTIONS_FILE = RESULTS_DIR / "em_lme_hard_collections.json"

OUT_JSON = RESULTS_DIR / "add_alpha_sweep.json"
OUT_MD = RESULTS_DIR / "add_alpha_sweep.md"

# Dedicated caches per standing convention (addsweep_*_cache.json).
ADDSWEEP_LLM_CACHE = CACHE_DIR / "addsweep_llm_cache.json"
ADDSWEEP_EMB_CACHE = CACHE_DIR / "addsweep_embedding_cache.json"

# Reuse the prior v2 steering caches as readers so we benefit from warm
# round-1 LLM + phrase embeddings.
STEERV2_LLM_CACHE = CACHE_DIR / "steerv2_llm_cache.json"
STEERV2_EMB_CACHE = CACHE_DIR / "steerv2_embedding_cache.json"

BUDGETS = (20, 50)
USER_PREFIX = "User: "
EXPAND_CONTEXT = 3
TOPK_FOR_LLM = 5
VECTOR_SEARCH_LIMIT = 50


def _ensure_user_prefix(text: str) -> str:
    t = text.lstrip()
    low = t.lower()
    if low.startswith("user:") or low.startswith("assistant:"):
        return t
    return USER_PREFIX + t


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# --------------------------------------------------------------------------
# Variant specs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    name: str
    mode: str  # "baseline" | "arithmetic" | "score_merge"
    alpha: float = 0.0
    rounds: int = 0


VARIANTS: list[Variant] = [
    Variant("add_baseline", "baseline"),
    Variant("add_a0.01_1r", "arithmetic", alpha=0.01, rounds=1),
    Variant("add_a0.03_1r", "arithmetic", alpha=0.03, rounds=1),
    Variant("add_a0.05_1r", "arithmetic", alpha=0.05, rounds=1),
    Variant("add_a0.1_1r",  "arithmetic", alpha=0.10, rounds=1),
    Variant("add_a0.2_1r",  "arithmetic", alpha=0.20, rounds=1),
    Variant("add_a0.05_3r", "arithmetic", alpha=0.05, rounds=3),
    Variant("add_score_merge", "score_merge", rounds=1),
]


# --------------------------------------------------------------------------
# Helpers reused from steerv2_eval
# --------------------------------------------------------------------------


async def _load_or_build_lme_v2f_cue(
    memory: EventMemory,
    question: str,
    v2f_cache: _MergedLLMCache,
    openai_client,
) -> tuple[str, list[str]]:
    prefixed_q = _ensure_user_prefix(question)
    primer_hits = _dedupe_by_turn_id(
        await _query_em(memory, prefixed_q, vector_search_limit=10, expand_context=0)
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text} for h in primer_hits
    ]
    context_section = format_primer_context(primer_segments)
    prompt = V2F_LME_MIXED_7030_PROMPT.format(
        question=question, context_section=context_section
    )
    cached = v2f_cache.get(V2F_MODEL, prompt)
    if cached is None:
        resp = await openai_client.chat.completions.create(
            model=V2F_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        cached = resp.choices[0].message.content or ""
        v2f_cache.put(V2F_MODEL, prompt, cached)
    cues = parse_speaker_cues(cached, max_cues=3)
    initial = cues[0] if cues else prefixed_q
    return initial, cues


async def _llm_steer_call(
    *,
    query_text: str,
    initial_cue_text: str,
    retrieved_str: str,
    openai_client,
    llm_cache: _MergedLLMCache,
) -> dict:
    prompt = STEER_V2_PROMPT.format(
        query=query_text,
        cue=initial_cue_text,
        topk=TOPK_FOR_LLM,
        retrieved_turns=retrieved_str,
    )
    cached = llm_cache.get(V2F_MODEL, prompt)
    if cached is None:
        resp = await openai_client.chat.completions.create(
            model=V2F_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        cached = resp.choices[0].message.content or ""
        llm_cache.put(V2F_MODEL, prompt, cached)
    return _parse_v2_json(cached)


def compute_recall(retrieved: set[int], gold: set[int]) -> float:
    if not gold:
        return 1.0
    return len(retrieved & gold) / len(gold)


# --------------------------------------------------------------------------
# Arithmetic (ADD-only) steering loop
# --------------------------------------------------------------------------


async def run_add_arithmetic(
    memory: EventMemory,
    *,
    query_text: str,
    initial_cue_text: str,
    alpha: float,
    rounds: int,
    embedder: OpenAIEmbedder,
    openai_client,
    llm_cache: _MergedLLMCache,
    emb_cache: EmbeddingCache,
    gold: set[int],
) -> dict:
    cue_vec = await cached_embed(embedder, initial_cue_text, emb_cache=emb_cache)
    probe = _normalize(np.array(cue_vec, dtype=np.float64))
    probe_0 = probe.copy()

    # Round 0 retrieve.
    r0_hits = _dedupe_by_turn_id(
        await _query_em_by_vector(
            memory,
            probe,
            vector_search_limit=VECTOR_SEARCH_LIMIT,
            expand_context=EXPAND_CONTEXT,
        )
    )

    trajectory: list[dict] = []
    r0_recall: dict[str, float] = {}
    for K in BUDGETS:
        r0_recall[f"r@{K}"] = round(
            compute_recall({h.turn_id for h in r0_hits[:K]}, gold), 4
        )
    trajectory.append({
        "round": 0,
        "add_magnitude": 0.0,
        "probe_drift": 1.0,
        "add_phrases": [],
        **r0_recall,
    })

    current_hits = r0_hits
    last_hits = r0_hits
    for rnd in range(1, rounds + 1):
        topk_hits = current_hits[:TOPK_FOR_LLM]
        retrieved_str = _format_retrieved_snippet_indexed(topk_hits, n=TOPK_FOR_LLM)
        parsed = await _llm_steer_call(
            query_text=query_text,
            initial_cue_text=initial_cue_text,
            retrieved_str=retrieved_str,
            openai_client=openai_client,
            llm_cache=llm_cache,
        )
        add_phrases = parsed["add_phrases"]

        add_sum = np.zeros_like(probe)
        for phrase in add_phrases:
            vec = await cached_embed(embedder, phrase, emb_cache=emb_cache)
            add_sum += _normalize(np.array(vec, dtype=np.float64))

        add_mag = float(alpha * np.linalg.norm(add_sum))
        probe = _normalize(probe + alpha * add_sum)
        drift = _cosine(probe, probe_0)

        next_hits = _dedupe_by_turn_id(
            await _query_em_by_vector(
                memory,
                probe,
                vector_search_limit=VECTOR_SEARCH_LIMIT,
                expand_context=EXPAND_CONTEXT,
            )
        )
        last_hits = next_hits

        rnd_recall = {
            f"r@{K}": round(
                compute_recall({h.turn_id for h in next_hits[:K]}, gold), 4
            )
            for K in BUDGETS
        }
        trajectory.append({
            "round": rnd,
            "add_magnitude": round(add_mag, 4),
            "probe_drift": round(drift, 4),
            "add_phrases": list(add_phrases),
            **rnd_recall,
        })

        if not add_phrases:
            break

        current_hits = next_hits

    final = {
        f"r@{K}": round(
            compute_recall({h.turn_id for h in last_hits[:K]}, gold), 4
        )
        for K in BUDGETS
    }
    return {
        "trajectory": trajectory,
        "final_hits": last_hits,
        **final,
    }


# --------------------------------------------------------------------------
# Baseline: round 0 only
# --------------------------------------------------------------------------


async def run_baseline(
    memory: EventMemory,
    *,
    initial_cue_text: str,
    embedder: OpenAIEmbedder,
    emb_cache: EmbeddingCache,
    gold: set[int],
) -> dict:
    cue_vec = await cached_embed(embedder, initial_cue_text, emb_cache=emb_cache)
    probe = _normalize(np.array(cue_vec, dtype=np.float64))
    r0_hits = _dedupe_by_turn_id(
        await _query_em_by_vector(
            memory,
            probe,
            vector_search_limit=VECTOR_SEARCH_LIMIT,
            expand_context=EXPAND_CONTEXT,
        )
    )
    final = {
        f"r@{K}": round(
            compute_recall({h.turn_id for h in r0_hits[:K]}, gold), 4
        )
        for K in BUDGETS
    }
    return {
        "trajectory": [
            {"round": 0, "add_magnitude": 0.0, "probe_drift": 1.0,
             "add_phrases": [], **final}
        ],
        "final_hits": r0_hits,
        **final,
    }


# --------------------------------------------------------------------------
# Score-merge variant: retrieve separately for each add phrase,
# merge by sum of cosine scores with baseline retrieval.
# --------------------------------------------------------------------------


async def run_score_merge(
    memory: EventMemory,
    *,
    query_text: str,
    initial_cue_text: str,
    embedder: OpenAIEmbedder,
    openai_client,
    llm_cache: _MergedLLMCache,
    emb_cache: EmbeddingCache,
    gold: set[int],
) -> dict:
    # Baseline retrieval (probe = embed(initial cue)).
    cue_vec = await cached_embed(embedder, initial_cue_text, emb_cache=emb_cache)
    probe = _normalize(np.array(cue_vec, dtype=np.float64))
    base_hits = _dedupe_by_turn_id(
        await _query_em_by_vector(
            memory,
            probe,
            vector_search_limit=VECTOR_SEARCH_LIMIT,
            expand_context=EXPAND_CONTEXT,
        )
    )

    # Use baseline's top-5 as context for the SAME STEER_V2_PROMPT that the
    # arithmetic variants use; this lets us share the cache.
    topk_hits = base_hits[:TOPK_FOR_LLM]
    retrieved_str = _format_retrieved_snippet_indexed(topk_hits, n=TOPK_FOR_LLM)
    parsed = await _llm_steer_call(
        query_text=query_text,
        initial_cue_text=initial_cue_text,
        retrieved_str=retrieved_str,
        openai_client=openai_client,
        llm_cache=llm_cache,
    )
    add_phrases = parsed["add_phrases"]

    # Accumulate sum-of-cosine-scores per turn_id across baseline + per-phrase
    # retrievals. Each retrieval contributes its hit's score where present.
    score_by_turn: dict[int, float] = defaultdict(float)
    hit_by_turn: dict[int, EMHit] = {}
    for h in base_hits:
        score_by_turn[h.turn_id] += float(h.score)
        if h.turn_id not in hit_by_turn:
            hit_by_turn[h.turn_id] = h

    for phrase in add_phrases:
        p_vec = await cached_embed(embedder, phrase, emb_cache=emb_cache)
        p_norm = _normalize(np.array(p_vec, dtype=np.float64))
        p_hits = _dedupe_by_turn_id(
            await _query_em_by_vector(
                memory,
                p_norm,
                vector_search_limit=VECTOR_SEARCH_LIMIT,
                expand_context=EXPAND_CONTEXT,
            )
        )
        for h in p_hits:
            score_by_turn[h.turn_id] += float(h.score)
            if h.turn_id not in hit_by_turn:
                hit_by_turn[h.turn_id] = h

    merged = sorted(
        hit_by_turn.values(),
        key=lambda h: -score_by_turn[h.turn_id],
    )

    final = {
        f"r@{K}": round(
            compute_recall({h.turn_id for h in merged[:K]}, gold), 4
        )
        for K in BUDGETS
    }
    return {
        "trajectory": [
            {"round": 0, "add_magnitude": 0.0, "probe_drift": 1.0,
             "add_phrases": [], **{
                 f"r@{K}": round(
                     compute_recall({h.turn_id for h in base_hits[:K]}, gold), 4
                 )
                 for K in BUDGETS
             }},
            {"round": 1, "add_magnitude": 0.0, "probe_drift": 1.0,
             "add_phrases": list(add_phrases), **final},
        ],
        "final_hits": merged,
        "n_phrases_merged": len(add_phrases),
        **final,
    }


# --------------------------------------------------------------------------
# Per-question driver
# --------------------------------------------------------------------------


async def run_one_question(
    variant: Variant,
    memory: EventMemory,
    question: dict,
    *,
    embedder: OpenAIEmbedder,
    openai_client,
    v2f_cache: _MergedLLMCache,
    steer_llm_cache: _MergedLLMCache,
    emb_cache: EmbeddingCache,
) -> dict:
    q_text = question["question"]
    gold = set(question.get("source_chat_ids", []))

    t0 = time.monotonic()
    initial_text, v2f_cues = await _load_or_build_lme_v2f_cue(
        memory, q_text, v2f_cache, openai_client
    )

    if variant.mode == "baseline":
        core = await run_baseline(
            memory,
            initial_cue_text=initial_text,
            embedder=embedder,
            emb_cache=emb_cache,
            gold=gold,
        )
    elif variant.mode == "arithmetic":
        core = await run_add_arithmetic(
            memory,
            query_text=q_text,
            initial_cue_text=initial_text,
            alpha=variant.alpha,
            rounds=variant.rounds,
            embedder=embedder,
            openai_client=openai_client,
            llm_cache=steer_llm_cache,
            emb_cache=emb_cache,
            gold=gold,
        )
    elif variant.mode == "score_merge":
        core = await run_score_merge(
            memory,
            query_text=q_text,
            initial_cue_text=initial_text,
            embedder=embedder,
            openai_client=openai_client,
            llm_cache=steer_llm_cache,
            emb_cache=emb_cache,
            gold=gold,
        )
    else:
        raise ValueError(variant.mode)

    elapsed = time.monotonic() - t0

    row: dict = {
        "question_id": question.get("question_id"),
        "category": question.get("category", "unknown"),
        "question": q_text,
        "initial_cue_text": initial_text,
        "v2f_cues": v2f_cues,
        "time_s": round(elapsed, 3),
        "trajectory": core["trajectory"],
    }
    for K in BUDGETS:
        row[f"r@{K}"] = core[f"r@{K}"]
    if "n_phrases_merged" in core:
        row["n_phrases_merged"] = core["n_phrases_merged"]
    return row


# --------------------------------------------------------------------------
# Load questions (same as steerv2_eval)
# --------------------------------------------------------------------------


def load_lme_questions(limit_per_cat: int = 10) -> list[dict]:
    with open(LME_QUESTIONS_FILE) as f:
        qs = json.load(f)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for q in qs:
        by_cat[q.get("category", "unknown")].append(q)
    out: list[dict] = []
    for cat in sorted(by_cat):
        out.extend(by_cat[cat][:limit_per_cat])
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


async def main() -> None:
    lme_qs = load_lme_questions(limit_per_cat=10)
    print(
        f"[add_alpha_sweep] n_questions={len(lme_qs)} variants={len(VARIANTS)}",
        flush=True,
    )

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    engines = []
    segment_stores = []

    openai_client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model="text-embedding-3-small",
            dimensions=1536,
            max_input_length=8192,
        )
    )

    # Caches: read from both the dedicated addsweep caches AND the prior
    # steerv2 caches, so round-1 LLM + phrase embeddings are warm.
    lme_v2f_cache = _MergedLLMCache(
        reader_paths=[LMETUNE_V2F_MIXED7030_CACHE],
        writer_path=LMETUNE_V2F_MIXED7030_CACHE,
    )
    steer_llm_cache = _MergedLLMCache(
        reader_paths=[STEERV2_LLM_CACHE, ADDSWEEP_LLM_CACHE],
        writer_path=ADDSWEEP_LLM_CACHE,
    )
    # Embedding cache: pre-seed addsweep from steerv2 entries on disk
    # (EmbeddingCache only reads from its own path), so we merge the two.
    emb_cache = EmbeddingCache(ADDSWEEP_EMB_CACHE)
    if STEERV2_EMB_CACHE.exists():
        try:
            with open(STEERV2_EMB_CACHE) as f:
                prior = json.load(f)
            for k, v in prior.items():
                if k not in emb_cache._cache:  # noqa: SLF001
                    emb_cache._cache[k] = v  # noqa: SLF001
        except Exception:
            pass

    memories: dict[str, EventMemory] = {}
    opened: list = []
    try:
        with open(LME_COLLECTIONS_FILE) as f:
            lme_meta = json.load(f)
        qid_to_meta = {r["question_id"]: r for r in lme_meta["questions"]}
        sql_url = lme_meta["sql_url"]
        if sql_url.startswith("sqlite"):
            engine = create_async_engine(sql_url)
        else:
            engine = create_async_engine(sql_url, pool_size=20, max_overflow=20)
        engines.append(engine)
        seg_store = SQLAlchemySegmentStore(
            SQLAlchemySegmentStoreParams(engine=engine)
        )
        await seg_store.startup()
        segment_stores.append(seg_store)

        for q in lme_qs:
            qm = qid_to_meta[q["question_id"]]
            coll = await vector_store.open_collection(
                namespace=qm["namespace"], name=qm["collection_name"]
            )
            part = await seg_store.open_or_create_partition(qm["partition_key"])
            mem = EventMemory(
                EventMemoryParams(
                    vector_store_collection=coll,
                    segment_store_partition=part,
                    embedder=embedder,
                    reranker=None,
                    derive_sentences=False,
                    max_text_chunk_length=500,
                )
            )
            memories[q["question_id"]] = mem
            opened.append((seg_store, coll, part))

        results: dict = {"variants": {}, "budgets": list(BUDGETS)}
        sem = asyncio.Semaphore(8)

        async def run_wrap(variant: Variant, q: dict):
            async with sem:
                mem = memories[q["question_id"]]
                return await run_one_question(
                    variant,
                    mem,
                    q,
                    embedder=embedder,
                    openai_client=openai_client,
                    v2f_cache=lme_v2f_cache,
                    steer_llm_cache=steer_llm_cache,
                    emb_cache=emb_cache,
                )

        for variant in VARIANTS:
            t_start = time.monotonic()
            tasks = [run_wrap(variant, q) for q in lme_qs]
            rows = await asyncio.gather(*tasks)
            elapsed = time.monotonic() - t_start

            n = len(rows)
            summary = {"n": n, "time_s": round(elapsed, 1)}
            for K in BUDGETS:
                summary[f"mean_r@{K}"] = round(
                    sum(r[f"r@{K}"] for r in rows) / max(n, 1), 4
                )

            by_cat: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                by_cat[r.get("category", "unknown")].append(r)
            cat_summary: dict[str, dict] = {}
            for cat, cr in by_cat.items():
                d = {"n": len(cr)}
                for K in BUDGETS:
                    d[f"mean_r@{K}"] = round(
                        sum(r[f"r@{K}"] for r in cr) / max(len(cr), 1), 4
                    )
                cat_summary[cat] = d

            # Trajectory aggregates per round.
            drift_sum: dict[int, list[float]] = defaultdict(list)
            add_mag_sum: dict[int, list[float]] = defaultdict(list)
            recall_traj: dict[int, dict[str, list[float]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for r in rows:
                for tr in r["trajectory"]:
                    rnd = tr["round"]
                    drift_sum[rnd].append(tr["probe_drift"])
                    add_mag_sum[rnd].append(tr["add_magnitude"])
                    for K in BUDGETS:
                        if f"r@{K}" in tr:
                            recall_traj[rnd][f"r@{K}"].append(tr[f"r@{K}"])

            traj_agg: dict[str, dict] = {}
            for rnd in sorted(drift_sum.keys()):
                entry: dict = {
                    "n": len(drift_sum[rnd]),
                    "mean_drift_vs_probe_0": round(
                        sum(drift_sum[rnd]) / max(len(drift_sum[rnd]), 1), 4
                    ),
                    "mean_add_mag": round(
                        sum(add_mag_sum[rnd]) / max(len(add_mag_sum[rnd]), 1), 4
                    ),
                }
                for K in BUDGETS:
                    vals = recall_traj[rnd][f"r@{K}"]
                    if vals:
                        entry[f"mean_r@{K}"] = round(sum(vals) / len(vals), 4)
                traj_agg[str(rnd)] = entry

            results["variants"][variant.name] = {
                "spec": {
                    "mode": variant.mode,
                    "alpha": variant.alpha,
                    "rounds": variant.rounds,
                },
                "summary": summary,
                "by_category": cat_summary,
                "trajectory": traj_agg,
                "per_question": rows,
            }

            # Save caches after each variant.
            lme_v2f_cache.save()
            steer_llm_cache.save()
            emb_cache.save()

            print(
                f"[{variant.name}] n={n} "
                f"r@20={summary['mean_r@20']:.4f} r@50={summary['mean_r@50']:.4f} "
                f"in {elapsed:.1f}s",
                flush=True,
            )

    finally:
        for seg_store, coll, part in opened:
            await seg_store.close_partition(part)
            await vector_store.close_collection(collection=coll)
        for seg_store in segment_stores:
            await seg_store.shutdown()
        await vector_store.shutdown()
        for engine in engines:
            await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUT_JSON}", flush=True)
    write_report(results)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def write_report(results: dict) -> None:
    variants = results["variants"]
    lines: list[str] = []
    lines.append("# Add-only Embedding Steering: Alpha Sweep")
    lines.append("")
    lines.append(
        "ADD-only evidence-grounded steering on LME-hard-POC (30 questions). "
        "Prior v2 add-only at alpha=0.1 hit -1.3pp (0.8040 vs 0.8169). This "
        "sweep asks: does a smaller alpha find positive territory, and does "
        "score-merge beat embedding arithmetic?"
    )
    lines.append("")
    lines.append(
        "Fixed: text-embedding-3-small, gpt-5-mini (via `em_v2f_lme_mixed_7030` + "
        "`expand_3`), 30 Qs, reuses steerv2 LLM/embedding caches."
    )
    lines.append("")

    # Main recall table
    lines.append("## Recall by variant")
    lines.append("")
    lines.append(
        "| Variant | mode | alpha | rounds | R@20 | R@50 | Δ R@20 vs baseline | Δ R@50 vs baseline |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    base_r20 = variants.get("add_baseline", {}).get("summary", {}).get("mean_r@20", 0.0)
    base_r50 = variants.get("add_baseline", {}).get("summary", {}).get("mean_r@50", 0.0)
    for name, per in variants.items():
        spec = per["spec"]
        s = per["summary"]
        d20 = s["mean_r@20"] - base_r20
        d50 = s["mean_r@50"] - base_r50
        alpha = f"{spec['alpha']:.2f}" if spec["alpha"] else "--"
        rounds = spec["rounds"] if spec["rounds"] else "--"
        lines.append(
            f"| `{name}` | {spec['mode']} | {alpha} | {rounds} | "
            f"{s['mean_r@20']:.4f} | {s['mean_r@50']:.4f} | "
            f"{d20:+.4f} | {d50:+.4f} |"
        )
    lines.append("")

    # Round-by-round R@50 (focus on add_a0.05_3r)
    lines.append("## Round-by-round R@50 (arithmetic variants)")
    lines.append("")
    lines.append("| Variant | rd 0 | rd 1 | rd 2 | rd 3 | drift@final |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name, per in variants.items():
        if per["spec"]["mode"] != "arithmetic":
            continue
        traj = per["trajectory"]
        cells = []
        for rnd in range(0, 4):
            entry = traj.get(str(rnd))
            if entry and "mean_r@50" in entry:
                cells.append(f"{entry['mean_r@50']:.4f}")
            else:
                cells.append("--")
        if traj:
            max_rnd = max(int(k) for k in traj.keys())
            final_drift = traj[str(max_rnd)].get("mean_drift_vs_probe_0", 1.0)
        else:
            final_drift = 1.0
        lines.append(
            f"| `{name}` | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{cells[3]} | {final_drift:.4f} |"
        )
    lines.append("")

    # Category breakdown at R@50
    lines.append("## Category R@50")
    lines.append("")
    lines.append(
        "| Variant | multi-session | single-session-preference | temporal-reasoning |"
    )
    lines.append("| --- | --- | --- | --- |")
    for name, per in variants.items():
        bc = per["by_category"]
        lines.append(
            f"| `{name}` | "
            f"{bc.get('multi-session', {}).get('mean_r@50', 0):.4f} | "
            f"{bc.get('single-session-preference', {}).get('mean_r@50', 0):.4f} | "
            f"{bc.get('temporal-reasoning', {}).get('mean_r@50', 0):.4f} |"
        )
    lines.append("")

    # Score-merge vs best arithmetic
    lines.append("## Score-merge vs arithmetic")
    lines.append("")
    sm = variants.get("add_score_merge", {}).get("summary", {})
    lines.append(
        f"- `add_score_merge` R@20 = {sm.get('mean_r@20', 0):.4f}, "
        f"R@50 = {sm.get('mean_r@50', 0):.4f}"
    )
    arith_best_r50 = 0.0
    arith_best_name = None
    for name, per in variants.items():
        if per["spec"]["mode"] != "arithmetic":
            continue
        r50 = per["summary"]["mean_r@50"]
        if r50 > arith_best_r50:
            arith_best_r50 = r50
            arith_best_name = name
    if arith_best_name:
        lines.append(
            f"- best arithmetic variant: `{arith_best_name}` R@50 = {arith_best_r50:.4f}"
        )
        lines.append(
            f"- Δ(score_merge - best_arith) R@50 = "
            f"{sm.get('mean_r@50', 0) - arith_best_r50:+.4f}"
        )
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    best_alpha_name = None
    best_delta_r50 = -1.0
    for name, per in variants.items():
        if per["spec"]["mode"] != "arithmetic":
            continue
        delta = per["summary"]["mean_r@50"] - base_r50
        if delta > best_delta_r50:
            best_delta_r50 = delta
            best_alpha_name = name
    lines.append(f"- baseline R@50: {base_r50:.4f}")
    lines.append(
        f"- best arithmetic: `{best_alpha_name}` (Δ R@50 = {best_delta_r50:+.4f})"
    )
    lines.append(
        f"- score-merge: `add_score_merge` "
        f"(Δ R@50 = {sm.get('mean_r@50', 0) - base_r50:+.4f})"
    )
    lines.append("")
    if best_delta_r50 >= 0.005:
        lines.append(
            f"**Additive arithmetic has a working regime.** Best alpha = "
            f"`{best_alpha_name}` beats baseline by {best_delta_r50 * 100:.2f}pp."
        )
    else:
        lines.append(
            "**Additive arithmetic is substrate-incompatible, just less "
            f"dramatically than subtractive** (best Δ R@50 = {best_delta_r50 * 100:+.2f}pp, "
            "nothing clears the +0.5pp bar)."
        )
    sm_vs_best = sm.get("mean_r@50", 0) - arith_best_r50
    if sm_vs_best >= 0.005:
        lines.append(
            f"Score-merge beats arithmetic by {sm_vs_best * 100:.2f}pp — "
            "confirms `additive = separate probe merged by score` is the "
            "right formulation, not embedding arithmetic. This is essentially "
            "what v2f already does."
        )
    else:
        lines.append(
            f"Score-merge does not meaningfully beat arithmetic (Δ = "
            f"{sm_vs_best * 100:+.2f}pp); the formulation choice is not the "
            "bottleneck."
        )
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- JSON: `{OUT_JSON.relative_to(ASSOC_DIR)}`")
    lines.append("- Source: `add_alpha_sweep.py` (framework files untouched)")
    lines.append(
        "- Caches: `cache/addsweep_llm_cache.json`, "
        "`cache/addsweep_embedding_cache.json` "
        "(reads from `steerv2_*` caches for warm hits)."
    )
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"Saved: {OUT_MD}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
