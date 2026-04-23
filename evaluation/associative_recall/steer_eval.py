"""Evaluate active embedding steering on LoCoMo-30 and LME-hard-30 (POC).

Variants (each run on both corpora):
  - steer_v2f_1round / _2round / _3round        : start with v2f cue, N rounds
  - steer_v2f_modelweight                       : 2 rounds, LLM chooses per-phrase magnitudes
  - steer_v2f_addonly                           : 2 rounds, ADD primitive only (ablation)
  - steer_v2f_subonly                           : 2 rounds, SUB primitive only (ablation)
  - steer_query_direct                          : start with raw query embedding, 2 rounds

Baselines (rerun with the same direct-vector retrieval path so comparisons are
apples-to-apples):
  - baseline_query_direct  : just embed(question) -> top-K (round 0 snapshot)
  - baseline_v2f_direct    : embed(v2f cue 1)  -> top-K (round 0 snapshot)

Outputs:
  results/active_steering.json
  results/active_steering.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

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

from active_steering import (
    STEER_EMB_CACHE,
    STEER_LLM_CACHE,
    EmbeddingCache,
    SteerConfig,
    active_steer,
)
from em_architectures import (
    BESTSHOT_LLM_CACHE,
    EM_V2F_LLM_CACHE,
    V2F_MODEL,
    V2F_PROMPT,
    _MergedLLMCache,
    _dedupe_by_turn_id,
    _query_em,
    format_primer_context,
    parse_v2f_cues,
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

LOCOMO_QUESTIONS_FILE = DATA_DIR / "questions_extended.json"
LME_QUESTIONS_FILE = DATA_DIR / "questions_longmemeval_hard.json"
LOCOMO_COLLECTIONS_FILE = RESULTS_DIR / "eventmemory_collections.json"
LME_COLLECTIONS_FILE = RESULTS_DIR / "em_lme_hard_collections.json"

OUT_JSON = RESULTS_DIR / "active_steering.json"
OUT_MD = RESULTS_DIR / "active_steering.md"

BUDGETS = (20, 50)
LOCOMO_CONV_IDS = ("locomo_conv-26", "locomo_conv-30", "locomo_conv-41")

USER_PREFIX = "User: "


def _ensure_user_prefix(text: str) -> str:
    t = text.lstrip()
    low = t.lower()
    if low.startswith("user:") or low.startswith("assistant:"):
        return t
    return USER_PREFIX + t


# --------------------------------------------------------------------------
# Initial-cue builders
# --------------------------------------------------------------------------


async def _load_or_build_locomo_v2f_cue(
    memory: EventMemory,
    question: str,
    v2f_cache: _MergedLLMCache,
    openai_client,
) -> tuple[str, list[str]]:
    """Return (initial_cue_text, v2f_cues_list).

    Uses existing em_v2f LoCoMo prompt (em_architectures.V2F_PROMPT). Primer
    top-10 built exactly as em_v2f does so prompt text matches the cached
    entries byte-for-byte.
    """
    primer_hits = _dedupe_by_turn_id(
        await _query_em(memory, question, vector_search_limit=10, expand_context=0)
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text} for h in primer_hits
    ]
    context_section = format_primer_context(primer_segments)
    prompt = V2F_PROMPT.format(question=question, context_section=context_section)

    cached = v2f_cache.get(V2F_MODEL, prompt)
    if cached is None:
        resp = await openai_client.chat.completions.create(
            model=V2F_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        cached = resp.choices[0].message.content or ""
        v2f_cache.put(V2F_MODEL, prompt, cached)
    cues = parse_v2f_cues(cached, max_cues=2)
    initial = cues[0] if cues else question
    return initial, cues


async def _load_or_build_lme_v2f_cue(
    memory: EventMemory,
    question: str,
    v2f_cache: _MergedLLMCache,
    openai_client,
) -> tuple[str, list[str]]:
    """LME variant: mixed_7030 prompt (best LME baseline). Cues already start
    with "User: ".
    """
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


# --------------------------------------------------------------------------
# Variant runner
# --------------------------------------------------------------------------


def compute_recall(retrieved: set[int], gold: set[int]) -> float:
    if not gold:
        return 1.0
    return len(retrieved & gold) / len(gold)


VARIANT_CONFIGS = {
    # v2f-initialized steering
    "steer_v2f_1round": dict(
        cue_source="v2f", max_rounds=1, weighted=False, use_add=True, use_sub=True
    ),
    "steer_v2f_2round": dict(
        cue_source="v2f", max_rounds=2, weighted=False, use_add=True, use_sub=True
    ),
    "steer_v2f_3round": dict(
        cue_source="v2f", max_rounds=3, weighted=False, use_add=True, use_sub=True
    ),
    "steer_v2f_modelweight": dict(
        cue_source="v2f", max_rounds=2, weighted=True, use_add=True, use_sub=True
    ),
    "steer_v2f_addonly": dict(
        cue_source="v2f", max_rounds=2, weighted=False, use_add=True, use_sub=False
    ),
    "steer_v2f_subonly": dict(
        cue_source="v2f", max_rounds=2, weighted=False, use_add=False, use_sub=True
    ),
    # Raw-query-initialized steering
    "steer_query_direct": dict(
        cue_source="query", max_rounds=2, weighted=False, use_add=True, use_sub=True
    ),
    # Baselines via direct-vector path (no steering rounds, just round 0)
    "baseline_query_direct": dict(
        cue_source="query", max_rounds=0, weighted=False, use_add=True, use_sub=True
    ),
    "baseline_v2f_direct": dict(
        cue_source="v2f", max_rounds=0, weighted=False, use_add=True, use_sub=True
    ),
}


def _expand_ctx_for_corpus(corpus: str) -> int:
    # LME recipe uses expand=3; LoCoMo uses expand=0.
    return 3 if corpus == "lme" else 0


async def run_one_question(
    variant: str,
    corpus: str,
    memory: EventMemory,
    question: dict,
    *,
    embedder: OpenAIEmbedder,
    openai_client,
    v2f_cache: _MergedLLMCache,
    steer_llm_cache: _MergedLLMCache,
    emb_cache: EmbeddingCache,
    max_K: int,
) -> dict:
    q_text = question["question"]
    gold = set(question.get("source_chat_ids", []))
    conf = VARIANT_CONFIGS[variant]

    t0 = time.monotonic()

    # Build initial cue text.
    if conf["cue_source"] == "v2f":
        if corpus == "locomo":
            initial_text, v2f_cues = await _load_or_build_locomo_v2f_cue(
                memory, q_text, v2f_cache, openai_client
            )
        else:
            initial_text, v2f_cues = await _load_or_build_lme_v2f_cue(
                memory, q_text, v2f_cache, openai_client
            )
    else:
        # raw query
        if corpus == "lme":
            initial_text = _ensure_user_prefix(q_text)
        else:
            initial_text = q_text
        v2f_cues = []

    config = SteerConfig(
        max_rounds=conf["max_rounds"],
        alpha=0.1,
        beta=0.1,
        topk_for_llm=5,
        vector_search_limit=max_K,
        expand_context=_expand_ctx_for_corpus(corpus),
        weighted_mode=conf["weighted"],
        use_add=conf["use_add"],
        use_sub=conf["use_sub"],
    )

    result = await active_steer(
        memory,
        initial_cue_text=initial_text,
        embedder=embedder,
        openai_client=openai_client,
        llm_cache=steer_llm_cache,
        emb_cache=emb_cache,
        config=config,
        gold=gold,
        K_budgets=BUDGETS,
    )

    elapsed = time.monotonic() - t0

    # Final recall from the last round snapshot.
    final_hits = result.hits_by_round[-1]
    row: dict = {
        "question_id": question.get("question_id") or question.get("conversation_id"),
        "conversation_id": question.get("conversation_id"),
        "category": question.get("category", "unknown"),
        "question": q_text,
        "initial_cue_text": initial_text,
        "v2f_cues": v2f_cues,
        "n_rounds": len(result.traces) - 1,
        "time_s": round(elapsed, 3),
    }
    for K in BUDGETS:
        retrieved = {h.turn_id for h in final_hits[:K]}
        row[f"r@{K}"] = round(compute_recall(retrieved, gold), 4)

    # Round-by-round trajectory.
    trajectory: list[dict] = []
    for t in result.traces:
        tr = {
            "round": t.round_idx,
            "add_magnitude": round(t.add_magnitude, 4),
            "sub_magnitude": round(t.sub_magnitude, 4),
            "probe_drift": round(t.probe_drift, 4),
            "add_phrases": [[p, round(w, 3)] for p, w in t.add_phrases],
            "sub_phrases": [[p, round(w, 3)] for p, w in t.sub_phrases],
        }
        tr.update({k: round(v, 4) for k, v in t.recall_deltas.items()})
        trajectory.append(tr)
    row["trajectory"] = trajectory

    return row


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def load_locomo_questions(limit: int | None = None) -> list[dict]:
    with open(LOCOMO_QUESTIONS_FILE) as f:
        qs = json.load(f)
    locomo = [q for q in qs if q.get("benchmark") == "locomo"][:30]
    if limit is not None:
        return locomo[:limit]
    return locomo


def load_lme_questions(limit_per_cat: int = 10) -> list[dict]:
    """POC subset: take first `limit_per_cat` questions per category."""
    with open(LME_QUESTIONS_FILE) as f:
        qs = json.load(f)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for q in qs:
        by_cat[q.get("category", "unknown")].append(q)
    out: list[dict] = []
    for cat in sorted(by_cat):
        out.extend(by_cat[cat][:limit_per_cat])
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        default=",".join(VARIANT_CONFIGS.keys()),
        help="Comma-separated variants.",
    )
    parser.add_argument(
        "--corpora",
        default="locomo,lme",
        help="Comma-separated: locomo,lme",
    )
    parser.add_argument(
        "--locomo-limit", type=int, default=None, help="Limit LoCoMo question count."
    )
    parser.add_argument(
        "--lme-per-cat",
        type=int,
        default=10,
        help="Per-category LME question count (POC subset).",
    )
    parser.add_argument("--arch-concurrency", type=int, default=8)
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    corpora = [c.strip() for c in args.corpora.split(",") if c.strip()]

    # Load datasets.
    locomo_qs = load_locomo_questions(args.locomo_limit) if "locomo" in corpora else []
    lme_qs = load_lme_questions(args.lme_per_cat) if "lme" in corpora else []
    print(
        f"[steer_eval] LoCoMo n={len(locomo_qs)}, LME n={len(lme_qs)}, "
        f"variants={len(variants)}",
        flush=True,
    )

    # Connect EM backend.
    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    # Two possible segment stores (LoCoMo vs LME might use different sqlite).
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

    # Caches.
    locomo_v2f_cache = _MergedLLMCache(
        reader_paths=[
            BESTSHOT_LLM_CACHE,
            ASSOC_DIR / "cache" / "meta_llm_cache.json",
            EM_V2F_LLM_CACHE,
        ],
        writer_path=EM_V2F_LLM_CACHE,
    )
    lme_v2f_cache = _MergedLLMCache(
        reader_paths=[LMETUNE_V2F_MIXED7030_CACHE],
        writer_path=LMETUNE_V2F_MIXED7030_CACHE,
    )
    steer_llm_cache = _MergedLLMCache(
        reader_paths=[STEER_LLM_CACHE],
        writer_path=STEER_LLM_CACHE,
    )
    emb_cache = EmbeddingCache(STEER_EMB_CACHE)

    # Open collections.
    memories: dict[tuple[str, str], EventMemory] = {}
    opened: list = []
    try:
        if locomo_qs:
            with open(LOCOMO_COLLECTIONS_FILE) as f:
                locomo_meta = json.load(f)
            conv_to_meta = {r["conversation_id"]: r for r in locomo_meta["conversations"]}
            sql_url = locomo_meta.get("sql_url") or os.getenv("SQL_URL")
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

            for conv_id in LOCOMO_CONV_IDS:
                m = conv_to_meta[conv_id]
                coll = await vector_store.open_collection(
                    namespace=m["namespace"], name=m["collection_name"]
                )
                part = await seg_store.open_or_create_partition(m["partition_key"])
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
                memories[("locomo", conv_id)] = mem
                opened.append((seg_store, coll, part))

        if lme_qs:
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
                memories[("lme", q["question_id"])] = mem
                opened.append((seg_store, coll, part))

        max_K = max(BUDGETS)
        results: dict = {"variants": {}, "budgets": list(BUDGETS)}

        sem = asyncio.Semaphore(args.arch_concurrency)

        async def run_wrap(variant, corpus, q):
            async with sem:
                mem_key = (
                    ("locomo", q["conversation_id"])
                    if corpus == "locomo"
                    else ("lme", q["question_id"])
                )
                mem = memories[mem_key]
                v2fc = locomo_v2f_cache if corpus == "locomo" else lme_v2f_cache
                return await run_one_question(
                    variant,
                    corpus,
                    mem,
                    q,
                    embedder=embedder,
                    openai_client=openai_client,
                    v2f_cache=v2fc,
                    steer_llm_cache=steer_llm_cache,
                    emb_cache=emb_cache,
                    max_K=max_K,
                )

        for variant in variants:
            results["variants"][variant] = {}
            for corpus, qs in [("locomo", locomo_qs), ("lme", lme_qs)]:
                if not qs:
                    continue
                t_start = time.monotonic()
                tasks = [run_wrap(variant, corpus, q) for q in qs]
                rows = await asyncio.gather(*tasks)
                elapsed = time.monotonic() - t_start

                # Summary.
                n = len(rows)
                summary = {"n": n, "time_s": round(elapsed, 1)}
                for K in BUDGETS:
                    summary[f"mean_r@{K}"] = round(
                        sum(r[f"r@{K}"] for r in rows) / max(n, 1), 4
                    )

                # Category breakdown.
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

                # Aggregate trajectory stats.
                drift_sum: dict[int, list[float]] = defaultdict(list)
                add_mag_sum: dict[int, list[float]] = defaultdict(list)
                sub_mag_sum: dict[int, list[float]] = defaultdict(list)
                recall_traj: dict[int, dict[str, list[float]]] = defaultdict(
                    lambda: defaultdict(list)
                )
                sub_nonempty = 0
                sub_total = 0
                for r in rows:
                    for tr in r["trajectory"]:
                        rnd = tr["round"]
                        drift_sum[rnd].append(tr["probe_drift"])
                        add_mag_sum[rnd].append(tr["add_magnitude"])
                        sub_mag_sum[rnd].append(tr["sub_magnitude"])
                        for K in BUDGETS:
                            if f"r@{K}" in tr:
                                recall_traj[rnd][f"r@{K}"].append(tr[f"r@{K}"])
                        if rnd > 0:
                            sub_total += 1
                            if tr["sub_phrases"]:
                                sub_nonempty += 1

                traj_agg = {}
                for rnd in sorted(drift_sum.keys()):
                    entry: dict = {
                        "n": len(drift_sum[rnd]),
                        "mean_drift_vs_probe_0": round(
                            sum(drift_sum[rnd]) / max(len(drift_sum[rnd]), 1), 4
                        ),
                        "mean_add_mag": round(
                            sum(add_mag_sum[rnd]) / max(len(add_mag_sum[rnd]), 1), 4
                        ),
                        "mean_sub_mag": round(
                            sum(sub_mag_sum[rnd]) / max(len(sub_mag_sum[rnd]), 1), 4
                        ),
                    }
                    for K in BUDGETS:
                        vals = recall_traj[rnd][f"r@{K}"]
                        if vals:
                            entry[f"mean_r@{K}"] = round(sum(vals) / len(vals), 4)
                    traj_agg[str(rnd)] = entry

                results["variants"][variant][corpus] = {
                    "summary": summary,
                    "by_category": cat_summary,
                    "trajectory": traj_agg,
                    "sub_nonempty_fraction": (
                        round(sub_nonempty / sub_total, 3) if sub_total else 0.0
                    ),
                    "per_question": rows,
                }

                # Persist caches after each variant/corpus.
                locomo_v2f_cache.save()
                lme_v2f_cache.save()
                steer_llm_cache.save()
                emb_cache.save()

                print(
                    f"[{variant} / {corpus}] n={n} "
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

    # Save.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUT_JSON}", flush=True)
    write_report(results)


def write_report(results: dict) -> None:
    variants = results["variants"]
    lines: list[str] = []
    lines.append("# Active Embedding Steering")
    lines.append("")
    lines.append(
        "Probe = normalize(initial_embedding + alpha*sum(add_phrase_embs) - "
        "beta*sum(sub_phrase_embs)) applied cumulatively over LLM-generated "
        "add/sub phrases."
    )
    lines.append("")
    lines.append("Fixed: alpha=beta=0.1, text-embedding-3-small, gpt-5-mini, LoCoMo-30 and LME-hard-30 POC subset.")
    lines.append("")

    # Overall recall matrix
    for corpus in ("locomo", "lme"):
        lines.append(f"## Recall matrix -- {corpus}")
        lines.append("")
        lines.append("| Variant | n | R@20 | R@50 | time (s) |")
        lines.append("| --- | --- | --- | --- | --- |")
        for variant, per_corpus in variants.items():
            data = per_corpus.get(corpus)
            if not data:
                continue
            s = data["summary"]
            lines.append(
                f"| `{variant}` | {s['n']} | {s['mean_r@20']:.4f} | "
                f"{s['mean_r@50']:.4f} | {s['time_s']:.1f} |"
            )
        lines.append("")

    # Per-category recall (LME only: 3 categories)
    if any("lme" in per_corpus for per_corpus in variants.values()):
        for K in BUDGETS:
            lines.append(f"## LME category recall (R@{K})")
            lines.append("")
            lines.append(
                "| Variant | multi-session | single-session-preference | temporal-reasoning |"
            )
            lines.append("| --- | --- | --- | --- |")
            for variant, per_corpus in variants.items():
                data = per_corpus.get("lme")
                if not data:
                    continue
                bc = data["by_category"]
                lines.append(
                    f"| `{variant}` | "
                    f"{bc.get('multi-session', {}).get(f'mean_r@{K}', 0):.4f} | "
                    f"{bc.get('single-session-preference', {}).get(f'mean_r@{K}', 0):.4f} | "
                    f"{bc.get('temporal-reasoning', {}).get(f'mean_r@{K}', 0):.4f} |"
                )
            lines.append("")

    # Round-by-round trajectory for key variants
    lines.append("## Round-by-round recall trajectory (mean R@50)")
    lines.append("")
    lines.append(
        "Rows: variants. Columns: round 0 (initial cue) .. final round."
    )
    lines.append("")
    for corpus in ("locomo", "lme"):
        lines.append(f"### {corpus}")
        lines.append("")
        lines.append("| Variant | rd 0 | rd 1 | rd 2 | rd 3 | drift@final |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for variant, per_corpus in variants.items():
            data = per_corpus.get(corpus)
            if not data:
                continue
            traj = data["trajectory"]
            cells = []
            max_rnd = max(int(k) for k in traj.keys())
            for rnd in range(0, 4):
                entry = traj.get(str(rnd))
                if entry and f"mean_r@50" in entry:
                    cells.append(f"{entry['mean_r@50']:.4f}")
                else:
                    cells.append("--")
            final_drift = traj[str(max_rnd)].get("mean_drift_vs_probe_0", 1.0)
            lines.append(
                f"| `{variant}` | {cells[0]} | {cells[1]} | {cells[2]} | "
                f"{cells[3]} | {final_drift:.4f} |"
            )
        lines.append("")

    # SUBTRACT-primitive usage
    lines.append("## SUBTRACT primitive usage")
    lines.append("")
    lines.append(
        "Fraction of steering rounds where the LLM emitted a non-empty SUBTRACT list."
    )
    lines.append("")
    lines.append("| Variant | corpus | sub_nonempty_fraction |")
    lines.append("| --- | --- | --- |")
    for variant, per_corpus in variants.items():
        for corpus in ("locomo", "lme"):
            data = per_corpus.get(corpus)
            if not data:
                continue
            frac = data.get("sub_nonempty_fraction", 0.0)
            lines.append(f"| `{variant}` | {corpus} | {frac:.3f} |")
    lines.append("")

    # Magnitude stats
    lines.append("## Update magnitudes (alpha*||add|| and beta*||sub||)")
    lines.append("")
    for corpus in ("locomo", "lme"):
        lines.append(f"### {corpus}")
        lines.append("")
        lines.append("| Variant | rd1 add_mag | rd1 sub_mag | rd2 add_mag | rd2 sub_mag |")
        lines.append("| --- | --- | --- | --- | --- |")
        for variant, per_corpus in variants.items():
            data = per_corpus.get(corpus)
            if not data:
                continue
            traj = data["trajectory"]
            cells = []
            for rnd in (1, 2):
                entry = traj.get(str(rnd)) or {}
                cells.append(entry.get("mean_add_mag", 0.0))
                cells.append(entry.get("mean_sub_mag", 0.0))
            lines.append(
                f"| `{variant}` | {cells[0]:.4f} | {cells[1]:.4f} | "
                f"{cells[2]:.4f} | {cells[3]:.4f} |"
            )
        lines.append("")

    # Sample add/sub phrases: 3 questions from LME (where steering should matter more)
    lines.append("## Sample add/sub phrases (LME)")
    lines.append("")
    for variant in ("steer_v2f_2round",):
        per = variants.get(variant, {}).get("lme")
        if not per:
            continue
        rows = per["per_question"][:3]
        lines.append(f"### `{variant}`")
        lines.append("")
        for r in rows:
            lines.append(f"- Q `{r['question_id']}` (`{r['category']}`): {r['question']}")
            lines.append(f"  - initial cue: `{r['initial_cue_text']}`")
            for tr in r["trajectory"]:
                if tr["round"] == 0:
                    continue
                adds = "; ".join(f"{p} (w={w})" for p, w in tr["add_phrases"])
                subs = "; ".join(f"{p} (w={w})" for p, w in tr["sub_phrases"])
                lines.append(
                    f"  - round {tr['round']}: drift={tr['probe_drift']:.3f}, "
                    f"R@50={tr.get('r@50','--')}"
                )
                lines.append(f"    - ADD: {adds or '(empty)'}")
                lines.append(f"    - SUB: {subs or '(empty)'}")
        lines.append("")

    # Verdict section
    lines.append("## Verdict")
    lines.append("")
    v2f_2 = variants.get("steer_v2f_2round", {})
    v2f_add = variants.get("steer_v2f_addonly", {})
    v2f_sub = variants.get("steer_v2f_subonly", {})
    base_v2f = variants.get("baseline_v2f_direct", {})
    base_q = variants.get("baseline_query_direct", {})

    def _get_r50(d, corpus):
        return d.get(corpus, {}).get("summary", {}).get("mean_r@50", 0.0)

    for corpus in ("locomo", "lme"):
        lines.append(f"### {corpus}")
        lines.append("")
        base = _get_r50(base_v2f, corpus)
        steer = _get_r50(v2f_2, corpus)
        addo = _get_r50(v2f_add, corpus)
        subo = _get_r50(v2f_sub, corpus)
        bq = _get_r50(base_q, corpus)
        lines.append(f"- baseline (v2f cue direct): R@50 = {base:.4f}")
        lines.append(f"- baseline (query direct): R@50 = {bq:.4f}")
        lines.append(f"- steer_v2f_2round: R@50 = {steer:.4f} (Δ vs baseline = {steer - base:+.4f})")
        lines.append(f"- steer_v2f_addonly: R@50 = {addo:.4f}")
        lines.append(f"- steer_v2f_subonly: R@50 = {subo:.4f}")
        lines.append("")

    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- JSON: `{OUT_JSON.relative_to(ASSOC_DIR)}`")
    lines.append("- Sources: `active_steering.py`, `steer_eval.py`")
    lines.append("- Caches: `cache/steer_llm_cache.json`, `cache/steer_embedding_cache.json`")
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"Saved: {OUT_MD}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
