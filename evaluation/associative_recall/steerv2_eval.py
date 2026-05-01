"""Evaluate v2 evidence-grounded active steering on LME-hard-POC (30 questions).

V2 variants:
  - steerv2_full    : add + sub (default)
  - steerv2_subonly : only subtract distractor turns
  - steerv2_addonly : only evidence-grounded adds

Baselines:
  - baseline_v2f_speakerformat : round 0 snapshot starting from v2f speaker-format cue
  - baseline_v2f_direct        : same (alias; kept for explicit naming parity)

Outputs:
  results/steering_v2.json
  results/steering_v2.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import openai
from active_steering import EmbeddingCache
from active_steering_v2 import SteerV2Config, active_steer_v2
from dotenv import load_dotenv
from em_architectures import (
    V2F_MODEL,
    _dedupe_by_turn_id,
    _MergedLLMCache,
    _query_em,
    format_primer_context,
)
from em_lme_tuned_cues import (
    LMETUNE_V2F_MIXED7030_CACHE,
    V2F_LME_MIXED_7030_PROMPT,
    parse_speaker_cues,
)
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
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

ASSOC_DIR = Path(__file__).resolve().parent
DATA_DIR = ASSOC_DIR / "data"
RESULTS_DIR = ASSOC_DIR / "results"
CACHE_DIR = ASSOC_DIR / "cache"

LME_QUESTIONS_FILE = DATA_DIR / "questions_longmemeval_hard.json"
LME_COLLECTIONS_FILE = RESULTS_DIR / "em_lme_hard_collections.json"

OUT_JSON = RESULTS_DIR / "steering_v2.json"
OUT_MD = RESULTS_DIR / "steering_v2.md"

STEERV2_LLM_CACHE = CACHE_DIR / "steerv2_llm_cache.json"
STEERV2_EMB_CACHE = CACHE_DIR / "steerv2_embedding_cache.json"

BUDGETS = (20, 50)
USER_PREFIX = "User: "


def _ensure_user_prefix(text: str) -> str:
    t = text.lstrip()
    low = t.lower()
    if low.startswith("user:") or low.startswith("assistant:"):
        return t
    return USER_PREFIX + t


# --------------------------------------------------------------------------
# Initial-cue builder (LME v2f speaker-format)
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


# --------------------------------------------------------------------------
# Variant configs
# --------------------------------------------------------------------------


VARIANT_CONFIGS = {
    "steerv2_full": dict(max_rounds=3, use_add=True, use_sub=True),
    "steerv2_subonly": dict(max_rounds=3, use_add=False, use_sub=True),
    "steerv2_addonly": dict(max_rounds=3, use_add=True, use_sub=False),
    "baseline_v2f_direct": dict(max_rounds=0, use_add=True, use_sub=True),
}


def compute_recall(retrieved: set[int], gold: set[int]) -> float:
    if not gold:
        return 1.0
    return len(retrieved & gold) / len(gold)


async def run_one_question(
    variant: str,
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

    # Build initial cue (v2f speaker-format; same as v1 baseline_v2f_direct).
    initial_text, v2f_cues = await _load_or_build_lme_v2f_cue(
        memory, q_text, v2f_cache, openai_client
    )

    config = SteerV2Config(
        max_rounds=conf["max_rounds"],
        alpha=0.1,
        beta=0.1,
        topk_for_llm=5,
        vector_search_limit=max_K,
        expand_context=3,  # LME expand recipe
        use_add=conf["use_add"],
        use_sub=conf["use_sub"],
    )

    result = await active_steer_v2(
        memory,
        query_text=q_text,
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

    final_hits = result.hits_by_round[-1]
    row: dict = {
        "question_id": question.get("question_id"),
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

    # Round-by-round trajectory with rich diagnostics.
    trajectory: list[dict] = []
    for idx, t in enumerate(result.traces):
        # Also record the turn_ids in top-K and whether each is gold, for
        # diagnosing classification quality.
        hits_this_round = result.hits_by_round[idx]
        topk_ids = [h.turn_id for h in hits_this_round[: config.topk_for_llm]]
        topk_is_gold = [tid in gold for tid in topk_ids]
        tr = {
            "round": t.round_idx,
            "add_magnitude": round(t.add_magnitude, 4),
            "sub_magnitude": round(t.sub_magnitude, 4),
            "probe_drift": round(t.probe_drift, 4),
            "add_phrases": list(t.add_phrases),
            "distractor_indices": list(t.distractor_indices),
            "gold_likely_indices": list(t.gold_likely_indices),
            "subtracted_texts": list(t.subtracted_texts),
            "reasoning": t.reasoning,
            "topk_turn_ids": topk_ids,
            "topk_is_gold": topk_is_gold,
        }
        tr.update({k: round(v, 4) for k, v in t.recall_deltas.items()})
        trajectory.append(tr)
    row["trajectory"] = trajectory

    return row


# --------------------------------------------------------------------------
# Main
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


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        default=",".join(VARIANT_CONFIGS.keys()),
    )
    parser.add_argument("--lme-per-cat", type=int, default=10)
    parser.add_argument("--arch-concurrency", type=int, default=8)
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    lme_qs = load_lme_questions(args.lme_per_cat)
    print(f"[steerv2_eval] LME n={len(lme_qs)}, variants={len(variants)}", flush=True)

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
    lme_v2f_cache = _MergedLLMCache(
        reader_paths=[LMETUNE_V2F_MIXED7030_CACHE],
        writer_path=LMETUNE_V2F_MIXED7030_CACHE,
    )
    steer_llm_cache = _MergedLLMCache(
        reader_paths=[STEERV2_LLM_CACHE],
        writer_path=STEERV2_LLM_CACHE,
    )
    emb_cache = EmbeddingCache(STEERV2_EMB_CACHE)

    # Open LME collections.
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
        seg_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
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

        max_K = max(BUDGETS)
        results: dict = {"variants": {}, "budgets": list(BUDGETS)}

        sem = asyncio.Semaphore(args.arch_concurrency)

        async def run_wrap(variant, q):
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
                    max_K=max_K,
                )

        for variant in variants:
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

            # Trajectory aggregates.
            drift_sum: dict[int, list[float]] = defaultdict(list)
            add_mag_sum: dict[int, list[float]] = defaultdict(list)
            sub_mag_sum: dict[int, list[float]] = defaultdict(list)
            recall_traj: dict[int, dict[str, list[float]]] = defaultdict(
                lambda: defaultdict(list)
            )
            # Classification accuracy: of LLM-flagged distractor turn_ids, how
            # many are NOT in gold? (true-positive distractor flag)
            flagged_distractors = 0
            correct_distractors = 0  # flagged AND not-gold
            flagged_gold_likely = 0
            correct_gold_likely = 0  # flagged AND is-gold
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
                        for i in tr["distractor_indices"]:
                            if 0 <= i < len(tr["topk_is_gold"]):
                                flagged_distractors += 1
                                if not tr["topk_is_gold"][i]:
                                    correct_distractors += 1
                        for i in tr["gold_likely_indices"]:
                            if 0 <= i < len(tr["topk_is_gold"]):
                                flagged_gold_likely += 1
                                if tr["topk_is_gold"][i]:
                                    correct_gold_likely += 1

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

            classification_stats = {
                "flagged_distractors": flagged_distractors,
                "correct_distractors": correct_distractors,
                "distractor_precision": (
                    round(correct_distractors / flagged_distractors, 4)
                    if flagged_distractors
                    else None
                ),
                "flagged_gold_likely": flagged_gold_likely,
                "correct_gold_likely": correct_gold_likely,
                "gold_likely_precision": (
                    round(correct_gold_likely / flagged_gold_likely, 4)
                    if flagged_gold_likely
                    else None
                ),
            }

            results["variants"][variant] = {
                "summary": summary,
                "by_category": cat_summary,
                "trajectory": traj_agg,
                "classification_stats": classification_stats,
                "per_question": rows,
            }

            # Persist caches after each variant.
            lme_v2f_cache.save()
            steer_llm_cache.save()
            emb_cache.save()

            print(
                f"[{variant}] n={n} "
                f"r@20={summary['mean_r@20']:.4f} r@50={summary['mean_r@50']:.4f} "
                f"dist_prec={classification_stats.get('distractor_precision')} "
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

# V1 reference numbers (from results/active_steering.md) for comparison.
V1_REFERENCE_LME = {
    "steer_v2f_2round": {"r@20": 0.6358, "r@50": 0.7994},
    "steer_v2f_addonly": {"r@20": 0.6583, "r@50": 0.8085},
    "steer_v2f_subonly": {"r@20": 0.6339, "r@50": 0.8128},
    "baseline_v2f_direct": {"r@20": 0.6303, "r@50": 0.8169},
}


def write_report(results: dict) -> None:
    variants = results["variants"]
    lines: list[str] = []
    lines.append("# Active Embedding Steering V2 (evidence-grounded)")
    lines.append("")
    lines.append(
        "V2 fix: SUBTRACT uses embeddings of LLM-classified distractor turns "
        "(actual retrieved text), not imagined opposite concepts. ADD phrases "
        "grounded in query vocabulary or extracted from gold-likely retrieved "
        "turns; fabrication of specific details forbidden."
    )
    lines.append("")
    lines.append(
        "Fixed: alpha=beta=0.1, text-embedding-3-small, gpt-5-mini, "
        "LME-hard-30 POC subset, up to 3 rounds with early-stop."
    )
    lines.append("")

    # Recall matrix
    lines.append("## Recall matrix (LME-hard-30)")
    lines.append("")
    lines.append("| Variant | n | R@20 | R@50 | time (s) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for variant, per in variants.items():
        s = per["summary"]
        lines.append(
            f"| `{variant}` | {s['n']} | {s['mean_r@20']:.4f} | "
            f"{s['mean_r@50']:.4f} | {s['time_s']:.1f} |"
        )
    lines.append("")

    # V1 comparison
    lines.append("## V2 vs V1 (LME R@50)")
    lines.append("")
    lines.append("| Variant | V1 R@50 | V2 R@50 | Δ |")
    lines.append("| --- | --- | --- | --- |")
    name_map = {
        "steerv2_full": "steer_v2f_2round",
        "steerv2_subonly": "steer_v2f_subonly",
        "steerv2_addonly": "steer_v2f_addonly",
        "baseline_v2f_direct": "baseline_v2f_direct",
    }
    for v2_name, v1_name in name_map.items():
        if v2_name not in variants:
            continue
        v1_r = V1_REFERENCE_LME.get(v1_name, {}).get("r@50", 0.0)
        v2_r = variants[v2_name]["summary"]["mean_r@50"]
        delta = v2_r - v1_r
        lines.append(
            f"| `{v2_name}` (vs `{v1_name}`) | {v1_r:.4f} | {v2_r:.4f} | {delta:+.4f} |"
        )
    lines.append("")

    # Per-category
    for K in BUDGETS:
        lines.append(f"## LME category recall (R@{K})")
        lines.append("")
        lines.append(
            "| Variant | multi-session | single-session-preference | temporal-reasoning |"
        )
        lines.append("| --- | --- | --- | --- |")
        for variant, per in variants.items():
            bc = per["by_category"]
            lines.append(
                f"| `{variant}` | "
                f"{bc.get('multi-session', {}).get(f'mean_r@{K}', 0):.4f} | "
                f"{bc.get('single-session-preference', {}).get(f'mean_r@{K}', 0):.4f} | "
                f"{bc.get('temporal-reasoning', {}).get(f'mean_r@{K}', 0):.4f} |"
            )
        lines.append("")

    # Classification stats
    lines.append("## LLM classification quality")
    lines.append("")
    lines.append(
        "Precision = fraction of LLM-flagged turns where the classification "
        "matches ground truth (distractor = not-gold, gold-likely = is-gold)."
    )
    lines.append("")
    lines.append(
        "| Variant | flagged_distractors | distractor_precision | "
        "flagged_gold_likely | gold_likely_precision |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for variant, per in variants.items():
        cs = per["classification_stats"]
        dp = cs["distractor_precision"]
        gp = cs["gold_likely_precision"]
        lines.append(
            f"| `{variant}` | {cs['flagged_distractors']} | "
            f"{dp if dp is not None else '--'} | "
            f"{cs['flagged_gold_likely']} | "
            f"{gp if gp is not None else '--'} |"
        )
    lines.append("")

    # Round-by-round trajectory
    lines.append("## Round-by-round R@50")
    lines.append("")
    lines.append("| Variant | rd 0 | rd 1 | rd 2 | rd 3 | drift@final |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for variant, per in variants.items():
        traj = per["trajectory"]
        cells = []
        for rnd in range(4):
            entry = traj.get(str(rnd))
            if entry and "mean_r@50" in entry:
                cells.append(f"{entry['mean_r@50']:.4f}")
            else:
                cells.append("--")
        max_rnd = max(int(k) for k in traj.keys())
        final_drift = traj[str(max_rnd)].get("mean_drift_vs_probe_0", 1.0)
        lines.append(
            f"| `{variant}` | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{cells[3]} | {final_drift:.4f} |"
        )
    lines.append("")

    # Sample classifications -- 3 questions from steerv2_full
    lines.append("## Sample classifications (steerv2_full, first 3 questions)")
    lines.append("")
    per_full = variants.get("steerv2_full", {})
    if per_full:
        rows = per_full["per_question"][:3]
        for r in rows:
            lines.append(f"### Q `{r['question_id']}` ({r['category']})")
            lines.append("")
            lines.append(f"- question: {r['question']}")
            lines.append(f"- initial cue: `{r['initial_cue_text']}`")
            for tr in r["trajectory"]:
                rnd = tr["round"]
                if rnd == 0:
                    lines.append(
                        f"- round 0: R@50={tr.get('r@50', '--')}, "
                        f"top-5 gold mask: {tr['topk_is_gold']}"
                    )
                    continue
                dist_correct = [
                    tr["topk_is_gold"][i] is False
                    for i in tr["distractor_indices"]
                    if 0 <= i < len(tr["topk_is_gold"])
                ]
                gold_correct = [
                    tr["topk_is_gold"][i] is True
                    for i in tr["gold_likely_indices"]
                    if 0 <= i < len(tr["topk_is_gold"])
                ]
                lines.append(
                    f"- round {rnd}: R@50={tr.get('r@50', '--')}, "
                    f"drift={tr['probe_drift']:.3f}"
                )
                lines.append(
                    f"  - distractor_indices={tr['distractor_indices']} "
                    f"(correct: {sum(dist_correct)}/{len(dist_correct)}), "
                    f"gold_likely={tr['gold_likely_indices']} "
                    f"(correct: {sum(gold_correct)}/{len(gold_correct)})"
                )
                lines.append(f"  - top-5 gold mask: {tr['topk_is_gold']}")
                lines.append(f"  - ADD phrases: {tr['add_phrases']}")
                if tr["subtracted_texts"]:
                    subs_preview = [t[:100] for t in tr["subtracted_texts"]]
                    lines.append(f"  - SUBTRACTED (previews): {subs_preview}")
                if tr["reasoning"]:
                    lines.append(f"  - reasoning: {tr['reasoning']}")
            lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    base = (
        variants.get("baseline_v2f_direct", {}).get("summary", {}).get("mean_r@50", 0.0)
    )
    full = variants.get("steerv2_full", {}).get("summary", {}).get("mean_r@50", 0.0)
    addo = variants.get("steerv2_addonly", {}).get("summary", {}).get("mean_r@50", 0.0)
    subo = variants.get("steerv2_subonly", {}).get("summary", {}).get("mean_r@50", 0.0)
    lines.append(f"- baseline (v2f speaker-format direct): R@50 = {base:.4f}")
    lines.append(
        f"- steerv2_full: R@50 = {full:.4f} (Δ vs baseline = {full - base:+.4f})"
    )
    lines.append(f"- steerv2_addonly: R@50 = {addo:.4f} (Δ = {addo - base:+.4f})")
    lines.append(f"- steerv2_subonly: R@50 = {subo:.4f} (Δ = {subo - base:+.4f})")
    lines.append("")

    # Decision rule outcome.
    best = max(full, addo, subo)
    if best - base >= 0.01:
        lines.append(
            f"**Evidence-grounded steering clears the +1pp bar** (best Δ = {best - base:+.4f})."
        )
    else:
        cls_prec = (
            variants.get("steerv2_full", {})
            .get("classification_stats", {})
            .get("distractor_precision")
        )
        if cls_prec is not None and cls_prec >= 0.7:
            lines.append(
                f"**Classification is accurate ({cls_prec:.2f}) but recall does NOT lift** "
                f"(best Δ = {best - base:+.4f}). This supports the substrate hypothesis: "
                "text-embedding-3-small does not reward vector arithmetic in this "
                "retrieval geometry, regardless of phrase discipline."
            )
        else:
            lines.append(
                f"**No lift (best Δ = {best - base:+.4f}) and classification precision "
                f"= {cls_prec}**; classification quality could be the bottleneck."
            )
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- JSON: `{OUT_JSON.relative_to(ASSOC_DIR)}`")
    lines.append("- Sources: `active_steering_v2.py`, `steerv2_eval.py`")
    lines.append(
        "- Caches: `cache/steerv2_llm_cache.json`, `cache/steerv2_embedding_cache.json`"
    )
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"Saved: {OUT_MD}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
