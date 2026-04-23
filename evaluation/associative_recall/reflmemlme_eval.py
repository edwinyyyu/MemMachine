"""Evaluate reflective-memory variants on LongMemEval-hard EventMemory.

Variants:
  - reflmemlme_1round  : 1 round of (cue_gen, retrieve, reflect-write)
  - reflmemlme_2round  : 2 rounds
  - reflmemlme_3round  : 3 rounds

All variants use:
  - Round 1 cue prompt = em_v2f_lme_mixed_7030 (cache-reuse via identical
    prompt text)
  - Rounds 2+ cue prompt = ROUND_N_CUE_PROMPT (with scratch memory context)
  - Reflection prompt after each round = REFLECTION_PROMPT (JSON output)
  - EventMemory.query with expand_context=3
  - "User: " prefix on queries and cues
  - Scratch memory scored via additional EM queries using scratch-entry
    text (equivalent to cosine against the turn corpus)

Reuses em_lme_setup.py's ingested collections + sqlite.
Does NOT modify any framework file or existing em_*.py / em_lme_tuned_*.py.

Outputs:
  results/reflective_memory_lme.json
  results/reflective_memory_lme.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
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

from em_architectures import (
    EMHit,
    V2F_MODEL,
    _MergedLLMCache,
    _dedupe_by_turn_id,
    _merge_by_max_score,
    _query_em,
)
from em_lme_tuned_cues import LMETUNE_V2F_MIXED7030_CACHE
from reflective_memory_lme import (
    REFLMEMLME_CUE_ROUND1_CACHE,
    REFLMEMLME_CUE_ROUNDN_CACHE,
    REFLMEMLME_REFLECT_CACHE,
    REFLECTION_PROMPT,
    ROUND_N_CUE_PROMPT,
    V2F_LME_MIXED_7030_PROMPT,
    ScratchMemory,
    format_prev_cues_section,
    format_primer_context_lme,
    format_scratch_section,
    format_turns_section,
    parse_reflection_json,
    parse_speaker_cues,
)


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

ASSOC_DIR = Path(__file__).resolve().parent
DATA_DIR = ASSOC_DIR / "data"
RESULTS_DIR = ASSOC_DIR / "results"

HARD_QUESTIONS_JSON = DATA_DIR / "questions_longmemeval_hard.json"
COLLECTIONS_JSON = RESULTS_DIR / "em_lme_hard_collections.json"
RESULTS_JSON = RESULTS_DIR / "reflective_memory_lme.json"
RESULTS_MD = RESULTS_DIR / "reflective_memory_lme.md"

BUDGETS = (20, 50)
USER_PREFIX = "User: "
EXPAND_CONTEXT = 3
ARCH_CONCURRENCY = 8

# References from prior LME runs.
REF_EM_V2F_EXPAND_3 = 0.832              # em_v2f_expand_3 R@50 overall
REF_EM_V2F_LME_MIXED_7030 = 0.8631       # em_v2f_lme_mixed_7030 R@50 overall
REF_EM_ENS_2_LME_RETUNED = 0.8499        # em_ens_2_lme_retuned R@50 overall
# per-category on em_v2f_lme_mixed_7030 (R@50)
REF_MS_MIXED_7030 = 0.8484               # multi-session
REF_SSP_MIXED_7030 = 0.9449              # single-session-preference
REF_TR_MIXED_7030 = 0.7961               # temporal-reasoning


def _ensure_user_prefix(text: str) -> str:
    t = text.lstrip()
    low = t.lower()
    if low.startswith("user:") or low.startswith("assistant:"):
        return t
    return USER_PREFIX + t


# -----------------------------------------------------------------------------
# Primer (for round-1 prompt cache compatibility with em_v2f_lme_mixed_7030).
# -----------------------------------------------------------------------------


async def _primer_context_for_round1(memory: EventMemory, question: str) -> str:
    """Build the primer context section that matches em_v2f_lme_mixed_7030.

    em_v2f_lme_mixed_7030 uses _primer_and_context from em_lme_tuned_eval.py,
    which:
      - prefixes "User: " to the question
      - queries EM with vector_search_limit=10, expand_context=0
      - dedupes by turn_id, takes top-10
      - passes through format_primer_context(segments).
    We replicate byte-for-byte here so the hash match the mixed_7030 cache.
    """
    prefixed_q = _ensure_user_prefix(question)
    primer_hits = _dedupe_by_turn_id(
        await _query_em(
            memory, prefixed_q, vector_search_limit=10, expand_context=0
        )
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text}
        for h in primer_hits
    ]
    return format_primer_context_lme(primer_segments)


# -----------------------------------------------------------------------------
# LLM-call wrappers with caching.
# -----------------------------------------------------------------------------


async def _call_llm_cached(
    *,
    prompt: str,
    cache: _MergedLLMCache,
    openai_client,
) -> tuple[str, bool]:
    """Return (response_text, cache_hit)."""
    cached = cache.get(V2F_MODEL, prompt)
    if cached is not None:
        return cached, True
    if openai_client is None:
        return "", False
    resp = await openai_client.chat.completions.create(
        model=V2F_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content or ""
    cache.put(V2F_MODEL, prompt, text)
    return text, False


# -----------------------------------------------------------------------------
# Retrieval utility: primer + cue list + scratch entries all merged by max.
# -----------------------------------------------------------------------------


async def _retrieve_with_cues_and_scratch(
    memory: EventMemory,
    *,
    question: str,
    cues: list[str],
    scratch: ScratchMemory,
    K: int,
    expand_context: int = EXPAND_CONTEXT,
) -> list[EMHit]:
    """Primer (prefixed question) + each cue + each scratch-entry, merged by
    max score per turn_id."""
    prefixed_q = _ensure_user_prefix(question)
    vsl = max(K, 20)

    batches: list[list[EMHit]] = []
    # Primer uses the raw question with same expand_context.
    batches.append(
        await _query_em(
            memory, prefixed_q, vector_search_limit=vsl, expand_context=expand_context
        )
    )
    # Each cue (with User:/Assistant: prefix already).
    for cue in cues:
        batches.append(
            await _query_em(
                memory,
                _ensure_user_prefix(cue),
                vector_search_limit=vsl,
                expand_context=expand_context,
            )
        )
    # Each scratch entry: prepend "User: " for embedding-space consistency
    # with the corpus (which is embedded as "User: ..." / "Assistant: ...").
    for e in scratch.entries:
        batches.append(
            await _query_em(
                memory,
                _ensure_user_prefix(e.text),
                vector_search_limit=vsl,
                expand_context=expand_context,
            )
        )
    merged = _merge_by_max_score(batches)
    return merged[:K]


# -----------------------------------------------------------------------------
# Embedding helper for scratch-memory book-keeping.
# -----------------------------------------------------------------------------


async def _embed_texts(
    embedder: OpenAIEmbedder, texts: list[str]
) -> list[np.ndarray]:
    if not texts:
        return []
    prefixed = [_ensure_user_prefix(t) for t in texts]
    embs = await embedder.search_embed(prefixed, max_attempts=3)
    return [np.asarray(e, dtype=np.float32) for e in embs]


# -----------------------------------------------------------------------------
# Per-question driver.
# -----------------------------------------------------------------------------


async def run_reflmem_query(
    memory: EventMemory,
    *,
    question: str,
    num_rounds: int,
    max_K: int,
    caches: dict[str, _MergedLLMCache],
    embedder: OpenAIEmbedder,
    openai_client,
) -> dict:
    """Run N rounds of reflective retrieval. Returns a diagnostic dict
    containing per-round hits, cues, reflections, scratch state, etc.
    """
    scratch = ScratchMemory()
    all_cues: list[str] = []
    round_records: list[dict] = []
    round_hits: list[list[EMHit]] = []
    cache_hits = {"cue_round1": 0, "cue_roundn": 0, "reflect": 0}
    llm_calls = 0

    # ---------------------------------------------------------------------
    # Round 1: em_v2f_lme_mixed_7030 prompt (cache-compatible).
    # ---------------------------------------------------------------------
    primer_ctx = await _primer_context_for_round1(memory, question)
    r1_prompt = V2F_LME_MIXED_7030_PROMPT.format(
        question=question, context_section=primer_ctx
    )
    # Prefer the mixed_7030 cache (exact same prompt) then our dedicated cache.
    r1_text, r1_hit = await _call_llm_cached(
        prompt=r1_prompt,
        cache=caches["cue_round1"],
        openai_client=openai_client,
    )
    if r1_hit:
        cache_hits["cue_round1"] += 1
    else:
        llm_calls += 1
    r1_cues = parse_speaker_cues(r1_text, max_cues=3)
    all_cues.extend(r1_cues)

    # Retrieve round 1 (no scratch yet).
    r1_hits = await _retrieve_with_cues_and_scratch(
        memory,
        question=question,
        cues=r1_cues,
        scratch=scratch,
        K=max_K,
    )
    round_hits.append(r1_hits)

    # Reflect after round 1.
    turns_ctx = format_turns_section(
        [
            {"turn_id": h.turn_id, "role": h.role, "text": h.text}
            for h in r1_hits[:20]
        ]
    )
    reflect_prompt_1 = REFLECTION_PROMPT.format(
        question=question,
        scratch_section=format_scratch_section(scratch),
        turns_section=turns_ctx,
    )
    reflect_text_1, rhit1 = await _call_llm_cached(
        prompt=reflect_prompt_1,
        cache=caches["reflect"],
        openai_client=openai_client,
    )
    if rhit1:
        cache_hits["reflect"] += 1
    else:
        llm_calls += 1
    learned_1, still_need_1 = parse_reflection_json(reflect_text_1)

    # Embed reflection and add to scratch.
    r1_scratch_texts = learned_1 + still_need_1
    r1_kinds = (
        ["learned"] * len(learned_1) + ["still_need"] * len(still_need_1)
    )
    r1_embs = await _embed_texts(embedder, r1_scratch_texts)
    for t, k, e in zip(r1_scratch_texts, r1_kinds, r1_embs):
        scratch.add(text=t, kind=k, embedding=e)

    round_records.append(
        {
            "round": 1,
            "cues": r1_cues,
            "learned": learned_1,
            "still_need": still_need_1,
            "n_hits": len(r1_hits),
        }
    )

    # ---------------------------------------------------------------------
    # Rounds 2..num_rounds.
    # ---------------------------------------------------------------------
    for r in range(2, num_rounds + 1):
        # Generate new cues informed by scratch.
        cue_prompt = ROUND_N_CUE_PROMPT.format(
            question=question,
            scratch_section=format_scratch_section(scratch),
            prev_cues_section=format_prev_cues_section(all_cues),
        )
        cue_text, chit = await _call_llm_cached(
            prompt=cue_prompt,
            cache=caches["cue_roundn"],
            openai_client=openai_client,
        )
        if chit:
            cache_hits["cue_roundn"] += 1
        else:
            llm_calls += 1
        new_cues = parse_speaker_cues(cue_text, max_cues=3)
        all_cues.extend(new_cues)

        # Retrieve with all cues so far AND scratch entries as extra anchors.
        r_hits = await _retrieve_with_cues_and_scratch(
            memory,
            question=question,
            cues=all_cues,
            scratch=scratch,
            K=max_K,
        )
        round_hits.append(r_hits)

        # Reflect after round r.
        turns_ctx = format_turns_section(
            [
                {"turn_id": h.turn_id, "role": h.role, "text": h.text}
                for h in r_hits[:20]
            ]
        )
        reflect_prompt_r = REFLECTION_PROMPT.format(
            question=question,
            scratch_section=format_scratch_section(scratch),
            turns_section=turns_ctx,
        )
        reflect_text_r, rhit_r = await _call_llm_cached(
            prompt=reflect_prompt_r,
            cache=caches["reflect"],
            openai_client=openai_client,
        )
        if rhit_r:
            cache_hits["reflect"] += 1
        else:
            llm_calls += 1
        learned_r, still_need_r = parse_reflection_json(reflect_text_r)

        new_texts = learned_r + still_need_r
        new_kinds = (
            ["learned"] * len(learned_r) + ["still_need"] * len(still_need_r)
        )
        new_embs = await _embed_texts(embedder, new_texts)
        for t, k, e in zip(new_texts, new_kinds, new_embs):
            scratch.add(text=t, kind=k, embedding=e)

        round_records.append(
            {
                "round": r,
                "cues": new_cues,
                "learned": learned_r,
                "still_need": still_need_r,
                "n_hits": len(r_hits),
            }
        )

    return {
        "round_hits": round_hits,
        "round_records": round_records,
        "all_cues": all_cues,
        "scratch_text_snapshot": scratch.as_text_lines(),
        "cache_hits": cache_hits,
        "llm_calls": llm_calls,
    }


# -----------------------------------------------------------------------------
# Evaluation helpers.
# -----------------------------------------------------------------------------


def compute_recall(retrieved: set[int], gold: set[int]) -> float:
    if not gold:
        return 1.0
    return len(retrieved & gold) / len(gold)


def load_questions() -> list[dict]:
    with open(HARD_QUESTIONS_JSON) as f:
        return json.load(f)


def load_collections_meta() -> dict:
    with open(COLLECTIONS_JSON) as f:
        return json.load(f)


async def evaluate_one(
    arch_name: str,
    memory: EventMemory,
    question: dict,
    caches: dict[str, _MergedLLMCache],
    embedder: OpenAIEmbedder,
    openai_client,
    *,
    num_rounds: int,
    max_K: int,
) -> dict:
    q_text = question["question"]
    gold = set(question.get("source_chat_ids", []))

    t0 = time.monotonic()
    out = await run_reflmem_query(
        memory,
        question=q_text,
        num_rounds=num_rounds,
        max_K=max_K,
        caches=caches,
        embedder=embedder,
        openai_client=openai_client,
    )
    elapsed = time.monotonic() - t0

    # Recall per round per K (for diagnostic novelty analysis).
    per_round_metrics: list[dict] = []
    for i, hits in enumerate(out["round_hits"], start=1):
        row = {"round": i}
        for K in BUDGETS:
            topk = hits[:K]
            retrieved = {h.turn_id for h in topk}
            row[f"r@{K}"] = round(compute_recall(retrieved, gold), 4)
            row[f"retrieved_set_at_{K}"] = sorted(retrieved)
        per_round_metrics.append(row)

    # Final round hits (the variant's "output").
    final_hits = out["round_hits"][-1]
    result: dict = {
        "question_id": question["question_id"],
        "category": question.get("category", "unknown"),
        "question": q_text,
        "num_gold": len(gold),
        "n_hits": len(final_hits),
        "time_s": round(elapsed, 3),
        "llm_calls": out["llm_calls"],
        "cache_hits": out["cache_hits"],
        "num_rounds_executed": num_rounds,
        "rounds": [
            {
                "round": rec["round"],
                "cues": rec["cues"],
                "learned": rec["learned"],
                "still_need": rec["still_need"],
                "n_hits": rec["n_hits"],
            }
            for rec in out["round_records"]
        ],
        "scratch_final": out["scratch_text_snapshot"],
        "per_round_metrics": per_round_metrics,
    }

    for K in BUDGETS:
        topk = final_hits[:K]
        retrieved = {h.turn_id for h in topk}
        result[f"r@{K}"] = round(compute_recall(retrieved, gold), 4)
    return result


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archs",
        default="reflmemlme_1round,reflmemlme_2round,reflmemlme_3round",
        help="Comma-separated variants to run.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    variant_rounds = {
        "reflmemlme_1round": 1,
        "reflmemlme_2round": 2,
        "reflmemlme_3round": 3,
    }
    for a in archs:
        if a not in variant_rounds:
            raise ValueError(f"Unknown variant: {a}")

    questions = load_questions()
    if args.limit is not None:
        questions = questions[: args.limit]
    print(f"[reflmemlme_eval] n_questions={len(questions)}", flush=True)

    meta = load_collections_meta()
    qid_to_meta = {r["question_id"]: r for r in meta["questions"]}

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    sql_url = meta["sql_url"]
    if sql_url.startswith("sqlite"):
        engine = create_async_engine(sql_url)
    else:
        engine = create_async_engine(sql_url, pool_size=20, max_overflow=20)
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await segment_store.startup()

    openai_client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model="text-embedding-3-small",
            dimensions=1536,
            max_input_length=8192,
        )
    )

    caches: dict[str, _MergedLLMCache] = {
        # Round 1 cue cache: read-through lmetune_v2f_mixed7030_cache.json
        # (identical prompt text) then our dedicated cache.
        "cue_round1": _MergedLLMCache(
            reader_paths=[
                LMETUNE_V2F_MIXED7030_CACHE,
                REFLMEMLME_CUE_ROUND1_CACHE,
            ],
            writer_path=REFLMEMLME_CUE_ROUND1_CACHE,
        ),
        "cue_roundn": _MergedLLMCache(
            reader_paths=[REFLMEMLME_CUE_ROUNDN_CACHE],
            writer_path=REFLMEMLME_CUE_ROUNDN_CACHE,
        ),
        "reflect": _MergedLLMCache(
            reader_paths=[REFLMEMLME_REFLECT_CACHE],
            writer_path=REFLMEMLME_REFLECT_CACHE,
        ),
    }

    # Open memories for each question.
    memories: dict[str, EventMemory] = {}
    opened: list = []
    for q in questions:
        qm = qid_to_meta[q["question_id"]]
        coll = await vector_store.open_collection(
            namespace=qm["namespace"], name=qm["collection_name"]
        )
        part = await segment_store.open_or_create_partition(qm["partition_key"])
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
        opened.append((coll, part))

    max_K = max(BUDGETS)
    results: dict = {
        "archs": {},
        "budgets": list(BUDGETS),
        "questions": len(questions),
    }
    semaphore = asyncio.Semaphore(ARCH_CONCURRENCY)

    async def run_one(arch: str, q: dict) -> dict:
        async with semaphore:
            return await evaluate_one(
                arch,
                memories[q["question_id"]],
                q,
                caches,
                embedder,
                openai_client,
                num_rounds=variant_rounds[arch],
                max_K=max_K,
            )

    try:
        for arch in archs:
            t_arch = time.monotonic()
            tasks = [run_one(arch, q) for q in questions]
            rows = await asyncio.gather(*tasks)
            arch_elapsed = time.monotonic() - t_arch

            n = len(rows)
            summary = {
                "n": n,
                "time_s": round(arch_elapsed, 1),
                "avg_llm_calls": round(
                    sum(r["llm_calls"] for r in rows) / max(n, 1), 2
                ),
                "avg_rounds_executed": round(
                    sum(r["num_rounds_executed"] for r in rows) / max(n, 1), 2
                ),
            }
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

            # Round-novelty analysis: across rounds, fraction of queries
            # where round r's retrieved set at K contains ≥1 gold id NOT
            # in round (r-1)'s set.
            round_novelty: dict[str, dict] = {}
            for K in BUDGETS:
                entry: dict = {"K": K, "rounds": {}}
                for r_idx in range(1, variant_rounds[arch] + 1):
                    n_gain_any = 0
                    n_nontrivial = 0
                    recall_delta_sum = 0.0
                    for row in rows:
                        gold = set()
                        # Recover gold from recall and retrieved sets.
                        pr = row["per_round_metrics"]
                        # pr entries are ordered by round; index r_idx-1 is this round.
                        if r_idx - 1 >= len(pr):
                            continue
                        this_set = set(pr[r_idx - 1][f"retrieved_set_at_{K}"])
                        q = next(
                            (qq for qq in questions if qq["question_id"] == row["question_id"]),
                            None,
                        )
                        if q is None:
                            continue
                        gold = set(q.get("source_chat_ids", []))
                        if not gold:
                            continue
                        this_r = len(this_set & gold) / len(gold)
                        if r_idx == 1:
                            prev_r = 0.0
                            prev_set: set[int] = set()
                        else:
                            prev_set = set(pr[r_idx - 2][f"retrieved_set_at_{K}"])
                            prev_r = len(prev_set & gold) / len(gold)
                        delta = this_r - prev_r
                        novel_gold = (this_set & gold) - (prev_set & gold)
                        if len(novel_gold) >= 1:
                            n_gain_any += 1
                        if delta > 0.0001:
                            n_nontrivial += 1
                        recall_delta_sum += delta
                    total = sum(
                        1
                        for qq in questions
                        if qq.get("source_chat_ids")
                    )
                    entry["rounds"][str(r_idx)] = {
                        "frac_queries_with_novel_gold": round(
                            n_gain_any / max(total, 1), 4
                        ),
                        "frac_queries_with_recall_gain": round(
                            n_nontrivial / max(total, 1), 4
                        ),
                        "mean_recall_delta": round(
                            recall_delta_sum / max(total, 1), 4
                        ),
                    }
                round_novelty[f"K={K}"] = entry

            # Strip bulky intermediate fields from persisted rows before
            # writing JSON to keep output small; keep rounds + scratch.
            persisted_rows = []
            for r in rows:
                rp = {k: v for k, v in r.items() if k != "per_round_metrics"}
                # shrink per_round_metrics (drop retrieved sets)
                prm = []
                for m in r["per_round_metrics"]:
                    prm.append({k: v for k, v in m.items() if not k.startswith("retrieved_set")})
                rp["per_round_metrics"] = prm
                persisted_rows.append(rp)

            results["archs"][arch] = {
                "summary": summary,
                "by_category": cat_summary,
                "round_novelty": round_novelty,
                "per_question": persisted_rows,
            }

            for c in caches.values():
                c.save()

            cat_str = " ".join(
                f"{c}={cat_summary[c].get('mean_r@50', 0):.3f}"
                for c in sorted(cat_summary)
            )
            print(
                f"[{arch}] n={n} "
                f"r@20={summary['mean_r@20']:.4f} r@50={summary['mean_r@50']:.4f} "
                f"llm_calls={summary['avg_llm_calls']:.2f} "
                f"({cat_str}) in {arch_elapsed:.1f}s",
                flush=True,
            )
    finally:
        for coll, part in opened:
            await segment_store.close_partition(part)
            await vector_store.close_collection(collection=coll)
        await segment_store.shutdown()
        await vector_store.shutdown()
        await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {RESULTS_JSON}", flush=True)
    write_markdown_report(results)


def write_markdown_report(results: dict) -> None:
    archs = results["archs"]
    md: list[str] = []
    md += [
        "# LME-hard Reflective Memory (scratch-memory write during query)",
        "",
        "All variants use `expand_context=3` + `User: ` prefix. Round-1 "
        "cue prompt is `em_v2f_lme_mixed_7030` (cache-reused). "
        "Reflection writes `learned` + `still_need` scratch entries; "
        "retrieval uses corpus + scratch entries as additional anchors, "
        "merged by max-per-turn.",
        "",
        "## References (LME-hard R@50 overall)",
        "",
        f"- `em_v2f_expand_3` baseline: {REF_EM_V2F_EXPAND_3:.3f}",
        f"- `em_ens_2_lme_retuned`: {REF_EM_ENS_2_LME_RETUNED:.3f}",
        f"- `em_v2f_lme_mixed_7030` ceiling: {REF_EM_V2F_LME_MIXED_7030:.3f}",
        "",
        "## Per-variant summary",
        "",
        "| Variant | R@20 | R@50 | Δ vs mixed_7030 (0.8631) | avg LLM calls | avg rounds | time (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arch, data in archs.items():
        s = data["summary"]
        delta = s["mean_r@50"] - REF_EM_V2F_LME_MIXED_7030
        md.append(
            f"| `{arch}` | {s['mean_r@20']:.4f} | {s['mean_r@50']:.4f} | "
            f"{delta:+.4f} | {s.get('avg_llm_calls', 0):.2f} | "
            f"{s.get('avg_rounds_executed', 0):.2f} | {s['time_s']:.1f} |"
        )

    for K in BUDGETS:
        md += [
            "",
            f"## Recall matrix (R@{K})",
            "",
            "| Variant | multi-session | single-session-preference | "
            "temporal-reasoning |",
            "| --- | --- | --- | --- |",
        ]
        for arch, data in archs.items():
            bc = data["by_category"]
            md.append(
                f"| `{arch}` | "
                f"{bc.get('multi-session', {}).get(f'mean_r@{K}', 0):.4f} | "
                f"{bc.get('single-session-preference', {}).get(f'mean_r@{K}', 0):.4f} | "
                f"{bc.get('temporal-reasoning', {}).get(f'mean_r@{K}', 0):.4f} |"
            )

    # Reference per-category row for mixed_7030.
    md += [
        "",
        "### Reference: `em_v2f_lme_mixed_7030` per-category @ K=50",
        "",
        f"- multi-session: {REF_MS_MIXED_7030:.4f}",
        f"- single-session-preference: {REF_SSP_MIXED_7030:.4f}",
        f"- temporal-reasoning: {REF_TR_MIXED_7030:.4f}",
    ]

    # Round-novelty analysis (for 2-round and 3-round variants).
    md += ["", "## Round-novelty analysis"]
    for arch, data in archs.items():
        md += ["", f"### `{arch}`"]
        rn = data.get("round_novelty", {})
        for K_key, entry in rn.items():
            md.append(f"- {K_key}:")
            for r_idx, r_data in entry["rounds"].items():
                md.append(
                    f"  - Round {r_idx}: frac_queries_with_novel_gold="
                    f"{r_data['frac_queries_with_novel_gold']:.3f}, "
                    f"frac_queries_with_recall_gain="
                    f"{r_data['frac_queries_with_recall_gain']:.3f}, "
                    f"mean_recall_delta={r_data['mean_recall_delta']:+.4f}"
                )

    # Sample: pick 2 questions from 2round where round-2 added novel gold,
    # fall back to first 2.
    sample_arch = (
        "reflmemlme_2round"
        if "reflmemlme_2round" in archs
        else list(archs.keys())[0]
    )
    md += ["", f"## Sample scratch state + round-2 gains (from `{sample_arch}`)"]
    sample_rows = archs[sample_arch]["per_question"]

    def _has_round2_gain(row: dict) -> bool:
        prm = row.get("per_round_metrics", [])
        if len(prm) < 2:
            return False
        return prm[1].get("r@50", 0.0) > prm[0].get("r@50", 0.0) + 0.0001

    samples = [r for r in sample_rows if _has_round2_gain(r)][:2]
    if len(samples) < 2:
        for r in sample_rows:
            if r not in samples:
                samples.append(r)
            if len(samples) >= 2:
                break

    for r in samples:
        md += ["", f"### Q `{r['question_id']}` ({r['category']})"]
        md.append(f"- Question: {r['question']}")
        md.append(f"- Gold turns: {r['num_gold']}")
        for rec in r.get("rounds", []):
            md.append(f"- Round {rec['round']} cues:")
            for c in rec["cues"]:
                md.append(f"  - `{c}`")
            if rec["learned"]:
                md.append(f"- Round {rec['round']} LEARNED:")
                for ln in rec["learned"]:
                    md.append(f"  - `{ln}`")
            if rec["still_need"]:
                md.append(f"- Round {rec['round']} STILL_NEED:")
                for sn in rec["still_need"]:
                    md.append(f"  - `{sn}`")
        prm = r.get("per_round_metrics", [])
        md.append("- Per-round recall:")
        for m in prm:
            md.append(
                f"  - Round {m['round']}: r@20={m.get('r@20', 0):.3f} "
                f"r@50={m.get('r@50', 0):.3f}"
            )

    # Verdict
    best_name, best_r50 = None, -1.0
    for arch, data in archs.items():
        v = data["summary"]["mean_r@50"]
        if v > best_r50:
            best_name, best_r50 = arch, v

    md += [
        "",
        "## Verdict",
        "",
        f"- Top variant: `{best_name}` R@50 = {best_r50:.4f} "
        f"(Δ vs em_v2f_lme_mixed_7030 {REF_EM_V2F_LME_MIXED_7030:.4f} = "
        f"{best_r50 - REF_EM_V2F_LME_MIXED_7030:+.4f})",
        "",
        "Decision rule check:",
        f"- If best R@50 > 0.863 overall: reflective memory lifts the "
        f"mixed_7030 ceiling. (best = {best_r50:.4f}, "
        f"{'LIFT' if best_r50 > REF_EM_V2F_LME_MIXED_7030 else 'NO-LIFT'})",
    ]
    # Temporal-reasoning check at K=50.
    if best_name is not None:
        tr_best = archs[best_name]["by_category"].get(
            "temporal-reasoning", {}
        ).get("mean_r@50", 0.0)
        md.append(
            f"- Temporal-reasoning (best variant @ K=50): {tr_best:.4f} "
            f"(ref {REF_TR_MIXED_7030:.4f}; "
            f"{'LIFT' if tr_best > REF_TR_MIXED_7030 else 'NO-LIFT'})"
        )

    md += [
        "",
        "## Outputs",
        "",
        f"- JSON: `{RESULTS_JSON.relative_to(ASSOC_DIR)}`",
        "- Sources: `reflective_memory_lme.py`, `reflmemlme_eval.py`",
        "- Caches: "
        "`cache/reflmemlme_cue_round1_cache.json` (reads from "
        "`cache/lmetune_v2f_mixed7030_cache.json`), "
        "`cache/reflmemlme_cue_roundn_cache.json`, "
        "`cache/reflmemlme_reflect_cache.json`",
    ]

    RESULTS_MD.write_text("\n".join(md))
    print(f"Saved: {RESULTS_MD}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
