"""Additive-only steering with varying phrase granularity.

Tests whether short keyword embeddings concentrate semantic mass more cleanly
than full-sentence embeddings, yielding more targeted probe-direction updates.

Three granularity modes (single-round additive only, no subtract):
  - keyword      : 2-4 keywords (1-3 words each)
  - short_phrase : 2-3 short phrases (4-8 words)
  - sentence     : 2-3 sentence-fragments (control; echoes v2 addonly)

Two aggregation modes:
  - arithmetic   : new_probe = normalize(probe + alpha * sum(normalize(emb_i)))
  - probe_union  : retrieve once per phrase, score-merge by sum_cosine

Evaluated on LME-hard POC (30 questions), baseline = round-0 snapshot from
v2f speaker-format cue (`baseline_v2f_direct` ~ 0.8169 R@50 in prior sweep).

Outputs:
  results/keyword_add_sweep.json
  results/keyword_add_sweep.md
Caches:
  cache/keyadd_llm_cache.json
  cache/keyadd_embedding_cache.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from uuid import UUID

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

OUT_JSON = RESULTS_DIR / "keyword_add_sweep.json"
OUT_MD = RESULTS_DIR / "keyword_add_sweep.md"

# Dedicated caches per-task (don't overlap other agents).
KEYADD_LLM_CACHE = CACHE_DIR / "keyadd_llm_cache.json"
KEYADD_EMB_CACHE = CACHE_DIR / "keyadd_embedding_cache.json"

BUDGETS = (20, 50)
USER_PREFIX = "User: "
EXPAND_CONTEXT = 3  # LME recipe
TOPK_FOR_LLM = 5
VECTOR_SEARCH_LIMIT = 50


# --------------------------------------------------------------------------
# Prompts per granularity
# --------------------------------------------------------------------------


_PROMPT_SHARED_HEADER = """\
You are steering a semantic retrieval probe toward gold-answering content.

Query: {query}
Original cue: {cue}

Current top-{topk} retrieved turns:
{retrieved_turns}

Task:
- CLASSIFY which retrieved turns look gold-likely (answering the query).
- Then extract ADD terms grounded in (a) exact words from the query, or
  (b) specific terms from gold-likely turns. NO fabricated specifics
  (no invented dates/names/titles/numbers).
"""


PROMPT_KEYWORD = _PROMPT_SHARED_HEADER + """
Extract 2-4 KEYWORDS (each 1-3 words) representing the specific concept the
probe should move toward.

Output STRICT JSON (no code fence):
{{"gold_likely_indices": [idx, ...], "add_terms": ["kw1", "kw2", ...], \
"reasoning": "one short sentence"}}

Rules:
- 2-4 items, each 1-3 words.
- No questions, no generic words like "information" or "detail".
"""


PROMPT_SHORT_PHRASE = _PROMPT_SHARED_HEADER + """
Extract 2-3 SHORT PHRASES (each 4-8 words) that target the specific concept
the probe should move toward.

Output STRICT JSON (no code fence):
{{"gold_likely_indices": [idx, ...], "add_terms": ["phrase1", "phrase2", ...], \
"reasoning": "one short sentence"}}

Rules:
- 2-3 items, each 4-8 words.
- Concrete, no questions.
"""


PROMPT_SENTENCE = _PROMPT_SHARED_HEADER + """
Extract 2-3 sentence-fragments (each < 20 words) that target the specific
concept the probe should move toward.

Output STRICT JSON (no code fence):
{{"gold_likely_indices": [idx, ...], "add_terms": ["frag1", "frag2", ...], \
"reasoning": "one short sentence"}}

Rules:
- 2-3 items, each < 20 words.
- Concrete, no questions.
"""


PROMPT_BY_GRAN = {
    "keyword": PROMPT_KEYWORD,
    "short_phrase": PROMPT_SHORT_PHRASE,
    "sentence": PROMPT_SENTENCE,
}


# --------------------------------------------------------------------------
# Helpers (self-contained; mirror v2 style without importing the module)
# --------------------------------------------------------------------------


def _ensure_user_prefix(text: str) -> str:
    t = text.lstrip()
    low = t.lower()
    if low.startswith("user:") or low.startswith("assistant:"):
        return t
    return USER_PREFIX + t


def _format_retrieved_snippet_indexed(hits: list[EMHit], n: int) -> str:
    lines = []
    for i, h in enumerate(hits[:n]):
        text = (h.text or "").replace("\n", " ")[:280]
        lines.append(f"[{i}] ({h.role}) {text}")
    return "\n".join(lines) if lines else "(no turns retrieved yet)"


def _parse_llm_json(response: str) -> dict:
    empty = {"gold_likely_indices": [], "add_terms": [], "reasoning": ""}
    if not response:
        return empty
    m = re.search(r"\{.*\}", response, re.DOTALL)
    raw = m.group(0) if m else response.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(raw)
    except Exception:
        return empty
    out = dict(empty)
    try:
        out["gold_likely_indices"] = [
            int(x) for x in obj.get("gold_likely_indices", []) if isinstance(x, (int, float))
        ]
    except Exception:
        out["gold_likely_indices"] = []
    adds = obj.get("add_terms", [])
    if isinstance(adds, list):
        out["add_terms"] = [
            str(p).strip() for p in adds if isinstance(p, str) and p.strip()
        ]
    out["reasoning"] = str(obj.get("reasoning", ""))
    return out


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# --------------------------------------------------------------------------
# Initial cue builder (matches steerv2_eval)
# --------------------------------------------------------------------------


async def _build_lme_v2f_cue(
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
# Variant runner: single-round add-only with granularity G, alpha A,
# aggregation M in {arithmetic, probe_union}.
# --------------------------------------------------------------------------


@dataclass
class VariantConfig:
    name: str
    granularity: str            # keyword | short_phrase | sentence
    alpha: float                # scale for arithmetic mode
    aggregation: str            # arithmetic | probe_union | baseline


VARIANTS: list[VariantConfig] = [
    VariantConfig("baseline", "sentence", 0.0, "baseline"),
    VariantConfig("keyadd_kw_a0.05",     "keyword",      0.05, "arithmetic"),
    VariantConfig("keyadd_kw_a0.1",      "keyword",      0.10, "arithmetic"),
    VariantConfig("keyadd_kw_a0.2",      "keyword",      0.20, "arithmetic"),
    VariantConfig("keyadd_short_a0.05",  "short_phrase", 0.05, "arithmetic"),
    VariantConfig("keyadd_short_a0.1",   "short_phrase", 0.10, "arithmetic"),
    VariantConfig("keyadd_sent_a0.05",   "sentence",     0.05, "arithmetic"),
    VariantConfig("keyadd_probe_union",  "keyword",      0.0,  "probe_union"),
]


async def _classify_and_extract(
    *,
    granularity: str,
    query_text: str,
    initial_cue_text: str,
    topk_hits: list[EMHit],
    openai_client,
    llm_cache: _MergedLLMCache,
) -> dict:
    prompt = PROMPT_BY_GRAN[granularity].format(
        query=query_text,
        cue=initial_cue_text,
        topk=TOPK_FOR_LLM,
        retrieved_turns=_format_retrieved_snippet_indexed(topk_hits, TOPK_FOR_LLM),
    )
    cached = llm_cache.get(V2F_MODEL, prompt)
    if cached is None:
        resp = await openai_client.chat.completions.create(
            model=V2F_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        cached = resp.choices[0].message.content or ""
        llm_cache.put(V2F_MODEL, prompt, cached)
    return _parse_llm_json(cached)


async def run_one_question(
    variant: VariantConfig,
    memory: EventMemory,
    question: dict,
    *,
    embedder: OpenAIEmbedder,
    openai_client,
    v2f_cache: _MergedLLMCache,
    llm_cache: _MergedLLMCache,
    emb_cache: EmbeddingCache,
) -> dict:
    q_text = question["question"]
    gold = set(question.get("source_chat_ids", []))

    t0 = time.monotonic()

    # 1. Initial v2f cue + round-0 retrieval.
    initial_text, v2f_cues = await _build_lme_v2f_cue(
        memory, q_text, v2f_cache, openai_client
    )
    cue_vec = await cached_embed(embedder, initial_text, emb_cache=emb_cache)
    probe = _normalize(np.array(cue_vec, dtype=np.float64))

    r0_hits = _dedupe_by_turn_id(await _query_em_by_vector(
        memory,
        probe,
        vector_search_limit=VECTOR_SEARCH_LIMIT,
        expand_context=EXPAND_CONTEXT,
    ))
    baseline_r = {}
    for K in BUDGETS:
        r = {h.turn_id for h in r0_hits[:K]}
        baseline_r[f"r@{K}"] = len(r & gold) / len(gold) if gold else 1.0

    row: dict = {
        "variant": variant.name,
        "question_id": question.get("question_id"),
        "category": question.get("category", "unknown"),
        "question": q_text,
        "initial_cue_text": initial_text,
        "v2f_cues": v2f_cues,
        "baseline_recall": {k: round(v, 4) for k, v in baseline_r.items()},
    }

    if variant.aggregation == "baseline":
        row.update({f"r@{K}": round(baseline_r[f"r@{K}"], 4) for K in BUDGETS})
        row["add_terms"] = []
        row["gold_likely_indices"] = []
        row["time_s"] = round(time.monotonic() - t0, 3)
        return row

    # 2. LLM classify + extract ADD terms at requested granularity.
    parsed = await _classify_and_extract(
        granularity=variant.granularity,
        query_text=q_text,
        initial_cue_text=initial_text,
        topk_hits=r0_hits[:TOPK_FOR_LLM],
        openai_client=openai_client,
        llm_cache=llm_cache,
    )
    gold_likely_idx = [
        i for i in parsed["gold_likely_indices"] if 0 <= i < min(len(r0_hits), TOPK_FOR_LLM)
    ]
    add_terms = parsed["add_terms"]
    row["add_terms"] = add_terms
    row["gold_likely_indices"] = gold_likely_idx
    row["reasoning"] = parsed.get("reasoning", "")

    # 3. Aggregate.
    if variant.aggregation == "arithmetic":
        add_sum = np.zeros_like(probe)
        for term in add_terms:
            vec = await cached_embed(embedder, term, emb_cache=emb_cache)
            add_sum += _normalize(np.array(vec, dtype=np.float64))
        new_probe = _normalize(probe + variant.alpha * add_sum)

        hits = _dedupe_by_turn_id(await _query_em_by_vector(
            memory,
            new_probe,
            vector_search_limit=VECTOR_SEARCH_LIMIT,
            expand_context=EXPAND_CONTEXT,
        ))
        for K in BUDGETS:
            r = {h.turn_id for h in hits[:K]}
            row[f"r@{K}"] = round(
                len(r & gold) / len(gold) if gold else 1.0, 4
            )
        row["time_s"] = round(time.monotonic() - t0, 3)
        return row

    # 4. probe_union: retrieve once per term, merge by sum_cosine. Also
    # include the baseline-cue probe in the union (so keywords AUGMENT,
    # not replace, the baseline).
    if variant.aggregation == "probe_union":
        per_probe_vectors: list[np.ndarray] = [probe]
        for term in add_terms:
            vec = await cached_embed(embedder, term, emb_cache=emb_cache)
            per_probe_vectors.append(_normalize(np.array(vec, dtype=np.float64)))

        # Merge by summing cosine scores across probes. Each probe retrieves
        # up to VECTOR_SEARCH_LIMIT seeds, expand_context applied per seed.
        score_by_turn: dict[int, float] = defaultdict(float)
        repr_hit: dict[int, EMHit] = {}
        for pv in per_probe_vectors:
            hits = _dedupe_by_turn_id(await _query_em_by_vector(
                memory,
                pv,
                vector_search_limit=VECTOR_SEARCH_LIMIT,
                expand_context=EXPAND_CONTEXT,
            ))
            for h in hits:
                score_by_turn[h.turn_id] += float(h.score)
                if h.turn_id not in repr_hit:
                    repr_hit[h.turn_id] = h

        merged = sorted(repr_hit.values(), key=lambda h: score_by_turn[h.turn_id], reverse=True)
        for K in BUDGETS:
            r = {h.turn_id for h in merged[:K]}
            row[f"r@{K}"] = round(
                len(r & gold) / len(gold) if gold else 1.0, 4
            )
        row["n_probes"] = len(per_probe_vectors)
        row["time_s"] = round(time.monotonic() - t0, 3)
        return row

    raise ValueError(f"Unknown aggregation: {variant.aggregation}")


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
        default=",".join(v.name for v in VARIANTS),
    )
    parser.add_argument("--lme-per-cat", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    selected_names = {v.strip() for v in args.variants.split(",") if v.strip()}
    variants = [v for v in VARIANTS if v.name in selected_names]
    lme_qs = load_lme_questions(args.lme_per_cat)
    print(f"[keyword_add_sweep] LME n={len(lme_qs)}, variants={len(variants)}", flush=True)

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
    llm_cache = _MergedLLMCache(
        reader_paths=[KEYADD_LLM_CACHE],
        writer_path=KEYADD_LLM_CACHE,
    )
    emb_cache = EmbeddingCache(KEYADD_EMB_CACHE)

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
        sem = asyncio.Semaphore(args.concurrency)

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
                    llm_cache=llm_cache,
                    emb_cache=emb_cache,
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

            results["variants"][variant.name] = {
                "config": {
                    "granularity": variant.granularity,
                    "alpha": variant.alpha,
                    "aggregation": variant.aggregation,
                },
                "summary": summary,
                "by_category": cat_summary,
                "per_question": rows,
            }

            # Persist after each variant.
            lme_v2f_cache.save()
            llm_cache.save()
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


def write_report(results: dict) -> None:
    variants = results["variants"]
    lines: list[str] = []
    lines.append("# Keyword/Granularity Add-only Steering Sweep")
    lines.append("")
    lines.append(
        "Single-round, additive-only probe update across granularities "
        "(keyword | short_phrase | sentence) and alpha scales. Unit-normalized "
        "phrase embeddings. Baseline = round-0 retrieval from v2f_lme_mixed_7030 "
        "speaker-format cue + expand_3."
    )
    lines.append("")
    lines.append(
        f"Fixed: text-embedding-3-small, {V2F_MODEL}, LME-hard-30 POC, "
        "vector_search_limit=50, topk_for_llm=5."
    )
    lines.append("")

    # Recall matrix.
    lines.append("## Recall matrix")
    lines.append("")
    lines.append("| Variant | granularity | α | aggregation | R@20 | R@50 | time (s) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, per in variants.items():
        cfg = per["config"]
        s = per["summary"]
        lines.append(
            f"| `{name}` | {cfg['granularity']} | {cfg['alpha']} | "
            f"{cfg['aggregation']} | {s['mean_r@20']:.4f} | "
            f"{s['mean_r@50']:.4f} | {s['time_s']:.1f} |"
        )
    lines.append("")

    # Δ vs baseline.
    base = variants.get("baseline", {}).get("summary", {})
    base_r50 = base.get("mean_r@50", 0.0)
    base_r20 = base.get("mean_r@20", 0.0)
    lines.append("## Δ vs baseline")
    lines.append("")
    lines.append("| Variant | R@20 Δ | R@50 Δ |")
    lines.append("| --- | --- | --- |")
    for name, per in variants.items():
        if name == "baseline":
            continue
        s = per["summary"]
        lines.append(
            f"| `{name}` | {s['mean_r@20'] - base_r20:+.4f} | "
            f"{s['mean_r@50'] - base_r50:+.4f} |"
        )
    lines.append("")

    # Per-category R@50.
    lines.append("## Per-category R@50")
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

    # Sample add terms for 3 questions from each non-baseline variant's first 3 questions.
    lines.append("## Sample add terms")
    lines.append("")
    shown = set()
    for name, per in variants.items():
        if name == "baseline":
            continue
        rows = per.get("per_question", [])[:3]
        if not rows:
            continue
        lines.append(f"### {name}")
        lines.append("")
        for r in rows:
            qid = r.get("question_id")
            if qid in shown:
                # Still show -- terms differ per variant.
                pass
            shown.add(qid)
            lines.append(f"- Q `{qid}` ({r.get('category')}): `{r['question'][:100]}`")
            lines.append(f"  - add_terms: {r.get('add_terms', [])}")
            if "reasoning" in r and r["reasoning"]:
                lines.append(f"  - reasoning: {r['reasoning']}")
        lines.append("")

    # Granularity comparison at best α per granularity.
    lines.append("## Granularity verdict")
    lines.append("")
    arith = {
        name: per for name, per in variants.items()
        if per["config"]["aggregation"] == "arithmetic"
    }
    if arith:
        best_by_gran: dict[str, tuple[str, float]] = {}
        for name, per in arith.items():
            gran = per["config"]["granularity"]
            r50 = per["summary"]["mean_r@50"]
            if gran not in best_by_gran or r50 > best_by_gran[gran][1]:
                best_by_gran[gran] = (name, r50)
        for gran, (name, r50) in best_by_gran.items():
            delta = r50 - base_r50
            lines.append(
                f"- {gran}: best variant `{name}` R@50 = {r50:.4f} "
                f"(Δ vs baseline = {delta:+.4f})"
            )
        lines.append("")

    # Arithmetic vs probe_union.
    union = variants.get("keyadd_probe_union")
    if union and arith:
        lines.append("## Arithmetic vs probe_union")
        lines.append("")
        best_arith_name, best_arith_r50 = max(
            ((n, p["summary"]["mean_r@50"]) for n, p in arith.items()),
            key=lambda x: x[1],
        )
        union_r50 = union["summary"]["mean_r@50"]
        lines.append(f"- best arithmetic: `{best_arith_name}` R@50 = {best_arith_r50:.4f}")
        lines.append(f"- probe_union: R@50 = {union_r50:.4f}")
        lines.append(f"- Δ (union - best_arith) = {union_r50 - best_arith_r50:+.4f}")
        lines.append(f"- Δ (union - baseline) = {union_r50 - base_r50:+.4f}")
        lines.append("")

    # Verdict block.
    lines.append("## Verdict")
    lines.append("")
    if arith:
        best_name, best_r50 = max(
            ((n, p["summary"]["mean_r@50"]) for n, p in arith.items()),
            key=lambda x: x[1],
        )
        delta = best_r50 - base_r50
        if delta >= 0.01:
            lines.append(
                f"Shorter/tuned-α arithmetic add HELPS: best `{best_name}` "
                f"R@50 = {best_r50:.4f} (Δ = {delta:+.4f} ≥ +1pp). "
                f"Hypothesis supported."
            )
        elif abs(delta) < 0.005:
            lines.append(
                f"No meaningful lift from arithmetic add at any granularity "
                f"(best Δ = {delta:+.4f}). Shorter is not better under arithmetic."
            )
        else:
            lines.append(
                f"Arithmetic add at best granularity yields Δ = {delta:+.4f}. "
                f"Shorter phrases are {'less harmful' if delta > 0 else 'still harmful'} "
                f"but do not clear the +1pp bar."
            )
    if union:
        union_r50 = union["summary"]["mean_r@50"]
        if arith:
            best_arith_r50 = max(p["summary"]["mean_r@50"] for p in arith.values())
            if union_r50 - best_arith_r50 >= 0.005:
                lines.append(
                    f"Score-merge (probe_union) outperforms best arithmetic by "
                    f"{union_r50 - best_arith_r50:+.4f} — multi-probe > arithmetic."
                )
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- JSON: `{OUT_JSON.relative_to(ASSOC_DIR)}`")
    lines.append("- Source: `keyword_add_steering.py`")
    lines.append(
        f"- Caches: `{KEYADD_LLM_CACHE.relative_to(ASSOC_DIR)}`, "
        f"`{KEYADD_EMB_CACHE.relative_to(ASSOC_DIR)}`"
    )
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"Saved: {OUT_MD}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
