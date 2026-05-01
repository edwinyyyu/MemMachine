"""Evaluate deeper cue-gen tuning variants for EventMemory on LoCoMo-30.

Extends the retuning sweep that produced em_v2f_speakerformat with:
  - length/structure variants (short / 5-cues / natural-turn)
  - a properly rewritten type_enumerated (TYPE_ENUM_EM_RETUNED)
  - a chain-scratchpad specialist prompt
  - two composition variants:
      alias_expand_speakerformat  = per-variant v2f with speakerformat cues
      two_speaker_filter_sf_cues  = two_speaker_filter with speakerformat cues

Plus three baselines for anchoring:
  em_v2f_speakerformat      (existing retuned cue gen)
  em_two_speaker_filter     (v2f cues + property_filter; current K=20 leader)
  em_two_speaker_query_only (no cues, filter only; current K=50 leader)

All retrieval runs at max_K=50 then sliced to K=20 and K=50 (fair backfill).
Each variant writes to a DEDICATED cache under `cache/emtune_*_cache.json`
so no prior specialist caches are poisoned.

No framework files, no em_setup / em_architectures / em_retuned_cue_gen
source modifications -- only imports.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import openai
from alias_expansion import (
    build_expanded_queries,
    find_alias_matches,
)
from dotenv import load_dotenv
from em_architectures import (
    V2F_MODEL,
    EMHit,
    _dedupe_by_turn_id,
    _merge_by_max_score,
    _MergedLLMCache,
    _query_em,
    format_primer_context,
)
from em_deeper_tune import (
    build_chain_scratchpad_speakerformat_prompt,
    build_speakerformat_5cues_prompt,
    build_speakerformat_natural_turn_prompt,
    build_speakerformat_short_prompt,
    build_type_enum_em_retuned_prompt,
)
from em_deeper_tune import (
    parse_cues as parse_deeper_cues,
)
from em_retuned_cue_gen import (
    build_v2f_speakerformat_prompt,
)
from em_retuned_cue_gen import (
    parse_cues as parse_retuned_cues,
)
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.filter.filter_parser import Comparison
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
from speaker_attributed import extract_name_mentions
from sqlalchemy.ext.asyncio import create_async_engine

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
BUDGETS = (20, 50)
MAX_K = max(BUDGETS)
LOCOMO_CONV_IDS = ("locomo_conv-26", "locomo_conv-30", "locomo_conv-41")


# --------------------------------------------------------------------------
# Variant registry
# --------------------------------------------------------------------------

# Each cue-gen variant is (builder_fn, n_cues, cache_file_name).
CUE_VARIANTS: dict[str, tuple] = {
    # Baseline re-run of the current retuned default.
    "em_v2f_speakerformat": (
        build_v2f_speakerformat_prompt,
        2,
        "emtune_v2f_speakerformat_cache.json",
    ),
    "v2f_speakerformat_short": (
        build_speakerformat_short_prompt,
        2,
        "emtune_sf_short_cache.json",
    ),
    "v2f_speakerformat_5cues": (
        build_speakerformat_5cues_prompt,
        5,
        "emtune_sf_5cues_cache.json",
    ),
    "v2f_speakerformat_natural_turn": (
        build_speakerformat_natural_turn_prompt,
        2,
        "emtune_sf_natural_turn_cache.json",
    ),
    "chain_with_scratchpad_speakerformat": (
        build_chain_scratchpad_speakerformat_prompt,
        2,
        "emtune_chain_scratchpad_cache.json",
    ),
    "type_enumerated_em_retuned": (
        build_type_enum_em_retuned_prompt,
        5,
        "emtune_type_enum_em_retuned_cache.json",
    ),
}

ALIAS_SPEAKERFORMAT_CACHE = "emtune_alias_speakerformat_cache.json"
TWO_SPEAKER_SF_CUES_CACHE = "emtune_two_speaker_sf_cues_cache.json"

# Composition variants reuse the speakerformat cue cache (same prompt applied
# per-variant / per-question), plus their own per-variant records.
COMPOSITION_VARIANTS = [
    "alias_expand_speakerformat",
    "two_speaker_filter_sf_cues",
]

# Baseline (no cue gen) variants computed directly.
BASELINE_NO_CUE_VARIANTS = [
    "em_two_speaker_query_only",
]

# Baseline using existing v2f speakerformat cues + property filter (different
# composition -- runs the same cue prompt as em_v2f_speakerformat, wrapped in
# the two_speaker_filter pipeline). Not "sf_cues" -- this is the prior
# em_two_speaker_filter (which uses the *old* V2F_PROMPT) for reference.
# We actually *replace* it with two_speaker_filter_sf_cues below, which
# should be >= two_speaker_filter.

ALL_VARIANTS = (
    list(CUE_VARIANTS.keys()) + COMPOSITION_VARIANTS + BASELINE_NO_CUE_VARIANTS
)

BASELINE_VARIANT = "em_v2f_speakerformat"


# --------------------------------------------------------------------------
# Data loading / helpers
# --------------------------------------------------------------------------


def load_questions() -> list[dict]:
    with open(DATA_DIR / "questions_extended.json") as f:
        qs = json.load(f)
    return [q for q in qs if q.get("benchmark") == "locomo"][:30]


def load_collections_meta() -> dict:
    with open(RESULTS_DIR / "eventmemory_collections.json") as f:
        return json.load(f)


def load_speaker_map() -> dict[str, dict[str, str]]:
    with open(RESULTS_DIR / "conversation_two_speakers.json") as f:
        return json.load(f).get("speakers", {}) or {}


def load_alias_groups() -> dict[str, list[list[str]]]:
    p = RESULTS_DIR / "conversation_alias_groups.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f).get("groups", {}) or {}


def compute_recall(retrieved: set[int], gold: set[int]) -> float:
    if not gold:
        return 1.0
    return len(retrieved & gold) / len(gold)


def classify_speaker_side(
    question: str,
    conversation_id: str,
    speaker_map: dict[str, dict[str, str]],
) -> tuple[str, str, str, list[str]]:
    pair = speaker_map.get(conversation_id, {})
    user_name = (pair.get("user") or "UNKNOWN").strip() or "UNKNOWN"
    asst_name = (pair.get("assistant") or "UNKNOWN").strip() or "UNKNOWN"
    tokens = extract_name_mentions(question)
    lowered = {t.lower() for t in tokens}
    hit_user = user_name != "UNKNOWN" and user_name.lower() in lowered
    hit_asst = asst_name != "UNKNOWN" and asst_name.lower() in lowered
    if hit_user and hit_asst:
        side = "both"
    elif hit_user:
        side = "user"
    elif hit_asst:
        side = "assistant"
    else:
        side = "none"
    return side, user_name, asst_name, tokens


# --------------------------------------------------------------------------
# Cue generation (cached, shared across variants by prompt hash)
# --------------------------------------------------------------------------


async def _llm_call(
    openai_client,
    prompt: str,
    cache: _MergedLLMCache,
) -> tuple[str, bool]:
    cached = cache.get(V2F_MODEL, prompt)
    if cached is not None:
        return cached, True
    resp = await openai_client.chat.completions.create(
        model=V2F_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content or ""
    cache.put(V2F_MODEL, prompt, text)
    return text, False


async def build_primer_context(memory: EventMemory, question: str) -> str:
    primer_hits = _dedupe_by_turn_id(
        await _query_em(memory, question, vector_search_limit=10, expand_context=0)
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text} for h in primer_hits
    ]
    return format_primer_context(primer_segments)


async def generate_cues_for(
    variant: str,
    question: str,
    context_section: str,
    user_name: str,
    asst_name: str,
    cache: _MergedLLMCache,
    openai_client,
) -> tuple[list[str], str, bool]:
    """Returns (cues, raw_response, cache_hit)."""
    builder, n_cues, _ = CUE_VARIANTS[variant]
    prompt = builder(question, context_section, user_name, asst_name)
    raw, hit = await _llm_call(openai_client, prompt, cache)
    # All variants use a single parser (same CUE_RE semantics).
    if variant == "em_v2f_speakerformat":
        cues = parse_retuned_cues(raw, max_cues=n_cues)
    else:
        cues = parse_deeper_cues(raw, max_cues=n_cues)
    return cues, raw, hit


# --------------------------------------------------------------------------
# Retrieval helpers
# --------------------------------------------------------------------------


async def _primer_then_cues_merge_max(
    memory: EventMemory,
    question: str,
    cues: list[str],
) -> list[EMHit]:
    primer_for_merge = await _query_em(
        memory, question, vector_search_limit=MAX_K, expand_context=0
    )
    cue_hits = []
    for cue in cues:
        cue_hits.append(
            await _query_em(memory, cue, vector_search_limit=MAX_K, expand_context=0)
        )
    return _merge_by_max_score([primer_for_merge, *cue_hits])


async def _speaker_filtered_query(
    memory: EventMemory,
    question: str,
    matched_name: str,
    K: int,
) -> list[EMHit]:
    prop_filter = Comparison(field="context.source", op="=", value=matched_name)
    raw = await memory.query(
        query=question,
        vector_search_limit=K,
        expand_context=0,
        property_filter=prop_filter,
    )
    hits: list[EMHit] = []
    for sc in raw.scored_segment_contexts:
        for seg in sc.segments:
            hits.append(
                EMHit(
                    turn_id=int(seg.properties.get("turn_id", -1)),
                    score=sc.score,
                    seed_segment_uuid=sc.seed_segment_uuid,
                    role=str(seg.properties.get("role", "")),
                    text=seg.block.text,
                )
            )
    return _dedupe_by_turn_id(hits)


# --------------------------------------------------------------------------
# Per-question evaluation dispatcher
# --------------------------------------------------------------------------


async def evaluate_question(
    variant: str,
    memory: EventMemory,
    question: dict,
    participants: tuple[str, str],
    caches: dict[str, _MergedLLMCache],
    openai_client,
    speaker_map: dict[str, dict[str, str]],
    alias_groups: dict[str, list[list[str]]],
) -> dict:
    q_text = question["question"]
    gold = set(question.get("source_chat_ids", []))
    user_name, asst_name = participants
    cid = question["conversation_id"]

    t0 = time.monotonic()

    row_extras: dict = {}
    cues_for_report: list[str] = []

    # ---- 1) Pure cue-gen variants ----
    if variant in CUE_VARIANTS:
        context_section = await build_primer_context(memory, q_text)
        cues, _raw, cache_hit = await generate_cues_for(
            variant,
            q_text,
            context_section,
            user_name,
            asst_name,
            caches[variant],
            openai_client,
        )
        cues_for_report = cues
        row_extras["n_cues"] = len(cues)
        row_extras["cache_hit"] = cache_hit
        hits = await _primer_then_cues_merge_max(memory, q_text, cues)
        hits = hits[:MAX_K]

    # ---- 2) Composition: alias_expand + speakerformat cues ----
    elif variant == "alias_expand_speakerformat":
        groups = alias_groups.get(cid, [])
        matches = find_alias_matches(q_text, groups) if groups else []
        cache = caches["alias_expand_speakerformat"]

        if not matches:
            # Fallback: single pass speakerformat (equiv. em_v2f_speakerformat
            # but with alias cache so no cross-poisoning).
            context_section = await build_primer_context(memory, q_text)
            prompt = build_v2f_speakerformat_prompt(
                q_text, context_section, user_name, asst_name
            )
            raw, cache_hit = await _llm_call(openai_client, prompt, cache)
            cues = parse_retuned_cues(raw, max_cues=2)
            cues_for_report = cues
            row_extras["alias_variants"] = 1
            row_extras["cache_hit"] = cache_hit
            hits = await _primer_then_cues_merge_max(memory, q_text, cues)
            hits = hits[:MAX_K]
        else:
            variants_list, match_records = build_expanded_queries(
                q_text, groups, max_siblings_per_match=4
            )
            batches: list[list[EMHit]] = []
            all_cues: list[str] = []
            cache_hits = 0
            cache_total = 0

            for v in variants_list:
                # primer retrieval per variant
                batches.append(
                    await _query_em(memory, v, vector_search_limit=10, expand_context=0)
                )
                # speakerformat cues per variant
                context_section_v = await build_primer_context(memory, v)
                prompt = build_v2f_speakerformat_prompt(
                    v, context_section_v, user_name, asst_name
                )
                raw, hit = await _llm_call(openai_client, prompt, cache)
                cache_total += 1
                cache_hits += 1 if hit else 0
                cues_v = parse_retuned_cues(raw, max_cues=2)
                all_cues.extend(cues_v)
                for c in cues_v:
                    batches.append(
                        await _query_em(
                            memory, c, vector_search_limit=10, expand_context=0
                        )
                    )

            # sibling-only probes
            sibling_probes: list[str] = []
            for rec in match_records:
                for sib in rec.get("siblings", [])[:4]:
                    if sib not in sibling_probes and sib != q_text:
                        sibling_probes.append(sib)
            for s in sibling_probes[:8]:
                batches.append(
                    await _query_em(memory, s, vector_search_limit=10, expand_context=0)
                )
            # full-K fallback on original
            batches.append(
                await _query_em(
                    memory, q_text, vector_search_limit=MAX_K, expand_context=0
                )
            )

            # sum-of-cosines per turn_id, one contribution per batch
            score_sum: dict[int, float] = {}
            representative: dict[int, EMHit] = {}
            for batch in batches:
                seen_in_batch: set[int] = set()
                for h in batch:
                    if h.turn_id in seen_in_batch:
                        continue
                    seen_in_batch.add(h.turn_id)
                    score_sum[h.turn_id] = score_sum.get(h.turn_id, 0.0) + h.score
                    if h.turn_id not in representative:
                        representative[h.turn_id] = h
            ranked = sorted(
                [
                    EMHit(
                        turn_id=tid,
                        score=score_sum[tid],
                        seed_segment_uuid=representative[tid].seed_segment_uuid,
                        role=representative[tid].role,
                        text=representative[tid].text,
                    )
                    for tid in score_sum
                ],
                key=lambda h: -h.score,
            )
            hits = ranked[:MAX_K]
            cues_for_report = all_cues[:2]  # first variant's cues, for display
            row_extras.update(
                {
                    "alias_variants": len(variants_list),
                    "alias_matches": [
                        {"matched": r["matched_in_query"], "siblings": r["siblings"]}
                        for r in match_records
                    ],
                    "n_sibling_probes": len(sibling_probes[:8]),
                    "cache_hit_rate": (cache_hits / cache_total)
                    if cache_total
                    else 1.0,
                }
            )

    # ---- 3) Composition: two_speaker_filter with speakerformat cues ----
    elif variant == "two_speaker_filter_sf_cues":
        cache = caches["two_speaker_filter_sf_cues"]
        # Generate speakerformat cues exactly like em_v2f_speakerformat.
        context_section = await build_primer_context(memory, q_text)
        prompt = build_v2f_speakerformat_prompt(
            q_text, context_section, user_name, asst_name
        )
        raw, cache_hit = await _llm_call(openai_client, prompt, cache)
        cues = parse_retuned_cues(raw, max_cues=2)
        cues_for_report = cues

        # Merge v2f hits (primer + cues, max_score).
        v2f_hits = await _primer_then_cues_merge_max(memory, q_text, cues)

        side, user_n, asst_n, name_tokens = classify_speaker_side(
            q_text, cid, speaker_map
        )
        row_extras.update(
            {
                "matched_side": side,
                "query_name_tokens": name_tokens,
                "cache_hit": cache_hit,
                "applied_speaker_filter": False,
            }
        )

        if side not in ("user", "assistant"):
            hits = v2f_hits[:MAX_K]
        else:
            matched_name = user_n if side == "user" else asst_n
            row_extras["applied_speaker_filter"] = True
            row_extras["matched_name"] = matched_name
            # Speaker-filtered query hits.
            speaker_hits = await _speaker_filtered_query(
                memory, q_text, matched_name, K=MAX_K + 10
            )
            # Filter v2f to matched role, then append speaker_hits.
            matched_role = side
            kept_v2f = [h for h in v2f_hits if h.role == matched_role]
            seen = {h.turn_id for h in kept_v2f}
            appended: list[EMHit] = []
            for h in speaker_hits:
                if h.turn_id in seen:
                    continue
                appended.append(h)
                seen.add(h.turn_id)
                if len(appended) >= 15:  # role_only_top_m=5 + 10
                    break
            row_extras["dropped_v2f"] = len(v2f_hits) - len(kept_v2f)
            row_extras["appended_speaker_hits"] = len(appended)
            hits = (kept_v2f + appended)[:MAX_K]

    # ---- 4) two_speaker_query_only baseline (no cues) ----
    elif variant == "em_two_speaker_query_only":
        side, user_n, asst_n, name_tokens = classify_speaker_side(
            q_text, cid, speaker_map
        )
        row_extras.update(
            {
                "matched_side": side,
                "query_name_tokens": name_tokens,
                "applied_speaker_filter": False,
            }
        )
        if side in ("user", "assistant"):
            matched_name = user_n if side == "user" else asst_n
            row_extras["applied_speaker_filter"] = True
            row_extras["matched_name"] = matched_name
            cosine_hits = _dedupe_by_turn_id(
                await _query_em(
                    memory, q_text, vector_search_limit=MAX_K, expand_context=0
                )
            )
            speaker_hits = await _speaker_filtered_query(
                memory, q_text, matched_name, K=MAX_K
            )
            matched_role = side
            kept = [h for h in cosine_hits if h.role == matched_role]
            seen = {h.turn_id for h in kept}
            for h in speaker_hits:
                if h.turn_id in seen:
                    continue
                kept.append(h)
                seen.add(h.turn_id)
            hits = kept[:MAX_K]
        else:
            hits = _dedupe_by_turn_id(
                await _query_em(
                    memory, q_text, vector_search_limit=MAX_K, expand_context=0
                )
            )[:MAX_K]
    else:
        raise KeyError(variant)

    elapsed = time.monotonic() - t0

    row: dict = {
        "conversation_id": cid,
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "gold_turn_ids": sorted(gold),
        "cues": cues_for_report,
        "time_s": round(elapsed, 3),
    }
    row.update(row_extras)

    for K in BUDGETS:
        topk = hits[:K]
        retrieved = {h.turn_id for h in topk}
        row[f"r@{K}"] = round(compute_recall(retrieved, gold), 4)
        row[f"retrieved_turn_ids@{K}"] = sorted(retrieved)
    return row


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


async def main() -> None:
    collections_meta = load_collections_meta()
    questions = load_questions()
    conv_to_meta = {r["conversation_id"]: r for r in collections_meta["conversations"]}
    speaker_map = load_speaker_map()
    alias_groups = load_alias_groups()

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    sql_url = collections_meta.get("sql_url") or os.getenv("SQL_URL")
    if sql_url is None:
        raise RuntimeError("No SQL_URL in collections meta or env")
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

    # Per-variant dedicated caches. Composition variants have their own caches
    # even though they call the same speakerformat prompt -- so that re-running
    # a variant does not pollute another variant's cache semantics.
    caches: dict[str, _MergedLLMCache] = {}
    for variant, (_b, _n, cache_name) in CUE_VARIANTS.items():
        # em_v2f_speakerformat can also READ from the existing emretune cache
        # to warm-start with zero additional LLM cost.
        extra_readers: list[Path] = []
        if variant == "em_v2f_speakerformat":
            legacy = CACHE_DIR / "emretune_v2f_speakerformat_cache.json"
            if legacy.exists():
                extra_readers.append(legacy)
        writer = CACHE_DIR / cache_name
        caches[variant] = _MergedLLMCache(
            reader_paths=[writer, *extra_readers],
            writer_path=writer,
        )

    # Composition caches also read from em_v2f_speakerformat's cache since
    # they use the same speakerformat prompt text (same sha) for cue gen.
    sf_legacy = CACHE_DIR / "emretune_v2f_speakerformat_cache.json"
    sf_new = CACHE_DIR / "emtune_v2f_speakerformat_cache.json"
    for variant, cache_name in [
        ("alias_expand_speakerformat", ALIAS_SPEAKERFORMAT_CACHE),
        ("two_speaker_filter_sf_cues", TWO_SPEAKER_SF_CUES_CACHE),
    ]:
        writer = CACHE_DIR / cache_name
        readers = [writer]
        if sf_legacy.exists():
            readers.append(sf_legacy)
        if sf_new.exists() and sf_new != writer:
            readers.append(sf_new)
        caches[variant] = _MergedLLMCache(reader_paths=readers, writer_path=writer)

    # Open EM per conversation.
    memories: dict[str, EventMemory] = {}
    participants_by_conv: dict[str, tuple[str, str]] = {}
    opened_resources: list = []
    for conv_id in LOCOMO_CONV_IDS:
        meta = conv_to_meta[conv_id]
        coll = await vector_store.open_collection(
            namespace=meta["namespace"], name=meta["collection_name"]
        )
        part = await segment_store.open_or_create_partition(meta["partition_key"])
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
        memories[conv_id] = mem
        participants_by_conv[conv_id] = (meta["user_name"], meta["assistant_name"])
        opened_resources.append((coll, part))

    results: dict = {
        "variants": {},
        "budgets": list(BUDGETS),
        "questions": len(questions),
        "anchors": {
            "em_v2f_speakerformat_prior": {"mean_r@20": 0.817, "mean_r@50": 0.892},
            "em_two_speaker_filter_prior": {"mean_r@20": 0.842, "mean_r@50": 0.900},
            "em_two_speaker_query_only_prior": {"mean_r@20": 0.800, "mean_r@50": 0.933},
            "em_alias_expand_v2f_prior": {"mean_r@20": 0.825, "mean_r@50": 0.883},
        },
    }

    for variant in ALL_VARIANTS:
        rows: list[dict] = []
        t_variant = time.monotonic()
        for q in questions:
            cid = q["conversation_id"]
            mem = memories[cid]
            participants = participants_by_conv[cid]
            row = await evaluate_question(
                variant,
                mem,
                q,
                participants,
                caches,
                openai_client,
                speaker_map,
                alias_groups,
            )
            rows.append(row)
        elapsed = time.monotonic() - t_variant

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
        for cat, cat_rows in by_cat.items():
            d = {"n": len(cat_rows)}
            for K in BUDGETS:
                d[f"mean_r@{K}"] = round(
                    sum(r[f"r@{K}"] for r in cat_rows) / max(len(cat_rows), 1), 4
                )
            cat_summary[cat] = d

        results["variants"][variant] = {
            "summary": summary,
            "by_category": cat_summary,
            "per_question": rows,
        }
        if variant in caches:
            caches[variant].save()

        print(
            f"[{variant}] n={summary['n']} "
            f"r@20={summary['mean_r@20']:.4f} r@50={summary['mean_r@50']:.4f} "
            f"in {summary['time_s']:.1f}s"
        )

    # W/T/L vs em_v2f_speakerformat baseline.
    baseline_rows = results["variants"][BASELINE_VARIANT]["per_question"]
    baseline_idx = {
        (r["conversation_id"], r["question_index"]): r for r in baseline_rows
    }
    for variant in ALL_VARIANTS:
        if variant == BASELINE_VARIANT:
            continue
        v_rows = results["variants"][variant]["per_question"]
        wtl = {}
        for K in BUDGETS:
            w = t = l = 0
            for r in v_rows:
                key = (r["conversation_id"], r["question_index"])
                br = baseline_idx[key]
                vr = r[f"r@{K}"]
                bv = br[f"r@{K}"]
                if vr > bv:
                    w += 1
                elif vr < bv:
                    l += 1
                else:
                    t += 1
            wtl[f"r@{K}"] = {"W": w, "T": t, "L": l}
        results["variants"][variant]["wtl_vs_baseline"] = wtl

    # Close.
    for coll, part in opened_resources:
        await segment_store.close_partition(part)
        await vector_store.close_collection(collection=coll)
    await segment_store.shutdown()
    await vector_store.shutdown()
    await engine.dispose()
    await qdrant_client.close()
    await openai_client.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "em_deeper_tune.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")

    md_lines = build_markdown_report(results, questions)
    out_md = RESULTS_DIR / "em_deeper_tune.md"
    out_md.write_text("\n".join(md_lines))
    print(f"Saved: {out_md}")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def build_markdown_report(results: dict, questions: list[dict]) -> list[str]:
    base = results["variants"][BASELINE_VARIANT]["summary"]
    anchors = results["anchors"]
    lines = [
        "# EventMemory Deeper Cue-Gen Tuning (LoCoMo-30)",
        "",
        "## Setup",
        "",
        f"- n_questions = {len(questions)} (benchmark=locomo, first 30)",
        "- EventMemory backend, `text-embedding-3-small`, `gpt-5-mini`",
        '- Speaker-baked embedded format: `"{source}: {text}"`',
        "- Retrieval at max_K=50, sliced to K=20 / K=50 (fair backfill)",
        "- Caches: `cache/emtune_<variant>_cache.json` (dedicated)",
        "",
        "## Recall table",
        "",
        "| Variant | R@20 | R@50 | d vs SF R@20 | d vs SF R@50 | time (s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    # Order variants: cue-gen first, then composition, then no-cue baselines.
    ordered = (
        list(CUE_VARIANTS.keys()) + COMPOSITION_VARIANTS + BASELINE_NO_CUE_VARIANTS
    )
    for variant in ordered:
        s = results["variants"][variant]["summary"]
        d20 = s["mean_r@20"] - base["mean_r@20"]
        d50 = s["mean_r@50"] - base["mean_r@50"]
        lines.append(
            f"| `{variant}` | {s['mean_r@20']:.4f} | {s['mean_r@50']:.4f} | "
            f"{d20:+.4f} | {d50:+.4f} | {s['time_s']:.1f} |"
        )
    lines += [
        "",
        "Anchors from prior runs:",
        f"- em_v2f_speakerformat prior: {anchors['em_v2f_speakerformat_prior']}",
        f"- em_two_speaker_filter (v2f cues) prior: "
        f"{anchors['em_two_speaker_filter_prior']}",
        f"- em_two_speaker_query_only prior: "
        f"{anchors['em_two_speaker_query_only_prior']}",
        f"- em_alias_expand_v2f prior: {anchors['em_alias_expand_v2f_prior']}",
        "",
        "## W/T/L per variant vs em_v2f_speakerformat",
        "",
        "| Variant | K=20 W/T/L | K=50 W/T/L |",
        "| --- | --- | --- |",
    ]
    for variant in ordered:
        if variant == BASELINE_VARIANT:
            continue
        w = results["variants"][variant].get("wtl_vs_baseline", {})
        w20 = w.get("r@20", {"W": 0, "T": 0, "L": 0})
        w50 = w.get("r@50", {"W": 0, "T": 0, "L": 0})
        lines.append(
            f"| `{variant}` | "
            f"{w20['W']}/{w20['T']}/{w20['L']} | "
            f"{w50['W']}/{w50['T']}/{w50['L']} |"
        )

    # Per-category R@50 for the baseline and top 2 variants by R@20 lift.
    lines += [
        "",
        "## Top-2 cue variants by R@20 lift over SF baseline",
        "",
    ]
    cue_only = list(CUE_VARIANTS.keys())
    cue_only_ranked = sorted(
        [
            (v, results["variants"][v]["summary"]["mean_r@20"] - base["mean_r@20"])
            for v in cue_only
            if v != BASELINE_VARIANT
        ],
        key=lambda t: -t[1],
    )
    for variant, delta in cue_only_ranked[:2]:
        s = results["variants"][variant]["summary"]
        lines.append(
            f"- `{variant}` : R@20={s['mean_r@20']:.4f} "
            f"(d={delta:+.4f}), R@50={s['mean_r@50']:.4f}"
        )
    lines.append("")

    # Sample cues (3 questions: baseline vs best cue variant).
    best_variant = cue_only_ranked[0][0] if cue_only_ranked else BASELINE_VARIANT
    lines += [
        f"## Sample cues: `em_v2f_speakerformat` (old) vs `{best_variant}` (best new)",
        "",
    ]
    base_rows = results["variants"][BASELINE_VARIANT]["per_question"]
    best_rows = results["variants"][best_variant]["per_question"]
    n = len(base_rows)
    sample_idxs = [0, n // 2, n - 1] if n >= 3 else list(range(n))
    for idx in sample_idxs:
        q_old = base_rows[idx]
        q_new = best_rows[idx]
        lines.append(
            f"### Q{idx}: `{q_old['conversation_id']}` -- {q_old['question']!r}"
        )
        lines.append(
            f"- OLD `{BASELINE_VARIANT}` "
            f"(R@20={q_old['r@20']:.2f}, R@50={q_old['r@50']:.2f}):"
        )
        for c in q_old["cues"]:
            lines.append(f"  - `{c}`")
        lines.append(
            f"- NEW `{best_variant}` "
            f"(R@20={q_new['r@20']:.2f}, R@50={q_new['r@50']:.2f}):"
        )
        for c in q_new["cues"]:
            lines.append(f"  - `{c}`")
        lines.append("")

    # Verdict per variant: ship / narrow / abandon.
    lines += [
        "## Verdict per variant (vs em_v2f_speakerformat)",
        "",
        "| Variant | R@20 d | R@50 d | Decision |",
        "| --- | --- | --- | --- |",
    ]
    ship_both: list[str] = []
    ship_k20_only: list[str] = []
    ship_k50_only: list[str] = []
    for variant in ordered:
        if variant == BASELINE_VARIANT:
            lines.append(f"| `{variant}` | +0.0000 | +0.0000 | baseline |")
            continue
        s = results["variants"][variant]["summary"]
        d20 = s["mean_r@20"] - base["mean_r@20"]
        d50 = s["mean_r@50"] - base["mean_r@50"]
        if d20 >= 0.01 and d50 >= 0.01:
            decision = "SHIP (both K)"
            ship_both.append(variant)
        elif d20 >= 0.01:
            decision = "narrow: K=20 only"
            ship_k20_only.append(variant)
        elif d50 >= 0.01:
            decision = "narrow: K=50 only"
            ship_k50_only.append(variant)
        else:
            decision = "abandon"
        lines.append(f"| `{variant}` | {d20:+.4f} | {d50:+.4f} | {decision} |")

    # Check composition vs their own references.
    ts_filter = (
        results["variants"].get("two_speaker_filter_sf_cues", {}).get("summary", {})
    )
    ts_qonly = (
        results["variants"].get("em_two_speaker_query_only", {}).get("summary", {})
    )

    lines += [
        "",
        "## Composition / regime ceilings",
        "",
    ]
    if ts_filter:
        d = ts_filter["mean_r@20"] - 0.842
        lines.append(
            f"- `two_speaker_filter_sf_cues` K=20 = {ts_filter['mean_r@20']:.4f} "
            f"(prior two_speaker_filter K=20 ceiling 0.842, d={d:+.4f})"
        )
    if ts_qonly:
        d = ts_qonly["mean_r@50"] - 0.933
        lines.append(
            f"- `em_two_speaker_query_only` K=50 = {ts_qonly['mean_r@50']:.4f} "
            f"(prior K=50 ceiling 0.933, d={d:+.4f})"
        )

    # Updated production recipe per K regime.
    lines += [
        "",
        "## Updated production recipe (per K regime)",
        "",
    ]
    # K=20 winner
    k20_candidates = []
    for v in ordered:
        s = results["variants"][v]["summary"]
        k20_candidates.append((v, s["mean_r@20"]))
    k20_candidates.sort(key=lambda t: -t[1])
    k50_candidates = []
    for v in ordered:
        s = results["variants"][v]["summary"]
        k50_candidates.append((v, s["mean_r@50"]))
    k50_candidates.sort(key=lambda t: -t[1])

    lines.append(
        f"- K=20 winner: `{k20_candidates[0][0]}` @ {k20_candidates[0][1]:.4f}"
    )
    lines.append(
        f"- K=50 winner: `{k50_candidates[0][0]}` @ {k50_candidates[0][1]:.4f}"
    )

    # Ship narrative.
    if ship_both:
        lines.append("")
        lines.append(
            f"**Ship cue-gen default**: `{ship_both[0]}` beats SF at BOTH "
            "K budgets by >=1pp."
        )
    elif ship_k20_only or ship_k50_only:
        lines.append("")
        lines.append("**Narrow ship per regime**:")
        for v in ship_k20_only:
            s = results["variants"][v]["summary"]
            lines.append(f"- K=20: `{v}` ({s['mean_r@20']:.4f})")
        for v in ship_k50_only:
            s = results["variants"][v]["summary"]
            lines.append(f"- K=50: `{v}` ({s['mean_r@50']:.4f})")
    else:
        lines.append("")
        lines.append(
            "**No cue-gen change.** No variant beats `em_v2f_speakerformat` "
            "by >=1pp at either K. Keep SF as EM cue default."
        )

    # Outputs.
    lines += [
        "",
        "## Outputs",
        "",
        "- `results/em_deeper_tune.json`",
        "- `results/em_deeper_tune.md`",
        "- Source: `em_deeper_tune.py`, `em_deeper_eval.py`",
        "- Caches: `cache/emtune_<variant>_cache.json`",
        "",
    ]
    return lines


if __name__ == "__main__":
    asyncio.run(main())
