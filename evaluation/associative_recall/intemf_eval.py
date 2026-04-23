"""Evaluate intent_em variants on LoCoMo-30 against em_two_speaker baselines.

Variants:
  intent_em_speaker_only
  intent_em_full_filter
  intent_em_filter_no_cues
  intent_em_with_speakerformat_cues

Reuses ingested EventMemory state (arc_em_lc30_v1_{26,30,41}) and the
shared intent_parse_cache. Does NOT re-ingest. Does NOT modify framework
or prior em_*.py files.

Outputs:
  results/intent_em.json       raw per-question + summary
  results/intent_em.md         schema findings, recall matrix,
                               constraint-firing analysis, verdict

Caches (dedicated, intemf_ prefix):
  cache/intemf_v2f_llm_cache.json            v2f (raw) cue responses
  cache/intemf_speakerformat_llm_cache.json  v2f_speakerformat cue responses
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict
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

from em_architectures import (
    BESTSHOT_LLM_CACHE,
    EM_V2F_LLM_CACHE,
    EMHit,
    _MergedLLMCache,
)
from intent_em import (
    IntentEMResult,
    intent_em_filter_no_cues,
    intent_em_full_filter,
    intent_em_speaker_only,
    intent_em_with_speakerformat_cues,
    load_intent_parse_cache,
    load_two_speaker_map,
)


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
BUDGETS = (20, 50)
LOCOMO_CONV_IDS = ("locomo_conv-26", "locomo_conv-30", "locomo_conv-41")

INTEMF_V2F_CACHE = CACHE_DIR / "intemf_v2f_llm_cache.json"
INTEMF_SPEAKERFMT_CACHE = CACHE_DIR / "intemf_speakerformat_llm_cache.json"

VARIANTS = (
    "intent_em_speaker_only",
    "intent_em_full_filter",
    "intent_em_filter_no_cues",
    "intent_em_with_speakerformat_cues",
)


# Reference numbers from prior runs (for the MD report).
REFERENCE = {
    "em_v2f_speakerformat": (0.817, 0.892),
    "em_two_speaker_filter": (0.842, 0.900),
    "em_two_speaker_query_only": (0.800, 0.933),
}


def load_locomo_questions() -> list[dict]:
    with open(DATA_DIR / "questions_extended.json") as f:
        qs = json.load(f)
    locomo = [q for q in qs if q.get("benchmark") == "locomo"]
    return locomo[:30]


def load_collections_meta() -> dict:
    with open(RESULTS_DIR / "eventmemory_collections.json") as f:
        return json.load(f)


def compute_recall(retrieved: set[int], gold: set[int]) -> float:
    if not gold:
        return 1.0
    return len(retrieved & gold) / len(gold)


async def evaluate_question(
    variant: str,
    memory: EventMemory,
    question: dict,
    *,
    plan: dict,
    speaker_map: dict[str, dict[str, str]],
    participants: tuple[str, str],
    v2f_cache: _MergedLLMCache,
    speakerfmt_cache: _MergedLLMCache,
    openai_client,
    max_K: int,
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    gold = set(question.get("source_chat_ids", []))

    t0 = time.monotonic()

    if variant == "intent_em_speaker_only":
        result: IntentEMResult = await intent_em_speaker_only(
            memory, q_text, conv_id,
            K=max_K, plan=plan, speaker_map=speaker_map,
            llm_cache=v2f_cache, openai_client=openai_client,
        )
    elif variant == "intent_em_full_filter":
        result = await intent_em_full_filter(
            memory, q_text, conv_id,
            K=max_K, plan=plan, speaker_map=speaker_map,
            llm_cache=v2f_cache, openai_client=openai_client,
        )
    elif variant == "intent_em_filter_no_cues":
        result = await intent_em_filter_no_cues(
            memory, q_text, conv_id,
            K=max_K, plan=plan, speaker_map=speaker_map,
        )
    elif variant == "intent_em_with_speakerformat_cues":
        result = await intent_em_with_speakerformat_cues(
            memory, q_text, conv_id,
            K=max_K, plan=plan, speaker_map=speaker_map,
            participants=participants,
            llm_cache=speakerfmt_cache, openai_client=openai_client,
        )
    else:
        raise KeyError(variant)

    elapsed = time.monotonic() - t0
    hits = result.hits
    meta = result.metadata
    analysis = meta.get("filter_analysis", {})

    row: dict = {
        "conversation_id": conv_id,
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "gold_turn_ids": sorted(gold),
        "n_hits": len(hits),
        "time_s": round(elapsed, 3),
        "detected_constraints": analysis.get("detected", []),
        "applied_constraints": analysis.get("applied", []),
        "dropped_constraints": analysis.get("dropped", []),
        "speaker_resolved": analysis.get("speaker_resolved"),
        "matched_side": analysis.get("matched_side"),
        "applied_filter": meta.get("applied_filter", False),
        "cues": meta.get("cues", []),
        "cache_hit": meta.get("cache_hit", False),
    }
    for K in BUDGETS:
        topk = hits[:K]
        retrieved = {h.turn_id for h in topk}
        row[f"r@{K}"] = round(compute_recall(retrieved, gold), 4)
        row[f"retrieved_turn_ids@{K}"] = sorted(retrieved)
    return row


async def run() -> None:
    collections_meta = load_collections_meta()
    questions = load_locomo_questions()
    conv_to_meta = {r["conversation_id"]: r for r in collections_meta["conversations"]}

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

    # Caches: v2f (raw) reads shared bestshot/em_v2f caches; speakerformat
    # reads the emretune-retuned cache. Both write to our dedicated files.
    v2f_cache = _MergedLLMCache(
        reader_paths=[
            BESTSHOT_LLM_CACHE,
            EM_V2F_LLM_CACHE,
            CACHE_DIR / "meta_llm_cache.json",
            CACHE_DIR / "arch_llm_cache.json",
            CACHE_DIR / "emdef_v2f_llm_cache.json",
            INTEMF_V2F_CACHE,
        ],
        writer_path=INTEMF_V2F_CACHE,
    )
    speakerfmt_cache = _MergedLLMCache(
        reader_paths=[
            CACHE_DIR / "emretune_v2f_speakerformat_cache.json",
            INTEMF_SPEAKERFMT_CACHE,
        ],
        writer_path=INTEMF_SPEAKERFMT_CACHE,
    )

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

    # Intent plans (already fully warmed from intent_parser.py runs).
    parse_cache = load_intent_parse_cache()
    speaker_map = load_two_speaker_map()

    max_K = max(BUDGETS)
    results: dict = {
        "variants": {},
        "budgets": list(BUDGETS),
        "questions": len(questions),
        "schema_notes": {
            "filterable_context_fields": ["context.source", "context.type"],
            "em_base_fields": ["timestamp", "segment_uuid"],
            "user_metadata_stored": ["arc_conversation_id", "turn_id", "role"],
            "note": (
                "Only context.source is semantically useful for intent-driven "
                "filtering. EM timestamps are synthesized (2023-01-01 + 60s "
                "per turn_id) so LoCoMo real-world temporal references "
                "('4 years ago', 'last weekend') do NOT map. Role "
                "duplicates context.source for two-speaker LoCoMo."
            ),
        },
    }

    # Precompute constraint firing rates from the plan cache.
    firing_counts = defaultdict(int)
    unresolved_speakers = 0
    for q in questions:
        plan = parse_cache.get(q["question"].strip(), {})
        cs = plan.get("constraints", {}) or {}
        if cs.get("speaker"):
            firing_counts["speaker"] += 1
        if cs.get("temporal_relation"):
            firing_counts["temporal_relation"] += 1
        if cs.get("negation"):
            firing_counts["negation"] += 1
        if cs.get("answer_form"):
            firing_counts["answer_form"] += 1
        if plan.get("needs_aggregation"):
            firing_counts["needs_aggregation"] += 1
    results["firing_counts"] = dict(firing_counts)

    for variant in VARIANTS:
        rows: list[dict] = []
        t0 = time.monotonic()
        for q in questions:
            cid = q["conversation_id"]
            mem = memories[cid]
            plan = parse_cache.get(q["question"].strip(), {}) or {
                "intent_type": "other", "constraints": {}, "entities": [],
                "primary_topic": None, "needs_aggregation": False,
                "parse_ok": False,
            }
            participants = participants_by_conv[cid]
            row = await evaluate_question(
                variant, mem, q,
                plan=plan,
                speaker_map=speaker_map,
                participants=participants,
                v2f_cache=v2f_cache,
                speakerfmt_cache=speakerfmt_cache,
                openai_client=openai_client,
                max_K=max_K,
            )
            rows.append(row)
        elapsed = time.monotonic() - t0

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

        # Firing-vs-recall analysis: recall by whether filter was applied.
        with_filter = [r for r in rows if r["applied_filter"]]
        without_filter = [r for r in rows if not r["applied_filter"]]
        firing_summary: dict = {
            "n_with_filter": len(with_filter),
            "n_without_filter": len(without_filter),
        }
        for K in BUDGETS:
            if with_filter:
                firing_summary[f"mean_r@{K}_with_filter"] = round(
                    sum(r[f"r@{K}"] for r in with_filter) / len(with_filter), 4
                )
            if without_filter:
                firing_summary[f"mean_r@{K}_without_filter"] = round(
                    sum(r[f"r@{K}"] for r in without_filter) / len(without_filter),
                    4,
                )

        results["variants"][variant] = {
            "summary": summary,
            "by_category": cat_summary,
            "firing_summary": firing_summary,
            "per_question": rows,
        }
        v2f_cache.save()
        speakerfmt_cache.save()

        print(
            f"[{variant}] n={summary['n']} "
            f"r@20={summary['mean_r@20']:.4f} r@50={summary['mean_r@50']:.4f} "
            f"in {summary['time_s']:.1f}s "
            f"(filter fired on {firing_summary['n_with_filter']}/{n})",
            flush=True,
        )

    for coll, part in opened_resources:
        await segment_store.close_partition(part)
        await vector_store.close_collection(collection=coll)
    await segment_store.shutdown()
    await vector_store.shutdown()
    await engine.dispose()
    await qdrant_client.close()
    await openai_client.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "intent_em.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")

    # Markdown report
    md_lines = build_markdown_report(results)
    out_md = RESULTS_DIR / "intent_em.md"
    out_md.write_text("\n".join(md_lines))
    print(f"Saved: {out_md}")


def build_markdown_report(results: dict) -> list[str]:
    schema = results["schema_notes"]
    firing = results.get("firing_counts", {})
    n_q = results["questions"]

    lines = [
        "# intent_parser on EventMemory (LoCoMo-30)",
        "",
        "## Schema available for intent-driven filtering",
        "",
        "- **Filterable Context fields**: `context.source` (speaker name), "
        "`context.type` (always 'message' for LoCoMo — useless).",
        "- **EM reserved base field**: `timestamp` (synthesized; see note).",
        "- **User metadata stored, NOT payload-indexed**: "
        "`arc_conversation_id`, `turn_id`, `role`.",
        "",
        f"> {schema['note']}",
        "",
        "## Constraint firing rates across 30 LoCoMo queries",
        "",
        "| Constraint | Fired | % |",
        "| --- | --- | --- |",
    ]
    for k in ("speaker", "temporal_relation", "negation",
              "answer_form", "needs_aggregation"):
        c = firing.get(k, 0)
        pct = round(100 * c / max(n_q, 1), 1)
        lines.append(f"| `{k}` | {c} | {pct}% |")
    lines += [
        "",
        "Only `speaker` maps to a usable EM filter; `temporal_relation` "
        "is dropped (synthesized timestamps don't align with LoCoMo's "
        "real-world dates); `negation` is applied as a post-retrieval "
        "score nudge; `answer_form` / `needs_aggregation` have no schema "
        "support on this corpus.",
        "",
        "## Recall matrix",
        "",
        "| Variant | R@20 | R@50 |",
        "| --- | --- | --- |",
    ]
    # Reference rows
    lines.append(
        f"| em_v2f_speakerformat (ref) | "
        f"{REFERENCE['em_v2f_speakerformat'][0]:.4f} | "
        f"{REFERENCE['em_v2f_speakerformat'][1]:.4f} |"
    )
    lines.append(
        f"| em_two_speaker_filter (ref) | "
        f"{REFERENCE['em_two_speaker_filter'][0]:.4f} | "
        f"{REFERENCE['em_two_speaker_filter'][1]:.4f} |"
    )
    lines.append(
        f"| em_two_speaker_query_only (ref K=50 leader) | "
        f"{REFERENCE['em_two_speaker_query_only'][0]:.4f} | "
        f"**{REFERENCE['em_two_speaker_query_only'][1]:.4f}** |"
    )
    for variant in VARIANTS:
        v = results["variants"].get(variant, {})
        s = v.get("summary", {})
        lines.append(
            f"| **{variant}** | "
            f"**{s.get('mean_r@20', 0):.4f}** | "
            f"**{s.get('mean_r@50', 0):.4f}** |"
        )
    lines += ["", "## Filter-firing vs recall (intent_em_speaker_only)", ""]
    sp_only = results["variants"].get("intent_em_speaker_only", {})
    fs = sp_only.get("firing_summary", {})
    if fs:
        lines += [
            f"- Filter fired on {fs['n_with_filter']}/{n_q} queries.",
            f"- Queries with filter: R@20={fs.get('mean_r@20_with_filter', 'n/a')} "
            f"R@50={fs.get('mean_r@50_with_filter', 'n/a')}",
            f"- Queries without filter: R@20={fs.get('mean_r@20_without_filter', 'n/a')} "
            f"R@50={fs.get('mean_r@50_without_filter', 'n/a')}",
        ]

    # Verdict
    best_r50 = 0.0
    best_variant = None
    for variant in VARIANTS:
        r50 = results["variants"].get(variant, {}).get("summary", {}).get("mean_r@50", 0)
        if r50 > best_r50:
            best_r50 = r50
            best_variant = variant
    r50_tsqo = REFERENCE["em_two_speaker_query_only"][1]
    r50_ts = REFERENCE["em_two_speaker_filter"][1]

    lines += ["", "## Verdict", ""]
    if best_r50 >= 0.95:
        lines.append(
            f"- `{best_variant}` R@50={best_r50:.4f} ≥ 0.95: "
            "**new LoCoMo ceiling for EM** (above em_two_speaker_query_only)."
        )
    elif best_r50 > r50_tsqo + 0.005:
        lines.append(
            f"- `{best_variant}` R@50={best_r50:.4f} beats "
            f"em_two_speaker_query_only ({r50_tsqo:.4f}): structured parse "
            "adds value beyond regex speaker mention."
        )
    elif best_r50 >= r50_tsqo - 0.005:
        lines.append(
            f"- `{best_variant}` R@50={best_r50:.4f} ≈ "
            f"em_two_speaker_query_only ({r50_tsqo:.4f}): LLM-parsed speaker "
            "is roughly equivalent to regex name-mention; intent parsing "
            "buys little on a corpus whose schema only supports a speaker filter."
        )
    else:
        lines.append(
            f"- Best intent_em variant R@50={best_r50:.4f} UNDER "
            f"em_two_speaker_query_only ({r50_tsqo:.4f}): LLM speaker "
            "parsing is either underfiring or hallucinating participants."
        )

    # Cue-composition check
    no_cues = results["variants"].get("intent_em_filter_no_cues", {}).get(
        "summary", {}
    ).get("mean_r@50", 0)
    full = results["variants"].get("intent_em_full_filter", {}).get(
        "summary", {}
    ).get("mean_r@50", 0)
    lines.append("")
    if no_cues > full + 0.005:
        lines.append(
            f"- `intent_em_filter_no_cues` ({no_cues:.4f}) > "
            f"`intent_em_full_filter` ({full:.4f}) at K=50: "
            "confirms the em_two_speaker_query_only pattern — "
            "cues hurt when a hard filter already constrains the "
            "candidate pool."
        )
    elif full > no_cues + 0.005:
        lines.append(
            f"- `intent_em_full_filter` ({full:.4f}) > "
            f"`intent_em_filter_no_cues` ({no_cues:.4f}) at K=50: "
            "cues compose with filter; invalidates the 'cues hurt when "
            "filter applies' finding from deferred_archs."
        )
    else:
        lines.append(
            f"- `intent_em_full_filter` ({full:.4f}) ≈ "
            f"`intent_em_filter_no_cues` ({no_cues:.4f}) at K=50: "
            "cues neither help nor hurt meaningfully on top of the filter."
        )

    lines += [
        "",
        "## Outputs",
        "",
        "- `results/intent_em.json`",
        "- `results/intent_em.md`",
        "- Source: `intent_em.py`, `intemf_eval.py`",
    ]
    return lines


if __name__ == "__main__":
    asyncio.run(run())
