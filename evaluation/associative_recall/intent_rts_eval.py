"""Evaluate intent_rts_* variants on LoCoMo-30 (real-timestamp ingest).

Uses the NEW EM collections from em_setup_rts.py
(prefix `arc_em_lc30_rts_v1_<conv>`) and the NEW SQLite at
`results/eventmemory_locomo_rts.sqlite3`. Does NOT touch the existing
non-rts collections / stores used by intemf_eval.py.

Variants:
  intent_rts_full           speaker + temporal filter + speakerformat cues
  intent_rts_temporal_only  temporal filter only + v2f cues
  intent_rts_speaker_only   speaker filter only + speakerformat cues (control)

Also runs a smoke test demonstrating the temporal filter narrows recall on
conv-26 (picks a [date_A, date_B] window, shows filtered vs unfiltered hits).

Caches:
  cache/locomo_rts_v2f_cache.json
  cache/locomo_rts_speakerformat_cache.json

Outputs:
  results/locomo_rts.json
  results/locomo_rts_eval.md
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import openai
from dotenv import load_dotenv
from em_architectures import (
    BESTSHOT_LLM_CACHE,
    EM_V2F_LLM_CACHE,
    _MergedLLMCache,
)
from em_setup_rts import datetime_from_locomo_time
from intent_em import load_intent_parse_cache, load_two_speaker_map
from intent_rts import (
    IntentRTSResult,
    intent_rts_full,
    intent_rts_speaker_only,
    intent_rts_temporal_only,
)
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.filter.filter_parser import And, Comparison
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

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

BUDGETS = (20, 50)
LOCOMO_CONV_IDS = ("locomo_conv-26", "locomo_conv-30", "locomo_conv-41")

LOCOMO_RTS_V2F_CACHE = CACHE_DIR / "locomo_rts_v2f_cache.json"
LOCOMO_RTS_SPEAKERFMT_CACHE = CACHE_DIR / "locomo_rts_speakerformat_cache.json"

VARIANTS = (
    "intent_rts_full",
    "intent_rts_temporal_only",
    "intent_rts_speaker_only",
)

# Reference numbers (from plan)
REFERENCE = {
    "em_v2f_speakerformat": (0.817, 0.892),
    "em_two_speaker_filter": (0.842, 0.900),
    "em_two_speaker_query_only": (0.800, 0.933),
    "intent_em_with_speakerformat_cues": (0.8167, 0.9083),
    "intent_em_filter_no_cues": (0.75, 0.900),
}


def load_locomo_questions() -> list[dict]:
    with open(DATA_DIR / "questions_extended.json") as f:
        qs = json.load(f)
    return [q for q in qs if q.get("benchmark") == "locomo"][:30]


def load_collections_meta() -> dict:
    with open(RESULTS_DIR / "eventmemory_locomo_rts_collections.json") as f:
        return json.load(f)


def compute_recall(retrieved: set[int], gold: set[int]) -> float:
    if not gold:
        return 1.0
    return len(retrieved & gold) / len(gold)


def compute_anchor_now(meta: dict) -> datetime:
    last = meta.get("last_session_dt")
    if last:
        try:
            dt = datetime.fromisoformat(last)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            pass
    return datetime(2024, 1, 1, tzinfo=UTC)


async def smoke_test_temporal(memory: EventMemory, conv_meta: dict) -> dict:
    """Verify temporal property_filter narrows retrieval to a date window."""
    sessions = conv_meta.get("sessions", [])
    if len(sessions) < 4:
        return {"skipped": True, "reason": "fewer than 4 sessions"}
    # Pick middle window: session index ~1/3 to session ~2/3.
    low = sessions[len(sessions) // 3]
    high = sessions[(2 * len(sessions)) // 3]
    start = datetime.fromisoformat(low["parsed_iso"])
    end = datetime.fromisoformat(high["parsed_iso"])
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    # Use a neutral, broad query to stress the filter.
    q = "what happened"
    unfiltered = await memory.query(q, vector_search_limit=30)
    filt = And(
        left=Comparison(field="timestamp", op=">=", value=start),
        right=Comparison(field="timestamp", op="<=", value=end),
    )
    filtered = await memory.query(q, vector_search_limit=30, property_filter=filt)

    def _extract(qr):
        rows = []
        for sc in qr.scored_segment_contexts:
            for seg in sc.segments:
                rows.append(
                    {
                        "turn_id": int(seg.properties.get("turn_id", -1)),
                        "session_idx": seg.properties.get("session_idx"),
                        "session_date_time": seg.properties.get("session_date_time"),
                    }
                )
        return rows

    u_rows = _extract(unfiltered)
    f_rows = _extract(filtered)
    # Check filter correctness: every filtered row's session date is in window
    violations = []
    for r in f_rows:
        try:
            ts = datetime_from_locomo_time(r["session_date_time"])
        except Exception:
            continue
        if ts < start or ts > end:
            violations.append(r)
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "unfiltered_n": len(u_rows),
        "filtered_n": len(f_rows),
        "unfiltered_session_indices": sorted(
            {r["session_idx"] for r in u_rows if r["session_idx"] is not None}
        ),
        "filtered_session_indices": sorted(
            {r["session_idx"] for r in f_rows if r["session_idx"] is not None}
        ),
        "violations": violations[:5],
        "ok": not violations and len(f_rows) > 0,
    }


async def evaluate_question(
    variant: str,
    memory: EventMemory,
    question: dict,
    *,
    plan: dict,
    speaker_map,
    participants,
    anchor_now: datetime,
    v2f_cache: _MergedLLMCache,
    speakerfmt_cache: _MergedLLMCache,
    openai_client,
    max_K: int,
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    gold = set(question.get("source_chat_ids", []))

    t0 = time.monotonic()
    if variant == "intent_rts_full":
        result: IntentRTSResult = await intent_rts_full(
            memory,
            q_text,
            conv_id,
            K=max_K,
            plan=plan,
            speaker_map=speaker_map,
            participants=participants,
            anchor_now=anchor_now,
            speakerfmt_cache=speakerfmt_cache,
            openai_client=openai_client,
        )
    elif variant == "intent_rts_temporal_only":
        result = await intent_rts_temporal_only(
            memory,
            q_text,
            conv_id,
            K=max_K,
            plan=plan,
            speaker_map=speaker_map,
            participants=participants,
            anchor_now=anchor_now,
            v2f_cache=v2f_cache,
            openai_client=openai_client,
        )
    elif variant == "intent_rts_speaker_only":
        result = await intent_rts_speaker_only(
            memory,
            q_text,
            conv_id,
            K=max_K,
            plan=plan,
            speaker_map=speaker_map,
            participants=participants,
            anchor_now=anchor_now,
            speakerfmt_cache=speakerfmt_cache,
            openai_client=openai_client,
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
        "temporal_window": analysis.get("temporal_window"),
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
        raise RuntimeError("no sql_url")
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

    v2f_cache = _MergedLLMCache(
        reader_paths=[
            BESTSHOT_LLM_CACHE,
            EM_V2F_LLM_CACHE,
            CACHE_DIR / "meta_llm_cache.json",
            CACHE_DIR / "arch_llm_cache.json",
            CACHE_DIR / "emdef_v2f_llm_cache.json",
            CACHE_DIR / "intemf_v2f_llm_cache.json",
            LOCOMO_RTS_V2F_CACHE,
        ],
        writer_path=LOCOMO_RTS_V2F_CACHE,
    )
    speakerfmt_cache = _MergedLLMCache(
        reader_paths=[
            CACHE_DIR / "emretune_v2f_speakerformat_cache.json",
            CACHE_DIR / "intemf_speakerformat_llm_cache.json",
            LOCOMO_RTS_SPEAKERFMT_CACHE,
        ],
        writer_path=LOCOMO_RTS_SPEAKERFMT_CACHE,
    )

    memories: dict[str, EventMemory] = {}
    participants_by_conv: dict[str, tuple[str, str]] = {}
    anchor_by_conv: dict[str, datetime] = {}
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
        anchor_by_conv[conv_id] = compute_anchor_now(meta)
        opened_resources.append((coll, part))

    # ----- Step 3: smoke test temporal filter on conv-26 -----
    smoke = {}
    for cid in LOCOMO_CONV_IDS:
        smoke[cid] = await smoke_test_temporal(memories[cid], conv_to_meta[cid])

    parse_cache = load_intent_parse_cache()
    speaker_map = load_two_speaker_map()

    max_K = max(BUDGETS)
    results: dict = {
        "variants": {},
        "budgets": list(BUDGETS),
        "questions": len(questions),
        "smoke_test": smoke,
        "anchor_now": {k: v.isoformat() for k, v in anchor_by_conv.items()},
    }

    # Firing counts (same plan cache)
    firing_counts = defaultdict(int)
    for q in questions:
        plan = parse_cache.get(q["question"].strip(), {}) or {}
        cs = plan.get("constraints", {}) or {}
        if cs.get("speaker"):
            firing_counts["speaker"] += 1
        if cs.get("temporal_relation"):
            firing_counts["temporal_relation"] += 1
        if cs.get("negation"):
            firing_counts["negation"] += 1
        if cs.get("answer_form"):
            firing_counts["answer_form"] += 1
    results["firing_counts"] = dict(firing_counts)

    for variant in VARIANTS:
        rows: list[dict] = []
        t0 = time.monotonic()
        for q in questions:
            cid = q["conversation_id"]
            mem = memories[cid]
            plan = parse_cache.get(q["question"].strip(), {}) or {
                "intent_type": "other",
                "constraints": {},
                "entities": [],
                "primary_topic": None,
                "needs_aggregation": False,
                "parse_ok": False,
            }
            row = await evaluate_question(
                variant,
                mem,
                q,
                plan=plan,
                speaker_map=speaker_map,
                participants=participants_by_conv[cid],
                anchor_now=anchor_by_conv[cid],
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
        cat_summary = {}
        for cat, cat_rows in by_cat.items():
            d = {"n": len(cat_rows)}
            for K in BUDGETS:
                d[f"mean_r@{K}"] = round(
                    sum(r[f"r@{K}"] for r in cat_rows) / max(len(cat_rows), 1), 4
                )
            cat_summary[cat] = d

        with_filter = [r for r in rows if r["applied_filter"]]
        without_filter = [r for r in rows if not r["applied_filter"]]
        with_temporal = [
            r for r in rows if "temporal_relation" in r.get("applied_constraints", [])
        ]
        without_temporal = [
            r
            for r in rows
            if "temporal_relation" not in r.get("applied_constraints", [])
        ]
        firing_summary = {
            "n_with_filter": len(with_filter),
            "n_without_filter": len(without_filter),
            "n_with_temporal": len(with_temporal),
            "n_without_temporal": len(without_temporal),
        }
        for K in BUDGETS:
            if with_filter:
                firing_summary[f"mean_r@{K}_with_filter"] = round(
                    sum(r[f"r@{K}"] for r in with_filter) / len(with_filter), 4
                )
            if without_filter:
                firing_summary[f"mean_r@{K}_without_filter"] = round(
                    sum(r[f"r@{K}"] for r in without_filter) / len(without_filter), 4
                )
            if with_temporal:
                firing_summary[f"mean_r@{K}_with_temporal"] = round(
                    sum(r[f"r@{K}"] for r in with_temporal) / len(with_temporal), 4
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
            f"(filter={firing_summary['n_with_filter']} temp={firing_summary['n_with_temporal']})",
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
    out_json = RESULTS_DIR / "locomo_rts.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {out_json}")

    md = build_markdown_report(results)
    out_md = RESULTS_DIR / "locomo_rts_eval.md"
    out_md.write_text("\n".join(md))
    print(f"Saved: {out_md}")


def build_markdown_report(results: dict) -> list[str]:
    firing = results.get("firing_counts", {})
    n_q = results["questions"]
    smoke = results.get("smoke_test", {})

    lines = [
        "# intent_parser on EventMemory with REAL LoCoMo timestamps",
        "",
        "## Schema changes",
        "",
        "- `timestamp` on each event now = session_date_time (parsed) + "
        "microsecond offset per in-session turn.",
        "- Per-event metadata adds `session_idx`, `in_session_idx`, "
        "`session_date_time`, `dia_id`.",
        "",
        "## Constraint firing rates (LoCoMo-30)",
        "",
        "| Constraint | Fired | % |",
        "| --- | --- | --- |",
    ]
    for k in ("speaker", "temporal_relation", "negation", "answer_form"):
        c = firing.get(k, 0)
        lines.append(f"| `{k}` | {c} | {round(100 * c / max(n_q, 1), 1)}% |")

    lines += ["", "## Temporal-filter smoke test", ""]
    for cid, r in smoke.items():
        if r.get("skipped"):
            lines.append(f"- {cid}: skipped ({r.get('reason')})")
            continue
        lines.append(
            f"- {cid}: window [{r['start']}, {r['end']}] — "
            f"unfiltered hits {r['unfiltered_n']} ({len(r['unfiltered_session_indices'])} sessions), "
            f"filtered hits {r['filtered_n']} ({len(r['filtered_session_indices'])} sessions), "
            f"ok={r.get('ok')}, violations={len(r.get('violations', []))}."
        )

    lines += [
        "",
        "## Recall matrix",
        "",
        "| Variant | R@20 | R@50 |",
        "| --- | --- | --- |",
    ]
    for ref_name, (r20, r50) in REFERENCE.items():
        lines.append(f"| {ref_name} (ref) | {r20:.4f} | {r50:.4f} |")
    for variant in VARIANTS:
        v = results["variants"].get(variant, {})
        s = v.get("summary", {})
        lines.append(
            f"| **{variant}** | **{s.get('mean_r@20', 0):.4f}** | "
            f"**{s.get('mean_r@50', 0):.4f}** |"
        )

    # Per-category for the full variant
    lines += ["", "## Per-category (intent_rts_full)", ""]
    by_cat = results["variants"].get("intent_rts_full", {}).get("by_category", {})
    lines.append("| Category | n | R@20 | R@50 |")
    lines.append("| --- | --- | --- | --- |")
    for cat, d in sorted(by_cat.items()):
        lines.append(
            f"| {cat} | {d['n']} | {d['mean_r@20']:.4f} | {d['mean_r@50']:.4f} |"
        )

    # Per-category temporal focus
    lines += [
        "",
        "## locomo_temporal focus (all variants)",
        "",
        "| Variant | n | R@20 | R@50 |",
        "| --- | --- | --- | --- |",
    ]
    for variant in VARIANTS:
        by_cat = results["variants"].get(variant, {}).get("by_category", {})
        t = by_cat.get("locomo_temporal")
        if t:
            lines.append(
                f"| {variant} | {t['n']} | {t['mean_r@20']:.4f} | {t['mean_r@50']:.4f} |"
            )

    lines += ["", "## Firing vs recall", ""]
    for variant in VARIANTS:
        fs = results["variants"].get(variant, {}).get("firing_summary", {})
        if not fs:
            continue
        lines.append(f"### {variant}")
        lines.append("")
        lines.append(
            f"- Filter fired on {fs['n_with_filter']}/{n_q} queries "
            f"(temporal filter: {fs['n_with_temporal']}/{n_q})."
        )
        for K in (20, 50):
            with_f = fs.get(f"mean_r@{K}_with_filter")
            without_f = fs.get(f"mean_r@{K}_without_filter")
            with_t = fs.get(f"mean_r@{K}_with_temporal")
            line = f"- K={K}: "
            if with_f is not None:
                line += f"with_filter={with_f} "
            if without_f is not None:
                line += f"without_filter={without_f} "
            if with_t is not None:
                line += f"with_temporal={with_t}"
            lines.append(line)
        lines.append("")

    # Verdict
    best_r50 = 0.0
    best_variant = None
    for variant in VARIANTS:
        r50 = (
            results["variants"].get(variant, {}).get("summary", {}).get("mean_r@50", 0)
        )
        if r50 > best_r50:
            best_r50 = r50
            best_variant = variant
    tsqo50 = REFERENCE["em_two_speaker_query_only"][1]
    v2fsf50 = REFERENCE["em_v2f_speakerformat"][1]

    lines += ["", "## Verdict", ""]
    if best_r50 >= tsqo50 + 0.005:
        lines.append(
            f"- `{best_variant}` R@50={best_r50:.4f} > "
            f"em_two_speaker_query_only ({tsqo50:.4f}): real-timestamp "
            "filter adds measurable recall over the speaker-only ceiling."
        )
    elif best_r50 >= tsqo50 - 0.005:
        lines.append(
            f"- `{best_variant}` R@50={best_r50:.4f} ≈ "
            f"em_two_speaker_query_only ({tsqo50:.4f}): temporal filter "
            "adds nothing on top of speaker filter at LoCoMo's granularity."
        )
    else:
        lines.append(
            f"- Best variant R@50={best_r50:.4f} UNDER "
            f"em_two_speaker_query_only ({tsqo50:.4f}): temporal filter "
            "is too narrow or too rarely resolvable; net negative."
        )

    temp_only = (
        results["variants"]
        .get("intent_rts_temporal_only", {})
        .get("summary", {})
        .get("mean_r@50", 0)
    )
    if temp_only > v2fsf50 + 0.005:
        lines.append(
            f"- `intent_rts_temporal_only` R@50={temp_only:.4f} > "
            f"em_v2f_speakerformat ({v2fsf50:.4f}): temporal signal is a "
            "real channel independent of speaker."
        )
    else:
        lines.append(
            f"- `intent_rts_temporal_only` R@50={temp_only:.4f} ≤ "
            f"em_v2f_speakerformat ({v2fsf50:.4f}): temporal signal alone "
            "does not beat speakerformat cues."
        )

    lines += [
        "",
        "## Outputs",
        "",
        "- `results/locomo_rts.json`",
        "- `results/locomo_rts_eval.md`",
        "- `results/locomo_rts_ingest.md`",
        "- `results/eventmemory_locomo_rts_collections.json`",
        "- Source: `em_setup_rts.py`, `intent_rts.py`, `intent_rts_eval.py`",
    ]
    return lines


if __name__ == "__main__":
    asyncio.run(run())
