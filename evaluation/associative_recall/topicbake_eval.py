"""Evaluate topic-baked EventMemory on LoCoMo-30.

Uses the NEW topic-baked collections produced by em_setup_topic.py
(`arc_em_lc30_topic_v1_<conv>` + `results/eventmemory_topic.sqlite3`). Does
NOT modify framework files, does NOT touch the standard em_setup collections.

Variants:
  em_cosine_baseline_topic  -- raw query cosine, expand_context=0
  em_v2f_topic              -- current V2F_PROMPT (primer + 2 cues, merge by
                                max score per turn_id)
  em_v2f_topic_prefix       -- v2f cues produced with a "speaker + topic"
                                retune prompt so cues mimic the embedded
                                format "[topic: <topic>] <text>"
  em_topic_plus_speaker_filter
                            -- em_v2f_topic + two_speaker-style property
                                filter on context.source (see em_two_speaker).

Compares against:
  em_cosine_baseline (standard ingest): 0.7333 / 0.8833
  em_v2f (standard ingest):             0.7417 / 0.8833
  v2f_em_speakerformat (retune):        0.8167 / 0.8917

Dedicated caches:
    cache/topicbake_v2f_cache.json           (vanilla V2F_PROMPT)
    cache/topicbake_v2f_prefix_cache.json    (topic-prefix retune)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import openai
from dotenv import load_dotenv
from em_architectures import (
    V2F_MODEL,
    EMHit,
    _dedupe_by_turn_id,
    _merge_by_max_score,
    _MergedLLMCache,
    _query_em,
    em_cosine,
    em_v2f,
    format_primer_context,
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
from sqlalchemy.ext.asyncio import create_async_engine

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

BUDGETS = (20, 50)
LOCOMO_CONV_IDS = ("locomo_conv-26", "locomo_conv-30", "locomo_conv-41")

VARIANTS = [
    "em_cosine_baseline_topic",
    "em_v2f_topic",
    "em_v2f_topic_prefix",
    "em_topic_plus_speaker_filter",
]

V2F_CACHE_FILE = CACHE_DIR / "topicbake_v2f_cache.json"
V2F_PREFIX_CACHE_FILE = CACHE_DIR / "topicbake_v2f_prefix_cache.json"


# --------------------------------------------------------------------------
# Topic-prefix retune prompt
# --------------------------------------------------------------------------


V2F_TOPIC_PREFIX_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history between {participant_1} and {participant_2}. Each turn is embedded \
in the format:
"<speaker>: [topic: <topic phrase>] <chat content>"
Example: "{participant_1}: [topic: LGBTQ support group visit] I went to the support group yesterday."

Your cues will be embedded and compared via cosine similarity against \
those turns. To align distributions, EACH cue MUST follow that exact \
format: begin with "<speaker>: [topic: <short topic phrase>] " where \
<speaker> is {participant_1} or {participant_2}. The topic phrase should \
be 2-6 words that naturally label what the turn is about.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is \
this search going? What kind of content is still missing? Should you \
search for similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep \
searching for more even if some are already found.

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns. Do NOT \
write questions ("Did you mention X?"). Write chat content that would \
actually appear.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <speaker>: [topic: <topic phrase>] <text>
CUE: <speaker>: [topic: <topic phrase>] <text>
Nothing else."""


CUE_RE = re.compile(
    r"^\s*(?:\[?[A-Z_]+\]?\s*[:\-]\s*)?CUE\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _parse_prefix_cues(response: str, max_cues: int = 2) -> list[str]:
    cues: list[str] = []
    for m in CUE_RE.finditer(response):
        cue = m.group(1).strip().strip('"').strip("'").strip()
        if cue:
            cues.append(cue)
        if len(cues) >= max_cues:
            break
    return cues


# --------------------------------------------------------------------------
# Name-mention speaker classification (reuse pattern from em_two_speaker).
# --------------------------------------------------------------------------


try:
    from speaker_attributed import extract_name_mentions
except Exception:  # pragma: no cover
    _NAME_RE = re.compile(r"\b([A-Z][a-z]+)\b")

    def extract_name_mentions(q: str) -> list[str]:
        # Fallback: naive capitalized-token extractor.
        return _NAME_RE.findall(q or "")


def classify_speaker_side(
    question: str,
    user_name: str,
    asst_name: str,
) -> tuple[str, list[str]]:
    tokens = extract_name_mentions(question)
    lowered = {t.lower() for t in tokens}
    hit_user = (user_name or "").lower() in lowered and user_name != "UNKNOWN"
    hit_asst = (asst_name or "").lower() in lowered and asst_name != "UNKNOWN"
    if hit_user and hit_asst:
        side = "both"
    elif hit_user:
        side = "user"
    elif hit_asst:
        side = "assistant"
    else:
        side = "none"
    return side, tokens


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------


def load_questions() -> list[dict]:
    with open(DATA_DIR / "questions_extended.json") as f:
        qs = json.load(f)
    return [q for q in qs if q.get("benchmark") == "locomo"][:30]


def load_collections_meta() -> dict:
    with open(RESULTS_DIR / "eventmemory_topic_collections.json") as f:
        return json.load(f)


def load_speaker_map() -> dict[str, dict[str, str]]:
    with open(RESULTS_DIR / "conversation_two_speakers.json") as f:
        return json.load(f)["speakers"]


def compute_recall(retrieved: set[int], gold: set[int]) -> float:
    if not gold:
        return 1.0
    return len(retrieved & gold) / len(gold)


async def _llm_call(
    openai_client, prompt: str, cache: _MergedLLMCache
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


async def _em_query_with_filter(
    memory: EventMemory,
    text: str,
    *,
    vector_search_limit: int,
    expand_context: int = 0,
    property_filter=None,
) -> list[EMHit]:
    qr = await memory.query(
        query=text,
        vector_search_limit=vector_search_limit,
        expand_context=expand_context,
        property_filter=property_filter,
    )
    hits: list[EMHit] = []
    for sc in qr.scored_segment_contexts:
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
    return hits


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------


async def run_cosine_baseline_topic(
    memory: EventMemory, question: str, *, max_K: int
) -> tuple[list[EMHit], dict]:
    hits = await em_cosine(memory, question, K=max_K, expand_context=0)
    return hits, {}


async def run_v2f_topic(
    memory: EventMemory,
    question: str,
    *,
    max_K: int,
    cache: _MergedLLMCache,
    openai_client,
) -> tuple[list[EMHit], dict]:
    hits, meta = await em_v2f(
        memory,
        question,
        K=max_K,
        llm_cache=cache,
        openai_client=openai_client,
        expand_context=0,
    )
    return hits, meta


async def run_v2f_topic_prefix(
    memory: EventMemory,
    question: str,
    *,
    max_K: int,
    cache: _MergedLLMCache,
    openai_client,
    user_name: str,
    asst_name: str,
) -> tuple[list[EMHit], dict]:
    # Primer: raw question against topic-baked memory, expand=0, K=10.
    primer_hits = _dedupe_by_turn_id(
        await _query_em(memory, question, vector_search_limit=10, expand_context=0)
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text} for h in primer_hits
    ]
    context_section = format_primer_context(primer_segments)

    prompt = V2F_TOPIC_PREFIX_PROMPT.format(
        question=question,
        context_section=context_section,
        participant_1=user_name,
        participant_2=asst_name,
    )
    raw, hit = await _llm_call(openai_client, prompt, cache)
    cues = _parse_prefix_cues(raw, max_cues=2)

    primer_for_merge = await _query_em(
        memory, question, vector_search_limit=max_K, expand_context=0
    )
    batches = [primer_for_merge]
    for cue in cues[:2]:
        batches.append(
            await _query_em(memory, cue, vector_search_limit=max_K, expand_context=0)
        )
    merged = _merge_by_max_score(batches)
    return merged[:max_K], {"cues": cues, "cache_hit": hit}


async def run_topic_plus_speaker_filter(
    memory: EventMemory,
    question: str,
    *,
    max_K: int,
    cache: _MergedLLMCache,
    openai_client,
    user_name: str,
    asst_name: str,
    role_only_top_m: int = 5,
) -> tuple[list[EMHit], dict]:
    """em_v2f_topic + name-mention -> property_filter on context.source."""
    v2f_hits, v2f_meta = await em_v2f(
        memory,
        question,
        K=max(max_K, 50),
        llm_cache=cache,
        openai_client=openai_client,
        expand_context=0,
    )
    side, tokens = classify_speaker_side(question, user_name, asst_name)

    meta: dict = {
        "v2f_cues": v2f_meta.get("cues", []),
        "v2f_cache_hit": v2f_meta.get("cache_hit", False),
        "matched_side": side,
        "query_name_tokens": tokens,
        "applied_speaker_filter": False,
        "appended_turn_ids": [],
        "dropped_v2f_turn_ids": [],
    }

    if side not in ("user", "assistant"):
        return v2f_hits[:max_K], meta

    matched_name = user_name if side == "user" else asst_name
    meta["applied_speaker_filter"] = True
    meta["matched_name"] = matched_name

    speaker_filter = Comparison(field="context.source", op="=", value=matched_name)
    speaker_hits = await _em_query_with_filter(
        memory,
        question,
        vector_search_limit=max_K + 10,
        expand_context=0,
        property_filter=speaker_filter,
    )
    speaker_hits = _dedupe_by_turn_id(speaker_hits)

    matched_role = side
    kept_v2f = [h for h in v2f_hits if h.role == matched_role]
    dropped = [h.turn_id for h in v2f_hits if h.role != matched_role]
    meta["dropped_v2f_turn_ids"] = dropped

    seen = {h.turn_id for h in kept_v2f}
    appended: list[EMHit] = []
    for h in speaker_hits:
        if h.turn_id in seen:
            continue
        appended.append(h)
        seen.add(h.turn_id)
        if len(appended) >= (role_only_top_m + 10):
            break
    meta["appended_turn_ids"] = [h.turn_id for h in appended]

    merged = kept_v2f + appended
    return merged[:max_K], meta


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------


async def evaluate_question(
    variant: str,
    memory: EventMemory,
    question: dict,
    *,
    user_name: str,
    asst_name: str,
    v2f_cache: _MergedLLMCache,
    v2f_prefix_cache: _MergedLLMCache,
    openai_client,
    max_K: int,
) -> dict:
    q_text = question["question"]
    gold = set(question.get("source_chat_ids", []))

    t0 = time.monotonic()
    meta: dict = {}

    if variant == "em_cosine_baseline_topic":
        hits, meta = await run_cosine_baseline_topic(memory, q_text, max_K=max_K)
    elif variant == "em_v2f_topic":
        hits, meta = await run_v2f_topic(
            memory,
            q_text,
            max_K=max_K,
            cache=v2f_cache,
            openai_client=openai_client,
        )
    elif variant == "em_v2f_topic_prefix":
        hits, meta = await run_v2f_topic_prefix(
            memory,
            q_text,
            max_K=max_K,
            cache=v2f_prefix_cache,
            openai_client=openai_client,
            user_name=user_name,
            asst_name=asst_name,
        )
    elif variant == "em_topic_plus_speaker_filter":
        hits, meta = await run_topic_plus_speaker_filter(
            memory,
            q_text,
            max_K=max_K,
            cache=v2f_cache,  # reuse vanilla v2f cache for consistency
            openai_client=openai_client,
            user_name=user_name,
            asst_name=asst_name,
        )
    else:
        raise KeyError(variant)

    elapsed = time.monotonic() - t0

    row: dict = {
        "conversation_id": question["conversation_id"],
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "gold_turn_ids": sorted(gold),
        "n_hits": len(hits),
        "time_s": round(elapsed, 3),
    }
    row.update(meta)
    for K in BUDGETS:
        topk = hits[:K]
        retrieved = {h.turn_id for h in topk}
        row[f"r@{K}"] = round(compute_recall(retrieved, gold), 4)
        row[f"retrieved_turn_ids@{K}"] = sorted(retrieved)
    return row


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated variants to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions (smoke test)",
    )
    args = parser.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    collections_meta = load_collections_meta()
    speaker_map = load_speaker_map()
    questions = load_questions()
    if args.limit is not None:
        questions = questions[: args.limit]

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
        raise RuntimeError("No SQL_URL in collections meta")
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

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    v2f_cache = _MergedLLMCache(
        reader_paths=[V2F_CACHE_FILE], writer_path=V2F_CACHE_FILE
    )
    v2f_prefix_cache = _MergedLLMCache(
        reader_paths=[V2F_PREFIX_CACHE_FILE], writer_path=V2F_PREFIX_CACHE_FILE
    )

    # Open EM per conversation.
    memories: dict[str, EventMemory] = {}
    speakers_by_conv: dict[str, tuple[str, str]] = {}
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
        pair = speaker_map[conv_id]
        speakers_by_conv[conv_id] = (
            pair.get("user") or "User",
            pair.get("assistant") or "Assistant",
        )
        opened_resources.append((coll, part))

    max_K = max(BUDGETS)
    results: dict = {
        "variants": {},
        "budgets": list(BUDGETS),
        "questions": len(questions),
    }

    for variant in variants:
        rows: list[dict] = []
        t0 = time.monotonic()
        for q in questions:
            cid = q["conversation_id"]
            user_name, asst_name = speakers_by_conv[cid]
            row = await evaluate_question(
                variant,
                memories[cid],
                q,
                user_name=user_name,
                asst_name=asst_name,
                v2f_cache=v2f_cache,
                v2f_prefix_cache=v2f_prefix_cache,
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

        results["variants"][variant] = {
            "summary": summary,
            "by_category": cat_summary,
            "per_question": rows,
        }
        v2f_cache.save()
        v2f_prefix_cache.save()

        print(
            f"[{variant}] n={summary['n']} "
            f"r@20={summary['mean_r@20']:.4f} r@50={summary['mean_r@50']:.4f} "
            f"in {summary['time_s']:.1f}s",
            flush=True,
        )

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
    out_json = RESULTS_DIR / "topic_baking.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")

    # Markdown.
    write_markdown(results, questions, out_md=RESULTS_DIR / "topic_baking.md")
    print(f"Saved: {RESULTS_DIR / 'topic_baking.md'}")


# Comparison baselines (standard EM, no topic baking, from em_eval):
BASELINES = {
    "em_cosine_baseline (standard ingest)": {"r@20": 0.7333, "r@50": 0.8833},
    "em_v2f (standard ingest)": {"r@20": 0.7417, "r@50": 0.8833},
    "v2f_em_speakerformat (retune)": {"r@20": 0.8167, "r@50": 0.8917},
    "em_two_speaker_filter (standard+filter)": {"r@20": 0.8417, "r@50": 0.9},
    "em_two_speaker_query_only (filter only)": {"r@20": 0.8, "r@50": 0.9333},
}


def write_markdown(results: dict, questions: list[dict], *, out_md: Path) -> None:
    lines: list[str] = [
        "# Topic-baking at ingestion (EventMemory, LoCoMo-30)",
        "",
        "## Setup",
        "",
        f"- n_questions = {len(questions)} (benchmark=locomo, first 30)",
        "- EventMemory backend, `text-embedding-3-small`, `gpt-5-mini`",
        '- Embedded text format: `"{source}: [topic: <topic>] <text>"` '
        "(speaker via MessageContext.source, topic prefix in Text item).",
        "- New Qdrant collection prefix `arc_em_lc30_topic_v1_{26,30,41}`, "
        "new SQLite `results/eventmemory_topic.sqlite3`.",
        "- Dedicated caches: `cache/topicbake_llm_cache.json` (ingest), "
        "`cache/topicbake_v2f_cache.json`, "
        "`cache/topicbake_v2f_prefix_cache.json` (eval).",
        "",
        "## Recall comparison",
        "",
        "| Variant | R@20 | R@50 | time (s) |",
        "| --- | --- | --- | --- |",
    ]
    for variant, data in results["variants"].items():
        s = data["summary"]
        lines.append(
            f"| `{variant}` | {s['mean_r@20']:.4f} | {s['mean_r@50']:.4f} | "
            f"{s['time_s']:.1f} |"
        )
    lines += [
        "",
        "## Standard EM baselines (no topic baking)",
        "",
        "| Baseline | R@20 | R@50 |",
        "| --- | --- | --- |",
    ]
    for name, s in BASELINES.items():
        lines.append(f"| {name} | {s['r@20']:.4f} | {s['r@50']:.4f} |")

    # Category breakdown for the most interesting variants
    lines += ["", "## By category", ""]
    for variant, data in results["variants"].items():
        lines.append(f"### `{variant}`")
        lines.append("")
        lines.append("| category | n | R@20 | R@50 |")
        lines.append("| --- | --- | --- | --- |")
        for cat, cs in sorted(data.get("by_category", {}).items()):
            lines.append(
                f"| {cat} | {cs['n']} | {cs['mean_r@20']:.4f} | {cs['mean_r@50']:.4f} |"
            )
        lines.append("")

    # Sample questions -> prefix cues
    prefix_rows = results["variants"].get("em_v2f_topic_prefix", {}).get("per_question")
    if prefix_rows:
        lines += ["## Sample `em_v2f_topic_prefix` cues (3 questions)", ""]
        n = len(prefix_rows)
        idxs = [0, n // 2, n - 1] if n >= 3 else list(range(n))
        for i in idxs:
            r = prefix_rows[i]
            lines.append(
                f"- Q{i} [{r['conversation_id']}] `{r['question']!r}` "
                f"(R@20={r['r@20']:.2f}, R@50={r['r@50']:.2f}):"
            )
            for c in r.get("cues", []):
                lines.append(f"  - `{c}`")
            lines.append("")

    lines += [
        "## Outputs",
        "",
        "- `results/topic_baking.json`",
        "- `results/topic_baking.md`",
        "- `results/eventmemory_topic_collections.json` (cleanup manifest)",
        "- `results/topic_baking_turns.json` (per-turn topic audit)",
        "- Source: `em_setup_topic.py`, `topicbake_eval.py`",
        "",
    ]
    out_md.write_text("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
