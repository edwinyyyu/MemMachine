"""Model-note augmented EventMemory ingestion for LoCoMo (v4).

v4 swaps v3's 3-labeled-line prompt (current_understanding/open_questions/
recent_realization) for the "neutral A'' " listener-observations prompt with
RESOLVED/FACT/COUNT/UPDATE/LINK/NAME labels and a PHATIC skip rule.

Differences from v3:
- Prompt format: ``{context_block}`` + ``{current_turn}`` (vs. v3's 5 fields).
- Output is either literal ``PHATIC`` or 1-4 labeled lines; the raw LLM text
  is stored as the note content (no JSON-style field parsing).
- If output is ``PHATIC`` (or empty), NO note event is ingested for that turn.
- Collection / partition / SQLite file prefixes use ``v4`` suffix so we never
  collide with v3 data.

All the rest (ingest_conversation shape, V_combined retrieval query, EM-canonical
context format, ``MessageContext(source="ModelNote")`` for note events,
timestamp+1us offsets, 3 LoCoMo conversations 26/30/41) is byte-equivalent
to v3.

Usage:
    uv run python evaluation/associative_recall/em_setup_notes_v4.py
    uv run python evaluation/associative_recall/em_setup_notes_v4.py --limit 5    # smoke
    uv run python evaluation/associative_recall/em_setup_notes_v4.py --conv locomo_conv-26
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import openai
from dotenv import load_dotenv
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.utils import async_with
from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    Content,
    Event,
    MessageContext,
    Text,
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

# v4: fresh collections so we never collide with v1/v2/v3.
COLLECTION_PREFIX = "arc_em_lc30_notes_v4"
LOGICAL_PREFIX = "arc_em_locomo30_notes_v4"
NAMESPACE = "arc_em_locomo30_notes_v4"

LOCOMO_CONV_IDS = ["locomo_conv-26", "locomo_conv-30", "locomo_conv-41"]

NOTES_MODEL = "gpt-5-mini"
NOTES_GEN_CACHE_FILE = CACHE_DIR / "notes_v4_gen_cache.json"
NOTES_SAMPLES_FILE = RESULTS_DIR / "notes_v4_samples.json"

EVENT_TYPE_MESSAGE = "message"
EVENT_TYPE_NOTE = "model_note"
NOTE_SPEAKER = "ModelNote"


# v4 neutral-examples listener-observations prompt (winning variant from
# notes_prompt_neutral.py, verbatim — only change is the {context_block} /
# {current_turn} placeholders).
NOTE_PROMPT = """\
You are a careful listener taking notes on a two-person conversation. For each CURRENT TURN you write 1-4 concrete, grounded observations — the kind a listener would jot so they could recall specifics later.

LABEL REFERENCE (use one label per line, at most 4 lines total):
- RESOLVED: <exact pronoun/deictic from current turn> -> <exact referent phrase from prior context>
    e.g. RESOLVED: "it" -> "the project we discussed yesterday"
- FACT: <a specific new detail explicitly stated in this turn>
    e.g. FACT: The deadline for the proposal is March 15th.
- COUNT: <entity> = <running total or duration with units>
    e.g. COUNT: houseplants = 3 (a succulent, a pothos, and a fern)
- UPDATE: <prior claim from context> -> <new/corrected claim in current turn>
    e.g. UPDATE: the flight was at 9am -> the flight is now at 11am (reschedule).
- LINK: <current element> refers to <earlier topic/event from context>
    e.g. LINK: "this plan" refers to the marketing plan from last month.
- NAME: <new proper noun or named entity introduced> = <short grounded description>
    e.g. NAME: Alex = a new hire on the design team.

PHATIC rule: output exactly "PHATIC" (and nothing else) if removing this turn from the conversation would lose no concrete information. Typical PHATIC turns are generic politeness, greetings, goodbyes, brief encouragement, or echo-agreements ("Thanks!", "Keep it up!", "Glad to hear", "Enjoy your day", "Bye!", "Great to see you", "That's awesome, keep it up!", "Your friendship means so much to me. Enjoy your day!", "Glad it helped!").

A turn is PHATIC even if it:
- mentions a topic already known from context (e.g. restating "running is good" when running was already discussed),
- expresses an emotion or compliment with no new fact ("your friendship means so much", "that's cool"),
- is a named address with no new content ("No worries, Mel!", "Thanks, Alex!").

A turn is NOT PHATIC if it introduces: a new number, date, name, object, place, quantity, or an update/correction to a prior claim.

Hard constraints:
- ONLY use information directly present in the CURRENT TURN (the context exists only to resolve references). Do NOT invent facts, numbers, dates, or names not stated.
- 1 observation per line; max ~20 words per line; no preamble, no explanations, no markdown.
- Prefer SPECIFIC referents (e.g. "the draft Sam sent Monday") over vague ones ("it", "that thing").
- Never write a thematic summary; write concrete listener observations.
- Restating a topic from context with no new detail is PHATIC, not a FACT. Meta-observations like "speaker expressed gratitude" or "speaker encouraged the other" are PHATIC, not FACTs.

CONTEXT (most recent last; each line is "<speaker>: <content>"):
{context_block}

CURRENT TURN:
{current_turn}

Observations (labeled lines, or PHATIC):
"""


# Label vocabulary — used only for stats, not generation.
NOTE_LABELS = ("RESOLVED", "FACT", "COUNT", "UPDATE", "LINK", "NAME")


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def _sha(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()


class _NotesCache:
    """Simple file-backed cache keyed by sha(model + prompt)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._cache: dict[str, str] = {}
        if path.exists():
            try:
                with open(path) as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}
        self._dirty = False

    def get(self, model: str, prompt: str) -> str | None:
        return self._cache.get(_sha(model, prompt))

    def put(self, model: str, prompt: str, response: str) -> None:
        self._cache[_sha(model, prompt)] = response
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._cache, f)
        tmp.replace(self._path)
        self._dirty = False


# --------------------------------------------------------------------------
# Data loading / timestamps
# --------------------------------------------------------------------------


def _conv_short(conv_id: str) -> str:
    return conv_id.rsplit("-", 1)[-1]


def load_speaker_map() -> dict[str, dict[str, str]]:
    with open(RESULTS_DIR / "conversation_two_speakers.json") as f:
        return json.load(f)["speakers"]


def load_conversation_segments(
    npz_path: Path,
) -> dict[str, list[tuple[int, str, str]]]:
    d = np.load(npz_path, allow_pickle=True)
    texts = d["texts"]
    cids = d["conversation_ids"]
    tids = d["turn_ids"]
    roles = d["roles"]
    out: dict[str, list[tuple[int, str, str]]] = {}
    for i in range(len(texts)):
        cid = str(cids[i])
        if cid not in LOCOMO_CONV_IDS:
            continue
        out.setdefault(cid, []).append((int(tids[i]), str(roles[i]), str(texts[i])))
    for cid in out:
        out[cid].sort(key=lambda t: t[0])
    return out


def _turn_timestamp(base: datetime, turn_id: int) -> datetime:
    return base + timedelta(seconds=60 * turn_id)


def _note_timestamp(turn_ts: datetime) -> datetime:
    return turn_ts + timedelta(microseconds=1)


# --------------------------------------------------------------------------
# EM-canonical formatting for LLM context
# --------------------------------------------------------------------------


def em_format(source: str, content: str, *, truncate_to: int | None = 220) -> str:
    """Canonical EM text form: "<source>: <content>"."""
    c = content.strip().replace("\n", " ")
    if truncate_to is not None and len(c) > truncate_to:
        c = c[:truncate_to] + "..."
    return f"{source}: {c}"


def _format_context_lines(
    prior_notes_prose: list[str],
    recent_turns: list[tuple[int, str, str]],
    related: list[dict],
) -> str:
    """v4 prompt is a single flat context block: recent turns + related events
    + prior notes, all EM-formatted, most-recent last.

    We interleave categories by rendering them in order: prior notes (oldest
    first), then earlier related events (oldest first), then recent turns
    (most recent last). This keeps the "most recent last" convention the
    prompt advertises. Prior context is strictly supplementary — the v4
    prompt pulls information only from CURRENT TURN — so exact ordering
    inside the block is low-stakes.
    """
    lines: list[str] = []
    # Earlier related events first (oldest first by turn_id if available).
    if related:
        rel_sorted = sorted(related, key=lambda r: r.get("turn_id", 0))
        for r in rel_sorted:
            lines.append(em_format(r["source"], r["text"]))
    # Prior notes in order of recency (oldest first — caller passes them
    # chronologically already; just render).
    for p in prior_notes_prose:
        lines.append(em_format(NOTE_SPEAKER, p, truncate_to=None))
    # Recent turns, chronological (most recent last).
    for _tid, sp, tx in recent_turns:
        lines.append(em_format(sp, tx, truncate_to=None))
    if not lines:
        return "(no prior context)"
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Note parsing
# --------------------------------------------------------------------------


def _clean_note_text(raw: str) -> str:
    """Strip code fences and normalize whitespace."""
    if not raw:
        return ""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _is_phatic(clean_text: str) -> bool:
    if not clean_text:
        return True
    # Accept "PHATIC" or whitespace variants exactly.
    head = clean_text.strip().splitlines()[0].strip().upper()
    return head == "PHATIC"


def _count_labels(clean_text: str) -> dict[str, int]:
    """Return a per-label count of lines beginning with each label keyword."""
    counts = dict.fromkeys(NOTE_LABELS, 0)
    if not clean_text:
        return counts
    for ln in clean_text.splitlines():
        s = ln.strip()
        if not s:
            continue
        for lbl in NOTE_LABELS:
            if s.upper().startswith(lbl + ":"):
                counts[lbl] += 1
                break
    return counts


# --------------------------------------------------------------------------
# Similarity retrieval for note context
# --------------------------------------------------------------------------


async def _retrieve_related(
    memory: EventMemory,
    query_text: str,
    *,
    K: int,
    exclude_turn_ids: set[int],
) -> list[dict]:
    if not query_text.strip():
        return []
    try:
        qr = await memory.query(query=query_text, vector_search_limit=K * 3)
    except Exception:
        return []
    out: list[dict] = []
    seen_seg_uuids: set = set()
    for sc in qr.scored_segment_contexts:
        if sc.seed_segment_uuid in seen_seg_uuids:
            continue
        for seg in sc.segments:
            props = seg.properties or {}
            tid = int(props.get("turn_id", -1))
            etype = str(props.get("event_type", ""))
            if tid in exclude_turn_ids and etype == EVENT_TYPE_MESSAGE:
                break
            source = str(props.get("speaker", "")) or (
                NOTE_SPEAKER if etype == EVENT_TYPE_NOTE else "Unknown"
            )
            text = seg.block.text if hasattr(seg.block, "text") else ""
            if not text:
                break
            out.append(
                {
                    "source": source,
                    "text": text,
                    "event_type": etype,
                    "turn_id": tid,
                }
            )
            seen_seg_uuids.add(sc.seed_segment_uuid)
            break
        if len(out) >= K:
            break
    return out[:K]


def _build_retrieval_query_v_combined(
    latest_note_prose: str | None,
    recent_turns: list[tuple[int, str, str]],
) -> str:
    lines: list[str] = []
    if latest_note_prose:
        lines.append(em_format(NOTE_SPEAKER, latest_note_prose, truncate_to=None))
    for _tid, sp, tx in recent_turns[-3:]:
        lines.append(em_format(sp, tx, truncate_to=None))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Note generation (v4)
# --------------------------------------------------------------------------


async def generate_note(
    *,
    openai_client,
    cache: _NotesCache,
    memory: EventMemory,
    participants: tuple[str, str],
    turn_id: int,
    turn_speaker: str,
    turn_text: str,
    prior_notes_prose: list[str],
    recent_turns: list[tuple[int, str, str]],
    K_related: int = 4,
) -> tuple[str, bool, bool, list[dict]]:
    """Generate one note.

    Returns:
        note_text: raw cleaned text (labeled lines); empty string if PHATIC
                   or model returned nothing.
        is_phatic: True if output was the PHATIC sentinel.
        cache_hit: True if prompt was already cached.
        related: related-events list used for context.
    """
    # V_combined similarity query: latest prior note + last 3 turns (excluding
    # the current turn, which was just added to recent_turns).
    latest_note = prior_notes_prose[-1] if prior_notes_prose else None
    sim_query = _build_retrieval_query_v_combined(latest_note, recent_turns[:-1])
    related = await _retrieve_related(
        memory,
        sim_query,
        K=K_related,
        exclude_turn_ids={turn_id},
    )

    # v4 prompt: single context_block + current_turn.
    context_block = _format_context_lines(
        prior_notes_prose=prior_notes_prose[-3:],
        recent_turns=recent_turns[-10:-1],  # exclude current turn
        related=related,
    )
    current_turn_line = em_format(turn_speaker, turn_text, truncate_to=None)

    prompt = NOTE_PROMPT.format(
        context_block=context_block,
        current_turn=current_turn_line,
    )

    cached = cache.get(NOTES_MODEL, prompt)
    cache_hit = cached is not None
    if cached is None:
        resp = await openai_client.chat.completions.create(
            model=NOTES_MODEL,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="low",
        )
        cached = resp.choices[0].message.content or ""
        cache.put(NOTES_MODEL, prompt, cached)

    cleaned = _clean_note_text(cached)
    if _is_phatic(cleaned):
        return "", True, cache_hit, related
    return cleaned, False, cache_hit, related


# --------------------------------------------------------------------------
# Ingestion per conversation
# --------------------------------------------------------------------------


async def ingest_conversation(
    conv_id: str,
    segments: list[tuple[int, str, str]],
    speakers: dict[str, str],
    vector_store: QdrantVectorStore,
    segment_store: SQLAlchemySegmentStore,
    embedder: OpenAIEmbedder,
    openai_client,
    notes_cache: _NotesCache,
    *,
    limit: int | None = None,
    resume: bool = False,
) -> dict:
    collection_name = f"{COLLECTION_PREFIX}_{_conv_short(conv_id)}"
    partition_key = collection_name

    if not resume:
        await vector_store.delete_collection(namespace=NAMESPACE, name=collection_name)
        await segment_store.delete_partition(partition_key)

    collection = await vector_store.open_or_create_collection(
        namespace=NAMESPACE,
        name=collection_name,
        config=VectorStoreCollectionConfig(
            vector_dimensions=embedder.dimensions,
            similarity_metric=embedder.similarity_metric,
            properties_schema=EventMemory.expected_vector_store_collection_schema(),
        ),
    )
    partition = await segment_store.open_or_create_partition(partition_key)

    memory = EventMemory(
        EventMemoryParams(
            vector_store_collection=collection,
            segment_store_partition=partition,
            embedder=embedder,
            reranker=None,
            derive_sentences=False,
            max_text_chunk_length=500,
        )
    )

    user_name = speakers.get("user") or "User"
    asst_name = speakers.get("assistant") or "Assistant"
    participants = (user_name, asst_name)
    base_ts = datetime(2023, 1, 1, tzinfo=timezone.utc)

    seg_iter = segments if limit is None else segments[:limit]
    n_total = len(seg_iter)

    prior_notes_prose: list[str] = []
    recent_turns: list[tuple[int, str, str]] = []
    samples: list[dict] = []
    already_ingested_turn_ids: set[int] = set()
    already_ingested_note_turn_ids: set[int] = set()

    t0 = time.monotonic()
    n_turns = 0
    n_notes = 0
    n_phatic = 0
    n_note_cache_hits = 0
    # Aggregate label counts over non-PHATIC notes in this conversation.
    label_totals = dict.fromkeys(NOTE_LABELS, 0)

    if resume:
        prior_by_tid: dict[int, str] = {}
        msgs_by_tid: dict[int, tuple[str, str]] = {}
        phatic_by_tid: set[int] = set()
        import json as _json

        from sqlalchemy import text as _sql_text

        async with segment_store._engine.connect() as conn:  # noqa: SLF001
            res = await conn.execute(
                _sql_text(
                    "SELECT properties, block FROM segment_store_sg WHERE partition_key = :pk"
                ),
                {"pk": partition_key},
            )
            for props_json, block_json in res:
                if not props_json:
                    continue
                props = _json.loads(props_json)
                tid = int(props.get("turn_id", {}).get("v", -1))
                etype = str(props.get("event_type", {}).get("v", ""))
                speaker = str(props.get("speaker", {}).get("v", ""))
                if tid < 0:
                    continue
                if etype == EVENT_TYPE_MESSAGE:
                    already_ingested_turn_ids.add(tid)
                    try:
                        block = _json.loads(block_json) if block_json else {}
                        txt = block.get("text", "")
                    except Exception:
                        txt = ""
                    msgs_by_tid[tid] = (speaker, txt)
                elif etype == EVENT_TYPE_NOTE:
                    already_ingested_note_turn_ids.add(tid)
                    try:
                        block = _json.loads(block_json) if block_json else {}
                        prose = block.get("text", "")
                    except Exception:
                        prose = ""
                    if prose:
                        prior_by_tid[tid] = prose

        sorted_msg_tids = sorted(msgs_by_tid.keys())
        for tid in sorted_msg_tids[-10:]:
            sp, tx = msgs_by_tid[tid]
            recent_turns.append((tid, sp, tx))
        sorted_note_tids = sorted(prior_by_tid.keys())
        for tid in sorted_note_tids[-8:]:
            prior_notes_prose.append(prior_by_tid[tid])

        n_turns = len(already_ingested_turn_ids)
        n_notes = len(already_ingested_note_turn_ids)
        print(
            f"[{conv_id}] resume: already have {n_turns} msgs, "
            f"{n_notes} notes (max_msg_tid={max(already_ingested_turn_ids) if already_ingested_turn_ids else -1}, "
            f"max_note_tid={max(already_ingested_note_turn_ids) if already_ingested_note_turn_ids else -1})",
            flush=True,
        )

    sample_idx = set()
    if n_total > 0:
        sample_idx = {0, n_total // 2, n_total - 1}

    for i, (turn_id, role, text) in enumerate(seg_iter):
        speaker_name = user_name if role == "user" else asst_name
        turn_ts = _turn_timestamp(base_ts, turn_id)
        t_iter = time.monotonic()

        # In v4 we may have skipped note ingestion for PHATIC turns; we can't
        # tell that from the DB alone. For resume simplicity, if the message
        # is ingested we still re-run note-gen (cache hit will short-circuit).
        if not (resume and turn_id in already_ingested_turn_ids):
            turn_event = Event(
                uuid=uuid4(),
                timestamp=turn_ts,
                body=Content(
                    context=MessageContext(source=speaker_name),
                    items=[Text(text=text.strip())],
                ),
                properties={
                    "arc_conversation_id": conv_id,
                    "turn_id": turn_id,
                    "role": role,
                    "speaker": speaker_name,
                    "event_type": EVENT_TYPE_MESSAGE,
                },
            )
            await memory.encode_events([turn_event])
            n_turns += 1
        recent_turns.append((turn_id, speaker_name, text.strip()))
        recent_turns = recent_turns[-10:]

        note_text = ""
        is_phatic = False
        cache_hit = False
        related: list[dict] = []

        if resume and turn_id in already_ingested_note_turn_ids:
            # Note already exists; skip regen.
            pass
        else:
            try:
                note_text, is_phatic, cache_hit, related = await generate_note(
                    openai_client=openai_client,
                    cache=notes_cache,
                    memory=memory,
                    participants=participants,
                    turn_id=turn_id,
                    turn_speaker=speaker_name,
                    turn_text=text.strip(),
                    prior_notes_prose=prior_notes_prose,
                    recent_turns=recent_turns,
                )
            except Exception as exc:
                print(f"[note-gen] conv={conv_id} turn={turn_id} err={exc!r}")
                note_text = ""
                is_phatic = False
                cache_hit = False
                related = []

            if cache_hit:
                n_note_cache_hits += 1

            if is_phatic:
                n_phatic += 1
            elif note_text:
                note_ts = _note_timestamp(turn_ts)
                note_event = Event(
                    uuid=uuid4(),
                    timestamp=note_ts,
                    body=Content(
                        context=MessageContext(source=NOTE_SPEAKER),
                        items=[Text(text=note_text)],
                    ),
                    properties={
                        "arc_conversation_id": conv_id,
                        "turn_id": turn_id,
                        "role": "model_note",
                        "speaker": NOTE_SPEAKER,
                        "event_type": EVENT_TYPE_NOTE,
                    },
                )
                await memory.encode_events([note_event])
                n_notes += 1
                prior_notes_prose.append(note_text)
                prior_notes_prose = prior_notes_prose[-8:]
                # Update label stats.
                for lbl, cnt in _count_labels(note_text).items():
                    label_totals[lbl] += cnt

        if i in sample_idx:
            samples.append(
                {
                    "conv_id": conv_id,
                    "position": "early"
                    if i == 0
                    else "late"
                    if i == n_total - 1
                    else "mid",
                    "turn_id": turn_id,
                    "speaker": speaker_name,
                    "turn_text": text.strip()[:400],
                    "is_phatic": is_phatic,
                    "note_text": note_text,
                    "label_counts": _count_labels(note_text) if note_text else {},
                    "related_used": [
                        {
                            "source": r["source"],
                            "text": r["text"][:200],
                            "event_type": r["event_type"],
                        }
                        for r in related
                    ],
                }
            )

        if (n_turns % 25) == 0:
            notes_cache.save()
        if os.environ.get("NOTES_V4_VERBOSE"):
            print(
                f"[{conv_id}] turn {turn_id} done in {time.monotonic() - t_iter:.1f}s (n_notes={n_notes}, n_phatic={n_phatic}, cache_hits={n_note_cache_hits})",
                flush=True,
            )

    ingest_time = time.monotonic() - t0
    notes_cache.save()

    await segment_store.close_partition(partition)
    await vector_store.close_collection(collection=collection)

    return {
        "conversation_id": conv_id,
        "collection_name": collection_name,
        "logical_collection_name": f"{LOGICAL_PREFIX}_{conv_id}",
        "partition_key": partition_key,
        "namespace": NAMESPACE,
        "n_turns_ingested": n_turns,
        "n_notes_ingested": n_notes,
        "n_phatic": n_phatic,
        "n_note_cache_hits": n_note_cache_hits,
        "label_totals": label_totals,
        "ingest_time_s": round(ingest_time, 2),
        "user_name": user_name,
        "assistant_name": asst_name,
        "samples": samples,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only ingest first N turns per conversation (smoke)",
    )
    parser.add_argument(
        "--conv", default=None, help="Single conversation id, e.g. locomo_conv-26"
    )
    parser.add_argument(
        "--concurrent_convs",
        type=int,
        default=3,
        help="# of conversations concurrently (each conv is sequential inside)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip delete of existing collection/partition; skip already-ingested turns",
    )
    args = parser.parse_args()

    speakers_map = load_speaker_map()
    conv_segments = load_conversation_segments(DATA_DIR / "segments_extended.npz")
    conv_ids = LOCOMO_CONV_IDS if args.conv is None else [args.conv]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    # Dedicated SQLite file for v4 — never touch v3 or standard-ingest data.
    sqlite_path = RESULTS_DIR / "eventmemory_notes_v4.sqlite3"
    sql_url = f"sqlite+aiosqlite:///{sqlite_path}"
    engine = create_async_engine(sql_url)
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

    notes_cache = _NotesCache(NOTES_GEN_CACHE_FILE)

    records: list[dict] = []
    try:
        semaphore = asyncio.Semaphore(max(1, args.concurrent_convs))
        tasks = [
            async_with(
                semaphore,
                ingest_conversation(
                    conv_id,
                    conv_segments[conv_id],
                    speakers_map[conv_id],
                    vector_store,
                    segment_store,
                    embedder,
                    openai_client,
                    notes_cache,
                    limit=args.limit,
                    resume=args.resume,
                ),
            )
            for conv_id in conv_ids
        ]
        records = await asyncio.gather(*tasks)
    finally:
        notes_cache.save()
        await segment_store.shutdown()
        await vector_store.shutdown()
        await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    out_collections = RESULTS_DIR / "eventmemory_notes_v4_collections.json"
    with open(out_collections, "w") as f:
        json.dump(
            {
                "namespace": NAMESPACE,
                "prefix": COLLECTION_PREFIX,
                "logical_prefix": LOGICAL_PREFIX,
                "sql_url": sql_url,
                "max_text_chunk_length": 500,
                "derive_sentences": False,
                "notes_model": NOTES_MODEL,
                "notes_cache_path": str(NOTES_GEN_CACHE_FILE),
                "retrieval_query_variant": "V_combined",
                "prompt_version": "v4_neutral_A_double_prime",
                "conversations": [
                    {k: v for k, v in r.items() if k != "samples"} for r in records
                ],
            },
            f,
            indent=2,
        )

    all_samples = [s for r in records for s in r.get("samples", [])]
    with open(NOTES_SAMPLES_FILE, "w") as f:
        json.dump({"samples": all_samples}, f, indent=2)

    for r in records:
        print(
            f"[ingested] {r['conversation_id']}: "
            f"turns={r['n_turns_ingested']} notes={r['n_notes_ingested']} "
            f"phatic={r['n_phatic']} cache_hits={r['n_note_cache_hits']} "
            f"labels={r['label_totals']} "
            f"in {r['ingest_time_s']}s -> {r['collection_name']}"
        )
    print(f"Saved: {out_collections}")
    print(f"Saved: {NOTES_SAMPLES_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
