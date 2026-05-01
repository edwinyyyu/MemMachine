"""Model-note augmented EventMemory ingestion for LoCoMo-30 (v3).

Per turn we ingest the message normally (speaker baked via
`MessageContext.source`), then at `timestamp + 1us` we ingest a short
internal-monologue note produced by gpt-5-mini. The note is written as
another EM event with `MessageContext(source="ModelNote")` so its embedded
text becomes:
    "ModelNote: current_understanding: ... open_questions: ... recent_realization: ..."

Critical formatting rule (v3): the note-generation LLM sees context in the
EXACT SAME format as EventMemory's embedded representation at query time,
i.e. `"<source>: <content>"` strings. No JSON, no bracket prefixes, no
[NOTE]/[CHAT] labels. This keeps the ingest distribution aligned with the
query distribution.

Similarity-retrieved context is queried with V_combined:
    query_text = "\n".join([latest_note_formatted, *last_3_turns_formatted])

This file creates fresh collections / SQLite (`v3`) so it never touches
existing em_setup / em_setup_notes data.

Usage:
    uv run python evaluation/associative_recall/em_setup_notes_v3.py
    uv run python evaluation/associative_recall/em_setup_notes_v3.py --limit 5    # smoke
    uv run python evaluation/associative_recall/em_setup_notes_v3.py --conv locomo_conv-26
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

# v3: fresh collections so we never collide with v1 or v2 from prior agents.
COLLECTION_PREFIX = "arc_em_lc30_notes_v3"
LOGICAL_PREFIX = "arc_em_locomo30_notes_v3"
NAMESPACE = "arc_em_locomo30_notes_v3"

LOCOMO_CONV_IDS = ["locomo_conv-26", "locomo_conv-30", "locomo_conv-41"]

NOTES_MODEL = "gpt-5-mini"
NOTES_GEN_CACHE_FILE = CACHE_DIR / "notes_v3_gen_cache.json"
NOTES_SAMPLES_FILE = RESULTS_DIR / "notes_v3_samples.json"

EVENT_TYPE_MESSAGE = "message"
EVENT_TYPE_NOTE = "model_note"
NOTE_SPEAKER = "ModelNote"


# The prompt presents PRIOR NOTES / RECENT TURNS / RELATED EARLIER EVENTS as
# lists of EM-canonical strings ("<source>: <content>"). No JSON, no brackets.
NOTE_PROMPT = """\
You are the internal monologue of a listener attending to a conversation \
between {participant_1} and {participant_2}. After each new turn you \
update a short private note capturing your current understanding, open \
questions, and the realization this turn prompted.

Your notes will later be used as long-term memory. Each note is stored \
and retrieved in the same textual form as chat turns: each line is \
"<speaker>: <content>". Message lines look like \
"{participant_1}: ..." and earlier notes look like "ModelNote: ...". \
Write your note so that when it later appears as "ModelNote: <your text>" \
it reads fluently among real chat lines.

Be specific. Reference concrete facts, plans, emotions, commitments, \
unresolved issues. Interpret what the turn means, how it updates your \
model of the person, and what you are still unsure about. Hedge \
uncertainty linguistically ("seems", "probably", "not confirmed"). \
Do NOT quote the turn verbatim.

PRIOR NOTES (most recent last; these were already stored as ModelNote events):
{prior_notes_section}

RECENT CONVERSATION (most recent last; each line is a stored event):
{recent_turns_section}

EARLIER RELATED EVENTS (retrieved by semantic similarity; each line is a \
stored event, either a message or an earlier ModelNote):
{related_section}

NEW TURN (just ingested as an event):
{new_turn_line}

Emit your note as plain text with EXACTLY these three labeled lines, in \
this order:
current_understanding: <1-2 sentences on your updated model of the \
conversation / participants / situation right now>
open_questions: <0-3 short comma-separated items; write "none" if \
genuinely nothing is open>
recent_realization: <1-2 sentences on what THIS turn specifically \
revealed or changed>

Constraints:
- Total length well under 180 words.
- No bullet lists, no markdown, no JSON; just the three labeled lines.
- Use natural prose for each field; no telegraphic fragments.
- Output the three lines and NOTHING ELSE.
"""


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
    # Bulk-load each npz key ONCE into memory; indexing into the lazy
    # d["..."] proxy re-decompresses on each access, which is catastrophically
    # slow on the ~3.7 GB segments_extended.npz.
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
    """Canonical EM text form: "<source>: <content>".

    Optionally truncates content so we stay inside note-LLM token budget;
    truncation here is only for the note-generator prompt, not for ingest
    (ingest uses untouched content).
    """
    c = content.strip().replace("\n", " ")
    if truncate_to is not None and len(c) > truncate_to:
        c = c[:truncate_to] + "..."
    return f"{source}: {c}"


def _format_prior_notes(prior_notes_prose: list[str]) -> str:
    if not prior_notes_prose:
        return "(no prior notes yet)"
    # Each prior note was stored as "ModelNote: <prose>"; re-render that same way.
    lines = [em_format(NOTE_SPEAKER, p, truncate_to=None) for p in prior_notes_prose]
    return "\n".join(lines)


def _format_recent_turns(recent_turns: list[tuple[int, str, str]]) -> str:
    # recent_turns is a list of (turn_id, speaker_name, text); we drop turn_id
    # from the rendered form so it matches EM's canonical text exactly.
    if not recent_turns:
        return "(no prior turns yet)"
    return "\n".join(em_format(sp, tx) for _tid, sp, tx in recent_turns)


def _format_related(related: list[dict]) -> str:
    if not related:
        return "(no earlier related events retrieved yet)"
    return "\n".join(em_format(r["source"], r["text"]) for r in related)


# --------------------------------------------------------------------------
# Note parsing
# --------------------------------------------------------------------------


def _extract_note_fields(raw: str) -> dict:
    """Parse the three labeled lines. Tolerant of minor format drift."""
    out = {
        "current_understanding": "",
        "open_questions": "",
        "recent_realization": "",
    }
    if not raw:
        return out
    text = raw.strip()
    # Strip code fences if any.
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Simple line-by-line parse.
    current_field: str | None = None
    buckets: dict[str, list[str]] = {k: [] for k in out}
    for ln in text.splitlines():
        lo = ln.strip()
        if not lo:
            continue
        lower = lo.lower()
        for key in out:
            prefix = key + ":"
            if lower.startswith(prefix):
                current_field = key
                rest = lo[len(prefix) :].strip()
                if rest:
                    buckets[key].append(rest)
                break
        else:
            if current_field is not None:
                buckets[current_field].append(lo)
    for k in out:
        out[k] = " ".join(buckets[k]).strip()
    return out


def _note_prose(fields: dict) -> str:
    """Reconstruct the labeled prose that will be embedded.

    The embedded text becomes "ModelNote: current_understanding: ... \
    open_questions: ... recent_realization: ..."
    """
    parts: list[str] = []
    cu = fields.get("current_understanding", "").strip()
    oq = fields.get("open_questions", "").strip()
    rr = fields.get("recent_realization", "").strip()
    if cu:
        parts.append(f"current_understanding: {cu}")
    if oq:
        parts.append(f"open_questions: {oq}")
    if rr:
        parts.append(f"recent_realization: {rr}")
    return " ".join(parts).strip()


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
    """Top-K earlier events by cosine, excluding the just-ingested turn.

    Returns dicts with keys: source, text, event_type, turn_id.
    """
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
            # Exclude the current turn (turn_id + event_type=="message") — the note
            # has turn_id + 1us timestamp but we stored it with SAME turn_id in props.
            # To exclude only the current MESSAGE (not the current-turn note which
            # wouldn't exist yet anyway), we check event_type too.
            tid = int(props.get("turn_id", -1))
            etype = str(props.get("event_type", ""))
            if tid in exclude_turn_ids and etype == EVENT_TYPE_MESSAGE:
                break
            # Determine source: notes have speaker="ModelNote", messages have the real name.
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
    """V_combined: latest note + last 3 turns, joined with newlines,
    rendered in EM-canonical "<source>: <content>" form."""
    lines: list[str] = []
    if latest_note_prose:
        lines.append(em_format(NOTE_SPEAKER, latest_note_prose, truncate_to=None))
    for _tid, sp, tx in recent_turns[-3:]:
        lines.append(em_format(sp, tx, truncate_to=None))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Note generation
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
) -> tuple[str, dict, bool, list[dict]]:
    """Generate one note. Returns (prose, struct_fields, cache_hit, related_used)."""

    # V_combined similarity query: latest prior note + last 3 turns, EM-formatted.
    latest_note = prior_notes_prose[-1] if prior_notes_prose else None
    sim_query = _build_retrieval_query_v_combined(latest_note, recent_turns[:-1])
    related = await _retrieve_related(
        memory,
        sim_query,
        K=K_related,
        exclude_turn_ids={turn_id},
    )

    prompt = NOTE_PROMPT.format(
        participant_1=participants[0],
        participant_2=participants[1],
        prior_notes_section=_format_prior_notes(prior_notes_prose[-3:]),
        recent_turns_section=_format_recent_turns(recent_turns[-10:]),
        related_section=_format_related(related),
        new_turn_line=em_format(turn_speaker, turn_text, truncate_to=None),
    )

    cached = cache.get(NOTES_MODEL, prompt)
    cache_hit = cached is not None
    if cached is None:
        # reasoning_effort="low" keeps per-call latency ~2-3s while still
        # producing well-structured notes (minimal produced occasional format drift
        # during smoke testing).
        resp = await openai_client.chat.completions.create(
            model=NOTES_MODEL,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="low",
        )
        cached = resp.choices[0].message.content or ""
        cache.put(NOTES_MODEL, prompt, cached)

    fields = _extract_note_fields(cached)
    prose = _note_prose(fields)
    if not prose:
        # Degenerate fallback so embedding still works: use the first 400 chars.
        prose = cached.strip().replace("\n", " ")[:400]
    return prose, fields, cache_hit, related


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
        # Fresh slate every run — safe because prefix/namespace are new.
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
    n_note_cache_hits = 0

    if resume:
        # Scan existing partition to find what's already ingested. Use
        # direct SQLAlchemy access — the segment store doesn't expose a
        # high-level iterator. Rebuild prior_notes_prose (last 8 notes
        # by turn_id) and recent_turns (last 10 messages by turn_id).
        prior_by_tid: dict[int, str] = {}
        msgs_by_tid: dict[int, tuple[str, str]] = {}
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

        # Rebuild recent_turns (last 10 by turn_id ascending, trimmed).
        sorted_msg_tids = sorted(msgs_by_tid.keys())
        for tid in sorted_msg_tids[-10:]:
            sp, tx = msgs_by_tid[tid]
            recent_turns.append((tid, sp, tx))
        # Rebuild prior_notes_prose (last 8 by turn_id).
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

    # Determine early/mid/late indices for sampling.
    sample_idx = set()
    if n_total > 0:
        sample_idx = {0, n_total // 2, n_total - 1}

    for i, (turn_id, role, text) in enumerate(seg_iter):
        speaker_name = user_name if role == "user" else asst_name
        turn_ts = _turn_timestamp(base_ts, turn_id)
        t_iter = time.monotonic()

        # Skip turns already ingested (resume mode).
        if (
            resume
            and turn_id in already_ingested_turn_ids
            and turn_id in already_ingested_note_turn_ids
        ):
            continue

        if os.environ.get("NOTES_V3_VERBOSE"):
            print(
                f"[{conv_id}] turn {turn_id} ({speaker_name}) starting...", flush=True
            )

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

        if resume and turn_id in already_ingested_note_turn_ids:
            # Note already exists; leave prior_notes_prose as reconstructed.
            prose = ""
            fields = {}
            cache_hit = False
            related = []
        else:
            try:
                prose, fields, cache_hit, related = await generate_note(
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
                prose = ""
                fields = {}
                cache_hit = False
                related = []

            if cache_hit:
                n_note_cache_hits += 1

            if prose:
                note_ts = _note_timestamp(turn_ts)
                note_event = Event(
                    uuid=uuid4(),
                    timestamp=note_ts,
                    body=Content(
                        context=MessageContext(source=NOTE_SPEAKER),
                        items=[Text(text=prose)],
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
                prior_notes_prose.append(prose)
                prior_notes_prose = prior_notes_prose[-8:]

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
                    "note_fields": fields,
                    "note_prose": prose,
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
        if os.environ.get("NOTES_V3_VERBOSE"):
            print(
                f"[{conv_id}] turn {turn_id} done in {time.monotonic() - t_iter:.1f}s (n_notes={n_notes}, cache_hits={n_note_cache_hits})",
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
        "n_note_cache_hits": n_note_cache_hits,
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

    # Dedicated SQLite file for v3 — never touch existing eventmemory*.sqlite3.
    sqlite_path = RESULTS_DIR / "eventmemory_notes_v3.sqlite3"
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

    out_collections = RESULTS_DIR / "eventmemory_notes_v3_collections.json"
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
            f"cache_hits={r['n_note_cache_hits']} "
            f"in {r['ingest_time_s']}s -> {r['collection_name']}"
        )
    print(f"Saved: {out_collections}")
    print(f"Saved: {NOTES_SAMPLES_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
