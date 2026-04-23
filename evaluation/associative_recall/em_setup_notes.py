"""Note-augmented EventMemory ingestion for LoCoMo-30.

After each ingested conversational turn, this pipeline asks a small LLM
(gpt-5-mini) to emit a short "internal monologue" note about the model's
current understanding, open questions, and recent realization. The note
is ingested as a SEPARATE EventMemory event at timestamp + 1 microsecond,
tagged `event_type="model_note"` (turns use `event_type="message"`).

New infrastructure (never touches the existing em_setup.py collections):
  Qdrant collections:  arc_em_lc30_notes_v1_<short conv id>
  Qdrant namespace:    arc_em_locomo30_notes
  SQLite segment store: results/eventmemory_notes.sqlite3
  LLM cache (note generation): cache/notes_gen_cache.json

Usage:
    uv run python evaluation/associative_recall/em_setup_notes.py
    # Smoke test (first N turns only per conversation):
    uv run python evaluation/associative_recall/em_setup_notes.py --limit 10

Dedicated cache means reruns are almost free once notes have been generated.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import openai
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

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


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

COLLECTION_PREFIX = "arc_em_lc30_notes_v1"
LOGICAL_PREFIX = "arc_em_locomo30_notes_v1"
NAMESPACE = "arc_em_locomo30_notes"

LOCOMO_CONV_IDS = ["locomo_conv-26", "locomo_conv-30", "locomo_conv-41"]

NOTES_MODEL = "gpt-5-mini"
NOTES_CACHE_FILE = CACHE_DIR / "notes_gen_cache.json"
NOTES_SAMPLES_FILE = RESULTS_DIR / "notes_samples.json"

# Event-type marker values.
EVENT_TYPE_MESSAGE = "message"
EVENT_TYPE_NOTE = "model_note"


NOTE_PROMPT = """\
You are the internal monologue of a listener who is attending to an \
ongoing conversation between {participant_1} and {participant_2}. After \
each new turn, you update a short private note summarizing your current \
understanding, open questions, and the realization prompted by this turn.

Your notes will later be used as memory to help answer complex questions \
about the participants. Be specific: reference concrete facts, plans, \
emotions, commitments, unresolved issues. Avoid vapid generalities like \
"{participant_1} said X"; instead interpret what it means, how it changes \
your model of the person, or what you are still unsure about.

PRIOR NOTES (most recent last, for continuity of thinking):
{prior_notes_section}

RECENT CONVERSATION (most recent last):
{recent_conv_section}

EARLIER RELATED EXCERPTS (retrieved by semantic similarity; may include \
earlier notes marked [NOTE]):
{related_section}

NEW TURN (just occurred):
[{speaker}]: {turn_text}

Emit a JSON object with EXACTLY these keys:
{{
  "current_understanding": "1-3 sentences summarizing your updated model \
of the conversation / participants / situation right now",
  "open_questions": ["what remains unclear", "what would you want to \
verify next time", "..."],
  "recent_realization": "1-2 sentences on what THIS turn specifically \
revealed or changed in your understanding"
}}

Constraints:
- Keep each field tight (no filler). Total length should be well under 220 words.
- open_questions is a list of 0-3 short phrases.
- Do NOT quote the turn verbatim; synthesize.
- Output JSON and nothing else."""


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def _sha(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()


class _NotesCache:
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
    out: dict[str, list[tuple[int, str, str]]] = {}
    for i in range(len(d["texts"])):
        cid = str(d["conversation_ids"][i])
        if cid not in LOCOMO_CONV_IDS:
            continue
        out.setdefault(cid, []).append(
            (int(d["turn_ids"][i]), str(d["roles"][i]), str(d["texts"][i]))
        )
    for cid in out:
        out[cid].sort(key=lambda t: t[0])
    return out


def _turn_timestamp(base: datetime, turn_id: int) -> datetime:
    # Same 60-second spacing as em_setup.py so conversational ordering matches.
    return base + timedelta(seconds=60 * turn_id)


def _note_timestamp(turn_ts: datetime) -> datetime:
    # Note goes 1 microsecond after the turn so it sorts immediately after.
    return turn_ts + timedelta(microseconds=1)


# --------------------------------------------------------------------------
# Note generation
# --------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        # Strip code fences.
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    m = _JSON_BLOCK_RE.search(t)
    if m is None:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _note_prose_from_struct(struct: dict) -> str:
    """Flatten the structured note to a prose string for embedding."""
    parts: list[str] = []
    cu = str(struct.get("current_understanding") or "").strip()
    if cu:
        parts.append(f"Current understanding: {cu}")
    oq = struct.get("open_questions") or []
    if isinstance(oq, list):
        oq_clean = [str(q).strip() for q in oq if str(q).strip()]
        if oq_clean:
            parts.append("Open questions: " + "; ".join(oq_clean))
    rr = str(struct.get("recent_realization") or "").strip()
    if rr:
        parts.append(f"Recent realization: {rr}")
    return " ".join(parts).strip()


def _format_prior_notes(prior: list[str]) -> str:
    if not prior:
        return "(no prior notes)"
    lines = []
    for i, n in enumerate(prior, 1):
        lines.append(f"  {i}. {n}")
    return "\n".join(lines)


def _format_recent_turns(recent: list[tuple[int, str, str]]) -> str:
    if not recent:
        return "(no prior turns)"
    lines = []
    for tid, speaker, text in recent:
        # Keep each line short to stay under token budgets.
        t = text.strip().replace("\n", " ")
        if len(t) > 220:
            t = t[:220] + "..."
        lines.append(f"  [turn {tid}] {speaker}: {t}")
    return "\n".join(lines)


def _format_related(related: list[dict]) -> str:
    if not related:
        return "(no earlier related excerpts available yet)"
    lines = []
    for r in related:
        label = "[NOTE]" if r.get("event_type") == EVENT_TYPE_NOTE else "[CHAT]"
        t = r.get("text", "").strip().replace("\n", " ")
        if len(t) > 220:
            t = t[:220] + "..."
        tid = r.get("turn_id", -1)
        spk = r.get("speaker", "")
        lines.append(f"  {label} [turn {tid}] {spk}: {t}")
    return "\n".join(lines)


async def _retrieve_related_context(
    memory: EventMemory,
    query_text: str,
    *,
    K: int = 5,
    exclude_turn_ids: set[int] | None = None,
) -> list[dict]:
    """Query the (partially-ingested) memory for similar earlier events."""
    exclude_turn_ids = exclude_turn_ids or set()
    try:
        qr = await memory.query(query=query_text, vector_search_limit=K * 2)
    except Exception:
        return []
    out: list[dict] = []
    seen_seg: set = set()
    for sc in qr.scored_segment_contexts:
        for seg in sc.segments:
            if sc.seed_segment_uuid in seen_seg:
                continue
            props = seg.properties or {}
            tid = int(props.get("turn_id", -1))
            if tid in exclude_turn_ids:
                continue
            out.append(
                {
                    "turn_id": tid,
                    "speaker": str(props.get("speaker", "")),
                    "event_type": str(props.get("event_type", "")),
                    "text": seg.block.text if hasattr(seg.block, "text") else "",
                }
            )
            seen_seg.add(sc.seed_segment_uuid)
            break
        if len(out) >= K:
            break
    return out[:K]


async def generate_note(
    *,
    openai_client,
    cache: _NotesCache,
    memory: EventMemory,
    conv_id: str,
    participants: tuple[str, str],
    turn_id: int,
    turn_speaker: str,
    turn_text: str,
    prior_notes_prose: list[str],
    recent_turns: list[tuple[int, str, str]],
) -> tuple[str, dict, bool]:
    """Generate one structured note for the current turn.

    Returns (prose_note, struct_dict, cache_hit).
    """
    # Retrieve earlier related events from the memory currently being built.
    # Exclude the CURRENT turn (just ingested) from results to avoid self-match.
    related = await _retrieve_related_context(
        memory,
        turn_text,
        K=5,
        exclude_turn_ids={turn_id},
    )

    prompt = NOTE_PROMPT.format(
        participant_1=participants[0],
        participant_2=participants[1],
        prior_notes_section=_format_prior_notes(prior_notes_prose[-3:]),
        recent_conv_section=_format_recent_turns(recent_turns[-10:]),
        related_section=_format_related(related),
        speaker=turn_speaker,
        turn_text=turn_text.strip(),
    )

    cached = cache.get(NOTES_MODEL, prompt)
    cache_hit = cached is not None
    if cached is None:
        resp = await openai_client.chat.completions.create(
            model=NOTES_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        cached = resp.choices[0].message.content or ""
        cache.put(NOTES_MODEL, prompt, cached)

    struct = _extract_json(cached) or {}
    prose = _note_prose_from_struct(struct)
    if not prose:
        # Fallback prose so embedding still works.
        prose = cached.strip().split("\n\n")[0][:500]
    return prose, struct, cache_hit


# --------------------------------------------------------------------------
# Ingestion
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
) -> dict:
    collection_name = f"{COLLECTION_PREFIX}_{_conv_short(conv_id)}"
    partition_key = collection_name

    # Start fresh every run (safe: NEW collections/partition keys).
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

    prior_notes_prose: list[str] = []  # last K notes (we slice last 3 when formatting)
    recent_turns: list[tuple[int, str, str]] = []  # last K turns (speaker_name, text)
    samples: list[dict] = []

    t0 = time.monotonic()
    n_turns = 0
    n_notes = 0
    n_note_cache_hits = 0

    # STRICT TEMPORAL ORDERING: ingest each (turn, note) synchronously in order
    # so that later notes can retrieve earlier (turns, notes) by similarity.
    for turn_id, role, text in seg_iter:
        speaker_name = user_name if role == "user" else asst_name

        turn_ts = _turn_timestamp(base_ts, turn_id)
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
        # Ingest the raw turn FIRST so note-generation can find it (via recent+related).
        await memory.encode_events([turn_event])
        n_turns += 1
        recent_turns.append((turn_id, speaker_name, text.strip()))
        recent_turns = recent_turns[-10:]

        # Generate the note.
        try:
            prose, struct, cache_hit = await generate_note(
                openai_client=openai_client,
                cache=notes_cache,
                memory=memory,
                conv_id=conv_id,
                participants=participants,
                turn_id=turn_id,
                turn_speaker=speaker_name,
                turn_text=text.strip(),
                prior_notes_prose=prior_notes_prose,
                recent_turns=recent_turns,
            )
        except Exception as exc:  # noqa: BLE001
            # Degrade gracefully: ingest a stub note so the pipeline continues.
            print(f"[note-gen] conv={conv_id} turn={turn_id} error={exc!r}")
            prose = ""
            struct = {}
            cache_hit = False

        if cache_hit:
            n_note_cache_hits += 1

        if prose:
            note_ts = _note_timestamp(turn_ts)
            # Note event: same substrate, MessageContext.source="ModelNote"
            # so embedded text becomes "ModelNote: <prose>".
            note_event = Event(
                uuid=uuid4(),
                timestamp=note_ts,
                body=Content(
                    context=MessageContext(source="ModelNote"),
                    items=[Text(text=prose)],
                ),
                properties={
                    "arc_conversation_id": conv_id,
                    # Reuse the same turn_id so notes remain addressable by turn.
                    "turn_id": turn_id,
                    "role": "model_note",
                    "speaker": "ModelNote",
                    "event_type": EVENT_TYPE_NOTE,
                },
            )
            await memory.encode_events([note_event])
            n_notes += 1
            prior_notes_prose.append(prose)
            prior_notes_prose = prior_notes_prose[-5:]  # only keep recent

        # Sample early/mid/late turns per conv for qualitative inspection.
        n_total_turns = len(segments) if limit is None else min(limit, len(segments))
        if n_total_turns > 0 and (
            turn_id in {segments[0][0],
                        segments[min(n_total_turns // 2, len(segments) - 1)][0],
                        segments[n_total_turns - 1][0]}
        ):
            samples.append(
                {
                    "conv_id": conv_id,
                    "turn_id": turn_id,
                    "speaker": speaker_name,
                    "turn_text": text.strip()[:400],
                    "note_struct": struct,
                    "note_prose": prose,
                }
            )

        # Periodically save cache to avoid losing work on interrupt.
        if (n_turns % 50) == 0:
            notes_cache.save()

    ingest_time = time.monotonic() - t0
    notes_cache.save()

    # Diagnostics.
    note_lengths = [len(p) for p in prior_notes_prose]  # only last 5 in buffer
    # Recompute avg note length from samples + last 5 prose in buffer is approximate
    # so instead we pull from cache prose if we want full stats; cheap approximation.

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
        help="Only ingest first N turns per conversation (smoke test)",
    )
    parser.add_argument(
        "--conv",
        default=None,
        help="Optional single conversation id (e.g. locomo_conv-26) to ingest",
    )
    parser.add_argument(
        "--concurrent_convs",
        type=int,
        default=1,
        help="Number of conversations to ingest concurrently (each conv is "
        "strictly sequential internally).",
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

    # Dedicated sqlite database so we never touch the existing eventmemory.sqlite3.
    sqlite_path = RESULTS_DIR / "eventmemory_notes.sqlite3"
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

    notes_cache = _NotesCache(NOTES_CACHE_FILE)

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

    # Save the collections manifest (for eval).
    out_collections = RESULTS_DIR / "eventmemory_notes_collections.json"
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
                "notes_cache_path": str(NOTES_CACHE_FILE),
                "conversations": [
                    {k: v for k, v in r.items() if k != "samples"} for r in records
                ],
            },
            f,
            indent=2,
        )

    # Save samples for qualitative inspection.
    all_samples = [s for r in records for s in r.get("samples", [])]
    with open(NOTES_SAMPLES_FILE, "w") as f:
        json.dump({"samples": all_samples}, f, indent=2)

    for r in records:
        print(
            f"[ingested] {r['conversation_id']}: "
            f"turns={r['n_turns_ingested']} notes={r['n_notes_ingested']} "
            f"note_cache_hits={r['n_note_cache_hits']} "
            f"in {r['ingest_time_s']}s -> {r['collection_name']}"
        )
    print(f"Saved: {out_collections}")
    print(f"Saved: {NOTES_SAMPLES_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
