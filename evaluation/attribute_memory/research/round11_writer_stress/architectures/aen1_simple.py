"""AEN-1 SIMPLIFIED — single ref type.

Every cross-entry reference is just `ref(prior_entry_uuid)` — no
clarify/refine/supersede/invalidate distinction.

The structural index `supersede_head[(@entity, predicate)]` is maintained as:
"latest entry in the ref chain for that (entity, predicate) pair." Every new
entry that has refs AND matches a (@entity, predicate) updates the head pointer
to itself.

When the reader needs to answer "was X ever true?" or "was this fact ever
incorrect?", it walks the chain and reads prose. Prose carries the nuance.

Data model:
  LogEntry(uuid, ts, text, mentions: list[str], refs: list[str])   # no Ref type

Indexes (all structural, no LLM):
  mention_index:  @entity -> list[uuid]
  supersede_head: (@entity, predicate) -> uuid of current head
  superseded_by:  uuid -> uuid that refs it
  embed_by_uuid:  uuid -> embedding
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND11 = HERE.parent
ROUND10 = ROUND11.parent / "round10_scale"
ROUND7 = ROUND11.parent / "round7"
sys.path.insert(0, str(ROUND7 / "experiments"))

from _common import Budget, Cache, cosine, embed_batch, extract_json, llm  # noqa: E402

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class LogEntry:
    uuid: str
    ts: int
    text: str
    mentions: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)  # list of prior uuids
    # metadata carried from writer for indexing (optional)
    predicate: str | None = None  # "@Entity.pred" if the writer emitted one


@dataclass
class IndexedLog:
    entries: list[LogEntry]
    by_uuid: dict[str, LogEntry]
    mention_index: dict[str, list[str]]
    superseded_by: dict[str, str]
    supersede_head: dict[tuple[str, str], str]
    embed_by_uuid: dict[str, list[float]]


def build_index(entries: list[LogEntry], cache: Cache, budget: Budget) -> IndexedLog:
    by_uuid = {e.uuid: e for e in entries}

    mention_index: dict[str, list[str]] = {}
    for e in entries:
        for m in e.mentions:
            mention_index.setdefault(m, []).append(e.uuid)

    # In the SIMPLIFIED model, we treat every ref as a potential supersede.
    # If entry X has refs=[uuid_prev, ...], then uuid_prev is superseded by X
    # (i.e. X is its successor in whatever chain they're in). We pick the
    # FIRST ref as the "primary prior" for chain purposes.
    superseded_by: dict[str, str] = {}
    for e in entries:
        for r_uuid in e.refs:
            if r_uuid in by_uuid and r_uuid not in superseded_by:
                superseded_by[r_uuid] = e.uuid

    # supersede_head: for each entry with a predicate and no successor,
    # it's the head of its chain. Keyed by (@entity, predicate).
    supersede_head: dict[tuple[str, str], str] = {}
    for e in entries:
        if e.predicate is None:
            continue
        if e.uuid in superseded_by:
            continue
        m = re.match(r"(@?[A-Za-z0-9_]+)\.(.+)", e.predicate)
        if not m:
            continue
        ent, pred = m.group(1), m.group(2)
        if not ent.startswith("@"):
            ent = "@" + ent
        supersede_head[(ent, pred)] = e.uuid

    texts = [e.text for e in entries]
    embs = embed_batch(texts, cache, budget) if entries else []
    embed_by_uuid = {e.uuid: embs[i] for i, e in enumerate(entries)}

    return IndexedLog(
        entries=entries,
        by_uuid=by_uuid,
        mention_index=mention_index,
        superseded_by=superseded_by,
        supersede_head=supersede_head,
        embed_by_uuid=embed_by_uuid,
    )


# ---------------------------------------------------------------------------
# Writer (single ref type)
# ---------------------------------------------------------------------------

WRITE_PROMPT = """You are a semantic-memory writer using a SINGLE APPEND-ONLY LOG.

Each entry you write is an atomic natural-language fact. Use @Name tags to
mention entities. Use `refs` to point at prior entries when this entry relates
to, updates, or corrects a prior fact.

REFS: a list of prior entry UUIDs — there is only ONE kind of ref. If this
entry updates/corrects/refines/clarifies a prior fact, include that prior entry
in `refs`. The prose text carries the nuance (replacement, retraction, detail,
etc.) — do NOT choose a relation label.

PREDICATE (optional but recommended for state-tracking facts):
  Format: "@Entity.predicate_name" — for example "@User.employer",
  "@Jamie.job", "@User.location". Omit for casual or multi-subject facts.

KNOWN ENTITIES so far: {known_entities}

PRIOR LOG SAMPLE (most recent + a few relevant older entries; cite these by
their uuid in `refs`):
{prior_log}

BATCH OF TURNS (process as a unit; emit 0+ entries covering the whole batch):
{turn_block}

Emit JSON:
{{
  "entries": [
    {{
      "text": "<atomic fact in one sentence, prose carries correction/supersede/nuance>",
      "mentions": ["@Name", ...],
      "refs": ["<prior-uuid>", ...],
      "predicate": "@Entity.pred" or null
    }}
  ]
}}

RULES
- Use @Name for every named entity (User, Jamie, Marcus, Luna, etc.). ALWAYS
  @User when the fact is about the speaker.
- If a turn is pure filler (weather, chitchat, jokes, noop), do NOT emit an
  entry for it. Skip silently.
- If a turn UPDATES or CORRECTS a prior fact (e.g. boss changed, moved cities,
  job changed, user says "actually X was wrong"), emit a new entry with
  `refs` pointing at the prior entry that stated the now-outdated value. Make
  the prose carry the supersede/correction nuance ("X is now ...", "Actually,
  X was wrong — it's really Y", etc.).
- For state-tracking facts (job, location, boss, employer, role, relationship,
  residence, partner, etc.), include the `predicate` in the form
  "@Entity.pred_name" — use stable lowercase predicate names.
- If the turn adds pure detail (clarification) about a prior fact without
  contradicting it, include the prior uuid in `refs` but the prose can just
  state the new detail.
- Prefer ONE entry per turn; emit multiple only if the turn bundles unrelated
  facts. If the batch has no memory-worthy content, output {{"entries": []}}.
- Do NOT invent entities. Do NOT add @User to facts where User isn't mentioned
  or implied.
- Output JSON ONLY.
"""


def _render_prior_log(
    prior_entries: list[LogEntry],
    max_recent: int = 12,
    relevant_heads: list[LogEntry] | None = None,
) -> str:
    recent = list(reversed(prior_entries[-max_recent:]))
    rel_ids = set()
    lines = []
    for e in recent:
        ref_str = f" refs=[{','.join(e.refs)}]" if e.refs else ""
        pred_str = f" pred={e.predicate}" if e.predicate else ""
        lines.append(
            f"[{e.uuid}] t{e.ts} mentions={','.join(e.mentions)} :: {e.text}{ref_str}{pred_str}"
        )
        rel_ids.add(e.uuid)
    if relevant_heads:
        rel_block = []
        for e in relevant_heads:
            if e.uuid in rel_ids:
                continue
            ref_str = f" refs=[{','.join(e.refs)}]" if e.refs else ""
            pred_str = f" pred={e.predicate}" if e.predicate else ""
            rel_block.append(
                f"[{e.uuid}] t{e.ts} mentions={','.join(e.mentions)} :: {e.text}{ref_str}{pred_str}  (chain head)"
            )
        if rel_block:
            lines = rel_block + ["  --- recent batch context below ---"] + lines
    return "\n".join(lines) if lines else "(empty)"


def write_batch(
    batch_turns: list[tuple[int, str]],  # [(turn_idx, text), ...]
    prior_entries: list[LogEntry],
    idx: IndexedLog | None,
    known_entities: set[str],
    cache: Cache,
    budget: Budget,
) -> list[LogEntry]:
    # Pick relevant prior-chain-heads by mention match for context:
    relevant: list[LogEntry] = []
    if idx is not None:
        batch_text = " ".join(t for _, t in batch_turns)
        # Very simple: any @Name in batch_text by string search
        tokens = re.findall(r"\b([A-Z][a-z]{1,20})\b", batch_text)
        ent_tags = {f"@{t}" for t in tokens} | {"@User"}
        seen = set()
        for (tag, pred), uuid in idx.supersede_head.items():
            if tag in ent_tags and uuid not in seen:
                seen.add(uuid)
                relevant.append(idx.by_uuid[uuid])
        # cap relevant to 8 most-recent
        relevant.sort(key=lambda e: e.ts, reverse=True)
        relevant = relevant[:8]

    prior_log = _render_prior_log(prior_entries, relevant_heads=relevant)
    turn_block = "\n".join(f"TURN {i}: {t}" for i, t in batch_turns)
    prompt = WRITE_PROMPT.format(
        known_entities=", ".join(sorted(known_entities))
        if known_entities
        else "(none)",
        prior_log=prior_log,
        turn_block=turn_block,
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return []
    entries_raw = obj.get("entries", []) or []
    entries: list[LogEntry] = []
    # Assign uuids based on the last turn_idx in the batch
    last_turn = batch_turns[-1][0] if batch_turns else 0
    for i, e in enumerate(entries_raw):
        if not isinstance(e, dict):
            continue
        text = (e.get("text") or "").strip()
        if not text:
            continue
        mentions = [m for m in (e.get("mentions") or []) if isinstance(m, str)]
        refs_raw = e.get("refs") or []
        refs = [r for r in refs_raw if isinstance(r, str)]
        predicate = e.get("predicate")
        if predicate is not None and not isinstance(predicate, str):
            predicate = None
        uuid = f"e{last_turn:04d}_{i}"
        entries.append(
            LogEntry(
                uuid=uuid,
                ts=last_turn,
                text=text,
                mentions=mentions,
                refs=refs,
                predicate=predicate,
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Retrieval (same as typed, but treats all refs as chain-links)
# ---------------------------------------------------------------------------

STOP = {
    "What",
    "Where",
    "When",
    "Who",
    "How",
    "Why",
    "Which",
    "Has",
    "Have",
    "Do",
    "Does",
    "Is",
    "Are",
    "Was",
    "Were",
    "Tell",
    "Before",
    "The",
    "A",
    "An",
    "And",
    "Or",
    "But",
    "I",
    "Know",
    "Me",
    "My",
    "Him",
    "Her",
    "Them",
    "They",
    "She",
    "He",
    "Currently",
    "Right",
    "Full",
    "Of",
    "User",
    "Users",
    "List",
    "Did",
    "Current",
    "It",
    "That",
    "This",
    "In",
    "On",
    "At",
    "About",
}


def extract_question_entities(question: str) -> list[str]:
    q = re.sub(r"[^a-zA-Z0-9\s']", " ", question)
    words = q.split()
    ents = []
    for w in words:
        if w.endswith("'s"):
            w = w[:-2]
        elif w.endswith("'"):
            w = w[:-1]
        if len(w) > 1 and w[0].isupper() and w not in STOP:
            ents.append(w)
    return ents + ["User"]


def _detect_kind(question: str) -> str:
    q = question.lower()
    if "history" in q or "in order" in q or "chronological" in q or "list" in q:
        return "history"
    if "ever" in q or "still" in q or "was" in q.split() or "did " in q:
        return "supersede"
    if (
        "current" in q
        or "now" in q
        or "who is" in q
        or "where does" in q
        or "where do" in q
        or "what is" in q
    ):
        return "current"
    if "tell me" in q or "everything" in q:
        return "entity"
    return "default"


def _rank_by_embedding(
    q_emb: list[float],
    candidate_uuids: list[str],
    embed_by_uuid: dict[str, list[float]],
    top_k: int,
) -> list[str]:
    scored = []
    for u in candidate_uuids:
        v = embed_by_uuid.get(u)
        if v is None:
            continue
        scored.append((cosine(q_emb, v), u))
    scored.sort(reverse=True)
    return [u for _, u in scored[:top_k]]


def retrieve(
    question: str,
    idx: IndexedLog,
    cache: Cache,
    budget: Budget,
    top_k: int = 12,
) -> list[LogEntry]:
    if not idx.entries:
        return []
    q_emb = embed_batch([question], cache, budget)[0]
    q_ents = extract_question_entities(question)
    q_tags = [f"@{e}" for e in q_ents]
    kind = _detect_kind(question)
    selected: set[str] = set()

    if kind in ("current", "supersede", "history", "entity"):
        for (tag, pred), uuid in idx.supersede_head.items():
            if tag in q_tags:
                selected.add(uuid)
                cur = uuid
                seen = {cur}
                for _ in range(50):
                    e = idx.by_uuid[cur]
                    prev_ids = [r for r in e.refs if r in idx.by_uuid]
                    if not prev_ids:
                        break
                    cur = prev_ids[0]
                    if cur in seen:
                        break
                    seen.add(cur)
                    selected.add(cur)

    mention_candidates: list[str] = []
    for tag in q_tags:
        mention_candidates.extend(idx.mention_index.get(tag, []))
    mention_topk = _rank_by_embedding(
        q_emb, list(set(mention_candidates)), idx.embed_by_uuid, top_k=top_k
    )
    selected.update(mention_topk)

    if kind in ("history", "default", "entity"):
        all_uuids = [e.uuid for e in idx.entries]
        full_topk = _rank_by_embedding(q_emb, all_uuids, idx.embed_by_uuid, top_k=top_k)
        selected.update(full_topk)

    to_add = set()
    for u in list(selected):
        if u in idx.superseded_by:
            to_add.add(idx.superseded_by[u])
        e = idx.by_uuid.get(u)
        if e:
            for r in e.refs:
                if r in idx.by_uuid:
                    to_add.add(r)
    selected.update(to_add)

    out = sorted(selected, key=lambda u: idx.by_uuid[u].ts)
    MAX = 60
    if len(out) > MAX:
        must_keep = set()
        for (tag, pred), uuid in idx.supersede_head.items():
            if tag in q_tags:
                must_keep.add(uuid)
        others = [u for u in out if u not in must_keep]
        ranked = _rank_by_embedding(
            q_emb, others, idx.embed_by_uuid, top_k=MAX - len(must_keep)
        )
        out = sorted(list(must_keep) + ranked, key=lambda u: idx.by_uuid[u].ts)
    return [idx.by_uuid[u] for u in out]


READ_PROMPT = """You are answering a question about User's life using a
semantic-memory log. Each entry is an atomic natural-language fact. Entries
tagged with @Name mention entities. The `refs` annotation shows PRIOR entries
this entry relates to (typically because this entry updates, corrects, or
refines the prior one). There is only one kind of ref — read the prose of both
the referring entry and the prior to understand the nature of the relation.

When answering:
- CURRENT STATE: the most recent entry in a ref-chain is the current value
  (unless the prose of that or a later entry says the fact was retracted).
- HISTORY: walk the chain from the earliest root to the head.
- If a claim was explicitly retracted/invalidated (prose says "was wrong",
  "scratch that", "actually never"), report accordingly.

RETRIEVED ENTRIES (chronological):
{entries_block}

QUESTION: {question}

Answer concisely. For yes/no questions, start your answer with "Yes" or "No".
"""


def format_entries(entries: list[LogEntry]) -> str:
    lines = []
    for e in entries:
        ref_str = f" refs=[{','.join(e.refs)}]" if e.refs else ""
        mentions = ",".join(e.mentions)
        lines.append(f"[{e.uuid}] t{e.ts} {mentions} :: {e.text}{ref_str}")
    return "\n".join(lines)


def answer_question(
    question: str,
    idx: IndexedLog,
    cache: Cache,
    budget: Budget,
    top_k: int = 12,
) -> str:
    retrieved = retrieve(question, idx, cache, budget, top_k=top_k)
    block = format_entries(retrieved)
    prompt = READ_PROMPT.format(entries_block=block, question=question)
    return llm(prompt, cache, budget).strip()


# ---------------------------------------------------------------------------
# End-to-end ingestion
# ---------------------------------------------------------------------------


def ingest_turns(
    turns: list[tuple[int, str]],  # (turn_idx, text)
    cache: Cache,
    budget: Budget,
    batch_size: int = 5,
    rebuild_index_every: int = 25,
) -> tuple[list[LogEntry], IndexedLog]:
    log: list[LogEntry] = []
    known: set[str] = {"User"}
    idx: IndexedLog | None = None

    for i in range(0, len(turns), batch_size):
        batch = turns[i : i + batch_size]
        new_entries = write_batch(batch, log, idx, known, cache, budget)
        for e in new_entries:
            for m in e.mentions:
                if m.startswith("@"):
                    known.add(m[1:])
        log.extend(new_entries)
        # Incrementally rebuild idx every N batches (cheap, purely local)
        if (i // batch_size) % rebuild_index_every == 0:
            idx = build_index(log, cache, budget)
    # Final build
    idx = build_index(log, cache, budget)
    return log, idx
