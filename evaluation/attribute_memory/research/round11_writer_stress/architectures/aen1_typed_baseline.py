"""AEN-1 TYPED BASELINE — four typed relations.

Same as round10's indexed AEN-1: refs carry relation labels
(clarify/refine/supersede/invalidate). The writer picks the relation.

Used as the comparison baseline for the simplified single-ref-type architecture.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

HERE = Path(__file__).resolve().parent
ROUND11 = HERE.parent
ROUND7 = ROUND11.parent / "round7"
sys.path.insert(0, str(ROUND7 / "experiments"))

from _common import Budget, Cache, cosine, embed_batch, extract_json, llm  # noqa: E402

Relation = Literal["clarify", "refine", "supersede", "invalidate"]


@dataclass
class Ref:
    uuid: str
    relation: Relation


@dataclass
class LogEntry:
    uuid: str
    ts: int
    text: str
    mentions: list[str] = field(default_factory=list)
    refs: list[Ref] = field(default_factory=list)
    predicate: str | None = None


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

    superseded_by: dict[str, str] = {}
    for e in entries:
        for r in e.refs:
            if r.relation in ("supersede", "invalidate") and r.uuid in by_uuid:
                if r.uuid not in superseded_by:
                    superseded_by[r.uuid] = e.uuid

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


WRITE_PROMPT = """You are a semantic-memory writer using a SINGLE APPEND-ONLY LOG.

Each entry you write is an atomic natural-language fact. Use @Name tags to
mention entities. Use `refs` to point at prior entries, WITH a relation label.

RELATION LABELS (pick ONE per ref):
- supersede: this new entry REPLACES the prior value (old state no longer current)
- invalidate: the prior entry was WRONG — retract without replacement
- refine: narrows/qualifies the prior entry
- clarify: adds detail without contradicting

PREDICATE (optional but recommended for state-tracking facts):
  Format: "@Entity.predicate_name" — for example "@User.employer".

KNOWN ENTITIES so far: {known_entities}

PRIOR LOG SAMPLE:
{prior_log}

BATCH OF TURNS (process as a unit; emit 0+ entries):
{turn_block}

Emit JSON:
{{
  "entries": [
    {{
      "text": "<atomic fact>",
      "mentions": ["@Name", ...],
      "refs": [{{"uuid": "<prior-uuid>", "relation": "supersede|invalidate|refine|clarify"}}],
      "predicate": "@Entity.pred" or null
    }}
  ]
}}

RULES
- Use @Name for every named entity. ALWAYS @User for speaker facts.
- Filler turns (weather, chitchat, jokes) → skip; no entries.
- UPDATE of a state fact → relation=supersede with ref to prior head.
- CORRECTION/retraction → relation=invalidate (with supersede entry if a new
  value is given in the same turn; use invalidate alone for pure retractions).
- Additional detail that doesn't contradict → relation=clarify.
- Narrowing/qualifying the prior → relation=refine.
- For state-tracking facts, include predicate "@Entity.pred_name".
- Output JSON ONLY.
"""


def _render_prior_log(
    prior_entries: list[LogEntry],
    max_recent: int = 12,
    relevant_heads: list[LogEntry] | None = None,
) -> str:
    recent = list(reversed(prior_entries[-max_recent:]))
    rel_ids = set(e.uuid for e in recent)
    lines = []
    for e in recent:
        ref_str = ""
        if e.refs:
            ref_str = (
                " refs=[" + ",".join(f"{r.uuid}:{r.relation}" for r in e.refs) + "]"
            )
        pred_str = f" pred={e.predicate}" if e.predicate else ""
        lines.append(
            f"[{e.uuid}] t{e.ts} mentions={','.join(e.mentions)} :: {e.text}{ref_str}{pred_str}"
        )
    if relevant_heads:
        rel_block = []
        for e in relevant_heads:
            if e.uuid in rel_ids:
                continue
            ref_str = ""
            if e.refs:
                ref_str = (
                    " refs=[" + ",".join(f"{r.uuid}:{r.relation}" for r in e.refs) + "]"
                )
            pred_str = f" pred={e.predicate}" if e.predicate else ""
            rel_block.append(
                f"[{e.uuid}] t{e.ts} mentions={','.join(e.mentions)} :: {e.text}{ref_str}{pred_str}  (chain head)"
            )
        if rel_block:
            lines = rel_block + ["  --- recent batch context below ---"] + lines
    return "\n".join(lines) if lines else "(empty)"


def write_batch(
    batch_turns: list[tuple[int, str]],
    prior_entries: list[LogEntry],
    idx: IndexedLog | None,
    known_entities: set[str],
    cache: Cache,
    budget: Budget,
) -> list[LogEntry]:
    relevant: list[LogEntry] = []
    if idx is not None:
        batch_text = " ".join(t for _, t in batch_turns)
        tokens = re.findall(r"\b([A-Z][a-z]{1,20})\b", batch_text)
        ent_tags = {f"@{t}" for t in tokens} | {"@User"}
        seen = set()
        for (tag, pred), uuid in idx.supersede_head.items():
            if tag in ent_tags and uuid not in seen:
                seen.add(uuid)
                relevant.append(idx.by_uuid[uuid])
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
    last_turn = batch_turns[-1][0] if batch_turns else 0
    for i, e in enumerate(entries_raw):
        if not isinstance(e, dict):
            continue
        text = (e.get("text") or "").strip()
        if not text:
            continue
        mentions = [m for m in (e.get("mentions") or []) if isinstance(m, str)]
        refs_raw = e.get("refs") or []
        refs = []
        for r in refs_raw:
            if isinstance(r, dict) and r.get("uuid") and r.get("relation"):
                rel = r["relation"]
                if rel in ("clarify", "refine", "supersede", "invalidate"):
                    refs.append(Ref(uuid=r["uuid"], relation=rel))
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


def _rank_by_embedding(q_emb, candidate_uuids, embed_by_uuid, top_k):
    scored = []
    for u in candidate_uuids:
        v = embed_by_uuid.get(u)
        if v is None:
            continue
        scored.append((cosine(q_emb, v), u))
    scored.sort(reverse=True)
    return [u for _, u in scored[:top_k]]


def retrieve(
    question: str, idx: IndexedLog, cache: Cache, budget: Budget, top_k: int = 12
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
                    prev_ids = [
                        r.uuid
                        for r in e.refs
                        if r.relation in ("supersede", "invalidate")
                        and r.uuid in idx.by_uuid
                    ]
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
                if r.uuid in idx.by_uuid:
                    to_add.add(r.uuid)
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
semantic-memory log with typed refs. Entries use @Name and carry refs with
relations:
- supersede: prior entry is REPLACED by this one
- invalidate: prior was wrong; retract without replacement
- refine: narrows/qualifies
- clarify: adds detail without contradicting

Current state = most recent non-superseded, non-invalidated claim.
History = chronological order from root to head.
If a claim was invalidated, report "no" unambiguously.

RETRIEVED ENTRIES (chronological):
{entries_block}

QUESTION: {question}

Answer concisely. For yes/no, start with "Yes" or "No".
"""


def format_entries(entries: list[LogEntry]) -> str:
    lines = []
    for e in entries:
        ref_str = ""
        if e.refs:
            ref_str = (
                " refs=[" + ",".join(f"{r.uuid}:{r.relation}" for r in e.refs) + "]"
            )
        mentions = ",".join(e.mentions)
        lines.append(f"[{e.uuid}] t{e.ts} {mentions} :: {e.text}{ref_str}")
    return "\n".join(lines)


def answer_question(
    question: str, idx: IndexedLog, cache: Cache, budget: Budget, top_k: int = 12
) -> str:
    retrieved = retrieve(question, idx, cache, budget, top_k=top_k)
    block = format_entries(retrieved)
    prompt = READ_PROMPT.format(entries_block=block, question=question)
    return llm(prompt, cache, budget).strip()


def ingest_turns(
    turns: list[tuple[int, str]],
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
        if (i // batch_size) % rebuild_index_every == 0:
            idx = build_index(log, cache, budget)
    idx = build_index(log, cache, budget)
    return log, idx
