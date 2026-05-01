"""AEN-1: Append-Entity-mention Network, single log.

Data model:
  LogEntry(uuid, ts, text, mentions: list[str], refs: list[(uuid, relation)])
  relation ∈ {clarify, refine, supersede, invalidate}

Storage: ONE append-only log for the entire memory. Entries reference each other
by uuid + relation.

Writer: single LLM call per turn, emits one or more entries:
  {
    "entries": [
      {"text": "...", "mentions": ["@Marcus", "@User"],
       "refs": [{"uuid": "<prior-uuid>", "relation": "supersede"}]}
    ]
  }

Reader: three complementary retrievals are merged and shown to the reader LLM:
  1. Embedding top-K of the question against entry texts
  2. Mention-filter: all entries that mention an entity extracted from the question
  3. Supersede-chain traversal: for each retrieved entry, walk its `refs` to
     find the chain up to root, including invalidate/supersede-nodes
The reader LLM sees entries in chronological order with relation annotations.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

HERE = Path(__file__).resolve().parent
ROUND9 = HERE.parent
ROUND7 = ROUND9.parent / "round7"
sys.path.insert(0, str(ROUND7 / "experiments"))

from _common import Budget, Cache, cosine, embed_batch, extract_json, llm  # noqa: E402

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

Relation = Literal["clarify", "refine", "supersede", "invalidate"]


@dataclass
class Ref:
    uuid: str
    relation: Relation


@dataclass
class LogEntry:
    uuid: str
    ts: int  # turn idx (we use turn idx as timestamp)
    text: str
    mentions: list[str] = field(default_factory=list)  # ["@User", "@Marcus"]
    refs: list[Ref] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

WRITE_PROMPT = """You are a semantic-memory writer using a SINGLE APPEND-ONLY LOG.

Each entry you write is atomic text about the world. Use @Name tags to mention
entities. Use refs to point at prior entries when THIS entry relates to them.

RELATIONS (for `refs`):
- clarify: adds detail to a prior entry without contradicting it
- refine: narrows or qualifies a prior entry
- supersede: replaces a prior entry with a new version (old stays in log, marked)
- invalidate: prior entry is wrong; retracts without replacement

KNOWN ENTITIES: {known_entities}

PRIOR LOG (newest first, show only last 15 entries for context):
{prior_log}

TURN: "{turn_text}"

Emit JSON:
{{
  "entries": [
    {{
      "text": "<atomic fact in one sentence>",
      "mentions": ["@Name", ...],
      "refs": [{{"uuid": "<prior-uuid>", "relation": "supersede|clarify|refine|invalidate"}}]
    }}
  ]
}}

RULES
- Use @Name for every named entity in the fact (User, Jamie, Marcus, Luna, etc.). ALWAYS @User when the fact is about the speaker.
- If turn is pure filler (weather, small chitchat, jokes), emit "entries": [].
- If the turn CORRECTS a prior claim, write a new entry with refs pointing at every prior entry it contradicts, with relation "invalidate" (if wrong) or "supersede" (if replaced by new value).
- If the turn REFINES a prior claim (more precise), use "refine".
- If the turn adds detail without replacing, use "clarify".
- Prefer ONE entry per turn; emit multiple only if the turn bundles multiple unrelated facts.
- Do NOT invent entities that aren't in the turn text.
- Output JSON only.
"""


def write_entries(
    turn_idx: int,
    turn_text: str,
    prior_entries: list[LogEntry],
    known_entities: set[str],
    cache: Cache,
    budget: Budget,
) -> list[LogEntry]:
    # Render prior log (newest first, last 15)
    recent = list(reversed(prior_entries[-15:]))
    lines = []
    for e in recent:
        ref_str = ""
        if e.refs:
            ref_str = " refs=" + ",".join(f"{r.uuid}:{r.relation}" for r in e.refs)
        lines.append(
            f"[{e.uuid}] t{e.ts} mentions={','.join(e.mentions)} {e.text}{ref_str}"
        )
    prior_block = "\n".join(lines) if lines else "(empty)"

    prompt = WRITE_PROMPT.format(
        known_entities=", ".join(sorted(known_entities))
        if known_entities
        else "(none)",
        prior_log=prior_block,
        turn_text=turn_text,
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return []
    entries_raw = obj.get("entries", []) or []
    entries: list[LogEntry] = []
    for i, e in enumerate(entries_raw):
        if not isinstance(e, dict):
            continue
        uuid = f"e{turn_idx:03d}_{i}"
        text = e.get("text", "").strip()
        if not text:
            continue
        mentions = e.get("mentions") or []
        mentions = [m for m in mentions if isinstance(m, str)]
        refs_raw = e.get("refs") or []
        refs: list[Ref] = []
        for r in refs_raw:
            if isinstance(r, dict) and r.get("uuid") and r.get("relation"):
                rel = r["relation"]
                if rel in ("clarify", "refine", "supersede", "invalidate"):
                    refs.append(Ref(uuid=r["uuid"], relation=rel))
        entries.append(
            LogEntry(uuid=uuid, ts=turn_idx, text=text, mentions=mentions, refs=refs)
        )
    return entries


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

READ_PROMPT = """You are answering a question about User's life using a
semantic-memory log. Each entry is atomic and mentions entities with @Name.
The refs annotation on an entry shows what prior entries it relates to:
- clarify: adds detail
- refine: narrows/qualifies
- supersede: replaces (old is no longer the current state)
- invalidate: prior entry was wrong

When answering, consider the most recent non-superseded, non-invalidated claims.
Be precise. If the answer is "the user was never X because that claim was
invalidated", say so.

RETRIEVED ENTRIES (chronological order; ref-chain included for each retrieved
entry):

{entries_block}

QUESTION: {question}

Answer concisely.
"""


def extract_question_entities(question: str) -> list[str]:
    """Very simple entity extraction for mention-filtering.

    We look for capitalized words likely to be entity names. This is a cheap
    pre-step — not LLM. Still adequate for our test set where entities are
    named plainly.
    """
    # Known entity list: we'll use any @Name the writer has emitted, but for
    # question-side we just parse capitalized words.
    import re

    # Strip punctuation
    q = re.sub(r"[^a-zA-Z0-9\s']", " ", question)
    words = q.split()
    # Common English stopwords that start with capital (start of sentence)
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
    }
    # User is always relevant but not a name to filter by extra.
    ents = []
    for w in words:
        if len(w) > 1 and w[0].isupper() and w not in STOP:
            ents.append(w)
    # Always include User
    return ents + ["User"]


def retrieve(
    question: str,
    log: list[LogEntry],
    cache: Cache,
    budget: Budget,
    top_k: int = 10,
) -> list[LogEntry]:
    """Combined retrieval: embedding top-K ∪ mention-filter ∪ supersede chain."""
    if not log:
        return []
    # 1. Embedding top-K
    texts = [e.text for e in log]
    embs = embed_batch(texts + [question], cache, budget)
    q_emb = embs[-1]
    entry_embs = embs[:-1]
    scores = [cosine(q_emb, e) for e in entry_embs]
    ranked_idx = sorted(range(len(log)), key=lambda i: scores[i], reverse=True)
    top_k_set = set(ranked_idx[:top_k])

    # 2. Mention-filter: any entry whose mentions include a question entity
    q_ents = extract_question_entities(question)
    q_ent_set = {f"@{e}" for e in q_ents}
    mention_idx = set()
    for i, e in enumerate(log):
        if any(m in q_ent_set for m in e.mentions):
            mention_idx.add(i)
    # Limit mention-filter to top 20 by embedding score among matches to avoid blowup
    mention_sorted = sorted(mention_idx, key=lambda i: scores[i], reverse=True)
    mention_set = set(mention_sorted[:20])

    # 3. Supersede/invalidate chain traversal
    # For any retrieved entry, follow refs backward AND find entries that ref it.
    selected = top_k_set | mention_set
    uuid_to_idx = {e.uuid: i for i, e in enumerate(log)}
    to_add = set()
    for i in selected:
        # Walk refs pointing out (older entries)
        for r in log[i].refs:
            if r.uuid in uuid_to_idx:
                to_add.add(uuid_to_idx[r.uuid])
        # Walk refs pointing IN (newer entries that ref this one)
        for j, e in enumerate(log):
            for r in e.refs:
                if r.uuid == log[i].uuid:
                    to_add.add(j)
    selected |= to_add

    return [log[i] for i in sorted(selected)]


def format_entries_for_reader(entries: list[LogEntry]) -> str:
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
    question: str,
    log: list[LogEntry],
    cache: Cache,
    budget: Budget,
    top_k: int = 10,
) -> str:
    retrieved = retrieve(question, log, cache, budget, top_k=top_k)
    block = format_entries_for_reader(retrieved)
    prompt = READ_PROMPT.format(entries_block=block, question=question)
    return llm(prompt, cache, budget).strip()


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def ingest_scenario(
    turns,
    cache: Cache,
    budget: Budget,
) -> list[LogEntry]:
    log: list[LogEntry] = []
    known: set[str] = {"User"}
    for turn in turns:
        new_entries = write_entries(turn.idx, turn.text, log, known, cache, budget)
        for e in new_entries:
            # harvest @Mentions into known
            for m in e.mentions:
                if m.startswith("@"):
                    known.add(m[1:])
        log.extend(new_entries)
    return log
