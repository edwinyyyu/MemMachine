"""AEN-1 + structural indexes.

Data model (identical backing log to plain AEN-1, with deterministic indexes):
  - log: list[Entry]
  - mention_index: @entity -> list[uuid]
  - supersede_head: (@entity, predicate) -> uuid of current head
  - supersede_chain_head: uuid -> uuid (the uuid that superseded it, if any)
  - embed_by_uuid: uuid -> embedding

Retrieval paths depending on question kind (route by simple regex over the
question):
  - current-state: consult supersede_head directly, render the head entry + 2
    recent mention-index entries for context.
  - entity-profile: mention-filter by the entity + embedding top-K within
    that subset.
  - history: mention-filter + walk supersede chain from head to root.
  - default: embedding top-K ∪ mention-filter.

All indexes are maintained deterministically at write time — no LLM.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND10 = HERE.parent
ROUND7 = ROUND10.parent / "round7"
sys.path.insert(0, str(ROUND10 / "scenarios"))
sys.path.insert(0, str(ROUND7 / "experiments"))

from _common import Budget, Cache, cosine, embed_batch, llm  # noqa: E402
from generators import Entry  # noqa: E402

# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------


@dataclass
class IndexedLog:
    entries: list[Entry]
    by_uuid: dict[str, Entry]
    mention_index: dict[str, list[str]]  # @entity -> uuids
    # supersede_chain: entries with supersede ref form chains. We compute the
    # current "head" as the last (highest-ts) entry that has no successor.
    superseded_by: dict[str, str]  # uuid -> newer uuid
    # supersede head keyed by (entity_tag, predicate_label) for quick lookup.
    # Since entries have a predicate field, this is trivially deterministic when
    # we have it; for entries without a predicate we fall back to
    # (sorted(mentions), "text-hash") which is noisy — only current-state queries
    # are answered from this map.
    supersede_head: dict[tuple[str, str], str]  # (tag, pred) -> uuid
    embed_by_uuid: dict[str, list[float]]


def build_index(entries: list[Entry], cache: Cache, budget: Budget) -> IndexedLog:
    by_uuid = {e.uuid: e for e in entries}
    # mention_index
    mention_index: dict[str, list[str]] = {}
    for e in entries:
        for m in e.mentions:
            mention_index.setdefault(m, []).append(e.uuid)
    # superseded_by: walk refs
    superseded_by: dict[str, str] = {}
    for e in entries:
        for r in e.refs:
            if r.relation == "supersede" and r.uuid in by_uuid:
                superseded_by[r.uuid] = e.uuid
            elif r.relation == "invalidate" and r.uuid in by_uuid:
                # invalidate marks prior as no-longer-true; we track it as
                # "superseded to the invalidate record" so the head is the
                # invalidate record (which carries the retraction).
                superseded_by[r.uuid] = e.uuid
    # supersede_head: for each entry with a predicate, if it's NOT in superseded_by
    # it is a head.
    supersede_head: dict[tuple[str, str], str] = {}
    for e in entries:
        if e.predicate is None:
            continue
        if e.uuid in superseded_by:
            continue
        # predicate format: "@Entity.pred" -> (@Entity, pred)
        m = re.match(r"(@?[A-Za-z0-9_]+)\.(.+)", e.predicate)
        if not m:
            continue
        ent, pred = m.group(1), m.group(2)
        if not ent.startswith("@"):
            ent = "@" + ent
        supersede_head[(ent, pred)] = e.uuid

    # Embed everything
    texts = [e.text for e in entries]
    embs = embed_batch(texts, cache, budget)
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
# Retrieval
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
}


def extract_question_entities(question: str) -> list[str]:
    q = re.sub(r"[^a-zA-Z0-9\s']", " ", question)
    words = q.split()
    ents = []
    for w in words:
        # Strip possessive 's
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


def retrieve_indexed(
    question: str,
    idx: IndexedLog,
    cache: Cache,
    budget: Budget,
    top_k: int = 12,
) -> list[Entry]:
    # 1. Question embedding
    q_emb = embed_batch([question], cache, budget)[0]

    # 2. Entities mentioned in question
    q_ents = extract_question_entities(question)
    q_tags = [f"@{e}" for e in q_ents]

    kind = _detect_kind(question)
    selected_uuids: set[str] = set()

    # A. Current-state, supersede, and history: check supersede_head directly
    # for any (tag, pred) where tag is in q_tags.  Walk the full chain back
    # to root so history queries can enumerate.
    if kind in ("current", "supersede", "history", "entity"):
        for (tag, pred), uuid in idx.supersede_head.items():
            if tag in q_tags:
                selected_uuids.add(uuid)
                # walk back the chain (full transitive closure)
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
                    selected_uuids.add(cur)

    # B. Mention-filter for entities in question
    mention_candidates: list[str] = []
    for tag in q_tags:
        mention_candidates.extend(idx.mention_index.get(tag, []))
    # embedding-rank within mention candidates
    mention_topk = _rank_by_embedding(
        q_emb, list(set(mention_candidates)), idx.embed_by_uuid, top_k=top_k
    )
    selected_uuids.update(mention_topk)

    # C. For history/default: embedding top-K across WHOLE log
    if kind in ("history", "default", "entity"):
        all_uuids = [e.uuid for e in idx.entries]
        full_topk = _rank_by_embedding(q_emb, all_uuids, idx.embed_by_uuid, top_k=top_k)
        selected_uuids.update(full_topk)

    # D. Walk supersede chains for any selected entry — include both directions
    to_add = set()
    for u in list(selected_uuids):
        # forward: the entry that supersedes u
        if u in idx.superseded_by:
            to_add.add(idx.superseded_by[u])
        # backward: refs of u
        e = idx.by_uuid.get(u)
        if e:
            for r in e.refs:
                if r.uuid in idx.by_uuid:
                    to_add.add(r.uuid)
    selected_uuids.update(to_add)

    # Sort chronologically
    selected = sorted(selected_uuids, key=lambda u: idx.by_uuid[u].ts)
    # Cap total to 60 entries to stay within reader context
    MAX = 60
    if len(selected) > MAX:
        # keep top_k by embedding plus the supersede_head-selected ones
        must_keep = set()
        for (tag, pred), uuid in idx.supersede_head.items():
            if tag in q_tags:
                must_keep.add(uuid)
        # rank others by embedding
        others = [u for u in selected if u not in must_keep]
        ranked = _rank_by_embedding(
            q_emb, others, idx.embed_by_uuid, top_k=MAX - len(must_keep)
        )
        final = list(must_keep) + ranked
        selected = sorted(final, key=lambda u: idx.by_uuid[u].ts)

    return [idx.by_uuid[u] for u in selected]


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

READ_PROMPT = """You are answering a question about User's life using a
semantic-memory log with STRUCTURAL INDEXES.

Each entry is an atomic fact. Entries tagged with @Name mention entities.
The `refs` annotation shows supersede/invalidate/clarify relations to prior
entries:
- supersede: the prior entry is replaced by this one (old is no longer current)
- invalidate: the prior entry was wrong; retract without replacement
- clarify: adds detail without contradicting

When answering:
- Current state = most recent non-superseded, non-invalidated claim.
- History = chronological order from root of chain to head.
- If a claim was invalidated, answer "no" unambiguously.
- If the question asks about a past state, walk the chain backwards.

RETRIEVED ENTRIES (chronological):
{entries_block}

QUESTION: {question}

Answer concisely. For yes/no questions, start your answer with "Yes" or "No".
"""


def format_entries(entries: list[Entry]) -> str:
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


def answer_indexed(
    question: str,
    idx: IndexedLog,
    cache: Cache,
    budget: Budget,
    top_k: int = 12,
) -> str:
    retrieved = retrieve_indexed(question, idx, cache, budget, top_k=top_k)
    block = format_entries(retrieved)
    prompt = READ_PROMPT.format(entries_block=block, question=question)
    return llm(prompt, cache, budget).strip()


# ---------------------------------------------------------------------------
# Plain AEN-1 retrieval (for comparison) — embedding + mention + ref-chain,
# NO supersede_head.
# ---------------------------------------------------------------------------


def retrieve_plain(
    question: str,
    entries: list[Entry],
    embed_by_uuid: dict[str, list[float]],
    cache: Cache,
    budget: Budget,
    top_k: int = 12,
) -> list[Entry]:
    by_uuid = {e.uuid: e for e in entries}
    q_emb = embed_batch([question], cache, budget)[0]
    # embedding top-K
    all_uuids = [e.uuid for e in entries]
    topk = _rank_by_embedding(q_emb, all_uuids, embed_by_uuid, top_k=top_k)
    selected = set(topk)
    # mention-filter
    q_ents = extract_question_entities(question)
    q_tags = {f"@{e}" for e in q_ents}
    mention_cands = [e.uuid for e in entries if any(m in q_tags for m in e.mentions)]
    mention_topk = _rank_by_embedding(q_emb, mention_cands, embed_by_uuid, top_k=20)
    selected.update(mention_topk)
    # Walk supersede chains
    to_add = set()
    for u in list(selected):
        e = by_uuid[u]
        for r in e.refs:
            if r.uuid in by_uuid:
                to_add.add(r.uuid)
        # and chase forward: any entry that references u
        for other in entries:
            for r in other.refs:
                if r.uuid == u:
                    to_add.add(other.uuid)
                    break
    selected.update(to_add)
    # Cap
    MAX = 60
    if len(selected) > MAX:
        ranked = _rank_by_embedding(q_emb, list(selected), embed_by_uuid, MAX)
        selected = set(ranked)
    return sorted([by_uuid[u] for u in selected], key=lambda e: e.ts)


def answer_plain(
    question: str,
    entries: list[Entry],
    embed_by_uuid: dict[str, list[float]],
    cache: Cache,
    budget: Budget,
    top_k: int = 12,
) -> str:
    retrieved = retrieve_plain(
        question, entries, embed_by_uuid, cache, budget, top_k=top_k
    )
    block = format_entries(retrieved)
    prompt = READ_PROMPT.format(entries_block=block, question=question)
    return llm(prompt, cache, budget).strip()
