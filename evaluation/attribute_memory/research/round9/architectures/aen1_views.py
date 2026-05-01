"""AEN-1 + materialized per-entity views.

Backing store: AEN-1 single log (same as aen1.py).
Layered on top: deterministic per-entity view derivation.

View rule:
  For each @entity, the "view" is the chronologically-ordered set of log entries
  that mention @entity, MINUS entries that are invalidated by a supersede/invalidate
  ref from a later entry.

Unlike the partitioned architecture, this view is derived from the log at read
time (no extra LLM write calls). We pre-compute it once when ingestion is done.
The view is a DENSE per-entity timeline that the reader LLM sees on any
entity-centric or supersede-chain question.

The read path:
  1. Identify target entities from question (as in AEN-1)
  2. Pull per-entity view for each target (all entries mentioning the entity,
     filtered for invalidation)
  3. ADDITIONALLY: embedding top-K across whole log for facts that don't name
     the entity directly (e.g. "Priya's team" -> entries about teams)
  4. Reader LLM over union
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND9 = HERE.parent
ROUND7 = ROUND9.parent / "round7"
sys.path.insert(0, str(ROUND7 / "experiments"))

from _common import Budget, Cache, cosine, embed_batch, llm  # noqa: E402
from aen1 import (  # noqa: E402
    LogEntry,
    extract_question_entities,
)
from aen1 import (
    ingest_scenario as aen1_ingest,
)

# ---------------------------------------------------------------------------
# View derivation
# ---------------------------------------------------------------------------


def build_views(log: list[LogEntry]) -> dict[str, list[LogEntry]]:
    """For each @entity mentioned anywhere, produce the chronological list of
    entries mentioning it, with an `invalidated` annotation.

    We do NOT drop invalidated entries — we keep them with a marker, so the
    reader can still see "what was X before" if asked.
    """
    # Build uuid -> idx
    uuid_to_idx = {e.uuid: i for i, e in enumerate(log)}
    # For each entry, flag if any LATER entry has a ref pointing at it with
    # relation=invalidate.
    invalidated_by: dict[str, list[str]] = {}
    superseded_by: dict[str, list[str]] = {}
    for e in log:
        for r in e.refs:
            if r.uuid not in uuid_to_idx:
                continue
            if r.relation == "invalidate":
                invalidated_by.setdefault(r.uuid, []).append(e.uuid)
            elif r.relation == "supersede":
                superseded_by.setdefault(r.uuid, []).append(e.uuid)

    # Views per entity name
    views: dict[str, list[LogEntry]] = {}
    for e in log:
        for m in e.mentions:
            if not m.startswith("@"):
                continue
            name = m[1:]
            views.setdefault(name, []).append(e)
    return views, invalidated_by, superseded_by


def render_view(
    name: str,
    entries: list[LogEntry],
    invalidated_by: dict[str, list[str]],
    superseded_by: dict[str, list[str]],
) -> str:
    """Chronological view for one entity, with annotations."""
    lines = [f"--- View: @{name} ---"]
    for e in entries:
        ann = []
        if e.uuid in invalidated_by:
            ann.append(f"INVALIDATED by {','.join(invalidated_by[e.uuid])}")
        if e.uuid in superseded_by:
            ann.append(f"SUPERSEDED by {','.join(superseded_by[e.uuid])}")
        if e.refs:
            ann.append(
                "refs=[" + ",".join(f"{r.uuid}:{r.relation}" for r in e.refs) + "]"
            )
        annotation = (" [" + " | ".join(ann) + "]") if ann else ""
        lines.append(f"[{e.uuid}] t{e.ts} :: {e.text}{annotation}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reader with views
# ---------------------------------------------------------------------------

VIEWS_READ_PROMPT = """You are answering a question about User's life using a
semantic-memory log with DERIVED per-entity views.

Each view is the chronological sequence of entries that mention a given entity
(@Name). Entries are annotated with:
- INVALIDATED by <uuid(s)>: the fact was retracted by a later entry (treat as NOT true)
- SUPERSEDED by <uuid(s)>: replaced by a later version (treat as past-state)
- refs=[...]: other entries this one relates to

When answering:
- Current state = most recent non-INVALIDATED, non-SUPERSEDED claim
- History = chronological list, including superseded items marked as past
- If everything about a topic was invalidated, say so

PER-ENTITY VIEWS:
{views_block}

ADDITIONAL NEIGHBORHOOD (embedding-retrieved, may or may not be entity-tagged):
{extra_block}

QUESTION: {question}

Answer concisely.
"""


def answer_question_with_views(
    question: str,
    log: list[LogEntry],
    views: dict[str, list[LogEntry]],
    invalidated_by: dict[str, list[str]],
    superseded_by: dict[str, list[str]],
    cache: Cache,
    budget: Budget,
    top_k_extra: int = 6,
) -> str:
    q_ents = extract_question_entities(question)
    # Select views for entities in question
    view_names = []
    for e in q_ents:
        if e in views and e not in view_names:
            view_names.append(e)
    # Always include User
    if "User" in views and "User" not in view_names:
        view_names.insert(0, "User")

    views_block_parts = []
    included_uuids: set[str] = set()
    for name in view_names:
        ents = views.get(name, [])
        views_block_parts.append(render_view(name, ents, invalidated_by, superseded_by))
        for e in ents:
            included_uuids.add(e.uuid)
    views_block = (
        "\n\n".join(views_block_parts) if views_block_parts else "(no matching views)"
    )

    # Extra retrieval: embedding top-K on log for stuff not already included
    if log:
        all_texts = [e.text for e in log]
        embs = embed_batch(all_texts + [question], cache, budget)
        q_emb = embs[-1]
        scores = [cosine(q_emb, v) for v in embs[:-1]]
        ranked = sorted(range(len(log)), key=lambda i: scores[i], reverse=True)
        extras = []
        for i in ranked:
            if log[i].uuid in included_uuids:
                continue
            extras.append(log[i])
            if len(extras) >= top_k_extra:
                break
        extra_lines = []
        for e in extras:
            ref_str = ""
            if e.refs:
                ref_str = (
                    " refs=[" + ",".join(f"{r.uuid}:{r.relation}" for r in e.refs) + "]"
                )
            mentions = ",".join(e.mentions)
            extra_lines.append(f"[{e.uuid}] t{e.ts} {mentions} :: {e.text}{ref_str}")
        extra_block = "\n".join(extra_lines) if extra_lines else "(none)"
    else:
        extra_block = "(empty log)"

    prompt = VIEWS_READ_PROMPT.format(
        views_block=views_block,
        extra_block=extra_block,
        question=question,
    )
    return llm(prompt, cache, budget).strip()


def ingest_and_build(
    turns,
    cache: Cache,
    budget: Budget,
):
    log = aen1_ingest(turns, cache, budget)
    views, inv, sup = build_views(log)
    return log, views, inv, sup
