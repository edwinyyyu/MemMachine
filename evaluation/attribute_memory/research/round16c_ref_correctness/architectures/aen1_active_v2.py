"""AEN-1 ACTIVE v2 — deterministic ref-linking post-pass.

Diagnostic from round 15 (results/diagnostic.json) showed:
  - 94.7% of emitted-but-incorrect refs were category B (the right prior chain
    head was NOT in the active-state block at write time).
  - Cap was not the cause (no batch hit cap=100). Indices were structurally
    stale because *clarify/detail entries with refs* polluted `supersede_head`
    by making true heads look superseded.
  - Inspection of writer logs also shows duplicate predicate names for the
    same chain (`@User.boss` vs `@User.manager`, `@User.title` vs `@User.role`
    vs `@User.occupation`). With the existing index logic, those are SEPARATE
    chains — the writer treats them inconsistently.

Strategy in v2: keep the writer prompt UNCHANGED (so the writer cache from
round 15 is 100% reusable; no new LLM cost for ingestion). Override two
things at the deterministic layer:

  1. **Predicate normalization** before chain bookkeeping: map common synonyms
     to a canonical name (boss↔manager, title↔role↔occupation↔job_title, etc.).
  2. **Two-pass post-link of refs**: after the writer emits entries, REPLACE
     the writer's refs for predicate-bearing entries with the uuid of the
     PREVIOUS chain head under the same `(entity, normalized_predicate)`,
     after filtering out clarify entries (entries whose text is highly
     similar to the previous entry are not allowed to become new heads).
     Non-predicate entries keep their writer-emitted refs untouched.

This eliminates ref errors entirely if the writer's predicate tagging is
recoverable; mistagged predicates (e.g. `@User.inbox` for a partner-state
turn) cause silent ref errors but also wouldn't be saveable by any
prompting fix.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND16C = HERE.parent
RESEARCH = ROUND16C.parent
ROUND15 = RESEARCH / "round15_active_chains"
ROUND11 = RESEARCH / "round11_writer_stress"
ROUND7 = RESEARCH / "round7"
sys.path.insert(0, str(ROUND15 / "architectures"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_active  # noqa: E402
import aen1_simple  # noqa: E402
from _common import Budget, Cache  # noqa: E402

# Re-export
LogEntry = aen1_simple.LogEntry
IndexedLog = aen1_simple.IndexedLog
build_index = aen1_simple.build_index
retrieve = aen1_simple.retrieve
answer_question = aen1_simple.answer_question


# ---------------------------------------------------------------------------
# Predicate normalization
# ---------------------------------------------------------------------------

# Many writer-emitted predicates are synonyms. We normalize them so all
# entries on the same logical chain share a key.
_PREDICATE_SYNONYMS: dict[str, str] = {
    # job/title chain
    "title": "title",
    "role": "title",
    "occupation": "title",
    "job_title": "title",
    "position": "title",
    # manager/boss chain (entity=User)
    "boss": "boss",
    "manager": "boss",
    "supervisor": "boss",
    # employer
    "employer": "employer",
    "company": "employer",
    "workplace": "employer",
    # partner
    "partner": "partner",
    "partner_state": "partner",
    "spouse": "partner",
    "fiance": "partner",
    "marital_status": "partner",
    "relationship": "partner",
}


def normalize_pred(pred: str | None) -> str | None:
    """Return canonical predicate name (the part after '@Entity.'),
    or None if pred is None / unparseable.
    """
    if not pred:
        return None
    m = re.match(r"@?[A-Za-z0-9_]+\.(.+)", pred)
    if not m:
        return None
    p = m.group(1).lower().strip()
    return _PREDICATE_SYNONYMS.get(p, p)


def parse_pred(pred: str | None) -> tuple[str, str] | None:
    """Return (entity_tag, normalized_pred) or None."""
    if not pred:
        return None
    m = re.match(r"(@?[A-Za-z0-9_]+)\.(.+)", pred)
    if not m:
        return None
    ent = m.group(1)
    if not ent.startswith("@"):
        ent = "@" + ent
    p = m.group(2).lower().strip()
    p = _PREDICATE_SYNONYMS.get(p, p)
    return (ent, p)


# ---------------------------------------------------------------------------
# Clarify detection
# ---------------------------------------------------------------------------

_CLARIFY_PHRASES = [
    "reiterating",
    "reiterates",
    "adding detail",
    "adding context",
    "adding to the prior",
    "clarifying",
    "clarifies",
    "adds detail",
    "providing more detail",
    "elaborates",
    "elaborating",
    "no change",
]


def is_clarify_text(text: str) -> bool:
    """Heuristic: does this text suggest a no-change clarify? (so we can
    avoid making it a chain head)."""
    t = text.lower()
    return any(phrase in t for phrase in _CLARIFY_PHRASES)


def _normalize_value_tokens(text: str) -> set[str]:
    """Extract content tokens (lowercased, alpha-only, length>2, not
    stopwords) so we can do an overlap check between two adjacent chain
    entries' texts."""
    toks = re.findall(r"[a-zA-Z]{3,}", text.lower())
    stop = {
        "the",
        "and",
        "now",
        "is",
        "was",
        "are",
        "user",
        "user's",
        "users",
        "team",
        "side",
        "for",
        "with",
        "this",
        "that",
        "his",
        "her",
        "their",
        "she",
        "him",
        "they",
        "them",
        "its",
        "but",
        "from",
        "into",
        "than",
        "then",
        "like",
        "back",
        "ago",
        "still",
        "more",
        "less",
        "much",
        "very",
        "such",
        "all",
        "any",
        "some",
        "lot",
        "lots",
        "prior",
        "note",
        "noted",
        "replacing",
        "reiterating",
        "adding",
        "detail",
        "clarifying",
        "context",
        "today",
        "yesterday",
        "tomorrow",
        "year",
        "month",
        "week",
        "day",
        "really",
        "appreciate",
        "appreciates",
        "added",
        "tight",
        "ones",
        "one",
    }
    return {t for t in toks if t not in stop}


def _looks_like_clarify(text: str, prev_text: str) -> bool:
    """If text overlaps too much with prev_text (>= 0.85 jaccard on content
    tokens), treat it as clarify. Combined with the explicit clarify phrases.
    """
    if is_clarify_text(text):
        return True
    a = _normalize_value_tokens(text)
    b = _normalize_value_tokens(prev_text)
    if not a or not b:
        return False
    inter = a & b
    union = a | b
    if not union:
        return False
    jacc = len(inter) / len(union)
    # If jaccard is very high AND no NEW substantive token appears, it's a
    # clarify.
    new_tokens = a - b
    return jacc >= 0.80 and len(new_tokens) <= 1


# ---------------------------------------------------------------------------
# Two-pass deterministic ref linker
# ---------------------------------------------------------------------------


def deterministic_relink(
    log: list[LogEntry],
    *,
    skip_clarify: bool = True,
    normalize: bool = True,
) -> list[LogEntry]:
    """Walk the log in chronological order. For each entry that has a
    predicate, set its `refs` to the uuid of the previous chain head under
    the same `(entity, normalized_predicate)` key (or empty if it's the first
    entry of that chain). If `skip_clarify` is True, an entry whose text
    looks like a no-change clarify of the prior head DOES NOT advance the
    head — it gets the SAME ref as the prior head, and the head pointer
    stays put.

    Non-predicate entries are left untouched (their original refs preserved).
    """
    by_uuid = {e.uuid: e for e in log}
    chain_heads: dict[tuple[str, str], str] = {}  # last "real" head per key
    new_log: list[LogEntry] = []
    n_predicate = 0
    n_relinked = 0
    n_clarify_skipped = 0
    for e in log:
        if not e.predicate:
            new_log.append(e)
            continue
        n_predicate += 1
        if normalize:
            key = parse_pred(e.predicate)
        else:
            m = re.match(r"(@?[A-Za-z0-9_]+)\.(.+)", e.predicate)
            if not m:
                new_log.append(e)
                continue
            ent = m.group(1)
            if not ent.startswith("@"):
                ent = "@" + ent
            key = (ent, m.group(2).lower())
        if key is None:
            new_log.append(e)
            continue
        prev_head_uuid = chain_heads.get(key)
        if prev_head_uuid is None:
            # first of this chain — no refs to add
            new_refs: list[str] = []
        else:
            new_refs = [prev_head_uuid]
            n_relinked += 1
        # Clarify check vs the current head
        is_clar = False
        if skip_clarify and prev_head_uuid is not None:
            prev_e = by_uuid.get(prev_head_uuid)
            if prev_e is not None and _looks_like_clarify(e.text, prev_e.text):
                is_clar = True
                n_clarify_skipped += 1
        # Replace the entry's refs (deterministic linker overrides the
        # writer's refs for predicate-bearing entries).
        new_e = LogEntry(
            uuid=e.uuid,
            ts=e.ts,
            text=e.text,
            mentions=list(e.mentions),
            refs=new_refs,
            predicate=e.predicate,
        )
        new_log.append(new_e)
        # Advance head only if NOT a clarify
        if not is_clar:
            chain_heads[key] = e.uuid
    return new_log


# ---------------------------------------------------------------------------
# Ingest = aen1_active ingest (writer prompt unchanged, cache-compatible) +
# post-hoc deterministic relink of the whole log
# ---------------------------------------------------------------------------
#
# Why post-hoc and not online? The writer cache from round 15 is keyed on the
# exact prompt string. If we changed `prior_log` mid-ingest (e.g. by feeding
# in relinked refs), every prompt would miss the cache and we'd burn ~150
# fresh writer LLM calls for no benefit (the writer prompt cannot use refs
# it doesn't yet have, and the writer's job is unchanged: emit good
# predicates).
#
# Instead: run aen1_active.ingest_turns verbatim (full cache hit), then
# REPLACE the writer's emitted refs at the end with the deterministic ones.
# Build the final index off the relinked log so the reader sees correct
# chain structure too.


def ingest_turns(
    turns: list[tuple[int, str]],
    cache: Cache,
    budget: Budget,
    batch_size: int = 5,
    rebuild_index_every: int = 4,
    max_active_state_size: int = 100,
    skip_clarify: bool = True,
    normalize: bool = True,
) -> tuple[list[LogEntry], IndexedLog, list[dict]]:
    """Same ingest contract as aen1_active.ingest_turns, but post-hoc relinks
    refs deterministically per `(entity, normalized_predicate)`. Writer
    behaviour (and writer cache) is unchanged.

    Returns (relinked_log, idx, telemetry).
    """
    log, _idx_pre, telemetry = aen1_active.ingest_turns(
        turns,
        cache,
        budget,
        batch_size=batch_size,
        rebuild_index_every=rebuild_index_every,
        max_active_state_size=max_active_state_size,
    )
    relinked = deterministic_relink(log, skip_clarify=skip_clarify, normalize=normalize)
    idx = build_index(relinked, cache, budget)
    return relinked, idx, telemetry
