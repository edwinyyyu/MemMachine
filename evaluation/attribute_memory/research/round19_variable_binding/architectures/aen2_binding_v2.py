"""AEN-2 BINDING V2 — simplified single-emit schema with stronger filler skip.

Lessons from v1:
  - Writer over-emits filler ("stomach hurts", "inbox at 412") as chains, then
    re-emits the same content across overlapping windows. 384 entries for 126
    turns; mostly redundant filler.
  - The two-output schema (Entry + Resolution items) was hard for the model.
    Most coref pairs got no anonymous cluster created at all.

V2 changes:
  - SINGLE emit type (entry). canonical_label is a field on the entry itself.
    When a turn names an existing anonymous cluster, the entry just states
    the binding ("@User's boss is named @Marcus") with cluster_id matching the
    existing anon cluster AND canonical_label="Marcus". Post-processor extracts
    a Resolution from any entry with canonical_label set on an existing cluster.
  - HARDER filler skip: writer is told to emit ONLY for chain-worthy events
    (durable life states: jobs, relationships, locations, decisions, named
    people, named entities). Body sensations, weather, transient feelings are
    explicitly listed as DO-NOT-EMIT examples.
  - Active state shows ONLY DURABLE chains (predicates from a curated list),
    not random predicates the writer invented earlier. This breaks the
    ephemeral-chain feedback loop.

The retrieval and IndexedLog machinery is identical to v1.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND19 = HERE.parent
RESEARCH = ROUND19.parent
ROUND7 = RESEARCH / "round7"
sys.path.insert(0, str(ROUND7 / "experiments"))

from _common import Budget, Cache, cosine, embed_batch, extract_json, llm  # noqa: E402

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class LogEntry:
    uuid: str
    ts: int
    cluster_id: str
    text: str
    subject: str | None = None
    predicate: str | None = None
    canonical_label: str | None = None
    mentions: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)


@dataclass
class Resolution:
    uuid: str
    ts: int
    cluster_id: str
    canonical_label: str
    evidence_entry_uuids: list[str] = field(default_factory=list)
    supersedes_resolution_id: str | None = None


@dataclass
class IndexedLog:
    entries: list[LogEntry]
    resolutions: list[Resolution]
    by_uuid: dict[str, LogEntry]
    cluster_entries: dict[str, list[LogEntry]]
    cluster_label: dict[str, str | None]
    chain_head: dict[tuple[str, str], str]
    chain_head_entry: dict[tuple[str, str], str]
    mention_index: dict[str, list[str]]
    embed_by_uuid: dict[str, list[float]]


def derive_resolutions(entries: list[LogEntry]) -> list[Resolution]:
    """Extract Resolution records from entries that carry a canonical_label.

    First time a cluster_id appears with a non-null canonical_label, that's
    a Resolution. Subsequent labels for the same cluster (if different) are
    superseding resolutions.
    """
    resolutions: list[Resolution] = []
    last_label: dict[str, str] = {}
    counter = 0
    for e in sorted(entries, key=lambda x: (x.ts, x.uuid)):
        if not e.canonical_label:
            continue
        prev = last_label.get(e.cluster_id)
        if prev == e.canonical_label:
            continue
        counter += 1
        resolutions.append(
            Resolution(
                uuid=f"r{e.ts:04d}_{counter}",
                ts=e.ts,
                cluster_id=e.cluster_id,
                canonical_label=e.canonical_label,
                evidence_entry_uuids=[e.uuid],
            )
        )
        last_label[e.cluster_id] = e.canonical_label
    return resolutions


def build_index(
    entries: list[LogEntry],
    cache: Cache,
    budget: Budget,
) -> tuple[IndexedLog, list[Resolution]]:
    by_uuid = {e.uuid: e for e in entries}
    resolutions = derive_resolutions(entries)

    cluster_entries: dict[str, list[LogEntry]] = {}
    for e in entries:
        cluster_entries.setdefault(e.cluster_id, []).append(e)

    cluster_label: dict[str, str | None] = dict.fromkeys(cluster_entries)
    res_by_cluster: dict[str, list[Resolution]] = {}
    for r in resolutions:
        res_by_cluster.setdefault(r.cluster_id, []).append(r)
    for cid, rs in res_by_cluster.items():
        rs.sort(key=lambda r: (r.ts, r.uuid))
        cluster_label[cid] = rs[-1].canonical_label

    chain_head: dict[tuple[str, str], str] = {}
    chain_head_entry: dict[tuple[str, str], str] = {}
    for e in sorted(entries, key=lambda x: (x.ts, x.uuid)):
        if e.subject and e.predicate:
            key = (e.subject, e.predicate)
            chain_head[key] = e.cluster_id
            chain_head_entry[key] = e.uuid

    mention_index: dict[str, list[str]] = {}
    for e in entries:
        for m in e.mentions:
            mention_index.setdefault(m, []).append(e.uuid)

    texts = [e.text for e in entries]
    embs = embed_batch(texts, cache, budget) if entries else []
    embed_by_uuid = {e.uuid: embs[i] for i, e in enumerate(entries)}

    return (
        IndexedLog(
            entries=entries,
            resolutions=resolutions,
            by_uuid=by_uuid,
            cluster_entries=cluster_entries,
            cluster_label=cluster_label,
            chain_head=chain_head,
            chain_head_entry=chain_head_entry,
            mention_index=mention_index,
            embed_by_uuid=embed_by_uuid,
        ),
        resolutions,
    )


# Curated list of durable predicates for active-state filtering. Predicates
# outside this list are deemphasized (still indexed, but not surfaced as
# active state to the writer, breaking the ephemeral-chain feedback loop).
DURABLE_PREDICATE_KEYWORDS = {
    "boss",
    "manager",
    "supervisor",
    "employer",
    "company",
    "workplace",
    "job",
    "occupation",
    "title",
    "role",
    "position",
    "team",
    "colleague",
    "coworker",
    "mentor",
    "advisor",
    "partner",
    "spouse",
    "fiance",
    "girlfriend",
    "boyfriend",
    "friend",
    "old_friend",
    "best_friend",
    "neighbor",
    "gym_buddy",
    "gym_friend",
    "senior",
    "junior",
    "location",
    "city",
    "country",
    "neighborhood",
    "residence",
    "home",
    "school",
    "university",
    "college",
    "hobby",
    "pet",
    "child",
    "parent",
    "sibling",
    "family",
    "name",
}


def is_durable_predicate(predicate: str | None) -> bool:
    if not predicate:
        return False
    pred_lower = predicate.lower()
    for kw in DURABLE_PREDICATE_KEYWORDS:
        if kw in pred_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Active-state rendering (DURABLE chains only)
# ---------------------------------------------------------------------------


def render_active_state(idx: IndexedLog | None, target_entities: set[str]) -> str:
    if idx is None:
        return "(empty)"
    relevant: list[tuple[tuple[str, str], str]] = []
    for key, cid in idx.chain_head.items():
        subj, pred = key
        if not is_durable_predicate(pred):
            continue
        if subj in target_entities or subj == "@User":
            relevant.append((key, cid))

    def _ts_for(key: tuple[str, str]) -> int:
        return idx.by_uuid[idx.chain_head_entry[key]].ts

    relevant.sort(key=lambda kv: _ts_for(kv[0]), reverse=True)

    lines = []
    for (subj, pred), cid in relevant[:20]:
        head_uuid = idx.chain_head_entry[(subj, pred)]
        head_entry = idx.by_uuid[head_uuid]
        label = idx.cluster_label.get(cid)
        label_str = f"label={label!r}" if label else "label=<unnamed>"
        lines.append(
            f"  {pred}: cluster={cid} {label_str}  "
            f"head=[{head_uuid} t={head_entry.ts}] :: {head_entry.text[:80]}"
        )
    return "\n".join(lines) if lines else "(none)"


def render_recent_log(prior_entries: list[LogEntry], cap: int = 8) -> str:
    if not prior_entries:
        return "(empty)"
    recent = list(reversed(prior_entries[-cap:]))
    lines = []
    for e in recent:
        bits = []
        if e.subject:
            bits.append(f"subject={e.subject}")
        if e.predicate:
            bits.append(f"pred={e.predicate}")
        if e.canonical_label:
            bits.append(f"label={e.canonical_label!r}")
        meta = " ".join(bits)
        lines.append(
            f"  [{e.uuid}] t={e.ts} cluster={e.cluster_id} {meta} :: {e.text[:90]}"
        )
    return "\n".join(lines)


def extract_window_entities(turns: list[tuple[int, str]]) -> set[str]:
    text = " ".join(t for _, t in turns)
    tokens = re.findall(r"\b([A-Z][a-z]{1,20})\b", text)
    ents = {f"@{t}" for t in tokens}
    ents.add("@User")
    return ents


# ---------------------------------------------------------------------------
# Writer prompt
# ---------------------------------------------------------------------------


WRITE_PROMPT = """You are a semantic-memory writer. Memory uses VARIABLE BINDING:

  An anonymous descriptor like "my new boss" creates an ENTITY (cluster_id)
  with NO label yet. Later, "his name is Marcus" assigns canonical_label
  ="Marcus" to that same cluster_id. The chain @User.boss continues to point
  at the same cluster — only the LABEL changed.

You emit ONE kind of item: an ENTRY. Schema:
  text:              atomic prose fact
  mentions:          @entity tags appearing in the text
  subject:           "@entity" the predicate is about (typically "@User")
  predicate:         "@subject.predicate_name" (e.g., "@User.boss"). Optional.
  cluster_id:        identifier for the OBJECT entity (the boss, the team).
                     - Anonymous: "anon_<topic>_<TURN>" (e.g., "anon_boss_3")
                     - Named entity: lowercase surface (e.g., "marcus", "priya")
                     - Reuse existing cluster_id from ACTIVE STATE when this entry
                       is about an entity already introduced.
  canonical_label:   the realized name if known. NULL if entity is still anonymous.
                     Setting this BINDS the cluster's label going forward.

ACTIVE STATE (current durable chain heads — only DURABLE predicates shown):
{active_state}

RECENT LOG (for context):
{recent_log}

KNOWN LABELED CLUSTERS:
{known_clusters}

CONVERSATION WINDOW
-------------------
{window_block}

Emit JSON for TARGET turns only:
{{
  "entries": [
    {{"turn": <int>, "text": "...", "mentions": ["@..."],
      "subject": "@User", "predicate": "@User.boss",
      "cluster_id": "anon_boss_3", "canonical_label": null}}
  ]
}}

EMIT for DURABLE LIFE FACTS. A fact is durable if it would still be true /
relevant a week or month later — it's about User's life, not their moment.

DURABLE (DO emit):
  - Job, employer, title, team, role
  - Boss, manager, mentor, colleague (named introductions especially)
  - Relationship: partner, friend, neighbor, family
  - Location: home, city, neighborhood
  - Possessions: car, bike, pet — ANY new car/bike/pet is durable, even
    "leased a Tesla Model 3" or "got a used Civic"
  - Hobbies & recurring routines: "started climbing", "joined a gym",
    "biking to work now", "took up pottery" — these are stable life patterns,
    NOT filler
  - School, university, education
  - Plans, decisions, named expectations
  - Naming an entity that was previously anonymous

FILLER (skip silently — emit `{{"entries": []}}` if all targets are filler):
  - Momentary body sensations: "stomach hurts NOW", "tired right now",
    "need a nap" (one-off, not "I have insomnia")
  - Weather, time-of-day, atmosphere: "rainy day", "slow afternoon"
  - This-instant feelings: "annoyed", "happy", "frustrated" (not durable
    moods or named emotional states like "User has been depressed")
  - One-off mundane actions: "had coffee this morning", "going to lunch"
    (not "User drinks coffee every morning" — that's a habit, durable)
  - Notification noise: "412 unread", "email avalanche", "Slack laggy"
  - Generic chitchat with no durable content

DISTINCTION TEST: would User's friend or assistant still want to know this in
a month? If yes, durable. If not, filler.

VARIABLE-BINDING RULES (the key payoff of this system):
  1. Anonymous descriptor ("my new boss started this week"):
     emit ENTRY with cluster_id="anon_boss_<TURN>", canonical_label=null,
     subject=@User, predicate=@User.boss.
  2. Name reveal for an EXISTING anonymous cluster ("his name is Marcus"
     about the boss already in active state as anon_boss_3):
     emit ENTRY with cluster_id="anon_boss_3" (matching the existing one),
     canonical_label="Marcus", subject=@User, predicate=@User.boss.
     (Reuse the SAME cluster_id from active state — don't create a new one.)
  3. Named entity introduced fresh: emit ENTRY with cluster_id="marcus",
     canonical_label="Marcus", subject/predicate as appropriate.
  4. Chain transition (User got a NEW boss after Marcus): emit ENTRY with a
     fresh cluster_id ("anon_boss_<NEW_TURN>" or the new name), no
     canonical_label unless name is given.

Output JSON ONLY. No commentary.
"""


def write_window(
    window_turns: list[tuple[int, str]],
    target_turn_lo: int,
    target_turns: list[tuple[int, str]],
    prior_entries: list[LogEntry],
    idx: IndexedLog | None,
    cache: Cache,
    budget: Budget,
) -> tuple[list[LogEntry], dict]:
    target_entities = extract_window_entities(target_turns)
    active_state = render_active_state(idx, target_entities)
    recent_log = render_recent_log(prior_entries, cap=8)

    if idx is None:
        known_clusters_str = "(none)"
    else:
        labeled = [
            f"{cid}={label!r}" for cid, label in idx.cluster_label.items() if label
        ]
        known_clusters_str = "; ".join(labeled[:30]) or "(none labeled yet)"

    window_lines = []
    in_target = False
    for tidx, text in window_turns:
        if not in_target and tidx >= target_turn_lo:
            window_lines.append("--- TARGET TURNS (emit entries for these) ---")
            in_target = True
        prefix = "  TARGET" if in_target else "  CONTEXT"
        window_lines.append(f"{prefix} TURN {tidx}: {text}")
    if not in_target:
        window_lines.insert(0, "--- TARGET TURNS ---")
    window_block = "\n".join(window_lines)

    prompt = WRITE_PROMPT.format(
        active_state=active_state,
        recent_log=recent_log,
        known_clusters=known_clusters_str,
        window_block=window_block,
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    telemetry = {
        "prompt_chars": len(prompt),
        "active_state_chars": len(active_state),
        "window_size": len(window_turns),
        "target_size": len(target_turns),
    }
    if not isinstance(obj, dict):
        return [], telemetry

    target_turn_set = {t for t, _ in target_turns}
    items = obj.get("entries", []) or []
    entries: list[LogEntry] = []
    counter = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        text = (it.get("text") or "").strip()
        if not text:
            continue
        cluster_id = (it.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        ts_raw = it.get("turn")
        try:
            ts = int(ts_raw) if ts_raw is not None else target_turns[-1][0]
        except (TypeError, ValueError):
            ts = target_turns[-1][0]
        if ts not in target_turn_set:
            ts = target_turns[-1][0]
        mentions = [m for m in (it.get("mentions") or []) if isinstance(m, str)]
        subject = it.get("subject") if isinstance(it.get("subject"), str) else None
        predicate = (
            it.get("predicate") if isinstance(it.get("predicate"), str) else None
        )
        canonical_label = it.get("canonical_label")
        if canonical_label is not None and not isinstance(canonical_label, str):
            canonical_label = None
        if isinstance(canonical_label, str):
            canonical_label = canonical_label.strip() or None
        refs = [r for r in (it.get("refs") or []) if isinstance(r, str)]
        uuid = f"e{ts:04d}_{counter}"
        counter += 1
        entries.append(
            LogEntry(
                uuid=uuid,
                ts=ts,
                cluster_id=cluster_id,
                text=text,
                subject=subject,
                predicate=predicate,
                canonical_label=canonical_label,
                mentions=mentions,
                refs=refs,
            )
        )
    telemetry["n_entries_emitted"] = len(entries)
    telemetry["n_with_label"] = sum(1 for e in entries if e.canonical_label)
    return entries, telemetry


# ---------------------------------------------------------------------------
# Retrieval (label-substitution at read-time)
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
    selected: set[str] = set()

    for (subj, pred), cid in idx.chain_head.items():
        if subj in q_tags:
            selected.add(idx.chain_head_entry[(subj, pred)])
            for prior in idx.cluster_entries.get(cid, []):
                selected.add(prior.uuid)

    mention_candidates: list[str] = []
    for tag in q_tags:
        mention_candidates.extend(idx.mention_index.get(tag, []))
    selected.update(
        _rank_by_embedding(
            q_emb, list(set(mention_candidates)), idx.embed_by_uuid, top_k=top_k
        )
    )

    all_uuids = [e.uuid for e in idx.entries]
    selected.update(
        _rank_by_embedding(q_emb, all_uuids, idx.embed_by_uuid, top_k=top_k)
    )

    out = sorted(selected, key=lambda u: idx.by_uuid[u].ts)
    if len(out) > 60:
        out = out[-60:]
    return [idx.by_uuid[u] for u in out]


READ_PROMPT = """You are answering a question about User using a
variable-binding semantic memory. Each entry has a cluster_id (entity
identity); some clusters have a canonical_label, others are anonymous.

When an entry references cluster_id X and X has a canonical_label, treat the
entry as being about that named entity.

CLUSTER LABELS (cluster_id -> name):
{cluster_labels}

RETRIEVED ENTRIES (chronological):
{entries_block}

QUESTION: {question}

Answer concisely. For yes/no questions, start with "Yes" or "No".
"""


def format_entries_for_read(entries):
    lines = []
    for e in entries:
        bits = []
        if e.subject:
            bits.append(f"subject={e.subject}")
        if e.predicate:
            bits.append(f"pred={e.predicate}")
        bits.append(f"cluster={e.cluster_id}")
        if e.canonical_label:
            bits.append(f"label={e.canonical_label!r}")
        meta = " ".join(bits)
        lines.append(f"[{e.uuid}] t{e.ts} :: {e.text}  ({meta})")
    return "\n".join(lines)


def answer_question(question, idx, cache, budget, top_k=12):
    retrieved = retrieve(question, idx, cache, budget, top_k=top_k)
    block = format_entries_for_read(retrieved)
    label_lines = [
        f"  {cid} -> {label}" for cid, label in idx.cluster_label.items() if label
    ]
    cluster_labels_str = "\n".join(label_lines) if label_lines else "(none)"
    prompt = READ_PROMPT.format(
        cluster_labels=cluster_labels_str,
        entries_block=block,
        question=question,
    )
    return llm(prompt, cache, budget).strip()


# ---------------------------------------------------------------------------
# Ingestion driver — K-block centered window
# ---------------------------------------------------------------------------


def ingest_turns(
    turns: list[tuple[int, str]],
    cache: Cache,
    budget: Budget,
    *,
    w_past: int = 7,
    w_future: int = 14,
    k: int = 3,
    rebuild_index_every: int = 4,
) -> tuple[list[LogEntry], list[Resolution], IndexedLog, list[dict]]:
    log: list[LogEntry] = []
    idx: IndexedLog | None = None
    telemetry: list[dict] = []

    n_turns = len(turns)
    fire_no = 0
    target_lo = 0
    while target_lo < n_turns:
        target_hi = min(n_turns, target_lo + k)
        win_lo = max(0, target_lo - w_past)
        win_hi = min(n_turns, target_hi + w_future)
        window_turns = turns[win_lo:win_hi]
        target_turns = turns[target_lo:target_hi]
        if not target_turns:
            break
        target_turn_lo = target_turns[0][0]

        new_entries, tele = write_window(
            window_turns,
            target_turn_lo,
            target_turns,
            log,
            idx,
            cache,
            budget,
        )
        log.extend(new_entries)
        tele["fire_no"] = fire_no
        tele["last_turn"] = target_turns[-1][0]
        tele["w_past_actual"] = target_lo - win_lo
        tele["w_future_actual"] = win_hi - target_hi
        tele["k_actual"] = target_hi - target_lo
        telemetry.append(tele)

        if fire_no % rebuild_index_every == 0:
            idx, _ = build_index(log, cache, budget)
        fire_no += 1
        target_lo = target_hi

    idx, resolutions = build_index(log, cache, budget)
    return log, resolutions, idx, telemetry
