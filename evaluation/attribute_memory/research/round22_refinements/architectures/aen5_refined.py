"""AEN-4 PARTITIONED — multi-collection storage.

Variable binding writer (R19 v2 baseline) + cognition pass, but each entry
goes into its own COLLECTION (storage partition):

  observations:  factual extracted entries (writer output)
  cognition:     mental-state entries (cognizer output)

Each collection has its own indexes:
  - by_uuid, mention_index, cluster_entries, cluster_label
  - chain_head per (subject, predicate)
  - embed_by_uuid

Retrieval decides per-question which collection(s) to query:
  - factual question: observations only
  - cognitive question (expect/plan/think/feel): observations + cognition

Active-state injection at write-time uses ONLY observations (cognition doesn't
influence the writer's chain decisions; that would be a feedback loop).

Cluster IDs are a SHARED namespace across collections (cognition can mention
an observation cluster_id and vice versa). Mentions are the cross-collection
linkage; the reader sees retrieved entries from both partitions and infers
relationships from text + cluster_id matches.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND21 = HERE.parent
RESEARCH = ROUND21.parent
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
    collection: str = "observations"  # "observations" | "cognition"


@dataclass
class Resolution:
    uuid: str
    ts: int
    cluster_id: str
    canonical_label: str
    evidence_entry_uuids: list[str] = field(default_factory=list)
    collection: str = "observations"


@dataclass
class IndexedCollection:
    """Index for a single collection (observations OR cognition)."""

    name: str
    entries: list[LogEntry]
    resolutions: list[Resolution]
    by_uuid: dict[str, LogEntry]
    cluster_entries: dict[str, list[LogEntry]]
    cluster_label: dict[str, str | None]
    chain_head: dict[tuple[str, str], str]
    chain_head_entry: dict[tuple[str, str], str]
    mention_index: dict[str, list[str]]
    embed_by_uuid: dict[str, list[float]]


@dataclass
class MemoryStore:
    """Holds multiple collections."""

    collections: dict[str, IndexedCollection] = field(default_factory=dict)

    def get(self, name: str) -> IndexedCollection | None:
        return self.collections.get(name)


def derive_resolutions(entries: list[LogEntry], collection: str) -> list[Resolution]:
    resolutions: list[Resolution] = []
    last_label: dict[str, str] = {}
    counter = 0
    for e in sorted(entries, key=lambda x: (x.ts, x.uuid)):
        if not e.canonical_label:
            continue
        if last_label.get(e.cluster_id) == e.canonical_label:
            continue
        counter += 1
        resolutions.append(
            Resolution(
                uuid=f"r{e.ts:04d}_{counter}",
                ts=e.ts,
                cluster_id=e.cluster_id,
                canonical_label=e.canonical_label,
                evidence_entry_uuids=[e.uuid],
                collection=collection,
            )
        )
        last_label[e.cluster_id] = e.canonical_label
    return resolutions


def build_collection(
    name: str,
    entries: list[LogEntry],
    cache: Cache,
    budget: Budget,
) -> IndexedCollection:
    by_uuid = {e.uuid: e for e in entries}
    resolutions = derive_resolutions(entries, name)

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

    return IndexedCollection(
        name=name,
        entries=entries,
        resolutions=resolutions,
        by_uuid=by_uuid,
        cluster_entries=cluster_entries,
        cluster_label=cluster_label,
        chain_head=chain_head,
        chain_head_entry=chain_head_entry,
        mention_index=mention_index,
        embed_by_uuid=embed_by_uuid,
    )


def build_store(
    obs_entries: list[LogEntry],
    cog_entries: list[LogEntry],
    cache: Cache,
    budget: Budget,
) -> MemoryStore:
    store = MemoryStore()
    store.collections["observations"] = build_collection(
        "observations", obs_entries, cache, budget
    )
    store.collections["cognition"] = build_collection(
        "cognition", cog_entries, cache, budget
    )
    return store


# Durable predicate keywords (factual chains).
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


def is_durable_predicate(predicate):
    if not predicate:
        return False
    pl = predicate.lower()
    return any(kw in pl for kw in DURABLE_PREDICATE_KEYWORDS)


def render_active_state(
    obs_idx: IndexedCollection | None, target_entities: set[str]
) -> str:
    """Render active state from OBSERVATIONS only — cognition shouldn't drive writer's chain decisions."""
    if obs_idx is None:
        return "(empty)"
    relevant: list[tuple[tuple[str, str], str]] = []
    for key, cid in obs_idx.chain_head.items():
        subj, pred = key
        if not is_durable_predicate(pred):
            continue
        if subj in target_entities or subj == "@User":
            relevant.append((key, cid))

    def _ts_for(key):
        return obs_idx.by_uuid[obs_idx.chain_head_entry[key]].ts

    relevant.sort(key=lambda kv: _ts_for(kv[0]), reverse=True)

    lines = []
    for (subj, pred), cid in relevant[:20]:
        head_uuid = obs_idx.chain_head_entry[(subj, pred)]
        head_entry = obs_idx.by_uuid[head_uuid]
        label = obs_idx.cluster_label.get(cid)
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


def extract_window_entities(turns):
    text = " ".join(t for _, t in turns)
    tokens = re.findall(r"\b([A-Z][a-z]{1,20})\b", text)
    ents = {f"@{t}" for t in tokens}
    ents.add("@User")
    return ents


# ---------------------------------------------------------------------------
# Writer prompt — restoring v2's exact text that scored 8/8 on multi_batch_coref
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

ONLY EMIT FOR CHAIN-WORTHY EVENTS. These ARE chain-worthy:
  - New job, new boss, new team, new colleague mention
  - New location, new home, new school
  - New relationship, new friend, new mentor, new neighbor
  - New possession: car, bike, pet (a new car/bike/pet IS durable; "picked up
    a Bianchi" = chain-worthy purchase)
  - Hobbies & recurring routines: "started climbing", "joined a gym",
    "biking to work" (stable life patterns, not filler)
  - Confirmed plans/decisions: "decided to move to Berlin", "starting at
    Notion next week"
  - Naming an entity that was previously anonymous
  - Update/change to any of the above (job change, etc.)

DO NOT EMIT for these (skip silently):
  - Body sensations: "stomach hurts", "tired", "need a nap"
  - Weather: "weather is nice", "rainy day"
  - Transient feelings: "long day", "slow afternoon", "mellow morning"
  - Routine activities: "had coffee", "going for lunch", "on calls all morning"
  - Inbox/notification noise: "412 unread", "email avalanche", "Slack laggy"
  - Generic chitchat
  Output `{{"entries": []}}` if the TARGET turns are entirely filler.

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


COGNIZE_PROMPT = """You are the COGNITION pass for a semantic-memory system.

The writer just extracted these OBSERVATION entries from the latest turns.
Your job: reason about implications. Emit COGNITION entries that capture
User's:
  - Expectations: things User now expects given new info ("@User expects @Sam to be the boss" because User said earlier this would happen if hired at @Notion)
  - Plans: actions User intends to take.
  - Beliefs: things User now believes (when not directly asserted as fact).
  - Fears / hopes: emotional valence about uncertain outcomes.
  - Confirmations / contradictions: prior expectations that just became true or false.

The cognition entries are MEMORIES of User's current mental state, not the
observations themselves. They reference observed clusters by mention or
cluster_id so retrieval surfaces them naturally.

Schema:
  text:               atomic prose ("@User expects @Sam to be his boss at @Notion")
  mentions:           @entity tags
  subject:            "@User"
  predicate:          "@User.expectation" / "@User.plan" / "@User.belief" /
                      "@User.hope" / "@User.fear" / "@User.confirmation"
  cluster_id:         can be a fresh "cog_<topic>_<TURN>" id, OR reuse the
                      cluster_id of the entity the cognition is ABOUT
  canonical_label:    optional

ACTIVE STATE (durable observation chain heads):
{active_state}

RECENT LOG (most recent committed entries — observations + cognitions):
{recent_log}

NEW OBSERVATIONS just emitted (the trigger for this cognition pass):
{new_observations}

CONVERSATION WINDOW (for context):
{window_block}

EMIT cognition entries ONLY for SPECIFIC TRIGGERS — not as routine commentary.
Default is ZERO entries. Emit one ONLY when one of these triggers fires:

  TRIGGER 1: CONDITIONAL — turn states "if X then Y" or describes a plan
  contingent on something.
    → emit @User.plan or @User.expectation: "If <condition>, @User intends/expects <consequence>".

  TRIGGER 2: CONFIRMATION — new observation matches a prior conditional/plan
  visible in active state or recent log.
    → emit @User.confirmation: "@User's prior plan to <X> is now confirmed by <observation>".

  TRIGGER 3: CONTRADICTION — new observation contradicts a prior expectation.
    → emit @User.contradiction: "@User's prior expectation about <X> is contradicted by <observation>".

  TRIGGER 4: NAMED HOPE/FEAR — User explicitly states a hope or fear.
    → emit @User.hope or @User.fear: matching User's stated valence.

DO NOT emit:
  - Trivial restatements of the observation itself
  - Generic "User now knows X" entries
  - Speculation without an explicit trigger above
  - Cognition entries with the same predicate as a factual chain (NEVER emit
    @User.boss/.employer/.location/.team/.partner/.title from cognition —
    those are FACTUAL chains owned by observations).

Default output is `{{"entries": []}}`. Emit non-empty ONLY when a trigger above clearly fires.
Aim for AT MOST one cognition entry per K-block, often zero.

Emit JSON:
{{"entries": [{{"turn": <int>, "text": "...", "mentions": ["@..."],
   "subject": "@User", "predicate": "@User.expectation",
   "cluster_id": "...", "canonical_label": null}}]}}
"""


def _parse_entries(
    items, target_turn_set, target_turns, collection: str, counter_offset: int = 0
):
    entries: list[LogEntry] = []
    counter = counter_offset
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
        if isinstance(canonical_label, str):
            canonical_label = canonical_label.strip() or None
        else:
            canonical_label = None
        prefix = "e" if collection == "observations" else "c"
        uuid = f"{prefix}{ts:04d}_{counter}"
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
                refs=[],
                collection=collection,
            )
        )
    return entries


def write_window(
    window_turns, target_turn_lo, target_turns, prior_obs, store, cache, budget
):
    target_entities = extract_window_entities(target_turns)
    obs_idx = store.get("observations") if store else None
    active_state = render_active_state(obs_idx, target_entities)
    recent_log = render_recent_log(prior_obs, cap=8)
    if obs_idx is None:
        known_clusters_str = "(none)"
    else:
        labeled = [
            f"{cid}={label!r}" for cid, label in obs_idx.cluster_label.items() if label
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
    telemetry = {"write_prompt_chars": len(prompt), "window_size": len(window_turns)}
    if not isinstance(obj, dict):
        return [], telemetry, window_block
    target_turn_set = {t for t, _ in target_turns}
    items = obj.get("entries", []) or []
    entries = _parse_entries(
        items, target_turn_set, target_turns, "observations", counter_offset=0
    )
    telemetry["n_observations_emitted"] = len(entries)
    return entries, telemetry, window_block


def cognize_window(
    target_turns, new_observations, prior_log, store, window_block, cache, budget
):
    if not new_observations:
        return [], {"n_cognitions_emitted": 0}
    target_entities = extract_window_entities(target_turns)
    for e in new_observations:
        for m in e.mentions:
            target_entities.add(m)
    obs_idx = store.get("observations") if store else None
    active_state = render_active_state(obs_idx, target_entities)
    recent_log = render_recent_log(prior_log, cap=8)

    obs_lines = []
    for e in new_observations:
        bits = []
        if e.subject:
            bits.append(f"subject={e.subject}")
        if e.predicate:
            bits.append(f"pred={e.predicate}")
        bits.append(f"cluster={e.cluster_id}")
        if e.canonical_label:
            bits.append(f"label={e.canonical_label!r}")
        obs_lines.append(f"  [{e.uuid}] t={e.ts} :: {e.text}  ({' '.join(bits)})")
    new_observations_str = "\n".join(obs_lines)

    prompt = COGNIZE_PROMPT.format(
        active_state=active_state,
        recent_log=recent_log,
        new_observations=new_observations_str,
        window_block=window_block,
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    telemetry = {"cognize_prompt_chars": len(prompt)}
    if not isinstance(obj, dict):
        return [], telemetry
    target_turn_set = {t for t, _ in target_turns}
    items = obj.get("entries", []) or []
    entries = _parse_entries(
        items, target_turn_set, target_turns, "cognition", counter_offset=1000
    )
    telemetry["n_cognitions_emitted"] = len(entries)
    return entries, telemetry


# ---------------------------------------------------------------------------
# Retrieval — multi-collection
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


def extract_question_entities(question):
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


COGNITIVE_QUESTION_KEYWORDS = {
    # Direct mental-state vocabulary
    "expect",
    "expected",
    "anticipate",
    "plan",
    "intend",
    "intended",
    "intention",
    "believe",
    "believed",
    "think",
    "thought",
    "fear",
    "worried",
    "concern",
    "hope",
    "wish",
    "wished",
    "predict",
    "predicted",
    "feel",
    "felt",
    # Confirmation-of-plan / outcome questions: surface cognition because the
    # answer often depends on a prior plan/intent that needs to be cross-checked.
    "end up",
    "ended up",
    "actually buy",
    "actually buying",
    "actually do",
    "actually did",
    "did user end",
    "does user end",
    # Questions framed around future/conditional outcomes
    "would happen",
    "will happen",
    "going to do",
    "going to be",
}


def is_cognitive_question(question):
    ql = question.lower()
    return any(kw in ql for kw in COGNITIVE_QUESTION_KEYWORDS)


def _rank_by_embedding(q_emb, candidate_uuids, embed_by_uuid, top_k):
    scored = []
    for u in candidate_uuids:
        v = embed_by_uuid.get(u)
        if v is None:
            continue
        scored.append((cosine(q_emb, v), u))
    scored.sort(reverse=True)
    return [u for _, u in scored[:top_k]]


def _retrieve_from_collection(idx: IndexedCollection, q_emb, q_tags, top_k):
    if not idx or not idx.entries:
        return []
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
    return [idx.by_uuid[u] for u in selected]


def retrieve(
    question,
    store: MemoryStore,
    cache,
    budget,
    top_k=12,
    include_cognition=None,
):
    if include_cognition is None:
        include_cognition = is_cognitive_question(question)
    q_emb = embed_batch([question], cache, budget)[0]
    q_ents = extract_question_entities(question)
    q_tags = [f"@{e}" for e in q_ents]

    obs_idx = store.get("observations")
    cog_idx = store.get("cognition") if include_cognition else None

    candidates: list[LogEntry] = []
    if obs_idx:
        candidates.extend(_retrieve_from_collection(obs_idx, q_emb, q_tags, top_k))
    if cog_idx:
        candidates.extend(_retrieve_from_collection(cog_idx, q_emb, q_tags, top_k))

    seen = set()
    deduped = []
    for e in sorted(candidates, key=lambda x: (x.ts, x.uuid)):
        if e.uuid in seen:
            continue
        seen.add(e.uuid)
        deduped.append(e)
    if len(deduped) > 60:
        deduped = deduped[-60:]
    return deduped


READ_PROMPT = """You are answering a question using a partitioned semantic memory.

Each entry has a cluster_id and lives in a COLLECTION:
  - observations: factual events in User's life (what was directly said/observed)
  - cognition: User's mental state — expectations, plans, beliefs, fears

Use observations as the source of truth for what User actually said/did.
Use cognition for what User expected, planned, or believed.

CLUSTER LABELS (cluster_id -> name, both collections):
{cluster_labels}

RETRIEVED ENTRIES (chronological; collection tagged):
{entries_block}

QUESTION: {question}

Answer concisely. For yes/no questions, start with "Yes" or "No".
"""


def format_entries_for_read(entries):
    lines = []
    for e in entries:
        bits = [f"col={e.collection}"]
        if e.subject:
            bits.append(f"subject={e.subject}")
        if e.predicate:
            bits.append(f"pred={e.predicate}")
        bits.append(f"cluster={e.cluster_id}")
        if e.canonical_label:
            bits.append(f"label={e.canonical_label!r}")
        lines.append(f"[{e.uuid}] t{e.ts} :: {e.text}  ({' '.join(bits)})")
    return "\n".join(lines)


def answer_question(question, store, cache, budget, top_k=12):
    retrieved = retrieve(question, store, cache, budget, top_k=top_k)
    block = format_entries_for_read(retrieved)
    label_lines = []
    for col in ("observations", "cognition"):
        idx = store.get(col)
        if idx is None:
            continue
        for cid, label in idx.cluster_label.items():
            if label:
                label_lines.append(f"  [{col}] {cid} -> {label}")
    cluster_labels_str = "\n".join(label_lines) if label_lines else "(none)"
    prompt = READ_PROMPT.format(
        cluster_labels=cluster_labels_str,
        entries_block=block,
        question=question,
    )
    return llm(prompt, cache, budget).strip()


# ---------------------------------------------------------------------------
# Ingestion driver — K-block centered + optional cognition pass
# ---------------------------------------------------------------------------


def ingest_turns(
    turns,
    cache,
    budget,
    *,
    w_past=7,
    w_future=7,
    k=3,
    rebuild_index_every=4,
    enable_cognition=True,
):
    obs_log: list[LogEntry] = []
    cog_log: list[LogEntry] = []
    store: MemoryStore | None = None
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

        new_observations, w_tele, window_block = write_window(
            window_turns,
            target_turn_lo,
            target_turns,
            obs_log,
            store,
            cache,
            budget,
        )
        obs_log.extend(new_observations)

        new_cognitions = []
        c_tele = {}
        if enable_cognition and new_observations:
            combined_log = obs_log + cog_log
            combined_log.sort(key=lambda e: (e.ts, e.uuid))
            new_cognitions, c_tele = cognize_window(
                target_turns,
                new_observations,
                combined_log,
                store,
                window_block,
                cache,
                budget,
            )
            cog_log.extend(new_cognitions)

        tele = {
            **w_tele,
            **c_tele,
            "fire_no": fire_no,
            "last_turn": target_turns[-1][0],
        }
        telemetry.append(tele)

        if fire_no % rebuild_index_every == 0:
            store = build_store(obs_log, cog_log, cache, budget)
        fire_no += 1
        target_lo = target_hi

    store = build_store(obs_log, cog_log, cache, budget)
    return obs_log, cog_log, store, telemetry
