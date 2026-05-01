"""AEN-3 COGNITION — variable-binding writer + cognition pass at write-time.

Adds a SECOND LLM pass after the writer (the "cognizer") that reasons about
the new observations + prior memories and emits cognition entries (User's
expectations, plans, beliefs, judgments, fears).

Key design:
  - LogEntry has `entry_type: "observation" | "cognition"`.
  - Cognize fires after each writer pass; sees retrieved relevant priors +
    the new observation entries; emits cognition entries with the same
    schema as observations (subject, predicate, cluster_id, mentions).
  - Retrieval type-filters based on question kind:
      * factual ("who is X?", "what is Y?"): observations only.
      * cognitive ("did User expect?", "what did User plan?"): both.
  - Cognition entries reference observation cluster_ids by mention/cluster_id,
    so spreading activation works through ordinary kNN + mention index.

This is the loop: observation → writer → cognition → next-turn writer sees
cognition in active state → cognition emits more entries → ...
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND20 = HERE.parent
RESEARCH = ROUND20.parent
ROUND19 = RESEARCH / "round19_variable_binding"
ROUND7 = RESEARCH / "round7"
sys.path.insert(0, str(ROUND19 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

from _common import cosine, embed_batch, extract_json, llm  # noqa: E402

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
    entry_type: str = "observation"  # "observation" | "cognition"


@dataclass
class Resolution:
    uuid: str
    ts: int
    cluster_id: str
    canonical_label: str
    evidence_entry_uuids: list[str] = field(default_factory=list)


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
            )
        )
        last_label[e.cluster_id] = e.canonical_label
    return resolutions


def build_index(entries, cache, budget):
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

COGNITIVE_PREDICATE_KEYWORDS = {
    "expectation",
    "expects",
    "anticipates",
    "plan",
    "intends",
    "intention",
    "belief",
    "believes",
    "thinks",
    "fear",
    "worried",
    "concern",
    "hope",
    "wishes",
    "judgment",
    "opinion",
    "prediction",
    "predicts",
}


def is_durable_predicate(predicate):
    if not predicate:
        return False
    pl = predicate.lower()
    return any(kw in pl for kw in DURABLE_PREDICATE_KEYWORDS)


def is_cognitive_predicate(predicate):
    if not predicate:
        return False
    pl = predicate.lower()
    return any(kw in pl for kw in COGNITIVE_PREDICATE_KEYWORDS)


# ---------------------------------------------------------------------------
# Active-state rendering
# ---------------------------------------------------------------------------


def render_active_state(idx, target_entities, include_cognitive=True):
    if idx is None:
        return "(empty)"
    relevant: list[tuple[tuple[str, str], str]] = []
    for key, cid in idx.chain_head.items():
        subj, pred = key
        if not (
            is_durable_predicate(pred)
            or (include_cognitive and is_cognitive_predicate(pred))
        ):
            continue
        if subj in target_entities or subj == "@User":
            relevant.append((key, cid))

    def _ts_for(key):
        return idx.by_uuid[idx.chain_head_entry[key]].ts

    relevant.sort(key=lambda kv: _ts_for(kv[0]), reverse=True)

    lines = []
    for (subj, pred), cid in relevant[:24]:
        head_uuid = idx.chain_head_entry[(subj, pred)]
        head_entry = idx.by_uuid[head_uuid]
        label = idx.cluster_label.get(cid)
        label_str = f"label={label!r}" if label else "label=<unnamed>"
        type_tag = (
            f" [{head_entry.entry_type}]"
            if head_entry.entry_type != "observation"
            else ""
        )
        lines.append(
            f"  {pred}{type_tag}: cluster={cid} {label_str}  "
            f"head=[{head_uuid} t={head_entry.ts}] :: {head_entry.text[:80]}"
        )
    return "\n".join(lines) if lines else "(none)"


def render_recent_log(prior_entries, cap=8):
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
        if e.entry_type != "observation":
            bits.append(f"type={e.entry_type}")
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
# Writer prompt (observation extraction)
# ---------------------------------------------------------------------------


WRITE_PROMPT = """You are a semantic-memory writer. Memory uses VARIABLE BINDING.

Anonymous descriptor "my new boss" creates a cluster_id with NO label.
Later "his name is Marcus" assigns canonical_label="Marcus" to the SAME
cluster_id. The chain @User.boss continues to point at the same cluster.

Emit ENTRIES with this schema:
  text:              atomic prose fact
  mentions:          @entity tags appearing in the text
  subject:           "@entity" the predicate is about (often "@User")
  predicate:         "@subject.predicate_name" (e.g., "@User.boss"). Optional.
  cluster_id:        identifier for the OBJECT entity:
                       - Anonymous: "anon_<topic>_<TURN>"
                       - Named:     lowercase surface (e.g., "marcus")
                       - Reuse from active state when extending an existing chain.
  canonical_label:   the realized name if known. NULL if anonymous.

ACTIVE STATE (durable + cognitive chain heads):
{active_state}

RECENT LOG:
{recent_log}

KNOWN LABELED CLUSTERS:
{known_clusters}

CONVERSATION WINDOW
-------------------
{window_block}

Emit JSON for TARGET turns only:
{{"entries": [{{"turn": <int>, "text": "...", "mentions": ["@..."],
   "subject": "@User", "predicate": "@User.boss",
   "cluster_id": "anon_boss_3", "canonical_label": null}}]}}

EMIT for DURABLE LIFE FACTS — facts that would still matter a month later:
  - Job, employer, title, team, role, boss, manager, mentor, colleague
  - Relationship: partner, friend, neighbor, family
  - Location: home, city
  - Possessions: car, bike, pet (any new car/bike/pet is durable)
  - Hobbies & recurring routines (started climbing, joined gym, biking to work)
  - School / education
  - Naming an entity that was previously anonymous

SKIP for FILLER (output {{"entries": []}} if all targets are filler):
  - Momentary body sensations, weather, time-of-day
  - This-instant feelings, one-off mundane actions
  - Notification noise (412 unread, etc.)

Output JSON ONLY.
"""


# ---------------------------------------------------------------------------
# Cognizer prompt (cognition emission after observation)
# ---------------------------------------------------------------------------


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

Schema is the same as observation entries plus `entry_type="cognition"`:
  text:               atomic prose ("@User expects @Sam to be his boss at @Notion")
  mentions:           @entity tags
  subject:            "@User" (cognition is User's mental state)
  predicate:          "@User.expectation" / "@User.plan" / "@User.belief" /
                      "@User.hope" / "@User.fear" / "@User.confirmation"
  cluster_id:         can be a fresh "cog_<topic>_<TURN>" id, OR reuse the
                      cluster_id of the entity the cognition is ABOUT (e.g.,
                      cluster_id="sam" for an expectation about Sam)
  canonical_label:    optional

ACTIVE STATE (durable + cognitive chain heads):
{active_state}

RECENT LOG (most recent committed entries):
{recent_log}

NEW OBSERVATIONS just emitted (the trigger for this cognition pass):
{new_observations}

CONVERSATION WINDOW (for context):
{window_block}

EMIT cognition entries ONLY when there's a non-trivial mental-state to capture:
  - Did the new observation FULFILL or CONTRADICT a prior expectation?
  - Does the new observation IMPLY a plan, fear, or hope?
  - Is User pointing at a future event with uncertainty?

DO NOT emit:
  - Trivial restatements of the observation itself
  - Generic "User now knows X" entries (the observation IS the knowledge)
  - Pure speculation with no grounding in the observed turns
  Output {{"entries": []}} if there's no genuine cognition to record.

Emit JSON:
{{"entries": [{{"turn": <int>, "text": "...", "mentions": ["@..."],
   "subject": "@User", "predicate": "@User.expectation",
   "cluster_id": "...", "canonical_label": null}}]}}
"""


def write_window(
    window_turns, target_turn_lo, target_turns, prior_entries, idx, cache, budget
):
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
        "write_prompt_chars": len(prompt),
        "window_size": len(window_turns),
    }
    if not isinstance(obj, dict):
        return [], telemetry, window_block
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
        if isinstance(canonical_label, str):
            canonical_label = canonical_label.strip() or None
        else:
            canonical_label = None
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
                entry_type="observation",
            )
        )
    telemetry["n_observations_emitted"] = len(entries)
    return entries, telemetry, window_block


def cognize_window(
    target_turns, new_observations, prior_entries, idx, window_block, cache, budget
):
    """Emit cognition entries based on the new observations + prior memory."""
    if not new_observations:
        return [], {"n_cognitions_emitted": 0, "skipped": "no_observations"}

    target_entities = extract_window_entities(target_turns)
    for e in new_observations:
        for m in e.mentions:
            target_entities.add(m)
    active_state = render_active_state(idx, target_entities)
    recent_log = render_recent_log(prior_entries, cap=8)

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
        meta = " ".join(bits)
        obs_lines.append(f"  [{e.uuid}] t={e.ts} :: {e.text}  ({meta})")
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
    entries: list[LogEntry] = []
    counter = 1000  # offset to avoid uuid collision with observation entries
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
        uuid = f"c{ts:04d}_{counter}"
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
                entry_type="cognition",
            )
        )
    telemetry["n_cognitions_emitted"] = len(entries)
    return entries, telemetry


# ---------------------------------------------------------------------------
# Retrieval with type-filter
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
    "what did user",
    "what does user",
    "what would user",
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


def retrieve(question, idx, cache, budget, top_k=12, include_cognition=None):
    if not idx.entries:
        return []
    if include_cognition is None:
        include_cognition = is_cognitive_question(question)

    q_emb = embed_batch([question], cache, budget)[0]
    q_ents = extract_question_entities(question)
    q_tags = [f"@{e}" for e in q_ents]

    def _allowed(e):
        if include_cognition:
            return True
        return e.entry_type == "observation"

    selected: set[str] = set()
    for (subj, pred), cid in idx.chain_head.items():
        if subj in q_tags:
            head_uuid = idx.chain_head_entry[(subj, pred)]
            head_entry = idx.by_uuid[head_uuid]
            if _allowed(head_entry):
                selected.add(head_uuid)
            for prior in idx.cluster_entries.get(cid, []):
                if _allowed(prior):
                    selected.add(prior.uuid)

    mention_candidates: list[str] = []
    for tag in q_tags:
        for u in idx.mention_index.get(tag, []):
            if _allowed(idx.by_uuid[u]):
                mention_candidates.append(u)
    selected.update(
        _rank_by_embedding(
            q_emb, list(set(mention_candidates)), idx.embed_by_uuid, top_k=top_k
        )
    )

    all_uuids = [e.uuid for e in idx.entries if _allowed(e)]
    selected.update(
        _rank_by_embedding(q_emb, all_uuids, idx.embed_by_uuid, top_k=top_k)
    )

    out = sorted(selected, key=lambda u: idx.by_uuid[u].ts)
    if len(out) > 60:
        out = out[-60:]
    return [idx.by_uuid[u] for u in out]


READ_PROMPT = """You are answering a question using a semantic memory.

Each entry has a cluster_id. Some entries are OBSERVATIONS (factual events
in User's life). Some are COGNITIONS (User's mental state — expectations,
plans, beliefs, fears) emitted by a separate cognition pass.

When the question asks about User's mental state, prefer cognition entries.
When the question asks about facts, prefer observations.

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
        bits.append(f"type={e.entry_type}")
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
# Ingestion driver — K-block centered + cognition pass
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

        new_observations, w_tele, window_block = write_window(
            window_turns,
            target_turn_lo,
            target_turns,
            log,
            idx,
            cache,
            budget,
        )
        log.extend(new_observations)

        new_cognitions = []
        c_tele = {}
        if enable_cognition and new_observations:
            new_cognitions, c_tele = cognize_window(
                target_turns,
                new_observations,
                log,
                idx,
                window_block,
                cache,
                budget,
            )
            log.extend(new_cognitions)

        tele = {
            **w_tele,
            **c_tele,
            "fire_no": fire_no,
            "last_turn": target_turns[-1][0],
        }
        telemetry.append(tele)

        if fire_no % rebuild_index_every == 0:
            idx, _ = build_index(log, cache, budget)
        fire_no += 1
        target_lo = target_hi

    idx, resolutions = build_index(log, cache, budget)
    return log, resolutions, idx, telemetry
