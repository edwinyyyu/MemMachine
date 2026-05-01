"""AEN-2 VARIABLE-BINDING.

Variable-binding framing for anonymous entities:

  Anonymous descriptor "my new boss started this week" creates an ENTITY
  cluster_id_X with no canonical label. The @User.boss chain points at
  cluster_X.

  Later: "his name is Marcus" emits a RESOLUTION binding cluster_X to
  canonical_label="Marcus". NO new chain entry. The chain still points at
  cluster_X; the chain just acquired a label.

  Retrieval: chain → latest entry → cluster_id → latest resolution → label
  substitution at read time.

This is framing A (existential variable assigned to realized entity), NOT
framing B (named-fact with descriptor as a property). Framing B is what
aen1_simple/sliding/active do via the `refs` field, and it conflates two
separate operations (FACT update vs LABEL binding).

Data model:
  LogEntry(uuid, ts, cluster_id, text, subject, predicate, world, mentions, refs)
  Resolution(uuid, ts, cluster_id, canonical_label, evidence_entry_uuids,
             supersedes_resolution_id)

Indexes:
  by_uuid:           uuid -> entry
  cluster_entries:   cluster_id -> [entries] in chronological order
  cluster_label:     cluster_id -> latest canonical_label (or None)
  chain_head:        (subject, predicate, world) -> latest cluster_id
  chain_head_entry:  (subject, predicate, world) -> latest entry uuid
  mention_index:     @entity -> [entry uuids]
  embed_by_uuid:     uuid -> embedding
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
    world: str = "real"
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
    chain_head: dict[tuple[str, str, str], str]
    chain_head_entry: dict[tuple[str, str, str], str]
    mention_index: dict[str, list[str]]
    embed_by_uuid: dict[str, list[float]]


def build_index(
    entries: list[LogEntry],
    resolutions: list[Resolution],
    cache: Cache,
    budget: Budget,
) -> IndexedLog:
    by_uuid = {e.uuid: e for e in entries}

    cluster_entries: dict[str, list[LogEntry]] = {}
    for e in entries:
        cluster_entries.setdefault(e.cluster_id, []).append(e)

    # Latest resolution per cluster (by ts, then by uuid for stability).
    cluster_label: dict[str, str | None] = dict.fromkeys(cluster_entries)
    res_by_cluster: dict[str, list[Resolution]] = {}
    for r in resolutions:
        res_by_cluster.setdefault(r.cluster_id, []).append(r)
    for cid, rs in res_by_cluster.items():
        rs.sort(key=lambda r: (r.ts, r.uuid))
        cluster_label[cid] = rs[-1].canonical_label

    # Chain head: latest entry per (subject, predicate, world).
    chain_head: dict[tuple[str, str, str], str] = {}
    chain_head_entry: dict[tuple[str, str, str], str] = {}
    for e in sorted(entries, key=lambda x: (x.ts, x.uuid)):
        if e.subject and e.predicate:
            key = (e.subject, e.predicate, e.world)
            chain_head[key] = e.cluster_id
            chain_head_entry[key] = e.uuid

    mention_index: dict[str, list[str]] = {}
    for e in entries:
        for m in e.mentions:
            mention_index.setdefault(m, []).append(e.uuid)

    texts = [e.text for e in entries]
    embs = embed_batch(texts, cache, budget) if entries else []
    embed_by_uuid = {e.uuid: embs[i] for i, e in enumerate(entries)}

    return IndexedLog(
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


# ---------------------------------------------------------------------------
# Active-state rendering
# ---------------------------------------------------------------------------


def render_active_state(idx: IndexedLog | None, target_entities: set[str]) -> str:
    """Render chain heads relevant to the target entities + a few global heads."""
    if idx is None:
        return "(empty)"
    lines: list[str] = []
    seen_keys: set[tuple[str, str, str]] = set()
    # Heads whose subject is mentioned in target_entities
    relevant: list[tuple[tuple[str, str, str], str]] = []
    for key, cid in idx.chain_head.items():
        subj = key[0]
        if subj in target_entities or subj == "@User":
            relevant.append((key, cid))

    # Sort by latest entry ts desc
    def _ts_for(key: tuple[str, str, str]) -> int:
        return idx.by_uuid[idx.chain_head_entry[key]].ts

    relevant.sort(key=lambda kv: _ts_for(kv[0]), reverse=True)

    for (subj, pred, world), cid in relevant[:30]:
        seen_keys.add((subj, pred, world))
        head_uuid = idx.chain_head_entry[(subj, pred, world)]
        head_entry = idx.by_uuid[head_uuid]
        label = idx.cluster_label.get(cid)
        label_str = f"label={label!r}" if label else "label=<unnamed>"
        lines.append(
            f"  {pred}[{world}]: cluster={cid} {label_str}  "
            f"head=[{head_uuid} t={head_entry.ts}] :: {head_entry.text[:80]}"
        )
    return "\n".join(lines) if lines else "(none)"


def render_recent_log(prior_entries: list[LogEntry], cap: int = 10) -> str:
    if not prior_entries:
        return "(empty)"
    recent = list(reversed(prior_entries[-cap:]))
    lines = []
    for e in recent:
        subj = f"subject={e.subject}" if e.subject else ""
        pred = f"pred={e.predicate}" if e.predicate else ""
        meta = " ".join(x for x in [subj, pred] if x)
        lines.append(f"  [{e.uuid}] t={e.ts} cluster={e.cluster_id} {meta} :: {e.text}")
    return "\n".join(lines)


def extract_window_entities(turns: list[tuple[int, str]]) -> set[str]:
    """Capitalized-token heuristic, plus @User."""
    text = " ".join(t for _, t in turns)
    tokens = re.findall(r"\b([A-Z][a-z]{1,20})\b", text)
    ents = {f"@{t}" for t in tokens}
    ents.add("@User")
    return ents


# ---------------------------------------------------------------------------
# Writer prompt
# ---------------------------------------------------------------------------


WRITE_PROMPT = """You are a semantic-memory writer using VARIABLE-BINDING for entities.

You emit two kinds of items:

(1) ENTRY — a new atomic fact about a chain.
  Fields:
    text:        atomic prose fact, includes @entity tags
    mentions:    list of @entity tags appearing in the text
    subject:     "@entity" the predicate is about (often "@User")
    predicate:   "@subject.predicate_name" (lowercase predicate). Optional.
    world:       "real" or "non_real"
    cluster_id:  identifier for the OBJECT of the predicate.
                 - For an anonymous entity (e.g. "my new boss" with no name yet),
                   generate a fresh ID like "anon_boss_<TURN>" using the target turn.
                 - If extending an existing chain (e.g., the boss is now changed
                   - new person), generate a new "anon_<topic>_<TURN>" or use a
                   known label if the new entity is already named.
                 - For named entities, use the lowercased surface (e.g.,
                   cluster_id="marcus").
    refs:        prior entry uuids ONLY if this entry CLARIFIES or UPDATES a
                 prior FACT. Do NOT use refs to bind a name to a descriptor —
                 use a RESOLUTION for that.

(2) RESOLUTION — binding a name to an existing anonymous cluster.
  Fields:
    cluster_id:               existing cluster_id from ACTIVE STATE (must match exactly)
    canonical_label:          the realized name (e.g., "Marcus")
    evidence_entry_uuids:     uuids of entries that support this binding

ACTIVE STATE (current chain heads in this window):
{active_state}

RECENT LOG (for context):
{recent_log}

KNOWN CLUSTERS (cluster_id → label):
{known_clusters}

CONVERSATION WINDOW
-------------------
{window_block}

Emit JSON with TARGET-TURN items only:
{{
  "items": [
    {{"type": "entry", "turn": <int from TARGET turns>, "text": "...",
      "mentions": ["@..."], "subject": "@User", "predicate": "@User.boss",
      "world": "real", "cluster_id": "anon_boss_3", "refs": []}},
    {{"type": "resolution", "turn": <int from TARGET turns>,
      "cluster_id": "anon_boss_3", "canonical_label": "Marcus",
      "evidence_entry_uuids": ["e0003_0"]}}
  ]
}}

RULES
- Emit items only for TARGET turns. CONTEXT turns are read-only — they're
  there for coreference and binding lookahead.
- Anonymous descriptor in a slot ("my new boss started this week"):
    → ENTRY with cluster_id="anon_<topic>_<TURN>". subject=@User. predicate=@User.boss.
- Name reveal for an anonymous cluster ("his name is Marcus"):
    → RESOLUTION binding the cluster_id to canonical_label="Marcus". NO new entry.
- Named entity introduced fresh (no anon slot, e.g., "Marcus is in town"):
    → ENTRY with cluster_id="marcus" + RESOLUTION binding cluster_id="marcus"
      to canonical_label="Marcus" (so future references know Marcus is labeled).
- Existing labeled cluster mentioned again with new info:
    → ENTRY with cluster_id matching the known label.
- Chain transition (e.g., User got a NEW boss): ENTRY with a NEW anon cluster_id.
- Filler turns (weather/chitchat): output no items for them.
- Output JSON ONLY.
"""


def write_window(
    window_turns: list[tuple[int, str]],
    target_turn_lo: int,
    target_turns: list[tuple[int, str]],
    prior_entries: list[LogEntry],
    idx: IndexedLog | None,
    cache: Cache,
    budget: Budget,
) -> tuple[list[LogEntry], list[Resolution], dict]:
    target_entities = extract_window_entities(target_turns)
    active_state = render_active_state(idx, target_entities)
    recent_log = render_recent_log(prior_entries, cap=10)

    # Known clusters render
    if idx is None:
        known_clusters_str = "(none)"
    else:
        known_pairs = []
        for cid, label in idx.cluster_label.items():
            if label is not None:
                known_pairs.append(f"{cid}={label!r}")
        known_clusters_str = "; ".join(known_pairs[:40]) or "(none labeled yet)"

    window_lines = []
    in_target = False
    for tidx, text in window_turns:
        if not in_target and tidx >= target_turn_lo:
            window_lines.append("--- TARGET TURNS ---")
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
        return [], [], telemetry

    target_turn_set = {t for t, _ in target_turns}
    items = obj.get("items", []) or []
    entries: list[LogEntry] = []
    resolutions: list[Resolution] = []
    counter = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        kind = it.get("type")
        ts_raw = it.get("turn")
        try:
            ts = int(ts_raw) if ts_raw is not None else target_turns[-1][0]
        except (TypeError, ValueError):
            ts = target_turns[-1][0]
        if ts not in target_turn_set:
            ts = target_turns[-1][0]

        if kind == "entry":
            text = (it.get("text") or "").strip()
            if not text:
                continue
            cluster_id = (it.get("cluster_id") or "").strip()
            if not cluster_id:
                continue
            mentions = [m for m in (it.get("mentions") or []) if isinstance(m, str)]
            subject = it.get("subject") or None
            predicate = it.get("predicate") or None
            world = (it.get("world") or "real").strip() or "real"
            refs = [r for r in (it.get("refs") or []) if isinstance(r, str)]
            uuid = f"e{ts:04d}_{counter}"
            counter += 1
            entries.append(
                LogEntry(
                    uuid=uuid,
                    ts=ts,
                    cluster_id=cluster_id,
                    text=text,
                    subject=subject if isinstance(subject, str) else None,
                    predicate=predicate if isinstance(predicate, str) else None,
                    world=world,
                    mentions=mentions,
                    refs=refs,
                )
            )
        elif kind == "resolution":
            cluster_id = (it.get("cluster_id") or "").strip()
            label = (it.get("canonical_label") or "").strip()
            if not cluster_id or not label:
                continue
            ev = [
                u for u in (it.get("evidence_entry_uuids") or []) if isinstance(u, str)
            ]
            uuid = f"r{ts:04d}_{counter}"
            counter += 1
            resolutions.append(
                Resolution(
                    uuid=uuid,
                    ts=ts,
                    cluster_id=cluster_id,
                    canonical_label=label,
                    evidence_entry_uuids=ev,
                )
            )
    telemetry["n_entries_emitted"] = len(entries)
    telemetry["n_resolutions_emitted"] = len(resolutions)
    return entries, resolutions, telemetry


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
    selected: set[str] = set()

    # Pull all chain heads whose subject matches a question entity.
    for (subj, pred, world), cid in idx.chain_head.items():
        if subj in q_tags:
            head_uuid = idx.chain_head_entry[(subj, pred, world)]
            selected.add(head_uuid)
            for prior in idx.cluster_entries.get(cid, []):
                selected.add(prior.uuid)

    # Embedding ranks over mention-filtered + global candidates.
    mention_candidates: list[str] = []
    for tag in q_tags:
        mention_candidates.extend(idx.mention_index.get(tag, []))
    mention_topk = _rank_by_embedding(
        q_emb, list(set(mention_candidates)), idx.embed_by_uuid, top_k=top_k
    )
    selected.update(mention_topk)

    all_uuids = [e.uuid for e in idx.entries]
    full_topk = _rank_by_embedding(q_emb, all_uuids, idx.embed_by_uuid, top_k=top_k)
    selected.update(full_topk)

    out = sorted(selected, key=lambda u: idx.by_uuid[u].ts)
    MAX = 60
    if len(out) > MAX:
        out = out[-MAX:]
    return [idx.by_uuid[u] for u in out]


READ_PROMPT = """You are answering a question about User's life using a
variable-binding semantic memory. Each entry has a cluster_id (entity
identity); some clusters have a canonical_label, others are anonymous (label
=<unnamed>).

When an entry mentions a cluster_id, look up the cluster_label table to
substitute the actual name if known.

CLUSTER LABELS (cluster_id -> label, omitted if unknown):
{cluster_labels}

RETRIEVED ENTRIES (chronological):
{entries_block}

QUESTION: {question}

Answer concisely. For yes/no questions, start your answer with "Yes" or "No".
"""


def format_entries_for_read(entries: list[LogEntry]) -> str:
    lines = []
    for e in entries:
        meta = []
        if e.subject:
            meta.append(f"subject={e.subject}")
        if e.predicate:
            meta.append(f"pred={e.predicate}")
        meta.append(f"cluster={e.cluster_id}")
        meta_str = " ".join(meta)
        lines.append(f"[{e.uuid}] t{e.ts} :: {e.text}  ({meta_str})")
    return "\n".join(lines)


def answer_question(
    question: str,
    idx: IndexedLog,
    cache: Cache,
    budget: Budget,
    top_k: int = 12,
) -> str:
    retrieved = retrieve(question, idx, cache, budget, top_k=top_k)
    block = format_entries_for_read(retrieved)
    label_lines = []
    for cid, label in idx.cluster_label.items():
        if label:
            label_lines.append(f"  {cid} -> {label}")
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
    resolutions: list[Resolution] = []
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

        new_entries, new_res, tele = write_window(
            window_turns,
            target_turn_lo,
            target_turns,
            log,
            idx,
            cache,
            budget,
        )
        log.extend(new_entries)
        resolutions.extend(new_res)
        tele["fire_no"] = fire_no
        tele["last_turn"] = target_turns[-1][0]
        tele["w_past_actual"] = target_lo - win_lo
        tele["w_future_actual"] = win_hi - target_hi
        tele["k_actual"] = target_hi - target_lo
        telemetry.append(tele)

        if fire_no % rebuild_index_every == 0:
            idx = build_index(log, resolutions, cache, budget)
        fire_no += 1
        target_lo = target_hi

    idx = build_index(log, resolutions, cache, budget)
    return log, resolutions, idx, telemetry
