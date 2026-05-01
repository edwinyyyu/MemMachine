"""AEN-7 RECURSIVE — unified write→retrieve→write loop.

Built on aen6_prose_v2 (DSU + prose facts). Replaces R22's trigger-gated
cognition with budget-bounded recursive reflection.

Loop shape per K-block fire:
    initial = writer(target, active, retrieval_for_window, scratchpad=[])
    store all; scratchpad = list(initial); queue = list(initial)
    budget = REFLECTION_BUDGET (default 2), max_used = REFLECTION_MAX (default 3)
    while budget > 0 and queue:
        seed = queue.pop(0)
        related = retrieve(seed.text)        # spans BOTH observations + cognition
        if not related: continue
        refl = reflector(seed, related, scratchpad)
        budget -= 1
        if refl.facts:
            store; scratchpad += refl.facts; queue += refl.facts
        if refl.continue_thinking and used < max_used:
            budget = min(budget+1, max_used - used)

Properties:
  - No explicit triggers — the LLM decides whether to emit based on what
    retrieval surfaces, not on categorical keywords.
  - Scratchpad prevents re-derivation: every reflector prompt includes facts
    already emitted this turn (writer + prior reflections).
  - Reflection facts go to collection="cognition"; observations to
    "observations". Retrieval defaults to BOTH collections (simulating free
    associative recall over direct experience and prior thinking).
  - Budget is dynamic: empty reflections auto-stop, continue_thinking=true
    extends up to a hard cap.

Reused from aen6_prose_v2 (imported, not copied):
  - Schema: Mention, Fact, BindingEvent, EntityRegistry, IndexedCollection,
    MemoryStore
  - Index machinery: build_collection, _normalize_surface
  - Active rendering: render_active_entities, render_recent_facts,
    extract_window_surfaces
  - Writer: WRITE_PROMPT, write_window
  - Retrieval primitives: extract_question_surfaces, _rank_by_embedding,
    retrieve, format_facts_for_read, format_resolution_map
"""

from __future__ import annotations

import sys
import uuid as _uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND24 = HERE.parent
RESEARCH = ROUND24.parent
ROUND23 = RESEARCH / "round23_prose_facts"
ROUND7 = RESEARCH / "round7"

sys.path.insert(0, str(ROUND23 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen6_prose_v2 as v2  # noqa: E402
from _common import Budget, Cache, cosine, extract_json, llm  # noqa: E402

# Re-export schema + helpers for convenience
Mention = v2.Mention
Fact = v2.Fact
BindingEvent = v2.BindingEvent
EntityRegistry = v2.EntityRegistry
IndexedCollection = v2.IndexedCollection
MemoryStore = v2.MemoryStore
build_collection = v2.build_collection
write_window = v2.write_window
format_facts_for_read = v2.format_facts_for_read
format_resolution_map = v2.format_resolution_map
READ_PROMPT = v2.READ_PROMPT


# ---------------------------------------------------------------------------
# Retrieval — wraps v2.retrieve, then enforces a PER-COLLECTION recency cap
# so cognition cannot displace older observations (the dense-history failure).
# ---------------------------------------------------------------------------


DEDUP_PROMPT = """For each fact pair below, decide if A and B are RESTATEMENTS
of the EXACT SAME EVENT — just paraphrased differently. Be VERY CONSERVATIVE.
False merge is worse than redundant facts. When in doubt, output false.

OUTPUT TRUE only if:
- A and B describe the SAME single event/fact with same actors, same
  values, same time
- One is just a rephrasing (different words, same meaning, same details)
- e.g., "User ate steak yesterday" and "Yesterday User had steak" → TRUE
- e.g., "Alice fixed the deploy bug" and "Alice fixed a deploy bug last
  sprint" → TRUE if context confirms same incident

OUTPUT FALSE if ANY of these apply:
- Argument roles differ ("Alice paid Bob" vs "Bob paid Alice")
- Different values for the same role (User's manager is Marcus / is Alice
  → DIFFERENT, these are two different points in time)
- Different titles, jobs, locations, possessions, hobbies
  (e.g., "User is a senior engineer" vs "User is a principal engineer"
  → FALSE: different job titles even if same person)
- Outcomes differ (planned vs done, success vs failure)
- Different details (different dates, different cities, different events)
- Same topic but distinct events (3 different cache bugs at different
  layers are 3 distinct events, not 1)
- One is a CHANGE/UPDATE to the other (Marcus is manager → Alice is now
  manager). State transitions are NEVER duplicates.
- Different timestamps or "this is happening" vs "this happened"
- One mentions an attribute the other doesn't (e.g., adds a new detail)

KEY HEURISTIC: if you'd want to keep BOTH facts in a chronicle of events,
output FALSE. Only output TRUE if one fact is literally redundant given
the other.

Output JSON ONLY: {{"dup": [<bool>, <bool>, ...]}} — one bool per pair, in order.

PAIRS:
{pairs_block}
"""


def _llm_dedup_filter(facts, store, cache, budget, threshold=0.80):
    """Cosine prefilter + LLM judge for near-duplicate detection.

    Cosine alone is unreliable (argument-order flips look near-identical).
    LLM alone is too expensive (O(N^2) calls). Hybrid: cosine pre-filters
    candidate pairs at threshold; one batched LLM call decides.
    """
    if len(facts) < 2:
        return facts

    embs: list = []
    for f in facts:
        idx = store.collections.get(f.collection)
        emb = idx.embed_by_uuid.get(f.fact_uuid) if idx else None
        embs.append(emb)

    candidate_pairs: list[tuple[int, int]] = []
    for i in range(len(facts)):
        if embs[i] is None:
            continue
        for j in range(i + 1, len(facts)):
            if embs[j] is None:
                continue
            if cosine(embs[i], embs[j]) > threshold:
                candidate_pairs.append((i, j))

    if not candidate_pairs:
        return facts

    pairs_lines = []
    for k, (i, j) in enumerate(candidate_pairs):
        pairs_lines.append(f"Pair {k}:\n  A: {facts[i].text}\n  B: {facts[j].text}")
    prompt = DEDUP_PROMPT.format(pairs_block="\n\n".join(pairs_lines))
    raw = llm(prompt, cache, budget, reasoning_effort="medium")
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return facts
    decisions = obj.get("dup")
    if not isinstance(decisions, list) or len(decisions) != len(candidate_pairs):
        return facts

    parent = list(range(len(facts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for k, (i, j) in enumerate(candidate_pairs):
        if k < len(decisions) and bool(decisions[k]):
            union(i, j)

    seen_groups: set = set()
    kept = []
    for i, f in enumerate(facts):
        g = find(i)
        if g in seen_groups:
            continue
        seen_groups.add(g)
        kept.append(f)
    return kept


def retrieve(
    question,
    store,
    cache,
    budget,
    top_k=14,
    collections=None,
    obs_cap=50,
    cog_cap=15,
    llm_dedup=False,
    expand_hops=0,
    expand_top_k=6,
):
    """Hybrid retrieval: surface-match keeps EVERYTHING (entity-relevant facts
    in any time period); kNN top-k caps the semantic-only matches.

    Why: history-aggregation queries on long sessions ("ever in Chicago?",
    "first manager?") need mid-history obs facts that surface-match the
    queried entity. A chronological cap drops these even when they were
    found by surface match. Solution: don't cap surface-matched facts —
    only cap the kNN-only additions.

    `llm_dedup`: if True, after exact-text dedup, run cosine-prefilter +
    LLM-judge near-dup pass. Use for read-time queries (1 extra LLM call
    per question). Skip for reflection-time retrieval (recursive cost).

    `expand_hops`: number of multi-hop expansion passes. Each hop adds
    facts about the top-K entities mentioned in the current retrieved set.
    Helps queries like "User's boss's mentor's son" where the answer entity
    isn't surfaced in the question. expand_top_k controls how many entities
    per hop are expanded.
    """
    from _common import embed_batch  # local import; same module as v2

    if collections is None:
        collections = (
            ["observations", "cognition"]
            if "cognition" in store.collections
            else ["observations"]
        )
    q_emb = embed_batch([question], cache, budget)[0]
    surfaces = v2.extract_question_surfaces(question)
    norm_surfaces = [v2._normalize_surface(s) for s in surfaces]

    # Per-collection: surface-matched (uncapped) and knn-only (capped)
    surface_by_col: dict[str, dict] = {c: {} for c in collections}
    knn_only_by_col: dict[str, dict] = {c: {} for c in collections}

    for col_name in collections:
        idx = store.collections.get(col_name)
        if idx is None or not idx.facts:
            continue
        # Surface match — uncapped
        for nsurf in norm_surfaces:
            if not nsurf:
                continue
            for surface_norm, mids in idx.mentions_by_surface.items():
                if nsurf in surface_norm or surface_norm in nsurf:
                    for mid in mids:
                        eid = store.registry.get_canonical(mid)
                        for fu in idx.facts_by_entity.get(eid, []):
                            f = idx.by_uuid.get(fu)
                            if f:
                                surface_by_col[col_name][fu] = f
        # kNN — only add facts not already in surface set
        all_uuids = [f.fact_uuid for f in idx.facts]
        for fu in v2._rank_by_embedding(
            q_emb, all_uuids, idx.embed_by_uuid, top_k=top_k
        ):
            if fu in surface_by_col[col_name]:
                continue
            f = idx.by_uuid[fu]
            knn_only_by_col[col_name][fu] = f

    # Multi-hop expansion: add facts about entities frequently mentioned in
    # the current retrieval set. Helps "X's Y's Z" chain queries where the
    # answer entity (Z) isn't named in the question.
    for hop in range(expand_hops):
        ent_freq: dict[str, int] = {}
        # Aggregate entity frequency across both surface and knn sets so far
        for col_name in collections:
            for col_dict in (surface_by_col[col_name], knn_only_by_col[col_name]):
                for f in col_dict.values():
                    seen_in_fact: set[str] = set()
                    for mid in f.mention_ids:
                        eid = store.registry.get_canonical(mid)
                        if eid in seen_in_fact:
                            continue
                        seen_in_fact.add(eid)
                        ent_freq[eid] = ent_freq.get(eid, 0) + 1
        # Skip the User entity and any entity already heavily represented
        ent_freq.pop("e_user", None)
        # Pick top-K entities by frequency
        top_entities = sorted(ent_freq.items(), key=lambda kv: -kv[1])[:expand_top_k]
        if not top_entities:
            break
        added_any = False
        for col_name in collections:
            idx = store.collections.get(col_name)
            if idx is None:
                continue
            for eid, _ in top_entities:
                for fu in idx.facts_by_entity.get(eid, []):
                    if (
                        fu in surface_by_col[col_name]
                        or fu in knn_only_by_col[col_name]
                    ):
                        continue
                    f = idx.by_uuid.get(fu)
                    if f:
                        surface_by_col[col_name][fu] = f
                        added_any = True
        if not added_any:
            break

    # Cap kNN-only chronologically per collection; keep all surface-matched.
    # EXACT-TEXT dedup only — embedding cosine is unreliable for argument-order
    # ("Alice paid Bob" vs "Bob paid Alice" have near-identical embeddings).
    # Near-duplicate paraphrases (same meaning, different wording) are NOT
    # deduped; they pass through as separate facts and the LLM reader is
    # responsible for handling them. This is intentional: false-merge is
    # worse than redundant retrieval.
    def _normalize_text(s: str) -> str:
        return " ".join(s.lower().split())[:200]

    capped: list = []
    seen_texts: set[str] = set()

    for col_name in collections:
        surface_facts = list(surface_by_col[col_name].values())
        knn_facts = sorted(
            knn_only_by_col[col_name].values(), key=lambda f: (f.ts, f.fact_uuid)
        )
        cap = obs_cap if col_name == "observations" else cog_cap

        # Surface-matched: dedup, earliest-wins (chronological)
        for f in sorted(surface_facts, key=lambda f: (f.ts, f.fact_uuid)):
            nt = _normalize_text(f.text)
            if nt in seen_texts:
                continue
            seen_texts.add(nt)
            capped.append(f)

        # kNN-only: dedup-then-cap
        kept_knn = []
        for f in knn_facts:
            nt = _normalize_text(f.text)
            if nt in seen_texts:
                continue
            seen_texts.add(nt)
            kept_knn.append(f)
        capped.extend(kept_knn[-cap:])

    capped.sort(key=lambda f: (f.ts, f.fact_uuid))

    if llm_dedup:
        capped = _llm_dedup_filter(capped, store, cache, budget)

    resolution_map: dict[str, set] = {}
    for f in capped:
        for mid in f.mention_ids:
            eid = store.registry.get_canonical(mid)
            resolution_map.setdefault(eid, set()).add(mid)
    return capped, resolution_map


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------


REFLECT_PROMPT = """You are a semantic-memory REFLECTOR. You emit COGNITION
facts — entries about User's MENTAL STATE — separate from observation facts.

OBSERVATIONS describe REALITY: "User's manager is Marcus.", "User lives in
Boston.", "User picked up a Bianchi." Verbs are factual: IS, HAS, LIVES,
WORKS, OWNS, DRIVES, etc.

COGNITIONS describe MENTAL STATE: "User expects Sam to be his boss when he
joins Notion.", "User plans to move to Berlin.", "User's earlier hope to
work at Anthropic has now come true.", "User felt frustrated by the third
deploy failure this week.", "Marcus believed the migration would go smoothly."
Verbs are mental-state:
  - EXPECTS / EXPECTED — what someone predicts/predicted given prior evidence
  - PLANS / INTENDS — actions someone intends to take
  - BELIEVES / BELIEVED — things someone holds as true (when not directly asserted as fact)
  - HOPES — desired outcomes (uncertain)
  - FEARS — feared outcomes (uncertain)
  - CONFIRMS — a prior plan/expectation that just became true
  - CONTRADICTS / WAS_WRONG_ABOUT — a prior expectation that just became false
  - FELT (frustrated / angry / joyful / relieved / anxious / sad / grateful /
    excited / disappointed / proud / overwhelmed / lonely / surprised) —
    emotional response to an event. Extract from TONE even when no explicit
    "I feel X" appears. Emotional cues: ALL CAPS / repeated punctuation /
    "ugh" "argh" "WHAT" "FINALLY" / hyperbole / sighs / exclamation chains.
  - WORRIED_ABOUT — ongoing concern (vs. acute fear)

A cognition ALWAYS attributes a mental state to a person. It is never a
flat factual assertion. Compare:
  WRONG (this is observation territory): "User's boss is Sam."
  RIGHT  (this is cognition):             "User EXPECTED Sam to be his boss."
  WRONG: "Marcus is no longer the manager."
  RIGHT: "User CONFIRMS the transition away from Marcus." (only when triggered
         by an explicit confirmation cue from User)

DEFAULT IS TO EMIT NOTHING. Most reflections produce zero new facts.

EMIT a cognition fact ONLY when one of these triggers fires:

  TRIGGER 1 — CONDITIONAL: a turn states "if X then Y" or describes a plan
    contingent on something.
    => emit "User plans/expects <consequence> if <condition>" or similar.

  TRIGGER 2 — CONFIRMATION: a new observation matches a prior conditional
    or plan visible in related memories.
    => emit "User's earlier plan/expectation about <X> has been confirmed
       by <observation>."

  TRIGGER 3 — CONTRADICTION: a new observation contradicts a prior fact,
    claim, or expectation visible in related memories. Also fires for
    SILENT contradictions — when User's seed fact is incompatible with a
    prior CLAIM (e.g., prior: "I'm vegetarian"; seed: "Had a steak"), even
    when User does not explicitly retract.
    => emit "User contradicts their earlier <claim> with <new fact>."

  TRIGGER 4 — NAMED HOPE/FEAR: User explicitly states a hope or fear.
    => emit "User hopes/fears <X>."

  TRIGGER 5 — EMOTIONAL STATE: a turn carries a clear emotional tone — joy,
    frustration, anxiety, relief, sadness, anger, gratitude, pride,
    disappointment, surprise. Detect from TONE and word choice (caps,
    "ugh"/"argh"/"WHAT"/"FINALLY", exclamation chains, hyperbole), not
    only explicit "I feel X" statements. Skip if WRITER already emitted a
    named-emotion fact for this event (no need to duplicate).
    => emit "User felt <emotion> about <event>." Choose ONE primary emotion.

If none of these triggers fires, emit nothing.

DO NOT EMIT — these are the failure modes:
  - NEVER emit a cognition with a FACTUAL verb. Cognitions own
    expects/plans/hopes/fears/confirms/contradicts/believes; observations own
    is/has/lives/works/manages/owns/drives. If your text reads as a factual
    claim about manager/employer/location/team/partner/title/possession,
    REFUSE — that's the observation writer's job.
  - DO NOT emit negations of prior facts ("X is no longer the manager"). State
    changes are carried by new affirmative observations; cognition does not
    duplicate them.
  - DO NOT recombine attributes across entities ("Both X and Y are into Z").
  - DO NOT restate the seed in alternate phrasing.
  - DO NOT emit anything substantively similar to a fact in RELATED MEMORIES.
  - DO NOT speculate about implications the seed doesn't directly warrant.

If unsure, EMIT NOTHING.

CONTINUE_THINKING flag:
  - true ONLY when you emitted at least one fact AND that fact obviously
    chains to ANOTHER inference (rare).
  - false otherwise (default).

SEED FACT (just emitted; reflect on this):
{seed_fact}

RELATED PRIOR MEMORIES (what retrieval surfaced from existing memory):
{related_facts}

ENTITY RESOLUTIONS (mention -> canonical entity):
{resolution_map}

ALREADY EMITTED THIS TURN (scratchpad — DO NOT RESTATE these):
{scratchpad}

Schema:
{{
  "new_facts": [
    {{
      "text": "<atomic prose sentence>",
      "mentions": [
        {{"surface": "<literal text token>", "resolves_to": "<entity_id or 'new'>"}}
      ]
    }}
  ],
  "continue_thinking": <bool>
}}

If nothing new: {{"new_facts": [], "continue_thinking": false}}.

Output JSON ONLY.
"""


def _materialize_reflection(
    facts_raw: list,
    seed_ts: int,
    store: MemoryStore,
    fire_idx: int,
    refl_step: int,
) -> tuple[list[Fact], list[Mention], list[BindingEvent]]:
    """Convert reflector JSON output into Fact/Mention objects with DSU updates."""
    new_facts: list[Fact] = []
    new_mentions: list[Mention] = []
    new_bindings: list[BindingEvent] = []
    fact_counter = 0

    for fr in facts_raw:
        if not isinstance(fr, dict):
            continue
        text = (fr.get("text") or "").strip()
        if not text:
            continue
        ts = seed_ts
        suffix = _uuid.uuid4().hex[:6]
        fact_uuid = f"r{ts:04d}_{fire_idx}_{refl_step}_{fact_counter}_{suffix}"
        fact_counter += 1

        mention_specs = fr.get("mentions") or []
        local_resolutions: list[tuple[str, str]] = []
        intra_resolve: dict[str, str] = {}

        for i, ms in enumerate(mention_specs):
            if not isinstance(ms, dict):
                continue
            surface = (ms.get("surface") or "").strip()
            if not surface:
                continue
            resolves_to = (ms.get("resolves_to") or "new").strip()
            mention_id = (
                f"mr{ts:04d}_{fire_idx}_{refl_step}_{fact_counter}_{i}_{suffix}"
            )
            new_mentions.append(
                Mention(
                    mention_id=mention_id,
                    surface=surface,
                    fact_uuid=fact_uuid,
                    ts=ts,
                )
            )
            store.registry.register(mention_id)
            local_resolutions.append((mention_id, resolves_to))

        for mention_id, resolves_to in local_resolutions:
            if resolves_to == "new" or not resolves_to:
                continue
            if resolves_to in store.registry.entity_members:
                existing_member = next(iter(store.registry.entity_members[resolves_to]))
                store.registry.merge(
                    mention_id,
                    existing_member,
                    ts=ts,
                    evidence_fact_uuids=[fact_uuid],
                    rationale=f"reflector resolved to {resolves_to}",
                )
                new_bindings.append(store.registry.binding_events[-1])
            elif resolves_to in intra_resolve:
                store.registry.merge(
                    mention_id,
                    intra_resolve[resolves_to],
                    ts=ts,
                    evidence_fact_uuids=[fact_uuid],
                    rationale="intra-fact coref (reflection)",
                )
                new_bindings.append(store.registry.binding_events[-1])
            else:
                intra_resolve[resolves_to] = mention_id

        new_facts.append(
            Fact(
                fact_uuid=fact_uuid,
                ts=ts,
                text=text,
                mention_ids=[m for m, _ in local_resolutions],
                collection="cognition",
            )
        )

    return new_facts, new_mentions, new_bindings


def _format_scratchpad(scratchpad: list[Fact], store: MemoryStore) -> str:
    if not scratchpad:
        return "(empty)"
    lines = []
    for f in scratchpad:
        tag = "OBS" if f.collection == "observations" else "COG"
        eids = []
        for mid in f.mention_ids:
            eid = store.registry.get_canonical(mid)
            if eid not in eids:
                eids.append(eid)
        eid_block = f" entities=[{', '.join(eids)}]" if eids else ""
        lines.append(f"  [{f.fact_uuid} t={f.ts} {tag}]{eid_block} {f.text[:140]}")
    return "\n".join(lines)


def reflect_on_fact(
    seed: Fact,
    store: MemoryStore,
    scratchpad: list[Fact],
    cache: Cache,
    budget: Budget,
    fire_idx: int,
    refl_step: int,
    top_k: int = 8,
) -> tuple[list[Fact], list[Mention], list[BindingEvent], bool]:
    """Run one reflection step. Returns (new_facts, new_mentions, bindings, continue_thinking)."""
    related, resolution_map = retrieve(
        seed.text,
        store,
        cache,
        budget,
        top_k=top_k,
        collections=["observations", "cognition"],
    )
    # Drop the seed itself; drop scratchpad facts (they're already in scratchpad block)
    scratch_uuids = {f.fact_uuid for f in scratchpad}
    related = [
        f
        for f in related
        if f.fact_uuid != seed.fact_uuid and f.fact_uuid not in scratch_uuids
    ]
    if not related:
        return [], [], [], False

    seed_str = f"[{seed.fact_uuid} t={seed.ts}] {seed.text}"
    related_str = format_facts_for_read(related, store)
    resolution_str = format_resolution_map(resolution_map, store)
    scratchpad_str = _format_scratchpad(scratchpad, store)

    prompt = REFLECT_PROMPT.format(
        seed_fact=seed_str,
        related_facts=related_str,
        resolution_map=resolution_str,
        scratchpad=scratchpad_str,
    )
    raw = llm(prompt, cache, budget, reasoning_effort="medium")
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return [], [], [], False

    facts_raw = obj.get("new_facts", []) or []
    continue_thinking = bool(obj.get("continue_thinking", False))
    new_facts, new_mentions, new_bindings = _materialize_reflection(
        facts_raw, seed.ts, store, fire_idx, refl_step
    )
    return new_facts, new_mentions, new_bindings, continue_thinking


# ---------------------------------------------------------------------------
# Ingestion driver — K=3 centered window + recursive reflection
# ---------------------------------------------------------------------------


def ingest_turns(
    turns,
    cache,
    budget,
    *,
    w_past: int = 7,
    w_future: int = 7,
    k: int = 3,
    rebuild_index_every: int = 4,
    reflection_budget: int = 2,
    reflection_max: int = 3,
    enable_reflection: bool = True,
    inline_anchors: bool = False,
):
    obs_facts: list[Fact] = []
    obs_mentions: list[Mention] = []
    cog_facts: list[Fact] = []
    cog_mentions: list[Mention] = []
    store = MemoryStore()

    # Pre-register the User entity as e_user (matches v2 convention)
    store.registry.register("m_user_root")
    store.registry.mention_to_entity["m_user_root"] = "e_user"
    store.registry.entity_members["e_user"] = {"m_user_root"}
    store.registry.entity_members.pop("e_m_user_root", None)

    telemetry = []
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

        obs_idx = store.collections.get("observations")
        new_facts, new_mentions, new_bindings, tele = write_window(
            window_turns,
            target_turn_lo,
            target_turns,
            obs_facts,
            obs_idx,
            store.registry,
            cache,
            budget,
            inline_anchors=inline_anchors,
            all_mentions=obs_mentions if inline_anchors else None,
        )
        obs_facts.extend(new_facts)
        obs_mentions.extend(new_mentions)

        scratchpad: list[Fact] = list(new_facts)
        n_refl_calls = 0
        n_cog_emitted = 0

        if enable_reflection and new_facts:
            queue: list[Fact] = list(new_facts)
            r_budget = reflection_budget
            r_used = 0
            while r_budget > 0 and r_used < reflection_max and queue:
                seed = queue.pop(0)
                refl_facts, refl_mentions, refl_bindings, cont = reflect_on_fact(
                    seed,
                    store,
                    scratchpad,
                    cache,
                    budget,
                    fire_idx=fire_no,
                    refl_step=r_used,
                )
                r_used += 1
                r_budget -= 1
                n_refl_calls += 1
                if refl_facts:
                    cog_facts.extend(refl_facts)
                    cog_mentions.extend(refl_mentions)
                    scratchpad.extend(refl_facts)
                    queue.extend(refl_facts)
                    n_cog_emitted += len(refl_facts)
                if cont and r_used < reflection_max:
                    r_budget = max(r_budget, min(1, reflection_max - r_used))

        tele["fire_no"] = fire_no
        tele["last_turn"] = target_turns[-1][0]
        tele["n_reflection_calls"] = n_refl_calls
        tele["n_cog_emitted"] = n_cog_emitted
        telemetry.append(tele)

        if fire_no % rebuild_index_every == 0:
            store.collections["observations"] = build_collection(
                "observations",
                obs_facts,
                obs_mentions,
                store.registry,
                cache,
                budget,
            )
            if cog_facts:
                store.collections["cognition"] = build_collection(
                    "cognition",
                    cog_facts,
                    cog_mentions,
                    store.registry,
                    cache,
                    budget,
                )
        fire_no += 1
        target_lo = target_hi

    # Final rebuild
    store.collections["observations"] = build_collection(
        "observations",
        obs_facts,
        obs_mentions,
        store.registry,
        cache,
        budget,
    )
    if cog_facts:
        store.collections["cognition"] = build_collection(
            "cognition",
            cog_facts,
            cog_mentions,
            store.registry,
            cache,
            budget,
        )
    return obs_facts, obs_mentions, cog_facts, cog_mentions, store, telemetry


# ---------------------------------------------------------------------------
# Reader — defaults to retrieving over BOTH collections
# ---------------------------------------------------------------------------


def answer_question(
    question: str,
    store: MemoryStore,
    cache: Cache,
    budget: Budget,
    top_k: int = 14,
    collections: list[str] | None = None,
) -> str:
    if collections is None:
        # Default: associative recall spans observations AND cognition
        collections = (
            ["observations", "cognition"]
            if "cognition" in store.collections
            else ["observations"]
        )
    facts, resolution_map = retrieve(
        question,
        store,
        cache,
        budget,
        top_k=top_k,
        collections=collections,
        llm_dedup=True,
        expand_hops=1,
    )
    eid_alias = v2._build_eid_alias(resolution_map)
    facts_block = format_facts_for_read(facts, store, eid_alias=eid_alias)
    resolution_block = format_resolution_map(resolution_map, store, eid_alias=eid_alias)
    prompt = READ_PROMPT.format(
        resolution_map=resolution_block,
        facts_block=facts_block,
        question=question,
    )
    return llm(prompt, cache, budget, reasoning_effort="medium").strip()
