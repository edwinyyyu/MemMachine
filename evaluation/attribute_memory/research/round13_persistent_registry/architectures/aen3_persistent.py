"""AEN-3 — persistent entity registry with LRU active cache.

Round 13 fix for round 12's bug: in aen2, the LRU dict WAS the registry, so
when an entity got evicted the system effectively forgot it. Later
descriptor-only mentions ("the recruiter from Q1") created a fresh entity
instead of resolving to the original.

aen3 separates *persistence* from *context*:

  - PersistentRegistry stores every entity ever created. Indexes:
      by_id          : ent_id -> Entity
      aliases_index  : alias_lower -> list[ent_id]    (exact match)
      desc_embeds    : ent_id -> np.ndarray           (description embedding)
  - ActiveCache is the LRU window (~20 entries) used by the coref LLM.

Per-turn pipeline:
  1. Coref LLM sees the LRU active cache + new turn. For each mention it
     emits action in {resolve, lookup, create, skip}:
       - resolve: cache hit by ent_id (cheap)
       - lookup: not in cache; ask the pipeline to find a match in the full
                 registry. Pipeline tries:
                   (a) exact alias index lookup (no extra LLM)
                   (b) if a `descriptor_query` is provided AND no exact-alias
                       hit, embedding search over description embeddings,
                       returning top-K candidates. LLM picks one or says
                       create_new.
       - create: fresh entity (after defensive registry-side alias check)
       - skip: not really an entity
  2. Resolve/create writes back: the entity is touched into the LRU and
     persistent updates (aliases, last_seen_turn, expanded description).
  3. Turn text is rewritten with `[@ent_NNNNN / surface]` for the writer.

Key decisions:
  - Embedding search is a CANDIDATE FILTER only. The LLM picks the final
    answer (or says create_new). We do NOT use cosine as final answer.
  - Embedding search fires only for descriptor lookups that MISSED the
    exact-alias index. Named lookups try exact alias first; if multiple
    candidates -> alias-disambig; if none -> create.
  - The description for an entity is maintained by the coref LLM (when
    creating it gives a short clause). When new identifying detail appears
    in later resolves, we append. Description embedding is recomputed only
    when description changes.
  - Top-K = 5 from embedding search.

Compatibility: still produces aen1_simple-style LogEntry's (writer is the
same as aen2's, see _write_batch_with_eids).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROUND13 = HERE.parent
ROUND12 = ROUND13.parent / "round12_entity_registry"
ROUND11 = ROUND13.parent / "round11_writer_stress"
ROUND7 = ROUND13.parent / "round7"
sys.path.insert(0, str(ROUND7 / "experiments"))
sys.path.insert(0, str(ROUND11 / "architectures"))

import aen1_simple  # noqa: E402
from _common import Budget, Cache, embed_batch, extract_json, llm  # noqa: E402

LogEntry = aen1_simple.LogEntry
IndexedLog = aen1_simple.IndexedLog
build_index = aen1_simple.build_index


# ---------------------------------------------------------------------------
# Entity + registry
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    id: str
    aliases: set[str] = field(default_factory=set)
    description: str = ""
    first_seen_turn: int = 0
    last_seen_turn: int = 0


@dataclass
class PersistentRegistry:
    """Durable storage of all entities ever created. Never forgets."""

    by_id: dict[str, Entity] = field(default_factory=dict)
    next_id: int = 1
    # alias (lowercased) -> list of ent_ids that have this alias
    aliases_index: dict[str, list[str]] = field(default_factory=dict)
    # ent_id -> embedding of (current) description
    desc_embeds: dict[str, np.ndarray] = field(default_factory=dict)
    # ent_id -> the description text used to compute the embedding (so we
    # don't re-embed if unchanged)
    desc_embed_source: dict[str, str] = field(default_factory=dict)
    # LRU
    lru_order: list[str] = field(default_factory=list)
    lru_size: int = 20

    # ---- creation / mutation -------------------------------------------------

    def make_id(self) -> str:
        eid = f"ent_{self.next_id:05d}"
        self.next_id += 1
        return eid

    def create(
        self,
        aliases: list[str],
        description: str,
        turn_idx: int,
        eid: str | None = None,
    ) -> Entity:
        if eid is None:
            eid = self.make_id()
        e = Entity(
            id=eid,
            aliases={a for a in aliases if a},
            description=description or "",
            first_seen_turn=turn_idx,
            last_seen_turn=turn_idx,
        )
        self.by_id[eid] = e
        for a in e.aliases:
            self._index_alias(a, eid)
        self._touch(eid)
        return e

    def add_alias(self, entity_id: str, alias: str) -> None:
        if entity_id not in self.by_id or not alias:
            return
        e = self.by_id[entity_id]
        if alias in e.aliases:
            return
        e.aliases.add(alias)
        self._index_alias(alias, entity_id)

    def _index_alias(self, alias: str, entity_id: str) -> None:
        key = alias.strip().lower()
        if not key:
            return
        lst = self.aliases_index.setdefault(key, [])
        if entity_id not in lst:
            lst.append(entity_id)

    def update_description(self, entity_id: str, new_desc: str) -> None:
        if entity_id not in self.by_id or not new_desc:
            return
        cur = self.by_id[entity_id].description
        new_desc = new_desc.strip()
        if not new_desc:
            return
        if not cur:
            self.by_id[entity_id].description = new_desc
        elif new_desc.lower() in cur.lower():
            return
        else:
            # cap total length at ~400 chars to bound prompt size
            joined = (cur + "; " + new_desc).strip("; ")
            if len(joined) > 400:
                joined = joined[-400:]
            self.by_id[entity_id].description = joined

    def touch(self, entity_id: str, turn_idx: int) -> None:
        if entity_id in self.by_id:
            self.by_id[entity_id].last_seen_turn = turn_idx
        self._touch(entity_id)

    def _touch(self, entity_id: str) -> None:
        if entity_id in self.lru_order:
            self.lru_order.remove(entity_id)
        self.lru_order.append(entity_id)

    # ---- lookup --------------------------------------------------------------

    def lru_entities(self) -> list[Entity]:
        out: list[Entity] = []
        for eid in self.lru_order[-self.lru_size :]:
            if eid in self.by_id:
                out.append(self.by_id[eid])
        return out

    def candidates_by_alias(self, alias: str) -> list[Entity]:
        key = alias.strip().lower()
        if not key:
            return []
        eids = self.aliases_index.get(key, [])
        return [self.by_id[eid] for eid in eids if eid in self.by_id]

    # ---- embedding-search ----------------------------------------------------

    def ensure_desc_embed(self, entity_id: str, cache: Cache, budget: Budget) -> None:
        """(Re)compute the description embedding if needed."""
        if entity_id not in self.by_id:
            return
        e = self.by_id[entity_id]
        if not e.description.strip():
            return
        if (
            entity_id in self.desc_embeds
            and self.desc_embed_source.get(entity_id) == e.description
        ):
            return
        # Build embedding text: aliases + description
        alias_str = "/".join(sorted(e.aliases)) if e.aliases else ""
        text = (alias_str + " :: " + e.description).strip()
        vec = embed_batch([text], cache, budget)[0]
        self.desc_embeds[entity_id] = np.asarray(vec, dtype=np.float32)
        self.desc_embed_source[entity_id] = e.description

    def embedding_search(
        self,
        query: str,
        cache: Cache,
        budget: Budget,
        top_k: int = 5,
        exclude: set[str] | None = None,
    ) -> list[tuple[Entity, float]]:
        """Return top-K (entity, cosine_score) for query among entities that
        have a description embedding."""
        exclude = exclude or set()
        # Make sure all entities with descriptions have embeddings
        # (lazy: only those we don't yet have, embedded one at a time within
        # the cache; embed_batch dedupes via cache key).
        to_embed: list[tuple[str, str]] = []  # (eid, embed_text)
        for eid, e in self.by_id.items():
            if eid in exclude or not e.description.strip():
                continue
            if (
                eid in self.desc_embeds
                and self.desc_embed_source.get(eid) == e.description
            ):
                continue
            alias_str = "/".join(sorted(e.aliases)) if e.aliases else ""
            text = (alias_str + " :: " + e.description).strip()
            to_embed.append((eid, text))
        if to_embed:
            vecs = embed_batch([t for _, t in to_embed], cache, budget)
            for (eid, _), v in zip(to_embed, vecs, strict=True):
                self.desc_embeds[eid] = np.asarray(v, dtype=np.float32)
                self.desc_embed_source[eid] = self.by_id[eid].description

        # Embed the query
        q_vec = np.asarray(embed_batch([query], cache, budget)[0], dtype=np.float32)
        q_norm = float(np.linalg.norm(q_vec)) or 1.0

        scored: list[tuple[Entity, float]] = []
        for eid, vec in self.desc_embeds.items():
            if eid in exclude:
                continue
            if eid not in self.by_id:
                continue
            n = float(np.linalg.norm(vec)) or 1.0
            score = float(q_vec @ vec) / (q_norm * n)
            scored.append((self.by_id[eid], score))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Coref prompts
# ---------------------------------------------------------------------------

COREF_PROMPT = """You are a coreference resolver for a chat assistant's memory.

For the user's NEW TURN below, identify every entity mention (named people,
pronouns, descriptive references like "the manager"). For each mention,
decide what to do.

The LRU cache below holds the most recently active entities — each has a
stable internal id and a short description. PREFER to RESOLVE mentions to a
cached entity when context is consistent.

LRU CACHE (most-recently-active LAST):
{lru_block}

NEW TURN (turn {turn_idx}): "{turn_text}"

Output JSON of the form:
{{
  "mentions": [
    {{
      "surface": "<exact substring from the turn>",
      "kind": "named" | "pronoun" | "descriptor",
      "action": "resolve" | "lookup" | "create" | "skip",
      "entity_id": "<ent_id from cache, ONLY if action=resolve>",
      "lookup_alias": "<canonical name to alias-search the registry, for action=lookup>",
      "lookup_descriptor": "<descriptive phrase for embedding-search the registry, for action=lookup; usually for descriptor mentions>",
      "description": "<short identifying detail (1 short clause): role, relationship, place — for create OR for enriching a resolve>"
    }}
  ]
}}

ACTIONS
- "resolve" — the mention refers to an entity in the LRU CACHE above. Use
  the cache entity_id verbatim. Use this whenever you can.
- "lookup" — the mention refers to an entity that is NOT in the cache but may
  exist in the long-term registry (an entity introduced earlier and pushed
  out of the LRU). The pipeline will alias-search and/or embedding-search the
  full registry and pick a match (or create new if no match).
    * For NAMED mentions ("Alice", "Marcus"), set lookup_alias to the name.
    * For DESCRIPTOR mentions ("the recruiter", "my dog walker"), set
      lookup_descriptor to a SEMANTIC paraphrase including any role,
      relationship, or distinguishing detail (e.g. "recruiter who placed
      me at my current job in Q1"). Set lookup_alias to the literal phrase
      too if useful. Be DESCRIPTIVE — this is what we embed.
- "create" — only when the turn EXPLICITLY introduces a new entity whose
  identity differs from anyone in the cache, AND you can give a useful
  description.
- "skip" — only for non-entity tokens (a date, a generic noun) or for
  pronouns that are genuinely ambiguous between multiple cached entities.

RULES
- For pronouns (he/she/they/his/her/their/them): if exactly one cached
  entity could be the antecedent, use action="resolve" with that entity_id.
  If ambiguous, action="skip".
- The user's "I"/"me"/"my"/"User" should resolve to the entity whose alias
  is "User" (already in the cache).
- For NAMED mentions: if exactly one cached entity has that name as an
  alias AND context is consistent (or silent), action="resolve". Otherwise
  action="lookup" with lookup_alias=name.
- For DESCRIPTOR mentions: if a cached entity's description matches,
  action="resolve". Otherwise action="lookup" with a rich
  lookup_descriptor.
- SILENT context (turn just says the name with no extra detail) is the
  default. Do NOT create a new entity just because a turn lacks detail.
- Only use action="create" when an INTRODUCTION is explicit and the
  entity's name does not appear in the cache. Provide a short description.
- Do NOT invent entity_ids. Only use ids you literally see in the cache.

Output ONLY the JSON.
"""


ALIAS_DISAMBIG_PROMPT = """A name in the user's turn matches multiple entities
in the long-term registry. Decide which one is meant — or whether the turn
introduces a NEW distinct entity.

NEW TURN (turn {turn_idx}): "{turn_text}"

NAME OR DESCRIPTOR: "{alias}"

CANDIDATE ENTITIES:
{candidates_block}

Output JSON:
{{
  "decision": "match" | "create_new",
  "entity_id": "<ent_id if match>",
  "rationale": "<one short sentence>"
}}

RULES
- If the turn's context (role, relationship, location, attribute) is consistent
  with one candidate and not the others, pick that candidate.
- If the context is SILENT (no extra detail), pick the most-recently-active
  candidate (the one with the largest last_seen_turn). Do NOT create a new
  entity.
- Only create a new entity when the turn EXPLICITLY signals a distinct
  identity ("X from high school", "another X", "X — last name Foster",
  "different X — my dentist").
- Output ONLY JSON.
"""


DESCRIPTOR_PICK_PROMPT = """The user's turn includes a descriptor reference
("{descriptor}") that is NOT in the active cache. We searched the long-term
registry by description-embedding and found these top-{k} candidates.

Decide which one matches — or say "create_new" if none of them fit.

NEW TURN (turn {turn_idx}): "{turn_text}"

DESCRIPTOR FROM TURN: "{descriptor}"

CANDIDATES (highest-similarity first):
{candidates_block}

Output JSON:
{{
  "decision": "match" | "create_new",
  "entity_id": "<ent_id if match>",
  "rationale": "<one short sentence>"
}}

RULES
- Pick a match when the role/relationship/specifics in the turn are
  consistent with ONE candidate's description. The candidate's description
  must contain a feature that aligns with the descriptor (e.g.
  descriptor="the recruiter from Q1" + candidate description "recruiter
  that placed me back in Q1" -> match).
- Do NOT match purely on topical overlap. The descriptor must point to a
  SPECIFIC entity, not a category.
- If multiple candidates fit equally well, pick the one with the largest
  last_seen_turn (most recently active).
- If no candidate's description specifically matches, decision="create_new".
- Output ONLY JSON.
"""


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_lru_block(reg: PersistentRegistry) -> str:
    ents = reg.lru_entities()
    if not ents:
        return "(empty cache)"
    lines = []
    for e in ents:
        aliases = "/".join(sorted(e.aliases)) if e.aliases else "(no aliases)"
        desc = e.description or "(no description)"
        lines.append(
            f"  [{e.id}] aliases={aliases} :: {desc} (last_seen=t{e.last_seen_turn})"
        )
    return "\n".join(lines)


def _render_candidates(cands: list[Entity]) -> str:
    lines = []
    for e in cands:
        aliases = "/".join(sorted(e.aliases))
        desc = e.description or "(no description)"
        lines.append(
            f"  [{e.id}] aliases={aliases} :: {desc} (last_seen=t{e.last_seen_turn})"
        )
    return "\n".join(lines)


def _render_scored_candidates(scored: list[tuple[Entity, float]]) -> str:
    lines = []
    for e, score in scored:
        aliases = "/".join(sorted(e.aliases))
        desc = e.description or "(no description)"
        lines.append(
            f"  [{e.id}] sim={score:.3f} aliases={aliases} :: "
            f"{desc} (last_seen=t{e.last_seen_turn})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def disambiguate_alias(
    alias: str,
    candidates: list[Entity],
    turn_idx: int,
    turn_text: str,
    cache: Cache,
    budget: Budget,
) -> tuple[str | None, str]:
    if len(candidates) == 1:
        return candidates[0].id, "single-candidate"
    if not candidates:
        return None, "no-candidates"
    prompt = ALIAS_DISAMBIG_PROMPT.format(
        turn_idx=turn_idx,
        turn_text=turn_text,
        alias=alias,
        candidates_block=_render_candidates(candidates),
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        cands_sorted = sorted(candidates, key=lambda e: -e.last_seen_turn)
        return cands_sorted[0].id, "fallback (parse error)"
    decision = obj.get("decision")
    if decision == "match":
        eid = obj.get("entity_id")
        if eid in {c.id for c in candidates}:
            return eid, obj.get("rationale", "")
    if decision == "create_new":
        return None, obj.get("rationale", "create_new")
    cands_sorted = sorted(candidates, key=lambda e: -e.last_seen_turn)
    return cands_sorted[0].id, "fallback"


def descriptor_pick(
    descriptor: str,
    scored: list[tuple[Entity, float]],
    turn_idx: int,
    turn_text: str,
    cache: Cache,
    budget: Budget,
    k: int,
) -> tuple[str | None, str]:
    """LLM picks one of the top-K embedding hits, or says create_new."""
    if not scored:
        return None, "no-embed-candidates"
    prompt = DESCRIPTOR_PICK_PROMPT.format(
        descriptor=descriptor,
        turn_idx=turn_idx,
        turn_text=turn_text,
        k=len(scored),
        candidates_block=_render_scored_candidates(scored),
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return None, "fallback-parse-error"
    decision = obj.get("decision")
    if decision == "match":
        eid = obj.get("entity_id")
        if eid in {c.id for c, _ in scored}:
            return eid, obj.get("rationale", "")
    if decision == "create_new":
        return None, obj.get("rationale", "create_new")
    return None, "fallback-unknown-decision"


# ---------------------------------------------------------------------------
# Per-turn coref + rewrite
# ---------------------------------------------------------------------------


@dataclass
class CorefDecision:
    surface: str
    kind: str
    entity_id: str | None
    rationale: str = ""
    used_embedding_search: bool = False


def coref_turn(
    turn_idx: int,
    turn_text: str,
    reg: PersistentRegistry,
    cache: Cache,
    budget: Budget,
    top_k: int = 5,
) -> tuple[str, list[CorefDecision]]:
    # Pre-seed user
    if "ent_user" not in reg.by_id:
        e = Entity(
            id="ent_user",
            aliases={"User", "I", "me", "my"},
            description="The user / speaker (first-person).",
            first_seen_turn=0,
            last_seen_turn=turn_idx,
        )
        reg.by_id["ent_user"] = e
        for a in e.aliases:
            reg._index_alias(a, "ent_user")
        reg._touch("ent_user")

    prompt = COREF_PROMPT.format(
        lru_block=_render_lru_block(reg),
        turn_idx=turn_idx,
        turn_text=turn_text,
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    decisions: list[CorefDecision] = []
    if not isinstance(obj, dict):
        return turn_text, decisions
    mentions = obj.get("mentions", []) or []

    replacements: list[tuple[str, str]] = []
    cache_eids = {e.id for e in reg.lru_entities()}

    for m in mentions:
        if not isinstance(m, dict):
            continue
        surface = (m.get("surface") or "").strip()
        if not surface:
            continue
        kind = m.get("kind") or "named"
        action = m.get("action") or "skip"
        eid: str | None = None
        rationale = ""
        used_embed = False

        if action == "resolve":
            cand_eid = m.get("entity_id")
            if cand_eid in reg.by_id:
                eid = cand_eid
                if kind in ("named", "descriptor"):
                    reg.add_alias(eid, surface)
            rationale = "cache-resolve"
        elif action == "lookup":
            lookup_alias = (m.get("lookup_alias") or surface).strip()
            lookup_desc = (m.get("lookup_descriptor") or "").strip()

            # (a) Try exact alias match across the full registry first
            cands = reg.candidates_by_alias(lookup_alias)
            # If kind=descriptor, also probe with the surface itself
            if kind == "descriptor" and not cands:
                cands = reg.candidates_by_alias(surface)

            if cands:
                eid, rationale = disambiguate_alias(
                    lookup_alias or surface, cands, turn_idx, turn_text, cache, budget
                )
                if eid is None:
                    desc = (m.get("description") or "").strip()
                    new_e = reg.create([lookup_alias or surface], desc, turn_idx)
                    eid = new_e.id
                    rationale = "alias-disambig-create-new"
                else:
                    rationale = "alias-resolve: " + rationale
            else:
                # (b) embedding-search fallback (descriptor cases primarily)
                # Build the query: prefer the rich descriptor; fall back to surface.
                query = lookup_desc or surface
                # Skip embedding search if the query is degenerate (only an
                # article-noun like "the dog") — actually we always try, the
                # LLM rejects bad matches.
                if query and reg.by_id:
                    used_embed = True
                    scored = reg.embedding_search(
                        query,
                        cache,
                        budget,
                        top_k=top_k,
                        exclude={"ent_user"},
                    )
                    eid, rat = descriptor_pick(
                        query,
                        scored,
                        turn_idx,
                        turn_text,
                        cache,
                        budget,
                        k=top_k,
                    )
                    rationale = "embed-search: " + rat
                if eid is None:
                    # Create new
                    desc = (m.get("description") or lookup_desc or "").strip()
                    new_e = reg.create([lookup_alias or surface], desc, turn_idx)
                    eid = new_e.id
                    rationale = (
                        rationale + " | create-fresh" if rationale else "create-fresh"
                    )
                else:
                    # Add the surface form as an alias too (helps next time)
                    if surface and surface.lower() not in (
                        a.lower() for a in reg.by_id[eid].aliases
                    ):
                        reg.add_alias(eid, surface)
        elif action == "create":
            alias = m.get("lookup_alias") or surface
            desc = (m.get("description") or "").strip()
            cands = reg.candidates_by_alias(alias)
            if cands:
                eid, rationale = disambiguate_alias(
                    alias, cands, turn_idx, turn_text, cache, budget
                )
                if eid is None:
                    new_e = reg.create([alias], desc, turn_idx)
                    eid = new_e.id
                    rationale = "create-after-disambig"
            else:
                new_e = reg.create([alias], desc, turn_idx)
                eid = new_e.id
                rationale = "create-fresh"
        else:  # skip
            decisions.append(CorefDecision(surface, kind, None, "skipped", False))
            continue

        if eid is not None:
            reg.touch(eid, turn_idx)
            extra_desc = (m.get("description") or "").strip()
            if extra_desc:
                reg.update_description(eid, extra_desc)

        decisions.append(CorefDecision(surface, kind, eid, rationale, used_embed))
        if eid is not None:
            replacements.append((surface, f"[@{eid} / {surface}]"))

    rewritten = turn_text
    for surface, repl in sorted(replacements, key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(surface))
        rewritten = pattern.sub(repl, rewritten, count=1)
    return rewritten, decisions


# ---------------------------------------------------------------------------
# End-to-end ingestion (writer copied from aen2_registry)
# ---------------------------------------------------------------------------

WRITE_PROMPT = """You are a semantic-memory writer using a SINGLE APPEND-ONLY LOG.

Each entry you write is an atomic natural-language fact. Use entity tags of
the form `@ent_NNNNN` (or @User for the speaker) to mention entities — these
are stable internal ids; they have already been resolved for you in the turn
text below, in the form [@ent_NNNNN / surface_form].

When you emit an entry, set `mentions` to the list of `@ent_*` tags for the
entities involved. Use `refs` to point at prior entry uuids when this entry
relates to, updates, or corrects a prior fact. There is only ONE kind of ref.
For state-tracking facts include `predicate` as `@ent_NNNNN.pred_name`.

KNOWN ENTITY TAGS so far: {known_entities}

PRIOR LOG SAMPLE (most recent + relevant chain heads):
{prior_log}

BATCH OF TURNS (entity tags already resolved inline):
{turn_block}

Emit JSON:
{{
  "entries": [
    {{
      "text": "<atomic fact in one sentence>",
      "mentions": ["@ent_NNNNN", ...],
      "refs": ["<prior-uuid>", ...],
      "predicate": "@ent_NNNNN.pred" or null
    }}
  ]
}}

RULES
- ALWAYS use the `@ent_*` tags exactly as they appear in the turn text. Do
  NOT use bare names like "Alice" or "Marcus" — only the resolved tags.
- If a turn is filler (weather, chitchat), skip silently.
- If a turn UPDATES or CORRECTS a prior fact, emit a new entry with `refs`
  pointing at the prior entry that stated the now-outdated value.
- For state-tracking facts (job, location, boss, employer, role,
  relationship), include `predicate` in the form `@ent_NNNNN.pred_name`.
- Output JSON ONLY.
"""


def _write_batch_with_eids(
    batch_turns: list[tuple[int, str]],
    prior_entries: list[LogEntry],
    idx: IndexedLog | None,
    known_entities: set[str],
    reg: PersistentRegistry,
    cache: Cache,
    budget: Budget,
) -> list[LogEntry]:
    relevant: list[LogEntry] = []
    if idx is not None:
        batch_text = " ".join(t for _, t in batch_turns)
        ent_tags = set(re.findall(r"@ent_\d{5}", batch_text)) | {"@User"}
        seen: set[str] = set()
        for (tag, pred), uuid in idx.supersede_head.items():
            if tag in ent_tags and uuid not in seen:
                seen.add(uuid)
                relevant.append(idx.by_uuid[uuid])
        relevant.sort(key=lambda e: e.ts, reverse=True)
        relevant = relevant[:8]

    prior_log = aen1_simple._render_prior_log(prior_entries, relevant_heads=relevant)
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
        refs = [r for r in refs_raw if isinstance(r, str)]
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


def ingest_turns_with_registry(
    turns: list[tuple[int, str]],
    cache: Cache,
    budget: Budget,
    batch_size: int = 5,
    rebuild_index_every: int = 25,
    lru_size: int = 20,
    top_k: int = 5,
    run_writer: bool = True,
) -> tuple[
    list[LogEntry],
    IndexedLog | None,
    PersistentRegistry,
    dict[int, list[CorefDecision]],
]:
    reg = PersistentRegistry(lru_size=lru_size)
    rewritten_turns: list[tuple[int, str]] = []
    coref_log: dict[int, list[CorefDecision]] = {}

    for tidx, text in turns:
        new_text, decisions = coref_turn(tidx, text, reg, cache, budget, top_k=top_k)
        rewritten_turns.append((tidx, new_text))
        coref_log[tidx] = decisions
        cache.save()

    log: list[LogEntry] = []
    idx: IndexedLog | None = None
    if run_writer:
        known: set[str] = {"User"}
        for i in range(0, len(rewritten_turns), batch_size):
            batch = rewritten_turns[i : i + batch_size]
            new_entries = _write_batch_with_eids(
                batch, log, idx, known, reg, cache, budget
            )
            for e in new_entries:
                for mn in e.mentions:
                    if mn.startswith("@"):
                        known.add(mn[1:])
            log.extend(new_entries)
            if (i // batch_size) % rebuild_index_every == 0:
                idx = build_index(log, cache, budget)
        idx = build_index(log, cache, budget)
    return log, idx, reg, coref_log


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def registry_snapshot(reg: PersistentRegistry) -> dict:
    return {
        "by_id": {
            eid: {
                "aliases": sorted(e.aliases),
                "description": e.description,
                "first_seen": e.first_seen_turn,
                "last_seen": e.last_seen_turn,
            }
            for eid, e in reg.by_id.items()
        },
        "lru_order": list(reg.lru_order),
        "n_with_desc_embed": len(reg.desc_embeds),
    }
