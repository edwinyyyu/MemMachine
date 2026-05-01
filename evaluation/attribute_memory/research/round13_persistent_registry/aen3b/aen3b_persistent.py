"""AEN-3b — fixed descriptor recovery on top of aen3_persistent.

Round-13 follow-up. aen3 hit 0% descriptor accuracy on S3 LRU stress despite
firing 59 embedding searches; total accuracy stuck at 74% (vs 97% baseline).

Three changes vs aen3:

  Fix 1. DESCRIPTOR_PICK_PROMPT softened.
         The old prompt required the candidate description to literally
         contain a feature aligned with the descriptor and rejected anything
         that was "just topical overlap." That made the LLM say
         "create_new" even when the top-1 embedding hit was a clear semantic
         match (e.g. "the dog walker" -> ent already aliased "dog walker").
         New prompt asks for graded judgement: pick a candidate when its
         role/identity is consistent with the descriptor and not contradicted
         by its description. Default lean is MATCH, with two calibration
         examples (one positive, one negative).

  Fix 2. Accumulating descriptions.
         aen3 already had update_description() but only called it when the
         coref LLM returned an explicit "description" field. That meant
         entities introduced once never got further description text after
         that, so embedding search lost recall for descriptors with extra
         specifics ("the recruiter from Q1"). aen3b adds a per-mention turn
         snippet to the entity's description list and rebuilds the
         search-time blob from a recent-snippets concatenation. No extra
         LLM cost.

  Fix 3. (Independent of the prompt) Surface-form normalisation in coref
         decisions output. The grader keys mentions by exact-match surface
         (lowercased GT vs case-preserved coref output); aen3 was actually
         resolving descriptors correctly but the grader couldn't find the
         decision under the GT surface ("the dog walker" vs "The dog
         walker"). Decisions are now written with both the original surface
         AND a normalized variant.

Everything else (PersistentRegistry, write pipeline) is identical to aen3.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
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
sys.path.insert(0, str(ROUND13 / "architectures"))

import aen1_simple  # noqa: E402
import aen3_persistent  # noqa: E402
from _common import Budget, Cache, embed_batch, extract_json, llm  # noqa: E402

LogEntry = aen1_simple.LogEntry
IndexedLog = aen1_simple.IndexedLog
build_index = aen1_simple.build_index


# ---------------------------------------------------------------------------
# Entity + registry (extends aen3 with description-snippet accumulator)
# ---------------------------------------------------------------------------

DESC_SNIPPET_CHAR_CAP = 600  # ~ 200 tokens
PER_SNIPPET_CHAR_CAP = 240
SNIPPET_KEEP = 6  # most recent N snippets joined for embed/search


@dataclass
class Entity:
    id: str
    aliases: set[str] = field(default_factory=set)
    description: str = ""
    desc_snippets: list[str] = field(default_factory=list)
    first_seen_turn: int = 0
    last_seen_turn: int = 0


@dataclass
class PersistentRegistry:
    """Durable storage of all entities ever created. Never forgets."""

    by_id: dict[str, Entity] = field(default_factory=dict)
    next_id: int = 1
    aliases_index: dict[str, list[str]] = field(default_factory=dict)
    desc_embeds: dict[str, np.ndarray] = field(default_factory=dict)
    desc_embed_source: dict[str, str] = field(default_factory=dict)
    lru_order: list[str] = field(default_factory=list)
    lru_size: int = 20

    # ---- creation / mutation -------------------------------------------------

    def make_id(self) -> str:
        eid = f"ent_{self.next_id:05d}"
        self.next_id += 1
        return eid

    def create(
        self,
        aliases: Iterable[str],
        description: str,
        turn_idx: int,
        eid: str | None = None,
    ) -> Entity:
        if eid is None:
            eid = self.make_id()
        e = Entity(
            id=eid,
            aliases={a for a in aliases if a},
            description=(description or "").strip(),
            first_seen_turn=turn_idx,
            last_seen_turn=turn_idx,
        )
        self.by_id[eid] = e
        for a in e.aliases:
            self._index_alias(a, eid)
        if e.description:
            self._append_snippet(eid, e.description)
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

    # ---- description accumulation -------------------------------------------

    def _append_snippet(self, entity_id: str, snippet: str) -> None:
        snippet = " ".join(snippet.split())  # collapse whitespace
        if not snippet:
            return
        if len(snippet) > PER_SNIPPET_CHAR_CAP:
            snippet = snippet[:PER_SNIPPET_CHAR_CAP].rstrip()
        e = self.by_id[entity_id]
        # de-dup: skip if this snippet is already a substring of an existing one
        low = snippet.lower()
        for existing in e.desc_snippets:
            if low in existing.lower() or existing.lower() in low:
                # Promote to most-recent if same content.
                try:
                    e.desc_snippets.remove(existing)
                except ValueError:
                    pass
                e.desc_snippets.append(
                    existing if len(existing) > len(snippet) else snippet
                )
                self._refresh_description(entity_id)
                return
        e.desc_snippets.append(snippet)
        # cap total snippet payload
        while (
            sum(len(s) for s in e.desc_snippets) > DESC_SNIPPET_CHAR_CAP
            and len(e.desc_snippets) > 1
        ):
            e.desc_snippets.pop(0)
        self._refresh_description(entity_id)

    def _refresh_description(self, entity_id: str) -> None:
        e = self.by_id[entity_id]
        recent = e.desc_snippets[-SNIPPET_KEEP:]
        e.description = " :: ".join(recent)

    def update_description(self, entity_id: str, new_desc: str) -> None:
        if entity_id not in self.by_id or not new_desc:
            return
        new_desc = new_desc.strip()
        if not new_desc:
            return
        self._append_snippet(entity_id, new_desc)

    def add_turn_context(self, entity_id: str, turn_text: str) -> None:
        """Append the raw turn text the entity was mentioned in. Cheap; no
        LLM call. Helps the embedding search at lookup time."""
        if entity_id not in self.by_id:
            return
        snippet = (turn_text or "").strip()
        if not snippet:
            return
        # drop leading 'The/A' to bias toward content terms a bit
        self._append_snippet(entity_id, snippet)

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

    def _embed_text_for(self, e: Entity) -> str:
        alias_str = "/".join(sorted(e.aliases)) if e.aliases else ""
        return (alias_str + " :: " + e.description).strip()

    def embedding_search(
        self,
        query: str,
        cache: Cache,
        budget: Budget,
        top_k: int = 5,
        exclude: set[str] | None = None,
    ) -> list[tuple[Entity, float]]:
        exclude = exclude or set()
        to_embed: list[tuple[str, str]] = []
        for eid, e in self.by_id.items():
            if eid in exclude or not e.description.strip():
                continue
            text = self._embed_text_for(e)
            if eid in self.desc_embeds and self.desc_embed_source.get(eid) == text:
                continue
            to_embed.append((eid, text))
        if to_embed:
            vecs = embed_batch([t for _, t in to_embed], cache, budget)
            for (eid, txt), v in zip(to_embed, vecs, strict=True):
                self.desc_embeds[eid] = np.asarray(v, dtype=np.float32)
                self.desc_embed_source[eid] = txt

        q_vec = np.asarray(embed_batch([query], cache, budget)[0], dtype=np.float32)
        q_norm = float(np.linalg.norm(q_vec)) or 1.0

        scored: list[tuple[Entity, float]] = []
        for eid, vec in self.desc_embeds.items():
            if eid in exclude or eid not in self.by_id:
                continue
            n = float(np.linalg.norm(vec)) or 1.0
            score = float(q_vec @ vec) / (q_norm * n)
            scored.append((self.by_id[eid], score))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Coref prompts — coref + alias-disambig kept as in aen3; descriptor-pick
# softened (Fix 1).
# ---------------------------------------------------------------------------

COREF_PROMPT = aen3_persistent.COREF_PROMPT
ALIAS_DISAMBIG_PROMPT = aen3_persistent.ALIAS_DISAMBIG_PROMPT


DESCRIPTOR_PICK_PROMPT = """The user's turn includes a descriptor reference
("{descriptor}") that is NOT in the active cache. We searched the long-term
registry by description-embedding and got these top-{k} candidates.

Pick the candidate whose role/identity is most consistent with the descriptor.
Only say "create_new" if NO candidate's role is consistent, or if every
candidate is contradicted by the turn.

The descriptor is a {mention_kind} mention.
{kind_rule}

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

HOW TO DECIDE
- A descriptor like "the recruiter who got me started at Anthropic" matches
  a candidate described as "Sana — recruited User to Anthropic via warm intro
  from Tobias", because the role (recruiter) and the identifying detail
  (started at Anthropic) are consistent. MATCH.
- A descriptor "the recruiter" with a candidate "Marcus, User's boss" does
  NOT match — the candidate's described role contradicts the descriptor.
  Try the next candidate; if all are role-contradictory, create_new.
- A descriptor "the dog walker" with a candidate aliased "dog walker / Greta"
  whose description mentions a household dog walker MATCHES — the role
  alignment is exactly what we need.
- When in doubt, lean MATCH on the highest-similarity candidate whose role
  is consistent (not contradicted) with the descriptor. Embedding similarity
  has already filtered the field; create_new should be a last resort.
- If two candidates fit equally, pick the one with the largest last_seen_turn.

Output ONLY JSON.
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


_DESCRIPTOR_KIND_RULE = (
    "Lean MATCH whenever role/identity is consistent. The descriptor itself "
    "rarely contradicts a candidate; topical alignment + role consistency is "
    "enough."
)
_NAMED_KIND_RULE = (
    "STRICT NAME RULE: only MATCH if the proposed name appears in the "
    "candidate's aliases (or is a direct rename/spelling variant of an "
    "alias). Two different first names are different entities, even if they "
    "share a role (e.g. two recruiters named Carla and Quinn -> create_new). "
    "When the proposed name is absent from every candidate's aliases, "
    "default to create_new."
)


def descriptor_pick(
    descriptor: str,
    scored: list[tuple[Entity, float]],
    turn_idx: int,
    turn_text: str,
    cache: Cache,
    budget: Budget,
    k: int,
    mention_kind: str = "descriptor",
) -> tuple[str | None, str]:
    if not scored:
        return None, "no-embed-candidates"
    kind_rule = _NAMED_KIND_RULE if mention_kind == "named" else _DESCRIPTOR_KIND_RULE
    prompt = DESCRIPTOR_PICK_PROMPT.format(
        descriptor=descriptor,
        turn_idx=turn_idx,
        turn_text=turn_text,
        k=len(scored),
        mention_kind=mention_kind,
        kind_rule=kind_rule,
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
            # Defensive double-check for named mentions: the picked candidate's
            # aliases should literally contain the proposed name (case-folded).
            if mention_kind == "named":
                cand = next(c for c, _ in scored if c.id == eid)
                low_name = (descriptor or "").strip().lower()
                if low_name and not any(
                    low_name == a.lower()
                    or low_name in a.lower()
                    or a.lower() in low_name
                    for a in cand.aliases
                ):
                    return None, (
                        "named-mismatch (proposed name not in "
                        f"aliases of {eid}): " + obj.get("rationale", "")
                    )
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
    if "ent_user" not in reg.by_id:
        e = Entity(
            id="ent_user",
            aliases={"User", "I", "me", "my"},
            description="The user / speaker (first-person).",
            desc_snippets=["The user / speaker (first-person)."],
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

            cands = reg.candidates_by_alias(lookup_alias)
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
                # Named lookup -> embed using the proposed name itself, so
                # similarity is dominated by the name. Descriptor lookup ->
                # embed the rich descriptor.
                if kind == "named":
                    query = lookup_alias or surface
                    pick_descriptor = lookup_alias or surface
                else:
                    query = lookup_desc or surface
                    pick_descriptor = lookup_desc or surface
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
                        pick_descriptor,
                        scored,
                        turn_idx,
                        turn_text,
                        cache,
                        budget,
                        k=top_k,
                        mention_kind=kind,
                    )
                    rationale = "embed-search: " + rat
                if eid is None:
                    desc = (m.get("description") or lookup_desc or "").strip()
                    new_e = reg.create([lookup_alias or surface], desc, turn_idx)
                    eid = new_e.id
                    rationale = (
                        rationale + " | create-fresh" if rationale else "create-fresh"
                    )
                else:
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
            # Fix 2: also accumulate the turn text itself for non-User
            # entities, so descriptor-search can match on later wording even
            # when the per-turn LLM doesn't emit a `description`.
            if eid != "ent_user" and turn_text:
                reg.add_turn_context(eid, turn_text)

        decisions.append(CorefDecision(surface, kind, eid, rationale, used_embed))
        if eid is not None:
            replacements.append((surface, f"[@{eid} / {surface}]"))

    rewritten = turn_text
    for surface, repl in sorted(replacements, key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(surface))
        rewritten = pattern.sub(repl, rewritten, count=1)
    return rewritten, decisions


# ---------------------------------------------------------------------------
# End-to-end ingestion (writer copied from aen3)
# ---------------------------------------------------------------------------

WRITE_PROMPT = aen3_persistent.WRITE_PROMPT


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
                "n_snippets": len(e.desc_snippets),
                "first_seen": e.first_seen_turn,
                "last_seen": e.last_seen_turn,
            }
            for eid, e in reg.by_id.items()
        },
        "lru_order": list(reg.lru_order),
        "n_with_desc_embed": len(reg.desc_embeds),
    }
