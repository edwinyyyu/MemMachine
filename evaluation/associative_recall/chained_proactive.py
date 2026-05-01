"""Chained proactive retrieval with entity discovery (DAG-structured).

Task -> LLM plans info-node DAG (parametric reasoning) -> executes retrievals
in topological order where downstream nodes depend on entities discovered in
upstream nodes.

Three systems are defined here:

  System A -- `single_shot` (em_v2f_speakerformat baseline)
              1 cue-gen LLM call over the raw task prompt, retrieve top-K.

  System B -- `flat_proactive`
              Decompose into 3-6 info needs in parallel, retrieve for each
              using only the task prompt as context. No cross-node entity
              feeding. (Structurally matches proactive_memory.run_proactive
              without the sufficiency audit round.)

  System C -- `chained_proactive`
              Two-phase: (1) LLM emits a JSON plan of info_nodes with
              dependencies: `entity_discovery` nodes that discover entities,
              and `per_entity_fact` nodes parameterised by upstream-discovered
              entities. (2) Execute DAG in topological order, using LLM-driven
              entity extraction on each node's hits to feed downstream.
              Final: LLM sufficiency check; if not done and under-budget, add
              more nodes and iterate (max 2 iterations).

Caches (dedicated — do not collide with other agents):
  cache/chained_plan_cache.json
  cache/chained_cuegen_cache.json
  cache/chained_entity_cache.json
  cache/chained_suff_cache.json
  cache/chained_flat_cache.json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from em_architectures import (
    V2F_MODEL,
    EMHit,
    _dedupe_by_turn_id,
    _merge_by_max_score,
    _MergedLLMCache,
    _query_em,
)
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory

CACHE_DIR = Path(__file__).resolve().parent / "cache"

CHAINED_PLAN_CACHE = CACHE_DIR / "chained_plan_cache.json"
CHAINED_CUEGEN_CACHE = CACHE_DIR / "chained_cuegen_cache.json"
CHAINED_ENTITY_CACHE = CACHE_DIR / "chained_entity_cache.json"
CHAINED_SUFF_CACHE = CACHE_DIR / "chained_suff_cache.json"
CHAINED_FLAT_CACHE = CACHE_DIR / "chained_flat_cache.json"


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


PLAN_PROMPT = """\
You are planning retrieval over a conversation memory between {participant_1} \
and {participant_2}. The memory only contains chat turns from that \
conversation. You will later run semantic retrieval calls against it.

TASK:
{task_prompt}

You must output a JSON plan describing the INFO NODES you will retrieve and \
how they depend on each other. Use parametric knowledge ABOUT THE TASK \
STRUCTURE — do NOT try to answer the task yet.

Two node types:
- "entity_discovery": retrieves content that lets us ENUMERATE a set of \
  entities (e.g. a list of people, projects, pets, concerns).
- "per_entity_fact": retrieves facts ABOUT EACH entity discovered by an \
  upstream node. Use this for lookups like "dietary restrictions for each \
  friend", "progress state for each project".

Required fields per node:
- id: short string like "n1", "n2"
- type: "entity_discovery" | "per_entity_fact"
- target: short declarative description of what this node retrieves
- expected_vocab: 3-6 words/phrases likely to appear in matching chat turns
- depends_on: list of upstream node ids (empty for roots)
- for_each: upstream node id whose discovered entities parameterise this \
  node (REQUIRED for per_entity_fact, null otherwise)
- entity_category: short label for the entity class this node yields \
  (required for entity_discovery; null otherwise) e.g. "person name", \
  "project name", "pet name"

Prefer 1-3 entity_discovery nodes and 1-3 per_entity_fact nodes. If the \
task names explicit entities, you may still emit per_entity_fact nodes \
with empty depends_on and list those entities in `explicit_entities`.

Output ONLY a JSON object:
{{
  "explicit_entities": ["..."],
  "info_nodes": [
    {{"id": "n1", "type": "entity_discovery", "target": "...",
      "expected_vocab": ["..."], "depends_on": [], "for_each": null,
      "entity_category": "..."}},
    {{"id": "n2", "type": "per_entity_fact", "target": "...",
      "expected_vocab": ["..."], "depends_on": ["n1"], "for_each": "n1",
      "entity_category": null}}
  ]
}}"""


CUEGEN_NODE_PROMPT = """\
Generate search cues for semantic retrieval over a conversation between \
{participant_1} and {participant_2}. Turns are embedded in the format \
"<speaker_name>: <chat content>"; cues will be embedded the same way.

Overall task (for context):
{task_prompt}

Info node target:
{target}
Expected vocabulary: {expected_vocab}
{entity_clause}

Generate 2 cues. Each cue MUST begin with "{participant_1}: " or \
"{participant_2}: ". Use chat-register text that would actually appear in \
the conversation — NOT questions.

Format:
CUE: <speaker_name>: <text>
CUE: <speaker_name>: <text>
Nothing else."""


ENTITY_EXTRACT_PROMPT = """\
You just retrieved chat turns for an entity-discovery info node. Extract \
the distinct entities of the requested category from these turns.

Entity category: {entity_category}
Node target: {target}

Retrieved excerpts:
{hits_block}

Output ONLY a JSON object listing up to 6 distinct entities as short \
strings (names or noun phrases). If the turns don't clearly name any \
entity of this category, output an empty list.

{{"entities": ["..."]}}"""


SUFFICIENCY_PROMPT = """\
You are auditing whether the retrieved content (across a DAG of info nodes) \
is SUFFICIENT to complete the task.

TASK:
{task_prompt}

Plan and per-node retrievals:
{plan_block}

Decide:
- answerable: true|false — can a competent AI use these retrievals to \
  produce a good answer?
- if false, suggest up to 2 EXTRA nodes to add (same schema as the plan).

Output ONLY JSON:
{{
  "answerable": true|false,
  "extra_nodes": [ ... ]
}}"""


FLAT_DECOMPOSE_PROMPT = """\
You are planning retrieval over a conversation memory between {participant_1} \
and {participant_2} to complete a task. Unlike a DAG, here you will issue \
all retrievals in PARALLEL from the task prompt alone.

TASK:
{task_prompt}

Decompose into 3-6 INFO NEEDS (flat list, no dependencies). Each need \
describes a category of information to retrieve. For each need:
- need: short declarative phrase
- expected_vocab: 3-6 words/phrases likely in matching chat turns

Output ONLY JSON:
{{"needs": [
  {{"need": "...", "expected_vocab": ["..."]}},
  ...
]}}"""


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


CUE_RE = re.compile(r"^\s*CUE\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def _strip_quotes(s: str) -> str:
    return s.strip().strip('"').strip("'").strip()


def parse_cues(response: str, max_cues: int = 2) -> list[str]:
    cues: list[str] = []
    for m in CUE_RE.finditer(response):
        cue = _strip_quotes(m.group(1))
        if cue:
            cues.append(cue)
        if len(cues) >= max_cues:
            break
    return cues


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(t[start : end + 1])
    except Exception:
        return None


def parse_plan(response: str) -> dict:
    obj = _extract_json(response) or {}
    explicit = obj.get("explicit_entities") or []
    nodes_raw = obj.get("info_nodes") or []
    nodes: list[dict] = []
    for n in nodes_raw:
        if not isinstance(n, dict):
            continue
        nid = str(n.get("id") or "").strip()
        ntype = str(n.get("type") or "").strip().lower()
        target = str(n.get("target") or "").strip()
        if (
            not nid
            or not target
            or ntype not in ("entity_discovery", "per_entity_fact")
        ):
            continue
        depends_on = [
            str(x).strip() for x in (n.get("depends_on") or []) if str(x).strip()
        ]
        for_each = n.get("for_each")
        for_each = str(for_each).strip() if for_each else None
        exp_vocab = [
            str(x).strip() for x in (n.get("expected_vocab") or []) if str(x).strip()
        ]
        entity_cat = n.get("entity_category")
        entity_cat = str(entity_cat).strip() if entity_cat else None
        nodes.append(
            {
                "id": nid,
                "type": ntype,
                "target": target,
                "expected_vocab": exp_vocab,
                "depends_on": depends_on,
                "for_each": for_each,
                "entity_category": entity_cat,
            }
        )
    return {
        "explicit_entities": [str(e).strip() for e in explicit if str(e).strip()],
        "info_nodes": nodes,
    }


def parse_entities(response: str) -> list[str]:
    obj = _extract_json(response) or {}
    ents = obj.get("entities") or []
    out: list[str] = []
    for e in ents:
        s = str(e).strip()
        if s and s.lower() not in {"none", "n/a", "unknown"}:
            out.append(s)
    # dedupe preserving order, case-insensitive
    seen: set[str] = set()
    dedup: list[str] = []
    for e in out:
        k = e.lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(e)
    return dedup[:6]


def parse_sufficiency(response: str) -> dict:
    obj = _extract_json(response) or {}
    ans = obj.get("answerable")
    answerable = (
        bool(ans)
        if isinstance(ans, bool)
        else str(ans).strip().lower() in ("true", "yes", "1")
    )
    extra = obj.get("extra_nodes") or []
    # Re-run node schema validation.
    extra_nodes: list[dict] = []
    for n in extra:
        if not isinstance(n, dict):
            continue
        nid = str(n.get("id") or "").strip()
        ntype = str(n.get("type") or "").strip().lower()
        target = str(n.get("target") or "").strip()
        if (
            not nid
            or not target
            or ntype not in ("entity_discovery", "per_entity_fact")
        ):
            continue
        extra_nodes.append(
            {
                "id": nid,
                "type": ntype,
                "target": target,
                "expected_vocab": [
                    str(x).strip()
                    for x in (n.get("expected_vocab") or [])
                    if str(x).strip()
                ],
                "depends_on": [
                    str(x).strip()
                    for x in (n.get("depends_on") or [])
                    if str(x).strip()
                ],
                "for_each": str(n.get("for_each")).strip()
                if n.get("for_each")
                else None,
                "entity_category": str(n.get("entity_category")).strip()
                if n.get("entity_category")
                else None,
            }
        )
    return {"answerable": answerable, "extra_nodes": extra_nodes}


def parse_flat_needs(response: str) -> list[dict]:
    obj = _extract_json(response) or {}
    raw = obj.get("needs") or []
    out: list[dict] = []
    for n in raw:
        if not isinstance(n, dict):
            continue
        need = str(n.get("need") or "").strip()
        if not need:
            continue
        out.append(
            {
                "need": need,
                "expected_vocab": [
                    str(x).strip()
                    for x in (n.get("expected_vocab") or [])
                    if str(x).strip()
                ],
            }
        )
    return out


# --------------------------------------------------------------------------
# LLM helper
# --------------------------------------------------------------------------


async def _llm_call(
    openai_client,
    prompt: str,
    cache: _MergedLLMCache,
) -> tuple[str, bool]:
    cached = cache.get(V2F_MODEL, prompt)
    if cached is not None:
        return cached, True
    resp = await openai_client.chat.completions.create(
        model=V2F_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content or ""
    cache.put(V2F_MODEL, prompt, text)
    return text, False


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------


def _format_hits_for_prompt(
    hits: list[EMHit], max_items: int = 8, max_len: int = 160
) -> str:
    if not hits:
        return "(no retrievals)"
    top = sorted(hits, key=lambda h: -h.score)[:max_items]
    top = sorted(top, key=lambda h: h.turn_id)
    lines = []
    for h in top:
        txt = h.text.replace("\n", " ")
        if len(txt) > max_len:
            txt = txt[:max_len] + "..."
        lines.append(f"[Turn {h.turn_id}, {h.role}]: {txt}")
    return "\n".join(lines)


def _topological_order(nodes: list[dict]) -> list[dict]:
    ids = {n["id"] for n in nodes}
    by_id = {n["id"]: n for n in nodes}
    visited: set[str] = set()
    order: list[dict] = []

    def visit(nid: str, stack: set[str]) -> None:
        if nid in visited or nid not in by_id:
            return
        if nid in stack:
            # cycle; break
            return
        stack.add(nid)
        for d in by_id[nid]["depends_on"]:
            if d in ids:
                visit(d, stack)
        stack.discard(nid)
        visited.add(nid)
        order.append(by_id[nid])

    for n in nodes:
        visit(n["id"], set())
    return order


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    hits: list[EMHit]
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# System A: single-shot v2f baseline
# --------------------------------------------------------------------------


SINGLE_SHOT_PROMPT = """\
You are generating search cues for semantic retrieval over a conversation \
between {participant_1} and {participant_2}. Turns are embedded in the \
format "<speaker_name>: <chat content>" and your cues will be embedded \
the same way.

Task:
{task_prompt}

Generate 2 search cues. Each cue MUST begin with "{participant_1}: " or \
"{participant_2}: ". Use specific vocabulary that would appear in target \
turns. Do NOT write questions; write text that would actually appear in a \
chat message.

Format:
CUE: <speaker_name>: <text>
CUE: <speaker_name>: <text>
Nothing else."""


async def run_single_shot(
    memory: EventMemory,
    task_prompt: str,
    participants: tuple[str, str],
    *,
    K: int,
    cuegen_cache: _MergedLLMCache,
    openai_client,
) -> RetrievalResult:
    p1, p2 = participants
    prompt = SINGLE_SHOT_PROMPT.format(
        task_prompt=task_prompt,
        participant_1=p1,
        participant_2=p2,
    )
    raw, cache_hit = await _llm_call(openai_client, prompt, cuegen_cache)
    cues = parse_cues(raw, max_cues=2)

    primer = await _query_em(
        memory, task_prompt, vector_search_limit=K, expand_context=0
    )
    per_cue: list[list[EMHit]] = []
    for cue in cues:
        per_cue.append(
            await _query_em(memory, cue, vector_search_limit=K, expand_context=0)
        )
    merged = _merge_by_max_score([primer, *per_cue])
    merged = _dedupe_by_turn_id(merged)[:K]
    return RetrievalResult(
        hits=merged,
        metadata={
            "system": "single_shot",
            "cues": cues,
            "cuegen_cache_hit": cache_hit,
            "n_llm_calls": 1,
            "n_turns_retrieved": len(merged),
        },
    )


# --------------------------------------------------------------------------
# System B: flat proactive (parallel decomposition, task-prompt-only context)
# --------------------------------------------------------------------------


async def run_flat_proactive(
    memory: EventMemory,
    task_prompt: str,
    participants: tuple[str, str],
    *,
    K_per_need: int = 15,
    K_final: int = 50,
    flat_cache: _MergedLLMCache,
    cuegen_cache: _MergedLLMCache,
    openai_client,
) -> RetrievalResult:
    p1, p2 = participants

    decompose_prompt = FLAT_DECOMPOSE_PROMPT.format(
        participant_1=p1,
        participant_2=p2,
        task_prompt=task_prompt,
    )
    decompose_raw, decompose_hit = await _llm_call(
        openai_client, decompose_prompt, flat_cache
    )
    needs = parse_flat_needs(decompose_raw)
    if not needs:
        needs = [
            {"need": task_prompt.strip().split("\n")[0][:120], "expected_vocab": []}
        ]

    n_llm_calls = 1
    all_batches: list[list[EMHit]] = []
    per_need_meta: list[dict] = []

    for need in needs:
        vocab_str = ", ".join(need.get("expected_vocab") or []) or "(none)"
        cue_prompt = CUEGEN_NODE_PROMPT.format(
            participant_1=p1,
            participant_2=p2,
            task_prompt=task_prompt,
            target=need["need"],
            expected_vocab=vocab_str,
            entity_clause="",
        )
        cue_raw, cue_hit = await _llm_call(openai_client, cue_prompt, cuegen_cache)
        n_llm_calls += 1
        cues = parse_cues(cue_raw, max_cues=2)

        # Primer on the need text + per-cue retrievals.
        primer = await _query_em(
            memory, need["need"], vector_search_limit=K_per_need, expand_context=0
        )
        batches_this_need: list[list[EMHit]] = [primer]
        for cue in cues:
            batches_this_need.append(
                await _query_em(
                    memory, cue, vector_search_limit=K_per_need, expand_context=0
                )
            )
        merged_need = _dedupe_by_turn_id(_merge_by_max_score(batches_this_need))
        all_batches.extend(batches_this_need)
        per_need_meta.append(
            {
                "need": need["need"],
                "cues": cues,
                "cue_cache_hit": cue_hit,
                "n_hits": len(merged_need),
                "top_turn_ids": [
                    h.turn_id for h in sorted(merged_need, key=lambda h: -h.score)[:10]
                ],
            }
        )

    global_merged = _dedupe_by_turn_id(_merge_by_max_score(all_batches))[:K_final]
    return RetrievalResult(
        hits=global_merged,
        metadata={
            "system": "flat_proactive",
            "n_llm_calls": n_llm_calls,
            "n_turns_retrieved": len(global_merged),
            "decompose_cache_hit": decompose_hit,
            "needs": per_need_meta,
        },
    )


# --------------------------------------------------------------------------
# System C: chained proactive (DAG + entity discovery)
# --------------------------------------------------------------------------


async def _retrieve_for_node(
    memory: EventMemory,
    task_prompt: str,
    node: dict,
    participants: tuple[str, str],
    upstream_entities: list[str],
    *,
    K_per_cue: int,
    cuegen_cache: _MergedLLMCache,
    openai_client,
) -> tuple[list[EMHit], dict]:
    """Retrieve for a single info node. If it depends on upstream entities,
    we parameterise the cue-gen once per entity (capped)."""
    p1, p2 = participants
    vocab_str = ", ".join(node.get("expected_vocab") or []) or "(none)"
    batches: list[list[EMHit]] = []
    cuegen_info: list[dict] = []
    n_llm_calls = 0

    # Primer: raw target text (no LLM call).
    batches.append(
        await _query_em(
            memory, node["target"], vector_search_limit=K_per_cue, expand_context=0
        )
    )

    if node["type"] == "per_entity_fact" and upstream_entities:
        # One cue-gen per upstream entity (capped at 5 to contain budget).
        for entity in upstream_entities[:5]:
            entity_clause = (
                f"This node is parameterised by the entity: {entity!r}. "
                f"Cues should retrieve {node['target']} SPECIFICALLY for {entity}."
            )
            prompt = CUEGEN_NODE_PROMPT.format(
                participant_1=p1,
                participant_2=p2,
                task_prompt=task_prompt,
                target=node["target"],
                expected_vocab=vocab_str,
                entity_clause=entity_clause,
            )
            raw, hit = await _llm_call(openai_client, prompt, cuegen_cache)
            if not hit:
                n_llm_calls += 1
            cues = parse_cues(raw, max_cues=2)
            # Also query the entity name directly as a primer.
            batches.append(
                await _query_em(
                    memory, entity, vector_search_limit=K_per_cue, expand_context=0
                )
            )
            for cue in cues:
                batches.append(
                    await _query_em(
                        memory, cue, vector_search_limit=K_per_cue, expand_context=0
                    )
                )
            cuegen_info.append({"entity": entity, "cues": cues, "cache_hit": hit})
    else:
        # entity_discovery OR per_entity_fact without resolved upstream.
        entity_clause = ""
        if node["type"] == "entity_discovery" and node.get("entity_category"):
            entity_clause = (
                f"Target retrieves ENUMERATION of entities of category "
                f"{node['entity_category']!r}."
            )
        prompt = CUEGEN_NODE_PROMPT.format(
            participant_1=p1,
            participant_2=p2,
            task_prompt=task_prompt,
            target=node["target"],
            expected_vocab=vocab_str,
            entity_clause=entity_clause,
        )
        raw, hit = await _llm_call(openai_client, prompt, cuegen_cache)
        if not hit:
            n_llm_calls += 1
        cues = parse_cues(raw, max_cues=2)
        for cue in cues:
            batches.append(
                await _query_em(
                    memory, cue, vector_search_limit=K_per_cue, expand_context=0
                )
            )
        cuegen_info.append({"entity": None, "cues": cues, "cache_hit": hit})

    merged = _dedupe_by_turn_id(_merge_by_max_score(batches))
    return merged, {
        "n_llm_calls": n_llm_calls,
        "cuegen": cuegen_info,
        "n_hits": len(merged),
        "batches_all": batches,  # keep for global merge
    }


async def _extract_entities_for_node(
    node: dict,
    hits: list[EMHit],
    *,
    entity_cache: _MergedLLMCache,
    openai_client,
) -> tuple[list[str], bool, int]:
    if node["type"] != "entity_discovery":
        return [], False, 0
    if not hits:
        return [], False, 0
    hits_block = _format_hits_for_prompt(hits, max_items=8, max_len=200)
    prompt = ENTITY_EXTRACT_PROMPT.format(
        entity_category=node.get("entity_category") or "entity",
        target=node["target"],
        hits_block=hits_block,
    )
    raw, hit = await _llm_call(openai_client, prompt, entity_cache)
    n_llm_calls = 0 if hit else 1
    ents = parse_entities(raw)
    return ents, hit, n_llm_calls


async def run_chained_proactive(
    memory: EventMemory,
    task_prompt: str,
    participants: tuple[str, str],
    *,
    K_per_cue: int = 10,
    K_final: int = 50,
    max_iterations: int = 2,
    plan_cache: _MergedLLMCache,
    cuegen_cache: _MergedLLMCache,
    entity_cache: _MergedLLMCache,
    suff_cache: _MergedLLMCache,
    openai_client,
) -> RetrievalResult:
    p1, p2 = participants
    n_llm_calls = 0
    iterations: list[dict] = []

    # ----- Phase 1: plan DAG -----
    plan_prompt = PLAN_PROMPT.format(
        participant_1=p1,
        participant_2=p2,
        task_prompt=task_prompt,
    )
    plan_raw, plan_hit = await _llm_call(openai_client, plan_prompt, plan_cache)
    if not plan_hit:
        n_llm_calls += 1
    plan = parse_plan(plan_raw)

    if not plan["info_nodes"]:
        # Fallback: treat entire task as one entity_discovery node.
        plan = {
            "explicit_entities": [],
            "info_nodes": [
                {
                    "id": "n1",
                    "type": "entity_discovery",
                    "target": task_prompt.strip().split("\n")[0][:160],
                    "expected_vocab": [],
                    "depends_on": [],
                    "for_each": None,
                    "entity_category": "mention",
                }
            ],
        }

    # ----- Phase 2: iterate (plan exec + sufficiency loop) -----
    node_hits: dict[str, list[EMHit]] = {}
    node_entities: dict[str, list[str]] = {}
    node_meta: dict[str, dict] = {}
    all_batches: list[list[EMHit]] = []
    discovered_entity_count = 0

    current_nodes = list(plan["info_nodes"])

    for iter_idx in range(max_iterations):
        # Topological order over current_nodes (not-yet-executed).
        pending = [n for n in current_nodes if n["id"] not in node_hits]
        if not pending:
            break
        ordered = _topological_order(pending)

        iter_rec: dict = {"iteration": iter_idx, "nodes": []}

        for node in ordered:
            upstream: list[str] = []
            if node.get("for_each"):
                upstream = list(node_entities.get(node["for_each"], []))
                # If upstream has no discovered entities yet, also fall back
                # to explicit_entities if any.
                if not upstream and plan.get("explicit_entities"):
                    upstream = list(plan["explicit_entities"])

            hits, rmeta = await _retrieve_for_node(
                memory,
                task_prompt,
                node,
                participants,
                upstream,
                K_per_cue=K_per_cue,
                cuegen_cache=cuegen_cache,
                openai_client=openai_client,
            )
            n_llm_calls += rmeta["n_llm_calls"]
            all_batches.extend(rmeta["batches_all"])
            node_hits[node["id"]] = hits

            ents, ent_hit, ent_calls = await _extract_entities_for_node(
                node,
                hits,
                entity_cache=entity_cache,
                openai_client=openai_client,
            )
            n_llm_calls += ent_calls
            node_entities[node["id"]] = ents
            if ents:
                discovered_entity_count += len(ents)

            node_meta[node["id"]] = {
                "id": node["id"],
                "type": node["type"],
                "target": node["target"],
                "depends_on": node["depends_on"],
                "for_each": node.get("for_each"),
                "entity_category": node.get("entity_category"),
                "upstream_entities_used": upstream,
                "cuegen": rmeta["cuegen"],
                "n_hits": rmeta["n_hits"],
                "top_turn_ids": [
                    h.turn_id for h in sorted(hits, key=lambda h: -h.score)[:10]
                ],
                "extracted_entities": ents,
                "entity_cache_hit": ent_hit,
            }
            iter_rec["nodes"].append(node["id"])

        iterations.append(iter_rec)

        # Sufficiency check: skip on last iteration to save budget.
        if iter_idx == max_iterations - 1:
            break

        plan_block_lines: list[str] = []
        for node in current_nodes:
            if node["id"] not in node_meta:
                continue
            m = node_meta[node["id"]]
            plan_block_lines.append(
                f"### Node {node['id']} ({node['type']}): {node['target']}"
            )
            if m.get("extracted_entities"):
                plan_block_lines.append(
                    f"Discovered entities: {m['extracted_entities']}"
                )
            plan_block_lines.append(
                _format_hits_for_prompt(node_hits[node["id"]], max_items=5, max_len=150)
            )
            plan_block_lines.append("")
        plan_block = "\n".join(plan_block_lines)

        suff_prompt = SUFFICIENCY_PROMPT.format(
            task_prompt=task_prompt,
            plan_block=plan_block,
        )
        suff_raw, suff_hit = await _llm_call(openai_client, suff_prompt, suff_cache)
        if not suff_hit:
            n_llm_calls += 1
        suff = parse_sufficiency(suff_raw)
        iter_rec["sufficiency"] = {
            "answerable": suff["answerable"],
            "n_extra_nodes": len(suff["extra_nodes"]),
            "cache_hit": suff_hit,
        }
        if suff["answerable"] or not suff["extra_nodes"]:
            break
        # Add new nodes for next iteration.
        # Give them fresh unique ids to avoid collision.
        added = 0
        for en in suff["extra_nodes"]:
            base_id = en["id"] or f"n{len(current_nodes) + 1}"
            new_id = base_id
            k = 0
            while new_id in {n["id"] for n in current_nodes}:
                k += 1
                new_id = f"{base_id}_x{k}"
            en["id"] = new_id
            current_nodes.append(en)
            added += 1
        if added == 0:
            break

    # ----- Final global merge -----
    global_merged = _dedupe_by_turn_id(_merge_by_max_score(all_batches))[:K_final]

    return RetrievalResult(
        hits=global_merged,
        metadata={
            "system": "chained_proactive",
            "n_llm_calls": n_llm_calls,
            "n_turns_retrieved": len(global_merged),
            "plan_cache_hit": plan_hit,
            "plan": {
                "explicit_entities": plan.get("explicit_entities", []),
                "info_nodes": [
                    {
                        "id": n["id"],
                        "type": n["type"],
                        "target": n["target"],
                        "depends_on": n["depends_on"],
                        "for_each": n.get("for_each"),
                        "entity_category": n.get("entity_category"),
                    }
                    for n in current_nodes
                ],
            },
            "nodes": [
                node_meta[n["id"]] for n in current_nodes if n["id"] in node_meta
            ],
            "n_iterations": len(iterations),
            "iterations": iterations,
            "n_entities_discovered": discovered_entity_count,
        },
    )
