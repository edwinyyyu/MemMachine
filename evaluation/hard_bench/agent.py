"""Spreading-activation agent loop (per DESIGN.md).

Phase 1: planning probes (iterative concept → see → re-probe).
Phase 2: plan-only.
Phase 3: per-step execution with mid-step probes.

LLM model: gpt-5-mini for all calls.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .system import Hit, UnifiedSystem

# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------


SPREADING_PROBE_SYSTEM = """You are an assistant helping a user with a task. You have access to a memory system holding the user's team's prior conversations, decisions, and notes. To find relevant prior context, you generate PROBES — concise queries that the memory system will match against stored notes.

Your goal: surface the most relevant memory content for the task. Each round, see what's been retrieved so far and decide what's still missing — emit fresh probes that target the missing concepts.

Format each round:
  THINKING: <2-4 sentence reasoning about what's missing or worth probing for>
  PROBE: <a single concise query, 1-2 phrases>
  PROBE: <another query>
  PROBE: <up to 3 probes per round>

If the retrieved context already covers everything you need, emit just:
  THINKING: <reasoning>
  STOP

Stop after at most 6 rounds total."""


SPREADING_PROBE_USER_INITIAL = """Current time (now): {current_time}

Task: {task_prompt}

No memory context retrieved yet. Generate your first probes. If the task references relative times ("last week", "yesterday"), resolve them against the current time when generating probes."""


SPREADING_PROBE_USER_FOLLOWUP = """New memory items retrieved this round (ordered chronologically):

{new_snippets}

All probes so far: {prior_probes}

What's still missing? Generate next-round probes (or STOP)."""


PLAN_ONLY_SYSTEM = """You are an assistant. Given a task and accumulated memory context, write a numbered plan addressing the task. Reference specific memory items where relevant. Do NOT execute the plan; just outline the steps.

Output format:
1. <step>
2. <step>
...

Aim for 3-7 numbered steps. Be specific."""


PLAN_ONLY_USER = """Current time (now): {current_time}

Task: {task_prompt}

Accumulated memory context (ordered chronologically):
{context_block}

Write the numbered plan."""


EXEC_STEP_PROBE_SYSTEM = """For the current step, generate up to 3 specific PROBES to find any additional memory content needed to execute this step well. If existing context already covers it, emit:
  PROBE: none

Format:
  PROBE: <query>
  ..."""


EXEC_STEP_PROBE_USER = """Current time (now): {current_time}

Task: {task_prompt}

Plan:
{plan}

Currently executing step {step_num}: {step_text}

Context already accumulated (ordered chronologically):
{context_block}

What probes (if any) for this specific step?"""


EXEC_STEP_WRITE_SYSTEM = """You are executing one step of a plan. Use the accumulated memory context to write the step's deliverable in 1-3 concrete sentences. Cite specific memory items if they constrain or inform the step.

If the user's task is asking about a guideline-applicable plan and memory contains a rule that the current plan violates, FLAG IT and propose the correct alternative."""


EXEC_STEP_WRITE_USER = """Current time (now): {current_time}

Task: {task_prompt}

Plan:
{plan}

Step {step_num}: {step_text}

Accumulated memory context (ordered chronologically):
{context_block}

Write the step's deliverable (1-3 sentences). Cite memory items by [date, time] when referencing them."""


# -----------------------------------------------------------------------------
# Agent loop
# -----------------------------------------------------------------------------


@dataclass
class AgentResult:
    task_prompt: str
    phase1_probes: list[str] = field(default_factory=list)
    phase1_hits: list[Hit] = field(default_factory=list)  # event-memory hits
    # Entity-memory facts accumulated across probes (parallel block).
    # Stored as a flat list of dicts for serialization; the source-turn
    # property dicts let plant_retrieved track plants surfaced via entity.
    phase1_entity_facts: list[dict] = field(default_factory=list)
    plan: str = ""
    plan_steps: list[str] = field(default_factory=list)
    step_outputs: list[dict] = field(
        default_factory=list
    )  # {step_num, step_text, hits, entity_facts, output}
    final_transcript: str = ""

    def to_dict(self) -> dict:
        return {
            "task_prompt": self.task_prompt,
            "phase1_probes": self.phase1_probes,
            "phase1_hit_turn_ids": [h.turn_id for h in self.phase1_hits],
            "phase1_entity_fact_ids": [
                f.get("fact_uuid") for f in self.phase1_entity_facts
            ],
            "phase1_entity_fact_source_turn_ids": [
                f.get("source_turn_id") for f in self.phase1_entity_facts
            ],
            "plan": self.plan,
            "plan_steps": self.plan_steps,
            "step_outputs": [
                {
                    **{
                        k: v
                        for k, v in so.items()
                        if k not in ("hits", "entity_facts")
                    },
                    "hits": [
                        {"turn_id": h.turn_id, "score": h.score, "channel": h.channel}
                        for h in so.get("hits", [])
                    ],
                    "entity_fact_ids": [
                        f.get("fact_uuid") for f in so.get("entity_facts", [])
                    ],
                }
                for so in self.step_outputs
            ],
            "final_transcript": self.final_transcript,
        }


def _parse_probe_output(text: str) -> tuple[list[str], bool]:
    """Returns (probes, stopped)."""
    probes: list[str] = []
    stopped = False
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("STOP") or line.upper() == "STOP":
            stopped = True
        elif line.upper().startswith("PROBE:"):
            probe = line[6:].strip()
            if probe and probe.lower() != "none":
                probes.append(probe)
    return probes, stopped


def _format_hits_block(hits: list[Hit]) -> str:
    if not hits:
        return "(no memory items)"
    # Always present items in chronological order so the model does not
    # assume retrieval-rank ordering. turn_id is monotonic with time.
    sorted_hits = sorted(hits, key=lambda h: h.turn_id)
    return "\n".join(f"  - {h.text}" for h in sorted_hits)


def _format_entity_block(
    entity_facts_by_uuid: dict,
    entity_store,
) -> str:
    """Render the reference's reader-side rendering for accumulated entity
    facts: `format_facts_for_read` lines (with `[surface → Entity X]`
    markers) plus the `format_resolution_map` sidebar (per-entity surfaces
    + discriminating excerpts, with `⚠ COLLIDING SURFACES` warning).

    The resolution map is REBUILT from the accumulated facts' mentions so
    it reflects the agent's full accumulated view (not just the most
    recent probe's slice).
    """
    if not entity_facts_by_uuid or entity_store is None:
        return ""
    from hard_bench.entity import v2 as _v2

    facts_sorted = sorted(
        entity_facts_by_uuid.values(), key=lambda f: (f.ts, f.fact_uuid)
    )
    # Rebuild resolution_map from accumulated facts' mentions.
    resolution_map: dict[str, set[str]] = {}
    for f in facts_sorted:
        col = entity_store.collections.get(f.collection)
        for mid in f.mention_ids:
            eid = entity_store.registry.get_canonical(mid)
            if col is not None and mid in col.mentions_by_id:
                surface = col.mentions_by_id[mid].surface
                resolution_map.setdefault(eid, set()).add(surface)
    eid_alias = _v2._build_eid_alias(resolution_map)
    facts_block = _v2.format_facts_for_read(
        facts_sorted, entity_store, eid_alias=eid_alias
    )
    resmap_block = _v2.format_resolution_map(
        resolution_map, entity_store, eid_alias=eid_alias
    )
    return (
        f"## Resolution map\n{resmap_block}\n\n## Facts (chronological)\n{facts_block}"
    )


def _format_two_blocks(
    em_hits: list[Hit],
    entity_facts_by_uuid: dict,
    entity_store,
) -> str:
    """Concatenate the two-channel context: event memory block (cosine /
    temporal-filtered hits) and entity memory block (reference rendering
    of entity-resolved facts + resolution map). Two separate blocks; the
    agent sees both side-by-side.
    """
    parts: list[str] = []
    em_block = _format_hits_block(em_hits)
    if em_block != "(no memory items)":
        parts.append(f"=== EVENT MEMORY (ordered chronologically) ===\n{em_block}")
    ent_block = _format_entity_block(entity_facts_by_uuid, entity_store)
    if ent_block:
        parts.append(
            "=== ENTITY MEMORY (entity-resolved facts + resolution map) ===\n"
            f"{ent_block}"
        )
    if not parts:
        return "(no memory items)"
    return "\n\n".join(parts)


async def run_agent(
    task_prompt: str,
    system: UnifiedSystem,
    *,
    channels: tuple[str, ...] = ("em_cosine",),
    k_per_probe: int = 3,
    max_phase1_rounds: int = 6,
    per_step_probe_rounds: int = 2,
) -> AgentResult:
    """Run the spreading-activation agent loop on one task."""
    result = AgentResult(task_prompt=task_prompt)
    # Format current_time consistently for all prompts (ISO date+time, UTC).
    current_time_str = system.current_time.strftime("%Y-%m-%d %H:%M UTC (%A)")
    # Set task_prompt as the anchor phrase so em_temporal channel preserves
    # the temporal anchor across all spreading-activation probes (probes drop
    # anchors as they get more concept-specific).
    system.task_anchor_phrase = task_prompt

    want_entity = "em_entity" in channels
    entity_store_ref = (
        getattr(system, "_entity_store", None) if want_entity else None
    )

    async def _retrieve_both(p):
        em_coro = system.retrieve(p, channels=channels, k=k_per_probe)
        if want_entity and entity_store_ref is not None:
            ent_coro = system.retrieve_entity(p, k=k_per_probe)
            em_hits, ent_bundle = await asyncio.gather(em_coro, ent_coro)
        else:
            em_hits = await em_coro
            ent_bundle = None
        return em_hits, ent_bundle

    # ---- Phase 1: planning-time spreading activation ---------------------
    em_acc: dict[int, Hit] = {}
    entity_facts_acc: dict = {}  # fact_uuid -> Fact
    prior_probes: list[str] = []

    # Round 0: initial probe set
    sys_prompt = SPREADING_PROBE_SYSTEM
    user_prompt = SPREADING_PROBE_USER_INITIAL.format(
        task_prompt=task_prompt, current_time=current_time_str
    )
    raw = await system.llm(f"{sys_prompt}\n\n---\n\n{user_prompt}")
    new_probes, stopped = _parse_probe_output(raw)

    for round_idx in range(max_phase1_rounds):
        if not new_probes or stopped:
            break
        prior_probes.extend(new_probes)
        # Run all probes this round in parallel; collect both em hits and entity bundles
        round_results = await asyncio.gather(
            *(_retrieve_both(p) for p in new_probes)
        )
        new_em_this_round: list[Hit] = []
        new_entity_this_round: dict = {}
        for em_hits, ent_bundle in round_results:
            for h in em_hits:
                if h.turn_id not in em_acc:
                    em_acc[h.turn_id] = h
                    new_em_this_round.append(h)
            if ent_bundle is not None:
                for f in ent_bundle.get("facts", []):
                    if f.fact_uuid not in entity_facts_acc:
                        entity_facts_acc[f.fact_uuid] = f
                        new_entity_this_round[f.fact_uuid] = f
        if not new_em_this_round and not new_entity_this_round:
            break  # no fresh hits → stop

        # Generate next-round probes
        new_snippets_block = _format_two_blocks(
            new_em_this_round, new_entity_this_round, entity_store_ref
        )
        followup = SPREADING_PROBE_USER_FOLLOWUP.format(
            new_snippets=new_snippets_block,
            prior_probes=", ".join(prior_probes[-10:]),
        )
        raw = await system.llm(f"{sys_prompt}\n\n---\n\n{followup}")
        new_probes, stopped = _parse_probe_output(raw)
        if stopped:
            break

    result.phase1_probes = prior_probes
    result.phase1_hits = list(em_acc.values())
    # Serialize accumulated entity facts with source-turn props for plant tracking
    result.phase1_entity_facts = []
    for f in entity_facts_acc.values():
        src_tid = int(f.ts)
        src_entry = system.turn_table.get(src_tid)
        src_props = (
            src_entry[3]
            if src_entry is not None and len(src_entry) > 3
            else {"turn_id": src_tid}
        )
        result.phase1_entity_facts.append(
            {
                "fact_uuid": f.fact_uuid,
                "ts": f.ts,
                "text": f.text,
                "source_turn_id": src_tid,
                "source_turn_properties": dict(src_props),
            }
        )

    # ---- Phase 2: plan-only ----------------------------------------------
    context_block = _format_two_blocks(
        result.phase1_hits, entity_facts_acc, entity_store_ref
    )
    plan_raw = await system.llm(
        f"{PLAN_ONLY_SYSTEM}\n\n---\n\n{PLAN_ONLY_USER.format(task_prompt=task_prompt, context_block=context_block, current_time=current_time_str)}"
    )
    result.plan = plan_raw.strip()

    # Parse numbered steps
    steps: list[str] = []
    for line in result.plan.splitlines():
        line = line.strip()
        if line and (line[0].isdigit() and ("." in line[:4] or ")" in line[:4])):
            # strip leading "1. " or "1) "
            after = (
                line.split(".", 1)[-1] if "." in line[:4] else line.split(")", 1)[-1]
            )
            steps.append(after.strip())
    result.plan_steps = steps

    # ---- Phase 3: per-step exec ------------------------------------------
    em_exec: dict[int, Hit] = dict(em_acc)
    entity_exec_facts: dict = dict(entity_facts_acc)
    plan_block = result.plan

    for step_num, step_text in enumerate(steps, start=1):
        # Mid-step probes
        for _ in range(per_step_probe_rounds):
            probe_prompt = (
                f"{EXEC_STEP_PROBE_SYSTEM}\n\n---\n\n"
                f"{EXEC_STEP_PROBE_USER.format(task_prompt=task_prompt, plan=plan_block, step_num=step_num, step_text=step_text, context_block=_format_two_blocks(list(em_exec.values()), entity_exec_facts, entity_store_ref), current_time=current_time_str)}"
            )
            probe_raw = await system.llm(probe_prompt, reasoning_effort="low")
            probes, _stopped = _parse_probe_output(probe_raw)
            if not probes:
                break
            step_results = await asyncio.gather(*(_retrieve_both(p) for p in probes))
            for em_hits, ent_bundle in step_results:
                for h in em_hits:
                    if h.turn_id not in em_exec:
                        em_exec[h.turn_id] = h
                if ent_bundle is not None:
                    for f in ent_bundle.get("facts", []):
                        if f.fact_uuid not in entity_exec_facts:
                            entity_exec_facts[f.fact_uuid] = f

        # Write step output
        write_prompt = (
            f"{EXEC_STEP_WRITE_SYSTEM}\n\n---\n\n"
            f"{EXEC_STEP_WRITE_USER.format(task_prompt=task_prompt, plan=plan_block, step_num=step_num, step_text=step_text, context_block=_format_two_blocks(list(em_exec.values()), entity_exec_facts, entity_store_ref), current_time=current_time_str)}"
        )
        step_out = await system.llm(write_prompt)
        # Serialize entity facts new at this step (or all accumulated — we
        # serialize ALL so plant_retrieved sees per-step contribution too).
        step_entity_facts = []
        for f in entity_exec_facts.values():
            src_tid = int(f.ts)
            src_entry = system.turn_table.get(src_tid)
            src_props = (
                src_entry[3]
                if src_entry is not None and len(src_entry) > 3
                else {"turn_id": src_tid}
            )
            step_entity_facts.append(
                {
                    "fact_uuid": f.fact_uuid,
                    "ts": f.ts,
                    "text": f.text,
                    "source_turn_id": src_tid,
                    "source_turn_properties": dict(src_props),
                }
            )
        result.step_outputs.append(
            {
                "step_num": step_num,
                "step_text": step_text,
                "hits": list(em_exec.values()),
                "entity_facts": step_entity_facts,
                "output": step_out.strip(),
            }
        )

    # ---- Final transcript ------------------------------------------------
    transcript_parts = [f"Task: {task_prompt}", "", "Plan:", result.plan, ""]
    for so in result.step_outputs:
        transcript_parts.append(f"Step {so['step_num']}: {so['step_text']}")
        transcript_parts.append(f"  → {so['output']}")
        transcript_parts.append("")
    result.final_transcript = "\n".join(transcript_parts)

    return result
