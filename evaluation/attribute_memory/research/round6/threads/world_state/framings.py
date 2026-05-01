"""F1-F4 prompt variants for the world-state framing experiment.

Each framing shares the SAME subject-heuristics, output format, and embedding-dedup
pass from R7. Only the opening framing paragraph varies. This isolates the effect
of framing from other prompt-engineering variables.

F3 additionally emits an optional second line with multi-label routes ("<E1>/<C1>,
<E2>/<C2>") when the fact primarily describes more than one entity. The caller
may parse the first line only (single-label mode) or both lines (multi-label mode).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
R6_DIR = HERE.parent
sys.path.insert(0, str(R6_DIR))

import openai
from strategies import (
    CallCounter,
    EmbedCache,
    LLMCache,
    RouteDecision,
    TopicState,
    cosine,
    embed,
    llm,
)

# Shared scaffolding: the "heuristics + output format" that R7 uses.
# Each framing slots an opening paragraph above this scaffolding.

_SCAFFOLD = """Heuristics for primary subject:
- A fact about a person's pet/child/partner has that pet/child/partner as subject.
- A fact about the user's business/employer: the USER is the subject (their career/business), unless the fact is strictly a property of the business as an independent entity.
- Reuse an existing topic if the subject matches.

FACT: {fact_text}

EXISTING TOPICS:
{existing_list}

Output exactly one line: <Subject>/<Category> (e.g. "Luna/Profile", "User/Employment", "User/Business"). Nothing else.
"""

# F3 scaffold allows optional multi-label second line.
_SCAFFOLD_F3 = """Heuristics for primary subject:
- A fact about a person's pet/child/partner has that pet/child/partner as subject.
- A fact about the user's business/employer: the USER is the subject (their career/business), unless the fact is strictly a property of the business as an independent entity.
- Reuse an existing topic if the subject matches.
- If a single fact PRIMARILY updates MORE THAN ONE entity's state (e.g. "X and Y got engaged"), list each as a separate route.

FACT: {fact_text}

EXISTING TOPICS:
{existing_list}

Output format: one line per primary subject. Each line: <Subject>/<Category>.
Most facts have exactly one primary subject -- output one line. Only emit multiple
lines when multiple entities' states are primarily affected.
"""


F1_OPENING = """Route this fact to an append-only topic log. Each topic is keyed by the PRIMARY SUBJECT the fact describes (not merely mentioned).
"""

F2_OPENING = """You are maintaining a model of each entity's current state in the world. Given the new fact, identify which entity's state is being described or changed, and route to that entity's topic log. Be consistent: if an entity is referred to by multiple names (Luna/the cat/my kitty, Jamie/J/my wife), route all mentions to the same log.
"""

F3_OPENING = """You are tracking how the world evolves across a conversation. Each fact is an observation that updates the state of one or more entities (people, pets, places, jobs, beliefs). Identify which entity's state this fact updates. If multiple entities are primarily affected (e.g. "X and Y got engaged", "Son got a hamster"), identify all of them.
"""

F4_OPENING = """You are simulating the user's life as a set of persistent objects (people, pets, places, jobs, beliefs). Each conversation reveals properties of these objects or creates new ones. For this fact, identify which object it primarily describes, and route to that object's topic log. Treat every alias for the same object (e.g. Luna = the cat = the kitty = the furball) as the same object.
"""


def _build_prompt(
    opening: str, fact_text: str, existing_list: str, scaffold: str = _SCAFFOLD
) -> str:
    return (
        opening
        + "\n"
        + scaffold.format(fact_text=fact_text, existing_list=existing_list)
    )


def _existing_topic_block(topics: list[TopicState]) -> str:
    if not topics:
        return "(none yet)"
    return "\n".join(f"- {t.name} (entries={t.entry_count})" for t in topics)


def _normalize_route(line: str) -> str:
    line = line.strip().strip("`'\"").strip()
    if not line:
        return ""
    if "/" not in line:
        line = f"User/{line}"
    parts = line.split("/", 1)
    return f"{parts[0].strip()}/{parts[1].strip()}"


MERGE_THRESHOLD = 0.72


def _embed_merge(
    proposed: str,
    fact_text: str,
    topics: list[TopicState],
    client: openai.OpenAI,
    embed_cache: EmbedCache,
    counter: CallCounter,
) -> tuple[str, bool, str]:
    """R7's embedding dedup step.

    Returns (final_topic_name, is_new, reason)."""
    existing_names = {t.name for t in topics}
    if proposed in existing_names:
        return proposed, False, "accept_existing"
    if not topics:
        return proposed, True, "accept_new"
    fact_emb = embed(client, embed_cache, counter, fact_text)
    best: tuple[str, float] | None = None
    for t in topics:
        if not t.centroid:
            continue
        s = cosine(fact_emb, t.centroid)
        if best is None or s > best[1]:
            best = (t.name, s)
    if best is not None and best[1] >= MERGE_THRESHOLD:
        return (
            best[0],
            False,
            f"embed_merge(proposed={proposed}, merged={best[0]}@{best[1]:.2f})",
        )
    return proposed, True, "accept_new"


def make_strategy(opening: str, label: str) -> Callable:
    """Build a single-label framing strategy with R7's scaffolding + embed dedup."""

    def strat(
        fact_text: str,
        topics: list[TopicState],
        client: openai.OpenAI,
        llm_cache: LLMCache,
        embed_cache: EmbedCache,
        counter: CallCounter,
    ) -> RouteDecision:
        existing_list = _existing_topic_block(topics)
        prompt = _build_prompt(opening, fact_text, existing_list, _SCAFFOLD)
        raw = llm(client, llm_cache, counter, prompt).strip()
        first_line = raw.splitlines()[0] if raw else ""
        proposed = _normalize_route(first_line) or "User/Other"
        final, is_new, reason = _embed_merge(
            proposed, fact_text, topics, client, embed_cache, counter
        )
        return RouteDecision(
            topic_name=final,
            is_new=is_new,
            reason=f"{label}:{reason}",
        )

    strat.__name__ = f"strategy_{label}"
    return strat


def make_f3_multilabel() -> Callable:
    """F3 with multi-label emission. Returns a RouteDecision whose multi_topics
    is populated if the LLM emits >1 line. topic_name == first line (for single-label
    scoring compatibility)."""

    def strat(
        fact_text: str,
        topics: list[TopicState],
        client: openai.OpenAI,
        llm_cache: LLMCache,
        embed_cache: EmbedCache,
        counter: CallCounter,
    ) -> RouteDecision:
        existing_list = _existing_topic_block(topics)
        prompt = _build_prompt(F3_OPENING, fact_text, existing_list, _SCAFFOLD_F3)
        raw = llm(client, llm_cache, counter, prompt).strip()
        lines = [ln for ln in (raw.splitlines() if raw else []) if ln.strip()]
        # Parse up to ~3 topic lines; ignore anything after
        raw_topics: list[str] = []
        for ln in lines[:3]:
            # Strip leading bullet chars
            t = ln.strip().lstrip("-*0123456789.) ").strip()
            norm = _normalize_route(t)
            if norm and norm not in raw_topics:
                raw_topics.append(norm)
        if not raw_topics:
            raw_topics = ["User/Other"]

        # Dedup each topic against existing via embeddings
        resolved: list[tuple[str, bool, str]] = []
        for p in raw_topics:
            final, is_new, reason = _embed_merge(
                p, fact_text, topics, client, embed_cache, counter
            )
            resolved.append((final, is_new, reason))

        primary = resolved[0]
        multi = [r[0] for r in resolved] if len(resolved) > 1 else None
        return RouteDecision(
            topic_name=primary[0],
            is_new=primary[1],
            reason=f"f3_multi:{primary[2]}",
            multi_topics=multi,
        )

    strat.__name__ = "strategy_f3_multilabel"
    return strat


F1 = make_strategy(F1_OPENING, "F1_baseline")
F2 = make_strategy(F2_OPENING, "F2_world_state_entity")
F3_MULTI = make_f3_multilabel()
F4 = make_strategy(F4_OPENING, "F4_world_state_simulation")

FRAMINGS: dict[str, Callable] = {
    "F1_baseline": F1,
    "F2_entity_state": F2,
    "F3_state_change_multi": F3_MULTI,
    "F4_simulation": F4,
}
