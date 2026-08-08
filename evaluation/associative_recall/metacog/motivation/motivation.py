"""Motivation generator for autonomous agents.

A standalone module — does NOT import any harness. Harnesses import this.

The generator emits a categorical motivation state plus an imperative
``drive_directive`` string suitable for injection into the agent's
SYSTEM_PROMPT or per-turn followup. The generator is one ``gpt-5-mini``
call per invocation with structured ``json_schema`` output and
``reasoning_effort="low"``.

Design notes (kept terse here; see BUILD-NOTES.md for the full design):

- State set is FIXED at six categorical labels (see ``VALID_STATES``).
- Decay rules and forced-rotation rules are encoded INSIDE the prompt
  rather than in Python so the LLM is the single source of truth for
  the transition (avoids split logic that disagrees).
- Anti-paraphrase guard: prompt requires explicit justification when
  the chosen state matches the input state, and pushes toward switching
  by default after the dwell minimum.
- Anti-wandering guard: a Python-side soft minimum dwell time of 3
  turns is checked; if the LLM wants to switch before that, we accept
  only when an external trigger fires (large turns_since_last_completion
  or turns_since_last_user_input). The prompt explains this rule.
- The Python wrapper is intentionally thin: it formats the prompt,
  invokes the LLM, parses JSON, and clamps a few fields. It does NOT
  itself decide the next state.

Public interface:

    state = initial_motivation_state(turn=0)
    state = await update_motivation(
        openai_client,
        current_state=state,
        recent_activity_summary="...",
        unresolved_goals=["..."],
        turns_since_last_user_input=2,
        turns_since_last_completion=4,
        turns_since_motivation_update=3,
        current_turn=10,
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---------- State definition ----------


VALID_STATES: frozenset[str] = frozenset(
    {
        "curious",
        "focused",
        "anxious",
        "restless",
        "satisfied",
        "bored",
    }
)


# Tunables. Kept module-level so harnesses can override before calling.
MIN_DWELL_TURNS: int = 3       # Soft floor: don't switch state earlier than this.
FORCED_ROTATION_TURNS: int = 8  # Hard ceiling: must switch by this point.
DEFAULT_MODEL: str = "gpt-5-mini"
DEFAULT_REASONING_EFFORT: str = "low"
DEFAULT_MAX_COMPLETION_TOKENS: int = 800


@dataclass
class MotivationState:
    """Categorical motivation + intensity + bookkeeping.

    ``state``           one of ``VALID_STATES``.
    ``intensity``       float in [0.0, 1.0]; 0 = barely there, 1 = peak.
    ``since_turn``      turn index at which this state began (used to
                        compute dwell length downstream).
    ``rationale``       1-2 sentence reasoning the generator emitted.
    ``drive_directive`` imperative sentence to inject into agent prompt.
    """

    state: str = "curious"
    intensity: float = 0.4
    since_turn: int = 0
    rationale: str = "Initial state at task start; no prior activity."
    drive_directive: str = (
        "Because the task is just starting, take one concrete step toward "
        "the most actionable open goal now."
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "intensity": self.intensity,
            "since_turn": self.since_turn,
            "rationale": self.rationale,
            "drive_directive": self.drive_directive,
        }


def initial_motivation_state(turn: int = 0) -> MotivationState:
    """Construct the seed state used at task start."""
    return MotivationState(since_turn=turn)


# ---------- JSON schema for the LLM call ----------


MOTIVATION_SCHEMA: dict[str, Any] = {
    "name": "motivation_update",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "enum": sorted(VALID_STATES),
                "description": (
                    "The chosen motivation state for this turn. Must be one "
                    "of the six fixed categories."
                ),
            },
            "intensity": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": (
                    "How strongly the state is felt this turn, in [0,1]. "
                    "0 = barely; 1 = peak. Calibrate against the input "
                    "intensity rather than treating each call as fresh."
                ),
            },
            "rationale": {
                "type": "string",
                "description": (
                    "1-2 sentences explaining why this state and "
                    "intensity. If state is unchanged from input, the "
                    "rationale MUST justify why no shift is warranted."
                ),
            },
            "drive_directive": {
                "type": "string",
                "description": (
                    "A single IMPERATIVE sentence to inject into the "
                    "agent's prompt, of the shape 'Because you're <state> "
                    "about <object>, do <concrete action> now.' Must "
                    "command, not suggest. Must reference a concrete "
                    "object drawn from the recent activity summary or "
                    "the unresolved goals list."
                ),
            },
        },
        "required": ["state", "intensity", "rationale", "drive_directive"],
        "additionalProperties": False,
    },
}


# ---------- Prompt template ----------


MOTIVATION_SYSTEM_PROMPT = """\
You are a motivation generator for an autonomous problem-solving agent. \
Your job each invocation: read the inputs, decide which of six \
categorical motivation states the agent should be in this turn, and \
emit one IMPERATIVE drive_directive sentence that the agent will \
literally read in its prompt.

The six states (mutually exclusive):
- curious     — explore something new; no specific task urgency.
- focused     — push the current task to completion.
- anxious     — high urgency on a deadline, bug, or unresolved issue.
- restless    — hasn't made progress recently; needs to switch approach.
- satisfied  — just completed something; low urgency; taking stock.
- bored       — no input, no obvious next action; needs to find something.

You receive these inputs (the user message will fill them in):
- current_state: the previous motivation state and its intensity.
- since_turn / turns_since_motivation_update: how long the current
  state has been held.
- recent_activity_summary: concise text of what just happened.
- unresolved_goals: list of open sub-decisions / incomplete tasks.
- turns_since_last_user_input.
- turns_since_last_completion.
- current_turn.

DECAY AND TRANSITION RULES (apply these — they are NOT suggestions):

1. focused: if turns_since_last_completion is high relative to dwell
   length and the recent_activity_summary shows no concrete progress,
   shift toward restless.
2. anxious: if the urgency-source (deadline, bug, unresolved issue)
   is no longer present in recent_activity_summary or unresolved_goals,
   shift toward focused.
3. satisfied: if a completion is no longer fresh (several turns since
   last completion) and unresolved_goals is non-empty, shift toward
   focused; if unresolved_goals is empty, shift toward bored.
4. bored: if any latent topic of interest has surfaced in working
   memory (recent_activity_summary mentions a new entity, question,
   or anomaly), shift toward curious.
5. restless: if a fresh angle or new approach is now visible in
   recent_activity_summary, shift toward curious or focused.
6. curious: if a concrete unresolved_goal is dominating attention,
   shift toward focused.

DWELL AND ROTATION RULES (hard):

- MINIMUM DWELL: a state should generally hold for at least 3 turns
  before switching. EXCEPTIONS that justify earlier switching:
  (a) turns_since_last_user_input == 0 with a clearly different topic,
  (b) a completion just happened (state -> satisfied),
  (c) a sharp new urgency just appeared (state -> anxious).
- FORCED ROTATION: if the same state has been held for more than 8
  turns (turns_since_motivation_update > 8), you MUST pick a different
  state. Choose the one whose decay rule above best fits the inputs;
  do not pick at random.

ANTI-PARAPHRASE RULE (read carefully):

- DEFAULT TO SHIFTING. If you are tempted to keep the input state,
  the rationale field MUST explicitly justify why no shift is warranted
  (cite which decay rule does NOT yet fire and why). If you cannot
  produce that justification cleanly, switch.
- The drive_directive must be a COMMAND, not advice. Forbidden
  phrasings: "you might consider", "perhaps try", "it could help to",
  "feel free to". Required shape: "Because you're <state> about
  <object>, do <action> now." The <object> must be a concrete noun
  taken from recent_activity_summary or unresolved_goals — never
  abstract ("the task", "things"). The <action> must be a single
  observable next step.

INTENSITY CALIBRATION:

- Default to intensity in [0.4, 0.7]. Use >0.8 only when the inputs
  clearly justify peak (anxious near a deadline, satisfied right after
  a major completion, bored after many empty turns).
- Avoid drift: if inputs are similar to the previous call, intensity
  should not jump by more than ~0.2.

Output strictly the JSON schema you've been given. No prose.
"""


MOTIVATION_USER_TEMPLATE = """\
INPUTS:

current_state:
  state: {state}
  intensity: {intensity:.2f}
  since_turn: {since_turn}
  prior_rationale: {prior_rationale}

current_turn: {current_turn}
turns_since_motivation_update: {turns_since_motivation_update}
turns_since_last_user_input: {turns_since_last_user_input}
turns_since_last_completion: {turns_since_last_completion}

recent_activity_summary:
{recent_activity_summary}

unresolved_goals ({n_goals}):
{unresolved_goals_block}

Decide the motivation state for this turn. Apply the decay, dwell, and
rotation rules in order. Then emit the imperative drive_directive."""


# ---------- Helpers ----------


def _format_goals_block(goals: Iterable[str]) -> str:
    items = [g.strip() for g in goals if g and g.strip()]
    if not items:
        return "  (none)"
    return "\n".join(f"  - {g}" for g in items)


def _clamp_intensity(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.5
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _trigger_present(
    *,
    turns_since_last_user_input: int,
    turns_since_last_completion: int,
    turns_since_motivation_update: int,
) -> bool:
    """External trigger that justifies bypassing the dwell minimum.

    Mirrors the "exceptions" listed in the prompt so Python and prompt
    agree about when an early switch is OK.
    """
    if turns_since_motivation_update > FORCED_ROTATION_TURNS:
        return True
    if turns_since_last_user_input == 0:
        return True
    if turns_since_last_completion == 0:
        return True
    return False


def _enforce_dwell(
    *,
    proposed: MotivationState,
    previous: MotivationState,
    turns_since_motivation_update: int,
    trigger_present: bool,
    current_turn: int,
) -> MotivationState:
    """Soft enforcement of MIN_DWELL_TURNS.

    If the LLM tries to switch state before the dwell minimum and no
    external trigger justifies it, we revert the categorical state to
    the previous one but KEEP the new intensity, rationale, and
    directive (so the agent still benefits from updated reasoning).
    """
    if proposed.state == previous.state:
        # No switch — leave since_turn anchored to the original onset.
        proposed.since_turn = previous.since_turn
        return proposed

    if turns_since_motivation_update < MIN_DWELL_TURNS and not trigger_present:
        logger.info(
            "motivation: blocked early switch %s -> %s (dwell=%d, no trigger)",
            previous.state,
            proposed.state,
            turns_since_motivation_update,
        )
        proposed.state = previous.state
        proposed.since_turn = previous.since_turn
        # Mark in rationale that the dwell guard fired.
        proposed.rationale = (
            f"[dwell-guard: kept '{previous.state}'] " + proposed.rationale
        )
        return proposed

    # Genuine switch.
    proposed.since_turn = current_turn
    return proposed


def _force_rotation_if_stuck(
    *,
    proposed: MotivationState,
    previous: MotivationState,
    turns_since_motivation_update: int,
    current_turn: int,
) -> MotivationState:
    """Hard rotation guard.

    If the LLM ignored the forced-rotation rule in the prompt, we pick
    a fallback state mechanically. The fallback aims to be plausible
    rather than optimal — runtime testing should validate whether this
    branch ever fires in practice.
    """
    if proposed.state != previous.state:
        return proposed
    if turns_since_motivation_update <= FORCED_ROTATION_TURNS:
        return proposed

    fallback_map = {
        "curious": "focused",
        "focused": "restless",
        "anxious": "focused",
        "restless": "curious",
        "satisfied": "bored",
        "bored": "curious",
    }
    new_state = fallback_map.get(previous.state, "curious")
    logger.info(
        "motivation: forced rotation %s -> %s after %d turns",
        previous.state,
        new_state,
        turns_since_motivation_update,
    )
    proposed.state = new_state
    proposed.since_turn = current_turn
    proposed.rationale = (
        f"[forced-rotation after {turns_since_motivation_update} turns] "
        + proposed.rationale
    )
    return proposed


# ---------- Main entrypoint ----------


async def update_motivation(
    openai_client: Any,
    *,
    current_state: MotivationState,
    recent_activity_summary: str,
    unresolved_goals: list[str],
    turns_since_last_user_input: int,
    turns_since_last_completion: int,
    turns_since_motivation_update: int,
    current_turn: int,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> MotivationState:
    """Run one motivation update.

    On any LLM/parse failure, returns ``current_state`` unchanged but
    bumps ``since_turn`` only if the dwell-guard exceptions would have
    allowed it. Failure is non-fatal: the agent keeps its prior state.
    """

    user_msg = MOTIVATION_USER_TEMPLATE.format(
        state=current_state.state,
        intensity=current_state.intensity,
        since_turn=current_state.since_turn,
        prior_rationale=(current_state.rationale or "(none)").strip(),
        current_turn=current_turn,
        turns_since_motivation_update=turns_since_motivation_update,
        turns_since_last_user_input=turns_since_last_user_input,
        turns_since_last_completion=turns_since_last_completion,
        recent_activity_summary=(recent_activity_summary or "(empty)").strip(),
        n_goals=len(unresolved_goals or []),
        unresolved_goals_block=_format_goals_block(unresolved_goals or []),
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": MOTIVATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_completion_tokens": max_completion_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": MOTIVATION_SCHEMA,
        },
        "reasoning_effort": reasoning_effort,
    }

    try:
        resp = await openai_client.chat.completions.create(**kwargs)
    except Exception as exc:
        # Older deployments may reject reasoning_effort; retry without it.
        msg = str(exc).lower()
        if "reasoning_effort" in msg or "unsupported" in msg:
            kwargs.pop("reasoning_effort", None)
            try:
                resp = await openai_client.chat.completions.create(**kwargs)
            except Exception as exc2:
                logger.warning("motivation: LLM call failed: %r", exc2)
                return current_state
        else:
            logger.warning("motivation: LLM call failed: %r", exc)
            return current_state

    raw = (resp.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("motivation: JSON parse failed: %r raw=%r", exc, raw)
        return current_state

    chosen_state = parsed.get("state", "")
    if chosen_state not in VALID_STATES:
        logger.warning(
            "motivation: invalid state %r (allowed=%s)",
            chosen_state,
            sorted(VALID_STATES),
        )
        return current_state

    proposed = MotivationState(
        state=chosen_state,
        intensity=_clamp_intensity(parsed.get("intensity", 0.5)),
        since_turn=current_turn,  # placeholder; corrected by dwell guard
        rationale=str(parsed.get("rationale", "")).strip(),
        drive_directive=str(parsed.get("drive_directive", "")).strip(),
    )

    trigger = _trigger_present(
        turns_since_last_user_input=turns_since_last_user_input,
        turns_since_last_completion=turns_since_last_completion,
        turns_since_motivation_update=turns_since_motivation_update,
    )

    proposed = _enforce_dwell(
        proposed=proposed,
        previous=current_state,
        turns_since_motivation_update=turns_since_motivation_update,
        trigger_present=trigger,
        current_turn=current_turn,
    )

    proposed = _force_rotation_if_stuck(
        proposed=proposed,
        previous=current_state,
        turns_since_motivation_update=turns_since_motivation_update,
        current_turn=current_turn,
    )

    return proposed
