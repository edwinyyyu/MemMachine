"""LLM-based ingestion-side alt-key generation.

For each conversation turn we ask an LLM (gpt-5-mini) to decide, given up to
two preceding turns as context, whether the current turn would benefit from
alt-keys — short rephrased/expanded versions of the turn that add retrieval
handles (anaphora resolution, alias linking, structured-fact annotation,
rare-entity context, etc.). The LLM is instructed to emit `SKIP` for turns
that don't need alt-keys.

Parallelism + caching:
  - Uses `BestshotLLMCache` shared with the rest of the bench so repeated
    runs cost nothing.
  - Uses a simple thread pool for uncached turns (default 8 workers).

This module is deliberately isolated from `ingest_regex_altkeys.py` — it
imports `AltKey` from there for type compatibility only.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterable

from openai import OpenAI

# We reuse the AltKey dataclass layout from the regex module so downstream
# augmented-index code works without change.
from ingest_regex_altkeys import AltKey  # noqa: F401

from best_shot import BestshotLLMCache

# ---------------------------------------------------------------------------
# Prompt versions
# ---------------------------------------------------------------------------
PROMPT_V1 = """\
You are reading one turn of a conversation at ingestion time. Decide whether \
this turn contains information that would be hard to find via its own text \
alone — information where a future query might fail to match the turn directly.

If yes, generate 1-3 short alt-keys: short rephrased/expanded versions of the \
turn that would help retrieval, each on its own line prefixed "ALT: ".

Common cases where alt-keys help:
- Anaphora: turn uses "it/he/they" referring to something named in prior \
turns — expand to include the referent.
- Short/ambiguous response: "yes"/"sure" answers whose content depends on \
prior turn — include prior context.
- Correction/retraction: turn overrides an earlier fact — tie correction to \
the thing it retracted.
- Alias: turn introduces a new name for something — link old and new names.
- Structured fact: turn states a specific numeric/dated/named fact (price, \
date, medication, person) — annotate the fact type.
- Rare entity: turn contains an uncommon proper noun or jargon term — \
include context that would help identify it.

If none of these apply and the turn is already clear on its own, output \
exactly:
SKIP

Prior context (up to 2 preceding turns):
{prev_context}

Current turn ({role}): {text}

Output either "SKIP" or one or more lines starting with "ALT: ". Nothing \
else."""


PROMPT_V2 = """\
You are reading one turn of a conversation at ingestion time and deciding \
whether to add alt-keys for retrieval. Alt-keys should only be generated for \
turns whose own text alone would likely NOT match a plausible future query.

Default behavior is SKIP. Most turns should SKIP. Only add alt-keys when the \
turn genuinely depends on outside context to be retrievable, or names a \
specific fact that a query might paraphrase.

Emit alt-keys only in these cases:
1. Anaphora that hides the referent: the turn uses "it/he/they/that" and the \
   referent named in the prior turn is semantically essential. Alt-key: same \
   statement with the referent substituted in.
2. Short reply that inherits meaning from the prior turn: e.g. user asks \
   "What year did you graduate?" and the turn is "2011". Alt-key: restate \
   the reply with the question subject.
3. Correction/retraction of an earlier fact. Alt-key: "corrected X: now Y".
4. Explicit alias introduction ("also called", "we renamed it"). Alt-key: \
   "<old name> is also <new name>".
5. Specific personal fact (name, date, number, location, medication, price, \
   pet, job, hobby) that a future question might paraphrase. Alt-key: the \
   fact stated concisely in third-person form, e.g. "Sarah's dog is named \
   Max".

Do NOT generate alt-keys for:
- Generic small talk, opinions, questions, emotional expressions
- Turns that are already self-contained
- Redundant rephrasings of the turn's own words
- Lists of capitalized words with no specific associative cue

Alt-key rules:
- Each alt-key 4-20 words, on its own line prefixed "ALT: "
- Max 2 alt-keys per turn; prefer 1
- Alt-keys must add new retrieval handles, not duplicate the original text
- If in doubt, output SKIP

Prior context (up to 2 preceding turns):
{prev_context}

Current turn ({role}): {text}

Output exactly "SKIP" or 1-2 lines beginning "ALT: ". Nothing else."""


PROMPT_V3 = """\
You generate alt-keys for conversation turns at ingestion time. An alt-key is \
a short rephrased statement that helps future semantic retrieval find this \
turn when the turn's own wording would miss a paraphrased query.

Default: SKIP. Only emit alt-keys for turns that clearly benefit.

Emit an alt-key when the turn falls into one of these cases:

A. Anaphoric short reply. The turn's meaning is unrecoverable without the \
   prior turn (pronoun, "yeah", "2011", "mine too"). Alt-key: restate the \
   reply with the prior-turn subject substituted in, 6-15 words.

B. Correction. The turn explicitly retracts or changes something. Alt-key: \
   "corrected: <new fact>", 5-15 words.

C. Alias. The turn introduces a new name for something already named. \
   Alt-key: "<alias A> is also known as <alias B>", 5-15 words.

D. Personal fact. The turn states a specific personal fact about a speaker \
   or named entity that a question might paraphrase: name, age, date, \
   location, profession, hobby, pet, medication, price, preference. \
   Alt-key: third-person fact restatement, 6-15 words, e.g. "Melanie works \
   as a nurse in Boston".

Do NOT emit alt-keys for:
- Opinions, feelings, questions, greetings, acknowledgements
- Turns already self-contained (e.g. "I love painting landscapes on weekends")
- Capitalized tokens alone — only emit if there is a specific fact the \
  capitalized token participates in
- Mere topic mentions without a fact

Format:
- Output "SKIP" (exactly) OR 1-2 lines starting "ALT: "
- Each alt-key 5-20 words
- No other text, no commentary, no numbering

Prior context (up to 2 preceding turns):
{prev_context}

Current turn ({role}): {text}

Output:"""


# Latest prompt used for ingestion
PROMPT_VERSIONS = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
    "v3": PROMPT_V3,
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class LLMAltKey:
    """An alt-key emitted by the LLM."""
    parent_index: int
    text: str
    source: str = "llm"  # e.g. "llm:v3"


@dataclass
class LLMTurnDecision:
    """The raw decision for one turn (for analysis/inspection)."""
    parent_index: int
    conversation_id: str
    turn_id: int
    role: str
    text: str
    prev_context: str
    raw_response: str
    skipped: bool
    alt_keys: list[str]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_prev_context(prev_turns: list[tuple[str, str]]) -> str:
    """prev_turns is a list of (role, text) tuples, most recent LAST."""
    if not prev_turns:
        return "(no prior context — this is the first turn)"
    lines = []
    for role, text in prev_turns:
        snippet = text[:200].replace("\n", " ").strip()
        lines.append(f"[{role}]: {snippet}")
    return "\n".join(lines)


def build_prompt(prompt_version: str, role: str, text: str,
                 prev_turns: list[tuple[str, str]]) -> str:
    template = PROMPT_VERSIONS[prompt_version]
    prev_ctx = build_prev_context(prev_turns[-2:])
    return template.format(
        prev_context=prev_ctx,
        role=role,
        text=text,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def parse_response(response: str) -> tuple[bool, list[str]]:
    """Return (skipped, alt_keys).

    - SKIP (case-insensitive, anywhere as the sole content of a line) means
      skip.
    - Otherwise, collect lines starting with "ALT: ".
    - Robust to minor format drift.
    """
    text = (response or "").strip()
    if not text:
        return True, []

    # If the whole thing is just "SKIP"
    stripped = text.strip().strip("'\"`")
    if stripped.upper() == "SKIP":
        return True, []

    alts: list[str] = []
    saw_skip = False
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        up = line.upper()
        if up == "SKIP":
            saw_skip = True
            continue
        if up.startswith("ALT:") or up.startswith("ALT "):
            # strip the prefix after the colon or space
            idx = line.find(":")
            if idx >= 0:
                alt = line[idx + 1:].strip()
            else:
                alt = line[3:].strip()
            # Trim trailing quotes/punct if wrapped
            alt = alt.strip().strip("'\"")
            if alt and len(alt) >= 2:
                alts.append(alt)

    if alts:
        return False, alts[:3]
    if saw_skip:
        return True, []
    # unknown format -> treat as SKIP (safer default)
    return True, []


# ---------------------------------------------------------------------------
# LLM call (threaded, cached)
# ---------------------------------------------------------------------------
class LLMAltKeyGenerator:
    def __init__(
        self,
        client: OpenAI | None = None,
        model: str = "gpt-5-mini",
        prompt_version: str = "v3",
        max_workers: int = 8,
        cache: BestshotLLMCache | None = None,
    ):
        self.client = client or OpenAI(timeout=60.0)
        self.model = model
        self.prompt_version = prompt_version
        self.max_workers = max_workers
        self.cache = cache or BestshotLLMCache()
        self._cache_lock = Lock()
        self._counter_lock = Lock()
        self.n_cached = 0
        self.n_uncached = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _cache_key_prompt(self, prompt: str) -> str:
        # Include prompt_version in the cached "prompt string" so key space
        # doesn't collide across prompt versions.
        return f"[ingest_llm_altkeys/{self.prompt_version}]\n" + prompt

    def call_one(self, role: str, text: str,
                 prev_turns: list[tuple[str, str]]) -> str:
        prompt = build_prompt(self.prompt_version, role, text, prev_turns)
        cache_key_prompt = self._cache_key_prompt(prompt)

        with self._cache_lock:
            cached = self.cache.get(self.model, cache_key_prompt)
        if cached is not None:
            with self._counter_lock:
                self.n_cached += 1
            return cached

        raw = ""
        pt = 0
        ct = 0
        last_err: Exception | None = None
        for tok_budget in (1200, 2400, 4000):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=tok_budget,
                )
                raw = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                last_err = None
                break
            except Exception as e:  # retry w/ higher budget on length errors
                msg = str(e)
                last_err = e
                if "max_tokens" in msg or "output limit" in msg:
                    continue
                # other errors: one retry
                if tok_budget == 1200:
                    continue
                raise
        if last_err is not None:
            # If the final try still failed, swallow as empty -> parsed as SKIP.
            # Preserves pipeline flow and is logged via empty raw response.
            raw = ""

        with self._cache_lock:
            self.cache.put(self.model, cache_key_prompt, raw)
        with self._counter_lock:
            self.n_uncached += 1
            self.total_prompt_tokens += int(pt)
            self.total_completion_tokens += int(ct)
        return raw

    def save(self) -> None:
        with self._cache_lock:
            self.cache.save()


# ---------------------------------------------------------------------------
# Pipeline over a list of segments
# ---------------------------------------------------------------------------
def generate_altkeys_for_conversation(
    generator: LLMAltKeyGenerator,
    segments: list,   # sorted by turn_id, same conversation
    progress_cb=None,
) -> list[LLMTurnDecision]:
    """For a single conversation, build prev-context windows and call LLM
    per turn. Sequential within conversation (prev_turns dependency is cheap
    to compute but ordering isn't required because prev_turns window uses the
    previous turns' original text, not their decisions)."""
    # We actually can parallelize: each turn's prompt depends only on the
    # TWO PREVIOUS original turn texts, not on prior decisions. Precompute
    # all prompts, then map in parallel.
    prev_by_idx: list[list[tuple[str, str]]] = []
    window: list[tuple[str, str]] = []
    for seg in segments:
        prev_by_idx.append(list(window[-2:]))
        window.append((seg.role, seg.text))

    decisions: list[LLMTurnDecision | None] = [None] * len(segments)

    def _do(i: int) -> tuple[int, LLMTurnDecision]:
        seg = segments[i]
        prev = prev_by_idx[i]
        raw = generator.call_one(seg.role, seg.text, prev)
        skipped, alts = parse_response(raw)
        dec = LLMTurnDecision(
            parent_index=seg.index,
            conversation_id=seg.conversation_id,
            turn_id=seg.turn_id,
            role=seg.role,
            text=seg.text,
            prev_context=build_prev_context(prev),
            raw_response=raw,
            skipped=skipped,
            alt_keys=alts,
        )
        if progress_cb is not None:
            progress_cb()
        return i, dec

    with ThreadPoolExecutor(max_workers=generator.max_workers) as ex:
        futures = [ex.submit(_do, i) for i in range(len(segments))]
        for f in as_completed(futures):
            i, dec = f.result()
            decisions[i] = dec

    return [d for d in decisions if d is not None]


def generate_altkeys_for_all(
    generator: LLMAltKeyGenerator,
    segments: list,
    log_every: int = 100,
) -> list[LLMTurnDecision]:
    """Group segments by conversation_id, order by turn_id, run LLM."""
    by_conv: dict[str, list] = {}
    for s in segments:
        by_conv.setdefault(s.conversation_id, []).append(s)
    all_decisions: list[LLMTurnDecision] = []
    total = len(segments)
    done = [0]
    t0 = time.time()
    last_save = [t0]

    def cb():
        done[0] += 1
        if done[0] % log_every == 0:
            el = time.time() - t0
            rate = done[0] / max(el, 1e-6)
            eta = (total - done[0]) / max(rate, 1e-6)
            print(
                f"  [{done[0]}/{total}] cached={generator.n_cached} "
                f"uncached={generator.n_uncached} "
                f"rate={rate:.1f} seg/s eta={eta:.0f}s",
                flush=True,
            )
            # checkpoint cache every ~30s
            if time.time() - last_save[0] > 30:
                generator.save()
                last_save[0] = time.time()

    for cid in sorted(by_conv.keys()):
        segs_sorted = sorted(by_conv[cid], key=lambda s: s.turn_id)
        decisions = generate_altkeys_for_conversation(generator, segs_sorted, cb)
        all_decisions.extend(decisions)

    generator.save()
    return all_decisions


# ---------------------------------------------------------------------------
# Convenience: convert decisions to AltKey list
# ---------------------------------------------------------------------------
def decisions_to_altkeys(
    decisions: Iterable[LLMTurnDecision],
    source_tag: str = "llm",
) -> list[AltKey]:
    out: list[AltKey] = []
    for d in decisions:
        if d.skipped:
            continue
        for alt in d.alt_keys:
            out.append(AltKey(
                parent_index=d.parent_index,
                heuristic=source_tag,
                text=alt,
            ))
    return out
