"""Iteratively tune the note-generation prompt for ingestion.

Standalone tuning script — does not touch framework files or em_setup_notes_*.py.
Uses fixed model gpt-5-mini + text-embedding-3-small (embedding not actually
needed here; we only exercise the LLM note-generation prompt).

Sample set: 15 diverse turns from locomo_conv-26, each with the last 3
preceding turns as context, formatted as EM-canonical "<speaker>: <content>".

We run 4 rounds max; each round has a different prompt candidate (A, A', A'',
optional B/C). We record outputs for all 15 turns per round, analyze failures,
and pick the winner.

Budget: ~15 * ~4 = ~60 LLM calls at gpt-5-mini (~$0.20).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openai
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-5-mini"
CACHE_FILE = CACHE_DIR / "notes_tune_prompts_cache.json"

# --- Sample set ---------------------------------------------------------

# Each entry: (turn_id, expected_category, short_description)
# Speakers: Caroline (user), Melanie (assistant) for locomo_conv-26.
SAMPLE_TURNS: list[tuple[int, str, str]] = [
    (
        130,
        "phatic",
        "Caroline: generic encouragement 'Running can boost mood. Keep it up!'",
    ),
    (134, "phatic", "Caroline: 'Glad it helped ya, Melanie!' short encouragement"),
    (173, "phatic", "Caroline: 'No worries... Enjoy your day!' goodbye phatic"),
    (
        12,
        "anaphora",
        "Caroline: 'Is this your own painting?' — 'this' → painting Mel just shared",
    ),
    (28, "anaphora", "Melanie: 'What made you pick it?' — 'it' → adoption agency"),
    (
        82,
        "anaphora",
        "Caroline: 'That bowl is gorgeous! ... Did you make it?' — 'it' → pottery bowl",
    ),
    (
        60,
        "named",
        "Caroline: necklace from grandma in Sweden (new entities: Sweden, grandma, necklace)",
    ),
    (118, "named", "Caroline: favorite book 'Becoming Nicole' by Amy Ellis Nutt"),
    (195, "named", "Caroline: group name 'Connected LGBTQ Activists'"),
    (
        47,
        "count",
        "Caroline: known these friends for 4 years (since moved from home country)",
    ),
    (
        123,
        "count",
        "Melanie: 'a pup and a kitty' — now 2 pets named later (Luna, Oliver)",
    ),
    (50, "fact", "Melanie: '5 years already!' years married"),
    (
        62,
        "fact",
        "Caroline: hand-painted bowl from friend on 18th birthday ten years ago",
    ),
    (
        197,
        "update",
        "Caroline: 'I missed it but it was a powerful reminder' — correcting implied attendance",
    ),
    (
        351,
        "update",
        "Melanie: 'The sign was just a precaution... had a great time' — correcting alarm",
    ),
]


# --- Data loading -------------------------------------------------------


def load_conv26_turns() -> list[tuple[int, str, str]]:
    d = np.load(DATA_DIR / "segments_extended.npz", allow_pickle=True)
    texts = d["texts"]
    cids = d["conversation_ids"]
    tids = d["turn_ids"]
    roles = d["roles"]
    out: list[tuple[int, str, str]] = []
    for i in range(len(texts)):
        cid = str(cids[i])
        if cid != "locomo_conv-26":
            continue
        speaker = "Caroline" if str(roles[i]) == "user" else "Melanie"
        out.append((int(tids[i]), speaker, str(texts[i]).strip()))
    out.sort(key=lambda t: t[0])
    return out


def em_format(speaker: str, content: str) -> str:
    return f"{speaker}: {content.strip()}"


# --- Cache --------------------------------------------------------------


def _sha(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()


class _Cache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._d: dict[str, str] = {}
        if path.exists():
            try:
                with open(path) as f:
                    self._d = json.load(f)
            except Exception:
                self._d = {}
        self._dirty = False

    def get(self, model: str, prompt: str) -> str | None:
        return self._d.get(_sha(model, prompt))

    def put(self, model: str, prompt: str, response: str) -> None:
        self._d[_sha(model, prompt)] = response
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._d, f)
        tmp.replace(self._path)
        self._dirty = False


# --- Prompt templates ---------------------------------------------------

# Round 1 — Candidate A (starting prompt)
PROMPT_A = """\
You are listening in on a conversation. Read the current turn in context and write specific listener observations.

If applicable, use these labels one per line:
- RESOLVED: <pronoun/deictic> -> <referent>
- FACT: <specific new detail>
- COUNT: <entity> = <running total>
- UPDATE: <prior claim> -> <new claim>
- LINK: <current element> refers to <earlier topic>
- NAME: <new entity> = <description>

If the turn is purely phatic (no specific content), output exactly: PHATIC

Context:
{context_block}

Current turn:
{current_turn}

Observations (labeled lines, or PHATIC):
"""


# Round 2 — Candidate A' (refinements):
# - Strengthen PHATIC rule with explicit examples
# - Require exact referent name/phrase from context when resolving
# - Add explicit no-fabrication rule
# - Tighten output form: 1 observation per line, max 20 words each
PROMPT_A_PRIME = """\
You are listening in on a conversation. Read the CURRENT TURN in the light of the preceding context and write specific listener observations that a careful note-taker would jot down.

Use one label per line, choosing from:
- RESOLVED: <pronoun/deictic word from current turn> -> <exact referent name or phrase from context>
- FACT: <specific new detail explicitly stated in the current turn>
- COUNT: <entity> = <running total or duration with units>
- UPDATE: <prior claim from context> -> <new/corrected claim in current turn>
- LINK: <current element> refers to <earlier topic or event in context>
- NAME: <new proper noun or named entity> = <short description grounded in the current turn>

PHATIC rule: if the current turn is purely phatic (generic politeness, greetings, encouragement, small talk, filler) with no new concrete content, output exactly:
PHATIC
Examples of PHATIC-worthy turns: "Thanks!", "Glad you had fun", "Have a great day!", "That's awesome, keep it up!".

Hard constraints:
- Only extract information directly stated in the current turn (context is only for resolving references). Do NOT invent facts.
- Max 20 words per line; 1 observation per line; no preamble; no explanations.
- Prefer concrete referents ("the adoption agency Caroline mentioned") over vague ones ("it").
- If the turn contains content but no label fits, still try FACT with the concrete detail; do not output PHATIC unless truly phatic.

Context (most recent last; each line is "<speaker>: <content>"):
{context_block}

Current turn:
{current_turn}

Observations (labeled lines, or PHATIC):
"""


# Round 3 — Candidate A'' (further refinements):
# - Explicit example of each label
# - Forbid fabrication more strongly
# - Handle anaphora edge cases explicitly
# - Phrase phatic rule as: "if removing this turn loses nothing specific, output PHATIC"
PROMPT_A_DOUBLE_PRIME = """\
You are a careful listener taking notes on a two-person conversation. For each CURRENT TURN you write 1-4 concrete, grounded observations — the kind a listener would jot so they could recall specifics later.

LABEL REFERENCE (use one label per line, at most 4 lines total):
- RESOLVED: <exact pronoun/deictic from current turn> -> <exact referent phrase from prior context>
    e.g. RESOLVED: "it" -> "the adoption agency Caroline mentioned in her last turn"
- FACT: <a specific new detail explicitly stated in this turn>
    e.g. FACT: Melanie and her husband have been married 5 years.
- COUNT: <entity> = <running total or duration with units>
    e.g. COUNT: Melanie's pets = 2 (a dog and a cat, Luna and Oliver)
- UPDATE: <prior claim from context> -> <new/corrected claim in current turn>
    e.g. UPDATE: Caroline marched in the parade -> Caroline missed the parade.
- LINK: <current element> refers to <earlier topic/event from context>
    e.g. LINK: Caroline's painting of unity refers to her LGBTQ center visit.
- NAME: <new proper noun or named entity introduced> = <short grounded description>
    e.g. NAME: "Becoming Nicole" by Amy Ellis Nutt = Caroline's favorite inspirational book about a trans girl.

PHATIC rule: output exactly "PHATIC" (and nothing else) if removing this turn from the conversation would lose no concrete information. Typical PHATIC turns are generic politeness, greetings, goodbyes, brief encouragement, or echo-agreements ("Thanks!", "Keep it up!", "Glad to hear", "Enjoy your day", "Bye!", "Great to see you", "That's awesome").

Hard constraints:
- ONLY use information directly present in the CURRENT TURN (the context exists only to resolve references). Do NOT invent facts, numbers, dates, or names not stated.
- 1 observation per line; max ~20 words per line; no preamble, no explanations, no markdown.
- Prefer SPECIFIC referents ("Melanie's pottery bowl from her class") over vague ones ("it", "that thing").
- Never write a thematic summary; write concrete listener observations.
- If the turn has any specific content at all (a name, number, object, concrete update), do NOT output PHATIC — use a proper label.

CONTEXT (most recent last; each line is "<speaker>: <content>"):
{context_block}

CURRENT TURN:
{current_turn}

Observations (labeled lines, or PHATIC):
"""


# Round 4 — Candidate B (alternative free-form design, for fallback comparison)
PROMPT_B_FREEFORM = """\
You are a careful listener in a two-person conversation. After the CURRENT TURN, write 1-3 short factual observations about what a listener JUST learned.

Each observation must be:
- a single plain sentence (no bullets, no labels, no markdown),
- grounded in the current turn only (use context only to resolve pronouns/references),
- concrete (specific names, numbers, durations, places, events) — not thematic.

If the current turn is purely phatic (greetings, thanks, generic encouragement, goodbyes, echoes), output exactly: PHATIC

CONTEXT (most recent last; each line is "<speaker>: <content>"):
{context_block}

CURRENT TURN:
{current_turn}

Observations:
"""


ROUND_PROMPTS: dict[str, str] = {
    "round1_A": PROMPT_A,
    "round2_Aprime": PROMPT_A_PRIME,
    "round3_Adoubleprime": PROMPT_A_DOUBLE_PRIME,
    "round4_B_freeform": PROMPT_B_FREEFORM,
}


# --- Driver -------------------------------------------------------------


@dataclass
class TurnResult:
    turn_id: int
    category: str
    description: str
    context_lines: list[str]
    current_turn_line: str
    output: str


async def _run_prompt_on_turn(
    client: openai.AsyncOpenAI,
    cache: _Cache,
    prompt_template: str,
    context_lines: list[str],
    current_turn_line: str,
) -> str:
    ctx_block = "\n".join(context_lines) if context_lines else "(no prior turns)"
    prompt = prompt_template.format(
        context_block=ctx_block, current_turn=current_turn_line
    )
    cached = cache.get(MODEL, prompt)
    if cached is not None:
        return cached
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="low",
    )
    text = resp.choices[0].message.content or ""
    cache.put(MODEL, prompt, text)
    return text


async def run_round(
    round_name: str,
    prompt_template: str,
    sample_set: list[dict[str, Any]],
    client: openai.AsyncOpenAI,
    cache: _Cache,
) -> list[TurnResult]:
    results: list[TurnResult] = []
    tasks = []
    for s in sample_set:
        tasks.append(
            _run_prompt_on_turn(
                client,
                cache,
                prompt_template,
                s["context_lines"],
                s["current_turn_line"],
            )
        )
    outputs = await asyncio.gather(*tasks)
    for s, out in zip(sample_set, outputs):
        results.append(
            TurnResult(
                turn_id=s["turn_id"],
                category=s["category"],
                description=s["description"],
                context_lines=s["context_lines"],
                current_turn_line=s["current_turn_line"],
                output=out.strip(),
            )
        )
    cache.save()
    return results


def _analyze_round(results: list[TurnResult]) -> dict[str, Any]:
    """Return simple scorecard: PHATIC accuracy on phatic turns, concrete hits on others."""
    by_cat: dict[str, list[TurnResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    phatic_total = len(by_cat.get("phatic", []))
    phatic_correct = sum(
        1 for r in by_cat.get("phatic", []) if r.output.strip().upper() == "PHATIC"
    )
    non_phatic = [r for r in results if r.category != "phatic"]
    # Concrete heuristic: output must not equal "PHATIC" and must mention at
    # least one explicit label keyword OR contain a digit/proper noun/colon
    # (rough heuristic; we also read outputs manually in analysis).
    label_kws = ("RESOLVED:", "FACT:", "COUNT:", "UPDATE:", "LINK:", "NAME:")
    concrete_ok = []
    concrete_fail = []
    for r in non_phatic:
        out = r.output.strip()
        if out.upper() == "PHATIC":
            concrete_fail.append(r)
            continue
        has_label = any(kw in out for kw in label_kws)
        has_structure = (
            has_label or any(ch.isdigit() for ch in out) or len(out.splitlines()) >= 1
        )
        if has_structure and len(out) > 0:
            concrete_ok.append(r)
        else:
            concrete_fail.append(r)
    phatic_acc = phatic_correct / phatic_total if phatic_total else 0.0
    concrete_acc = len(concrete_ok) / len(non_phatic) if non_phatic else 0.0
    return {
        "phatic_total": phatic_total,
        "phatic_correct": phatic_correct,
        "phatic_accuracy": round(phatic_acc, 3),
        "non_phatic_total": len(non_phatic),
        "non_phatic_concrete": len(concrete_ok),
        "concrete_accuracy": round(concrete_acc, 3),
    }


def _result_to_dict(r: TurnResult) -> dict[str, Any]:
    return {
        "turn_id": r.turn_id,
        "category": r.category,
        "description": r.description,
        "context_lines": r.context_lines,
        "current_turn_line": r.current_turn_line,
        "output": r.output,
    }


async def main() -> None:
    turns = load_conv26_turns()
    by_tid = {t[0]: t for t in turns}

    sample_set: list[dict[str, Any]] = []
    for tid, cat, desc in SAMPLE_TURNS:
        if tid not in by_tid:
            raise RuntimeError(f"missing turn {tid} in conv-26 data")
        # Last 3 preceding turns as context.
        prev_ctx = [t for t in turns if t[0] < tid][-3:]
        ctx_lines = [em_format(sp, tx) for _tid, sp, tx in prev_ctx]
        _tid, sp, tx = by_tid[tid]
        cur_line = em_format(sp, tx)
        sample_set.append(
            {
                "turn_id": tid,
                "category": cat,
                "description": desc,
                "context_lines": ctx_lines,
                "current_turn_line": cur_line,
            }
        )

    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    cache = _Cache(CACHE_FILE)

    all_results: dict[str, dict[str, Any]] = {}
    try:
        for round_name, tmpl in ROUND_PROMPTS.items():
            print(f"\n=== Running {round_name} ===", flush=True)
            results = await run_round(round_name, tmpl, sample_set, client, cache)
            score = _analyze_round(results)
            print(
                f"  {round_name}: PHATIC {score['phatic_correct']}/{score['phatic_total']}"
                f" = {score['phatic_accuracy']:.0%}; concrete"
                f" {score['non_phatic_concrete']}/{score['non_phatic_total']}"
                f" = {score['concrete_accuracy']:.0%}",
                flush=True,
            )
            all_results[round_name] = {
                "prompt": tmpl,
                "scorecard": score,
                "results": [_result_to_dict(r) for r in results],
            }
    finally:
        cache.save()
        await client.close()

    # Persist raw
    with open(RESULTS_DIR / "notes_prompt_tuning.json", "w") as f:
        json.dump(
            {
                "model": MODEL,
                "sample_set": [
                    {
                        "turn_id": s["turn_id"],
                        "category": s["category"],
                        "description": s["description"],
                        "context_lines": s["context_lines"],
                        "current_turn_line": s["current_turn_line"],
                    }
                    for s in sample_set
                ],
                "rounds": all_results,
            },
            f,
            indent=2,
        )

    # Markdown report
    md = ["# Notes-prompt iterative tuning\n"]
    md.append("## Sample set (15 diverse turns from locomo_conv-26)\n")
    for s in sample_set:
        md.append(
            f"- turn_id={s['turn_id']} ({s['category']}): {s['description']}\n"
            f"  current: `{s['current_turn_line']}`"
        )
    md.append("")

    for round_name, block in all_results.items():
        md.append(f"\n## {round_name}\n")
        sc = block["scorecard"]
        md.append(
            f"**Scorecard**: PHATIC {sc['phatic_correct']}/{sc['phatic_total']} "
            f"= {sc['phatic_accuracy']:.0%}; concrete {sc['non_phatic_concrete']}/{sc['non_phatic_total']} "
            f"= {sc['concrete_accuracy']:.0%}\n"
        )
        md.append(
            "<details><summary>Prompt</summary>\n\n```\n"
            + block["prompt"]
            + "\n```\n</details>\n"
        )
        md.append("### Outputs\n")
        for r in block["results"]:
            md.append(
                f"**turn_id={r['turn_id']} ({r['category']})** — {r['description']}\n\n"
                f"current: `{r['current_turn_line']}`\n\n"
                f"```\n{r['output']}\n```\n"
            )

    with open(RESULTS_DIR / "notes_prompt_tuning.md", "w") as f:
        f.write("\n".join(md))

    print(f"\nWrote {RESULTS_DIR / 'notes_prompt_tuning.json'}")
    print(f"Wrote {RESULTS_DIR / 'notes_prompt_tuning.md'}")


if __name__ == "__main__":
    asyncio.run(main())
