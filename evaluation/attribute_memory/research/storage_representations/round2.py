"""Round 2: test 3 finalists on readback + drift + proxy retrieval.

Finalists (chosen from Round 1 qualitative inspection):
 - C1_flat_triples: baseline (the current design)
 - C2_typed_hierarchy: strongest structure (declared kinds, confidence, corrections)
 - C3_dossier_markdown: best readable + confidence-aware + set-aware

For each finalist x all 12 scenarios we:
  1. Generate the stored representation (author-time call, reuses Round 1 cache
     for 6 overlapping scenarios).
  2. Readback: a fresh LLM call gets ONLY the representation (no source) and
     must emit a canonical fact list. We compare that to the hand-labeled
     truth.
  3. Drift: a paraphrase of the original source is written manually and
     fed through the author prompt; we check whether the resulting
     representation matches the original one on key fields.
  4. Proxy retrieval: given a "new fact" (e.g. "user got a hamster"), an
     LLM call is asked, given ONLY the stored representation, which prior
     fact(s) would be relevant. We check whether it surfaces the right
     existing facts (e.g. pets set).

Then Round-2 judge: per scenario, one judge call compares the three
readback outputs against the scenario truth. Judges which rep captured
nuance/confidence/set-cardinality/negations best.

Budget: new author (3 finalists * 6 new scenarios) = 18 calls,
        readback (3 * 12 = 36),
        drift-author (3 * ~6 drift scenarios = 18),
        proxy-retrieval (3 * ~6 = 18),
        judge (~12).
Total new ~= 100. Running total after Round 1: ~136.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
load_dotenv(ROOT / "evaluation" / ".env")

RESULTS_DIR = HERE / "results"
CACHE_DIR = HERE / "cache"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-5-mini"
# Separate cache file so we can see what round 2 cost.
CACHE_FILE_R2 = CACHE_DIR / "round2_cache.json"
# Re-use round 1 author cache (same prompts, hits for overlapping scenarios).
CACHE_FILE_R1 = CACHE_DIR / "round1_cache.json"

FINALISTS = ["C1_flat_triples", "C2_typed_hierarchy", "C3_dossier_markdown"]


# Scenarios on which to run the DRIFT test (paraphrase same fact).
DRIFT_SCENARIOS = [
    "simple_first_person",
    "set_valued_pets",
    "negation",
    "hedged_nuanced",
    "nuanced_preference",
]

# Scenarios on which to run the PROXY RETRIEVAL test.
RETRIEVAL_SCENARIOS = [
    "set_valued_pets",  # new pet addition
    "set_valued_allergies",  # new allergy
    "simple_first_person",  # "I'm moving again"
    "nuanced_preference",  # new jazz sub-genre liked
    "correction_retraction",  # already-corrected field updated again
]


# --- Cache --------------------------------------------------------------


def _sha(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()


class _LayeredCache:
    """Read-through from a set of files, write to one file."""

    def __init__(self, write_path: Path, read_paths: list[Path]) -> None:
        self._write_path = write_path
        self._d: dict[str, str] = {}
        for p in read_paths + [write_path]:
            if p.exists():
                try:
                    with open(p) as f:
                        self._d.update(json.load(f))
                except Exception:
                    pass
        self._new: dict[str, str] = {}

    def get(self, model: str, prompt: str) -> str | None:
        return self._d.get(_sha(model, prompt))

    def put(self, model: str, prompt: str, response: str) -> None:
        k = _sha(model, prompt)
        self._d[k] = response
        self._new[k] = response

    def save(self) -> None:
        if not self._new:
            return
        existing = {}
        if self._write_path.exists():
            try:
                with open(self._write_path) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing.update(self._new)
        tmp = self._write_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self._write_path)
        self._new = {}


# --- Author prompts (from Round 1) --------------------------------------

from round1 import CANDIDATES  # noqa: E402

# --- Readback prompt ----------------------------------------------------

READBACK_PROMPT = """\
You are given ONLY a stored memory representation about a subject. You cannot see the original source text.

Your task: emit a canonical flat fact list that captures EVERYTHING the representation asserts, including:
- Confidence qualifiers (stated, approximate, suspected, corrected, negated, third-party).
- Set cardinality (how many members; what the count was before a change).
- Negations (what the subject explicitly does NOT do/have).
- Corrections (what was retracted and what replaced it).

Output exactly one fact per line in this format:
- <attribute>: <value> [<confidence>]

If the fact is set-valued, write:
- <attribute>: item1, item2, item3 (cardinality=n, was=m if changed) [<confidence>]

If the fact is a correction, write:
- <attribute>: <final value>; retracted <prior value> [corrected]

If the fact is a negation, write:
- <attribute>: NOT <value> [negated]

No preamble, no prose, no markdown headings. Just fact lines.

Stored representation:
\"\"\"
{representation}
\"\"\"

Facts:
"""


# --- Proxy retrieval prompt ---------------------------------------------

RETRIEVAL_PROMPT = """\
You are a memory assistant. A user just said the following NEW statement:

NEW STATEMENT: {new_statement}

Here is the currently stored memory about the user (or subject):

STORED MEMORY:
\"\"\"
{representation}
\"\"\"

Task: identify any EXISTING facts in the stored memory that are relevant to the new statement — facts that would need to be read, updated, added-to, or contradicted based on the new statement.

Output one line per relevant stored fact in this format:
RELEVANT: <verbatim fact from stored memory> | <why relevant to new statement>

If no stored fact is relevant, output exactly: NONE

Do not invent facts not present in STORED MEMORY. Do not act on the new statement — only identify which prior facts are relevant.
"""


# --- Judge prompt -------------------------------------------------------

JUDGE_PROMPT = """\
You are judging how well three memory representations preserved the meaning of the source text, as evidenced by how well a reader can recover the facts from the representation alone.

SCENARIO: {scenario_id}
SOURCE TEXT: {source}

GROUND-TRUTH FACTS (hand-labeled, considered correct):
{truth}

READBACK FROM REPRESENTATION A ({a_name}) — reader had ONLY the stored rep, not the source:
{readback_a}

READBACK FROM REPRESENTATION B ({b_name}):
{readback_b}

READBACK FROM REPRESENTATION C ({c_name}):
{readback_c}

For each representation, score on the four properties below on a 1-5 scale (5 = perfect), then give a 1-sentence justification per score. Be honest about negative results — it is OK for representations to be similar or for the baseline to win.

Properties:
1. FIDELITY — readback captures all factual content in the truth.
2. CONFIDENCE — readback preserves hedges/confidence (stated vs approximate vs suspected vs corrected vs negated).
3. CARDINALITY — readback preserves set cardinality and any count changes.
4. NEGATION — readback preserves explicit negations and non-negations.

Output JSON only, in this exact shape (no preamble, no markdown fences):
{{
  "A_{a_name}": {{"fidelity": <int>, "confidence": <int>, "cardinality": <int>, "negation": <int>, "note": "<1 sentence>"}},
  "B_{b_name}": {{"fidelity": <int>, "confidence": <int>, "cardinality": <int>, "negation": <int>, "note": "<1 sentence>"}},
  "C_{c_name}": {{"fidelity": <int>, "confidence": <int>, "cardinality": <int>, "negation": <int>, "note": "<1 sentence>"}},
  "overall_winner": "A" | "B" | "C" | "tie",
  "overall_justification": "<1-2 sentences>"
}}
"""


# --- Drift scenarios: manual paraphrases --------------------------------

DRIFT_PARAPHRASES: dict[str, str] = {
    # Same facts, different wording.
    "simple_first_person": (
        "Jamie and I have been in Portland for about three years now — we came "
        "up from Austin together."
    ),
    "set_valued_pets": (
        "Right now we have five pets — the two cats (Luna, Milo), Rex the "
        "pitbull, and my daughter's hamster. We used to have six before our "
        "fish died last month."
    ),
    "negation": (
        "Coffee's not my thing — never touched it. I'm strictly tea. That "
        "said I do consume caffeine elsewhere; it's specifically coffee I "
        "dislike."
    ),
    "hedged_nuanced": (
        "Wondering if dairy is an issue for me — cheese can bother me at "
        "times but yogurt seems OK. Thinking about easing off it."
    ),
    "nuanced_preference": (
        "Big jazz fan, especially bebop and hard bop — free jazz I can "
        "handle a little but wouldn't put on at home."
    ),
}


# --- Proxy retrieval: new statements per scenario -----------------------

RETRIEVAL_NEW_STATEMENTS: dict[str, str] = {
    "set_valued_pets": "I just got a hamster for my son's birthday.",
    "set_valued_allergies": "Found out I'm also allergic to tree nuts now — not as bad as shellfish though.",
    "simple_first_person": "We're thinking of moving to Seattle next summer.",
    "nuanced_preference": "Been getting into cool jazz lately, especially Chet Baker.",
    "correction_retraction": "Actually I took a new role at OpenAI last week.",
}


# --- Helpers ------------------------------------------------------------


async def _call(
    client: openai.AsyncOpenAI,
    cache: _LayeredCache,
    prompt: str,
) -> str:
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


def _truth_str(truth: dict) -> str:
    """Pretty flat dump of the scenarios.json truth block for the judge."""
    lines: list[str] = []
    for f in truth.get("facts", []):
        vstr = f["value"]
        if isinstance(vstr, list):
            vstr = ", ".join(vstr)
        conf = f.get("confidence", "?")
        card = f" (cardinality={f['cardinality']})" if f.get("cardinality") else ""
        lines.append(
            f"  - subject={f.get('subject', '?')} | attribute={f['attribute']} | "
            f"value={vstr}{card} | confidence={conf}"
        )
    if truth.get("relationships"):
        for r in truth["relationships"]:
            lines.append(
                f"  - REL: {r.get('source', '?')} -[{r.get('type', '?')}]-> "
                f"{r.get('target', '?')}"
            )
    if truth.get("retracted"):
        for r in truth["retracted"]:
            lines.append(
                f"  - RETRACTED: attribute={r['attribute']} "
                f"previously_stated={r['previously_stated']}"
            )
    return "\n".join(lines)


# --- Driver -------------------------------------------------------------


async def main() -> None:
    with open(HERE / "scenarios.json") as f:
        bundle = json.load(f)
    scenarios = {s["id"]: s for s in bundle["scenarios"]}
    all_sids = [s["id"] for s in bundle["scenarios"]]

    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    cache = _LayeredCache(write_path=CACHE_FILE_R2, read_paths=[CACHE_FILE_R1])

    # ---- Step A: Author-time representations on all 12 scenarios ----
    author_tasks: list[tuple[str, str, Any]] = []
    for cname in FINALISTS:
        prompt_tmpl = CANDIDATES[cname]["prompt"]
        for sid in all_sids:
            p = prompt_tmpl.format(source=scenarios[sid]["source"])
            author_tasks.append((cname, sid, _call(client, cache, p)))
    author_outs = await asyncio.gather(*[t[2] for t in author_tasks])
    author: dict[tuple[str, str], str] = {}
    for (cname, sid, _), out in zip(author_tasks, author_outs):
        author[(cname, sid)] = out.strip()
    cache.save()
    print(f"[author] {len(author_tasks)} representations generated")

    # ---- Step B: Readback on all (finalist, scenario) ----
    rb_tasks: list[tuple[str, str, Any]] = []
    for cname in FINALISTS:
        for sid in all_sids:
            p = READBACK_PROMPT.format(representation=author[(cname, sid)])
            rb_tasks.append((cname, sid, _call(client, cache, p)))
    rb_outs = await asyncio.gather(*[t[2] for t in rb_tasks])
    readback: dict[tuple[str, str], str] = {}
    for (cname, sid, _), out in zip(rb_tasks, rb_outs):
        readback[(cname, sid)] = out.strip()
    cache.save()
    print(f"[readback] {len(rb_tasks)} readbacks generated")

    # ---- Step C: Drift-author on paraphrased sources ----
    drift_tasks: list[tuple[str, str, Any]] = []
    for cname in FINALISTS:
        prompt_tmpl = CANDIDATES[cname]["prompt"]
        for sid in DRIFT_SCENARIOS:
            p = prompt_tmpl.format(source=DRIFT_PARAPHRASES[sid])
            drift_tasks.append((cname, sid, _call(client, cache, p)))
    drift_outs = await asyncio.gather(*[t[2] for t in drift_tasks])
    drift_rep: dict[tuple[str, str], str] = {}
    for (cname, sid, _), out in zip(drift_tasks, drift_outs):
        drift_rep[(cname, sid)] = out.strip()
    cache.save()
    print(f"[drift-author] {len(drift_tasks)} paraphrase reps generated")

    # ---- Step D: Drift-readback (so we compare readbacks apples-to-apples) ----
    drift_rb_tasks: list[tuple[str, str, Any]] = []
    for cname in FINALISTS:
        for sid in DRIFT_SCENARIOS:
            p = READBACK_PROMPT.format(representation=drift_rep[(cname, sid)])
            drift_rb_tasks.append((cname, sid, _call(client, cache, p)))
    drift_rb_outs = await asyncio.gather(*[t[2] for t in drift_rb_tasks])
    drift_rb: dict[tuple[str, str], str] = {}
    for (cname, sid, _), out in zip(drift_rb_tasks, drift_rb_outs):
        drift_rb[(cname, sid)] = out.strip()
    cache.save()
    print(f"[drift-readback] {len(drift_rb_tasks)} drift readbacks generated")

    # ---- Step E: Proxy retrieval ----
    ret_tasks: list[tuple[str, str, Any]] = []
    for cname in FINALISTS:
        for sid in RETRIEVAL_SCENARIOS:
            p = RETRIEVAL_PROMPT.format(
                new_statement=RETRIEVAL_NEW_STATEMENTS[sid],
                representation=author[(cname, sid)],
            )
            ret_tasks.append((cname, sid, _call(client, cache, p)))
    ret_outs = await asyncio.gather(*[t[2] for t in ret_tasks])
    retrieval: dict[tuple[str, str], str] = {}
    for (cname, sid, _), out in zip(ret_tasks, ret_outs):
        retrieval[(cname, sid)] = out.strip()
    cache.save()
    print(f"[retrieval] {len(ret_tasks)} relevance checks generated")

    # ---- Step F: Judge comparisons (one per scenario) ----
    judge_tasks: list[tuple[str, Any]] = []
    for sid in all_sids:
        scen = scenarios[sid]
        a_name, b_name, c_name = FINALISTS
        p = JUDGE_PROMPT.format(
            scenario_id=sid,
            source=scen["source"],
            truth=_truth_str(scen["truth"]),
            a_name=a_name,
            b_name=b_name,
            c_name=c_name,
            readback_a=readback[(a_name, sid)],
            readback_b=readback[(b_name, sid)],
            readback_c=readback[(c_name, sid)],
        )
        judge_tasks.append((sid, _call(client, cache, p)))
    judge_outs = await asyncio.gather(*[t[1] for t in judge_tasks])
    judges: dict[str, str] = {}
    for (sid, _), out in zip(judge_tasks, judge_outs):
        judges[sid] = out.strip()
    cache.save()
    print(f"[judge] {len(judge_tasks)} judgments generated")

    await client.close()

    # --- Persist ---
    payload: dict[str, Any] = {
        "model": MODEL,
        "finalists": FINALISTS,
        "author_representations": {
            cname: {sid: author[(cname, sid)] for sid in all_sids}
            for cname in FINALISTS
        },
        "readbacks": {
            cname: {sid: readback[(cname, sid)] for sid in all_sids}
            for cname in FINALISTS
        },
        "drift_paraphrases": DRIFT_PARAPHRASES,
        "drift_representations": {
            cname: {sid: drift_rep[(cname, sid)] for sid in DRIFT_SCENARIOS}
            for cname in FINALISTS
        },
        "drift_readbacks": {
            cname: {sid: drift_rb[(cname, sid)] for sid in DRIFT_SCENARIOS}
            for cname in FINALISTS
        },
        "retrieval_new_statements": RETRIEVAL_NEW_STATEMENTS,
        "retrieval_outputs": {
            cname: {sid: retrieval[(cname, sid)] for sid in RETRIEVAL_SCENARIOS}
            for cname in FINALISTS
        },
        "judges": judges,
    }
    with open(RESULTS_DIR / "round2_outputs.json", "w") as f:
        json.dump(payload, f, indent=2)

    # --- Compute per-finalist judge totals ---
    totals: dict[str, dict[str, float]] = {
        c: {
            "fidelity": 0.0,
            "confidence": 0.0,
            "cardinality": 0.0,
            "negation": 0.0,
            "count": 0,
        }
        for c in FINALISTS
    }
    winner_tally: dict[str, int] = dict.fromkeys(FINALISTS, 0)
    winner_tally["tie"] = 0
    parse_fail = 0
    for sid, j in judges.items():
        try:
            s = j.strip()
            if s.startswith("```"):
                # strip fences
                s = s.strip("`")
                nl = s.find("\n")
                if nl != -1:
                    s = s[nl:].strip()
            d = json.loads(s)
        except Exception:
            parse_fail += 1
            continue
        a_name, b_name, c_name = FINALISTS
        keymap = {
            a_name: f"A_{a_name}",
            b_name: f"B_{b_name}",
            c_name: f"C_{c_name}",
        }
        for cname, k in keymap.items():
            block = d.get(k) or d.get(k.split("_", 1)[-1]) or {}
            for prop in ("fidelity", "confidence", "cardinality", "negation"):
                v = block.get(prop)
                if isinstance(v, (int, float)):
                    totals[cname][prop] += float(v)
            totals[cname]["count"] += 1
        w = d.get("overall_winner", "tie")
        letter_to_name = {"A": a_name, "B": b_name, "C": c_name, "tie": "tie"}
        winner_tally[letter_to_name.get(w, "tie")] = (
            winner_tally.get(letter_to_name.get(w, "tie"), 0) + 1
        )

    avg: dict[str, dict[str, float]] = {}
    for cname, t in totals.items():
        n = t["count"] or 1
        avg[cname] = {
            k: round(t[k] / n, 3)
            for k in ("fidelity", "confidence", "cardinality", "negation")
        }
        avg[cname]["count"] = int(t["count"])

    print("\n=== Judge averages (1-5 each property) ===")
    for c in FINALISTS:
        a = avg[c]
        print(
            f"  {c}: fidelity={a['fidelity']}  confidence={a['confidence']}  "
            f"cardinality={a['cardinality']}  negation={a['negation']}  "
            f"(n={a['count']})"
        )
    print(f"\n=== Winner tally across {len(judges)} scenarios ===")
    for c, n in winner_tally.items():
        print(f"  {c}: {n}")
    print(f"Parse failures: {parse_fail}")

    payload["judge_averages"] = avg
    payload["judge_winner_tally"] = winner_tally
    payload["judge_parse_fail"] = parse_fail
    with open(RESULTS_DIR / "round2_outputs.json", "w") as f:
        json.dump(payload, f, indent=2)

    # Markdown browse-friendly dump
    md: list[str] = ["# Round 2 — readback, drift, retrieval, judge\n"]
    md.append(f"Finalists: {', '.join(FINALISTS)}.\n")
    md.append(f"Model: {MODEL}.\n")
    md.append("\n## Judge averages (1-5)\n")
    md.append(
        "| Representation | Fidelity | Confidence | Cardinality | Negation | n |\n"
    )
    md.append("|---|---|---|---|---|---|\n")
    for c in FINALISTS:
        a = avg[c]
        md.append(
            f"| {c} | {a['fidelity']} | {a['confidence']} | {a['cardinality']} | {a['negation']} | {a['count']} |\n"
        )
    md.append(f"\nWinner tally: `{winner_tally}`, judge parse failures: {parse_fail}\n")

    md.append("\n## Per-scenario readbacks, drift, retrieval, judge\n")
    for sid in all_sids:
        scen = scenarios[sid]
        md.append(f"\n### {sid}\n")
        md.append(f"**Source:** `{scen['source']}`\n")
        md.append(f"**Truth:**\n```\n{_truth_str(scen['truth'])}\n```\n")
        for c in FINALISTS:
            md.append(f"\n#### {c}\n")
            md.append("**Author-time rep:**\n```\n" + author[(c, sid)] + "\n```\n")
            md.append("**Readback:**\n```\n" + readback[(c, sid)] + "\n```\n")
            if sid in DRIFT_SCENARIOS:
                md.append("**Drift paraphrase source:**\n")
                md.append(f"> {DRIFT_PARAPHRASES[sid]}\n")
                md.append("**Drift rep:**\n```\n" + drift_rep[(c, sid)] + "\n```\n")
                md.append("**Drift readback:**\n```\n" + drift_rb[(c, sid)] + "\n```\n")
            if sid in RETRIEVAL_SCENARIOS:
                md.append("**Retrieval new statement:**\n")
                md.append(f"> {RETRIEVAL_NEW_STATEMENTS[sid]}\n")
                md.append(
                    "**Retrieval output:**\n```\n" + retrieval[(c, sid)] + "\n```\n"
                )
        md.append("\n**Judge verdict:**\n```\n" + judges[sid] + "\n```\n")

    with open(RESULTS_DIR / "round2_outputs.md", "w") as f:
        f.write("\n".join(md))

    print(f"\nWrote {RESULTS_DIR / 'round2_outputs.json'}")
    print(f"Wrote {RESULTS_DIR / 'round2_outputs.md'}")


if __name__ == "__main__":
    asyncio.run(main())
