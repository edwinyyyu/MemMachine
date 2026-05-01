"""Round 1: shortlist 6 representation x framing combinations.

For each candidate, generate stored representations on a 6-scenario subset
covering the main axes (simple, hedged, set-valued, correction, negation,
novel chunk). Inspect outputs qualitatively (dumped to results/).

Budget: 6 candidates x 6 scenarios = 36 LLM calls at gpt-5-mini.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # semantic_memory/
load_dotenv(ROOT / "evaluation" / ".env")

RESULTS_DIR = HERE / "results"
CACHE_DIR = HERE / "cache"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-5-mini"
CACHE_FILE = CACHE_DIR / "round1_cache.json"

# Round-1 scenario subset covering the axes cheaply.
ROUND1_SCENARIO_IDS = [
    "simple_first_person",
    "hedged_nuanced",
    "set_valued_pets",
    "correction_retraction",
    "negation",
    "novel_chunk",
]


# --- Cache (identical pattern to notes_prompt_tuning.py) ----------------


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


# --- Candidate representations + framings -------------------------------
# Each candidate is (name, framing-description, prompt-template-with-{source}).

# --- C1: Flat triples (baseline, "fact-extraction table" framing) ---
C1_FLAT_TRIPLES = """\
You are building a fact-extraction table about the subject of the text below.

Output ONE fact per line in the format:
(topic | category | attribute | value)

Rules:
- topic: the thing the fact is about (e.g. the person, a place, a pet they own).
- category: a broad grouping (e.g. "biography", "preferences", "medical", "relationships").
- attribute: the specific property (e.g. "city", "allergy", "employer").
- value: the observed value. You may include qualifiers like "former", "suspected", "approximately", or set sizes in the value if needed.
- One fact per line. No preamble, no prose, no markdown.
- Extract only what the text states; do NOT invent values.

Text:
\"\"\"
{source}
\"\"\"

Facts:
"""

# --- C2: Strict hierarchy with declared cardinality, "structured profile" framing ---
C2_TYPED_HIERARCHY = """\
You are writing a structured profile about the subject of the text below.

Output a JSON object keyed by topic. Each topic maps to an object keyed by category. Each category maps to an object keyed by attribute. Each attribute has this shape:
{{
  "kind": "functional" | "partial_functional" | "set" | "ordered_list",
  "value": <string, list-of-strings, or null>,
  "confidence": "stated" | "approximate" | "suspected" | "corrected" | "negated",
  "note": <optional string for nuance, hedging, source qualifier, or set cardinality>
}}

kind semantics:
- functional: at most one value ever (e.g. birth_year). "value" is a string.
- partial_functional: typically one current value but could change (e.g. current_employer). "value" is a string.
- set: unordered multi-valued attribute (e.g. allergies, pets). "value" is a JSON array.
- ordered_list: sequence where order matters (e.g. work_mode_history). "value" is a JSON array in order.

For set-valued attributes, also add "cardinality" (integer) when stated or inferable.
For negated attributes, set "confidence": "negated" and put the negated value in "value" as a string like "no" or the specific thing denied.
For corrected/retracted claims, set "confidence": "corrected" and put the final value; use "note" for the prior (retracted) value.

Rules:
- Only include information directly stated in the text.
- Use lower_snake_case for category and attribute keys.
- Output JSON only, no preamble, no markdown fences.

Text:
\"\"\"
{source}
\"\"\"

JSON:
"""

# --- C3: Natural-language dossier, "dossier section" framing ---
C3_DOSSIER = """\
You are drafting a dossier section on the subject of the text below. A dossier is a concise, factual briefing.

Write markdown with:
- A `## <Subject>` header.
- `### <Category>` subheaders for major groupings (e.g. Biography, Preferences, Medical, Relationships).
- Bulleted facts, one per line, in plain sentences.
- Inline confidence hedges where applicable: `[confirmed]`, `[stated]`, `[approximate]`, `[suspected]`, `[corrected]`, `[negated]`. Place the tag at the end of the bullet.
- For sets, write the bullet as: "<attribute>: item1, item2, item3 (n total) [stated]". Keep the count explicit.
- For corrections, write: "<attribute>: <current value>; previously claimed <prior value> [corrected]".
- For negations, write: "Does NOT <attribute>: <value> [negated]".

Rules:
- Only include information directly stated.
- No preamble, no prose outside the dossier, no speculation.

Text:
\"\"\"
{source}
\"\"\"

Dossier:
"""

# --- C4: Observation log, "journalistic profile / observation journal" framing ---
C4_OBSERVATION_LOG = """\
You are keeping an observation journal about the subject of the text below. Each entry is a single timestamp-anchored observation noting what was just learned and how certain it is.

Write one observation per line in this format:
[confidence] <what was observed, as a concrete self-contained sentence mentioning the subject by name>

Confidence tags (use exactly one per line):
- [stated]: directly asserted in the text.
- [approximate]: hedged with "about", "around", "roughly", "a few".
- [suspected]: hedged with "I think", "maybe", "probably", "possibly".
- [corrected]: replaces a prior claim within the same text.
- [negated]: explicitly denies something.
- [third-party]: attributed to someone else, not the subject.

For sets, emit ONE observation that lists all members with explicit count, e.g.:
[stated] User has 3 pets: Luna (cat), Milo (cat), Rex (pitbull dog).

For corrections, emit ONE [corrected] observation noting both the final and retracted value.

Rules:
- Each line is self-contained: a future reader should need no other context.
- Do not invent anything. Only include what the text asserts.
- No preamble, no markdown, no bullets — just confidence-tagged lines.

Text:
\"\"\"
{source}
\"\"\"

Observations:
"""

# --- C5: Entity-centric graph, "character sheet" framing ---
C5_ENTITY_GRAPH = """\
You are building a character sheet that treats every named entity (person, pet, place, object, organization) as a first-class node with its own properties and relationships.

Output a JSON object with this shape:
{{
  "entities": {{
    "<entity_id>": {{
      "type": "person" | "pet" | "place" | "org" | "object" | "event" | "condition" | "role",
      "canonical_name": "<best name or phrase for this entity>",
      "properties": {{
        "<prop>": {{"value": <string-or-list>, "confidence": "stated|approximate|suspected|corrected|negated", "note": "<optional>"}}
      }}
    }}
  }},
  "relationships": [
    {{"source": "<entity_id>", "target": "<entity_id>", "type": "<relation>", "confidence": "stated|...", "note": "<optional>"}}
  ],
  "set_memberships": [
    {{"owner": "<entity_id>", "attribute": "<name>", "members": ["<entity_id>", ...], "cardinality": <int or null>, "confidence": "stated|..."}}
  ]
}}

Rules:
- Make every mentioned person, pet, or place its own entity, even if only implied (e.g. "user" if the text is first-person).
- Use `set_memberships` for multi-valued attributes like "pets", "allergies", "concerns".
- For corrected values, include the final in `properties.value` with confidence "corrected", and put the retracted value in `note`.
- For negations, use confidence "negated" and put the denied value in `value`.
- Output JSON only, no preamble, no markdown.

Text:
\"\"\"
{source}
\"\"\"

JSON:
"""

# --- C6: Hybrid (JSON skeleton + prose nuance), "study notes" framing ---
C6_HYBRID = """\
You are writing study notes about the subject of the text below. Study notes combine a structured card at the top with free-text nuance below.

Output exactly two sections:

=== CARD ===
A JSON object with this minimal structure:
{{
  "subject": "<primary subject>",
  "facts": [
    {{"attribute": "<snake_case>", "value": <string-or-list>, "kind": "functional|partial_functional|set|ordered_list", "confidence": "stated|approximate|suspected|corrected|negated", "cardinality": <int or null>}}
  ],
  "other_entities": [
    {{"name": "<name>", "type": "<person|pet|place|...>", "relation_to_subject": "<short phrase>"}}
  ]
}}

=== NOTES ===
Free-text paragraph(s) capturing nuance the JSON can't — tone, hedging language, interpretive caveats, multi-sentence corrections. Keep under ~120 words. Reference specific facts by attribute name when helpful.

Rules:
- Only include information directly stated.
- Output both sections in order, starting with `=== CARD ===`. No preamble, no markdown fences around the JSON.

Text:
\"\"\"
{source}
\"\"\"
"""

CANDIDATES: dict[str, dict[str, str]] = {
    "C1_flat_triples": {
        "framing": "fact-extraction table",
        "prompt": C1_FLAT_TRIPLES,
    },
    "C2_typed_hierarchy": {
        "framing": "structured profile with cardinality types",
        "prompt": C2_TYPED_HIERARCHY,
    },
    "C3_dossier_markdown": {
        "framing": "dossier section",
        "prompt": C3_DOSSIER,
    },
    "C4_observation_log": {
        "framing": "observation journal",
        "prompt": C4_OBSERVATION_LOG,
    },
    "C5_entity_graph": {
        "framing": "character sheet",
        "prompt": C5_ENTITY_GRAPH,
    },
    "C6_hybrid_card_notes": {
        "framing": "study notes (card + prose)",
        "prompt": C6_HYBRID,
    },
}


# --- Driver -------------------------------------------------------------


@dataclass
class CandidateRun:
    candidate: str
    scenario_id: str
    source: str
    output: str


async def _run_one(
    client: openai.AsyncOpenAI,
    cache: _Cache,
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


async def main() -> None:
    with open(HERE / "scenarios.json") as f:
        bundle = json.load(f)
    scenarios = {s["id"]: s for s in bundle["scenarios"]}

    subset = [scenarios[sid] for sid in ROUND1_SCENARIO_IDS]

    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    cache = _Cache(CACHE_FILE)

    all_runs: list[CandidateRun] = []
    tasks: list[tuple[str, str, str, Any]] = []
    for cname, cdef in CANDIDATES.items():
        for scen in subset:
            prompt = cdef["prompt"].format(source=scen["source"])
            tasks.append(
                (cname, scen["id"], scen["source"], _run_one(client, cache, prompt))
            )

    # Execute concurrently
    outs = await asyncio.gather(*[t[3] for t in tasks])
    for (cname, sid, src, _), out in zip(tasks, outs):
        all_runs.append(
            CandidateRun(
                candidate=cname, scenario_id=sid, source=src, output=out.strip()
            )
        )

    cache.save()
    await client.close()

    # Persist
    payload = {
        "model": MODEL,
        "candidates": {
            cname: {"framing": cdef["framing"], "prompt": cdef["prompt"]}
            for cname, cdef in CANDIDATES.items()
        },
        "scenario_ids_used": ROUND1_SCENARIO_IDS,
        "runs": [
            {
                "candidate": r.candidate,
                "scenario_id": r.scenario_id,
                "source": r.source,
                "output": r.output,
            }
            for r in all_runs
        ],
    }
    out_path = RESULTS_DIR / "round1_outputs.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Markdown browse-friendly dump
    md_lines: list[str] = ["# Round 1 outputs\n"]
    md_lines.append(f"Model: {MODEL}. Scenarios: {', '.join(ROUND1_SCENARIO_IDS)}.\n")
    by_scen: dict[str, list[CandidateRun]] = {}
    for r in all_runs:
        by_scen.setdefault(r.scenario_id, []).append(r)
    for sid in ROUND1_SCENARIO_IDS:
        scen = scenarios[sid]
        md_lines.append(f"\n## Scenario: {sid}\n")
        md_lines.append(f"**Source:** `{scen['source']}`\n")
        for r in by_scen.get(sid, []):
            md_lines.append(f"\n### {r.candidate}\n")
            md_lines.append("```\n" + r.output + "\n```\n")
    with open(RESULTS_DIR / "round1_outputs.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"Wrote {out_path}")
    print(f"Wrote {RESULTS_DIR / 'round1_outputs.md'}")
    print(f"Total runs: {len(all_runs)}")


if __name__ == "__main__":
    asyncio.run(main())
