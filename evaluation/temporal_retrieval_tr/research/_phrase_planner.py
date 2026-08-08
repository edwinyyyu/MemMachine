"""Minimal LLM phrase planner: LLM emits intent + phrase polarity only.

Calendar resolution is done by Duckling code-side. The LLM only needs to:
  1. Identify which text spans are temporal phrases
  2. Tag each as "include" or "exclude" based on query polarity
  3. Detect extremum intent

Output schema is tiny -> short prompt -> low latency.

Goals:
  - LLM call latency ~500ms with gpt-5-nano @ minimal reasoning
  - Quality close to full LLM planner (within 1-3pp)
  - Far better than rule-only planner on stress benches
"""
from __future__ import annotations

import asyncio, hashlib, json, os, re, time
from datetime import datetime, timedelta
from pathlib import Path

from openai import AsyncOpenAI
from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)
from openai.types.responses import ResponseTextConfigParam

from temporal_retrieval_min.schema import parse_iso, to_us
from temporal_retrieval_tr.planner import Plan
from temporal_retrieval_tr.time_range import (
    NEG_INF, POS_INF, Interval, IntervalSet,
)

from ._duckling_extractor import DucklingHTTPExtractor


# Cache dir for LLM phrase responses.
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "phrase_planner"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


_PROMPT = """Identify temporal phrases in the query and emit a minimal plan.

For each query, emit:
  - include_phrases: text spans naming periods the query asks ABOUT (positive scope)
  - exclude_phrases: text spans naming periods to EXCLUDE (negation, "not", "except")
  - extremum: "latest" if asking for most recent; "earliest" if asking for first;
              null otherwise

When negation has a bounded scope ("in 2024 except Y"), include the bounded scope
in include_phrases AND the excluded period in exclude_phrases.

For "Q2 or Q4" in a year context, qualify each (e.g., "Q2 2023", "Q4 2023").

Examples:
  Q: "What's my latest project Alpha update from Q4 2023?"
    -> {{"include_phrases":["Q4 2023"],"exclude_phrases":[],"extremum":"latest"}}
  Q: "Meetings not in 2023"
    -> {{"include_phrases":[],"exclude_phrases":["2023"],"extremum":null}}
  Q: "What were dental appointments in Q3 2023 and Q1 2024?"
    -> {{"include_phrases":["Q3 2023","Q1 2024"],"exclude_phrases":[],"extremum":null}}
  Q: "What did I do outside summer 2024?"
    -> {{"include_phrases":[],"exclude_phrases":["summer 2024"],"extremum":null}}
  Q: "Marathons in 2023 not in Q2 or Q4"
    -> {{"include_phrases":["2023"],"exclude_phrases":["Q2 2023","Q4 2023"],"extremum":null}}
  Q: "What year did I meet my wife?"
    -> {{"include_phrases":[],"exclude_phrases":[],"extremum":null}}

Reference time: {ref_time}
Query: {query}"""


_PHRASE_SCHEMA = {
    "type": "object",
    "properties": {
        "include_phrases": {
            "type": "array",
            "items": {"type": "string"},
        },
        "exclude_phrases": {
            "type": "array",
            "items": {"type": "string"},
        },
        "extremum": {
            "type": ["string", "null"],
            "enum": ["latest", "earliest", None],
        },
    },
    "required": ["include_phrases", "exclude_phrases", "extremum"],
    "additionalProperties": False,
}


_QUARTER_RE = re.compile(r"\b[Qq]([1-4])\s+((?:19|20)\d{2})\b")


def _quarter_interval(q: int, year: int) -> Interval:
    from datetime import UTC
    start_month = 1 + 3 * (q - 1)
    end_month = start_month + 3
    end_year = year + (end_month - 1) // 12
    end_month = ((end_month - 1) % 12) + 1
    start = datetime(year, start_month, 1, tzinfo=UTC)
    end = datetime(end_year, end_month, 1, tzinfo=UTC)
    return Interval(to_us(start), to_us(end))


def _complement(intervals: list[Interval]) -> list[Interval]:
    if not intervals:
        return [Interval(NEG_INF, POS_INF)]
    sorted_ivs = sorted(intervals, key=lambda iv: iv.earliest_us)
    merged: list[Interval] = [sorted_ivs[0]]
    for iv in sorted_ivs[1:]:
        prev = merged[-1]
        if iv.earliest_us <= prev.latest_us:
            new_end = iv.latest_us if iv.latest_us > prev.latest_us else prev.latest_us
            merged[-1] = Interval(prev.earliest_us, new_end)
        else:
            merged.append(iv)
    result: list[Interval] = []
    prev_end = NEG_INF
    for iv in merged:
        if prev_end < iv.earliest_us:
            result.append(Interval(prev_end, iv.earliest_us))
        prev_end = iv.latest_us
    if prev_end < POS_INF:
        result.append(Interval(prev_end, POS_INF))
    return result


def _intersect(a: list[Interval], b: list[Interval]) -> list[Interval]:
    out = []
    for ia in a:
        for ib in b:
            lo = ia.earliest_us if ia.earliest_us > ib.earliest_us else ib.earliest_us
            hi = ia.latest_us if ia.latest_us < ib.latest_us else ib.latest_us
            if lo < hi:
                out.append(Interval(lo, hi))
    return out


class PhrasePlanner:
    """LLM emits phrase polarity + intent; Duckling resolves dates."""

    def __init__(
        self,
        model: str = "gpt-5-nano",
        reasoning_effort: str = "minimal",
    ) -> None:
        self._model = model
        self._reasoning = reasoning_effort
        self._client = AsyncOpenAI()
        self._extractor = DucklingHTTPExtractor()
        self._cache_file = CACHE_DIR / f"{model}_{reasoning_effort}_cache.json"
        self._cache: dict[str, dict] = {}
        if self._cache_file.exists():
            try:
                self._cache = json.loads(self._cache_file.read_text())
            except Exception:
                pass
        self._dirty = False

    def save_caches(self) -> None:
        if self._dirty:
            self._cache_file.write_text(json.dumps(self._cache))
            self._dirty = False
        self._extractor.save_caches()

    def _key(self, query: str, ref_time: str) -> str:
        return hashlib.sha256(f"{query}|{ref_time}".encode()).hexdigest()

    async def _llm_call(
        self, query: str, ref_time: str
    ) -> tuple[dict, float]:
        prompt = _PROMPT.format(query=query, ref_time=ref_time)
        format_cfg: ResponseFormatTextJSONSchemaConfigParam = {
            "type": "json_schema",
            "name": "phrase_plan",
            "strict": True,
            "schema": _PHRASE_SCHEMA,
        }
        text_cfg: ResponseTextConfigParam = {"format": format_cfg}
        kwargs = dict(model=self._model, input=prompt, text=text_cfg)
        if self._reasoning:
            kwargs["reasoning"] = {"effort": self._reasoning}
        t0 = time.perf_counter()
        resp = await self._client.responses.create(**kwargs)
        dt = time.perf_counter() - t0
        obj = json.loads(resp.output_text)
        return obj, dt

    async def _resolve_phrase(
        self, phrase: str, ref_dt: datetime
    ) -> list[Interval]:
        """Resolve a text phrase to intervals via Duckling + quarter regex."""
        ivs: list[Interval] = []
        # Quarter abbreviation fallback (Duckling can't parse Q3 2024).
        for m in _QUARTER_RE.finditer(phrase):
            ivs.append(_quarter_interval(int(m.group(1)), int(m.group(2))))
        # Strip quarters from phrase to avoid Duckling confusion.
        cleaned = _QUARTER_RE.sub("", phrase).strip(" ,")
        if cleaned:
            entities = await self._extractor.extract_anchors(cleaned, ref_dt)
            for ent in entities:
                ivs.extend(ent.intervals)
        return ivs

    async def plan(self, query: str, ref_time: str) -> Plan:
        key = self._key(query, ref_time)
        cached_obj = self._cache.get(key)
        if cached_obj is None:
            try:
                obj, _ = await self._llm_call(query, ref_time)
                self._cache[key] = obj
                self._dirty = True
            except Exception as e:
                return Plan(parse_error=str(e))
        else:
            obj = cached_obj

        try:
            ref_dt = parse_iso(ref_time)
        except Exception as e:
            return Plan(parse_error=str(e))

        include_ivs_per_phrase: list[list[Interval]] = []
        for ph in obj.get("include_phrases", []):
            include_ivs_per_phrase.append(
                await self._resolve_phrase(ph, ref_dt)
            )
        exclude_ivs_per_phrase: list[list[Interval]] = []
        for ph in obj.get("exclude_phrases", []):
            exclude_ivs_per_phrase.append(
                await self._resolve_phrase(ph, ref_dt)
            )
        extremum = obj.get("extremum")
        if extremum not in ("latest", "earliest"):
            extremum = None

        include_flat = [iv for group in include_ivs_per_phrase for iv in group]
        exclude_flat = [iv for group in exclude_ivs_per_phrase for iv in group]

        # Build plan from include/exclude
        if not include_flat and not exclude_flat:
            targets = []
        elif exclude_flat and not include_flat:
            # Pure complement
            comp = _complement(exclude_flat)
            targets = [IntervalSet.from_intervals(comp)] if comp else []
        elif exclude_flat and include_flat:
            # Bounded scope binding: intersect(include) ∩ complement(exclude)
            comp = _complement(exclude_flat)
            bounded = _intersect(include_flat, comp)
            targets = [IntervalSet.from_intervals(bounded)] if bounded else []
        else:
            # Only include: each phrase's intervals become one target
            # (graded coverage when multiple phrases).
            targets = []
            for group in include_ivs_per_phrase:
                if group:
                    targets.append(IntervalSet.from_intervals(group))

        return Plan(targets=targets, extremum=extremum)
