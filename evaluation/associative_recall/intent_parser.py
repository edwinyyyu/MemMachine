"""Structured intent parser + constraint-based retrieval.

Different from multichannel_weighted (blind channel weighting) and
keyword_router (picks one specialist): this architecture does EXPLICIT
structured query planning.

Pipeline
--------
1. ONE LLM call per query produces a typed JSON intent plan:
    {
        "intent_type": "factual-lookup" | "preference" |
            "temporal-compare" | "multi-hop-inference" |
            "commitment-tracking" | "synthesis" | "counterfactual" |
            "other",
        "entities": ["Caroline", "Phoenix"],
        "constraints": {
            "speaker": "Caroline" | null,
            "temporal_relation": {"marker": "after",
                                    "reference": "Monday meeting"} | null,
            "negation": true | false,
            "quantity_bound": null | {...},
            "answer_form": "date" | "person" | "number" |
                "description" | "list" | null
        },
        "primary_topic": "Phoenix status",
        "needs_aggregation": false
    }

2. A retrieval plan is derived from the parsed structure:
    - v2f cosine on primary_topic (or raw query if topic missing)
    - speaker filter if constraints.speaker set and we can resolve it
    - temporal-token restricted cosine if temporal_relation set
    - negation boost when constraints.negation is true
    - answer-form boost (dates -> temporal tokens; person -> proper nouns)
    - intent-type boost (preference -> first-person self-statements)
    - K bump when answer_form=list or needs_aggregation

3. Signals are MERGED as stacked addition (NOT max-score merge, which
    displaces v2f's good picks). Each signal contributes w * normalized
    signal score (default w=0.05).

Variants
--------
  intent_parser_full               — full parse + all constraints
  intent_parser_critical_only      — speaker + temporal only
  intent_parser_no_plan_exec       — parse but run v2f only (parser-cost
                                     control)

Caches
------
  cache/intent_embedding_cache.json
  cache/intent_llm_cache.json
  cache/intent_parse_cache.json    (structured JSON plans by query-hash)

Reads-only from existing shared caches to warm-start.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)
from best_shot import V2F_PROMPT, _format_segments, _parse_cues
from dotenv import load_dotenv
from openai import OpenAI
from speaker_attributed import extract_name_mentions

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"

# Dedicated caches — writes go ONLY here.
_INTENT_EMB_FILE = CACHE_DIR / "intent_embedding_cache.json"
_INTENT_LLM_FILE = CACHE_DIR / "intent_llm_cache.json"
_INTENT_PARSE_FILE = CACHE_DIR / "intent_parse_cache.json"

_SHARED_EMB_READ = (
    "embedding_cache.json",
    "arch_embedding_cache.json",
    "frontier_embedding_cache.json",
    "meta_embedding_cache.json",
    "optim_embedding_cache.json",
    "bestshot_embedding_cache.json",
    "antipara_embedding_cache.json",
    "alias_embedding_cache.json",
    "speaker_embedding_cache.json",
    "two_speaker_embedding_cache.json",
    "multich_embedding_cache.json",
    "intent_embedding_cache.json",
)
_SHARED_LLM_READ = (
    "llm_cache.json",
    "arch_llm_cache.json",
    "tree_llm_cache.json",
    "frontier_llm_cache.json",
    "meta_llm_cache.json",
    "optim_llm_cache.json",
    "bestshot_llm_cache.json",
    "alias_llm_cache.json",
    "speaker_llm_cache.json",
    "two_speaker_llm_cache.json",
    "antipara_llm_cache.json",
    "multich_llm_cache.json",
    "intent_llm_cache.json",
)


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
class IntentEmbeddingCache(EmbeddingCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in _SHARED_EMB_READ:
            p = self.cache_dir / name
            if not p.exists():
                continue
            try:
                with open(p) as f:
                    self._cache.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
        self.cache_file = _INTENT_EMB_FILE
        self._new_entries: dict[str, list[float]] = {}

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding.tolist()
        self._new_entries[key] = embedding.tolist()

    def save(self) -> None:
        if not self._new_entries:
            return
        existing: dict[str, list[float]] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.update(self._new_entries)
        tmp = self.cache_file.parent / (self.cache_file.name + f".tmp.{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        os.replace(tmp, self.cache_file)
        self._new_entries = {}


class IntentLLMCache(LLMCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in _SHARED_LLM_READ:
            p = self.cache_dir / name
            if not p.exists():
                continue
            try:
                with open(p) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            for k, v in data.items():
                if v:
                    self._cache[k] = v
        self.cache_file = _INTENT_LLM_FILE
        self._new_entries: dict[str, str] = {}

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        self._cache[key] = response
        self._new_entries[key] = response

    def save(self) -> None:
        if not self._new_entries:
            return
        existing: dict[str, str] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.update(self._new_entries)
        tmp = self.cache_file.parent / (self.cache_file.name + f".tmp.{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        os.replace(tmp, self.cache_file)
        self._new_entries = {}


class IntentParseCache:
    """Dedicated cache for structured intent plans, keyed by query text."""

    def __init__(self) -> None:
        self.cache_file = _INTENT_PARSE_FILE
        self._cache: dict[str, dict] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._cache = {}
        self._new_entries: dict[str, dict] = {}

    def get(self, query: str) -> dict | None:
        return self._cache.get(query.strip())

    def put(self, query: str, plan: dict) -> None:
        k = query.strip()
        self._cache[k] = plan
        self._new_entries[k] = plan

    def save(self) -> None:
        if not self._new_entries:
            return
        existing: dict[str, dict] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.update(self._new_entries)
        tmp = self.cache_file.parent / (self.cache_file.name + f".tmp.{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        os.replace(tmp, self.cache_file)
        self._new_entries = {}


# ---------------------------------------------------------------------------
# Intent schema + parsing prompt
# ---------------------------------------------------------------------------
INTENT_TYPES = (
    "factual-lookup",
    "preference",
    "temporal-compare",
    "multi-hop-inference",
    "commitment-tracking",
    "synthesis",
    "counterfactual",
    "other",
)

ANSWER_FORMS = (
    "date",
    "person",
    "number",
    "description",
    "list",
    "yes-no",
    None,
)

INTENT_PARSE_PROMPT = """\
You are a query planner for a conversational memory retrieval system. \
Given a user question about a past conversation, extract its structured \
intent so retrieval can apply typed constraints.

Output ONLY a single JSON object matching this schema (no prose before or \
after, no code fences):

{{
  "intent_type": one of {intent_types},
  "entities": array of proper-noun entities mentioned in the query,
  "constraints": {{
    "speaker": string or null — a speaker/participant named in the query \
whose words are being asked about (null if the query is agnostic about \
speaker),
    "temporal_relation": null or \
{{"marker": "after|before|during|when|between", "reference": "<event or \
time>"}},
    "negation": true if the query asks about what was NOT done / denied / \
refused, else false,
    "quantity_bound": null or {{"operator": "gte|lte|eq", "value": N}} \
when the query asks about a specific count or threshold,
    "answer_form": one of ["date","person","number","description","list",\
"yes-no"] or null
  }},
  "primary_topic": a short phrase (<= 6 words) naming the MOST specific \
topic to retrieve on. Prefer proper nouns and distinctive vocabulary over \
function words. Use the raw question's noun phrase, not a rewording.,
  "needs_aggregation": true if answering requires combining multiple turns \
("all", "every", "list of", "total", "overall"), else false
}}

Rules:
- Only set "speaker" when the query names a specific person whose \
statements are under scope (e.g. "What did Caroline say..."). Do NOT fill \
from general references ("the user", "my friend").
- Only set "temporal_relation" when the query explicitly anchors in time \
(e.g. "after the Monday meeting", "last month", "before the launch"). \
Generic mentions of dates in a topic do not count.
- Entities are capitalized multi-word proper nouns ONLY (people, places, \
products, project names).
- Do NOT invent constraints. If in doubt, leave null / false.

Query: {query}"""


def build_intent_parse_prompt(query: str) -> str:
    types_str = json.dumps(list(INTENT_TYPES))
    return INTENT_PARSE_PROMPT.format(intent_types=types_str, query=query.strip())


def _coerce_str(x) -> str | None:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        return s or None
    return None


def _coerce_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("true", "yes", "1")
    return False


def parse_intent_response(raw: str) -> dict:
    """Parse LLM response into a normalized intent plan dict.

    Always returns a dict with the full schema; missing fields become
    defaults (null/false/[]) so downstream code can assume fields exist.
    """
    empty = {
        "intent_type": "other",
        "entities": [],
        "constraints": {
            "speaker": None,
            "temporal_relation": None,
            "negation": False,
            "quantity_bound": None,
            "answer_form": None,
        },
        "primary_topic": None,
        "needs_aggregation": False,
        "parse_ok": False,
    }
    if not raw or not raw.strip():
        return empty

    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return empty

    if not isinstance(obj, dict):
        return empty

    intent_type = _coerce_str(obj.get("intent_type")) or "other"
    if intent_type not in INTENT_TYPES:
        intent_type = "other"

    entities_raw = obj.get("entities", []) or []
    entities: list[str] = []
    if isinstance(entities_raw, list):
        for e in entities_raw:
            se = _coerce_str(e)
            if se:
                entities.append(se)

    constraints_raw = obj.get("constraints", {}) or {}
    if not isinstance(constraints_raw, dict):
        constraints_raw = {}

    speaker = _coerce_str(constraints_raw.get("speaker"))

    tr_raw = constraints_raw.get("temporal_relation")
    temporal_relation: dict | None = None
    if isinstance(tr_raw, dict):
        marker = _coerce_str(tr_raw.get("marker"))
        reference = _coerce_str(tr_raw.get("reference"))
        if marker or reference:
            temporal_relation = {
                "marker": (marker or "").lower() or None,
                "reference": reference,
            }

    negation = _coerce_bool(constraints_raw.get("negation"))

    qb_raw = constraints_raw.get("quantity_bound")
    quantity_bound: dict | None = None
    if isinstance(qb_raw, dict):
        op = _coerce_str(qb_raw.get("operator"))
        val = qb_raw.get("value")
        if op or (val is not None):
            quantity_bound = {
                "operator": (op or "").lower() or None,
                "value": val,
            }

    answer_form = _coerce_str(constraints_raw.get("answer_form"))
    if answer_form:
        answer_form = answer_form.lower()
    if answer_form not in ("date", "person", "number", "description", "list", "yes-no"):
        answer_form = None

    primary_topic = _coerce_str(obj.get("primary_topic"))
    needs_aggregation = _coerce_bool(obj.get("needs_aggregation"))

    return {
        "intent_type": intent_type,
        "entities": entities,
        "constraints": {
            "speaker": speaker,
            "temporal_relation": temporal_relation,
            "negation": negation,
            "quantity_bound": quantity_bound,
            "answer_form": answer_form,
        },
        "primary_topic": primary_topic,
        "needs_aggregation": needs_aggregation,
        "parse_ok": True,
    }


# ---------------------------------------------------------------------------
# Lexical helpers (replicated locally per brief, do not depend on
# multichannel_weighted)
# ---------------------------------------------------------------------------
_RE_DATE_WORDS = re.compile(
    r"\b("
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun|"
    r"today|tomorrow|yesterday|tonight|week|month|year|weekend|weekday|"
    r"morning|afternoon|evening|night|noon|midnight|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"last|next|previous|before|after|recent|recently|later|earlier|"
    r"once|twice|ago|"
    r"spring|summer|fall|autumn|winter"
    r")\b",
    re.IGNORECASE,
)
_RE_TIME = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)\b")
_RE_DATE_DIGIT = re.compile(
    r"\b(?:\d{4}|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|"
    r"\d{1,2}(?:st|nd|rd|th))\b"
)

_RE_NEGATION = re.compile(
    r"\b("
    r"not|never|didn'?t|don'?t|doesn'?t|won'?t|wouldn'?t|can'?t|cannot|"
    r"couldn'?t|refuse[ds]?|decline[ds]?|rejected?|avoid(?:ed|ing)?|"
    r"against|no longer|stop(?:ped)?|denied|deny|nope"
    r")\b",
    re.IGNORECASE,
)

_RE_FIRST_PERSON_PREF = re.compile(
    r"\bI\s+("
    r"like|love|prefer|hate|enjoy|want|need|use|drink|eat|avoid|own|have|"
    r"work|think|believe|wish|live|am|'m"
    r")\b",
    re.IGNORECASE,
)

_NAME_STOPWORDS = {
    "I",
    "You",
    "Me",
    "My",
    "We",
    "Us",
    "Our",
    "Your",
    "They",
    "Them",
    "He",
    "She",
    "It",
    "This",
    "That",
    "These",
    "Those",
    "There",
    "Here",
    "What",
    "When",
    "Where",
    "Who",
    "Why",
    "How",
    "Which",
    "Whom",
    "Whose",
    "The",
    "A",
    "An",
    "Is",
    "Are",
    "Was",
    "Were",
    "Be",
    "Been",
    "Being",
    "Have",
    "Has",
    "Had",
    "Do",
    "Does",
    "Did",
    "Will",
    "Would",
    "Should",
    "Could",
    "Can",
    "May",
    "Might",
    "Must",
    "Shall",
    "Yes",
    "No",
    "Not",
    "Never",
    "Always",
    "Maybe",
    "And",
    "Or",
    "But",
    "If",
    "So",
    "Then",
    "Else",
    "Because",
    "Since",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "January",
    "February",
    "March",
    "April",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
}
_RE_CAPWORD = re.compile(r"\b([A-Z][a-zA-Z\-']+)\b")


def has_temporal_tokens(text: str) -> bool:
    if _RE_DATE_WORDS.search(text):
        return True
    if _RE_TIME.search(text):
        return True
    if _RE_DATE_DIGIT.search(text):
        return True
    return False


def has_date_tokens(text: str) -> bool:
    """Specifically a date (not just a generic temporal word)."""
    if _RE_DATE_DIGIT.search(text):
        return True
    # Month/day names
    if re.search(
        r"\b(january|february|march|april|may|june|july|august|"
        r"september|october|november|december|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    return False


def has_proper_noun(text: str) -> bool:
    # Look beyond the first-word capitalization (sentences start capitalized)
    parts = text.split()
    for i, tok in enumerate(parts):
        clean = re.sub(r"[^A-Za-z\-']", "", tok)
        if not clean:
            continue
        if i == 0:
            continue
        if clean[0].isupper() and clean not in _NAME_STOPWORDS:
            return True
    return False


def has_negation_markers(text: str) -> bool:
    return bool(_RE_NEGATION.search(text))


def has_first_person_preference(text: str) -> bool:
    return bool(_RE_FIRST_PERSON_PREF.search(text))


# ---------------------------------------------------------------------------
# Speaker-map loading (reuse ingest artifacts if present)
# ---------------------------------------------------------------------------
_CONV_SPEAKERS_FILE = (
    Path(__file__).resolve().parent / "results" / "conversation_speakers.json"
)
_CONV_TWO_SPEAKERS_FILE = (
    Path(__file__).resolve().parent / "results" / "conversation_two_speakers.json"
)


def load_speaker_map() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if _CONV_TWO_SPEAKERS_FILE.exists():
        try:
            with open(_CONV_TWO_SPEAKERS_FILE) as f:
                data = json.load(f)
            raw = data.get("speakers", {}) or {}
            for cid, pair in raw.items():
                if isinstance(pair, dict):
                    out[cid] = {
                        "user": (pair.get("user") or "UNKNOWN").strip() or "UNKNOWN",
                        "assistant": (pair.get("assistant") or "UNKNOWN").strip()
                        or "UNKNOWN",
                    }
        except (json.JSONDecodeError, OSError):
            pass
    if _CONV_SPEAKERS_FILE.exists():
        try:
            with open(_CONV_SPEAKERS_FILE) as f:
                data = json.load(f)
            one = data.get("speakers", {}) or {}
            for cid, name in one.items():
                if cid not in out:
                    out[cid] = {
                        "user": name or "UNKNOWN",
                        "assistant": "UNKNOWN",
                    }
        except (json.JSONDecodeError, OSError):
            pass
    return out


# ---------------------------------------------------------------------------
# Main architecture
# ---------------------------------------------------------------------------
@dataclass
class IntentResult:
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


class IntentParserArch:
    """Parse query into structured intent, then execute typed constraints.

    mode:
        "full"          — parse + apply all constraint signals
        "critical_only" — parse + only speaker + temporal signals (+v2f base)
        "no_plan_exec"  — parse but ignore structure (isolates parser cost)
    """

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        mode: str = "full",
        per_channel_top_k: int = 50,
        signal_bonus: float = 0.05,
    ):
        self.store = store
        self.client = client or OpenAI(timeout=60.0, max_retries=3)
        self.mode = mode
        self.per_channel_top_k = per_channel_top_k
        self.signal_bonus = signal_bonus

        self.embedding_cache = IntentEmbeddingCache()
        self.llm_cache = IntentLLMCache()
        self.parse_cache = IntentParseCache()
        self.embed_calls = 0
        self.llm_calls = 0

        self.speaker_map = load_speaker_map()

        # Precomputed masks/flags per store (fixed per store).
        self.role_masks = {
            "user": np.array([s.role == "user" for s in store.segments], dtype=bool),
            "assistant": np.array(
                [s.role == "assistant" for s in store.segments], dtype=bool
            ),
        }
        self.temporal_mask = np.array(
            [has_temporal_tokens(s.text) for s in store.segments],
            dtype=bool,
        )
        self.date_mask = np.array(
            [has_date_tokens(s.text) for s in store.segments], dtype=bool
        )
        self.proper_noun_mask = np.array(
            [has_proper_noun(s.text) for s in store.segments], dtype=bool
        )
        self.negation_mask = np.array(
            [has_negation_markers(s.text) for s in store.segments], dtype=bool
        )
        self.first_person_pref_mask = np.array(
            [has_first_person_preference(s.text) for s in store.segments],
            dtype=bool,
        )

        if mode == "full":
            self.arch_name = "intent_parser_full"
        elif mode == "critical_only":
            self.arch_name = "intent_parser_critical_only"
        elif mode == "no_plan_exec":
            self.arch_name = "intent_parser_no_plan_exec"
        else:
            raise KeyError(mode)

    # --- cache helpers ---
    def embed_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
        cached = self.embedding_cache.get(text)
        if cached is not None:
            self.embed_calls += 1
            return cached
        response = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        emb = np.array(response.data[0].embedding, dtype=np.float32)
        self.embedding_cache.put(text, emb)
        self.embed_calls += 1
        return emb

    def llm_call(self, prompt: str, model: str = MODEL) -> str:
        cached = self.llm_cache.get(model, prompt)
        if cached is not None:
            self.llm_calls += 1
            return cached
        last_exc = None
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=2000,
                )
                text = resp.choices[0].message.content or ""
                self.llm_cache.put(model, prompt, text)
                self.llm_calls += 1
                return text
            except Exception as e:
                last_exc = e
                time.sleep(1.5 * (attempt + 1))
        print(f"    LLM call failed after 3 attempts: {last_exc}", flush=True)
        self.llm_cache.put(model, prompt, "")
        self.llm_calls += 1
        return ""

    def save_caches(self) -> None:
        try:
            self.embedding_cache.save()
        except Exception as e:
            print(f"  (warn) embedding_cache.save failed: {e}", flush=True)
        try:
            self.llm_cache.save()
        except Exception as e:
            print(f"  (warn) llm_cache.save failed: {e}", flush=True)
        try:
            self.parse_cache.save()
        except Exception as e:
            print(f"  (warn) parse_cache.save failed: {e}", flush=True)

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0

    # --- intent parse ---
    def parse_intent(self, query: str) -> dict:
        cached = self.parse_cache.get(query)
        if cached:
            # Cache hit: zero LLM cost for planning
            return cached
        prompt = build_intent_parse_prompt(query)
        raw = self.llm_call(prompt)
        plan = parse_intent_response(raw)
        self.parse_cache.put(query, plan)
        return plan

    # --- retrieval helpers ---
    def _cosine_search(
        self,
        query_emb: np.ndarray,
        conversation_id: str,
        mask: np.ndarray | None = None,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        k = top_k or self.per_channel_top_k
        q = query_emb.astype(np.float32)
        qn = max(float(np.linalg.norm(q)), 1e-10)
        q = q / qn
        sims = self.store.normalized_embeddings @ q
        conv_mask = self.store.conversation_ids == conversation_id
        combined = conv_mask.copy()
        if mask is not None:
            combined = combined & mask
        sims = np.where(combined, sims, -1.0)
        order = np.argsort(sims)[::-1][: max(k, 1)]
        out: list[tuple[int, float]] = []
        for i in order:
            if sims[i] <= -0.5:
                break
            out.append((int(i), float(sims[i])))
        return out

    def _v2f_cosine(self, query: str, conversation_id: str) -> list[tuple[int, float]]:
        """V2f-style retrieval matching MetaV2f's ordering.

        Returns an ordered list of (seg_index, rank_score) where the ordering
        is: primer top-10 first, then each cue's top-10 hits appended in
        order (deduped). rank_score is a synthetic descending value so that
        the merge step later can still normalize by "top score" — but the
        primary ordering signal is LIST POSITION, not cosine score.
        """
        query_emb = self.embed_text(query)
        primer = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        primer_segs = list(primer.segments)
        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + _format_segments(primer_segs)
        )
        prompt = V2F_PROMPT.format(question=query, context_section=context_section)
        output = self.llm_call(prompt)
        cues = _parse_cues(output)[:2]

        ordered_indices: list[int] = []
        seen: set[int] = set()
        for seg in primer_segs:
            if seg.index not in seen:
                ordered_indices.append(seg.index)
                seen.add(seg.index)

        for cue in cues:
            cue_emb = self.embed_text(cue)
            res = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=seen,
            )
            for seg in res.segments:
                if seg.index not in seen:
                    ordered_indices.append(seg.index)
                    seen.add(seg.index)

        # Assign synthetic descending scores so list position is preserved
        # under later rank-by-score. A small epsilon gap ensures ties break
        # toward list-order.
        out: list[tuple[int, float]] = []
        total = len(ordered_indices)
        for rank, idx in enumerate(ordered_indices):
            score = 1.0 - (rank / max(total, 1)) * 0.5  # range [0.5, 1.0]
            out.append((idx, score))
        return out

    # --- core ---
    @staticmethod
    def _normalize(cands: list[tuple[int, float]]) -> list[tuple[int, float]]:
        if not cands:
            return cands
        top = max(sc for _, sc in cands)
        if top <= 0:
            return [(i, 0.0) for i, _ in cands]
        return [(i, sc / top) for i, sc in cands]

    def _resolve_speaker_role(
        self, speaker_name: str, conversation_id: str, query: str
    ) -> str | None:
        """Map a plan's 'speaker' to 'user' or 'assistant' using the
        conversation's speaker map. Returns None if speaker can't be resolved.

        Also returns None if the parsed speaker doesn't appear in the
        query's tokens (defensive: LLM hallucinated a speaker)."""
        if not speaker_name:
            return None
        speaker_lower = speaker_name.lower().strip()
        # Defensive: parsed speaker must appear in the original query.
        q_mentions = {t.lower() for t in extract_name_mentions(query)}
        q_tokens = {t.lower() for t in re.findall(r"[A-Za-z]+", query)}
        if speaker_lower not in q_mentions and speaker_lower.split()[0] not in q_tokens:
            return None

        pair = self.speaker_map.get(conversation_id, {})
        user_name = (pair.get("user") or "UNKNOWN").lower()
        asst_name = (pair.get("assistant") or "UNKNOWN").lower()

        def _name_match(a: str, b: str) -> bool:
            if not a or not b:
                return False
            if a == b:
                return True
            a_parts = a.split()
            b_parts = b.split()
            if a_parts and b_parts:
                if a_parts[0] == b_parts[0]:
                    return True
            return False

        if user_name != "unknown" and _name_match(speaker_lower, user_name):
            return "user"
        if asst_name != "unknown" and _name_match(speaker_lower, asst_name):
            return "assistant"
        return None

    def retrieve(self, question: str, conversation_id: str) -> IntentResult:
        plan = self.parse_intent(question)
        constraints = plan.get("constraints", {}) or {}
        signals_applied: list[str] = []
        signals_detected: list[str] = []

        # ---------- Base channel: v2f on raw question ---------------------
        # We always run v2f on the raw question (NOT on primary_topic):
        # primary_topic is a compression; using it as the v2f query drops
        # context that v2f's cue-generation would otherwise leverage. The
        # primary_topic IS used downstream as a secondary retrieval signal.
        primary_topic = plan.get("primary_topic") or question
        if not primary_topic or len(primary_topic) < 3:
            primary_topic = question

        # no_plan_exec: parse but completely ignore structure.
        if self.mode == "no_plan_exec":
            base = self._v2f_cosine(question, conversation_id)
            final = dict(base)
            ranked = sorted(final.items(), key=lambda x: -x[1])
            segments = [self.store.segments[i] for i, _ in ranked]
            return IntentResult(
                segments=segments,
                metadata={
                    "name": self.arch_name,
                    "plan": plan,
                    "signals_detected": [],
                    "signals_applied": [],
                    "mode": self.mode,
                },
            )

        base = self._v2f_cosine(question, conversation_id)
        base_norm = self._normalize(base)
        final_scores: dict[int, float] = {i: sc for i, sc in base_norm}

        # ---------- Signals ----------
        # Detect which signals are present in the plan (independent of
        # whether they're applied in this mode).
        if constraints.get("speaker"):
            signals_detected.append("speaker")
        if constraints.get("temporal_relation"):
            signals_detected.append("temporal_relation")
        if constraints.get("negation"):
            signals_detected.append("negation")
        if constraints.get("answer_form") == "date":
            signals_detected.append("answer_form:date")
        if constraints.get("answer_form") == "person":
            signals_detected.append("answer_form:person")
        if constraints.get("answer_form") == "list":
            signals_detected.append("answer_form:list")
        if plan.get("needs_aggregation"):
            signals_detected.append("needs_aggregation")
        if plan.get("intent_type") == "preference":
            signals_detected.append("intent_type:preference")

        # In critical_only mode, only apply speaker + temporal.
        allowed = None
        if self.mode == "critical_only":
            allowed = {"speaker", "temporal_relation"}

        def _allowed(name: str) -> bool:
            return allowed is None or name in allowed

        query_emb = self.embed_text(question)
        topic_emb = self.embed_text(primary_topic)

        # 1. Speaker filter
        sp_name = constraints.get("speaker")
        if sp_name and _allowed("speaker"):
            role = self._resolve_speaker_role(sp_name, conversation_id, question)
            if role in ("user", "assistant"):
                mask = self.role_masks[role]
                res = self._cosine_search(topic_emb, conversation_id, mask=mask)
                for idx, sc in self._normalize(res):
                    final_scores[idx] = (
                        final_scores.get(idx, 0.0) + self.signal_bonus * sc
                    )
                signals_applied.append("speaker")

        # 2. Temporal-token restricted cosine
        tr = constraints.get("temporal_relation")
        if tr and _allowed("temporal_relation"):
            res = self._cosine_search(
                query_emb, conversation_id, mask=self.temporal_mask
            )
            for idx, sc in self._normalize(res):
                final_scores[idx] = final_scores.get(idx, 0.0) + self.signal_bonus * sc
            # Also embed the temporal reference itself if available
            ref = tr.get("reference") if isinstance(tr, dict) else None
            if ref and len(ref) >= 3:
                ref_emb = self.embed_text(ref)
                res2 = self._cosine_search(
                    ref_emb, conversation_id, mask=self.temporal_mask
                )
                for idx, sc in self._normalize(res2):
                    final_scores[idx] = (
                        final_scores.get(idx, 0.0) + self.signal_bonus * sc
                    )
            signals_applied.append("temporal_relation")

        # 3. Negation boost
        if constraints.get("negation") and _allowed("negation"):
            res = self._cosine_search(
                query_emb, conversation_id, mask=self.negation_mask
            )
            for idx, sc in self._normalize(res):
                final_scores[idx] = final_scores.get(idx, 0.0) + self.signal_bonus * sc
            signals_applied.append("negation")

        # 4. Answer-form specific boosts
        af = constraints.get("answer_form")
        if af == "date" and _allowed("answer_form:date"):
            res = self._cosine_search(query_emb, conversation_id, mask=self.date_mask)
            for idx, sc in self._normalize(res):
                final_scores[idx] = final_scores.get(idx, 0.0) + self.signal_bonus * sc
            signals_applied.append("answer_form:date")
        elif af == "person" and _allowed("answer_form:person"):
            res = self._cosine_search(
                query_emb, conversation_id, mask=self.proper_noun_mask
            )
            for idx, sc in self._normalize(res):
                final_scores[idx] = final_scores.get(idx, 0.0) + self.signal_bonus * sc
            signals_applied.append("answer_form:person")

        # 5. Preference intent: boost first-person self-statements
        if plan.get("intent_type") == "preference" and _allowed(
            "intent_type:preference"
        ):
            res = self._cosine_search(
                query_emb,
                conversation_id,
                mask=self.first_person_pref_mask,
            )
            for idx, sc in self._normalize(res):
                final_scores[idx] = final_scores.get(idx, 0.0) + self.signal_bonus * sc
            signals_applied.append("intent_type:preference")

        # 6. Answer-form list / needs_aggregation: expand top-K via extra
        #    raw-cosine pull (append un-seen candidates at modest score).
        expand = False
        if af == "list" and _allowed("answer_form:list"):
            expand = True
            signals_applied.append("answer_form:list")
        if plan.get("needs_aggregation") and _allowed("needs_aggregation"):
            expand = True
            if "needs_aggregation" not in signals_applied:
                signals_applied.append("needs_aggregation")
        if expand:
            # Pull a wider raw-cosine set and add any new candidates at a
            # small score so they land in the backfill zone.
            wide = self._cosine_search(
                query_emb, conversation_id, top_k=self.per_channel_top_k * 2
            )
            for idx, sc in wide:
                if idx not in final_scores:
                    final_scores[idx] = self.signal_bonus * 0.5

        # ---------- Rank ----------
        ranked = sorted(final_scores.items(), key=lambda x: -x[1])
        segments = [self.store.segments[i] for i, _ in ranked]

        metadata: dict = {
            "name": self.arch_name,
            "plan": plan,
            "signals_detected": signals_detected,
            "signals_applied": signals_applied,
            "num_candidates": len(final_scores),
            "mode": self.mode,
            "top5_turn_ids": [
                (self.store.segments[i].turn_id, float(sc)) for i, sc in ranked[:5]
            ],
        }
        return IntentResult(segments=segments, metadata=metadata)


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------
def build_variant(
    name: str, store: SegmentStore, client: OpenAI | None = None
) -> IntentParserArch:
    if name == "intent_parser_full":
        return IntentParserArch(store, client=client, mode="full")
    if name == "intent_parser_critical_only":
        return IntentParserArch(store, client=client, mode="critical_only")
    if name == "intent_parser_no_plan_exec":
        return IntentParserArch(store, client=client, mode="no_plan_exec")
    raise KeyError(name)


VARIANTS = (
    "intent_parser_full",
    "intent_parser_critical_only",
    "intent_parser_no_plan_exec",
)
