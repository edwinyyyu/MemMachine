"""Answer-type aware reranking.

Classify each query's expected answer type (DATE, PERSON, NUMBER, LOCATION,
REASON, DESCRIPTION), then rerank retrieved top-K candidates by presence of
answer-type tokens.

Two components:

1) Classification: rule-based (covers ~95% of cases), with gpt-5-mini fallback
   for ambiguous "what/which" questions.
2) Token detection: regex-based per-type for turn text.

Rerank options (applied to v2f's already-retrieved ranked candidate pool):
 - additive bonus at various alphas (0.05, 0.1, 0.2): new_score = cosine + alpha * hit
 - hard filter: if any turn hits, keep hits first, then non-hits.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from associative_recall import CACHE_DIR, Segment
from openai import OpenAI

ANSWER_TYPES = ("DATE", "PERSON", "NUMBER", "LOCATION", "REASON", "DESCRIPTION")


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
class AnswerTypeLLMCache:
    """LLM cache for answer-type classification (gpt-5-mini)."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "answer_type_llm_cache.json"
        self._cache: dict[str, str] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}
        self._new: dict[str, str] = {}

    def _key(self, model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()

    def get(self, model: str, prompt: str) -> str | None:
        return self._cache.get(self._key(model, prompt))

    def put(self, model: str, prompt: str, response: str) -> None:
        k = self._key(model, prompt)
        self._cache[k] = response
        self._new[k] = response

    def save(self) -> None:
        if not self._new:
            return
        existing = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing.update(self._new)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


class AnswerTypeDecisionCache:
    """Cache classifier decisions (string -> AnswerType) per-question."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "answer_type_decision_cache.json"
        self._cache: dict[str, str] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}
        self._new: dict[str, str] = {}

    def _key(self, q: str) -> str:
        return hashlib.sha256(q.encode()).hexdigest()

    def get(self, q: str) -> str | None:
        return self._cache.get(self._key(q))

    def put(self, q: str, answer_type: str) -> None:
        k = self._key(q)
        self._cache[k] = answer_type
        self._new[k] = answer_type

    def save(self) -> None:
        if not self._new:
            return
        existing = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing.update(self._new)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
TIME_WORD_RE = re.compile(
    r"\b(year|month|week|day|date|time|when|schedule|hour|morning|afternoon|evening|night|"
    r"weekday|weekend|today|yesterday|tomorrow|last week|next week|season|holiday|"
    r"birthday|anniversary|semester|quarter|ago|since|before|after|recent)\b",
    re.IGNORECASE,
)
PERSON_WORD_RE = re.compile(
    r"\b(who|whose|whom|name|person|people|friend|colleague|roommate|partner|spouse|"
    r"sibling|parent|brother|sister|mom|dad|child|son|daughter|boyfriend|girlfriend)\b",
    re.IGNORECASE,
)
NUMBER_WORD_RE = re.compile(
    r"\b(how many|how much|how old|how long|how often|number of|amount of|quantity|count|"
    r"price|cost|percentage|percent)\b",
    re.IGNORECASE,
)
LOCATION_WORD_RE = re.compile(
    r"\b(where|place|city|town|country|location|venue|restaurant|store|shop|address)\b",
    re.IGNORECASE,
)
REASON_WORD_RE = re.compile(
    r"\b(why|reason|because|purpose)\b",
    re.IGNORECASE,
)


def classify_answer_type_rule(question: str) -> tuple[str, bool]:
    """Return (answer_type, confident). confident=False triggers LLM fallback.

    Covers most cases. Returns confident=False only for ambiguous what/which.
    """
    q = question.strip()
    ql = q.lower()

    # Strip leading punctuation/whitespace
    ql_stripped = ql.lstrip()

    # Priority order: most specific first
    if NUMBER_WORD_RE.search(ql_stripped[:40]) and (
        ql_stripped.startswith("how many")
        or ql_stripped.startswith("how much")
        or ql_stripped.startswith("how old")
        or ql_stripped.startswith("how long")
        or ql_stripped.startswith("how often")
    ):
        return "NUMBER", True
    if (
        ql_stripped.startswith("when")
        or ql_stripped.startswith("on what")
        or ql_stripped.startswith("at what")
    ):
        return "DATE", True
    if (
        ql_stripped.startswith("who")
        or ql_stripped.startswith("whose")
        or ql_stripped.startswith("whom")
    ):
        return "PERSON", True
    if ql_stripped.startswith("where"):
        return "LOCATION", True
    if ql_stripped.startswith("why"):
        return "REASON", True

    # what/which ambiguous — rely on time/number/location/person words
    if ql_stripped.startswith("what") or ql_stripped.startswith("which"):
        if TIME_WORD_RE.search(ql_stripped):
            return "DATE", True
        if NUMBER_WORD_RE.search(ql_stripped):
            return "NUMBER", True
        if LOCATION_WORD_RE.search(ql_stripped):
            return "LOCATION", True
        if PERSON_WORD_RE.search(ql_stripped):
            return "PERSON", True
        # Ambiguous -> LLM fallback (marked confident=False)
        return "DESCRIPTION", False

    # Other structures
    if TIME_WORD_RE.search(ql_stripped[:60]):
        return "DATE", True
    if NUMBER_WORD_RE.search(ql_stripped[:60]):
        return "NUMBER", True

    return "DESCRIPTION", True


LLM_CLASSIFY_PROMPT = """\
What is the expected answer type for this question? Pick ONE: \
DATE, PERSON, NUMBER, LOCATION, REASON, DESCRIPTION.
Question: {q}
Output only the label."""


def classify_answer_type(
    question: str,
    llm_cache: AnswerTypeLLMCache,
    decision_cache: AnswerTypeDecisionCache,
    client: OpenAI | None = None,
    model: str = "gpt-5-mini",
) -> str:
    """Classify the expected answer type, with caching and rule-first strategy."""
    cached = decision_cache.get(question)
    if cached is not None:
        return cached

    at, confident = classify_answer_type_rule(question)
    if confident:
        decision_cache.put(question, at)
        return at

    # LLM fallback
    if client is None:
        # Fall back to rule decision
        decision_cache.put(question, at)
        return at

    prompt = LLM_CLASSIFY_PROMPT.format(q=question)
    cached_llm = llm_cache.get(model, prompt)
    if cached_llm is not None:
        resp = cached_llm
    else:
        try:
            comp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            resp = (comp.choices[0].message.content or "").strip()
            llm_cache.put(model, prompt, resp)
        except Exception:
            decision_cache.put(question, at)
            return at

    # Parse
    resp_u = resp.upper()
    for t in ANSWER_TYPES:
        if t in resp_u:
            decision_cache.put(question, t)
            return t
    decision_cache.put(question, at)
    return at


# ---------------------------------------------------------------------------
# Token detection per turn
# ---------------------------------------------------------------------------
MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
    re.IGNORECASE,
)
WEEKDAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b",
    re.IGNORECASE,
)
REL_DATE_RE = re.compile(
    r"\b(today|yesterday|tomorrow|tonight|this week|last week|next week|"
    r"this month|last month|next month|this year|last year|next year|"
    r"last night|this weekend|last weekend|next weekend|recently|"
    r"few days ago|a week ago|weeks ago|months ago|years ago|ago)\b",
    re.IGNORECASE,
)
DATE_NUM_RE = re.compile(r"\b\d{1,4}[/\-]\d{1,2}(?:[/\-]\d{1,4})?\b")
TIME_RE = re.compile(r"\b\d{1,2}\s*(?:am|pm|AM|PM)\b|\b\d{1,2}:\d{2}\b")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
SEQ_RE = re.compile(
    r"\b(then|after|before|earlier|later|afterwards|afterward|first|finally|eventually)\b",
    re.IGNORECASE,
)

NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
QUANTIFIER_RE = re.compile(
    r"\b(couple|few|several|many|dozens|dozen|hundred|thousand|twice|thrice|both)\b",
    re.IGNORECASE,
)
MONEY_RE = re.compile(r"\$\d+|\d+\s*(dollars?|bucks?|cents?|usd|\$)\b", re.IGNORECASE)
PERCENT_RE = re.compile(r"\d+\s*%|\d+\s*percent\b", re.IGNORECASE)

# Person heuristic: Capitalized words that aren't sentence starts.
CAPITALIZED_RE = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]{2,}\b")
# Also match via simple capitalized name at sentence start
STANDALONE_CAP_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")

PRONOUN_RE = re.compile(
    r"\b(he|she|him|her|his|hers|they|them|their|theirs)\b",
    re.IGNORECASE,
)

# LOCATION heuristic
LOC_PREP_RE = re.compile(
    r"\b(at|in|to|from|near)\s+[A-Z][a-zA-Z]+",
    re.IGNORECASE,
)
PLACE_WORDS_RE = re.compile(
    r"\b(city|street|road|avenue|park|restaurant|cafe|coffee shop|bar|mall|store|"
    r"hotel|beach|mountain|hospital|home|house|office|school|university|college|"
    r"airport|station|country)\b",
    re.IGNORECASE,
)

REASON_MARKER_RE = re.compile(
    r"\b(because|since|so that|due to|that's why|thats why|"
    r"in order to|the reason|as a result|thanks to)\b",
    re.IGNORECASE,
)


def turn_has_answer_type_tokens(text: str, answer_type: str) -> bool:
    """Return True if the turn text plausibly contains tokens of the answer type."""
    if not text:
        return False
    t = text
    if answer_type == "DATE":
        return bool(
            MONTH_RE.search(t)
            or WEEKDAY_RE.search(t)
            or REL_DATE_RE.search(t)
            or DATE_NUM_RE.search(t)
            or TIME_RE.search(t)
            or YEAR_RE.search(t)
            or SEQ_RE.search(t)
        )
    if answer_type == "NUMBER":
        return bool(
            NUM_RE.search(t)
            or QUANTIFIER_RE.search(t)
            or MONEY_RE.search(t)
            or PERCENT_RE.search(t)
        )
    if answer_type == "PERSON":
        # Has a capitalized name-like token (mid-sentence), or pronoun reference
        # (Pronouns are weaker signal, don't count alone.)
        for m in STANDALONE_CAP_RE.finditer(t):
            tok = m.group(0)
            # Filter common non-name capitalized words
            if tok.lower() in {
                "i",
                "i'm",
                "im",
                "ill",
                "ive",
                "id",
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "so",
                "this",
                "that",
                "there",
                "then",
                "when",
                "where",
                "why",
                "how",
                "what",
                "which",
                "who",
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            }:
                continue
            return True
        return False
    if answer_type == "LOCATION":
        return bool(LOC_PREP_RE.search(t) or PLACE_WORDS_RE.search(t))
    if answer_type == "REASON":
        return bool(REASON_MARKER_RE.search(t))
    # DESCRIPTION: every turn passes
    return True


# ---------------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------------
@dataclass
class RerankedResult:
    ranked: list[tuple[Segment, float]]  # (segment, new_score), already deduped


def rerank_additive_bonus(
    ranked: list[tuple[Segment, float]],
    answer_type: str,
    alpha: float,
) -> list[tuple[Segment, float]]:
    """Add alpha to score if turn contains answer-type tokens; stable-resort desc.

    Input ranked is list of (segment, score) in current order.
    """
    scored: list[tuple[float, int, Segment]] = []
    for i, (s, sc) in enumerate(ranked):
        bonus = alpha if turn_has_answer_type_tokens(s.text, answer_type) else 0.0
        # use original index as tiebreaker to preserve original order for ties
        scored.append((sc + bonus, i, s))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(s, sc) for sc, _, s in scored]


def rerank_hard_filter(
    ranked: list[tuple[Segment, float]],
    answer_type: str,
) -> list[tuple[Segment, float]]:
    """If any segment has answer-type tokens, put those first (stable), then
    non-matches. If none match, keep original order."""
    hits: list[tuple[Segment, float]] = []
    misses: list[tuple[Segment, float]] = []
    for s, sc in ranked:
        if turn_has_answer_type_tokens(s.text, answer_type):
            hits.append((s, sc))
        else:
            misses.append((s, sc))
    if not hits:
        return list(ranked)
    return hits + misses
