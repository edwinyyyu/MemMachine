"""Model-agnostic cue-generation specification.

Motivation: prior tests showed gpt-5-nano catastrophically underperforms
gpt-5-mini on the v2f cue-generation prompt (-22 to -27pp r@20). The fix can't
be "find the right prompt for nano" — it must be a **specification** that any
competent model can satisfy. Structure the output contract (constraints +
verify-repair loop) so different models produce different prose but all
satisfy the same rules.

Spec constraints (post-hoc verifiable, per-cue):
  1. Length: 8-35 words
  2. Entity overlap: >= 1 non-stopword token shared with the query
  3. Anti-paraphrase: does not start with {what, when, how, why, who, which,
     where}
  4. Anti-duplication: Jaccard similarity with query < 0.4
  5. Register: casual chat fragment, 1-2 sentences (enforced by prompt +
     length bound; mechanical check rejects more than 2 sentence terminators)

Set-level constraints:
  6. Pairwise cosine < 0.85 (anti-redundant)
  7. Cosine with query > 0.3 (anti-random)

Architectures defined here (all use V2f-style hop0-top-10 + per-cue-top-10):
  - CueSpecMini         : gpt-5-mini + spec prompt, 1 repair max
  - CueSpecNano         : gpt-5-nano + spec prompt + repair loop
  - CueSpecNanoNoRepair : gpt-5-nano + spec prompt, no repair (ablation)
  - V2fNano             : gpt-5-nano + vanilla V2f prompt (replicates the
                          known failure mode)

The reference `meta_v2f` (mini + vanilla V2f) baseline is read from existing
result files; we don't re-run it here.

Dedicated caches: cuespec_embedding_cache.json, cuespec_llm_cache.json.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

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
from openai import OpenAI

# ---------------------------------------------------------------------------
# Dedicated caches — read all existing caches for maximum hits, write to
# dedicated files to avoid collisions with concurrent agents.
# ---------------------------------------------------------------------------
_CUESPEC_EMB_FILE = CACHE_DIR / "cuespec_embedding_cache.json"
_CUESPEC_LLM_FILE = CACHE_DIR / "cuespec_llm_cache.json"

_EMB_READ_CACHES = (
    "embedding_cache.json",
    "arch_embedding_cache.json",
    "agent_embedding_cache.json",
    "frontier_embedding_cache.json",
    "meta_embedding_cache.json",
    "optim_embedding_cache.json",
    "synth_test_embedding_cache.json",
    "bestshot_embedding_cache.json",
    "antipara_embedding_cache.json",
    "type_enum_embedding_cache.json",
    "cuespec_embedding_cache.json",
)
_LLM_READ_CACHES = (
    "llm_cache.json",
    "arch_llm_cache.json",
    "agent_llm_cache.json",
    "tree_llm_cache.json",
    "frontier_llm_cache.json",
    "meta_llm_cache.json",
    "optim_llm_cache.json",
    "synth_test_llm_cache.json",
    "bestshot_llm_cache.json",
    "antipara_llm_cache.json",
    "type_enum_llm_cache.json",
    "nano_llm_cache.json",
    "cuespec_llm_cache.json",
)


class CueSpecEmbeddingCache(EmbeddingCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in _EMB_READ_CACHES:
            p = self.cache_dir / name
            if not p.exists():
                continue
            try:
                with open(p) as f:
                    self._cache.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
        self.cache_file = _CUESPEC_EMB_FILE
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
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new_entries = {}


class CueSpecLLMCache(LLMCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in _LLM_READ_CACHES:
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
        self.cache_file = _CUESPEC_LLM_FILE
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
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new_entries = {}


# ---------------------------------------------------------------------------
# Constraint definitions
# ---------------------------------------------------------------------------

# Minimal English stopword list — big enough to catch "what/where/which"
# bloat without over-filtering useful content words.
_STOPWORDS: frozenset[str] = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "his", "i", "if", "in", "into", "is", "it", "its", "me", "my", "not", "of", "on", "or", "our", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "through", "to", "was", "we", "were", "what", "when", "where", "which", "who", "whom", "whose", "why", "will", "with", "would", "you", "your", "yours", "about", "above", "after", "all", "also", "am", "any", "below", "both", "can", "could", "even", "from", "had", "her", "hers", "him", "himself", "itself", "just", "like", "many", "may", "might", "must", "nor", "now", "off", "only", "other", "out", "over", "same", "several", "since", "still", "tell", "than", "though", "too", "very", "via", "when", "whenever", "whether", "while", "whom", "with", "without", "yet"]
)

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
_INTERROGATIVE_PREFIXES: frozenset[str] = frozenset(
    {"what", "when", "how", "why", "who", "which", "where"}
)

# Set-level thresholds
_MAX_PAIRWISE_COSINE = 0.85
_MIN_QUERY_COSINE = 0.30

# Per-cue thresholds
_MIN_WORDS = 8
_MAX_WORDS = 35
_MAX_JACCARD_QUERY = 0.40
_MAX_SENTENCES = 2


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _content_tokens(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in _STOPWORDS and len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = a & b
    union = a | b
    if not union:
        return 0.0
    return len(inter) / len(union)


def _sentence_count(text: str) -> int:
    # Count terminators; empty cues return 0, single unterminated sentence
    # returns 1 (treated as 1 sentence).
    term = len(re.findall(r"[.!?](?=\s|$)", text.strip()))
    if term == 0:
        return 1 if text.strip() else 0
    return term


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class CueFailure:
    """Per-cue failure record."""

    index: int
    cue: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Outcome of validating a set of cues against the spec."""

    cues: list[str]
    query: str
    failures: list[CueFailure]
    set_level_failures: list[str]

    @property
    def ok(self) -> bool:
        return not self.failures and not self.set_level_failures

    def as_feedback(self) -> str:
        """Human-readable feedback string for the repair prompt."""
        if self.ok:
            return ""
        lines: list[str] = []
        if self.failures:
            lines.append("Per-cue failures:")
            for f in self.failures:
                reasons = "; ".join(f.reasons)
                lines.append(f"  cue {f.index + 1} ({reasons}): {f.cue!r}")
        if self.set_level_failures:
            lines.append("Set-level failures:")
            for msg in self.set_level_failures:
                lines.append(f"  {msg}")
        return "\n".join(lines)


def validate_cues(
    cues: list[str],
    query: str,
    cue_embeddings: list[np.ndarray] | None = None,
    query_embedding: np.ndarray | None = None,
    context_text: str = "",
) -> ValidationReport:
    """Check cues against the spec. Embeddings are optional; if omitted,
    set-level cosine checks are skipped (useful for a cheap pre-check).

    `context_text` is the hop0 excerpts that were given to the generator.
    Entity-overlap is satisfied when a cue shares a content word with EITHER
    the query OR the retrieved context — this lets the LLM use corpus
    vocabulary (names, objects mentioned in the excerpts) that isn't in the
    question itself.
    """
    query_tokens = _content_tokens(query)
    query_all_tokens = set(_tokens(query))
    context_tokens = _content_tokens(context_text) if context_text else set()
    allowed_overlap_vocab = query_tokens | context_tokens

    per_cue_failures: list[CueFailure] = []
    for i, cue in enumerate(cues):
        reasons: list[str] = []
        stripped = cue.strip()
        if not stripped:
            reasons.append("empty")
            per_cue_failures.append(CueFailure(index=i, cue=cue, reasons=reasons))
            continue

        words = _tokens(stripped)
        wc = len(words)
        if wc < _MIN_WORDS:
            reasons.append(f"too_short({wc}w)")
        elif wc > _MAX_WORDS:
            reasons.append(f"too_long({wc}w)")

        cue_content = _content_tokens(stripped)
        if not (cue_content & allowed_overlap_vocab) and allowed_overlap_vocab:
            reasons.append("no_entity_overlap")

        first_word = words[0] if words else ""
        if first_word in _INTERROGATIVE_PREFIXES:
            reasons.append(f"interrogative_prefix({first_word})")

        cue_all = set(words)
        j = _jaccard(cue_all, query_all_tokens)
        if j >= _MAX_JACCARD_QUERY:
            reasons.append(f"jaccard_query({j:.2f})")

        sc = _sentence_count(stripped)
        if sc > _MAX_SENTENCES:
            reasons.append(f"too_many_sentences({sc})")

        if reasons:
            per_cue_failures.append(CueFailure(index=i, cue=cue, reasons=reasons))

    set_level: list[str] = []
    if cue_embeddings is not None and query_embedding is not None:
        # Anti-redundancy
        for i in range(len(cue_embeddings)):
            for j in range(i + 1, len(cue_embeddings)):
                cos = _cosine(cue_embeddings[i], cue_embeddings[j])
                if cos > _MAX_PAIRWISE_COSINE:
                    set_level.append(
                        f"cues_{i + 1}_and_{j + 1}_too_similar(cos={cos:.2f})"
                    )
        # Anti-random
        for i, emb in enumerate(cue_embeddings):
            cos = _cosine(emb, query_embedding)
            if cos < _MIN_QUERY_COSINE:
                set_level.append(f"cue_{i + 1}_too_unrelated_to_query(cos={cos:.2f})")

    return ValidationReport(
        cues=list(cues),
        query=query,
        failures=per_cue_failures,
        set_level_failures=set_level,
    )


# ---------------------------------------------------------------------------
# Prompts — specification-driven (same text for any model)
# ---------------------------------------------------------------------------

CUESPEC_PROMPT = """\
You generate search cues for retrieval over a conversation history. Cues \
are embedded and compared via cosine similarity against stored chat turns.

Question: {question}

{context_section}

Generate {num_cues} cues. Use specific vocabulary that would appear in the \
target conversation turns. If the retrieved excerpts clearly mention \
relevant names, objects, or phrases, use them. If the excerpts don't \
contain the answer, generate plausible content — names, specifics, \
events — that such a conversation might have said.

Each cue MUST satisfy every rule below:

1. LENGTH: between 8 and 35 words.
2. ENTITY OVERLAP: contains at least one specific content word taken from \
the question OR from the retrieved excerpts (a name, object, place, \
event, topic — NOT question words like "what/when/how/why/who/which/\
where" and NOT common words like "the/is/did/had").
3. NOT INTERROGATIVE: does not start with "what", "when", "how", "why", \
"who", "which", or "where".
4. NOT A PARAPHRASE: does not restate the question. Generate CONTENT \
THAT WOULD ANSWER the question — the kind of text that someone in the \
conversation would actually have typed. First-person ("My research \
focused on...", "I went to...") often works well because it mimics how \
people describe their own experiences.
5. CASUAL REGISTER: 1-2 sentences of chat-message prose. No bullet lists, \
no boolean operators, no search syntax.
6. DIVERSE: different cues must target DIFFERENT aspects (different \
keywords, different sub-topic). Cues must not paraphrase each other.

Output strict JSON on a single line, nothing else:
{{"cues": ["cue 1 text", "cue 2 text"]}}
"""


CUESPEC_REPAIR_PROMPT = """\
You generate search cues for retrieval over a conversation history.

Question: {question}

{context_section}

Your previous attempt failed the spec. Rules to satisfy (ALL must hold):

1. LENGTH: 8-35 words per cue.
2. ENTITY OVERLAP: each cue must contain a specific content word copied \
from the question (name, object, topic — not "what/when/how/the/is").
3. NOT INTERROGATIVE: do not start with "what/when/how/why/who/which/where".
4. NOT A PARAPHRASE: generate content that would ANSWER the question in \
a chat message, NOT a restated question. First-person phrasing often \
works well ("My research focused on...", "I went to the...").
5. CASUAL REGISTER: 1-2 sentences of chat prose.
6. DIVERSE: each of the {num_cues} cues targets a DIFFERENT aspect.

Your previous attempt:
{previous_cues_block}

Failures found:
{failure_feedback}

Regenerate all {num_cues} cues to fix these specific failures. Keep good \
cues if they passed; replace the bad ones. Output strict JSON on a single \
line:
{{"cues": ["cue 1 text", "cue 2 text"]}}
"""


# JSON parser that tolerates ``` fences, leading/trailing prose.
_JSON_OBJ_RE = re.compile(r"\{[^\{\}]*\"cues\"\s*:\s*\[.*?\][^\{\}]*\}", re.DOTALL)


def _parse_cuespec_output(response: str) -> list[str]:
    """Parse {"cues": [...]} from LLM output. Fallback: parse CUE: lines."""
    if not response:
        return []
    # Strip markdown fences
    text = response.strip()
    if text.startswith("```"):
        # drop first line of fence and trailing fence
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # First try direct JSON
    for candidate in (text,):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and isinstance(obj.get("cues"), list):
                return [str(c).strip() for c in obj["cues"] if str(c).strip()]
        except json.JSONDecodeError:
            pass

    # Scan for a JSON object with "cues"
    m = _JSON_OBJ_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and isinstance(obj.get("cues"), list):
                return [str(c).strip() for c in obj["cues"] if str(c).strip()]
        except json.JSONDecodeError:
            pass

    # Last resort: v2f-style CUE: lines
    fallback = _parse_cues(response)
    return fallback


# ---------------------------------------------------------------------------
# Base retrieval class with configurable model + verify-repair loop.
# ---------------------------------------------------------------------------


@dataclass
class CueSpecResult:
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


class CueSpecBase:
    """Base retrieval class. Subclasses set `model`, `arch_name`, whether
    to use the spec prompt, and how many repair rounds to allow."""

    model: str = "gpt-5-mini"
    arch_name: str = "cuespec_base"
    use_spec_prompt: bool = True
    max_repair_rounds: int = 2
    num_cues: int = 2

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
    ) -> None:
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = CueSpecEmbeddingCache()
        self.llm_cache = CueSpecLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0

    def embed_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
        cached = self.embedding_cache.get(text)
        if cached is not None:
            self.embed_calls += 1
            return cached
        response = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        self.embedding_cache.put(text, embedding)
        self.embed_calls += 1
        return embedding

    def llm_call(self, prompt: str) -> str:
        cached = self.llm_cache.get(self.model, prompt)
        # Treat empty cached responses as cache-miss (nano's reasoning tokens
        # sometimes exhaust the budget; we want to retry at higher cap, not
        # permanently cache the empty string).
        if cached is not None and cached.strip():
            self.llm_calls += 1
            return cached
        # nano burns ~2000 reasoning tokens per call for these spec prompts;
        # 4000 is not always enough. 6000 leaves headroom for visible output.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=6000,
        )
        text = response.choices[0].message.content or ""
        if text.strip():
            self.llm_cache.put(self.model, prompt, text)
        self.llm_calls += 1
        return text

    def save_caches(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0

    # -- Generation with verify-repair loop -------------------------------
    def _generate_cues(
        self,
        question: str,
        context_section: str,
        context_text: str = "",
    ) -> tuple[list[str], list[dict]]:
        """Return (final_cues, attempt_log). Attempt log records each
        generation + validation cycle so evaluator can report repair rates."""
        attempts: list[dict] = []

        if not self.use_spec_prompt:
            # Vanilla V2f path — no repair, no validation.
            prompt = V2F_PROMPT.format(
                question=question, context_section=context_section
            )
            output = self.llm_call(prompt)
            cues = _parse_cues(output)[: self.num_cues]
            attempts.append(
                {
                    "attempt": 0,
                    "prompt_kind": "v2f",
                    "output": output,
                    "parsed_cues": cues,
                    "validation": None,
                }
            )
            return cues, attempts

        prompt = CUESPEC_PROMPT.format(
            question=question,
            context_section=context_section,
            num_cues=self.num_cues,
        )
        output = self.llm_call(prompt)
        cues = _parse_cuespec_output(output)[: self.num_cues]

        # Cheap pre-check (no embeddings) — rejects per-cue issues. Set-level
        # embedding checks run after pre-check passes.
        rpt = validate_cues(cues, question, context_text=context_text)
        attempts.append(
            {
                "attempt": 0,
                "prompt_kind": "spec",
                "output": output,
                "parsed_cues": list(cues),
                "validation": {
                    "ok": rpt.ok,
                    "failures": [
                        {"index": f.index, "cue": f.cue, "reasons": f.reasons}
                        for f in rpt.failures
                    ],
                    "set_level_failures": rpt.set_level_failures,
                },
            }
        )

        # Repair loop
        last_cues = cues
        last_report = rpt
        for round_i in range(self.max_repair_rounds):
            if last_report.ok:
                break
            # Build repair prompt
            prev_block_lines = []
            for idx, c in enumerate(last_cues):
                prev_block_lines.append(f"  cue {idx + 1}: {c}")
            previous_block = "\n".join(prev_block_lines) or "  (none)"
            repair_prompt = CUESPEC_REPAIR_PROMPT.format(
                question=question,
                context_section=context_section,
                num_cues=self.num_cues,
                previous_cues_block=previous_block,
                failure_feedback=last_report.as_feedback(),
            )
            output = self.llm_call(repair_prompt)
            new_cues = _parse_cuespec_output(output)[: self.num_cues]
            new_rpt = validate_cues(new_cues, question, context_text=context_text)
            attempts.append(
                {
                    "attempt": round_i + 1,
                    "prompt_kind": "repair",
                    "output": output,
                    "parsed_cues": list(new_cues),
                    "validation": {
                        "ok": new_rpt.ok,
                        "failures": [
                            {"index": f.index, "cue": f.cue, "reasons": f.reasons}
                            for f in new_rpt.failures
                        ],
                        "set_level_failures": new_rpt.set_level_failures,
                    },
                }
            )
            last_cues = new_cues
            last_report = new_rpt

        return last_cues, attempts

    def retrieve(self, question: str, conversation_id: str) -> CueSpecResult:
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        context_text = _format_segments(all_segments)
        context_section = "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + context_text

        cues, attempts = self._generate_cues(
            question, context_section, context_text=context_text
        )

        # Compute set-level embedding checks (for reporting) on final cues
        cue_embs = [self.embed_text(c) for c in cues if c.strip()]
        final_set_report = (
            validate_cues(
                cues,
                question,
                cue_embs,
                query_emb,
                context_text=context_text,
            )
            if cues
            else None
        )

        # Retrieve per-cue top-10
        for cue in cues:
            if not cue.strip():
                continue
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        metadata: dict = {
            "name": self.arch_name,
            "model": self.model,
            "cues": cues,
            "num_attempts": len(attempts),
            "num_repair_rounds": max(len(attempts) - 1, 0),
            "attempts": attempts,
            "final_validation_ok": (
                final_set_report.ok if final_set_report is not None else False
            ),
            "final_validation": (
                {
                    "ok": final_set_report.ok,
                    "failures": [
                        {"index": f.index, "cue": f.cue, "reasons": f.reasons}
                        for f in final_set_report.failures
                    ],
                    "set_level_failures": final_set_report.set_level_failures,
                }
                if final_set_report is not None
                else None
            ),
        }
        return CueSpecResult(segments=all_segments, metadata=metadata)


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class CueSpecMini(CueSpecBase):
    """gpt-5-mini + spec prompt + up to 1 repair (control: should match
    pristine v2f on mini)."""

    model = "gpt-5-mini"
    arch_name = "cuespec_mini"
    use_spec_prompt = True
    max_repair_rounds = 1


class CueSpecNano(CueSpecBase):
    """gpt-5-nano + spec prompt + up to 2 repair rounds. PRIMARY TEST."""

    model = "gpt-5-nano"
    arch_name = "cuespec_nano"
    use_spec_prompt = True
    max_repair_rounds = 2


class CueSpecNanoNoRepair(CueSpecBase):
    """gpt-5-nano + spec prompt, no repair (ablation: is repair doing the
    work, or is the spec prompt alone sufficient?)"""

    model = "gpt-5-nano"
    arch_name = "cuespec_nano_no_repair"
    use_spec_prompt = True
    max_repair_rounds = 0


class V2fNano(CueSpecBase):
    """gpt-5-nano + vanilla V2f prompt. Replicates the known failure mode
    (no spec, no repair)."""

    model = "gpt-5-nano"
    arch_name = "v2f_nano"
    use_spec_prompt = False
    max_repair_rounds = 0


def build_variants(store: SegmentStore) -> dict[str, CueSpecBase]:
    return {
        "cuespec_mini": CueSpecMini(store),
        "cuespec_nano": CueSpecNano(store),
        "cuespec_nano_no_repair": CueSpecNanoNoRepair(store),
        "v2f_nano": V2fNano(store),
    }
