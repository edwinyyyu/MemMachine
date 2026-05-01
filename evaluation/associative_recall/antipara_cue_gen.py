"""Anti-paraphrase + verbatim-quote cue generation variants.

Background: per-cue attribution analysis showed losing cues are dominantly
"interrogative paraphrases" — short cues that restate the question or guess
an answer. These concentrate in `locomo_temporal` (25% winner rate) and
`locomo_single_hop`. Winning cues are corpus-grounded (longer + entity-dense).

This module tests three minimal prompt variants over the v2f baseline:

  V2fAntiParaphrase            — v2f + 2 explicit "Do NOT" instructions
  V2fVerbatimQuote             — v2f + verbatim-phrase requirement (+ post filter)
  V2fAntiParaParaVerbatim      — both together

All use MetaV2f's hop0-top-10 + 2-cue-top-10 structure. No framework edits.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
from associative_recall import (
    CACHE_DIR,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)
from best_shot import (
    V2F_PROMPT,
    BestshotBase,
    BestshotResult,
    _format_segments,
    _parse_cues,
)
from openai import OpenAI

# ---------------------------------------------------------------------------
# Dedicated caches — read-only on shared caches, write to own files
# to avoid concurrent-writer corruption with other agents.
# ---------------------------------------------------------------------------

_ANTIPARA_EMB_FILE = CACHE_DIR / "antipara_embedding_cache.json"
_ANTIPARA_LLM_FILE = CACHE_DIR / "antipara_llm_cache.json"

# Read these existing caches to maximize cache hits (questions, segments,
# cosine-retrieval embeddings are shared with prior runs). Writes ONLY go
# to the dedicated antipara_* files.
_SHARED_EMB_READ = (
    "embedding_cache.json",
    "arch_embedding_cache.json",
    "agent_embedding_cache.json",
    "frontier_embedding_cache.json",
    "meta_embedding_cache.json",
    "optim_embedding_cache.json",
    "synth_test_embedding_cache.json",
    "bestshot_embedding_cache.json",
    "fewshot_embedding_cache.json",  # may or may not exist
    "antipara_embedding_cache.json",
)
_SHARED_LLM_READ = (
    "llm_cache.json",
    "arch_llm_cache.json",
    "agent_llm_cache.json",
    "tree_llm_cache.json",
    "frontier_llm_cache.json",
    "meta_llm_cache.json",
    "optim_llm_cache.json",
    "synth_test_llm_cache.json",
    "bestshot_llm_cache.json",
    "fewshot_llm_cache.json",
    "antipara_llm_cache.json",
)


class AntiparaEmbeddingCache(EmbeddingCache):
    """Reads shared embedding caches (best-effort), writes only to dedicated file."""

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
                # Skip corrupt/partial shared caches silently.
                continue
        self.cache_file = _ANTIPARA_EMB_FILE
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


class AntiparaLLMCache(LLMCache):
    """Reads shared LLM caches (best-effort), writes only to dedicated file."""

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
        self.cache_file = _ANTIPARA_LLM_FILE
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
# Prompt variants
# ---------------------------------------------------------------------------

# V2f + two explicit negative instructions. Minimal diff: insert two bullets
# right after the existing "Do NOT write questions" line.
V2F_ANTIPARA_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.
Do NOT restate or paraphrase the question.
Do NOT guess the answer.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


# V2f + verbatim-quote requirement. When hop0 excerpts exist the cue must
# include at least one 2-5 word phrase copied verbatim from them.
V2F_VERBATIM_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

RULE: each cue you generate MUST include at least one EXACT 2-5 word phrase \
copied verbatim from the excerpts above. This grounds the cue in the \
actual conversation vocabulary. Pick phrases that are distinctive (names, \
objects, events) rather than generic filler.

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Each cue must be \
15-40 words and must quote a 2-5 word phrase from the excerpts.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


# V2f + anti-paraphrase + verbatim-quote. Combined.
V2F_ANTIPARA_VERBATIM_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

RULE: each cue you generate MUST include at least one EXACT 2-5 word phrase \
copied verbatim from the excerpts above. This grounds the cue in the \
actual conversation vocabulary. Pick phrases that are distinctive (names, \
objects, events) rather than generic filler.

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Each cue must be \
15-40 words and must quote a 2-5 word phrase from the excerpts.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.
Do NOT restate or paraphrase the question.
Do NOT guess the answer.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


# ---------------------------------------------------------------------------
# Verbatim check
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def verbatim_check(cue: str, context: str, min_n: int = 2, max_n: int = 5) -> bool:
    """True if cue contains any 2-5 word n-gram that also appears in context.

    Case-insensitive, word-tokenized.
    """
    cue_toks = _tokens(cue)
    ctx_toks = _tokens(context)
    if not cue_toks or not ctx_toks:
        return False
    for n in range(min_n, max_n + 1):
        if _ngrams(cue_toks, n) & _ngrams(ctx_toks, n):
            return True
    return False


# ---------------------------------------------------------------------------
# Base class: v2f skeleton with a swappable prompt template
# ---------------------------------------------------------------------------


@dataclass
class _CueOutcome:
    cue: str
    kept: bool
    reason: str  # "ok" | "no_verbatim" | "empty"


class _V2fVariantBase(BestshotBase):
    """V2f-structured retrieval with a swappable prompt and optional filter.

    Subclasses set `prompt_template` and `enforce_verbatim`.
    """

    prompt_template: str = V2F_PROMPT
    enforce_verbatim: bool = False
    arch_name: str = "v2f_variant"

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
    ):
        super().__init__(store, client)
        # Override caches: read shared caches for hits, write only to
        # dedicated antipara_* files to avoid colliding with concurrent agents.
        self.embedding_cache = AntiparaEmbeddingCache()
        self.llm_cache = AntiparaLLMCache()

    def _hop0_context_text(self, segments: list[Segment]) -> str:
        """Return the bare excerpts text used both for the prompt and the
        verbatim filter. Mirrors `_format_segments` output."""
        return _format_segments(segments)

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        hop0_text = self._hop0_context_text(all_segments)
        context_section = "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + hop0_text

        # Fallback behavior: if hop0 is empty, verbatim variants degrade to
        # vanilla v2f to avoid asking the LLM to quote from nothing.
        if self.enforce_verbatim and not all_segments:
            prompt = V2F_PROMPT.format(
                question=question, context_section=context_section
            )
        else:
            prompt = self.prompt_template.format(
                question=question, context_section=context_section
            )

        output = self.llm_call(prompt)
        cues = _parse_cues(output)[:2]

        outcomes: list[_CueOutcome] = []
        kept_cues: list[str] = []
        for cue in cues:
            if not cue.strip():
                outcomes.append(_CueOutcome(cue=cue, kept=False, reason="empty"))
                continue
            if self.enforce_verbatim and all_segments:
                if not verbatim_check(cue, hop0_text):
                    outcomes.append(
                        _CueOutcome(cue=cue, kept=False, reason="no_verbatim")
                    )
                    continue
            outcomes.append(_CueOutcome(cue=cue, kept=True, reason="ok"))
            kept_cues.append(cue)

        for cue in kept_cues:
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

        return BestshotResult(
            segments=all_segments,
            metadata={
                "name": self.arch_name,
                "output": output,
                "cues": kept_cues,
                "cues_raw": [o.cue for o in outcomes],
                "cue_outcomes": [
                    {"cue": o.cue, "kept": o.kept, "reason": o.reason} for o in outcomes
                ],
                "hop0_empty": len(hop0.segments) == 0,
            },
        )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class MetaV2fDedicated(_V2fVariantBase):
    """Reference v2f baseline — same logic as `best_shot.MetaV2f` but using
    the antipara dedicated caches so concurrent agents can't corrupt them.
    """

    prompt_template = V2F_PROMPT
    enforce_verbatim = False
    arch_name = "meta_v2f"


class V2fAntiParaphrase(_V2fVariantBase):
    """V2f prompt + two explicit 'Do NOT restate/guess' instructions."""

    prompt_template = V2F_ANTIPARA_PROMPT
    enforce_verbatim = False
    arch_name = "v2f_anti_paraphrase"


class V2fVerbatimQuote(_V2fVariantBase):
    """V2f prompt + require 2-5 word verbatim phrase from hop0 excerpts.

    Post-filter drops cues that fail `verbatim_check`.
    """

    prompt_template = V2F_VERBATIM_PROMPT
    enforce_verbatim = True
    arch_name = "v2f_verbatim_quote"


class V2fAntiParaphraseVerbatim(_V2fVariantBase):
    """Combination of anti-paraphrase + verbatim-quote."""

    prompt_template = V2F_ANTIPARA_VERBATIM_PROMPT
    enforce_verbatim = True
    arch_name = "v2f_anti_paraphrase_verbatim"
