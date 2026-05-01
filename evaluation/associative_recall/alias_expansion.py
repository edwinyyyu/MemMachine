"""Query-time alias expansion via ingest-extracted alias groups.

Motivation
----------
Evolving terminology / multi-referent entities defeat cue generation (v2f et al.)
because aliases are corpus-specific ("Project Phoenix" = "the portal thing" =
"v2"). A cue-generating LLM cannot invent these aliases without having read the
conversation. The aliases USED in answer-bearing turns get missed even when the
alias-DECLARING turn is retrieved.

Pipeline
--------
Ingest-time (once per conversation):
  Single LLM call over the full conversation text (~10-25K tokens — fits
  gpt-5-mini context). Prompt asks for groups of names/terms that refer to the
  same thing.

Query-time (per question):
  1. Find candidate mentions in the query (capitalized multi-words, quoted
     terms, also single-word capitalized entities excluding sentence-start).
  2. For each mention e, look up e in this conversation's alias groups. If
     matched to a group G, the expanded query set is
         { q_original } U { q with e replaced by a : a in G minus {e} }
     Also add the "alias-keyword" sibling as a standalone cue (each sibling as
     its own short retrieval probe) to force retrieval of any turn that uses
     the alias without paraphrasing the full question.
  3. Retrieve with each variant (top-10), union the segment pool, rank by max
     cosine across all probes.

Variants
--------
  alias_expand_cosine     — expansion on cosine baseline only, no cue gen.
  alias_expand_v2f        — v2f cue-gen + alias expansion cosine probes (the
                            standard stacking test).
  alias_expand_v2f_cheap  — v2f runs ONCE on original query; alias-expanded
                            queries only contribute cosine retrievals (no extra
                            v2f cue gen per variant). This is the cheap-stacking
                            baseline the plan asks for.

Caches
------
Dedicated alias_*_cache.json (extraction prompts are large → big string keys).
Reads shared caches for embedding hits but writes only to the dedicated files
to avoid corrupting other agents' caches.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
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
from best_shot import (
    MODEL,
    V2F_PROMPT,
    BestshotBase,
    BestshotResult,
    _format_segments,
    _parse_cues,
)
from openai import OpenAI

# ---------------------------------------------------------------------------
# Dedicated caches
# ---------------------------------------------------------------------------

_ALIAS_EMB_FILE = CACHE_DIR / "alias_embedding_cache.json"
_ALIAS_LLM_FILE = CACHE_DIR / "alias_llm_cache.json"
_ALIAS_GROUPS_FILE = (
    Path(__file__).resolve().parent / "results" / "conversation_alias_groups.json"
)

# Best-effort list of shared caches to warm from. Other agents' writes will
# naturally be picked up next time this module is imported.
_SHARED_EMB_READ = (
    "embedding_cache.json",
    "arch_embedding_cache.json",
    "agent_embedding_cache.json",
    "frontier_embedding_cache.json",
    "meta_embedding_cache.json",
    "optim_embedding_cache.json",
    "synth_test_embedding_cache.json",
    "bestshot_embedding_cache.json",
    "fewshot_embedding_cache.json",
    "antipara_embedding_cache.json",
    "inv_query_embedding_cache.json",
    "anchor_embedding_cache.json",
    "alias_embedding_cache.json",
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
    "inv_query_llm_cache.json",
    "anchor_llm_cache.json",
    "alias_llm_cache.json",
)


class AliasEmbeddingCache(EmbeddingCache):
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
        self.cache_file = _ALIAS_EMB_FILE
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


class AliasLLMCache(LLMCache):
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
        self.cache_file = _ALIAS_LLM_FILE
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
# Prompts
# ---------------------------------------------------------------------------

ALIAS_EXTRACTION_PROMPT = """\
You are reading a long conversation transcript. Your job is to identify groups \
of names, phrases, or terms that refer to the SAME underlying entity (person, \
project, place, object, concept). Include formal names alongside any informal \
nicknames, aliases, code names, shortened forms, or alternative phrasings the \
participants use to refer to the same thing across the conversation.

Examples of what counts as an alias group (illustrative only — do NOT copy):
  - ["Project Phoenix", "the portal thing", "Portal 2.0", "the bird", "v2"]
  - ["Sarah", "Sara", "Sar"]
  - ["the Monterey trip", "our California road trip", "the drive down the coast"]
  - ["LGBTQ support group", "the queer group", "Tuesday night group"]

Rules:
  - Only output GENUINE alias groups (2+ distinct ways of referring to the same \
thing, BOTH actually used in the conversation).
  - A person's first name alone is an alias of their full name only if both \
appear.
  - Do NOT include generic words ("the kids", "work") unless they clearly and \
consistently refer to a specific entity introduced elsewhere by a distinct name.
  - Do NOT list one-off mentions or synonyms that only appear once.
  - Err on the side of recall: if two terms plausibly refer to the same thing \
and both are used, include them.

Conversation transcript:
{conversation_text}

Output ONLY a JSON array of alias groups. Each group is an array of strings.
If no alias groups exist, output [].

Example output:
[
  ["Project Phoenix", "Portal 2.0", "the bird", "v2"],
  ["Sara", "Sarah"]
]"""


# ---------------------------------------------------------------------------
# Alias extraction
# ---------------------------------------------------------------------------


def _format_conversation_for_extraction(
    segments: list[Segment], max_chars_per_turn: int = 350
) -> str:
    """Chronological transcript, one line per turn."""
    ordered = sorted(segments, key=lambda s: s.turn_id)
    lines: list[str] = []
    for seg in ordered:
        txt = seg.text.replace("\n", " ").strip()
        if len(txt) > max_chars_per_turn:
            txt = txt[:max_chars_per_turn].rstrip() + "..."
        lines.append(f"[{seg.turn_id}] {seg.role}: {txt}")
    return "\n".join(lines)


def _parse_alias_groups(response: str) -> list[list[str]]:
    """Pull the JSON array out of the LLM response (handles code fences and
    surrounding prose)."""
    if not response:
        return []
    text = response.strip()
    # Strip code fences if present
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1)
    else:
        # Grab the first top-level array
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    groups: list[list[str]] = []
    for group in parsed:
        if not isinstance(group, list):
            continue
        cleaned = [str(x).strip() for x in group if str(x).strip()]
        # Need at least 2 distinct members
        distinct = []
        seen_lower: set[str] = set()
        for c in cleaned:
            cl = c.lower()
            if cl not in seen_lower:
                distinct.append(c)
                seen_lower.add(cl)
        if len(distinct) >= 2:
            groups.append(distinct)
    return groups


class AliasExtractor:
    """Runs a single LLM call per conversation to extract alias groups.

    Persists results to results/conversation_alias_groups.json so the expensive
    call is only made once across variants/runs.
    """

    def __init__(self, client: OpenAI | None = None) -> None:
        if client is None:
            client = OpenAI(timeout=90.0, max_retries=3)
        self.client = client
        self.llm_cache = AliasLLMCache()
        self._groups: dict[str, list[list[str]]] = {}
        self._raw: dict[str, str] = {}
        if _ALIAS_GROUPS_FILE.exists():
            try:
                with open(_ALIAS_GROUPS_FILE) as f:
                    data = json.load(f)
                self._groups = data.get("groups", {}) or {}
                self._raw = data.get("raw_responses", {}) or {}
            except (json.JSONDecodeError, OSError):
                self._groups = {}
                self._raw = {}

    def get_groups(self, conversation_id: str) -> list[list[str]]:
        return list(self._groups.get(conversation_id, []))

    def extract_for_store(
        self, store: SegmentStore, conversation_ids: list[str] | None = None
    ) -> dict[str, list[list[str]]]:
        """Extract alias groups for the given conversations (default: all in
        store). Uses persistent results file + LLM cache.
        """
        all_cids = sorted({s.conversation_id for s in store.segments})
        if conversation_ids is None:
            conversation_ids = all_cids

        pending: list[str] = [
            cid for cid in conversation_ids if cid not in self._groups
        ]
        if not pending:
            return {cid: self._groups[cid] for cid in conversation_ids}

        print(
            f"  [alias] Extracting alias groups for {len(pending)} conversation(s):"
            f" {pending}",
            flush=True,
        )
        for cid in pending:
            segs = [s for s in store.segments if s.conversation_id == cid]
            if not segs:
                self._groups[cid] = []
                continue
            transcript = _format_conversation_for_extraction(segs)
            prompt = ALIAS_EXTRACTION_PROMPT.format(conversation_text=transcript)

            cached = self.llm_cache.get(MODEL, prompt)
            if cached is not None:
                response = cached
            else:
                response = ""
                last_exc: Exception | None = None
                for attempt in range(3):
                    try:
                        completion = self.client.chat.completions.create(
                            model=MODEL,
                            messages=[{"role": "user", "content": prompt}],
                            max_completion_tokens=16000,
                        )
                        response = completion.choices[0].message.content or ""
                        break
                    except Exception as e:
                        last_exc = e
                        time.sleep(2.0 * (attempt + 1))
                if not response and last_exc is not None:
                    print(
                        f"    [alias] LLM call failed for {cid}: {last_exc}",
                        flush=True,
                    )
                self.llm_cache.put(MODEL, prompt, response)
            groups = _parse_alias_groups(response)
            self._groups[cid] = groups
            self._raw[cid] = response
            print(f"    {cid}: {len(groups)} alias groups extracted", flush=True)

        # Persist to disk
        self.llm_cache.save()
        _ALIAS_GROUPS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_ALIAS_GROUPS_FILE, "w") as f:
            json.dump(
                {"groups": self._groups, "raw_responses": self._raw},
                f,
                indent=2,
                default=str,
            )

        return {cid: self._groups[cid] for cid in conversation_ids}


# ---------------------------------------------------------------------------
# Query-time expansion
# ---------------------------------------------------------------------------


_STOPWORDS_SENTENCE_START = {
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "did",
    "does",
    "do",
    "was",
    "were",
    "is",
    "are",
    "has",
    "have",
    "had",
    "can",
    "could",
    "would",
    "should",
    "will",
    "the",
    "a",
    "an",
    "on",
    "in",
    "of",
    "to",
    "and",
    "or",
    "but",
    "for",
    "with",
    "about",
    "from",
    "i",
    "my",
    "me",
    "our",
    "we",
    "you",
    "your",
    "after",
    "before",
    "during",
    "at",
    "as",
    "by",
    "if",
    "so",
    "then",
}


def find_alias_matches(
    query: str, alias_groups: list[list[str]]
) -> list[tuple[str, list[str]]]:
    """Find occurrences of any alias term in the query. Returns a list of
    (matched_term_as_it_appears_in_query, sibling_aliases_list) pairs. The
    sibling list excludes the matched term.

    Matching strategy: case-insensitive whole-word-ish substring match against
    each alias term. Prefer longest-first so that "Project Phoenix" is matched
    before "Phoenix" when both are in different groups.
    """
    matches: list[tuple[str, list[str]]] = []
    ql = query.lower()
    # Flatten to (term, group) pairs sorted by term length desc
    flat: list[tuple[str, list[str]]] = []
    for group in alias_groups:
        for term in group:
            flat.append((term, group))
    flat.sort(key=lambda t: len(t[0]), reverse=True)

    used_groups: set[int] = set()
    consumed_spans: list[tuple[int, int]] = []

    def overlaps(span: tuple[int, int]) -> bool:
        for s, e in consumed_spans:
            if not (span[1] <= s or span[0] >= e):
                return True
        return False

    for term, group in flat:
        gid = id(group)
        if gid in used_groups:
            continue
        tl = term.lower()
        if not tl or len(tl) < 2:
            continue
        # Use a relaxed word boundary: check match and ensure the boundary on
        # each side is non-alnum (or the string edge).
        start = 0
        while True:
            idx = ql.find(tl, start)
            if idx < 0:
                break
            end = idx + len(tl)
            left_ok = idx == 0 or not ql[idx - 1].isalnum()
            right_ok = end == len(ql) or not ql[end].isalnum()
            if left_ok and right_ok and not overlaps((idx, end)):
                matched_in_query = query[idx:end]
                siblings = [a for a in group if a.lower() != tl]
                matches.append((matched_in_query, siblings))
                used_groups.add(gid)
                consumed_spans.append((idx, end))
                break
            start = idx + 1
    return matches


def _replace_first_occurrence(text: str, needle: str, replacement: str) -> str:
    """Replace the first whole-word-ish occurrence of needle (case-insensitive)
    with replacement, preserving the rest of the text verbatim.
    """
    tl = needle.lower()
    ql = text.lower()
    start = 0
    while True:
        idx = ql.find(tl, start)
        if idx < 0:
            return text  # shouldn't happen given upstream matching
        end = idx + len(tl)
        left_ok = idx == 0 or not ql[idx - 1].isalnum()
        right_ok = end == len(ql) or not ql[end].isalnum()
        if left_ok and right_ok:
            return text[:idx] + replacement + text[end:]
        start = idx + 1


def build_expanded_queries(
    query: str, alias_groups: list[list[str]], max_siblings_per_match: int = 4
) -> tuple[list[str], list[dict]]:
    """Build the set of expanded queries for this (query, conversation) pair.

    Returns (variants, match_records):
      variants: list of query strings starting with the original. Each sibling
        replacement produces a new variant. If multiple matches exist, we only
        replace one at a time to avoid combinatorial blowup; the first (longest)
        matched entity is the primary pivot.
      match_records: metadata for inspection/reporting.
    """
    variants: list[str] = [query]
    records: list[dict] = []
    matches = find_alias_matches(query, alias_groups)
    for matched_term, siblings in matches:
        record = {
            "matched_in_query": matched_term,
            "siblings": siblings[:max_siblings_per_match],
        }
        for sib in siblings[:max_siblings_per_match]:
            variant = _replace_first_occurrence(query, matched_term, sib)
            if variant and variant not in variants:
                variants.append(variant)
        records.append(record)
    return variants, records


# ---------------------------------------------------------------------------
# Base arch
# ---------------------------------------------------------------------------


class _AliasExpansionBase(BestshotBase):
    arch_name: str = "alias_expansion"
    per_variant_top_k: int = 10
    run_v2f: bool = False  # if True, run v2f cue-gen on original query
    run_v2f_per_variant: bool = False  # if True, run v2f for each variant too
    include_sibling_probes: bool = True  # retrieve with each sibling alone

    _extractor_cache: dict[int, AliasExtractor] = {}

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        if client is None:
            client = OpenAI(timeout=60.0, max_retries=3)
        super().__init__(store, client)
        self.embedding_cache = AliasEmbeddingCache()
        self.llm_cache = AliasLLMCache()

        # Per-process singleton-ish extractor keyed by store id so we reuse
        # extracted alias groups across repeated arch instantiations.
        key = id(store)
        ext = self._extractor_cache.get(key)
        if ext is None:
            ext = AliasExtractor(client=self.client)
            conv_ids = sorted({s.conversation_id for s in store.segments})
            ext.extract_for_store(store, conv_ids)
            self._extractor_cache[key] = ext
        self.extractor = ext

    def llm_call(self, prompt: str, model: str = MODEL) -> str:
        cached = self.llm_cache.get(model, prompt)
        if cached is not None:
            self.llm_calls += 1
            return cached
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=2000,
                )
                text = response.choices[0].message.content or ""
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

    def embed_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
        cached = self.embedding_cache.get(text)
        if cached is not None:
            self.embed_calls += 1
            return cached
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.embeddings.create(
                    model=EMBED_MODEL, input=[text]
                )
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                self.embedding_cache.put(text, embedding)
                self.embed_calls += 1
                return embedding
            except Exception as e:
                last_exc = e
                time.sleep(1.5 * (attempt + 1))
        print(f"    Embed failed after 3 attempts: {last_exc}", flush=True)
        self.embed_calls += 1
        return np.zeros(1536, dtype=np.float32)

    def _run_v2f(
        self,
        question: str,
        primer_segments: list[Segment],
        conversation_id: str,
        score_map: dict[int, float],
        seg_map: dict[int, Segment],
    ) -> tuple[list[str], list[dict]]:
        """Run the v2f cue-gen pipeline on a question and fold retrievals into
        score_map/seg_map. Returns (cues, outcomes)."""
        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(primer_segments)
        )
        prompt = V2F_PROMPT.format(question=question, context_section=context_section)
        output = self.llm_call(prompt)
        cues = _parse_cues(output)[:2]
        outcomes: list[dict] = []
        for cue in cues:
            cue_emb = self.embed_text(cue)
            res = self.store.search(cue_emb, top_k=10, conversation_id=conversation_id)
            rids: list[int] = []
            for seg, sc in zip(res.segments, res.scores):
                rids.append(seg.index)
                if seg.index not in score_map or sc > score_map[seg.index]:
                    score_map[seg.index] = sc
                if seg.index not in seg_map:
                    seg_map[seg.index] = seg
            outcomes.append(
                {
                    "cue": cue,
                    "retrieved_turn_ids": [seg_map[i].turn_id for i in rids],
                }
            )
        return cues, outcomes

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        alias_groups = self.extractor.get_groups(conversation_id)
        variants, match_records = build_expanded_queries(question, alias_groups)

        score_map: dict[int, float] = {}
        seg_map: dict[int, Segment] = {}

        # --- Retrieve with each query variant (cosine, top-K) ---
        variant_outcomes: list[dict] = []
        # Embed all variants in parallel (typically 1-5 total)
        variant_embs: list[np.ndarray]
        if variants:
            with ThreadPoolExecutor(max_workers=max(1, len(variants))) as pool:
                variant_embs = list(pool.map(self.embed_text, variants))
        else:
            variant_embs = []

        for variant, emb in zip(variants, variant_embs):
            res = self.store.search(
                emb, top_k=self.per_variant_top_k, conversation_id=conversation_id
            )
            rids: list[int] = []
            for seg, sc in zip(res.segments, res.scores):
                rids.append(seg.index)
                if seg.index not in score_map or sc > score_map[seg.index]:
                    score_map[seg.index] = sc
                if seg.index not in seg_map:
                    seg_map[seg.index] = seg
            variant_outcomes.append(
                {
                    "variant": variant,
                    "retrieved_turn_ids": [seg_map[i].turn_id for i in rids],
                }
            )

        # --- Optional: sibling-only probes (retrieve turns that just mention
        # the alias, even if the rest of the question doesn't fit). ---
        sibling_probe_outcomes: list[dict] = []
        if self.include_sibling_probes and match_records:
            sib_texts: list[str] = []
            for rec in match_records:
                for sib in rec["siblings"]:
                    if sib not in sib_texts:
                        sib_texts.append(sib)
            if sib_texts:
                with ThreadPoolExecutor(
                    max_workers=max(1, min(len(sib_texts), 6))
                ) as pool:
                    sib_embs = list(pool.map(self.embed_text, sib_texts))
                for sib_text, sib_emb in zip(sib_texts, sib_embs):
                    res = self.store.search(
                        sib_emb,
                        top_k=self.per_variant_top_k,
                        conversation_id=conversation_id,
                    )
                    rids = []
                    for seg, sc in zip(res.segments, res.scores):
                        rids.append(seg.index)
                        if seg.index not in score_map or sc > score_map[seg.index]:
                            score_map[seg.index] = sc
                        if seg.index not in seg_map:
                            seg_map[seg.index] = seg
                    sibling_probe_outcomes.append(
                        {
                            "sibling": sib_text,
                            "retrieved_turn_ids": [seg_map[i].turn_id for i in rids],
                        }
                    )

        # Sort hop0 (top-10 of original) for v2f primer
        primer_segments: list[Segment] = []
        if variants:
            res0 = self.store.search(
                variant_embs[0], top_k=10, conversation_id=conversation_id
            )
            primer_segments = list(res0.segments)

        # --- Optional: v2f on original query (and per-variant for _full mode) ---
        v2f_cues: list[str] = []
        v2f_outcomes: list[dict] = []
        per_variant_v2f: list[dict] = []
        if self.run_v2f and primer_segments:
            cues, outs = self._run_v2f(
                question, primer_segments, conversation_id, score_map, seg_map
            )
            v2f_cues.extend(cues)
            v2f_outcomes.extend(outs)

        if self.run_v2f_per_variant and primer_segments:
            for v in variants[1:]:  # skip original, already handled above
                cues_v, outs_v = self._run_v2f(
                    v, primer_segments, conversation_id, score_map, seg_map
                )
                per_variant_v2f.append(
                    {"variant": v, "cues": cues_v, "outcomes": outs_v}
                )

        # Rank by max score
        ranked = sorted(score_map.keys(), key=lambda i: score_map[i], reverse=True)
        all_segments = [seg_map[i] for i in ranked]

        metadata = {
            "name": self.arch_name,
            "alias_groups": alias_groups,
            "match_records": match_records,
            "num_variants": len(variants),
            "query_variants": variants,
            "variant_outcomes": variant_outcomes,
            "sibling_probe_outcomes": sibling_probe_outcomes,
            "v2f_cues": v2f_cues,
            "v2f_outcomes": v2f_outcomes,
            "per_variant_v2f": per_variant_v2f,
            "num_probes": (
                len(variants)
                + len(sibling_probe_outcomes)
                + len(v2f_cues)
                + sum(len(p["cues"]) for p in per_variant_v2f)
            ),
        }

        return BestshotResult(segments=all_segments, metadata=metadata)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


class AliasExpandCosine(_AliasExpansionBase):
    """Pure alias expansion on cosine baseline; no cue generation."""

    arch_name = "alias_expand_cosine"
    run_v2f = False
    run_v2f_per_variant = False
    include_sibling_probes = True


class AliasExpandV2fCheap(_AliasExpansionBase):
    """v2f on original query + cosine expansion on alias variants."""

    arch_name = "alias_expand_v2f_cheap"
    run_v2f = True
    run_v2f_per_variant = False
    include_sibling_probes = True


class AliasExpandV2fFull(_AliasExpansionBase):
    """v2f on original + v2f on each alias variant. Expensive but fair stacking
    test (the plan's ``alias_expand_v2f`` variant)."""

    arch_name = "alias_expand_v2f"
    run_v2f = True
    run_v2f_per_variant = True
    include_sibling_probes = True


ARCH_CLASSES: dict[str, type] = {
    "alias_expand_cosine": AliasExpandCosine,
    "alias_expand_v2f_cheap": AliasExpandV2fCheap,
    "alias_expand_v2f": AliasExpandV2fFull,
}
