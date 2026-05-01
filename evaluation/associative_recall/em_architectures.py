"""EventMemory-backed retrieval architectures for LoCoMo-30 re-evaluation.

Each architecture takes an open `EventMemory` and a question, and returns
an ordered list of `(turn_id, score, seed_segment_uuid)` triples.

The architectures re-use the existing SegmentStore-era LLM/embedding caches
where possible (e.g. v2f cue text is deterministic for a given question), and
delegate retrieval entirely to `EventMemory.query(...)`.

Architectures implemented here:
  - em_cosine_baseline  : raw question -> EventMemory.query, expand=0
  - em_cosine_expand_6  : raw question -> EventMemory.query, expand=6
  - em_v2f              : raw-query hit + 2 v2f cues, each -> EventMemory.query,
                          merge by max score per segment (dedup by segment uuid)
  - em_v2f_expand_6     : same, with expand_context=6

No framework files are modified. No LLM calls are made unless the bestshot
LLM cache does not contain the v2f prompt (cache miss); in that case we
fall back to a fresh OpenAI call and write the response back into a
DEDICATED cache `em_v2f_llm_cache.json` so we never poison other
specialists' caches.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory

CACHE_DIR = Path(__file__).resolve().parent / "cache"
BESTSHOT_LLM_CACHE = CACHE_DIR / "bestshot_llm_cache.json"
TYPE_ENUM_LLM_CACHE = CACHE_DIR / "type_enum_llm_cache.json"
EM_V2F_LLM_CACHE = CACHE_DIR / "em_v2f_llm_cache.json"
EM_CUE_LLM_CACHE = CACHE_DIR / "em_cue_llm_cache.json"

V2F_MODEL = "gpt-5-mini"


TYPE_ENUMERATED_PROMPT = """\
Generate cues to find scattered constraints/details in a conversation. Each cue \
should mimic how someone would ACTUALLY phrase that type of information in chat.

Question: {question}

RETRIEVED SO FAR:
{context_section}

Generate ONE cue per type below. Use casual first-person chat register. Use \
deictic pronouns (she, he, they) NOT named entities. No quotes around phrases.

[ARRIVAL]: when someone says they arrived/showed up somewhere
[PREFERENCE]: when someone expresses a like/dislike
[CONFLICT]: when a disagreement or issue is discussed
[UPDATE]: informal updates like "oh I forgot to mention" or "just got a message"
[RESOLUTION]: resolutions like "we cleared the air" or "actually it's fine now"
[AFTERTHOUGHT]: casual additions like "wait one more thing" or "btw"
[PHYSICAL]: spatial/physical details like seating, location, position

Format:
CUE: <casual chat text for this type>
(7 cues total, one per type)"""


# type_enumerated uses its own _format_segments (max_chars=250, max_items=12)
# in associative_recall.py. Our format_primer_context matches those defaults.
# type_enumerated's _build_context_section wraps with "RETRIEVED CONVERSATION
# EXCERPTS SO FAR:\n" too, identical to best_shot's.


# Exact V2F prompt text from best_shot.py (MetaV2f) — keep byte-equivalent so
# hashing hits the bestshot cache.
V2F_PROMPT = """\
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

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


def _sha(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()


class _MergedLLMCache:
    """Read from multiple sources, write only to a dedicated file."""

    def __init__(self, reader_paths: list[Path], writer_path: Path) -> None:
        self._cache: dict[str, str] = {}
        for p in reader_paths:
            if p.exists():
                try:
                    with open(p) as f:
                        self._cache.update(json.load(f))
                except Exception:
                    pass
        self._writer_path = writer_path
        self._pending: dict[str, str] = {}

    def get(self, model: str, prompt: str) -> str | None:
        key = _sha(model, prompt)
        return self._cache.get(key)

    def put(self, model: str, prompt: str, response: str) -> None:
        key = _sha(model, prompt)
        self._cache[key] = response
        self._pending[key] = response

    def save(self) -> None:
        if not self._pending:
            return
        existing: dict[str, str] = {}
        if self._writer_path.exists():
            try:
                with open(self._writer_path) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing.update(self._pending)
        tmp = self._writer_path.with_suffix(".json.tmp")
        self._writer_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self._writer_path)
        self._pending.clear()


CUE_RE = re.compile(r"^\s*CUE\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def parse_v2f_cues(response: str, max_cues: int = 2) -> list[str]:
    cues: list[str] = []
    for m in CUE_RE.finditer(response):
        cue = m.group(1).strip().strip('"').strip()
        if cue:
            cues.append(cue)
        if len(cues) >= max_cues:
            break
    return cues


def format_primer_context(
    segments: list,
    *,
    max_items: int = 12,
    max_len: int = 250,
) -> str:
    """Replicate best_shot._format_segments context style so prompt text
    matches existing cache entries byte-for-byte when possible.

    Input `segments` is a list of dicts with keys turn_id, role, text.
    """
    if not segments:
        return (
            "No conversation excerpts retrieved yet. Generate cues based on "
            "what you'd expect to find in a conversation about this topic."
        )
    # best_shot sorts by turn_id, displays first 12.
    sorted_segs = sorted(segments, key=lambda s: s["turn_id"])
    lines = []
    for s in sorted_segs[:max_items]:
        lines.append(f"[Turn {s['turn_id']}, {s['role']}]: {s['text'][:max_len]}")
    return "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + "\n".join(lines)


@dataclass
class EMHit:
    turn_id: int
    score: float
    seed_segment_uuid: UUID
    role: str
    text: str


async def _query_em(
    memory: EventMemory,
    text: str,
    *,
    vector_search_limit: int,
    expand_context: int,
) -> list[EMHit]:
    qr = await memory.query(
        query=text,
        vector_search_limit=vector_search_limit,
        expand_context=expand_context,
    )
    hits: list[EMHit] = []
    for sc in qr.scored_segment_contexts:
        # Seed segment is the one anchoring the match. For turn_id purposes
        # we use the seed segment's turn_id when expand_context=0; with
        # expand_context>0 we still surface the seed as the "primary"
        # retrieved turn for recall counting, and the expanded segments
        # as additional turns.
        for seg in sc.segments:
            hits.append(
                EMHit(
                    turn_id=int(seg.properties.get("turn_id", -1)),
                    score=sc.score,
                    seed_segment_uuid=sc.seed_segment_uuid,
                    role=str(seg.properties.get("role", "")),
                    text=seg.block.text,
                )
            )
    return hits


def _dedupe_by_turn_id(hits: list[EMHit]) -> list[EMHit]:
    """Keep first occurrence per turn_id (best-ranked)."""
    seen: set[int] = set()
    out: list[EMHit] = []
    for h in hits:
        if h.turn_id in seen:
            continue
        seen.add(h.turn_id)
        out.append(h)
    return out


def _merge_by_max_score(batches: list[list[EMHit]]) -> list[EMHit]:
    """Merge multiple ranked result lists into one, deduped by turn_id,
    score = max across batches."""
    best: dict[int, EMHit] = {}
    for batch in batches:
        for h in batch:
            prev = best.get(h.turn_id)
            if prev is None or h.score > prev.score:
                best[h.turn_id] = h
    return sorted(best.values(), key=lambda h: -h.score)


# --------------------------------------------------------------------------
# Architectures
# --------------------------------------------------------------------------


async def em_cosine(
    memory: EventMemory,
    question: str,
    *,
    K: int,
    expand_context: int = 0,
) -> list[EMHit]:
    # Deduplicate by turn_id and keep top-K turn-level hits.
    hits = await _query_em(
        memory, question, vector_search_limit=K, expand_context=expand_context
    )
    return _dedupe_by_turn_id(hits)[:K]


async def em_v2f(
    memory: EventMemory,
    question: str,
    *,
    K: int,
    llm_cache: _MergedLLMCache,
    openai_client=None,
    expand_context: int = 0,
) -> tuple[list[EMHit], dict]:
    """V2F: raw-query primer + 2 v2f cues, merged by max score per turn_id.

    Primer retrieval uses EventMemory.query with expand=0 (matches MetaV2f's
    top-10 hop0). Context is built from its segments to preserve the exact
    prompt text used by best_shot.MetaV2f, which lets us hit the existing
    bestshot_llm_cache.
    """
    # Hop 0: raw-query retrieval with K=10 (same as best_shot MetaV2f).
    primer_hits = _dedupe_by_turn_id(
        await _query_em(memory, question, vector_search_limit=10, expand_context=0)
    )[:10]
    primer_segments = [
        {
            "turn_id": h.turn_id,
            "role": h.role,
            "text": h.text,
        }
        for h in primer_hits
    ]
    context_section = format_primer_context(primer_segments)
    prompt = V2F_PROMPT.format(question=question, context_section=context_section)

    cached = llm_cache.get(V2F_MODEL, prompt)
    if cached is None:
        if openai_client is None:
            # No client: skip cue generation, just use primer.
            cues = []
        else:
            resp = await openai_client.chat.completions.create(
                model=V2F_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
            llm_cache.put(V2F_MODEL, prompt, text)
            cached = text
            cues = parse_v2f_cues(text, max_cues=2)
    else:
        cues = parse_v2f_cues(cached, max_cues=2)

    # Per-cue retrieval.
    cue_hits = []
    for cue in cues[:2]:
        cue_hits.append(
            await _query_em(
                memory, cue, vector_search_limit=K, expand_context=expand_context
            )
        )

    # Merge: primer (expand=0) + each cue (expand=expand_context),
    # dedup by turn_id, rank by max score.
    primer_for_merge = await _query_em(
        memory, question, vector_search_limit=K, expand_context=expand_context
    )
    merged = _merge_by_max_score([primer_for_merge, *cue_hits])
    return merged[:K], {"cues": cues, "cache_hit": cached is not None}


# ---------------------------------------------------------------------------
# type_enumerated cue generation (for em_ens_2)
# ---------------------------------------------------------------------------


TYPE_ENUM_CUE_RE = re.compile(
    r"^\s*(?:\[?[A-Z]+\]?\s*[:\-]\s*)?CUE\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def parse_type_enum_cues(response: str, max_cues: int = 7) -> list[str]:
    """Parse type_enumerated cues. The existing type_enumerated.py _parse_cues
    accepts lines beginning with CUE: (possibly prefixed by a [TYPE]: tag)."""
    cues: list[str] = []
    for m in TYPE_ENUM_CUE_RE.finditer(response):
        cue = m.group(1).strip().strip('"').strip()
        if cue:
            cues.append(cue)
        if len(cues) >= max_cues:
            break
    return cues


async def em_ens_2(
    memory: EventMemory,
    question: str,
    *,
    K: int,
    v2f_cache: _MergedLLMCache,
    type_enum_cache: _MergedLLMCache,
    openai_client=None,
    expand_context: int = 0,
) -> tuple[list[EMHit], dict]:
    """Ensemble: v2f's 2 cues + type_enumerated's 7 cues, all via EM.query,
    merged by sum of scores per turn_id (sum_cosine)."""
    primer_hits = _dedupe_by_turn_id(
        await _query_em(memory, question, vector_search_limit=10, expand_context=0)
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text} for h in primer_hits
    ]
    context_section = format_primer_context(primer_segments)

    # v2f cues
    v2f_prompt = V2F_PROMPT.format(question=question, context_section=context_section)
    v2f_resp = v2f_cache.get(V2F_MODEL, v2f_prompt)
    v2f_cache_hit = v2f_resp is not None
    if v2f_resp is None and openai_client is not None:
        r = await openai_client.chat.completions.create(
            model=V2F_MODEL,
            messages=[{"role": "user", "content": v2f_prompt}],
        )
        v2f_resp = r.choices[0].message.content or ""
        v2f_cache.put(V2F_MODEL, v2f_prompt, v2f_resp)
    v2f_cues = parse_v2f_cues(v2f_resp or "", max_cues=2)

    # type_enumerated cues (reuse the same context_section)
    te_prompt = TYPE_ENUMERATED_PROMPT.format(
        question=question, context_section=context_section
    )
    te_resp = type_enum_cache.get(V2F_MODEL, te_prompt)
    te_cache_hit = te_resp is not None
    if te_resp is None and openai_client is not None:
        r = await openai_client.chat.completions.create(
            model=V2F_MODEL,
            messages=[{"role": "user", "content": te_prompt}],
        )
        te_resp = r.choices[0].message.content or ""
        type_enum_cache.put(V2F_MODEL, te_prompt, te_resp)
    te_cues = parse_type_enum_cues(te_resp or "", max_cues=7)

    # Run all cues + primer.
    batches = [
        await _query_em(
            memory, question, vector_search_limit=K, expand_context=expand_context
        )
    ]
    for cue in v2f_cues[:2] + te_cues[:7]:
        batches.append(
            await _query_em(
                memory, cue, vector_search_limit=K, expand_context=expand_context
            )
        )

    # Merge by sum of scores (sum_cosine).
    score_sum: dict[int, float] = {}
    representative: dict[int, EMHit] = {}
    for batch in batches:
        seen_in_batch: set[int] = set()
        for h in batch:
            if h.turn_id in seen_in_batch:
                continue  # one contribution per turn_id per batch
            seen_in_batch.add(h.turn_id)
            score_sum[h.turn_id] = score_sum.get(h.turn_id, 0.0) + h.score
            if h.turn_id not in representative:
                representative[h.turn_id] = h
    ranked = sorted(
        [
            EMHit(
                turn_id=tid,
                score=score_sum[tid],
                seed_segment_uuid=representative[tid].seed_segment_uuid,
                role=representative[tid].role,
                text=representative[tid].text,
            )
            for tid in score_sum
        ],
        key=lambda h: -h.score,
    )
    return ranked[:K], {
        "v2f_cues": v2f_cues,
        "type_enum_cues": te_cues,
        "v2f_cache_hit": v2f_cache_hit,
        "te_cache_hit": te_cache_hit,
    }
