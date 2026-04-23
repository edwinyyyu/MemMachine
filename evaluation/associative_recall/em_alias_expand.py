"""EventMemory-native alias_expand_v2f.

Reuses `conversation_alias_groups.json` (already extracted on SS side; 15/7/16
groups for LoCoMo 26/30/41). For each query:

  1. Find alias terms in the question (alias_expansion.find_alias_matches).
  2. Build variants by sibling substitution (alias_expansion.build_expanded_queries).
  3. For each variant run em_v2f (primer+v2f cues+per-variant retrieval via
     EventMemory.query).
  4. Add sibling-only probes (each sibling as a standalone EM.query).
  5. Merge all retrievals across variants/probes/v2f-cues by max score per
     turn_id (sum_cosine also available but max tracked for fairness).

If no alias term found in the query, falls back to em_v2f.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory

from em_architectures import (
    EMHit,
    V2F_MODEL,
    V2F_PROMPT,
    _MergedLLMCache,
    _dedupe_by_turn_id,
    _query_em,
    em_v2f,
    format_primer_context,
    parse_v2f_cues,
)
from alias_expansion import (
    build_expanded_queries,
    find_alias_matches,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALIAS_GROUPS_FILE = RESULTS_DIR / "conversation_alias_groups.json"


def load_alias_groups() -> dict[str, list[list[str]]]:
    if not ALIAS_GROUPS_FILE.exists():
        return {}
    with open(ALIAS_GROUPS_FILE) as f:
        data = json.load(f)
    return data.get("groups", {}) or {}


@dataclass
class AliasExpandResult:
    hits: list[EMHit]
    metadata: dict


async def _run_v2f_cues_for(
    memory: EventMemory,
    question: str,
    *,
    K: int,
    llm_cache: _MergedLLMCache,
    openai_client,
) -> tuple[list[str], list[list[EMHit]], bool]:
    """Generate v2f cues for `question`, retrieve per-cue via EM.query.

    Returns (cues, per_cue_hits, cache_hit). The primer hits are NOT
    returned here -- caller handles primer separately so it can dedupe
    across variants.
    """
    primer_hits = _dedupe_by_turn_id(
        await _query_em(memory, question, vector_search_limit=10, expand_context=0)
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text}
        for h in primer_hits
    ]
    context_section = format_primer_context(primer_segments)
    prompt = V2F_PROMPT.format(
        question=question, context_section=context_section
    )

    cached = llm_cache.get(V2F_MODEL, prompt)
    if cached is None:
        if openai_client is None:
            return [], [], False
        resp = await openai_client.chat.completions.create(
            model=V2F_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        llm_cache.put(V2F_MODEL, prompt, text)
        cached = text
        cache_hit = False
    else:
        cache_hit = True

    cues = parse_v2f_cues(cached, max_cues=2)
    per_cue_hits: list[list[EMHit]] = []
    for cue in cues[:2]:
        per_cue_hits.append(
            await _query_em(
                memory, cue, vector_search_limit=K, expand_context=0
            )
        )
    return cues, per_cue_hits, cache_hit


async def em_alias_expand_v2f(
    memory: EventMemory,
    question: str,
    conversation_id: str,
    *,
    K: int,
    alias_groups_by_conv: dict[str, list[list[str]]],
    llm_cache: _MergedLLMCache,
    openai_client=None,
    max_siblings_per_match: int = 4,
    per_variant_top_k: int = 10,
) -> AliasExpandResult:
    """Alias expansion + per-variant v2f, fused by sum_cosine per turn_id.

    If no alias match, falls back to em_v2f (single pass).
    """
    groups = alias_groups_by_conv.get(conversation_id, [])
    matches = find_alias_matches(question, groups) if groups else []

    if not matches:
        # Fallback to plain em_v2f.
        v2f_hits, v2f_meta = await em_v2f(
            memory,
            question,
            K=K,
            llm_cache=llm_cache,
            openai_client=openai_client,
            expand_context=0,
        )
        return AliasExpandResult(
            hits=v2f_hits,
            metadata={
                "alias_matches": [],
                "variants": [question],
                "fallback": "em_v2f",
                "v2f_cues": v2f_meta.get("cues", []),
            },
        )

    variants, match_records = build_expanded_queries(
        question, groups, max_siblings_per_match=max_siblings_per_match
    )

    # Gather all retrieval batches to merge by sum_cosine.
    batches: list[list[EMHit]] = []
    per_variant_cues: list[dict] = []

    # For each variant: primer retrieval + v2f cues + per-cue retrievals.
    for variant in variants:
        # Primer (raw variant retrieval).
        variant_primary = await _query_em(
            memory, variant, vector_search_limit=per_variant_top_k,
            expand_context=0,
        )
        batches.append(variant_primary)

        # V2f cues for this variant.
        cues, cue_batches, cache_hit = await _run_v2f_cues_for(
            memory, variant, K=per_variant_top_k,
            llm_cache=llm_cache, openai_client=openai_client,
        )
        per_variant_cues.append({
            "variant": variant, "cues": cues, "cache_hit": cache_hit
        })
        batches.extend(cue_batches)

    # Sibling-only probes: each sibling alias as its own standalone query.
    sibling_probes: list[str] = []
    for rec in match_records:
        for sib in rec.get("siblings", [])[:max_siblings_per_match]:
            if sib not in sibling_probes and sib != question:
                sibling_probes.append(sib)
    for sib_text in sibling_probes[:8]:
        batches.append(
            await _query_em(
                memory, sib_text, vector_search_limit=per_variant_top_k,
                expand_context=0,
            )
        )

    # Full-K fallback retrieval on the original so we have enough coverage.
    orig_fullk = await _query_em(
        memory, question, vector_search_limit=K, expand_context=0
    )
    batches.append(orig_fullk)

    # Merge by sum of scores per turn_id (one contribution per batch).
    score_sum: dict[int, float] = {}
    representative: dict[int, EMHit] = {}
    for batch in batches:
        seen_in_batch: set[int] = set()
        for h in batch:
            if h.turn_id in seen_in_batch:
                continue
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

    metadata = {
        "alias_matches": [
            {"matched_in_query": r["matched_in_query"], "siblings": r["siblings"]}
            for r in match_records
        ],
        "variants": variants,
        "num_variants": len(variants),
        "num_sibling_probes": len(sibling_probes[:8]),
        "per_variant_cues": per_variant_cues,
    }
    return AliasExpandResult(hits=ranked[:K], metadata=metadata)
