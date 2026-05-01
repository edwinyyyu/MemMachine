"""Active embedding steering v2 -- evidence-grounded.

V2 fixes the phrase-discipline issue in v1 where the LLM was inventing
distractor concepts and fabricating specific details for ADD. V2 grounds both
operations in actual retrieved evidence:

SUBTRACT: LLM classifies retrieved top-K turns as gold-like vs distractor,
outputs turn INDICES. The subtract vector is the sum of embeddings of those
actual turn texts (not invented opposite concepts).

ADD: LLM generates 2-3 short phrases grounded in (a) query vocabulary or
(b) extracted phrases from gold-likely retrieved turns. Forbidden: fabricating
specific dates/names/titles/numbers not present in query or evidence.

Probe update (unchanged from v1):
    probe = normalize(probe + alpha * sum(add_embs) - beta * sum(sub_turn_embs))

Reuses the direct-vector EM query + EMHit machinery from v1.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
from active_steering import (
    EmbeddingCache,
    _query_em_by_vector,
    cached_embed,
)
from em_architectures import (
    V2F_MODEL,
    EMHit,
    _dedupe_by_turn_id,
    _MergedLLMCache,
)
from memmachine_server.common.embedder.openai_embedder import OpenAIEmbedder
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory

# --------------------------------------------------------------------------
# V2 prompt (evidence-grounded)
# --------------------------------------------------------------------------


STEER_V2_PROMPT = """\
You are steering a semantic retrieval probe. You'll classify current \
retrievals and extract grounded phrases.

Query: {query}
Original cue: {cue}

Current top-{topk} retrieved turns:
{retrieved_turns}

Task:
1. CLASSIFY: which of these turns are DISTRACTORS (topically similar but not \
answering the query)? List their indices.
2. GOLD-LIKELY: which indices (if any) look like they ARE answering the query?
3. EXTRACT ADD phrases: 2-3 short phrases using (a) vocabulary from the \
query, or (b) specific words/fragments copied from the turns at \
GOLD-LIKELY indices. DO NOT fabricate specific details (names, dates, \
titles, numbers) that aren't in the query or these turns.

Output STRICT JSON (no code fence, no commentary):
{{"distractor_indices": [idx, ...], \
"gold_likely_indices": [idx, ...], \
"add_phrases": ["phrase1", "phrase2", "phrase3"], \
"reasoning": "one short sentence"}}

Rules:
- Indices are 0-based, only from the list above.
- Each ADD phrase < 20 words, concrete, no questions.
- If nothing looks like a distractor, return [] for distractor_indices.
- If you can't find a gold-likely turn, use only query vocabulary for ADD."""


def _format_retrieved_snippet_indexed(hits: list[EMHit], n: int = 5) -> str:
    lines = []
    for i, h in enumerate(hits[:n]):
        text = (h.text or "").replace("\n", " ")[:280]
        lines.append(f"[{i}] ({h.role}) {text}")
    return "\n".join(lines) if lines else "(no turns retrieved yet)"


def _parse_v2_json(response: str) -> dict:
    """Extract JSON from response text, tolerating code fences / trailing text."""
    empty = {
        "distractor_indices": [],
        "gold_likely_indices": [],
        "add_phrases": [],
        "reasoning": "",
    }
    if not response:
        return empty
    m = re.search(r"\{.*\}", response, re.DOTALL)
    raw = m.group(0) if m else response.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(raw)
    except Exception:
        return empty
    # Normalize.
    out = dict(empty)
    try:
        out["distractor_indices"] = [
            int(x)
            for x in obj.get("distractor_indices", [])
            if isinstance(x, (int, float))
        ]
    except Exception:
        out["distractor_indices"] = []
    try:
        out["gold_likely_indices"] = [
            int(x)
            for x in obj.get("gold_likely_indices", [])
            if isinstance(x, (int, float))
        ]
    except Exception:
        out["gold_likely_indices"] = []
    adds = obj.get("add_phrases", [])
    if isinstance(adds, list):
        out["add_phrases"] = [
            str(p).strip() for p in adds if isinstance(p, str) and p.strip()
        ]
    out["reasoning"] = str(obj.get("reasoning", ""))
    return out


# --------------------------------------------------------------------------
# Vector helpers
# --------------------------------------------------------------------------


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# --------------------------------------------------------------------------
# Active steering v2
# --------------------------------------------------------------------------


@dataclass
class SteerV2Config:
    max_rounds: int = 3
    alpha: float = 0.1  # add scale
    beta: float = 0.1  # sub scale
    topk_for_llm: int = 5
    vector_search_limit: int = 50
    expand_context: int = 0
    use_sub: bool = True
    use_add: bool = True


@dataclass
class RoundTraceV2:
    round_idx: int
    add_phrases: list[str]
    distractor_indices: list[int]
    gold_likely_indices: list[int]
    # Text previews of what we actually subtracted/added (for diagnostics).
    subtracted_texts: list[str]
    add_magnitude: float
    sub_magnitude: float
    probe_drift: float  # cosine(probe_0, probe_after)
    recall_deltas: dict[str, float]
    reasoning: str


@dataclass
class SteerV2Result:
    hits_by_round: list[list[EMHit]]
    traces: list[RoundTraceV2]
    initial_cue_text: str
    query_text: str
    final_probe: list[float]


async def active_steer_v2(
    memory: EventMemory,
    *,
    query_text: str,
    initial_cue_text: str,
    embedder: OpenAIEmbedder,
    openai_client,
    llm_cache: _MergedLLMCache,
    emb_cache: EmbeddingCache,
    config: SteerV2Config,
    gold: set[int] | None = None,
    K_budgets: tuple[int, ...] = (20, 50),
) -> SteerV2Result:
    """Run v2 evidence-grounded active steering for up to config.max_rounds."""
    # Starting probe from initial cue.
    cue_vec = await cached_embed(embedder, initial_cue_text, emb_cache=emb_cache)
    probe = _normalize(np.array(cue_vec, dtype=np.float64))
    probe_0 = probe.copy()

    hits_by_round: list[list[EMHit]] = []
    traces: list[RoundTraceV2] = []

    # Round 0: initial retrieve.
    r0_hits = await _query_em_by_vector(
        memory,
        probe,
        vector_search_limit=config.vector_search_limit,
        expand_context=config.expand_context,
    )
    r0_dedup = _dedupe_by_turn_id(r0_hits)
    hits_by_round.append(r0_dedup)

    initial_recall: dict[str, float] = {}
    if gold is not None:
        for K in K_budgets:
            r = {h.turn_id for h in r0_dedup[:K]}
            initial_recall[f"r@{K}"] = len(r & gold) / len(gold) if gold else 1.0
    traces.append(
        RoundTraceV2(
            round_idx=0,
            add_phrases=[],
            distractor_indices=[],
            gold_likely_indices=[],
            subtracted_texts=[],
            add_magnitude=0.0,
            sub_magnitude=0.0,
            probe_drift=1.0,
            recall_deltas=initial_recall,
            reasoning="",
        )
    )

    current_hits = r0_dedup
    for rnd in range(1, config.max_rounds + 1):
        topk_hits = current_hits[: config.topk_for_llm]
        retrieved_str = _format_retrieved_snippet_indexed(
            topk_hits, n=config.topk_for_llm
        )
        prompt = STEER_V2_PROMPT.format(
            query=query_text,
            cue=initial_cue_text,
            topk=config.topk_for_llm,
            retrieved_turns=retrieved_str,
        )

        cached = llm_cache.get(V2F_MODEL, prompt)
        if cached is None:
            resp = await openai_client.chat.completions.create(
                model=V2F_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            cached = resp.choices[0].message.content or ""
            llm_cache.put(V2F_MODEL, prompt, cached)

        parsed = _parse_v2_json(cached)
        distractor_idx = parsed["distractor_indices"]
        gold_likely_idx = parsed["gold_likely_indices"]
        add_phrases = parsed["add_phrases"]

        # Filter indices to valid range.
        distractor_idx = [i for i in distractor_idx if 0 <= i < len(topk_hits)]
        gold_likely_idx = [i for i in gold_likely_idx if 0 <= i < len(topk_hits)]

        # Apply ablation flags.
        if not config.use_add:
            add_phrases = []
        if not config.use_sub:
            distractor_idx = []

        # Compute sub vector from actual distractor turn embeddings.
        # We embed the turn text directly (not the seed segment vector) because
        # text-embedding-3-small on the raw text is the natural dual of the
        # probe space, and the underlying EM indexing embeds segment text.
        subtracted_texts: list[str] = []
        sub_sum = np.zeros_like(probe)
        for i in distractor_idx:
            turn_text = topk_hits[i].text or ""
            if not turn_text.strip():
                continue
            # Truncate to keep embedding cache keys stable & reasonable size.
            truncated = turn_text[:2000]
            subtracted_texts.append(truncated[:200])
            vec = await cached_embed(embedder, truncated, emb_cache=emb_cache)
            sub_sum += _normalize(np.array(vec, dtype=np.float64))

        # Compute add vector from LLM-generated (but grounded) phrases.
        add_sum = np.zeros_like(probe)
        for phrase in add_phrases:
            vec = await cached_embed(embedder, phrase, emb_cache=emb_cache)
            add_sum += _normalize(np.array(vec, dtype=np.float64))

        add_mag = float(config.alpha * np.linalg.norm(add_sum))
        sub_mag = float(config.beta * np.linalg.norm(sub_sum))

        new_probe = probe + config.alpha * add_sum - config.beta * sub_sum
        new_probe = _normalize(new_probe)

        drift = _cosine(new_probe, probe_0)
        probe = new_probe

        # Retrieve with new probe.
        next_hits = await _query_em_by_vector(
            memory,
            probe,
            vector_search_limit=config.vector_search_limit,
            expand_context=config.expand_context,
        )
        next_dedup = _dedupe_by_turn_id(next_hits)
        hits_by_round.append(next_dedup)

        round_recall: dict[str, float] = {}
        if gold is not None:
            for K in K_budgets:
                r = {h.turn_id for h in next_dedup[:K]}
                round_recall[f"r@{K}"] = len(r & gold) / len(gold) if gold else 1.0

        traces.append(
            RoundTraceV2(
                round_idx=rnd,
                add_phrases=list(add_phrases),
                distractor_indices=list(distractor_idx),
                gold_likely_indices=list(gold_likely_idx),
                subtracted_texts=subtracted_texts,
                add_magnitude=add_mag,
                sub_magnitude=sub_mag,
                probe_drift=drift,
                recall_deltas=round_recall,
                reasoning=parsed.get("reasoning", ""),
            )
        )

        # Early stop if LLM flags nothing actionable.
        if not distractor_idx and not add_phrases:
            break

        current_hits = next_dedup

    return SteerV2Result(
        hits_by_round=hits_by_round,
        traces=traces,
        initial_cue_text=initial_cue_text,
        query_text=query_text,
        final_probe=probe.tolist(),
    )
