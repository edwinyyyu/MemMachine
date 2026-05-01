"""Active embedding steering via LLM-generated add/subtract phrases.

Novel iterative-probe-refinement mechanism that does explicit vector arithmetic
in embedding space, with LLM-chosen directions (both attract AND repel).

Mechanism:
    probe_0 = normalize(embed(cue))    # starting cue (v2f text or raw query)
    for round in 1..max_rounds:
        retrieved = retrieve_top_K(probe, em)
        llm_output = { add: [phrases...], sub: [phrases...] }
        add_embs = [normalize(embed(p)) for p in add]
        sub_embs = [normalize(embed(p)) for p in sub]
        probe = normalize(probe + alpha*sum(add) - beta*sum(sub))

Distinct from `iterative_query_refinement` (centroid-pull, no LLM semantic
judgment) -- here the LLM picks BOTH add and subtract directions, which
gives us an explicit repulsion primitive against high-scoring non-gold content.

Reuses EM collections ingested for LoCoMo-30 and LME-hard.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import numpy as np
from em_architectures import (
    V2F_MODEL,
    EMHit,
    _dedupe_by_turn_id,
    _MergedLLMCache,
)
from memmachine_server.common.embedder.openai_embedder import OpenAIEmbedder
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory

CACHE_DIR = Path(__file__).resolve().parent / "cache"

STEER_LLM_CACHE = CACHE_DIR / "steer_llm_cache.json"
STEER_EMB_CACHE = CACHE_DIR / "steer_embedding_cache.json"


# --------------------------------------------------------------------------
# Caches
# --------------------------------------------------------------------------


class EmbeddingCache:
    """JSON-backed embedding cache keyed by sha256(model:text)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cache: dict[str, list[float]] = {}
        self._pending: dict[str, list[float]] = {}
        if path.exists():
            try:
                with open(path) as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    @staticmethod
    def _key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()

    def get(self, model: str, text: str) -> list[float] | None:
        return self._cache.get(self._key(model, text))

    def put(self, model: str, text: str, vec: Sequence[float]) -> None:
        k = self._key(model, text)
        self._cache[k] = list(vec)
        self._pending[k] = list(vec)

    def save(self) -> None:
        if not self._pending:
            return
        existing: dict[str, list[float]] = {}
        if self.path.exists():
            try:
                with open(self.path) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing.update(self._pending)
        tmp = self.path.with_suffix(".json.tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.path)
        self._pending.clear()


async def cached_embed(
    embedder: OpenAIEmbedder,
    text: str,
    *,
    emb_cache: EmbeddingCache,
    emb_model: str = "text-embedding-3-small",
) -> list[float]:
    hit = emb_cache.get(emb_model, text)
    if hit is not None:
        return hit
    [vec] = await embedder.search_embed([text])
    emb_cache.put(emb_model, text, vec)
    return list(vec)


# --------------------------------------------------------------------------
# LLM prompt + parsing
# --------------------------------------------------------------------------


STEER_PROMPT = """\
You're steering a semantic retrieval probe toward relevant content and away \
from distractors. The probe is compared to conversation turns via cosine \
similarity.

Original query/cue:
{cue}

Current top-5 retrieved turns (rank: text):
{retrieved_turns}

Your job:
1) ADD phrases: short text snippets that look like what you'd EXPECT a gold \
turn to contain -- concrete first-person chat phrasing, specific vocabulary, \
dates/entities if implied. These pull the probe toward missing-gold content.
2) SUBTRACT phrases: short snippets describing the DISTRACTOR pattern you see \
in current retrievals -- on-topic but not what the query needs. These repel \
the probe from dominant non-gold content.

Output STRICT JSON only (no code fence, no commentary):
{{"add": ["phrase1", "phrase2", "phrase3"], \
"sub": ["phrase1", "phrase2"], \
"reasoning": "one short sentence"}}

Rules:
- 2-3 ADD phrases, 1-3 SUBTRACT phrases.
- Each phrase must be a SHORT specific noun/verb phrase or sentence fragment \
(under 20 words). NO questions.
- Phrases must be CONCRETE; avoid generic words like "information" or "detail".
- If current retrievals already look right, return fewer SUB phrases (even 0)."""


STEER_PROMPT_WEIGHTED = """\
You're steering a semantic retrieval probe toward relevant content and away \
from distractors. The probe is compared to conversation turns via cosine \
similarity.

Original query/cue:
{cue}

Current top-5 retrieved turns (rank: text):
{retrieved_turns}

Your job:
1) ADD phrases with per-phrase magnitude (0.5=mild, 1.0=standard, 2.0=strong).
2) SUBTRACT phrases with per-phrase magnitude likewise.

Output STRICT JSON only:
{{"add": [{{"phrase": "...", "w": 1.0}}, ...], \
"sub": [{{"phrase": "...", "w": 1.0}}, ...], \
"reasoning": "one sentence"}}

Rules:
- 2-3 ADD, 1-3 SUB.
- Each phrase < 20 words, concrete, chat-style, no questions.
- Magnitudes in [0.25, 3.0]."""


def _format_retrieved_snippet(hits: list[EMHit], n: int = 5) -> str:
    lines = []
    for i, h in enumerate(hits[:n], start=1):
        text = (h.text or "").replace("\n", " ")[:240]
        lines.append(f"{i}. [{h.role}] {text}")
    return "\n".join(lines) if lines else "(no turns retrieved yet)"


def _parse_steer_json(response: str) -> dict:
    """Extract JSON from response text, tolerating code fences / trailing text."""
    if not response:
        return {"add": [], "sub": [], "reasoning": ""}
    # Try to find first {...} JSON object.
    m = re.search(r"\{.*\}", response, re.DOTALL)
    raw = m.group(0) if m else response.strip()
    # Strip code fence markers if any.
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(raw)
    except Exception:
        return {"add": [], "sub": [], "reasoning": ""}
    return obj


def _extract_phrases(items) -> list[tuple[str, float]]:
    """Normalize to [(phrase, weight)]. Tolerates strings or dicts."""
    out: list[tuple[str, float]] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if isinstance(it, str):
            p = it.strip()
            if p:
                out.append((p, 1.0))
        elif isinstance(it, dict):
            p = str(it.get("phrase") or it.get("text") or "").strip()
            w = it.get("w", it.get("weight", 1.0))
            try:
                w = float(w)
            except Exception:
                w = 1.0
            w = max(0.0, min(3.0, w))
            if p:
                out.append((p, w))
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
# Direct-vector EM query (bypass EM.query's text embedding step)
# --------------------------------------------------------------------------


async def _query_em_by_vector(
    memory: EventMemory,
    probe: Sequence[float],
    *,
    vector_search_limit: int,
    expand_context: int,
) -> list[EMHit]:
    """Replicate EventMemory.query semantics but with a raw probe vector.

    Returns the same EMHit objects as _query_em, so recall computation
    downstream is identical.
    """
    vs = memory._vector_store_collection  # noqa: SLF001
    seg = memory._segment_store_partition  # noqa: SLF001

    [query_result] = await vs.query(
        query_vectors=[list(probe)],
        limit=vector_search_limit,
        return_vector=False,
        return_properties=True,
    )

    seed_embedding_scores: dict[UUID, float] = {}
    for match in query_result.matches:
        segment_uuid = UUID(str(match.record.properties["_segment_uuid"]))
        if segment_uuid not in seed_embedding_scores:
            seed_embedding_scores[segment_uuid] = match.score
    seed_uuids = list(seed_embedding_scores)

    max_back = expand_context // 3
    max_fwd = expand_context - max_back
    contexts_by_seed = await seg.get_segment_contexts(
        seed_segment_uuids=seed_uuids,
        max_backward_segments=max_back,
        max_forward_segments=max_fwd,
    )

    hits: list[EMHit] = []
    for seed_uuid in seed_uuids:
        if seed_uuid not in contexts_by_seed:
            continue
        score = seed_embedding_scores[seed_uuid]
        for s in contexts_by_seed[seed_uuid]:
            hits.append(
                EMHit(
                    turn_id=int(s.properties.get("turn_id", -1)),
                    score=score,
                    seed_segment_uuid=seed_uuid,
                    role=str(s.properties.get("role", "")),
                    text=s.block.text,
                )
            )
    return hits


# --------------------------------------------------------------------------
# Active steering
# --------------------------------------------------------------------------


@dataclass
class SteerConfig:
    max_rounds: int = 3
    alpha: float = 0.1  # add scale
    beta: float = 0.1  # sub scale
    topk_for_llm: int = 5
    vector_search_limit: int = 50
    expand_context: int = 0
    weighted_mode: bool = False
    use_sub: bool = True
    use_add: bool = True
    # Early stopping: if round adds no novel gold delta in top-K we can break.
    early_stop_recall_delta_eps: float = 0.0


@dataclass
class RoundTrace:
    round_idx: int
    add_phrases: list[tuple[str, float]]
    sub_phrases: list[tuple[str, float]]
    add_magnitude: float  # alpha * ||sum add||
    sub_magnitude: float  # beta * ||sum sub||
    probe_drift: float  # cosine(probe_0, probe_after)
    recall_deltas: dict[str, float]  # optional per-K recall


@dataclass
class SteerResult:
    hits_by_round: list[list[EMHit]]  # top-K snapshot each round
    traces: list[RoundTrace]
    initial_cue_text: str
    final_probe: list[float]


async def active_steer(
    memory: EventMemory,
    *,
    initial_cue_text: str,
    embedder: OpenAIEmbedder,
    openai_client,
    llm_cache: _MergedLLMCache,
    emb_cache: EmbeddingCache,
    config: SteerConfig,
    gold: set[int] | None = None,
    K_budgets: tuple[int, ...] = (20, 50),
) -> SteerResult:
    """Run active steering for up to config.max_rounds.

    Returns the top-K snapshot at each round (round 0 = initial probe),
    plus per-round trace metadata.

    When gold is provided, RoundTrace.recall_deltas gives per-K recall for
    that round's retrieval -- used in reporting only, not for steering.
    """
    # Starting probe from initial cue.
    cue_vec = await cached_embed(embedder, initial_cue_text, emb_cache=emb_cache)
    probe = _normalize(np.array(cue_vec, dtype=np.float64))
    probe_0 = probe.copy()

    hits_by_round: list[list[EMHit]] = []
    traces: list[RoundTrace] = []

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
        RoundTrace(
            round_idx=0,
            add_phrases=[],
            sub_phrases=[],
            add_magnitude=0.0,
            sub_magnitude=0.0,
            probe_drift=1.0,
            recall_deltas=initial_recall,
        )
    )

    prompt_template = STEER_PROMPT_WEIGHTED if config.weighted_mode else STEER_PROMPT

    current_hits = r0_dedup
    for rnd in range(1, config.max_rounds + 1):
        retrieved_str = _format_retrieved_snippet(current_hits, n=config.topk_for_llm)
        prompt = prompt_template.format(
            cue=initial_cue_text,
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

        parsed = _parse_steer_json(cached)
        add_items = _extract_phrases(parsed.get("add", []))
        sub_items = _extract_phrases(parsed.get("sub", []))

        # Apply ablation flags.
        if not config.use_add:
            add_items = []
        if not config.use_sub:
            sub_items = []

        # Embed phrases & aggregate.
        add_sum = np.zeros_like(probe)
        for phrase, w in add_items:
            vec = await cached_embed(embedder, phrase, emb_cache=emb_cache)
            add_sum += w * _normalize(np.array(vec, dtype=np.float64))

        sub_sum = np.zeros_like(probe)
        for phrase, w in sub_items:
            vec = await cached_embed(embedder, phrase, emb_cache=emb_cache)
            sub_sum += w * _normalize(np.array(vec, dtype=np.float64))

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
            RoundTrace(
                round_idx=rnd,
                add_phrases=add_items,
                sub_phrases=sub_items,
                add_magnitude=add_mag,
                sub_magnitude=sub_mag,
                probe_drift=drift,
                recall_deltas=round_recall,
            )
        )

        # Early stop if no delta vs prior round.
        if (
            config.early_stop_recall_delta_eps > 0
            and gold is not None
            and len(traces) >= 2
        ):
            prev_r = traces[-2].recall_deltas.get(f"r@{max(K_budgets)}", 0.0)
            cur_r = traces[-1].recall_deltas.get(f"r@{max(K_budgets)}", 0.0)
            if cur_r - prev_r < config.early_stop_recall_delta_eps:
                break

        current_hits = next_dedup

    return SteerResult(
        hits_by_round=hits_by_round,
        traces=traces,
        initial_cue_text=initial_cue_text,
        final_probe=probe.tolist(),
    )
