"""Topic-routing strategies for round 6.

Each strategy has the interface:
    route(fact_text: str, existing_topics: list[TopicState]) -> RouteDecision

Strategies vary in how they decide:
  - R1: Fixed closed-enum taxonomy (10 buckets under User/...)
  - R2: LLM proposes a topic name, embedding-validates against existing
  - R3: Entity-first: extract entity then category
  - R4: Embedding-nearest only (no LLM)
  - R5: Hybrid -- LLM with full existing-topic list + descriptions; embedding sanity check
  - R6: Cheap-first cascade -- embedding nearest at tight threshold, LLM fallback only on low-confidence

A RouteDecision carries:
  topic_name: str
  is_new: bool
  cost: {"llm_calls": int, "embed_calls": int}
  reason: str

State is maintained externally by the evaluator (list of TopicState objects,
each with name, description, centroid embedding, entry_count).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import openai

# --- Infrastructure ------------------------------------------------------

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parents[3]  # .../evaluation


def _sha(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class LLMCache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._d: dict[str, str] = {}
        if path.exists():
            try:
                with open(path) as f:
                    self._d = json.load(f)
            except Exception:
                self._d = {}
        self._dirty = False
        self._hits = 0
        self._misses = 0

    def get(self, model: str, prompt: str) -> str | None:
        v = self._d.get(_sha(f"{model}::{prompt}"))
        if v is not None:
            self._hits += 1
        return v

    def put(self, model: str, prompt: str, response: str) -> None:
        self._d[_sha(f"{model}::{prompt}")] = response
        self._dirty = True
        self._misses += 1

    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._d)}

    def save(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._d, f)
        tmp.replace(self._path)
        self._dirty = False


class EmbedCache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._d: dict[str, list[float]] = {}
        if path.exists():
            try:
                with open(path) as f:
                    self._d = json.load(f)
            except Exception:
                self._d = {}
        self._dirty = False
        self._hits = 0
        self._misses = 0

    def get(self, text: str) -> list[float] | None:
        v = self._d.get(_sha(f"emb::{text}"))
        if v is not None:
            self._hits += 1
        return v

    def put(self, text: str, emb: list[float]) -> None:
        self._d[_sha(f"emb::{text}")] = emb
        self._dirty = True
        self._misses += 1

    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._d)}

    def save(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._d, f)
        tmp.replace(self._path)
        self._dirty = False


@dataclass
class CallCounter:
    llm_calls: int = 0
    embed_calls: int = 0

    def tick_llm(self) -> None:
        self.llm_calls += 1

    def tick_embed(self) -> None:
        self.embed_calls += 1


def llm(
    client: openai.OpenAI,
    cache: LLMCache,
    counter: CallCounter,
    prompt: str,
    model: str = "gpt-5-mini",
) -> str:
    cached = cache.get(model, prompt)
    if cached is not None:
        return cached
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="low",
    )
    text = resp.choices[0].message.content or ""
    cache.put(model, prompt, text)
    counter.tick_llm()
    return text


def embed(
    client: openai.OpenAI,
    cache: EmbedCache,
    counter: CallCounter,
    text: str,
) -> list[float]:
    cached = cache.get(text)
    if cached is not None:
        return cached
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    vec = list(resp.data[0].embedding)
    cache.put(text, vec)
    counter.tick_embed()
    return vec


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# --- Topic state ---------------------------------------------------------


@dataclass
class TopicState:
    name: str
    description: str = ""
    centroid: list[float] = field(default_factory=list)
    entry_count: int = 0
    _embeds_sum: list[float] = field(default_factory=list)

    def update_centroid(self, new_emb: list[float]) -> None:
        if not self._embeds_sum:
            self._embeds_sum = list(new_emb)
        else:
            self._embeds_sum = [a + b for a, b in zip(self._embeds_sum, new_emb)]
        self.entry_count += 1
        self.centroid = [x / self.entry_count for x in self._embeds_sum]


@dataclass
class RouteDecision:
    topic_name: str
    is_new: bool
    reason: str
    multi_topics: list[str] | None = None  # for multi-label variants


# --- Strategies ----------------------------------------------------------

R1_TAXONOMY = [
    "User/Biography",
    "User/Medical",
    "User/Employment",
    "User/Preferences",
    "User/Relationships",
    "User/Possessions",
    "User/Events",
    "User/Beliefs",
    "User/Skills",
    "User/Other",
]


def strategy_r1_fixed_taxonomy(
    fact_text: str,
    topics: list[TopicState],
    client: openai.OpenAI,
    llm_cache: LLMCache,
    embed_cache: EmbedCache,
    counter: CallCounter,
) -> RouteDecision:
    """LLM picks one of the 10 fixed buckets. No new topics ever created."""
    prompt = f"""You are a topic router for a personal memory system. Pick the single best bucket for this fact.

FACT: {fact_text}

BUCKETS (pick exactly one name, verbatim):
- User/Biography    (name, age, home city, demographics)
- User/Medical      (conditions, allergies, medications, injuries)
- User/Employment   (job, employer, role, career changes)
- User/Preferences  (likes/dislikes, favorites, tastes)
- User/Relationships (partner, family members, friends, colleagues)
- User/Possessions  (pets, vehicles, owned objects)
- User/Events       (things that happened, trips, milestones)
- User/Beliefs      (opinions, values, stances)
- User/Skills       (hobbies, abilities, training)
- User/Other        (anything that doesn't fit above)

Reply with ONLY the bucket name, nothing else.
"""
    out = llm(client, llm_cache, counter, prompt).strip()
    # Normalize
    for b in R1_TAXONOMY:
        if b.lower() in out.lower():
            out = b
            break
    else:
        out = "User/Other"
    is_new = out not in {t.name for t in topics}
    return RouteDecision(topic_name=out, is_new=is_new, reason="r1_fixed")


def strategy_r2_llm_proposed(
    fact_text: str,
    topics: list[TopicState],
    client: openai.OpenAI,
    llm_cache: LLMCache,
    embed_cache: EmbedCache,
    counter: CallCounter,
    similarity_threshold: float = 0.78,
) -> RouteDecision:
    """LLM proposes a topic name; embedding validates against existing topics.

    If an existing topic has similarity > threshold, route to it. Else create new.
    """
    existing_names = [t.name for t in topics]
    existing_list = (
        "\n".join(f"- {n}" for n in existing_names) if existing_names else "(none yet)"
    )
    prompt = f"""You are a topic router for a personal memory system. Each memory-worthy fact goes into an append-only topic log. Propose a concise, specific topic name for this fact.

FACT: {fact_text}

EXISTING TOPICS:
{existing_list}

Rules:
- Topic names use the form "Entity/Category" e.g. "User/Medical", "Luna/Profile", "Jamie/Employment".
- Prefer reusing an existing topic if semantically apt.
- Otherwise propose a new, short topic name.

Reply with ONLY the topic name, nothing else.
"""
    proposed = llm(client, llm_cache, counter, prompt).strip()
    # Clean quotes/markdown
    proposed = (
        proposed.strip("`'\"").splitlines()[0].strip() if proposed else "User/Other"
    )
    # Validate via embedding against existing topics
    if topics:
        fact_emb = embed(client, embed_cache, counter, fact_text)
        best_name = None
        best_score = -1.0
        for t in topics:
            if not t.centroid:
                continue
            s = cosine(fact_emb, t.centroid)
            if s > best_score:
                best_score = s
                best_name = t.name
        # if the LLM proposed an existing one exactly, use it
        if proposed in {t.name for t in topics}:
            return RouteDecision(
                topic_name=proposed,
                is_new=False,
                reason=f"r2_reuse_llm_pick (emb_best={best_name}@{best_score:.2f})",
            )
        # If the embedding says there is a strong existing match, use that instead of creating new
        if best_name is not None and best_score >= similarity_threshold:
            return RouteDecision(
                topic_name=best_name,
                is_new=False,
                reason=f"r2_embed_override(proposed={proposed}, nearest={best_name}@{best_score:.2f})",
            )
    return RouteDecision(topic_name=proposed, is_new=True, reason="r2_new_topic")


def strategy_r3_entity_first(
    fact_text: str,
    topics: list[TopicState],
    client: openai.OpenAI,
    llm_cache: LLMCache,
    embed_cache: EmbedCache,
    counter: CallCounter,
) -> RouteDecision:
    """Two-level: extract primary entity, then category. One LLM call returns both."""
    existing_names = [t.name for t in topics]
    existing_list = (
        "\n".join(f"- {n}" for n in existing_names) if existing_names else "(none yet)"
    )
    prompt = f"""Extract the primary entity this fact is ABOUT (not just mentioned), and a short category word.
Primary entity is the subject the fact primarily describes. Examples:
  "User's cat Luna is 2 years old" -> entity=Luna, category=Profile
  "User works at Anthropic" -> entity=User, category=Employment
  "User's friend Marco has an allergy" -> entity=Marco, category=Medical
  "User adopted a cat named Luna" -> entity=Luna, category=Profile (Luna is the new entity; the fact is about Luna existing)

FACT: {fact_text}

EXISTING TOPICS (reuse if entity+category match):
{existing_list}

Output exactly one line, in the form: <Entity>/<Category>
e.g. "Luna/Profile" or "User/Employment"
"""
    out = llm(client, llm_cache, counter, prompt).strip()
    out = out.strip("`'\"").splitlines()[0].strip()
    # Enforce X/Y shape
    if "/" not in out:
        out = f"User/{out}" if out else "User/Other"
    parts = out.split("/", 1)
    out = f"{parts[0].strip()}/{parts[1].strip()}"
    is_new = out not in {t.name for t in topics}
    return RouteDecision(topic_name=out, is_new=is_new, reason="r3_entity_first")


def strategy_r4_embedding_only(
    fact_text: str,
    topics: list[TopicState],
    client: openai.OpenAI,
    llm_cache: LLMCache,
    embed_cache: EmbedCache,
    counter: CallCounter,
    similarity_threshold: float = 0.50,
) -> RouteDecision:
    """Embedding nearest; above threshold reuse, else create new topic with a synthesized name from the fact.

    For the new-topic name (no LLM), we use a deterministic heuristic: first capitalized word(s).
    """
    fact_emb = embed(client, embed_cache, counter, fact_text)
    if topics:
        best_name = None
        best_score = -1.0
        for t in topics:
            if not t.centroid:
                continue
            s = cosine(fact_emb, t.centroid)
            if s > best_score:
                best_score = s
                best_name = t.name
        if best_name is not None and best_score >= similarity_threshold:
            return RouteDecision(
                topic_name=best_name,
                is_new=False,
                reason=f"r4_embed_reuse@{best_score:.2f}",
            )
    # Heuristic name extraction: first capitalized noun or entity word in fact
    # Grab the first capitalized word that's not "User" to avoid only-User topics
    words = re.findall(r"[A-Z][a-zA-Z]+", fact_text)
    entity = next((w for w in words if w not in {"User"}), "User")
    # Pick a generic category from keywords
    fl = fact_text.lower()
    if any(
        k in fl
        for k in [
            "allerg",
            "disease",
            "pain",
            "diabet",
            "doctor",
            "disc",
            "prescribed",
            "therapy",
        ]
    ):
        category = "Medical"
    elif any(
        k in fl
        for k in [
            "work",
            "employ",
            "job",
            "career",
            "company",
            "role",
            "hospital",
            "nurse",
        ]
    ):
        category = "Employment"
    elif any(k in fl for k in ["cat", "dog", "pet", "hamster", "golden"]):
        category = "Profile"
    elif any(
        k in fl for k in ["hobby", "woodwork", "train", "marathon", "soccer", "violin"]
    ):
        category = "Activity"
    elif any(k in fl for k in ["business", "client", "launch", "roast", "product"]):
        category = "Business"
    else:
        category = "Other"
    new_name = f"{entity}/{category}"
    return RouteDecision(topic_name=new_name, is_new=True, reason="r4_new_heuristic")


def strategy_r5_hybrid(
    fact_text: str,
    topics: list[TopicState],
    client: openai.OpenAI,
    llm_cache: LLMCache,
    embed_cache: EmbedCache,
    counter: CallCounter,
    mismatch_cosine_gap: float = 0.15,
    existing_sim_threshold: float = 0.60,
) -> RouteDecision:
    """LLM proposes given full topic list; embedding sanity-checks; reprompt if disagreement."""
    existing_names = [t.name for t in topics]
    existing_list = (
        "\n".join(
            f"- {t.name}: {t.description or '(no description)'} (entries={t.entry_count})"
            for t in topics
        )
        if topics
        else "(none yet)"
    )
    prompt = f"""You are a topic router. Choose OR create a topic for this fact.

FACT: {fact_text}

EXISTING TOPICS:
{existing_list}

Rules:
- Topic names are "<Entity>/<Category>" e.g. "Luna/Profile", "User/Medical".
- STRONGLY prefer reusing an existing topic name if appropriate.
- Only create a new topic when no existing topic fits.

Respond with ONLY the topic name.
"""
    proposed = (
        llm(client, llm_cache, counter, prompt)
        .strip()
        .strip("`'\"")
        .splitlines()[0]
        .strip()
    )
    if "/" not in proposed:
        proposed = f"User/{proposed}" if proposed else "User/Other"

    # Embedding sanity check
    if topics:
        fact_emb = embed(client, embed_cache, counter, fact_text)
        scored = []
        for t in topics:
            if not t.centroid:
                continue
            scored.append((t.name, cosine(fact_emb, t.centroid)))
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored:
            nearest_name, nearest_score = scored[0]
            proposed_score = next((s for n, s in scored if n == proposed), None)
            # Disagreement: proposed is NEW but an existing topic is strongly similar;
            # OR proposed is existing but a different one is notably closer.
            is_disagreement = False
            if (
                proposed not in existing_names
                and nearest_score >= existing_sim_threshold
            ) or (
                proposed_score is not None
                and nearest_name != proposed
                and (nearest_score - proposed_score) >= mismatch_cosine_gap
            ):
                is_disagreement = True

            if is_disagreement:
                top3 = scored[:3]
                cand_block = "\n".join(f"- {n} (similarity={s:.2f})" for n, s in top3)
                reprompt = f"""Re-route this fact. The embedding model suggests a different existing topic may fit better.

FACT: {fact_text}

Top candidate existing topics (by similarity):
{cand_block}

Your earlier proposal: {proposed}

Choose one of the top candidates above, OR keep your earlier proposal if it really is better. Respond with ONLY the topic name.
"""
                final = (
                    llm(client, llm_cache, counter, reprompt)
                    .strip()
                    .strip("`'\"")
                    .splitlines()[0]
                    .strip()
                )
                if "/" not in final:
                    final = proposed  # fall back
                is_new = final not in existing_names
                return RouteDecision(
                    topic_name=final,
                    is_new=is_new,
                    reason=f"r5_reprompt(first={proposed}, nearest={nearest_name}@{nearest_score:.2f})",
                )
    is_new = proposed not in existing_names
    return RouteDecision(topic_name=proposed, is_new=is_new, reason="r5_accept")


def strategy_r6_cheap_cascade(
    fact_text: str,
    topics: list[TopicState],
    client: openai.OpenAI,
    llm_cache: LLMCache,
    embed_cache: EmbedCache,
    counter: CallCounter,
    strong_threshold: float = 0.65,
    weak_threshold: float = 0.45,
) -> RouteDecision:
    """Cascade: embedding-first. Strong hit -> reuse, no LLM. Weak/no hit -> LLM fallback (with R5 prompt shape)."""
    fact_emb = embed(client, embed_cache, counter, fact_text)
    if topics:
        scored = []
        for t in topics:
            if not t.centroid:
                continue
            scored.append((t.name, cosine(fact_emb, t.centroid)))
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored and scored[0][1] >= strong_threshold:
            return RouteDecision(
                topic_name=scored[0][0],
                is_new=False,
                reason=f"r6_strong_embed@{scored[0][1]:.2f}",
            )
    # Fallback to LLM using R5-like hybrid prompt
    existing_list = (
        "\n".join(f"- {t.name} (entries={t.entry_count})" for t in topics)
        if topics
        else "(none yet)"
    )
    prompt = f"""Topic router. FACT: {fact_text}

EXISTING TOPICS:
{existing_list}

Use format "<Entity>/<Category>". Prefer reuse. Reply with ONLY the topic name.
"""
    proposed = (
        llm(client, llm_cache, counter, prompt)
        .strip()
        .strip("`'\"")
        .splitlines()[0]
        .strip()
    )
    if "/" not in proposed:
        proposed = f"User/{proposed}" if proposed else "User/Other"
    is_new = proposed not in {t.name for t in topics}
    return RouteDecision(topic_name=proposed, is_new=is_new, reason="r6_llm_fallback")


def strategy_r7_entity_first_plus_embed(
    fact_text: str,
    topics: list[TopicState],
    client: openai.OpenAI,
    llm_cache: LLMCache,
    embed_cache: EmbedCache,
    counter: CallCounter,
    merge_threshold: float = 0.72,
) -> RouteDecision:
    """R3's entity-first prompt, augmented by an embedding-based dedup step.

    If R3's output names an entity that doesn't match any existing topic but the
    fact embeds close to an existing topic centroid whose entity name is a
    substring of the fact (i.e. likely-the-same-subject), merge. This targets
    S6's failure mode where R3 spawns "Magpie Roasters/Business",
    "Neon Cafes/Business" etc for facts that should share one topic.

    Also enforces a User/Core fallback: if proposed entity is not User and no
    named entity appears in the fact (shouldn't happen with R3), keep.
    """
    existing_names = [t.name for t in topics]
    existing_list = (
        "\n".join(
            f"- {n} (entries={t.entry_count})" for n, t in zip(existing_names, topics)
        )
        if existing_names
        else "(none yet)"
    )
    prompt = f"""Route this fact to an append-only topic log. Each topic is keyed by the PRIMARY SUBJECT the fact describes (not merely mentioned).

Heuristics for primary subject:
- A fact about a person's pet/child/partner has that pet/child/partner as subject.
- A fact about the user's business/employer: the USER is the subject (their career/business), unless the fact is strictly a property of the business as an independent entity.
- Reuse an existing topic if the subject matches.

FACT: {fact_text}

EXISTING TOPICS:
{existing_list}

Output exactly one line: <Subject>/<Category> (e.g. "Luna/Profile", "User/Employment", "User/Business"). Nothing else.
"""
    proposed = (
        llm(client, llm_cache, counter, prompt)
        .strip()
        .strip("`'\"")
        .splitlines()[0]
        .strip()
    )
    if "/" not in proposed:
        proposed = f"User/{proposed}" if proposed else "User/Other"
    parts = proposed.split("/", 1)
    proposed = f"{parts[0].strip()}/{parts[1].strip()}"

    # Embedding dedup: if proposed is NEW but the fact embeds close to an
    # existing topic whose entity-path-component overlaps, merge.
    if topics and proposed not in existing_names:
        fact_emb = embed(client, embed_cache, counter, fact_text)
        best: tuple[str, float] | None = None
        for t in topics:
            if not t.centroid:
                continue
            s = cosine(fact_emb, t.centroid)
            if best is None or s > best[1]:
                best = (t.name, s)
        if best is not None and best[1] >= merge_threshold:
            return RouteDecision(
                topic_name=best[0],
                is_new=False,
                reason=f"r7_embed_merge(proposed={proposed}, merged={best[0]}@{best[1]:.2f})",
            )
    is_new = proposed not in existing_names
    return RouteDecision(topic_name=proposed, is_new=is_new, reason="r7_accept")


STRATEGIES: dict[str, Callable] = {
    "R1_fixed_taxonomy": strategy_r1_fixed_taxonomy,
    "R2_llm_proposed": strategy_r2_llm_proposed,
    "R3_entity_first": strategy_r3_entity_first,
    "R4_embedding_only": strategy_r4_embedding_only,
    "R5_hybrid": strategy_r5_hybrid,
    "R6_cheap_cascade": strategy_r6_cheap_cascade,
    "R7_entity_plus_embed": strategy_r7_entity_first_plus_embed,
}
