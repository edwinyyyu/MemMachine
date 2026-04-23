"""Critical-info vector store.

A small subset of very-important turns (medications, allergies, family sizes,
key commitments, specific dates, etc.) gets flagged at ingestion time. These
turns go into a SEPARATE vector store with a lower retrieval threshold / score
bonus. Additional alt-keys may be generated per critical turn, all pointing
back to the same original turn.

This differs from `ingest_llm_altkeys.py` (prompt v3): there, alt-keys were
merged into the main pool via max-over-keys scoring and displaced clean v2f
retrievals. Here, critical items form a disjoint path that only surfaces items
that are critical anyway.

Pipeline:
  1. LLM classification: for each turn, decide SKIP vs CRITICAL; if critical,
     emit 3 short focused alt-keys.
  2. Build a SeparatePoolStore keyed by (critical_alt_idx -> parent_turn_idx).
  3. At query time: main cosine -> top-K. Critical cosine -> top-M. Merge by
     parent_turn_idx, breaking ties toward main.

Exports:
  - CriticalInfoGenerator (LLM classifier + alt-key emitter, threaded + cached)
  - CriticalTurnDecision (raw output)
  - CriticalInfoStore (separate pool of critical-turn embeddings)
  - merge_with_main (merge strategies)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterable

import numpy as np
from openai import OpenAI

from associative_recall import EMBED_MODEL, Segment, SegmentStore
from best_shot import BestshotEmbeddingCache, BestshotLLMCache


# ---------------------------------------------------------------------------
# Classification prompt — strict "critical info" filter
# ---------------------------------------------------------------------------
CRITICAL_PROMPT_V1 = """\
You are tagging conversation turns at ingestion time for a small critical-info \
memory store. ONLY tag a turn if it contains an enduring fact that would be \
catastrophic to miss later. Examples:
- Medication names, dosages, allergies
- Health conditions, diagnoses
- Numbers of family members, names and relationships
- Specific commitments with deadlines or people
- Specific addresses, account numbers, IDs
- Important preferences ("I never eat X")
- Critical decisions that set future constraints

Do NOT tag turns with:
- Casual chitchat, opinions, speculation
- Non-specific statements ("I like coffee")
- Narrative commentary
- Work context unless it contains a specific commitment or fact

Turn ({role}): {text}

If critical, output 3 alt-keys — short focused rephrasings that surface the \
critical fact from different query angles. Format:
ALT: {{key 1}}
ALT: {{key 2}}
ALT: {{key 3}}

If not critical, output exactly:
SKIP"""


CRITICAL_PROMPT_V2 = """\
You are tagging conversation turns at ingestion time for a small CRITICAL-INFO \
memory store. This store is reserved for a TINY fraction (<10%) of turns that \
contain enduring facts whose loss would be catastrophic. Default to SKIP.

Tag as CRITICAL ONLY if the turn contains at least one of:
- Specific medication, dosage, allergy, diagnosis, or condition
- Specific number/names of family members, pets, or close relationships
- Specific commitment with a deadline, person, amount, or place
  (e.g. "I'll meet Dr. Chen at 3pm Friday", "I'm paying $500 by April 20")
- Specific address, account number, ID, or credential
- Hard preference/prohibition ("I never drink alcohol", "I can't eat shellfish")
- Critical decision that sets a future constraint (e.g. "I've decided to move \
to Toronto", "we cancelled the wedding")

Do NOT tag (these should SKIP):
- Casual chitchat, opinions, speculation, feelings
- Generic statements ("I like coffee", "I enjoy running")
- Narrative backstory or past events without an enduring constraint
- Plans under discussion that haven't been committed to
- Work context unless it contains a specific commitment or fact
- Repeated re-mention of a previously-stated fact
- Questions, greetings, acknowledgements
- Hobbies, interests, general preferences without a prohibition
- Any turn under ~10 tokens unless it states a hard number/name/date

Turn ({role}): {text}

If CRITICAL, output 3 short alt-keys (5-15 words each) that surface the \
critical fact from different query angles. Format EXACTLY:
ALT: <key 1>
ALT: <key 2>
ALT: <key 3>

If not critical, output EXACTLY:
SKIP"""


CRITICAL_PROMPT_V3 = """\
You are tagging conversation turns for a VERY SMALL critical-info memory store. \
Budget: at most 5% of turns may be tagged. Default to SKIP. The bar is whether \
losing this fact would cause a clear real-world harm (missed medication, \
allergic reaction, missed appointment, wrong name, wrong number, broken \
commitment, safety failure).

Tag as CRITICAL only if the turn names ALL of:
(a) a specific entity (a medication name + dose, a specific allergy, a named \
person/relationship with role, a specific date/time/address/account, a hard \
prohibition)
(b) an enduring property (persistent condition, permanent relationship, future-\
binding commitment) — not a one-off or a discussion in progress
(c) the speaker is STATING the fact (not asking about it, not hypothesizing, \
not recapping something the assistant already said, not acknowledging)

If the turn is the assistant confirming or elaborating on a user-stated fact, \
SKIP — we already have the fact on the user turn.

Hard SKIP list (even if a fact is mentioned):
- Opinions, feelings, prefer-not-to-say, work-in-progress discussion
- Price/spec/quantity numbers about options being compared or planned
- Product names, brand names, gadget specs — unless they're a prescription
- Hobbies, interests, general preferences, casual mentions of friends
- Re-mentions, reminders, restatements of previously-stated facts
- Any turn that is a question, greeting, or acknowledgement
- Narrative commentary, backstory, or context
- Numbers that are merely discussed as part of budget/shopping/planning

Turn ({role}): {text}

If CRITICAL, output exactly 3 short alt-keys (5-15 words each), each on its \
own line:
ALT: <key 1>
ALT: <key 2>
ALT: <key 3>

If not critical, output EXACTLY:
SKIP"""


PROMPT_VERSIONS = {
    "v1": CRITICAL_PROMPT_V1,
    "v2": CRITICAL_PROMPT_V2,
    "v3": CRITICAL_PROMPT_V3,
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class CriticalTurnDecision:
    parent_index: int
    conversation_id: str
    turn_id: int
    role: str
    text: str
    raw_response: str
    critical: bool
    alt_keys: list[str]


@dataclass
class CriticalAltKey:
    parent_index: int
    text: str


# ---------------------------------------------------------------------------
# Prompt builder + response parser
# ---------------------------------------------------------------------------
def build_prompt(version: str, role: str, text: str) -> str:
    tpl = PROMPT_VERSIONS[version]
    # Truncate very long turns to keep prompt cost bounded.
    t = text[:1200]
    return tpl.format(role=role, text=t)


def parse_response(response: str) -> tuple[bool, list[str]]:
    """Return (critical, alt_keys). On any parse failure, return (False, [])."""
    text = (response or "").strip()
    if not text:
        return False, []
    stripped = text.strip().strip("'\"`")
    if stripped.upper() == "SKIP":
        return False, []

    alts: list[str] = []
    saw_skip = False
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        up = line.upper()
        if up == "SKIP":
            saw_skip = True
            continue
        if up.startswith("ALT:") or up.startswith("ALT "):
            idx = line.find(":")
            alt = line[idx + 1:].strip() if idx >= 0 else line[3:].strip()
            alt = alt.strip().strip("'\"<>").strip()
            if alt and len(alt) >= 3:
                alts.append(alt)

    if alts:
        return True, alts[:3]
    if saw_skip:
        return False, []
    return False, []  # unknown -> safer default: skip


# ---------------------------------------------------------------------------
# LLM classifier (threaded, cached)
# ---------------------------------------------------------------------------
class CriticalInfoGenerator:
    def __init__(
        self,
        client: OpenAI | None = None,
        model: str = "gpt-5-mini",
        prompt_version: str = "v2",
        max_workers: int = 8,
        cache: BestshotLLMCache | None = None,
    ):
        self.client = client or OpenAI(timeout=60.0)
        self.model = model
        self.prompt_version = prompt_version
        self.max_workers = max_workers
        self.cache = cache or BestshotLLMCache()
        self._cache_lock = Lock()
        self._counter_lock = Lock()
        self.n_cached = 0
        self.n_uncached = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _cache_key_prompt(self, prompt: str) -> str:
        # Namespace by classifier identity so we don't collide with
        # ingest_llm_altkeys' prompts (those ask for different outputs).
        return f"[critical_info_store/{self.prompt_version}]\n" + prompt

    def call_one(self, role: str, text: str) -> str:
        prompt = build_prompt(self.prompt_version, role, text)
        ck = self._cache_key_prompt(prompt)

        with self._cache_lock:
            cached = self.cache.get(self.model, ck)
        if cached is not None:
            with self._counter_lock:
                self.n_cached += 1
            return cached

        raw = ""
        pt = 0
        ct = 0
        last_err: Exception | None = None
        for tok_budget in (800, 1600, 3200):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=tok_budget,
                )
                raw = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                last_err = None
                break
            except Exception as e:
                last_err = e
                msg = str(e)
                if "max_tokens" in msg or "output limit" in msg:
                    continue
                if tok_budget == 800:
                    continue
                raise
        if last_err is not None:
            raw = ""

        with self._cache_lock:
            self.cache.put(self.model, ck, raw)
        with self._counter_lock:
            self.n_uncached += 1
            self.total_prompt_tokens += int(pt)
            self.total_completion_tokens += int(ct)
        return raw

    def save(self) -> None:
        with self._cache_lock:
            self.cache.save()


def classify_turns(
    generator: CriticalInfoGenerator,
    segments: Iterable[Segment],
    log_every: int = 100,
) -> list[CriticalTurnDecision]:
    """Classify all segments (parallel across the whole corpus — no context
    dependency)."""
    segs = list(segments)
    n = len(segs)
    decisions: list[CriticalTurnDecision | None] = [None] * n
    t0 = time.time()
    done = [0]
    last_save = [t0]

    def _do(i: int) -> tuple[int, CriticalTurnDecision]:
        s = segs[i]
        raw = generator.call_one(s.role, s.text)
        critical, alts = parse_response(raw)
        dec = CriticalTurnDecision(
            parent_index=s.index,
            conversation_id=s.conversation_id,
            turn_id=s.turn_id,
            role=s.role,
            text=s.text,
            raw_response=raw,
            critical=critical,
            alt_keys=alts,
        )
        done[0] += 1
        if done[0] % log_every == 0:
            el = time.time() - t0
            rate = done[0] / max(el, 1e-6)
            eta = (n - done[0]) / max(rate, 1e-6)
            print(
                f"  [{done[0]}/{n}] cached={generator.n_cached} "
                f"uncached={generator.n_uncached} "
                f"rate={rate:.1f}/s eta={eta:.0f}s",
                flush=True,
            )
            if time.time() - last_save[0] > 30:
                generator.save()
                last_save[0] = time.time()
        return i, dec

    with ThreadPoolExecutor(max_workers=generator.max_workers) as ex:
        futures = [ex.submit(_do, i) for i in range(n)]
        for f in as_completed(futures):
            i, dec = f.result()
            decisions[i] = dec

    generator.save()
    return [d for d in decisions if d is not None]


# ---------------------------------------------------------------------------
# Separate vector store
# ---------------------------------------------------------------------------
class CriticalInfoStore:
    """A disjoint vector store of critical-turn alt-keys.

    Keeps a list of alt-key embeddings and a parallel array mapping each
    alt-key row to its parent original-segment index. Also tracks the original
    turn_id per alt-key for conversation filtering.
    """

    def __init__(
        self,
        base: SegmentStore,
        alt_keys: list[CriticalAltKey],
        alt_embeddings: np.ndarray,
    ):
        self._base = base
        self.alt_keys = alt_keys
        if len(alt_keys) == 0 or alt_embeddings.size == 0:
            dim = base.normalized_embeddings.shape[1]
            self.alt_normalized = np.zeros((0, dim), dtype=np.float32)
            self.alt_parent_index = np.zeros(0, dtype=np.int64)
            self.alt_conversation_ids = np.zeros(0, dtype=object)
        else:
            norms = np.linalg.norm(alt_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            self.alt_normalized = (alt_embeddings / norms).astype(np.float32)
            self.alt_parent_index = np.array(
                [k.parent_index for k in alt_keys], dtype=np.int64,
            )
            parent_convs = [
                base.segments[k.parent_index].conversation_id for k in alt_keys
            ]
            self.alt_conversation_ids = np.array(parent_convs, dtype=object)

    def search_per_parent(
        self,
        query_embedding: np.ndarray,
        top_m: int,
        conversation_id: str,
        min_score: float = -1.0,
    ) -> list[tuple[int, float, Segment]]:
        """Return up to top_m (parent_index, score, Segment) tuples, one per
        parent, scored as the MAX alt-key similarity for that parent.

        Results are filtered to `conversation_id` and to scores >= min_score.
        Sorted descending by score.
        """
        if self.alt_normalized.shape[0] == 0:
            return []
        q = query_embedding.astype(np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-10)
        alt_sims = self.alt_normalized @ q  # (M,)

        # conversation filter
        mask = self.alt_conversation_ids == conversation_id
        if not np.any(mask):
            return []
        alt_sims = np.where(mask, alt_sims, -np.inf)

        # Per-parent max
        base_n = len(self._base.segments)
        per_parent = np.full(base_n, -np.inf, dtype=np.float32)
        np.maximum.at(per_parent, self.alt_parent_index, alt_sims)

        # Candidates above min_score
        candidates = np.argsort(per_parent)[::-1]
        out: list[tuple[int, float, Segment]] = []
        for idx in candidates:
            sc = float(per_parent[idx])
            if sc == -np.inf or sc < min_score:
                break
            out.append((int(idx), sc, self._base.segments[int(idx)]))
            if len(out) >= top_m:
                break
        return out


# ---------------------------------------------------------------------------
# Merge strategies
# ---------------------------------------------------------------------------
def merge_additive_bonus(
    main_ranked: list[tuple[Segment, float]],
    crit_ranked: list[tuple[int, float, Segment]],
    K: int,
    bonus: float = 0.1,
) -> list[Segment]:
    """Merge: take main pool as-is, plus critical hits with +bonus score.

    Union by original parent_index; keep max score; stable-sort by score
    descending; break ties toward main (main entries keep their position if
    scores equal).
    """
    # Build score map: parent_index -> (score, segment, from_main?)
    best: dict[int, tuple[float, Segment, bool]] = {}
    for seg, sc in main_ranked:
        best[seg.index] = (sc, seg, True)
    for parent_idx, sc, seg in crit_ranked:
        boosted = sc + bonus
        if parent_idx in best:
            cur_sc, cur_seg, cur_main = best[parent_idx]
            if boosted > cur_sc:
                # higher-scored crit wins
                best[parent_idx] = (boosted, cur_seg, cur_main)  # keep main's seg ref
        else:
            best[parent_idx] = (boosted, seg, False)

    items = list(best.values())
    # Sort: score desc, main-first on ties
    items.sort(key=lambda x: (-x[0], 0 if x[2] else 1))
    return [seg for _, seg, _ in items[:K]]


def merge_always_top_m(
    main_ranked: list[tuple[Segment, float]],
    crit_ranked: list[tuple[int, float, Segment]],
    K: int,
    top_m: int = 5,
    min_score: float = 0.2,
) -> list[Segment]:
    """Always-include top-M critical hits (that clear min_score), then fill
    with main hits up to K. Deduplicate by parent_index."""
    out_segs: list[Segment] = []
    seen: set[int] = set()

    # First: top-M crit above min_score
    for parent_idx, sc, seg in crit_ranked[:top_m]:
        if sc < min_score:
            break
        if parent_idx in seen:
            continue
        out_segs.append(seg)
        seen.add(parent_idx)
        if len(out_segs) >= K:
            return out_segs

    # Then: main hits in order
    for seg, _ in main_ranked:
        if seg.index in seen:
            continue
        out_segs.append(seg)
        seen.add(seg.index)
        if len(out_segs) >= K:
            break
    return out_segs[:K]


# ---------------------------------------------------------------------------
# Convenience: decisions -> CriticalAltKey list (deduped by text)
# ---------------------------------------------------------------------------
def decisions_to_altkeys(
    decisions: Iterable[CriticalTurnDecision],
) -> list[CriticalAltKey]:
    out: list[CriticalAltKey] = []
    seen: set[str] = set()
    for d in decisions:
        if not d.critical:
            continue
        for alt in d.alt_keys:
            key = alt.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(CriticalAltKey(parent_index=d.parent_index, text=key))
    return out
