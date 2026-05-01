"""Pronoun-resolution ingest index.

At ingestion time an LLM is asked, per-turn (given 2-3 preceding turns as
context), to rewrite the turn with any pronouns/deictics ("it", "they",
"this", "that one", etc.) substituted by their specific referents. Turns
that are already self-contained get SKIP. The RESOLVED texts form a
SEPARATE vector index -- a disjoint retrieval path.

At query time: v2f runs as usual; we also cosine-search the resolved
index with the query, then STACKED-MERGE into v2f's top-K (v2f first,
then resolved hits fill remaining slots if their parent_turn isn't
already present).

Motivation: idiosyncratic_analysis found 35% of hard-to-retrieve turns
use implicit references ("it", "they") whose text alone can't match a
query that names the specific entity. v2f cue generation cannot invent
the referent. Ingest-time resolution does.

Distinct from `ingest_llm_altkeys.py` (which emitted broad alt-keys and
competed with v2f in max-score merge):
  * narrower: only turns with unresolved pronouns are flagged (target
    SKIP >= 50%)
  * stacked merge (validated by `critical_info_store`) -- resolved hits
    only fill slots v2f leaves empty.

Exports:
  - PronounResolver  (LLM classifier + rewriter, threaded + cached)
  - PronounTurnDecision
  - ResolvedTurnIndex (separate vector store of resolved texts)
  - resolve_turns
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock

import numpy as np
from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)
from openai import OpenAI

# ---------------------------------------------------------------------------
# Dedicated caches (pronoun_*_cache.json) -- do not pollute other agents'
# caches. We still READ from the shared pool for warm-starts.
# ---------------------------------------------------------------------------
_PRONOUN_EMB_FILE = CACHE_DIR / "pronoun_embedding_cache.json"
_PRONOUN_LLM_FILE = CACHE_DIR / "pronoun_llm_cache.json"

_SHARED_EMB_READ = (
    "pronoun_embedding_cache.json",
    "bestshot_embedding_cache.json",
    "embedding_cache.json",
    "arch_embedding_cache.json",
    "agent_embedding_cache.json",
    "meta_embedding_cache.json",
)
_SHARED_LLM_READ = (
    "pronoun_llm_cache.json",
    "bestshot_llm_cache.json",
    "llm_cache.json",
    "arch_llm_cache.json",
    "agent_llm_cache.json",
    "meta_llm_cache.json",
)


class PronounEmbeddingCache(EmbeddingCache):
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
        self.cache_file = _PRONOUN_EMB_FILE
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


class PronounLLMCache(LLMCache):
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
        self.cache_file = _PRONOUN_LLM_FILE
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
# Prompt: narrow -- only rewrite when pronouns/deictics genuinely hide the
# referent. Everything else is SKIP.
# ---------------------------------------------------------------------------
PROMPT_V1 = """\
You are resolving pronouns and deictic expressions in a conversation turn. \
Given the turn and preceding context, rewrite the turn with unambiguous \
references substituted for words like: it, they, them, this, that, these, \
those, here, there, he/she (ONLY if the referent is unclear in the turn \
itself).

ONLY rewrite if the turn has ambiguous references that would be unclear \
to a reader seeing ONLY the turn. If the turn is already self-contained \
(names the entities it talks about), output exactly: SKIP

Preceding context (up to 3 most-recent prior turns):
{prev_context}

Turn ({role}): {text}

Rules:
- Substitute only the ambiguous pronoun/deictic with the concrete referent \
from context.
- Preserve the rest of the turn's wording.
- Do NOT expand with extra commentary.
- If multiple plausible referents exist and you cannot pick one, SKIP.
- If the turn has no ambiguous reference, SKIP.
- Output 4-40 words, or SKIP.

Output exactly one of:
RESOLVED: <rewritten turn>
SKIP"""


PROMPT_VERSIONS = {"v1": PROMPT_V1}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class PronounTurnDecision:
    parent_index: int
    conversation_id: str
    turn_id: int
    role: str
    text: str
    prev_context: str
    raw_response: str
    resolved: bool
    resolved_text: str  # empty if SKIP


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------
def build_prev_context(prev_turns: list[tuple[str, str]]) -> str:
    """prev_turns is a list of (role, text); most recent LAST."""
    if not prev_turns:
        return "(no prior context -- this is the first turn)"
    lines = []
    for role, text in prev_turns:
        snippet = text[:300].replace("\n", " ").strip()
        lines.append(f"[{role}]: {snippet}")
    return "\n".join(lines)


def build_prompt(
    prompt_version: str, role: str, text: str, prev_turns: list[tuple[str, str]]
) -> str:
    tpl = PROMPT_VERSIONS[prompt_version]
    prev_ctx = build_prev_context(prev_turns[-3:])
    return tpl.format(prev_context=prev_ctx, role=role, text=text[:1200])


def parse_response(response: str) -> tuple[bool, str]:
    """Return (resolved, resolved_text). On any parse failure, return
    (False, "")."""
    if not response:
        return False, ""
    text = response.strip()
    # Strip code fences / backticks / quotes if wrapped.
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.startswith("```")]
        text = "\n".join(lines).strip()

    stripped = text.strip().strip("'\"`")
    if stripped.upper() == "SKIP":
        return False, ""

    saw_resolved = False
    resolved_text = ""
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        up = line.upper()
        if up == "SKIP":
            return False, ""
        if up.startswith("RESOLVED:") or up.startswith("RESOLVED "):
            idx = line.find(":")
            payload = line[idx + 1 :].strip() if idx >= 0 else line[9:].strip()
            payload = payload.strip().strip("'\"`").strip()
            if payload:
                saw_resolved = True
                resolved_text = payload
                break

    if saw_resolved and 3 < len(resolved_text) < 600:
        return True, resolved_text
    return False, ""


# ---------------------------------------------------------------------------
# LLM resolver (threaded, cached)
# ---------------------------------------------------------------------------
class PronounResolver:
    def __init__(
        self,
        client: OpenAI | None = None,
        model: str = "gpt-5-mini",
        prompt_version: str = "v1",
        max_workers: int = 8,
        cache: PronounLLMCache | None = None,
    ):
        self.client = client or OpenAI(timeout=60.0)
        self.model = model
        self.prompt_version = prompt_version
        self.max_workers = max_workers
        self.cache = cache or PronounLLMCache()
        self._cache_lock = Lock()
        self._counter_lock = Lock()
        self.n_cached = 0
        self.n_uncached = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _cache_key_prompt(self, prompt: str) -> str:
        # Namespace so we never collide with ingest_llm_altkeys /
        # critical_info_store prompt-cache entries.
        return f"[pronoun_resolution/{self.prompt_version}]\n" + prompt

    def call_one(self, role: str, text: str, prev_turns: list[tuple[str, str]]) -> str:
        prompt = build_prompt(self.prompt_version, role, text, prev_turns)
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


def _resolve_one_conversation(
    resolver: PronounResolver,
    segments: list[Segment],
    progress_cb=None,
) -> list[PronounTurnDecision]:
    """Per conversation, compute prev-context windows (2-3 prior turns) and
    fan out to LLM in parallel. Prev-context only uses ORIGINAL turn text
    (not resolved output) so order doesn't matter for correctness."""
    prev_by_idx: list[list[tuple[str, str]]] = []
    window: list[tuple[str, str]] = []
    for seg in segments:
        prev_by_idx.append(list(window[-3:]))
        window.append((seg.role, seg.text))

    decisions: list[PronounTurnDecision | None] = [None] * len(segments)

    def _do(i: int) -> tuple[int, PronounTurnDecision]:
        seg = segments[i]
        prev = prev_by_idx[i]
        raw = resolver.call_one(seg.role, seg.text, prev)
        resolved, rtxt = parse_response(raw)
        dec = PronounTurnDecision(
            parent_index=seg.index,
            conversation_id=seg.conversation_id,
            turn_id=seg.turn_id,
            role=seg.role,
            text=seg.text,
            prev_context=build_prev_context(prev),
            raw_response=raw,
            resolved=resolved,
            resolved_text=rtxt,
        )
        if progress_cb is not None:
            progress_cb()
        return i, dec

    with ThreadPoolExecutor(max_workers=resolver.max_workers) as ex:
        futures = [ex.submit(_do, i) for i in range(len(segments))]
        for f in as_completed(futures):
            i, dec = f.result()
            decisions[i] = dec

    return [d for d in decisions if d is not None]


def resolve_turns(
    resolver: PronounResolver,
    segments: Iterable[Segment],
    log_every: int = 100,
) -> list[PronounTurnDecision]:
    """Group by conversation, resolve each turn. Returns one decision per
    input segment (some RESOLVED, rest SKIP)."""
    segs = list(segments)
    by_conv: dict[str, list[Segment]] = {}
    for s in segs:
        by_conv.setdefault(s.conversation_id, []).append(s)

    total = len(segs)
    done = [0]
    t0 = time.time()
    last_save = [t0]

    def cb():
        done[0] += 1
        if done[0] % log_every == 0:
            el = time.time() - t0
            rate = done[0] / max(el, 1e-6)
            eta = (total - done[0]) / max(rate, 1e-6)
            print(
                f"  [{done[0]}/{total}] cached={resolver.n_cached} "
                f"uncached={resolver.n_uncached} "
                f"rate={rate:.1f}/s eta={eta:.0f}s",
                flush=True,
            )
            if time.time() - last_save[0] > 30:
                resolver.save()
                last_save[0] = time.time()

    all_decisions: list[PronounTurnDecision] = []
    for cid in sorted(by_conv.keys()):
        segs_sorted = sorted(by_conv[cid], key=lambda s: s.turn_id)
        decs = _resolve_one_conversation(resolver, segs_sorted, cb)
        all_decisions.extend(decs)

    resolver.save()
    return all_decisions


# ---------------------------------------------------------------------------
# Separate vector store of RESOLVED turn texts
# ---------------------------------------------------------------------------
class ResolvedTurnIndex:
    """Disjoint vector store of resolved turn texts.

    Each row: (resolved_text, parent_turn_index, conversation_id) with a
    normalized embedding. search_per_parent() returns up to top_m unique
    parents scoped to a conversation.
    """

    def __init__(
        self,
        base: SegmentStore,
        resolved_parent_indices: list[int],
        resolved_texts: list[str],
        resolved_embeddings: np.ndarray,
    ):
        self._base = base
        assert len(resolved_parent_indices) == len(resolved_texts)
        n = len(resolved_parent_indices)
        dim = base.normalized_embeddings.shape[1]
        self.resolved_texts = resolved_texts
        if n == 0 or resolved_embeddings.size == 0:
            self.normalized = np.zeros((0, dim), dtype=np.float32)
            self.parent_indices = np.zeros(0, dtype=np.int64)
            self.conversation_ids = np.zeros(0, dtype=object)
            return
        norms = np.linalg.norm(resolved_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self.normalized = (resolved_embeddings / norms).astype(np.float32)
        self.parent_indices = np.array(resolved_parent_indices, dtype=np.int64)
        parent_convs = [
            base.segments[p].conversation_id for p in resolved_parent_indices
        ]
        self.conversation_ids = np.array(parent_convs, dtype=object)

    @property
    def n(self) -> int:
        return int(self.normalized.shape[0])

    def search_per_parent(
        self,
        query_embedding: np.ndarray,
        top_m: int,
        conversation_id: str,
    ) -> list[tuple[int, float, Segment]]:
        """Return up to top_m (parent_index, score, Segment) tuples, one per
        parent, scored as the MAX resolved-row similarity for that parent.
        Filtered to `conversation_id`. Sorted by score desc."""
        if self.n == 0:
            return []
        q = query_embedding.astype(np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-10)
        sims = self.normalized @ q  # (N,)

        mask = self.conversation_ids == conversation_id
        if not np.any(mask):
            return []
        sims = np.where(mask, sims, -np.inf)

        base_n = len(self._base.segments)
        per_parent = np.full(base_n, -np.inf, dtype=np.float32)
        np.maximum.at(per_parent, self.parent_indices, sims)

        order = np.argsort(per_parent)[::-1]
        out: list[tuple[int, float, Segment]] = []
        for idx in order:
            sc = float(per_parent[idx])
            if sc == -np.inf:
                break
            out.append((int(idx), sc, self._base.segments[int(idx)]))
            if len(out) >= top_m:
                break
        return out


# ---------------------------------------------------------------------------
# Convenience: embed resolved texts with a provided embedding cache.
# ---------------------------------------------------------------------------
def embed_resolved_texts(
    client: OpenAI,
    cache: EmbeddingCache,
    texts: list[str],
    batch_size: int = 96,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    out: list[np.ndarray | None] = [None] * len(texts)
    pending: list[tuple[int, str]] = []
    for i, t in enumerate(texts):
        tt = (t or "").strip()
        if not tt:
            out[i] = np.zeros(1536, dtype=np.float32)
            continue
        cached = cache.get(tt)
        if cached is not None:
            out[i] = cached.astype(np.float32)
        else:
            pending.append((i, tt))

    if pending:
        print(
            f"  [pronoun_resolution] embedding {len(pending)} new resolved texts ...",
            flush=True,
        )
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        batch_texts = [bt for _, bt in batch]
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch_texts)
                break
            except Exception as e:
                last_exc = e
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"embed failed: {last_exc}")
        for (i, t), ed in zip(batch, resp.data):
            emb = np.array(ed.embedding, dtype=np.float32)
            cache.put(t, emb)
            out[i] = emb

    cache.save()
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Stacked merge -- exact pattern from stacked_alias / critical_info_store
# ---------------------------------------------------------------------------
def stacked_merge(
    main_ranked: list[tuple[Segment, float]],
    resolved_ranked: list[tuple[int, float, Segment]],
    K: int,
) -> list[Segment]:
    """Main (v2f+backfill) ordering is preserved as the prefix. Resolved hits,
    in score order, fill only slots not already occupied by a parent_index
    present in main, up to K."""
    out: list[Segment] = []
    seen: set[int] = set()
    for seg, _ in main_ranked:
        if seg.index in seen:
            continue
        out.append(seg)
        seen.add(seg.index)
        if len(out) >= K:
            return out

    for pidx, _sc, seg in resolved_ranked:
        if pidx in seen:
            continue
        out.append(seg)
        seen.add(pidx)
        if len(out) >= K:
            break
    return out[:K]


def stacked_merge_with_bonus(
    main_ranked: list[tuple[Segment, float]],
    resolved_ranked: list[tuple[int, float, Segment]],
    K: int,
    bonus: float = 0.05,
) -> list[Segment]:
    """Variant: resolved hit only enters the top-K if (resolved_score + bonus)
    beats main_ranked's Kth item's score AND the parent isn't already in
    main. main_ranked's relative ordering is otherwise preserved; a winning
    resolved hit displaces the weakest main item beyond position K.
    """
    # Split main into a mutable list (seg, score) keeping relative order.
    main_list: list[tuple[Segment, float]] = [(s, sc) for s, sc in main_ranked]
    seen: set[int] = {s.index for s, _ in main_list[:K]}

    # Resolved hits sorted by boosted score.
    for pidx, sc, seg in resolved_ranked:
        boosted = sc + bonus
        if pidx in seen:
            continue
        # Find the current Kth score in the (preserved-order) list.
        if len(main_list) < K:
            # Room to spare -- append.
            main_list.append((seg, boosted))
            seen.add(pidx)
            continue
        # Find weakest item within the current top-K by score.
        weakest_pos = None
        weakest_score = None
        for pos in range(min(K, len(main_list))):
            s2, sc2 = main_list[pos]
            if weakest_score is None or sc2 < weakest_score:
                weakest_score = sc2
                weakest_pos = pos
        if weakest_pos is None or weakest_score is None:
            continue
        if boosted > weakest_score:
            # Replace that weakest position with the resolved seg.
            displaced = main_list[weakest_pos][0]
            main_list[weakest_pos] = (seg, boosted)
            seen.discard(displaced.index)
            seen.add(pidx)
        # Otherwise: drop this resolved hit -- v2f's Kth beats it.

    # Truncate to K preserving the new ordering (stable).
    return [s for s, _ in main_list[:K]]
