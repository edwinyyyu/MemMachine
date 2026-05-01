"""Meta-router composition: conditional dispatch between two_speaker_filter
and gated_overlay.

Motivation
----------
From session findings:
  - `two_speaker_filter` STRICTLY DOMINATES `gated_overlay` on every LoCoMo
    shape tested. Zero per-query LLM cost.
  - Only 60% of LoCoMo queries name a participant (the other 40% need a
    fallback).
  - `gated_overlay` (+3.3pp LoCoMo K=50) is the natural fallback.

Composition
-----------
For each (query, conversation_id):
  1. Extract capitalized proper-noun tokens from the query (regex, zero-LLM).
  2. Look up known conversation participants (from
     `results/conversation_two_speakers.json`).
  3. If any token matches a known participant's first name:
       -> dispatch to `two_speaker_filter` (zero LLM)
  4. Else:
       -> dispatch to `gated_overlay` (v1, confidence_threshold=0.7)

Variants
--------
  - MetaRouter         : as above (two_speaker when speaker matches, else gated)
  - MetaRouterInverted : CONTROL. gated when speaker matches, two_speaker else.
                          Should lose — confirms dispatch logic matters.

Do NOT modify framework files, `two_speaker_filter.py`, or `gated_overlay.py`.
This module imports those classes and orchestrates them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from associative_recall import Segment, SegmentStore
from gated_overlay import GatedOverlay
from openai import OpenAI
from speaker_attributed import extract_name_mentions
from two_speaker_filter import _CONV_TWO_SPEAKERS_FILE, TwoSpeakerFilter

# ---------------------------------------------------------------------------
# Route constants
# ---------------------------------------------------------------------------
ROUTE_TWO_SPEAKER = "two_speaker"
ROUTE_GATED = "gated"


# ---------------------------------------------------------------------------
# Result dataclass (matches GatedResult shape)
# ---------------------------------------------------------------------------
@dataclass
class MetaRouterResult:
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared speaker-map loader
# ---------------------------------------------------------------------------
def load_speaker_pairs() -> dict[str, dict[str, str]]:
    """Load known (user, assistant) first-name pairs per conversation."""
    if not _CONV_TWO_SPEAKERS_FILE.exists():
        return {}
    try:
        with open(_CONV_TWO_SPEAKERS_FILE) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    raw = data.get("speakers", {}) or {}
    out: dict[str, dict[str, str]] = {}
    for cid, pair in raw.items():
        if not isinstance(pair, dict):
            continue
        out[cid] = {
            "user": (pair.get("user") or "UNKNOWN") or "UNKNOWN",
            "assistant": (pair.get("assistant") or "UNKNOWN") or "UNKNOWN",
        }
    return out


def query_mentions_known_speaker(
    query: str, conversation_id: str, pairs: dict[str, dict[str, str]]
) -> tuple[bool, list[str], list[str]]:
    """Zero-LLM regex dispatch check.

    Returns (matches, query_name_tokens, matched_names).
    matches is True iff any proper-noun token in the query equals a
    known participant's first name for this conversation.
    """
    pair = pairs.get(conversation_id, {})
    user_name = pair.get("user") or "UNKNOWN"
    asst_name = pair.get("assistant") or "UNKNOWN"
    tokens = extract_name_mentions(query)
    tlow = {t.lower() for t in tokens}

    matched: list[str] = []
    if user_name != "UNKNOWN" and user_name.lower() in tlow:
        matched.append(user_name)
    if asst_name != "UNKNOWN" and asst_name.lower() in tlow:
        matched.append(asst_name)
    return (len(matched) > 0), tokens, matched


# ---------------------------------------------------------------------------
# Meta-router composition
# ---------------------------------------------------------------------------
class MetaRouter:
    """Dispatch between two_speaker_filter and gated_overlay by regex
    speaker-mention check. Zero per-query LLM cost for the routing decision.

    Parameters
    ----------
    store : SegmentStore
    client : OpenAI
    inverted : bool
        If False (default): speaker-mentioned -> two_speaker, else -> gated.
        If True (control): speaker-mentioned -> gated, else -> two_speaker.
    gated_threshold : float
        Confidence threshold for gated_overlay (v1 default 0.7).
    name : str
        Architecture name for reporting.
    """

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        inverted: bool = False,
        gated_threshold: float = 0.7,
        name: str | None = None,
    ) -> None:
        self.store = store
        self.client = client or OpenAI(timeout=60.0, max_retries=3)
        self.inverted = inverted
        self.gated_threshold = gated_threshold
        self.arch_name = name or ("meta_router_inverted" if inverted else "meta_router")

        # Instantiate both architectures (they have their own caches).
        self.two_speaker = TwoSpeakerFilter(store, client=self.client)
        self.gated = GatedOverlay(
            store,
            client=self.client,
            threshold=gated_threshold,
            name=f"gated_threshold_{gated_threshold}",
        )

        # Shared speaker-pair lookup (same JSON TwoSpeakerFilter already
        # ingested; we load it directly for the pure-regex routing decision).
        self.speaker_pairs = load_speaker_pairs()

        # Counters
        self.embed_calls = 0
        self.llm_calls = 0

    # --- cache/counter helpers ---
    def reset_counters(self) -> None:
        self.two_speaker.embed_calls = 0
        self.two_speaker.llm_calls = 0
        self.gated.reset_counters()
        self.embed_calls = 0
        self.llm_calls = 0

    def save_caches(self) -> None:
        try:
            self.two_speaker.save_caches()
        except Exception as e:
            print(f"  (warn) two_speaker.save_caches: {e}", flush=True)
        try:
            self.gated.save_caches()
        except Exception as e:
            print(f"  (warn) gated.save_caches: {e}", flush=True)

    def embed_text(self, text: str) -> np.ndarray:
        """Expose a single embedding helper (used by the eval driver for
        its own cosine backfill computation). Delegates to gated's cache
        (which warm-reads from most other caches)."""
        return self.gated.embed_text(text)

    # --- dispatch ---
    def _collect_counters(self) -> None:
        self.embed_calls = getattr(self.two_speaker, "embed_calls", 0) + getattr(
            self.gated, "embed_calls", 0
        )
        self.llm_calls = getattr(self.two_speaker, "llm_calls", 0) + getattr(
            self.gated, "llm_calls", 0
        )

    def retrieve(
        self,
        question: str,
        conversation_id: str,
        K: int = 50,
    ) -> MetaRouterResult:
        matches, name_tokens, matched_names = query_mentions_known_speaker(
            question, conversation_id, self.speaker_pairs
        )

        # Primary routing rule
        use_two_speaker = matches
        # Inverted variant flips the rule.
        if self.inverted:
            use_two_speaker = not matches

        route = ROUTE_TWO_SPEAKER if use_two_speaker else ROUTE_GATED

        if use_two_speaker:
            # TwoSpeakerFilter.retrieve returns a BestshotResult with
            # .segments and .metadata.
            inner = self.two_speaker.retrieve(question, conversation_id)
            inner_segments = list(inner.segments)
            inner_meta = inner.metadata or {}
        else:
            # GatedOverlay.retrieve takes K.
            inner = self.gated.retrieve(question, conversation_id, K=K)
            inner_segments = list(inner.segments)
            inner_meta = inner.metadata or {}

        self._collect_counters()

        metadata: dict = {
            "name": self.arch_name,
            "route": route,
            "routing_rule": ("inverted" if self.inverted else "primary"),
            "query_name_tokens": name_tokens,
            "matched_names": matched_names,
            "speaker_mention_matches": matches,
            "inner_arch": (
                "two_speaker_filter" if use_two_speaker else "gated_overlay"
            ),
            "inner_metadata": inner_meta,
        }
        return MetaRouterResult(segments=inner_segments, metadata=metadata)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def build_meta_router(
    name: str,
    store: SegmentStore,
    client: OpenAI | None = None,
) -> MetaRouter:
    if name == "meta_router":
        return MetaRouter(store, client=client, inverted=False, name="meta_router")
    if name == "meta_router_inverted":
        return MetaRouter(
            store, client=client, inverted=True, name="meta_router_inverted"
        )
    raise KeyError(name)


VARIANTS = ("meta_router", "meta_router_inverted")
