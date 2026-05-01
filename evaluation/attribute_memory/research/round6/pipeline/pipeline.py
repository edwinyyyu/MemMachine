"""Semantic-memory ingestion pipeline prototype.

Sits on top of event memory. Converts a stream of (ts, source, text) turns
into per-topic append-only logs that a downstream agent can query.

Architecture in one paragraph:

   event stream
       -> [salience prefilter: cheap keyword/embedding heuristic]
       -> [batcher: silence-gap OR rolling window of N turns]
       -> [extractor LLM: emits {"commands":[{op:"append"/"append_ref"/"noop",...}]}]
       -> [router: each command names a topic; matched to existing topic via
           embedding-keyed topic index; new topic created if no near match]
       -> [log store: append-only per topic; batch UUID attached for rollback]
       -> [background consolidator: when a topic grows past K entries, a
           single LLM call produces a 'compaction' entry that summarizes
           prior entries and marks them consolidated. Full history preserved]
       -> [query interface: free-text query -> topic-selection (embedding) ->
           topic log(s) rendered -> LLM answers over log(s)]

Seven research questions this file answers in code form:

  1. Trigger: silence-gap OR N-turn rolling window (whichever fires first).
     Per-turn is too expensive (~30x cost in the S2 scenario); per-conversation
     has no coherent boundary in persistent chat. Default: N=5, gap=30min.
  2. Batching: small enough that a single LLM call can hold full context;
     big enough that multi-turn correction/set-add opportunities coexist.
     Default: 5 turns. Overrides: conversation end, silence gap >30min.
  3. Extraction: multi-append per batch, with noop as a first-class op.
     The batch LLM emits one command per memory-worthy fact; typically 0-3
     per batch. Phatic batches produce a single noop.
  4. Routing: the extractor chooses the topic name in the same call. An
     embedding-based topic-alias index deduplicates routing ("User/Cats/Luna"
     vs "User/Pets/Luna" collapse to whichever appeared first).
  5. Consolidation: triggered when a topic exceeds 8 live entries; single
     LLM call produces a 'supersede' entry that rolls up prior content.
     Prior entries stay in log but are marked consolidated.
  6. Query: free text -> top-3 topic logs by embedding cosine on topic
     name+text -> render logs -> LLM generates answer.
  7. Entity linking: the extractor is given a running alias index at prompt
     time ("known entities: Luna, Miso, ..."). Embeddings catch paraphrase
     drift; the extractor reuses canonical names when they're in the index.

All LLM calls are cached by (model, prompt) hash. Embeddings by (model, text).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import openai
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parents[3]  # evaluation/
load_dotenv(EVAL_ROOT / ".env")

MODEL = "gpt-5-mini"
EMBED_MODEL = "text-embedding-3-small"
PRICE_PER_CALL_APPROX = 0.003  # conservative
PRICE_PER_EMBED_APPROX = 0.00002


# -----------------------------------------------------------------------------
# Cache + budget
# -----------------------------------------------------------------------------


def _sha(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


class LLMCache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._d: dict[str, str] = {}
        if path.exists():
            try:
                self._d = json.loads(path.read_text())
            except Exception:
                self._d = {}
        self._dirty = False

    def get(self, key: str) -> str | None:
        return self._d.get(key)

    def put(self, key: str, value: str) -> None:
        self._d[key] = value
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._d))
        self._dirty = False


class EmbedCache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._d: dict[str, list[float]] = {}
        if path.exists():
            try:
                self._d = json.loads(path.read_text())
            except Exception:
                self._d = {}
        self._dirty = False

    def get(self, key: str) -> list[float] | None:
        return self._d.get(key)

    def put(self, key: str, value: list[float]) -> None:
        self._d[key] = value
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._d))
        self._dirty = False


@dataclass
class Budget:
    max_llm: int = 300
    max_embed: int = 100
    stop_at_llm: int = 240  # 80%
    stop_at_embed: int = 80  # 80%
    llm_calls: int = 0
    embed_calls: int = 0

    def tick_llm(self) -> None:
        self.llm_calls += 1
        if self.llm_calls >= self.stop_at_llm:
            raise RuntimeError(
                f"Budget stop: {self.llm_calls}/{self.max_llm} LLM calls."
            )

    def tick_embed(self) -> None:
        self.embed_calls += 1
        if self.embed_calls >= self.stop_at_embed:
            raise RuntimeError(
                f"Budget stop: {self.embed_calls}/{self.max_embed} embed calls."
            )

    def approx_cost(self) -> float:
        return (
            self.llm_calls * PRICE_PER_CALL_APPROX
            + self.embed_calls * PRICE_PER_EMBED_APPROX
        )


_client: openai.OpenAI | None = None


def client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def llm_call(
    prompt: str, cache: LLMCache, budget: Budget, reasoning_effort: str = "low"
) -> str:
    key = _sha(MODEL, reasoning_effort, prompt)
    cached = cache.get(key)
    if cached is not None:
        return cached
    resp = client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort=reasoning_effort,
    )
    text = resp.choices[0].message.content or ""
    cache.put(key, text)
    budget.tick_llm()
    return text


def embed(texts: Iterable[str], cache: EmbedCache, budget: Budget) -> list[list[float]]:
    texts = list(texts)
    out: list[list[float] | None] = [None] * len(texts)
    to_fetch_idx: list[int] = []
    to_fetch_text: list[str] = []
    for i, t in enumerate(texts):
        key = _sha(EMBED_MODEL, t)
        c = cache.get(key)
        if c is not None:
            out[i] = c
        else:
            to_fetch_idx.append(i)
            to_fetch_text.append(t)
    if to_fetch_text:
        resp = client().embeddings.create(model=EMBED_MODEL, input=to_fetch_text)
        budget.tick_embed()  # batched, count as one
        for idx, datum in zip(to_fetch_idx, resp.data, strict=True):
            vec = list(datum.embedding)
            out[idx] = vec
            cache.put(_sha(EMBED_MODEL, texts[idx]), vec)
    assert all(v is not None for v in out)
    return [v for v in out]  # type: ignore


def cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(va @ vb / (na * nb))


# -----------------------------------------------------------------------------
# Data types
# -----------------------------------------------------------------------------


@dataclass
class Event:
    ts: str
    source: str  # "user", "assistant", "document"
    text: str

    def format(self) -> str:
        return f"[{self.ts}] {self.source}: {self.text}"


@dataclass
class LogEntry:
    entry_id: int
    topic: str
    text: str
    ts: str
    source_event_indexes: list[int]
    batch_id: str
    refs: list[int] = field(default_factory=list)
    relation: str | None = None  # clarify|refine|supersede|invalidate|consolidate|None
    consolidated: bool = False  # set when a supersede/consolidate entry replaces this


@dataclass
class TopicLog:
    name: str
    entries: list[LogEntry] = field(default_factory=list)
    embedding: list[float] | None = None  # of topic name + concatenated recent texts

    def live_entries(self) -> list[LogEntry]:
        return [e for e in self.entries if not e.consolidated]


# -----------------------------------------------------------------------------
# Salience pre-filter (cheap: keyword + regex, no LLM)
# -----------------------------------------------------------------------------

_PHATIC_PATTERNS = [
    r"^(hi|hey|hello|morning|good morning|good evening|yo|sup|howdy)\b",
    r"^(bye|cya|goodbye|see ya|later|ttyl)\b",
    r"^(thanks|thank you|thx|ty)\b",
    r"^(ok|okay|got it|cool|nice|sweet|awesome|understood|sounds good)\.?$",
    r"^(lol|haha|lmao|rofl)\.?$",
    r"^happy (birthday|valentine|new year|holidays)",
    r"weather|raining|sunny",  # weak; let LLM handle
]

_PHATIC_RE = [re.compile(p, re.IGNORECASE) for p in _PHATIC_PATTERNS]


def is_clearly_phatic(text: str) -> bool:
    """Conservative pre-filter: true only if the turn is obviously pure chitchat
    AND short. We prefer false negatives (let LLM noop) to false positives
    (silently drop a fact)."""
    t = text.strip()
    if not t:
        return True
    if len(t) > 120:
        return False  # longer messages likely carry content
    for r in _PHATIC_RE:
        if r.search(t):
            return True
    # Very short one-word replies.
    if len(t.split()) <= 2 and not any(c.isdigit() for c in t):
        return True
    return False


# -----------------------------------------------------------------------------
# Batching policy
# -----------------------------------------------------------------------------


@dataclass
class BatchPolicy:
    # Window over non-phatic turns.
    n_turns: int = 5
    # Silence gap (minutes) forces flush.
    silence_gap_minutes: float = 30.0
    # Any single turn over this length flushes immediately (document ingest).
    long_turn_flush_chars: int = 500

    def should_flush(self, pending: list[Event], next_event: Event | None) -> bool:
        if not pending:
            return False
        if next_event is None:
            return True  # end of stream
        if len(pending) >= self.n_turns:
            return True
        # Check silence gap.
        try:
            last = _parse_ts(pending[-1].ts)
            nxt = _parse_ts(next_event.ts)
            if (nxt - last) / 60.0 >= self.silence_gap_minutes:
                return True
        except Exception:
            pass
        # Flush on big chunks (documents)
        if any(len(e.text) >= self.long_turn_flush_chars for e in pending):
            return True
        return False


def _parse_ts(ts: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(ts).timestamp()


# -----------------------------------------------------------------------------
# Extraction + routing
# -----------------------------------------------------------------------------

EXTRACTION_PROMPT_TMPL = """You are a semantic memory extractor. Convert a batch of raw conversation turns into append-only memory commands.

SCHEMA
Each command is JSON. Shapes:
  append        : op=append, topic=<topic_name>, text=<natural-language fact>
  append_ref    : op=append_ref, topic=<topic_name>, refs=[<entry_id>,...],
                  relation=clarify|refine|supersede|invalidate, text=<fact>
  noop          : op=noop

TOPIC NAMING
- Use hierarchical slash-paths: "User/Name", "User/Work", "User/Cats/Luna",
  "Partner/Riley/Career", "Elena/Family/Sofia".
- REUSE topics from the known-topics list below whenever the new fact fits.
  Consistency matters more than precision.
- Each entity gets its own subtree. "User/Cats/Luna" is a single topic even
  when we add more facts about Luna.

RELATIONS (only for append_ref)
- supersede: new claim replaces prior (e.g. age correction).
- invalidate: prior claim was wrong / retracted without replacement.
- clarify: adds non-conflicting detail to prior entry.
- refine: narrows or qualifies prior entry.

WHEN TO NOOP
- Pure chitchat, weather, greetings, jokes, filler.
- Assistant turns (usually just acknowledgements).
- Vague emotional venting without factual content.
- Return a single noop command if the entire batch is noop-worthy.

WHEN TO APPEND
- Factual claim about user, their relations, their world.
- Preferences, allergies, jobs, relationships, pets, places.
- Explicit updates or corrections on prior facts ("actually she's 4, not 3")
  -> append_ref supersede.

KNOWN TOPICS (reuse these names when applicable)
__KNOWN_TOPICS_BLOCK__

KNOWN ENTITIES (canonical names -- reuse)
__KNOWN_ENTITIES_BLOCK__

PRIOR RELEVANT LOG ENTRIES (use entry_id for refs)
__PRIOR_ENTRIES_BLOCK__

BATCH
__BATCH_BLOCK__

Output a JSON array of commands and NOTHING else. Valid JSON only, no markdown.
Example output:
  [{"op":"append","topic":"User/Name","text":"User's name is Alex"}]
One fact per command. Emit AS MANY commands as there are distinct factual
claims in the batch -- a dense paragraph about a person can yield 10+ appends.
Do not compress multiple unrelated facts into a single append.
"""


CONSOLIDATION_PROMPT_TMPL = """You are compacting an append-only semantic memory log.

TOPIC: __TOPIC__

LOG ENTRIES (entry_id, text; chronological):
__ENTRIES_BLOCK__

Produce a SINGLE consolidated fact that captures the current state of this topic,
resolving supersedes, skipping invalidated claims, merging clarifications.
Keep it concise but preserve all live distinct facts.
Output JSON only, no markdown:
  {"text":"<consolidated current-state paragraph>","refs":[<all_entry_ids_consolidated>]}
"""


QUERY_PROMPT_TMPL = """You are answering a user's query from their semantic memory.

QUERY: __QUERY__

RELEVANT MEMORY TOPICS (ranked by relevance):
__TOPICS_BLOCK__

Answer the query using ONLY the memory above. Be concise. If the memory
lacks information to answer, say so. Cite topic names in brackets when useful.
"""


def _fill(template: str, **kwargs: str) -> str:
    out = template
    for k, v in kwargs.items():
        out = out.replace(f"__{k.upper()}__", v)
    return out


def _extract_json(text: str) -> Any:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\[.*\]|\{.*\}", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    batch_policy: BatchPolicy = field(default_factory=BatchPolicy)
    # Default OFF: empirically the cheap keyword filter removes conversational
    # context around corrections (weather chitchat sandwiched between real
    # facts), and the LLM's own noop discipline is good enough. See
    # no_prefilter ablation: 6/8 vs 5/8 rubric. Keep the option for use cases
    # where LLM cost is the binding constraint.
    salience_prefilter: bool = False
    consolidation_threshold: int = 10
    topic_alias_cosine_threshold: float = 0.78
    query_top_k_topics: int = 6
    # Broad queries ("tell me about the user") benefit from pulling every
    # topic whose entity prefix matches the query. We union prefix-matched
    # topics with top-K similarity topics.
    query_prefix_union: bool = True
    skip_all_phatic_batches: bool = True


class SemanticMemoryPipeline:
    def __init__(
        self, cache_dir: Path, budget: Budget, config: PipelineConfig | None = None
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.llm_cache = LLMCache(cache_dir / "llm.json")
        self.embed_cache = EmbedCache(cache_dir / "embed.json")
        self.budget = budget
        self.config = config or PipelineConfig()

        self.topics: dict[str, TopicLog] = {}
        self.entries: dict[int, LogEntry] = {}  # entry_id -> entry
        self.next_entry_id: int = 1
        self.all_events: list[Event] = []
        self.batches: list[dict[str, Any]] = []  # for rollback / debugging

        # Topic alias index: list of (topic_name, embedding) for fuzzy routing.
        self._topic_embeddings: dict[str, list[float]] = {}

        # Entity alias index: canonical name -> list of occurrences.
        self.entities: set[str] = set()

    # --- ingest ---

    def ingest(self, events: list[Event]) -> dict[str, Any]:
        """Drive a full stream of events through the pipeline. Returns stats."""
        stats: dict[str, Any] = {
            "num_events": len(events),
            "num_batches": 0,
            "num_phatic_filtered": 0,
            "num_extraction_calls": 0,
            "num_consolidation_calls": 0,
            "num_noop_batches": 0,
            "num_append_commands": 0,
            "num_append_ref_commands": 0,
        }

        pending: list[Event] = []
        for i, ev in enumerate(events):
            self.all_events.append(ev)
            pending.append(ev)
            nxt = events[i + 1] if i + 1 < len(events) else None
            if self.config.batch_policy.should_flush(pending, nxt):
                self._process_batch(pending, stats)
                pending = []
        if pending:
            self._process_batch(pending, stats)

        # Consolidation pass (background trigger: after ingest ends, also
        # in-line per batch below).
        self._maybe_consolidate(stats)

        self.llm_cache.save()
        self.embed_cache.save()
        return stats

    def _process_batch(self, pending: list[Event], stats: dict[str, Any]) -> None:
        stats["num_batches"] += 1

        # Salience pre-filter
        filtered: list[Event] = []
        for ev in pending:
            if self.config.salience_prefilter and is_clearly_phatic(ev.text):
                stats["num_phatic_filtered"] += 1
                continue
            filtered.append(ev)
        if not filtered:
            stats["num_noop_batches"] += 1
            return

        if self.config.skip_all_phatic_batches and not any(
            ev.source in ("user", "document") for ev in filtered
        ):
            # Assistant-only batch with no user/document content; unlikely to
            # carry semantic content. Skip LLM call.
            stats["num_noop_batches"] += 1
            return

        # Build prompt
        known_topics = list(self.topics.keys())
        known_entities = sorted(self.entities)
        prior_entries = self._prior_entries_for_batch(filtered)
        batch_block = "\n".join(ev.format() for ev in pending)  # pass ALL for context

        prompt = _fill(
            EXTRACTION_PROMPT_TMPL,
            known_topics_block="\n".join(f"- {t}" for t in known_topics) or "(none)",
            known_entities_block=", ".join(known_entities) or "(none)",
            prior_entries_block=prior_entries or "(none)",
            batch_block=batch_block,
        )
        raw = llm_call(prompt, self.llm_cache, self.budget)
        stats["num_extraction_calls"] += 1

        cmds = _extract_json(raw)
        if not isinstance(cmds, list):
            cmds = []
        batch_id = _sha(*(ev.ts for ev in pending))[:8]
        self.batches.append(
            {
                "batch_id": batch_id,
                "events": [asdict(ev) for ev in pending],
                "raw": raw,
                "commands": cmds,
            }
        )

        any_append = False
        for cmd in cmds:
            if not isinstance(cmd, dict):
                continue
            op = cmd.get("op")
            if op == "noop":
                continue
            topic = cmd.get("topic")
            text = cmd.get("text")
            if not topic or not text:
                continue
            # Route topic through alias index.
            topic = self._route_topic(topic)
            event_indexes = [self.all_events.index(ev) for ev in pending]
            entry = LogEntry(
                entry_id=self.next_entry_id,
                topic=topic,
                text=text,
                ts=pending[-1].ts,
                source_event_indexes=event_indexes,
                batch_id=batch_id,
                refs=cmd.get("refs", []) or [],
                relation=cmd.get("relation"),
            )
            self.next_entry_id += 1
            self._append_entry(entry)
            # Mark refs: if supersede/invalidate, set the old to consolidated=True.
            # But DO NOT mark a consolidate entry as consolidated when a narrower
            # supersede refs it -- that would evict all the other live facts
            # that the consolidate rolled up. Instead, let both live side-by-side
            # and trust the reader LLM to resolve.
            if entry.relation in ("supersede", "invalidate"):
                for r in entry.refs:
                    if r in self.entries and self.entries[r].relation != "consolidate":
                        self.entries[r].consolidated = True
            if op == "append":
                stats["num_append_commands"] += 1
            elif op == "append_ref":
                stats["num_append_ref_commands"] += 1
            any_append = True
            # Extract entities from topic path (tail segments).
            for seg in topic.split("/"):
                if seg and seg[0].isupper() and seg not in ("User", "Partner"):
                    self.entities.add(seg)

        if not any_append:
            stats["num_noop_batches"] += 1

        # Inline consolidation trigger.
        self._maybe_consolidate(stats)

    def _prior_entries_for_batch(self, batch: list[Event]) -> str:
        """Render a small window of prior entries for this batch -- by topic
        similarity to the batch text."""
        if not self.topics:
            return ""
        # Very cheap: take the last N entries across all topics.
        recent = sorted(self.entries.values(), key=lambda e: e.ts)[-20:]
        return "\n".join(
            f"- id={e.entry_id} topic={e.topic} rel={e.relation} "
            f"[{'consolidated' if e.consolidated else 'live'}]: {e.text}"
            for e in recent
        )

    def _append_entry(self, entry: LogEntry) -> None:
        topic = entry.topic
        if topic not in self.topics:
            self.topics[topic] = TopicLog(name=topic)
        self.topics[topic].entries.append(entry)
        self.entries[entry.entry_id] = entry
        # Reset topic embedding (lazy-recompute at query time).
        self._topic_embeddings.pop(topic, None)

    # --- routing ---

    def _route_topic(self, proposed: str) -> str:
        """Fuzzy topic routing: if proposed is very close to an existing topic
        by embedding similarity, return the existing topic. Otherwise create."""
        proposed = proposed.strip().strip("/")
        if proposed in self.topics:
            return proposed
        if not self.topics:
            return proposed
        # Embed proposed + each candidate and compute cosine.
        candidates = list(self.topics.keys())
        texts_to_embed = [proposed] + [
            c for c in candidates if c not in self._topic_embeddings
        ]
        vecs = embed(texts_to_embed, self.embed_cache, self.budget)
        proposed_vec = vecs[0]
        vec_iter = iter(vecs[1:])
        for c in candidates:
            if c not in self._topic_embeddings:
                self._topic_embeddings[c] = next(vec_iter)
        best = max(
            candidates,
            key=lambda c: cosine(proposed_vec, self._topic_embeddings[c]),
        )
        best_sim = cosine(proposed_vec, self._topic_embeddings[best])
        if best_sim >= self.config.topic_alias_cosine_threshold:
            return best
        # Cache proposed embedding for future routing.
        self._topic_embeddings[proposed] = proposed_vec
        return proposed

    # --- consolidation ---

    def _maybe_consolidate(self, stats: dict[str, Any]) -> None:
        for topic_name, log in list(self.topics.items()):
            live = log.live_entries()
            if len(live) < self.config.consolidation_threshold:
                continue
            # Consolidate: single LLM call summarizing live entries.
            entries_block = "\n".join(
                f"[id={e.entry_id} rel={e.relation or 'append'}] {e.text}" for e in live
            )
            prompt = _fill(
                CONSOLIDATION_PROMPT_TMPL,
                topic=topic_name,
                entries_block=entries_block,
            )
            raw = llm_call(prompt, self.llm_cache, self.budget)
            stats["num_consolidation_calls"] += 1
            obj = _extract_json(raw)
            if not isinstance(obj, dict):
                continue
            text = obj.get("text", "")
            refs = obj.get("refs", [e.entry_id for e in live])
            consolidated_entry = LogEntry(
                entry_id=self.next_entry_id,
                topic=topic_name,
                text=text,
                ts=live[-1].ts,
                source_event_indexes=[],
                batch_id=f"consolidate_{self.next_entry_id}",
                refs=refs,
                relation="consolidate",
            )
            self.next_entry_id += 1
            self._append_entry(consolidated_entry)
            for r in refs:
                if r in self.entries:
                    self.entries[r].consolidated = True

    # --- query ---

    def query(self, q: str) -> dict[str, Any]:
        """Free-text query -> answer string + matched topics."""
        if not self.topics:
            return {"answer": "I don't have any memory yet.", "topics": []}

        # Embed query + topic names (cached).
        topic_names = list(self.topics.keys())
        to_embed = [q] + [self._render_topic_for_embedding(tn) for tn in topic_names]
        vecs = embed(to_embed, self.embed_cache, self.budget)
        q_vec = vecs[0]
        sims = [
            (tn, cosine(q_vec, v)) for tn, v in zip(topic_names, vecs[1:], strict=True)
        ]
        sims.sort(key=lambda p: p[1], reverse=True)
        top = sims[: self.config.query_top_k_topics]
        chosen = {tn for tn, _ in top}

        # Prefix-union: if the query mentions an entity name whose prefix
        # matches a topic root, include all its sub-topics. Also: detect
        # broad "tell me about X" patterns and union everything under X/.
        if self.config.query_prefix_union:
            ql = q.lower()
            entity_names = {tn.split("/")[0] for tn in topic_names}
            for entity in entity_names:
                if entity.lower() in ql or entity == "User":
                    # Only auto-include User/ topics if the query clearly
                    # references the user.
                    if entity.lower() in ql or any(
                        w in ql for w in ("user", "me ", "my ", "i ", "you ")
                    ):
                        for tn in topic_names:
                            if tn.split("/")[0] == entity:
                                chosen.add(tn)

        sim_by_topic = dict(sims)
        ordered = sorted(chosen, key=lambda tn: sim_by_topic.get(tn, 0.0), reverse=True)
        topics_block = "\n\n".join(
            self._render_topic_for_query(tn, sim_by_topic.get(tn, 0.0))
            for tn in ordered
        )
        top = [(tn, sim_by_topic.get(tn, 0.0)) for tn in ordered]
        prompt = _fill(QUERY_PROMPT_TMPL, query=q, topics_block=topics_block)
        answer = llm_call(prompt, self.llm_cache, self.budget, reasoning_effort="low")
        return {
            "answer": answer.strip(),
            "topics": [tn for tn, _ in top],
            "topics_block": topics_block,
        }

    def _render_topic_for_embedding(self, topic: str) -> str:
        log = self.topics[topic]
        live = log.live_entries()
        recent_text = " ".join(e.text for e in live[-4:])
        return f"{topic} :: {recent_text}"

    def _render_topic_for_query(self, topic: str, sim: float) -> str:
        log = self.topics[topic]
        live = log.live_entries()
        if not live:
            # Use last consolidated entry if all live are gone (unusual).
            live = log.entries[-2:]
        body = "\n".join(
            f"  [{e.entry_id}] ({e.ts}) {e.text}"
            + (f" [rel={e.relation}, refs={e.refs}]" if e.relation else "")
            for e in live
        )
        return f"TOPIC: {topic} (sim={sim:.2f})\n{body}"

    # --- rollback ---

    def rollback_batch(self, batch_id: str) -> int:
        """Remove all entries produced by a batch. Returns number removed."""
        to_remove = [e_id for e_id, e in self.entries.items() if e.batch_id == batch_id]
        for e_id in to_remove:
            entry = self.entries.pop(e_id)
            topic = entry.topic
            if topic in self.topics:
                self.topics[topic].entries = [
                    e for e in self.topics[topic].entries if e.entry_id != e_id
                ]
                if not self.topics[topic].entries:
                    self.topics.pop(topic)
                self._topic_embeddings.pop(topic, None)
        # Un-consolidate any entries that were marked consolidated by removed refs.
        for e in self.entries.values():
            if e.consolidated:
                # If the superseder is gone, revive this.
                superseders = [
                    other
                    for other in self.entries.values()
                    if other.relation in ("supersede", "invalidate", "consolidate")
                    and e.entry_id in other.refs
                ]
                if not superseders:
                    e.consolidated = False
        return len(to_remove)

    # --- serialization ---

    def snapshot(self) -> dict[str, Any]:
        return {
            "topics": {
                name: {
                    "name": log.name,
                    "entries": [asdict(e) for e in log.entries],
                }
                for name, log in self.topics.items()
            },
            "batches": self.batches,
            "entities": sorted(self.entities),
        }
