# Prior Art: Production Semantic / Long-Term Memory Systems

Survey conducted 2026-04-23 to ground design decisions for the `attribute_memory`
component. The current design uses flat `(topic, category, attribute, value)`
rows addressed by `add` / `delete` commands. This document maps that design
against what leading production systems actually ship.

Systems covered: mem0, Letta / MemGPT, Zep / Graphiti, Cognee, ChatGPT memory,
Anthropic Claude memory tool + Claude consumer memory, Character.AI. Plus
benchmark-paper cross-references.

Sections: 1 per-system writeup, 2 comparison table, 3 synthesis + concrete
gaps in attribute_memory.

---

## 1. Per-system writeups

### 1.1 mem0 / Mem0g

**Storage.** Each memory is a short natural-language string (15-80 words per
the prompt) with a stable `id`, plus dense embedding. Mem0g (graph variant)
additionally represents memories as a directed labeled graph `G=(V,E,L)`:
nodes are typed entities with embeddings + timestamps, edges are
`(source, relation, destination)` triplets. There is no declared cardinality
(no notion that `allergies` is a set) — plurality is handled implicitly by the
LLM during the update step.

**Update mechanism.** Two-phase pipeline:

1. *Extraction* — an LLM reads the latest turn + rolling summary + last m=10
   messages and emits `{"facts": ["fact1", "fact2", ...]}`.
2. *Update* — each candidate fact is embedded, top-k similar existing
   memories are retrieved, and a second LLM call emits a `tool-call`-style
   decision per memory: **ADD / UPDATE / DELETE / NONE** (early versions
   also supported MERGE). UPDATE carries `old_memory` so downstream systems
   can track what changed.

A newer "additive" extraction prompt drops UPDATE/DELETE in favor of
append-only storage with `attributed_to` (user / assistant) and
`linked_memory_ids` cross-references (UUIDs).

**Retrieval.** Vector kNN over embeddings. The paper reports a
hybrid/multi-signal retrieval stack ("Mem0 clearly outperforms… F1 28.64"
on LoCoMo, more recent token-efficient variant claimed 91.6): semantic
similarity + BM25 keyword + entity match, fused. Mem0g additionally does
entity-centric graph traversal (find query entities, expand to neighbors)
and triplet-embedding matching.

**Delete / retract.** DELETE op removes an entry by id; UPDATE reuses the id
so history is implicit not explicit. No validity intervals.

**Provenance / confidence.** No per-fact confidence score. Provenance only
appears in the newer additive schema (`attributed_to: user|assistant`).

**Entity canonicalisation.** Only in Mem0g: entities are extracted, embedded,
linked across memories. No strong canonical-id system — similarity-based.

**Notable design choice.** The update prompt is the heart of mem0. Memory
editing is delegated almost entirely to an LLM-driven
"choose one of ADD/UPDATE/DELETE/NONE" decision. Criticism (MemU) is that
flat fact extraction without an organizing schema "flattens intelligence"
because the LLM has no representational grip to reason about relations.

Sources:
- https://arxiv.org/html/2504.19413v1 (Mem0 paper)
- https://docs.mem0.ai/open-source/features/custom-update-memory-prompt
- https://github.com/mem0ai/mem0
- https://memu.pro/blog/mem0-ai-memory-layer-agent-personalization (critique)
- https://mem0.ai/blog/benchmarked-openai-memory-vs-langmem-vs-memgpt-vs-mem0-for-long-term-memory-here-s-how-they-stacked-up

---

### 1.2 Letta / MemGPT

**Storage.** Three tiers, inherited from the MemGPT OS-metaphor paper:

- *Core memory* (in-context) — structured **blocks** pinned to the system
  prompt. A block is `{label, description, value, limit}` where `value` is
  **free-form text** up to a character limit (default 2k). The canonical
  blocks are `human` and `persona` but apps define their own. Blocks can be
  shared across agents.
- *Recall memory* — full conversation log, searchable but not in context.
- *Archival memory* — external vector DB; holds "processed and indexed
  knowledge" and external data sources.

**Update mechanism.** Pure **tool-call** surface. The agent edits its own
memory by calling typed tools: `memory_replace(label, old_str, new_str)`,
`memory_insert(label, text)`, `memory_rethink(label, new_value)`,
`memory_finish_edits()`, `archival_memory_insert(text)`,
`archival_memory_search(query)`. This is a text-editor-style diff patch over
the block's free-form string, not a structured schema update.

**Set-valued facts.** Handled by the LLM writing them into the free-text
block in whatever prose form it chooses. No typed cardinality.

**Delete / retract.** `memory_replace` with empty `new_str`, or rethinking
the whole block. Archival insertions are append-only; the LLM has to search
and then decide to rewrite core memory to reflect a retraction.

**Provenance / confidence.** Not modeled. Block text is the LLM's own prose.

**Retrieval.** Vector search over archival + recall; core memory is always
in-context so needs no retrieval.

**Canonicalisation.** None baked in — it's just text.

**Long novel-style input.** Handled by paging through the context window:
the agent uses `archival_memory_search` iteratively and copies the relevant
excerpts into core memory blocks via `memory_insert`.

**Notable design choice.** Letta's own benchmark paper argues *agent
capability > retrieval mechanism*: simple filesystem/text tools hit 74% on
LoCoMo, beating graph/structured approaches, because LLMs are heavily
trained on filesystem / text-edit operations. Structure can actually hurt
because the LLM has a harder time reasoning over it.

Sources:
- https://docs.letta.com/concepts/memgpt/
- https://docs.letta.com/guides/agents/memory/
- https://www.letta.com/blog/agent-memory
- https://www.letta.com/blog/benchmarking-ai-agent-memory
- https://docs.letta.com/guides/legacy/memgpt_agents_legacy

---

### 1.3 Zep / Graphiti

**Storage.** Temporal knowledge graph with three subgraphs:

- *Episode subgraph* — raw input (chat turns, JSON docs, etc.) with
  temporal boundaries; these are the provenance nodes.
- *Semantic entity subgraph* — extracted entities (Pydantic-typable or
  learned). Nodes carry an embedding and an evolving summary.
- *Community subgraph* — clusters built via label propagation / Leiden,
  used for higher-level retrieval.

Facts are edges: triples `(entity, relation, entity)` with validity
metadata. Every edge carries four timestamps — **bi-temporal**:

| field        | meaning                                    |
| ------------ | ------------------------------------------ |
| `created_at` | when Zep ingested it                       |
| `valid_at`   | when the fact became true in the world     |
| `invalid_at` | when the fact was superseded               |
| `expired_at` | internal expiry / retention                |

**Update mechanism.** Append-only log with invalidation markers. When a new
fact arrives, Graphiti retrieves semantically-related existing edges and
asks an LLM to compare them for contradiction. On contradiction it *does
not delete* — it stamps `invalid_at` on the superseded edge and inserts
the new one. Queries can be point-in-time ("what was true at T?") because
history is preserved.

**Entity resolution.** LLM-guided deduplication at ingest: extracted entity
candidates are compared (semantic + keyword) against existing entity nodes
and merged if they refer to the same thing. Entity summaries evolve over
time as new episodes are ingested.

**Set-valued facts.** Natural — multiple edges of the same relation from
the same subject coexist, each with its own validity window.

**Retrieval.** Hybrid: cosine similarity over entity/edge embeddings +
BM25 + breadth-first graph traversal + reranking (optional MMR / cross-
encoder). Can return edges, nodes, or community summaries.

**Provenance.** Every semantic edge/node traces back to an episode node.

**Notable design choice.** Bi-temporal model (when did it happen vs when
did we learn it) is Zep's distinguishing primitive. There has been
methodological dispute on their LoCoMo claims (Zep: 84% → Mem0
reanalysis: 58.44% → Zep counter: 75.14%).

Sources:
- https://arxiv.org/abs/2501.13956 (Zep paper)
- https://arxiv.org/pdf/2501.13956
- https://github.com/getzep/graphiti
- https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/
- https://help.getzep.com/graph-overview
- https://github.com/getzep/zep-papers/issues/5 (benchmark dispute)

---

### 1.4 Cognee

**Storage.** Knowledge graph (Neo4j or similar) combined with a vector
store. Nodes are typed entities; edges are typed relations. Crucially,
Cognee supports a pluggable **ontology layer** (RDF / Pydantic) — a
`BaseOntologyResolver` can be plugged in to ground entities against a
formal ontology, and a `FuzzyMatchingStrategy` handles natural-language
drift from formal types. Without an ontology entities get
`ontology_valid=False` and the graph is purely LLM-extracted.

**Pipeline.** ECL — *Extract, Cognify, Load*. Extract schema/entities from
sources (including relational DBs where every row becomes a node and
every foreign key becomes an edge). Cognify validates and enriches
(e.g. `ElectricCar ⊂ Car` inferred from ontology even if not stated).
Load writes to graph + vector stores.

**Update mechanism.** Public API: `remember`, `recall`, `forget`,
`improve`. Specifics of in-place update vs append-only are not documented
clearly; the focus is on re-running the pipeline and letting ontology
validation + dedupe reconcile.

**Retrieval.** Auto-routing — a dispatcher picks graph traversal vs vector
search per query. Hybrid.

**Notable design choice.** Ontology as an *additive* layer — you can start
with pure LLM extraction and layer structure in later without rewriting.
This is the only surveyed system that treats schema as a first-class
pluggable primitive.

Sources:
- https://github.com/topoteretes/cognee
- https://www.cognee.ai/blog/deep-dives/grounding-ai-memory
- https://www.cognee.ai/blog/deep-dives/ontology-ai-memory
- https://memgraph.com/blog/from-rag-to-graphs-cognee-ai-memory

---

### 1.5 OpenAI ChatGPT memory (consumer)

**Storage.** Flat list of short natural-language strings scoped per user
account. No vector DB, no RAG. Per reverse-engineering: memories are
injected as a numbered list under a `# Model Set Context` header after
the system prompt, with a bracketed date per entry:

```
1. [2024-04-26]. User loves dogs.
2. [2024-04-30]. User prefers Python over JavaScript.
```

**Update mechanism.** A server-side `bio` tool. The model writes
`to=bio <text>` and the service appends. Existing similar memories are
consolidated ("I love dogs" + "I love cats" → combined pet statement).

**Delete.** User-facing trash-bin UI, or natural-language "forget X".
Additionally "Reference chat history" (a separate mode, 2025/2026) lets
memory implicitly draw from past chats without an explicit store; the
Jan 2026 upgrade extended reliable recall to 12+ months back.

**Set / cardinality / confidence / provenance.** None of these are
modeled. It is fundamentally a flat string list.

**Entity canonicalisation.** None.

**Notable design choice.** Remarkable simplicity. The entire system is "a
list of sentences appended to the prompt." Users can read and edit every
memory. Most of the work is done by the model's in-context reasoning.

Sources:
- https://help.openai.com/en/articles/8590148-memory-faq
- https://help.openai.com/en/articles/8983136-what-is-memory
- https://openai.com/index/memory-and-new-controls-for-chatgpt/
- https://help.openai.com/en/articles/11146739-how-does-reference-saved-memories-work
- https://llmrefs.com/blog/reverse-engineering-chatgpt-memory
- https://github.com/0xeb/TheBigPromptLibrary/blob/main/Articles/chatgpt-bio-tool-and-memory/chatgpt-bio-and-memory.md
- https://simonwillison.net/2024/Feb/14/memory-and-new-controls-for-chatgpt/

---

### 1.6 Anthropic Claude memory (API memory tool + consumer)

**Two distinct products.**

**A. API `memory_20250818` tool (Claude API).** A **filesystem** scoped to
`/memories`. Tool commands are a text-editor surface:

| command       | semantics                                           |
| ------------- | --------------------------------------------------- |
| `view`        | list dir or read file (optional line range)         |
| `create`      | new file with full contents                         |
| `str_replace` | in-place text replacement, errors on duplicates     |
| `insert`      | insert at a specific line number                    |
| `delete`      | remove file or directory (recursive)                |
| `rename`      | move / rename                                       |

Storage is client-side (caller implements the handlers). There is **no
schema** — memories are arbitrary files, often markdown/XML. The
system-prompted protocol instructs the model to view the dir first,
record progress, assume interruption. Anthropic's guidance is explicitly
that the primitive enables *just-in-time context retrieval*, not upfront
loading.

"Auto Dream" / periodic reorg: Claude Code auto-reviews memory files,
prunes stale entries, reconciles contradictions, reorganises.

**B. Consumer Chat Memory + Projects (2026).** Auto-synthesised cross-
conversation memory, rolled out March 2026. Every 24 hours Claude
synthesises standalone-chat history into a summary (name, preferences,
ongoing projects, style) that is carried into new chats. Projects get an
**isolated memory space** — preferences set in a Project don't leak to
standalone chats or to other Projects.

**Set / provenance / confidence.** Not modeled in either product; it's
prose synthesis.

**Notable design choice.** Anthropic's API side commits hard to
unstructured filesystem as the primitive and puts *all* schema
responsibility in the agent / user prompt. The Auto-Dream pattern is
essentially a scheduled re-compaction / conflict-resolution pass over
free text — the "garbage collection" analog of Graphiti's invalidation.

Sources:
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool
- https://docs.anthropic.com/en/docs/claude-code/memory
- https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context
- https://claudefa.st/blog/guide/mechanics/auto-dream
- https://www.leoniemonigatti.com/blog/claude-memory-tool.html

---

### 1.7 Character.AI persistent memory

Public technical detail is thin. What I looked at:

- https://grokipedia.com/page/Character.ai
- https://aiagentmemory.org/articles/how-to-make-character-ai-memory-better/
- https://convai.com/blog/long-term-memeory (competitor with relevant detail)
- https://www.quora.com/Will-the-memory-of-AI-characters-get-better

**What's publicly confirmed.** Character.AI launched persistent memory
features in 2022; the product has "pinned memories" (short textual facts
users can explicitly set on a character) and implicit long-term recall
tied to the user-character pair. Technical architecture (storage format,
update mechanism, whether vector DB / graph / summaries) is not
documented publicly. Competitors in the same market (Convai) describe
hybrid RAG + custom ranking weighted by recency and emotional salience,
which is a reasonable prior for what Character.AI does but cannot be
attributed to them.

**Verdict.** Not enough public info to compare on the detailed axes.
Listed here for completeness; I would not base design on speculation.

---

### 1.8 Benchmark / comparison literature

LoCoMo is the dominant benchmark (long multi-session conversations).
Headline numbers are contested:

- Mem0 paper: Mem0 F1=28.64 (single), 48.93 (temporal); Mem0 blog
  claims 91.6 with token-efficient variant.
- Zep: originally claimed 84.5% → Mem0 reanalysis 58.44% → Zep counter
  75.14%.
- Letta: filesystem-only agent reaches 74% — filesystem > graph memory
  on this benchmark when agent quality is held constant.

**What benchmarks reveal about representation.**

1. **Agent quality > representation choice** at current model scale.
   Letta's experiment is the clearest evidence: held-constant filesystem
   tools with a capable agent beat fancier structured systems. Structure
   helps if and only if the agent reasons well over it.
2. **Bi-temporal / validity windows** (Graphiti style) are
   benchmark-relevant only if questions test *temporal* reasoning; they
   are near-neutral on single-hop factual queries.
3. **Hybrid retrieval** (semantic + BM25 + entity) is a consistent
   winner over any single-signal retrieval for this task family.
4. LoCoMo is small (81 QA pairs) and heavily overfit — any numbers from
   it should be treated as weak evidence. Most recent papers use it
   alongside AgentBench-like long tasks.

Sources:
- https://mem0.ai/blog/benchmarked-openai-memory-vs-langmem-vs-memgpt-vs-mem0-for-long-term-memory-here-s-how-they-stacked-up
- https://dev.to/varun_pratapbhardwaj_b13/5-ai-agent-memory-systems-compared-mem0-zep-letta-supermemory-superlocalmemory-2026-benchmark-59p3
- https://github.com/getzep/zep-papers/issues/5
- https://atlan.com/know/zep-vs-mem0/
- https://www.graphlit.com/blog/survey-of-ai-agent-memory-frameworks
- https://www.letta.com/blog/benchmarking-ai-agent-memory

---

## 2. Comparison table

| System                    | Primary repr.                          | Schema?         | Cardinality declared? | Update primitives                                        | Delete / retract                          | Provenance           | Confidence    | Canonical entities    | Retrieval                     |
| ------------------------- | -------------------------------------- | --------------- | --------------------- | -------------------------------------------------------- | ----------------------------------------- | -------------------- | ------------- | --------------------- | ----------------------------- |
| **mem0**                  | NL string + embedding (flat)           | No              | No (LLM-implicit)     | ADD / UPDATE / DELETE / NONE LLM tool-call per candidate | DELETE op by id                           | No (additive: `attributed_to`) | No  | Only in Mem0g (fuzzy) | Vector + BM25 + entity fusion |
| **mem0g**                 | Typed entity + `(s,r,o)` triples       | Graph schema    | No                    | Same 4-op prompt over edges                              | DELETE edge                               | Weak                 | No            | Yes (entity nodes)    | Graph traversal + triplet sim |
| **Letta / MemGPT**        | Free-text core-memory blocks + vector archival | Labelled blocks w/ char-limit; block body is prose | No                    | `memory_replace`, `memory_insert`, `memory_rethink`, `archival_memory_*` (tool calls) | `str_replace` with empty; rewrite block   | No (prose)           | No            | No                    | Vector over archival / recall; core in-context |
| **Zep / Graphiti**        | Triples `(s,r,o)` as graph edges       | Typed entities + edges (Pydantic) | Cardinality via time intervals (multi-edge OK) | Episode ingest → LLM dedupe + LLM contradict-check → append + stamp `invalid_at` | **Invalidate, never delete** (bi-temporal) | Episode node lineage | No (implicit via recency) | Yes (LLM-guided at ingest) | Hybrid: cosine + BM25 + BFS graph + rerank |
| **Cognee**                | Graph + vectors, ontology-grounded     | Pluggable RDF/Pydantic ontology | Declared in ontology   | Pipeline re-run; `remember` / `forget`                   | `forget` op                               | Episode lineage      | `ontology_valid` bool | Ontology + fuzzy match | Auto-routed hybrid            |
| **ChatGPT (bio)**         | Numbered list of NL strings            | No              | No                    | `to=bio` append; auto-consolidation                      | Trash-bin UI or NL "forget"               | Bracketed date       | No            | No                    | None (injected wholesale)     |
| **Claude API memory**     | Filesystem — arbitrary files in `/memories` | No (files are whatever) | No                    | `view`/`create`/`str_replace`/`insert`/`delete`/`rename` | `delete` file or `str_replace`            | No                   | No            | No                    | Agent-driven `view`           |
| **Claude consumer chat**  | Auto-synth per-user summary + per-Project isolated summary | Prose          | No                    | Auto re-synth every 24h                                  | Implicit (re-synth drops it)              | Source-chat links    | No            | No                    | Synthesis into prompt         |
| **Character.AI**          | (public) pinned short facts + implicit LTM | Unknown | Unknown               | Unknown (pinned mem user-facing)                         | User-facing edit                          | Unknown              | Unknown       | Unknown               | Unknown                       |

---

## 3. Synthesis

### 3.1 Dominant patterns

1. **Natural-language strings remain the unit of memory.** mem0, Letta,
   ChatGPT, and Claude all store *prose*. Even Graphiti's edges carry a
   summary string. No surveyed production system stores memory as typed
   attribute-value rows the way `attribute_memory` does. The strongest
   argument for this (Letta) is that LLMs are best at reading and
   editing prose; structure tends to lose more in retrieval than it
   gains in precision.
2. **Updates go through an LLM decision step.** Every system either
   (a) calls an LLM to pick ADD / UPDATE / DELETE / MERGE (mem0), or
   (b) exposes typed tool-calls that the agent itself invokes (Letta,
   Claude API, Graphiti ingest). None use "just overwrite the value."
3. **Retrieval is almost always hybrid.** Vector + keyword + some form
   of graph/entity signal. Pure kNN is rare in production.
4. **Contradictions are handled at ingest, not at query time.** Graphiti
   invalidates; mem0 runs DELETE; Claude Auto-Dream runs a scheduled
   pass. None defer conflict handling until retrieval.
5. **Entity canonicalisation is a distinguishing feature.** Systems
   that do it (Graphiti, Mem0g, Cognee) pitch it as their big
   differentiator; systems that don't (ChatGPT, Letta, Claude)
   explicitly punt it to the LLM at read time.

### 3.2 Notable disagreements

- **Structured vs unstructured.** Graphiti and Cognee go all-in on
  typed graphs. Letta and Anthropic go all-in on unstructured prose /
  filesystem. mem0 sits in the middle (flat strings + optional graph
  variant). The Letta benchmark is the strongest evidence that
  unstructured can match or beat structured given a capable agent;
  Graphiti's counter is that *temporal* queries and multi-hop
  reasoning need structure.
- **Delete vs invalidate.** mem0 and filesystems truly delete.
  Graphiti (and by extension anything with validity intervals) only
  invalidates — nothing is ever removed from history.
- **Canonicalisation cost.** Graphiti/Cognee pay an ingestion-time
  LLM cost for dedup; mem0 and Letta amortise this cost to retrieval
  time (or skip it).
- **Who owns the write decision?** Letta and Claude API: the agent
  itself, via typed tool calls. mem0: a separate post-hoc LLM pass.
  ChatGPT: a server-side `bio` tool opaque to the user model. These
  are genuinely different architectural commitments.

### 3.3 Concrete gaps in `attribute_memory` relative to the field

Given the current `(topic, category, attribute, value)` + add/delete row
design:

1. **No provenance.** Every surveyed system except mem0-classic has
   *some* notion of where a fact came from (Zep episodes, mem0-additive
   `attributed_to`, ChatGPT bracketed dates, Graphiti source edge).
   attribute_memory has none. This matters for correction and for
   trust when the LLM composes retrieved facts.
2. **No temporal validity.** attribute_memory rows are timeless. The
   state-of-the-art in this space (Graphiti) is bi-temporal. Even
   ChatGPT carries an ingestion date. Without `valid_at` / `invalid_at`
   you cannot answer "what did the user prefer last month?" or
   "is this still current?"
3. **No explicit contradiction handling.** add/delete implicitly
   requires the LLM to issue the delete. mem0 runs a dedicated
   contradiction-detection pass per ingest. Graphiti does LLM-guided
   comparison against semantic neighbours. attribute_memory lacks
   the analog — a new row for the same `(topic, category, attribute)`
   with a different `value` is silently accepted.
4. **No cardinality declaration.** Without a set/single marker on
   `attribute`, the LLM does not know whether "pet" should replace or
   append. mem0 punts this to the LLM; Cognee uses ontology cardinality;
   Graphiti uses bi-temporal edges. attribute_memory currently has no
   signal either way.
5. **No entity canonicalisation.** "Luna", "my cat Luna", and "the cat"
   resolve to three different `topic` strings. Every structured system
   surveyed solves this explicitly.
6. **Flat schema loses relational context.** The `(topic, category,
   attribute, value)` row cannot express `Luna → owned_by → Edwin →
   lives_in → Brooklyn` without denormalising. Graph systems handle this
   natively; prose systems handle it implicitly.
7. **No confidence / soft delete.** Graphiti invalidates rather than
   deletes. add/delete is hard. Retractions overwrite history.
8. **No handling of novel-style long input.** The field mostly pairs
   memory with a separate *extraction* LLM step. attribute_memory's
   public API does not distinguish ingestion from extraction.

### 3.4 Design primitives worth borrowing

Ranked by expected ROI for attribute_memory's goals:

**(a) Bi-temporal validity on rows** (from Graphiti). Add
`valid_from`, `valid_to`, `ingested_at` to the row. Expose a
"retract" that sets `valid_to=now` instead of hard-deleting. This is
cheap, orthogonal to the existing schema, unlocks temporal queries,
and gives a soft-delete audit trail. Matches what Zep pitches as its
core differentiator.

**(b) LLM-gated update pipeline with an explicit op set** (from mem0).
Before writing a row, retrieve the top-k nearest existing rows on
`(topic, category, attribute)` and ask the LLM to classify the new
candidate as ADD / UPDATE / DELETE / NOOP. This is the single most
load-bearing pattern in mem0 and it plugs cleanly into the existing
attribute schema without changing storage. Pair with (a) so UPDATE
invalidates the old and inserts the new.

**(c) Provenance per row** (from Graphiti episodes /
mem0-additive). Store the `event.id` or message-range that produced the
row. Cheap to add and pays dividends at correction time. Optional
extension: record the extraction model and prompt version so you can
re-extract on upgrade.

**(d) Cardinality hint on `attribute`** (from Cognee-style ontology).
One bit: `is_set` / `is_scalar`. Lets the LLM-gated pipeline in (b)
make the right UPDATE-vs-ADD decision without re-asking. Matches how
set-valued attributes (allergies, pets, languages) empirically behave
differently from scalar ones (name, age).

Optional / further-out:

**(e) Entity canonicalisation on `topic`** (from Graphiti). Give
`topic` a stable UUID distinct from its surface form; resolve surface
forms via an alias table. Unlocks "Luna" ≡ "my cat" ≡ "the cat" at
retrieval. Higher cost — needs its own dedupe pass — so only worth it
once (a)–(d) are in.

**(f) Scheduled reorg pass** (Auto-Dream style). Periodically sweep
and reconcile contradictions that escaped ingest-time handling.

The minimal package — (a)+(b)+(c)+(d) — keeps the current flat row
model while closing the gap to mem0- and Graphiti-level hygiene, at
the cost of one extra LLM call per ingest and four extra columns.
