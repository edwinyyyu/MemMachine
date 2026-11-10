# Methodology

This document explains what the benchmark measures, why it measures it that way, and what each choice rules in or out. Each module's docstring restates the part it implements. The tests in `tests/` check the parts where a silent mistake would change numbers, including that replaying configurations from cached scores gives exactly what MemMachine's EventMemory returns.

## 1. The question

Does reranking EventMemory's results with a cross-encoder (Qwen3-Reranker-8B) improve retrieval enough to justify its cost and latency? If the gain is marginal, the service could make reranking optional instead of on by default.

To answer that, the benchmark also measures how the gain depends on:
- the size of the units retrieved (EventMemory's segment length);
- whether the reranker scores the retrieved segments themselves or their expanded windows;
- how deep it reranks;
- how many tokens the reader LLM receives;
- the kind of question.

It also reports the reranker's own cost, and total cost over a continuous range of LLM prices.

LongMemEval-M is evaluated first. The other datasets prepared earlier are inactive until we decide which to add back (`rbench/datasets/__init__.py`).

## 2. What is measured

**No retrieved item is ever counted as a "hit".** Every measure is a per-query value computed from the gold. Reported numbers are means over test queries, each query weighted equally (a macro-average); nothing is pooled across queries. The full definitions are in the glossary at the top of `rbench/metrics.py`.

**Gold.** LongMemEval marks each turn that holds the answer (`has_answer`). A question's marked turns together are its one answer, and all of them are needed, so every-gold coverage is the headline measure (break-even, single-measure charts).

**Three readings of a gold turn.** Each is a separate metric, reported side by side and never combined:

| reading | a gold turn counts once… | use |
|---|---|---|
| complete | every segment of the turn has been read | strict; right for the short user turns (94% of the evidence, median 67 tokens) |
| touched | any segment of the turn has been read | a potential hit; LongMemEval's own notion of a retrieved unit; optimistic for long assistant turns (median 248 tokens) |
| fraction (recall) | — the share of the turn's characters read | between the two |

**Per query, at reader budget B:**
- first-gold coverage: some gold turn is read within B tokens (complete, or touched);
- every-gold coverage: every gold turn is read within B (complete, or touched);
- recall: the share of gold characters read;
- precision: the share of characters read that are gold;
- F1: computed per query, then averaged;
- the share of evidence turns found (`share`: complete; `share_touch`: touched): per query, the share of its evidence turns read within B, each turn counted once, averaged over queries (the user's decision, 2026-11-09, beside the others; every-gold stays the headline). With one evidence turn it equals every-gold; touched is generous on long assistant turns and complete is strict (`rbench/report/readings.py`);
- the same within the first k segments;
- coverage at equal counts instead of tokens: at the same vector search limit L (seeds retrieved), seeds kept after reranking k (contexts read), and segments read m (EventMemory's `max_num_segments`);

**Reader tokens are the tokens of the string EventMemory would hand the LLM** (`reading.py`):
- segments are admitted in EventMemory's order under a growing segment limit;
- the admitted set is rendered as EventMemory renders it: chronological components, one `[date and time] speaker:` header per run of a turn's segments, text as JSON;
- overlapping windows are counted once;
- a prefix costs the most any step up to it cost: the rendered string can shrink by a header when a segment joins two windows, but a reader that stops at the first segment that does not fit never gets there.

The cost is tracked incrementally with an additive token model. Tests check it against a from-scratch recomputation (exact) and against tokenizing the rendered string (within 3%, under 1% on average; within 1.5% on real LongMemEval-M contexts).

**Which contexts are in the query result.** EventMemory's `string_from_query_result` restricts every context of the result to the admitted segments before joining components, so a window can link components even if the segment limit stopped before it. A reranked configuration's result holds all its L seeds (its vector search limit), so every window links from the start. The first stage without a limit reads as many seeds as the budget takes, and that count is its vector search limit, so a window links only once admission reaches it.

**Why there is no reader LLM and no judge.** An answering model adds its own variance and cost, and it mixes retrieval quality with reading skill. Coverage measures what retrieval controls. An end-to-end answer-accuracy check on a few final configurations remains possible if wanted.

## 3. The system under test: EventMemory, faithfully

The benchmark runs MemMachine's EventMemory itself, from the repository it lives in: a branch based on `feat/horizontal-scaling` (at `043fb7538` when it was branched; its changes since `bbcc8c3da` are partition handles and store plumbing, none of them in retrieval). Each history is encoded into its own partition in MemMachine's SQLite segment and vector stores (`rbench/memory.py`). The benchmark supplies only what a deployment supplies, as implementations of MemMachine's interfaces: an exact vector search engine, the embedder (MemMachine's `OpenAIEmbedder`) and a reranker that reads the score cache.
- **Segmentation:** MemMachine's `TextSegmenter` (LangChain's recursive splitter with EventMemory's separators, at most N characters), applied to each turn (event) separately. On 46,390 LongMemEval-M turns its output equals the separately pinned LangChain splitter's at every length tested. Option: the simple chunker (`cut<N>`, `memory.FixedCutSegmenter`), a Segmenter that cuts each turn every N characters regardless of content (the last piece shorter, nothing stripped); stage `lme-cut`. Recursive splitting at a limit of N gives segments well under N on average (seg500's mean segment is about 300 characters on the 8 lme-test histories), so the two are compared through their measured sizes, not their N.
- **Events and their order:** speaker User or Assistant. EventMemory orders a partition by timestamp. LongMemEval dates sessions only to the minute, gives turns no time, and its files are not in date order (40 of 94 sampled M histories; most also have sessions starting in the same minute). So sessions are sorted by date (ties in file order), and each turn's timestamp is its session's date plus one microsecond per earlier turn in the history: every turn is ordered, and nothing rendered or embedded changes.
- **Embedding text** (`WholeTextDeriver`): `[<date>] <speaker>: "<JSON text>"`, so a session's segments embed differently in each history it appears in. Queries embed the question, with Qwen3-Embedding-8B's instruction prefix where that setup is chosen (below). The paid stages embed with Qwen3-Embedding-8B on DeepInfra; the free local runs use embeddinggemma-300m on this machine.
- **Embedding prompts (embeddinggemma, the comparison first stage):** the model card recommends `task: search result | query: ` for queries and `title: none | text: ` for documents; they are not required. MemMachine's sentence-transformers embedder uses only the query prompt (embeddinggemma's `default_prompt_name` is null), and its OpenAI-compatible embedder uses none. Four setups (`gemma300m`: both; `-q`: query only; `-raw`: none; `-d`: document only) are compared on the first stage alone over the first 200 questions of the sample order (the pilot's 100 and the next 100), and the best is used, since the strongest baseline tells the most about what a reranker adds. Telling them apart does not need whole histories: each question searches its evidence sessions plus 25 random other sessions of its own history, built exactly as the benchmark builds histories, and the setups are compared on how they rank the gold segments (`python -m rbench compare-embedders`, about half an hour locally instead of about 8 hours). Result on 200 questions (2026-11-06, `results/summary/embedders__longmemeval_m__seg500__200q__gemma300m.md`): the prompts make a small difference, seen at the top of the ranking. Taken as a 2 × 2 design (a framing chosen after a first look at the pilot's 100, which the 200 include, so this is not an independent replication), the query prompt makes it slightly worse (every gold turn in the top 5 segments −0.05, MRR −0.06, intervals excluding 0); the document prompt's effect (+0.01) cannot be told from none. The benchmark therefore embeds with no prompts (`gemma300m-raw`, `gemma300m-di-raw`), which is tied for best and exactly what MemMachine's OpenAI-compatible embedder sends. Choosing the setup on the evaluated questions can only flatter the first stage, which makes the reranker's measured gain conservative.
- **The embedder: Qwen3-Embedding-8B** (the user's decision, 2026-11-09: production may use it; embeddinggemma-300m stays for comparison). DeepInfra, $0.010 per million tokens, at 1,024 dimensions (Matryoshka; a vector truncated to its first 1,024 and re-normalized). Documents are embedded as EventMemory sends them, with no instruction; queries carry the model card's `Instruct: {task}\nQuery:` prefix unless compare-embedders shows the plain question does better (`qwen3-8b-di` / `qwen3-8b-di-raw`). Before anything is embedded at scale, `python -m rbench embed-checks` must pass (cents): the card's example scores reproduce within 0.005, a text embeds the same alone and in a batch, the 1,024-dimension vectors equal the truncated full ones, and billed tokens equal our count. Then `compare-embedders --embedders qwen3-8b-di,qwen3-8b-di-raw,gemma300m-di-raw` on the pilot's 100 reduced haystacks; if Qwen3 does not beat embeddinggemma, that is reported before going on. Embedding LongMemEval-M costs about $6.29 per segment length (628M tokens) against embeddinggemma's $1.37, and its vectors take about 15 GB per segment length (float16).
- **First stage:** `EventMemory.query` without a reranker, over the question's own history (~16k segments at 500 characters), with exact search: no approximate index, so no index recall loss is mixed into the result. Its vector search limit is 400 seeds, enough to fill the largest reader budget.
- **Expansion** (`expand_context` E; EventMemory's default is 0): MemMachine's own, E // 3 segments before the seed and the rest after, from its segment store. E = 1, 2, 3 and 6 are replayed.
- **Reranking:** Qwen3-Reranker in its model card's format, called as EventMemory calls any reranker: `score(question, contexts rendered with date and short time)`. With E > 0 EventMemory scores each seed's window (expand, then rerank). The pairs a paid run scores are collected by querying EventMemory with a reranker that records what it is asked. Replays read the cached scores; `tests/test_memory.py` checks that a replay at any vector search limit gives exactly `EventMemory.query`'s order.
- **Depth is the vector search limit:** EventMemory reranks every seed its vector search returns, so a reranked configuration reads only its reranked seeds. The first stage alone reads as many seeds as the budget takes, or, as `first_stage@nL`, only its top L, for comparison at the same vector search limit.
- **Reranker input format:** the model card's format (system prompt, instruction, query, document, empty think block). The user's rule was to use whichever of it and raw pairs does better on a small sample with the 0.6B. Raw pairs are what MemMachine's `CrossEncoderReranker` (a sentence-transformers CrossEncoder) sends: `tokenizer(query, text)`, which for Qwen3's tokenizer is the two texts concatenated (`rerankers.RAW_PAIRS`, scored the same way). Stage `format-test` compared them, free, on the local 0.6B over 100 LongMemEval-S questions (the M questions over smaller histories), reranking the same top L seeds at 500 characters (2026-11-09). The card's format won at every reading. At L 50, raw pairs read all evidence in full for 0.27 fewer of the questions at 500 reader tokens (95% interval [−0.39, −0.16]) and 0.25 fewer at 1,000 ([−0.36, −0.16]), and some evidence for 0.20 fewer at 500, falling below the first stage alone (0.26 vs 0.51 at 500); at L 20 the gaps are −0.13 and −0.19 at 500; at L 10 or past 2,000 tokens the two tie. Raw pairs cost about 40% fewer reranker tokens, but the card's format is what the benchmark uses, and the only one DeepInfra serves (it applies the format itself). A service that sends raw pairs to Qwen3-Reranker should expect results well below these.

**Processing is one history at a time.** Histories share sessions but not their dates, so nothing is shared across histories. Each history's stores are kept on disk (vectors as float16, ~26 MB per history at 500 characters); a history is encoded once, and one interrupted mid-way is encoded again. The segment table the simulation reads (texts, order, token counts) is read back from the segment store and refreshed whenever the history is encoded again.

**The inactive datasets run through the same path** (`rbench/partitions.py`, `rbench/simulation.py`). A query searches one partition chosen by its dataset's scope: its own document (doc), its candidate documents (scope_docs: SWE-bench's repository snapshot), or the whole corpus (corpus, one partition every query shares). A document is an event with no producer (`NullContext`) and one text block, in corpus order; non-chat text has no author or time, so documents are embedded and rendered with no date (`FormatOptions(date_style=None, time_style=None)`): the embedding text, the reranker's input and the reader's context are the segment's JSON text alone. Gold is character spans in event texts (LongMemEval's gold turns are whole-turn spans). Each segment's offsets in its event are found in order, since TextSegmenter's segments are its text's substrings with edge whitespace stripped; a span is trimmed to its non-whitespace extent and maps to the segments overlapping it. On every query of the seven datasets, at seg500 and cut500, every gold span maps to segments, none empty (`tests/test_partitions.py` checks the tiny samples). In a corpus partition a window can reach across a document boundary into the neighbouring document, as EventMemory's expansion does whenever documents share a partition. Where a window joins consecutive segments of one event, EventMemory concatenates them, so whitespace TextSegmenter stripped at a boundary is lost from the rendered text (production behaviour; the gold mapping is unaffected).

## 4. What is varied, and what is replayed for free

| factor | values | how |
|---|---|---|
| segment length | 500 characters (core); 250, 1,000, 2,000 | one embedding per length |
| chunker | EventMemory's TextSegmenter (`seg<N>`); the simple chunker (`cut<N>`, optional stage `lme-cut`) | one embedding per chunking |
| reranking | none; rerank the seeds; rerank each seed's window (expand, then rerank) | seeds scored; windows scored at `expand_context` 3 |
| reading (`expand_context`) | the seeds (0), or their windows at 1, 2, 3, 6 | replayed; free except expand-then-rerank, which needs its windows scored |
| rerank depth = vector search limit L | every L up to the pool D: the depth that holds all of a question's evidence for 95% of the sample's questions, measured per segment length and chunker after embedding (`rbench/depth.py`; D at 99% is reported too). The whole pool is scored, so 5, 10, 20 (EventMemory's default), 30, 50, … up to D replay | replayed from cached scores |
| first stage's vector search limit | the same L values, or none (as many seeds as the budget reads) | replayed |
| reader budget | continuous: every ranking is read to the end of its pool, so no curve stops short of its plateau (500–4,000 tokens is the realistic range); charts mark every 500, recall and precision are measured every 250 up to 16,000 | replayed |
| seeds kept (k), segments read (m) | any count | replayed |

**Pool depth.** Each question's pool (the vector search limit the reranker scores, all of it) is deep enough that reading it reaches the end of the reader-token axis (16,000 tokens) for 95% of questions, per segment length, chunker and scoring position (`rbench/depth.py`): curves keep rising across the axis instead of flattening because the pool ran out. The user's rule (2026-11-09): the curve should get close to about 0.95 on LongMemEval-S without flattening early, and where that is not achievable at reasonable cost, the pool is picked by price, the axis ending at 16,000 tokens or less. Measured at 500 characters: on LongMemEval-S (100 questions) the seed pool of 221 holds all of a question's evidence for 96% of questions and the window pool of 116 for 97%, so its curves can reach about 0.95; on LongMemEval-M (8 histories, before the paid embeddings) the seed pool of 199 holds it for 7 of 8 (one question needs 361 seeds), and that ceiling is accepted rather than chased. `estimate` reports each pool's evidence share beside its depth, and the charts' bound line shows the ceiling: where it levels off below 1, the remaining evidence is outside the pool.

Units follow EventMemory's settings: segment length in characters, `expand_context` and the vector search limit in counts. Tokens measure what things cost: reader budgets, reranker input, and the first stage's head (16,384 tokens of seed text, enough to fill the largest budget). Pools and depths are seed counts rather than tokens because the vector search limit is a count; equal reranker input across segment lengths is compared through each configuration's measured reranker tokens (§5 of the report). A segment length is a limit, not a size: the simulation records each history's measured mean segment (characters and tokens) and the mean unit each configuration reads (the seed, or its window), and the reports state those measured sizes. Where a comparison pairs units by size (the expansion comparison, §11), it pairs by measured size and flags pairs further apart than `config.EVAL.size_tolerance` (25%).

Qwen3-Reranker scores each pair independently, so a pool scored to depth D answers every shallower depth exactly. The first stage runs once per question, deep enough for every configuration. (A real vector index asked for the top 20 may return slightly different results from the top 20 of a deeper search, but that is an index effect outside this evaluation.)

## 5. Normalizations

| held equal | where |
|---|---|
| reader tokens (the rendered context) | report §3–4, `coverage_tokens_*.png` |
| reranker tokens, as an API bills them and as a self-hosted server with a prefix cache computes them | §5–6, `rerank_cost.png` |
| total cost, at any LLM-to-reranker price ratio | §7 (exact break-even values), `cost_saving.png` (% saved at every LLM price and coverage target, with the break-even LLM price per target drawn over it); both compare always reranking with never reranking |
| segments read (k) | `rpf_k.png`, §12 |
| vector search limit L, seeds kept after reranking k, segments read after expansion m | §13, `coverage_counts.png` |
| what a budget implies for EventMemory's settings (`max_num_segments`, `vector_search_limit`) | §14 |

**Break-even.** Scaling both prices by the same factor changes no decision, so whether reranking pays depends only on the price ratio. For each every-gold coverage target:
1. choose the cheapest configuration without a reranker (any segment length and expansion) and the reranked configuration that pays off at the lowest ratio (seeds or windows reranked, any expansion, depth and segment length);
2. compute r* = reranker tokens ÷ reader tokens saved;
3. bootstrap questions for a 95% interval, each resample making its own choice in step 1.

Budgets are exact: a target's budget is the order statistic of the per-question costs, not a grid point.

There is no dev split (§6), so the choice and the measurement use the same questions. The minimum over many more reranked than unreranked configurations leans optimistic for the reranker; the interval includes the selection's variability.

**Latency.** Call times are recorded as a by-product only and reported in §15. They do not represent a self-hosted deployment (DeepInfra requests run 16 at a time, with queueing and any retries), and nothing is spent to measure them.

**Settings are chosen overall, never per question** (the user's rule, 2026-11-09). Every choice the reports make (the vector search limit, seeds kept, segment length, expansion, scoring position, and each side of the cost comparison) is made once for all questions at a given reader budget or coverage target, because a service runs one configuration. The pool rule is overall too: a depth that fills the axis for 95% of the questions. The only per-question optima are the true bounds (`rbench/bounds.py`), labelled as bounds.

## 6. Statistics

- **No split** (decided 2026-11-06): all 470 questions are evaluated. Samples take questions in a fixed hash order, so the pilot's 100 are part of the core's 470.
- **Selection:** choices among configurations are made and measured on the same questions; where a choice is reported (§5–6 of the report), the report says so.
- **Intervals:** 95% paired bootstrap intervals resample questions (1,000 resamples). Histories are independent: evidence sessions are essentially unique to their question, with 8 of 940 shared.
- **Many comparisons:** single intervals that just exclude zero are hypotheses. Conclusions need patterns that hold across segment lengths and question types.

## 7. Threats to validity

| threat | handling |
|---|---|
| A reranker can only reorder what the first stage retrieved | the headroom table (§2 of the report) gives, per vector search limit, how often all of a question's evidence is among the seeds retrieved; the coverage charts show the first stage cut to the reranker's limit beside the reranker |
| Choosing the best of many configurations on the same questions flatters the winner | stated with every chosen number; bootstrap resamples re-choose; conclusions rest on patterns across segment lengths and question types |
| LongMemEval's timestamps are minute-level and its files out of date order | sessions sorted by date, turns ordered by unrendered microsecond increments (§3); tests check the order is EventMemory's |
| Whole-turn gold is strict for long assistant turns | complete, touched and fraction are reported separately |
| Questions and evidence sessions involve LLM generation | human-rewritten and human-screened; filler is real user–assistant chat (ShareGPT, UltraChat) |
| The simulation might not match EventMemory | it runs MemMachine's EventMemory; the two replayed pieces (admission at every segment limit, the reader-cost model) are tested against `build_query_result_context` and `string_from_query_result`, and replayed reranking against `EventMemory.query` |
| Token counts differ between tokenizers | reader budgets are embeddinggemma tokens, reranker costs Qwen3 tokens; an LLM bills in its own tokenizer, so budgets are a yardstick, not an exact bill |
| Spend | `--max-usd` caps total DeepInfra reranker spend across models and runs; it is checked before every request group, and each request is cached as it returns. Embedding spend is not capped; `estimate` prices it (Qwen3-Embedding-8B: about $6.29 per segment length on LongMemEval-M), and the user approves the price before any paid run |
| Interruptions | reranker scores are saved as each request returns; embeddings in shards of 1,024 segments; reruns skip finished work |

## 8. Before a paid run

1. `python -m pytest -q tests`
2. `python -m rbench run lme-test`: local and free, 8 LongMemEval-S histories end to end.
3. `python -m rbench smoke`: the rerankers on DeepInfra (fractions of a cent).
4. `python -m rbench embed-checks`: Qwen3-Embedding-8B reproduces its model card's scores, embeds a text the same alone and in a batch, gives the 1,024-dimension vectors at unit norm and as the full ones truncated, bills the tokens we count, and gives the same vectors through MemMachine's OpenAIEmbedder (the path ingest uses) as through the API directly (cents). All five must pass.
5. `python -m rbench compare-embedders --embedders qwen3-8b-di,qwen3-8b-di-raw,gemma300m-di-raw`: the pilot's 100 reduced haystacks, with and without Qwen3's query instruction, against embeddinggemma (cents). The better instruction setting becomes `embedding.PAID`; if Qwen3 does not beat embeddinggemma, stop and report.
6. `python -m rbench estimate lme-pilot lme-core lme-sizes lme-cut`: embedding and reranking prices per stage, for the user to approve.
7. `python -m rbench run lme-pilot --max-usd 3`: 100 questions (embedded with Qwen3, which also measures each pool's depth and evidence share); reused by the core.
8. Review, then `run lme-core`, `lme-sizes` and `lme-cut`, raising the cap stage by stage.
