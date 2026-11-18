# Prompt templates

This page lists every prompt and text format the benchmark sends to a model, with the exact
strings. The strings come from the code: `rbench/rerankers.py`, `rbench/embedding.py`, and
MemMachine's EventMemory, which renders the documents. The ones the results use are marked
**used**. The rest were tested and not adopted; METHODOLOGY section 8 has why.

## How the reranker was run (no vLLM)

The Qwen3 rerankers (`Qwen/Qwen3-Reranker-4B` and `-8B`) ran on **DeepInfra's hosted inference
API**, not on vLLM, so no template file was involved. Each request sends one query and up to
32 documents (at most 24,000 pair tokens) to `https://api.deepinfra.com/v1/inference/<model>`:

```json
{"queries": ["<query>", "..."], "documents": ["<document>", "..."], "instruction": "<instruction>"}
```

DeepInfra wraps each pair in the model card's format, shown below. It returns P(yes), and
rbench converts that to log-odds. rbench rebuilds the same text locally to count tokens, and
the tokens DeepInfra bills match that count (`python -m rbench smoke`).

## The reranker's format

The format is the Qwen3-Reranker model card's. Every pair reads as follows; the three
placeholders are filled in, and nothing else changes:

```text
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {instruction}
<Query>: {query}
<Document>: {document}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

(It ends with `</think>` followed by a blank line.)

**`{instruction}`**
- **used**, the memory instruction (`q4-mem`, `q8-mem`): `Given a user's query, retrieve relevant passages that provide the facts, preferences, or events needed to answer it.`
- the model card's (`q4`, `q8`): `Given a web search query, retrieve relevant passages that answer the query`
- six candidates tested on held-out questions, none adopted (`rerankers.TUNING`): `RERANKER_INSTRUCTION.md` in the results bundle lists each one with its results.

**`{query}`**
- **used**: the question as it is, which is what EventMemory passes its reranker. For example, `What should I serve for dinner this weekend with my homegrown ingredients?`
- tested, not adopted: as the user's message, `User: "<question>"` (`-user`), and that headed by the question's date, `[Tuesday, May 30, 2023, 11:56 PM] User: "<question>"` (`-date-user`).

**`{document}`**: a retrieved segment exactly as EventMemory renders it for its reranker
(`EventMemory.string_from_segment_context`, `FormatOptions(time_style="short")`). That is the
date and time, the speaker, then the text as a JSON string:

```text
[Tuesday, May 23, 2023, 5:46 AM] User: "I've been using basil and mint in my cooking lately. I've even harvested some cherry tomatoes from my garden. Do you have any suggestions for companion plants that could help my cherry tomatoes grow better?"
```

A seed's window (expand_context above 0) renders several such lines, one per segment.

## Running the same reranker on vLLM

`qwen3_reranker_memory.jinja`, in this folder, is vLLM's own `qwen3_reranker.jinja`
(vllm-project/vllm, `examples/pooling/score/template/`, last changed in commit 70c00163ffa8,
2026-05-14) with **one change**: its default instruction is the memory instruction instead of
the card's. `tests/test_templates.py` renders it with Jinja and checks two things:
- **By default**, it produces exactly the text above with the memory instruction.
- **With a request's `instruction` field**, that instruction overrides the default, as in vLLM's
  stock template.

Serve the original Qwen3 Reranker with it. The `--hf_overrides` value is vLLM's documented one
for this model:

```sh
vllm serve Qwen/Qwen3-Reranker-4B \
  --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}' \
  --chat-template qwen3_reranker_memory.jinja
```

MemMachine's `cohere` reranker provider can point at a Cohere-compatible rerank API through
its `base_url`, and vLLM serves one. That provider sends only the query and the documents, with
no instruction field. Behind it, vLLM's stock template therefore scores with the model card's
instruction, and this file's default makes it score with the memory instruction instead.
(Clients that can send the field can instead keep vLLM's stock template and pass
`"instruction": "..."` in each `/score` or `/rerank` request, per vLLM's scoring docs.)

We have not run vLLM ourselves. The template text is verified to be identical to what
DeepInfra received from us. That vLLM's sequence-classification conversion gives the same
scores as DeepInfra's is expected (it reads the same yes/no logits), but not verified.

## The embedders

### embeddinggemma-300m, the first stage (**used**: `gemma300m-di-raw`, no prompts)

**Query:** the question as it is.

```text
What should I serve for dinner this weekend with my homegrown ingredients?
```

**Documents:** each segment as EventMemory's WholeTextDeriver renders it for embedding, with
MemMachine's default options. That is the date (no time), the speaker, then the text as a
JSON string:

```text
[Tuesday, May 23, 2023] User: "I've been using basil and mint in my cooking lately. I've even harvested some cherry tomatoes from my garden. Do you have any suggestions for companion plants that could help my cherry tomatoes grow better?"
```

Tested on embeddinggemma and not adopted (no setup beat this one clearly, or it was worse):
- the model card's prompts: query `task: search result | query: <question>`, document `title: none | text: <document>`, and either one alone;
- the query as `User: "<question>"`, and as `[Tuesday, May 30, 2023] User: "<question>"`;
- documents without the date (`User: "<text>"`), without the speaker (`[<date>] "<text>"`), or without both (`"<text>"`). The variants without the speaker scored slightly higher, but EventMemory's format was kept (METHODOLOGY section 8).

### Qwen3-Embedding 0.6B, 4B and 8B (compared, not used)

**Query:** the card's format, with no space after `Query:`, as in the card's code:

```text
Instruct: {task}
Query:{question}
```

Here `{task}` is the card's `Given a web search query, retrieve relevant passages that
answer the query`, or the memory instruction above. A third setup sends the question alone,
with no instruction. Each of the three was also tried with the question as
`User: "<question>"`.

**Documents:** with no instruction, as EventMemory renders them for embedding (as above).
