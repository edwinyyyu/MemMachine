# Evaluation

> Retroactive design document: it describes the system as built, written
> after the decisions it covers were made.

## What the benchmarks measure

The benchmarks in `evaluation/` measure answer quality: whether a model
given MemMachine's search results answers questions about earlier
conversations or documents correctly. Each run follows the same steps:

1. **Ingest** a dataset's conversations or documents into MemMachine.
2. **Search** MemMachine for each question.
3. **Answer** each question with a language model, given the search results.
4. **Judge** each answer against the reference with a language model, and
   aggregate the scores by question category.

The datasets cover long multi-session conversations (LoCoMo, LongMemEval,
BEAM) and multi-hop questions over documents (HotpotQA, WikiMultiHop).
[`evaluation/README.md`](../evaluation/README.md) has the commands.

The retrieval-agent suite runs the server's library in its own process
rather than calling a server, and measures episodic long-term memory on the
declarative backend, without short-term or semantic memory.

## Targets

The retrieval-agent suite compares three targets on the same questions:

- **memmachine**: one search per question.
- **retrieval_agent**: agent mode, where a language model plans the searches.
- **llm**: no memory; the model gets the dataset's full context, as a
  reference point.

The older `episodic_memory` suite runs LoCoMo and LongMemEval; its REST
scripts call a running server and, by default, use both episodic and
semantic memory.

## Conditions that move a score

A score is comparable only with scores taken under the same conditions:

- the answering model, the judging model, and the embedder;
- the reranker, the search limit, and context expansion;
- the long-term backend and its settings, and which memory types are on;
- the size of the context the answering model receives.

A result quoted without these is not comparable to another result.

## Capacity

The benchmarks measure quality on one dataset at a time. Capacity planning
for a deployment's traffic uses the sizing calculator in `tools/sizing`,
which turns a peak request rate and traffic mix into server, database, GPU,
and storage requirements; the server's metrics (see
[server and API](server_and_api.md#observability)) show whether a deployment
keeps up.
