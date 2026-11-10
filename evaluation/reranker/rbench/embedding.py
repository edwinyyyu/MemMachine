"""First-stage embedders: their endpoints and prompts, and direct embedding for the checks
(compare-embedders, embed-fidelity). The benchmark itself embeds through MemMachine's
OpenAIEmbedder inside EventMemory (memory.QueryCachingEmbedder).

An embedder is an OpenAI-compatible embeddings endpoint plus the model's prompts and
context limit. Two serve google/embeddinggemma-300m:
  gemma300m     llama.cpp's llama-server on this machine (127.0.0.1:9931), free, ~7-10k tokens/s
  gemma300m-di  DeepInfra's hosted model, $0.002 per million tokens (checked 2026-11-05)
each with four prompt setups (SETUPS: "", "-q", "-raw", "-d"); the benchmark uses "-raw".
The client adds the model's prompts itself, as sentence-transformers does. Vectors live
under the embedder's name, so the two never mix; `preflight embed-fidelity` compares them.
Adding an embedder (another server, an API) adds a first stage whose results come out of the
same pipeline: the benchmark's first-stage tables are an embedding and chunking benchmark in
their own right.

Inputs longer than the embedder's context are averaged as MemMachine's embedders do
(#1449): the text is cut greedily into `max_input_chars` pieces, each piece is embedded
with the prompt, and the vectors are averaged weighted by piece length, then normalized
(the vector store scores by cosine). A piece still over the context is cut at the
context, as sentence-transformers truncates. Only long texts and queries (SWE-bench issues)
take this path.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path

import httpx
import numpy as np

from rbench.dbs import replace_atomically
from rbench.paths import SECRETS, VECTORS
from rbench.schema import queries
from rbench.text import GEMMA, QWEN_EMBED, token_starts


@dataclass(frozen=True)
class Embedder:
    name: str
    url: str  # base URL; requests go to url + route
    query_prompt: str
    doc_prompt: str
    max_tokens: int  # context, special tokens included
    max_input_chars: int  # MemMachine's max_input_length for the averaging path
    route: str = "/v1/embeddings"
    model: str | None = None  # the request's "model" field, for hosted APIs
    key_file: str | None = None  # file under .secrets/ holding a bearer token
    usd_per_mtok: float = 0.0
    request_inputs: int = 32
    in_flight: int = 3  # the shared local server has 4 slots; one stays free for other clients
    doc_store: str | None = None  # where document vectors live: shared by setups with the same server and document prompt
    dimensions: int = 768  # stored vector size
    request_dimensions: int | None = None  # what the API is asked for (None: dimensions); longer vectors are truncated
    search_dimensions: int | None = None  # what searches use (None: dimensions): the stored vectors' leading ones, re-normalized
    send_dimensions: bool = False  # whether direct requests carry a dimensions field (hosted Matryoshka models)
    tokenizer: str = GEMMA  # counts the tokens an input is billed for
    special_tokens: int = 2  # added per input (BOS and EOS for embeddinggemma; EOS for Qwen3)

    @property
    def documents(self) -> str:
        return self.doc_store or self.name


QUERY_PROMPT, DOC_PROMPT = "task: search result | query: ", "title: none | text: "  # the model card's recommended prompts
# max_input_chars is MemMachine's max_input_length (code points per embedded piece). 6,000 let a
# dense SWE-bench issue reach 2,160 tokens, past the 2,048-token context (llama-server refuses
# it); 3,000 is the largest round value whose pieces fit for every query of every dataset (at
# most 1,827 tokens, measured 2026-11-09; 3,500 still gives one of 2,225). Segments (at most
# 2,000 characters) and LongMemEval's questions are shorter, so nothing else changes.
GEMMA_LIMITS = dict(max_tokens=2048, max_input_chars=3000)
SERVERS = {
    "gemma300m": dict(url="http://127.0.0.1:9931"),
    "gemma300m-di": dict(url="https://api.deepinfra.com/v1/openai", route="/embeddings", model="google/embeddinggemma-300m",
                         key_file="deepinfra_api_key", usd_per_mtok=0.002, request_inputs=64, in_flight=16),
}
# Prompt setups per server. The bare name uses both recommended prompts. "-q" is MemMachine's
# SentenceTransformerEmbedder (query prompt only: embeddinggemma's default_prompt_name is null, so
# documents get none). "-raw" is MemMachine's OpenAIEmbedder (no prompts). "-d" is the document
# prompt alone, for completeness. Setups with the same document prompt share document vectors.
SETUPS = {"": (QUERY_PROMPT, DOC_PROMPT), "-q": (QUERY_PROMPT, ""), "-raw": ("", ""), "-d": ("", DOC_PROMPT)}
EMBEDDERS = {
    f"{server}{suffix}": Embedder(f"{server}{suffix}", query_prompt=q, doc_prompt=d,
                                  doc_store=server if d else f"{server}-nodoc", **GEMMA_LIMITS, **settings)
    for server, settings in SERVERS.items() for suffix, (q, d) in SETUPS.items()
}
# Qwen3-Embedding-8B on DeepInfra (the user's decision, 2026-11-09: the default first stage, at
# 1,024 dimensions; production may use it). Model card (checked 2026-11-09): queries are
# "Instruct: {task}\nQuery:{query}" (the card's Python has no space after "Query:"), documents get
# no instruction, last-token pooling then L2 normalization, 32k context, Matryoshka 32-4,096.
# DeepInfra $0.010 per million tokens; its "Custom Instruction" field (prepended to every input)
# stays empty, so queries are formatted here. The instruction setting is chosen by
# compare-embedders (the user's rule: whatever performs best): "-raw" sends no instruction.
QWEN3_TASK = "Given a web search query, retrieve relevant passages that answer the query"  # the card's example
QWEN3_QUERY = f"Instruct: {QWEN3_TASK}\nQuery:"
QWEN3 = dict(url="https://api.deepinfra.com/v1/openai", route="/embeddings", model="Qwen/Qwen3-Embedding-8B",
             key_file="deepinfra_api_key", usd_per_mtok=0.010, request_inputs=64, in_flight=16, max_tokens=32768,
             max_input_chars=40000, dimensions=1024, send_dimensions=True, tokenizer=QWEN_EMBED, special_tokens=1,
             doc_store="qwen3-8b-di")
EMBEDDERS |= {"qwen3-8b-di": Embedder("qwen3-8b-di", query_prompt=QWEN3_QUERY, doc_prompt="", **QWEN3),
              "qwen3-8b-di-raw": Embedder("qwen3-8b-di-raw", query_prompt="", doc_prompt="", **QWEN3)}
# Full width (4,096): compare-embedders --widths scores 4,096, 2,048 and 1,024 from one response
# (truncated and re-normalized), and keeps these vectors in their own cache and document store.
FULL = 4096
EMBEDDERS |= {f"{name}-{FULL}": replace(EMBEDDERS[name], name=f"{name}-{FULL}", dimensions=FULL,
                                         doc_store=f"qwen3-8b-di-{FULL}") for name in ("qwen3-8b-di", "qwen3-8b-di-raw")}
# max_input_chars 40,000: pieces of dense code (about 1.7 characters per token) stay under 32k tokens.
PAID = "qwen3-8b-di"  # the paid stages' first stage; the instruction setting is pending compare-embedders

# No prompts (compare_embedders.py, 200 LongMemEval-M questions, 2026-11-06): the query prompt
# makes the first stage slightly worse (every@5 -0.05, MRR -0.06, intervals excluding 0); the
# document prompt's effect (+0.01) cannot be told from none. No prompts is also exactly what
# MemMachine's OpenAI-compatible embedder sends.
DEFAULT = "gemma300m-raw"
SHARD = 8192  # vectors per resumable shard



def path(embedder: str, dataset: str, name: str) -> Path:
    return VECTORS / embedder / dataset / f"{name}.npy"


def query_vectors(embedder: str, dataset: str, variant: str) -> np.ndarray:
    """Query vectors in queries(dataset) order; remade when the questions or the prompt change."""
    import hashlib
    import json

    target = path(embedder, dataset, f"query_{variant}")
    spec, rows = EMBEDDERS[embedder], queries(dataset)
    texts = [q["variants"][variant] for q in rows]
    key = hashlib.sha1(json.dumps([spec.query_prompt, spec.model or spec.url, texts]).encode()).hexdigest()[:16]
    meta = target.with_suffix(".key")
    if target.exists() and (not meta.exists() or meta.read_text() != key):
        target.unlink()  # made for other questions or another prompt: cheap to redo
    if not target.exists():
        _run(spec, texts, spec.query_prompt, target)
        meta.write_text(key)
    return np.load(target)


def embed(spec: Embedder, texts: list[str], prompt: str) -> np.ndarray:
    """Unit vectors (float32), with MemMachine's averaging for long texts."""
    pieces, owners, weights = [], [], []
    for i, t in enumerate(texts):
        parts = [t[k : k + spec.max_input_chars] for k in range(0, len(t), spec.max_input_chars)] or [t]
        for part in parts:
            pieces.append(prompt + part)
            owners.append(i)
            weights.append(max(len(part), 1))
    vectors = asyncio.run(_embed_all(spec, pieces))
    if len(pieces) != len(texts):
        out = np.zeros((len(texts), vectors.shape[1]))
        np.add.at(out, np.asarray(owners), vectors * np.asarray(weights, dtype=np.float64)[:, None])
        vectors = out.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def _run(spec: Embedder, texts: list[str], prompt: str, target: Path) -> None:
    """Embed in resumable shards under <target>.parts/, then join into <target> (float16)."""
    parts = target.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    shards = range(0, len(texts), SHARD)
    started = time.monotonic()
    for i, offset in enumerate(shards):
        shard = parts / f"{i:05d}.npy"
        if shard.exists():
            try:
                np.load(shard)
            except (OSError, ValueError):  # unreadable: embedded again
                shard.unlink()
        if not shard.exists():
            vectors = embed(spec, texts[offset : offset + SHARD], prompt).astype(np.float16)
            replace_atomically(shard, lambda temporary: np.save(temporary, vectors))  # noqa: B023 - called at once
            print(f"{target}: shard {i + 1}/{len(shards)}", flush=True)
    joined = np.concatenate([np.load(parts / f"{i:05d}.npy") for i in range(len(shards))])
    assert len(joined) == len(texts)
    replace_atomically(target, lambda temporary: np.save(temporary, joined))
    shutil.rmtree(parts)
    print(f"{target}: {len(texts)} vectors in {time.monotonic() - started:.0f}s", flush=True)


async def _post(client: httpx.AsyncClient, route: str, body: dict) -> httpx.Response:
    """One request, retried as the reranker's are (limits.py): 429, 5xx and transport errors
    back off for up to an hour; out of credits and other answers stop at once."""
    from rbench import credits, limits

    failing, attempt, last = None, 0, ""
    while True:
        if limits.CANCEL.is_set():
            raise limits.Interrupted()
        try:
            response = await client.post(route, json=body)
            last = f"HTTP {response.status_code}"
            if credits.refused(response.status_code, response.text):
                raise credits.OutOfCredits(f"embedding: HTTP {response.status_code}")
            if response.status_code not in (429, 500, 502, 503, 504):  # 503: the local server is loading
                response.raise_for_status()
                return response
        except httpx.TransportError as err:
            last = type(err).__name__
        failing = failing or time.monotonic()
        if time.monotonic() - failing > limits.PATIENCE:
            raise limits.GaveUp("embedding requests", last)
        attempt += 1
        await limits.sleep(limits.backoff(attempt))


def fit_dimensions(spec: Embedder, vectors: np.ndarray) -> np.ndarray:
    """Vectors longer than the stored size truncated to it and re-normalized (a Matryoshka
    model's prefix is itself an embedding)."""
    if vectors.shape[1] <= spec.dimensions:
        return vectors
    v = vectors[:, : spec.dimensions]
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, norms, out=np.zeros_like(v), where=norms > 0)


def _fit(spec: Embedder, text: str) -> str:
    """The text, cut at the last whole token that fits the context (as sentence-transformers
    truncates), measured with the model's own tokenizer. A text that cannot reach the limit is
    returned without loading it (so embeddinggemma's gated tokenizer is never needed for
    segments, about 500 characters, against its 2,048 tokens): every token covers at least one
    UTF-8 byte, and SentencePiece adds at most one leading-space token."""
    budget = spec.max_tokens - spec.special_tokens
    if len(text.encode()) + 1 <= budget:
        return text
    starts = token_starts([text], spec.tokenizer)[0]
    return text if len(starts) <= budget else text[: int(starts[budget])]


MAX_REQUEST_CHARS = 75_000  # MemMachine's OpenAIEmbedder per-request cap: the ingest path sends no more either


async def _embed_all(spec: Embedder, texts: list[str]) -> np.ndarray:
    semaphore = asyncio.Semaphore(spec.in_flight)
    batches, size = [[]], 0  # at most request_inputs inputs and MAX_REQUEST_CHARS characters per request
    for t in texts:
        if batches[-1] and (len(batches[-1]) == spec.request_inputs or size + len(t) > MAX_REQUEST_CHARS):
            batches.append([])
            size = 0
        batches[-1].append(t)
        size += len(t)
    batches = [b for b in batches if b]
    out: list[np.ndarray | None] = [None] * len(batches)
    headers = {"Authorization": f"bearer {(SECRETS / spec.key_file).read_text().strip()}"} if spec.key_file else {}
    extra = {"model": spec.model, "encoding_format": "float"} if spec.model else {}
    if spec.send_dimensions:
        extra["dimensions"] = spec.request_dimensions or spec.dimensions
    async with httpx.AsyncClient(base_url=spec.url, timeout=600, headers=headers) as client:

        async def run(i: int) -> None:
            inputs = [_fit(spec, t) for t in batches[i]]
            async with semaphore:
                data = (await _post(client, spec.route, {"input": inputs, **extra})).json()["data"]
            out[i] = fit_dimensions(spec, np.asarray([d["embedding"] for d in sorted(data, key=lambda d: d["index"])],
                                                     dtype=np.float32))

        window = spec.in_flight * 4  # bounded fan-out keeps memory flat
        for start in range(0, len(batches), window):
            await asyncio.gather(*(run(i) for i in range(start, min(start + window, len(batches)))))
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)
