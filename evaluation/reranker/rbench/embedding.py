"""First-stage embedders: chunk and query vectors.

An embedder is an OpenAI-compatible embeddings endpoint plus the model's prompts and
context limit. Two serve google/embeddinggemma-300m:
  gemma300m     llama.cpp's llama-server on this machine (127.0.0.1:9931), free, ~7-10k tokens/s
  gemma300m-di  DeepInfra's hosted model, $0.002 per million tokens (checked 2026-11-05)
each with four prompt setups (SETUPS: "", "-q", "-raw", "-d"); the benchmark uses "-raw".
The client adds the model's prompts itself, as sentence-transformers does. Vectors live
under the embedder's name, so the two never mix; `preflight embed-fidelity` compares them. Adding an embedder (another server, an API) adds a first stage
whose results come out of the same pipeline: the benchmark's first-stage tables are an
embedding and chunking benchmark in their own right.

Inputs longer than the embedder's context are averaged as MemMachine's embedders do
(#1449): the text is cut greedily into `max_input_chars` pieces, each piece is embedded
with the prompt, and the vectors are averaged weighted by piece length, then normalized
(the vector store scores by cosine). A piece still over the context is cut at the
context, as sentence-transformers truncates. Only 2,048-token chunks and long queries
(SWE-bench issues) take this path.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from rbench import chunking
from rbench.paths import SECRETS, VECTORS
from rbench.schema import queries
from rbench.text import GEMMA, token_starts


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

    @property
    def documents(self) -> str:
        return self.doc_store or self.name


QUERY_PROMPT, DOC_PROMPT = "task: search result | query: ", "title: none | text: "  # the model card's recommended prompts
GEMMA_LIMITS = dict(max_tokens=2048, max_input_chars=6000)
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
# No prompts (compare_embedders.py, 200 LongMemEval-M questions, 2026-11-06): the query prompt
# makes the first stage slightly worse (every@5 -0.05, MRR -0.06, intervals excluding 0); the
# document prompt's effect (+0.01) cannot be told from none. No prompts is also exactly what
# MemMachine's OpenAI-compatible embedder sends.
DEFAULT = "gemma300m-raw"
SHARD = 8192  # vectors per resumable shard
ATTEMPTS = 10


def path(embedder: str, dataset: str, name: str) -> Path:
    return VECTORS / embedder / dataset / f"{name}.npy"


def chunk_vectors(embedder: str, dataset: str, chunking_name: str) -> np.ndarray:
    target = path(embedder, dataset, chunking_name)
    if not target.exists():
        texts = chunking.load(dataset, chunking_name).texts(dataset)
        _run(EMBEDDERS[embedder], texts, EMBEDDERS[embedder].doc_prompt, target)
    return np.load(target)


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
        if not shard.exists():
            np.save(shard, embed(spec, texts[offset : offset + SHARD], prompt).astype(np.float16))
            print(f"{target}: shard {i + 1}/{len(shards)}", flush=True)
    joined = np.concatenate([np.load(parts / f"{i:05d}.npy") for i in range(len(shards))])
    assert len(joined) == len(texts)
    np.save(target, joined)
    shutil.rmtree(parts)
    print(f"{target}: {len(texts)} vectors in {time.monotonic() - started:.0f}s", flush=True)


async def _post(client: httpx.AsyncClient, route: str, body: dict) -> httpx.Response:
    for attempt in range(ATTEMPTS):
        try:
            response = await client.post(route, json=body)
            if response.status_code not in (429, 500, 502, 503, 504):  # 503: the local server is loading
                response.raise_for_status()
                return response
        except httpx.TransportError:
            if attempt == ATTEMPTS - 1:
                raise
        await asyncio.sleep(min(60, 2**attempt))
    raise RuntimeError(f"{route}: no answer after {ATTEMPTS} attempts")


def _fit(spec: Embedder, text: str) -> str:
    """The text, cut at the last whole token that fits the context (as sentence-transformers
    truncates), measured with the model's own tokenizer."""
    budget = spec.max_tokens - 2  # BOS and EOS
    if len(text.encode()) <= budget:
        return text
    starts = token_starts([text], GEMMA)[0]
    return text if len(starts) <= budget else text[: int(starts[budget])]


async def _embed_all(spec: Embedder, texts: list[str]) -> np.ndarray:
    semaphore = asyncio.Semaphore(spec.in_flight)
    batches = [texts[i : i + spec.request_inputs] for i in range(0, len(texts), spec.request_inputs)]
    out: list[np.ndarray | None] = [None] * len(batches)
    headers = {"Authorization": f"bearer {(SECRETS / spec.key_file).read_text().strip()}"} if spec.key_file else {}
    extra = {"model": spec.model, "encoding_format": "float"} if spec.model else {}
    async with httpx.AsyncClient(base_url=spec.url, timeout=600, headers=headers) as client:

        async def run(i: int) -> None:
            inputs = [_fit(spec, t) for t in batches[i]]
            async with semaphore:
                data = (await _post(client, spec.route, {"input": inputs, **extra})).json()["data"]
            out[i] = np.asarray([d["embedding"] for d in sorted(data, key=lambda d: d["index"])], dtype=np.float32)

        window = spec.in_flight * 4  # bounded fan-out keeps memory flat
        for start in range(0, len(batches), window):
            await asyncio.gather(*(run(i) for i in range(start, min(start + window, len(batches)))))
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)
