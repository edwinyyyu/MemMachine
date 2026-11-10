"""Qwen3 rerankers: input format, backends, the score cache, and the spend log.

Each model's entry in RERANKERS holds everything specific to it: its backend and price, its
chart label, color and markers, and its Template. The Qwen3 model card's template: the string
"<Instruct>: {instruction}\\n<Query>: {query}\\n<Document>: {document}" is tokenized in one
piece and cut from the end to fit, then wrapped in a fixed system prompt (PREFIX) and an
assistant turn with an empty think block (SUFFIX). A pair is at most 8,192 tokens. The
score is the next-token logit of "yes" minus that of "no" (locally); DeepInfra returns
P(yes), the sigmoid of the same quantity, and `to_logit` puts both on one scale. Documents
sent to DeepInfra are cut client-side the same way, so billing matches our count.

Every pair is scored independently, so a pool scored to depth D answers every shallower
depth, top-k and reader budget without scoring again. Scores are cached by content:
(prompt fingerprint, render, query text, document text). The fingerprint hashes the
instruction's text with the prompt template, so editing either never reuses old scores.

Cache keys carry a tag for what the document is: "plain" chunk text (the non-conversation
datasets) or an "em" context EventMemory rendered (memory.RENDER).

Spend: every paid request is cached and logged as soon as it returns, so a failure loses
nothing already paid for. (A request retried after a timeout may have been billed for the
attempt that timed out; that attempt is not logged, so logged spend can slightly undercount.)
There is no spend cap: DeepInfra's prepaid credits are the cap, and running out of them stops
the run cleanly with everything finished kept (credits.py).
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import time
from dataclasses import dataclass
from functools import cache

import numpy as np

from rbench import credits, limits
from rbench.paths import SCORES, SECRETS
from rbench.text import QWEN, tokenizer

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
FORMAT = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"  # the model card's default
SATURATED = 1e-15  # P(yes) this close to 0 or 1 is clipped; exact 0s and 1s tie (preflight.smoke reports them)


@dataclass(frozen=True)
class Template:
    """What a model reads around each (query, document) pair: the text before, the pair's own
    format, the text after, the instruction it carries, the longest pair, and the tokenizer
    that counts it. Its fingerprint (prompt_key) is part of every cache key."""

    prefix: str
    format: str  # with {instruction}, {query} and {document}
    suffix: str
    instruction: str
    max_pair_tokens: int
    tokenizer: str  # Hugging Face repo


QWEN3_CARD = Template(PREFIX, FORMAT, SUFFIX, INSTRUCTION, 8192, QWEN)  # the Qwen3-Reranker model card's format
# Raw pairs: what a sentence-transformers CrossEncoder (MemMachine's CrossEncoderReranker) sends,
# tokenizer(query, document), which for Qwen3's tokenizer is the two texts concatenated with no
# separator or special tokens. Scored as the card's format is (yes minus no at the last token),
# which is what the sequence-classification conversions of Qwen3-Reranker compute. Local only:
# DeepInfra applies the card's format itself.
RAW_PAIRS = Template("", "{query}{document}", "", "", 8192, QWEN)


@dataclass(frozen=True)
class Reranker:
    """A reranker under evaluation: adding one to RERANKERS is all it takes for runs, the score
    cache, the report and the charts to include it."""

    name: str
    hf_model: str
    backend: str  # "local" (transformers on this machine) or "deepinfra"
    usd_per_mtok: float | None  # DeepInfra list price, checked 2026-11-05; None for a local model
    label: str  # in tables and charts
    color: str  # chart color: a slot of the validated categorical palette (report/figures.py)
    template: Template = QWEN3_CARD
    priced_as: str | None = None  # a local model's stand-in price: this hosted model's
    local_tok_per_s: float | None = None  # pair tokens per second on this machine (M3 Pro), measured 2026-11-05
    # chart markers, line-only shapes: (scoring the seeds, scoring the windows). Color tells
    # models apart; the marker tells what the reranker scored, so the two lines of one model
    # differ by shape, never by dashes (report/figures.py)
    markers: tuple[str, str] = ("2", "1")  # up caltrop: seeds; down caltrop: windows

    @property
    def price(self) -> float:
        """$ per million pair tokens (a local model: its stand-in's)."""
        if self.usd_per_mtok is not None:
            return self.usd_per_mtok
        return RERANKERS[self.priced_as].usd_per_mtok if self.priced_as else 0.0


RERANKERS = {
    "q06-local": Reranker("q06-local", "Qwen/Qwen3-Reranker-0.6B", "local", None, "Qwen3-Reranker-0.6B (local)", "#e87ba4",
                          priced_as="q06", local_tok_per_s=2500),
    "q06-local-raw": Reranker("q06-local-raw", "Qwen/Qwen3-Reranker-0.6B", "local", None,
                              "Qwen3-Reranker-0.6B (local, raw pairs)", "#6250d6", template=RAW_PAIRS, priced_as="q06",
                              local_tok_per_s=2500),
    "q06": Reranker("q06", "Qwen/Qwen3-Reranker-0.6B", "deepinfra", 0.010, "Qwen3-Reranker-0.6B", "#eb6834",
                    local_tok_per_s=2500),
    "q4": Reranker("q4", "Qwen/Qwen3-Reranker-4B", "deepinfra", 0.025, "Qwen3-Reranker-4B", "#1baf7a"),
    "q8": Reranker("q8", "Qwen/Qwen3-Reranker-8B", "deepinfra", 0.050, "Qwen3-Reranker-8B", "#eda100"),
}
PAID = [m for m, r in RERANKERS.items() if r.backend == "deepinfra"]


def prompt_key(t: Template) -> str:
    """Fingerprint of everything around the query and document that the model reads."""
    return _h(t.prefix + t.format + t.suffix + t.instruction)[:12]


def _h(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


@cache
def _fixed_ids(t: Template = QWEN3_CARD) -> tuple[tuple[int, ...], tuple[int, ...]]:
    tok = tokenizer(t.tokenizer)
    return tuple(tok.encode(t.prefix, add_special_tokens=False).ids), tuple(tok.encode(t.suffix, add_special_tokens=False).ids)


def _room(t: Template) -> int:
    prefix, suffix = _fixed_ids(t)
    return t.max_pair_tokens - len(prefix) - len(suffix)


def pair_ids(query: str, documents: list[str], t: Template = QWEN3_CARD) -> list[list[int]]:
    """Token ids of each pair exactly as the template builds them."""
    prefix, suffix = _fixed_ids(t)
    texts = [t.format.format(instruction=t.instruction, query=query, document=d) for d in documents]
    return [list(prefix) + e.ids[: _room(t)] + list(suffix)
            for e in tokenizer(t.tokenizer).encode_batch(texts, add_special_tokens=False)]


def fit_document(query: str, document: str, t: Template = QWEN3_CARD) -> str:
    """The document cut so its pair fits the template's longest pair (what is sent to an API)."""
    text = t.format.format(instruction=t.instruction, query=query, document=document)
    e = tokenizer(t.tokenizer).encode(text, add_special_tokens=False)
    if len(e.ids) <= _room(t):
        return document
    head = len(text) - len(document)
    return text[head : e.offsets[_room(t)][0]]


def pair_overhead(query: str, t: Template = QWEN3_CARD) -> int:
    """Tokens every pair repeats before its document (prompt, instruction, query): what a
    server with a prefix cache computes once per query."""
    head = t.format.format(instruction=t.instruction, query=query, document="").rstrip()
    return len(_fixed_ids(t)[0]) + len(tokenizer(t.tokenizer).encode(head, add_special_tokens=False).ids)


def to_logit(reranker: str, scores: np.ndarray) -> np.ndarray:
    """Cached scores on one scale (yes-no logit) whatever the backend returned."""
    if RERANKERS[reranker].backend == "local":
        return scores
    p = np.clip(scores, SATURATED, 1 - SATURATED)  # keeps all the resolution a double carries
    return np.log(p) - np.log1p(-p)


# ------------------------------------------------------------------------- cache


class ScoreCache:
    def __init__(self, reranker: str):
        SCORES.mkdir(parents=True, exist_ok=True)
        self.reranker = reranker
        self.db = sqlite3.connect(SCORES / f"{reranker}.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL, tokens INTEGER)")
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (at REAL, pairs INTEGER, tokens INTEGER, billed INTEGER, "
                        "cost REAL, seconds REAL, note TEXT)")

    @staticmethod
    def key(prompt: str, mode: str, query: str, document: str) -> str:
        return f"{prompt}|{mode}|{_h(query)[:16]}|{_h(document)[:16]}"

    def get(self, keys: list[str]) -> dict[str, tuple[float, int]]:
        out = {}
        for start in range(0, len(keys), 900):  # SQLite's bound-parameter limit
            part = keys[start : start + 900]
            sql = f"SELECT key, score, tokens FROM scores WHERE key IN ({','.join('?' * len(part))})"
            out.update({k: (s, n) for k, s, n in self.db.execute(sql, part)})
        return out

    def put(self, rows: list[tuple[str, float, int]], call: tuple) -> None:
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO scores VALUES (?, ?, ?)", rows)
            self.db.execute("INSERT INTO calls VALUES (?, ?, ?, ?, ?, ?, ?)", call)

    def spent_usd(self) -> float:
        """Cost so far: the API's reported cost per call, else billed (or counted) tokens at list price."""
        price = RERANKERS[self.reranker].usd_per_mtok or 0.0
        row = self.db.execute("SELECT SUM(CASE WHEN cost > 0 THEN cost ELSE COALESCE(NULLIF(billed, 0), tokens) * ? / 1e6 END) "
                              "FROM calls", (price,)).fetchone()
        return float(row[0] or 0.0)


def total_spent_usd() -> float:
    return sum(ScoreCache(m).spent_usd() for m in PAID)


# ---------------------------------------------------------------------- backends

Work = list[tuple[str, list[tuple[str, str]]]]  # (query, [(cache key, document)])
DEEPINFRA_URL = "https://api.deepinfra.com/v1/inference/"
REQUEST_DOCS = 32
REQUEST_TOKENS = 24000
GROUP_QUERIES = 16  # local scoring's groups (memory, progress)


def _local(model: str, work: Work, t: Template, cache_: ScoreCache, note: str) -> None:
    from rbench import qwen_local

    rows, keys = [], []
    for query, items in work:
        rows += pair_ids(query, [d for _, d in items], t)
        keys += [k for k, _ in items]
    t0 = time.monotonic()
    scores = qwen_local.score_rows(rows, model)
    out = [(k, float(s), len(r)) for k, s, r in zip(keys, scores, rows, strict=True)]
    cache_.put(out, (time.time(), len(out), sum(n for *_, n in out), 0, 0.0, time.monotonic() - t0, note))


RETRY = (429, 500, 502, 503, 504)


async def _deepinfra(model: str, work: Work, t: Template, cache_: ScoreCache, note: str, progress=None) -> None:
    """Score with DeepInfra through the model's adaptive concurrency limit (limits.py), caching
    and logging each request as soon as it returns. Retryable failures back off for up to an
    hour per request. A request that cannot succeed (out of credits, a non-retryable answer, or
    an hour of failures) stops the rest from being sent; those in flight finish and are cached."""
    import httpx

    key = (SECRETS / "deepinfra_api_key").read_text().strip()
    requests = []
    for query, items in work:
        lengths = [len(r) for r in pair_ids(query, [d for _, d in items], t)]
        batch, size = [], 0
        for (k, d), n in zip(items, lengths, strict=True):
            if batch and (len(batch) == REQUEST_DOCS or size + n > REQUEST_TOKENS):
                requests.append((query, batch))
                batch, size = [], 0
            batch.append((k, fit_document(query, d, t), n))
            size += n
        if batch:
            requests.append((query, batch))
    gate = limits.Gate(limits.limit_for(f"deepinfra:{model}"))
    stop: list[BaseException] = []
    async with httpx.AsyncClient(timeout=300, headers={"Authorization": f"bearer {key}"}) as client:

        async def run(query: str, batch: list[tuple[str, str, int]]) -> bool:
            body = {"queries": [query] * len(batch), "documents": [d for _, d, _ in batch], "instruction": t.instruction}
            t0, failing, attempt = time.monotonic(), None, 0
            while True:
                async with gate:
                    if stop:
                        return False  # another request cannot succeed: send nothing more
                    try:
                        response = await client.post(DEEPINFRA_URL + model, json=body)
                        last = f"HTTP {response.status_code}"
                    except httpx.TransportError as err:
                        response, last = None, type(err).__name__
                if response is not None and response.status_code == 200:
                    gate.state.success()
                    await gate.wake()
                    break
                if response is not None and credits.refused(response.status_code, response.text):
                    raise credits.OutOfCredits(f"reranking: HTTP {response.status_code}")
                if response is not None and response.status_code not in RETRY:
                    raise RuntimeError(f"DeepInfra {response.status_code}: {response.text[:300]}")
                if response is not None and response.status_code == 429:
                    gate.state.throttled()
                failing = failing or time.monotonic()
                if time.monotonic() - failing > limits.PATIENCE:
                    raise limits.GaveUp("reranking requests", last)
                attempt += 1
                await asyncio.sleep(limits.backoff(attempt))
            data = response.json()
            if len(data["scores"]) != len(batch):
                raise RuntimeError(f"DeepInfra returned {len(data['scores'])} scores for {len(batch)} documents")
            status = data.get("inference_status") or {}
            billed = int(data.get("input_tokens") or status.get("tokens_input") or 0)
            rows = [(k, float(s), n) for (k, _, n), s in zip(batch, data["scores"], strict=True)]
            cache_.put(rows, (time.time(), len(rows), sum(n for *_, n in rows), billed, float(status.get("cost") or 0.0),
                              time.monotonic() - t0, note))
            if progress is not None:
                progress.step(len(rows), note=f"at most {int(gate.state.limit)} in flight")
            return True

        async def guarded(query: str, batch: list[tuple[str, str, int]]) -> bool:
            try:
                return await run(query, batch)
            except Exception as err:
                stop.append(err)
                raise

        results = await asyncio.gather(*(guarded(q, b) for q, b in requests), return_exceptions=True)
    failures = [r for r in results if isinstance(r, BaseException)]
    cached = sum(r is True for r in results)
    for kind in (credits.OutOfCredits, limits.GaveUp):
        hit = [r for r in failures if isinstance(r, kind)]
        if hit:  # every request in flight has finished and is cached: stop cleanly
            raise SystemExit(f"{hit[0]}; {cached} of {len(requests)} requests of this job are cached")
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(requests)} requests failed ({cached} are cached): {failures[0]}")


def score(reranker: str, work: Work, note: str) -> None:
    """Score work, each request cached as it returns: a local model in groups of GROUP_QUERIES
    queries, a hosted one all at once through its adaptive limit (limits.py). A hosted model
    stops cleanly when DeepInfra's credits run out (credits.py) or it fails for an hour."""
    from rbench.progress import Progress

    spec, cache_ = RERANKERS[reranker], ScoreCache(reranker)
    progress = Progress(f"  {reranker} {note}", sum(len(items) for _, items in work), "pairs")
    if spec.backend == "local":
        for g in range(0, len(work), GROUP_QUERIES):
            group = work[g : g + GROUP_QUERIES]
            _local(spec.hf_model, group, spec.template, cache_, note)
            progress.step(sum(len(items) for _, items in group))
        return
    if spec.template != QWEN3_CARD:
        raise SystemExit(f"{reranker}: DeepInfra applies the model card's format itself; other templates run locally")
    asyncio.run(_deepinfra(spec.hf_model, work, spec.template, cache_, note, progress))
