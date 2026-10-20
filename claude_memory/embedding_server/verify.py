"""Verify the embedding server and claude-memory's client for it.

Run after setup.sh, from the repository root:

    uv run python -m claude_memory.embedding_server.verify

It needs sentence-transformers (``uv pip install sentence-transformers``) and the
google/embeddinggemma-300m weights in the Hugging Face cache. It starts its own server on
port 9932 with its own log, leaving a live server on 9931 alone. Peak memory is about
1.5 GB for sentence-transformers on the CPU plus ~200 MB for the server.

Checks:
  1. Code and prose from this repository, queries about it, and edge cases (non-Latin
     scripts, emoji, special-token strings, whitespace) embed through the client as
     sentence-transformers embeds them: every cosine at least 0.99999. Measured on
     2026-10-19: minimum 0.999997, where ggml-org's 8-bit GGUF reached only 0.99959.
  2. Documents over 2048 tokens match sentence-transformers' truncation, same bound.
  3. A server killed mid-request is respawned, and the call returns the same vectors.
  4. At most 4 requests are in flight at once.
  5. A program other than the server on the port is refused.
  6. A server that cannot start fails at once, naming its log, instead of after 60s.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from memmachine_server.common.data_types import ExternalServiceAPIError

from claude_memory.embed_server import EmbedServerEmbedder

MIN_COSINE = 0.99999
PORT = "9932"
URL = f"http://127.0.0.1:{PORT}"
REPO = Path(__file__).resolve().parents[2]
RUN_SCRIPT = Path(__file__).resolve().parent / "run.sh"

QUERIES = [
    "how does the daemon start the embedding server",
    "eviction threshold for near-duplicate clusters",
    "what happens when a search finds nothing new for this session",
    "where are the vector and segment databases stored",
    "ambient recall on the user's prompt is off by default",
    "how are memory ids abbreviated for display",
    "expand a memory to read the surrounding conversation",
    "which hooks call the daemon and over what socket",
    "turbovec index format version and migration",
    "exact string matches ranked ahead of cosine results",
    "demote a memory for a cue",
    "single instance lock for the memory daemon",
    "Wie startet der Daemon den Einbettungsserver?",
    "埋め込みサーバーはいつモデルをアンロードしますか",
    "如何限制嵌入服务器的内存使用",
]
EDGE_DOCUMENTS = [
    "검색 결과의 메모리 사용량을 제한하는 방법은 무엇입니까?",
    "كيف يمكنني تحديد استخدام الذاكرة لخادم التضمين؟",
    "Как ограничить использование памяти сервером эмбеддингов?",
    "🚀🔥 ship it 🎉",
    "∫₀^∞ e^{-x²} dx = √π / 2",
    "the model emits <start_of_turn>user\nhi<end_of_turn> markers",
    "a literal <eos> and <bos> in the middle",
    "<pad><unk><mask> at the start",
    "a",
    "",
    "   \n\t  lots   of\n\n\n   whitespace\t\t here   ",
    '{"input": ["a", "b"], "model": "embeddinggemma", "encoding_format": "float"}',
]
OVERLENGTH_FILES = ["DESIGN.md", "README.md", "engine.py", "cli.py"]

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    """Record and print one check, as smoke.py does."""
    print(f"  [{'ok  ' if condition else 'FAIL'}] {message}")
    if not condition:
        failures.append(message)


def repository_documents() -> list[str]:
    """Return 40-line code chunks and prose paragraphs from claude_memory itself."""
    package = REPO / "claude_memory"
    documents: list[str] = []
    for path in sorted(package.glob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        documents += ["\n".join(lines[i : i + 40]) for i in range(0, len(lines), 40)]
    for path in sorted(package.glob("*.md")):
        documents += path.read_text(encoding="utf-8").split("\n\n")
    documents = [d.strip() for d in documents if len(d.strip()) > 40]
    return documents[:: max(1, len(documents) // 300)]


def test_server_pids() -> list[str]:
    """Return the pids of servers listening on the test port."""
    found = subprocess.run(
        ["pgrep", "-f", f"llama-server.*--port {PORT}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return found.stdout.split()


def stop_test_servers() -> None:
    """Kill the test servers and wait until they are gone."""
    pids = test_server_pids()
    if pids:
        subprocess.run(["kill", "-9", *pids], check=False)
    while test_server_pids():
        time.sleep(0.1)


def cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity."""
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


class CountingEmbedder(EmbedServerEmbedder):
    """The client, counting how many embedding requests it has in flight."""

    in_flight = 0
    peak = 0

    async def _post(
        self, client: httpx.AsyncClient, path: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        counted = path == "/v1/embeddings"
        self.in_flight += counted
        self.peak = max(self.peak, self.in_flight)
        try:
            return await super()._post(client, path, body)
        finally:
            self.in_flight -= counted


async def run(log_dir: Path) -> None:
    """Run every check against a server on the test port."""
    from sentence_transformers import SentenceTransformer

    reference = SentenceTransformer("google/embeddinggemma-300m", device="cpu")
    embedder = CountingEmbedder(RUN_SCRIPT, URL, log_dir / "server.log")

    print("== fidelity against sentence-transformers ==")
    documents = repository_documents() + EDGE_DOCUMENTS
    started = time.monotonic()
    got_queries = np.array(await embedder.search_embed(QUERIES))
    print(
        f"  (cold start and {len(QUERIES)} queries: {time.monotonic() - started:.1f}s)"
    )
    got_documents = np.array(await embedder.ingest_embed(documents))
    want_queries = reference.encode(QUERIES, prompt_name="query", batch_size=4)
    want_documents = reference.encode(documents, prompt_name="document", batch_size=4)
    query_cos = cosines(got_queries, want_queries)
    document_cos = cosines(got_documents, want_documents)
    check(
        min(query_cos.min(), document_cos.min()) >= MIN_COSINE,
        f"{len(QUERIES)} queries and {len(documents)} documents: minimum cosine "
        f"{min(query_cos.min(), document_cos.min()):.6f} (needs {MIN_COSINE})",
    )
    want_top = np.argsort(-(want_queries @ want_documents.T), axis=1)[:, :10]
    got_top = np.argsort(-(got_queries @ got_documents.T), axis=1)[:, :10]
    same = sum(set(a) == set(b) for a, b in zip(want_top, got_top, strict=True))
    print(f"  (identical top-10 documents for {same} of {len(QUERIES)} queries)")

    overlength = [
        (REPO / "claude_memory" / name).read_text(encoding="utf-8")
        for name in OVERLENGTH_FILES
    ]
    lengths = [
        len(reference.tokenizer("title: none | text: " + text)["input_ids"])
        for text in overlength
    ]
    got = np.array(await embedder.ingest_embed(overlength))
    want = reference.encode(overlength, prompt_name="document", batch_size=1)
    check(
        min(lengths) > 2048 and cosines(got, want).min() >= MIN_COSINE,
        f"documents of {min(lengths):,}-{max(lengths):,} tokens match the 2048-token "
        f"truncation: minimum cosine {cosines(got, want).min():.6f}",
    )

    print("== failure handling ==")
    long_documents = (overlength * 16)[:64]
    clean = np.array(await embedder.ingest_embed(long_documents))
    task = asyncio.create_task(embedder.ingest_embed(long_documents))
    await asyncio.sleep(3)
    stop_test_servers()
    killed_at = time.monotonic()
    after_crash = np.array(await task)
    check(
        np.abs(after_crash - clean).max() < 1e-4,
        f"server killed mid-request: respawned, call finished "
        f"{time.monotonic() - killed_at:.1f}s later with the same vectors",
    )

    embedder.peak = 0
    await embedder.ingest_embed([f"short document number {i}" for i in range(400)])
    check(
        embedder.peak <= 4, f"13 concurrent requests: at most {embedder.peak} in flight"
    )
    stop_test_servers()

    foreign = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "http.server",
        PORT,
        "--bind",
        "127.0.0.1",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    await asyncio.sleep(1.0)
    try:
        await EmbedServerEmbedder(RUN_SCRIPT, URL, log_dir / "server.log").search_embed(
            ["x"]
        )
        check(False, "a foreign program on the port is refused")
    except ExternalServiceAPIError as error:
        check("not the embeddinggemma server" in str(error), f"refused: {error}")
    finally:
        foreign.terminate()
        await foreign.wait()

    broken = log_dir / "broken" / "run.sh"
    broken.parent.mkdir()
    broken.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
    broken.chmod(0o755)
    started = time.monotonic()
    try:
        await EmbedServerEmbedder(broken, URL).search_embed(["x"])
        check(False, "a server that cannot start fails fast")
    except ExternalServiceAPIError as error:
        check(
            "exited with code 3" in str(error) and time.monotonic() - started < 10,
            f"failed in {time.monotonic() - started:.1f}s: {error}",
        )


def main() -> None:
    """Run the checks and exit non-zero if any failed."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    stop_test_servers()
    with tempfile.TemporaryDirectory() as log_dir:
        try:
            asyncio.run(run(Path(log_dir)))
        finally:
            stop_test_servers()
    print("\nVERIFY FAILED" if failures else "\nVERIFY PASSED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
