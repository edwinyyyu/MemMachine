"""Shared embedder factory — OpenAI or sentence-transformers.

Used by locomo_ingest.py and locomo_search.py so both build the SAME
embedder from --embedding-model (ingest and search MUST match).

OpenAI: text-embedding-3-small / -large.
Sentence-transformers (local, via the package's SentenceTransformerEmbedder):
  embeddinggemma -> google/embeddinggemma-300m   (768-d)
  minilm         -> sentence-transformers/all-MiniLM-L6-v2  (384-d)
Both ST models expose query/document prompts; the package embedder hard-
codes prompt_name="query" for search, so default_prompt_name is set to
"document" for the ingest side.
"""

from __future__ import annotations

ST_MODELS = {
    "embeddinggemma": "google/embeddinggemma-300m",
    # Ablation variant: same model + same document embeddings, but the
    # query side uses the QA-task prompt instead of the default
    # retrieval-query prompt. embeddinggemma's packaged prompts dict has
    # "query" = "task: search result | query: " (current default); the
    # QA prompt "task: question answering | query: " is on the model
    # card but not in the dict, so build_embedder injects it.
    "embeddinggemma-qa": "google/embeddinggemma-300m",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}
# Query-prompt override applied to ST models, keyed by --embedding-model.
_ST_QUERY_PROMPT_OVERRIDE = {
    "embeddinggemma-qa": "task: question answering | query: ",
}
EMBEDDING_CHOICES = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "embeddinggemma-300m",  # SERVER (canonical spelling) via OpenAIEmbedder
    *ST_MODELS,
]


def build_embedder(model_name: str, openai_client):
    """Construct the Embedder selected by `model_name`."""
    if model_name in ("text-embedding-3-small", "text-embedding-3-large"):
        from memmachine_server.common.embedder.openai_embedder import (
            OpenAIEmbedder,
            OpenAIEmbedderParams,
        )

        dims = 1536 if model_name.endswith("small") else 3072
        return OpenAIEmbedder(
            OpenAIEmbedderParams(
                client=openai_client,
                model=model_name,
                dimensions=dims,
                max_input_length=8192,
            )
        )

    if model_name == "embeddinggemma-300m":
        # Server-hosted google/embeddinggemma-300m via OpenAI-compatible
        # /v1/embeddings. The server does NOT auto-apply the model's
        # task: prompts (probed 2026-05-26: cos(raw, prompted) ≈ 0.84,
        # not 1.0). To match local SentenceTransformer behavior we wrap
        # OpenAIEmbedder in a prompt-prepender that uses the EXACT
        # prompts from the local model's `prompts` dict -- "query"
        # and "document" entries. The document prompt uses a
        # title-prefix format ("title: none | text: ") rather than the
        # parallel-structure I'd guessed first; that mismatch produced
        # cos(local-DOC, server-DOC)=0.89 on the same input and
        # regressed cat1 by ~5pp on bo-natural. With the right prompts
        # cos becomes 0.99995 (verified 2026-05-27).
        from memmachine_server.common.embedder.embedder import Embedder
        from memmachine_server.common.embedder.openai_embedder import (
            OpenAIEmbedder,
            OpenAIEmbedderParams,
        )

        inner = OpenAIEmbedder(
            OpenAIEmbedderParams(
                client=openai_client,
                model="embeddinggemma-300m",
                dimensions=768,
                max_input_length=8192,
            )
        )

        DOC_PROMPT = "title: none | text: "
        QUERY_PROMPT = "task: search result | query: "

        class _EmbeddinggemmaServerEmbedder(Embedder):
            def __init__(self, inner_embedder):
                # Let the inner OpenAIEmbedder handle batching.
                super().__init__(batch_size=None)
                self._inner = inner_embedder

            async def _ingest_embed(self, inputs, max_attempts=1):
                # Override max_attempts upward -- callers (EventMemory)
                # pass max_attempts=1 by default, which makes a single
                # APITimeoutError kill the whole ingest. The vmnet4-200
                # embedding backend has been intermittently flaky under
                # concurrent load; retry up to 5x with exponential
                # backoff (inner client handles the backoff).
                return await self._inner.ingest_embed(
                    [DOC_PROMPT + s for s in inputs],
                    max(max_attempts, 5),
                )

            async def _search_embed(self, queries, max_attempts=1):
                return await self._inner.search_embed(
                    [QUERY_PROMPT + q for q in queries],
                    max(max_attempts, 5),
                )

            @property
            def model_id(self):
                return self._inner.model_id

            @property
            def dimensions(self):
                return self._inner.dimensions

            @property
            def similarity_metric(self):
                return self._inner.similarity_metric

        return _EmbeddinggemmaServerEmbedder(inner)

    if model_name in ST_MODELS:
        from sentence_transformers import SentenceTransformer
        from memmachine_server.common.embedder.sentence_transformer_embedder import (  # noqa: E501
            SentenceTransformerEmbedder,
            SentenceTransformerEmbedderParams,
        )

        hf_name = ST_MODELS[model_name]
        st = SentenceTransformer(hf_name)
        # The package embedder passes prompt_name="query" for search and
        # None for ingest; None falls back to default_prompt_name. Both
        # ST models here carry query/document prompts -> ingest gets the
        # document prompt, search the query prompt.
        if "document" in (st.prompts or {}):
            st.default_prompt_name = "document"
        # Ablation: swap the query-side prompt. _search_embed calls
        # encode(prompt_name="query"), which reads st.prompts["query"];
        # overriding that entry changes only the query instruction
        # (document embeddings in the DB are unaffected).
        query_override = _ST_QUERY_PROMPT_OVERRIDE.get(model_name)
        if query_override is not None and st.prompts is not None:
            st.prompts["query"] = query_override
        return SentenceTransformerEmbedder(
            SentenceTransformerEmbedderParams(
                model_name=hf_name,
                sentence_transformer=st,
                batch_size=256,
            )
        )

    raise ValueError(f"Unknown embedding model: {model_name}")
