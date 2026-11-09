"""rbench: when does a reranker help retrieval for chatbots and coding agents?

Pipeline (each stage caches its output under data/ and can be rerun alone):

    datasets  prepare     download a source and write docs.jsonl / queries.jsonl (schema.py)
    chunking  chunk       fixed token windows and EventMemory's recursive segmenter, size-matched
    embedding embed       chunk and query vectors from a first-stage embedder
    retrieval pool        rank each query's candidates; keep the head of the ranking
    rerankers rerank      score the head with a reranker; every pair cached once
    simulate  simulate    replay reading under every ranker, depth and budget -> tidy tables
    report    report      markdown tables and charts

The first stage alone is itself an embedding/chunking benchmark: every report includes it.
"""
