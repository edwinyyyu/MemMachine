"""Research / ablation harnesses for the temporal retriever.

These modules are not part of the public API. They are scripts that
exercise the production stack (`temporal_retrieval.*`) for ablation,
prompt optimization, and validation. Run as modules from `evaluation/`:

    uv run python -m temporal_retrieval.research._ablation_proper
    uv run python -m temporal_retrieval.research._ablation_hard
    uv run python -m temporal_retrieval.research._sensitivity_curated_bench
    uv run python -m temporal_retrieval.research._prompt_optimizer
    uv run python -m temporal_retrieval.research._validate_best_prompt
    uv run python -m temporal_retrieval.research._smoke_e2e
"""
