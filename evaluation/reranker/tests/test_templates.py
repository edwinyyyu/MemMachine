"""templates/qwen3_reranker_memory.jinja is vLLM's qwen3_reranker.jinja with only its default
instruction changed, and renders exactly the prompt the benchmark's reranker reads."""

from pathlib import Path

import jinja2

from rbench import rerankers

TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "qwen3_reranker_memory.jinja"


def _render(**kwargs) -> str:
    q, d = kwargs.pop("query"), kwargs.pop("document")
    messages = [{"role": "query", "content": q}, {"role": "document", "content": d}]
    return jinja2.Environment(keep_trailing_newline=True).from_string(TEMPLATE.read_text()).render(messages=messages, **kwargs)


def _prompt(model: str, q: str, d: str) -> str:
    t = rerankers.RERANKERS[model].template
    return t.prefix + t.format.format(instruction=t.instruction, query=q, document=d) + t.suffix


def test_the_default_is_the_memory_instruction_and_a_request_instruction_still_overrides_it():
    q, d = "What should I serve for dinner?", '[Tuesday, May 23, 2023, 5:46 AM] User: "basil and mint"'
    assert _render(query=q, document=d) == _prompt("q4-mem", q, d)
    assert _render(query=q, document=d, instruction=rerankers.INSTRUCTION) == _prompt("q4", q, d)
    assert rerankers.MEMORY_INSTRUCTION in TEMPLATE.read_text()
