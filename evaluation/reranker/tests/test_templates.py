"""templates/qwen3_reranker_memory.jinja is vLLM's qwen3_reranker.jinja with its default
instruction changed and one newline added at its end, and renders, as vLLM renders it, exactly
the prompt the benchmark's reranker read (the Qwen3-Reranker model card's)."""

from pathlib import Path

import jinja2
from transformers.utils.chat_template_utils import _compile_jinja_template

from rbench import rerankers

TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "qwen3_reranker_memory.jinja"
MESSAGES = [{"role": "query", "content": "What should I serve for dinner?"},
            {"role": "document", "content": '[Tuesday, May 23, 2023, 5:46 AM] User: "basil and mint"'}]


def _as_vllm(**kwargs) -> str:
    """vLLM's scoring renders a --chat-template through tokenizer.apply_chat_template, i.e. Hugging
    Face's compiled Jinja environment, which drops a template's final newline."""
    return _compile_jinja_template(TEMPLATE.read_text()).render(messages=MESSAGES, **kwargs)


def _prompt(model: str) -> str:
    t = rerankers.RERANKERS[model].template
    q, d = (m["content"] for m in MESSAGES)
    return t.prefix + t.format.format(instruction=t.instruction, query=q, document=d) + t.suffix


def test_as_vllm_renders_it_the_default_is_the_memory_instruction_and_a_request_instruction_overrides_it():
    assert _as_vllm() == _prompt("q4-mem")
    assert _as_vllm(instruction=rerankers.INSTRUCTION) == _prompt("q4")
    assert _prompt("q4-mem").endswith("</think>\n\n")  # the answer position the reranker was trained at


def test_the_extra_newline_is_what_the_renderer_drops():
    plain = jinja2.Environment().from_string(TEMPLATE.read_text()).render(messages=MESSAGES)  # Jinja's default too
    kept = jinja2.Environment(keep_trailing_newline=True).from_string(TEMPLATE.read_text()).render(messages=MESSAGES)
    assert plain == _prompt("q4-mem") and kept == _prompt("q4-mem") + "\n"
