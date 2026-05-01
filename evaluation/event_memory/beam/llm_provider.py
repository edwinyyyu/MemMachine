"""Provider-agnostic chat client for BEAM answerers and judges.

Supports two providers behind a single `ChatClient` interface:

- `openai` — `openai.AsyncOpenAI`, via `OPENAI_API_KEY`.
- `google` — `google.genai.Client`, via `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).
  Used to match Vectorize's leaderboard (`gemini-3.1-pro-preview` answerer,
  `gemini-2.5-flash-lite` judge).

`ChatClient.create` returns a normalized dict with `content`,
`prompt_tokens`, `completion_tokens`, `total_tokens` — the fields all BEAM
call sites need. OpenAI-style `response_format` kwargs are translated into
Gemini's `response_mime_type` + `response_schema` on the Gemini path. Because
Gemini's schema dialect rejects OpenAI-specific fields like
`additionalProperties` and `strict`, we strip those (recursively) before
handing the schema off to Gemini.

The OpenAI embedder (text-embedding-3-small) is not covered by this module —
callers instantiate `AsyncOpenAI` directly for that, since the embedder choice
is fixed by the benchmark.
"""

from __future__ import annotations

import os
from typing import Any

PROVIDER_OPENAI = "openai"
PROVIDER_GOOGLE = "google"
PROVIDERS: tuple[str, ...] = (PROVIDER_OPENAI, PROVIDER_GOOGLE)

DEFAULT_ANSWER_MODEL = "gpt-4.1-nano"
DEFAULT_JUDGE_MODEL = "gpt-4.1-mini"


def _build_gemini_schema(schema_dict: dict[str, Any]) -> Any:
    """Translate an OpenAI-style JSON Schema dict to a Gemini `types.Schema`.

    Mirrors the translation in Vectorize's `GeminiLLM._build_schema`
    (https://github.com/vectorize-io/agent-memory-benchmark/blob/main/src/memory_bench/llm/gemini.py).
    Fields OpenAI strict mode requires (like `additionalProperties`) are
    dropped; only the subset Gemini accepts is forwarded.
    """
    from google.genai import types

    type_map = {
        "string": types.Type.STRING,
        "boolean": types.Type.BOOLEAN,
        "integer": types.Type.INTEGER,
        "number": types.Type.NUMBER,
    }

    properties_dict = schema_dict.get("properties", {}) or {}
    properties: dict[str, Any] = {}
    for name, spec in properties_dict.items():
        spec_type = spec.get("type", "string") if isinstance(spec, dict) else "string"
        prop = types.Schema(type=type_map.get(spec_type, types.Type.STRING))
        description = spec.get("description") if isinstance(spec, dict) else None
        if description:
            prop.description = description
        properties[name] = prop

    return types.Schema(
        type=types.Type.OBJECT,
        properties=properties,
        required=list(schema_dict.get("required", [])),
    )


class ChatClient:
    """Provider-agnostic async chat client."""

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Run a chat completion and return a normalized result.

        Result dict always contains: `content`, `prompt_tokens`,
        `completion_tokens`, `total_tokens`.
        """
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


def _openai_max_tokens_kwargs(model: str, max_tokens: int) -> dict[str, Any]:
    """gpt-5 / o-series reject `max_tokens`; use `max_completion_tokens` instead.

    Matches the routing in mem0ai/memory-benchmarks' LLMClient.
    """
    m = model.lower()
    if m.startswith(("gpt-5", "o1", "o3", "o4")):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


class OpenAIChatClient(ChatClient):
    def __init__(self, api_key: str | None = None) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if response_format is not None:
            kwargs["response_format"] = response_format
        if max_tokens is not None:
            kwargs.update(_openai_max_tokens_kwargs(model, max_tokens))
        resp = await self._client.chat.completions.create(**kwargs)
        return {
            "content": resp.choices[0].message.content or "",
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }

    async def close(self) -> None:
        await self._client.close()


class GoogleChatClient(ChatClient):
    def __init__(self, api_key: str | None = None) -> None:
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai is not installed. "
                "Run `uv pip install google-genai` to enable the google provider."
            ) from e
        self._genai = genai
        resolved_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY must be set for the google provider."
            )
        self._client = genai.Client(api_key=resolved_key)

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        from google.genai import types

        # Split OpenAI-style messages into Gemini system_instruction + contents.
        system_parts: list[str] = []
        contents: list[types.Content] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "") or ""
            if role == "system":
                system_parts.append(content)
                continue
            gemini_role = "user" if role == "user" else "model"
            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content)],
                )
            )

        config_kwargs: dict[str, Any] = {}
        if system_parts:
            config_kwargs["system_instruction"] = "\n\n".join(system_parts)
        if response_format is not None:
            kind = response_format.get("type")
            if kind == "json_schema":
                schema = response_format.get("json_schema", {}).get("schema", {})
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_schema"] = _build_gemini_schema(schema)
            elif kind == "json_object":
                config_kwargs["response_mime_type"] = "application/json"
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        resp = await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        # Prefer google-genai's auto-parsed structured output when present
        # (this is what Vectorize's GeminiLLM reads first). Fall back to the
        # raw text, which our callers re-parse with json.loads.
        import json as _json

        parsed = getattr(resp, "parsed", None)
        if parsed is not None and not isinstance(parsed, str):
            content = _json.dumps(parsed)
        else:
            content = resp.text or ""

        usage = getattr(resp, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        total_tokens = getattr(usage, "total_token_count", 0) if usage else 0

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    async def close(self) -> None:
        # google-genai Client has no close method; nothing to release.
        return None


def make_chat_client(provider: str, api_key: str | None = None) -> ChatClient:
    if provider == PROVIDER_OPENAI:
        return OpenAIChatClient(api_key=api_key)
    if provider == PROVIDER_GOOGLE:
        return GoogleChatClient(api_key=api_key)
    raise ValueError(
        f"Unknown provider: {provider!r}. Use one of: {', '.join(PROVIDERS)}"
    )
