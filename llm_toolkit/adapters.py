# Adapters for different LLM providers (Anthropic, OpenAI, Gemini).


from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Response:
    provider: str
    model: str
    text: str
    usage: Usage


class AdapterError(RuntimeError):
    pass


TokenCallback = Optional[Callable[[str], None]]


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise AdapterError(f"Missing required environment variable: {var_name}")
    return value


def ask_anthropic(
    *,
    model: str,
    messages: list[Message],
    stream: bool = False,
    on_token: TokenCallback = None,
    max_tokens: int = 1024,
) -> Response:
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise AdapterError("Anthropic SDK is not installed. Install with: pip install anthropic") from exc

    client = Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))
    sdk_messages = [{"role": m.role, "content": m.content} for m in messages]

    if stream:
        text_parts: list[str] = []
        with client.messages.stream(model=model, messages=sdk_messages, max_tokens=max_tokens) as stream_handle:
            for text in stream_handle.text_stream:
                if text:
                    text_parts.append(text)
                    if on_token:
                        on_token(text)
            final_message = stream_handle.get_final_message()

        usage = Usage(
            input_tokens=getattr(final_message.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(final_message.usage, "output_tokens", 0) or 0,
        )
        return Response(provider="claude", model=model, text="".join(text_parts), usage=usage)

    response = client.messages.create(model=model, messages=sdk_messages, max_tokens=max_tokens)
    text = "".join(getattr(block, "text", "") for block in response.content)
    usage = Usage(
        input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
    )
    return Response(provider="claude", model=model, text=text, usage=usage)


def ask_openai(
    *,
    model: str,
    messages: list[Message],
    stream: bool = False,
    on_token: TokenCallback = None,
) -> Response:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AdapterError("OpenAI SDK is not installed. Install with: pip install openai") from exc

    client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
    sdk_messages = [{"role": m.role, "content": m.content} for m in messages]

    if stream:
        text_parts: list[str] = []
        input_tokens = 0
        output_tokens = 0
        stream_handle = client.chat.completions.create(
            model=model,
            messages=sdk_messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        for event in stream_handle:
            if event.choices:
                delta = event.choices[0].delta.content or ""
                if delta:
                    text_parts.append(delta)
                    if on_token:
                        on_token(delta)
            if getattr(event, "usage", None):
                input_tokens = getattr(event.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(event.usage, "completion_tokens", 0) or 0
        usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
        return Response(provider="gpt", model=model, text="".join(text_parts), usage=usage)

    response = client.chat.completions.create(model=model, messages=sdk_messages, stream=False)
    text = response.choices[0].message.content or ""
    usage = Usage(
        input_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
    )
    return Response(provider="gpt", model=model, text=text, usage=usage)


def ask_gemini(
    *,
    model: str,
    messages: list[Message],
    stream: bool = False,
    on_token: TokenCallback = None,
) -> Response:
    try:
        from google import genai
    except ImportError as exc:
        raise AdapterError("Gemini SDK is not installed. Install with: pip install google-genai") from exc

    client = genai.Client(api_key=_require_env("GEMINI_API_KEY"))
    prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)

    if stream:
        text_parts: list[str] = []
        usage_metadata = None
        for chunk in client.models.generate_content_stream(model=model, contents=prompt):
            chunk_text = getattr(chunk, "text", "") or ""
            if chunk_text:
                text_parts.append(chunk_text)
                if on_token:
                    on_token(chunk_text)
            if getattr(chunk, "usage_metadata", None):
                usage_metadata = chunk.usage_metadata
        usage = Usage(
            input_tokens=getattr(usage_metadata, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage_metadata, "candidates_token_count", 0) or 0,
        )
        return Response(provider="gemini", model=model, text="".join(text_parts), usage=usage)

    response = client.models.generate_content(model=model, contents=prompt)
    usage_metadata = getattr(response, "usage_metadata", None)
    usage = Usage(
        input_tokens=getattr(usage_metadata, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage_metadata, "candidates_token_count", 0) or 0,
    )
    return Response(provider="gemini", model=model, text=getattr(response, "text", "") or "", usage=usage)
