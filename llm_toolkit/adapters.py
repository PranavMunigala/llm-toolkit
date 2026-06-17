from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .tool_types import ToolCall


@dataclass(frozen=True)
class Message:
    role: str
    content: Any


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
    tool_calls: list[ToolCall] | None = None


class AdapterError(RuntimeError):
    pass


TokenCallback = Optional[Callable[[str], None]]


def _require_env(var_name: str) -> str:
    return os.getenv(var_name, "") or ""


def _tool_id(index: int, provider_id: str | None = None) -> str:
    return provider_id or f"tool_call_{index}"


def _anthropic_tools(tools) -> list[dict] | None:
    if not tools:
        return None
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in tools
    ]


def _openai_tools(tools) -> list[dict] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


def _openai_message(message: Message) -> dict:
    if isinstance(message.content, dict):
        payload = {"role": message.role, **message.content}
        if message.role == "assistant" and payload.get("content") is None:
            payload["content"] = None
        return payload
    return {"role": message.role, "content": message.content}


# =========================================================
# ANTHROPIC
# =========================================================

def ask_anthropic(
    *,
    model: str,
    messages: list[Message],
    tools=None,
    tool_choice: str | None = None,
    stream: bool = False,
    on_token: TokenCallback = None,
    max_tokens: int = 1024,
) -> Response:

    anth_key = os.getenv("ANTHROPIC_API_KEY", "") or ""
    if not anth_key:
        return Response(
            provider="claude",
            model=model,
            text="No Anthropic API key configured.",
            usage=Usage(0, 0),
            tool_calls=None,
        )

    try:
        from anthropic import Anthropic
    except ImportError:
        return Response(
            provider="claude",
            model=model,
            text="Anthropic SDK not installed.",
            usage=Usage(0, 0),
            tool_calls=None,
        )

    client = Anthropic(api_key=anth_key)

    sdk_messages = [{"role": m.role, "content": m.content} for m in messages]
    request_kwargs = {}
    provider_tools = _anthropic_tools(tools)
    if provider_tools:
        request_kwargs["tools"] = provider_tools
        if tool_choice == "required":
            request_kwargs["tool_choice"] = {"type": "any"}

    try:
        response = client.messages.create(
            model=model,
            messages=sdk_messages,
            max_tokens=max_tokens,
            **request_kwargs,
        )

        text = "".join(
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", None) == "text"
        )

        usage = Usage(
            input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
        )

        tool_calls = parse_anthropic_tool_calls(response.content)

        return Response(
            provider="claude",
            model=model,
            text=text,
            usage=usage,
            tool_calls=tool_calls or None,
        )

    except Exception as exc:
        return Response(
            provider="claude",
            model=model,
            text=f"Anthropic error: {exc}",
            usage=Usage(0, 0),
            tool_calls=None,
        )


def parse_anthropic_tool_calls(content_blocks) -> list[ToolCall]:
    calls = []

    for index, block in enumerate(content_blocks):
        if getattr(block, "type", None) == "tool_use":
            calls.append(
                ToolCall(
                    name=block.name,
                    arguments=block.input,
                    id=_tool_id(index, getattr(block, "id", None)),
                )
            )

    return calls


# =========================================================
# OPENAI
# =========================================================

def ask_openai(
    *,
    model: str,
    messages: list[Message],
    tools=None,
    tool_choice: str | None = None,
    stream: bool = False,
    on_token: TokenCallback = None,
) -> Response:

    openai_key = os.getenv("OPENAI_API_KEY", "") or ""
    if not openai_key:
        return Response(
            provider="gpt",
            model=model,
            text="No OpenAI API key configured.",
            usage=Usage(0, 0),
            tool_calls=None,
        )

    try:
        from openai import OpenAI
    except ImportError:
        return Response(
            provider="gpt",
            model=model,
            text="OpenAI SDK not installed.",
            usage=Usage(0, 0),
            tool_calls=None,
        )

    client = OpenAI(api_key=openai_key)

    sdk_messages = [_openai_message(m) for m in messages]
    request_kwargs = {}
    provider_tools = _openai_tools(tools)
    if provider_tools:
        request_kwargs["tools"] = provider_tools
        request_kwargs["tool_choice"] = tool_choice or "auto"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=sdk_messages,
            stream=False,
            **request_kwargs,
        )

        message = response.choices[0].message
        text = message.content or ""

        usage = Usage(
            input_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
        )

        tool_calls = parse_openai_tool_calls(message)

        return Response(
            provider="gpt",
            model=model,
            text=text,
            usage=usage,
            tool_calls=tool_calls or None,
        )

    except Exception as exc:
        return Response(
            provider="gpt",
            model=model,
            text=f"OpenAI error: {exc}",
            usage=Usage(0, 0),
            tool_calls=None,
        )


def parse_openai_tool_calls(message) -> list[ToolCall]:
    calls = []

    for index, tool_call in enumerate(message.tool_calls or []):

        try:
            args = json.loads(tool_call.function.arguments)
        except Exception:
            args = {}

        calls.append(
            ToolCall(
                name=tool_call.function.name,
                arguments=args,
                id=_tool_id(index, getattr(tool_call, "id", None)),
            )
        )

    return calls


# =========================================================
# GEMINI (kept but NOT tool-enabled yet)
# =========================================================

def ask_gemini(
    *,
    model: str,
    messages: list[Message],
    tools=None,
    tool_choice: str | None = None,
    stream: bool = False,
    on_token: TokenCallback = None,
) -> Response:
    # Gemini tool calling is intentionally not wired in this adapter yet.
    # If tools are supplied, this wrapper ignores them and returns no tool_calls
    # so callers can fall back gracefully instead of entering a tool loop.

    gemini_key = os.getenv("GEMINI_API_KEY", "") or ""
    if not gemini_key:
        return Response(
            provider="gemini",
            model=model,
            text="No Gemini API key configured.",
            usage=Usage(0, 0),
            tool_calls=None,
        )

    try:
        from google import genai
    except ImportError:
        return Response(
            provider="gemini",
            model=model,
            text="Gemini SDK not installed.",
            usage=Usage(0, 0),
            tool_calls=None,
        )

    client = genai.Client(api_key=gemini_key)

    prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

        usage_metadata = getattr(response, "usage_metadata", None)

        usage = Usage(
            input_tokens=getattr(usage_metadata, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage_metadata, "candidates_token_count", 0) or 0,
        )

        return Response(
            provider="gemini",
            model=model,
            text=getattr(response, "text", "") or "",
            usage=usage,
            tool_calls=None,
        )

    except Exception as exc:
        return Response(
            provider="gemini",
            model=model,
            text=f"Gemini error: {exc}",
            usage=Usage(0, 0),
            tool_calls=None,
        )
