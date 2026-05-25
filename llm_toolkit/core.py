from __future__ import annotations

from time import perf_counter
from typing import Callable, Optional

from .adapters import AdapterError, Message, Response, Usage, ask_anthropic, ask_gemini, ask_openai
from .constants import PRICES

DEFAULT_MODELS = {
    "claude": "claude-3-5-haiku-latest",
    "gpt": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
}


def cost_cents(usage: Usage, model: str) -> dict[str, float]:
    if model not in PRICES:
        raise AdapterError(f"Model '{model}' is not configured in PRICES.")

    price = PRICES[model]
    input_cents = (usage.input_tokens / 1_000_000) * price["input_per_million_usd"] * 100
    output_cents = (usage.output_tokens / 1_000_000) * price["output_per_million_usd"] * 100
    total_cents = input_cents + output_cents
    return {
        "input_cents": round(input_cents, 6),
        "output_cents": round(output_cents, 6),
        "total_cents": round(total_cents, 6),
    }


def ask(
    prompt: str,
    provider: str,
    model: Optional[str] = None,
    stream: bool = False,
    on_token: Optional[Callable[[str], None]] = None,
) -> Response:
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    provider_key = provider.strip().lower()
    resolved_model = model or DEFAULT_MODELS.get(provider_key)
    if not resolved_model:
        raise ValueError(f"Unknown provider: {provider}")

    messages = [Message(role="user", content=prompt)]

    if provider_key == "claude":
        return ask_anthropic(model=resolved_model, messages=messages, stream=stream, on_token=on_token)
    if provider_key == "gpt":
        return ask_openai(model=resolved_model, messages=messages, stream=stream, on_token=on_token)
    if provider_key == "gemini":
        return ask_gemini(model=resolved_model, messages=messages, stream=stream, on_token=on_token)
    raise ValueError(f"Unsupported provider: {provider}")


def ask_all(prompt: str, stream: bool = False) -> tuple[dict[str, Response], float]:
    started = perf_counter()
    results = {
        provider: ask(prompt=prompt, provider=provider, stream=stream)
        for provider in ("claude", "gpt", "gemini")
    }
    elapsed_seconds = perf_counter() - started
    return results, elapsed_seconds
