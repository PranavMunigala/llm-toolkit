from __future__ import annotations

import argparse
from typing import Callable

from llm_toolkit.adapters import Message, Response, ask_anthropic, ask_gemini, ask_openai
from llm_toolkit.core import DEFAULT_MODELS
from llm_toolkit.tool_loop import agent_loop
from llm_toolkit.tools.calculator import CALCULATOR_TOOL


ASK_FNS: dict[str, Callable[..., Response]] = {
    "claude": ask_anthropic,
    "gpt": ask_openai,
    "gemini": ask_gemini,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Live tool-calling smoke test.")
    parser.add_argument("--provider", choices=["gpt", "claude", "gemini"], default="gpt")
    parser.add_argument("--model", help="Override the provider default model.")
    parser.add_argument("--prompt", default="What is 365 * 263?")
    parser.add_argument("--max-steps", type=int, default=4)
    args = parser.parse_args()

    provider = args.provider
    model = args.model or DEFAULT_MODELS[provider]
    saw_tool_call = False

    def ask_fn(
        *,
        model: str,
        messages: list[Message],
        tools=None,
        tool_choice: str | None = None,
    ) -> Response:
        nonlocal saw_tool_call
        response = ASK_FNS[provider](
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        if response.tool_calls:
            saw_tool_call = True
        return response

    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Prompt: {args.prompt}")
    print("Expected: calculator MUST be called for this arithmetic prompt.")

    response = agent_loop(
        ask_fn=ask_fn,
        provider=provider,
        model=model,
        messages=[Message(role="user", content=args.prompt)],
        tools=[CALCULATOR_TOOL],
        max_steps=args.max_steps,
        verbose=True,
        require_tool=provider in {"gpt", "claude"},
    )

    print("\nFINAL ANSWER:")
    print(response.text)

    if provider == "gemini":
        print("\nGemini adapter is not tool-enabled in this implementation; no tool call is expected.")
        return 0

    if not saw_tool_call:
        print("\nFAIL: response.tool_calls was empty or None when calculator was expected.")
        return 1

    print("\nPASS: calculator tool call was observed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
