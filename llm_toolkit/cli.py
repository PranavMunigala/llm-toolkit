# CLI entry point for llm-toolkit, providing a unified interface to interact with multiple LLM providers.

from __future__ import annotations

import argparse
import sys
from time import perf_counter

from .adapters import AdapterError, Message, Response, ask_anthropic, ask_gemini, ask_openai
from .core import DEFAULT_MODELS, ask, cost_cents
from .tool_loop import agent_loop
from .tools.calculator import CALCULATOR_TOOL


ASK_FNS = {
    "claude": ask_anthropic,
    "gpt": ask_openai,
    "gemini": ask_gemini,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-toolkit", description="Unified CLI for Claude, GPT, and Gemini.")
    parser.add_argument("prompt", help="Prompt to send to the model.")
    parser.add_argument("--provider", choices=["claude", "gpt", "gemini"], default="gpt")
    parser.add_argument("--all", action="store_true", dest="all_providers", help="Run prompt on all providers.")
    parser.add_argument("--stream", action="store_true", help="Stream tokens as they arrive.")
    parser.add_argument("--model", help="Override the default model for a provider.")
    parser.add_argument("--tools", action="store_true", help="Enable the built-in calculator tool.")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum tool-loop steps when tools are enabled.")
    parser.add_argument("--verbose", action="store_true", help="Print tool-loop execution details.")
    return parser


def _print_response(response: Response) -> None:
    try:
        costs = cost_cents(response.usage, response.model)
    except Exception:
        costs = {"input_cents": 0.0, "output_cents": 0.0, "total_cents": 0.0}

    print(f"\n[{response.provider}:{response.model}]")
    text = response.text.strip()
    print(text)
    print(
        "usage: "
        f"in={response.usage.input_tokens} "
        f"out={response.usage.output_tokens} "
        f"cost(cents) in={costs['input_cents']:.6f} "
        f"out={costs['output_cents']:.6f} "
        f"total={costs['total_cents']:.6f}"
    )

    lowered = text.lower()
    if "404" in text or "not_found" in lowered or ("model" in lowered and "not" in lowered and "found" in lowered):
        print("\nHint: The provider reported the model was not found. Try specifying a different model with --model or check your provider account for available models.")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    providers = ["claude", "gpt", "gemini"] if args.all_providers else [args.provider]
    started = perf_counter()

    try:
        for provider in providers:
            print(f"\n=== {provider} ===")
            callback = (lambda token: print(token, end="", flush=True)) if args.stream else None
            model = args.model if not args.all_providers else None
            if args.tools:
                resolved_model = model or DEFAULT_MODELS[provider]
                tools = [CALCULATOR_TOOL]

                def ask_fn(
                    *,
                    model: str,
                    messages: list[Message],
                    tools=None,
                    tool_choice: str | None = None,
                ) -> Response:
                    return ASK_FNS[provider](
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        stream=args.stream,
                        on_token=callback,
                    )

                response = agent_loop(
                    ask_fn=ask_fn,
                    provider=provider,
                    model=resolved_model,
                    messages=[Message(role="user", content=args.prompt)],
                    tools=tools,
                    max_steps=args.max_steps,
                    verbose=args.verbose,
                )
            else:
                response = ask(
                    prompt=args.prompt,
                    provider=provider,
                    model=model,
                    stream=args.stream,
                    on_token=callback,
                )
            if args.stream:
                print()
            _print_response(response)
    except (AdapterError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    elapsed = perf_counter() - started
    print(f"\nTotal elapsed time: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
