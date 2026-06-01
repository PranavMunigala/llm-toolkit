# CLI entry point for llm-toolkit, providing a unified interface to interact with multiple LLM providers.

from __future__ import annotations

import argparse
import sys
from time import perf_counter

from .adapters import AdapterError, Response
from .core import ask, cost_cents


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-toolkit", description="Unified CLI for Claude, GPT, and Gemini.")
    parser.add_argument("prompt", help="Prompt to send to the model.")
    parser.add_argument("--provider", choices=["claude", "gpt", "gemini"], default="gpt")
    parser.add_argument("--all", action="store_true", dest="all_providers", help="Run prompt on all providers.")
    parser.add_argument("--stream", action="store_true", help="Stream tokens as they arrive.")
    parser.add_argument("--model", help="Override the default model for a provider.")
    return parser


def _print_response(response: Response) -> None:
    costs = cost_cents(response.usage, response.model)
    print(f"\n[{response.provider}:{response.model}]")
    print(response.text.strip())
    print(
        "usage: "
        f"in={response.usage.input_tokens} "
        f"out={response.usage.output_tokens} "
        f"cost(cents) in={costs['input_cents']:.6f} "
        f"out={costs['output_cents']:.6f} "
        f"total={costs['total_cents']:.6f}"
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    providers = ["claude", "gpt", "gemini"] if args.all_providers else [args.provider]
    started = perf_counter()

    try:
        for provider in providers:
            print(f"\n=== {provider} ===")
            callback = (lambda token: print(token, end="", flush=True)) if args.stream else None
            response = ask(
                prompt=args.prompt,
                provider=provider,
                model=args.model if not args.all_providers else None,
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
