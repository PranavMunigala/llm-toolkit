from __future__ import annotations

import json
from typing import Dict, List

from .adapters import Message, Response, Usage
from .tool_types import Tool, ToolCall


def _assistant_tool_message(provider: str, response: Response, calls: list[ToolCall]) -> Message:
    if provider == "gpt":
        return Message(
            role="assistant",
            content={
                "content": response.text or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in calls
                ],
            },
        )

    content = []
    if response.text:
        content.append({"type": "text", "text": response.text})
    content.extend(
        {
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": call.arguments,
        }
        for call in calls
    )
    return Message(role="assistant", content=content)


def _tool_result_message(provider: str, call: ToolCall, result: object) -> Message:
    content = str(result)
    if provider == "gpt":
        return Message(
            role="tool",
            content={
                "tool_call_id": call.id,
                "content": content,
            },
        )

    return Message(
        role="user",
        content=[
            {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": content,
            }
        ],
    )


def agent_loop(
    *,
    ask_fn,
    provider: str,
    model: str,
    messages: List[Message],
    tools: List[Tool],
    max_steps: int = 8,
    verbose: bool = True,
    require_tool: bool = False,
) -> Response:
    """
    Tool-calling agent loop:

    1. Call model
    2. Extract tool calls
    3. Execute tools
    4. Feed results back
    5. Repeat
    """

    tool_registry: Dict[str, Tool] = {t.name: t for t in tools}
    last_response: Response | None = None
    provider_key = provider.strip().lower()

    if provider_key == "gemini":
        if verbose:
            print("Gemini adapter is not tool-enabled; running one model call without tool loop.")
        return ask_fn(model=model, messages=messages, tools=None)

    for step in range(1, max_steps + 1):

        request_kwargs = {
            "model": model,
            "messages": messages,
            "tools": tools,
        }
        if require_tool and step == 1:
            request_kwargs["tool_choice"] = "required"

        response = ask_fn(**request_kwargs)

        last_response = response

        if verbose:
            print(f"\n===== STEP {step} =====")
            print("MODEL OUTPUT:")
            print(response.text)

        tool_calls = getattr(response, "tool_calls", None)

        if not tool_calls:
            return response

        messages.append(_assistant_tool_message(provider_key, response, tool_calls))

        for call in tool_calls:

            tool = tool_registry.get(call.name)

            if tool is None:
                result = f"ERROR: Unknown tool '{call.name}'"
            else:
                try:
                    result = tool.run(**call.arguments)
                except Exception as exc:
                    result = f"ERROR: {str(exc)}"

            if verbose:
                print(f"\nTOOL: {call.name}")
                print("ARGS:")
                print(json.dumps(call.arguments, indent=2))
                print("RESULT:")
                print(result)

            messages.append(_tool_result_message(provider_key, call, result))

    return Response(
        provider=provider,
        model=model,
        text=f"Stopped after reaching max_steps={max_steps}",
        usage=last_response.usage if last_response else Usage(0, 0),
        tool_calls=None,
    )
