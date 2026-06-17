from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class Tool:
    """
    Universal tool definition.
    """

    name: str
    description: str
    input_schema: dict
    run: Callable[..., Any]


@dataclass
class ToolCall:
    """
    Normalized tool call format used by all providers.
    """

    name: str
    arguments: dict
    id: str | None = None
