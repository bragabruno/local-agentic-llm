"""Tool abstraction: a named callable plus an OpenAI-style JSON schema."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ToolError(Exception):
    """Raised when a tool receives invalid input or cannot complete.

    Surfaced back to the model as an error tool-result so it can recover — never
    swallowed silently.
    """


@dataclass(frozen=True)
class Tool:
    """A callable tool exposed to the model.

    Attributes:
        name: Stable identifier the model calls.
        description: What the tool does (shown to the model).
        parameters: JSON Schema for the arguments object.
        func: Implementation; receives validated kwargs, returns a string result.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]

    def openai_schema(self) -> dict[str, Any]:
        """Render this tool as an OpenAI `tools` entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __call__(self, **kwargs: Any) -> str:
        return self.func(**kwargs)
