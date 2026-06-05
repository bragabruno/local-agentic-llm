"""Agent tools (LLM-2.2).

Tools are plain Python callables paired with an OpenAI-style JSON schema, kept free of
LangGraph/LangChain imports so they stay unit-testable without a running model. The graph
layer adapts them to the framework's tool interface.
"""

from __future__ import annotations

from .base import Tool, ToolError
from .calculator import calculator_tool
from .file_search import file_search_tool

# Retrieval is registered by the RAG layer at runtime (agent/rag) to avoid a hard
# dependency on a vector store at import time.
REGISTRY: dict[str, Tool] = {
    calculator_tool.name: calculator_tool,
    file_search_tool.name: file_search_tool,
}


def register(tool: Tool) -> None:
    """Add or replace a tool in the registry."""
    REGISTRY[tool.name] = tool


__all__ = ["Tool", "ToolError", "REGISTRY", "register", "calculator_tool", "file_search_tool"]
