"""Expose retrieval as an agent tool.

The retrieval tool is built at runtime against a live `VectorStore` and registered into
the tool registry — keeping the store (and its heavy deps) out of import-time wiring.
"""

from __future__ import annotations

from ..tools import register
from ..tools.base import Tool, ToolError


def build_retrieval_tool(store: object, k: int = 4) -> Tool:
    """Wrap a `VectorStore`-like object (anything with `.query(text, k)`) as a Tool.

    The result demonstrably influences answers (LLM-2.6 acceptance): retrieved chunks are
    returned verbatim for the model to ground on, and an empty result is reported plainly
    rather than silently returning nothing.
    """

    def retrieve(query: str) -> str:
        if not query:
            raise ToolError("query must be non-empty")
        chunks = store.query(query, k=k)  # type: ignore[attr-defined]
        if not chunks:
            return "No relevant documents found."
        return "\n\n---\n\n".join(f"[chunk {i}] {c}" for i, c in enumerate(chunks))

    tool = Tool(
        name="retrieve",
        description="Retrieve relevant passages from the indexed knowledge base for a query.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up in the knowledge base."}
            },
            "required": ["query"],
        },
        func=retrieve,
    )
    register(tool)
    return tool
