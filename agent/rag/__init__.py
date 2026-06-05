"""RAG layer (LLM-2.3): chunk → embed → store → retrieve, exposed as an agent tool.

Heavy deps (chromadb, sentence-transformers) are imported lazily inside the store so the
chunking logic and tool wiring stay importable — and unit-testable — without them.
"""

from __future__ import annotations

from .chunk import chunk_text
from .retrieve import build_retrieval_tool

__all__ = ["chunk_text", "build_retrieval_tool"]
