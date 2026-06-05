"""Vector store wrapper (Chroma + sentence-transformers).

Small embedding model by default so it shares the 48 GB budget without crowding the
resident MoE: ``all-MiniLM-L6-v2`` (~80 MB) — [verify] before relying on it. Heavy
imports are deferred to construction time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .chunk import chunk_text

DEFAULT_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")
DEFAULT_PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", ".chroma")
DEFAULT_COLLECTION = "documents"

_TEXT_SUFFIXES = {".py", ".md", ".txt", ".rst"}


class VectorStore:
    """Thin wrapper over a persistent Chroma collection with local embeddings."""

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection: str = DEFAULT_COLLECTION,
        embed_model: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        import chromadb
        from chromadb.utils import embedding_functions

        # Access through an Any alias: chromadb dynamically exports the embedding functions
        # (so the attribute isn't statically known) and its EmbeddingFunction protocol is
        # invariant over input type. Both trip static checks that don't reflect runtime.
        ef_module: Any = embedding_functions
        embedder: Any = ef_module.SentenceTransformerEmbeddingFunction(model_name=embed_model)
        self._embedder = embedder
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection, embedding_function=embedder
        )

    def add_text(self, doc_id: str, text: str, max_chars: int = 1000) -> int:
        """Chunk and index ``text``. Returns the number of chunks added."""
        chunks = chunk_text(text, max_chars=max_chars)
        if not chunks:
            return 0
        self._collection.add(
            ids=[f"{doc_id}:{c.index}" for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": doc_id, "chunk": c.index} for c in chunks],
        )
        return len(chunks)

    def ingest_path(self, root: str | Path, max_chars: int = 1000) -> int:
        """Index every text file under ``root``. Returns total chunks added."""
        base = Path(root)
        total = 0
        files = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in files:
            if path.is_file() and path.suffix in _TEXT_SUFFIXES:
                try:
                    total += self.add_text(str(path), path.read_text(encoding="utf-8"), max_chars)
                except (UnicodeDecodeError, OSError):
                    continue
        return total

    def query(self, text: str, k: int = 4) -> list[str]:
        """Return the ``k`` most relevant chunk texts for ``text``."""
        result = self._collection.query(query_texts=[text], n_results=k)
        docs = result.get("documents") or [[]]
        return list(docs[0])
