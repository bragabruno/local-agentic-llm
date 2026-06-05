"""Text chunking — pure, dependency-free, and unit-tested.

Splits on paragraph boundaries first, then packs paragraphs into chunks of at most
``max_chars`` with a small overlap so retrieval context isn't severed mid-thought.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100) -> list[Chunk]:
    """Split ``text`` into overlapping chunks no longer than ``max_chars``.

    Args:
        text: Source document text.
        max_chars: Hard cap on chunk length (characters).
        overlap: Characters of tail context prepended to the next chunk.

    Returns:
        Ordered chunks. Empty input yields an empty list.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be in [0, max_chars)")

    normalized = text.strip()
    if not normalized:
        return []

    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + 2 + len(para) <= max_chars:
            current = f"{current}\n\n{para}"
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}".strip() if tail else para

        # A single paragraph longer than max_chars is hard-split.
        while len(current) > max_chars:
            chunks.append(current[:max_chars])
            current = current[max_chars - overlap :] if overlap else current[max_chars:]

    if current:
        chunks.append(current)

    return [Chunk(text=c, index=i) for i, c in enumerate(chunks)]
