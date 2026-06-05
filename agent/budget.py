"""Context / prompt-budget management (LLM-2.5).

The resident model's usable context is bounded by the KV-cache headroom measured in
Phase 1 (LLM-1.4). This module keeps the prompt inside that budget so multi-step loops
never overflow mid-run: it counts tokens and trims the *oldest* conversational turns
while always preserving the system prompt and the most recent message.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class _Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...


def _load_encoder() -> _Encoder | None:
    """Best-effort tiktoken encoder; None if unavailable (falls back to a heuristic)."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


_ENCODER = _load_encoder()

# Heuristic when no tokenizer is present: ~4 chars/token is a standard rough estimate.
_CHARS_PER_TOKEN = 4


def count_tokens(text: str) -> int:
    """Count tokens in ``text``, using tiktoken when available else a char heuristic."""
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _message_text(message: Any) -> str:
    """Extract text from a dict-style or object-style chat message."""
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def message_tokens(message: Any) -> int:
    """Approximate token cost of a single chat message (content + small framing overhead)."""
    # +4 approximates per-message role/delimiter framing in chat formats.
    return count_tokens(_message_text(message)) + 4


def total_tokens(messages: Sequence[Any]) -> int:
    """Total approximate token cost of a message list."""
    return sum(message_tokens(m) for m in messages)


def trim_to_budget(messages: Sequence[Any], budget: int) -> list[Any]:
    """Drop oldest non-system messages until the list fits ``budget`` tokens.

    Invariants:
      * A leading system message is always kept.
      * The final (most recent) message is always kept, even if it alone exceeds the
        budget — truncating the live turn is the caller's concern, not ours.

    Args:
        messages: Ordered chat history (oldest first).
        budget: Maximum total tokens permitted (e.g. ``AgentConfig.prompt_token_budget``).

    Returns:
        A new list within budget where possible, preserving order.
    """
    if not messages:
        return []

    msgs = list(messages)
    has_system = _is_system(msgs[0])
    head = [msgs[0]] if has_system else []
    body = msgs[1:] if has_system else msgs

    if not body:
        return head

    # Always keep the most recent message; trim from the front of the remaining body.
    last = body[-1]
    middle = body[:-1]

    fixed_cost = total_tokens(head) + message_tokens(last)
    available = budget - fixed_cost

    kept_reversed: list[Any] = []
    running = 0
    for msg in reversed(middle):
        cost = message_tokens(msg)
        if running + cost > available:
            break
        kept_reversed.append(msg)
        running += cost

    return head + list(reversed(kept_reversed)) + [last]


def _is_system(message: Any) -> bool:
    if isinstance(message, dict):
        return message.get("role") == "system"
    return getattr(message, "type", None) == "system" or getattr(message, "role", None) == "system"
