"""Runtime configuration for the agent.

Everything is driven from the environment so the same code points at any local
OpenAI-compatible engine (Ollama, LM Studio, mlx-lm, vllm-mlx) without edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Default resident endpoint (Phase 1 — Ollama). [verify] model tag before use.
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen3-coder:30b-a3b-q5"


@dataclass(frozen=True)
class AgentConfig:
    """Immutable agent settings resolved once at startup.

    Attributes:
        base_url: OpenAI-compatible endpoint of the local serving engine.
        api_key: Placeholder accepted by local engines; never a real secret.
        model: Served model tag.
        temperature: Sampling temperature for the agent model.
        context_window: Total token window the served model exposes.
        kv_reserve_tokens: Tokens held back from the window for generation +
            KV-cache headroom. Measured empirically in Phase 1 (LLM-1.4); the
            default is a conservative placeholder until then.
        max_steps: Hard cap on agent loop iterations (tool-call rounds).
    """

    base_url: str = DEFAULT_BASE_URL
    api_key: str = "local"
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    context_window: int = 32_768
    kv_reserve_tokens: int = 4_096
    max_steps: int = 12

    @property
    def prompt_token_budget(self) -> int:
        """Tokens available for the prompt after reserving generation/KV headroom."""
        return max(0, self.context_window - self.kv_reserve_tokens)

    @classmethod
    def from_env(cls) -> AgentConfig:
        """Build config from environment, falling back to the resident-path defaults."""
        return cls(
            base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
            api_key=os.getenv("OPENAI_API_KEY", "local"),
            model=os.getenv("AGENT_MODEL", DEFAULT_MODEL),
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0.0")),
            context_window=int(os.getenv("AGENT_CONTEXT_WINDOW", "32768")),
            kv_reserve_tokens=int(os.getenv("AGENT_KV_RESERVE_TOKENS", "4096")),
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "12")),
        )
