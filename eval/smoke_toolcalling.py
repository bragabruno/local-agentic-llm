"""Quant sensitivity smoke probe (LLM-1.5).

Sends a deterministic tool-calling prompt to the local endpoint and checks the model
emits a *well-formed* tool call with correct arguments. Run it against each quant (Q5/Q6)
to pick the one that holds tool-calling — aggressive quant degrades structured output
first.

Usage:
    AGENT_MODEL=qwen3-coder:30b-a3b-q5 python -m eval.smoke_toolcalling
    AGENT_MODEL=qwen3-coder:30b-a3b-q6 python -m eval.smoke_toolcalling

Exits non-zero if the probe fails, so it can gate a quant choice in a script.
"""

from __future__ import annotations

import sys

from agent.config import AgentConfig
from agent.tools.calculator import calculator_tool


def run_probe(config: AgentConfig | None = None) -> bool:
    """Return True if the model emits a correct calculator tool call."""
    config = config or AgentConfig.from_env()
    from openai import OpenAI

    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    resp = client.chat.completions.create(
        model=config.model,
        temperature=0,
        tools=[calculator_tool.openai_schema()],
        messages=[
            {"role": "system", "content": "Use the calculator tool for any arithmetic."},
            {"role": "user", "content": "What is (137 + 49) * 3? Use the tool."},
        ],
    )
    msg = resp.choices[0].message
    calls = msg.tool_calls or []
    if not calls:
        print("FAIL: model did not emit a tool call")
        return False
    call = calls[0]
    print(f"tool={call.function.name} args={call.function.arguments}")
    ok = call.function.name == "calculator" and "137" in call.function.arguments
    print("PASS" if ok else "FAIL: wrong tool/args")
    return ok


def main() -> int:
    config = AgentConfig.from_env()
    print(f"Probing {config.model} at {config.base_url}")
    try:
        return 0 if run_probe(config) else 1
    except Exception as exc:  # noqa: BLE001 — surface connection/endpoint errors plainly
        print(f"ERROR: probe could not reach the endpoint: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
