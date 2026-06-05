"""Representative end-to-end task (LLM-2.6).

Exercises ≥2 tools + RAG + ≥3 reasoning steps against the local endpoint:

  1. Indexes a small in-memory knowledge base (a fact the model can't know a priori).
  2. Asks a question that forces a `retrieve` call to get a number, then a `calculator`
     call to transform it — at least three loop steps.
  3. Checks the final answer contains the expected result, proving the retrieved fact
     actually influenced the output.

Requires the resident model served locally (Phase 1). Run:

    export OPENAI_BASE_URL=http://localhost:11434/v1
    export OPENAI_API_KEY=local
    export AGENT_MODEL=qwen3-coder:30b-a3b-q5   # [verify]
    python -m eval.e2e_task
"""

from __future__ import annotations

import sys

from langchain_core.messages import HumanMessage, SystemMessage

from agent.config import AgentConfig
from agent.graph import build_agent
from agent.rag.retrieve import build_retrieval_tool
from agent.tools import REGISTRY

# A fact the model cannot know without retrieval.
KNOWLEDGE = """
Project Zephyr operations manual.

The Zephyr reactor's nominal output is 1450 megawatts per unit.
The site operates 4 identical units in total.
Scheduled maintenance takes one unit offline each quarter.
"""

QUESTION = (
    "Using the knowledge base, find the Zephyr reactor's nominal output per unit and the "
    "number of units, then calculate the total nominal output across all units. "
    "State the final number in megawatts."
)

EXPECTED_TOTAL = "5800"  # 1450 * 4


class _InMemoryStore:
    """Minimal store for the e2e demo; avoids needing a persistent Chroma dir."""

    def __init__(self) -> None:
        from agent.rag.store import VectorStore

        self._store = VectorStore(persist_dir=".chroma_e2e", collection="e2e")
        self._store.add_text("zephyr-manual", KNOWLEDGE)

    def query(self, text: str, k: int = 4) -> list[str]:
        return self._store.query(text, k=k)


def main() -> int:
    config = AgentConfig.from_env()
    build_retrieval_tool(_InMemoryStore())  # registers `retrieve` into REGISTRY

    print(f"Running e2e task against {config.model} at {config.base_url}")
    print(f"Tools available: {sorted(REGISTRY)}")

    agent = build_agent(config=config, tools=REGISTRY)
    try:
        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(
                        content=(
                            "You are a precise agent. Use the `retrieve` tool to look up "
                            "facts and the `calculator` tool for arithmetic. Do not guess."
                        )
                    ),
                    HumanMessage(content=QUESTION),
                ],
                "steps": 0,
            }
        )
    except Exception as exc:  # noqa: BLE001 — endpoint/connection errors are the likely cause
        print(f"ERROR: could not run against the endpoint: {exc}")
        return 2

    final = result["messages"][-1].content
    steps = result["steps"]
    print(f"\n--- final answer (after {steps} steps) ---\n{final}\n")

    if EXPECTED_TOTAL in str(final):
        print(f"PASS: answer contains expected total {EXPECTED_TOTAL} MW")
        return 0
    print(f"FAIL: expected {EXPECTED_TOTAL} MW in the answer")
    return 1


if __name__ == "__main__":
    sys.exit(main())
