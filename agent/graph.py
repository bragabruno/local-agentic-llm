"""The multi-step agent loop (LLM-2.1, LLM-2.4).

A LangGraph state machine: the model proposes tool calls, tools execute, results feed
back, and the loop repeats until the model answers without a tool call or the step cap
(`AgentConfig.max_steps`) is hit. Prompt history is trimmed to the measured KV budget
before every model call (LLM-2.5).

The model is any OpenAI-compatible chat endpoint (Phase 1's local engine by default).
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .budget import trim_to_budget
from .config import AgentConfig
from .tools import REGISTRY, Tool, ToolError


class AgentState(TypedDict):
    """Loop state. `messages` accumulates via LangGraph's reducer; `steps` bounds it."""

    messages: Annotated[list[AnyMessage], add_messages]
    steps: int


def _build_model(config: AgentConfig, tools: dict[str, Tool]) -> Any:
    """Construct a tool-bound ChatOpenAI pointed at the local endpoint."""
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    model = ChatOpenAI(
        base_url=config.base_url,
        api_key=SecretStr(config.api_key),
        model=config.model,
        temperature=config.temperature,
    )
    return model.bind_tools([t.openai_schema() for t in tools.values()])


def _run_tool(tools: dict[str, Tool], name: str, args: dict[str, Any]) -> str:
    """Execute one tool call, converting failures into a recoverable error string."""
    tool = tools.get(name)
    if tool is None:
        return f"ERROR: unknown tool {name!r}"
    try:
        return tool(**args)
    except ToolError as exc:
        return f"ERROR: {exc}"


def build_agent(
    config: AgentConfig | None = None,
    tools: dict[str, Tool] | None = None,
    model: Any | None = None,
) -> Any:
    """Compile the agent graph.

    Args:
        config: Runtime settings; defaults to `AgentConfig.from_env()`.
        tools: Tool registry; defaults to the global `REGISTRY`.
        model: Pre-built tool-bound chat model (for tests); built from config if omitted.

    Returns:
        A compiled LangGraph runnable invoked with `{"messages": [...], "steps": 0}`.
    """
    config = config or AgentConfig.from_env()
    tools = tools if tools is not None else REGISTRY
    bound_model = model if model is not None else _build_model(config, tools)
    budget = config.prompt_token_budget

    def agent_node(state: AgentState) -> dict[str, Any]:
        trimmed = trim_to_budget(state["messages"], budget)
        response = bound_model.invoke(trimmed)
        return {"messages": [response], "steps": state["steps"] + 1}

    def tool_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        results: list[AnyMessage] = []
        for call in getattr(last, "tool_calls", []) or []:
            output = _run_tool(tools, call["name"], call.get("args", {}))
            results.append(ToolMessage(content=output, tool_call_id=call["id"]))
        return {"messages": results}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        has_calls = bool(getattr(last, "tool_calls", None))
        if has_calls and state["steps"] < config.max_steps:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()
