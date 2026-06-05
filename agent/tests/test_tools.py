import pytest

from agent.router import Phase3Locked, Route, escalate, route
from agent.tools.base import ToolError
from agent.tools.calculator import calculate
from agent.tools.file_search import file_search


def test_calculator_basic() -> None:
    assert calculate("(3 + 4) * 2") == "14"
    assert calculate("2 ** 10") == "1024"
    assert calculate("7 / 2") == "3.5"


def test_calculator_rejects_code() -> None:
    with pytest.raises(ToolError):
        calculate("__import__('os').system('echo hi')")
    with pytest.raises(ToolError):
        calculate("a + 1")  # names are disallowed


def test_calculator_invalid_syntax() -> None:
    with pytest.raises(ToolError):
        calculate("3 +")


def test_file_search_finds_known_string() -> None:
    # This repo contains the marker below in its README.
    out = file_search("Local Agentic LLM System", subdir=".")
    assert "README.md" in out


def test_file_search_blocks_traversal() -> None:
    with pytest.raises(ToolError):
        file_search("anything", subdir="../../etc")


def test_router_locked() -> None:
    assert route("any task") is Route.RESIDENT
    with pytest.raises(Phase3Locked):
        escalate("a very hard sub-task")
