"""A safe arithmetic calculator tool.

Uses an AST walk restricted to arithmetic nodes — never ``eval`` on raw input — so a
model (or a prompt-injected tool call) cannot execute arbitrary code.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from .base import Tool, ToolError

_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ToolError(f"unsupported constant: {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ToolError(f"disallowed expression element: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Evaluate an arithmetic ``expression`` and return the numeric result as a string."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {expression!r}") from exc
    result = _eval_node(tree)
    # Render integers without a trailing .0 for readability.
    if result == int(result):
        return str(int(result))
    return repr(result)


calculator_tool = Tool(
    name="calculator",
    description="Evaluate a basic arithmetic expression (+ - * / // % ** and parentheses).",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, e.g. '(3 + 4) * 2'.",
            }
        },
        "required": ["expression"],
    },
    func=calculate,
)
