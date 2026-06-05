"""Escalation router (Phase 3 — LOCKED).

Per the Phase 0 decision (`docs/decisions/0000-why-local.md`): the driver is
hard-offline and does **not** justify a model larger than 48 GB, so Phase 3
(large-MoE streaming + escalation) is locked. Every step is served by the resident
model. This module exists so callers have a stable seam, but escalation raises until
the gate is reopened.
"""

from __future__ import annotations

from enum import Enum


class Route(str, Enum):
    RESIDENT = "resident"


class Phase3Locked(RuntimeError):
    """Raised when something attempts to escalate while Phase 3 is locked."""


def route(_task: str) -> Route:
    """Decide which backend handles a sub-task.

    While Phase 3 is locked this is unconditional: everything stays on the resident
    model. The signature is kept task-aware so reopening the gate is a local change
    here, not at every call site.
    """
    return Route.RESIDENT


def escalate(task: str) -> None:
    """Escalate a hard sub-task. Locked — see module docstring and ADR 0000."""
    raise Phase3Locked(
        "Phase 3 is locked: the hard-offline driver did not justify a model >48 GB. "
        "To enable escalation, reopen LLM-0.3 with a documented capability gap and a "
        "local streamed big-MoE endpoint (remote APIs are excluded by the driver). "
        f"Rejected task: {task!r}"
    )
