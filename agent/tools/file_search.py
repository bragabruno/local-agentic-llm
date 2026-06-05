"""A sandboxed file-search tool.

Searches plain-text files under a fixed root for a substring, returning matching
`path:line: text` hits. The root is resolved once and every candidate path is checked
to stay inside it, so the model cannot read outside the sandbox via `../` traversal.
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import Tool, ToolError

# Sandbox root: overridable, defaults to the current working directory.
_ROOT = Path(os.getenv("AGENT_FILE_SEARCH_ROOT", ".")).resolve()
_MAX_HITS = 50
_TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".json", ".yaml", ".yml"}


def _inside_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(_ROOT)
        return True
    except ValueError:
        return False


def file_search(query: str, subdir: str = ".") -> str:
    """Find lines containing ``query`` in text files under ``subdir`` (within the sandbox)."""
    if not query:
        raise ToolError("query must be non-empty")

    base = (_ROOT / subdir).resolve()
    if not _inside_root(base):
        raise ToolError(f"path escapes sandbox root: {subdir!r}")
    if not base.exists():
        raise ToolError(f"path does not exist: {subdir!r}")

    hits: list[str] = []
    for path in sorted(base.rglob("*")):
        if len(hits) >= _MAX_HITS:
            break
        if not path.is_file() or path.suffix not in _TEXT_SUFFIXES:
            continue
        if not _inside_root(path):
            continue
        try:
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if query in line:
                    rel = path.relative_to(_ROOT)
                    hits.append(f"{rel}:{lineno}: {line.strip()}")
                    if len(hits) >= _MAX_HITS:
                        break
        except (UnicodeDecodeError, OSError):
            continue

    if not hits:
        return f"No matches for {query!r}."
    return "\n".join(hits)


file_search_tool = Tool(
    name="file_search",
    description="Search text files under the project for a substring; returns path:line: text hits.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Substring to search for."},
            "subdir": {
                "type": "string",
                "description": "Subdirectory to search within (default: project root).",
            },
        },
        "required": ["query"],
    },
    func=file_search,
)
