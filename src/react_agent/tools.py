"""Tool definitions for the agent.

Includes web search plus local file inspection helpers for code review flows.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, cast

from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime

from react_agent.context import Context


def _resolve_base_dir() -> Path:
    runtime = get_runtime(Context)
    base_dir = runtime.context.base_dir or os.getcwd()
    base_path = Path(base_dir).expanduser().resolve()
    if not base_path.exists() or not base_path.is_dir():
        raise ValueError(f"Base directory does not exist: {base_path}")
    return base_path


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in {
        ".py",
        ".md",
        ".txt",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".js",
        ".ts",
        ".tsx",
        ".css",
        ".html",
        ".sh",
        ".cfg",
        ".ini",
    }


async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results."""
    runtime = get_runtime(Context)
    wrapped = TavilySearch(max_results=runtime.context.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


async def list_source_files(pattern: str = "**/*") -> list[str]:
    """List text-like source files under the configured base_dir."""
    base_path = _resolve_base_dir()
    paths: List[str] = []
    for path in base_path.glob(pattern):
        if path.is_file() and _is_text_file(path):
            rel = path.relative_to(base_path)
            paths.append(str(rel))
    return sorted(paths)


async def read_file(path: str, max_bytes: int = 200_000) -> dict[str, Any]:
    """Read a text file relative to base_dir with a size limit."""
    base_path = _resolve_base_dir()
    target = (base_path / path).resolve()
    if not str(target).startswith(str(base_path)):
        raise ValueError("Path escapes base_dir")
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    data = target.read_bytes()[:max_bytes]
    if b"\x00" in data:
        raise ValueError("Binary file detected; skipping")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")
    return {"path": str(target.relative_to(base_path)), "content": text}


async def write_file(path: str, content: str, **_: Any) -> dict[str, str]:
    """Create or overwrite a text file under base_dir."""
    base_path = _resolve_base_dir()
    target = (base_path / path).resolve()
    if not str(target).startswith(str(base_path)):
        raise ValueError("Path escapes base_dir")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"path": str(target.relative_to(base_path)), "status": "written"}


async def run_python(
    path: str,
    args: Optional[List[str]] = None,
    timeout: float = 120.0,
    **_: Any,
) -> dict[str, Any]:
    """Execute a Python file under base_dir and return stdout/stderr."""
    base_path = _resolve_base_dir()
    target = (base_path / path).resolve()
    if not str(target).startswith(str(base_path)):
        raise ValueError("Path escapes base_dir")
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    cmd = ["python", str(target), *(args or [])]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(base_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return {
            "path": str(target.relative_to(base_path)),
            "status": "timeout",
            "log": f"Timed out after {timeout}s while running: {' '.join(cmd)}",
        }

    def _clip(data: bytes) -> str:
        return data.decode("utf-8", errors="replace")[:200_000]

    stdout_txt = _clip(stdout)
    stderr_txt = _clip(stderr)
    log_str = (
        f"Command: {' '.join(cmd)}\n"
        f"Return code: {proc.returncode}\n"
        f"--- stdout ---\n{stdout_txt}\n"
        f"--- stderr ---\n{stderr_txt}"
    )

    return {
        "path": str(target.relative_to(base_path)),
        "returncode": proc.returncode,
        "stdout": stdout_txt,
        "stderr": stderr_txt,
        "log": log_str,
    }


TOOLS: List[Callable[..., Any]] = [search, list_source_files, read_file, write_file, run_python]