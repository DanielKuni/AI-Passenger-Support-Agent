"""
MCP client bridge.

Spawns the MCP server (mcp_server/server.py) as a subprocess over stdio, discovers its tools,
converts them to Claude tool definitions, and forwards tool calls. Every call is appended to an
in-memory log that the web UI's developer panel and the console show.
"""
from __future__ import annotations

import json
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, get_default_environment, stdio_client

from .redact import redact_mapping

# Tools whose string arguments are redacted before they are sent or logged (a case must never carry secrets).
REDACTED_TOOLS = {"prepare_support_case"}


def _attr(obj, *names, default=None):
    """mcp 2.x uses snake_case attributes; 1.x used camelCase. Accept both."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


class MCPBridge:
    def __init__(self, server_script: Path):
        self.server_script = server_script
        self._stack: AsyncExitStack | None = None
        self.session: ClientSession | None = None
        self.tools: list[dict] = []          # Claude-format tool definitions
        self.call_log: list[dict] = []       # every call, newest last

    async def start(self) -> None:
        self._stack = AsyncExitStack()
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_script)],
            # Merge, don't replace: a bare env dict strips PATH/SYSTEMROOT and the child hangs on Windows.
            env={**get_default_environment(), "PYTHONIOENCODING": "utf-8"},
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        listed = await self.session.list_tools()
        self.tools = [
            {"name": t.name, "description": t.description or "", "input_schema": _attr(t, "input_schema", "inputSchema")}
            for t in listed.tools
        ]
        print(f"[mcp-client] connected; tools: {[t['name'] for t in self.tools]}", flush=True)

    async def stop(self) -> None:
        if self._stack:
            await self._stack.aclose()
            self._stack = None

    async def call_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        """Returns (text_for_model, is_error). Logs the call."""
        assert self.session is not None, "MCP bridge not started"
        redaction_kinds: list[str] = []
        if name in REDACTED_TOOLS:
            arguments, redaction_kinds = redact_mapping(arguments)
        t0 = time.perf_counter()
        try:
            result = await self.session.call_tool(name, arguments)
            is_error = bool(_attr(result, "is_error", "isError", default=False))
            structured = _attr(result, "structured_content", "structuredContent")
            if structured is not None and not is_error:
                text = json.dumps(structured, ensure_ascii=False)
            else:
                text = "\n".join(getattr(c, "text", "") for c in result.content if getattr(c, "type", "") == "text")
        except Exception as e:  # transport-level failure
            text, is_error = f"MCP call failed: {e}", True
        entry = {
            "tool": name,
            "input": arguments,
            "output": text,
            "is_error": is_error,
            "duration_ms": round((time.perf_counter() - t0) * 1000),
            "demo": True,
            "redacted": sorted(set(redaction_kinds)),
        }
        self.call_log.append(entry)
        print(f"[mcp-client] {name}({json.dumps(arguments, ensure_ascii=False)}) -> "
              f"{'ERROR ' if is_error else ''}{text[:300]}", flush=True)
        return text, is_error
