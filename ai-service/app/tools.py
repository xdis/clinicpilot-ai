"""Tool gateways used by the agent.

Both gateways expose the same five tools and return the same shapes, so the agent does not
care how a tool is executed:
  - RestTools: calls the backend REST API directly (default).
  - McpTools:  calls the same tools through the bundled MCP server (app.mcp_server) over stdio,
               using one long-lived MCP session owned by a background event loop.
Every call is recorded (name, transport, args, ok, latency) for the agent trace.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Protocol

from .backend_client import BackendClient, ToolError
from .schemas import ToolCall

TOOL_NAMES = ("get_patient_profile", "list_bookings", "get_catalog", "validate_booking", "create_booking")


class ToolGateway(Protocol):
    transport: str
    calls: list[ToolCall]

    def call(self, name: str, **args: Any) -> Any: ...


class _Recorder:
    transport = "none"

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def call(self, name: str, **args: Any) -> Any:
        if name not in TOOL_NAMES:
            raise ToolError("UNKNOWN_TOOL", f"Unknown tool {name}")
        started = time.perf_counter()
        ok = False
        try:
            result = self._invoke(name, args)
            ok = True
            return result
        finally:
            self.calls.append(
                ToolCall(
                    name=name,
                    transport=self.transport,
                    args=args,
                    ok=ok,
                    ms=int((time.perf_counter() - started) * 1000),
                )
            )

    def _invoke(self, name: str, args: dict[str, Any]) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def reset(self) -> None:
        self.calls = []


class RestTools(_Recorder):
    transport = "rest"

    def __init__(self, backend: BackendClient):
        super().__init__()
        self._backend = backend

    def _invoke(self, name: str, args: dict[str, Any]) -> Any:
        return getattr(self._backend, name)(**args)


class _McpWorker:
    """Owns a single MCP client session on a private event loop; requests are queued to it."""

    def __init__(self, backend_url: str):
        self._backend_url = backend_url
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._queue: asyncio.Queue | None = None
        self._error: BaseException | None = None
        threading.Thread(target=self._run, name="mcp-client", daemon=True).start()
        if not self._ready.wait(timeout=30):
            raise ToolError("MCP_START_TIMEOUT", "MCP server did not start within 30s")
        if self._error:
            raise ToolError("MCP_START_FAILED", f"MCP server failed to start: {self._error}")

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        from mcp import Client, StdioServerParameters

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.mcp_server"],
            cwd=str(Path(__file__).resolve().parent.parent),
            env={**os.environ, "BACKEND_URL": self._backend_url},
        )
        try:
            async with Client(params, read_timeout_seconds=30) as client:
                self._queue = asyncio.Queue()
                self._ready.set()
                while True:
                    name, args, fut = await self._queue.get()
                    try:
                        result = await client.call_tool(name, args)
                        fut.set_result(result)
                    except BaseException as exc:  # noqa: BLE001 - forwarded to caller
                        fut.set_exception(exc)
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            self._ready.set()

    def call(self, name: str, args: dict[str, Any]) -> Any:
        fut: Future = Future()
        assert self._queue is not None
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (name, args, fut))
        try:
            result = fut.result(timeout=30)
        except TimeoutError as exc:  # concurrent.futures.TimeoutError is TimeoutError on 3.11+
            raise ToolError("MCP_TIMEOUT", f"MCP tool {name} did not answer within 30s") from exc
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - transport failure (e.g. server process died)
            raise ToolError("MCP_TRANSPORT_ERROR", f"MCP call {name} failed: {exc}") from exc
        payload: Any = result.structured_content
        if payload is None and result.content:
            text = getattr(result.content[0], "text", "")
            try:
                payload = json.loads(text)
            except ValueError:
                payload = {"error": {"code": "MCP_TOOL_ERROR", "message": text}}
        if isinstance(payload, dict) and "result" in payload and len(payload) == 1:
            payload = payload["result"]  # MCP wraps non-object return values
        if result.is_error or (isinstance(payload, dict) and "error" in payload and len(payload) == 1):
            err = payload.get("error", {}) if isinstance(payload, dict) else {}
            raise ToolError(err.get("code", "MCP_TOOL_ERROR"), err.get("message", str(payload)), err.get("details"))
        return payload


_workers: dict[str, _McpWorker] = {}
_workers_lock = threading.Lock()


class McpTools(_Recorder):
    transport = "mcp"

    def __init__(self, backend_url: str):
        super().__init__()
        with _workers_lock:
            if backend_url not in _workers:
                _workers[backend_url] = _McpWorker(backend_url)
            self._worker = _workers[backend_url]

    def _invoke(self, name: str, args: dict[str, Any]) -> Any:
        return self._worker.call(name, args)
