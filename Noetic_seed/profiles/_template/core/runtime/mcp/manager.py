"""McpServerManager — MCP server ライフサイクル管理。

claw-code 参照: rust/crates/runtime/src/mcp_stdio.rs (lifecycle)
                 rust/crates/runtime/src/mcp_lifecycle_hardened.rs

責務:
  - transport の起動/停止
  - initialize handshake
  - tools/list / resources/list の discovery
  - tools/call 呼出
  - request/response id 管理
  - timeout / 切断時の graceful degradation
"""
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from core.runtime.mcp.client import BaseTransport
from core.runtime.mcp.protocol import (
    JsonRpcRequest,
    JsonRpcError,
    METHOD_NOT_FOUND,
    INTERNAL_ERROR,
    PROTOCOL_VERSION,
)


class ConnectionStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTH_REQUIRED = "auth_required"
    ERROR = "error"


@dataclass
class McpToolInfo:
    name: str
    description: str = ""
    input_schema: Optional[dict] = None


@dataclass
class McpResourceInfo:
    uri: str
    name: str = ""
    description: str = ""
    mime_type: Optional[str] = None


# claw-code の timeout 設定 (mcp_stdio.rs:22-29)
DEFAULT_INITIALIZE_TIMEOUT = 10.0   # seconds
DEFAULT_LIST_TIMEOUT = 30.0
DEFAULT_CALL_TIMEOUT = 60.0


class McpServerManager:
    """1 つの MCP server の接続・ツール一覧・呼出を管理する。"""

    def __init__(self, server_name: str, transport: BaseTransport):
        self.server_name = server_name
        self.transport = transport
        self.status = ConnectionStatus.DISCONNECTED
        self.tools: list = []       # [McpToolInfo, ...]
        self.resources: list = []   # [McpResourceInfo, ...]
        self.server_info: Optional[dict] = None
        self.last_error: Optional[str] = None

        self._id_counter = 0
        self._id_lock = threading.Lock()

    # ---- id 生成 ----

    def _next_id(self) -> int:
        with self._id_lock:
            self._id_counter += 1
            return self._id_counter

    # ---- lifecycle ----

    def start(self,
              initialize_timeout: float = DEFAULT_INITIALIZE_TIMEOUT) -> bool:
        """transport 起動 + initialize handshake。成功なら True。"""
        self.status = ConnectionStatus.CONNECTING
        try:
            self.transport.start()
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.last_error = f"transport start failed: {e}"
            return False

        # initialize RPC
        req = JsonRpcRequest(
            method="initialize",
            params={
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "noetic-seed-mcp",
                    "version": "0.5",
                },
            },
            id=self._next_id(),
        )
        response = self._send_and_wait(req, timeout=initialize_timeout)
        if response is None:
            self.status = ConnectionStatus.ERROR
            self.last_error = "initialize timeout"
            return False
        if "error" in response:
            err = response["error"]
            self.status = ConnectionStatus.ERROR
            self.last_error = f"initialize error: {err.get('message')}"
            return False

        result = response.get("result") or {}
        self.server_info = result.get("serverInfo")

        # initialized notification (notifications/initialized)
        try:
            self.transport.send(JsonRpcRequest(
                method="notifications/initialized",
                params={},
                id=None,
            ))
        except Exception:
            pass

        self.status = ConnectionStatus.CONNECTED
        return True

    def stop(self) -> None:
        try:
            self.transport.stop()
        finally:
            self.status = ConnectionStatus.DISCONNECTED

    # ---- discovery ----

    def discover_tools(self,
                       timeout: float = DEFAULT_LIST_TIMEOUT) -> list:
        """tools/list を呼んで self.tools に格納。"""
        response = self._send_and_wait(
            JsonRpcRequest(method="tools/list", params={},
                           id=self._next_id()),
            timeout=timeout,
        )
        if response is None or "error" in response:
            return []
        result = response.get("result") or {}
        raw_tools = result.get("tools") or []
        self.tools = [
            McpToolInfo(
                name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema"),
            )
            for t in raw_tools
        ]
        return self.tools

    def discover_resources(self,
                           timeout: float = DEFAULT_LIST_TIMEOUT) -> list:
        """resources/list を呼んで self.resources に格納。"""
        response = self._send_and_wait(
            JsonRpcRequest(method="resources/list", params={},
                           id=self._next_id()),
            timeout=timeout,
        )
        if response is None or "error" in response:
            return []
        result = response.get("result") or {}
        raw_res = result.get("resources") or []
        self.resources = [
            McpResourceInfo(
                uri=r.get("uri", ""),
                name=r.get("name", ""),
                description=r.get("description", ""),
                mime_type=r.get("mimeType"),
            )
            for r in raw_res
        ]
        return self.resources

    # ---- tool call ----

    def call_tool(self, tool_name: str, arguments: dict,
                  timeout: float = DEFAULT_CALL_TIMEOUT) -> dict:
        """tools/call を呼ぶ。戻り値: result dict (raw JSON-RPC result)。

        エラー時: {'error': {code, message}} を返す。
        """
        if self.status != ConnectionStatus.CONNECTED:
            return {"error": {"code": INTERNAL_ERROR,
                              "message": f"server not connected (status={self.status.value})"}}
        response = self._send_and_wait(
            JsonRpcRequest(
                method="tools/call",
                params={"name": tool_name, "arguments": arguments},
                id=self._next_id(),
            ),
            timeout=timeout,
        )
        if response is None:
            return {"error": {"code": INTERNAL_ERROR, "message": "timeout"}}
        if "error" in response:
            return {"error": response["error"]}
        return {"result": response.get("result") or {}}

    def read_resource(self, uri: str,
                      timeout: float = DEFAULT_CALL_TIMEOUT) -> dict:
        if self.status != ConnectionStatus.CONNECTED:
            return {"error": {"code": INTERNAL_ERROR,
                              "message": "server not connected"}}
        response = self._send_and_wait(
            JsonRpcRequest(
                method="resources/read",
                params={"uri": uri},
                id=self._next_id(),
            ),
            timeout=timeout,
        )
        if response is None:
            return {"error": {"code": INTERNAL_ERROR, "message": "timeout"}}
        if "error" in response:
            return {"error": response["error"]}
        return {"result": response.get("result") or {}}

    # ---- request/response 待機 ----

    def _send_and_wait(self, request: JsonRpcRequest,
                       timeout: float) -> Optional[dict]:
        """request を送って同じ id の response を受け取る。

        id が合わない message は discard (notification 等)。
        timeout で None。
        """
        import time
        try:
            self.transport.send(request)
        except Exception as e:
            self.last_error = f"send failed: {e}"
            return None

        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = max(0.01, deadline - time.time())
            msg = self.transport.recv(timeout=min(remaining, 1.0))
            if msg is None:
                continue
            if msg.get("id") == request.id:
                return msg
            # 別 id の response or notification は無視
        return None
