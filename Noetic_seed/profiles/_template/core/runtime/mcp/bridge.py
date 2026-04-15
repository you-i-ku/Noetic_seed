"""McpToolBridge — 複数 MCP server の統合 registry。

claw-code 参照: rust/crates/runtime/src/mcp_tool_bridge.rs:1-921

責務:
  - 複数 server の起動/停止をまとめて管理
  - 各 server の tools を Noetic ToolRegistry に prefix 付きで登録
  - tool 呼出時に prefix から server を特定して dispatch
  - health monitoring (status / last_error)
"""
import threading
from dataclasses import dataclass
from typing import Optional

from core.runtime.mcp.client import BaseTransport
from core.runtime.mcp.manager import (
    ConnectionStatus, McpServerManager, McpResourceInfo, McpToolInfo,
)
from core.runtime.mcp.naming import (
    mcp_tool_name as make_mcp_name,
    parse_mcp_tool_name,
)
from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


@dataclass
class ServerHealth:
    """health monitor 用のスナップショット。"""
    name: str
    status: str
    tool_count: int
    resource_count: int
    last_error: Optional[str]


class McpToolBridge:
    """MCP servers を Noetic ToolRegistry に統合する。"""

    def __init__(self, tool_registry: ToolRegistry):
        self._registry = tool_registry
        self._managers: dict = {}   # server_name -> McpServerManager
        self._lock = threading.RLock()

    # ---- server 登録 ----

    def add_server(self, server_name: str, transport: BaseTransport,
                   start: bool = True,
                   auto_discover: bool = True) -> McpServerManager:
        """server を登録 (optionally 起動 + discover)。"""
        with self._lock:
            if server_name in self._managers:
                raise ValueError(f"server '{server_name}' already registered")
            mgr = McpServerManager(server_name, transport)
            self._managers[server_name] = mgr

        if start:
            ok = mgr.start()
            if ok and auto_discover:
                mgr.discover_tools()
                mgr.discover_resources()
                self._register_tools_to_registry(server_name, mgr.tools)
        return mgr

    def remove_server(self, server_name: str) -> bool:
        with self._lock:
            mgr = self._managers.pop(server_name, None)
        if mgr is None:
            return False
        # 登録されていた tool を unregister
        self._unregister_tools(server_name, mgr.tools)
        mgr.stop()
        return True

    def stop_all(self) -> None:
        with self._lock:
            names = list(self._managers.keys())
        for name in names:
            self.remove_server(name)

    # ---- 動的 discovery / refresh ----

    def refresh_tools(self, server_name: str) -> list:
        """server に tools/list を再問合せして registry に反映。"""
        mgr = self._managers.get(server_name)
        if mgr is None:
            return []
        old = list(mgr.tools)
        new = mgr.discover_tools()
        self._unregister_tools(server_name, old)
        self._register_tools_to_registry(server_name, new)
        return new

    # ---- dispatch ----

    def call(self, server_name: str, tool_name: str,
             arguments: dict) -> dict:
        """raw call (prefix 剥がし済みの tool_name を指定)。"""
        mgr = self._managers.get(server_name)
        if mgr is None:
            return {"error": {"code": -32000,
                              "message": f"server '{server_name}' not registered"}}
        return mgr.call_tool(tool_name, arguments)

    def call_by_full_name(self, full_name: str,
                          arguments: dict) -> dict:
        """'mcp__slack__post_msg' のような prefix 付き名前で呼出。"""
        server, tool = parse_mcp_tool_name(full_name)
        if server is None:
            return {"error": {"code": -32000,
                              "message": f"not an MCP tool name: {full_name}"}}
        return self.call(server, tool, arguments)

    def list_resources(self, server_name: str) -> list:
        mgr = self._managers.get(server_name)
        if mgr is None:
            return []
        return mgr.resources

    def read_resource(self, server_name: str, uri: str) -> dict:
        mgr = self._managers.get(server_name)
        if mgr is None:
            return {"error": {"code": -32000,
                              "message": f"server '{server_name}' not registered"}}
        return mgr.read_resource(uri)

    # ---- health ----

    def health_snapshot(self) -> list:
        """全 server の健康状態。"""
        out: list = []
        with self._lock:
            for name, mgr in self._managers.items():
                out.append(ServerHealth(
                    name=name,
                    status=mgr.status.value,
                    tool_count=len(mgr.tools),
                    resource_count=len(mgr.resources),
                    last_error=mgr.last_error,
                ))
        return out

    def get_manager(self, server_name: str) -> Optional[McpServerManager]:
        return self._managers.get(server_name)

    # ---- Noetic ToolRegistry 連携 ----

    def _register_tools_to_registry(self, server_name: str,
                                     tools: list) -> None:
        """MCP tools を prefix 付き名前で Noetic ToolRegistry に登録。"""
        for info in tools:
            full_name = make_mcp_name(server_name, info.name)
            spec = ToolSpec(
                name=full_name,
                description=info.description or f"MCP tool from {server_name}",
                input_schema=info.input_schema or {"type": "object"},
                # MCP tool はデフォルトで DANGER (外部プロセス接続なので)
                required_permission=PermissionMode.DANGER_FULL_ACCESS,
                handler=self._make_handler(server_name, info.name),
            )
            self._registry.register(spec)

    def _unregister_tools(self, server_name: str, tools: list) -> None:
        for info in tools:
            full_name = make_mcp_name(server_name, info.name)
            self._registry.unregister(full_name)

    def _make_handler(self, server_name: str, tool_name: str):
        """MCP tool を呼び出すクロージャを生成 (ToolSpec.handler 用)。"""
        def _handler(inp: dict) -> str:
            import json as _json
            r = self.call(server_name, tool_name, inp)
            if "error" in r:
                err = r["error"]
                return (f"[MCP error {err.get('code','?')}] "
                        f"{err.get('message','unknown')}")
            result = r.get("result") or {}
            # MCP tools/call response 形式:
            #   {"content": [{"type":"text","text":"..."}, ...]}
            content = result.get("content")
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict):
                        ct = c.get("type")
                        if ct == "text":
                            parts.append(c.get("text", ""))
                        elif ct == "resource":
                            parts.append(f"[resource: {c.get('resource', {}).get('uri', '')}]")
                        else:
                            parts.append(_json.dumps(c, ensure_ascii=False))
                return "\n".join(parts) if parts else "(empty response)"
            # plain result
            return _json.dumps(result, ensure_ascii=False)

        return _handler
