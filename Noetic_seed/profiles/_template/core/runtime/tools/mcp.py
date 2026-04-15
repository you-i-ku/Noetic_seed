"""MCP — MCP / ListMcpResources / ReadMcpResource / McpAuth.

claw-code 参照:
  - rust/crates/runtime/src/mcp_tool_bridge.rs:1-921
  - rust/crates/runtime/src/mcp.rs:26-37

Phase 2 では MCP server protocol (JSON-RPC 2.0) 本体は実装せず、
registry + dispatch bridge の interface のみ提供。
set_mcp_bridge() で実 MCP client を差し込む設計。
"""
from typing import Callable, Optional

from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


# 実 MCP client を外部から注入するための bridge
_bridge: dict = {
    "call_tool": None,       # callable(server_name, tool_name, arguments) -> str
    "list_resources": None,  # callable(server_name) -> list
    "read_resource": None,   # callable(server_name, uri) -> str
    "auth": None,            # callable(server_name) -> str
}


def set_mcp_bridge(
    call_tool: Optional[Callable] = None,
    list_resources: Optional[Callable] = None,
    read_resource: Optional[Callable] = None,
    auth: Optional[Callable] = None,
) -> None:
    if call_tool is not None:
        _bridge["call_tool"] = call_tool
    if list_resources is not None:
        _bridge["list_resources"] = list_resources
    if read_resource is not None:
        _bridge["read_resource"] = read_resource
    if auth is not None:
        _bridge["auth"] = auth


# ============================================================
# MCP (generic tool dispatch)
# ============================================================

def mcp_call(inp: dict) -> str:
    server = (inp.get("server") or "").strip()
    tool = (inp.get("tool") or "").strip()
    arguments = inp.get("arguments") or {}
    if not server:
        return "Error: server is required"
    if not tool:
        return "Error: tool is required"

    fn = _bridge.get("call_tool")
    if fn is None:
        return (f"[MCP pending — bridge not configured]\n"
                f"server: {server}\n"
                f"tool: {tool}\n"
                f"arguments: {arguments}")
    try:
        return fn(server, tool, arguments)
    except Exception as e:
        return f"Error: MCP call failed: {e}"


# ============================================================
# ListMcpResources
# ============================================================

def list_mcp_resources(inp: dict) -> str:
    server = (inp.get("server") or "").strip()
    if not server:
        return "Error: server is required"

    fn = _bridge.get("list_resources")
    if fn is None:
        return f"[ListMcpResources pending — bridge not configured]\nserver: {server}"
    try:
        resources = fn(server)
    except Exception as e:
        return f"Error: list_resources failed: {e}"
    if not resources:
        return f"No resources on server '{server}'"
    lines = [f"Resources on '{server}' ({len(resources)}):"]
    for r in resources:
        uri = r.get("uri") if isinstance(r, dict) else str(r)
        name = r.get("name", "") if isinstance(r, dict) else ""
        lines.append(f"  - {uri}{'  ' + name if name else ''}")
    return "\n".join(lines)


# ============================================================
# ReadMcpResource
# ============================================================

def read_mcp_resource(inp: dict) -> str:
    server = (inp.get("server") or "").strip()
    uri = (inp.get("uri") or "").strip()
    if not server:
        return "Error: server is required"
    if not uri:
        return "Error: uri is required"

    fn = _bridge.get("read_resource")
    if fn is None:
        return (f"[ReadMcpResource pending — bridge not configured]\n"
                f"server: {server}\nuri: {uri}")
    try:
        return fn(server, uri)
    except Exception as e:
        return f"Error: read_resource failed: {e}"


# ============================================================
# McpAuth
# ============================================================

def mcp_auth(inp: dict) -> str:
    server = (inp.get("server") or "").strip()
    if not server:
        return "Error: server is required"

    fn = _bridge.get("auth")
    if fn is None:
        return f"[McpAuth pending — bridge not configured]\nserver: {server}"
    try:
        return fn(server)
    except Exception as e:
        return f"Error: auth failed: {e}"


# ============================================================
# Helpers (shared with registry.mcp_tool_name)
# ============================================================

def mcp_tool_name(server_name: str, tool_name: str) -> str:
    """claw-code/rust/crates/runtime/src/mcp.rs:26-37 準拠の名前正規化。"""
    return ToolRegistry.mcp_tool_name(server_name, tool_name)


# ============================================================
# register
# ============================================================

def register(registry: ToolRegistry) -> None:
    danger = PermissionMode.DANGER_FULL_ACCESS
    ro = PermissionMode.READ_ONLY
    specs = [
        ToolSpec(name="MCP",
                 description="Invoke an MCP server tool.",
                 input_schema={
                     "type": "object",
                     "properties": {
                         "server": {"type": "string"},
                         "tool": {"type": "string"},
                         "arguments": {"type": "object"},
                     },
                     "required": ["server", "tool"],
                 },
                 required_permission=danger, handler=mcp_call),
        ToolSpec(name="ListMcpResources",
                 description="List resources exposed by an MCP server.",
                 input_schema={
                     "type": "object",
                     "properties": {"server": {"type": "string"}},
                     "required": ["server"],
                 },
                 required_permission=ro, handler=list_mcp_resources),
        ToolSpec(name="ReadMcpResource",
                 description="Read an MCP resource by URI.",
                 input_schema={
                     "type": "object",
                     "properties": {
                         "server": {"type": "string"},
                         "uri": {"type": "string"},
                     },
                     "required": ["server", "uri"],
                 },
                 required_permission=ro, handler=read_mcp_resource),
        ToolSpec(name="McpAuth",
                 description="Authenticate with an MCP server (OAuth/credentials).",
                 input_schema={
                     "type": "object",
                     "properties": {"server": {"type": "string"}},
                     "required": ["server"],
                 },
                 required_permission=danger, handler=mcp_auth),
    ]
    for s in specs:
        registry.register(s)
