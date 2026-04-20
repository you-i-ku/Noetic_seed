"""Noetic_seed MCP server (段階6-C v3)。

FastMCP SDK 経由で 4 tool を expose し、外部 MCP client (Claude Code 等)
との接続を成立させる。

設計指針:
- FastMCP を採用 (低レベル Server より ergonomic)
- stdio transport (ローカル stdin/stdout 経由、認証不要)
- stdout は JSON-RPC 専用、ログは stderr へ (print() 禁止、logging 経由推奨)
- tool 関数は seed_tools.py で定義、ここでは mcp.tool() で登録するだけ
"""
from mcp.server.fastmcp import FastMCP

from core.runtime.mcp.server.seed_tools import (
    noetic_seed_get_state,
    noetic_seed_send_message,
    noetic_seed_get_recent_outputs,
    noetic_seed_get_wm_snapshot,
)


mcp_server = FastMCP("noetic-seed")

# 4 tool を登録 (FastMCP は型ヒント + docstring から inputSchema を自動生成)
mcp_server.tool()(noetic_seed_get_state)
mcp_server.tool()(noetic_seed_send_message)
mcp_server.tool()(noetic_seed_get_recent_outputs)
mcp_server.tool()(noetic_seed_get_wm_snapshot)
