"""MCP 名前正規化。

claw-code 参照: rust/crates/runtime/src/mcp.rs:26-37
"""


def normalize_name_for_mcp(s: str) -> str:
    """英数字とアンダースコア以外を _ に置換。"""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in s)


def mcp_tool_prefix(server_name: str) -> str:
    """例: 'slack-server' -> 'mcp__slack_server__'"""
    return f"mcp__{normalize_name_for_mcp(server_name)}__"


def mcp_tool_name(server_name: str, tool_name: str) -> str:
    """例: ('slack-server', 'post_message')
        -> 'mcp__slack_server__post_message'"""
    return f"{mcp_tool_prefix(server_name)}{normalize_name_for_mcp(tool_name)}"


def parse_mcp_tool_name(full_name: str) -> tuple:
    """逆変換: 'mcp__<server>__<tool>' -> (server, tool)。
    非 MCP 名なら (None, full_name)。"""
    if not full_name.startswith("mcp__"):
        return (None, full_name)
    rest = full_name[5:]
    idx = rest.find("__")
    if idx < 0:
        return (None, full_name)
    return (rest[:idx], rest[idx + 2:])
