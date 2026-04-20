"""MCP server (段階6-C v3) — Noetic を MCP server 化して外部 client (Claude Code 等) と接続。

- FastMCP SDK 経由で 4 tool (iku_get_state / iku_send_message /
  iku_get_recent_outputs / iku_get_wm_snapshot) を expose
- 接続元 client_info から channel_registry の判定関数で channel spec 生成
- accessor-only 原則: state.json / WM の書き込みは pending_claude_inputs.jsonl 経由、
  直接 WM を操作しない
"""
