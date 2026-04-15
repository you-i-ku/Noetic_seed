"""Tool Registry — 名前 → ToolSpec のマップ。

claw-code の global_tool_registry + mcp_tool_bridge の Python port。

責務:
  - native tool の登録
  - MCP tool の登録 (prefix "mcp__<server>__")
  - tool 名から spec を検索
  - フィルタリング (tool_level / channel / permission)

TODO: 別セッションで実装。
"""
from typing import Optional

from core.runtime.tool_schema import ToolSpec


class ToolRegistry:
    """tool の集中レジストリ。

    まだ未実装。
    """

    def __init__(self):
        self._tools: dict = {}

    def register(self, spec: ToolSpec) -> None:
        """tool を登録。"""
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        """tool を取得。"""
        return self._tools.get(name)

    def list(self, channel: Optional[str] = None,
             max_permission: Optional[str] = None) -> list:
        """フィルタ付き tool 一覧。"""
        raise NotImplementedError

    def execute(self, name: str, tool_input: dict) -> str:
        """tool を実行して結果を返す。"""
        raise NotImplementedError
