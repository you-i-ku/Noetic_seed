"""Tool Registry — 名前 → ToolSpec のマップ。

claw-code の global_tool_registry (rust/crates/tools/src/lib.rs:1-80) +
mcp_tool_bridge (rust/crates/runtime/src/mcp_tool_bridge.rs) の Python port。

厳密 claw-code 準拠。Noetic 固有機能は含めない。
"""
from typing import Optional

from core.runtime.tool_schema import ToolSpec


class ToolRegistry:
    """tool の集中レジストリ。"""

    def __init__(self):
        self._tools: dict = {}

    def register(self, spec: ToolSpec) -> None:
        """tool を登録。同名は上書き。"""
        self._tools[spec.name] = spec

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def all_names(self) -> list:
        return list(self._tools.keys())

    def list(
        self,
        max_permission=None,  # PermissionMode
        allowlist: Optional[list] = None,
        denylist: Optional[list] = None,
    ) -> list:
        """フィルタ付き tool 一覧を返す。

        max_permission:  この permission 以下で動く tool のみ
        allowlist:       含まれる tool 名のみ
        denylist:        除外する tool 名
        """
        from core.runtime.permissions import _MODE_LEVEL

        max_level = _MODE_LEVEL.get(max_permission, 99) if max_permission else 99
        out = []
        for spec in self._tools.values():
            if allowlist is not None and spec.name not in allowlist:
                continue
            if denylist is not None and spec.name in denylist:
                continue
            spec_level = _MODE_LEVEL.get(spec.required_permission, 99)
            if spec_level > max_level:
                continue
            out.append(spec)
        return out

    def execute(self, name: str, tool_input: dict) -> str:
        """tool を実行して結果 (str) を返す。未登録なら ValueError。"""
        spec = self.get(name)
        if spec is None:
            raise ValueError(f"Tool not registered: {name}")
        return spec.handler(tool_input)

    # ---- MCP tool 用ヘルパ ----

    @staticmethod
    def mcp_tool_name(server_name: str, tool_name: str) -> str:
        """MCP tool の prefix 付き正規化名。
        claw-code/rust/crates/runtime/src/mcp.rs:26-37 に準拠。"""
        def normalize(s: str) -> str:
            return "".join(c if c.isalnum() or c == "_" else "_" for c in s)

        return f"mcp__{normalize(server_name)}__{normalize(tool_name)}"

    def is_mcp_tool(self, name: str) -> bool:
        return name.startswith("mcp__")
