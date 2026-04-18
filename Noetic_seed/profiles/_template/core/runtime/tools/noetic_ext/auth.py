"""Noetic 認証・秘密層 tool 群 — auth_profile_info / secret_read / secret_write。

claw 文法準拠 ToolSpec。handler は legacy (tools/auth_tools.py, tools/secret_tools.py)
を温存。secrets は sandbox/secrets/ に隔離され、読取は承認不要 / 書込は承認必須。
"""
from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


def _approval_props() -> dict:
    return {
        "tool_intent": {
            "type": "string",
            "description": "あなたの内部理由 (1 文、80 字目安)",
        },
        "tool_expected_outcome": {
            "type": "string",
            "description": "期待する結果 (1 文、80 字目安)",
        },
        "message": {
            "type": "string",
            "description": "端末前の協力者への一言",
        },
    }


_APPROVAL_REQUIRED = ["tool_intent", "tool_expected_outcome", "message"]


def _build_specs(tools_dict: dict) -> list:
    specs = [
        ToolSpec(
            name="auth_profile_info",
            description=(
                "認証プロファイルのメタ情報を取得する。name 省略で一覧、"
                "指定で詳細。機密フィールド (token / key / secret 等) は除外される。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "プロファイル名 (省略時は登録プロファイル一覧)",
                    },
                    **_approval_props(),
                },
                "required": list(_APPROVAL_REQUIRED),
                "additionalProperties": False,
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=tools_dict["auth_profile_info"]["func"],
        ),
        ToolSpec(
            name="secret_read",
            description=(
                "sandbox/secrets/ に保存された秘密情報を読む。承認不要だが、"
                "name パターン (英数字 / _ . -) に従う必要あり。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "secret 名 (英数字 / _ . -、最大 120 字)",
                    },
                    **_approval_props(),
                },
                "required": ["name", *_APPROVAL_REQUIRED],
                "additionalProperties": False,
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=tools_dict["secret_read"]["func"],
        ),
        ToolSpec(
            name="secret_write",
            description=(
                "sandbox/secrets/ に秘密情報を書き込む。承認必須、最大 1 MB、"
                "name は英数字 / _ . - のみ。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "secret 名 (英数字 / _ . -、最大 120 字)",
                    },
                    "content": {
                        "type": "string",
                        "description": "書き込む内容 (最大 1 MB)",
                    },
                    **_approval_props(),
                },
                "required": ["name", "content", *_APPROVAL_REQUIRED],
                "additionalProperties": False,
            },
            required_permission=PermissionMode.DANGER_FULL_ACCESS,
            handler=tools_dict["secret_write"]["func"],
        ),
    ]
    return specs


def register(registry: ToolRegistry, tools_dict: dict) -> int:
    """auth family の 3 tool を registry に登録。"""
    specs = _build_specs(tools_dict)
    for spec in specs:
        registry.register(spec)
    return len(specs)
