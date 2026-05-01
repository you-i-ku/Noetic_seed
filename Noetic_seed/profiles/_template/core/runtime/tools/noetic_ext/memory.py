"""Noetic 記憶層 tool 群 — search_memory / memory_store / memory_update / memory_forget。

claw 文法準拠 ToolSpec。handler は legacy (tools/memory_tool.py) を温存。
記憶ネットワーク 4 層 (world / experience / opinion / entity) を操作する。
"""
from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


# 段階7: _NETWORKS 撤去 → tag_registry で動的検証 (schema の enum も削除)


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
            name="search_memory",
            description=(
                "記憶ネットワーク (動的拡張可) を "
                "ベクトル + キーワード検索する。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "検索クエリ",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "最大件数 (default 5)",
                    },
                    **_approval_props(),
                },
                "required": ["query", *_APPROVAL_REQUIRED],
                "additionalProperties": False,
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=tools_dict["search_memory"]["func"],
        ),
        ToolSpec(
            name="memory_store",
            description=(
                "記憶ネットワークに新エントリを保存する。network 省略 → untagged 経路 "
                "(11-D Phase 1)、network 指定 + 未登録 → auto register、rules 省略可。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "network": {
                        "type": "string",
                        "description": "保存先のタグ名 (省略可、動的拡張可、未登録タグは auto register)",
                    },
                    "content": {
                        "type": "string",
                        "description": "記憶内容",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "確信度 (opinion/entity で使用)",
                    },
                    "entity_name": {
                        "type": "string",
                        "description": "エンティティ名 (entity network の場合)",
                    },
                    "relationship": {
                        "type": "string",
                        "description": "関係性ラベル (entity network の場合)",
                    },
                    "rules": {
                        "type": "object",
                        "description": "(11-D 以降 任意) bitemporal=True (既存 fact 凍結) や write_protected=True (書込み禁止 pseudo-tag) を指定したい時のみ渡す。省略時は全 False default で auto register",
                        "properties": {
                            "beta_plus": {"type": "boolean"},
                            "bitemporal": {"type": "boolean"},
                            "c_gradual_source": {"type": "boolean"},
                            "write_protected": {"type": "boolean"},
                        },
                    },
                    "display_format": {
                        "type": "string",
                        "description": "(段階7: 新タグ発明時 任意) format_memories_for_prompt 用テンプレート。省略時は [{name}] {content}",
                    },
                    **_approval_props(),
                },
                "required": ["network", "content", *_APPROVAL_REQUIRED],
                "additionalProperties": False,
            },
            required_permission=PermissionMode.WORKSPACE_WRITE,
            handler=tools_dict["memory_store"]["func"],
        ),
        ToolSpec(
            name="memory_update",
            description=(
                "既存の記憶エントリを更新する。memory_id で対象指定、"
                "content / confidence を部分上書き。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "対象記憶の ID",
                    },
                    "content": {
                        "type": "string",
                        "description": "新しい内容 (省略時は confidence のみ更新)",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "新しい確信度",
                    },
                    **_approval_props(),
                },
                "required": ["memory_id", *_APPROVAL_REQUIRED],
                "additionalProperties": False,
            },
            required_permission=PermissionMode.WORKSPACE_WRITE,
            handler=tools_dict["memory_update"]["func"],
        ),
        ToolSpec(
            name="memory_forget",
            description="記憶エントリを削除する。memory_id で対象指定。",
            input_schema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "削除する記憶の ID",
                    },
                    **_approval_props(),
                },
                "required": ["memory_id", *_APPROVAL_REQUIRED],
                "additionalProperties": False,
            },
            required_permission=PermissionMode.WORKSPACE_WRITE,
            handler=tools_dict["memory_forget"]["func"],
        ),
    ]
    return specs


def register(registry: ToolRegistry, tools_dict: dict) -> int:
    """memory family の 4 tool を registry に登録。"""
    specs = _build_specs(tools_dict)
    for spec in specs:
        registry.register(spec)
    return len(specs)
