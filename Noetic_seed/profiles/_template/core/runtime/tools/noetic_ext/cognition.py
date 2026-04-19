"""Noetic 認知層 tool 群 — reflect / update_self / output_display / wait。

claw 文法準拠 ToolSpec。handler は legacy (tools/ 配下) を温存。
"""
from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


def _approval_props() -> dict:
    """全 Noetic tool 共通の承認 3 層 properties。"""
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
            "description": "端末前の協力者への一言 (対等な口調、報告・共有)",
        },
    }


_APPROVAL_REQUIRED = ["tool_intent", "tool_expected_outcome", "message"]


def _build_specs(tools_dict: dict) -> list:
    specs = [
        ToolSpec(
            name="reflect",
            description=(
                "内省を実行し、Opinion / Disposition / Entity の更新と記憶保存を行う。"
                "E4 多様性指標にも影響。引数なし。"
            ),
            input_schema={
                "type": "object",
                "properties": _approval_props(),
                "required": list(_APPROVAL_REQUIRED),
                "additionalProperties": False,
            },
            required_permission=PermissionMode.WORKSPACE_WRITE,
            handler=tools_dict["reflect"]["func"],
        ),
        ToolSpec(
            name="update_self",
            description=(
                "自己モデル (state.self) の属性を更新する。key-value pair で記録、"
                "name は一度確定すると変更不可。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "self モデルのキー名",
                    },
                    "value": {
                        "type": "string",
                        "description": "格納する値",
                    },
                    **_approval_props(),
                },
                "required": ["key", "value", *_APPROVAL_REQUIRED],
                "additionalProperties": False,
            },
            required_permission=PermissionMode.WORKSPACE_WRITE,
            handler=tools_dict["update_self"]["func"],
        ),
        ToolSpec(
            name="output_display",
            description=(
                "発話を channel 指定で届ける。送信先 channel は WM.channels を観察して決定、"
                "受信 channel に対応させて返すこと (段階6-C v3: 観察駆動)。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "発話内容",
                    },
                    "channel": {
                        "type": "string",
                        "description": (
                            "送信先 channel id (必須)。WM.channels を観察して利用可能な "
                            "channel から選び、受信 channel に対応させて返す。"
                        ),
                    },
                    **_approval_props(),
                },
                "required": ["content", "channel", *_APPROVAL_REQUIRED],
                "additionalProperties": False,
            },
            required_permission=PermissionMode.WORKSPACE_WRITE,
            handler=tools_dict["output_display"]["func"],
        ),
        ToolSpec(
            name="wait",
            description=(
                "待機する。dismiss で特定 pending を明示的に却下できる "
                "(省略時は単純待機)。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "dismiss": {
                        "type": "string",
                        "description": (
                            "却下する pending の ID (省略時は単純待機)。"
                            "必ず 'p_' から始まる ID を指定する "
                            "(例: p_30c4d130_0008_listen_a_123)。"
                            "[未対応事項] 欄に dismiss_id=... として表示されているものを使う。"
                            "[STM log] 欄の log ID (prefix なしの {session}_{cycle} 形式) は "
                            "dismiss に渡してはいけない。"
                        ),
                    },
                    **_approval_props(),
                },
                "required": list(_APPROVAL_REQUIRED),
                "additionalProperties": False,
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=tools_dict["wait"]["func"],
        ),
    ]
    return specs


def register(registry: ToolRegistry, tools_dict: dict) -> int:
    """cognition family の 4 tool を registry に登録。"""
    specs = _build_specs(tools_dict)
    for spec in specs:
        registry.register(spec)
    return len(specs)
