"""Noetic legacy tool stub (Phase 4 限定、Phase 5 で削除予定)。

Phase 4 Step E-2b で追加。ConversationRuntime.run_turn_with_forced_tool が
呼び出せる最小限の Noetic 固有 tool を ToolSpec 化する目的。

対象 5 個 (Step H 50 cycle 走行テストに必要):
  - output_display  : channel=device 発話、M1 channel_match_ratio 測定に必須
  - wait            : fallback、dismiss 機能
  - reflect         : E 値 + Opinion/Disposition 更新検証
  - update_self     : self モデル更新、state 変化量計測
  - search_memory   : memory network 検索

### 配置の理由
claw-code 準拠の `core/runtime/tools/` 配下には意図的に置かない
(`core/runtime/tools/__init__.py` docstring:
 "Noetic 固有機能 (recall / modify_self / camera_stream 等) は含めない")。
Phase 5 で削除する際に claw-code 準拠部分を汚さないための分離。

### Phase 5 移行方針
- `memory/project_phase4_noetic_stub_temp.md` 参照
- `IKU_REGEN_GUIDE.md` Step 0-B で削除
- 汎用操作 (post_message / observe_visual / query_memory 等) を claw-code
  準拠で実装完了時点で本ファイル削除 + main.py の register 呼出削除
"""
from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


# ============================================================
# 共通 input_schema (承認 3 層 + tool 固有 args)
# ============================================================

def _schema_with_approval(
    extra_properties: dict = None,
    required_extra: list = None,
) -> dict:
    """承認 3 層 + tool 固有フィールドを含む JSON schema を生成。

    承認 3 層: tool_intent / tool_expected_outcome / message
    (APPROVAL_PROMPT_SPEC §3 の仕様。PreToolUse hook が検証する)。

    additionalProperties: True で、LLM が schema にない任意 args を付与
    できる (Noetic 既存 tool は args の自由度が高いため緩く許容)。
    """
    props = {
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
    if extra_properties:
        props.update(extra_properties)
    required = ["tool_intent", "tool_expected_outcome", "message"]
    if required_extra:
        required.extend(required_extra)
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": True,
    }


# ============================================================
# Tool 仕様定義
# ============================================================

_OUTPUT_DISPLAY_SCHEMA = _schema_with_approval(
    extra_properties={
        "content": {"type": "string", "description": "発話内容"},
    },
    required_extra=["content"],
)

_WAIT_SCHEMA = _schema_with_approval(
    extra_properties={
        "dismiss": {
            "type": "string",
            "description": "任意: dismiss する pending の ID",
        },
    },
)

_REFLECT_SCHEMA = _schema_with_approval()

_UPDATE_SELF_SCHEMA = _schema_with_approval(
    extra_properties={
        "key": {"type": "string", "description": "self モデルのキー"},
        "value": {"type": "string", "description": "値"},
    },
    required_extra=["key", "value"],
)

_SEARCH_MEMORY_SCHEMA = _schema_with_approval(
    extra_properties={
        "query": {"type": "string", "description": "検索クエリ"},
        "limit": {"type": "integer", "default": 5},
    },
    required_extra=["query"],
)


# (name, description, required_permission, input_schema)
_SPECS: list[tuple] = [
    ("output_display",
     "端末前の協力者への発話 (channel=device)",
     PermissionMode.WORKSPACE_WRITE,
     _OUTPUT_DISPLAY_SCHEMA),
    ("wait",
     "待機、または pending の dismiss",
     PermissionMode.READ_ONLY,
     _WAIT_SCHEMA),
    ("reflect",
     "内省: Opinion / Disposition / Entity 更新",
     PermissionMode.WORKSPACE_WRITE,
     _REFLECT_SCHEMA),
    ("update_self",
     "self モデルの key=value 更新",
     PermissionMode.WORKSPACE_WRITE,
     _UPDATE_SELF_SCHEMA),
    ("search_memory",
     "memory network (entity / opinion / experience) からの検索",
     PermissionMode.READ_ONLY,
     _SEARCH_MEMORY_SCHEMA),
]


STUB_TOOL_NAMES = tuple(name for name, _, _, _ in _SPECS)


# ============================================================
# 登録関数
# ============================================================

def register_noetic_stubs(
    registry: ToolRegistry,
    tools_dict: dict,
) -> int:
    """5 個の Noetic 固有 tool を ToolSpec 化して registry に登録。

    handler は tools_dict[name]["func"] を参照するので、Noetic 既存の
    tool 実装がそのまま ConversationRuntime 経由で呼べる (動作は維持、
    呼出形式だけ function calling 化)。

    Args:
        registry: 登録先 ToolRegistry
        tools_dict: 既存 Noetic TOOLS dict ({"name": {"desc": ..., "func": ...}})

    Returns:
        登録した tool 数。

    Raises:
        KeyError: tools_dict に必要な tool が無い場合 (設定不整合)
    """
    count = 0
    for name, desc, perm, schema in _SPECS:
        if name not in tools_dict:
            raise KeyError(
                f"Noetic tool '{name}' が tools_dict に見つかりません。"
                f"tools/__init__.py の TOOLS 定義を確認してください。"
            )
        handler = tools_dict[name]["func"]
        spec = ToolSpec(
            name=name,
            description=desc,
            input_schema=schema,
            required_permission=perm,
            handler=handler,
        )
        registry.register(spec)
        count += 1
    return count
