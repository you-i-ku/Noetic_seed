"""Legacy Noetic tool bridge — H-2 移行期の一時橋渡し。

Phase 4 Step H-2 A (2026-04-18) で新設、H-2 D で削除予定。

### 目的

legacy TOOLS dict (tools/__init__.py の 40 tool) を ToolSpec 化して
ToolRegistry に自動登録する。これにより H-2 B の controller 切替後、
legacy tool も ConversationRuntime 経由で LLM② から呼出可能になる。

### 動作

- 全 legacy TOOLS エントリを走査
- registry.has(name) が True の場合 (claw-code / noetic_stub が先に登録済)
  は skip。claw / stub 側を尊重する
- skip_names で追加除外可能
- handler は tools_dict[name]["func"] 呼出の単純な passthrough
- input_schema は承認 3 層 + additionalProperties=True の緩い形
  (H-2 C.4 で native ToolSpec に昇格時に厳密化される)

### Phase 5 方針

H-2 D で本モジュール削除 + TOOLS dict 廃止。Phase 5 の個別 tool 廃止思想
(memory/feedback_no_individual_tools.md) に沿って、汎用操作 + MCP server
構成に移行する時点で本モジュールは役目を終える。
"""
from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


# READ_ONLY 判定: 承認不要で動作する legacy tool の一覧。
# それ以外は WORKSPACE_WRITE (承認必要) として扱う。
# 注: mic_record / camera_stream / screen_peek は録音・撮影なので承認必要。
_READ_ONLY_LEGACY_TOOLS = frozenset({
    "wait",
    "list_files", "read_file",
    "search_memory", "memory_store",
    "web_search", "fetch_url",
    "elyth_info", "elyth_get", "elyth_mark_read",
    "x_timeline", "x_search", "x_get_notifications",
    "auth_profile_info", "secret_read",
    "view_image", "listen_audio",
})


def _make_passthrough_schema() -> dict:
    """承認 3 層 + free-form args の最小 schema。

    bridge 経由の tool は args 形式が多様なので additionalProperties=True で
    LLM の任意指定を許容する。H-2 C.4 で native ToolSpec に昇格時に厳密な
    schema に置換する。
    """
    return {
        "type": "object",
        "properties": {
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
        },
        "required": ["tool_intent", "tool_expected_outcome", "message"],
        "additionalProperties": True,
    }


def register_legacy_bridge(
    registry: ToolRegistry,
    tools_dict: dict,
    *,
    skip_names: frozenset = frozenset(),
) -> int:
    """legacy TOOLS dict の全エントリを ToolSpec で passthrough 登録する。

    既に同名が registry に存在する場合 (claw-code / noetic_stub が先に登録済)
    は skip する。skip_names で追加除外可能。

    Args:
        registry: 登録先 ToolRegistry
        tools_dict: legacy TOOLS dict ({name: {"desc": str, "func": callable}})
        skip_names: 除外する tool 名の set

    Returns:
        実際に登録した tool 数 (skip した数は含まない)。
    """
    schema = _make_passthrough_schema()
    count = 0
    for name, meta in tools_dict.items():
        if name in skip_names:
            continue
        if registry.has(name):
            continue
        perm = (
            PermissionMode.READ_ONLY
            if name in _READ_ONLY_LEGACY_TOOLS
            else PermissionMode.WORKSPACE_WRITE
        )
        spec = ToolSpec(
            name=name,
            description=meta.get("desc", name),
            input_schema=schema,
            required_permission=perm,
            handler=meta["func"],
        )
        registry.register(spec)
        count += 1
    return count
