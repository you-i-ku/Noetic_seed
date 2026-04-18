"""Legacy Noetic tool bridge — H-2 移行期の一時橋渡し。

Phase 4 Step H-2 A (2026-04-18) で新設、H-2 D で削除予定。

### 目的

legacy TOOLS dict (tools/__init__.py の 40 tool) を ToolSpec 化して
ToolRegistry に自動登録する。これにより H-2 B の controller 切替後、
legacy tool も ConversationRuntime 経由で LLM② から呼出可能になる。

### 動作

- 全 legacy TOOLS エントリを走査し、ToolSpec 化して registry に登録
- registry に既登録の name は **上書き** (overwrite)。legacy 側のセキュリティ
  guard (read_file の sandbox/secrets/ ガード、write_file の書込先制限等) を
  保つため、claw 同名 tool があっても legacy が勝つ
- skip_names で明示除外可能 (noetic_stub 等、bridge より後で登録したい場合に使用)
- handler は tools_dict[name]["func"] 呼出の単純な passthrough
- input_schema は承認 3 層 + additionalProperties=True の緩い形
  (H-2 C.4 で native ToolSpec に昇格時に厳密化される)

### 登録順序 (main.py)

```
register_claw_tools(registry, ...)      # 1. claw 50 を配置
register_legacy_bridge(registry, TOOLS) # 2. legacy 40 で上書き
                                         #    (read_file/write_file は legacy の
                                         #     secrets guard 版が勝つ)
register_noetic_stubs(registry, TOOLS)  # 3. stub 5 の厳密 schema で最終上書き
```

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

    既登録 name は上書きする (overwrite)。legacy のセキュリティガードを保つため
    claw 同名 tool を上書きする設計。明示的に除外したい場合は skip_names で指定。

    Args:
        registry: 登録先 ToolRegistry
        tools_dict: legacy TOOLS dict ({name: {"desc": str, "func": callable}})
        skip_names: 除外する tool 名の set (上書き対象外にしたい name)

    Returns:
        実際に登録した tool 数 (skip した数は含まない)。
    """
    schema = _make_passthrough_schema()
    count = 0
    for name, meta in tools_dict.items():
        if name in skip_names:
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
