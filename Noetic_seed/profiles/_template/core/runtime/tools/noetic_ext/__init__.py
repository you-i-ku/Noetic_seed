"""Noetic 固有 tool 群 (claw 文法準拠 ToolSpec)。

Phase 4 Step H-2 C.4 Session B (2026-04-18) で新設。

### 位置づけ

claw-code の `core/runtime/tools/` は runtime 汎用 tool 群 (file_ops/shell/web/
task/worker/mcp/…) を扱う。Noetic 固有機能 (認知 / 記憶 / 身体化 / 認証) は
本 package に per-family modules として分離配置する。

### 設計原則 (claw 文法準拠)

- tool 定義: `ToolSpec` (name / description / input_schema / permission / handler)
- input_schema: `{"type": "object", "properties": {...}, "required": [...], "additionalProperties": False}`
- 引数名: snake_case 原則、型明示 (string / integer / number / enum)
- description: 機能中心の 1-2 sentence (日本語、Noetic 既存規約)
- handler: **legacy func 温存**。tools/__init__.py の TOOLS[name]["func"] を参照
- permission: READ_ONLY (観測・query) / WORKSPACE_WRITE (state 変更) /
  DANGER_FULL_ACCESS (外部装置作動・機密書込)
- 承認 3 層 (tool_intent / tool_expected_outcome / message): 全 tool で required

### 本 package の責務外

- legacy handler 実装: tools/*.py (builtin.py / memory_tool.py / device_tools.py 等) のまま
- dangerous pattern / sandbox 境界等の tool 横断 security: hook 層 (core/runtime/hooks.py の make_file_access_guard 等)
- Phase 5 で handler を claw 相当に差し替える際、本 package の ToolSpec 記述は
  そのまま流用可能 (handler 参照だけ変わる)

### family 分割

- cognition.py : reflect / update_self / output_display / wait
- memory.py    : search_memory / memory_store / memory_update / memory_forget
- sense.py     : view_image / listen_audio / mic_record / camera_stream /
                  camera_stream_stop / screen_peek
- auth.py      : auth_profile_info / secret_read / secret_write
"""
from core.runtime.registry import ToolRegistry


def register_noetic_tools(registry: ToolRegistry, tools_dict: dict) -> int:
    """Noetic 固有 17 tool を per-family で一括登録する。

    claw-code の register_all pattern 準拠。各 family module の register()
    を順に呼び、登録総数を返す。

    Args:
        registry: 登録先 ToolRegistry
        tools_dict: legacy TOOLS dict (handler 参照元)

    Returns:
        登録した tool 総数 (想定 17)。
    """
    from core.runtime.tools.noetic_ext import auth, cognition, memory, sense
    total = 0
    total += cognition.register(registry, tools_dict)
    total += memory.register(registry, tools_dict)
    total += sense.register(registry, tools_dict)
    total += auth.register(registry, tools_dict)
    return total


# 登録対象の tool 名一覧 (bridge の skip_names 用、test の検証用)
NOETIC_TOOL_NAMES = frozenset({
    # cognition
    "reflect", "update_self", "output_display", "wait",
    # memory
    "search_memory", "memory_store", "memory_update", "memory_forget",
    # sense
    "view_image", "listen_audio", "mic_record",
    "camera_stream", "camera_stream_stop", "screen_peek",
    # auth
    "auth_profile_info", "secret_read", "secret_write",
})
