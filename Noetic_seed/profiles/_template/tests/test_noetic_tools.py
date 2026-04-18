"""noetic_ext.register_noetic_tools テスト。

Phase 4 Step H-2 C.4 Session B (2026-04-18): 17 個の Noetic 固有 tool を
claw 文法準拠 ToolSpec で ToolRegistry に登録する動作を検証。
handler は tools_dict 経由で legacy Noetic 実装を温存する想定。

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_noetic_tools.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tools.noetic_ext import NOETIC_TOOL_NAMES, register_noetic_tools


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _fake_tools_dict():
    """Noetic 17 tool + 余分を含む TOOLS dict 模倣。"""
    names = [
        "output_display", "wait", "reflect", "update_self",
        "search_memory", "memory_store", "memory_update", "memory_forget",
        "view_image", "listen_audio", "mic_record",
        "camera_stream", "camera_stream_stop", "screen_peek",
        "auth_profile_info", "secret_read", "secret_write",
        # 余分 (noetic_ext 対象外、無視されるべき)
        "elyth_post", "x_post", "create_tool",
    ]
    d = {}
    for n in names:
        d[n] = {"desc": n, "func": (lambda name: lambda inp: f"[{name}] " + str(inp)[:40])(n)}
    return d


# ============================================================
# 登録の基本動作
# ============================================================

def test_register_all_17():
    print("== 17 個全部登録される ==")
    reg = ToolRegistry()
    n = register_noetic_tools(reg, _fake_tools_dict())
    return all([
        _assert(n == 17, f"登録数=17 (実={n})"),
        _assert(len(reg.list()) == 17, f"registry 件数=17 (実={len(reg.list())})"),
    ])


def test_tool_names_exposed():
    print("== NOETIC_TOOL_NAMES が 17 個を export ==")
    return all([
        _assert(len(NOETIC_TOOL_NAMES) == 17, f"17 個 (実={len(NOETIC_TOOL_NAMES)})"),
        _assert("reflect" in NOETIC_TOOL_NAMES, "reflect"),
        _assert("memory_store" in NOETIC_TOOL_NAMES, "memory_store"),
        _assert("camera_stream" in NOETIC_TOOL_NAMES, "camera_stream"),
        _assert("secret_write" in NOETIC_TOOL_NAMES, "secret_write"),
    ])


def test_registered_tool_names_match():
    print("== 登録された tool 名が NOETIC_TOOL_NAMES と一致 ==")
    reg = ToolRegistry()
    register_noetic_tools(reg, _fake_tools_dict())
    names = {spec.name for spec in reg.list()}
    return _assert(names == set(NOETIC_TOOL_NAMES),
                   f"一致: 差分={names ^ set(NOETIC_TOOL_NAMES)}")


def test_extra_tools_ignored():
    print("== 対象外 tool (elyth_post / x_post / create_tool) は登録されない ==")
    reg = ToolRegistry()
    register_noetic_tools(reg, _fake_tools_dict())
    return all([
        _assert(not reg.has("elyth_post"), "elyth_post 登録なし"),
        _assert(not reg.has("x_post"), "x_post 登録なし"),
        _assert(not reg.has("create_tool"), "create_tool 登録なし"),
    ])


# ============================================================
# input_schema 検証 (claw 文法準拠チェック)
# ============================================================

def test_schema_has_approval_3_layer():
    print("== 全 tool の input_schema に 3 層 required ==")
    reg = ToolRegistry()
    register_noetic_tools(reg, _fake_tools_dict())
    results = []
    for name in NOETIC_TOOL_NAMES:
        spec = reg.get(name)
        required = set(spec.input_schema.get("required", []))
        ok = {"tool_intent", "tool_expected_outcome", "message"} <= required
        results.append(_assert(ok, f"{name}"))
    return all(results)


def test_schema_type_object_everywhere():
    print("== 全 tool の schema が type=object ==")
    reg = ToolRegistry()
    register_noetic_tools(reg, _fake_tools_dict())
    results = []
    for name in NOETIC_TOOL_NAMES:
        spec = reg.get(name)
        results.append(_assert(
            spec.input_schema.get("type") == "object",
            f"{name}: type=object"))
    return all(results)


def test_schema_additional_properties_false():
    print("== 全 tool の schema が additionalProperties=False (claw 厳密モード) ==")
    reg = ToolRegistry()
    register_noetic_tools(reg, _fake_tools_dict())
    results = []
    for name in NOETIC_TOOL_NAMES:
        spec = reg.get(name)
        results.append(_assert(
            spec.input_schema.get("additionalProperties") is False,
            f"{name}: additionalProperties=False"))
    return all(results)


def test_schema_tool_specific_required():
    print("== tool 固有 required 引数の検証 ==")
    reg = ToolRegistry()
    register_noetic_tools(reg, _fake_tools_dict())
    cases = [
        ("output_display", {"content"}),
        ("update_self", {"key", "value"}),
        ("search_memory", {"query"}),
        ("memory_store", {"network", "content"}),
        ("memory_update", {"memory_id"}),
        ("memory_forget", {"memory_id"}),
        ("view_image", {"path"}),
        ("listen_audio", {"path"}),
        ("mic_record", {"duration_sec"}),
        ("secret_read", {"name"}),
        ("secret_write", {"name", "content"}),
    ]
    results = []
    for name, must in cases:
        spec = reg.get(name)
        required = set(spec.input_schema.get("required", []))
        ok = must <= required
        results.append(_assert(ok, f"{name}: {must} required"))
    return all(results)


def test_schema_enum_network():
    print("== memory_store.network が enum 制約を持つ ==")
    reg = ToolRegistry()
    register_noetic_tools(reg, _fake_tools_dict())
    spec = reg.get("memory_store")
    props = spec.input_schema.get("properties", {})
    enum_vals = set(props.get("network", {}).get("enum", []))
    return _assert(
        enum_vals == {"world", "experience", "opinion", "entity"},
        f"network enum 一致: {enum_vals}")


def test_schema_enum_facing():
    print("== camera_stream.facing が enum 制約 (front/back) を持つ ==")
    reg = ToolRegistry()
    register_noetic_tools(reg, _fake_tools_dict())
    spec = reg.get("camera_stream")
    props = spec.input_schema.get("properties", {})
    enum_vals = set(props.get("facing", {}).get("enum", []))
    return _assert(
        enum_vals == {"front", "back"},
        f"facing enum 一致: {enum_vals}")


def test_schema_numeric_constraints():
    print("== 数値引数に minimum / maximum 制約が付く ==")
    reg = ToolRegistry()
    register_noetic_tools(reg, _fake_tools_dict())
    spec = reg.get("mic_record")
    duration = spec.input_schema["properties"]["duration_sec"]
    return all([
        _assert(duration.get("type") == "number", "duration_sec: number"),
        _assert(duration.get("minimum") == 1.0, "minimum=1.0"),
        _assert(duration.get("maximum") == 30.0, "maximum=30.0"),
    ])


# ============================================================
# permission 検証 (合意済割当)
# ============================================================

def test_permissions_per_family():
    print("== 各 tool の required_permission が合意通り ==")
    reg = ToolRegistry()
    register_noetic_tools(reg, _fake_tools_dict())
    expected = {
        # cognition
        "reflect":        PermissionMode.WORKSPACE_WRITE,
        "update_self":    PermissionMode.WORKSPACE_WRITE,
        "output_display": PermissionMode.WORKSPACE_WRITE,
        "wait":           PermissionMode.READ_ONLY,
        # memory
        "search_memory":  PermissionMode.READ_ONLY,
        "memory_store":   PermissionMode.WORKSPACE_WRITE,
        "memory_update":  PermissionMode.WORKSPACE_WRITE,
        "memory_forget":  PermissionMode.WORKSPACE_WRITE,
        # sense
        "view_image":         PermissionMode.READ_ONLY,
        "listen_audio":       PermissionMode.READ_ONLY,
        "mic_record":         PermissionMode.DANGER_FULL_ACCESS,
        "camera_stream":      PermissionMode.DANGER_FULL_ACCESS,
        "camera_stream_stop": PermissionMode.READ_ONLY,
        "screen_peek":        PermissionMode.DANGER_FULL_ACCESS,
        # auth
        "auth_profile_info":  PermissionMode.READ_ONLY,
        "secret_read":        PermissionMode.READ_ONLY,
        "secret_write":       PermissionMode.DANGER_FULL_ACCESS,
    }
    results = []
    for name, perm in expected.items():
        spec = reg.get(name)
        results.append(_assert(
            spec.required_permission == perm,
            f"{name}: {perm.name}"))
    return all(results)


# ============================================================
# handler 連動
# ============================================================

def test_handler_delegates_to_tools_dict():
    print("== handler が tools_dict[name]['func'] を呼ぶ ==")
    reg = ToolRegistry()
    register_noetic_tools(reg, _fake_tools_dict())
    out = reg.execute("output_display", {
        "content": "hello",
        "tool_intent": "t", "tool_expected_outcome": "t", "message": "t",
    })
    return _assert("output_display" in out, f"fake func 経由: {out[:60]}")


def test_handler_view_image():
    print("== view_image handler が legacy func を呼ぶ ==")
    reg = ToolRegistry()
    register_noetic_tools(reg, _fake_tools_dict())
    out = reg.execute("view_image", {
        "path": "x.jpg",
        "tool_intent": "t", "tool_expected_outcome": "t", "message": "t",
    })
    return _assert("view_image" in out, f"legacy 呼出: {out[:60]}")


# ============================================================
# 異常系
# ============================================================

def test_missing_tool_raises():
    print("== tools_dict に必要 tool 欠落 → KeyError ==")
    reg = ToolRegistry()
    incomplete = _fake_tools_dict()
    del incomplete["reflect"]
    try:
        register_noetic_tools(reg, incomplete)
        return _assert(False, "KeyError 期待")
    except KeyError as e:
        return _assert("reflect" in str(e), f"欠落 tool が msg に含む: {e}")


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("17 個登録", test_register_all_17),
        ("NOETIC_TOOL_NAMES export", test_tool_names_exposed),
        ("tool 名一致", test_registered_tool_names_match),
        ("対象外 tool 無視", test_extra_tools_ignored),
        ("schema: 3 層 required", test_schema_has_approval_3_layer),
        ("schema: type=object", test_schema_type_object_everywhere),
        ("schema: additionalProperties=False", test_schema_additional_properties_false),
        ("schema: tool 固有 required", test_schema_tool_specific_required),
        ("schema: network enum", test_schema_enum_network),
        ("schema: facing enum", test_schema_enum_facing),
        ("schema: 数値 min/max", test_schema_numeric_constraints),
        ("permission 割当", test_permissions_per_family),
        ("handler delegate: output_display", test_handler_delegates_to_tools_dict),
        ("handler delegate: view_image", test_handler_view_image),
        ("tool 欠落 → KeyError", test_missing_tool_raises),
    ]
    results = []
    for _label, fn in groups:
        print()
        ok = fn()
        results.append((_label, ok))
    print()
    print("=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for _label, ok in results:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {_label}")
    print(f"\n  {passed}/{total} groups passed")
    sys.exit(0 if passed == total else 1)
