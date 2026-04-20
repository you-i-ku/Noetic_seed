"""legacy_bridge.register_legacy_bridge テスト。

Phase 4 Step H-2 A: legacy TOOLS dict を ToolSpec で passthrough 登録する。
既に claw / stub で登録済の name は skip される。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.legacy_bridge import (
    _READ_ONLY_LEGACY_TOOLS,
    _make_passthrough_schema,
    register_legacy_bridge,
)
from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _fake_tools_dict():
    return {
        "output_display": {"desc": "端末発話", "func": lambda a: "said"},
        "wait": {"desc": "待機", "func": lambda a: "waited"},
        "reflect": {"desc": "内省", "func": lambda a: "reflected"},
        "update_self": {"desc": "自己更新", "func": lambda a: "updated"},
        "search_memory": {"desc": "記憶検索", "func": lambda a: "found"},
        "elyth_post": {"desc": "Elyth 投稿", "func": lambda a: "posted"},
        "view_image": {"desc": "画像認識", "func": lambda a: "saw"},
        "self_modify": {"desc": "自己改変", "func": lambda a: "modified"},
        "read_file": {"desc": "ファイル読取", "func": lambda a: "read"},
    }


def test_bridge_registers_all_when_empty():
    print("== 空 registry に全 tool が登録される ==")
    reg = ToolRegistry()
    n = register_legacy_bridge(reg, _fake_tools_dict())
    return all([
        _assert(n == 9, f"登録数=9 (実={n})"),
        _assert(reg.has("output_display"), "output_display 登録"),
        _assert(reg.has("elyth_post"), "elyth_post 登録"),
        _assert(reg.has("self_modify"), "self_modify 登録"),
    ])


def test_bridge_overwrites_existing_names():
    print("== 既登録 name は上書きされる (legacy guard 優先) ==")
    reg = ToolRegistry()
    # claw 側が read_file を先に登録してると仮定
    reg.register(ToolSpec(
        name="read_file", description="claw read_file",
        input_schema={"type": "object", "properties": {}},
        required_permission=PermissionMode.READ_ONLY,
        handler=lambda a: "claw_read",
    ))
    n = register_legacy_bridge(reg, _fake_tools_dict())
    return all([
        _assert(n == 9, f"登録数=9 (overwrite、全件登録) (実={n})"),
        _assert(reg.get("read_file").handler({}) == "read",
                "legacy handler で上書き (secrets guard 保護)"),
    ])


def test_bridge_skip_names_param():
    print("== skip_names パラメータで追加除外できる ==")
    reg = ToolRegistry()
    skip = frozenset({"elyth_post", "self_modify"})
    n = register_legacy_bridge(reg, _fake_tools_dict(), skip_names=skip)
    return all([
        _assert(n == 7, f"登録数=7 (2 skip 後) (実={n})"),
        _assert(not reg.has("elyth_post"), "elyth_post 未登録"),
        _assert(not reg.has("self_modify"), "self_modify 未登録"),
        _assert(reg.has("wait"), "wait は登録される"),
    ])


def test_bridge_permissions():
    print("== READ_ONLY 判定 ==")
    reg = ToolRegistry()
    register_legacy_bridge(reg, _fake_tools_dict())
    return all([
        _assert(reg.get("wait").required_permission == PermissionMode.READ_ONLY,
                "wait = READ_ONLY"),
        _assert(reg.get("search_memory").required_permission == PermissionMode.READ_ONLY,
                "search_memory = READ_ONLY"),
        _assert(reg.get("view_image").required_permission == PermissionMode.READ_ONLY,
                "view_image = READ_ONLY"),
        _assert(reg.get("output_display").required_permission == PermissionMode.WORKSPACE_WRITE,
                "output_display = WORKSPACE_WRITE"),
        _assert(reg.get("self_modify").required_permission == PermissionMode.WORKSPACE_WRITE,
                "self_modify = WORKSPACE_WRITE"),
        _assert(reg.get("elyth_post").required_permission == PermissionMode.WORKSPACE_WRITE,
                "elyth_post = WORKSPACE_WRITE"),
    ])


def test_bridge_handler_passthrough():
    print("== handler が legacy func を passthrough で呼ぶ ==")
    reg = ToolRegistry()
    register_legacy_bridge(reg, _fake_tools_dict())
    return all([
        _assert(reg.get("output_display").handler({}) == "said",
                "output_display handler = legacy func"),
        _assert(reg.get("elyth_post").handler({}) == "posted",
                "elyth_post handler = legacy func"),
        _assert(reg.execute("wait", {}) == "waited",
                "execute() 経由でも legacy func 呼出"),
    ])


def test_bridge_schema_has_approval_layer():
    print("== schema に承認 3 層が含まれる ==")
    schema = _make_passthrough_schema()
    props = schema.get("properties", {})
    return all([
        _assert("tool_intent" in props, "tool_intent フィールド"),
        _assert("tool_expected_outcome" in props, "tool_expected_outcome フィールド"),
        _assert("message" in props, "message フィールド"),
        _assert("tool_intent" in schema.get("required", []),
                "tool_intent は required"),
        _assert(schema.get("additionalProperties") is True,
                "additionalProperties=True (free-form args 許容)"),
    ])


def test_readonly_set_reasonable():
    print("== _READ_ONLY_LEGACY_TOOLS に危険 tool が含まれないこと ==")
    dangerous = {"self_modify", "exec_code", "create_tool",
                 "elyth_post", "x_post", "output_display",
                 "mic_record", "camera_stream", "screen_peek"}
    overlap = dangerous & _READ_ONLY_LEGACY_TOOLS
    return _assert(not overlap, f"危険 tool なし (実={overlap})")


if __name__ == "__main__":
    groups = [
        ("bridge: 空 registry 登録", test_bridge_registers_all_when_empty),
        ("bridge: 既登録 overwrite", test_bridge_overwrites_existing_names),
        ("bridge: skip_names パラメータ", test_bridge_skip_names_param),
        ("bridge: permission 判定", test_bridge_permissions),
        ("bridge: handler passthrough", test_bridge_handler_passthrough),
        ("bridge: schema 構造", test_bridge_schema_has_approval_layer),
        ("bridge: READ_ONLY リスト健全性", test_readonly_set_reasonable),
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
