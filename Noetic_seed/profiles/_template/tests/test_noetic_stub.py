"""noetic_stub_tools.register_noetic_stubs テスト。

Phase 4 Step E-2b: 5 個の Noetic 固有 tool stub を ToolRegistry に登録する
動作を検証。handler は tools_dict 経由で既存 Noetic 実装を呼ぶ想定。

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_noetic_stub.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.noetic_stub_tools import (
    STUB_TOOL_NAMES,
    register_noetic_stubs,
)
from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _fake_tools_dict():
    """テスト用: stub が参照する既存 Noetic TOOLS dict を模倣。"""
    return {
        "output_display": {"desc": "発話", "func": lambda inp: f"[spoke] {inp.get('content','')}"},
        "wait":           {"desc": "待機", "func": lambda inp: "[waited]"},
        "reflect":        {"desc": "内省", "func": lambda inp: "[reflected]"},
        "update_self":    {"desc": "self 更新", "func": lambda inp: f"[self[{inp.get('key')}]={inp.get('value')}]"},
        "search_memory":  {"desc": "記憶検索", "func": lambda inp: f"[mem q={inp.get('query')}]"},
        # 余分な tool (stub 対象外、無視されるべき)
        "elyth_post":     {"desc": "Elyth", "func": lambda inp: "[elyth]"},
    }


# ============================================================
# 登録の基本動作
# ============================================================

def test_register_all_five():
    print("== 5 個全部登録される ==")
    reg = ToolRegistry()
    n = register_noetic_stubs(reg, _fake_tools_dict())
    return all([
        _assert(n == 5, f"登録数=5 (実={n})"),
        _assert(len(reg.list()) == 5, "registry にも 5 個"),
    ])


def test_stub_names_exposed():
    print("== STUB_TOOL_NAMES が 5 個の名前を export ==")
    return all([
        _assert(len(STUB_TOOL_NAMES) == 5, "5 個"),
        _assert("output_display" in STUB_TOOL_NAMES, "output_display"),
        _assert("wait" in STUB_TOOL_NAMES, "wait"),
        _assert("reflect" in STUB_TOOL_NAMES, "reflect"),
        _assert("update_self" in STUB_TOOL_NAMES, "update_self"),
        _assert("search_memory" in STUB_TOOL_NAMES, "search_memory"),
    ])


def test_registered_tools_have_correct_names():
    print("== 登録された tool 名が期待通り ==")
    reg = ToolRegistry()
    register_noetic_stubs(reg, _fake_tools_dict())
    names = {spec.name for spec in reg.list()}
    return _assert(names == set(STUB_TOOL_NAMES),
                   f"名前一致: {sorted(names)}")


def test_extra_tools_ignored():
    print("== 対象外 tool (elyth_post 等) は登録されない ==")
    reg = ToolRegistry()
    register_noetic_stubs(reg, _fake_tools_dict())
    return _assert(not reg.has("elyth_post"),
                   "elyth_post は登録されない")


# ============================================================
# input_schema 検証
# ============================================================

def test_schema_has_approval_fields():
    print("== 全 tool の input_schema に 3 層必須 ==")
    reg = ToolRegistry()
    register_noetic_stubs(reg, _fake_tools_dict())
    results = []
    for name in STUB_TOOL_NAMES:
        spec = reg.get(name)
        schema = spec.input_schema
        required = set(schema.get("required", []))
        props = schema.get("properties", {})
        results.append(_assert(
            {"tool_intent", "tool_expected_outcome", "message"} <= required,
            f"{name}: 3 層 required"))
        results.append(_assert(
            "tool_intent" in props and "message" in props,
            f"{name}: 3 層 properties"))
    return all(results)


def test_schema_output_display_content_required():
    print("== output_display: content も required ==")
    reg = ToolRegistry()
    register_noetic_stubs(reg, _fake_tools_dict())
    spec = reg.get("output_display")
    required = set(spec.input_schema.get("required", []))
    return _assert("content" in required, "content required")


def test_schema_update_self_key_value_required():
    print("== update_self: key / value も required ==")
    reg = ToolRegistry()
    register_noetic_stubs(reg, _fake_tools_dict())
    spec = reg.get("update_self")
    required = set(spec.input_schema.get("required", []))
    return all([
        _assert("key" in required, "key required"),
        _assert("value" in required, "value required"),
    ])


def test_schema_search_memory_query_required():
    print("== search_memory: query required ==")
    reg = ToolRegistry()
    register_noetic_stubs(reg, _fake_tools_dict())
    spec = reg.get("search_memory")
    required = set(spec.input_schema.get("required", []))
    return _assert("query" in required, "query required")


def test_schema_allows_additional_properties():
    print("== schema は additionalProperties=True (LLM の自由 args を許容) ==")
    reg = ToolRegistry()
    register_noetic_stubs(reg, _fake_tools_dict())
    results = []
    for name in STUB_TOOL_NAMES:
        spec = reg.get(name)
        results.append(_assert(
            spec.input_schema.get("additionalProperties") is True,
            f"{name}: additionalProperties=True"))
    return all(results)


# ============================================================
# permission 検証
# ============================================================

def test_permissions_assigned():
    print("== 各 tool の required_permission が期待通り ==")
    reg = ToolRegistry()
    register_noetic_stubs(reg, _fake_tools_dict())
    expected = {
        "output_display": PermissionMode.WORKSPACE_WRITE,
        "wait":           PermissionMode.READ_ONLY,
        "reflect":        PermissionMode.WORKSPACE_WRITE,
        "update_self":    PermissionMode.WORKSPACE_WRITE,
        "search_memory":  PermissionMode.READ_ONLY,
    }
    results = []
    for name, expected_perm in expected.items():
        spec = reg.get(name)
        results.append(_assert(
            spec.required_permission == expected_perm,
            f"{name}: {expected_perm.name}"))
    return all(results)


# ============================================================
# handler 連動
# ============================================================

def test_handler_delegates_to_tools_dict():
    print("== handler が tools_dict[name]['func'] を呼び出す ==")
    reg = ToolRegistry()
    register_noetic_stubs(reg, _fake_tools_dict())
    out = reg.execute("output_display", {
        "content": "hello",
        "tool_intent": "test",
        "tool_expected_outcome": "ok",
        "message": "test",
    })
    return _assert("hello" in out, f"fake func 経由: {out}")


def test_handler_update_self_args():
    print("== update_self handler が key/value を受け取る ==")
    reg = ToolRegistry()
    register_noetic_stubs(reg, _fake_tools_dict())
    out = reg.execute("update_self", {
        "key": "mood", "value": "curious",
        "tool_intent": "x", "tool_expected_outcome": "y", "message": "z",
    })
    return all([
        _assert("mood" in out, "key 反映"),
        _assert("curious" in out, "value 反映"),
    ])


# ============================================================
# 異常系
# ============================================================

def test_missing_tool_raises():
    print("== tools_dict に必要 tool が無い → KeyError ==")
    reg = ToolRegistry()
    incomplete = _fake_tools_dict()
    del incomplete["reflect"]
    try:
        register_noetic_stubs(reg, incomplete)
        return _assert(False, "KeyError 期待")
    except KeyError as e:
        return all([
            _assert(True, "KeyError 発生"),
            _assert("reflect" in str(e), "欠落 tool 名が msg 含む"),
        ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("5 個登録", test_register_all_five),
        ("STUB_TOOL_NAMES export", test_stub_names_exposed),
        ("tool 名一致", test_registered_tools_have_correct_names),
        ("対象外 tool 無視", test_extra_tools_ignored),
        ("schema: 3 層", test_schema_has_approval_fields),
        ("schema: output_display content", test_schema_output_display_content_required),
        ("schema: update_self key/value", test_schema_update_self_key_value_required),
        ("schema: search_memory query", test_schema_search_memory_query_required),
        ("schema: additionalProperties", test_schema_allows_additional_properties),
        ("permission 割当", test_permissions_assigned),
        ("handler delegate", test_handler_delegates_to_tools_dict),
        ("update_self args", test_handler_update_self_args),
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
