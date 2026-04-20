"""PermissionEnforcer 動作確認テスト (claw-code 準拠版)。

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_permissions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.permissions import (
    PermissionEnforcer,
    PermissionMode,
    PermissionDecision,
    PermissionRules,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def test_read_only_mode():
    print("== mode=READ_ONLY ==")
    enf = PermissionEnforcer(mode=PermissionMode.READ_ONLY)
    r = [
        _assert(enf.check("read_file") == PermissionDecision.ALLOW,
                "read_file ALLOW"),
        _assert(enf.check("write_file") == PermissionDecision.ASK,
                "write_file ASK"),
        _assert(enf.check("bash") == PermissionDecision.ASK, "bash ASK"),
        _assert(enf.check("WebFetch") == PermissionDecision.ALLOW,
                "WebFetch ALLOW"),
    ]
    return all(r)


def test_workspace_write_mode():
    print("== mode=WORKSPACE_WRITE ==")
    enf = PermissionEnforcer(mode=PermissionMode.WORKSPACE_WRITE)
    r = [
        _assert(enf.check("read_file") == PermissionDecision.ALLOW,
                "read_file ALLOW"),
        _assert(enf.check("write_file") == PermissionDecision.ALLOW,
                "write_file ALLOW"),
        _assert(enf.check("edit_file") == PermissionDecision.ALLOW,
                "edit_file ALLOW"),
        _assert(enf.check("bash") == PermissionDecision.ASK, "bash ASK"),
        _assert(enf.check("MCP") == PermissionDecision.ASK, "MCP ASK"),
    ]
    return all(r)


def test_danger_mode():
    print("== mode=DANGER_FULL_ACCESS ==")
    enf = PermissionEnforcer(mode=PermissionMode.DANGER_FULL_ACCESS)
    r = [
        _assert(enf.check("read_file") == PermissionDecision.ALLOW,
                "read_file ALLOW"),
        _assert(enf.check("bash") == PermissionDecision.ALLOW, "bash ALLOW"),
        _assert(enf.check("MCP") == PermissionDecision.ALLOW, "MCP ALLOW"),
        _assert(enf.check("UnknownTool") == PermissionDecision.ALLOW,
                "unknown ALLOW"),
    ]
    return all(r)


def test_prompt_mode():
    print("== mode=PROMPT ==")
    enf = PermissionEnforcer(mode=PermissionMode.PROMPT)
    r = [
        _assert(enf.check("read_file") == PermissionDecision.ASK,
                "read_file ASK"),
        _assert(enf.check("bash") == PermissionDecision.ASK, "bash ASK"),
    ]
    return all(r)


def test_allow_mode():
    print("== mode=ALLOW ==")
    enf = PermissionEnforcer(mode=PermissionMode.ALLOW)
    r = [
        _assert(enf.check("bash") == PermissionDecision.ALLOW, "bash ALLOW"),
    ]
    return all(r)


def test_rules_priority():
    print("== rules > mode 優先 ==")
    rules = PermissionRules(deny=["bash"], allow=["read_file"],
                            ask=["write_file"])
    enf = PermissionEnforcer(mode=PermissionMode.DANGER_FULL_ACCESS,
                             rules=rules)
    r = [
        _assert(enf.check("bash") == PermissionDecision.DENY,
                "bash rules.deny で DENY"),
        _assert(enf.check("read_file") == PermissionDecision.ALLOW,
                "read_file rules.allow で ALLOW"),
        _assert(enf.check("write_file") == PermissionDecision.ASK,
                "write_file rules.ask で ASK"),
        _assert(enf.check("edit_file") == PermissionDecision.ALLOW,
                "未指定 edit_file は mode=DANGER で ALLOW"),
    ]
    return all(r)


def test_prefix_pattern():
    print("== prefix pattern Worker* ==")
    rules = PermissionRules(deny=["Worker*"])
    enf = PermissionEnforcer(mode=PermissionMode.DANGER_FULL_ACCESS,
                             rules=rules)
    r = [
        _assert(enf.check("WorkerCreate") == PermissionDecision.DENY,
                "WorkerCreate DENY"),
        _assert(enf.check("WorkerGet") == PermissionDecision.DENY,
                "WorkerGet DENY"),
        _assert(enf.check("TaskCreate") == PermissionDecision.ALLOW,
                "TaskCreate ALLOW"),
    ]
    return all(r)


def test_required_mode_lookup():
    print("== required_mode_for ==")
    enf = PermissionEnforcer(mode=PermissionMode.READ_ONLY)
    r = [
        _assert(enf.required_mode_for("bash")
                == PermissionMode.DANGER_FULL_ACCESS, "bash DANGER"),
        _assert(enf.required_mode_for("read_file")
                == PermissionMode.READ_ONLY, "read_file RO"),
        _assert(enf.required_mode_for("NonExistent") is None, "unknown None"),
    ]
    return all(r)


def test_register_tool_permission():
    print("== register_tool_permission (MCP 動的追加想定) ==")
    enf = PermissionEnforcer(mode=PermissionMode.DANGER_FULL_ACCESS)
    before = enf.check("mcp__slack__post")
    enf.register_tool_permission("mcp__slack__post",
                                 PermissionMode.DANGER_FULL_ACCESS)
    after = enf.check("mcp__slack__post")
    r = [
        _assert(before == PermissionDecision.ALLOW,
                "登録前 mode=DANGER なら unknown でも ALLOW"),
        _assert(after == PermissionDecision.ALLOW, "登録後も ALLOW"),
    ]
    return all(r)


def main():
    tests = [
        test_read_only_mode, test_workspace_write_mode, test_danger_mode,
        test_prompt_mode, test_allow_mode, test_rules_priority,
        test_prefix_pattern, test_required_mode_lookup,
        test_register_tool_permission,
    ]
    print(f"Running {len(tests)} test groups...\n")
    passed = 0
    for t in tests:
        if t():
            passed += 1
        print()
    print(f"=== Result: {passed}/{len(tests)} test groups passed ===")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
