"""Approval 3 層チェッカー (PreToolUse hook) テスト。

APPROVAL_PROMPT_SPEC.md §8.2 のテスト項目を網羅:
  - 3 層全揃い → pass
  - 各フィールド欠損 → deny
  - 空白 / None / キー無し → 欠損扱い
  - policy="warn" / "auto_fill" 切替
  - HookRunner 統合 (updated_input の伝播)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_approval_spec.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.hooks import (
    HookRunner,
    HookRunResult,
    make_pre_tool_use_approval_check,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _full_input(**overrides):
    """3 層揃った tool_input。overrides で一部を差し替える。"""
    base = {
        "tool_intent": "テスト実行",
        "tool_expected_outcome": "何か起こる",
        "message": "テストします",
        "path": "/tmp/foo.py",
    }
    base.update(overrides)
    return base


# ============================================================
# Policy=deny
# ============================================================

def test_deny_all_present():
    print("== deny: 3 層全揃い → allow ==")
    check = make_pre_tool_use_approval_check(missing_field_policy="deny")
    r = check("write_file", _full_input())
    return all([
        _assert(not r.denied, "denied=False"),
        _assert(not r.failed, "failed=False"),
        _assert(r.messages == [], "messages 空"),
        _assert(r.updated_input is None, "updated_input 無し"),
    ])


def test_deny_missing_intent():
    print("== deny: tool_intent 欠損 → deny ==")
    check = make_pre_tool_use_approval_check(missing_field_policy="deny")
    r = check("write_file", _full_input(tool_intent=""))
    return all([
        _assert(r.denied, "denied=True"),
        _assert(any("tool_intent" in m for m in r.messages),
                "msg に intent 含む"),
    ])


def test_deny_missing_expected():
    print("== deny: tool_expected_outcome 欠損 → deny ==")
    check = make_pre_tool_use_approval_check(missing_field_policy="deny")
    r = check("write_file", _full_input(tool_expected_outcome=""))
    return all([
        _assert(r.denied, "denied=True"),
        _assert(any("tool_expected_outcome" in m for m in r.messages),
                "msg に expected 含む"),
    ])


def test_deny_missing_message():
    print("== deny: message 欠損 → deny ==")
    check = make_pre_tool_use_approval_check(missing_field_policy="deny")
    r = check("write_file", _full_input(message=""))
    return all([
        _assert(r.denied, "denied=True"),
        _assert(any("message" in m for m in r.messages),
                "msg に message 含む"),
    ])


def test_deny_multiple_missing():
    print("== deny: 2 フィールド同時欠損 → deny (全件表示) ==")
    check = make_pre_tool_use_approval_check(missing_field_policy="deny")
    r = check("write_file", _full_input(tool_intent="", message=""))
    return all([
        _assert(r.denied, "denied=True"),
        _assert(any("tool_intent" in m and "message" in m for m in r.messages),
                "両フィールドとも msg に含む"),
    ])


def test_deny_whitespace_only():
    print("== deny: 空白のみ ('   ') → 欠損扱い ==")
    check = make_pre_tool_use_approval_check(missing_field_policy="deny")
    r = check("write_file", _full_input(tool_intent="   "))
    return _assert(r.denied, "空白のみは欠損")


def test_deny_none_value():
    print("== deny: None 値 → 欠損扱い ==")
    check = make_pre_tool_use_approval_check(missing_field_policy="deny")
    r = check("write_file", _full_input(message=None))
    return _assert(r.denied, "None は欠損")


def test_deny_missing_key():
    print("== deny: キー自体が無い → 欠損扱い ==")
    check = make_pre_tool_use_approval_check(missing_field_policy="deny")
    inp = _full_input()
    del inp["tool_intent"]
    r = check("write_file", inp)
    return _assert(r.denied, "キー無しは欠損")


# ============================================================
# Policy=warn
# ============================================================

def test_warn_missing_pass():
    print("== warn: 欠損でも pass、warning メッセージ付き ==")
    check = make_pre_tool_use_approval_check(missing_field_policy="warn")
    r = check("write_file", _full_input(tool_intent=""))
    return all([
        _assert(not r.denied, "denied=False"),
        _assert(any("warning" in m for m in r.messages),
                "warning メッセージ含む"),
        _assert(any("tool_intent" in m for m in r.messages),
                "欠損フィールド名含む"),
    ])


def test_warn_all_present():
    print("== warn: 全揃いは無警告で allow ==")
    check = make_pre_tool_use_approval_check(missing_field_policy="warn")
    r = check("write_file", _full_input())
    return all([
        _assert(not r.denied, "denied=False"),
        _assert(r.messages == [], "messages 空"),
    ])


# ============================================================
# Policy=auto_fill
# ============================================================

def test_autofill_missing():
    print("== auto_fill: 欠損は補完して allow ==")
    check = make_pre_tool_use_approval_check(missing_field_policy="auto_fill")
    r = check("write_file", _full_input(tool_intent="", message=""))
    return all([
        _assert(not r.denied, "denied=False"),
        _assert(r.updated_input is not None, "updated_input あり"),
        _assert(r.updated_input["tool_intent"].startswith("[auto_fill]"),
                "intent 補完"),
        _assert(r.updated_input["message"].startswith("[auto_fill]"),
                "message 補完"),
        _assert(r.updated_input["tool_expected_outcome"] == "何か起こる",
                "既存値は保持"),
        _assert(r.updated_input["path"] == "/tmp/foo.py",
                "tool 固有 args も保持"),
    ])


def test_autofill_all_present():
    print("== auto_fill: 全揃いは触らない ==")
    check = make_pre_tool_use_approval_check(missing_field_policy="auto_fill")
    r = check("write_file", _full_input())
    return all([
        _assert(not r.denied, "denied=False"),
        _assert(r.updated_input is None, "updated_input 無し (補完不要)"),
        _assert(r.messages == [], "messages 空"),
    ])


def test_autofill_preserves_original():
    print("== auto_fill: 元 dict を破壊しない ==")
    check = make_pre_tool_use_approval_check(missing_field_policy="auto_fill")
    original = _full_input(tool_intent="")
    r = check("write_file", original)
    return all([
        _assert(r.updated_input is not original,
                "新 dict を返す (同一オブジェクト不可)"),
        _assert(original["tool_intent"] == "",
                "元 dict の tool_intent は空のまま"),
    ])


# ============================================================
# Invalid policy
# ============================================================

def test_invalid_policy():
    print("== 未知の policy → ValueError (factory 時点で検出) ==")
    try:
        make_pre_tool_use_approval_check(missing_field_policy="xxx")
        return _assert(False, "ValueError 期待")
    except ValueError as e:
        return all([
            _assert(True, "ValueError 発生"),
            _assert("xxx" in str(e), "エラーメッセージに policy 名含む"),
        ])


# ============================================================
# HookRunner 統合
# ============================================================

def test_hook_runner_integration():
    print("== HookRunner register → run_pre_tool_use 経由で動作 ==")
    runner = HookRunner()
    runner.register_pre(
        make_pre_tool_use_approval_check(missing_field_policy="deny")
    )
    r1 = runner.run_pre_tool_use("write_file", _full_input())
    r2 = runner.run_pre_tool_use("write_file", _full_input(message=""))
    return all([
        _assert(not r1.denied, "全揃い: denied=False"),
        _assert(r2.denied, "欠損: denied=True"),
    ])


def test_hook_runner_autofill_propagates():
    print("== HookRunner + auto_fill: updated_input が次 handler に伝播 ==")
    runner = HookRunner()
    runner.register_pre(
        make_pre_tool_use_approval_check(missing_field_policy="auto_fill")
    )
    captured = {}

    def _second(tool_name, tool_input):
        captured["input"] = dict(tool_input)
        return HookRunResult.allow()

    runner.register_pre(_second)

    r = runner.run_pre_tool_use("write_file", _full_input(tool_intent=""))
    return all([
        _assert(not r.denied, "denied=False"),
        _assert("input" in captured, "2 段目 handler が呼ばれた"),
        _assert(captured["input"]["tool_intent"].startswith("[auto_fill]"),
                "2 段目に補完後 input が渡る"),
        _assert(r.updated_input is not None,
                "最終結果にも updated_input 残る"),
    ])


def test_hook_runner_deny_stops_chain():
    print("== HookRunner + deny: 後続 handler は呼ばれない ==")
    runner = HookRunner()
    runner.register_pre(
        make_pre_tool_use_approval_check(missing_field_policy="deny")
    )
    called = {"n": 0}

    def _second(tool_name, tool_input):
        called["n"] += 1
        return HookRunResult.allow()

    runner.register_pre(_second)

    r = runner.run_pre_tool_use("write_file", _full_input(message=""))
    return all([
        _assert(r.denied, "denied=True"),
        _assert(called["n"] == 0, "後続 handler 未呼出"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("deny: 全揃い", test_deny_all_present),
        ("deny: intent 欠損", test_deny_missing_intent),
        ("deny: expected 欠損", test_deny_missing_expected),
        ("deny: message 欠損", test_deny_missing_message),
        ("deny: 複数欠損", test_deny_multiple_missing),
        ("deny: 空白のみ", test_deny_whitespace_only),
        ("deny: None", test_deny_none_value),
        ("deny: キー無し", test_deny_missing_key),
        ("warn: 欠損 pass", test_warn_missing_pass),
        ("warn: 全揃い", test_warn_all_present),
        ("auto_fill: 欠損補完", test_autofill_missing),
        ("auto_fill: 全揃い", test_autofill_all_present),
        ("auto_fill: 元 dict 非破壊", test_autofill_preserves_original),
        ("invalid policy", test_invalid_policy),
        ("HookRunner 統合", test_hook_runner_integration),
        ("HookRunner auto_fill 伝播", test_hook_runner_autofill_propagates),
        ("HookRunner deny で chain 停止", test_hook_runner_deny_stops_chain),
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
