"""PostToolUse hook (E 値評価 / Action Ledger / unresolved) テスト。

PHASE4_TASKS.md Step B の成功条件を網羅:
  - tool 実行 → hook 発火 → state に E 値/ledger/pending 更新
  - 既存 E2 cap (0.3 + eff*0.7) が効く
  - LLM 評価失敗時のフォールバック
  - tool 失敗時は post_tool_use_failure で tool_errors に記録

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_post_tool_hooks.py
"""
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.hooks import (
    HookRunner,
    make_post_tool_use_evaluation,
    make_post_tool_use_failure_logger,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _fresh_state():
    return {
        "cycle_id": 10,
        "log": [],
        "pending": [],
        "action_ledger": [],
        "self": {},
        "files_read": [],
        "files_written": [],
        "energy": 50,
        "e_values": {},
    }


def _full_input(**overrides):
    base = {
        "tool_intent": "設定を書き込む",
        "tool_expected_outcome": "file が作成される",
        "message": "設定を書き込みます",
        "path": "/tmp/test.py",
    }
    base.update(overrides)
    return base


def _mock_llm_ok(prompt, max_tokens=None, temperature=None):
    """E1=70, E2=60, E3=80, E4=90 を返す mock。"""
    return "E1=70\nE2=60\nE3=80\nE4=90"


def _mock_llm_low_e2(prompt, max_tokens=None, temperature=None):
    """E2 を高めに返して E2 cap の効きを確認するための mock。"""
    return "E1=90\nE2=100\nE3=80\nE4=90"


def _mock_llm_fail(prompt, max_tokens=None, temperature=None):
    """eval_with_llm 内で例外を吐く mock。フォールバック動作確認用。"""
    raise RuntimeError("LLM 接続失敗")


def _make_hook(state, state_before,
               call_llm=_mock_llm_ok, cycle_id=10, recent_intents=None):
    """factory 呼出のボイラプレート集約。"""
    return make_post_tool_use_evaluation(
        state=state,
        get_state_before=lambda: state_before,
        call_llm_fn=call_llm,
        get_cycle_id=lambda: cycle_id,
        get_recent_intents=lambda: list(recent_intents or []),
    )


# ============================================================
# E 値評価 hook
# ============================================================

def test_eval_basic():
    print("== eval hook: state[e_values] に E1-E4 と eff 格納 ==")
    state = _fresh_state()
    before = deepcopy(state)
    hook = _make_hook(state, before)
    r = hook("write_file", _full_input(), "ファイル書き込み完了")
    e_values = state.get("e_values", {})
    return all([
        _assert(not r.denied, "denied=False"),
        _assert("e1" in e_values, "e1 保存"),
        _assert("e2" in e_values, "e2 保存"),
        _assert("e3" in e_values, "e3 保存"),
        _assert("e4" in e_values, "e4 保存"),
        _assert("eff" in e_values, "eff 保存"),
        _assert(e_values["e1"].endswith("%"), "% 表記"),
        _assert(isinstance(e_values["eff"], float), "eff は float"),
    ])


def test_eval_updates_ledger():
    print("== eval hook: action_ledger に追記 ==")
    state = _fresh_state()
    before = deepcopy(state)
    hook = _make_hook(state, before)
    hook("write_file", _full_input(), "完了")
    ledger = state.get("action_ledger", [])
    entry = ledger[-1] if ledger else {}
    return all([
        _assert(len(ledger) == 1, "1 件追記"),
        _assert(entry.get("tool") == "write_file", "tool 名"),
        _assert(entry.get("action_key", "").startswith("write_file:"),
                "action_key に path ベース"),
        _assert("設定を書き込む" in entry.get("intent", ""), "intent 保存"),
        _assert(entry.get("cycle") == 10, "cycle 保存"),
    ])


def test_eval_updates_unresolved_intent():
    print("== eval hook: UPS v2 pending に追加 (source_action=tool_name) ==")
    state = _fresh_state()
    before = deepcopy(state)
    hook = _make_hook(state, before)
    hook("write_file", _full_input(), "完了")
    # Step C-2 以降: type='pending', semantic_merge=True, source_action=tool_name
    ups_entries = [
        p for p in state.get("pending", [])
        if p.get("type") == "pending"
        and p.get("semantic_merge") is True
    ]
    entry = ups_entries[0] if ups_entries else {}
    # E3=80 → gap = 1 - 0.8 = 0.2
    return all([
        _assert(len(ups_entries) == 1, "UPS v2 pending 1 件追加"),
        _assert(abs(entry.get("gap", 0) - 0.2) < 0.01,
                "gap=0.2 (E3=80 由来)"),
        _assert(entry.get("origin_cycle") == 10, "origin_cycle"),
        _assert(entry.get("source_action") == "write_file",
                "source_action=tool_name"),
        _assert(entry.get("observation_lag_kind") == "cycles",
                "lag_kind=cycles (unresolved 系の default)"),
        _assert(entry.get("expected_channel") == "self",
                "expected_channel=self (内省由来)"),
    ])


def test_eval_e2_cap_zero_eff():
    print("== eval hook: eff=0 (変化なし) → E2 は上限 30% ==")
    state = _fresh_state()
    before = deepcopy(state)  # 完全同一 → state 差分ゼロ
    hook = _make_hook(state, before, call_llm=_mock_llm_low_e2)
    # bash を選んでいる理由: eval.py の calc_effective_change は
    # ACTIONABLE_TOOLS (reflect / output_display / SNS 系) に対して
    # content_novelty 加点 (log 空で 0.7) が入るため、純粋な eff=0 を
    # 作るには ACTIONABLE にも file 操作にも入らない tool を選ぶ必要が
    # ある。bash はそれに該当する (eval.py §218-258 参照)。
    hook("bash", _full_input(), "")
    e2 = state["e_values"]["e2"]
    e2_raw = state["e_values"]["e2_raw"]
    eff = state["e_values"]["eff"]
    # E2_raw = 100%、eff=0 → cap = 0.3 → E2 = 30%
    return all([
        _assert(eff == 0.0, f"eff={eff} = 0.0 (純粋 eff=0)"),
        _assert(e2_raw == "100%", "e2_raw=100% (生スコア)"),
        _assert(e2 == "30%", f"e2={e2} が 30% (cap 下限)"),
    ])


def test_eval_e2_cap_with_eff():
    print("== eval hook: eff 大 → E2 cap 緩む ==")
    state = _fresh_state()
    before = deepcopy(state)
    # tool 実行で self に新 key → effective_change 加点
    state["self"]["new_insight"] = "何かわかった"
    state["files_written"] = ["/tmp/foo.py"]
    hook = _make_hook(state, before, call_llm=_mock_llm_low_e2)
    hook("reflect", _full_input(), "新しい発見があった")
    e2_raw_pct = int(state["e_values"]["e2_raw"].rstrip("%"))
    e2_pct = int(state["e_values"]["e2"].rstrip("%"))
    eff = state["e_values"]["eff"]
    # eff > 0 → cap = 0.3 + eff*0.7 > 0.3
    # だから E2 > E2_raw * 0.3 = 30
    return all([
        _assert(eff > 0, f"eff={eff} > 0 (state 変化あり)"),
        _assert(e2_pct > 30, f"e2={e2_pct}% > 30% (cap 緩み)"),
        _assert(e2_pct <= e2_raw_pct, f"e2={e2_pct} <= raw={e2_raw_pct}"),
    ])


def test_eval_llm_failure_fallback():
    print("== eval hook: LLM 失敗でもクラッシュせず default スコアで進む ==")
    state = _fresh_state()
    before = deepcopy(state)
    hook = _make_hook(state, before, call_llm=_mock_llm_fail)
    r = hook("write_file", _full_input(), "完了")
    e_values = state.get("e_values", {})
    return all([
        _assert(not r.denied, "denied=False"),
        _assert(e_values.get("e1") == "50%", "e1 default=50%"),
        _assert(e_values.get("e3") == "50%", "e3 default=50%"),
        _assert(len(state.get("action_ledger", [])) == 1, "ledger は追記される"),
    ])


def test_eval_empty_intent():
    print("== eval hook: tool_intent 空でも crash しない ==")
    state = _fresh_state()
    before = deepcopy(state)
    hook = _make_hook(state, before)
    r = hook("write_file", _full_input(tool_intent=""), "完了")
    return _assert(not r.denied, "denied=False")


def test_eval_message_format():
    print("== eval hook: 戻り値 messages に要約 ==")
    state = _fresh_state()
    before = deepcopy(state)
    hook = _make_hook(state, before)
    r = hook("write_file", _full_input(), "完了")
    msg = r.messages[0] if r.messages else ""
    return all([
        _assert(len(r.messages) == 1, "messages 1 件"),
        _assert("post_eval" in msg, "prefix 含む"),
        _assert("write_file" in msg, "tool 名含む"),
        _assert("eff=" in msg, "eff 含む"),
    ])


# ============================================================
# Failure hook
# ============================================================

def test_failure_basic():
    print("== failure hook: tool_errors に追記 ==")
    state = _fresh_state()
    hook = make_post_tool_use_failure_logger(
        state=state, get_cycle_id=lambda: 10
    )
    r = hook("bash", _full_input(), "ExitCode=1: permission denied")
    errors = state.get("tool_errors", [])
    entry = errors[-1] if errors else {}
    return all([
        _assert(not r.denied, "denied=False"),
        _assert(len(errors) == 1, "1 件追記"),
        _assert(entry.get("tool") == "bash", "tool 名"),
        _assert("permission" in entry.get("error", ""), "エラー内容保存"),
        _assert(entry.get("cycle") == 10, "cycle 保存"),
        _assert("設定を書き込む" in entry.get("intent", ""),
                "intent も保存 (承認審査の証跡に使える)"),
    ])


def test_failure_size_cap():
    print("== failure hook: max_entries 超過で古い順に捨てる ==")
    state = _fresh_state()
    hook = make_post_tool_use_failure_logger(
        state=state, get_cycle_id=lambda: 0, max_entries=3,
    )
    for i in range(5):
        hook("bash", _full_input(tool_intent=f"call {i}"), f"err {i}")
    errors = state.get("tool_errors", [])
    return all([
        _assert(len(errors) == 3, f"3 件に truncate (実={len(errors)})"),
        _assert("err 2" in errors[0]["error"], "古い方は残る最古 err 2"),
        _assert("err 4" in errors[-1]["error"], "最新 err 4"),
    ])


def test_failure_error_truncation():
    print("== failure hook: 長大エラー文字列は 500 字で切り詰め ==")
    state = _fresh_state()
    hook = make_post_tool_use_failure_logger(
        state=state, get_cycle_id=lambda: 0
    )
    long_err = "x" * 1000
    hook("bash", _full_input(), long_err)
    entry = state["tool_errors"][-1]
    return _assert(len(entry["error"]) == 500, f"500 字 (実={len(entry['error'])})")


# ============================================================
# HookRunner 統合
# ============================================================

def test_hook_runner_integration():
    print("== HookRunner に register → run_post_tool_use 経由で動作 ==")
    state = _fresh_state()
    before = deepcopy(state)
    runner = HookRunner()
    runner.register_post(_make_hook(state, before))
    runner.register_failure(make_post_tool_use_failure_logger(
        state=state, get_cycle_id=lambda: 10,
    ))
    # 成功パス
    r_ok = runner.run_post_tool_use("write_file", _full_input(), "完了")
    # 失敗パス
    r_fail = runner.run_post_tool_use_failure(
        "bash", _full_input(), "error: xxx"
    )
    return all([
        _assert(not r_ok.denied, "post success: denied=False"),
        _assert(len(state["action_ledger"]) == 1, "ledger 追記"),
        _assert(not r_fail.denied, "post fail: denied=False"),
        _assert(len(state["tool_errors"]) == 1, "errors 追記"),
    ])


def test_hook_runner_chain_success_and_fail():
    print("== HookRunner + 複数 post handler: 両方呼ばれる ==")
    state = _fresh_state()
    before = deepcopy(state)
    runner = HookRunner()
    runner.register_post(_make_hook(state, before))

    called = {"second": 0}

    def _secondary(tool_name, tool_input, output):
        called["second"] += 1
        from core.runtime.hooks import HookRunResult
        return HookRunResult.allow(messages=["[secondary] tagged"])

    runner.register_post(_secondary)

    r = runner.run_post_tool_use("write_file", _full_input(), "完了")
    return all([
        _assert(called["second"] == 1, "secondary も呼ばれた"),
        _assert(len(state["action_ledger"]) == 1, "primary が state 更新"),
        _assert(any("secondary" in m for m in r.messages),
                "secondary の messages も merge"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("eval: 基本動作 (E1-E4 保存)", test_eval_basic),
        ("eval: ledger 追記", test_eval_updates_ledger),
        ("eval: unresolved_intent 追加", test_eval_updates_unresolved_intent),
        ("eval: E2 cap (eff=0 → 30%)", test_eval_e2_cap_zero_eff),
        ("eval: E2 cap (eff>0 → 緩む)", test_eval_e2_cap_with_eff),
        ("eval: LLM 失敗 fallback", test_eval_llm_failure_fallback),
        ("eval: intent 空でも動作", test_eval_empty_intent),
        ("eval: 戻り値 messages", test_eval_message_format),
        ("failure: 基本", test_failure_basic),
        ("failure: size cap", test_failure_size_cap),
        ("failure: error 500 字 truncate", test_failure_error_truncation),
        ("HookRunner 統合", test_hook_runner_integration),
        ("HookRunner chain post 複数", test_hook_runner_chain_success_and_fail),
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
