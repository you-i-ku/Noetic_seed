"""approval_callback factory テスト。

Phase 4 Step E-2c: make_approval_callback の動作検証。
  - 3 層 preview 整形
  - pause_on_await 連動 (is_paused 発動/解放)
  - request_approval 戻り値の透過
  - 例外時の pause 解放保証 (try/finally)
  - 依存の注入テスト (request_approval_fn / set_paused_fn)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_approval_callback.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.approval_callback import _format_preview, make_approval_callback


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _full_input(**overrides):
    base = {
        "tool_intent": "設定を書く",
        "tool_expected_outcome": "file 作成",
        "message": "設定を書きます",
        "path": "/tmp/foo.py",
        "content": "x",
    }
    base.update(overrides)
    return base


# ============================================================
# _format_preview
# ============================================================

def test_preview_has_three_layers():
    print("== preview: 3 層構造 (① what / ② why / ③ to_human) ==")
    preview = _format_preview("write_file", _full_input(), [])
    return all([
        _assert("① what (args)" in preview, "① what"),
        _assert("② why:" in preview, "② why"),
        _assert("③ to you" in preview, "③ to you"),
    ])


def test_preview_hides_approval_fields_from_args():
    print("== preview: args に承認 3 層は混ざらない ==")
    preview = _format_preview("write_file", _full_input(), [])
    # args 行だけ切り出してチェック
    args_line = [l for l in preview.split("\n") if "① what" in l][0]
    return all([
        _assert("path" in args_line, "tool 固有 args (path) は表示"),
        _assert("tool_intent" not in args_line,
                "tool_intent は args から除外"),
        _assert("tool_expected_outcome" not in args_line,
                "tool_expected_outcome は args から除外"),
    ])


def test_preview_shows_intent_and_expected():
    print("== preview: intent / expected が表示される ==")
    preview = _format_preview("write_file", _full_input(), [])
    return all([
        _assert("intent:   設定を書く" in preview, "intent"),
        _assert("expected: file 作成" in preview, "expected"),
    ])


def test_preview_shows_message_section():
    print("== preview: message が ③ 区画に表示 ==")
    preview = _format_preview("write_file", _full_input(), [])
    return _assert("設定を書きます" in preview, "message text")


def test_preview_missing_fields_placeholder():
    print("== preview: 3 層欠損は '(空)' 表示 ==")
    inp = _full_input(tool_intent="", tool_expected_outcome="", message="")
    preview = _format_preview("write_file", inp, [])
    return all([
        _assert("intent:   (空)" in preview, "intent 空"),
        _assert("expected: (空)" in preview, "expected 空"),
        _assert("(空)" in preview, "message 空"),
    ])


def test_preview_includes_pre_hook():
    print("== preview: pre_hook_messages が末尾に含まれる ==")
    preview = _format_preview("write_file", _full_input(),
                              ["[approval] warning: x", "[other] y"])
    return all([
        _assert("[pre_hook]" in preview, "[pre_hook] セクション"),
        _assert("warning: x" in preview, "1 件目"),
        _assert("[other] y" in preview, "2 件目"),
    ])


def test_preview_no_pre_hook_section_when_empty():
    print("== preview: pre_hook なしならセクション出ない ==")
    preview = _format_preview("write_file", _full_input(), [])
    return _assert("[pre_hook]" not in preview, "[pre_hook] 無し")


# ============================================================
# make_approval_callback: 動作
# ============================================================

def _stub_approval(returns: bool = True, capture: dict = None):
    def _f(tool_name, preview, timeout_sec):
        if capture is not None:
            capture["tool_name"] = tool_name
            capture["preview"] = preview
            capture["timeout_sec"] = timeout_sec
        return returns
    return _f


def _stub_pause(history: list):
    def _f(value):
        history.append(bool(value))
    return _f


def test_callback_calls_request_approval():
    print("== callback: request_approval に tool_name/preview 渡す ==")
    cap = {}
    cb = make_approval_callback(
        pause_on_await=False,
        request_approval_fn=_stub_approval(True, cap),
        set_paused_fn=_stub_pause([]),
    )
    result = cb("write_file", _full_input(), [])
    return all([
        _assert(result is True, "True 透過"),
        _assert(cap.get("tool_name") == "write_file", "tool_name 渡る"),
        _assert("① what" in cap.get("preview", ""), "preview 構造"),
        _assert(cap.get("timeout_sec") == 300, "default timeout"),
    ])


def test_callback_passthrough_false():
    print("== callback: request_approval が False → callback も False ==")
    cb = make_approval_callback(
        pause_on_await=False,
        request_approval_fn=_stub_approval(False),
        set_paused_fn=_stub_pause([]),
    )
    return _assert(cb("write_file", _full_input(), []) is False,
                   "False 透過")


def test_callback_custom_timeout():
    print("== callback: timeout_sec のカスタム値が渡る ==")
    cap = {}
    cb = make_approval_callback(
        pause_on_await=False, timeout_sec=60,
        request_approval_fn=_stub_approval(True, cap),
        set_paused_fn=_stub_pause([]),
    )
    cb("bash", _full_input(), [])
    return _assert(cap.get("timeout_sec") == 60, "timeout=60")


# ============================================================
# pause_on_await 連動
# ============================================================

def test_pause_on_await_true_sets_and_clears():
    print("== pause_on_await=True: set_paused(True)→False 順序 ==")
    history: list = []
    cb = make_approval_callback(
        pause_on_await=True,
        request_approval_fn=_stub_approval(True),
        set_paused_fn=_stub_pause(history),
    )
    cb("write_file", _full_input(), [])
    return all([
        _assert(history == [True, False],
                f"set_paused 呼出順序: {history}"),
    ])


def test_pause_on_await_false_no_setpaused():
    print("== pause_on_await=False: set_paused 触らない ==")
    history: list = []
    cb = make_approval_callback(
        pause_on_await=False,
        request_approval_fn=_stub_approval(True),
        set_paused_fn=_stub_pause(history),
    )
    cb("write_file", _full_input(), [])
    return _assert(history == [], f"set_paused 未呼出 (実={history})")


def test_pause_released_on_exception():
    print("== 例外発生でも pause が解放される (try/finally) ==")
    history: list = []

    def _raising(tool_name, preview, timeout_sec):
        raise RuntimeError("承認中エラー")

    cb = make_approval_callback(
        pause_on_await=True,
        request_approval_fn=_raising,
        set_paused_fn=_stub_pause(history),
    )
    try:
        cb("write_file", _full_input(), [])
        return _assert(False, "例外伝搬期待")
    except RuntimeError:
        return _assert(history == [True, False],
                       f"例外後も False 解放 (実={history})")


def test_pause_released_on_denial():
    print("== 拒否 (False) でも pause が解放される ==")
    history: list = []
    cb = make_approval_callback(
        pause_on_await=True,
        request_approval_fn=_stub_approval(False),
        set_paused_fn=_stub_pause(history),
    )
    result = cb("write_file", _full_input(), [])
    return all([
        _assert(result is False, "False 透過"),
        _assert(history == [True, False], "pause 解放"),
    ])


# ============================================================
# ws_server 実連動 (set_paused が本物で動くか)
# ============================================================

def test_real_set_paused_integration():
    print("== ws_server.set_paused と is_paused 実連動 ==")
    from core import ws_server
    # initial state
    ws_server.set_paused(False)
    initial = ws_server.is_paused()

    cb = make_approval_callback(
        pause_on_await=True,
        request_approval_fn=_stub_approval(True),
        # set_paused_fn は省略 → 実 ws_server.set_paused を遅延 import
    )
    # callback 内で set_paused(True) → request_approval → set_paused(False)
    cb("write_file", _full_input(), [])
    # callback 完了後は pause が解放されているはず
    after = ws_server.is_paused()
    return all([
        _assert(initial is False, "初期 False"),
        _assert(after is False, "callback 後 False (解放済)"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("preview: 3 層構造", test_preview_has_three_layers),
        ("preview: args 分離", test_preview_hides_approval_fields_from_args),
        ("preview: intent/expected", test_preview_shows_intent_and_expected),
        ("preview: message ③", test_preview_shows_message_section),
        ("preview: 空欄 placeholder", test_preview_missing_fields_placeholder),
        ("preview: pre_hook 含む", test_preview_includes_pre_hook),
        ("preview: pre_hook 無し", test_preview_no_pre_hook_section_when_empty),
        ("callback: request_approval 渡す",
         test_callback_calls_request_approval),
        ("callback: False 透過", test_callback_passthrough_false),
        ("callback: timeout カスタム", test_callback_custom_timeout),
        ("pause: True で発動/解放", test_pause_on_await_true_sets_and_clears),
        ("pause: False で無触り", test_pause_on_await_false_no_setpaused),
        ("pause: 例外時解放", test_pause_released_on_exception),
        ("pause: 拒否時解放", test_pause_released_on_denial),
        ("ws_server 実連動", test_real_set_paused_integration),
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
