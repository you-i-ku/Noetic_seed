"""make_post_body_modify_pending_hook の身体改変 pending 自動追加 test
(段階12 Step 5, PLAN §9 / §13-1)。

検証ケース:
  - core/foo.py への write_file 成功 → pending_add 呼ばれる
  - tools/sandbox.py への edit_file 成功 → pending_add 呼ばれる
  - main.py / .mcp.json への write_file 成功 → pending_add 呼ばれる
  - sandbox/memo.md / pref.json への write_file → pending_add 呼ばれない (対象外)
  - read_file 等の対象外 tool → pending_add 呼ばれない
  - pending_add が exception 上げても hook は allow 返す (書換えは続行扱い)
  - 追加された pending の content_intent / source_action / match_pattern が
    PLAN §9-2 の翻訳どおり

pending_add は test 用 mock に置き換え、実 state の mutation はしない。

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_pending_body_modify.py
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.hooks import make_post_body_modify_pending_hook


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _make_hook_with_mock_state():
    """mock state + cycle_id getter + hook を組合せて返す。"""
    state = {"pending": [], "cycle_id": 42, "session_id": "testsess"}
    hook = make_post_body_modify_pending_hook(
        state_getter=lambda: state,
        get_cycle_id=lambda: state.get("cycle_id", 0),
    )
    return state, hook


def test_pending_added_for_core_path():
    print("== write_file(core/foo.py) で pending 追加 ==")
    state, hook = _make_hook_with_mock_state()
    with patch("core.pending_unified.pending_add") as mock_add:
        r = hook("write_file", {"path": "core/foo.py", "content": "x"}, "")
    # pending_add が呼ばれていなければ後続 kwargs assertion は無意味なので早期 return
    if not mock_add.called:
        return _assert(False, "pending_add が呼ばれていない (前提崩れ)")
    kwargs = mock_add.call_args.kwargs
    return all([
        _assert(not r.denied, "allow"),
        _assert(mock_add.called, "pending_add 呼ばれた"),
        _assert(kwargs.get("source_action") == "write_file",
                "source_action=write_file"),
        _assert("core/foo.py" in kwargs.get("content_intent", ""),
                "content_intent に path 含む"),
        _assert(kwargs.get("lag_kind") == "cycles", "lag_kind=cycles"),
        _assert(kwargs.get("semantic_merge") is True, "semantic_merge=True"),
        _assert(kwargs.get("match_pattern") == {"tool_name": "reboot"},
                "match_pattern={tool_name:reboot}"),
        _assert(kwargs.get("cycle_id") == 42, "cycle_id=42"),
    ])


def test_pending_added_for_tools_path():
    print("== edit_file(tools/sandbox.py) で pending 追加 ==")
    state, hook = _make_hook_with_mock_state()
    with patch("core.pending_unified.pending_add") as mock_add:
        r = hook("edit_file", {"path": "tools/sandbox.py",
                                "old_string": "a", "new_string": "b"}, "")
    return all([
        _assert(mock_add.called, "pending_add 呼ばれた"),
        _assert(mock_add.call_args.kwargs.get("source_action") == "edit_file",
                "source_action=edit_file"),
    ])


def test_pending_added_for_main_py():
    print("== write_file(main.py) で pending 追加 ==")
    state, hook = _make_hook_with_mock_state()
    with patch("core.pending_unified.pending_add") as mock_add:
        hook("write_file", {"path": "main.py", "content": "x"}, "")
    return _assert(mock_add.called, "pending_add 呼ばれた (main.py = 神経中枢)")


def test_pending_added_for_mcp_json():
    print("== write_file(.mcp.json) で pending 追加 ==")
    state, hook = _make_hook_with_mock_state()
    with patch("core.pending_unified.pending_add") as mock_add:
        hook("write_file", {"path": ".mcp.json", "content": "{}"}, "")
    return _assert(mock_add.called, "pending_add 呼ばれた (.mcp.json = 関係性の器)")


def test_pending_skipped_for_sandbox_path():
    print("== write_file(sandbox/memo.md) で pending 追加されない ==")
    state, hook = _make_hook_with_mock_state()
    with patch("core.pending_unified.pending_add") as mock_add:
        hook("write_file", {"path": "sandbox/memo.md", "content": "x"}, "")
    return _assert(not mock_add.called, "pending_add 呼ばれない")


def test_pending_skipped_for_pref_json():
    print("== write_file(pref.json) で pending 追加されない ==")
    state, hook = _make_hook_with_mock_state()
    with patch("core.pending_unified.pending_add") as mock_add:
        hook("write_file", {"path": "pref.json", "content": "{}"}, "")
    return _assert(not mock_add.called, "pending_add 呼ばれない")


def test_pending_skipped_for_read_tool():
    print("== read_file は対象外 tool (pending 追加されない) ==")
    state, hook = _make_hook_with_mock_state()
    with patch("core.pending_unified.pending_add") as mock_add:
        hook("read_file", {"path": "core/foo.py"}, "")
    return _assert(not mock_add.called, "pending_add 呼ばれない")


def test_pending_add_exception_is_swallowed():
    """pending_add が exception 上げても hook は allow を返す (書換え続行)。"""
    print("== pending_add 例外時も hook は allow ==")
    state, hook = _make_hook_with_mock_state()
    with patch("core.pending_unified.pending_add",
               side_effect=RuntimeError("simulated")):
        r = hook("write_file", {"path": "core/foo.py", "content": "x"}, "")
    return _assert(not r.denied, "allow (例外は warning に吸収、書換え続行)")


if __name__ == "__main__":
    groups = [
        ("write: core/ で pending 追加", test_pending_added_for_core_path),
        ("edit: tools/ で pending 追加", test_pending_added_for_tools_path),
        ("write: main.py で pending 追加", test_pending_added_for_main_py),
        ("write: .mcp.json で pending 追加", test_pending_added_for_mcp_json),
        ("write: sandbox は対象外", test_pending_skipped_for_sandbox_path),
        ("write: pref.json は対象外", test_pending_skipped_for_pref_json),
        ("read_file は対象外 tool", test_pending_skipped_for_read_tool),
        ("pending_add 例外時も allow", test_pending_add_exception_is_swallowed),
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
