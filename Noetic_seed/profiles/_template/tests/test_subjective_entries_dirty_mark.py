"""subjective_entries dirty mark 漏れ fix テスト (Slice 1 hotfix 2、別 review P1-A).

x_tools._resolve_x_feedback と elyth_tools._resolve_elyth_feedback で
state["subjective_entries"] の e2 を直接更新する経路で、明示的に
mark_view_dirty("subjective_entries") を呼出する fix の verify。

背景: subjective_entries は materialized view (段階13 Phase 0.1.D)、save_state は
dirty view だけ JSONL に flush する設計。dirty mark なしだと再起動後 E2 修正が
JSONL に書き戻されず消える (別 review 指摘 P1-A)。

成功条件:
  - e2 書換が成立 (regex match で +40% bump) → mark_view_dirty 呼ばれる
  - 書換が成立しない (regex unmatch / id 不一致) → mark_view_dirty 呼ばれない
    (不要発火しない、識別力強化)

識別力 (CLAUDE.md §5):
  - dirty mark 抜け実装 (本 hotfix の reverse) で test_*_dirty_marked が fail
  - dirty mark 過剰呼出 (e2 不変でも呼ぶ) で test_*_no_e2_change_no_mark が fail

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_subjective_entries_dirty_mark.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.view_persistence as vp_mod
import core.state as state_mod
import tools.x_tools as x_mod
import tools.elyth_tools as elyth_mod


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# spy state
_DIRTY_CALLS = []


def _reset_spy():
    _DIRTY_CALLS.clear()


def _spy_mark_view_dirty(view_name):
    _DIRTY_CALLS.append(view_name)


# patch backup
_ORIG = {}


def _patch():
    # ORIG 保存は初回のみ (連続 test で _restore 後の本物を再保存しない)
    if not _ORIG:
        _ORIG["mark_view_dirty"] = vp_mod.mark_view_dirty
        _ORIG["load_state"] = state_mod.load_state
        _ORIG["save_state"] = state_mod.save_state
    # spy 設定は毎回 (連続 test で _restore 後でも有効化)
    vp_mod.mark_view_dirty = _spy_mark_view_dirty


def _restore():
    if not _ORIG:
        return
    vp_mod.mark_view_dirty = _ORIG["mark_view_dirty"]
    state_mod.load_state = _ORIG["load_state"]
    state_mod.save_state = _ORIG["save_state"]


def _make_state_with_pending(tool_prefix: str, log_entry_id: str = "le1",
                              e2_initial: str = "30%"):
    """test 用 state: pending_feedback (awaiting) + subjective_entries 1 件。"""
    return {
        "pending_feedback": [{
            "status": "awaiting",
            "tool": f"{tool_prefix}_post",
            "text_snippet": "test snippet content",
            "entity": "",
            "log_entry_id": log_entry_id,
        }],
        "subjective_entries": [{
            "id": log_entry_id,
            "e2": e2_initial,
        }],
    }


# ============================================================
# Section A: x_tools._resolve_x_feedback
# ============================================================

def test_x_e2_change_marks_dirty():
    print("== x_tools: e2 書換成立 → mark_view_dirty('subjective_entries') 呼ばれる ==")
    _patch()
    _reset_spy()
    state = _make_state_with_pending("x", e2_initial="30%")
    state_mod.load_state = lambda: state
    state_mod.save_state = lambda s: None
    notifications = [{
        "type": "like",
        "from": "alice",
        "text": "test snippet content reply text",
    }]
    try:
        x_mod._resolve_x_feedback(notifications)
    finally:
        _restore()
    return all([
        _assert("subjective_entries" in _DIRTY_CALLS,
                f"mark_view_dirty('subjective_entries') 呼出 (got: {_DIRTY_CALLS})"),
        _assert(state["subjective_entries"][0]["e2"] == "70%",
                f"e2 が 30% + 40 = 70% に bump (got: {state['subjective_entries'][0]['e2']!r})"),
    ])


def test_x_no_e2_change_no_mark():
    print("== x_tools: id 不一致で e2 不変 → mark_view_dirty 呼ばれない ==")
    _patch()
    _reset_spy()
    state = _make_state_with_pending("x", log_entry_id="le1", e2_initial="30%")
    # subjective_entries の id を別物に変えて、log_entry_id 一致 entry なし状態に
    state["subjective_entries"][0]["id"] = "different_id"
    state_mod.load_state = lambda: state
    state_mod.save_state = lambda s: None
    notifications = [{
        "type": "like",
        "from": "alice",
        "text": "test snippet content reply text",
    }]
    try:
        x_mod._resolve_x_feedback(notifications)
    finally:
        _restore()
    return all([
        _assert("subjective_entries" not in _DIRTY_CALLS,
                f"mark_view_dirty 呼ばれない (got: {_DIRTY_CALLS})"),
        _assert(state["subjective_entries"][0]["e2"] == "30%",
                f"e2 不変 (got: {state['subjective_entries'][0]['e2']!r})"),
    ])


# ============================================================
# Section B: elyth_tools._resolve_elyth_feedback
# ============================================================

def test_elyth_e2_change_marks_dirty():
    print("== elyth_tools: e2 書換成立 → mark_view_dirty('subjective_entries') 呼ばれる ==")
    _patch()
    _reset_spy()
    state = _make_state_with_pending("elyth", e2_initial="20%")
    notifications = [{
        "notification_type": "like",
        "post_author_handle": "anyone",
    }]
    try:
        result = elyth_mod._resolve_elyth_feedback(notifications, state)
    finally:
        _restore()
    return all([
        _assert(result is True, "_resolve_elyth_feedback returns True (resolved)"),
        _assert("subjective_entries" in _DIRTY_CALLS,
                f"mark_view_dirty('subjective_entries') 呼出 (got: {_DIRTY_CALLS})"),
        _assert(state["subjective_entries"][0]["e2"] == "60%",
                f"e2 が 20% + 40 = 60% に bump (got: {state['subjective_entries'][0]['e2']!r})"),
    ])


def test_elyth_no_e2_change_no_mark():
    print("== elyth_tools: id 不一致で e2 不変 → mark_view_dirty 呼ばれない ==")
    _patch()
    _reset_spy()
    state = _make_state_with_pending("elyth", log_entry_id="le1", e2_initial="30%")
    state["subjective_entries"][0]["id"] = "different_id"
    notifications = [{
        "notification_type": "like",
        "post_author_handle": "anyone",
    }]
    try:
        elyth_mod._resolve_elyth_feedback(notifications, state)
    finally:
        _restore()
    return all([
        _assert("subjective_entries" not in _DIRTY_CALLS,
                f"mark_view_dirty 呼ばれない (got: {_DIRTY_CALLS})"),
        _assert(state["subjective_entries"][0]["e2"] == "30%",
                f"e2 不変 (got: {state['subjective_entries'][0]['e2']!r})"),
    ])


def run_all():
    print("=" * 60)
    print("test_subjective_entries_dirty_mark.py (Slice 1 hotfix 2、別 review P1-A)")
    print("=" * 60)
    results = [
        test_x_e2_change_marks_dirty(),
        test_x_no_e2_change_no_mark(),
        test_elyth_e2_change_marks_dirty(),
        test_elyth_no_e2_change_no_mark(),
    ]
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"結果: {passed}/{total} passed")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
