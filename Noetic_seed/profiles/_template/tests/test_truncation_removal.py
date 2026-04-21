"""段階10 Step 4 付帯 D: truncation 撤去テスト。

Fix 5 (ui_tools.py content[:80] 撤去) 精神の一般化。log.result に届く
「iku 発話 echo back」系 return での content 短縮を 6 ファイル 7 箇所で撤去。

対象:
  - tools/memory_tool.py:145 — content[:60]
  - tools/x_tools.py:506 — text[:80]
  - tools/elyth_tools.py:32 — content[:80] (投稿)
  - tools/elyth_tools.py:59 — content[:80] (返信)
  - core/runtime/tools/ui.py:78 — message[:100]
  - core/runtime/tools/task.py:92 — description[:100]
  - core/runtime/tools/plan.py:30,38 — [:200] x 2

維持対象 (判断保留):
  - http_tool の url[:80] / str(e)[:200] (error msg、iku 発話じゃない)
  - elyth_tools の JSON[:3000] (API response 部分出力)
  - sandbox の output[:5000] (bash 外部出力、ui_tools とは別 path)
  - x_tools:414 の cell[:200] (X 通知、iku 発話じゃない)
  - builtin の dismiss content[:50] (境界、priority 低で維持)

戦略: 実行可能な unit test (plan/task/ui/runtime) + 静的 src check
(memory/x/elyth、外部依存が重い箇所は src inspection で撤去確認)。

使い方:
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_truncation_removal.py
"""
import sys
import inspect
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# ============================================================
# core/runtime/tools/plan.py — enter_plan_mode / exit_plan_mode
# ============================================================

def test_plan_enter_full_content():
    print("== plan.enter_plan_mode: 200 字超の plan を full 返す ==")
    from core.runtime.tools import plan
    long_plan = "計画内容" * 100  # 400 字
    result = plan.enter_plan_mode({"plan": long_plan})
    try:
        plan.exit_plan_mode({})  # クリーンアップ
    except Exception:
        pass
    return all([
        _assert(long_plan in result, f"full plan 含む (len={len(long_plan)})"),
        _assert("[:200]" not in result, "明示的 truncation なし"),
    ])


def test_plan_exit_full_content():
    print("== plan.exit_plan_mode: 200 字超の previous plan を full 返す ==")
    from core.runtime.tools import plan
    long_plan = "保存" * 150  # 300 字
    plan.enter_plan_mode({"plan": long_plan})
    result = plan.exit_plan_mode({})
    return _assert(
        long_plan in result,
        f"full saved plan 含む (len={len(long_plan)})",
    )


def test_plan_empty_fallback():
    print("== plan.enter_plan_mode: 空 plan で '(empty)' fallback ==")
    from core.runtime.tools import plan
    result = plan.enter_plan_mode({})
    plan.exit_plan_mode({})  # クリーンアップ
    return _assert("(empty)" in result, "空 plan は '(empty)' 表示")


# ============================================================
# core/runtime/tools/task.py — task_create
# ============================================================

def test_task_create_full_description():
    print("== task.task_create: 100 字超の description を full 返す ==")
    from core.runtime.tools import task as task_mod
    long_desc = "タスク詳細" * 30  # 150 字
    result = task_mod.task_create({"description": long_desc})
    return all([
        _assert(long_desc in result, f"full description 含む (len={len(long_desc)})"),
        _assert("[:100]" not in result, "明示的 truncation なし"),
    ])


# ============================================================
# core/runtime/tools/ui.py — send_user_message
# ============================================================

def test_ui_sent_full_message_via_mock():
    print("== ui.send_user_message: 100 字超の message を full 返す (callback mock) ==")
    from core.runtime.tools import ui as ui_mod

    # send_user callback を _ui_bridge 経由で mock して Sent path に入る
    original_fn = ui_mod._ui_bridge.get("send_user")
    ui_mod._ui_bridge["send_user"] = lambda msg, attach, status: None
    try:
        long_msg = "メッセージ" * 30  # 150 字
        result = ui_mod.send_user_message({"message": long_msg})
        return all([
            _assert(long_msg in result, f"full message 含む (len={len(long_msg)})"),
            _assert(result.startswith("Sent"), "Sent path"),
        ])
    finally:
        ui_mod._ui_bridge["send_user"] = original_fn


# ============================================================
# tools/memory_tool.py — 静的 src inspection (state 依存で重い)
# ============================================================

def test_memory_tool_src_no_content_truncation():
    print("== memory_tool._tool_memory_store: src に content[:60] なし ==")
    from tools import memory_tool
    src = inspect.getsource(memory_tool._tool_memory_store)
    return all([
        _assert("content[:60]" not in src, "content[:60] 撤去済"),
        _assert("記憶保存完了" in src, "return format は維持"),
    ])


# ============================================================
# tools/x_tools.py — 静的 src inspection (playwright 依存)
# ============================================================

def test_x_tools_src_no_text_truncation():
    print("== x_tools._x_reply: src に text[:80] なし ==")
    from tools import x_tools
    # _x_reply は playwright 依存でモジュール全体 src 検査
    src = inspect.getsource(x_tools)
    # 対象行: "返信完了: {text[:80]}" が消えていること
    return _assert(
        "返信完了: {text[:80]}" not in src,
        "text[:80] が返信完了 format から撤去",
    )


# ============================================================
# tools/elyth_tools.py — 静的 src inspection (httpx 依存)
# ============================================================

def test_elyth_tools_src_no_content_truncation():
    print("== elyth_tools._elyth_post / _elyth_reply: src に content[:80] なし ==")
    from tools import elyth_tools
    post_src = inspect.getsource(elyth_tools._elyth_post)
    reply_src = inspect.getsource(elyth_tools._elyth_reply)
    return all([
        _assert("投稿完了: {content[:80]}" not in post_src, "投稿完了 format 撤去"),
        _assert("返信完了: {content[:80]}" not in reply_src, "返信完了 format 撤去"),
        # 維持対象 (JSON[:3000]) は別の関数内にあるので影響なし確認不要
    ])


# ============================================================
# 逆方向 test: 維持対象が撤去されてないこと
# ============================================================

def test_preserved_truncations_intact():
    print("== 維持対象の truncation が誤って撤去されてない ==")
    from tools import http_tool, elyth_tools, sandbox
    # http_tool の error msg 切り詰めは維持
    http_src = inspect.getsource(http_tool)
    # elyth の JSON[:3000] は維持
    elyth_src = inspect.getsource(elyth_tools)
    # sandbox の output[:5000] は維持
    sandbox_src = inspect.getsource(sandbox)
    return all([
        _assert("str(e)[:200]" in http_src, "http_tool error msg 切り詰め維持"),
        _assert("[:3000]" in elyth_src, "elyth JSON response 切り詰め維持"),
        _assert("[:5000]" in sandbox_src, "sandbox bash output 切り詰め維持"),
    ])


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    tests = [
        test_plan_enter_full_content,
        test_plan_exit_full_content,
        test_plan_empty_fallback,
        test_task_create_full_description,
        test_ui_sent_full_message_via_mock,
        test_memory_tool_src_no_content_truncation,
        test_x_tools_src_no_text_truncation,
        test_elyth_tools_src_no_content_truncation,
        test_preserved_truncations_intact,
    ]
    results = [t() for t in tests]
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
