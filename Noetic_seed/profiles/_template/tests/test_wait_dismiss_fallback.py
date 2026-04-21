"""test_wait_dismiss_fallback.py — 案 C: wait dismiss の log id fallback。

背景:
  log id ({session}_{cycle} 形式、例: 728c156c_0025) と pending id
  (p_{session}_{cycle}_{source_action}_{uuid} 形式) が session+cycle prefix
  共通で、LLM が視覚的に混同しやすい構造的欠陥。
  2026-04-18 3bf6a6e で schema description に明記したが、2026-04-21 smoke の
  cycle 26 で LLM が依然 log id (`728c156c_0025`) を dismiss に指定して失敗。

設計判断 (ゆう 2026-04-21):
  「構造的欠陥なので wait 指定を LLM に学ばせる意味が薄い」→ 案 C (system 側で
  log id fallback map)。feedback_llm_as_brain 整合 (LLM を brain として扱う =
  id 区別のような認知負荷は system が吸収する)。

検証対象:
  - 既存 p_ prefix id で dismiss 成功 (回帰)
  - log id ({session}_{cycle}) で dismiss 成功 (新動作)
  - log id で cycle に pending なしで失敗
  - 不正形式で dismiss 失敗 (回帰)
  - log id + 複数 pending で priority 最高を選ぶ
  - 消化済 pending は fallback 対象外
  - 段階10.5 Fix 2 漏れ: dismiss メッセージの content_intent 表示

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_wait_dismiss_fallback.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import builtin as _builtin_module


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    return bool(cond)


def _with_tmp_state(fake_state):
    """_wait_or_dismiss 実行時の load_state/save_state を差し替え。"""
    saved = []
    orig_load = _builtin_module.load_state
    orig_save = _builtin_module.save_state
    _builtin_module.load_state = lambda: fake_state
    _builtin_module.save_state = lambda s: saved.append(s)
    return saved, orig_load, orig_save


def _restore(orig_load, orig_save):
    _builtin_module.load_state = orig_load
    _builtin_module.save_state = orig_save


def _make_state(pendings):
    return {"pending": pendings, "unresponded_external_count": 0}


# ============================================================
# Section 1: 既存動作維持 (p_ prefix)
# ============================================================
print("=== Section 1: 既存動作維持 ===")

state = _make_state([
    {"id": "p_abc12345_0024_response_xyz", "type": "pending",
     "source_action": "response_to_external", "origin_cycle": 24,
     "observed_content": None, "priority": 5.0,
     "content_intent": "ゆうへの応答"},
])
saved, ol, os_ = _with_tmp_state(state)
try:
    result = _builtin_module._wait_or_dismiss({"dismiss": "p_abc12345_0024_response_xyz"})
    _assert(not result.startswith("[dismiss] id="),
            f"1-1 p_ prefix で dismiss 成功: {result!r}")
    _assert(len(saved) == 1, "1-1b save_state 呼出し")
    _assert(len(saved[0]["pending"]) == 0, "1-1c pending 削除")
    _assert("ゆうへの応答" in result, "1-1d content_intent 表示 (Fix 2 漏れ補完)")
finally:
    _restore(ol, os_)


# ============================================================
# Section 2: log id fallback 成功
# ============================================================
print("=== Section 2: log id fallback ===")

state = _make_state([
    {"id": "p_abc12345_0025_response_xyz", "type": "pending",
     "source_action": "response_to_external", "origin_cycle": 25,
     "observed_content": None, "priority": 5.0,
     "content_intent": "cycle 25 の応答意図"},
])
saved, ol, os_ = _with_tmp_state(state)
try:
    result = _builtin_module._wait_or_dismiss({"dismiss": "abc12345_0025"})
    _assert(not result.startswith("[dismiss] id="),
            f"2-1 log id で dismiss 成功: {result!r}")
    _assert(len(saved) == 1, "2-1b save_state 呼出し")
    _assert(len(saved[0]["pending"]) == 0, "2-1c pending 削除")
finally:
    _restore(ol, os_)


# ============================================================
# Section 3: log id で cycle に pending なし → 失敗
# ============================================================
print("=== Section 3: log id でヒットなし ===")

state = _make_state([
    {"id": "p_abc12345_0030_response_xyz", "type": "pending",
     "source_action": "response_to_external", "origin_cycle": 30,
     "observed_content": None, "priority": 5.0, "content_intent": "別 cycle"},
])
saved, ol, os_ = _with_tmp_state(state)
try:
    result = _builtin_module._wait_or_dismiss({"dismiss": "abc12345_0025"})
    _assert(result.startswith("[dismiss] id="),
            "3-1 cycle 25 に pending なしで失敗")
finally:
    _restore(ol, os_)


# ============================================================
# Section 4: 不正形式 (既存動作、回帰)
# ============================================================
print("=== Section 4: 不正形式 ===")

state = _make_state([])
saved, ol, os_ = _with_tmp_state(state)
try:
    result = _builtin_module._wait_or_dismiss({"dismiss": "garbage_invalid"})
    _assert(result.startswith("[dismiss] id="), "4-1 不正形式で失敗")
finally:
    _restore(ol, os_)


# ============================================================
# Section 5: log id + 複数 pending で priority 最高選択
# ============================================================
print("=== Section 5: 複数 pending で priority ===")

state = _make_state([
    {"id": "p_abc12345_0025_a_1", "type": "pending",
     "source_action": "response_to_external", "origin_cycle": 25,
     "observed_content": None, "priority": 3.0,
     "content_intent": "low priority"},
    {"id": "p_abc12345_0025_b_2", "type": "pending",
     "source_action": "reflect", "origin_cycle": 25,
     "observed_content": None, "priority": 8.0,
     "content_intent": "high priority"},
])
saved, ol, os_ = _with_tmp_state(state)
try:
    result = _builtin_module._wait_or_dismiss({"dismiss": "abc12345_0025"})
    _assert(not result.startswith("[dismiss] id="),
            f"5-1 log id で dismiss 成功: {result!r}")
    remaining_ids = [p["id"] for p in saved[0]["pending"]]
    _assert("p_abc12345_0025_a_1" in remaining_ids,
            "5-1b low priority が残る (high priority が削除された)")
    _assert("p_abc12345_0025_b_2" not in remaining_ids,
            "5-1c high priority が削除対象に選ばれた")
finally:
    _restore(ol, os_)


# ============================================================
# Section 6: 消化済 pending は fallback 対象外
# ============================================================
print("=== Section 6: 消化済 skip ===")

state = _make_state([
    {"id": "p_abc12345_0025_response_xyz", "type": "pending",
     "source_action": "response_to_external", "origin_cycle": 25,
     "observed_content": "既に観察済",  # 消化済
     "priority": 5.0, "content_intent": "observed"},
])
saved, ol, os_ = _with_tmp_state(state)
try:
    result = _builtin_module._wait_or_dismiss({"dismiss": "abc12345_0025"})
    _assert(result.startswith("[dismiss] id="),
            "6-1 消化済 pending は fallback 対象外")
finally:
    _restore(ol, os_)


# ============================================================
# 結果サマリ
# ============================================================
print("\n========== SUMMARY ==========")
passed = sum(1 for ok, _ in results if ok)
failed = [msg for ok, msg in results if not ok]
print(f"passed: {passed}/{len(results)}")
if failed:
    print(f"failed: {len(failed)}")
    for msg in failed:
        print(f"  - {msg}")
    sys.exit(1)
print("all pass")
