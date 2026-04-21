"""test_pending_observable_split.py — 段階10.5 Fix 2 (案 P 忠実実装)。

検証対象:
  - pending_add の signature: content → content_intent (必須引数)
  - content_observable 自動生成ルール: f"{source_action} to channel={channel or 'self'} @ cycle {cycle_id}"
  - match_pattern 新構造: source_action / expected_channel / observable_similarity_threshold
  - _matches の新構造判定 (observable 類似度 + 構造 match)
  - migration helper: 旧 pending (content_observable 欠落) を drop

段階10.5 Fix 2 設計判断 (ゆう 2026-04-21 確定、案 P 選択):
  - 新スキーマ: content_observable (機械生成、match 用) + content_intent (LLM 生成、表示用)
  - 旧 content フィールドは撤去
  - match_pattern 新構造で role 明確化 (tool_name_any→source_action、channel_match→expected_channel、
    content_similarity_threshold→observable_similarity_threshold)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_pending_observable_split.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import pending_unified
from core.pending_unified import (
    pending_add,
    pending_add_response_intent,
    migrate_pending_observable_split,
    _matches,
)


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    return bool(cond)


def _install_sim_mock(fn):
    original = pending_unified._sim_check
    pending_unified._sim_check = fn
    return original


# ============================================================
# Section 1: content_observable 自動生成ルール (5 ケース)
# ============================================================
print("=== Section 1: content_observable 自動生成ルール ===")

# 1-1: source_action + channel + cycle_id で機械生成
state = {"pending": [], "cycle_id": 5, "session_id": "t"}
p = pending_add(
    state,
    source_action="output_display",
    expected_observation="応答",
    lag_kind="cycles",
    content_intent="ゆうへの応答",
    cycle_id=5,
    channel="self",
)
_assert(
    p["content_observable"] == "output_display to channel=self @ cycle 5",
    f"1-1 observable = 'output_display to channel=self @ cycle 5' (actual: {p['content_observable']!r})",
)

# 1-2: channel=None → 'self' に fallback
state2 = {"pending": [], "cycle_id": 7, "session_id": "t"}
p2 = pending_add(
    state2,
    source_action="reflect",
    expected_observation="内省結果",
    lag_kind="cycles",
    content_intent="自己整理",
    cycle_id=7,
    channel=None,
)
_assert(
    p2["content_observable"] == "reflect to channel=self @ cycle 7",
    f"1-2 channel=None → 'self' fallback (actual: {p2['content_observable']!r})",
)

# 1-3: content_intent と content_observable が異なる (意図 vs 観測事実)
_assert(
    p["content_intent"] == "ゆうへの応答",
    "1-3 content_intent は引数値そのまま",
)
_assert(
    p["content_intent"] != p["content_observable"],
    "1-3b intent != observable (意図と観測事実の分離)",
)

# 1-4: 旧 content フィールドは撤去されている
_assert(
    "content" not in p or p.get("content") is None,
    f"1-4 旧 content フィールド不在 (actual keys: {sorted(p.keys())})",
)

# 1-5: 同 cycle 同 source_action 同 channel なら observable 完全一致 (semantic merge 用)
state3 = {"pending": [], "cycle_id": 10, "session_id": "t"}
p3a = pending_add(
    state3,
    source_action="output_display",
    expected_observation="応答A",
    lag_kind="cycles",
    content_intent="ゆうへの応答その1 (表現A)",
    cycle_id=10,
    channel="self",
)
p3b = pending_add(
    state3,
    source_action="output_display",
    expected_observation="応答B",
    lag_kind="cycles",
    content_intent="ゆうへの応答その1 (表現B微差)",  # intent は微差
    cycle_id=10,
    channel="self",
)
_assert(
    p3a["content_observable"] == p3b["content_observable"],
    "1-5 同 cycle 同 source_action 同 channel で observable 完全一致 (intent 微差でも)",
)


# ============================================================
# Section 2: pending_add signature (3 ケース)
# ============================================================
print("=== Section 2: pending_add signature ===")

# 2-1: content_intent 引数で受け取る (旧 content は撤去)
state4 = {"pending": [], "cycle_id": 0, "session_id": "t"}
p4 = pending_add(
    state4,
    source_action="memory_store",
    expected_observation="記憶化",
    lag_kind="cycles",
    content_intent="新 signature test",
    cycle_id=0,
)
_assert(p4["content_intent"] == "新 signature test", "2-1 content_intent 受け取り")
_assert("content_observable" in p4, "2-1b content_observable 自動生成")

# 2-2: content_intent 過長時は truncate (500 文字)
state5 = {"pending": [], "cycle_id": 0, "session_id": "t"}
long_text = "x" * 1000
p5 = pending_add(
    state5,
    source_action="bash",
    expected_observation="実行",
    lag_kind="cycles",
    content_intent=long_text,
    cycle_id=0,
)
_assert(len(p5["content_intent"]) == 500, f"2-2 content_intent 500 字 truncate (actual len={len(p5['content_intent'])})")

# 2-3: pending_add_response_intent は新 signature で pending 生成
state6 = {"pending": [], "cycle_id": 3, "session_id": "t"}
p6 = pending_add_response_intent(
    state6, channel="device", text="こんにちは", cycle_id=3,
)
_assert("content_intent" in p6, "2-3 response_intent が content_intent 設定")
_assert("content_observable" in p6, "2-3b response_intent が content_observable 設定")
_assert(
    p6["content_observable"] == "response_to_external to channel=device @ cycle 3",
    f"2-3c observable = 'response_to_external to channel=device @ cycle 3' (actual: {p6['content_observable']!r})",
)


# ============================================================
# Section 3: match_pattern 新構造 (5 ケース)
# ============================================================
print("=== Section 3: match_pattern 新構造 ===")

# 3-1: source_action 一致で match (単一 tool 名)
pending_s = {
    "content_observable": "output_display to channel=self @ cycle 5",
    "content_intent": "応答",
    "expected_channel": "self",
}
mp = {"source_action": "output_display"}
ok = _matches(mp, "output_display", {}, "result", "self", pending_s)
_assert(ok is True, "3-1 source_action 一致で True")

# 3-2: source_action 不一致で False
ok = _matches(mp, "bash", {}, "result", "self", pending_s)
_assert(ok is False, "3-2 source_action 不一致で False")

# 3-3: expected_channel 一致で match
mp_ch = {"expected_channel": "self"}
ok = _matches(mp_ch, "output_display", {}, "result", "self", pending_s)
_assert(ok is True, "3-3 expected_channel 一致で True")

# 3-4: expected_channel 不一致で False
ok = _matches(mp_ch, "output_display", {}, "result", "device", pending_s)
_assert(ok is False, "3-4 expected_channel 不一致で False")

# 3-5: observable_similarity_threshold で類似度判定 (source_action 併用時のみ有効)
# 段階11-A hotfix (2026-04-22): tool 特定 field なしで類似度のみの match は
# 誤消化の温床のため拒否される。source_action と併用時に類似度機能が働く
original = _install_sim_mock(lambda a, b, t: True)
try:
    mp_sim = {"source_action": "output_display",
              "observable_similarity_threshold": 0.7}
    ok = _matches(mp_sim, "output_display", {}, "result text", "self", pending_s)
    _assert(ok is True, "3-5 observable_similarity True (source_action 併用)")
finally:
    pending_unified._sim_check = original


# ============================================================
# Section 4: migration helper (2 ケース)
# ============================================================
print("=== Section 4: migrate_pending_observable_split ===")

# 4-1: 旧 pending (content_observable 欠落) を drop
state_old = {
    "pending": [
        {"type": "pending", "id": "p_old_1", "content": "旧形式 1"},
        {"type": "pending", "id": "p_old_2", "content": "旧形式 2"},
        {
            "type": "pending",
            "id": "p_new_1",
            "content_observable": "new obs",
            "content_intent": "new intent",
        },
    ]
}
dropped = migrate_pending_observable_split(state_old)
_assert(dropped == 2, f"4-1 drop 件数 = 2 (actual: {dropped})")
_assert(len(state_old["pending"]) == 1, "4-1b 残存 1 件")
_assert(
    state_old["pending"][0]["id"] == "p_new_1",
    "4-1c 新形式 pending のみ残存",
)

# 4-2: 旧 pending なしで drop=0
state_clean = {"pending": []}
dropped_clean = migrate_pending_observable_split(state_clean)
_assert(dropped_clean == 0, "4-2 空 state で drop=0")


# ============================================================
# Section 5: pending_add_response_intent の match_pattern 新構造 (2 ケース)
# ============================================================
print("=== Section 5: response_intent match_pattern 新構造 ===")

state7 = {"pending": [], "cycle_id": 8, "session_id": "t"}
p7 = pending_add_response_intent(state7, channel="claude", text="質問", cycle_id=8)
mp7 = p7.get("match_pattern") or {}

# 5-1: source_action = output_display
_assert(
    mp7.get("source_action") == "output_display",
    f"5-1 source_action='output_display' (actual: {mp7.get('source_action')!r})",
)

# 5-2: expected_channel = 受信 channel と一致
_assert(
    mp7.get("expected_channel") == "claude",
    f"5-2 expected_channel='claude' (actual: {mp7.get('expected_channel')!r})",
)


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
