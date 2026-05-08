"""段階8 Step 2 — 外部入力応答意図 + pending burden → pressure + 安全弁テスト。

WORLD_MODEL_DESIGN/STAGE8_REPETITION_AND_PREDICTOR_PLAN.md §4-1 / §4-3 / §4-4:
  - pending_add_response_intent (改善5 案 5-A): 外部入力 → 内部応答意図
  - pending_burden signal (改善6-D): 未消化 pending 総 priority → pressure
  - pending_prune 安全弁 (PENDING_ATTEMPTS_SAFETY_CAP=50): deprecated マーク

哲学:
  - feedback_failed_observation_no_pending_effect: 失敗観測は pending 無影響
  - feedback_cognitive_load_via_pressure: 認知負荷は pressure で情報理論的に
  - feedback_freedom_to_die: 安全弁はメモリ保護のみ、認知的諦めは wait(dismiss)

使い方:
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_pending_external_and_burden.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pending_unified import (
    PENDING_ATTEMPTS_SAFETY_CAP,
    pending_add,
    pending_add_response_intent,
    pending_prune,
)
from core.entropy import calc_pressure_signals, ENTROPY_PARAMS


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _fresh_state():
    return {
        "cycle_id": 10,
        "raw_events": [], "subjective_entries": [],
        "pending": [],
        "session_id": "test",
        "entropy": 0.5,
        "last_e2": 0.5,
        "last_e3": 0.5,
        "last_e4": 0.3,
        "unresolved_external": 0.0,
    }


# ============================================================
# pending_add_response_intent (改善5)
# ============================================================

def test_response_intent_basic():
    print("== pending_add_response_intent: response_to_external pending 生成 ==")
    state = _fresh_state()
    p = pending_add_response_intent(
        state=state, channel="device",
        text="おねーたん、今日もお疲れ〜",
        cycle_id=10,
    )
    return all([
        _assert(p["source_action"] == "response_to_external", "source_action"),
        _assert(p["expected_channel"] == "device", "expected_channel=device"),
        _assert(p["observation_lag_kind"] == "cycles", "lag=cycles"),
        _assert(p["semantic_merge"] is True, "semantic_merge=True"),
        _assert(p["gap"] == 1.0, "初期 gap=1.0"),
        _assert(len(state["pending"]) == 1, "state に追加"),
    ])


def test_response_intent_match_pattern():
    print("== pending_add_response_intent: match_pattern が正しい ==")
    state = _fresh_state()
    p = pending_add_response_intent(
        state=state, channel="claude",
        text="チャネル指定できるよ", cycle_id=10,
    )
    mp = p.get("match_pattern") or {}
    return all([
        # 段階10.5 Fix 2: match_pattern 新構造 (source_action / expected_channel / observable_similarity_threshold)
        _assert(mp.get("source_action") == "output_display",
                "source_action=output_display"),
        _assert(mp.get("expected_channel") == "claude",
                "expected_channel=claude (受信 channel と一致)"),
        _assert(mp.get("observable_similarity_threshold") is None,
                "observable_similarity_threshold=未指定 (skip)"),
    ])


def test_response_intent_content_preview():
    print("== pending_add_response_intent: content に channel + snippet ==")
    state = _fresh_state()
    long_text = "あ" * 200
    p = pending_add_response_intent(
        state=state, channel="device", text=long_text, cycle_id=0,
    )
    # 段階10.5 Fix 2: content → content_intent (LLM 生成、表示用) に rename
    content = p.get("content_intent", "")
    return all([
        _assert("device" in content, "channel 名が含まれる"),
        _assert("..." in content, "text 切詰め marker あり"),
        _assert(len(content) <= 500, "500 文字 cap 内"),
    ])


# ============================================================
# 段階14.8 — 同 channel 未消化 response_to_external pending の merge logic
# ============================================================
# 識別力 (CLAUDE.md §5): 4 条件 AND の merge 判定で、誤実装 1 つにつき
# 1 つ以上の test が fail するように fixture を設計。
#   - source_action チェック skip → test_no_merge_different_source_action fail
#   - expected_channel チェック skip → test_no_merge_different_channel fail
#   - observed_content チェック skip → test_no_merge_observed_pending fail
#   - 全 pending 無条件 merge → test_no_merge_different_channel fail


def test_response_intent_merge_same_channel_unobserved():
    print("== merge: 同 channel 未消化 → 既存 entry に merge、新規作成しない ==")
    state = _fresh_state()
    p1 = pending_add_response_intent(
        state=state, channel="device", text="ねえ", cycle_id=10,
    )
    p2 = pending_add_response_intent(
        state=state, channel="device", text="聞いてる？", cycle_id=11,
    )
    return all([
        _assert(len(state["pending"]) == 1,
                f"pending は 1 件のまま ({len(state['pending'])})"),
        _assert(p1 is p2, "同一 entry が返される (merge による上書き)"),
        _assert(p2["attempts"] == 2, f"attempts=2 ({p2['attempts']})"),
        _assert("ねえ" in p2["content_intent"], "1 回目の text 残存"),
        _assert("聞いてる？" in p2["content_intent"], "2 回目の text 追記"),
        _assert(p2["last_cycle"] == 11, f"last_cycle=11 ({p2['last_cycle']})"),
    ])


def test_response_intent_merge_three_consecutive():
    print("== merge: 3 連続発話 → 1 件 attempts=3、全 text 累積 ==")
    state = _fresh_state()
    pending_add_response_intent(
        state=state, channel="device", text="ねえ", cycle_id=10,
    )
    pending_add_response_intent(
        state=state, channel="device", text="聞いてる？", cycle_id=11,
    )
    p3 = pending_add_response_intent(
        state=state, channel="device", text="いまどう？", cycle_id=12,
    )
    return all([
        _assert(len(state["pending"]) == 1, "1 件のまま"),
        _assert(p3["attempts"] == 3, f"attempts=3 ({p3['attempts']})"),
        _assert("ねえ" in p3["content_intent"], "1 回目残存"),
        _assert("聞いてる？" in p3["content_intent"], "2 回目残存"),
        _assert("いまどう？" in p3["content_intent"], "3 回目残存"),
    ])


def test_response_intent_no_merge_different_channel():
    print("== no merge: device pending + claude 入力 → 2 件 (channel 識別力) ==")
    state = _fresh_state()
    p_device = pending_add_response_intent(
        state=state, channel="device", text="ねえ", cycle_id=10,
    )
    p_claude = pending_add_response_intent(
        state=state, channel="claude", text="hi there", cycle_id=11,
    )
    return all([
        _assert(len(state["pending"]) == 2,
                f"別 channel は別 pending ({len(state['pending'])})"),
        _assert(p_device is not p_claude, "別 entry"),
        _assert(p_device["expected_channel"] == "device", "device pending 不変"),
        _assert(p_claude["expected_channel"] == "claude", "claude pending 新規"),
        _assert(p_device["attempts"] == 1, "device 側 attempts=1 のまま"),
    ])


def test_response_intent_no_merge_observed_pending():
    print("== no merge: 消化済 device pending + 同 device 入力 → 2 件 ==")
    state = _fresh_state()
    p_old = pending_add_response_intent(
        state=state, channel="device", text="ねえ", cycle_id=10,
    )
    # 手動で消化済マーク
    p_old["observed_content"] = "返事した"
    p_old["gap"] = 0.0
    # 同 channel に新規入力
    p_new = pending_add_response_intent(
        state=state, channel="device", text="また話そう", cycle_id=20,
    )
    return all([
        _assert(len(state["pending"]) == 2,
                f"消化済 + 新規で 2 件 ({len(state['pending'])})"),
        _assert(p_old is not p_new, "別 entry"),
        _assert(p_old["observed_content"] == "返事した",
                "古い pending の消化状態は不変"),
        _assert(p_new["observed_content"] is None, "新 pending は未消化"),
        _assert(p_new["attempts"] == 1, "新 pending attempts=1 (merge せず新規)"),
    ])


def test_response_intent_no_merge_different_source_action():
    print("== no merge: 同 channel 別 source_action は影響しない ==")
    state = _fresh_state()
    # source_action="reflection" の pending を device channel で先置き
    p_other = pending_add(
        state=state, source_action="reflection",
        expected_observation="x", lag_kind="cycles",
        content_intent="別 source_action", cycle_id=10,
        channel="device",
        match_pattern={
            "source_action": "reflect", "expected_channel": "device",
        },
    )
    # 同 channel に response_to_external 入力 → 新規作成 (other は影響なし)
    p_resp = pending_add_response_intent(
        state=state, channel="device", text="ねえ", cycle_id=11,
    )
    return all([
        _assert(len(state["pending"]) == 2,
                f"別 source_action とは混ぜない ({len(state['pending'])})"),
        _assert(p_other["source_action"] == "reflection", "他 pending 不変"),
        _assert(p_resp["source_action"] == "response_to_external",
                "response 新規作成"),
        _assert(p_other["attempts"] == 1, "他 pending attempts 不変"),
        _assert(p_resp["attempts"] == 1, "response 新規 (merge 対象なし)"),
    ])


def test_response_intent_merge_resets_gap_and_priority():
    print("== merge: gap=1.0 リセット + priority 再算 + timestamp 更新 ==")
    state = _fresh_state()
    p1 = pending_add_response_intent(
        state=state, channel="device", text="ねえ", cycle_id=10,
    )
    # 既存 pending の gap を手動で下げて、merge 時にリセットされる識別力を作る
    p1["gap"] = 0.3
    old_timestamp = p1["timestamp"]
    p2 = pending_add_response_intent(
        state=state, channel="device", text="聞いてる？", cycle_id=11,
    )
    return all([
        _assert(p2["gap"] == 1.0, f"gap=1.0 リセット ({p2['gap']})"),
        _assert(p2["timestamp"] >= old_timestamp,
                "timestamp 更新 (>= 旧、now_ts 同秒許容)"),
        _assert(p2["last_cycle"] == 11, "last_cycle=11"),
        _assert(p2["priority"] is not None, "priority 再算済"),
    ])


# ============================================================
# pending_burden signal (改善6-D)
# ============================================================

def test_pending_burden_signal_rises_with_unobserved():
    print("== pending_burden: 未消化 pending 複数で signal 上昇 ==")
    state0 = _fresh_state()
    sig0 = calc_pressure_signals(state0)
    base = sig0.get("pending_burden", 0.0)

    state1 = _fresh_state()
    # 3 件の未消化 pending (semantic_merge=True)
    for i in range(3):
        pending_add_response_intent(
            state=state1, channel="device",
            text=f"msg {i}", cycle_id=0,
        )
    sig1 = calc_pressure_signals(state1)
    burden1 = sig1.get("pending_burden", 0.0)

    return all([
        _assert(base == 0.0, "pending ゼロなら burden=0"),
        _assert(burden1 > 0.0, f"未消化 pending ありで burden 上昇 ({burden1:.3f})"),
    ])


def test_pending_burden_excludes_observed():
    print("== pending_burden: 消化済 pending は除外 ==")
    state = _fresh_state()
    p = pending_add_response_intent(
        state=state, channel="device", text="msg", cycle_id=0,
    )
    # 手動で消化済マーク
    p["observed_content"] = "replied"
    p["gap"] = 0.0
    sig = calc_pressure_signals(state)
    return _assert(sig.get("pending_burden", -1.0) == 0.0,
                   "消化済は burden に入らない")


def test_pending_burden_excludes_deprecated():
    print("== pending_burden: deprecated pending は除外 ==")
    state = _fresh_state()
    p = pending_add_response_intent(
        state=state, channel="device", text="msg", cycle_id=0,
    )
    p["expiry_policy"] = "deprecated"
    sig = calc_pressure_signals(state)
    return _assert(sig.get("pending_burden", -1.0) == 0.0,
                   "deprecated は burden に入らない")


def test_pending_burden_param_exists():
    print("== pending_burden: ENTROPY_PARAMS に w_pending_burden 登録 ==")
    return _assert("w_pending_burden" in ENTROPY_PARAMS,
                   f"w_pending_burden={ENTROPY_PARAMS.get('w_pending_burden')}")


# ============================================================
# pending_prune 安全弁 (attempts >= 50)
# ============================================================

def test_safety_cap_constant():
    print("== 安全弁: PENDING_ATTEMPTS_SAFETY_CAP == 50 ==")
    return _assert(PENDING_ATTEMPTS_SAFETY_CAP == 50,
                   f"閾値 {PENDING_ATTEMPTS_SAFETY_CAP}")


def test_safety_cap_triggers_deprecated():
    print("== 安全弁: attempts >= 50 で expiry_policy=deprecated ==")
    state = _fresh_state()
    p = pending_add(
        state, source_action="test",
        expected_observation="x", lag_kind="cycles",
        content_intent="c", cycle_id=0,
    )
    p["attempts"] = 50
    pending_prune(state, current_cycle=1)
    return all([
        _assert(p["expiry_policy"] == "deprecated", "deprecated マーク付与"),
        _assert(p in state["pending"], "state に残る (削除されない)"),
    ])


def test_safety_cap_below_threshold_no_op():
    print("== 安全弁: attempts < 50 は無影響 ==")
    state = _fresh_state()
    p = pending_add(
        state, source_action="test",
        expected_observation="x", lag_kind="cycles",
        content_intent="c", cycle_id=0,
        expiry_policy="dynamic_n",
    )
    p["attempts"] = 49
    original_policy = p["expiry_policy"]
    pending_prune(state, current_cycle=1)
    return _assert(p.get("expiry_policy") == original_policy,
                   "expiry_policy 変化なし")


def test_deprecated_survives_prune():
    print("== 安全弁: deprecated は dynamic_n 競争外で state に残る ==")
    state = _fresh_state()
    # dynamic_n で競争しそうな pending を 3 件追加、うち 1 件を deprecated
    for i in range(3):
        p = pending_add(
            state, source_action="test",
            expected_observation=f"x{i}", lag_kind="cycles",
            content_intent=f"c{i}", cycle_id=0,
            expiry_policy="dynamic_n",
        )
    # 最初の 1 件を attempts=55 に、deprecated 化狙い
    state["pending"][0]["attempts"] = 55
    pending_prune(state, current_cycle=1, dynamic_n=1)
    # deprecated + dynamic_candidate 上位 1 件 = 合計 2 件残る想定
    deprecated_present = any(
        p.get("expiry_policy") == "deprecated"
        for p in state["pending"]
    )
    return all([
        _assert(deprecated_present, "deprecated が state に残存"),
        _assert(len(state["pending"]) >= 1,
                f"state 空にならない (現 {len(state['pending'])})"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("response_intent: 基本生成", test_response_intent_basic),
        ("response_intent: match_pattern", test_response_intent_match_pattern),
        ("response_intent: content preview", test_response_intent_content_preview),
        ("merge: 同 channel 未消化 → 1 件 attempts=2",
         test_response_intent_merge_same_channel_unobserved),
        ("merge: 3 連続 → 1 件 attempts=3",
         test_response_intent_merge_three_consecutive),
        ("no merge: 別 channel は別 pending",
         test_response_intent_no_merge_different_channel),
        ("no merge: 消化済 pending とは混ぜない",
         test_response_intent_no_merge_observed_pending),
        ("no merge: 別 source_action とは混ぜない",
         test_response_intent_no_merge_different_source_action),
        ("merge: gap リセット + priority 再算",
         test_response_intent_merge_resets_gap_and_priority),
        ("burden: 未消化で上昇", test_pending_burden_signal_rises_with_unobserved),
        ("burden: 消化済 除外", test_pending_burden_excludes_observed),
        ("burden: deprecated 除外", test_pending_burden_excludes_deprecated),
        ("burden: ENTROPY_PARAMS 登録", test_pending_burden_param_exists),
        ("safety: 定数=50", test_safety_cap_constant),
        ("safety: 50 で deprecated", test_safety_cap_triggers_deprecated),
        ("safety: 49 は無影響", test_safety_cap_below_threshold_no_op),
        ("safety: deprecated は生き残る", test_deprecated_survives_prune),
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
